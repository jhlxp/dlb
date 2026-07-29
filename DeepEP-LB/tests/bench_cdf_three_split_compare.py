"""Run raw, full-quota, and MCRB under one empirical CDF split-stage case."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dlb_test_extension import load_dlb_test_extension
from test_dlb_loopback import (
    RAILS,
    WORLD_SIZE,
    compute_quota_rows,
    event_ms,
    find_free_port,
    gather_object,
    global_max_samples,
    make_records,
    summarize,
)


THIS_DIR = Path(__file__).resolve().parent
SIM_DIR = Path("/home/chen/workspace/infra/code/DeepEP-MCRB/tests")
sys.path.insert(0, str(SIM_DIR))

from mcrb_2x4_sim import (  # noqa: E402
    DEFAULT_EMPIRICAL_DISTRIBUTION,
    Message,
    balance_messages,
    choose_rail_from_quota,
    compute_quota_row,
    empirical_receive_messages,
    mask_contains,
)


LOG_ROOT = THIS_DIR / "logs"


def selected_loads(rows: list[list[int]]) -> list[int]:
    """Return load per selected Rail."""

    return [sum(row[rail] for row in rows) for rail in range(RAILS)]


def moved_messages(rows: list[list[int]]) -> int:
    """Return source-side moved message count."""

    return sum(
        rows[source][selected]
        for source in range(RAILS)
        for selected in range(RAILS)
        if source != selected
    )


def mcrb_rows(messages: list[Message]) -> list[list[int]]:
    """Return MCRB source-Rail by selected-Rail counts."""

    loads = [0] * RAILS
    for message in messages:
        loads[message.source_rail] += 1
    quotas = [compute_quota_row(loads, source) for source in range(RAILS)]
    used = [[0] * RAILS for _ in range(RAILS)]
    rows = [[0] * RAILS for _ in range(RAILS)]
    for message in messages:
        selected = choose_rail_from_quota(
            message.source_rail,
            message.dst_mask,
            quotas[message.source_rail],
            used[message.source_rail],
            allow_extra_forward=False,
        )
        rows[message.source_rail][selected] += 1
    return rows


def selected_destinations(
    messages: list[Message],
    rows: list[list[int]],
    selected_rail: int,
) -> list[int]:
    """Return destination Rails that arrive at one selected Rail."""

    grouped = [[] for _ in range(RAILS)]
    for message in messages:
        grouped[message.source_rail].append(message)

    destinations: list[int] = []
    for source_rail, source_messages in enumerate(grouped):
        cursor = 0
        for rail in [source_rail] + [
            other for other in range(RAILS) if other != source_rail
        ]:
            count = rows[source_rail][rail]
            chunk = source_messages[cursor: cursor + count]
            if rail == selected_rail:
                for message in chunk:
                    destination = next(
                        dst for dst in range(RAILS)
                        if mask_contains(message.dst_mask, dst)
                    )
                    destinations.append(destination)
            cursor += count
    return destinations


def raw_destinations(messages: list[Message], source_rail: int) -> list[int]:
    """Return destination Rails for raw same-Rail sends."""

    destinations: list[int] = []
    for message in messages:
        if message.source_rail != source_rail:
            continue
        destination = next(
            dst for dst in range(RAILS)
            if mask_contains(message.dst_mask, dst)
        )
        destinations.append(destination)
    return destinations


def seed_post_forward(runtime, records, destinations: list[int], device) -> None:
    """Seed a receive ring with destination Rail IDs."""

    if not destinations:
        return
    rails = torch.tensor(destinations, dtype=torch.int32, device=device)
    runtime.seed_receive_ring(records, rails, len(destinations))
    torch.cuda.synchronize()


def worker(local_rank: int, port: int, log_dir: str, dtype_name: str,
           args_dict: dict) -> None:
    """Run one Rank of the split-stage CDF comparison."""

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=local_rank,
        world_size=WORLD_SIZE,
    )
    rank = dist.get_rank()
    source_server = rank // RAILS
    source_rail = rank % RAILS
    loads = [5000, 6000, 7000, 8000]
    hidden = int(args_dict["hidden"])
    topk = int(args_dict["topk"])
    warmup = int(args_dict["warmup"])
    iterations = int(args_dict["iterations"])
    seed = int(args_dict["seed"])
    distribution_path = Path(args_dict["distribution_path"])
    mcrb_first = bool(args_dict["mcrb_first"])

    messages, metadata = empirical_receive_messages(
        source_loads=loads,
        seed=seed,
        distribution_path=distribution_path,
    )
    full_rows = compute_quota_rows(loads, source_server, 1 - source_server, 0)
    mcrb_route_rows = mcrb_rows(messages)
    mcrb_algo = balance_messages(messages, allow_extra_forward=False)
    full_targets = selected_loads(full_rows)
    mcrb_targets = selected_loads(mcrb_route_rows)

    element_bytes = 2 if dtype_name == "bf16" else 1
    num_scales = 0 if dtype_name == "bf16" else (hidden + 127) // 128
    raw_record_bytes = hidden * element_bytes + num_scales * 4 + 16 + topk * 8
    record_bytes = (raw_record_bytes + 15) // 16 * 16
    max_tokens = max(max(loads), max(full_targets), max(mcrb_targets))

    extension = load_dlb_test_extension()
    runtime = extension.DlbP2PLoopbackRuntime(
        rank, WORLD_SIZE, max_tokens, record_bytes
    )
    runtime.sync(gather_object(runtime.get_ipc_handle()))
    stride = int(runtime.get_record_stride())
    device = torch.device("cuda", local_rank)
    records = make_records(rank, max_tokens, stride, device)
    combined = torch.empty_like(records)

    full_quota_rows = torch.tensor(full_rows, dtype=torch.int32, device=device)
    mcrb_quota_rows = torch.tensor(
        mcrb_route_rows, dtype=torch.int32, device=device
    )
    full_incoming = sum(
        full_rows[producer][source_rail]
        for producer in range(RAILS)
        if producer != source_rail
    )
    mcrb_incoming = sum(
        mcrb_route_rows[producer][source_rail]
        for producer in range(RAILS)
        if producer != source_rail
    )

    raw_post_dest = raw_destinations(messages, source_rail)
    full_post_dest = selected_destinations(messages, full_rows, source_rail)
    mcrb_post_dest = selected_destinations(
        messages, mcrb_route_rows, source_rail
    )

    # Correctness smoke.
    runtime.reset()
    runtime.pack_off(records, loads[source_rail])
    torch.cuda.synchronize()
    runtime.combine(combined, loads[source_rail], loads[source_rail], 11)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        combined[: loads[source_rail]],
        records[: loads[source_rail]],
    )
    runtime.reset()
    runtime.dispatch_on(
        records, full_quota_rows, loads[source_rail], full_incoming, 12
    )
    torch.cuda.synchronize()
    runtime.combine(combined, loads[source_rail], full_targets[source_rail], 13)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        combined[: loads[source_rail]],
        records[: loads[source_rail]],
    )

    for index in range(warmup):
        runtime.reset()
        runtime.pack_off(records, loads[source_rail])
        runtime.combine(combined, loads[source_rail], loads[source_rail],
                        100 + index)
        runtime.reset()
        runtime.dispatch_on(records, full_quota_rows, loads[source_rail],
                            full_incoming, 200 + index)
        runtime.combine(combined, loads[source_rail], full_targets[source_rail],
                        300 + index)
        runtime.reset()
        runtime.dispatch_on(records, mcrb_quota_rows, loads[source_rail],
                            mcrb_incoming, 400 + index)
        runtime.combine(combined, loads[source_rail], mcrb_targets[source_rail],
                        500 + index)
        runtime.reset()
        seed_post_forward(runtime, records, raw_post_dest, device)
        runtime.clear_post_forward_counter()
        runtime.post_forward(len(raw_post_dest))
        dist.barrier()

    samples = {
        "raw_pre": [],
        "raw_combine": [],
        "raw_post": [],
        "full_pre": [],
        "full_combine": [],
        "full_post": [],
        "mcrb_pre": [],
        "mcrb_combine": [],
        "mcrb_post": [],
    }
    for index in range(iterations):
        runtime.reset()
        samples["raw_pre"].append(event_ms(
            lambda: runtime.pack_off(records, loads[source_rail])
        ))
        samples["raw_combine"].append(event_ms(
            lambda: runtime.combine(
                combined, loads[source_rail], loads[source_rail], 1000 + index
            )
        ))
        runtime.reset()
        seed_post_forward(runtime, records, raw_post_dest, device)
        runtime.clear_post_forward_counter()
        samples["raw_post"].append(event_ms(
            lambda: runtime.post_forward(len(raw_post_dest))
        ))
        dist.barrier()

        def measure_lb_path(name, quota_rows, incoming, targets, post_dest,
                            dispatch_epoch, combine_epoch):
            runtime.reset()
            samples[f"{name}_pre"].append(event_ms(
                lambda: runtime.dispatch_on(
                    records, quota_rows, loads[source_rail],
                    incoming, dispatch_epoch + index
                )
            ))
            samples[f"{name}_combine"].append(event_ms(
                lambda: runtime.combine(
                    combined, loads[source_rail],
                    targets[source_rail], combine_epoch + index
                )
            ))
            runtime.reset()
            seed_post_forward(runtime, records, post_dest, device)
            runtime.clear_post_forward_counter()
            samples[f"{name}_post"].append(event_ms(
                lambda: runtime.post_forward(len(post_dest))
            ))
            dist.barrier()

        if mcrb_first:
            measure_lb_path(
                "mcrb", mcrb_quota_rows, mcrb_incoming,
                mcrb_targets, mcrb_post_dest, 4000, 5000
            )
            measure_lb_path(
                "full", full_quota_rows, full_incoming,
                full_targets, full_post_dest, 2000, 3000
            )
        else:
            measure_lb_path(
                "full", full_quota_rows, full_incoming,
                full_targets, full_post_dest, 2000, 3000
            )
            measure_lb_path(
                "mcrb", mcrb_quota_rows, mcrb_incoming,
                mcrb_targets, mcrb_post_dest, 4000, 5000
            )

    reduced_values = {
        name: global_max_samples(values)
        for name, values in samples.items()
    }

    if rank == 0:
        reduced = {
            name: summarize(values)
            for name, values in reduced_values.items()
        }

        def total(prefix: str) -> float:
            return (
                reduced[f"{prefix}_pre"]["p50_ms"]
                + reduced[f"{prefix}_combine"]["p50_ms"]
                + reduced[f"{prefix}_post"]["p50_ms"]
            )

        report = {
            "status": "passed",
            "case": "empirical_cdf_hotspot",
            "scope": (
                "2x4 CUDA-IPC/NVLink split stages; no RDMA/HCA; "
                "no expert GEMM"
            ),
            "mcrb_first": mcrb_first,
            "workload": {
                "dtype": dtype_name,
                "hidden": hidden,
                "topk": topk,
                "record_stride_bytes": stride,
                "source_rail_loads": loads,
                "cdf_destination_rail_hits": metadata[
                    "destination_rail_message_hits"
                ],
            },
            "full_quota": {
                "source_by_selected_rows": full_rows,
                "selected_rail_loads": full_targets,
                "moved_messages": moved_messages(full_rows),
            },
            "mcrb": {
                "source_by_selected_rows": mcrb_route_rows,
                "selected_rail_loads": mcrb_targets,
                "moved_messages": moved_messages(mcrb_route_rows),
                "extra_forwarding_hops": mcrb_algo["extra_forwarding_hops"],
                "third_party_rail_violations":
                    mcrb_algo["third_party_rail_violations"],
            },
            "latency": {
                "deepep_raw_pre_transfer": reduced["raw_pre"],
                "deepep_raw_combine": reduced["raw_combine"],
                "deepep_raw_post_transfer": reduced["raw_post"],
                "deepep_raw_total_p50_ms": total("raw"),
                "full_quota_pre_transfer": reduced["full_pre"],
                "full_quota_combine": reduced["full_combine"],
                "full_quota_post_transfer": reduced["full_post"],
                "full_quota_total_p50_ms": total("full"),
                "mcrb_pre_transfer": reduced["mcrb_pre"],
                "mcrb_combine": reduced["mcrb_combine"],
                "mcrb_post_transfer": reduced["mcrb_post"],
                "mcrb_total_p50_ms": total("mcrb"),
            },
        }
        Path(log_dir, f"{dtype_name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Run the split-stage CDF comparison."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--mcrb-first", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp8", "both"), default="both")
    parser.add_argument(
        "--empirical-distribution",
        type=Path,
        default=DEFAULT_EMPIRICAL_DISTRIBUTION,
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / f"{timestamp}-cdf-three-split-compare"
    log_dir.mkdir(parents=True, exist_ok=False)
    args_dict = {
        "hidden": args.hidden,
        "topk": args.topk,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "distribution_path": str(args.empirical_distribution),
        "mcrb_first": args.mcrb_first,
    }
    Path(log_dir, "config.json").write_text(
        json.dumps(args_dict, indent=2), encoding="utf-8"
    )
    dtypes = ("bf16", "fp8") if args.dtype == "both" else (args.dtype,)
    for dtype_name in dtypes:
        mp.spawn(
            worker,
            args=(find_free_port(), str(log_dir), dtype_name, args_dict),
            nprocs=WORLD_SIZE,
            join=True,
        )
    print(f"log_dir={log_dir}", flush=True)


if __name__ == "__main__":
    main()
