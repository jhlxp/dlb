"""Measure identical full-quota/MCRB pre-transfer paths with alternating order."""

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


SIM_DIR = Path("/home/chen/workspace/infra/code/DeepEP-MCRB/tests")
sys.path.insert(0, str(SIM_DIR))

from mcrb_2x4_sim import (  # noqa: E402
    DEFAULT_EMPIRICAL_DISTRIBUTION,
    Message,
    choose_rail_from_quota,
    compute_quota_row,
    empirical_receive_messages,
)


LOG_ROOT = Path(__file__).resolve().parent / "logs"


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


def selected_loads(rows: list[list[int]]) -> list[int]:
    """Return selected-Rail loads."""

    return [sum(row[rail] for row in rows) for rail in range(RAILS)]


def worker(local_rank: int, port: int, log_dir: str, dtype_name: str,
           args_dict: dict) -> None:
    """Run one Rank."""

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
    iterations = int(args_dict["iterations"])
    warmup = int(args_dict["warmup"])

    messages, metadata = empirical_receive_messages(
        source_loads=loads,
        seed=int(args_dict["seed"]),
        distribution_path=Path(args_dict["distribution_path"]),
    )
    full_rows = compute_quota_rows(loads, source_server, 1 - source_server, 0)
    route_rows = mcrb_rows(messages)
    if full_rows != route_rows:
        raise AssertionError(f"rows differ: full={full_rows}, mcrb={route_rows}")

    element_bytes = 2 if dtype_name == "bf16" else 1
    num_scales = 0 if dtype_name == "bf16" else (hidden + 127) // 128
    record_bytes = (
        hidden * element_bytes + num_scales * 4 + 16 + topk * 8 + 15
    ) // 16 * 16
    max_tokens = max(max(loads), max(selected_loads(full_rows)))

    extension = load_dlb_test_extension()
    runtime = extension.DlbP2PLoopbackRuntime(
        rank, WORLD_SIZE, max_tokens, record_bytes
    )
    runtime.sync(gather_object(runtime.get_ipc_handle()))
    stride = int(runtime.get_record_stride())
    records = make_records(rank, max_tokens, stride, torch.device("cuda", local_rank))
    full_quota = torch.tensor(full_rows, dtype=torch.int32, device=records.device)
    mcrb_quota = torch.tensor(route_rows, dtype=torch.int32, device=records.device)
    incoming = sum(
        full_rows[producer][source_rail]
        for producer in range(RAILS)
        if producer != source_rail
    )

    for index in range(warmup):
        runtime.reset()
        runtime.dispatch_on(records, full_quota, loads[source_rail], incoming, 100 + index)
        runtime.reset()
        runtime.dispatch_on(records, mcrb_quota, loads[source_rail], incoming, 200 + index)
        dist.barrier()

    samples = {"full_first": [], "mcrb_second": [], "mcrb_first": [], "full_second": []}
    for index in range(iterations):
        if index % 2 == 0:
            runtime.reset()
            samples["full_first"].append(event_ms(
                lambda: runtime.dispatch_on(
                    records, full_quota, loads[source_rail], incoming, 1000 + index
                )
            ))
            dist.barrier()
            runtime.reset()
            samples["mcrb_second"].append(event_ms(
                lambda: runtime.dispatch_on(
                    records, mcrb_quota, loads[source_rail], incoming, 2000 + index
                )
            ))
        else:
            runtime.reset()
            samples["mcrb_first"].append(event_ms(
                lambda: runtime.dispatch_on(
                    records, mcrb_quota, loads[source_rail], incoming, 3000 + index
                )
            ))
            dist.barrier()
            runtime.reset()
            samples["full_second"].append(event_ms(
                lambda: runtime.dispatch_on(
                    records, full_quota, loads[source_rail], incoming, 4000 + index
                )
            ))
        dist.barrier()

    reduced = {
        name: global_max_samples(values)
        for name, values in samples.items()
    }
    if rank == 0:
        report = {
            "status": "passed",
            "case": "identical_pre_transfer_order_effect",
            "workload": {
                "dtype": dtype_name,
                "hidden": hidden,
                "topk": topk,
                "record_stride_bytes": stride,
                "source_rail_loads": loads,
                "cdf_destination_rail_hits": metadata["destination_rail_message_hits"],
            },
            "rows_equal": True,
            "source_by_selected_rows": full_rows,
            "latency": {
                name: summarize(values)
                for name, values in reduced.items()
            },
        }
        Path(log_dir, f"{dtype_name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Run the diagnostic."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--empirical-distribution",
        type=Path,
        default=DEFAULT_EMPIRICAL_DISTRIBUTION,
    )
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / f"{timestamp}-pre-transfer-order-effect"
    log_dir.mkdir(parents=True, exist_ok=False)
    args_dict = {
        "hidden": args.hidden,
        "topk": args.topk,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "distribution_path": str(args.empirical_distribution),
    }
    Path(log_dir, "config.json").write_text(
        json.dumps(args_dict, indent=2), encoding="utf-8"
    )
    mp.spawn(
        worker,
        args=(find_free_port(), str(log_dir), args.dtype, args_dict),
        nprocs=WORLD_SIZE,
        join=True,
    )
    print(f"log_dir={log_dir}", flush=True)


if __name__ == "__main__":
    main()
