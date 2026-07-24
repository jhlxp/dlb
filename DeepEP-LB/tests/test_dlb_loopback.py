"""Validate and benchmark the fused DLB NVLink Rail patch.

The test uses eight GPUs as two logical four-GPU servers. CUDA IPC exposes
only the four local Rail buffers to each Rank. It validates source staging up
to the selected Rail send ring and moved-token repair back to the origin Rail;
it does not pretend that the omitted network boundary is RDMA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dlb_test_extension import load_dlb_test_extension


WORLD_SIZE = 8
RAILS = 4
LOG_ROOT = Path(__file__).resolve().parent / "logs"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "csrc/deep_ep.cpp",
    "csrc/kernels/internode.cu",
    "csrc/kernels/rail_lb.cuh",
    "csrc/kernels/rail_lb.cu",
    "tests/csrc/dlb_loopback_test.cu",
    "tests/test_dlb_loopback.py",
)


def source_sha256() -> str:
    """Identify the production Rail patch and this test backend."""
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def find_free_port() -> int:
    """Return an unused localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def gather_object(value):
    """Gather one picklable value from every EP Rank."""
    values = [None] * WORLD_SIZE
    dist.all_gather_object(values, value)
    return values


def compute_quota_rows(
    loads: list[int],
    source_server: int,
    destination_server: int,
    round_id: int,
) -> list[list[int]]:
    """Mirror the compact deterministic production quota calculation."""
    total = sum(loads)
    targets = [total // RAILS] * RAILS
    first = (
        source_server * 17 + destination_server * 31 + round_id
    ) % RAILS
    for offset in range(total % RAILS):
        targets[(first + offset) % RAILS] += 1
    deficits = [
        max(targets[rail] - loads[rail], 0)
        for rail in range(RAILS)
    ]
    rows = [[0] * RAILS for _ in range(RAILS)]
    for producer in range(RAILS):
        direct = min(loads[producer], targets[producer])
        rows[producer][producer] = direct
        surplus = loads[producer] - direct
        for selected in range(RAILS):
            moved = min(surplus, deficits[selected])
            rows[producer][selected] += moved
            surplus -= moved
            deficits[selected] -= moved
    return rows


def selected_ordinals(
    row: list[int],
    source_rail: int,
    selected_rail: int,
) -> range:
    """Return source-flow ordinals assigned to one selected Rail."""
    begin = 0
    order = [source_rail] + [
        rail for rail in range(RAILS) if rail != source_rail
    ]
    for rail in order:
        end = begin + row[rail]
        if rail == selected_rail:
            return range(begin, end)
        begin = end
    raise AssertionError("selected Rail was not found")


def expected_ids(
    source_server: int,
    selected_rail: int,
    rows: list[list[int]],
) -> torch.Tensor:
    """Build the token identities expected in one selected Rail ring."""
    values: list[int] = []
    for producer in range(RAILS):
        source_rank = source_server * RAILS + producer
        for ordinal in selected_ordinals(
            rows[producer], producer, selected_rail
        ):
            values.append((source_rank << 32) | ordinal)
    return torch.tensor(sorted(values), dtype=torch.int64)


def make_records(
    rank: int,
    num_tokens: int,
    record_stride: int,
    device: torch.device,
) -> torch.Tensor:
    """Create deterministic byte records without Python-side token loops."""
    token = torch.arange(
        num_tokens, device=device, dtype=torch.int64
    )[:, None]
    byte = torch.arange(
        record_stride, device=device, dtype=torch.int64
    )[None, :]
    return (
        token * 13 + byte * 7 + rank * 29
    ).remainder(251).to(torch.uint8).contiguous()


def event_ms(operation) -> float:
    """Measure one local CUDA operation."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def percentile(values: list[float], quantile: float) -> float:
    """Return a nearest-rank percentile."""
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize latency samples in milliseconds."""
    return {
        "min_ms": min(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": sum(values) / len(values),
    }


def global_max_samples(
    local_samples: list[float],
) -> list[float] | None:
    """Return the slowest Rank for each measured iteration."""
    all_samples = gather_object(local_samples)
    if dist.get_rank() != 0:
        return None
    return [
        max(rank_samples[index] for rank_samples in all_samples)
        for index in range(len(local_samples))
    ]


def worker(
    local_rank: int,
    port: int,
    log_dir: str,
    dtype_name: str,
    hidden: int,
    topk: int,
    warmup: int,
    iterations: int,
) -> None:
    """Run one BF16 or FP8 source-staging validation."""
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
    max_tokens = max(loads)
    element_bytes = 2 if dtype_name == "bf16" else 1
    num_scales = 0 if dtype_name == "bf16" else (hidden + 127) // 128
    source_meta_bytes = 16
    raw_record_bytes = (
        hidden * element_bytes
        + num_scales * 4
        + source_meta_bytes
        + topk * (4 + 4)
    )
    record_bytes = (raw_record_bytes + 15) // 16 * 16

    extension = load_dlb_test_extension()
    runtime = extension.DlbP2PLoopbackRuntime(
        rank, WORLD_SIZE, max_tokens, record_bytes
    )
    runtime.sync(gather_object(runtime.get_ipc_handle()))
    stride = int(runtime.get_record_stride())
    device = torch.device("cuda", local_rank)
    records = make_records(rank, max_tokens, stride, device)
    combined = torch.empty_like(records)
    source_loads = torch.tensor(
        loads, dtype=torch.int32, device=device
    )
    rows = compute_quota_rows(
        loads, source_server, 1 - source_server, 0
    )
    target_loads = [
        sum(rows[producer][selected] for producer in range(RAILS))
        for selected in range(RAILS)
    ]
    quota_rows = torch.tensor(
        rows, dtype=torch.int32, device=device
    )
    incoming_moved = sum(
        rows[producer][source_rail]
        for producer in range(RAILS)
        if producer != source_rail
    )

    # Validate the production quota helper before exercising the P2P path.
    stats = runtime.quota_stats(
        source_loads, source_server, 0
    ).cpu().tolist()
    if stats[:RAILS] != rows[source_rail]:
        raise AssertionError(
            f"quota mismatch: CUDA={stats[:RAILS]}, "
            f"reference={rows[source_rail]}"
        )

    epoch = 1
    runtime.reset()
    runtime.pack_off(records, loads[source_rail])
    torch.cuda.synchronize()
    dist.barrier()
    off_ids, off_records = runtime.materialize_ring(
        loads[source_rail]
    )
    expected_off_ids = torch.tensor(
        [
            (rank << 32) | token
            for token in range(loads[source_rail])
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(off_ids, expected_off_ids)
    torch.testing.assert_close(
        off_records[:, :32], records[: loads[source_rail], :32]
    )
    runtime.combine(
        combined,
        loads[source_rail],
        loads[source_rail],
        11,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        combined[: loads[source_rail]],
        records[: loads[source_rail]],
    )

    runtime.reset()
    dist.barrier()
    runtime.dispatch_on(
        records,
        quota_rows,
        loads[source_rail],
        incoming_moved,
        epoch,
    )
    torch.cuda.synchronize()
    dist.barrier()
    on_ids, on_records = runtime.materialize_ring(
        target_loads[source_rail]
    )
    order = on_ids.argsort()
    expected = expected_ids(
        source_server, source_rail, rows
    ).to(device)
    if not torch.equal(on_ids[order], expected):
        invalid = int((on_ids == -1).sum().item())
        actual_sorted = on_ids[order]
        mismatch = torch.nonzero(
            actual_sorted != expected, as_tuple=False
        ).flatten()
        stage_ids, stage_ready = runtime.materialize_stage(
            incoming_moved
        )
        print(
            f"[rank {rank}] invalid_ring_ids={invalid}, "
            f"first_ids={on_ids[:16].cpu().tolist()}, "
            f"actual_sorted_first={actual_sorted[:16].cpu().tolist()}, "
            f"actual_tail={actual_sorted[-8:].cpu().tolist()}, "
            f"expected_first={expected[:16].cpu().tolist()}, "
            f"expected_tail={expected[-8:].cpu().tolist()}, "
            f"mismatch_first={mismatch[:16].cpu().tolist()}, "
            f"unique={int(torch.unique(on_ids).numel())}, "
            f"stage_zero={int((stage_ids == 0).sum().item())}, "
            f"stage_unique={int(torch.unique(stage_ids).numel())}, "
            f"stage_ready_values="
            f"{torch.unique(stage_ready).cpu().tolist()}",
            flush=True,
        )
    torch.testing.assert_close(on_ids[order], expected)

    # Payload bytes must still match the source identity after peer staging.
    sorted_ids = on_ids[order]
    source_ranks = torch.bitwise_right_shift(sorted_ids, 32)
    source_tokens = torch.bitwise_and(
        sorted_ids, (1 << 32) - 1
    )
    byte = torch.arange(32, device=device, dtype=torch.int64)[None, :]
    expected_prefix = (
        source_tokens[:, None] * 13
        + byte * 7
        + source_ranks[:, None] * 29
    ).remainder(251).to(torch.uint8)
    torch.testing.assert_close(on_records[order, :32], expected_prefix)
    dist.barrier()
    runtime.combine(
        combined,
        loads[source_rail],
        target_loads[source_rail],
        12,
    )
    torch.cuda.synchronize()
    dist.barrier()
    torch.testing.assert_close(
        combined[: loads[source_rail]],
        records[: loads[source_rail]],
    )

    # A second dispatch/combine epoch catches stale flags and slot reuse.
    epoch = 2
    runtime.reset()
    dist.barrier()
    runtime.dispatch_on(
        records,
        quota_rows,
        loads[source_rail],
        incoming_moved,
        epoch,
    )
    torch.cuda.synchronize()
    dist.barrier()
    second_ids, _ = runtime.materialize_ring(
        target_loads[source_rail]
    )
    torch.testing.assert_close(second_ids.sort().values, expected)
    dist.barrier()
    runtime.combine(
        combined,
        loads[source_rail],
        target_loads[source_rail],
        13,
    )
    torch.cuda.synchronize()
    dist.barrier()
    torch.testing.assert_close(
        combined[: loads[source_rail]],
        records[: loads[source_rail]],
    )

    for index in range(warmup):
        runtime.reset()
        runtime.pack_off(records, loads[source_rail])
        torch.cuda.synchronize()
        runtime.combine(
            combined,
            loads[source_rail],
            loads[source_rail],
            100 + index,
        )
        torch.cuda.synchronize()
        dist.barrier()
        runtime.reset()
        runtime.dispatch_on(
            records,
            quota_rows,
            loads[source_rail],
            incoming_moved,
            100 + index,
        )
        torch.cuda.synchronize()
        dist.barrier()
        runtime.combine(
            combined,
            loads[source_rail],
            target_loads[source_rail],
            200 + index,
        )
        torch.cuda.synchronize()
        dist.barrier()

    quota_samples: list[float] = []
    off_samples: list[float] = []
    balanced_control_samples: list[float] = []
    off_combine_samples: list[float] = []
    on_dispatch_samples: list[float] = []
    on_combine_samples: list[float] = []
    for index in range(iterations):
        quota_samples.append(event_ms(
            lambda: runtime.quota_stats(
                source_loads, source_server, index
            )
        ))

        runtime.reset()
        balanced_control_samples.append(event_ms(
            lambda: runtime.pack_off(
                records, target_loads[source_rail]
            )
        ))
        runtime.reset()
        off_samples.append(event_ms(
            lambda: runtime.pack_off(
                records, loads[source_rail]
            )
        ))
        off_combine_samples.append(event_ms(
            lambda: runtime.combine(
                combined,
                loads[source_rail],
                loads[source_rail],
                1000 + index,
            )
        ))
        dist.barrier()

        current_epoch = 1000 + index
        runtime.reset()
        dist.barrier()
        on_dispatch_samples.append(event_ms(
            lambda: runtime.dispatch_on(
                records,
                quota_rows,
                loads[source_rail],
                incoming_moved,
                current_epoch,
            )
        ))
        dist.barrier()
        on_combine_samples.append(event_ms(
            lambda: runtime.combine(
                combined,
                loads[source_rail],
                target_loads[source_rail],
                2000 + index,
            )
        ))
        dist.barrier()

    quota_max = global_max_samples(quota_samples)
    off_max = global_max_samples(off_samples)
    balanced_control_max = global_max_samples(
        balanced_control_samples
    )
    off_combine_max = global_max_samples(off_combine_samples)
    on_dispatch_max = global_max_samples(on_dispatch_samples)
    on_combine_max = global_max_samples(on_combine_samples)
    rank_result = {
        "rank": rank,
        "source_server": source_server,
        "source_rail": source_rail,
        "off_messages": loads[source_rail],
        "on_messages": target_loads[source_rail],
        "direct_messages": rows[source_rail][source_rail],
        "moved_messages": int(stats[RAILS + 1]),
        "quota_row": rows[source_rail],
    }
    rank_results = gather_object(rank_result)

    if rank == 0:
        quota_summary = summarize(quota_max)
        off_summary = summarize(off_max)
        balanced_control_summary = summarize(balanced_control_max)
        off_combine_summary = summarize(off_combine_max)
        on_dispatch_summary = summarize(on_dispatch_max)
        on_combine_summary = summarize(on_combine_max)
        dispatch_incremental_p50 = (
            on_dispatch_summary["p50_ms"] -
            off_summary["p50_ms"]
        )
        staging_incremental_p50 = (
            on_dispatch_summary["p50_ms"] -
            balanced_control_summary["p50_ms"]
        )
        combine_incremental_p50 = (
            on_combine_summary["p50_ms"] -
            off_combine_summary["p50_ms"]
        )
        moved_per_server = sum(
            row["moved_messages"]
            for row in rank_results
            if row["source_server"] == 0
        )
        moved_bytes = moved_per_server * stride
        logical_bytes = sum(loads) * stride
        off_dispatch_gbps = (
            logical_bytes / off_summary["p50_ms"] / 1e6
        )
        on_dispatch_gbps = (
            logical_bytes / on_dispatch_summary["p50_ms"] / 1e6
        )
        off_combine_gbps = (
            logical_bytes / off_combine_summary["p50_ms"] / 1e6
        )
        on_combine_gbps = (
            logical_bytes / on_combine_summary["p50_ms"] / 1e6
        )
        rail_patch_gbps = (
            moved_bytes / staging_incremental_p50 / 1e6
            if staging_incremental_p50 > 0
            else None
        )
        report = {
            "status": "passed",
            "source_sha256": source_sha256(),
            "scope": (
                "test-only CUDA-IPC/NVLink source staging and "
                "moved-token reverse repair around an omitted "
                "network boundary"
            ),
            "topology": {
                "logical_servers": 2,
                "rails_per_server": 4,
                "ep_size": 8,
            },
            "workload": {
                "dtype": dtype_name,
                "hidden": hidden,
                "topk": topk,
                "record_stride_bytes": stride,
                "source_rail_loads": loads,
                "selected_rail_loads": target_loads,
            },
            "traffic": {
                "moved_messages_per_server": moved_per_server,
                "moved_bytes_per_server": moved_bytes,
                "moved_fraction": moved_per_server / sum(loads),
            },
            "correctness": {
                "token_conservation": "passed",
                "token_identity": "passed",
                "payload_bytes": "passed",
                "direct_path": "passed",
                "moved_path": "passed",
                "reverse_repair": "passed",
                "cached_schedule_reuse": "passed",
                "multi_epoch": "passed",
            },
            "latency": {
                "compact_quota_isolated": quota_summary,
                "lb_off_dispatch_pack": off_summary,
                "balanced_local_pack_control":
                    balanced_control_summary,
                "lb_on_fused_dispatch_pack": on_dispatch_summary,
                "on_minus_off_dispatch_p50_us":
                    dispatch_incremental_p50 * 1000,
                "rail_patch_over_balanced_control_p50_us":
                    staging_incremental_p50 * 1000,
                "lb_off_combine": off_combine_summary,
                "lb_on_combine_with_reverse_repair":
                    on_combine_summary,
                "combine_incremental_p50_us":
                    combine_incremental_p50 * 1000,
                "total_lb_incremental_p50_us":
                    (
                        dispatch_incremental_p50 +
                        combine_incremental_p50
                    ) * 1000,
            },
            "throughput_gbps": {
                "logical_payload_bytes_per_server": logical_bytes,
                "lb_off_dispatch_pack": off_dispatch_gbps,
                "lb_on_fused_dispatch_pack": on_dispatch_gbps,
                "lb_off_combine": off_combine_gbps,
                "lb_on_combine_with_reverse_repair":
                    on_combine_gbps,
                "moved_payload_over_rail_patch_delta":
                    rail_patch_gbps,
            },
            "rank_results": rank_results,
        }
        output = Path(log_dir) / f"{dtype_name}.json"
        output.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Run production-shaped BF16 and FP8 source-staging cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--dtype", choices=("bf16", "fp8", "both"), default="both"
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_ROOT / f"{timestamp}-dlb-p2p-data-plane"
    log_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES", ""
        ),
        "hidden": args.hidden,
        "topk": args.topk,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "dtype": args.dtype,
        "source_loads": [5000, 6000, 7000, 8000],
        "source_sha256": source_sha256(),
        "source_files": SOURCE_FILES,
    }
    (log_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    dtypes = ("bf16", "fp8") if args.dtype == "both" else (args.dtype,)
    for dtype_name in dtypes:
        mp.spawn(
            worker,
            args=(
                find_free_port(),
                str(log_dir),
                dtype_name,
                args.hidden,
                args.topk,
                args.warmup,
                args.iterations,
            ),
            nprocs=WORLD_SIZE,
            join=True,
        )
    print(f"log_dir={log_dir}")


if __name__ == "__main__":
    main()
