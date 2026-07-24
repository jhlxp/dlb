"""Correctness and launch-cost checks for the compact DLB quota core."""

from __future__ import annotations

import argparse
import statistics

import torch

from dlb_test_extension import load_dlb_test_extension


@torch.no_grad()
def run(iterations: int) -> None:
    extension = load_dlb_test_extension()
    source_loads = torch.tensor(
        [
            [0, 0, 0, 0],
            [5000, 6000, 7000, 8000],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    quotas, balanced = extension.run_dlb_compact_quota(source_loads, 0, 0)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        quotas.sum(dim=-1, dtype=torch.int32),
        source_loads,
    )
    torch.testing.assert_close(
        balanced[1],
        torch.full((4,), 6500, dtype=torch.int32, device="cuda"),
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us: list[float] = []
    for _ in range(20):
        extension.run_dlb_compact_quota(source_loads, 0, 0)
    torch.cuda.synchronize()
    for _ in range(iterations):
        start.record()
        extension.run_dlb_compact_quota(source_loads, 0, 0)
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    print(f"before={source_loads[1].tolist()}")
    print(f"after={balanced[1].tolist()}")
    print(f"quota_rows={quotas[1].tolist()}")
    print(
        "compact_quota_cuda_us: "
        f"p50={statistics.median(samples_us):.3f}, "
        f"min={min(samples_us):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    run(args.iterations)
