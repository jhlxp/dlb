#!/usr/bin/env python3
"""Benchmark fused DLB non-transport dispatch preparation at EP8."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dlb import DLBBuffer  # noqa: E402


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[int(fraction * (len(ordered) - 1))]


def reduce_max(value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def run(args: argparse.Namespace) -> None:
    dist.init_process_group("cpu:gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.get_world_size() != 8:
        raise RuntimeError("DLB fused prepare benchmark requires EP8")
    torch.cuda.set_device(local_rank)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + rank)
    x = torch.randn(
        (args.tokens_per_rank, args.hidden),
        generator=generator,
        dtype=torch.float32,
    ).to(device=local_rank, dtype=torch.bfloat16)
    scores = torch.randn(
        (args.tokens_per_rank, args.num_experts),
        generator=generator,
        dtype=torch.float32,
    ).to(local_rank)
    values, topk_idx = torch.topk(
        scores, args.num_topk, dim=-1, sorted=False
    )
    topk_idx = topk_idx.to(torch.int64).contiguous()
    topk_weights = torch.softmax(values, dim=-1).to(torch.float32).contiguous()
    del scores, values

    buffer = DLBBuffer(
        dist.group.WORLD,
        gpus_per_server=4,
        num_max_tokens_per_rank=args.tokens_per_rank,
        hidden=args.hidden,
        num_experts=args.num_experts,
        num_topk=args.num_topk,
        dtype=torch.bfloat16,
        pipeline_depth=2,
    )

    epoch = 1
    round_id = 0

    def measure() -> tuple[list[float], list[float]]:
        nonlocal epoch, round_id
        method = buffer._comm.benchmark_prepare_moe_device
        for _ in range(args.warmups):
            result = method(
                x, topk_idx, topk_weights, epoch, round_id, args.num_experts
            )
            torch.cuda.synchronize(local_rank)
            epoch += 1
            round_id += 1
        del result
        dist.barrier()

        cuda_samples: list[float] = []
        wall_samples: list[float] = []
        for _ in range(args.iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(local_rank)
            wall_start = time.perf_counter()
            start_event.record()
            result = method(
                x, topk_idx, topk_weights, epoch, round_id, args.num_experts
            )
            end_event.record()
            end_event.synchronize()
            wall_us = (time.perf_counter() - wall_start) * 1e6
            cuda_us = start_event.elapsed_time(end_event) * 1000.0
            cuda_samples.append(reduce_max(cuda_us))
            wall_samples.append(reduce_max(wall_us))
            epoch += 1
            round_id += 1
        del result
        return cuda_samples, wall_samples

    device_cuda, device_wall = measure()

    if rank == 0:
        print(
            f"DLB fused prepare: EP8 tokens/rank={args.tokens_per_rank} "
            f"experts={args.num_experts} topk={args.num_topk} hidden={args.hidden}"
        )
        for name, samples in (
            ("GPU-plan CUDA timeline", device_cuda),
            ("GPU-plan wall", device_wall),
        ):
            print(
                f"{name}: mean={statistics.fmean(samples):.3f} us "
                f"p50={percentile(samples, 0.50):.3f} us "
                f"p95={percentile(samples, 0.95):.3f} us"
            )

    buffer.close()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
