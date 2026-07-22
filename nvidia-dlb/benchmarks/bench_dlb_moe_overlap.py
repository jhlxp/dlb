"""Compares sequential and split-phase DLB MoE microbatch execution.

The benchmark uses the same routed microbatches for both modes.  The
split-phase mode posts dispatch for microbatch ``n + 1`` while the current
stream evaluates the local expert MLP for microbatch ``n``.  It then finishes
``n + 1`` and combines the completed output of ``n``.  This is the safe
cross-microbatch overlap boundary exposed by :class:`dlb.DLBBuffer`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist


NVIDIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NVIDIA_ROOT))
sys.path.insert(0, str(NVIDIA_ROOT / "tests"))

from dlb import DLBBuffer  # noqa: E402
from test_dlb_moe_network import (  # noqa: E402
    TinyMoE,
    max_across_ranks,
    percentile,
    reference_forward,
    run_local_experts,
)


@dataclass(frozen=True)
class MoEMicrobatch:
    """Owns one deterministic routed input used by both benchmark modes."""

    x: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class MaterializedDispatch:
    """Owns one receive layout until its local expert computation completes."""

    batch_index: int
    batch: MoEMicrobatch
    recv_x: torch.Tensor
    handle: object


def build_microbatches(
    model: TinyMoE,
    *,
    count: int,
    tokens_per_rank: int,
    hidden: int,
    num_topk: int,
    rank: int,
    device: torch.device,
    seed: int,
) -> list[MoEMicrobatch]:
    """Builds deterministic router outputs outside the timed communication path."""
    batches: list[MoEMicrobatch] = []
    for index in range(count):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1000 + rank * 97 + index)
        x = torch.randn(
            (tokens_per_rank, hidden), generator=generator, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        topk_idx, topk_weights = model.route(x, num_topk)
        batches.append(MoEMicrobatch(x, topk_idx, topk_weights))
    return batches


def run_expert_mlp(
    model: TinyMoE,
    dispatch: MaterializedDispatch,
    *,
    rank: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Executes the correctness reference MLP on one materialized receive layout."""
    return run_local_experts(
        model,
        dispatch.recv_x,
        rank,
        num_local_experts,
        dispatch.handle.num_recv_tokens_per_expert_list,
        dispatch.handle.num_unaligned_recv_tokens_per_expert_list,
    )


def materialize_ticket(
    buffer: DLBBuffer, batch_index: int, batch: MoEMicrobatch
) -> MaterializedDispatch:
    """Posts and finishes a single bootstrap dispatch."""
    ticket = buffer.post_dispatch(
        batch.x, batch.topk_idx, batch.topk_weights, expert_alignment=8
    )
    recv_x, _, _, handle, event = ticket.finish()
    event.current_stream_wait()
    return MaterializedDispatch(batch_index, batch, recv_x, handle)


def run_sequential(
    buffer: DLBBuffer,
    model: TinyMoE,
    batches: list[MoEMicrobatch],
    *,
    warmup_steps: int,
    steps: int,
    rank: int,
) -> tuple[list[float], list[torch.Tensor], list[float]]:
    """Runs dispatch, expert MLP, and combine serially for each microbatch."""
    outputs: list[torch.Tensor] = []
    samples: list[float] = []
    expert_samples: list[float] = []
    for index, batch in enumerate(batches[: warmup_steps + steps]):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        ticket = buffer.post_dispatch(
            batch.x, batch.topk_idx, batch.topk_weights, expert_alignment=8
        )
        recv_x, _, _, handle, event = ticket.finish()
        event.current_stream_wait()
        expert_begin = torch.cuda.Event(enable_timing=True)
        expert_end = torch.cuda.Event(enable_timing=True)
        expert_begin.record(torch.cuda.current_stream())
        expert_output = run_local_experts(
            model,
            recv_x,
            rank,
            buffer.num_local_experts,
            handle.num_recv_tokens_per_expert_list,
            handle.num_unaligned_recv_tokens_per_expert_list,
        )
        expert_end.record(torch.cuda.current_stream())
        combined, _, combine_event = buffer.combine(expert_output, handle)
        combine_event.synchronize()
        elapsed_ms = max_across_ranks((time.perf_counter() - start) * 1000.0)
        if index >= warmup_steps:
            samples.append(elapsed_ms)
            expert_samples.append(max_across_ranks(expert_begin.elapsed_time(expert_end)))
            outputs.append(combined)
    return samples, outputs, expert_samples


def run_pipelined(
    buffer: DLBBuffer,
    model: TinyMoE,
    batches: list[MoEMicrobatch],
    reference_outputs: list[torch.Tensor],
    *,
    warmup_steps: int,
    steps: int,
    rank: int,
    atol: float,
    rtol: float,
) -> tuple[list[float], list[float]]:
    """Overlaps dispatch ``n + 1`` with the expert MLP for dispatch ``n``.

    The final extra microbatch is intentionally received but not evaluated. It
    supplies the transport that overlaps the final measured expert MLP, so no
    timed sample includes a pipeline drain bubble.
    """
    ready = materialize_ticket(buffer, 0, batches[0])
    samples: list[float] = []
    expert_samples: list[float] = []
    for next_index in range(1, warmup_steps + steps + 1):
        output_index = next_index - 1
        timed = output_index >= warmup_steps
        if timed:
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()

        # The current stream queues source-side planning and packing first.
        # Transport then proceeds on DLB's transport stream while the current
        # stream computes the previous microbatch's local experts.
        ticket = buffer.post_dispatch(
            batches[next_index].x,
            batches[next_index].topk_idx,
            batches[next_index].topk_weights,
            expert_alignment=8,
        )
        expert_begin = torch.cuda.Event(enable_timing=True)
        expert_end = torch.cuda.Event(enable_timing=True)
        expert_begin.record(torch.cuda.current_stream())
        expert_output = run_expert_mlp(
            model,
            ready,
            rank=rank,
            num_local_experts=buffer.num_local_experts,
        )
        expert_end.record(torch.cuda.current_stream())
        recv_x, _, _, next_handle, next_event = ticket.finish()
        next_event.current_stream_wait()
        next_ready = MaterializedDispatch(
            next_index, batches[next_index], recv_x, next_handle
        )

        combined, _, combine_event = buffer.combine(expert_output, ready.handle)
        combine_event.synchronize()
        if timed:
            elapsed_ms = max_across_ranks((time.perf_counter() - start) * 1000.0)
            samples.append(elapsed_ms)
            expert_samples.append(max_across_ranks(expert_begin.elapsed_time(expert_end)))
            expected = reference_outputs[output_index - warmup_steps]
            torch.testing.assert_close(combined.float(), expected.float(), atol=atol, rtol=rtol)
            reference = reference_forward(
                model,
                ready.batch.x,
                ready.batch.topk_idx,
                ready.batch.topk_weights,
                model.router.out_features,
            )
            torch.testing.assert_close(combined.float(), reference.float(), atol=atol, rtol=rtol)
        ready = next_ready
    return samples, expert_samples


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    """Runs the distributed sequential-versus-overlap comparison."""
    dist.init_process_group(backend="cpu:gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise RuntimeError("the DLB overlap benchmark requires exactly eight ranks")
    if args.num_experts != 256:
        raise ValueError("the DLB overlap benchmark fixes the EP8 model at 256 experts")
    global_tokens = args.global_batch_size * args.sequence_length
    if global_tokens % world_size != 0:
        raise ValueError("global tokens must be divisible by the EP world size")
    if args.warmup_steps < 0 or args.steps <= 0:
        raise ValueError("warmup steps must be non-negative and steps must be positive")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tokens_per_rank = global_tokens // world_size
    model = TinyMoE(args.hidden, args.ffn_hidden, args.num_experts, device, args.seed).eval()
    buffer = DLBBuffer(
        dist.group.WORLD,
        gpus_per_server=4,
        num_max_tokens_per_rank=tokens_per_rank,
        hidden=args.hidden,
        num_experts=args.num_experts,
        num_topk=args.num_topk,
        dtype=torch.bfloat16,
        pipeline_depth=2,
        chunk_bytes=args.chunk_bytes,
        num_comm_sms=args.num_comm_sms,
        transport_backend=args.transport_backend,
    )
    # Pipeline execution needs one additional microbatch to avoid timing a
    # terminal drain. The sequential path intentionally uses the same measured
    # microbatches as the pipelined path.
    batches = build_microbatches(
        model,
        count=args.warmup_steps + args.steps + 1,
        tokens_per_rank=tokens_per_rank,
        hidden=args.hidden,
        num_topk=args.num_topk,
        rank=rank,
        device=device,
        seed=args.seed,
    )
    torch.cuda.synchronize()

    # Precondition CUDA kernels, the allocator, and persistent DLB workers
    # before collecting either mode. The measured sequential phase therefore
    # does not pay the one-time cold-start progression of the data plane.
    precondition_steps = min(len(batches), max(args.warmup_steps, args.steps))
    _, precondition_outputs, _ = run_sequential(
        buffer,
        model,
        batches,
        warmup_steps=0,
        steps=precondition_steps,
        rank=rank,
    )
    del precondition_outputs

    sequential_samples, sequential_outputs, sequential_expert_samples = run_sequential(
        buffer,
        model,
        batches,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        rank=rank,
    )
    pipelined_samples, pipelined_expert_samples = run_pipelined(
        buffer,
        model,
        batches,
        sequential_outputs,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        rank=rank,
        atol=args.atol,
        rtol=args.rtol,
    )
    if (
        len(sequential_samples) != args.steps
        or len(pipelined_samples) != args.steps
        or len(pipelined_expert_samples) != args.steps
    ):
        raise AssertionError("benchmark did not collect the requested measured microbatches")

    sequential_p50 = percentile(sequential_samples, 0.50)
    pipelined_p50 = percentile(pipelined_samples, 0.50)
    result = {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "topology": {"ep_size": 8, "logical_servers": 2, "gpus_per_server": 4},
        "model": {
            "global_batch_size": args.global_batch_size,
            "sequence_length": args.sequence_length,
            "global_tokens": global_tokens,
            "tokens_per_rank": tokens_per_rank,
            "hidden": args.hidden,
            "ffn_hidden": args.ffn_hidden,
            "num_experts": args.num_experts,
            "topk": args.num_topk,
            "num_comm_sms": args.num_comm_sms,
            "transport_backend": args.transport_backend,
        },
        "precondition_microbatches": precondition_steps,
        "sequential": {
            "samples_ms": sequential_samples,
            "expert_kernel_samples_ms": sequential_expert_samples,
            "expert_kernel_p50_ms": percentile(sequential_expert_samples, 0.50),
            "p50_ms": sequential_p50,
            "p90_ms": percentile(sequential_samples, 0.90),
            "global_tokens_per_second": global_tokens / sequential_p50 * 1000.0,
        },
        "pipelined": {
            "samples_ms": pipelined_samples,
            "expert_kernel_samples_ms": pipelined_expert_samples,
            "expert_kernel_p50_ms": percentile(pipelined_expert_samples, 0.50),
            "p50_ms": pipelined_p50,
            "p90_ms": percentile(pipelined_samples, 0.90),
            "global_tokens_per_second": global_tokens / pipelined_p50 * 1000.0,
        },
        "p50_speedup": sequential_p50 / pipelined_p50,
    }
    if rank == 0:
        print("DLB MoE overlap benchmark passed")
        print(f"sequential P50: {sequential_p50:.3f} ms")
        print(f"pipelined P50: {pipelined_p50:.3f} ms")
        print(f"P50 speedup: {result['p50_speedup']:.3f}x")
        if args.result_json:
            path = Path(args.result_json).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    buffer.close()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    """Parses the fixed-shape EP8 MoE benchmark configuration."""
    parser = argparse.ArgumentParser(description="DLB split-phase MoE overlap benchmark")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn-hidden", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-comm-sms", type=int, default=24)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--transport-backend", choices=("nvshmem", "loopback"), default="loopback")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--atol", type=float, default=0.125)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--result-json", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
