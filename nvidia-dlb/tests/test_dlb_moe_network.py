"""EP8 forward test for a small neural-network MoE backed by DLB.

The distributed path owns 32 of 256 experts per rank.  A deterministic full
expert bank is retained only as a test oracle, allowing every source rank to
compare DLB's result with direct local execution of exactly the same experts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


NVIDIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NVIDIA_ROOT))

from dlb import DLBBuffer  # noqa: E402


class TinyMoE(nn.Module):
    """Deterministic router and expert MLP bank used by the integration test."""

    def __init__(
        self,
        hidden: int,
        ffn_hidden: int,
        num_experts: int,
        device: torch.device,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        self.router = nn.Linear(hidden, num_experts, bias=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            router_weight = torch.randn(
                (num_experts, hidden), generator=generator, dtype=torch.float32
            ) / math.sqrt(hidden)
            router_bias = torch.randn(
                (num_experts,), generator=generator, dtype=torch.float32
            ) * 0.01
            self.router.weight.copy_(router_weight.to(device))
            self.router.bias.copy_(router_bias.to(device))

        weight_1 = torch.randn(
            (num_experts, hidden, ffn_hidden), generator=generator, dtype=torch.float32
        ) / math.sqrt(hidden)
        bias_1 = torch.randn(
            (num_experts, ffn_hidden), generator=generator, dtype=torch.float32
        ) * 0.01
        weight_2 = torch.randn(
            (num_experts, ffn_hidden, hidden), generator=generator, dtype=torch.float32
        ) / math.sqrt(ffn_hidden)
        bias_2 = torch.randn(
            (num_experts, hidden), generator=generator, dtype=torch.float32
        ) * 0.01
        self.expert_weight_1 = nn.Parameter(weight_1.to(device=device, dtype=torch.bfloat16))
        self.expert_bias_1 = nn.Parameter(bias_1.to(device=device, dtype=torch.bfloat16))
        self.expert_weight_2 = nn.Parameter(weight_2.to(device=device, dtype=torch.bfloat16))
        self.expert_bias_2 = nn.Parameter(bias_2.to(device=device, dtype=torch.bfloat16))
        self.requires_grad_(False)

    def route(self, x: torch.Tensor, num_topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x.float())
        topk_logits, topk_idx = torch.topk(logits, k=num_topk, dim=-1, sorted=True)
        topk_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
        return topk_idx.contiguous(), topk_weights.contiguous()

    def run_expert(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        hidden = torch.matmul(x, self.expert_weight_1[expert])
        hidden = F.silu(hidden + self.expert_bias_1[expert])
        return torch.matmul(hidden, self.expert_weight_2[expert]) + self.expert_bias_2[expert]


def run_local_experts(
    model: TinyMoE,
    recv_x: torch.Tensor,
    rank: int,
    num_local_experts: int,
    aligned_counts: list[int],
    actual_counts: list[int],
) -> torch.Tensor:
    output = torch.zeros_like(recv_x)
    offset = 0
    for local_expert, (aligned, actual) in enumerate(zip(aligned_counts, actual_counts)):
        if actual:
            global_expert = rank * num_local_experts + local_expert
            output[offset : offset + actual] = model.run_expert(
                recv_x[offset : offset + actual], global_expert
            )
        offset += aligned
    if offset != recv_x.shape[0]:
        raise AssertionError("expert segment sizes do not cover the dispatched tensor")
    return output


def reference_forward(
    model: TinyMoE,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Execute the selected experts directly without any communication."""
    num_tokens, num_topk = topk_idx.shape
    contributions = torch.zeros(
        (num_tokens, num_topk, x.shape[1]), dtype=x.dtype, device=x.device
    )
    for expert in range(num_experts):
        positions = (topk_idx == expert).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token_idx = positions[:, 0]
        slot_idx = positions[:, 1]
        expert_output = model.run_expert(x.index_select(0, token_idx), expert)
        weighted = (
            expert_output.float() * topk_weights[token_idx, slot_idx, None]
        ).to(x.dtype)
        contributions[token_idx, slot_idx] = weighted

    result = torch.zeros_like(x)
    for slot in range(num_topk):
        result.add_(contributions[:, slot])
    return result


def max_across_ranks(value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def sum_across_ranks(value: int) -> int:
    tensor = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def controlled_route(
    *,
    mode: str,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    rank: int,
    world_size: int,
    gpus_per_server: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic all-intra-server or all-cross-server routes."""
    if mode not in ("intra-server", "cross-server"):
        raise ValueError(f"unsupported controlled routing mode: {mode}")
    server_count = world_size // gpus_per_server
    if server_count != 2:
        raise ValueError("the controlled EP8 routes require exactly two servers")
    num_local_experts = num_experts // world_size
    source_server = rank // gpus_per_server
    source_local = rank % gpus_per_server

    token = torch.arange(num_tokens, device=device, dtype=torch.int64)[:, None]
    slot = torch.arange(num_topk, device=device, dtype=torch.int64)[None, :]
    ordinal = token * num_topk + slot
    if mode == "intra-server":
        # Keep every route on the source server but exclude the source GPU,
        # so the measured dispatch payload always crosses local NVLink/P2P.
        destination_server = source_server
        destination_local = (
            source_local + 1 + ordinal.remainder(gpus_per_server - 1)
        ).remainder(gpus_per_server)
    else:
        destination_server = 1 - source_server
        destination_local = ordinal.remainder(gpus_per_server)

    destination_rank = destination_server * gpus_per_server + destination_local
    local_expert = ordinal.remainder(num_local_experts)
    topk_idx = (destination_rank * num_local_experts + local_expert).contiguous()
    topk_weights = torch.full(
        (num_tokens, num_topk),
        1.0 / num_topk,
        dtype=torch.float32,
        device=device,
    )
    return topk_idx, topk_weights


def percentile(samples: list[float], fraction: float) -> float:
    if not samples or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile requires samples and a fraction in [0, 1]")
    ordered = sorted(samples)
    # Nearest-rank percentile. In particular, P90 of three measurements is
    # the largest sample rather than the median.
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    dist.init_process_group(backend="cpu:gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise RuntimeError("the DLB neural-network test requires exactly eight ranks")
    if args.num_experts != 256:
        raise ValueError("this regression test intentionally fixes the EP8 model at 256 experts")
    if args.global_batch_size <= 0 or args.sequence_length <= 0:
        raise ValueError("global batch size and sequence length must be positive")
    if args.steps <= 0 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup steps cannot be negative")
    global_tokens = args.global_batch_size * args.sequence_length
    if global_tokens % world_size != 0:
        raise ValueError("global tokens must be divisible by the EP world size")
    tokens_per_rank = global_tokens // world_size

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.reset_peak_memory_stats(device)
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
        enable_profiling=args.profile_stages,
        transport_backend=args.transport_backend,
    )

    step_results: list[dict[str, object]] = []
    last_rail_traffic = None
    last_channel_traffic = None
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
        input_generator = torch.Generator(device="cpu")
        input_generator.manual_seed(args.seed + 1000 + rank * 97 + step)
        x = torch.randn(
            (tokens_per_rank, args.hidden), generator=input_generator, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        if args.routing_mode == "model":
            topk_idx, topk_weights = model.route(x, args.num_topk)
        else:
            topk_idx, topk_weights = controlled_route(
                mode=args.routing_mode,
                num_tokens=tokens_per_rank,
                num_topk=args.num_topk,
                num_experts=args.num_experts,
                rank=rank,
                world_size=world_size,
                gpus_per_server=buffer.gpus_per_server,
                device=device,
            )

        # One dispatch wire record carries a token's hidden state plus all of
        # its expert selections for one destination rank. Count those records
        # outside the timed region so Top-K routes and transferred vectors are
        # not conflated in the benchmark report.
        destination_ranks = torch.sort(
            topk_idx.div(buffer.num_local_experts, rounding_mode="floor"), dim=1
        ).values
        local_wire_records = int(
            (
                1
                + (destination_ranks[:, 1:] != destination_ranks[:, :-1]).sum(dim=1)
            ).sum().item()
        )
        aggregate_wire_records = sum_across_ranks(local_wire_records)

        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        torch.cuda.nvtx.range_push(
            f"DLB.dispatch.{args.routing_mode}.rank{rank}.step{step}"
        )
        try:
            recv_x, _, _, handle, dispatch_event = buffer.dispatch(
                x, topk_idx, topk_weights, expert_alignment=args.expert_alignment
            )
            dispatch_event.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        dispatch_ms = (time.perf_counter() - started) * 1000.0
        dispatch_profile = None
        if args.profile_stages:
            local_profile = buffer.get_last_dispatch_profile()
            dispatch_profile = {
                name: max_across_ranks(value)
                for name, value in local_profile.items()
            }
        last_rail_traffic = handle.dispatch_rail_traffic_records
        last_channel_traffic = handle.dispatch_channel_traffic_records
        aggregate_received_groups = sum_across_ranks(
            int(handle.dispatch_group_output_indices.shape[0])
        )
        if aggregate_received_groups != aggregate_wire_records:
            raise AssertionError(
                "dispatch group conservation failed: "
                f"sent={aggregate_wire_records}, received={aggregate_received_groups}"
            )

        started = time.perf_counter()
        expert_output = run_local_experts(
            model,
            recv_x,
            rank,
            buffer.num_local_experts,
            handle.num_recv_tokens_per_expert_list,
            handle.num_unaligned_recv_tokens_per_expert_list,
        )
        torch.cuda.synchronize(device)
        expert_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        torch.cuda.nvtx.range_push(
            f"DLB.combine.{args.routing_mode}.rank{rank}.step{step}"
        )
        try:
            combined_x, combined_weights, combine_event = buffer.combine(expert_output, handle)
            combine_event.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        combine_ms = (time.perf_counter() - started) * 1000.0
        combine_profile = None
        if args.profile_stages:
            local_profile = buffer.get_last_combine_profile()
            combine_profile = {
                name: max_across_ranks(value)
                for name, value in local_profile.items()
            }

        reference_x = reference_forward(
            model, x, topk_idx, topk_weights, args.num_experts
        )
        difference = (combined_x.float() - reference_x.float()).abs()
        local_max_error = float(difference.max().item())
        local_mean_error = float(difference.mean().item())
        global_max_error = max_across_ranks(local_max_error)
        global_mean_error = max_across_ranks(local_mean_error)
        torch.testing.assert_close(
            combined_x.float(), reference_x.float(), atol=args.atol, rtol=args.rtol
        )
        if not torch.equal(combined_weights, topk_weights):
            raise AssertionError("combine did not restore the router weights exactly")
        if any(count % args.expert_alignment for count in handle.num_recv_tokens_per_expert_list):
            raise AssertionError("an expert segment is not aligned")

        max_dispatch_ms = max_across_ranks(dispatch_ms)
        max_expert_ms = max_across_ranks(expert_ms)
        max_combine_ms = max_across_ranks(combine_ms)
        if step >= args.warmup_steps:
            step_results.append(
                {
                    "step": step - args.warmup_steps,
                    "global_max_abs_error": global_max_error,
                    "global_max_mean_abs_error": global_mean_error,
                    "max_dispatch_ms": max_dispatch_ms,
                    "max_expert_ms": max_expert_ms,
                    "max_combine_ms": max_combine_ms,
                    "aggregate_dispatch_wire_records": aggregate_wire_records,
                    "aggregate_combine_wire_records": aggregate_received_groups,
                    **(
                        {"dispatch_stage_max_ms": dispatch_profile}
                        if dispatch_profile is not None
                        else {}
                    ),
                    **(
                        {"combine_stage_max_ms": combine_profile}
                        if combine_profile is not None
                        else {}
                    ),
                }
            )

    assert last_rail_traffic is not None
    assert last_channel_traffic is not None
    gathered_traffic = [torch.empty_like(last_rail_traffic) for _ in range(world_size)]
    dist.all_gather(gathered_traffic, last_rail_traffic)
    server_traffic: list[list[int]] = []
    for server in range(2):
        reference = gathered_traffic[server * 4]
        for source_rank in range(server * 4, (server + 1) * 4):
            if not torch.equal(gathered_traffic[source_rank], reference):
                raise AssertionError("source ranks in one server materialized different Rail plans")
        if int(reference.max() - reference.min()) > 1:
            raise AssertionError(f"server {server} Rail traffic is imbalanced: {reference.tolist()}")
        server_traffic.append([int(value) for value in reference.tolist()])

    gathered_channels = [torch.empty_like(last_channel_traffic) for _ in range(world_size)]
    dist.all_gather(gathered_channels, last_channel_traffic)
    channel_traffic = [
        [int(value) for value in per_rank.tolist()]
        for per_rank in gathered_channels
    ]
    for rank_index, per_rank in enumerate(channel_traffic):
        local_rail = rank_index % buffer.gpus_per_server
        if sum(per_rank) != server_traffic[rank_index // buffer.gpus_per_server][local_rail]:
            raise AssertionError(
                f"rank {rank_index} channel traffic does not conserve its Rail records"
            )

    peak_memory = max_across_ranks(float(torch.cuda.max_memory_allocated(device)))
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    dispatch_samples = [float(step["max_dispatch_ms"]) for step in step_results]
    combine_samples = [float(step["max_combine_ms"]) for step in step_results]
    expert_routes_per_rank = tokens_per_rank * args.num_topk
    semantic_hidden_bytes_per_rank = (
        expert_routes_per_rank * args.hidden * torch.bfloat16.itemsize
    )
    aggregate_semantic_hidden_bytes = semantic_hidden_bytes_per_rank * world_size
    aggregate_dispatch_wire_records = int(statistics.median(
        [int(step["aggregate_dispatch_wire_records"]) for step in step_results]
    ))
    dispatch_wire_records_per_rank = aggregate_dispatch_wire_records / world_size
    dispatch_wire_bytes_per_rank = (
        dispatch_wire_records_per_rank * buffer.size_hint.record_bytes
    )
    aggregate_dispatch_wire_bytes = (
        aggregate_dispatch_wire_records * buffer.size_hint.record_bytes
    )
    aggregate_combine_wire_records = int(statistics.median(
        [int(step["aggregate_combine_wire_records"]) for step in step_results]
    ))
    combine_wire_records_per_rank = aggregate_combine_wire_records / world_size
    combine_wire_bytes_per_rank = (
        combine_wire_records_per_rank * buffer.size_hint.record_bytes
    )
    aggregate_combine_wire_bytes = (
        aggregate_combine_wire_records * buffer.size_hint.record_bytes
    )
    dispatch_p50_ms = percentile(dispatch_samples, 0.50)
    combine_p50_ms = percentile(combine_samples, 0.50)
    performance = {
        "record_bytes": buffer.size_hint.record_bytes,
        "expert_routes_per_rank": expert_routes_per_rank,
        "semantic_hidden_bytes_per_rank": semantic_hidden_bytes_per_rank,
        "aggregate_semantic_hidden_bytes": aggregate_semantic_hidden_bytes,
        "dispatch_wire_records_per_rank_average": dispatch_wire_records_per_rank,
        "aggregate_dispatch_wire_records": aggregate_dispatch_wire_records,
        "dispatch_wire_bytes_per_rank_average": dispatch_wire_bytes_per_rank,
        "aggregate_dispatch_wire_bytes": aggregate_dispatch_wire_bytes,
        "combine_wire_records_per_rank": combine_wire_records_per_rank,
        "aggregate_combine_wire_records": aggregate_combine_wire_records,
        "combine_wire_bytes_per_rank": combine_wire_bytes_per_rank,
        "aggregate_combine_wire_bytes": aggregate_combine_wire_bytes,
        "dispatch_p50_ms": dispatch_p50_ms,
        "dispatch_p90_ms": percentile(dispatch_samples, 0.90),
        "combine_p50_ms": combine_p50_ms,
        "combine_p90_ms": percentile(combine_samples, 0.90),
        "dispatch_wire_per_rank_gbps": dispatch_wire_bytes_per_rank / dispatch_p50_ms / 1e6,
        "dispatch_wire_aggregate_gbps": aggregate_dispatch_wire_bytes / dispatch_p50_ms / 1e6,
        "dispatch_semantic_hidden_per_rank_gbps": semantic_hidden_bytes_per_rank / dispatch_p50_ms / 1e6,
        "dispatch_semantic_hidden_aggregate_gbps": aggregate_semantic_hidden_bytes / dispatch_p50_ms / 1e6,
        "combine_wire_per_rank_gbps": combine_wire_bytes_per_rank / combine_p50_ms / 1e6,
        "combine_wire_aggregate_gbps": aggregate_combine_wire_bytes / combine_p50_ms / 1e6,
    }
    if args.profile_stages:
        profile_names = step_results[0]["dispatch_stage_max_ms"].keys()
        performance["dispatch_stage_p50_ms"] = {
            name: percentile(
                [float(step["dispatch_stage_max_ms"][name]) for step in step_results],
                0.50,
            )
            for name in profile_names
        }
        combine_profile_names = step_results[0]["combine_stage_max_ms"].keys()
        performance["combine_stage_p50_ms"] = {
            name: percentile(
                [float(step["combine_stage_max_ms"][name]) for step in step_results],
                0.50,
            )
            for name in combine_profile_names
        }
    result = {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "topology": {"ep_size": 8, "logical_servers": 2, "gpus_per_server": 4},
        "model": {
            "routing_mode": args.routing_mode,
            "transport_backend": args.transport_backend,
            "global_batch_size": args.global_batch_size,
            "sequence_length": args.sequence_length,
            "global_tokens": global_tokens,
            "tokens_per_rank": tokens_per_rank,
            "hidden": args.hidden,
            "ffn_hidden": args.ffn_hidden,
            "num_experts": args.num_experts,
            "experts_per_rank": args.num_experts // world_size,
            "topk": args.num_topk,
            "chunk_bytes": args.chunk_bytes,
            "num_comm_sms": args.num_comm_sms,
            "num_rail_channels": args.num_comm_sms // 2,
            "dtype": "bfloat16",
            "parameter_bytes_per_rank_with_oracle": parameter_bytes,
        },
        "steps": step_results,
        "performance": performance,
        "last_dispatch_rail_records": server_traffic,
        "last_dispatch_channel_records_per_rank": channel_traffic,
        "max_peak_torch_cuda_bytes": int(peak_memory),
    }

    dist.barrier()
    if rank == 0:
        print("DLB tiny MoE network passed")
        print(
            f"EP8, experts=256 (32/rank), hidden={args.hidden}, "
            f"ffn_hidden={args.ffn_hidden}, topk={args.num_topk}"
        )
        print(f"routing mode: {args.routing_mode}")
        print(f"transport backend: {args.transport_backend}")
        print(
            f"global batch={args.global_batch_size}, sequence={args.sequence_length}, "
            f"global tokens={global_tokens}, tokens/rank={tokens_per_rank}"
        )
        print(f"last dispatch Rail records: {server_traffic}")
        print(f"last dispatch channel records per rank: {channel_traffic}")
        print(
            f"dispatch records: expert routes/rank={expert_routes_per_rank}, "
            f"rank-deduplicated wire/rank(avg)={dispatch_wire_records_per_rank:.1f}"
        )
        print(
            "combine records: "
            f"rank-deduplicated wire/rank(avg)={combine_wire_records_per_rank:.1f}"
        )
        print(
            f"dispatch P50={dispatch_p50_ms:.3f} ms, "
            f"wire={performance['dispatch_wire_per_rank_gbps']:.2f} GB/s/rank, "
            f"wire aggregate={performance['dispatch_wire_aggregate_gbps']:.2f} GB/s, "
            f"semantic hidden={performance['dispatch_semantic_hidden_aggregate_gbps']:.2f} GB/s"
        )
        print(
            f"combine P50={combine_p50_ms:.3f} ms, "
            f"wire={performance['combine_wire_per_rank_gbps']:.2f} GB/s/rank, "
            f"wire aggregate={performance['combine_wire_aggregate_gbps']:.2f} GB/s"
        )
        if args.profile_stages:
            print("dispatch stage P50 (max rank):")
            for name, value in performance["dispatch_stage_p50_ms"].items():
                print(f"  {name}: {value:.3f} ms")
            print("combine stage P50 (max rank):")
            for name, value in performance["combine_stage_p50_ms"].items():
                print(f"  {name}: {value:.3f} ms")
        print(f"global max abs error: {max(x['global_max_abs_error'] for x in step_results):.6f}")
        print(f"max peak PyTorch CUDA memory: {int(peak_memory) / 1024**2:.2f} MiB")
        if args.result_json:
            result_path = Path(args.result_json).resolve()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    buffer.close()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DLB EP8 tiny MoE neural-network test")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn-hidden", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--expert-alignment", type=int, default=8)
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=0,
        help="transport chunk bytes; 0 selects 4 MiB for loopback and 64 KiB for NVSHMEM",
    )
    parser.add_argument(
        "--num-comm-sms",
        type=int,
        default=24,
        help="even sender+receiver CTA (approximate SM) budget; half is the channel count",
    )
    parser.add_argument(
        "--routing-mode",
        choices=("model", "intra-server", "cross-server"),
        default="model",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--atol", type=float, default=0.125)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--result-json", type=str, default="")
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="record detailed CUDA-event timings for dispatch and combine",
    )
    parser.add_argument(
        "--transport-backend",
        choices=("nvshmem", "loopback"),
        default="loopback",
        help="real NVSHMEM/RDMA or same-host NVLink Rail emulation",
    )
    args = parser.parse_args()
    if args.chunk_bytes == 0:
        args.chunk_bytes = (
            4 * 1024 * 1024
            if args.transport_backend == "loopback"
            else 64 * 1024
        )
    return args


if __name__ == "__main__":
    run(parse_args())
