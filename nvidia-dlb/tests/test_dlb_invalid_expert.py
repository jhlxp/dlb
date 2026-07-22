"""Verify that an out-of-range router expert fails instead of dropping a record."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dlb import DLBBuffer  # noqa: E402


@torch.no_grad()
def run() -> None:
    dist.init_process_group("cpu:gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError("the DLB invalid-expert test requires exactly eight ranks")

    torch.cuda.set_device(local_rank)
    hidden = 64
    num_experts = 16
    num_topk = 2
    tokens = 8
    buffer = DLBBuffer(
        dist.group.WORLD,
        gpus_per_server=4,
        num_max_tokens_per_rank=tokens,
        hidden=hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        dtype=torch.bfloat16,
    )

    x = torch.randn(tokens, hidden, device=local_rank, dtype=torch.bfloat16)
    topk_idx = torch.zeros(tokens, num_topk, device=local_rank, dtype=torch.int64)
    topk_weights = torch.full(
        (tokens, num_topk), 0.5, device=local_rank, dtype=torch.float32
    )
    if rank == 0:
        topk_idx[0, 0] = num_experts

    failed_as_expected = 0
    try:
        _, _, _, _, event = buffer.dispatch(x, topk_idx, topk_weights)
        event.synchronize()
    except RuntimeError as error:
        if "expert IDs outside" not in str(error):
            raise
        failed_as_expected = 1

    failures = torch.tensor([failed_as_expected], dtype=torch.int64)
    dist.all_reduce(failures)
    if int(failures.item()) != 1:
        raise AssertionError(
            f"expected exactly the rank with the invalid route to fail, got {failures.item()}"
        )

    dist.barrier()
    if rank == 0:
        print("DLB invalid expert fail-fast passed")
    buffer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
