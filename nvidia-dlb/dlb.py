"""MoE-facing DLB dispatch/combine interface.

This module deliberately keeps the source-local DLB scheduler and NVSHMEM
transport behind a tensor API.  The transport record is an internal ABI; model
code supplies hidden states and router decisions and receives expert-grouped
tensors plus a handle for the reverse combine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from dlb_nvshmem_utils import get_nvshmem_init_id, nvshmem_comm_t


_PAYLOAD_OFFSET = 128
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_DISPATCH_PROFILE_NAMES = (
    "prepare",
    "stage_publish",
    "producer_wait",
    "staging_copy",
    "rail_transport",
    "receive_repair",
    "completion_wait",
    "count_readback",
    "materialize_scatter",
    "end_to_end",
)
_COMBINE_PROFILE_NAMES = (
    "prepare",
    "stage_publish",
    "producer_wait",
    "staging_copy",
    "rail_transport",
    "receive_repair",
    "completion_wait",
    "post_transport",
    "accumulate_cast",
    "end_to_end",
)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class DLBBufferSizeHint:
    """Per-rank capacities derived from a bounded MoE shape."""

    record_bytes: int
    receive_slot_bytes: int
    repair_slot_bytes: int
    rail_send_capacity_bytes: int
    estimated_runtime_bytes: int


class EventOverlap:
    """CUDA event returned by DLB communication operations."""

    def __init__(self, event: torch.cuda.Event, tensors: tuple[torch.Tensor, ...] = ()) -> None:
        self.event: Optional[torch.cuda.Event] = event
        self._tensors = tensors

    def current_stream_wait(self, release_handle: bool = False) -> None:
        if self.event is None:
            return
        torch.cuda.current_stream().wait_event(self.event)
        if release_handle:
            self.event = None
            self._tensors = ()

    def synchronize(self) -> None:
        if self.event is not None:
            self.event.synchronize()

    def __enter__(self) -> "EventOverlap":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.current_stream_wait()


@dataclass
class DLBHandle:
    """Routing state produced by :meth:`DLBDispatchTicket.finish`.

    The received metadata is kept in the same expert-expanded, padded order as
    ``recv_x``.  ``combine`` uses it to return every expert result to the
    original rank/token/top-k slot.
    """

    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    num_tokens: int
    num_experts: int
    expert_alignment: int
    num_recv_tokens: int
    num_recv_tokens_per_expert_list: list[int]
    num_unaligned_recv_tokens_per_expert_list: list[int]
    recv_headers: torch.Tensor
    recv_source_rank: torch.Tensor
    recv_source_token: torch.Tensor
    recv_expert_idx: torch.Tensor
    recv_topk_slot: torch.Tensor
    recv_topk_weights: torch.Tensor
    valid_mask: torch.Tensor
    dispatch_group_output_indices: torch.Tensor
    dispatch_rail_traffic_records: torch.Tensor
    dispatch_channel_traffic_records: torch.Tensor


@dataclass
class DLBDispatchTicket:
    """Owns one DLB dispatch that has been posted but not materialized.

    A posted dispatch has already planned, packed, and launched transport on
    the communication streams.  Its receive layout is unavailable until
    :meth:`finish` waits for transport and materializes the expert-major
    tensors.  The ticket keeps routing inputs alive for all asynchronous GPU
    work issued during the post phase.
    """

    _buffer: "DLBBuffer"
    _x: torch.Tensor
    _topk_idx: torch.Tensor
    _topk_weights: torch.Tensor
    _epoch: int
    _expert_alignment: int
    _finished: bool = False

    def finish(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DLBHandle, EventOverlap]:
        """Waits for transport and returns the normal DLB dispatch result."""
        return self._buffer._finish_posted_dispatch(self)


class DLBBuffer:
    """Reusable MoE communication buffer backed by the DLB NVSHMEM runtime.

    The public receive layout is expert-expanded and grouped by local expert.
    On the dispatch wire, however, all selections of one token that target the
    same rank share one hidden-state record; the destination expands that group
    back into expert rows.  The public contract is independent of whether the
    runtime uses local NVLink, same-host loopback emulation, or inter-node RDMA.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        gpus_per_server: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        num_topk: int,
        dtype: torch.dtype = torch.bfloat16,
        pipeline_depth: int = 2,
        chunk_bytes: int = 64 * 1024,
        num_comm_sms: int = 24,
        enable_profiling: bool = False,
        transport_backend: str = "nvshmem",
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before DLBBuffer")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported DLB payload dtype: {dtype}")

        self.group = group
        self.group_ranks = dist.get_process_group_ranks(group)
        self.global_rank = dist.get_rank()
        if self.global_rank not in self.group_ranks:
            raise RuntimeError("the current process is not a member of the DLB process group")
        self.rank = self.group_ranks.index(self.global_rank)
        self.world_size = len(self.group_ranks)
        self.gpus_per_server = int(gpus_per_server)
        if self.gpus_per_server <= 0 or self.world_size % self.gpus_per_server != 0:
            raise ValueError("DLB requires world_size == server_count * gpus_per_server")
        self.server_count = self.world_size // self.gpus_per_server
        if self.server_count < 2:
            raise ValueError("DLB Rail scheduling requires at least two logical servers")
        if num_max_tokens_per_rank <= 0 or hidden <= 0 or num_topk <= 0:
            raise ValueError("token, hidden, and top-k bounds must be positive")
        if num_topk > 8:
            raise ValueError("the grouped dispatch record currently supports top-k <= 8")
        if num_experts <= 0 or num_experts % self.world_size != 0:
            raise ValueError("num_experts must be positive and divisible by the EP world size")
        if pipeline_depth <= 0 or chunk_bytes <= 0:
            raise ValueError("pipeline_depth and chunk_bytes must be positive")
        if num_comm_sms <= 0 or num_comm_sms % 2 != 0:
            raise ValueError("num_comm_sms must be a positive even number")
        if transport_backend not in ("nvshmem", "loopback"):
            raise ValueError("transport_backend must be 'nvshmem' or 'loopback'")

        self.server_rank = self.rank // self.gpus_per_server
        self.local_rank = self.rank % self.gpus_per_server
        self.num_max_tokens_per_rank = int(num_max_tokens_per_rank)
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.num_topk = int(num_topk)
        self.num_local_experts = self.num_experts // self.world_size
        self.dtype = dtype
        self.pipeline_depth = int(pipeline_depth)
        self.chunk_bytes = int(chunk_bytes)
        self.num_comm_sms = int(num_comm_sms)
        self.num_rail_channels = self.num_comm_sms // 2
        self.enable_profiling = bool(enable_profiling)
        self.transport_backend = transport_backend
        self.device = torch.cuda.current_device()
        self.closed = False
        self._epoch = 0
        self._round_id = 0
        self._pending_dispatch: Optional[DLBDispatchTicket] = None
        self.size_hint = self.get_buffer_size_hint(
            self.world_size,
            self.gpus_per_server,
            self.num_max_tokens_per_rank,
            self.hidden,
            self.num_topk,
            self.dtype,
            self.pipeline_depth,
        )

        if self.rank == 0:
            uid = get_nvshmem_init_id()
        else:
            # NVSHMEM's unique-id ABI is 128 bytes in the linked runtime.
            uid = torch.zeros(128, dtype=torch.uint8)
        dist.broadcast(uid, src=self.group_ranks[0], group=self.group)

        self._comm = nvshmem_comm_t(
            self.rank,
            self.gpus_per_server,
            self.world_size,
            self.device,
            uid,
            self.size_hint.record_bytes,
            self.size_hint.receive_slot_bytes,
            self.size_hint.repair_slot_bytes,
            self.pipeline_depth,
            self.chunk_bytes,
            self.size_hint.rail_send_capacity_bytes,
            self.num_comm_sms,
            self.enable_profiling,
            self.transport_backend == "loopback",
        )

    @staticmethod
    def get_buffer_size_hint(
        world_size: int,
        gpus_per_server: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        dtype: torch.dtype = torch.bfloat16,
        pipeline_depth: int = 2,
    ) -> DLBBufferSizeHint:
        """Derive fixed-capacity transport buffers from the MoE shape bound."""
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported DLB payload dtype: {dtype}")
        if world_size <= 0 or gpus_per_server <= 0 or world_size % gpus_per_server != 0:
            raise ValueError("invalid SxM topology")
        server_count = world_size // gpus_per_server
        if server_count < 2:
            raise ValueError("DLB Rail scheduling requires at least two logical servers")
        if num_max_tokens_per_rank <= 0 or hidden <= 0 or num_topk <= 0 or pipeline_depth <= 0:
            raise ValueError("buffer shape bounds and pipeline_depth must be positive")
        if num_topk > 8:
            raise ValueError("the grouped dispatch record currently supports top-k <= 8")
        record_bytes = _align(_PAYLOAD_OFFSET + hidden * dtype.itemsize, 16)
        # DLB balances each source-server/destination-server tile over M Rails.
        # One Rail therefore needs at most one rank's bounded Top-K volume.
        slot_records = num_max_tokens_per_rank * num_topk
        slot_bytes = _align(slot_records * record_bytes, 16)
        rail_send_capacity = max(slot_bytes, (server_count - 1) * slot_bytes)
        # Approximate the dominant allocations owned by DlbRuntime.  Descriptor
        # and signal arrays are intentionally omitted from the byte-level hint.
        estimated = pipeline_depth * (
            world_size * slot_bytes
            + rail_send_capacity
            + server_count * slot_bytes
        )
        return DLBBufferSizeHint(
            record_bytes=record_bytes,
            receive_slot_bytes=slot_bytes,
            repair_slot_bytes=slot_bytes,
            rail_send_capacity_bytes=rail_send_capacity,
            estimated_runtime_bytes=estimated,
        )

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("DLBBuffer is closed")

    def _prepare_dispatch_inputs(
        self,
        x: torch.Tensor,
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        handle: Optional[DLBHandle],
        expert_alignment: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validates and normalizes one public MoE routing request."""
        if x.dim() != 2 or x.shape[1] != self.hidden or x.shape[0] > self.num_max_tokens_per_rank:
            raise ValueError(
                f"x must have shape [T, {self.hidden}] with T <= {self.num_max_tokens_per_rank}"
            )
        if x.dtype != self.dtype or not x.is_cuda or not x.is_contiguous():
            raise ValueError(f"x must be a contiguous CUDA {self.dtype} tensor")
        if expert_alignment <= 0:
            raise ValueError("expert_alignment must be positive")

        if handle is not None:
            if topk_idx is not None:
                raise ValueError("topk_idx must be omitted when a cached handle is supplied")
            topk_idx = handle.topk_idx
            topk_weights = handle.topk_weights if topk_weights is None else topk_weights
        if topk_idx is None:
            raise ValueError("topk_idx is required for a non-cached dispatch")
        if topk_idx.shape != (x.shape[0], self.num_topk) or not topk_idx.is_cuda:
            raise ValueError(f"topk_idx must be a CUDA tensor shaped [T, {self.num_topk}]")
        topk_idx = topk_idx.to(torch.int64).contiguous()
        if topk_weights is None:
            topk_weights = torch.ones_like(topk_idx, dtype=torch.float32)
        if topk_weights.shape != topk_idx.shape or not topk_weights.is_cuda:
            raise ValueError("topk_weights must be a CUDA tensor with the same shape as topk_idx")
        return x, topk_idx, topk_weights.to(torch.float32).contiguous()

    def post_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        *,
        expert_alignment: int = 1,
        handle: Optional[DLBHandle] = None,
    ) -> DLBDispatchTicket:
        """Posts DLB routing and transport without waiting for received inputs.

        The application may run work that does not consume this dispatch's
        expert inputs before calling :meth:`DLBDispatchTicket.finish`.  One
        outstanding post is allowed per buffer because it owns the current
        transport and repair slots.
        """
        self._check_open()
        if self._pending_dispatch is not None:
            raise RuntimeError("finish the outstanding DLB dispatch before posting another one")
        x, topk_idx, topk_weights = self._prepare_dispatch_inputs(
            x, topk_idx, topk_weights, handle, expert_alignment
        )
        epoch = self._epoch + 1
        self._comm.post_dispatch_moe(
            x,
            topk_idx,
            topk_weights,
            epoch,
            self._round_id,
            self.num_experts,
            expert_alignment,
        )
        self._epoch = epoch
        self._round_id += 1
        ticket = DLBDispatchTicket(
            self, x, topk_idx, topk_weights, epoch, expert_alignment
        )
        self._pending_dispatch = ticket
        return ticket

    def _finish_posted_dispatch(
        self, ticket: DLBDispatchTicket
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DLBHandle, EventOverlap]:
        """Materializes the expert inputs associated with one posted ticket."""
        self._check_open()
        if ticket._buffer is not self or ticket is not self._pending_dispatch:
            raise RuntimeError("the DLB dispatch ticket does not belong to this active buffer")
        if ticket._finished:
            raise RuntimeError("the DLB dispatch ticket has already been finished")
        (
            recv_x,
            headers,
            recv_weights,
            valid_mask,
            actual_counts_t,
            aligned_counts_t,
            rail_traffic,
            channel_traffic,
            group_output_indices,
        ) = self._comm.finish_dispatch_moe(ticket._epoch)
        actual_counts = [int(value) for value in actual_counts_t.tolist()]
        aligned_counts = [int(value) for value in aligned_counts_t.tolist()]
        result_handle = DLBHandle(
            topk_idx=ticket._topk_idx,
            topk_weights=ticket._topk_weights,
            num_tokens=ticket._x.shape[0],
            num_experts=self.num_experts,
            expert_alignment=ticket._expert_alignment,
            num_recv_tokens=sum(actual_counts),
            num_recv_tokens_per_expert_list=aligned_counts,
            num_unaligned_recv_tokens_per_expert_list=actual_counts,
            recv_headers=headers,
            recv_source_rank=headers[:, 1],
            recv_source_token=headers[:, 2],
            recv_expert_idx=headers[:, 3],
            recv_topk_slot=headers[:, 4],
            recv_topk_weights=recv_weights,
            valid_mask=valid_mask,
            dispatch_group_output_indices=group_output_indices,
            dispatch_rail_traffic_records=rail_traffic,
            dispatch_channel_traffic_records=channel_traffic,
        )
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        ticket._finished = True
        self._pending_dispatch = None
        return recv_x, result_handle.recv_expert_idx, recv_weights, result_handle, EventOverlap(
            event, (recv_x, result_handle.recv_expert_idx, recv_weights)
        )

    def combine(
        self,
        x: torch.Tensor,
        handle: DLBHandle,
        topk_weights: Optional[torch.Tensor] = None,
        *,
        apply_router_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, EventOverlap]:
        """Return expert outputs and reduce them into original token order."""
        self._check_open()
        if self._pending_dispatch is not None:
            raise RuntimeError("finish the outstanding DLB dispatch before calling combine")
        if x.shape != (handle.valid_mask.numel(), self.hidden):
            raise ValueError(
                f"combine x must have shape [{handle.valid_mask.numel()}, {self.hidden}]"
            )
        if x.dtype != self.dtype or not x.is_cuda or not x.is_contiguous():
            raise ValueError(f"combine x must be a contiguous CUDA {self.dtype} tensor")
        if handle.num_experts != self.num_experts or handle.topk_idx.shape != (
            handle.num_tokens,
            self.num_topk,
        ):
            raise ValueError("combine handle is incompatible with this DLBBuffer")

        if topk_weights is not None:
            if (
                topk_weights.shape != handle.recv_topk_weights.shape
                or not topk_weights.is_cuda
                or topk_weights.device != x.device
            ):
                raise ValueError("combine topk_weights must match the padded dispatched layout")
        weights = handle.recv_topk_weights if topk_weights is None else topk_weights
        weights = weights.to(torch.float32).contiguous()
        epoch = self._epoch + 1
        combined, combined_weights = self._comm.combine_moe(
            x,
            handle.recv_headers,
            weights,
            handle.valid_mask,
            handle.dispatch_group_output_indices,
            epoch,
            self._round_id,
            handle.num_tokens,
            self.num_topk,
            apply_router_weights,
        )
        self._epoch = epoch
        self._round_id += 1

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        return combined, combined_weights, EventOverlap(event, (combined, combined_weights))

    def get_last_dispatch_profile(self) -> dict[str, float]:
        """Synchronize and return CUDA stage timings for the last dispatch.

        The method is deliberately opt-in and must be called after dispatch
        and before combine. Normal production launches allocate no timed CUDA
        events when ``enable_profiling`` is false.
        """
        self._check_open()
        if not self.enable_profiling:
            raise RuntimeError("DLB profiling was not enabled for this buffer")
        values = self._comm.get_last_dispatch_profile().tolist()
        return dict(zip(_DISPATCH_PROFILE_NAMES, (float(value) for value in values)))

    def get_last_combine_profile(self) -> dict[str, float]:
        """Synchronize and return CUDA stage timings for the last combine.

        This must be called after combine and before the next communication
        operation. Profiling remains completely disabled in normal launches.
        """
        self._check_open()
        if not self.enable_profiling:
            raise RuntimeError("DLB profiling was not enabled for this buffer")
        values = self._comm.get_last_combine_profile().tolist()
        return dict(zip(_COMBINE_PROFILE_NAMES, (float(value) for value in values)))

    def close(self) -> None:
        if self.closed:
            return
        if self._pending_dispatch is not None:
            raise RuntimeError("finish the outstanding DLB dispatch before closing the buffer")
        self._comm.close()
        self.closed = True

    def __enter__(self) -> "DLBBuffer":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
