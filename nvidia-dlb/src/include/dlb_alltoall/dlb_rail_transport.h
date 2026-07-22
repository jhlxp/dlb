#pragma once

#include "dlb_alltoall/dlb_rail_protocol.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace dlb_alltoall {

// Submit every transfer owned by the calling Rail rank. The receive buffer,
// signal, progress, and credit buffers are NVSHMEM symmetric allocations. The
// transfer set includes zero-byte arrivals so the destination can wait only on
// static topology-derived slots.
cudaError_t launch_dlb_internode_rail(
    const std::uint8_t* source_buffer,
    std::uint8_t* symmetric_receive_buffer,
    std::uint64_t* symmetric_receive_signals,
    std::uint64_t* symmetric_progress_signals,
    std::uint64_t* symmetric_credits,
    const DlbRailTransfer* transfers,
    std::uint32_t transfer_count,
    std::uint32_t channel_count,
    std::uint64_t credit_epoch,
    std::uint64_t receive_slot_bytes,
    std::uint64_t chunk_bytes,
    cudaStream_t stream);

// Same-host implementation of the identical Rail descriptors. Each chunk is
// copied directly through a peer pointer and published with system-scope
// atomics. This backend exists for faithful multi-server algorithm testing on
// one NVSwitch host; it is never selected implicitly for real multi-node jobs.
cudaError_t launch_dlb_loopback_rail(
    const std::uint8_t* source_buffer,
    std::uint8_t* const* receive_buffers,
    std::uint64_t* const* receive_signals,
    std::uint64_t* const* progress_signals,
    std::uint64_t* symmetric_credits,
    const DlbRailTransfer* transfers,
    std::uint32_t transfer_count,
    std::uint32_t channel_count,
    std::uint64_t credit_epoch,
    std::uint64_t receive_slot_bytes,
    std::uint64_t chunk_bytes,
    std::uint64_t receive_epoch_offset_bytes,
    std::uint64_t signal_epoch_offset,
    cudaStream_t stream);

// Receive each Rail slice as a sequence of put+signal chunks and immediately
// repair every visible chunk into its real destination GPU.  Credit is
// returned only after the final chunk is system-visible at the destination.
cudaError_t launch_dlb_receive_and_repair_chunks(
    const std::uint8_t* symmetric_receive_buffer,
    std::uint64_t* symmetric_receive_signals,
    std::uint64_t* symmetric_progress_signals,
    std::uint64_t* symmetric_credits,
    const DlbRailArrival* arrivals,
    std::uint32_t arrival_count,
    std::uint32_t channel_count,
    std::uint64_t arrival_epoch,
    std::uint64_t receive_slot_bytes,
    std::uint32_t gpus_per_server,
    std::uint8_t* const* final_destination_buffers,
    std::uint64_t repair_slot_bytes,
    std::uint64_t repair_epoch_offset_bytes,
    std::uint64_t chunk_bytes,
    bool use_loopback_transport,
    std::uint64_t* const* loopback_credit_buffers,
    std::uint64_t loopback_credit_offset,
    cudaStream_t stream);

}  // namespace dlb_alltoall
