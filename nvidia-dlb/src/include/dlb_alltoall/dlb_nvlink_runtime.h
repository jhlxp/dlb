#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace dlb_alltoall {

// Minimal dynamic-module smoke check. It intentionally touches neither
// CUDA-IPC nor NVSHMEM, so an error here points at CUDA module loading rather
// than the DLB transport protocol.
cudaError_t launch_dlb_cuda_module_probe(std::uint32_t* result, cudaStream_t stream);

// CUDA events do not synchronize independent processes. These helpers use
// CUDA-IPC-visible system-scope atomics to publish and wait for every local
// GPU process participating in direct pack or destination repair.
cudaError_t launch_dlb_publish_ipc_epoch(
    std::uint64_t* const* target_signal_buffers,
    std::uint32_t target_count,
    std::uint32_t signal_offset,
    std::uint64_t epoch,
    cudaStream_t stream);

cudaError_t launch_dlb_wait_ipc_epochs(
    const std::uint64_t* local_signal_buffer,
    std::uint32_t signal_offset,
    std::uint32_t producer_count,
    std::uint64_t epoch,
    cudaStream_t stream);

}  // namespace dlb_alltoall
