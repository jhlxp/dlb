#include "dlb_alltoall/dlb_nvlink_runtime.h"

#include <cstdint>

// Deliberately at global C++ linkage: this is the same CUDA registration shape
// used by the DLB Torch extension and makes the probe useful when diagnosing
// dynamic-module loading behaviour.
__global__ void dlb_cuda_module_probe_kernel(std::uint32_t* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *result = 0x444c42u;
    }
}

namespace dlb_alltoall {

extern "C" __global__ void dlb_publish_ipc_epoch_kernel(
    std::uint64_t* const* target_signal_buffers,
    std::uint32_t target_count,
    std::uint32_t signal_offset,
    std::uint64_t epoch) {
    const std::uint32_t target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target >= target_count) return;
    atomicExch_system(
        reinterpret_cast<unsigned long long*>(target_signal_buffers[target] + signal_offset),
        static_cast<unsigned long long>(epoch));
}

extern "C" __global__ void dlb_wait_ipc_epochs_kernel(
    const std::uint64_t* local_signal_buffer,
    std::uint32_t signal_offset,
    std::uint32_t producer_count,
    std::uint64_t epoch) {
    const std::uint32_t producer = blockIdx.x * blockDim.x + threadIdx.x;
    if (producer >= producer_count) return;
    auto* signal = reinterpret_cast<unsigned long long*>(
        const_cast<std::uint64_t*>(local_signal_buffer + signal_offset + producer));
    while (atomicAdd_system(signal, 0ULL) < static_cast<unsigned long long>(epoch)) {
        __nanosleep(64);
    }
}

cudaError_t launch_dlb_cuda_module_probe(std::uint32_t* result, cudaStream_t stream) {
    if (result == nullptr) return cudaErrorInvalidValue;
    dlb_cuda_module_probe_kernel<<<1, 1, 0, stream>>>(result);
    return cudaGetLastError();
}

cudaError_t launch_dlb_publish_ipc_epoch(
    std::uint64_t* const* target_signal_buffers,
    std::uint32_t target_count,
    std::uint32_t signal_offset,
    std::uint64_t epoch,
    cudaStream_t stream) {
    if (target_count == 0) return cudaSuccess;
    if (target_signal_buffers == nullptr || epoch == 0) return cudaErrorInvalidValue;
    constexpr unsigned kThreads = 128;
    void* args[] = {&target_signal_buffers, &target_count, &signal_offset, &epoch};
    const cudaError_t status = cudaLaunchKernel(
        reinterpret_cast<const void*>(dlb_publish_ipc_epoch_kernel),
        dim3((target_count + kThreads - 1) / kThreads), dim3(kThreads), args, 0, stream);
    return status == cudaSuccess ? cudaGetLastError() : status;
}

cudaError_t launch_dlb_wait_ipc_epochs(
    const std::uint64_t* local_signal_buffer,
    std::uint32_t signal_offset,
    std::uint32_t producer_count,
    std::uint64_t epoch,
    cudaStream_t stream) {
    if (producer_count == 0) return cudaSuccess;
    if (local_signal_buffer == nullptr || epoch == 0) return cudaErrorInvalidValue;
    constexpr unsigned kThreads = 128;
    void* args[] = {&local_signal_buffer, &signal_offset, &producer_count, &epoch};
    const cudaError_t status = cudaLaunchKernel(
        reinterpret_cast<const void*>(dlb_wait_ipc_epochs_kernel),
        dim3((producer_count + kThreads - 1) / kThreads), dim3(kThreads), args, 0, stream);
    return status == cudaSuccess ? cudaGetLastError() : status;
}

}  // namespace dlb_alltoall
