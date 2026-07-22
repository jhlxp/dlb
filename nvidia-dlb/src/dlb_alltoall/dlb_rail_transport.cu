#include "dlb_alltoall/dlb_rail_transport.h"

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdint>

namespace dlb_alltoall {
namespace {

// A wide communication CTA keeps enough independent payload copies in flight.
// DLB uses 640 threads (20 warps): its generic receive/repair kernel needs
// enough registers that 768 threads would exceed the per-SM register budget
// on Hopper.
constexpr unsigned kCommThreadsPerBlock = 640;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = kCommThreadsPerBlock / kWarpSize;

__device__ __forceinline__ void wait_for_credit_warp(
    std::uint64_t* credits, std::uint32_t credit_index, std::uint64_t epoch) {
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        nvshmem_uint64_wait_until(credits + credit_index, NVSHMEM_CMP_GE, epoch);
    }
    __syncwarp();
}

__global__ void dlb_internode_rail_kernel(
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
    std::uint64_t chunk_bytes) {
    const std::uint32_t channel = blockIdx.x;
    if (channel >= channel_count) return;
    const std::uint32_t warp = threadIdx.x / kWarpSize;
    const std::uint32_t lane = threadIdx.x & (kWarpSize - 1);
    // A fixed sender CTA owns one channel across every destination group;
    // its warps process independent groups concurrently. Consequently the CTA
    // count does not grow with S or M while all 20 warps remain useful.
    for (std::uint32_t transfer_index = warp * channel_count + channel;
         transfer_index < transfer_count;
         transfer_index += kWarpsPerBlock * channel_count) {
        const DlbRailTransfer transfer = transfers[transfer_index];
        wait_for_credit_warp(symmetric_credits, transfer.credit_index, credit_epoch);

        // Encode the source-server-local receive offset and byte count in the
        // signal itself. `+1` keeps an empty transfer distinguishable from an
        // idle (zero) signal.
        const std::uint64_t slot_offset =
            transfer.destination_offset_bytes % receive_slot_bytes;
        const std::uint64_t encoded =
            ((slot_offset + 1) << 32) |
            static_cast<std::uint32_t>(transfer.bytes + 1);
        if (lane == 0) {
            nvshmemx_signal_op(symmetric_receive_signals + transfer.signal_index,
                               encoded, NVSHMEM_SIGNAL_SET,
                               transfer.destination_rank);
        }
        __syncwarp();
        for (std::uint64_t offset = 0; offset < transfer.bytes;
             offset += chunk_bytes) {
            const std::uint64_t bytes = min(chunk_bytes, transfer.bytes - offset);
            // On a P2P peer this becomes a cooperative remote load/store copy;
            // on an inter-node peer NVSHMEM selects its RDMA transport. The
            // blocking primitive makes the channel progress prefix exact.
            nvshmemx_putmem_signal_warp(
                symmetric_receive_buffer + transfer.destination_offset_bytes + offset,
                source_buffer + transfer.source_offset_bytes + offset,
                bytes,
                symmetric_progress_signals + transfer.signal_index,
                bytes,
                NVSHMEM_SIGNAL_ADD,
                transfer.destination_rank);
        }
    }
}

__device__ __forceinline__ void copy_repair_bytes_warp(
    const std::uint8_t* source, std::uint8_t* destination, std::uint64_t bytes) {
    constexpr unsigned kUnroll = 4;
    const unsigned lane = threadIdx.x & (kWarpSize - 1);
    const std::uintptr_t alignment = reinterpret_cast<std::uintptr_t>(source) |
                                     reinterpret_cast<std::uintptr_t>(destination) | bytes;
    if ((alignment & (alignof(uint4) - 1)) == 0) {
        const std::uint64_t count = bytes / sizeof(uint4);
        const uint4* input = reinterpret_cast<const uint4*>(source);
        uint4* output = reinterpret_cast<uint4*>(destination);
        for (std::uint64_t base = static_cast<std::uint64_t>(lane) * kUnroll;
             base < count; base += static_cast<std::uint64_t>(kWarpSize) * kUnroll) {
            uint4 values[kUnroll];
#pragma unroll
            for (unsigned item = 0; item < kUnroll; ++item) {
                if (base + item < count) values[item] = input[base + item];
            }
#pragma unroll
            for (unsigned item = 0; item < kUnroll; ++item) {
                if (base + item < count) output[base + item] = values[item];
            }
        }
        return;
    }
    for (std::uint64_t offset = lane; offset < bytes; offset += kWarpSize) {
        destination[offset] = source[offset];
    }
}

__device__ __forceinline__ std::uint64_t load_u64_system(
    const std::uint64_t* pointer) {
    return atomicAdd_system(
        reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(pointer)),
        0ULL);
}

__device__ __forceinline__ void wait_u64_system_warp(
    const std::uint64_t* pointer, std::uint64_t value) {
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        while (load_u64_system(pointer) < value) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
    }
    __syncwarp();
}

__global__ void dlb_loopback_rail_kernel(
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
    std::uint64_t signal_epoch_offset) {
    const std::uint32_t channel = blockIdx.x;
    if (channel >= channel_count) return;
    const std::uint32_t warp = threadIdx.x / kWarpSize;
    const std::uint32_t lane = threadIdx.x & (kWarpSize - 1);
    for (std::uint32_t transfer_index = warp * channel_count + channel;
         transfer_index < transfer_count;
         transfer_index += kWarpsPerBlock * channel_count) {
        const DlbRailTransfer transfer = transfers[transfer_index];
        wait_u64_system_warp(symmetric_credits + transfer.credit_index, credit_epoch);
        std::uint64_t* metadata =
            receive_signals[transfer.destination_rank] + signal_epoch_offset +
            transfer.signal_index;
        std::uint64_t* progress =
            progress_signals[transfer.destination_rank] + signal_epoch_offset +
            transfer.signal_index;
        if (lane == 0) {
            const std::uint64_t slot_offset =
                transfer.destination_offset_bytes % receive_slot_bytes;
            const std::uint64_t encoded =
                ((slot_offset + 1) << 32) |
                static_cast<std::uint32_t>(transfer.bytes + 1);
            atomicExch_system(reinterpret_cast<unsigned long long*>(metadata), encoded);
        }
        __syncwarp();
        for (std::uint64_t chunk_offset = 0; chunk_offset < transfer.bytes;
             chunk_offset += chunk_bytes) {
            const std::uint64_t bytes = min(chunk_bytes, transfer.bytes - chunk_offset);
            copy_repair_bytes_warp(
                source_buffer + transfer.source_offset_bytes + chunk_offset,
                receive_buffers[transfer.destination_rank] +
                    receive_epoch_offset_bytes +
                    transfer.destination_offset_bytes + chunk_offset,
                bytes);
            // Every lane orders its own peer stores before lane 0 advances
            // the remotely visible contiguous-prefix counter.
            __threadfence_system();
            __syncwarp();
            if (lane == 0) {
                atomicAdd_system(reinterpret_cast<unsigned long long*>(progress), bytes);
            }
            __syncwarp();
        }
    }
}

__global__ void dlb_receive_and_repair_chunks_kernel(
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
    std::uint64_t loopback_credit_offset) {
    const std::uint32_t channel = blockIdx.x;
    if (channel >= channel_count) return;
    const std::uint32_t warp = threadIdx.x / kWarpSize;
    const std::uint32_t lane = threadIdx.x & (kWarpSize - 1);
    for (std::uint32_t arrival_index = warp * channel_count + channel;
         arrival_index < arrival_count;
         arrival_index += kWarpsPerBlock * channel_count) {
        const DlbRailArrival arrival = arrivals[arrival_index];
        std::uint64_t metadata_value = 0;
        if (lane == 0) {
            std::uint64_t* metadata = symmetric_receive_signals + arrival.signal_index;
            if (use_loopback_transport) {
                while (load_u64_system(metadata) < 1) {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(64);
#endif
                }
            } else {
                nvshmem_uint64_wait_until(metadata, NVSHMEM_CMP_GE, 1);
            }
            metadata_value = *metadata;
        }
        const std::uint64_t encoded = __shfl_sync(0xffffffffu, metadata_value, 0);
        const std::uint64_t slot_offset = (encoded >> 32) - 1;
        const std::uint64_t bytes = static_cast<std::uint32_t>(encoded) - 1;
        const std::uint64_t source_server = arrival.source_rank / gpus_per_server;
        const std::uint64_t receive_offset =
            source_server * receive_slot_bytes + slot_offset;
        const std::uint64_t repair_offset = receive_offset % repair_slot_bytes;
        const bool valid = repair_offset <= repair_slot_bytes &&
                           bytes <= repair_slot_bytes - repair_offset;
        if (valid) {
            for (std::uint64_t offset = 0; offset < bytes; offset += chunk_bytes) {
                const std::uint64_t current = min(chunk_bytes, bytes - offset);
                if (lane == 0) {
                    std::uint64_t* progress =
                        symmetric_progress_signals + arrival.signal_index;
                    if (use_loopback_transport) {
                        while (load_u64_system(progress) < offset + current) {
#if __CUDA_ARCH__ >= 700
                            __nanosleep(64);
#endif
                        }
                    } else {
                        nvshmem_uint64_wait_until(
                            progress, NVSHMEM_CMP_GE, offset + current);
                    }
                }
                __syncwarp();
                const std::uint8_t* source =
                    symmetric_receive_buffer + receive_offset + offset;
                std::uint8_t* destination =
                    final_destination_buffers[arrival.final_destination_rank] +
                    repair_epoch_offset_bytes +
                    static_cast<std::uint64_t>(arrival.source_rank) * repair_slot_bytes +
                    repair_offset + offset;
                copy_repair_bytes_warp(source, destination, current);
                __syncwarp();
            }
            __threadfence_system();
        }
        if (lane == 0) {
            symmetric_receive_signals[arrival.signal_index] = 0;
            symmetric_progress_signals[arrival.signal_index] = 0;
            __threadfence_system();
            if (use_loopback_transport) {
                atomicExch_system(
                    reinterpret_cast<unsigned long long*>(
                        loopback_credit_buffers[arrival.source_rank] +
                        loopback_credit_offset + arrival.credit_index),
                    arrival_epoch + 1);
            } else {
                nvshmemx_signal_op(symmetric_credits + arrival.credit_index,
                                   arrival_epoch + 1, NVSHMEM_SIGNAL_SET,
                                   arrival.source_rank);
            }
        }
        __syncwarp();
    }
}

cudaError_t launch_1d(dim3 grid, dim3 block, const void* kernel,
                      void** args, cudaStream_t stream) {
    cudaError_t status = cudaLaunchKernel(kernel, grid, block, args, 0, stream);
    return status == cudaSuccess ? cudaGetLastError() : status;
}

}  // namespace

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
    cudaStream_t stream) {
    if (transfer_count == 0) {
        return cudaSuccess;
    }
    if (source_buffer == nullptr || symmetric_receive_buffer == nullptr ||
        symmetric_receive_signals == nullptr || symmetric_progress_signals == nullptr ||
        symmetric_credits == nullptr || transfers == nullptr || channel_count == 0 ||
        receive_slot_bytes == 0 || chunk_bytes == 0) {
        return cudaErrorInvalidValue;
    }

    void* args[] = {
        &source_buffer,
        &symmetric_receive_buffer,
        &symmetric_receive_signals,
        &symmetric_progress_signals,
        &symmetric_credits,
        &transfers,
        &transfer_count,
        &channel_count,
        &credit_epoch,
        &receive_slot_bytes,
        &chunk_bytes,
    };
    const dim3 block(kCommThreadsPerBlock);
    const dim3 grid(channel_count);
    return launch_1d(grid, block, reinterpret_cast<const void*>(dlb_internode_rail_kernel), args, stream);
}

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
    cudaStream_t stream) {
    if (transfer_count == 0) return cudaSuccess;
    if (source_buffer == nullptr || receive_buffers == nullptr ||
        receive_signals == nullptr || progress_signals == nullptr ||
        symmetric_credits == nullptr || transfers == nullptr || channel_count == 0 ||
        receive_slot_bytes == 0 || chunk_bytes == 0) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {
        &source_buffer, &receive_buffers, &receive_signals, &progress_signals,
        &symmetric_credits, &transfers, &transfer_count, &channel_count, &credit_epoch,
        &receive_slot_bytes, &chunk_bytes, &receive_epoch_offset_bytes,
        &signal_epoch_offset,
    };
    return launch_1d(dim3(channel_count), dim3(kCommThreadsPerBlock),
                     reinterpret_cast<const void*>(dlb_loopback_rail_kernel),
                     args, stream);
}

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
    cudaStream_t stream) {
    if (arrival_count == 0) return cudaSuccess;
    if (symmetric_receive_buffer == nullptr || symmetric_receive_signals == nullptr ||
        symmetric_progress_signals == nullptr || symmetric_credits == nullptr ||
        arrivals == nullptr || channel_count == 0 || final_destination_buffers == nullptr ||
        receive_slot_bytes == 0 || gpus_per_server == 0 || repair_slot_bytes == 0 ||
        chunk_bytes == 0) {
        return cudaErrorInvalidValue;
    }
    if (use_loopback_transport && loopback_credit_buffers == nullptr) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&symmetric_receive_buffer, &symmetric_receive_signals,
                    &symmetric_progress_signals, &symmetric_credits, &arrivals,
                    &arrival_count, &channel_count, &arrival_epoch, &receive_slot_bytes,
                    &gpus_per_server, &final_destination_buffers, &repair_slot_bytes,
                    &repair_epoch_offset_bytes, &chunk_bytes,
                    &use_loopback_transport, &loopback_credit_buffers,
                    &loopback_credit_offset};
    return launch_1d(dim3(channel_count), dim3(kCommThreadsPerBlock),
                     reinterpret_cast<const void*>(dlb_receive_and_repair_chunks_kernel),
                     args, stream);
}

}  // namespace dlb_alltoall
