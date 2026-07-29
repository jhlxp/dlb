#include "dlb_loopback_test.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "kernels/rail_lb.cuh"
#include "kernels/utils.cuh"

#include <algorithm>
#include <cstring>

namespace deep_ep_lb_test {
namespace {

constexpr int kRails = NUM_MAX_NVL_PEERS;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr int kPersistentBlocks = 20;
constexpr size_t kAlignment = 128;

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void check_cuda_ints(const torch::Tensor& tensor, int device) {
    TORCH_CHECK(tensor.is_cuda());
    TORCH_CHECK(tensor.is_contiguous());
    TORCH_CHECK(tensor.get_device() == device);
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32);
    TORCH_CHECK(tensor.numel() == kRails);
}

void check_cuda_quota_rows(const torch::Tensor& tensor, int device) {
    TORCH_CHECK(tensor.is_cuda());
    TORCH_CHECK(tensor.is_contiguous());
    TORCH_CHECK(tensor.get_device() == device);
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32);
    TORCH_CHECK(tensor.dim() == 2);
    TORCH_CHECK(tensor.size(0) == kRails);
    TORCH_CHECK(tensor.size(1) == kRails);
}

void check_cuda_destination_rails(const torch::Tensor& tensor, int device) {
    TORCH_CHECK(tensor.is_cuda());
    TORCH_CHECK(tensor.is_contiguous());
    TORCH_CHECK(tensor.get_device() == device);
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32);
    TORCH_CHECK(tensor.dim() == 1);
}

__device__ __forceinline__ void copy_record(
    uint8_t* destination,
    const uint8_t* source,
    int record_bytes,
    int lane) {
    for (int offset = lane * static_cast<int>(sizeof(int4));
         offset < record_bytes;
         offset += 32 * static_cast<int>(sizeof(int4))) {
        *reinterpret_cast<int4*>(destination + offset) =
            *reinterpret_cast<const int4*>(source + offset);
    }
}

__device__ __forceinline__ int selected_ordinal(
    const int* quotas,
    int source_rail,
    int selected_rail,
    int ordinal) {
    if (selected_rail == source_rail)
        return ordinal;
    ordinal -= quotas[source_rail];
    #pragma unroll
    for (int rail = 0; rail < kRails; ++rail) {
        if (rail == source_rail)
            continue;
        if (rail == selected_rail)
            return ordinal;
        ordinal -= quotas[rail];
    }
    return -1;
}

__device__ __forceinline__ int staging_offset(
    const int* quota_rows,
    int source_rail,
    int selected_rail) {
    int offset = 0;
    #pragma unroll
    for (int producer = 0; producer < source_rail; ++producer) {
        if (producer == selected_rail)
            continue;
        offset += quota_rows[producer * kRails + selected_rail];
    }
    return offset;
}

__global__ void pack_off_kernel(
    const uint8_t* records,
    uint8_t* local_buffer,
    size_t ring_ids_offset,
    size_t ring_payload_offset,
    size_t record_stride,
    int record_bytes,
    int source_rank,
    int num_messages) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int first_message =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int message_stride =
        static_cast<int>(gridDim.x) * kWarpsPerBlock;
    auto ring_ids = reinterpret_cast<int64_t*>(
        local_buffer + ring_ids_offset
    );
    for (int message = first_message;
         message < num_messages;
         message += message_stride) {
        auto destination =
            local_buffer + ring_payload_offset +
            static_cast<size_t>(message) * record_stride;
        copy_record(
            destination,
            records + static_cast<size_t>(message) * record_stride,
            record_bytes,
            lane
        );
        if (lane == 0) {
            ring_ids[message] =
                (static_cast<int64_t>(source_rank) << 32) |
                static_cast<uint32_t>(message);
        }
    }
}

__global__ void dispatch_on_kernel(
    const uint8_t* records,
    void** local_buffers,
    const int* quota_rows,
    size_t sync_offset,
    size_t stage_ready_offset,
    size_t staging_ids_offset,
    size_t staging_payload_offset,
    size_t ring_ids_offset,
    size_t ring_payload_offset,
    size_t record_stride,
    int record_bytes,
    int source_rank,
    int source_rail,
    int num_messages,
    int num_incoming,
    int epoch) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int first_ordinal =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int ordinal_stride =
        static_cast<int>(gridDim.x) * kWarpsPerBlock;
    const int* quotas = quota_rows + source_rail * kRails;
    for (int ordinal = first_ordinal;
         ordinal < num_messages;
         ordinal += ordinal_stride) {
        const int selected = deep_ep::rail_lb::select_rail(
            quotas, source_rail, ordinal
        );
        const int local_ordinal = selected_ordinal(
            quotas, source_rail, selected, ordinal
        );
        auto destination_buffer =
            static_cast<uint8_t*>(local_buffers[selected]);
        uint8_t* destination = nullptr;
        int64_t* destination_ids = nullptr;
        int* ready = nullptr;
        int destination_slot = local_ordinal;

        if (selected == source_rail) {
            destination =
                destination_buffer + ring_payload_offset +
                static_cast<size_t>(destination_slot) * record_stride;
            destination_ids = reinterpret_cast<int64_t*>(
                destination_buffer + ring_ids_offset
            );
        } else {
            destination_slot += staging_offset(
                quota_rows,
                source_rail,
                selected
            );
            destination =
                destination_buffer + staging_payload_offset +
                static_cast<size_t>(destination_slot) * record_stride;
            destination_ids = reinterpret_cast<int64_t*>(
                destination_buffer + staging_ids_offset
            );
            ready = reinterpret_cast<int*>(
                destination_buffer + stage_ready_offset
            ) + destination_slot;
        }

        copy_record(
            destination,
            records + static_cast<size_t>(ordinal) * record_stride,
            record_bytes,
            lane
        );
        if (lane == 0) {
            destination_ids[destination_slot] =
                (static_cast<int64_t>(source_rank) << 32) |
                static_cast<uint32_t>(ordinal);
        }
        __syncwarp();
        if (lane == 0 and ready != nullptr)
            deep_ep::st_release_sys_global(ready, epoch);
    }

    // Keep all test blocks resident and complete local production before any
    // block waits on peer-ready flags. This mirrors one persistent dispatch
    // launch without adding a second staging-consumer kernel.
    __syncthreads();
    auto sync = reinterpret_cast<int*>(local_buffers[source_rail]) +
        sync_offset / sizeof(int);
    if (threadIdx.x == 0) {
        __threadfence_system();
        atomicAdd(sync, 1);
        while (atomicAdd(sync, 0) != static_cast<int>(gridDim.x))
            __nanosleep(64);
    }
    __syncthreads();

    auto local_buffer =
        static_cast<uint8_t*>(local_buffers[source_rail]);
    const int* selected_quotas =
        quota_rows + source_rail * kRails;
    const int direct = selected_quotas[source_rail];
    auto staging_ids = reinterpret_cast<const int64_t*>(
        local_buffer + staging_ids_offset
    );
    auto ring_ids = reinterpret_cast<int64_t*>(
        local_buffer + ring_ids_offset
    );
    for (int stage_slot = first_ordinal;
         stage_slot < num_incoming;
         stage_slot += ordinal_stride) {
        auto ready = reinterpret_cast<int*>(
            local_buffer + stage_ready_offset
        ) + stage_slot;
        while (deep_ep::ld_acquire_sys_global(ready) != epoch)
            __nanosleep(64);

        const int ring_slot = direct + stage_slot;
        copy_record(
            local_buffer + ring_payload_offset +
                static_cast<size_t>(ring_slot) * record_stride,
            local_buffer + staging_payload_offset +
                static_cast<size_t>(stage_slot) * record_stride,
            record_bytes,
            lane
        );
        if (lane == 0)
            ring_ids[ring_slot] = staging_ids[stage_slot];
    }
}

__global__ void seed_receive_ring_kernel(
    const uint8_t* records,
    const int* destination_rails,
    uint8_t* local_buffer,
    size_t ring_ids_offset,
    size_t ring_payload_offset,
    size_t record_stride,
    int record_bytes,
    int num_messages) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int first_message =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int message_stride =
        static_cast<int>(gridDim.x) * kWarpsPerBlock;
    auto ring_ids = reinterpret_cast<int64_t*>(
        local_buffer + ring_ids_offset
    );
    for (int message = first_message;
         message < num_messages;
         message += message_stride) {
        copy_record(
            local_buffer + ring_payload_offset +
                static_cast<size_t>(message) * record_stride,
            records + static_cast<size_t>(message) * record_stride,
            record_bytes,
            lane
        );
        if (lane == 0)
            ring_ids[message] = destination_rails[message];
    }
}

__global__ void post_forward_kernel(
    void** local_buffers,
    size_t sync_offset,
    size_t staging_payload_offset,
    size_t ring_ids_offset,
    size_t ring_payload_offset,
    size_t record_stride,
    int record_bytes,
    int source_rail,
    int num_ring_messages) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int first_message =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int message_stride =
        static_cast<int>(gridDim.x) * kWarpsPerBlock;
    auto local_buffer =
        static_cast<uint8_t*>(local_buffers[source_rail]);
    auto ring_ids = reinterpret_cast<const int64_t*>(
        local_buffer + ring_ids_offset
    );
    for (int message = first_message;
         message < num_ring_messages;
         message += message_stride) {
        const int destination_rail = static_cast<int>(ring_ids[message]);
        auto destination_buffer =
            static_cast<uint8_t*>(local_buffers[destination_rail]);
        auto counter = reinterpret_cast<int*>(
            destination_buffer + sync_offset
        );
        int slot = 0;
        if (lane == 0)
            slot = atomicAdd(counter, 1);
        slot = __shfl_sync(0xffffffff, slot, 0);
        copy_record(
            destination_buffer + staging_payload_offset +
                static_cast<size_t>(slot) * record_stride,
            local_buffer + ring_payload_offset +
                static_cast<size_t>(message) * record_stride,
            record_bytes,
            lane
        );
    }
}

__global__ void combine_kernel(
    uint8_t* combined_records,
    void** local_buffers,
    size_t sync_offset,
    size_t repair_ready_offset,
    size_t staging_payload_offset,
    size_t ring_ids_offset,
    size_t ring_payload_offset,
    size_t record_stride,
    int record_bytes,
    int source_rail,
    int num_source_messages,
    int num_ring_messages,
    int epoch) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int first_message =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int message_stride =
        static_cast<int>(gridDim.x) * kWarpsPerBlock;
    auto local_buffer =
        static_cast<uint8_t*>(local_buffers[source_rail]);
    const auto ring_ids = reinterpret_cast<const int64_t*>(
        local_buffer + ring_ids_offset
    );

    // Return each selected-Rail result to its original Rail. Direct messages
    // stay local; only moved messages cross CUDA IPC in this phase.
    for (int message = first_message;
         message < num_ring_messages;
         message += message_stride) {
        const int64_t identity = ring_ids[message];
        const int origin_rank = static_cast<int>(identity >> 32);
        const int origin_rail = origin_rank % kRails;
        const int origin_token = static_cast<int>(
            static_cast<uint32_t>(identity)
        );
        auto origin_buffer =
            static_cast<uint8_t*>(local_buffers[origin_rail]);
        copy_record(
            origin_buffer + staging_payload_offset +
                static_cast<size_t>(origin_token) * record_stride,
            local_buffer + ring_payload_offset +
                static_cast<size_t>(message) * record_stride,
            record_bytes,
            lane
        );
        __syncwarp();
        if (lane == 0) {
            deep_ep::st_release_sys_global(
                reinterpret_cast<int*>(
                    origin_buffer + repair_ready_offset
                ) + origin_token,
                epoch
            );
        }
    }

    __syncthreads();
    auto sync = reinterpret_cast<int*>(
        local_buffer + sync_offset
    );
    if (threadIdx.x == 0) {
        __threadfence_system();
        atomicAdd(sync, 1);
        while (atomicAdd(sync, 0) != static_cast<int>(gridDim.x))
            __nanosleep(64);
    }
    __syncthreads();

    // Every origin Rail materializes its original token order after all
    // direct and moved partial rows have returned.
    for (int token = first_message;
         token < num_source_messages;
         token += message_stride) {
        auto ready = reinterpret_cast<int*>(
            local_buffer + repair_ready_offset
        ) + token;
        while (deep_ep::ld_acquire_sys_global(ready) != epoch)
            __nanosleep(64);
        copy_record(
            combined_records +
                static_cast<size_t>(token) * record_stride,
            local_buffer + staging_payload_offset +
                static_cast<size_t>(token) * record_stride,
            record_bytes,
            lane
        );
    }
}

__global__ void quota_stats_kernel(
    const int* source_loads,
    int source_server,
    int destination_server,
    int source_rail,
    int round_id,
    int64_t* output) {
    if (blockIdx.x != 0 or threadIdx.x != 0)
        return;
    int quotas[kRails];
    deep_ep::rail_lb::compute_quota_row(
        source_loads,
        source_server,
        destination_server,
        source_rail,
        round_id,
        quotas
    );
    int moved = 0;
    for (int rail = 0; rail < kRails; ++rail) {
        output[rail] = quotas[rail];
        if (rail != source_rail)
            moved += quotas[rail];
    }
    output[kRails] = source_loads[source_rail];
    output[kRails + 1] = moved;
}

}  // namespace

DlbP2PLoopbackRuntime::DlbP2PLoopbackRuntime(
    int rank,
    int world_size,
    int max_tokens,
    int record_bytes):
        rank_(rank),
        world_size_(world_size),
        max_tokens_(max_tokens),
        source_rail_(rank % kRails),
        queue_capacity_(max_tokens * kRails),
        record_stride_(align_up(record_bytes, sizeof(int4))) {
    TORCH_CHECK(kRails == 4);
    TORCH_CHECK(world_size == 8);
    TORCH_CHECK(rank >= 0 and rank < world_size);
    TORCH_CHECK(max_tokens > 0);
    TORCH_CHECK(record_bytes > 0 and record_bytes % sizeof(int4) == 0);
    C10_CUDA_CHECK(cudaGetDevice(&device_id_));

    size_t cursor = 0;
    sync_offset_ = cursor;
    cursor = align_up(cursor + sizeof(int), kAlignment);
    stage_ready_offset_ = cursor;
    cursor = align_up(
        cursor + static_cast<size_t>(queue_capacity_) * sizeof(int),
        kAlignment
    );
    repair_ready_offset_ = cursor;
    cursor = align_up(
        cursor + static_cast<size_t>(max_tokens_) * sizeof(int),
        kAlignment
    );
    staging_ids_offset_ = cursor;
    cursor = align_up(
        cursor + static_cast<size_t>(queue_capacity_) * sizeof(int64_t),
        kAlignment
    );
    staging_payload_offset_ = cursor;
    cursor = align_up(
        cursor + static_cast<size_t>(queue_capacity_) * record_stride_,
        kAlignment
    );
    ring_ids_offset_ = cursor;
    cursor = align_up(
        cursor + static_cast<size_t>(queue_capacity_) * sizeof(int64_t),
        kAlignment
    );
    ring_payload_offset_ = cursor;
    cursor = align_up(
        cursor + static_cast<size_t>(queue_capacity_) * record_stride_,
        kAlignment
    );
    total_bytes_ = cursor;

    C10_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&local_buffer_), total_bytes_
    ));
    C10_CUDA_CHECK(cudaMemset(local_buffer_, 0, total_bytes_));
    C10_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&local_ptrs_device_),
        sizeof(void*) * kRails
    ));
    local_ptrs_host_.resize(kRails, nullptr);
}

DlbP2PLoopbackRuntime::~DlbP2PLoopbackRuntime() {
    try {
        set_device();
        cudaDeviceSynchronize();
        for (int rail = 0; rail < kRails; ++rail) {
            if (rail != source_rail_ and local_ptrs_host_[rail] != nullptr)
                cudaIpcCloseMemHandle(local_ptrs_host_[rail]);
        }
        if (local_ptrs_device_ != nullptr)
            cudaFree(local_ptrs_device_);
        if (local_buffer_ != nullptr)
            cudaFree(local_buffer_);
    } catch (...) {
    }
}

void DlbP2PLoopbackRuntime::set_device() const {
    C10_CUDA_CHECK(cudaSetDevice(device_id_));
}

pybind11::bytes DlbP2PLoopbackRuntime::get_ipc_handle() const {
    set_device();
    cudaIpcMemHandle_t handle{};
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, local_buffer_));
    return pybind11::bytes(
        reinterpret_cast<const char*>(&handle), sizeof(handle)
    );
}

void DlbP2PLoopbackRuntime::sync(
    const std::vector<pybind11::bytes>& handles) {
    TORCH_CHECK(not synced_);
    TORCH_CHECK(static_cast<int>(handles.size()) == world_size_);
    set_device();
    const int server = rank_ / kRails;
    for (int rail = 0; rail < kRails; ++rail) {
        const int peer = server * kRails + rail;
        if (peer == rank_) {
            local_ptrs_host_[rail] = local_buffer_;
            continue;
        }
        const auto encoded = static_cast<std::string>(handles[peer]);
        TORCH_CHECK(encoded.size() == sizeof(cudaIpcMemHandle_t));
        cudaIpcMemHandle_t handle{};
        std::memcpy(&handle, encoded.data(), sizeof(handle));
        C10_CUDA_CHECK(cudaIpcOpenMemHandle(
            &local_ptrs_host_[rail],
            handle,
            cudaIpcMemLazyEnablePeerAccess
        ));
    }
    C10_CUDA_CHECK(cudaMemcpy(
        local_ptrs_device_,
        local_ptrs_host_.data(),
        sizeof(void*) * kRails,
        cudaMemcpyHostToDevice
    ));
    synced_ = true;
}

void DlbP2PLoopbackRuntime::reset() {
    TORCH_CHECK(synced_);
    set_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device_id_);
    C10_CUDA_CHECK(cudaMemsetAsync(
        local_buffer_ + ring_ids_offset_,
        0xff,
        static_cast<size_t>(queue_capacity_) * sizeof(int64_t),
        stream
    ));
}

void DlbP2PLoopbackRuntime::pack_off(
    const torch::Tensor& records,
    int num_messages) {
    TORCH_CHECK(synced_);
    TORCH_CHECK(records.is_cuda() and records.is_contiguous());
    TORCH_CHECK(records.get_device() == device_id_);
    TORCH_CHECK(records.scalar_type() == torch::kUInt8);
    TORCH_CHECK(records.dim() == 2);
    TORCH_CHECK(records.size(0) <= max_tokens_);
    TORCH_CHECK(
        records.size(1) == static_cast<int64_t>(record_stride_)
    );
    TORCH_CHECK(0 <= num_messages and num_messages <= records.size(0));
    if (num_messages == 0)
        return;
    pack_off_kernel<<<
        kPersistentBlocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream(device_id_)
    >>>(
        records.data_ptr<uint8_t>(),
        local_buffer_,
        ring_ids_offset_,
        ring_payload_offset_,
        record_stride_,
        static_cast<int>(record_stride_),
        rank_,
        num_messages
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void DlbP2PLoopbackRuntime::dispatch_on(
    const torch::Tensor& records,
    const torch::Tensor& quota_rows,
    int num_messages,
    int num_incoming,
    int epoch) {
    TORCH_CHECK(synced_);
    check_cuda_quota_rows(quota_rows, device_id_);
    TORCH_CHECK(records.is_cuda() and records.is_contiguous());
    TORCH_CHECK(records.get_device() == device_id_);
    TORCH_CHECK(records.scalar_type() == torch::kUInt8);
    TORCH_CHECK(
        records.dim() == 2 and
        records.size(0) <= max_tokens_ and
        records.size(1) == static_cast<int64_t>(record_stride_)
    );
    TORCH_CHECK(
        0 <= num_messages and num_messages <= records.size(0)
    );
    TORCH_CHECK(0 <= num_incoming and num_incoming <= queue_capacity_);
    if (num_messages == 0 and num_incoming == 0)
        return;
    const auto stream = at::cuda::getCurrentCUDAStream(device_id_);
    C10_CUDA_CHECK(cudaMemsetAsync(
        local_buffer_ + sync_offset_, 0, sizeof(int), stream
    ));
    dispatch_on_kernel<<<
        kPersistentBlocks,
        kThreads,
        0,
        stream
    >>>(
        records.data_ptr<uint8_t>(),
        local_ptrs_device_,
        quota_rows.data_ptr<int>(),
        sync_offset_,
        stage_ready_offset_,
        staging_ids_offset_,
        staging_payload_offset_,
        ring_ids_offset_,
        ring_payload_offset_,
        record_stride_,
        static_cast<int>(record_stride_),
        rank_,
        source_rail_,
        num_messages,
        num_incoming,
        epoch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void DlbP2PLoopbackRuntime::combine(
    const torch::Tensor& combined_records,
    int num_source_messages,
    int num_ring_messages,
    int epoch) {
    TORCH_CHECK(synced_);
    TORCH_CHECK(
        combined_records.is_cuda() and
        combined_records.is_contiguous()
    );
    TORCH_CHECK(combined_records.get_device() == device_id_);
    TORCH_CHECK(combined_records.scalar_type() == torch::kUInt8);
    TORCH_CHECK(
        combined_records.dim() == 2 and
        combined_records.size(0) >= num_source_messages and
        combined_records.size(1) ==
            static_cast<int64_t>(record_stride_)
    );
    TORCH_CHECK(
        0 <= num_source_messages and
        num_source_messages <= max_tokens_
    );
    TORCH_CHECK(
        0 <= num_ring_messages and
        num_ring_messages <= queue_capacity_
    );
    if (num_source_messages == 0)
        return;
    const auto stream = at::cuda::getCurrentCUDAStream(device_id_);
    C10_CUDA_CHECK(cudaMemsetAsync(
        local_buffer_ + sync_offset_, 0, sizeof(int), stream
    ));
    combine_kernel<<<
        kPersistentBlocks,
        kThreads,
        0,
        stream
    >>>(
        combined_records.data_ptr<uint8_t>(),
        local_ptrs_device_,
        sync_offset_,
        repair_ready_offset_,
        staging_payload_offset_,
        ring_ids_offset_,
        ring_payload_offset_,
        record_stride_,
        static_cast<int>(record_stride_),
        source_rail_,
        num_source_messages,
        num_ring_messages,
        epoch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void DlbP2PLoopbackRuntime::clear_post_forward_counter() {
    TORCH_CHECK(synced_);
    set_device();
    C10_CUDA_CHECK(cudaMemsetAsync(
        local_buffer_ + sync_offset_,
        0,
        sizeof(int),
        at::cuda::getCurrentCUDAStream(device_id_)
    ));
}

void DlbP2PLoopbackRuntime::seed_receive_ring(
    const torch::Tensor& records,
    const torch::Tensor& destination_rails,
    int num_messages) {
    TORCH_CHECK(synced_);
    TORCH_CHECK(records.is_cuda() and records.is_contiguous());
    TORCH_CHECK(records.get_device() == device_id_);
    TORCH_CHECK(records.scalar_type() == torch::kUInt8);
    TORCH_CHECK(records.dim() == 2);
    TORCH_CHECK(records.size(0) <= max_tokens_);
    TORCH_CHECK(
        records.size(1) == static_cast<int64_t>(record_stride_)
    );
    check_cuda_destination_rails(destination_rails, device_id_);
    TORCH_CHECK(destination_rails.size(0) >= num_messages);
    TORCH_CHECK(0 <= num_messages and num_messages <= records.size(0));
    if (num_messages == 0)
        return;
    seed_receive_ring_kernel<<<
        kPersistentBlocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream(device_id_)
    >>>(
        records.data_ptr<uint8_t>(),
        destination_rails.data_ptr<int>(),
        local_buffer_,
        ring_ids_offset_,
        ring_payload_offset_,
        record_stride_,
        static_cast<int>(record_stride_),
        num_messages
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void DlbP2PLoopbackRuntime::post_forward(int num_ring_messages) {
    TORCH_CHECK(synced_);
    TORCH_CHECK(0 <= num_ring_messages and
                num_ring_messages <= queue_capacity_);
    if (num_ring_messages == 0)
        return;
    post_forward_kernel<<<
        kPersistentBlocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream(device_id_)
    >>>(
        local_ptrs_device_,
        sync_offset_,
        staging_payload_offset_,
        ring_ids_offset_,
        ring_payload_offset_,
        record_stride_,
        static_cast<int>(record_stride_),
        source_rail_,
        num_ring_messages
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor>
DlbP2PLoopbackRuntime::materialize_ring(int count) {
    TORCH_CHECK(synced_);
    TORCH_CHECK(0 <= count and count <= queue_capacity_);
    auto device = torch::TensorOptions().device(
        torch::kCUDA, device_id_
    );
    auto ids = torch::empty({count}, device.dtype(torch::kInt64));
    auto records = torch::empty(
        {count, static_cast<int64_t>(record_stride_)},
        device.dtype(torch::kUInt8)
    );
    const auto stream = at::cuda::getCurrentCUDAStream(device_id_);
    if (count > 0) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            ids.data_ptr(),
            local_buffer_ + ring_ids_offset_,
            static_cast<size_t>(count) * sizeof(int64_t),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            records.data_ptr(),
            local_buffer_ + ring_payload_offset_,
            static_cast<size_t>(count) * record_stride_,
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }
    return {ids, records};
}

std::vector<torch::Tensor>
DlbP2PLoopbackRuntime::materialize_stage(int count) {
    TORCH_CHECK(synced_);
    TORCH_CHECK(0 <= count and count <= queue_capacity_);
    auto device = torch::TensorOptions().device(
        torch::kCUDA, device_id_
    );
    auto ids = torch::empty({count}, device.dtype(torch::kInt64));
    auto ready = torch::empty({count}, device.dtype(torch::kInt32));
    const auto stream = at::cuda::getCurrentCUDAStream(device_id_);
    if (count > 0) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            ids.data_ptr(),
            local_buffer_ + staging_ids_offset_,
            static_cast<size_t>(count) * sizeof(int64_t),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            ready.data_ptr(),
            local_buffer_ + stage_ready_offset_,
            static_cast<size_t>(count) * sizeof(int),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }
    return {ids, ready};
}

torch::Tensor DlbP2PLoopbackRuntime::quota_stats(
    const torch::Tensor& source_loads,
    int source_server,
    int round_id) {
    check_cuda_ints(source_loads, device_id_);
    auto output = torch::empty(
        {kRails + 2},
        torch::TensorOptions().
            device(torch::kCUDA, device_id_).
            dtype(torch::kInt64)
    );
    quota_stats_kernel<<<
        1, 1, 0, at::cuda::getCurrentCUDAStream(device_id_)
    >>>(
        source_loads.data_ptr<int>(),
        source_server,
        1 - source_server,
        source_rail_,
        round_id,
        output.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

int64_t DlbP2PLoopbackRuntime::get_record_stride() const {
    return static_cast<int64_t>(record_stride_);
}

void bind_dlb_p2p_loopback(pybind11::module_& module) {
    pybind11::class_<DlbP2PLoopbackRuntime>(
        module, "DlbP2PLoopbackRuntime"
    )
        .def(pybind11::init<int, int, int, int>())
        .def(
            "get_ipc_handle",
            &DlbP2PLoopbackRuntime::get_ipc_handle
        )
        .def("sync", &DlbP2PLoopbackRuntime::sync)
        .def("reset", &DlbP2PLoopbackRuntime::reset)
        .def("pack_off", &DlbP2PLoopbackRuntime::pack_off)
        .def("dispatch_on", &DlbP2PLoopbackRuntime::dispatch_on)
        .def("combine", &DlbP2PLoopbackRuntime::combine)
        .def(
            "clear_post_forward_counter",
            &DlbP2PLoopbackRuntime::clear_post_forward_counter
        )
        .def(
            "seed_receive_ring",
            &DlbP2PLoopbackRuntime::seed_receive_ring
        )
        .def("post_forward", &DlbP2PLoopbackRuntime::post_forward)
        .def(
            "materialize_ring",
            &DlbP2PLoopbackRuntime::materialize_ring
        )
        .def(
            "materialize_stage",
            &DlbP2PLoopbackRuntime::materialize_stage
        )
        .def("quota_stats", &DlbP2PLoopbackRuntime::quota_stats)
        .def(
            "get_record_stride",
            &DlbP2PLoopbackRuntime::get_record_stride
        );
}

}  // namespace deep_ep_lb_test
