#include "rail_lb.cuh"

#include "buffer.cuh"
#include "utils.cuh"

namespace deep_ep {
namespace rail_lb {

namespace {

template <typename T>
__device__ __forceinline__ void copy_handle_range(
        T* destination, const T* source, size_t count) {
    const auto thread =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = thread; index < count; index += stride)
        destination[index] = source[index];
}

template <bool kLoad>
__global__ void transfer_handle_kernel(
        void* local_workspace,
        void* handle_buffer,
        int num_tokens,
        int num_incoming,
        int num_outgoing,
        int max_tokens,
        int num_servers,
        int num_channels,
        size_t record_stride) {
    Workspace workspace(
        local_workspace,
        max_tokens,
        num_servers,
        num_channels,
        record_stride
    );
    HandleLayout handle(
        handle_buffer,
        num_tokens,
        num_incoming,
        num_outgoing,
        num_servers,
        num_channels
    );
    const auto route_entries =
        static_cast<size_t>(NUM_MAX_NVL_PEERS) *
        num_servers * num_channels;

    if constexpr (kLoad) {
        if (blockIdx.x == 0 and threadIdx.x == 0) {
            EP_DEVICE_ASSERT(handle.header[0] == HandleLayout::kVersion);
            EP_DEVICE_ASSERT(handle.header[1] == num_tokens);
            EP_DEVICE_ASSERT(handle.header[2] == num_incoming);
            EP_DEVICE_ASSERT(handle.header[3] == num_outgoing);
            *workspace.num_tokens = num_tokens;
            *workspace.num_outgoing = num_outgoing;
        }
        copy_handle_range(
            workspace.quotas,
            handle.quotas,
            static_cast<size_t>(num_servers) * NUM_MAX_NVL_PEERS
        );
        copy_handle_range(
            workspace.source_channel_prefix,
            handle.source_channel_prefix,
            static_cast<size_t>(num_servers) * num_channels
        );
        copy_handle_range(
            workspace.route_counts, handle.route_counts, route_entries
        );
        copy_handle_range(
            workspace.mask_counts,
            handle.mask_counts,
            route_entries * NUM_MAX_NVL_PEERS
        );
        copy_handle_range(
            workspace.stage_offsets, handle.stage_offsets, route_entries
        );
        copy_handle_range(
            workspace.repair_offsets, handle.repair_offsets, route_entries
        );
        copy_handle_range(
            workspace.repair_heads, handle.repair_heads, num_tokens
        );
        copy_handle_range(
            workspace.virtual_token_indices,
            handle.virtual_token_indices,
            num_incoming
        );
        copy_handle_range(
            workspace.virtual_repair_slots,
            handle.virtual_repair_slots,
            num_incoming
        );
        copy_handle_range(
            workspace.repair_next, handle.repair_next, num_outgoing
        );
        copy_handle_range(
            workspace.virtual_source_rails,
            handle.virtual_source_rails,
            num_incoming
        );
        copy_handle_range(
            workspace.virtual_channels,
            handle.virtual_channels,
            num_incoming
        );
    } else {
        if (blockIdx.x == 0 and threadIdx.x == 0) {
            handle.header[0] = HandleLayout::kVersion;
            handle.header[1] = num_tokens;
            handle.header[2] = num_incoming;
            handle.header[3] = num_outgoing;
        }
        copy_handle_range(
            handle.quotas,
            workspace.quotas,
            static_cast<size_t>(num_servers) * NUM_MAX_NVL_PEERS
        );
        copy_handle_range(
            handle.source_channel_prefix,
            workspace.source_channel_prefix,
            static_cast<size_t>(num_servers) * num_channels
        );
        copy_handle_range(
            handle.route_counts, workspace.route_counts, route_entries
        );
        copy_handle_range(
            handle.mask_counts,
            workspace.mask_counts,
            route_entries * NUM_MAX_NVL_PEERS
        );
        copy_handle_range(
            handle.stage_offsets, workspace.stage_offsets, route_entries
        );
        copy_handle_range(
            handle.repair_offsets, workspace.repair_offsets, route_entries
        );
        copy_handle_range(
            handle.repair_heads, workspace.repair_heads, num_tokens
        );
        copy_handle_range(
            handle.virtual_token_indices,
            workspace.virtual_token_indices,
            num_incoming
        );
        copy_handle_range(
            handle.virtual_repair_slots,
            workspace.virtual_repair_slots,
            num_incoming
        );
        copy_handle_range(
            handle.repair_next, workspace.repair_next, num_outgoing
        );
        copy_handle_range(
            handle.virtual_source_rails,
            workspace.virtual_source_rails,
            num_incoming
        );
        copy_handle_range(
            handle.virtual_channels,
            workspace.virtual_channels,
            num_incoming
        );
    }
}

__global__ void finalize_combined_heads_kernel(
        int* combined_rdma_head,
        void* local_workspace,
        int num_virtual_tokens,
        int max_tokens,
        int num_servers,
        int num_channels,
        size_t record_stride) {
    if (threadIdx.x != 0)
        return;
    const int destination_server =
        static_cast<int>(blockIdx.x) / num_channels;
    const int channel = static_cast<int>(blockIdx.x) % num_channels;
    Workspace workspace(
        local_workspace,
        max_tokens,
        num_servers,
        num_channels,
        record_stride
    );
    const int num_original_tokens = *workspace.num_tokens;

    int last_head = 1 << 25;
    for (int row = num_virtual_tokens - 1; row >= 0; --row) {
        int row_channel = 0;
        if (row < num_original_tokens) {
            int begin = 0, end = 0;
            get_channel_task_range(
                num_original_tokens,
                num_channels,
                channel,
                begin,
                end
            );
            if (row < begin or row >= end)
                continue;
            row_channel = channel;
        } else {
            const int moved = row - num_original_tokens;
            row_channel = workspace.virtual_channels[moved];
        }
        if (row_channel != channel)
            continue;

        auto head =
            combined_rdma_head + row * num_servers + destination_server;
        const int current = *head;
        if (current < 0)
            *head = -last_head - 1;
        else
            last_head = current;
    }
}

__global__ void activate_cached_kernel(void* local_workspace,
                                       int max_tokens,
                                       int num_servers,
                                       int num_channels,
                                       int round_id,
                                       size_t record_stride) {
    if (threadIdx.x != 0)
        return;
    Workspace workspace(
        local_workspace,
        max_tokens,
        num_servers,
        num_channels,
        record_stride
    );
    const int epoch = round_id + 1;
    *workspace.epoch = epoch;
    memory_fence();
    st_release_sys_global(workspace.schedule_ready, epoch);
}

}  // namespace

size_t get_record_stride(int hidden, int num_topk, int source_meta_bytes) {
    const size_t bytes =
        static_cast<size_t>(hidden) * sizeof(uint16_t) +
        static_cast<size_t>(source_meta_bytes) +
        static_cast<size_t>(num_topk) * (sizeof(int) + sizeof(float)) +
        static_cast<size_t>(ceil_div(hidden, 128)) * sizeof(float);
    return align(bytes, static_cast<size_t>(sizeof(int4)));
}

size_t get_workspace_size(int max_tokens,
                          int num_servers,
                          int num_channels,
                          size_t record_stride) {
    Workspace layout(
        reinterpret_cast<void*>(NUM_BUFFER_ALIGNMENT_BYTES),
        max_tokens,
        num_servers,
        num_channels,
        record_stride
    );
    return align(
        layout.total_bytes,
        static_cast<size_t>(NUM_BUFFER_ALIGNMENT_BYTES)
    );
}

size_t get_handle_size(int num_tokens,
                       int num_incoming,
                       int num_outgoing,
                       int num_servers,
                       int num_channels) {
    EP_HOST_ASSERT(
        num_tokens >= 0 and num_incoming >= 0 and num_outgoing >= 0
    );
    HandleLayout layout(
        reinterpret_cast<void*>(NUM_BUFFER_ALIGNMENT_BYTES),
        num_tokens,
        num_incoming,
        num_outgoing,
        num_servers,
        num_channels
    );
    return layout.total_bytes;
}

int get_handle_outgoing_count(size_t handle_bytes,
                              int num_tokens,
                              int num_incoming,
                              int num_servers,
                              int num_channels) {
    const auto base_bytes = get_handle_size(
        num_tokens, num_incoming, 0, num_servers, num_channels
    );
    EP_HOST_ASSERT(handle_bytes >= base_bytes);
    const auto outgoing_bytes = handle_bytes - base_bytes;
    EP_HOST_ASSERT(outgoing_bytes % sizeof(int) == 0);
    return static_cast<int>(outgoing_bytes / sizeof(int));
}

void save_handle(void* destination,
                 const void* local_workspace,
                 int num_tokens,
                 int num_incoming,
                 int num_outgoing,
                 int max_tokens,
                 int num_servers,
                 int num_channels,
                 size_t record_stride,
                 cudaStream_t stream) {
    EP_HOST_ASSERT(destination != nullptr and local_workspace != nullptr);
    EP_HOST_ASSERT(
        num_tokens <= max_tokens and
        num_incoming >= 0 and num_outgoing >= 0
    );
    const auto handle_bytes = get_handle_size(
        num_tokens, num_incoming, num_outgoing,
        num_servers, num_channels
    );
    const auto blocks = static_cast<int>(max(
        static_cast<size_t>(1),
        ceil_div(handle_bytes, sizeof(int4) * 256)
    ));
    transfer_handle_kernel<false><<<blocks, 256, 0, stream>>>(
        const_cast<void*>(local_workspace),
        destination,
        num_tokens,
        num_incoming,
        num_outgoing,
        max_tokens,
        num_servers,
        num_channels,
        record_stride
    );
    CUDA_CHECK(cudaGetLastError());
}

void load_handle(void* local_workspace,
                 const void* source,
                 int num_tokens,
                 int num_incoming,
                 int num_outgoing,
                 int max_tokens,
                 int num_servers,
                 int num_channels,
                 size_t record_stride,
                 cudaStream_t stream) {
    EP_HOST_ASSERT(local_workspace != nullptr and source != nullptr);
    EP_HOST_ASSERT(
        num_tokens <= max_tokens and
        num_incoming >= 0 and num_outgoing >= 0
    );
    const auto handle_bytes = get_handle_size(
        num_tokens, num_incoming, num_outgoing,
        num_servers, num_channels
    );
    const auto blocks = static_cast<int>(max(
        static_cast<size_t>(1),
        ceil_div(handle_bytes, sizeof(int4) * 256)
    ));
    transfer_handle_kernel<true><<<blocks, 256, 0, stream>>>(
        local_workspace,
        const_cast<void*>(source),
        num_tokens,
        num_incoming,
        num_outgoing,
        max_tokens,
        num_servers,
        num_channels,
        record_stride
    );
    CUDA_CHECK(cudaGetLastError());
}

void activate_cached(void* local_workspace,
                     int max_tokens,
                     int num_servers,
                     int num_channels,
                     int round_id,
                     size_t record_stride,
                     cudaStream_t stream) {
    EP_HOST_ASSERT(local_workspace != nullptr);
    activate_cached_kernel<<<1, 1, 0, stream>>>(
        local_workspace,
        max_tokens,
        num_servers,
        num_channels,
        round_id,
        record_stride
    );
    CUDA_CHECK(cudaGetLastError());
}

void finalize_combined_heads(int* combined_rdma_head,
                             void* local_workspace,
                             int num_virtual_tokens,
                             int max_tokens,
                             int num_servers,
                             int num_channels,
                             size_t record_stride,
                             cudaStream_t stream) {
    EP_HOST_ASSERT(
        combined_rdma_head != nullptr and local_workspace != nullptr
    );
    finalize_combined_heads_kernel<<<
        num_servers * num_channels,
        32,
        0,
        stream
    >>>(
        combined_rdma_head,
        local_workspace,
        num_virtual_tokens,
        max_tokens,
        num_servers,
        num_channels,
        record_stride
    );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace rail_lb
}  // namespace deep_ep
