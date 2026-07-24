#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "kernels/rail_lb.cuh"

#include <vector>

namespace {

void check_cuda_tensor(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), "input must be contiguous");
}

template <bool kCheck, typename T>
__device__ __forceinline__ void fill_or_check(
    T* values,
    size_t count,
    int seed,
    int* mismatches) {
    const auto thread =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = thread; index < count; index += stride) {
        const auto expected = static_cast<T>(seed + index % 97);
        if constexpr (kCheck) {
            if (values[index] != expected)
                atomicAdd(mismatches, 1);
        } else {
            values[index] = expected;
        }
    }
}

template <bool kCheck>
__global__ void handle_fields_kernel(
    void* workspace_buffer,
    int max_tokens,
    int num_tokens,
    int num_incoming,
    int num_outgoing,
    int num_servers,
    int num_channels,
    size_t record_stride,
    int* mismatches) {
    deep_ep::rail_lb::Workspace workspace(
        workspace_buffer,
        max_tokens,
        num_servers,
        num_channels,
        record_stride
    );
    const auto route_entries =
        static_cast<size_t>(NUM_MAX_NVL_PEERS) *
        num_servers * num_channels;
    fill_or_check<kCheck>(
        workspace.quotas,
        static_cast<size_t>(num_servers) * NUM_MAX_NVL_PEERS,
        11,
        mismatches
    );
    fill_or_check<kCheck>(
        workspace.source_channel_prefix,
        static_cast<size_t>(num_servers) * num_channels,
        23,
        mismatches
    );
    fill_or_check<kCheck>(
        workspace.route_counts, route_entries, 31, mismatches
    );
    fill_or_check<kCheck>(
        workspace.mask_counts,
        route_entries * NUM_MAX_NVL_PEERS,
        43,
        mismatches
    );
    fill_or_check<kCheck>(
        workspace.stage_offsets, route_entries, 53, mismatches
    );
    fill_or_check<kCheck>(
        workspace.repair_offsets, route_entries, 61, mismatches
    );
    fill_or_check<kCheck>(
        workspace.repair_heads, num_tokens, 71, mismatches
    );
    fill_or_check<kCheck>(
        workspace.virtual_token_indices,
        num_incoming,
        79,
        mismatches
    );
    fill_or_check<kCheck>(
        workspace.virtual_repair_slots,
        num_incoming,
        89,
        mismatches
    );
    fill_or_check<kCheck>(
        workspace.repair_next, num_outgoing, 97, mismatches
    );
    fill_or_check<kCheck>(
        workspace.virtual_source_rails,
        num_incoming,
        7,
        mismatches
    );
    fill_or_check<kCheck>(
        workspace.virtual_channels,
        num_incoming,
        13,
        mismatches
    );
    if (blockIdx.x == 0 and threadIdx.x == 0) {
        if constexpr (kCheck) {
            if (*workspace.num_tokens != num_tokens)
                atomicAdd(mismatches, 1);
            if (*workspace.num_outgoing != num_outgoing)
                atomicAdd(mismatches, 1);
        } else {
            *workspace.num_tokens = num_tokens;
            *workspace.num_outgoing = num_outgoing;
        }
    }
}

__global__ void compact_quota_kernel(
    const int* source_loads,
    int num_servers,
    int source_server,
    int round_id,
    int* quotas,
    int* balanced_loads) {
    const int destination_server = static_cast<int>(blockIdx.x);
    const int source_rail = static_cast<int>(threadIdx.x);
    if (destination_server >= num_servers or
        source_rail >= NUM_MAX_NVL_PEERS)
        return;

    const auto loads =
        source_loads + destination_server * NUM_MAX_NVL_PEERS;
    auto quota_row =
        quotas +
        (destination_server * NUM_MAX_NVL_PEERS + source_rail) *
            NUM_MAX_NVL_PEERS;
    deep_ep::rail_lb::compute_quota_row(
        loads,
        source_server,
        destination_server,
        source_rail,
        round_id,
        quota_row
    );

    #pragma unroll
    for (int selected = 0;
         selected < NUM_MAX_NVL_PEERS;
         ++selected) {
        atomicAdd(
            balanced_loads +
                destination_server * NUM_MAX_NVL_PEERS + selected,
            quota_row[selected]
        );
    }
}

}  // namespace

std::vector<torch::Tensor> run_dlb_compact_quota_cuda(
    const torch::Tensor& source_loads,
    int64_t source_server,
    int64_t round_id) {
    check_cuda_tensor(source_loads);
    TORCH_CHECK(
        source_loads.scalar_type() == torch::kInt32,
        "source_loads must use int32"
    );
    TORCH_CHECK(
        source_loads.dim() == 2 and
            source_loads.size(1) == NUM_MAX_NVL_PEERS,
        "source_loads must have shape [num_servers, num_rails]"
    );
    const int num_servers = static_cast<int>(source_loads.size(0));
    TORCH_CHECK(
        source_server >= 0 and source_server < num_servers,
        "source_server is out of range"
    );

    auto quotas = torch::empty(
        {
            num_servers,
            NUM_MAX_NVL_PEERS,
            NUM_MAX_NVL_PEERS,
        },
        source_loads.options()
    );
    auto balanced_loads = torch::zeros_like(source_loads);
    const auto stream = at::cuda::getCurrentCUDAStream();
    compact_quota_kernel<<<
        num_servers,
        NUM_MAX_NVL_PEERS,
        0,
        stream
    >>>(
        source_loads.data_ptr<int>(),
        num_servers,
        static_cast<int>(source_server),
        static_cast<int>(round_id),
        quotas.data_ptr<int>(),
        balanced_loads.data_ptr<int>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {quotas, balanced_loads};
}

std::vector<torch::Tensor> run_dlb_handle_roundtrip_cuda(
    int64_t max_tokens,
    int64_t num_tokens,
    int64_t num_incoming,
    int64_t num_outgoing,
    int64_t num_servers,
    int64_t num_channels) {
    TORCH_CHECK(max_tokens >= num_tokens and num_tokens >= 0);
    TORCH_CHECK(num_incoming >= 0 and num_outgoing >= 0);
    TORCH_CHECK(num_servers > 1 and num_channels > 0);
    constexpr size_t record_stride = sizeof(int4);
    const auto workspace_bytes =
        deep_ep::rail_lb::get_workspace_size(
            static_cast<int>(max_tokens),
            static_cast<int>(num_servers),
            static_cast<int>(num_channels),
            record_stride
        );
    const auto handle_bytes =
        deep_ep::rail_lb::get_handle_size(
            static_cast<int>(num_tokens),
            static_cast<int>(num_incoming),
            static_cast<int>(num_outgoing),
            static_cast<int>(num_servers),
            static_cast<int>(num_channels)
        );
    auto byte_options = torch::TensorOptions().
        device(torch::kCUDA).
        dtype(torch::kUInt8);
    auto workspace = torch::empty(
        {static_cast<int64_t>(workspace_bytes)}, byte_options
    );
    auto handle = torch::empty(
        {static_cast<int64_t>(handle_bytes)}, byte_options
    );
    auto mismatches = torch::zeros(
        {1}, byte_options.dtype(torch::kInt32)
    );
    const auto stream = at::cuda::getCurrentCUDAStream();
    handle_fields_kernel<false><<<32, 256, 0, stream>>>(
        workspace.data_ptr(),
        static_cast<int>(max_tokens),
        static_cast<int>(num_tokens),
        static_cast<int>(num_incoming),
        static_cast<int>(num_outgoing),
        static_cast<int>(num_servers),
        static_cast<int>(num_channels),
        record_stride,
        mismatches.data_ptr<int>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    deep_ep::rail_lb::save_handle(
        handle.data_ptr(),
        workspace.data_ptr(),
        static_cast<int>(num_tokens),
        static_cast<int>(num_incoming),
        static_cast<int>(num_outgoing),
        static_cast<int>(max_tokens),
        static_cast<int>(num_servers),
        static_cast<int>(num_channels),
        record_stride,
        stream
    );
    C10_CUDA_CHECK(cudaMemsetAsync(
        workspace.data_ptr(), 0, workspace_bytes, stream
    ));
    deep_ep::rail_lb::load_handle(
        workspace.data_ptr(),
        handle.data_ptr(),
        static_cast<int>(num_tokens),
        static_cast<int>(num_incoming),
        static_cast<int>(num_outgoing),
        static_cast<int>(max_tokens),
        static_cast<int>(num_servers),
        static_cast<int>(num_channels),
        record_stride,
        stream
    );
    handle_fields_kernel<true><<<32, 256, 0, stream>>>(
        workspace.data_ptr(),
        static_cast<int>(max_tokens),
        static_cast<int>(num_tokens),
        static_cast<int>(num_incoming),
        static_cast<int>(num_outgoing),
        static_cast<int>(num_servers),
        static_cast<int>(num_channels),
        record_stride,
        mismatches.data_ptr<int>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {handle, mismatches};
}
