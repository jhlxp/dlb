#pragma once

#include "dlb_alltoall/dlb_rail_protocol.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace dlb_alltoall {

cudaError_t launch_dlb_count_moe_routes(
    const std::int64_t* topk_idx,
    std::uint64_t route_count,
    std::uint32_t num_topk,
    std::uint32_t num_experts,
    std::uint32_t num_local_experts,
    std::uint32_t world_size,
    std::uint64_t* destination_counts,
    std::uint64_t* invalid_route_count,
    cudaStream_t stream);

cudaError_t launch_dlb_count_received_experts(
    const std::uint8_t* repair_records,
    std::uint64_t record_count,
    std::uint64_t record_bytes,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint32_t first_local_expert,
    std::uint32_t num_local_experts,
    std::uint64_t* expert_counts,
    std::uint64_t* group_count,
    cudaStream_t stream);

cudaError_t launch_dlb_scatter_received_experts(
    const std::uint8_t* repair_records,
    std::uint64_t record_count,
    std::uint64_t record_bytes,
    std::uint64_t hidden_bytes,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint32_t first_local_expert,
    std::uint32_t num_local_experts,
    const std::uint64_t* expert_offsets,
    std::uint64_t* expert_cursors,
    std::uint8_t* recv_x,
    std::int64_t* recv_headers,
    float* recv_weights,
    bool* valid_mask,
    std::uint64_t output_records,
    std::uint64_t* group_cursor,
    std::uint64_t* group_output_indices,
    std::uint64_t group_capacity,
    cudaStream_t stream);

// GPU-resident DLB planner. It reproduces the reference scheduler's
// source-local surplus/deficit and direct-priority assignment, giving every
// Rail either floor(N/M) or ceil(N/M) records without host action tables.
cudaError_t launch_dlb_build_dynamic_rail_plan(
    const std::uint64_t* local_server_demand,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t local_rail,
    std::uint32_t round_id,
    std::uint32_t channel_count,
    std::uint64_t record_bytes,
    std::uint64_t receive_slot_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t* flow_rail_counts,
    DlbRailTransfer* transfers,
    std::uint32_t transfer_count,
    std::uint64_t* rail_record_counts,
    std::uint64_t* channel_record_counts,
    cudaStream_t stream);

cudaError_t launch_dlb_pack_moe_direct(
    const void* x,
    std::uint64_t num_tokens,
    std::uint64_t hidden_bytes,
    const std::int64_t* topk_idx,
    const float* topk_weights,
    std::uint32_t num_topk,
    std::uint32_t num_experts,
    std::uint32_t world_size,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t source_local_rank,
    std::uint32_t source_rank,
    std::uint32_t round_id,
    std::uint64_t epoch,
    std::uint64_t record_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t repair_slot_bytes,
    std::uint64_t repair_epoch_offset_bytes,
    const std::uint64_t* local_server_demand,
    const std::uint64_t* flow_rail_counts,
    std::uint64_t* destination_cursors,
    std::uint8_t* const* rail_send_buffers,
    std::uint8_t* const* repair_buffers,
    cudaStream_t stream);

cudaError_t launch_dlb_count_combine_routes(
    const std::uint64_t* group_output_indices,
    std::uint64_t group_count,
    const std::int64_t* headers,
    std::uint64_t output_record_count,
    std::uint32_t world_size,
    std::uint64_t* destination_counts,
    cudaStream_t stream);

cudaError_t launch_dlb_pack_combine_direct(
    const void* x,
    std::uint64_t hidden_elements,
    std::uint32_t scalar_type,
    const std::int64_t* headers,
    const float* weights,
    std::uint64_t output_record_count,
    const std::uint64_t* group_output_indices,
    std::uint64_t group_count,
    bool apply_weights,
    std::uint32_t world_size,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t source_local_rank,
    std::uint32_t source_rank,
    std::uint64_t epoch,
    std::uint64_t record_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t repair_slot_bytes,
    std::uint64_t repair_epoch_offset_bytes,
    const std::uint64_t* flow_rail_counts,
    std::uint64_t* destination_cursors,
    std::uint8_t* const* rail_send_buffers,
    std::uint8_t* const* repair_buffers,
    cudaStream_t stream);

// Rebuilds the original token order directly from returned combine records.
// Accumulation is FP32; the binding casts once to the public activation dtype.
cudaError_t launch_dlb_accumulate_combined_records(
    const std::uint8_t* repair_records,
    std::uint64_t repair_record_count,
    std::uint64_t record_bytes,
    std::uint64_t hidden_elements,
    std::uint32_t scalar_type,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint64_t num_tokens,
    std::uint32_t num_topk,
    float* combined,
    float* combined_weights,
    cudaStream_t stream);

}  // namespace dlb_alltoall
