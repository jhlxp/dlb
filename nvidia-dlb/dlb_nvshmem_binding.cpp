#include "dlb_alltoall/dlb_alltoall_nvshmem.h"
#include "dlb_alltoall/dlb_moe_runtime.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

at::Tensor get_dlb_nvshmem_init_id() {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    TORCH_CHECK(nvshmemx_get_uniqueid(&id) == NVSHMEMX_SUCCESS,
                "getting DLB NVSHMEM unique id failed");
    return at::from_blob(&id, sizeof(id), at::kByte).clone();
}

at::Tensor dlb_cuda_module_probe(int64_t device) {
    TORCH_CHECK(device >= 0, "DLB CUDA module probe requires a non-negative device");
    check_cuda(cudaSetDevice(static_cast<int>(device)), "selecting DLB CUDA probe device");
    std::uint32_t* device_result = nullptr;
    std::uint32_t host_result = 0;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(host_result)),
               "allocating DLB CUDA module probe");
    const cudaError_t launch_status =
        dlb_alltoall::launch_dlb_cuda_module_probe(device_result, nullptr);
    if (launch_status != cudaSuccess) {
        cudaFree(device_result);
        check_cuda(launch_status, "launching DLB CUDA module probe");
    }
    check_cuda(cudaMemcpy(&host_result, device_result, sizeof(host_result), cudaMemcpyDeviceToHost),
               "reading DLB CUDA module probe");
    check_cuda(cudaFree(device_result), "freeing DLB CUDA module probe");
    TORCH_CHECK(host_result == 0x444c42u, "DLB CUDA module probe returned an invalid value");
    return at::scalar_tensor(static_cast<std::int64_t>(host_result),
                             at::TensorOptions().dtype(at::kLong));
}

// Owns the persistent topology, buffers, and communication state of one rank.
// Each rank has its own instance and collects only source-server-local demand;
// no instance stores a cluster-wide demand matrix or global schedule.
struct dlb_nvshmem_comm_t : torch::CustomClassHolder {
    std::uint32_t rank;
    std::uint32_t world;
    std::uint32_t rails;
    std::uint32_t device;
    std::uint32_t server;
    std::uint32_t local_rank;
    // Remains invalid until the server-local NVSHMEM team is created successfully.
    nvshmem_team_t local_team = NVSHMEM_TEAM_INVALID;
    dlb_alltoall::DlbRuntime runtime{};
    bool initialized = false;
    bool closed = false;
    // `last_dispatch_epoch` orders every completed DLB communication phase
    // (dispatch or combine). The submitted/finished pair separately tracks
    // the one dispatch that may be posted without being materialized yet.
    std::uint64_t last_dispatch_epoch = 0;
    std::uint64_t submitted_dispatch_epoch = 0;
    std::uint64_t finished_dispatch_epoch = 0;
    bool dispatch_pending = false;
    at::Tensor pending_dispatch_x;
    at::Tensor pending_dispatch_topk_idx;
    at::Tensor pending_dispatch_topk_weights;
    std::uint32_t pending_dispatch_num_experts = 0;
    std::int64_t pending_dispatch_expert_alignment = 0;
    // The runtime's repair-ready event protects the writer. This additional
    // event protects the final scatter reader before a later dispatch reuses
    // the same repair slot from another CUDA stream.
    cudaEvent_t dispatch_materialize_ready = nullptr;
    bool dispatch_materialize_ready_recorded = false;
    std::uint64_t* symmetric_demand_row = nullptr;
    std::uint64_t* symmetric_demand_matrix = nullptr;
    std::uint64_t* device_destination_cursors = nullptr;
    std::uint64_t* device_flow_rail_counts = nullptr;
    std::uint64_t* device_rail_record_counts = nullptr;
    std::uint64_t* device_channel_record_counts = nullptr;
    std::uint64_t* device_invalid_route_count = nullptr;
    std::uint64_t* host_invalid_route_count = nullptr;
    bool profiling_enabled = false;
    int last_profile_kind = 0;  // 0: none/read, 1: dispatch, 2: combine
    std::uint64_t last_profile_epoch = 0;

    // Validates the runtime configuration and initializes rank-local resources.
    dlb_nvshmem_comm_t(int64_t rank_value, int64_t rails_value, int64_t world_value,
                        int64_t device_value, at::Tensor uid,
                        int64_t record_bytes_value,
                        int64_t receive_slot_bytes_value,
                        int64_t repair_slot_bytes_value,
                        int64_t pipeline_depth_value,
                        int64_t chunk_bytes_value,
                        int64_t num_comm_sms_value,
                        int64_t rail_send_capacity_bytes_value,
                        bool enable_profiling_value,
                        bool use_loopback_transport_value)
        : rank(static_cast<std::uint32_t>(rank_value)),
          world(static_cast<std::uint32_t>(world_value)),
          rails(static_cast<std::uint32_t>(rails_value)),
          device(static_cast<std::uint32_t>(device_value)) {
        profiling_enabled = enable_profiling_value;
        TORCH_CHECK(rank_value >= 0 && rails_value > 0 && world_value > 0 && device_value >= 0,
                    "DLB rank, topology, and device arguments are invalid");
        TORCH_CHECK(rank_value <= std::numeric_limits<std::uint32_t>::max() &&
                        rails_value <= std::numeric_limits<std::uint32_t>::max() &&
                        world_value <= std::numeric_limits<std::uint32_t>::max() &&
                        device_value <= std::numeric_limits<std::uint32_t>::max(),
                    "DLB rank, topology, and device arguments must fit uint32_t");
        TORCH_CHECK(rails > 0 && world % rails == 0 && world / rails >= 2 && rank < world,
                    "DLB requires an SxM topology with S >= 2 and world == S * rails");
        TORCH_CHECK(record_bytes_value > 0 && receive_slot_bytes_value > 0 &&
                        repair_slot_bytes_value > 0 && pipeline_depth_value > 0 &&
                        chunk_bytes_value > 0 && num_comm_sms_value > 0 &&
                        num_comm_sms_value % 2 == 0 &&
                        rail_send_capacity_bytes_value > 0,
                    "DLB runtime capacities must be positive");
        TORCH_CHECK(pipeline_depth_value <= std::numeric_limits<std::uint32_t>::max(),
                    "DLB pipeline depth must fit uint32_t");
        TORCH_CHECK(uid.device().is_cpu() && uid.scalar_type() == at::kByte &&
                        uid.numel() == sizeof(nvshmemx_uniqueid_t),
                    "uid must be a CPU NVSHMEM unique-id byte tensor");
        // Map the global rank to its logical server and Rail endpoint.
        server = rank / rails;
        local_rank = rank % rails;
        check_cuda(cudaSetDevice(device), "selecting DLB CUDA device");
        int device_sm_count = 0;
        check_cuda(cudaDeviceGetAttribute(&device_sm_count,
                                          cudaDevAttrMultiProcessorCount,
                                          static_cast<int>(device)),
                   "querying DLB CUDA SM count");
        TORCH_CHECK(num_comm_sms_value <= device_sm_count,
                    "num_comm_sms exceeds the CUDA device SM count");
        // Pair one sender CTA with one receiver CTA for every Rail channel.
        const std::uint32_t rail_channel_count =
            static_cast<std::uint32_t>(num_comm_sms_value / 2);

        // Join all EP ranks into one NVSHMEM world using the broadcast unique ID.
        nvshmemx_uniqueid_t id;
        std::memcpy(&id, uid.data_ptr(), sizeof(id));
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_set_attr_uniqueid_args(rank, world, &id, &attr);
        TORCH_CHECK(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) == NVSHMEMX_SUCCESS,
                    "initializing DLB NVSHMEM runtime failed");
        // Group the contiguous ranks of this logical server for local demand collection.
        TORCH_CHECK(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, server * rails, 1, rails,
                                                nullptr, 0, &local_team) == NVSHMEMX_SUCCESS,
                    "creating DLB local NVSHMEM team failed");

        // Freeze the validated topology, capacities, and backend policy for the runtime.
        const dlb_alltoall::DlbRuntimeConfig config = {
            rank, world, rails,
            static_cast<std::uint64_t>(record_bytes_value),
            static_cast<std::uint64_t>(receive_slot_bytes_value),
            static_cast<std::uint64_t>(repair_slot_bytes_value),
            1,
            static_cast<std::uint32_t>(pipeline_depth_value),
            static_cast<std::uint64_t>(chunk_bytes_value),
            rail_channel_count,
            static_cast<std::uint64_t>(rail_send_capacity_bytes_value),
            profiling_enabled,
            use_loopback_transport_value,
            local_team,
        };
        // Allocate the persistent transport resources owned by this rank.
        check_cuda(dlb_alltoall::initialize_dlb_nvshmem_rdma_runtime(&runtime, config),
                   "initializing DLB NVSHMEM/RDMA backend");
        symmetric_demand_row = static_cast<std::uint64_t*>(
            nvshmem_calloc(world, sizeof(std::uint64_t)));
        symmetric_demand_matrix = static_cast<std::uint64_t*>(
            nvshmem_calloc(static_cast<std::size_t>(rails) * world,
                           sizeof(std::uint64_t)));
        TORCH_CHECK(symmetric_demand_row != nullptr && symmetric_demand_matrix != nullptr,
                    "allocating DLB symmetric demand buffers failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_destination_cursors),
                              static_cast<std::size_t>(world) * sizeof(std::uint64_t)),
                   "allocating DLB destination cursors");
        const std::size_t remote_servers = runtime.server_count - 1;
        const std::size_t flow_count = remote_servers * rails * rails * rails;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_flow_rail_counts),
                              flow_count * sizeof(std::uint64_t)),
                   "allocating GPU-resident DLB flow plan");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_rail_record_counts),
                              static_cast<std::size_t>(rails) * sizeof(std::uint64_t)),
                   "allocating GPU-resident DLB Rail counters");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_channel_record_counts),
                              static_cast<std::size_t>(rail_channel_count) *
                                  sizeof(std::uint64_t)),
                   "allocating GPU-resident DLB channel counters");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_invalid_route_count),
                              sizeof(std::uint64_t)),
                   "allocating DLB invalid-route counter");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&host_invalid_route_count),
                                 sizeof(std::uint64_t), cudaHostAllocPortable),
                   "allocating DLB pinned invalid-route counter");
        check_cuda(cudaEventCreateWithFlags(&dispatch_materialize_ready,
                                            cudaEventDisableTiming),
                   "creating DLB dispatch materialization event");
        const std::size_t transfer_count =
            remote_servers * rails * rail_channel_count;
        TORCH_CHECK(transfer_count <= std::numeric_limits<std::uint32_t>::max(),
                    "DLB dynamic transfer count exceeds uint32_t");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&runtime.device_transfers),
                              transfer_count * sizeof(dlb_alltoall::DlbRailTransfer)),
                   "allocating GPU-resident DLB transfer plan");
        runtime.device_transfer_count = static_cast<std::uint32_t>(transfer_count);
        TORCH_CHECK(remote_servers <=
                        runtime.config.rail_send_capacity_bytes /
                            runtime.config.receive_slot_bytes,
                    "DLB Rail capacity cannot hold fixed remote-server slots");
        nvshmem_barrier_all();
        initialized = true;
    }

    void validate_moe_inputs(const at::Tensor& x, const at::Tensor& topk_idx,
                             const at::Tensor& topk_weights,
                             std::uint32_t num_experts) const {
        TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
                    "x must be a contiguous rank-2 CUDA tensor");
        TORCH_CHECK(x.get_device() == static_cast<int64_t>(device),
                    "x must live on the communicator CUDA device");
        TORCH_CHECK(x.scalar_type() == at::kBFloat16 ||
                        x.scalar_type() == at::kHalf ||
                        x.scalar_type() == at::kFloat,
                    "x must use BF16, FP16, or FP32");
        TORCH_CHECK(topk_idx.is_cuda() && topk_idx.is_contiguous() &&
                        topk_idx.scalar_type() == at::kLong && topk_idx.dim() == 2 &&
                        topk_idx.size(0) == x.size(0),
                    "topk_idx must be a contiguous CUDA int64 [tokens, topk] tensor");
        TORCH_CHECK(topk_weights.is_cuda() && topk_weights.is_contiguous() &&
                        topk_weights.scalar_type() == at::kFloat &&
                        topk_weights.sizes() == topk_idx.sizes(),
                    "topk_weights must be contiguous CUDA float32 matching topk_idx");
        TORCH_CHECK(topk_idx.get_device() == static_cast<int64_t>(device) &&
                        topk_weights.get_device() == static_cast<int64_t>(device),
                    "MoE routing tensors must live on the communicator CUDA device");
        TORCH_CHECK(num_experts > 0 && num_experts % world == 0,
                    "num_experts must be divisible by the DLB world size");
        TORCH_CHECK(topk_idx.size(1) > 0 && topk_idx.size(1) <= 8 &&
                        topk_idx.size(1) <=
                            std::numeric_limits<std::uint32_t>::max(),
                    "DLB rank-deduplicated dispatch supports topk in [1, 8]");
        const std::uint64_t hidden_bytes =
            static_cast<std::uint64_t>(x.size(1)) * x.element_size();
        TORCH_CHECK(runtime.config.record_bytes >= 128 + hidden_bytes,
                    "DLB record size is too small for the hidden payload");
        TORCH_CHECK(static_cast<std::uint64_t>(x.size(0)) <=
                        std::numeric_limits<std::uint64_t>::max() /
                            static_cast<std::uint64_t>(topk_idx.size(1)),
                    "MoE route count overflows uint64_t");
    }

    // Converts local MoE routing decisions into transport-ready DLB records.
    //
    // This method reuses the pipeline slot selected by epoch, collects the
    // server-local demand tile, builds a Rail-balanced device plan, and packs
    // rank-deduplicated records directly into the selected transport buffers.
    // All work is enqueued on caller_stream without a host-side synchronization.
    void prepare_moe_direct(at::Tensor x, at::Tensor topk_idx,
                            at::Tensor topk_weights, std::uint64_t epoch,
                            std::uint32_t round_id, std::uint32_t num_experts,
                            cudaStream_t caller_stream) {
        validate_moe_inputs(x, topk_idx, topk_weights, num_experts);
        const std::uint32_t local_experts = num_experts / world;
        const std::uint64_t route_count =
            static_cast<std::uint64_t>(x.size(0)) * topk_idx.size(1);
        const std::uint64_t hidden_bytes =
            static_cast<std::uint64_t>(x.size(1)) * x.element_size();
        const std::uint32_t slot = static_cast<std::uint32_t>(
            epoch % runtime.config.pipeline_depth);

        // Direct packing writes the reusable Rail and repair slots itself, so
        // it must observe the same slot-lifetime dependencies as the transport
        // pipeline before touching either buffer.
        check_cuda(cudaStreamWaitEvent(caller_stream, runtime.repair_ready[slot], 0),
                   "waiting for reusable DLB repair slot");
        check_cuda(cudaStreamWaitEvent(caller_stream, runtime.transport_ready[slot], 0),
                   "waiting for reusable DLB transport slot");
        check_cuda(cudaMemsetAsync(
                       symmetric_demand_row, 0,
                       static_cast<std::size_t>(world) * sizeof(std::uint64_t),
                       caller_stream),
                   "clearing GPU-resident DLB demand row");
        check_cuda(cudaMemsetAsync(device_invalid_route_count, 0,
                                   sizeof(std::uint64_t), caller_stream),
                   "clearing DLB invalid-route counter");
        // Build this rank's demand row, then concatenate all rows from the
        // server-local NVSHMEM team into the M-by-world demand matrix.
        check_cuda(dlb_alltoall::launch_dlb_count_moe_routes(
                       topk_idx.data_ptr<std::int64_t>(), route_count,
                       static_cast<std::uint32_t>(topk_idx.size(1)), num_experts,
                       local_experts, world, symmetric_demand_row,
                       device_invalid_route_count, caller_stream),
                   "counting GPU-resident DLB MoE routes");
        TORCH_CHECK(nvshmemx_uint64_fcollect_on_stream(
                        local_team, symmetric_demand_matrix, symmetric_demand_row,
                        world, caller_stream) == NVSHMEMX_SUCCESS,
                    "collecting DLB server-local demand on GPU failed");

        runtime.device_transfer_count = static_cast<std::uint32_t>(
            (runtime.server_count - 1) * rails *
            runtime.config.rail_channel_count);
        check_cuda(dlb_alltoall::launch_dlb_build_dynamic_rail_plan(
                       symmetric_demand_matrix, runtime.server_count, rails, server,
                       local_rank, round_id, runtime.config.rail_channel_count,
                       runtime.config.record_bytes,
                       runtime.config.receive_slot_bytes,
                       runtime.config.receive_slot_bytes, device_flow_rail_counts,
                       runtime.device_transfers, runtime.device_transfer_count,
                       device_rail_record_counts, device_channel_record_counts,
                       caller_stream),
                   "building GPU-resident DLB Rail plan");
        check_cuda(cudaMemsetAsync(
                       device_destination_cursors, 0,
                       static_cast<std::size_t>(world) * sizeof(std::uint64_t),
                       caller_stream),
                   "clearing GPU-resident DLB pack cursors");
        const std::uint64_t repair_epoch_offset =
            static_cast<std::uint64_t>(slot) * world *
            runtime.config.repair_slot_bytes;
        // The device pointer table exposes every local Rail's symmetric send
        // slot, allowing this source GPU to stage directly to selected_rail.
        check_cuda(dlb_alltoall::launch_dlb_pack_moe_direct(
                       x.data_ptr(), static_cast<std::uint64_t>(x.size(0)),
                       hidden_bytes, topk_idx.data_ptr<std::int64_t>(),
                       topk_weights.data_ptr<float>(),
                       static_cast<std::uint32_t>(topk_idx.size(1)), num_experts,
                       world, runtime.server_count, rails, server, local_rank, rank,
                       round_id, epoch, runtime.config.record_bytes,
                       runtime.config.receive_slot_bytes,
                       runtime.config.repair_slot_bytes, repair_epoch_offset,
                       symmetric_demand_matrix, device_flow_rail_counts,
                       device_destination_cursors,
                       reinterpret_cast<std::uint8_t* const*>(
                           runtime.rail_send_tables[slot].device_ptrs),
                       reinterpret_cast<std::uint8_t* const*>(
                           runtime.repair_table.device_ptrs),
                       caller_stream),
                   "direct-packing DLB MoE records on GPU");
    }

    at::Tensor benchmark_prepare_moe_device(at::Tensor x, at::Tensor topk_idx,
                                  at::Tensor topk_weights, int64_t epoch_value,
                                  int64_t round_id_value,
                                  int64_t num_experts_value) {
        TORCH_CHECK(!closed && initialized, "DLB communicator is closed");
        TORCH_CHECK(epoch_value > 0 && round_id_value >= 0 &&
                        round_id_value <=
                            std::numeric_limits<std::uint32_t>::max() &&
                        num_experts_value > 0 &&
                        num_experts_value <=
                            std::numeric_limits<std::uint32_t>::max(),
                    "DLB device prepare scalar arguments are invalid");
        check_cuda(cudaSetDevice(device), "selecting DLB device-prepare device");
        const cudaStream_t caller_stream =
            at::cuda::getCurrentCUDAStream(device).stream();
        prepare_moe_direct(
            x, topk_idx, topk_weights,
            static_cast<std::uint64_t>(epoch_value),
            static_cast<std::uint32_t>(round_id_value),
            static_cast<std::uint32_t>(num_experts_value), caller_stream);
        at::Tensor result = at::empty(
            {static_cast<int64_t>(rails)},
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device));
        check_cuda(cudaMemcpyAsync(
                       result.data_ptr(), device_rail_record_counts,
                       static_cast<std::size_t>(rails) * sizeof(std::uint64_t),
                       cudaMemcpyDeviceToDevice, caller_stream),
                   "copying device-prepared DLB Rail counters");
        return result;
    }

    // Posts the asynchronous transport portion of one MoE dispatch.
    //
    // Args:
    //   x: Contiguous [tokens, hidden] CUDA payload tensor.
    //   topk_idx: Contiguous [tokens, topk] CUDA expert IDs.
    //   topk_weights: Contiguous [tokens, topk] CUDA router weights.
    //   epoch_value: Globally ordered communication epoch for this dispatch.
    //   round_id_value: Deterministic Rail-balancing rotation identifier.
    //   num_experts_value: Total expert count across all DLB ranks.
    //   expert_alignment_value: Output alignment retained for finish_dispatch_moe.
    //
    // This method validates, plans, packs, and launches transport without
    // waiting for inbound repair or materializing expert-major outputs. Only
    // one posted dispatch may exist at a time because its metadata and repair
    // buffer are owned by this communicator instance until it is finished.
    void post_dispatch_moe(
        at::Tensor x, at::Tensor topk_idx, at::Tensor topk_weights,
        int64_t epoch_value, int64_t round_id_value, int64_t num_experts_value,
        int64_t expert_alignment_value) {
        TORCH_CHECK(!closed && initialized, "DLB communicator is closed");
        TORCH_CHECK(!dispatch_pending,
                    "DLB permits only one outstanding dispatch; call "
                    "finish_dispatch_moe before posting another dispatch");
        TORCH_CHECK(epoch_value > 0 &&
                        static_cast<std::uint64_t>(epoch_value) == last_dispatch_epoch + 1,
                    "DLB dispatch epochs must start at 1 and increase by one");
        TORCH_CHECK(static_cast<std::uint64_t>(epoch_value) > submitted_dispatch_epoch,
                    "DLB dispatch epoch must exceed the previously submitted epoch");
        TORCH_CHECK(round_id_value >= 0 &&
                        round_id_value <= std::numeric_limits<std::uint32_t>::max(),
                    "round_id must fit uint32_t");
        TORCH_CHECK(num_experts_value > 0 &&
                        num_experts_value <= std::numeric_limits<std::uint32_t>::max(),
                    "num_experts must fit uint32_t");
        TORCH_CHECK(expert_alignment_value > 0,
                    "expert_alignment must be positive");
        check_cuda(cudaSetDevice(device), "selecting DLB dispatch device");
        const cudaStream_t caller_stream =
            at::cuda::getCurrentCUDAStream(device).stream();
        const std::uint64_t epoch = static_cast<std::uint64_t>(epoch_value);
        const std::uint32_t profile_slot = static_cast<std::uint32_t>(
            epoch % runtime.config.pipeline_depth);
        if (profiling_enabled) {
            check_cuda(cudaEventRecord(runtime.profile_dispatch_begin[profile_slot],
                                       caller_stream),
                       "recording DLB dispatch profile start");
        }
        // A prior finish may have enqueued scatter on another caller stream.
        // Do not overwrite its repair slot until that reader has completed.
        if (dispatch_materialize_ready_recorded) {
            check_cuda(cudaStreamWaitEvent(caller_stream, dispatch_materialize_ready, 0),
                       "waiting for prior DLB dispatch materialization");
        }
        prepare_moe_direct(
            x, topk_idx, topk_weights, epoch,
            static_cast<std::uint32_t>(round_id_value),
            static_cast<std::uint32_t>(num_experts_value), caller_stream);
        check_cuda(dlb_alltoall::launch_dlb_nvshmem_rdma(
                       &runtime, epoch, caller_stream),
                   "launching fused DLB dispatch");

        // Keep source storage alive until finish has established that repair is
        // complete. This permits callers to release inputs after posting.
        pending_dispatch_x = std::move(x);
        pending_dispatch_topk_idx = std::move(topk_idx);
        pending_dispatch_topk_weights = std::move(topk_weights);
        pending_dispatch_num_experts = static_cast<std::uint32_t>(num_experts_value);
        pending_dispatch_expert_alignment = expert_alignment_value;
        submitted_dispatch_epoch = epoch;
        dispatch_pending = true;
    }

    // Materializes a previously posted dispatch after inbound repair is ready.
    //
    // This helper contains only the finish phase. Source preparation and the
    // asynchronous transport launch are performed by post_dispatch_moe.
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor> materialize_posted_dispatch(
        at::Tensor x, int64_t epoch_value, int64_t num_experts_value,
        int64_t expert_alignment_value) {
        TORCH_CHECK(!closed && initialized, "DLB communicator is closed");
        TORCH_CHECK(dispatch_pending,
                    "DLB materialization requires a posted dispatch");
        TORCH_CHECK(epoch_value > 0 &&
                        static_cast<std::uint64_t>(epoch_value) == submitted_dispatch_epoch,
                    "DLB materialization epoch does not match the posted dispatch");
        TORCH_CHECK(num_experts_value == pending_dispatch_num_experts &&
                        expert_alignment_value == pending_dispatch_expert_alignment,
                    "DLB materialization metadata does not match the posted dispatch");
        check_cuda(cudaSetDevice(device), "selecting DLB dispatch device");
        const cudaStream_t caller_stream =
            at::cuda::getCurrentCUDAStream(device).stream();
        const std::uint64_t epoch = static_cast<std::uint64_t>(epoch_value);
        const std::uint32_t profile_slot = static_cast<std::uint32_t>(
            epoch % runtime.config.pipeline_depth);
        std::uint8_t* repair =
            dlb_alltoall::dlb_nvshmem_repair_buffer_for_epoch(&runtime, epoch);
        const std::size_t output_bytes =
            static_cast<std::size_t>(world) * runtime.config.repair_slot_bytes;
        check_cuda(dlb_alltoall::wait_dlb_nvshmem_rdma_epoch(
                       &runtime, epoch, caller_stream),
                   "waiting for fused DLB dispatch");

        const std::uint32_t local_experts =
            static_cast<std::uint32_t>(num_experts_value) / world;
        const std::uint64_t hidden_bytes =
            static_cast<std::uint64_t>(x.size(1)) * x.element_size();
        TORCH_CHECK(output_bytes % runtime.config.record_bytes == 0,
                    "DLB repair buffer is not record aligned");
        const std::uint64_t repair_record_count =
            output_bytes / runtime.config.record_bytes;
        at::Tensor device_counts = at::zeros(
            {static_cast<int64_t>(local_experts)},
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device));
        at::Tensor device_group_count = at::zeros(
            {1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device));
        check_cuda(dlb_alltoall::launch_dlb_count_received_experts(
                       repair, repair_record_count, runtime.config.record_bytes,
                       epoch, rank, rank * local_experts, local_experts,
                       reinterpret_cast<std::uint64_t*>(device_counts.data_ptr<std::int64_t>()),
                       reinterpret_cast<std::uint64_t*>(
                           device_group_count.data_ptr<std::int64_t>()),
                       caller_stream),
                   "counting fused DLB received experts");
        const at::TensorOptions pinned_long =
            at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true);
        at::Tensor actual_counts_cpu = at::empty(
            {static_cast<int64_t>(local_experts)}, pinned_long);
        at::Tensor rail_counts_cpu = at::empty(
            {static_cast<int64_t>(rails)}, pinned_long);
        at::Tensor channel_counts_cpu = at::empty(
            {static_cast<int64_t>(runtime.config.rail_channel_count)},
            pinned_long);
        at::Tensor group_count_cpu = at::empty({1}, pinned_long);
        check_cuda(cudaMemcpyAsync(
                       actual_counts_cpu.data_ptr(), device_counts.data_ptr(),
                       static_cast<std::size_t>(local_experts) * sizeof(std::uint64_t),
                       cudaMemcpyDeviceToHost, caller_stream),
                   "copying fused DLB received expert counts");
        check_cuda(cudaMemcpyAsync(
                       rail_counts_cpu.data_ptr(), device_rail_record_counts,
                       static_cast<std::size_t>(rails) * sizeof(std::uint64_t),
                       cudaMemcpyDeviceToHost, caller_stream),
                   "copying GPU-resident DLB Rail counters");
        check_cuda(cudaMemcpyAsync(
                       channel_counts_cpu.data_ptr(), device_channel_record_counts,
                       static_cast<std::size_t>(runtime.config.rail_channel_count) *
                           sizeof(std::uint64_t),
                       cudaMemcpyDeviceToHost, caller_stream),
                   "copying GPU-resident DLB channel counters");
        check_cuda(cudaMemcpyAsync(
                       group_count_cpu.data_ptr(), device_group_count.data_ptr(),
                       sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                       caller_stream),
                   "copying fused DLB received group count");
        check_cuda(cudaMemcpyAsync(
                       host_invalid_route_count, device_invalid_route_count,
                       sizeof(std::uint64_t), cudaMemcpyDeviceToHost, caller_stream),
                   "copying DLB invalid-route counter");
        if (profiling_enabled) {
            check_cuda(cudaEventRecord(runtime.profile_counted[profile_slot], caller_stream),
                       "recording DLB dispatch count completion");
        }
        // Exact PyTorch output allocation needs only this compact expert,
        // Rail, channel, group, and validation-counter boundary.
        check_cuda(cudaStreamSynchronize(caller_stream),
                   "reading fused DLB received expert counts");
        TORCH_CHECK(*host_invalid_route_count == 0,
                    "topk_idx contains ", *host_invalid_route_count,
                    " expert IDs outside [0, ", num_experts_value, ")");
        const std::int64_t group_count_value =
            group_count_cpu.data_ptr<std::int64_t>()[0];
        TORCH_CHECK(group_count_value >= 0,
                    "DLB received group count cannot be negative");

        const auto* actual_counts = actual_counts_cpu.data_ptr<std::int64_t>();
        std::vector<std::int64_t> aligned_counts(local_experts, 0);
        std::vector<std::uint64_t> expert_offsets(local_experts, 0);
        std::uint64_t padded_records = 0;
        for (std::uint32_t expert = 0; expert < local_experts; ++expert) {
            TORCH_CHECK(actual_counts[expert] >= 0,
                        "DLB received expert count cannot be negative");
            expert_offsets[expert] = padded_records;
            const std::uint64_t actual = static_cast<std::uint64_t>(actual_counts[expert]);
            const std::uint64_t alignment =
                static_cast<std::uint64_t>(expert_alignment_value);
            TORCH_CHECK(actual <= std::numeric_limits<std::uint64_t>::max() - alignment + 1,
                        "DLB expert alignment overflows uint64_t");
            const std::uint64_t aligned = (actual + alignment - 1) / alignment * alignment;
            TORCH_CHECK(aligned <= std::numeric_limits<std::int64_t>::max() - padded_records,
                        "DLB padded receive size overflows int64_t");
            aligned_counts[expert] = static_cast<std::int64_t>(aligned);
            padded_records += aligned;
        }

        at::Tensor recv_x = at::empty(
            {static_cast<int64_t>(padded_records), x.size(1)}, x.options());
        at::Tensor recv_headers = at::empty(
            {static_cast<int64_t>(padded_records), 6},
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device));
        at::Tensor recv_weights = at::empty(
            {static_cast<int64_t>(padded_records)},
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device));
        at::Tensor valid_mask = at::empty(
            {static_cast<int64_t>(padded_records)},
            at::TensorOptions().dtype(at::kBool).device(at::kCUDA, device));
        at::Tensor group_output_indices = at::full(
            {group_count_value, 8}, -1,
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device));
        if (padded_records != 0) {
            check_cuda(cudaMemsetAsync(recv_x.data_ptr(), 0,
                                       padded_records * hidden_bytes, caller_stream),
                       "clearing fused DLB receive payload");
            check_cuda(cudaMemsetAsync(recv_headers.data_ptr(), 0xff,
                                       padded_records * 6 * sizeof(std::int64_t),
                                       caller_stream),
                       "clearing fused DLB receive headers");
            check_cuda(cudaMemsetAsync(recv_weights.data_ptr(), 0,
                                       padded_records * sizeof(float), caller_stream),
                       "clearing fused DLB receive weights");
            check_cuda(cudaMemsetAsync(valid_mask.data_ptr(), 0,
                                       padded_records * sizeof(bool), caller_stream),
                       "clearing fused DLB valid mask");
        }
        if (padded_records != 0) {
            at::Tensor device_offsets = at::empty(
                {static_cast<int64_t>(local_experts)},
                at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device));
            at::Tensor device_cursors = at::zeros_like(device_offsets);
            at::Tensor device_group_cursor = at::zeros(
                {1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device));
            check_cuda(cudaMemcpyAsync(device_offsets.data_ptr(), expert_offsets.data(),
                                       expert_offsets.size() * sizeof(expert_offsets.front()),
                                       cudaMemcpyHostToDevice, caller_stream),
                       "uploading fused DLB expert offsets");
            check_cuda(dlb_alltoall::launch_dlb_scatter_received_experts(
                           repair, repair_record_count, runtime.config.record_bytes,
                           hidden_bytes, epoch, rank, rank * local_experts,
                           local_experts,
                           reinterpret_cast<const std::uint64_t*>(device_offsets.data_ptr()),
                           reinterpret_cast<std::uint64_t*>(device_cursors.data_ptr()),
                           static_cast<std::uint8_t*>(recv_x.data_ptr()),
                           recv_headers.data_ptr<std::int64_t>(),
                           recv_weights.data_ptr<float>(), valid_mask.data_ptr<bool>(),
                           padded_records,
                           reinterpret_cast<std::uint64_t*>(
                               device_group_cursor.data_ptr<std::int64_t>()),
                           reinterpret_cast<std::uint64_t*>(
                               group_output_indices.data_ptr<std::int64_t>()),
                           static_cast<std::uint64_t>(group_count_value),
                           caller_stream),
                       "scattering fused DLB received experts");
        }
        if (profiling_enabled) {
            check_cuda(cudaEventRecord(runtime.profile_scattered[profile_slot], caller_stream),
                       "recording DLB dispatch scatter completion");
            last_profile_epoch = epoch;
            last_profile_kind = 1;
        }

        at::Tensor aligned_counts_cpu = at::empty_like(actual_counts_cpu);
        std::memcpy(aligned_counts_cpu.data_ptr(), aligned_counts.data(),
                    aligned_counts.size() * sizeof(aligned_counts.front()));
        // Record after scatter so a later post on any CUDA stream cannot
        // overwrite this epoch's repair slot while scatter still reads it.
        check_cuda(cudaEventRecord(dispatch_materialize_ready, caller_stream),
                   "recording DLB dispatch materialization completion");
        dispatch_materialize_ready_recorded = true;
        finished_dispatch_epoch = epoch;
        dispatch_pending = false;
        pending_dispatch_x = at::Tensor();
        pending_dispatch_topk_idx = at::Tensor();
        pending_dispatch_topk_weights = at::Tensor();
        pending_dispatch_num_experts = 0;
        pending_dispatch_expert_alignment = 0;
        last_dispatch_epoch = epoch;
        return std::make_tuple(recv_x, recv_headers, recv_weights, valid_mask,
                               actual_counts_cpu, aligned_counts_cpu,
                               rail_counts_cpu, channel_counts_cpu,
                               group_output_indices);
    }

    // Finishes a specific posted dispatch and returns expert-major local inputs.
    //
    // Args:
    //   epoch_value: The epoch passed to post_dispatch_moe.
    //
    // Returns:
    //   Nine tensors containing expert-major inputs, routing metadata, and
    //   Rail/channel accounting for the posted epoch.
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor> finish_dispatch_moe(int64_t epoch_value) {
        TORCH_CHECK(!closed && initialized, "DLB communicator is closed");
        TORCH_CHECK(dispatch_pending,
                    "DLB finish_dispatch_moe requires a posted dispatch");
        TORCH_CHECK(epoch_value > 0 &&
                        static_cast<std::uint64_t>(epoch_value) == submitted_dispatch_epoch,
                    "DLB finish_dispatch_moe epoch does not match the posted dispatch");
        return materialize_posted_dispatch(
            pending_dispatch_x, epoch_value,
            static_cast<int64_t>(pending_dispatch_num_experts),
            pending_dispatch_expert_alignment);
    }

    at::Tensor get_last_dispatch_profile() {
        TORCH_CHECK(profiling_enabled,
                    "DLB profiling is disabled; construct DLBBuffer with enable_profiling=True");
        TORCH_CHECK(last_profile_kind == 1 && last_profile_epoch != 0,
                    "no unread DLB dispatch profile is available");
        check_cuda(cudaSetDevice(device), "selecting DLB profiling device");
        float values[dlb_alltoall::kDlbProfileMetricCount]{};
        check_cuda(dlb_alltoall::read_dlb_nvshmem_profile(
                       &runtime, last_profile_epoch, values,
                       dlb_alltoall::kDlbProfileMetricCount),
                   "reading DLB dispatch profile");
        at::Tensor profile = at::empty(
            {static_cast<int64_t>(dlb_alltoall::kDlbProfileMetricCount)},
            at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
        std::memcpy(profile.data_ptr<float>(), values, sizeof(values));
        last_profile_kind = 0;
        return profile;
    }

    at::Tensor get_last_combine_profile() {
        TORCH_CHECK(profiling_enabled,
                    "DLB profiling is disabled; construct DLBBuffer with enable_profiling=True");
        TORCH_CHECK(last_profile_kind == 2 && last_profile_epoch != 0,
                    "no unread DLB combine profile is available");
        check_cuda(cudaSetDevice(device), "selecting DLB profiling device");
        float values[dlb_alltoall::kDlbProfileMetricCount]{};
        check_cuda(dlb_alltoall::read_dlb_nvshmem_profile(
                       &runtime, last_profile_epoch, values,
                       dlb_alltoall::kDlbProfileMetricCount),
                   "reading DLB combine profile");
        at::Tensor profile = at::empty(
            {static_cast<int64_t>(dlb_alltoall::kDlbProfileMetricCount)},
            at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
        std::memcpy(profile.data_ptr<float>(), values, sizeof(values));
        last_profile_kind = 0;
        return profile;
    }

    std::tuple<at::Tensor, at::Tensor> combine_moe(
        at::Tensor x, at::Tensor headers, at::Tensor weights,
        at::Tensor valid_mask, at::Tensor group_output_indices,
        int64_t epoch_value, int64_t round_id_value,
        int64_t num_tokens_value, int64_t num_topk_value,
        bool apply_router_weights) {
        TORCH_CHECK(!closed && initialized, "DLB communicator is closed");
        TORCH_CHECK(!dispatch_pending,
                    "finish_dispatch_moe must complete before combine_moe");
        last_profile_kind = 0;
        TORCH_CHECK(epoch_value > 0 &&
                        static_cast<std::uint64_t>(epoch_value) ==
                            last_dispatch_epoch + 1,
                    "DLB combine epochs must increase by one");
        TORCH_CHECK(round_id_value >= 0 &&
                        round_id_value <=
                            std::numeric_limits<std::uint32_t>::max(),
                    "round_id must fit uint32_t");
        TORCH_CHECK(num_tokens_value > 0 && num_topk_value > 0 &&
                        num_topk_value <=
                            std::numeric_limits<std::uint32_t>::max(),
                    "DLB combine token and top-k sizes are invalid");
        TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                        x.get_device() == static_cast<int64_t>(device),
                    "combine x must be a contiguous rank-2 tensor on this CUDA device");
        TORCH_CHECK(x.scalar_type() == at::kBFloat16 ||
                        x.scalar_type() == at::kHalf ||
                        x.scalar_type() == at::kFloat,
                    "combine x must use BF16, FP16, or FP32");
        TORCH_CHECK(headers.is_cuda() && headers.is_contiguous() &&
                        headers.scalar_type() == at::kLong && headers.dim() == 2 &&
                        headers.size(0) == x.size(0) && headers.size(1) == 6 &&
                        headers.get_device() == static_cast<int64_t>(device),
                    "combine headers must be contiguous CUDA int64 [records, 6]");
        TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
                        weights.scalar_type() == at::kFloat && weights.dim() == 1 &&
                        weights.size(0) == x.size(0) &&
                        weights.get_device() == static_cast<int64_t>(device),
                    "combine weights must be contiguous CUDA float32 [records]");
        TORCH_CHECK(valid_mask.is_cuda() && valid_mask.is_contiguous() &&
                        valid_mask.scalar_type() == at::kBool && valid_mask.dim() == 1 &&
                        valid_mask.size(0) == x.size(0) &&
                        valid_mask.get_device() == static_cast<int64_t>(device),
                    "combine valid mask must be contiguous CUDA bool [records]");
        TORCH_CHECK(group_output_indices.is_cuda() &&
                        group_output_indices.is_contiguous() &&
                        group_output_indices.scalar_type() == at::kLong &&
                        group_output_indices.dim() == 2 &&
                        group_output_indices.size(1) == 8 &&
                        group_output_indices.get_device() ==
                            static_cast<int64_t>(device),
                    "combine group mapping must be contiguous CUDA int64 [groups, 8]");
        const std::uint64_t hidden_bytes =
            static_cast<std::uint64_t>(x.size(1)) * x.element_size();
        TORCH_CHECK(runtime.config.record_bytes >= 128 + hidden_bytes,
                    "DLB record size is too small for the combine payload");
        std::uint32_t scalar_type = 2;
        if (x.scalar_type() == at::kBFloat16) scalar_type = 0;
        if (x.scalar_type() == at::kHalf) scalar_type = 1;

        check_cuda(cudaSetDevice(device), "selecting DLB combine device");
        const cudaStream_t caller_stream =
            at::cuda::getCurrentCUDAStream(device).stream();
        const std::uint64_t epoch = static_cast<std::uint64_t>(epoch_value);
        const std::uint32_t slot = static_cast<std::uint32_t>(
            epoch % runtime.config.pipeline_depth);
        if (profiling_enabled) {
            check_cuda(cudaEventRecord(runtime.profile_dispatch_begin[slot],
                                       caller_stream),
                       "recording DLB combine profile start");
        }
        check_cuda(cudaStreamWaitEvent(caller_stream, runtime.repair_ready[slot], 0),
                   "waiting for reusable DLB combine repair slot");
        check_cuda(cudaStreamWaitEvent(caller_stream, runtime.transport_ready[slot], 0),
                   "waiting for reusable DLB combine transport slot");
        check_cuda(cudaMemsetAsync(
                       symmetric_demand_row, 0,
                       static_cast<std::size_t>(world) * sizeof(std::uint64_t),
                       caller_stream),
                   "clearing GPU-resident DLB combine demand row");
        check_cuda(dlb_alltoall::launch_dlb_count_combine_routes(
                       reinterpret_cast<const std::uint64_t*>(
                           group_output_indices.data_ptr<std::int64_t>()),
                       static_cast<std::uint64_t>(group_output_indices.size(0)),
                       headers.data_ptr<std::int64_t>(),
                       static_cast<std::uint64_t>(x.size(0)), world,
                       symmetric_demand_row, caller_stream),
                   "counting GPU-resident DLB combine routes");
        TORCH_CHECK(nvshmemx_uint64_fcollect_on_stream(
                        local_team, symmetric_demand_matrix, symmetric_demand_row,
                        world, caller_stream) == NVSHMEMX_SUCCESS,
                    "collecting DLB combine demand on GPU failed");

        runtime.device_transfer_count = static_cast<std::uint32_t>(
            (runtime.server_count - 1) * rails *
            runtime.config.rail_channel_count);
        check_cuda(dlb_alltoall::launch_dlb_build_dynamic_rail_plan(
                       symmetric_demand_matrix, runtime.server_count, rails, server,
                       local_rank, static_cast<std::uint32_t>(round_id_value),
                       runtime.config.rail_channel_count,
                       runtime.config.record_bytes, runtime.config.receive_slot_bytes,
                       runtime.config.receive_slot_bytes, device_flow_rail_counts,
                       runtime.device_transfers, runtime.device_transfer_count,
                       device_rail_record_counts, device_channel_record_counts,
                       caller_stream),
                   "building GPU-resident DLB combine Rail plan");
        check_cuda(cudaMemsetAsync(
                       device_destination_cursors, 0,
                       static_cast<std::size_t>(world) * sizeof(std::uint64_t),
                       caller_stream),
                   "clearing GPU-resident DLB combine cursors");
        const std::uint64_t repair_epoch_offset =
            static_cast<std::uint64_t>(slot) * world *
            runtime.config.repair_slot_bytes;
        check_cuda(dlb_alltoall::launch_dlb_pack_combine_direct(
                       x.data_ptr(), static_cast<std::uint64_t>(x.size(1)),
                       scalar_type, headers.data_ptr<std::int64_t>(),
                       weights.data_ptr<float>(),
                       static_cast<std::uint64_t>(x.size(0)),
                       reinterpret_cast<const std::uint64_t*>(
                           group_output_indices.data_ptr<std::int64_t>()),
                       static_cast<std::uint64_t>(group_output_indices.size(0)),
                       apply_router_weights, world,
                       runtime.server_count, rails, server, local_rank, rank, epoch,
                       runtime.config.record_bytes,
                       runtime.config.receive_slot_bytes,
                       runtime.config.repair_slot_bytes, repair_epoch_offset,
                       device_flow_rail_counts, device_destination_cursors,
                       reinterpret_cast<std::uint8_t* const*>(
                           runtime.rail_send_tables[slot].device_ptrs),
                       reinterpret_cast<std::uint8_t* const*>(
                           runtime.repair_table.device_ptrs),
                       caller_stream),
                   "direct-packing DLB combine records on GPU");

        std::uint8_t* repair =
            dlb_alltoall::dlb_nvshmem_repair_buffer_for_epoch(&runtime, epoch);
        check_cuda(dlb_alltoall::launch_dlb_nvshmem_rdma(
                       &runtime, epoch, caller_stream),
                   "launching fused DLB combine");
        check_cuda(dlb_alltoall::wait_dlb_nvshmem_rdma_epoch(
                       &runtime, epoch, caller_stream),
                   "waiting for fused DLB combine");
        if (profiling_enabled) {
            check_cuda(cudaEventRecord(runtime.profile_counted[slot], caller_stream),
                       "recording DLB combine transport completion");
        }

        at::Tensor combined_float = at::zeros(
            {num_tokens_value, x.size(1)},
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device));
        at::Tensor combined_weights = at::zeros(
            {num_tokens_value, num_topk_value},
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device));
        const std::uint64_t repair_bytes =
            static_cast<std::uint64_t>(world) * runtime.config.repair_slot_bytes;
        TORCH_CHECK(repair_bytes % runtime.config.record_bytes == 0,
                    "DLB combine repair buffer is not record aligned");
        check_cuda(dlb_alltoall::launch_dlb_accumulate_combined_records(
                       repair, repair_bytes / runtime.config.record_bytes,
                       runtime.config.record_bytes,
                       static_cast<std::uint64_t>(x.size(1)), scalar_type, epoch,
                       rank, static_cast<std::uint64_t>(num_tokens_value),
                       static_cast<std::uint32_t>(num_topk_value),
                       combined_float.data_ptr<float>(),
                       combined_weights.data_ptr<float>(), caller_stream),
                   "accumulating fused DLB combine output");
        at::Tensor combined = combined_float.to(x.scalar_type());
        if (profiling_enabled) {
            check_cuda(cudaEventRecord(runtime.profile_scattered[slot], caller_stream),
                       "recording DLB combine accumulation completion");
            last_profile_epoch = epoch;
            last_profile_kind = 2;
        }
        last_dispatch_epoch = epoch;
        return std::make_tuple(combined, combined_weights);
    }

    void close() {
        if (closed) return;
        TORCH_CHECK(!dispatch_pending,
                    "finish_dispatch_moe must complete before closing DLB communicator");
        if (initialized) {
            check_cuda(cudaSetDevice(device), "selecting DLB CUDA device for close");
            check_cuda(dlb_alltoall::destroy_dlb_nvshmem_rdma_runtime(&runtime),
                       "destroying DLB NVSHMEM runtime");
            if (symmetric_demand_matrix != nullptr) {
                nvshmem_free(symmetric_demand_matrix);
                symmetric_demand_matrix = nullptr;
            }
            if (symmetric_demand_row != nullptr) {
                nvshmem_free(symmetric_demand_row);
                symmetric_demand_row = nullptr;
            }
            initialized = false;
        }
        nvshmem_barrier_all();
        if (device_destination_cursors != nullptr) {
            check_cuda(cudaFree(device_destination_cursors),
                       "releasing DLB destination cursors");
            device_destination_cursors = nullptr;
        }
        if (device_flow_rail_counts != nullptr) {
            check_cuda(cudaFree(device_flow_rail_counts),
                       "releasing GPU-resident DLB flow plan");
            device_flow_rail_counts = nullptr;
        }
        if (device_rail_record_counts != nullptr) {
            check_cuda(cudaFree(device_rail_record_counts),
                       "releasing GPU-resident DLB Rail counters");
            device_rail_record_counts = nullptr;
        }
        if (device_channel_record_counts != nullptr) {
            check_cuda(cudaFree(device_channel_record_counts),
                       "releasing GPU-resident DLB channel counters");
            device_channel_record_counts = nullptr;
        }
        if (device_invalid_route_count != nullptr) {
            check_cuda(cudaFree(device_invalid_route_count),
                       "releasing DLB invalid-route counter");
            device_invalid_route_count = nullptr;
        }
        if (host_invalid_route_count != nullptr) {
            check_cuda(cudaFreeHost(host_invalid_route_count),
                       "releasing DLB pinned invalid-route counter");
            host_invalid_route_count = nullptr;
        }
        if (dispatch_materialize_ready != nullptr) {
            check_cuda(cudaEventDestroy(dispatch_materialize_ready),
                       "destroying DLB dispatch materialization event");
            dispatch_materialize_ready = nullptr;
            dispatch_materialize_ready_recorded = false;
        }
        if (local_team != NVSHMEM_TEAM_INVALID) {
            nvshmem_team_destroy(local_team);
            local_team = NVSHMEM_TEAM_INVALID;
        }
        nvshmem_finalize();
        closed = true;
    }

    ~dlb_nvshmem_comm_t() {
        // Explicit `close()` is collective and is required by the tester.
        // Do not issue a hidden collective from an arbitrary Python destructor.
        if (!closed && initialized) cudaDeviceSynchronize();
    }
};

}  // namespace

// Expose the stateful DLB communicator through the Torch custom-class registry.
TORCH_LIBRARY(dlb_classes, m) {
    m.class_<dlb_nvshmem_comm_t>("nvshmem_comm_t")
        .def(torch::init<int64_t, int64_t, int64_t, int64_t, at::Tensor,
                         int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                         int64_t,
                         bool, bool>())
        .def("benchmark_prepare_moe_device",
             &dlb_nvshmem_comm_t::benchmark_prepare_moe_device)
        .def("post_dispatch_moe", &dlb_nvshmem_comm_t::post_dispatch_moe)
        .def("finish_dispatch_moe", &dlb_nvshmem_comm_t::finish_dispatch_moe)
        .def("get_last_dispatch_profile",
             &dlb_nvshmem_comm_t::get_last_dispatch_profile)
        .def("get_last_combine_profile",
             &dlb_nvshmem_comm_t::get_last_combine_profile)
        .def("combine_moe", &dlb_nvshmem_comm_t::combine_moe)
        .def("close", &dlb_nvshmem_comm_t::close);
}

// Expose stateless bootstrap and diagnostic operations through Torch operators.
TORCH_LIBRARY(dlb_nvshmem, m) {
    m.def("get_uniqueid", &get_dlb_nvshmem_init_id);
    m.def("cuda_module_probe", &dlb_cuda_module_probe);
}
