#pragma once

#include "dlb_alltoall/dlb_nvlink_runtime.h"
#include "dlb_alltoall/dlb_rail_transport.h"
#include "dlb_alltoall/dlb_rail_receive_plan.h"

#include <cuda_runtime_api.h>
#include <nvshmem.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dlb_alltoall {

// Real cross-machine backend. NVSHMEM must already be initialized; with an
// IBGDA-capable NVSHMEM deployment the Rail put is GPU-initiated RDMA. All
// ranks on one logical server call initialize/launch collectively.
struct DlbRuntimeConfig {
    std::uint32_t rank;
    std::uint32_t world_size;
    std::uint32_t gpus_per_server;
    std::uint64_t record_bytes;
    std::uint64_t receive_slot_bytes;
    std::uint64_t repair_slot_bytes;
    std::uint64_t initial_epoch;
    // DLB keeps several independent protocol slots.  Consecutive epochs use
    // different slots so staging of epoch N+1 can overlap transport/repair of
    // epoch N.  Two is the normal ping-pong configuration.
    std::uint32_t pipeline_depth;
    // Upper bound of one GPU copy / NVSHMEM put submitted by a CTA/warp.  It
    // is a transport tuning knob, not a scheduling knob.
    std::uint64_t chunk_bytes;
    // Independent Rail channels. Every channel owns one sender CTA, one
    // receiver CTA, and distinct metadata/progress/credit slots. The total
    // communication-SM budget is approximately 2 * rail_channel_count.
    std::uint32_t rail_channel_count;
    // Fixed per-Rail capacity. Peer-visible symmetric NVSHMEM/RDMA source
    // buffers are allocated once and reused by every plan/epoch; a plan
    // larger than this capacity is rejected explicitly.
    std::uint64_t rail_send_capacity_bytes;
    // Optional CUDA-event instrumentation. Disabled by default so production
    // launches pay no profiling-event cost.
    bool enable_profiling;
    // Same-host logical-server validation backend. It preserves the DLB Rail
    // mapping and repair protocol but uses direct NVLink/P2P peer pointers.
    // False selects the real cross-machine NVSHMEM/IBGDA backend.
    bool use_loopback_transport;
    nvshmem_team_t local_team;
};

struct DlbCudaIpcTable {
    void* local_buffer;
    void** device_ptrs;
    std::vector<void*> host_ptrs;
    // CUDA IPC mappings must be closed explicitly.  Pointer tables obtained
    // from nvshmem_ptr are non-owning views of the symmetric heap and must not
    // be passed to cudaIpcCloseMemHandle.
    bool owns_remote_mappings = false;
};

// One instance lives on every GPU rank. It owns the direct-pack DLB transport
// buffers and GPU descriptor tables for that rank's logical source server.
struct DlbRuntime {
    DlbRuntimeConfig config;
    std::uint32_t server_count;
    std::uint32_t server_rank;
    std::uint32_t local_rank;
    bool initialized;

    DlbRailReceivePlan rail_receive_plan;

    // One fixed-capacity symmetric send buffer per ping-pong slot.  Local GPU
    // producers use the per-slot NVSHMEM peer-pointer table to write directly
    // into the selected Rail's send queue.  This follows the receiver/queue
    // pointer-table design used by high-throughput GPU communication kernels
    // and avoids a second full-capacity device-to-device staging copy.
    std::uint8_t* symmetric_rail_send_buffer;
    std::uint8_t* local_repair_buffer;
    std::vector<DlbCudaIpcTable> rail_send_tables;
    DlbCudaIpcTable repair_table;
    std::uint64_t* local_stage_signals;
    std::uint64_t* local_completion_signals;
    DlbCudaIpcTable stage_signal_table;
    DlbCudaIpcTable completion_signal_table;

    std::uint8_t* symmetric_receive_buffer;
    std::uint64_t* symmetric_receive_signals;
    std::uint64_t* symmetric_progress_signals;
    std::uint64_t* symmetric_credits;
    DlbCudaIpcTable loopback_receive_table;
    DlbCudaIpcTable loopback_receive_signal_table;
    DlbCudaIpcTable loopback_progress_signal_table;
    DlbCudaIpcTable loopback_credit_table;

    DlbRailTransfer* device_transfers;
    std::uint32_t device_transfer_count;
    DlbRailArrival* device_arrivals;
    std::uint32_t device_arrival_count;

    // A launch first stages on `stage_stream`, injects/waits on
    // `transport_stream`, then repairs and returns credits on
    // `repair_stream`.  The events establish the only required ordering;
    // callers retain asynchronous dispatch and may explicitly wait per epoch.
    cudaStream_t stage_stream;
    cudaStream_t transport_stream;
    cudaStream_t repair_stream;
    std::vector<cudaEvent_t> source_ready;
    std::vector<cudaEvent_t> stage_ready;
    std::vector<cudaEvent_t> transport_ready;
    std::vector<cudaEvent_t> repair_ready;

    // Timed CUDA events are allocated only when enable_profiling is true.
    // They expose the full asynchronous pipeline without inserting a host
    // synchronization into the normal dispatch path.
    std::vector<cudaEvent_t> profile_dispatch_begin;
    std::vector<cudaEvent_t> profile_source_ready;
    std::vector<cudaEvent_t> profile_stage_ready;
    std::vector<cudaEvent_t> profile_producers_ready;
    std::vector<cudaEvent_t> profile_transport_ready;
    std::vector<cudaEvent_t> profile_repair_kernel_ready;
    std::vector<cudaEvent_t> profile_repair_ready;
    std::vector<cudaEvent_t> profile_counted;
    std::vector<cudaEvent_t> profile_scattered;
};

// Fixed metric order returned by the Torch/Python monitoring API:
// prepare, stage publish, producer wait, staging copy, Rail transport,
// receive/repair, completion wait, count/readback, materialize/scatter,
// end-to-end. Staging copy is retained as an explicit zero-valued metric so a
// regression that reintroduces it remains visible in logs and dashboards.
constexpr std::size_t kDlbProfileMetricCount = 10;

cudaError_t read_dlb_nvshmem_profile(DlbRuntime* runtime,
                                     std::uint64_t epoch,
                                     float* milliseconds,
                                     std::size_t metric_count);

// Allocates symmetric protocol buffers, direct-pack Rail buffers, and the
// local final-repair buffer.
cudaError_t initialize_dlb_nvshmem_rdma_runtime(DlbRuntime* runtime,
                                                const DlbRuntimeConfig& config);

// Direct pack has already written this epoch's Rail and same-server repair
// slots. The call asynchronously coordinates local producers, sends the Rail
// slices, repairs remote arrivals, and records destination completion.
cudaError_t launch_dlb_nvshmem_rdma(DlbRuntime* runtime,
                                    std::uint64_t epoch,
                                    cudaStream_t stream);

// Make `stream` wait only for the requested epoch's destination-side repair
// and credit return.  This is the async dispatch completion primitive.
cudaError_t wait_dlb_nvshmem_rdma_epoch(DlbRuntime* runtime,
                                        std::uint64_t epoch,
                                        cudaStream_t stream);

// The final-repair region belonging to `epoch`.  A caller that needs to read
// it directly must first wait for that epoch.  The address stays valid until
// runtime destruction or the same ping-pong slot is reused.
std::uint8_t* dlb_nvshmem_repair_buffer_for_epoch(DlbRuntime* runtime,
                                                  std::uint64_t epoch);

cudaError_t destroy_dlb_nvshmem_rdma_runtime(DlbRuntime* runtime);

}  // namespace dlb_alltoall
