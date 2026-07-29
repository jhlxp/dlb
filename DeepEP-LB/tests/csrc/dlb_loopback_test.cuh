#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace deep_ep_lb_test {

// Test-only CUDA-IPC backend for the DLB Rail patch. It models two logical
// four-GPU servers on one NVLink host and validates the source staging and
// moved-token reverse repair around the unchanged RDMA boundary. No logical
// topology or loopback transport enters production.
class DlbP2PLoopbackRuntime {
public:
    DlbP2PLoopbackRuntime(
        int rank,
        int world_size,
        int max_tokens,
        int record_bytes);

    ~DlbP2PLoopbackRuntime();

    pybind11::bytes get_ipc_handle() const;

    void sync(const std::vector<pybind11::bytes>& handles);

    void reset();

    void pack_off(
        const torch::Tensor& records,
        int num_messages);

    void dispatch_on(
        const torch::Tensor& records,
        const torch::Tensor& quota_rows,
        int num_messages,
        int num_incoming,
        int epoch);

    void combine(
        const torch::Tensor& combined_records,
        int num_source_messages,
        int num_ring_messages,
        int epoch);

    void clear_post_forward_counter();

    void seed_receive_ring(
        const torch::Tensor& records,
        const torch::Tensor& destination_rails,
        int num_messages);

    void post_forward(int num_ring_messages);

    std::vector<torch::Tensor> materialize_ring(int count);

    std::vector<torch::Tensor> materialize_stage(int count);

    torch::Tensor quota_stats(
        const torch::Tensor& source_loads,
        int source_server,
        int round_id);

    int64_t get_record_stride() const;

private:
    void set_device() const;

    int rank_;
    int world_size_;
    int max_tokens_;
    int source_rail_;
    int device_id_;
    int queue_capacity_;
    size_t record_stride_;
    size_t sync_offset_;
    size_t stage_ready_offset_;
    size_t repair_ready_offset_;
    size_t staging_ids_offset_;
    size_t staging_payload_offset_;
    size_t ring_ids_offset_;
    size_t ring_payload_offset_;
    size_t total_bytes_;
    uint8_t* local_buffer_ = nullptr;
    void** local_ptrs_device_ = nullptr;
    std::vector<void*> local_ptrs_host_;
    bool synced_ = false;
};

void bind_dlb_p2p_loopback(pybind11::module_& module);

}  // namespace deep_ep_lb_test
