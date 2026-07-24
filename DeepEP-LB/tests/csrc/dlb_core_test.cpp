#include <torch/extension.h>

#include "dlb_loopback_test.cuh"

#include <vector>

std::vector<torch::Tensor> run_dlb_compact_quota_cuda(
    const torch::Tensor& source_loads,
    int64_t source_server,
    int64_t round_id);

std::vector<torch::Tensor> run_dlb_handle_roundtrip_cuda(
    int64_t max_tokens,
    int64_t num_tokens,
    int64_t num_incoming,
    int64_t num_outgoing,
    int64_t num_servers,
    int64_t num_channels);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "run_dlb_compact_quota",
        &run_dlb_compact_quota_cuda,
        "Build compact deterministic DLB Rail quotas"
    );
    module.def(
        "run_dlb_handle_roundtrip",
        &run_dlb_handle_roundtrip_cuda,
        "Validate compact moved-only DLB handle serialization"
    );
    deep_ep_lb_test::bind_dlb_p2p_loopback(module);
}
