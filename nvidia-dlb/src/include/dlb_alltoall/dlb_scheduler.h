#pragma once

#include <cstdint>
#include <vector>

namespace dlb_alltoall {

// A tile preserves logical source-GPU -> real-destination-GPU demand. The row
// of assigned is a selected source NIC; the physical Rail is NIC q -> NIC q.
struct TilePlan {
    unsigned source_server;
    unsigned destination_server;
    std::vector<std::uint64_t> input;
    std::vector<std::uint64_t> assigned;
    // [source GPU][selected source NIC][real destination GPU]. This is the
    // executable source-side routing flow used by the NVLink staging path.
    std::vector<std::uint64_t> source_to_nic;
    std::vector<std::uint64_t> source_gpu_loads;
    std::vector<std::uint64_t> nic_targets;
    std::uint64_t total_records;
    std::uint64_t source_staging_records;
    std::uint64_t static_destination_repair_records;
    std::uint64_t destination_repair_records;
    std::uint64_t direct_priority_records;
};

// This is deliberately source-server local. `local_server_demand` has M rows
// (the source server GPUs) and S*M columns (all real destination GPUs), so
// DLB does not require a full N x N demand matrix or a global schedule.
struct ServerScheduler {
    unsigned server_count;
    unsigned gpus_per_server;
    unsigned source_server;
    unsigned round_id;
    std::vector<TilePlan> tiles;
    std::vector<std::uint64_t> source_nic_sent;
    std::uint64_t inter_server_records;
    std::uint64_t source_staging_records;
    std::uint64_t static_destination_repair_records;
    std::uint64_t destination_repair_records;
    std::uint64_t direct_priority_records;
};

void init_server_scheduler(ServerScheduler* scheduler,
                           unsigned server_count,
                           unsigned gpus_per_server,
                           unsigned source_server,
                           const std::vector<std::uint64_t>& local_server_demand,
                           unsigned round_id = 0,
                           bool direct_priority = true);

void update_server_scheduler(ServerScheduler* scheduler,
                             const std::vector<std::uint64_t>& local_server_demand,
                             unsigned round_id = 0,
                             bool direct_priority = true);

// Build one independent source-server to destination-server plan. A server
// controller can execute its S-1 remote-destination tiles in parallel.
TilePlan plan_destination_tile(unsigned server_count,
                               unsigned gpus_per_server,
                               unsigned source_server,
                               unsigned destination_server,
                               const std::vector<std::uint64_t>& local_server_demand,
                               unsigned round_id = 0,
                               bool direct_priority = true);

void validate_server_scheduler(const ServerScheduler& scheduler);

unsigned tile_nic_gap(const TilePlan& tile);

}  // namespace dlb_alltoall
