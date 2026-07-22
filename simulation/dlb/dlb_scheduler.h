#pragma once

#include <cstdint>
#include <vector>

namespace dlb {

typedef std::uint64_t Size;

// One logical source-GPU to real-destination-GPU server-pair tile. Rows of
// assigned are selected source NICs; Rail q carries that row to destination
// NIC q, which then forwards to the real destination GPU column when needed.
struct TilePlan {
    unsigned source_server;
    unsigned destination_server;
    std::vector<Size> input;
    std::vector<Size> assigned;
    std::vector<Size> source_gpu_loads;
    std::vector<Size> nic_targets;
    Size total_records;
    Size source_staging_records;
    Size static_destination_repair_records;
    Size destination_repair_records;
    Size direct_priority_records;
};

struct Plan {
    unsigned server_count;
    unsigned gpus_per_server;
    std::vector<TilePlan> tiles;
    std::vector<Size> logical_inter_server_records;
    std::vector<Size> source_nic_sent;
    std::vector<Size> destination_nic_received;
    Size inter_server_records;
    Size source_staging_records;
    Size static_destination_repair_records;
    Size destination_repair_records;
    Size direct_priority_records;
};

struct MetadataFootprint {
    unsigned dlb_counters_per_source_server;
    unsigned fast_global_server_pair_counters;
    unsigned fast_global_schedule_matrix_values;
};

// demand is a row-major (server_count * gpus_per_server)^2 logical GPU matrix.
// DLB does not inspect or schedule same-server entries.
Plan build_plan(const std::vector<Size>& demand,
                unsigned server_count,
                unsigned gpus_per_server,
                unsigned round_id = 0,
                bool direct_priority = true);

// Throws std::logic_error when any record-conservation or Rail-balance
// invariant is broken.
void validate_plan(const Plan& plan, const std::vector<Size>& demand);

MetadataFootprint metadata_footprint(unsigned server_count,
                                     unsigned gpus_per_server);

unsigned tile_load_gap(const TilePlan& tile);

}  // namespace dlb
