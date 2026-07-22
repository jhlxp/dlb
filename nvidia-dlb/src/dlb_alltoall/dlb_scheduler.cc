#include "dlb_alltoall/dlb_scheduler.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace dlb_alltoall {
namespace {

typedef std::uint64_t Size;

std::size_t index(unsigned width, unsigned row, unsigned column) {
    return static_cast<std::size_t>(row) * width + column;
}

std::size_t flow_index(unsigned gpu_count, unsigned source_gpu,
                       unsigned selected_nic, unsigned destination_gpu) {
    return (static_cast<std::size_t>(source_gpu) * gpu_count + selected_nic) * gpu_count +
           destination_gpu;
}

Size sum(const std::vector<Size>& values) {
    return std::accumulate(values.begin(), values.end(), static_cast<Size>(0));
}

std::vector<Size> targets_for_tile(Size total, unsigned gpu_count,
                                   unsigned source_server, unsigned destination_server,
                                   unsigned round_id) {
    std::vector<Size> targets(gpu_count, total / gpu_count);
    const unsigned first = (source_server * 17 + destination_server * 31 + round_id) % gpu_count;
    for (unsigned offset = 0; offset < total % gpu_count; ++offset) {
        ++targets[(first + offset) % gpu_count];
    }
    return targets;
}

TilePlan build_tile(unsigned server_count, unsigned gpu_count,
                    unsigned source_server, unsigned destination_server,
                    const std::vector<Size>& local_server_demand,
                    unsigned round_id, bool direct_priority) {
    const unsigned rank_count = server_count * gpu_count;
    TilePlan tile;
    tile.source_server = source_server;
    tile.destination_server = destination_server;
    tile.input.assign(gpu_count * gpu_count, 0);
    tile.assigned.assign(gpu_count * gpu_count, 0);
    tile.source_to_nic.assign(static_cast<std::size_t>(gpu_count) * gpu_count * gpu_count, 0);
    tile.source_gpu_loads.assign(gpu_count, 0);
    tile.total_records = 0;
    tile.source_staging_records = 0;
    tile.static_destination_repair_records = 0;
    tile.destination_repair_records = 0;
    tile.direct_priority_records = 0;

    for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
        for (unsigned destination_gpu = 0; destination_gpu < gpu_count; ++destination_gpu) {
            const Size records = local_server_demand[index(rank_count, source_gpu,
                                                           destination_server * gpu_count + destination_gpu)];
            tile.input[index(gpu_count, source_gpu, destination_gpu)] = records;
            tile.assigned[index(gpu_count, source_gpu, destination_gpu)] = records;
            tile.source_to_nic[flow_index(gpu_count, source_gpu, source_gpu, destination_gpu)] = records;
            tile.source_gpu_loads[source_gpu] += records;
            tile.total_records += records;
            if (source_gpu != destination_gpu) {
                tile.static_destination_repair_records += records;
            }
        }
    }

    tile.nic_targets = targets_for_tile(tile.total_records, gpu_count,
                                         source_server, destination_server, round_id);
    std::vector<Size> surplus(gpu_count, 0);
    std::vector<Size> deficit(gpu_count, 0);
    for (unsigned gpu = 0; gpu < gpu_count; ++gpu) {
        if (tile.source_gpu_loads[gpu] > tile.nic_targets[gpu]) {
            surplus[gpu] = tile.source_gpu_loads[gpu] - tile.nic_targets[gpu];
        } else {
            deficit[gpu] = tile.nic_targets[gpu] - tile.source_gpu_loads[gpu];
        }
    }

    unsigned selected_nic = 0;
    for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
        Size remaining_surplus = surplus[source_gpu];
        while (remaining_surplus != 0) {
            while (selected_nic < gpu_count && deficit[selected_nic] == 0) {
                ++selected_nic;
            }
            if (selected_nic == gpu_count) {
                throw std::logic_error("DLB source-local planner exhausted NIC deficits early");
            }
            Size remaining_deficit = deficit[selected_nic];
            std::vector<unsigned> destination_order;
            if (direct_priority) {
                destination_order.push_back(selected_nic);
            }
            for (unsigned destination_gpu = 0; destination_gpu < gpu_count; ++destination_gpu) {
                if (!direct_priority || destination_gpu != selected_nic) {
                    destination_order.push_back(destination_gpu);
                }
            }
            for (std::vector<unsigned>::const_iterator it = destination_order.begin();
                 it != destination_order.end() && remaining_surplus != 0 && remaining_deficit != 0;
                 ++it) {
                const unsigned real_destination_gpu = *it;
                const std::size_t source_entry = index(gpu_count, source_gpu, real_destination_gpu);
                const Size moved = std::min(remaining_surplus,
                                            std::min(remaining_deficit, tile.assigned[source_entry]));
                if (moved == 0) {
                    continue;
                }
                tile.assigned[source_entry] -= moved;
                tile.assigned[index(gpu_count, selected_nic, real_destination_gpu)] += moved;
                const std::size_t original_flow =
                    flow_index(gpu_count, source_gpu, source_gpu, real_destination_gpu);
                const std::size_t staged_flow =
                    flow_index(gpu_count, source_gpu, selected_nic, real_destination_gpu);
                if (tile.source_to_nic[original_flow] < moved) {
                    throw std::logic_error("DLB source-local planner moved records it does not own");
                }
                tile.source_to_nic[original_flow] -= moved;
                tile.source_to_nic[staged_flow] += moved;
                remaining_surplus -= moved;
                remaining_deficit -= moved;
                deficit[selected_nic] -= moved;
                tile.source_staging_records += moved;
                if (real_destination_gpu == selected_nic) {
                    tile.direct_priority_records += moved;
                }
            }
            if (remaining_surplus != 0 && remaining_deficit != 0) {
                throw std::logic_error("DLB source-local planner could not fill a NIC deficit");
            }
        }
    }

    for (unsigned destination_nic = 0; destination_nic < gpu_count; ++destination_nic) {
        for (unsigned real_destination_gpu = 0; real_destination_gpu < gpu_count; ++real_destination_gpu) {
            if (destination_nic != real_destination_gpu) {
                tile.destination_repair_records +=
                    tile.assigned[index(gpu_count, destination_nic, real_destination_gpu)];
            }
        }
    }
    return tile;
}

void rebuild(ServerScheduler* scheduler, const std::vector<Size>& local_server_demand,
             unsigned round_id, bool direct_priority) {
    const unsigned rank_count = scheduler->server_count * scheduler->gpus_per_server;
    if (local_server_demand.size() != static_cast<std::size_t>(scheduler->gpus_per_server) * rank_count) {
        throw std::invalid_argument("DLB server-local demand must be M by S*M");
    }
    scheduler->round_id = round_id;
    scheduler->tiles.clear();
    scheduler->source_nic_sent.assign(scheduler->gpus_per_server, 0);
    scheduler->inter_server_records = 0;
    scheduler->source_staging_records = 0;
    scheduler->static_destination_repair_records = 0;
    scheduler->destination_repair_records = 0;
    scheduler->direct_priority_records = 0;

    for (unsigned destination_server = 0; destination_server < scheduler->server_count; ++destination_server) {
        if (destination_server == scheduler->source_server) {
            continue;
        }
        TilePlan tile = build_tile(scheduler->server_count, scheduler->gpus_per_server,
                                   scheduler->source_server, destination_server,
                                   local_server_demand, round_id, direct_priority);
        for (unsigned nic = 0; nic < scheduler->gpus_per_server; ++nic) {
            scheduler->source_nic_sent[nic] += tile.nic_targets[nic];
        }
        scheduler->inter_server_records += tile.total_records;
        scheduler->source_staging_records += tile.source_staging_records;
        scheduler->static_destination_repair_records += tile.static_destination_repair_records;
        scheduler->destination_repair_records += tile.destination_repair_records;
        scheduler->direct_priority_records += tile.direct_priority_records;
        scheduler->tiles.push_back(tile);
    }
}

}  // namespace

void init_server_scheduler(ServerScheduler* scheduler, unsigned server_count,
                           unsigned gpus_per_server, unsigned source_server,
                           const std::vector<Size>& local_server_demand,
                           unsigned round_id, bool direct_priority) {
    if (scheduler == NULL || server_count < 2 || gpus_per_server == 0 || source_server >= server_count) {
        throw std::invalid_argument("DLB server scheduler topology is invalid");
    }
    scheduler->server_count = server_count;
    scheduler->gpus_per_server = gpus_per_server;
    scheduler->source_server = source_server;
    rebuild(scheduler, local_server_demand, round_id, direct_priority);
}

void update_server_scheduler(ServerScheduler* scheduler, const std::vector<Size>& local_server_demand,
                             unsigned round_id, bool direct_priority) {
    if (scheduler == NULL || scheduler->server_count < 2 || scheduler->gpus_per_server == 0) {
        throw std::invalid_argument("DLB server scheduler is not initialized");
    }
    rebuild(scheduler, local_server_demand, round_id, direct_priority);
}

TilePlan plan_destination_tile(unsigned server_count, unsigned gpus_per_server,
                               unsigned source_server, unsigned destination_server,
                               const std::vector<Size>& local_server_demand,
                               unsigned round_id, bool direct_priority) {
    if (server_count < 2 || gpus_per_server == 0 || source_server >= server_count ||
        destination_server >= server_count || source_server == destination_server) {
        throw std::invalid_argument("DLB destination-tile topology is invalid");
    }
    const unsigned rank_count = server_count * gpus_per_server;
    if (local_server_demand.size() != static_cast<std::size_t>(gpus_per_server) * rank_count) {
        throw std::invalid_argument("DLB destination-tile demand must be M by S*M");
    }
    return build_tile(server_count, gpus_per_server, source_server, destination_server,
                      local_server_demand, round_id, direct_priority);
}

void validate_server_scheduler(const ServerScheduler& scheduler) {
    const unsigned gpu_count = scheduler.gpus_per_server;
    if (scheduler.tiles.size() != scheduler.server_count - 1 || scheduler.source_nic_sent.size() != gpu_count) {
        throw std::logic_error("DLB server scheduler dimensions are invalid");
    }
    std::vector<Size> source_nic_sent(gpu_count, 0);
    Size total = 0;
    Size staging = 0;
    Size static_repair = 0;
    Size repair = 0;
    Size direct = 0;
    for (std::vector<TilePlan>::const_iterator it = scheduler.tiles.begin(); it != scheduler.tiles.end(); ++it) {
        const TilePlan& tile = *it;
        if (tile.source_server != scheduler.source_server || tile.destination_server == scheduler.source_server ||
            tile.input.size() != gpu_count * gpu_count || tile.assigned.size() != gpu_count * gpu_count ||
            tile.source_to_nic.size() != static_cast<std::size_t>(gpu_count) * gpu_count * gpu_count ||
            tile.nic_targets.size() != gpu_count) {
            throw std::logic_error("DLB server scheduler tile is invalid");
        }
        if (sum(tile.input) != tile.total_records || sum(tile.assigned) != tile.total_records) {
            throw std::logic_error("DLB server scheduler did not conserve a tile");
        }
        Size expected_staging = 0;
        Size expected_static_repair = 0;
        Size expected_repair = 0;
        Size expected_direct = 0;
        for (unsigned destination_gpu = 0; destination_gpu < gpu_count; ++destination_gpu) {
            Size input_column = 0;
            Size assigned_column = 0;
            for (unsigned row = 0; row < gpu_count; ++row) {
                input_column += tile.input[index(gpu_count, row, destination_gpu)];
                assigned_column += tile.assigned[index(gpu_count, row, destination_gpu)];
                Size source_flow = 0;
                for (unsigned selected_nic = 0; selected_nic < gpu_count; ++selected_nic) {
                    const Size records = tile.source_to_nic[
                        flow_index(gpu_count, row, selected_nic, destination_gpu)];
                    source_flow += records;
                    if (row != selected_nic && selected_nic == destination_gpu) {
                        expected_direct += records;
                    }
                }
                if (source_flow != tile.input[index(gpu_count, row, destination_gpu)]) {
                    throw std::logic_error("DLB source-to-NIC flow changed source demand");
                }
                if (row != destination_gpu) {
                    expected_static_repair += tile.input[index(gpu_count, row, destination_gpu)];
                }
            }
            if (input_column != assigned_column) {
                throw std::logic_error("DLB changed logical destination-GPU demand");
            }
        }
        for (unsigned nic = 0; nic < gpu_count; ++nic) {
            Size assigned_row = 0;
            Size input_row = 0;
            for (unsigned destination_gpu = 0; destination_gpu < gpu_count; ++destination_gpu) {
                input_row += tile.input[index(gpu_count, nic, destination_gpu)];
                assigned_row += tile.assigned[index(gpu_count, nic, destination_gpu)];
                if (nic != destination_gpu) {
                    expected_repair += tile.assigned[index(gpu_count, nic, destination_gpu)];
                }
                Size nic_flow = 0;
                for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
                    nic_flow += tile.source_to_nic[
                        flow_index(gpu_count, source_gpu, nic, destination_gpu)];
                }
                if (nic_flow != tile.assigned[index(gpu_count, nic, destination_gpu)]) {
                    throw std::logic_error("DLB source-to-NIC flow changed selected NIC demand");
                }
            }
            if (input_row != tile.source_gpu_loads[nic]) {
                throw std::logic_error("DLB source GPU load does not match its tile row");
            }
            if (assigned_row != tile.nic_targets[nic]) {
                throw std::logic_error("DLB selected NIC total does not match its target");
            }
            if (input_row > tile.nic_targets[nic]) {
                expected_staging += input_row - tile.nic_targets[nic];
            }
            source_nic_sent[nic] += assigned_row;
        }
        if (tile_nic_gap(tile) > 1) {
            throw std::logic_error("DLB per-pair NIC gap exceeds one record");
        }
        if (expected_staging != tile.source_staging_records ||
            expected_static_repair != tile.static_destination_repair_records ||
            expected_repair != tile.destination_repair_records ||
            expected_direct != tile.direct_priority_records) {
            throw std::logic_error("DLB server scheduler tile accounting is inconsistent");
        }
        total += tile.total_records;
        staging += tile.source_staging_records;
        static_repair += tile.static_destination_repair_records;
        repair += tile.destination_repair_records;
        direct += tile.direct_priority_records;
    }
    if (source_nic_sent != scheduler.source_nic_sent || total != scheduler.inter_server_records ||
        staging != scheduler.source_staging_records || static_repair != scheduler.static_destination_repair_records ||
        repair != scheduler.destination_repair_records || direct != scheduler.direct_priority_records) {
        throw std::logic_error("DLB server scheduler aggregate accounting is inconsistent");
    }
}

unsigned tile_nic_gap(const TilePlan& tile) {
    if (tile.nic_targets.empty()) {
        return 0;
    }
    const std::pair<std::vector<Size>::const_iterator, std::vector<Size>::const_iterator> limits =
        std::minmax_element(tile.nic_targets.begin(), tile.nic_targets.end());
    return static_cast<unsigned>(*limits.second - *limits.first);
}

}  // namespace dlb_alltoall
