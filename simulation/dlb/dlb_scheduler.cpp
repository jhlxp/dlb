#include "dlb_scheduler.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace dlb {
namespace {

std::size_t matrix_index(unsigned width, unsigned row, unsigned column) {
    return static_cast<std::size_t>(row) * width + column;
}

Size sum(const std::vector<Size>& values) {
    return std::accumulate(values.begin(), values.end(), static_cast<Size>(0));
}

std::vector<Size> make_targets(Size total,
                               unsigned gpu_count,
                               unsigned source_server,
                               unsigned destination_server,
                               unsigned round_id) {
    std::vector<Size> targets(gpu_count, total / gpu_count);
    const unsigned remainder = static_cast<unsigned>(total % gpu_count);
    const unsigned first = (source_server * 17 + destination_server * 31 + round_id) % gpu_count;
    for (unsigned offset = 0; offset < remainder; ++offset) {
        ++targets[(first + offset) % gpu_count];
    }
    return targets;
}

TilePlan build_tile(const std::vector<Size>& demand,
                    unsigned server_count,
                    unsigned gpu_count,
                    unsigned source_server,
                    unsigned destination_server,
                    unsigned round_id,
                    bool direct_priority) {
    const unsigned rank_count = server_count * gpu_count;
    TilePlan tile;
    tile.source_server = source_server;
    tile.destination_server = destination_server;
    tile.input.assign(gpu_count * gpu_count, 0);
    tile.assigned.assign(gpu_count * gpu_count, 0);
    tile.source_gpu_loads.assign(gpu_count, 0);
    tile.total_records = 0;
    tile.source_staging_records = 0;
    tile.static_destination_repair_records = 0;
    tile.destination_repair_records = 0;
    tile.direct_priority_records = 0;

    for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
        const unsigned source_rank = source_server * gpu_count + source_gpu;
        for (unsigned destination_gpu = 0; destination_gpu < gpu_count; ++destination_gpu) {
            const unsigned destination_rank = destination_server * gpu_count + destination_gpu;
            const Size records = demand[matrix_index(rank_count, source_rank, destination_rank)];
            tile.input[matrix_index(gpu_count, source_gpu, destination_gpu)] = records;
            tile.assigned[matrix_index(gpu_count, source_gpu, destination_gpu)] = records;
            tile.source_gpu_loads[source_gpu] += records;
            tile.total_records += records;
            if (source_gpu != destination_gpu) {
                tile.static_destination_repair_records += records;
            }
        }
    }

    tile.nic_targets = make_targets(tile.total_records, gpu_count, source_server, destination_server, round_id);
    std::vector<Size> surplus(gpu_count, 0);
    std::vector<Size> deficit(gpu_count, 0);
    for (unsigned gpu = 0; gpu < gpu_count; ++gpu) {
        if (tile.source_gpu_loads[gpu] > tile.nic_targets[gpu]) {
            surplus[gpu] = tile.source_gpu_loads[gpu] - tile.nic_targets[gpu];
        } else {
            deficit[gpu] = tile.nic_targets[gpu] - tile.source_gpu_loads[gpu];
        }
    }

    unsigned destination_nic = 0;
    for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
        Size remaining_surplus = surplus[source_gpu];
        while (remaining_surplus != 0) {
            while (destination_nic < gpu_count && deficit[destination_nic] == 0) {
                ++destination_nic;
            }
            if (destination_nic == gpu_count) {
                throw std::logic_error("DLB exhausted NIC deficits before donor surplus");
            }

            std::vector<unsigned> destination_order;
            if (direct_priority) {
                destination_order.push_back(destination_nic);
            }
            for (unsigned destination_gpu = 0; destination_gpu < gpu_count; ++destination_gpu) {
                if (!direct_priority || destination_gpu != destination_nic) {
                    destination_order.push_back(destination_gpu);
                }
            }

            Size remaining_deficit = deficit[destination_nic];
            for (std::vector<unsigned>::const_iterator it = destination_order.begin();
                 it != destination_order.end() && remaining_surplus != 0 && remaining_deficit != 0;
                 ++it) {
                const unsigned real_destination_gpu = *it;
                const std::size_t entry = matrix_index(gpu_count, source_gpu, real_destination_gpu);
                const Size moved = std::min(remaining_surplus,
                                            std::min(remaining_deficit, tile.assigned[entry]));
                if (moved == 0) {
                    continue;
                }
                tile.assigned[entry] -= moved;
                tile.assigned[matrix_index(gpu_count, destination_nic, real_destination_gpu)] += moved;
                remaining_surplus -= moved;
                remaining_deficit -= moved;
                deficit[destination_nic] -= moved;
                tile.source_staging_records += moved;
                if (real_destination_gpu == destination_nic) {
                    tile.direct_priority_records += moved;
                }
            }
            if (remaining_deficit != 0 && remaining_surplus != 0) {
                throw std::logic_error("DLB could not satisfy a NIC deficit from available donor records");
            }
        }
    }

    for (unsigned nic = 0; nic < gpu_count; ++nic) {
        for (unsigned destination_gpu = 0; destination_gpu < gpu_count; ++destination_gpu) {
            if (nic != destination_gpu) {
                tile.destination_repair_records += tile.assigned[matrix_index(gpu_count, nic, destination_gpu)];
            }
        }
    }
    return tile;
}

}  // namespace

Plan build_plan(const std::vector<Size>& demand,
                unsigned server_count,
                unsigned gpus_per_server,
                unsigned round_id,
                bool direct_priority) {
    if (server_count < 2 || gpus_per_server == 0) {
        throw std::invalid_argument("DLB requires at least two servers and one GPU/NIC per server");
    }
    const unsigned rank_count = server_count * gpus_per_server;
    if (demand.size() != static_cast<std::size_t>(rank_count) * rank_count) {
        throw std::invalid_argument("DLB demand matrix dimensions do not match the topology");
    }

    Plan plan;
    plan.server_count = server_count;
    plan.gpus_per_server = gpus_per_server;
    plan.logical_inter_server_records.assign(server_count * server_count, 0);
    plan.source_nic_sent.assign(server_count * gpus_per_server, 0);
    plan.destination_nic_received.assign(server_count * gpus_per_server, 0);
    plan.inter_server_records = 0;
    plan.source_staging_records = 0;
    plan.static_destination_repair_records = 0;
    plan.destination_repair_records = 0;
    plan.direct_priority_records = 0;

    for (unsigned source_server = 0; source_server < server_count; ++source_server) {
        for (unsigned destination_server = 0; destination_server < server_count; ++destination_server) {
            if (source_server == destination_server) {
                continue;
            }
            TilePlan tile = build_tile(demand, server_count, gpus_per_server,
                                       source_server, destination_server,
                                       round_id, direct_priority);
            plan.logical_inter_server_records[matrix_index(server_count, source_server, destination_server)] = tile.total_records;
            plan.inter_server_records += tile.total_records;
            plan.source_staging_records += tile.source_staging_records;
            plan.static_destination_repair_records += tile.static_destination_repair_records;
            plan.destination_repair_records += tile.destination_repair_records;
            plan.direct_priority_records += tile.direct_priority_records;
            for (unsigned nic = 0; nic < gpus_per_server; ++nic) {
                plan.source_nic_sent[source_server * gpus_per_server + nic] += tile.nic_targets[nic];
                plan.destination_nic_received[destination_server * gpus_per_server + nic] += tile.nic_targets[nic];
            }
            plan.tiles.push_back(tile);
        }
    }
    return plan;
}

void validate_plan(const Plan& plan, const std::vector<Size>& demand) {
    const unsigned server_count = plan.server_count;
    const unsigned gpu_count = plan.gpus_per_server;
    const unsigned rank_count = server_count * gpu_count;
    if (demand.size() != static_cast<std::size_t>(rank_count) * rank_count) {
        throw std::logic_error("DLB validation demand matrix dimensions do not match the plan");
    }
    if (plan.tiles.size() != static_cast<std::size_t>(server_count) * (server_count - 1)) {
        throw std::logic_error("DLB plan does not have one tile per directed server pair");
    }

    std::vector<Size> source_nic_sent(server_count * gpu_count, 0);
    std::vector<Size> destination_nic_received(server_count * gpu_count, 0);
    std::vector<Size> logical_server_records(server_count * server_count, 0);
    Size total = 0;
    Size source_staging = 0;
    Size static_repair = 0;
    Size repair = 0;
    Size direct = 0;

    for (std::vector<TilePlan>::const_iterator it = plan.tiles.begin(); it != plan.tiles.end(); ++it) {
        const TilePlan& tile = *it;
        if (tile.source_server == tile.destination_server ||
            tile.source_server >= server_count || tile.destination_server >= server_count) {
            throw std::logic_error("DLB tile has an invalid server pair");
        }
        if (tile.input.size() != gpu_count * gpu_count ||
            tile.assigned.size() != gpu_count * gpu_count ||
            tile.nic_targets.size() != gpu_count ||
            tile.source_gpu_loads.size() != gpu_count) {
            throw std::logic_error("DLB tile dimensions are invalid");
        }
        if (sum(tile.input) != tile.total_records || sum(tile.assigned) != tile.total_records) {
            throw std::logic_error("DLB tile did not conserve records");
        }

        Size expected_staging = 0;
        Size expected_static_repair = 0;
        Size expected_repair = 0;
        for (unsigned real_destination_gpu = 0; real_destination_gpu < gpu_count; ++real_destination_gpu) {
            Size input_column = 0;
            Size assigned_column = 0;
            for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
                const Size input = tile.input[matrix_index(gpu_count, source_gpu, real_destination_gpu)];
                const Size expected = demand[matrix_index(rank_count,
                                                          tile.source_server * gpu_count + source_gpu,
                                                          tile.destination_server * gpu_count + real_destination_gpu)];
                if (input != expected) {
                    throw std::logic_error("DLB tile input no longer matches logical GPU demand");
                }
                input_column += input;
                assigned_column += tile.assigned[matrix_index(gpu_count, source_gpu, real_destination_gpu)];
                if (source_gpu != real_destination_gpu) {
                    expected_static_repair += input;
                }
            }
            if (input_column != assigned_column) {
                throw std::logic_error("DLB changed a real destination GPU's logical demand");
            }
        }

        for (unsigned nic = 0; nic < gpu_count; ++nic) {
            Size assigned_row = 0;
            Size input_row = 0;
            for (unsigned real_destination_gpu = 0; real_destination_gpu < gpu_count; ++real_destination_gpu) {
                input_row += tile.input[matrix_index(gpu_count, nic, real_destination_gpu)];
                assigned_row += tile.assigned[matrix_index(gpu_count, nic, real_destination_gpu)];
                if (nic != real_destination_gpu) {
                    expected_repair += tile.assigned[matrix_index(gpu_count, nic, real_destination_gpu)];
                }
            }
            if (input_row != tile.source_gpu_loads[nic]) {
                throw std::logic_error("DLB source GPU load does not match its tile row");
            }
            if (assigned_row != tile.nic_targets[nic]) {
                throw std::logic_error("DLB selected NIC load differs from its target");
            }
            if (input_row > tile.nic_targets[nic]) {
                expected_staging += input_row - tile.nic_targets[nic];
            }
            source_nic_sent[tile.source_server * gpu_count + nic] += assigned_row;
            destination_nic_received[tile.destination_server * gpu_count + nic] += assigned_row;
        }

        if (tile_load_gap(tile) > 1) {
            throw std::logic_error("DLB pair Rail gap exceeds one indivisible record");
        }
        if (expected_staging != tile.source_staging_records ||
            expected_static_repair != tile.static_destination_repair_records ||
            expected_repair != tile.destination_repair_records ||
            tile.direct_priority_records > tile.source_staging_records) {
            throw std::logic_error("DLB tile staging or repair accounting is inconsistent");
        }
        logical_server_records[matrix_index(server_count, tile.source_server, tile.destination_server)] += tile.total_records;
        total += tile.total_records;
        source_staging += tile.source_staging_records;
        static_repair += tile.static_destination_repair_records;
        repair += tile.destination_repair_records;
        direct += tile.direct_priority_records;
    }

    if (source_nic_sent != plan.source_nic_sent ||
        destination_nic_received != plan.destination_nic_received ||
        logical_server_records != plan.logical_inter_server_records ||
        total != plan.inter_server_records ||
        source_staging != plan.source_staging_records ||
        static_repair != plan.static_destination_repair_records ||
        repair != plan.destination_repair_records ||
        direct != plan.direct_priority_records) {
        throw std::logic_error("DLB aggregate accounting is inconsistent");
    }
    for (unsigned server = 0; server < server_count; ++server) {
        const std::vector<Size>::const_iterator source_begin = source_nic_sent.begin() + server * gpu_count;
        const std::vector<Size>::const_iterator source_end = source_begin + gpu_count;
        const std::vector<Size>::const_iterator receive_begin = destination_nic_received.begin() + server * gpu_count;
        const std::vector<Size>::const_iterator receive_end = receive_begin + gpu_count;
        if (*std::max_element(source_begin, source_end) - *std::min_element(source_begin, source_end) > server_count - 1 ||
            *std::max_element(receive_begin, receive_end) - *std::min_element(receive_begin, receive_end) > server_count - 1) {
            throw std::logic_error("DLB aggregate NIC gap exceeds the number of server pairs");
        }
    }
}

MetadataFootprint metadata_footprint(unsigned server_count, unsigned gpus_per_server) {
    if (server_count < 2 || gpus_per_server == 0) {
        throw std::invalid_argument("metadata footprint requires a valid topology");
    }
    MetadataFootprint footprint;
    footprint.dlb_counters_per_source_server = gpus_per_server * (server_count - 1);
    footprint.fast_global_server_pair_counters = server_count * server_count;
    footprint.fast_global_schedule_matrix_values = server_count * server_count;
    return footprint;
}

unsigned tile_load_gap(const TilePlan& tile) {
    if (tile.nic_targets.empty()) {
        return 0;
    }
    const std::pair<std::vector<Size>::const_iterator, std::vector<Size>::const_iterator> bounds =
        std::minmax_element(tile.nic_targets.begin(), tile.nic_targets.end());
    return static_cast<unsigned>(*bounds.second - *bounds.first);
}

}  // namespace dlb
