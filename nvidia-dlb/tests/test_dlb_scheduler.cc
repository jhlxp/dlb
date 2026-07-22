#include "dlb_alltoall/dlb_scheduler.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

typedef std::uint64_t Size;

std::size_t index(unsigned width, unsigned row, unsigned column) {
    return static_cast<std::size_t>(row) * width + column;
}

std::vector<Size> make_global_demand(unsigned server_count, unsigned gpu_count) {
    const unsigned rank_count = server_count * gpu_count;
    std::mt19937 generator(20260718);
    std::uniform_int_distribution<unsigned> noise(0, 63);
    std::vector<Size> demand(static_cast<std::size_t>(rank_count) * rank_count, 0);
    for (unsigned source = 0; source < rank_count; ++source) {
        for (unsigned destination = 0; destination < rank_count; ++destination) {
            if (source / gpu_count == destination / gpu_count) {
                continue;
            }
            const unsigned hotspot = destination % gpu_count == (source + 2) % gpu_count ? 256 : 0;
            demand[index(rank_count, source, destination)] = hotspot + noise(generator);
        }
    }
    return demand;
}

std::vector<Size> local_source_rows(const std::vector<Size>& global_demand,
                                    unsigned server_count, unsigned gpu_count,
                                    unsigned source_server) {
    const unsigned rank_count = server_count * gpu_count;
    std::vector<Size> local(static_cast<std::size_t>(gpu_count) * rank_count, 0);
    for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
        for (unsigned destination_rank = 0; destination_rank < rank_count; ++destination_rank) {
            local[index(rank_count, source_gpu, destination_rank)] =
                global_demand[index(rank_count, source_server * gpu_count + source_gpu, destination_rank)];
        }
    }
    return local;
}

void test_direct_priority() {
    const unsigned server_count = 2;
    const unsigned gpu_count = 2;
    const unsigned rank_count = server_count * gpu_count;
    std::vector<Size> local(static_cast<std::size_t>(gpu_count) * rank_count, 0);
    local[index(rank_count, 0, 2)] = 4;
    local[index(rank_count, 0, 3)] = 6;
    local[index(rank_count, 1, 2)] = 2;
    dlb_alltoall::ServerScheduler direct;
    dlb_alltoall::ServerScheduler ordinary;
    dlb_alltoall::init_server_scheduler(&direct, server_count, gpu_count, 0, local, 0, true);
    dlb_alltoall::init_server_scheduler(&ordinary, server_count, gpu_count, 0, local, 0, false);
    dlb_alltoall::validate_server_scheduler(direct);
    dlb_alltoall::validate_server_scheduler(ordinary);
    assert(direct.destination_repair_records == 4);
    assert(ordinary.destination_repair_records == 12);
    assert(direct.direct_priority_records == 4);
}

void test_source_local_schedulers() {
    const unsigned server_count = 4;
    const unsigned gpu_count = 8;
    const std::vector<Size> global_demand = make_global_demand(server_count, gpu_count);
    std::vector<Size> received_by_nic(server_count * gpu_count, 0);
    Size total_sent = 0;
    Size total_received = 0;

    for (unsigned source_server = 0; source_server < server_count; ++source_server) {
        dlb_alltoall::ServerScheduler scheduler;
        dlb_alltoall::init_server_scheduler(
            &scheduler, server_count, gpu_count, source_server,
            local_source_rows(global_demand, server_count, gpu_count, source_server), 3, true);
        dlb_alltoall::validate_server_scheduler(scheduler);
        assert(scheduler.tiles.size() == server_count - 1);
        for (std::vector<dlb_alltoall::TilePlan>::const_iterator tile = scheduler.tiles.begin();
             tile != scheduler.tiles.end(); ++tile) {
            assert(dlb_alltoall::tile_nic_gap(*tile) <= 1);
            for (unsigned nic = 0; nic < gpu_count; ++nic) {
                received_by_nic[tile->destination_server * gpu_count + nic] += tile->nic_targets[nic];
            }
        }
        for (unsigned nic = 0; nic < gpu_count; ++nic) {
            total_sent += scheduler.source_nic_sent[nic];
        }
    }

    for (std::vector<Size>::const_iterator value = received_by_nic.begin();
         value != received_by_nic.end(); ++value) {
        total_received += *value;
    }
    assert(total_sent == total_received);
    for (unsigned destination_server = 0; destination_server < server_count; ++destination_server) {
        Size minimum = received_by_nic[destination_server * gpu_count];
        Size maximum = minimum;
        for (unsigned nic = 1; nic < gpu_count; ++nic) {
            const Size value = received_by_nic[destination_server * gpu_count + nic];
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        assert(maximum - minimum <= server_count - 1);
    }
    std::cout << "NVIDIA DLB host scheduler: local source-server plans passed; records="
              << total_sent << std::endl;
}

void test_zero_update_and_variable_topologies() {
    const unsigned server_count = 3;
    const unsigned gpu_count = 5;
    const unsigned rank_count = server_count * gpu_count;
    std::vector<Size> zero(static_cast<std::size_t>(gpu_count) * rank_count, 0);
    dlb_alltoall::ServerScheduler scheduler;
    dlb_alltoall::init_server_scheduler(&scheduler, server_count, gpu_count, 1, zero, 0, true);
    dlb_alltoall::validate_server_scheduler(scheduler);
    assert(scheduler.inter_server_records == 0);
    for (std::vector<dlb_alltoall::TilePlan>::const_iterator tile = scheduler.tiles.begin();
         tile != scheduler.tiles.end(); ++tile) {
        assert(dlb_alltoall::tile_nic_gap(*tile) == 0);
    }

    const std::vector<Size> global_demand = make_global_demand(server_count, gpu_count);
    dlb_alltoall::update_server_scheduler(
        &scheduler, local_source_rows(global_demand, server_count, gpu_count, 1), 11, true);
    dlb_alltoall::validate_server_scheduler(scheduler);
    assert(scheduler.inter_server_records != 0);

    const unsigned topologies[][2] = {{2, 1}, {2, 7}, {5, 3}};
    for (unsigned case_id = 0; case_id < sizeof(topologies) / sizeof(topologies[0]); ++case_id) {
        const unsigned servers = topologies[case_id][0];
        const unsigned gpus = topologies[case_id][1];
        const std::vector<Size> demand = make_global_demand(servers, gpus);
        for (unsigned source_server = 0; source_server < servers; ++source_server) {
            dlb_alltoall::ServerScheduler plan;
            dlb_alltoall::init_server_scheduler(
                &plan, servers, gpus, source_server,
                local_source_rows(demand, servers, gpus, source_server), case_id, true);
            dlb_alltoall::validate_server_scheduler(plan);
            for (std::vector<dlb_alltoall::TilePlan>::const_iterator tile = plan.tiles.begin();
                 tile != plan.tiles.end(); ++tile) {
                assert(dlb_alltoall::tile_nic_gap(*tile) <= 1);
            }
        }
    }
}

}  // namespace

int main() {
    test_direct_priority();
    test_source_local_schedulers();
    test_zero_update_and_variable_topologies();
    std::cout << "NVIDIA DLB scheduler tests passed" << std::endl;
    return 0;
}
