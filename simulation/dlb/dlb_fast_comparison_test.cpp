#include "dlb_scheduler.h"

#include "all2all.h"
#include "local.h"
#include "matrix.h"
#include <algorithm>

#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

typedef dlb::Size Size;

std::size_t index(unsigned width, unsigned row, unsigned column) {
    return static_cast<std::size_t>(row) * width + column;
}

std::vector<Size> make_workload(unsigned server_count, unsigned gpu_count, unsigned seed) {
    const unsigned rank_count = server_count * gpu_count;
    std::mt19937 generator(seed);
    std::uniform_int_distribution<unsigned> noise(0, 31);
    std::vector<Size> workload(static_cast<std::size_t>(rank_count) * rank_count, 0);
    for (unsigned source = 0; source < rank_count; ++source) {
        for (unsigned destination = 0; destination < rank_count; ++destination) {
            if (source / gpu_count == destination / gpu_count) {
                continue;
            }
            const unsigned hotspot = (destination % gpu_count == (source + 3) % gpu_count) ? 96 : 0;
            workload[index(rank_count, source, destination)] = hotspot + noise(generator);
        }
    }
    return workload;
}

std::vector<unsigned> fast_server_rail_matrix(const std::vector<Size>& workload,
                                              unsigned server_count,
                                              unsigned gpu_count) {
    const unsigned rank_count = server_count * gpu_count;
    std::vector<unsigned> matrix(server_count * server_count, 0);
    for (unsigned source_server = 0; source_server < server_count; ++source_server) {
        for (unsigned destination_server = 0; destination_server < server_count; ++destination_server) {
            if (source_server == destination_server) {
                continue;
            }
            Size total = 0;
            for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
                for (unsigned destination_gpu = 0; destination_gpu < gpu_count; ++destination_gpu) {
                    total += workload[index(rank_count,
                                            source_server * gpu_count + source_gpu,
                                            destination_server * gpu_count + destination_gpu)];
                }
            }
            matrix[index(server_count, source_server, destination_server)] =
                static_cast<unsigned>((total + gpu_count - 1) / gpu_count);
        }
    }
    return matrix;
}

Size max_row_or_column_sum(const std::vector<unsigned>& matrix, unsigned dimension) {
    Size maximum = 0;
    for (unsigned index_value = 0; index_value < dimension; ++index_value) {
        Size row = 0;
        Size column = 0;
        for (unsigned other = 0; other < dimension; ++other) {
            row += matrix[index(dimension, index_value, other)];
            column += matrix[index(dimension, other, index_value)];
        }
        maximum = std::max(maximum, std::max(row, column));
    }
    return maximum;
}

Size max_value(const std::vector<Size>& values) {
    return values.empty() ? 0 : *std::max_element(values.begin(), values.end());
}

struct FastMetrics {
    unsigned birkhoff_stages;
    Size source_staging_records;
    Size scaleout_bottleneck_records;
};

FastMetrics fast_full_planner_metrics(const std::vector<Size>& workload,
                                      unsigned server_count,
                                      unsigned gpu_count) {
    const unsigned rank_count = server_count * gpu_count;
    std::vector<unsigned> demand(static_cast<std::size_t>(rank_count) * rank_count, 0);
    for (std::size_t entry = 0; entry < demand.size(); ++entry) {
        if (workload[entry] > static_cast<Size>(UINT_MAX)) {
            throw std::logic_error("FAST reference input exceeds its uint scheduler element type");
        }
        demand[entry] = static_cast<unsigned>(workload[entry]);
    }

    // Exercise FAST's existing per-tile LocalScheduler before its global
    // Birkhoff phase. This is the path DLB is compared with, not Birkhoff alone.
    Size source_staging_records = 0;
    for (unsigned source_server = 0; source_server < server_count; ++source_server) {
        LocalScheduler local(&demand[source_server * rank_count * gpu_count],
                             gpu_count, server_count, source_server, H100);
        local.prepare_load_balance();
        for (unsigned destination_server = 0; destination_server < server_count; ++destination_server) {
            if (source_server == destination_server) {
                continue;
            }
            struct load_balance_result result;
            result.balance = new unsigned[gpu_count * gpu_count]();
            result.dispatch = new unsigned[gpu_count * gpu_count]();
            local.server2server_balance(destination_server, result);
            for (unsigned entry = 0; entry < gpu_count * gpu_count; ++entry) {
                source_staging_records += result.balance[entry];
            }
            delete[] result.balance;
            delete[] result.dispatch;
        }
    }

    std::vector<unsigned> server_matrix = fast_server_rail_matrix(workload, server_count, gpu_count);
    Matrix matrix(server_matrix.data(), server_count);
    FastAll2All fast(&matrix, gpu_count, ETHER400, H100);
    fast.to_scaled_doubly_stochastic_matrix();
    fast.decompose();
    if (!fast.verify_decomposition()) {
        throw std::logic_error("FAST Birkhoff decomposition did not verify");
    }
    FastMetrics metrics;
    metrics.birkhoff_stages = static_cast<unsigned>(fast.p_sets.size());
    metrics.source_staging_records = source_staging_records;
    metrics.scaleout_bottleneck_records = max_row_or_column_sum(server_matrix, server_count);
    return metrics;
}

void test_direct_priority() {
    const unsigned server_count = 2;
    const unsigned gpu_count = 2;
    const unsigned rank_count = server_count * gpu_count;
    std::vector<Size> workload(static_cast<std::size_t>(rank_count) * rank_count, 0);
    // S0 -> S1 tile: row loads are 10 and 2. NIC 1 has a deficit of four;
    // direct priority must take its destination-GPU-1 records first.
    workload[index(rank_count, 0, 2)] = 4;
    workload[index(rank_count, 0, 3)] = 6;
    workload[index(rank_count, 1, 2)] = 2;

    const dlb::Plan direct = dlb::build_plan(workload, server_count, gpu_count, 0, true);
    const dlb::Plan ordinary = dlb::build_plan(workload, server_count, gpu_count, 0, false);
    dlb::validate_plan(direct, workload);
    dlb::validate_plan(ordinary, workload);
    assert(direct.source_staging_records == 4);
    assert(direct.direct_priority_records == 4);
    assert(direct.destination_repair_records == 4);
    assert(ordinary.destination_repair_records == 12);
    assert(direct.destination_repair_records < ordinary.destination_repair_records);
}

void test_zero_and_invalid_inputs() {
    const unsigned server_count = 3;
    const unsigned gpu_count = 5;
    const unsigned rank_count = server_count * gpu_count;
    const std::vector<Size> zero(static_cast<std::size_t>(rank_count) * rank_count, 0);
    const dlb::Plan plan = dlb::build_plan(zero, server_count, gpu_count, 11, true);
    dlb::validate_plan(plan, zero);
    assert(plan.inter_server_records == 0);
    for (std::vector<dlb::TilePlan>::const_iterator tile = plan.tiles.begin(); tile != plan.tiles.end(); ++tile) {
        assert(dlb::tile_load_gap(*tile) == 0);
    }

    bool rejected = false;
    try {
        dlb::build_plan(std::vector<Size>(3, 0), server_count, gpu_count, 0, true);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

void test_dlb_and_fast(unsigned server_count, unsigned gpu_count, unsigned seed) {
    const std::vector<Size> workload = make_workload(server_count, gpu_count, seed);
    const dlb::Plan plan = dlb::build_plan(workload, server_count, gpu_count, 7, true);
    dlb::validate_plan(plan, workload);
    assert(plan.tiles.size() == static_cast<std::size_t>(server_count) * (server_count - 1));
    for (std::vector<dlb::TilePlan>::const_iterator tile = plan.tiles.begin(); tile != plan.tiles.end(); ++tile) {
        assert(dlb::tile_load_gap(*tile) <= 1);
    }

    const FastMetrics fast = fast_full_planner_metrics(workload, server_count, gpu_count);
    assert(fast.birkhoff_stages > 0);
    const dlb::MetadataFootprint metadata = dlb::metadata_footprint(server_count, gpu_count);
    assert(metadata.dlb_counters_per_source_server == gpu_count * (server_count - 1));
    assert(metadata.fast_global_server_pair_counters == server_count * server_count);

    std::cout << "case S=" << server_count << " M=" << gpu_count
              << ": records=" << plan.inter_server_records
              << ", DLB source staging=" << plan.source_staging_records
              << ", DLB destination repair=" << plan.destination_repair_records
              << ", FAST source staging=" << fast.source_staging_records
              << ", FAST Birkhoff stages=" << fast.birkhoff_stages
              << ", DLB local counters/server=" << metadata.dlb_counters_per_source_server
              << ", FAST global matrix=" << metadata.fast_global_schedule_matrix_values
              << std::endl;
}

void benchmark_scheduler_cpu() {
    const unsigned server_count = 4;
    const unsigned gpu_count = 8;
    const unsigned iterations = 100;
    const std::vector<Size> workload = make_workload(server_count, gpu_count, 20260718);

    const std::chrono::steady_clock::time_point dlb_start = std::chrono::steady_clock::now();
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        const dlb::Plan plan = dlb::build_plan(workload, server_count, gpu_count, iteration, true);
        assert(plan.inter_server_records != 0);
    }
    const std::chrono::steady_clock::time_point dlb_end = std::chrono::steady_clock::now();

    const std::chrono::steady_clock::time_point fast_start = std::chrono::steady_clock::now();
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        assert(fast_full_planner_metrics(workload, server_count, gpu_count).birkhoff_stages != 0);
    }
    const std::chrono::steady_clock::time_point fast_end = std::chrono::steady_clock::now();

    const double dlb_us = std::chrono::duration<double, std::micro>(dlb_end - dlb_start).count() / iterations;
    const double fast_us = std::chrono::duration<double, std::micro>(fast_end - fast_start).count() / iterations;
    std::cout << "CPU planner benchmark only (not GPU communication): DLB=" << dlb_us
              << " us/plan, FAST local+global Birkhoff=" << fast_us << " us/plan" << std::endl;
}

void write_benchmark_csv(const std::string& path) {
    const unsigned gpu_count = 8;
    const unsigned server_counts[] = {2, 3, 4, 5, 6, 8};
    std::ofstream output(path.c_str());
    if (!output) {
        throw std::runtime_error("could not open DLB/FAST benchmark CSV");
    }
    output << "servers,gpus,inter_server_records,dlb_planner_us,fast_planner_us,"
              "dlb_source_staging_records,fast_source_staging_records,"
              "dlb_destination_repair_records,dlb_scaleout_bottleneck_records,"
              "fast_scaleout_bottleneck_records,fast_birkhoff_stages,"
              "dlb_local_counters_per_server,fast_global_matrix_values\n";
    output << std::fixed << std::setprecision(3);

    for (unsigned case_id = 0; case_id < sizeof(server_counts) / sizeof(server_counts[0]); ++case_id) {
        const unsigned server_count = server_counts[case_id];
        const unsigned iterations = server_count <= 6 ? 100 : 50;
        const std::vector<Size> workload = make_workload(server_count, gpu_count, 20260718 + case_id);
        const dlb::Plan dlb_plan = dlb::build_plan(workload, server_count, gpu_count, case_id, true);
        dlb::validate_plan(dlb_plan, workload);
        const FastMetrics fast = fast_full_planner_metrics(workload, server_count, gpu_count);

        const std::chrono::steady_clock::time_point dlb_start = std::chrono::steady_clock::now();
        for (unsigned iteration = 0; iteration < iterations; ++iteration) {
            const dlb::Plan plan = dlb::build_plan(workload, server_count, gpu_count, iteration, true);
            assert(plan.inter_server_records == dlb_plan.inter_server_records);
        }
        const std::chrono::steady_clock::time_point dlb_end = std::chrono::steady_clock::now();

        const std::chrono::steady_clock::time_point fast_start = std::chrono::steady_clock::now();
        for (unsigned iteration = 0; iteration < iterations; ++iteration) {
            assert(fast_full_planner_metrics(workload, server_count, gpu_count).birkhoff_stages != 0);
        }
        const std::chrono::steady_clock::time_point fast_end = std::chrono::steady_clock::now();

        const dlb::MetadataFootprint metadata = dlb::metadata_footprint(server_count, gpu_count);
        const double dlb_us = std::chrono::duration<double, std::micro>(dlb_end - dlb_start).count() / iterations;
        const double fast_us = std::chrono::duration<double, std::micro>(fast_end - fast_start).count() / iterations;
        output << server_count << ',' << gpu_count << ',' << dlb_plan.inter_server_records << ','
               << dlb_us << ',' << fast_us << ','
               << dlb_plan.source_staging_records << ',' << fast.source_staging_records << ','
               << dlb_plan.destination_repair_records << ','
               << std::max(max_value(dlb_plan.source_nic_sent),
                           max_value(dlb_plan.destination_nic_received)) << ','
               << fast.scaleout_bottleneck_records << ','
               << fast.birkhoff_stages << ',' << metadata.dlb_counters_per_source_server << ','
               << metadata.fast_global_schedule_matrix_values << '\n';
    }
    std::cout << "wrote FAST/DLB benchmark CSV to " << path << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    test_direct_priority();
    test_zero_and_invalid_inputs();
    test_dlb_and_fast(2, 2, 7);
    test_dlb_and_fast(4, 8, 19);
    test_dlb_and_fast(6, 4, 31);
    if (argc == 1) {
        benchmark_scheduler_cpu();
    } else if (argc == 3 && std::string(argv[1]) == "--benchmark-csv") {
        write_benchmark_csv(argv[2]);
    } else {
        throw std::invalid_argument("usage: dlb_fast_comparison_test [--benchmark-csv path]");
    }
    std::cout << "DLB/FAST simulation comparison tests passed" << std::endl;
    return 0;
}
