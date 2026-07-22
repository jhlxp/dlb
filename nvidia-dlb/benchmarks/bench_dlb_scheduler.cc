#include "dlb_alltoall/dlb_scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Size = std::uint64_t;
using Clock = std::chrono::steady_clock;

struct Stats {
    double mean_us;
    double p50_us;
    double p95_us;
};

std::size_t index(unsigned width, unsigned row, unsigned column) {
    return static_cast<std::size_t>(row) * width + column;
}

Size mix(Size value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<Size> make_local_demand(unsigned server_count, unsigned gpu_count,
                                    unsigned source_server, unsigned variant) {
    const unsigned rank_count = server_count * gpu_count;
    std::vector<Size> demand(static_cast<std::size_t>(gpu_count) * rank_count, 0);
    for (unsigned source_gpu = 0; source_gpu < gpu_count; ++source_gpu) {
        for (unsigned destination_rank = 0; destination_rank < rank_count; ++destination_rank) {
            if (destination_rank / gpu_count == source_server) {
                continue;
            }
            const unsigned destination_gpu = destination_rank % gpu_count;
            const Size key = (static_cast<Size>(variant) << 32) ^
                             (static_cast<Size>(source_server) << 24) ^
                             (static_cast<Size>(source_gpu) << 16) ^ destination_rank;
            const Size hotspot = destination_gpu == (source_gpu + variant + 3) % gpu_count ? 256 : 0;
            demand[index(rank_count, source_gpu, destination_rank)] = hotspot + mix(key) % 64;
        }
    }
    return demand;
}

Stats summarize(std::vector<double> samples) {
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&samples](double fraction) {
        return samples[static_cast<std::size_t>(fraction * (samples.size() - 1))];
    };
    return {mean, percentile(0.50), percentile(0.95)};
}

template <typename Function>
double elapsed_us(Function&& function) {
    const Clock::time_point start = Clock::now();
    function();
    const Clock::time_point end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

void print_stats(const char* label, const Stats& stats) {
    std::cout << "  " << std::left << std::setw(37) << label << std::right
              << "mean=" << std::fixed << std::setprecision(3) << stats.mean_us
              << " us  p50=" << stats.p50_us << " us  p95=" << stats.p95_us << " us\n";
}

class TileWorkerPool {
public:
    TileWorkerPool(unsigned worker_count, unsigned server_count, unsigned gpu_count,
                   unsigned source_server, std::vector<unsigned> destinations)
        : server_count_(server_count), gpu_count_(gpu_count), source_server_(source_server),
          destinations_(std::move(destinations)), results_(destinations_.size()), next_(0),
          completed_(0), generation_(0), stop_(false), demand_(nullptr), round_id_(0) {
        for (unsigned worker = 0; worker < worker_count; ++worker) {
            workers_.emplace_back(&TileWorkerPool::worker_loop, this);
        }
    }

    ~TileWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        work_ready_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    const std::vector<dlb_alltoall::TilePlan>& run(const std::vector<Size>& demand,
                                                    unsigned round_id) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            demand_ = &demand;
            round_id_ = round_id;
            next_.store(0, std::memory_order_relaxed);
            completed_.store(0, std::memory_order_relaxed);
            ++generation_;
        }
        work_ready_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] {
            return completed_.load(std::memory_order_acquire) == destinations_.size();
        });
        return results_;
    }

private:
    void worker_loop() {
        unsigned observed_generation = 0;
        while (true) {
            const std::vector<Size>* demand = nullptr;
            unsigned round_id = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [this, &observed_generation] {
                    return stop_ || generation_ != observed_generation;
                });
                if (stop_) {
                    return;
                }
                observed_generation = generation_;
                demand = demand_;
                round_id = round_id_;
            }
            while (true) {
                const unsigned task = next_.fetch_add(1, std::memory_order_relaxed);
                if (task >= destinations_.size()) {
                    break;
                }
                results_[task] = dlb_alltoall::plan_destination_tile(
                    server_count_, gpu_count_, source_server_, destinations_[task],
                    *demand, round_id, true);
                if (completed_.fetch_add(1, std::memory_order_release) + 1 == destinations_.size()) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    done_.notify_one();
                }
            }
        }
    }

    unsigned server_count_;
    unsigned gpu_count_;
    unsigned source_server_;
    std::vector<unsigned> destinations_;
    std::vector<dlb_alltoall::TilePlan> results_;
    std::vector<std::thread> workers_;
    std::atomic<unsigned> next_;
    std::atomic<unsigned> completed_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable done_;
    unsigned generation_;
    bool stop_;
    const std::vector<Size>* demand_;
    unsigned round_id_;
};

void run_case(unsigned server_count, unsigned gpu_count, unsigned iterations) {
    constexpr unsigned kVariants = 16;
    constexpr unsigned kWarmups = 100;
    const unsigned source_server = 0;
    std::vector<std::vector<Size>> demands;
    for (unsigned variant = 0; variant < kVariants; ++variant) {
        demands.push_back(make_local_demand(server_count, gpu_count, source_server, variant));
    }
    std::vector<unsigned> destinations;
    for (unsigned destination_server = 0; destination_server < server_count; ++destination_server) {
        if (destination_server != source_server) {
            destinations.push_back(destination_server);
        }
    }

    dlb_alltoall::ServerScheduler scheduler;
    dlb_alltoall::init_server_scheduler(&scheduler, server_count, gpu_count, source_server,
                                        demands[0], 0, true);
    TileWorkerPool workers(destinations.size(), server_count, gpu_count, source_server, destinations);
    for (unsigned warmup = 0; warmup < kWarmups; ++warmup) {
        const std::vector<Size>& demand = demands[warmup % kVariants];
        dlb_alltoall::update_server_scheduler(&scheduler, demand, warmup, true);
        workers.run(demand, warmup);
    }

    std::vector<double> update_samples;
    std::vector<double> sequential_tile_samples;
    std::vector<double> parallel_tile_samples;
    std::vector<double> validation_samples;
    update_samples.reserve(iterations);
    sequential_tile_samples.reserve(iterations);
    parallel_tile_samples.reserve(iterations);
    validation_samples.reserve(iterations);

    Size checksum = 0;
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        const std::vector<Size>& demand = demands[iteration % kVariants];
        update_samples.push_back(elapsed_us([&] {
            dlb_alltoall::update_server_scheduler(&scheduler, demand, iteration, true);
        }));
        sequential_tile_samples.push_back(elapsed_us([&] {
            for (unsigned destination_server : destinations) {
                const dlb_alltoall::TilePlan tile = dlb_alltoall::plan_destination_tile(
                    server_count, gpu_count, source_server, destination_server, demand, iteration, true);
                checksum += tile.total_records;
            }
        }));
        parallel_tile_samples.push_back(elapsed_us([&] {
            const std::vector<dlb_alltoall::TilePlan>& tiles = workers.run(demand, iteration);
            for (const dlb_alltoall::TilePlan& tile : tiles) {
                checksum += tile.total_records;
            }
        }));
        validation_samples.push_back(elapsed_us([&] {
            dlb_alltoall::validate_server_scheduler(scheduler);
        }));
    }

    const std::size_t remote_count_bytes =
        static_cast<std::size_t>(gpu_count) * (server_count - 1) * gpu_count * sizeof(Size);
    const std::size_t current_input_bytes =
        static_cast<std::size_t>(gpu_count) * server_count * gpu_count * sizeof(Size);

    std::cout << "\nS=" << server_count << ", M=" << gpu_count
              << " (EP=" << server_count * gpu_count << "), independent tiles="
              << destinations.size() << '\n';
    std::cout << "  Local DLB remote-count collection: " << remote_count_bytes
              << " B/server; current input container: " << current_input_bytes << " B/server\n";
    print_stats("Current ServerScheduler update", summarize(update_samples));
    print_stats("Tile plans, sequential", summarize(sequential_tile_samples));
    print_stats("Tile plans, persistent worker pool", summarize(parallel_tile_samples));
    print_stats("Validation only (not production work)", summarize(validation_samples));
    std::cout << "  checksum=" << checksum << '\n';
}

unsigned parse_iterations(int argc, char** argv) {
    if (argc == 1) {
        return 5000;
    }
    if (argc == 3 && std::string(argv[1]) == "--iterations") {
        const unsigned iterations = static_cast<unsigned>(std::stoul(argv[2]));
        if (iterations != 0) {
            return iterations;
        }
    }
    throw std::invalid_argument("usage: bench_dlb_scheduler [--iterations N]");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const unsigned iterations = parse_iterations(argc, argv);
        std::cout << "DLB control-plane benchmark: CPU only, excluding local AllGather and GPU communication\n";
        std::cout << "One controller runs per server, so the parallel tile line is the planning critical path.\n";
        run_case(2, 4, iterations);
        for (unsigned server_count : {4U, 8U, 12U}) {
            run_case(server_count, 8, iterations);
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
