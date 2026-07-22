#include "fast_alltoall/alltoall_global_scheduler.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Size = std::uint64_t;

struct Stats {
    double mean_us;
    double p50_us;
    double p95_us;
};

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

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

std::vector<Size> make_workload(unsigned server_count, unsigned gpus_per_server,
                                unsigned variant) {
    const unsigned ranks = server_count * gpus_per_server;
    std::vector<Size> workload(static_cast<std::size_t>(ranks) * ranks, 0);
    for (unsigned source = 0; source < ranks; ++source) {
        for (unsigned destination = 0; destination < ranks; ++destination) {
            if (source / gpus_per_server == destination / gpus_per_server) {
                continue;
            }
            const Size key = (static_cast<Size>(variant) << 32) ^
                             (static_cast<Size>(source) << 16) ^ destination;
            const unsigned hotspot = destination % gpus_per_server ==
                                     (source + variant + 3) % gpus_per_server ? 256 : 0;
            workload[index(ranks, source, destination)] = hotspot + mix(key) % 64;
        }
    }
    return workload;
}

Stats summarize(std::vector<double> samples) {
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&samples](double fraction) {
        return samples[static_cast<std::size_t>(fraction * (samples.size() - 1))];
    };
    return {mean, percentile(0.50), percentile(0.95)};
}

void run_case(unsigned server_count, unsigned iterations) {
    constexpr unsigned kGpus = 8;
    constexpr unsigned kVariants = 8;
    constexpr unsigned kWarmups = 20;
    constexpr unsigned kBlocks = 32;
    constexpr Size kBlockTransferBytes = 1 << 20;
    constexpr unsigned kElementBytes = 16;
    std::vector<std::vector<Size>> workloads;
    for (unsigned variant = 0; variant < kVariants; ++variant) {
        workloads.push_back(make_workload(server_count, kGpus, variant));
    }

    GlobalScheduler scheduler{};
    init_global_scheduler(&scheduler, server_count, kGpus, workloads.front().data(), 0,
                          kBlocks, kBlockTransferBytes, kElementBytes);
    flash_scheduler(&scheduler);
    for (unsigned warmup = 0; warmup < kWarmups; ++warmup) {
        update_global_scheduler(&scheduler, workloads[warmup % kVariants].data());
        flash_scheduler(&scheduler);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize during FAST warmup");
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        const Clock::time_point start = Clock::now();
        update_global_scheduler(&scheduler, workloads[iteration % kVariants].data());
        flash_scheduler(&scheduler);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize during FAST measurement");
        const Clock::time_point end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    free_global_scheduler(&scheduler);

    const Stats stats = summarize(samples);
    const unsigned ranks = server_count * kGpus;
    std::cout << "FAST scheduler, S=" << server_count << ", M=" << kGpus
              << " (EP=" << ranks << "): mean=" << std::fixed << std::setprecision(3)
              << stats.mean_us << " us  p50=" << stats.p50_us << " us  p95="
              << stats.p95_us << " us\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1) {
            throw std::invalid_argument("usage: fast_scheduler_benchmark");
        }
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            throw std::runtime_error("FAST scheduler benchmark requires one CUDA device");
        }
        check_cuda(cudaSetDevice(0), "cudaSetDevice(0)");
        std::cout << "FAST control-plane benchmark: update_global_scheduler + flash_scheduler, "
                     "including its CUDA metadata clears/uploads; no payload communication\n";
        run_case(4, 300);
        run_case(8, 300);
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
