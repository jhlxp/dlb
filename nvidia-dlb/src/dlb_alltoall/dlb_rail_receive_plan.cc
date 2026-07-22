#include "dlb_alltoall/dlb_rail_receive_plan.h"

#include <limits>
#include <stdexcept>

namespace dlb_alltoall {
namespace {

std::uint32_t checked_slot(unsigned server, unsigned local_rank, unsigned gpus_per_server) {
    const std::uint64_t value = static_cast<std::uint64_t>(server) * gpus_per_server + local_rank;
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("DLB Rail receive slot exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(value);
}

std::uint32_t checked_channel_slot(unsigned server, unsigned local_rank,
                                   unsigned channel, unsigned gpus_per_server,
                                   unsigned channel_count) {
    const std::uint64_t base = checked_slot(server, local_rank, gpus_per_server);
    const std::uint64_t value = base * channel_count + channel;
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("DLB Rail channel slot exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(value);
}

}  // namespace

DlbRailReceivePlan materialize_rail_receive_plan(unsigned server_count,
                                                  unsigned gpus_per_server,
                                                  unsigned destination_server,
                                                  unsigned rail_rank,
                                                  unsigned channel_count) {
    if (server_count == 0 || gpus_per_server == 0 || destination_server >= server_count ||
        rail_rank >= gpus_per_server || channel_count == 0) {
        throw std::invalid_argument("DLB Rail receive plan topology is invalid");
    }
    DlbRailReceivePlan plan;
    plan.server_count = server_count;
    plan.gpus_per_server = gpus_per_server;
    plan.destination_server = destination_server;
    plan.rail_rank = rail_rank;
    plan.channel_count = channel_count;
    plan.arrivals.reserve(static_cast<std::size_t>(server_count - 1) *
                          gpus_per_server * channel_count);
    for (unsigned source_server = 0; source_server < server_count; ++source_server) {
        if (source_server == destination_server) {
            continue;
        }
        for (unsigned final_destination = 0; final_destination < gpus_per_server;
             ++final_destination) {
            const std::uint32_t source_rank =
                checked_slot(source_server, rail_rank, gpus_per_server);
            for (unsigned channel = 0; channel < channel_count; ++channel) {
                const std::uint32_t signal_index = checked_channel_slot(
                    source_server, final_destination, channel,
                    gpus_per_server, channel_count);
                const std::uint32_t credit_index = checked_channel_slot(
                    destination_server, final_destination, channel,
                    gpus_per_server, channel_count);
                plan.arrivals.push_back({signal_index, source_rank, credit_index,
                                         final_destination, channel});
            }
        }
    }
    validate_rail_receive_plan(plan);
    return plan;
}

void validate_rail_receive_plan(const DlbRailReceivePlan& plan) {
    if (plan.server_count == 0 || plan.gpus_per_server == 0 ||
        plan.destination_server >= plan.server_count || plan.rail_rank >= plan.gpus_per_server ||
        plan.channel_count == 0 ||
        plan.arrivals.size() != static_cast<std::size_t>(plan.server_count - 1) *
                                    plan.gpus_per_server * plan.channel_count) {
        throw std::logic_error("DLB Rail receive plan dimensions are inconsistent");
    }
    std::size_t index = 0;
    for (unsigned source_server = 0; source_server < plan.server_count; ++source_server) {
        if (source_server == plan.destination_server) {
            continue;
        }
        for (unsigned final_destination = 0; final_destination < plan.gpus_per_server;
            ++final_destination) {
            for (unsigned channel = 0; channel < plan.channel_count;
                 ++channel, ++index) {
                const DlbRailArrival& arrival = plan.arrivals[index];
                if (arrival.signal_index != checked_channel_slot(
                        source_server, final_destination, channel,
                        plan.gpus_per_server, plan.channel_count) ||
                    arrival.source_rank != checked_slot(
                        source_server, plan.rail_rank, plan.gpus_per_server) ||
                    arrival.credit_index != checked_channel_slot(
                        plan.destination_server, final_destination, channel,
                        plan.gpus_per_server, plan.channel_count) ||
                    arrival.final_destination_rank != final_destination ||
                    arrival.channel_index != channel) {
                    throw std::logic_error("DLB Rail receive arrival is inconsistent");
                }
            }
        }
    }
}

}  // namespace dlb_alltoall
