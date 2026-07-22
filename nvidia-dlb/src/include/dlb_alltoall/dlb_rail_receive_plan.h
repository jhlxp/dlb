#pragma once

#include "dlb_alltoall/dlb_rail_protocol.h"

#include <vector>

namespace dlb_alltoall {

// Receiver-local, topology-only plan. It contains every possible sender slot;
// zero-byte transfers still signal their epoch, so waiting never depends on a
// cross-server demand exchange.
struct DlbRailReceivePlan {
    unsigned server_count;
    unsigned gpus_per_server;
    unsigned destination_server;
    unsigned rail_rank;
    unsigned channel_count;
    std::vector<DlbRailArrival> arrivals;
};

DlbRailReceivePlan materialize_rail_receive_plan(unsigned server_count,
                                                  unsigned gpus_per_server,
                                                  unsigned destination_server,
                                                  unsigned rail_rank,
                                                  unsigned channel_count);
void validate_rail_receive_plan(const DlbRailReceivePlan& plan);

}  // namespace dlb_alltoall
