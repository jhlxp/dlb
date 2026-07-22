#pragma once

#include <cstdint>

namespace dlb_alltoall {

// A payload-aware DLB Rail operation. All offsets are bytes. The GPU dynamic
// planner chooses source/destination ranges before this descriptor reaches
// the transport kernel. signal_index must be unique among concurrent writers
// to destination_rank; credit_index identifies the symmetric sender-side slot
// returned after destination-side consumption.
struct DlbRailTransfer {
    std::uint32_t destination_rank;
    std::uint32_t signal_index;
    std::uint32_t credit_index;
    std::uint32_t channel_index;
    std::uint64_t source_offset_bytes;
    std::uint64_t destination_offset_bytes;
    std::uint64_t bytes;
};

// Static receiver-side identity for one (source server, final GPU) arrival on
// the calling Rail rank. It is derived from topology, not remote demand. The
// sender encodes the dynamic receive offset and byte count in the metadata
// signal, so the receiver needs no separate arrival header.
struct DlbRailArrival {
    std::uint32_t signal_index;
    std::uint32_t source_rank;
    std::uint32_t credit_index;
    std::uint32_t final_destination_rank;
    std::uint32_t channel_index;
};

static_assert(sizeof(DlbRailTransfer) == 40, "DLB Rail transfer layout changed");
static_assert(sizeof(DlbRailArrival) == 20, "DLB Rail arrival layout changed");

}  // namespace dlb_alltoall
