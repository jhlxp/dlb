#include "dlb_alltoall/dlb_moe_runtime.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace dlb_alltoall {
namespace {

constexpr unsigned kPackThreads = 128;
constexpr std::uint32_t kMaxRailGpus = 8;
constexpr std::uint32_t kMaxTopK = 8;
constexpr std::uint64_t kDispatchPayloadOffset = 128;
constexpr std::uint64_t kInvalidRecord = std::numeric_limits<std::uint64_t>::max();

struct alignas(16) DlbRankDispatchHeader {
    std::int64_t epoch;
    std::int64_t source_rank;
    std::int64_t source_token;
    std::int64_t destination_rank;
    std::uint32_t selection_count;
    std::uint32_t reserved;
    std::int32_t experts[kMaxTopK];
    std::uint8_t topk_slots[kMaxTopK];
    float weights[kMaxTopK];
    std::uint8_t padding[16];
};
static_assert(sizeof(DlbRankDispatchHeader) == kDispatchPayloadOffset,
              "DLB rank dispatch header layout changed");

__device__ __forceinline__ bool dispatch_header_is_valid(
    const DlbRankDispatchHeader* header,
    std::uint64_t epoch,
    std::uint32_t destination_rank);

__global__ void count_moe_routes_kernel(
    const std::int64_t* topk_idx,
    std::uint64_t route_count,
    std::uint32_t num_topk,
    std::uint32_t num_experts,
    std::uint32_t num_local_experts,
    std::uint32_t world_size,
    std::uint64_t* destination_counts,
    std::uint64_t* invalid_route_count) {
    const std::uint64_t route =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (route >= route_count) return;
    const std::int64_t expert = topk_idx[route];
    if (expert < 0 || expert >= static_cast<std::int64_t>(num_experts)) {
        atomicAdd(reinterpret_cast<unsigned long long*>(invalid_route_count), 1ULL);
        return;
    }
    const std::uint32_t destination =
        static_cast<std::uint32_t>(expert) / num_local_experts;
    if (destination < world_size) {
        const std::uint32_t topk_slot = route % num_topk;
        const std::uint64_t token_begin = route - topk_slot;
        for (std::uint32_t earlier = 0; earlier < topk_slot; ++earlier) {
            const std::int64_t earlier_expert = topk_idx[token_begin + earlier];
            if (earlier_expert >= 0 &&
                earlier_expert < static_cast<std::int64_t>(num_experts) &&
                static_cast<std::uint32_t>(earlier_expert) / num_local_experts ==
                    destination) {
                return;
            }
        }
        atomicAdd(reinterpret_cast<unsigned long long*>(destination_counts + destination),
                  1ULL);
    }
}

__device__ __forceinline__ std::uint32_t remote_server_id(
    std::uint32_t remote_index, std::uint32_t source_server) {
    return remote_index < source_server ? remote_index : remote_index + 1;
}

__device__ __forceinline__ std::uint64_t flow_rail_index(
    std::uint32_t remote_index,
    std::uint32_t source_local_rank,
    std::uint32_t rail,
    std::uint32_t destination_local_rank,
    std::uint32_t gpus_per_server) {
    return (((static_cast<std::uint64_t>(remote_index) * gpus_per_server +
              source_local_rank) * gpus_per_server + rail) * gpus_per_server +
            destination_local_rank);
}

// One block owns one remote-server tile.  The tile is deliberately tiny
// (normally 4x4 or 8x8), so one thread reproduces the reference scheduler's
// exact surplus/deficit decisions while different destination-server tiles
// run independently.  No host-visible plan is materialized.
__global__ void build_dynamic_rail_plan_kernel(
    const std::uint64_t* local_server_demand,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t local_rail,
    std::uint32_t round_id,
    std::uint32_t channel_count,
    std::uint64_t record_bytes,
    std::uint64_t receive_slot_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t* flow_rail_counts,
    DlbRailTransfer* transfers,
    std::uint64_t* rail_record_counts,
    std::uint64_t* channel_record_counts) {
    if (threadIdx.x != 0) return;
    const std::uint32_t remote_index = blockIdx.x;
    if (remote_index >= server_count - 1) return;
    const std::uint32_t destination_server =
        remote_server_id(remote_index, source_server);
    const std::uint32_t world_size = server_count * gpus_per_server;

    std::uint64_t source_loads[kMaxRailGpus] = {};
    std::uint64_t targets[kMaxRailGpus] = {};
    std::uint64_t surplus[kMaxRailGpus] = {};
    std::uint64_t deficit[kMaxRailGpus] = {};
    std::uint64_t tile_total = 0;

    // Start with every source GPU using its own Rail, exactly like
    // build_tile() in the C++ reference scheduler.
    for (std::uint32_t source = 0; source < gpus_per_server; ++source) {
        for (std::uint32_t rail = 0; rail < gpus_per_server; ++rail) {
            for (std::uint32_t destination = 0; destination < gpus_per_server;
                 ++destination) {
                const std::uint64_t records =
                    rail == source
                        ? local_server_demand[
                              static_cast<std::uint64_t>(source) * world_size +
                              destination_server * gpus_per_server + destination]
                        : 0;
                flow_rail_counts[flow_rail_index(
                    remote_index, source, rail, destination, gpus_per_server)] = records;
                if (rail == source) source_loads[source] += records;
            }
        }
        tile_total += source_loads[source];
    }

    const std::uint32_t first =
        (source_server * 17u + destination_server * 31u + round_id) %
        gpus_per_server;
    for (std::uint32_t rail = 0; rail < gpus_per_server; ++rail) {
        targets[rail] = tile_total / gpus_per_server;
    }
    for (std::uint32_t offset = 0; offset < tile_total % gpus_per_server; ++offset) {
        ++targets[(first + offset) % gpus_per_server];
    }
    for (std::uint32_t rail = 0; rail < gpus_per_server; ++rail) {
        if (source_loads[rail] > targets[rail]) {
            surplus[rail] = source_loads[rail] - targets[rail];
        } else {
            deficit[rail] = targets[rail] - source_loads[rail];
        }
    }

    std::uint32_t selected_rail = 0;
    for (std::uint32_t source = 0; source < gpus_per_server; ++source) {
        std::uint64_t remaining_surplus = surplus[source];
        while (remaining_surplus != 0) {
            while (selected_rail < gpus_per_server && deficit[selected_rail] == 0) {
                ++selected_rail;
            }
            if (selected_rail == gpus_per_server) break;
            std::uint64_t remaining_deficit = deficit[selected_rail];
            // Preserve direct-priority: first move records whose logical
            // destination matches the selected Rail, then visit the rest.
            for (std::uint32_t order = 0;
                 order < gpus_per_server && remaining_surplus != 0 &&
                 remaining_deficit != 0;
                 ++order) {
                const std::uint32_t destination =
                    order == 0 ? selected_rail
                               : (order <= selected_rail ? order - 1 : order);
                const std::uint64_t own_index = flow_rail_index(
                    remote_index, source, source, destination, gpus_per_server);
                const std::uint64_t staged_index = flow_rail_index(
                    remote_index, source, selected_rail, destination,
                    gpus_per_server);
                const std::uint64_t available = flow_rail_counts[own_index];
                const std::uint64_t moved = min(
                    remaining_surplus, min(remaining_deficit, available));
                flow_rail_counts[own_index] -= moved;
                flow_rail_counts[staged_index] += moved;
                remaining_surplus -= moved;
                remaining_deficit -= moved;
                deficit[selected_rail] -= moved;
            }
        }
    }

    std::uint64_t local_rail_prefix = 0;
    for (std::uint32_t destination = 0; destination < gpus_per_server;
         ++destination) {
        std::uint64_t group_records = 0;
        for (std::uint32_t source = 0; source < gpus_per_server; ++source) {
            group_records += flow_rail_counts[flow_rail_index(
                remote_index, source, local_rail, destination,
                gpus_per_server)];
        }
        std::uint64_t channel_prefix = 0;
        for (std::uint32_t channel = 0; channel < channel_count; ++channel) {
            const std::uint64_t channel_records =
                group_records / channel_count +
                (channel < group_records % channel_count ? 1 : 0);
            const std::uint32_t transfer_index =
                (remote_index * gpus_per_server + destination) * channel_count +
                channel;
            DlbRailTransfer transfer{};
            transfer.destination_rank =
                destination_server * gpus_per_server + local_rail;
            transfer.signal_index =
                (source_server * gpus_per_server + destination) * channel_count +
                channel;
            transfer.credit_index =
                (destination_server * gpus_per_server + destination) *
                    channel_count +
                channel;
            transfer.channel_index = channel;
            transfer.source_offset_bytes =
                static_cast<std::uint64_t>(remote_index) * rail_slot_bytes +
                (local_rail_prefix + channel_prefix) * record_bytes;
            transfer.destination_offset_bytes =
                static_cast<std::uint64_t>(source_server) * receive_slot_bytes +
                (local_rail_prefix + channel_prefix) * record_bytes;
            transfer.bytes = channel_records * record_bytes;
            transfers[transfer_index] = transfer;
            channel_prefix += channel_records;
            atomicAdd(reinterpret_cast<unsigned long long*>(
                          channel_record_counts + channel),
                      static_cast<unsigned long long>(channel_records));
        }
        local_rail_prefix += group_records;
    }

    for (std::uint32_t rail = 0; rail < gpus_per_server; ++rail) {
        std::uint64_t records = 0;
        for (std::uint32_t source = 0; source < gpus_per_server; ++source) {
            for (std::uint32_t destination = 0;
                 destination < gpus_per_server; ++destination) {
                records += flow_rail_counts[flow_rail_index(
                    remote_index, source, rail, destination,
                    gpus_per_server)];
            }
        }
        atomicAdd(reinterpret_cast<unsigned long long*>(rail_record_counts + rail),
                  static_cast<unsigned long long>(records));
    }
}

__global__ void pack_moe_direct_kernel(
    const std::uint8_t* x,
    std::uint64_t num_tokens,
    std::uint64_t hidden_bytes,
    const std::int64_t* topk_idx,
    const float* topk_weights,
    std::uint32_t num_topk,
    std::uint32_t num_experts,
    std::uint32_t world_size,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t source_local_rank,
    std::uint32_t source_rank,
    std::uint64_t epoch,
    std::uint64_t record_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t repair_slot_bytes,
    std::uint64_t repair_epoch_offset_bytes,
    const std::uint64_t* flow_rail_counts,
    std::uint64_t* destination_cursors,
    std::uint8_t* const* rail_send_buffers,
    std::uint8_t* const* repair_buffers) {
    const std::uint64_t token = blockIdx.x;
    if (token >= num_tokens) return;

    __shared__ std::uint8_t* record;
    __shared__ std::uint32_t destination_rank;
    const std::uint64_t token_begin = token * num_topk;
    const std::uint32_t num_local_experts = num_experts / world_size;

    // One block owns one token. It walks only the token's unique destination
    // ranks, avoiding the previous tokens*topk launch in which duplicate-rank
    // blocks did no useful payload work.
    for (std::uint32_t representative_slot = 0;
         representative_slot < num_topk; ++representative_slot) {
        if (threadIdx.x == 0) {
            record = nullptr;
            destination_rank = world_size;
            const std::int64_t expert =
                topk_idx[token_begin + representative_slot];
            if (expert >= 0 && expert < static_cast<std::int64_t>(num_experts)) {
                destination_rank =
                    static_cast<std::uint32_t>(expert) / num_local_experts;
                bool first_for_destination = true;
                for (std::uint32_t earlier = 0; earlier < representative_slot;
                     ++earlier) {
                    const std::int64_t earlier_expert =
                        topk_idx[token_begin + earlier];
                    if (earlier_expert >= 0 &&
                        earlier_expert < static_cast<std::int64_t>(num_experts) &&
                        static_cast<std::uint32_t>(earlier_expert) /
                                num_local_experts == destination_rank) {
                        first_for_destination = false;
                        break;
                    }
                }
                if (!first_for_destination) {
                    destination_rank = world_size;
                } else {
                    const std::uint64_t ordinal = atomicAdd(
                        reinterpret_cast<unsigned long long*>(
                            destination_cursors + destination_rank),
                        1ULL);
                    const std::uint32_t destination_server =
                        destination_rank / gpus_per_server;
                    const std::uint32_t destination_local_rank =
                        destination_rank % gpus_per_server;
                    if (destination_server == source_server) {
                        if (ordinal < repair_slot_bytes / record_bytes) {
                            record = repair_buffers[destination_local_rank] +
                                repair_epoch_offset_bytes +
                                static_cast<std::uint64_t>(source_rank) *
                                    repair_slot_bytes +
                                ordinal * record_bytes;
                        }
                    } else {
                        const std::uint32_t remote_index =
                            destination_server < source_server
                                ? destination_server
                                : destination_server - 1;
                        std::uint64_t flow_begin = 0;
                        std::uint32_t selected_rail = gpus_per_server;
                        std::uint64_t selected_ordinal = 0;
                        for (std::uint32_t rail = 0; rail < gpus_per_server;
                             ++rail) {
                            const std::uint64_t records =
                                flow_rail_counts[flow_rail_index(
                                    remote_index, source_local_rank, rail,
                                    destination_local_rank, gpus_per_server)];
                            if (ordinal < flow_begin + records) {
                                selected_rail = rail;
                                selected_ordinal = ordinal - flow_begin;
                                break;
                            }
                            flow_begin += records;
                        }
                        if (selected_rail < gpus_per_server) {
                            std::uint64_t group_prefix = 0;
                            for (std::uint32_t destination = 0;
                                 destination < destination_local_rank;
                                 ++destination) {
                                for (std::uint32_t source = 0;
                                     source < gpus_per_server; ++source) {
                                    group_prefix +=
                                        flow_rail_counts[flow_rail_index(
                                            remote_index, source, selected_rail,
                                            destination, gpus_per_server)];
                                }
                            }
                            for (std::uint32_t source = 0;
                                 source < source_local_rank; ++source) {
                                group_prefix += flow_rail_counts[flow_rail_index(
                                    remote_index, source, selected_rail,
                                    destination_local_rank, gpus_per_server)];
                            }
                            const std::uint64_t slot_record =
                                group_prefix + selected_ordinal;
                            if (slot_record < rail_slot_bytes / record_bytes) {
                                record = rail_send_buffers[selected_rail] +
                                    (static_cast<std::uint64_t>(remote_index) *
                                         rail_slot_bytes) +
                                    slot_record * record_bytes;
                            }
                        }
                    }
                }
            }

            if (record != nullptr) {
                auto* header = reinterpret_cast<DlbRankDispatchHeader*>(record);
                header->epoch = static_cast<std::int64_t>(epoch);
                header->source_rank = static_cast<std::int64_t>(source_rank);
                header->source_token = static_cast<std::int64_t>(token);
                header->destination_rank =
                    static_cast<std::int64_t>(destination_rank);
                header->selection_count = 0;
                header->reserved = 0;
#pragma unroll
                for (std::uint32_t index = 0; index < kMaxTopK; ++index) {
                    header->experts[index] = -1;
                    header->topk_slots[index] = 0xff;
                    header->weights[index] = 0.0f;
                }
                for (std::uint32_t slot = 0; slot < num_topk; ++slot) {
                    const std::int64_t selected_expert =
                        topk_idx[token_begin + slot];
                    if (selected_expert < 0 ||
                        selected_expert >=
                            static_cast<std::int64_t>(num_experts) ||
                        static_cast<std::uint32_t>(selected_expert) /
                                num_local_experts != destination_rank) {
                        continue;
                    }
                    const std::uint32_t selection =
                        header->selection_count++;
                    header->experts[selection] =
                        static_cast<std::int32_t>(selected_expert);
                    header->topk_slots[selection] =
                        static_cast<std::uint8_t>(slot);
                    header->weights[selection] = topk_weights[token_begin + slot];
                }
            }
        }
        __syncthreads();

        if (record != nullptr) {
            const std::uint8_t* source = x + token * hidden_bytes;
            std::uint8_t* payload = record + kDispatchPayloadOffset;
            const std::uintptr_t alignment =
                reinterpret_cast<std::uintptr_t>(source) |
                reinterpret_cast<std::uintptr_t>(payload) | hidden_bytes;
            if ((alignment & (alignof(uint4) - 1)) == 0) {
                const auto* input = reinterpret_cast<const uint4*>(source);
                auto* output = reinterpret_cast<uint4*>(payload);
                const std::uint64_t vectors = hidden_bytes / sizeof(uint4);
                for (std::uint64_t index = threadIdx.x; index < vectors;
                     index += blockDim.x) {
                    output[index] = input[index];
                }
            } else {
                for (std::uint64_t index = threadIdx.x; index < hidden_bytes;
                     index += blockDim.x) {
                    payload[index] = source[index];
                }
            }
            for (std::uint64_t index =
                     kDispatchPayloadOffset + hidden_bytes + threadIdx.x;
                 index < record_bytes; index += blockDim.x) {
                record[index] = 0;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && record != nullptr) {
            __threadfence_system();
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float load_activation(
    const std::uint8_t* payload, std::uint64_t index,
    std::uint32_t scalar_type) {
    if (scalar_type == 0) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(payload)[index]);
    }
    if (scalar_type == 1) {
        return __half2float(reinterpret_cast<const __half*>(payload)[index]);
    }
    return reinterpret_cast<const float*>(payload)[index];
}

__device__ __forceinline__ void store_activation(
    std::uint8_t* payload, std::uint64_t index, std::uint32_t scalar_type,
    float value) {
    if (scalar_type == 0) {
        reinterpret_cast<__nv_bfloat16*>(payload)[index] =
            __float2bfloat16_rn(value);
    } else if (scalar_type == 1) {
        reinterpret_cast<__half*>(payload)[index] = __float2half_rn(value);
    } else {
        reinterpret_cast<float*>(payload)[index] = value;
    }
}

__global__ void count_combine_routes_kernel(
    const std::uint64_t* group_output_indices,
    std::uint64_t group_count,
    const std::int64_t* headers,
    std::uint64_t output_record_count,
    std::uint32_t world_size,
    std::uint64_t* destination_counts) {
    const std::uint64_t group =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= group_count) return;
    const std::uint64_t first = group_output_indices[group * kMaxTopK];
    if (first >= output_record_count) return;
    const std::int64_t destination = headers[first * 6 + 1];
    if (destination >= 0 && destination < static_cast<std::int64_t>(world_size)) {
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      destination_counts + destination),
                  1ULL);
    }
}

__global__ void pack_combine_direct_kernel(
    const std::uint8_t* x,
    std::uint64_t hidden_elements,
    std::uint32_t scalar_type,
    const std::int64_t* headers,
    const float* weights,
    std::uint64_t output_record_count,
    const std::uint64_t* group_output_indices,
    std::uint64_t group_count,
    bool apply_weights,
    std::uint32_t world_size,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t source_local_rank,
    std::uint32_t source_rank,
    std::uint64_t epoch,
    std::uint64_t record_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t repair_slot_bytes,
    std::uint64_t repair_epoch_offset_bytes,
    const std::uint64_t* flow_rail_counts,
    std::uint64_t* destination_cursors,
    std::uint8_t* const* rail_send_buffers,
    std::uint8_t* const* repair_buffers) {
    const std::uint64_t group = blockIdx.x;
    if (group >= group_count) return;

    __shared__ std::uint8_t* record;
    __shared__ std::uint64_t input_indices[kMaxTopK];
    __shared__ std::uint32_t selection_count;
    __shared__ std::uint32_t destination_rank;
    if (threadIdx.x == 0) {
        record = nullptr;
        selection_count = 0;
        destination_rank = world_size;
#pragma unroll
        for (std::uint32_t selection = 0; selection < kMaxTopK; ++selection) {
            const std::uint64_t input =
                group_output_indices[group * kMaxTopK + selection];
            input_indices[selection] = input;
            if (input < output_record_count) ++selection_count;
        }
        if (selection_count != 0) {
            const std::int64_t destination = headers[input_indices[0] * 6 + 1];
            if (destination >= 0 &&
                destination < static_cast<std::int64_t>(world_size)) {
                destination_rank = static_cast<std::uint32_t>(destination);
                const std::uint64_t ordinal = atomicAdd(
                    reinterpret_cast<unsigned long long*>(
                        destination_cursors + destination_rank),
                    1ULL);
                const std::uint32_t destination_server =
                    destination_rank / gpus_per_server;
                const std::uint32_t destination_local_rank =
                    destination_rank % gpus_per_server;
                if (destination_server == source_server) {
                    if (ordinal < repair_slot_bytes / record_bytes) {
                        record = repair_buffers[destination_local_rank] +
                            repair_epoch_offset_bytes +
                            static_cast<std::uint64_t>(source_rank) *
                                repair_slot_bytes +
                            ordinal * record_bytes;
                    }
                } else {
                    const std::uint32_t remote_index =
                        destination_server < source_server
                            ? destination_server
                            : destination_server - 1;
                    std::uint64_t flow_begin = 0;
                    std::uint32_t selected_rail = gpus_per_server;
                    std::uint64_t selected_ordinal = 0;
                    for (std::uint32_t rail = 0; rail < gpus_per_server; ++rail) {
                        const std::uint64_t records =
                            flow_rail_counts[flow_rail_index(
                                remote_index, source_local_rank, rail,
                                destination_local_rank, gpus_per_server)];
                        if (ordinal < flow_begin + records) {
                            selected_rail = rail;
                            selected_ordinal = ordinal - flow_begin;
                            break;
                        }
                        flow_begin += records;
                    }
                    if (selected_rail < gpus_per_server) {
                        std::uint64_t group_prefix = 0;
                        for (std::uint32_t destination_local = 0;
                             destination_local < destination_local_rank;
                             ++destination_local) {
                            for (std::uint32_t source = 0;
                                 source < gpus_per_server; ++source) {
                                group_prefix += flow_rail_counts[flow_rail_index(
                                    remote_index, source, selected_rail,
                                    destination_local, gpus_per_server)];
                            }
                        }
                        for (std::uint32_t source = 0;
                             source < source_local_rank; ++source) {
                            group_prefix += flow_rail_counts[flow_rail_index(
                                remote_index, source, selected_rail,
                                destination_local_rank, gpus_per_server)];
                        }
                        const std::uint64_t slot_record =
                            group_prefix + selected_ordinal;
                        if (slot_record < rail_slot_bytes / record_bytes) {
                            record = rail_send_buffers[selected_rail] +
                                static_cast<std::uint64_t>(remote_index) *
                                    rail_slot_bytes +
                                slot_record * record_bytes;
                        }
                    }
                }
            }
        }

        if (record != nullptr) {
            const std::int64_t* first_header =
                headers + input_indices[0] * 6;
            auto* header = reinterpret_cast<DlbRankDispatchHeader*>(record);
            header->epoch = static_cast<std::int64_t>(epoch);
            header->source_rank = static_cast<std::int64_t>(source_rank);
            header->source_token = first_header[2];
            header->destination_rank =
                static_cast<std::int64_t>(destination_rank);
            header->selection_count = selection_count;
            header->reserved = 0;
#pragma unroll
            for (std::uint32_t selection = 0; selection < kMaxTopK; ++selection) {
                if (selection < selection_count) {
                    const std::uint64_t input = input_indices[selection];
                    header->experts[selection] =
                        static_cast<std::int32_t>(headers[input * 6 + 3]);
                    header->topk_slots[selection] =
                        static_cast<std::uint8_t>(headers[input * 6 + 4]);
                    header->weights[selection] = weights[input];
                } else {
                    header->experts[selection] = -1;
                    header->topk_slots[selection] = 0xff;
                    header->weights[selection] = 0.0f;
                }
            }
        }
    }
    __syncthreads();
    if (record == nullptr) return;

    const std::uint32_t element_bytes = scalar_type == 2 ? 4 : 2;
    std::uint8_t* payload = record + kDispatchPayloadOffset;
    for (std::uint64_t hidden = threadIdx.x; hidden < hidden_elements;
         hidden += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (std::uint32_t selection = 0; selection < kMaxTopK; ++selection) {
            if (selection >= selection_count) break;
            const std::uint64_t input = input_indices[selection];
            const std::uint8_t* source =
                x + input * hidden_elements * element_bytes;
            value += load_activation(source, hidden, scalar_type) *
                     (apply_weights ? weights[input] : 1.0f);
        }
        store_activation(payload, hidden, scalar_type, value);
    }
    const std::uint64_t hidden_bytes = hidden_elements * element_bytes;
    for (std::uint64_t index =
             kDispatchPayloadOffset + hidden_bytes + threadIdx.x;
         index < record_bytes; index += blockDim.x) {
        record[index] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) __threadfence_system();
}

__global__ void accumulate_combined_records_kernel(
    const std::uint8_t* repair_records,
    std::uint64_t repair_record_count,
    std::uint64_t record_bytes,
    std::uint64_t hidden_elements,
    std::uint32_t scalar_type,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint64_t num_tokens,
    std::uint32_t num_topk,
    float* combined,
    float* combined_weights) {
    const std::uint64_t record_index = blockIdx.x;
    if (record_index >= repair_record_count) return;
    const std::uint8_t* record = repair_records + record_index * record_bytes;

    __shared__ std::uint64_t token;
    __shared__ std::uint32_t selection_count;
    __shared__ bool valid;
    if (threadIdx.x == 0) {
        const auto* header =
            reinterpret_cast<const DlbRankDispatchHeader*>(record);
        valid = dispatch_header_is_valid(header, epoch, destination_rank) &&
                header->source_token >= 0 &&
                header->source_token < static_cast<std::int64_t>(num_tokens);
        if (valid) {
            token = static_cast<std::uint64_t>(header->source_token);
            selection_count = header->selection_count;
            for (std::uint32_t selection = 0; selection < selection_count;
                 ++selection) {
                const std::uint8_t slot = header->topk_slots[selection];
                if (slot < num_topk) {
                    combined_weights[token * num_topk + slot] =
                        header->weights[selection];
                }
            }
        }
    }
    __syncthreads();
    if (!valid) return;
    const std::uint8_t* payload = record + kDispatchPayloadOffset;
    float* output = combined + token * hidden_elements;
    for (std::uint64_t index = threadIdx.x; index < hidden_elements;
         index += blockDim.x) {
        atomicAdd(output + index, load_activation(payload, index, scalar_type));
    }
}

__device__ __forceinline__ bool dispatch_header_is_valid(
    const DlbRankDispatchHeader* header,
    std::uint64_t epoch,
    std::uint32_t destination_rank) {
    return header->epoch == static_cast<std::int64_t>(epoch) &&
           header->destination_rank == static_cast<std::int64_t>(destination_rank) &&
           header->selection_count > 0 &&
           header->selection_count <= kMaxTopK;
}

__global__ void count_received_experts_kernel(
    const std::uint8_t* repair_records,
    std::uint64_t record_count,
    std::uint64_t record_bytes,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint32_t first_local_expert,
    std::uint32_t num_local_experts,
    std::uint64_t* expert_counts,
    std::uint64_t* group_count) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= record_count) return;
    const std::uint8_t* record = repair_records + index * record_bytes;
    const auto* header = reinterpret_cast<const DlbRankDispatchHeader*>(record);
    if (!dispatch_header_is_valid(header, epoch, destination_rank)) return;
    atomicAdd(reinterpret_cast<unsigned long long*>(group_count), 1ULL);
    for (std::uint32_t selection = 0; selection < header->selection_count;
         ++selection) {
        const std::int32_t expert = header->experts[selection];
        if (expert < static_cast<std::int32_t>(first_local_expert) ||
            expert >= static_cast<std::int32_t>(first_local_expert +
                                                num_local_experts)) {
            continue;
        }
        const std::uint32_t local_expert =
            static_cast<std::uint32_t>(expert) - first_local_expert;
        atomicAdd(reinterpret_cast<unsigned long long*>(expert_counts + local_expert),
                  1ULL);
    }
}

__global__ void scatter_received_experts_kernel(
    const std::uint8_t* repair_records,
    std::uint64_t record_count,
    std::uint64_t record_bytes,
    std::uint64_t hidden_bytes,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint32_t first_local_expert,
    std::uint32_t num_local_experts,
    const std::uint64_t* expert_offsets,
    std::uint64_t* expert_cursors,
    std::uint8_t* recv_x,
    std::int64_t* recv_headers,
    float* recv_weights,
    bool* valid_mask,
    std::uint64_t output_records,
    std::uint64_t* group_cursor,
    std::uint64_t* group_output_indices,
    std::uint64_t group_capacity) {
    const std::uint64_t record_index = blockIdx.x;
    if (record_index >= record_count) return;
    const std::uint8_t* record = repair_records + record_index * record_bytes;

    __shared__ std::uint64_t output_indices[kMaxTopK];
    __shared__ std::uint64_t group_index;
    __shared__ std::uint32_t selection_count;
    if (threadIdx.x == 0) {
        const auto* header = reinterpret_cast<const DlbRankDispatchHeader*>(record);
        selection_count = dispatch_header_is_valid(header, epoch, destination_rank)
                              ? header->selection_count
                              : 0;
        group_index = selection_count != 0
                          ? atomicAdd(reinterpret_cast<unsigned long long*>(
                                          group_cursor),
                                      1ULL)
                          : kInvalidRecord;
#pragma unroll
        for (std::uint32_t selection = 0; selection < kMaxTopK; ++selection) {
            output_indices[selection] = kInvalidRecord;
        }
        for (std::uint32_t selection = 0; selection < selection_count; ++selection) {
            const std::int32_t expert = header->experts[selection];
            if (expert < static_cast<std::int32_t>(first_local_expert) ||
                expert >= static_cast<std::int32_t>(first_local_expert +
                                                    num_local_experts)) {
                continue;
            }
            const std::uint32_t local_expert =
                static_cast<std::uint32_t>(expert) - first_local_expert;
            const std::uint64_t output_index = expert_offsets[local_expert] + atomicAdd(
                reinterpret_cast<unsigned long long*>(expert_cursors + local_expert), 1ULL);
            output_indices[selection] = output_index;
            if (group_index < group_capacity &&
                output_index < output_records) {
                group_output_indices[group_index * kMaxTopK + selection] =
                    output_index;
            }
            if (output_index < output_records) {
                std::int64_t* destination_header = recv_headers + output_index * 6;
                destination_header[0] = header->epoch;
                destination_header[1] = header->source_rank;
                destination_header[2] = header->source_token;
                destination_header[3] = expert;
                destination_header[4] = header->topk_slots[selection];
                destination_header[5] = header->destination_rank;
                recv_weights[output_index] = header->weights[selection];
                valid_mask[output_index] = true;
            }
        }
    }
    __syncthreads();
    const std::uint8_t* source = record + kDispatchPayloadOffset;
    for (std::uint32_t selection = 0; selection < selection_count; ++selection) {
        const std::uint64_t output_index = output_indices[selection];
        if (output_index < output_records) {
            std::uint8_t* destination = recv_x + output_index * hidden_bytes;
            const std::uintptr_t alignment = reinterpret_cast<std::uintptr_t>(source) |
                                             reinterpret_cast<std::uintptr_t>(destination) |
                                             hidden_bytes;
            if ((alignment & (alignof(uint4) - 1)) == 0) {
                const auto* input = reinterpret_cast<const uint4*>(source);
                auto* output = reinterpret_cast<uint4*>(destination);
                const std::uint64_t vectors = hidden_bytes / sizeof(uint4);
                for (std::uint64_t index = threadIdx.x; index < vectors;
                     index += blockDim.x) {
                    output[index] = input[index];
                }
            } else {
                for (std::uint64_t index = threadIdx.x; index < hidden_bytes;
                     index += blockDim.x) {
                    destination[index] = source[index];
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace

cudaError_t launch_dlb_count_moe_routes(
    const std::int64_t* topk_idx,
    std::uint64_t route_count,
    std::uint32_t num_topk,
    std::uint32_t num_experts,
    std::uint32_t num_local_experts,
    std::uint32_t world_size,
    std::uint64_t* destination_counts,
    std::uint64_t* invalid_route_count,
    cudaStream_t stream) {
    if (topk_idx == nullptr || destination_counts == nullptr ||
        invalid_route_count == nullptr || num_experts == 0 ||
        num_local_experts == 0 || world_size == 0 || num_topk == 0 ||
        num_topk > kMaxTopK || route_count % num_topk != 0) {
        return cudaErrorInvalidValue;
    }
    if (route_count == 0) return cudaSuccess;
    constexpr unsigned threads = 256;
    const std::uint64_t blocks = (route_count + threads - 1) / threads;
    if (blocks > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidConfiguration;
    count_moe_routes_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        topk_idx, route_count, num_topk, num_experts, num_local_experts, world_size,
        destination_counts, invalid_route_count);
    return cudaGetLastError();
}

cudaError_t launch_dlb_build_dynamic_rail_plan(
    const std::uint64_t* local_server_demand,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t local_rail,
    std::uint32_t round_id,
    std::uint32_t channel_count,
    std::uint64_t record_bytes,
    std::uint64_t receive_slot_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t* flow_rail_counts,
    DlbRailTransfer* transfers,
    std::uint32_t transfer_count,
    std::uint64_t* rail_record_counts,
    std::uint64_t* channel_record_counts,
    cudaStream_t stream) {
    if (local_server_demand == nullptr || server_count < 2 ||
        gpus_per_server == 0 || gpus_per_server > kMaxRailGpus ||
        channel_count == 0 ||
        source_server >= server_count || local_rail >= gpus_per_server ||
        record_bytes == 0 || receive_slot_bytes < record_bytes ||
        rail_slot_bytes < receive_slot_bytes ||
        receive_slot_bytes % record_bytes != 0 ||
        rail_slot_bytes % record_bytes != 0 || flow_rail_counts == nullptr ||
        transfers == nullptr ||
        transfer_count != (server_count - 1) * gpus_per_server * channel_count ||
        rail_record_counts == nullptr || channel_record_counts == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMemsetAsync(
        rail_record_counts, 0,
        static_cast<std::size_t>(gpus_per_server) * sizeof(std::uint64_t), stream);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(
        channel_record_counts, 0,
        static_cast<std::size_t>(channel_count) * sizeof(std::uint64_t), stream);
    if (status != cudaSuccess) return status;
    build_dynamic_rail_plan_kernel<<<server_count - 1, 1, 0, stream>>>(
        local_server_demand, server_count, gpus_per_server, source_server,
        local_rail, round_id, channel_count, record_bytes, receive_slot_bytes,
        rail_slot_bytes, flow_rail_counts, transfers, rail_record_counts,
        channel_record_counts);
    return cudaGetLastError();
}

cudaError_t launch_dlb_pack_moe_direct(
    const void* x,
    std::uint64_t num_tokens,
    std::uint64_t hidden_bytes,
    const std::int64_t* topk_idx,
    const float* topk_weights,
    std::uint32_t num_topk,
    std::uint32_t num_experts,
    std::uint32_t world_size,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t source_local_rank,
    std::uint32_t source_rank,
    std::uint32_t round_id,
    std::uint64_t epoch,
    std::uint64_t record_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t repair_slot_bytes,
    std::uint64_t repair_epoch_offset_bytes,
    const std::uint64_t* local_server_demand,
    const std::uint64_t* flow_rail_counts,
    std::uint64_t* destination_cursors,
    std::uint8_t* const* rail_send_buffers,
    std::uint8_t* const* repair_buffers,
    cudaStream_t stream) {
    (void)round_id;
    (void)local_server_demand;
    if (x == nullptr || hidden_bytes == 0 || topk_idx == nullptr ||
        topk_weights == nullptr || num_topk == 0 || num_experts == 0 ||
        world_size == 0 || server_count < 2 || gpus_per_server == 0 ||
        gpus_per_server > kMaxRailGpus ||
        world_size != server_count * gpus_per_server ||
        source_server >= server_count || source_local_rank >= gpus_per_server ||
        source_rank >= world_size || epoch == 0 || num_topk > kMaxTopK ||
        record_bytes < kDispatchPayloadOffset + hidden_bytes ||
        rail_slot_bytes < record_bytes || repair_slot_bytes < record_bytes ||
        rail_slot_bytes % record_bytes != 0 ||
        repair_slot_bytes % record_bytes != 0 || flow_rail_counts == nullptr ||
        destination_cursors == nullptr || rail_send_buffers == nullptr ||
        repair_buffers == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (num_experts % world_size != 0 ||
        num_tokens > std::numeric_limits<std::uint64_t>::max() / num_topk) {
        return cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return cudaSuccess;
    if (num_tokens > std::numeric_limits<unsigned>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    pack_moe_direct_kernel<<<static_cast<unsigned>(num_tokens), kPackThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(x), num_tokens, hidden_bytes, topk_idx,
        topk_weights, num_topk, num_experts, world_size, server_count,
        gpus_per_server, source_server, source_local_rank, source_rank, epoch,
        record_bytes, rail_slot_bytes, repair_slot_bytes,
        repair_epoch_offset_bytes, flow_rail_counts, destination_cursors,
        rail_send_buffers, repair_buffers);
    return cudaGetLastError();
}

cudaError_t launch_dlb_count_combine_routes(
    const std::uint64_t* group_output_indices,
    std::uint64_t group_count,
    const std::int64_t* headers,
    std::uint64_t output_record_count,
    std::uint32_t world_size,
    std::uint64_t* destination_counts,
    cudaStream_t stream) {
    if (group_output_indices == nullptr || headers == nullptr || world_size == 0 ||
        destination_counts == nullptr || output_record_count == 0) {
        return cudaErrorInvalidValue;
    }
    if (group_count == 0) return cudaSuccess;
    constexpr unsigned threads = 256;
    const std::uint64_t blocks = (group_count + threads - 1) / threads;
    if (blocks > std::numeric_limits<unsigned>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    count_combine_routes_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        group_output_indices, group_count, headers, output_record_count,
        world_size, destination_counts);
    return cudaGetLastError();
}

cudaError_t launch_dlb_pack_combine_direct(
    const void* x,
    std::uint64_t hidden_elements,
    std::uint32_t scalar_type,
    const std::int64_t* headers,
    const float* weights,
    std::uint64_t output_record_count,
    const std::uint64_t* group_output_indices,
    std::uint64_t group_count,
    bool apply_weights,
    std::uint32_t world_size,
    std::uint32_t server_count,
    std::uint32_t gpus_per_server,
    std::uint32_t source_server,
    std::uint32_t source_local_rank,
    std::uint32_t source_rank,
    std::uint64_t epoch,
    std::uint64_t record_bytes,
    std::uint64_t rail_slot_bytes,
    std::uint64_t repair_slot_bytes,
    std::uint64_t repair_epoch_offset_bytes,
    const std::uint64_t* flow_rail_counts,
    std::uint64_t* destination_cursors,
    std::uint8_t* const* rail_send_buffers,
    std::uint8_t* const* repair_buffers,
    cudaStream_t stream) {
    const std::uint32_t element_bytes = scalar_type == 2 ? 4 : 2;
    if (x == nullptr || hidden_elements == 0 || scalar_type > 2 ||
        headers == nullptr || weights == nullptr || output_record_count == 0 ||
        group_output_indices == nullptr || world_size == 0 || server_count < 2 ||
        gpus_per_server == 0 || gpus_per_server > kMaxRailGpus ||
        world_size != server_count * gpus_per_server ||
        source_server >= server_count || source_local_rank >= gpus_per_server ||
        source_rank >= world_size || epoch == 0 ||
        record_bytes < kDispatchPayloadOffset + hidden_elements * element_bytes ||
        rail_slot_bytes < record_bytes || repair_slot_bytes < record_bytes ||
        rail_slot_bytes % record_bytes != 0 ||
        repair_slot_bytes % record_bytes != 0 || flow_rail_counts == nullptr ||
        destination_cursors == nullptr || rail_send_buffers == nullptr ||
        repair_buffers == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (group_count == 0) return cudaSuccess;
    if (group_count > std::numeric_limits<unsigned>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    pack_combine_direct_kernel<<<static_cast<unsigned>(group_count),
                                 kPackThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(x), hidden_elements, scalar_type,
        headers, weights, output_record_count, group_output_indices, group_count,
        apply_weights, world_size, server_count, gpus_per_server,
        source_server, source_local_rank, source_rank, epoch, record_bytes,
        rail_slot_bytes, repair_slot_bytes, repair_epoch_offset_bytes,
        flow_rail_counts, destination_cursors, rail_send_buffers, repair_buffers);
    return cudaGetLastError();
}

cudaError_t launch_dlb_accumulate_combined_records(
    const std::uint8_t* repair_records,
    std::uint64_t repair_record_count,
    std::uint64_t record_bytes,
    std::uint64_t hidden_elements,
    std::uint32_t scalar_type,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint64_t num_tokens,
    std::uint32_t num_topk,
    float* combined,
    float* combined_weights,
    cudaStream_t stream) {
    const std::uint32_t element_bytes = scalar_type == 2 ? 4 : 2;
    if (repair_records == nullptr || scalar_type > 2 || hidden_elements == 0 ||
        record_bytes < kDispatchPayloadOffset + hidden_elements * element_bytes ||
        epoch == 0 ||
        num_tokens == 0 || num_topk == 0 || combined == nullptr ||
        combined_weights == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (repair_record_count == 0) return cudaSuccess;
    if (repair_record_count > std::numeric_limits<unsigned>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    accumulate_combined_records_kernel<<<static_cast<unsigned>(repair_record_count),
                                         kPackThreads, 0, stream>>>(
        repair_records, repair_record_count, record_bytes, hidden_elements,
        scalar_type, epoch, destination_rank, num_tokens, num_topk,
        combined, combined_weights);
    return cudaGetLastError();
}

cudaError_t launch_dlb_count_received_experts(
    const std::uint8_t* repair_records,
    std::uint64_t record_count,
    std::uint64_t record_bytes,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint32_t first_local_expert,
    std::uint32_t num_local_experts,
    std::uint64_t* expert_counts,
    std::uint64_t* group_count,
    cudaStream_t stream) {
    if (repair_records == nullptr || record_bytes < kDispatchPayloadOffset || epoch == 0 ||
        num_local_experts == 0 || expert_counts == nullptr ||
        group_count == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (record_count == 0) return cudaSuccess;
    constexpr unsigned threads = 256;
    const std::uint64_t blocks = (record_count + threads - 1) / threads;
    if (blocks > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidConfiguration;
    count_received_experts_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        repair_records, record_count, record_bytes, epoch, destination_rank,
        first_local_expert, num_local_experts, expert_counts, group_count);
    return cudaGetLastError();
}

cudaError_t launch_dlb_scatter_received_experts(
    const std::uint8_t* repair_records,
    std::uint64_t record_count,
    std::uint64_t record_bytes,
    std::uint64_t hidden_bytes,
    std::uint64_t epoch,
    std::uint32_t destination_rank,
    std::uint32_t first_local_expert,
    std::uint32_t num_local_experts,
    const std::uint64_t* expert_offsets,
    std::uint64_t* expert_cursors,
    std::uint8_t* recv_x,
    std::int64_t* recv_headers,
    float* recv_weights,
    bool* valid_mask,
    std::uint64_t output_records,
    std::uint64_t* group_cursor,
    std::uint64_t* group_output_indices,
    std::uint64_t group_capacity,
    cudaStream_t stream) {
    if (repair_records == nullptr ||
        record_bytes < kDispatchPayloadOffset + hidden_bytes ||
        hidden_bytes == 0 || epoch == 0 || num_local_experts == 0 ||
        expert_offsets == nullptr || expert_cursors == nullptr || recv_x == nullptr ||
        recv_headers == nullptr || recv_weights == nullptr || valid_mask == nullptr ||
        output_records == 0 || group_cursor == nullptr ||
        group_output_indices == nullptr || group_capacity == 0) {
        return cudaErrorInvalidValue;
    }
    if (record_count == 0) return cudaSuccess;
    if (record_count > std::numeric_limits<unsigned>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    scatter_received_experts_kernel<<<static_cast<unsigned>(record_count),
                                      kPackThreads, 0, stream>>>(
        repair_records, record_count, record_bytes, hidden_bytes, epoch,
        destination_rank, first_local_expert, num_local_experts,
        expert_offsets, expert_cursors, recv_x, recv_headers, recv_weights,
        valid_mask, output_records, group_cursor, group_output_indices,
        group_capacity);
    return cudaGetLastError();
}

}  // namespace dlb_alltoall
