#pragma once

#include "configs.cuh"

#include <cstddef>
#include <cstdint>

namespace deep_ep {
namespace rail_lb {

constexpr int kMaxChannels = 32;
constexpr uint8_t kInvalidRail = 0xff;

#ifdef __CUDACC__
#define DEEP_EP_LB_UNROLL _Pragma("unroll")
#else
#define DEEP_EP_LB_UNROLL
#endif

// Computes the compact source-to-Rail quota row for one destination server.
// The input contains one token count per source Rail. The output contains the
// number of this source Rail's tokens that must use each selected Rail.
__host__ __device__ __forceinline__
void compute_quota_row(const int* source_loads,
                       int source_server,
                       int destination_server,
                       int source_rail,
                       int round_id,
                       int* quotas) {
    int targets[NUM_MAX_NVL_PEERS] = {};
    int deficits[NUM_MAX_NVL_PEERS] = {};
    int total = 0;

    DEEP_EP_LB_UNROLL
    for (int rail = 0; rail < NUM_MAX_NVL_PEERS; ++rail) {
        quotas[rail] = 0;
        total += source_loads[rail];
    }
    if (destination_server == source_server) {
        quotas[source_rail] = source_loads[source_rail];
        return;
    }

    const int first =
        (source_server * 17 + destination_server * 31 + round_id) %
        NUM_MAX_NVL_PEERS;
    DEEP_EP_LB_UNROLL
    for (int rail = 0; rail < NUM_MAX_NVL_PEERS; ++rail)
        targets[rail] = total / NUM_MAX_NVL_PEERS;
    DEEP_EP_LB_UNROLL
    for (int offset = 0; offset < NUM_MAX_NVL_PEERS; ++offset) {
        if (offset < total % NUM_MAX_NVL_PEERS)
            ++targets[(first + offset) % NUM_MAX_NVL_PEERS];
    }
    DEEP_EP_LB_UNROLL
    for (int rail = 0; rail < NUM_MAX_NVL_PEERS; ++rail) {
        const int deficit = targets[rail] - source_loads[rail];
        deficits[rail] = deficit > 0 ? deficit : 0;
    }

    // Reproduce source-order surplus/deficit matching. The direct share is
    // first, so the hot dispatch lookup favors the original zero-copy path.
    DEEP_EP_LB_UNROLL
    for (int producer = 0; producer < NUM_MAX_NVL_PEERS; ++producer) {
        const int direct =
            source_loads[producer] < targets[producer]
                ? source_loads[producer]
                : targets[producer];
        if (producer == source_rail)
            quotas[producer] = direct;

        int surplus = source_loads[producer] - direct;
        DEEP_EP_LB_UNROLL
        for (int selected = 0;
             selected < NUM_MAX_NVL_PEERS;
             ++selected) {
            const int moved =
                surplus < deficits[selected]
                    ? surplus
                    : deficits[selected];
            if (producer == source_rail)
                quotas[selected] += moved;
            surplus -= moved;
            deficits[selected] -= moved;
        }
    }
}

// Maps an ordinal in one producer-to-server token flow to its physical Rail.
__host__ __device__ __forceinline__
int select_rail(const int* quotas, int source_rail, int ordinal) {
    if (ordinal < quotas[source_rail])
        return source_rail;
    ordinal -= quotas[source_rail];

    DEEP_EP_LB_UNROLL
    for (int rail = 0; rail < NUM_MAX_NVL_PEERS; ++rail) {
        if (rail == source_rail)
            continue;
        if (ordinal < quotas[rail])
            return rail;
        ordinal -= quotas[rail];
    }
    return kInvalidRail;
}

#undef DEEP_EP_LB_UNROLL

size_t get_record_stride(int hidden, int num_topk, int source_meta_bytes);

// Per-GPU CUDA-IPC workspace for the fused DLB data plane. Fixed-size fields
// describe only server/Rail/channel quotas. Variable-size fields contain only
// messages that actually moved away from their source Rail; there is no
// token-by-server selected-Rail plan.
struct Workspace {
    int* epoch;
    int* demand_ready;
    int* quota_ready;
    int* schedule_ready;
    int* num_tokens;
    int* num_outgoing;

    int* source_loads;
    int* quotas;
    int* source_channel_prefix;
    int* route_ready;
    int* route_counts;
    int* mask_counts;
    int* stage_offsets;
    int* repair_offsets;

    int* repair_heads;
    int* direct_ready;

    int* stage_ready;
    int* repair_ready;
    int* virtual_token_indices;
    int* virtual_repair_slots;
    int* repair_next;
    uint8_t* virtual_source_rails;
    uint8_t* virtual_channels;
    uint8_t* payload;

    size_t metadata_bytes;
    size_t payload_capacity;
    size_t total_bytes;

    __host__ __device__ Workspace(void* base,
                                  int max_tokens,
                                  int num_servers,
                                  int num_channels,
                                  size_t record_stride) {
        auto cursor = reinterpret_cast<uint8_t*>(base);
        auto take_ints = [&](size_t count) {
            auto result = reinterpret_cast<int*>(cursor);
            cursor += sizeof(int) * count;
            return result;
        };

        epoch = take_ints(1);
        demand_ready = take_ints(1);
        quota_ready = take_ints(1);
        schedule_ready = take_ints(1);
        num_tokens = take_ints(1);
        num_outgoing = take_ints(1);

        source_loads = take_ints(num_servers);
        quotas = take_ints(
            static_cast<size_t>(num_servers) * NUM_MAX_NVL_PEERS
        );
        source_channel_prefix = take_ints(
            static_cast<size_t>(num_servers) * num_channels
        );
        route_ready = take_ints(num_servers);

        const auto route_entries =
            static_cast<size_t>(NUM_MAX_NVL_PEERS) *
            num_servers * num_channels;
        route_counts = take_ints(route_entries);
        mask_counts = take_ints(route_entries * NUM_MAX_NVL_PEERS);
        stage_offsets = take_ints(route_entries);
        repair_offsets = take_ints(route_entries);

        repair_heads = take_ints(max_tokens);
        direct_ready = take_ints(max_tokens);

        // One source token has at most one deduplicated message for each
        // remote server. Rail selection chooses exactly one physical Rail for
        // that message, so capacity does not multiply by the number of
        // non-origin Rails. Dispatch and combine reuse the same payload.
        payload_capacity =
            static_cast<size_t>(max_tokens) *
            (num_servers > 1 ? num_servers - 1 : 1);

        stage_ready = take_ints(payload_capacity);
        repair_ready = take_ints(payload_capacity);
        virtual_token_indices = take_ints(payload_capacity);
        virtual_repair_slots = take_ints(payload_capacity);
        repair_next = take_ints(payload_capacity);

        virtual_source_rails = cursor;
        cursor += payload_capacity;
        virtual_channels = cursor;
        cursor += payload_capacity;

        auto cursor_value = reinterpret_cast<uintptr_t>(cursor);
        cursor_value =
            (cursor_value + NUM_BUFFER_ALIGNMENT_BYTES - 1) /
            NUM_BUFFER_ALIGNMENT_BYTES * NUM_BUFFER_ALIGNMENT_BYTES;
        cursor = reinterpret_cast<uint8_t*>(cursor_value);
        metadata_bytes = static_cast<size_t>(
            cursor - reinterpret_cast<uint8_t*>(base)
        );
        payload = cursor;
        total_bytes = metadata_bytes + payload_capacity * record_stride;
    }

    __host__ __device__ __forceinline__
    int quota_index(int server, int selected_rail) const {
        return server * NUM_MAX_NVL_PEERS + selected_rail;
    }

    __host__ __device__ __forceinline__
    int* quota_row(int server) const {
        return quotas + server * NUM_MAX_NVL_PEERS;
    }

    __host__ __device__ __forceinline__
    int source_channel_index(int server, int channel,
                             int num_channels) const {
        return server * num_channels + channel;
    }

    __host__ __device__ __forceinline__
    int route_index(int rail, int server, int channel,
                    int num_servers, int num_channels) const {
        return (rail * num_servers + server) * num_channels + channel;
    }

    __host__ __device__ __forceinline__
    int mask_index(int rail, int server, int channel, int destination_rail,
                   int num_servers, int num_channels) const {
        return route_index(
                   rail, server, channel, num_servers, num_channels
               ) * NUM_MAX_NVL_PEERS + destination_rail;
    }
};

// Compact dispatch handle. Fixed-size entries cache the Rail schedule, while
// variable-size entries contain only the moved-token reverse mapping.
struct HandleLayout {
    static constexpr int kVersion = 1;
    static constexpr int kHeaderInts = 4;

    int* header;
    int* quotas;
    int* source_channel_prefix;
    int* route_counts;
    int* mask_counts;
    int* stage_offsets;
    int* repair_offsets;
    int* repair_heads;
    int* virtual_token_indices;
    int* virtual_repair_slots;
    int* repair_next;
    uint8_t* virtual_source_rails;
    uint8_t* virtual_channels;
    size_t total_bytes;

    __host__ __device__ HandleLayout(void* base,
                                     int num_tokens,
                                     int num_incoming,
                                     int num_outgoing,
                                     int num_servers,
                                     int num_channels) {
        auto cursor = reinterpret_cast<uint8_t*>(base);
        auto take_ints = [&](size_t count) {
            auto result = reinterpret_cast<int*>(cursor);
            cursor += sizeof(int) * count;
            return result;
        };
        header = take_ints(kHeaderInts);
        quotas = take_ints(
            static_cast<size_t>(num_servers) * NUM_MAX_NVL_PEERS
        );
        source_channel_prefix = take_ints(
            static_cast<size_t>(num_servers) * num_channels
        );
        const auto route_entries =
            static_cast<size_t>(NUM_MAX_NVL_PEERS) *
            num_servers * num_channels;
        route_counts = take_ints(route_entries);
        mask_counts = take_ints(route_entries * NUM_MAX_NVL_PEERS);
        stage_offsets = take_ints(route_entries);
        repair_offsets = take_ints(route_entries);
        repair_heads = take_ints(num_tokens);
        virtual_token_indices = take_ints(num_incoming);
        virtual_repair_slots = take_ints(num_incoming);
        repair_next = take_ints(num_outgoing);
        virtual_source_rails = cursor;
        cursor += num_incoming;
        virtual_channels = cursor;
        cursor += num_incoming;
        total_bytes = static_cast<size_t>(
            cursor - reinterpret_cast<uint8_t*>(base)
        );
    }
};

size_t get_workspace_size(int max_tokens,
                          int num_servers,
                          int num_channels,
                          size_t record_stride);

size_t get_handle_size(int num_tokens,
                       int num_incoming,
                       int num_outgoing,
                       int num_servers,
                       int num_channels);

int get_handle_outgoing_count(size_t handle_bytes,
                              int num_tokens,
                              int num_incoming,
                              int num_servers,
                              int num_channels);

// Saves and restores the fixed schedule plus exact moved-only reverse map.
// Readiness flags, payload storage and unused capacity are never copied.
void save_handle(void* destination,
                 const void* local_workspace,
                 int num_tokens,
                 int num_incoming,
                 int num_outgoing,
                 int max_tokens,
                 int num_servers,
                 int num_channels,
                 size_t record_stride,
                 cudaStream_t stream);

void load_handle(void* local_workspace,
                 const void* source,
                 int num_tokens,
                 int num_incoming,
                 int num_outgoing,
                 int max_tokens,
                 int num_servers,
                 int num_channels,
                 size_t record_stride,
                 cudaStream_t stream);

// Starts a cached dispatch with a fresh epoch while reusing quotas, prefixes
// and moved-message mappings loaded from the compact handle.
void activate_cached(void* local_workspace,
                     int max_tokens,
                     int num_servers,
                     int num_channels,
                     int round_id,
                     size_t record_stride,
                     cudaStream_t stream);

void finalize_combined_heads(int* combined_rdma_head,
                             void* local_workspace,
                             int num_virtual_tokens,
                             int max_tokens,
                             int num_servers,
                             int num_channels,
                             size_t record_stride,
                             cudaStream_t stream);

}  // namespace rail_lb
}  // namespace deep_ep
