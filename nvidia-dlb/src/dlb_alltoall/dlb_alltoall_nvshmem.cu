#include "dlb_alltoall/dlb_alltoall_nvshmem.h"

#include <nvshmemx.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <new>
#include <vector>

namespace dlb_alltoall {
namespace {

template <typename T>
void release_device(T** pointer) {
    if (*pointer != nullptr) {
        cudaFree(*pointer);
        *pointer = nullptr;
    }
}

template <typename T>
cudaError_t upload_vector(T** device, std::size_t* capacity,
                          const std::vector<T>& host) {
    if (host.empty()) {
        return cudaSuccess;
    }
    if (host.size() > *capacity) {
        std::size_t next_capacity = std::max<std::size_t>(host.size(), std::max<std::size_t>(4, *capacity * 2));
        T* replacement = nullptr;
        cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&replacement),
                                        next_capacity * sizeof(T));
        if (status != cudaSuccess) return status;
        release_device(device);
        *device = replacement;
        *capacity = next_capacity;
    }
    // The topology descriptor vector is ordinary pageable host memory and
    // goes out of scope after runtime initialization. Keep the copy synchronous;
    // allocation reuse removes the expensive free/malloc churn while this
    // preserves an unambiguous host-lifetime contract.
    return cudaMemcpy(*device, host.data(), host.size() * sizeof(T),
                      cudaMemcpyHostToDevice);
}

bool multiply_fits(std::size_t left, std::size_t right) {
    return left == 0 || right <= std::numeric_limits<std::size_t>::max() / left;
}

std::size_t protocol_slot_count(const DlbRuntime& runtime) {
    return static_cast<std::size_t>(runtime.server_count) *
           runtime.config.gpus_per_server * runtime.config.rail_channel_count;
}

std::size_t protocol_receive_bytes(const DlbRuntime& runtime) {
    return static_cast<std::size_t>(runtime.server_count) * runtime.config.receive_slot_bytes;
}

std::uint32_t pipeline_slot(const DlbRuntime& runtime, std::uint64_t epoch) {
    return static_cast<std::uint32_t>(epoch % runtime.config.pipeline_depth);
}

cudaError_t create_pipeline(DlbRuntime* runtime) {
    cudaError_t status = cudaStreamCreateWithFlags(&runtime->stage_stream, cudaStreamNonBlocking);
    if (status == cudaSuccess) {
        status = cudaStreamCreateWithFlags(&runtime->transport_stream, cudaStreamNonBlocking);
    }
    if (status == cudaSuccess) {
        status = cudaStreamCreateWithFlags(&runtime->repair_stream, cudaStreamNonBlocking);
    }
    const std::uint32_t depth = runtime->config.pipeline_depth;
    runtime->source_ready.resize(depth, nullptr);
    runtime->stage_ready.resize(depth, nullptr);
    runtime->transport_ready.resize(depth, nullptr);
    runtime->repair_ready.resize(depth, nullptr);
    if (runtime->config.enable_profiling) {
        runtime->profile_dispatch_begin.resize(depth, nullptr);
        runtime->profile_source_ready.resize(depth, nullptr);
        runtime->profile_stage_ready.resize(depth, nullptr);
        runtime->profile_producers_ready.resize(depth, nullptr);
        runtime->profile_transport_ready.resize(depth, nullptr);
        runtime->profile_repair_kernel_ready.resize(depth, nullptr);
        runtime->profile_repair_ready.resize(depth, nullptr);
        runtime->profile_counted.resize(depth, nullptr);
        runtime->profile_scattered.resize(depth, nullptr);
    }
    for (std::uint32_t slot = 0; status == cudaSuccess && slot < depth; ++slot) {
        status = cudaEventCreateWithFlags(&runtime->source_ready[slot], cudaEventDisableTiming);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(&runtime->stage_ready[slot], cudaEventDisableTiming);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(&runtime->transport_ready[slot], cudaEventDisableTiming);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(&runtime->repair_ready[slot], cudaEventDisableTiming);
        if (status == cudaSuccess && runtime->config.enable_profiling) {
            cudaEvent_t* timed_events[] = {
                &runtime->profile_dispatch_begin[slot],
                &runtime->profile_source_ready[slot],
                &runtime->profile_stage_ready[slot],
                &runtime->profile_producers_ready[slot],
                &runtime->profile_transport_ready[slot],
                &runtime->profile_repair_kernel_ready[slot],
                &runtime->profile_repair_ready[slot],
                &runtime->profile_counted[slot],
                &runtime->profile_scattered[slot],
            };
            for (cudaEvent_t* event : timed_events) {
                status = cudaEventCreateWithFlags(event, cudaEventDefault);
                if (status != cudaSuccess) break;
            }
        }
        // A slot is initially reusable.  Later launches wait for the repair
        // and transport events before overwriting its ping-pong buffers.
        if (status == cudaSuccess) status = cudaEventRecord(runtime->repair_ready[slot], runtime->repair_stream);
        if (status == cudaSuccess) status = cudaEventRecord(runtime->transport_ready[slot], runtime->transport_stream);
    }
    return status;
}

cudaError_t destroy_pipeline(DlbRuntime* runtime) {
    cudaError_t first_error = cudaSuccess;
    const auto destroy_events = [&first_error](std::vector<cudaEvent_t>* events) {
        for (cudaEvent_t event : *events) {
            if (event == nullptr) continue;
            const cudaError_t status = cudaEventDestroy(event);
            if (first_error == cudaSuccess && status != cudaSuccess) first_error = status;
        }
        events->clear();
    };
    destroy_events(&runtime->source_ready);
    destroy_events(&runtime->stage_ready);
    destroy_events(&runtime->transport_ready);
    destroy_events(&runtime->repair_ready);
    destroy_events(&runtime->profile_dispatch_begin);
    destroy_events(&runtime->profile_source_ready);
    destroy_events(&runtime->profile_stage_ready);
    destroy_events(&runtime->profile_producers_ready);
    destroy_events(&runtime->profile_transport_ready);
    destroy_events(&runtime->profile_repair_kernel_ready);
    destroy_events(&runtime->profile_repair_ready);
    destroy_events(&runtime->profile_counted);
    destroy_events(&runtime->profile_scattered);
    if (runtime->stage_stream != nullptr) {
        const cudaError_t status = cudaStreamDestroy(runtime->stage_stream);
        if (first_error == cudaSuccess && status != cudaSuccess) first_error = status;
    }
    if (runtime->transport_stream != nullptr) {
        const cudaError_t status = cudaStreamDestroy(runtime->transport_stream);
        if (first_error == cudaSuccess && status != cudaSuccess) first_error = status;
    }
    if (runtime->repair_stream != nullptr) {
        const cudaError_t status = cudaStreamDestroy(runtime->repair_stream);
        if (first_error == cudaSuccess && status != cudaSuccess) first_error = status;
    }
    runtime->stage_stream = nullptr;
    runtime->transport_stream = nullptr;
    runtime->repair_stream = nullptr;
    return first_error;
}

cudaError_t allocate_device_bytes(std::uint8_t** pointer, std::size_t bytes) {
    const std::size_t allocation = std::max<std::size_t>(bytes, 1);
    return cudaMalloc(reinterpret_cast<void**>(pointer), allocation);
}

cudaError_t destroy_ipc_table(DlbCudaIpcTable* table, std::uint32_t local_rank) {
    cudaError_t first_error = cudaSuccess;
    if (table->device_ptrs != nullptr) {
        const cudaError_t status = cudaFree(table->device_ptrs);
        if (first_error == cudaSuccess && status != cudaSuccess) {
            first_error = status;
        }
    }
    if (table->owns_remote_mappings) {
        for (std::size_t rank = 0; rank < table->host_ptrs.size(); ++rank) {
            if (rank == local_rank || table->host_ptrs[rank] == nullptr) {
                continue;
            }
            const cudaError_t status = cudaIpcCloseMemHandle(table->host_ptrs[rank]);
            if (first_error == cudaSuccess && status != cudaSuccess) {
                first_error = status;
            }
        }
    }
    table->local_buffer = nullptr;
    table->device_ptrs = nullptr;
    table->host_ptrs.clear();
    table->owns_remote_mappings = false;
    return first_error;
}

// Every local rank contributes its CUDA IPC handle through the initialized
// NVSHMEM team and receives a device-visible table of local Rail pointers.
cudaError_t build_ipc_table(DlbCudaIpcTable* table, void* local_buffer,
                            std::uint32_t local_rank, std::uint32_t local_count,
                            nvshmem_team_t local_team) {
    if (local_buffer == nullptr || local_count == 0 || local_rank >= local_count) {
        return cudaErrorInvalidValue;
    }
    const cudaError_t reset_status = destroy_ipc_table(table, local_rank);
    if (reset_status != cudaSuccess) {
        return reset_status;
    }

    cudaIpcMemHandle_t local_handle;
    cudaError_t status = cudaIpcGetMemHandle(&local_handle, local_buffer);
    if (status != cudaSuccess) {
        return status;
    }

    const std::size_t bytes = static_cast<std::size_t>(local_count) * sizeof(cudaIpcMemHandle_t);
    void* symmetric_source = nvshmem_malloc(bytes);
    void* symmetric_destination = nvshmem_malloc(bytes);
    if (symmetric_source == nullptr || symmetric_destination == nullptr) {
        if (symmetric_source != nullptr) nvshmem_free(symmetric_source);
        if (symmetric_destination != nullptr) nvshmem_free(symmetric_destination);
        return cudaErrorMemoryAllocation;
    }

    std::vector<cudaIpcMemHandle_t> send_handles(local_count, local_handle);
    std::vector<cudaIpcMemHandle_t> receive_handles(local_count);
    status = cudaMemcpy(symmetric_source, send_handles.data(), bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        nvshmem_barrier_all();
        const int collective_status = nvshmem_alltoallmem(
            local_team, symmetric_destination, symmetric_source, sizeof(cudaIpcMemHandle_t));
        if (collective_status != NVSHMEMX_SUCCESS) {
            status = cudaErrorUnknown;
        } else {
            // NVSHMEM host collectives may enqueue CUDA work internally.
            // Complete it here: otherwise a bootstrap error can surface at
            // the next unrelated DLB kernel launch and be misattributed.
            nvshmem_quiet();
            status = cudaDeviceSynchronize();
        }
        if (status == cudaSuccess) {
            status = cudaMemcpy(receive_handles.data(), symmetric_destination, bytes,
                                cudaMemcpyDeviceToHost);
        }
    }
    nvshmem_free(symmetric_destination);
    nvshmem_free(symmetric_source);
    if (status != cudaSuccess) {
        return status;
    }

    table->host_ptrs.assign(local_count, nullptr);
    table->local_buffer = local_buffer;
    table->owns_remote_mappings = true;
    for (std::uint32_t rank = 0; rank < local_count; ++rank) {
        if (rank == local_rank) {
            table->host_ptrs[rank] = local_buffer;
            continue;
        }
        status = cudaIpcOpenMemHandle(&table->host_ptrs[rank], receive_handles[rank],
                                      cudaIpcMemLazyEnablePeerAccess);
        if (status != cudaSuccess) {
            destroy_ipc_table(table, local_rank);
            return status;
        }
    }
    status = cudaMalloc(reinterpret_cast<void**>(&table->device_ptrs),
                        static_cast<std::size_t>(local_count) * sizeof(void*));
    if (status == cudaSuccess) {
        status = cudaMemcpy(table->device_ptrs, table->host_ptrs.data(),
                            static_cast<std::size_t>(local_count) * sizeof(void*),
                            cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        destroy_ipc_table(table, local_rank);
    }
    return status;
}

// Build a local-server GPU pointer table for a symmetric allocation.  Unlike
// CUDA IPC bootstrap this needs no handle exchange: NVSHMEM already knows the
// same allocation on every PE and returns the directly load/store-accessible
// peer address.  The returned pointers are non-owning.
cudaError_t build_symmetric_peer_table(DlbCudaIpcTable* table, void* local_buffer,
                                       std::uint32_t first_pe,
                                       std::uint32_t self_index,
                                       std::uint32_t peer_count) {
    if (local_buffer == nullptr || peer_count == 0 || self_index >= peer_count) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = destroy_ipc_table(table, self_index);
    if (status != cudaSuccess) return status;

    table->local_buffer = local_buffer;
    table->owns_remote_mappings = false;
    table->host_ptrs.assign(peer_count, nullptr);
    for (std::uint32_t rank = 0; rank < peer_count; ++rank) {
        const int pe = static_cast<int>(first_pe + rank);
        void* pointer = rank == self_index ? local_buffer : nvshmem_ptr(local_buffer, pe);
        if (pointer == nullptr) {
            std::fprintf(stderr,
                         "DLB cannot directly access local Rail %u (PE %d) symmetric send buffer\n",
                         rank, pe);
            destroy_ipc_table(table, self_index);
            return cudaErrorPeerAccessUnsupported;
        }
        table->host_ptrs[rank] = pointer;
    }
    status = cudaMalloc(reinterpret_cast<void**>(&table->device_ptrs),
                        static_cast<std::size_t>(peer_count) * sizeof(void*));
    if (status == cudaSuccess) {
        status = cudaMemcpy(table->device_ptrs, table->host_ptrs.data(),
                            static_cast<std::size_t>(peer_count) * sizeof(void*),
                            cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) destroy_ipc_table(table, self_index);
    return status;
}

cudaError_t initialize_receive_protocol(DlbRuntime* runtime) {
    const std::size_t one_slot_count = protocol_slot_count(*runtime);
    const std::size_t one_receive_bytes = protocol_receive_bytes(*runtime);
    if (!multiply_fits(one_slot_count, runtime->config.pipeline_depth) ||
        !multiply_fits(one_receive_bytes, runtime->config.pipeline_depth)) {
        return cudaErrorInvalidValue;
    }
    const std::size_t slot_count = one_slot_count * runtime->config.pipeline_depth;
    const std::size_t receive_bytes = one_receive_bytes * runtime->config.pipeline_depth;
    runtime->symmetric_receive_buffer =
        static_cast<std::uint8_t*>(nvshmem_malloc(std::max<std::size_t>(receive_bytes, 1)));
    runtime->symmetric_receive_signals =
        static_cast<std::uint64_t*>(nvshmem_calloc(slot_count, sizeof(std::uint64_t)));
    runtime->symmetric_progress_signals =
        static_cast<std::uint64_t*>(nvshmem_calloc(slot_count, sizeof(std::uint64_t)));
    runtime->symmetric_credits =
        static_cast<std::uint64_t*>(nvshmem_calloc(slot_count, sizeof(std::uint64_t)));
    if (runtime->symmetric_receive_buffer == nullptr || runtime->symmetric_receive_signals == nullptr ||
        runtime->symmetric_progress_signals == nullptr || runtime->symmetric_credits == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    cudaError_t status = cudaMemset(runtime->symmetric_receive_buffer, 0, receive_bytes);
    if (status != cudaSuccess) {
        return status;
    }
    std::vector<std::uint64_t> initial_credits(slot_count, runtime->config.initial_epoch);
    status = cudaMemcpy(runtime->symmetric_credits, initial_credits.data(),
                        slot_count * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        nvshmem_barrier_all();
    }
    return status;
}

void release_runtime_buffers(DlbRuntime* runtime) {
    release_device(&runtime->device_transfers);
    release_device(&runtime->device_arrivals);
    for (DlbCudaIpcTable& table : runtime->rail_send_tables) {
        destroy_ipc_table(&table, runtime->local_rank);
    }
    runtime->rail_send_tables.clear();
    destroy_ipc_table(&runtime->repair_table, runtime->local_rank);
    destroy_ipc_table(&runtime->stage_signal_table, runtime->local_rank);
    destroy_ipc_table(&runtime->completion_signal_table, runtime->local_rank);
    destroy_ipc_table(&runtime->loopback_receive_table, runtime->config.rank);
    destroy_ipc_table(&runtime->loopback_receive_signal_table, runtime->config.rank);
    destroy_ipc_table(&runtime->loopback_progress_signal_table, runtime->config.rank);
    destroy_ipc_table(&runtime->loopback_credit_table, runtime->config.rank);
    release_device(&runtime->local_stage_signals);
    release_device(&runtime->local_completion_signals);
    release_device(&runtime->local_repair_buffer);
    if (runtime->symmetric_rail_send_buffer != nullptr) nvshmem_free(runtime->symmetric_rail_send_buffer);
    if (runtime->symmetric_credits != nullptr) nvshmem_free(runtime->symmetric_credits);
    if (runtime->symmetric_progress_signals != nullptr) nvshmem_free(runtime->symmetric_progress_signals);
    if (runtime->symmetric_receive_signals != nullptr) nvshmem_free(runtime->symmetric_receive_signals);
    if (runtime->symmetric_receive_buffer != nullptr) nvshmem_free(runtime->symmetric_receive_buffer);
    runtime->symmetric_credits = nullptr;
    runtime->symmetric_progress_signals = nullptr;
    runtime->symmetric_receive_signals = nullptr;
    runtime->symmetric_receive_buffer = nullptr;
    runtime->symmetric_rail_send_buffer = nullptr;
    destroy_pipeline(runtime);
}

}  // namespace

cudaError_t initialize_dlb_nvshmem_rdma_runtime(DlbRuntime* runtime,
                                                const DlbRuntimeConfig& config) {
    if (runtime == nullptr || config.gpus_per_server == 0 || config.world_size == 0 ||
        config.world_size % config.gpus_per_server != 0 || config.rank >= config.world_size ||
        config.record_bytes == 0 || config.receive_slot_bytes == 0 ||
        config.repair_slot_bytes == 0 || config.initial_epoch == 0 ||
        config.pipeline_depth == 0 || config.chunk_bytes == 0 ||
        config.rail_channel_count == 0 ||
        config.rail_send_capacity_bytes == 0) {
        return cudaErrorInvalidValue;
    }
    if (config.receive_slot_bytes >= std::numeric_limits<std::uint32_t>::max()) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t server_count = config.world_size / config.gpus_per_server;
    if (server_count < 2) {
        // A one-server topology has no remote DLB tile. Use two logical
        // servers (for example 2 x 4 ranks on one 8-GPU host) for loopback
        // NVSHMEM transport testing.
        return cudaErrorInvalidConfiguration;
    }
    if (!multiply_fits(config.world_size, config.repair_slot_bytes) ||
        !multiply_fits(config.world_size, config.rail_channel_count) ||
        !multiply_fits(static_cast<std::size_t>(config.world_size) * config.repair_slot_bytes,
                       config.pipeline_depth) ||
        !multiply_fits(config.rail_send_capacity_bytes, config.pipeline_depth) ||
        !multiply_fits(static_cast<std::size_t>(config.gpus_per_server), config.pipeline_depth)) {
        return cudaErrorInvalidValue;
    }

    *runtime = DlbRuntime{};
    runtime->config = config;
    runtime->server_count = server_count;
    runtime->server_rank = config.rank / config.gpus_per_server;
    runtime->local_rank = config.rank % config.gpus_per_server;
    runtime->local_repair_buffer = nullptr;
    runtime->rail_send_tables.resize(config.pipeline_depth);

    cudaError_t status = allocate_device_bytes(
        &runtime->local_repair_buffer,
        static_cast<std::size_t>(config.world_size) * config.repair_slot_bytes *
            config.pipeline_depth);
    if (status == cudaSuccess) {
        status = cudaMemset(runtime->local_repair_buffer, 0,
                            static_cast<std::size_t>(config.world_size) * config.repair_slot_bytes *
                                config.pipeline_depth);
    }
    if (status == cudaSuccess) {
        status = build_ipc_table(&runtime->repair_table, runtime->local_repair_buffer,
                                 runtime->local_rank, config.gpus_per_server, config.local_team);
    }
    const std::size_t ipc_signal_count =
        static_cast<std::size_t>(config.pipeline_depth) * config.gpus_per_server;
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&runtime->local_stage_signals),
                            ipc_signal_count * sizeof(std::uint64_t));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&runtime->local_completion_signals),
                            ipc_signal_count * sizeof(std::uint64_t));
    }
    if (status == cudaSuccess) {
        status = cudaMemset(runtime->local_stage_signals, 0,
                            ipc_signal_count * sizeof(std::uint64_t));
    }
    if (status == cudaSuccess) {
        status = cudaMemset(runtime->local_completion_signals, 0,
                            ipc_signal_count * sizeof(std::uint64_t));
    }
    if (status == cudaSuccess) {
        status = build_ipc_table(&runtime->stage_signal_table,
                                 runtime->local_stage_signals, runtime->local_rank,
                                 config.gpus_per_server, config.local_team);
    }
    if (status == cudaSuccess) {
        status = build_ipc_table(&runtime->completion_signal_table,
                                 runtime->local_completion_signals, runtime->local_rank,
                                 config.gpus_per_server, config.local_team);
    }
    if (status == cudaSuccess) {
        const std::size_t symmetric_send_bytes =
            static_cast<std::size_t>(config.rail_send_capacity_bytes) * config.pipeline_depth;
        runtime->symmetric_rail_send_buffer = static_cast<std::uint8_t*>(
            nvshmem_align(16, symmetric_send_bytes));
        if (runtime->symmetric_rail_send_buffer == nullptr) {
            status = cudaErrorMemoryAllocation;
        } else {
            status = cudaMemset(runtime->symmetric_rail_send_buffer, 0, symmetric_send_bytes);
        }
    }
    if (status == cudaSuccess) {
        // nvshmem_align is collective.  Make every PE's allocation visible
        // before resolving direct peer pointers for the local Rail group.
        nvshmem_barrier_all();
        for (std::uint32_t slot = 0; status == cudaSuccess &&
                                      slot < config.pipeline_depth; ++slot) {
            void* slot_buffer = runtime->symmetric_rail_send_buffer +
                static_cast<std::size_t>(slot) * config.rail_send_capacity_bytes;
            status = build_symmetric_peer_table(
                &runtime->rail_send_tables[slot], slot_buffer,
                runtime->server_rank * config.gpus_per_server,
                runtime->local_rank,
                config.gpus_per_server);
        }
    }
    if (status == cudaSuccess) {
        status = initialize_receive_protocol(runtime);
    }
    if (status == cudaSuccess && config.use_loopback_transport) {
        status = build_symmetric_peer_table(
            &runtime->loopback_receive_table, runtime->symmetric_receive_buffer,
            0, config.rank, config.world_size);
    }
    if (status == cudaSuccess && config.use_loopback_transport) {
        status = build_symmetric_peer_table(
            &runtime->loopback_receive_signal_table,
            runtime->symmetric_receive_signals, 0, config.rank,
            config.world_size);
    }
    if (status == cudaSuccess && config.use_loopback_transport) {
        status = build_symmetric_peer_table(
            &runtime->loopback_progress_signal_table,
            runtime->symmetric_progress_signals, 0, config.rank,
            config.world_size);
    }
    if (status == cudaSuccess && config.use_loopback_transport) {
        status = build_symmetric_peer_table(
            &runtime->loopback_credit_table, runtime->symmetric_credits,
            0, config.rank, config.world_size);
    }
    if (status == cudaSuccess) {
        status = create_pipeline(runtime);
    }
    if (status == cudaSuccess) {
        runtime->rail_receive_plan = materialize_rail_receive_plan(
            runtime->server_count, config.gpus_per_server, runtime->server_rank,
            runtime->local_rank, config.rail_channel_count);
        runtime->device_arrival_count = static_cast<std::uint32_t>(runtime->rail_receive_plan.arrivals.size());
        std::size_t arrival_capacity = 0;
        status = upload_vector(&runtime->device_arrivals, &arrival_capacity,
                               runtime->rail_receive_plan.arrivals);
    }
    if (status != cudaSuccess) {
        release_runtime_buffers(runtime);
        *runtime = DlbRuntime{};
        return status;
    }
    runtime->initialized = true;
    return cudaSuccess;
}

cudaError_t launch_dlb_nvshmem_rdma(DlbRuntime* runtime,
                                    std::uint64_t epoch,
                                    cudaStream_t stream) {
    if (runtime == nullptr || !runtime->initialized || epoch < runtime->config.initial_epoch ||
        epoch == std::numeric_limits<std::uint64_t>::max()) {
        return cudaErrorInvalidValue;
    }
    // Keep asynchronous CUDA/NVSHMEM errors attributed to the operation that
    // caused them. In particular this distinguishes IPC bootstrap errors from
    // an error in the direct-pack/coordination pipeline.
    const cudaError_t pending_status = cudaGetLastError();
    if (pending_status != cudaSuccess) {
        std::fprintf(stderr, "DLB launch has a pending CUDA error before transport: %s\\n",
                     cudaGetErrorString(pending_status));
        return pending_status;
    }

    const std::uint32_t slot = pipeline_slot(*runtime, epoch);
    // Each protocol slot carries an independent monotonically increasing
    // sequence: epoch 1 and 2 use the two initial ping-pong slots at sequence
    // 1; epoch 3 returns to slot 1 at sequence 2.
    const std::uint64_t slot_epoch = runtime->config.initial_epoch +
        (epoch - runtime->config.initial_epoch) / runtime->config.pipeline_depth;
    const std::size_t receive_bytes = protocol_receive_bytes(*runtime);
    const std::size_t slot_count = protocol_slot_count(*runtime);
    const std::uint32_t ipc_slot_begin = slot * runtime->config.gpus_per_server;
    const std::uint32_t ipc_writer_offset = ipc_slot_begin + runtime->local_rank;
    const std::uint64_t repair_epoch_offset =
        static_cast<std::uint64_t>(slot) * runtime->config.world_size *
        runtime->config.repair_slot_bytes;

    cudaError_t status = cudaEventRecord(runtime->source_ready[slot], stream);
    if (status == cudaSuccess && runtime->config.enable_profiling) {
        status = cudaEventRecord(runtime->profile_source_ready[slot], stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamWaitEvent(runtime->stage_stream, runtime->repair_ready[slot], 0);
    }
    if (status == cudaSuccess) {
        status = cudaStreamWaitEvent(runtime->stage_stream, runtime->transport_ready[slot], 0);
    }
    if (status == cudaSuccess) {
        status = cudaStreamWaitEvent(runtime->stage_stream, runtime->source_ready[slot], 0);
    }
    if (status != cudaSuccess) return status;

    status = launch_dlb_publish_ipc_epoch(
        reinterpret_cast<std::uint64_t* const*>(runtime->stage_signal_table.device_ptrs),
        runtime->config.gpus_per_server, ipc_writer_offset, slot_epoch,
        runtime->stage_stream);
    if (status == cudaSuccess) {
        status = cudaEventRecord(runtime->stage_ready[slot], runtime->stage_stream);
    }
    if (status == cudaSuccess && runtime->config.enable_profiling) {
        status = cudaEventRecord(runtime->profile_stage_ready[slot], runtime->stage_stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamWaitEvent(runtime->transport_stream, runtime->stage_ready[slot], 0);
    }
    if (status == cudaSuccess) {
        status = launch_dlb_wait_ipc_epochs(
            runtime->local_stage_signals, ipc_slot_begin,
            runtime->config.gpus_per_server, slot_epoch, runtime->transport_stream);
    }
    if (status == cudaSuccess && runtime->config.enable_profiling) {
        status = cudaEventRecord(runtime->profile_producers_ready[slot],
                                 runtime->transport_stream);
    }
    std::uint8_t* symmetric_send_buffer = runtime->symmetric_rail_send_buffer +
        static_cast<std::size_t>(slot) * runtime->config.rail_send_capacity_bytes;
    if (status == cudaSuccess) {
        if (runtime->config.use_loopback_transport) {
            status = launch_dlb_loopback_rail(
                symmetric_send_buffer,
                reinterpret_cast<std::uint8_t* const*>(
                    runtime->loopback_receive_table.device_ptrs),
                reinterpret_cast<std::uint64_t* const*>(
                    runtime->loopback_receive_signal_table.device_ptrs),
                reinterpret_cast<std::uint64_t* const*>(
                    runtime->loopback_progress_signal_table.device_ptrs),
                runtime->symmetric_credits + slot * slot_count,
                runtime->device_transfers, runtime->device_transfer_count,
                runtime->config.rail_channel_count,
                slot_epoch, runtime->config.receive_slot_bytes,
                runtime->config.chunk_bytes,
                static_cast<std::uint64_t>(slot) * receive_bytes,
                static_cast<std::uint64_t>(slot) * slot_count,
                runtime->transport_stream);
        } else {
            status = launch_dlb_internode_rail(
                symmetric_send_buffer,
                runtime->symmetric_receive_buffer + slot * receive_bytes,
                runtime->symmetric_receive_signals + slot * slot_count,
                runtime->symmetric_progress_signals + slot * slot_count,
                runtime->symmetric_credits + slot * slot_count,
                runtime->device_transfers, runtime->device_transfer_count,
                runtime->config.rail_channel_count,
                slot_epoch, runtime->config.receive_slot_bytes,
                runtime->config.chunk_bytes, runtime->transport_stream);
        }
        if (status != cudaSuccess) {
            std::fprintf(stderr, "DLB launch failed at Rail put: %s\n", cudaGetErrorString(status));
        }
    }
    if (status == cudaSuccess) {
        status = cudaEventRecord(runtime->transport_ready[slot], runtime->transport_stream);
    }
    if (status == cudaSuccess && runtime->config.enable_profiling) {
        status = cudaEventRecord(runtime->profile_transport_ready[slot],
                                 runtime->transport_stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamWaitEvent(runtime->repair_stream, runtime->stage_ready[slot], 0);
    }
    if (status == cudaSuccess) {
        status = launch_dlb_receive_and_repair_chunks(
            runtime->symmetric_receive_buffer + slot * receive_bytes,
            runtime->symmetric_receive_signals + slot * slot_count,
            runtime->symmetric_progress_signals + slot * slot_count,
            runtime->symmetric_credits + slot * slot_count,
            runtime->device_arrivals, runtime->device_arrival_count,
            runtime->config.rail_channel_count, slot_epoch,
            runtime->config.receive_slot_bytes, runtime->config.gpus_per_server,
            reinterpret_cast<std::uint8_t* const*>(runtime->repair_table.device_ptrs),
            runtime->config.repair_slot_bytes, repair_epoch_offset,
            runtime->config.chunk_bytes,
            runtime->config.use_loopback_transport,
            reinterpret_cast<std::uint64_t* const*>(
                runtime->loopback_credit_table.device_ptrs),
            static_cast<std::uint64_t>(slot) * slot_count,
            runtime->repair_stream);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "DLB launch failed at chunk receive/repair: %s\n",
                         cudaGetErrorString(status));
        }
    }
    if (status == cudaSuccess) {
        status = launch_dlb_publish_ipc_epoch(
            reinterpret_cast<std::uint64_t* const*>(runtime->completion_signal_table.device_ptrs),
            runtime->config.gpus_per_server, ipc_writer_offset, slot_epoch,
            runtime->repair_stream);
    }
    if (status == cudaSuccess && runtime->config.enable_profiling) {
        status = cudaEventRecord(runtime->profile_repair_kernel_ready[slot],
                                 runtime->repair_stream);
    }
    if (status == cudaSuccess) {
        status = launch_dlb_wait_ipc_epochs(
            runtime->local_completion_signals, ipc_slot_begin,
            runtime->config.gpus_per_server, slot_epoch, runtime->repair_stream);
    }
    if (status == cudaSuccess) {
        status = cudaEventRecord(runtime->repair_ready[slot], runtime->repair_stream);
    }
    if (status == cudaSuccess && runtime->config.enable_profiling) {
        status = cudaEventRecord(runtime->profile_repair_ready[slot],
                                 runtime->repair_stream);
    }
    return status;
}

cudaError_t read_dlb_nvshmem_profile(DlbRuntime* runtime,
                                     std::uint64_t epoch,
                                     float* milliseconds,
                                     std::size_t metric_count) {
    if (runtime == nullptr || !runtime->initialized ||
        !runtime->config.enable_profiling || milliseconds == nullptr ||
        metric_count != kDlbProfileMetricCount ||
        epoch < runtime->config.initial_epoch) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t slot = pipeline_slot(*runtime, epoch);
    // The result/scatter stream waits for inbound repair, while the local
    // transport stream may still be completing this PE's outbound puts. Both
    // branches belong to the profiled operation and CUDA event elapsed-time
    // queries require their end events to have completed.
    cudaError_t status = cudaEventSynchronize(runtime->profile_transport_ready[slot]);
    if (status == cudaSuccess) {
        status = cudaEventSynchronize(runtime->profile_scattered[slot]);
    }
    if (status != cudaSuccess) return status;
    const auto elapsed = [&status](float* value, cudaEvent_t begin,
                                   cudaEvent_t end) {
        if (status == cudaSuccess) status = cudaEventElapsedTime(value, begin, end);
    };
    elapsed(&milliseconds[0], runtime->profile_dispatch_begin[slot],
            runtime->profile_source_ready[slot]);
    elapsed(&milliseconds[1], runtime->profile_source_ready[slot],
            runtime->profile_stage_ready[slot]);
    elapsed(&milliseconds[2], runtime->profile_stage_ready[slot],
            runtime->profile_producers_ready[slot]);
    milliseconds[3] = 0.0f;
    elapsed(&milliseconds[4], runtime->profile_producers_ready[slot],
            runtime->profile_transport_ready[slot]);
    elapsed(&milliseconds[5], runtime->profile_stage_ready[slot],
            runtime->profile_repair_kernel_ready[slot]);
    elapsed(&milliseconds[6], runtime->profile_repair_kernel_ready[slot],
            runtime->profile_repair_ready[slot]);
    elapsed(&milliseconds[7], runtime->profile_repair_ready[slot],
            runtime->profile_counted[slot]);
    elapsed(&milliseconds[8], runtime->profile_counted[slot],
            runtime->profile_scattered[slot]);
    elapsed(&milliseconds[9], runtime->profile_dispatch_begin[slot],
            runtime->profile_scattered[slot]);
    return status;
}

cudaError_t wait_dlb_nvshmem_rdma_epoch(DlbRuntime* runtime,
                                        std::uint64_t epoch,
                                        cudaStream_t stream) {
    if (runtime == nullptr || !runtime->initialized || epoch < runtime->config.initial_epoch) {
        return cudaErrorInvalidValue;
    }
    return cudaStreamWaitEvent(stream, runtime->repair_ready[pipeline_slot(*runtime, epoch)], 0);
}

std::uint8_t* dlb_nvshmem_repair_buffer_for_epoch(DlbRuntime* runtime,
                                                  std::uint64_t epoch) {
    if (runtime == nullptr || !runtime->initialized || epoch < runtime->config.initial_epoch) {
        return nullptr;
    }
    const std::uint64_t lane_bytes = static_cast<std::uint64_t>(runtime->config.world_size) *
                                     runtime->config.repair_slot_bytes;
    return runtime->local_repair_buffer + pipeline_slot(*runtime, epoch) * lane_bytes;
}

cudaError_t destroy_dlb_nvshmem_rdma_runtime(DlbRuntime* runtime) {
    if (runtime == nullptr) {
        return cudaErrorInvalidValue;
    }
    const cudaError_t sync_status = cudaDeviceSynchronize();
    release_runtime_buffers(runtime);
    *runtime = DlbRuntime{};
    return sync_status;
}

}  // namespace dlb_alltoall
