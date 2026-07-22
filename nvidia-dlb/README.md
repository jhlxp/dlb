# NVIDIA DLB

This directory is the standalone CUDA/NVSHMEM implementation of DLB. It owns
its reference scheduler, GPU-resident dynamic planner, communication runtimes,
Torch binding, model-facing Python API, tests, and logs.

## Layout

```text
nvidia-dlb/
├── CMakeLists.txt
├── dlb.py                         # MoE-facing dispatch/combine API
├── dlb_nvshmem_binding.cpp        # Torch custom-class binding
├── dlb_nvshmem_utils.py           # Shared-library loader
├── scripts/
│   └── run_dlb_moe_network_test.sh # Default logged EP8 regression
├── src/
│   ├── dlb_alltoall/              # Scheduler and CUDA/NVSHMEM runtime
│   └── include/dlb_alltoall/      # DLB headers
├── tests/                         # Scheduler, validation, and EP8 network tests
├── benchmarks/                    # Performance-only programs
└── log/                           # Timestamped test results
```

## Design

With `S` servers and `M` GPUs/Rails per server, every source server consumes
only its local `M x (S*M)` logical-demand rows. For each remote server it:

1. extracts one `M x M` destination tile;
2. balances that tile over the `M` source Rails;
3. preserves the real destination GPU for destination-side repair;
4. materializes direct-pack offsets, Rail transfers, and repair metadata.

The resulting path is:

```text
source GPU -> selected local Rail -> matching destination Rail
           -> real destination GPU -> local expert
```

DLB does not collect a cluster-wide demand matrix and does not construct a
global communication schedule.

The data plane is split into four layers:

1. `dlb_moe_runtime` counts routes, rebuilds the source-local DLB plan, packs
   records directly into Rail/repair buffers, and restores expert/token order
   in CUDA kernels.
2. `dlb_alltoall_nvshmem` owns buffers, CUDA IPC tables, streams/events, and
   dispatch/combine epoch orchestration.
3. `dlb_nvlink_runtime` publishes and waits for CUDA-IPC-visible epoch
   signals between the GPU processes of one logical server.
4. `dlb_rail_transport` performs chunked NVSHMEM puts (or explicit same-host
   loopback copies), progress/credit handling, and destination-side repair.
   Every Rail group is divided across fixed sender/receiver channels; the
   retained `dlb_rail_receive_plan` builds their topology-only arrival table
   once during initialization.

There is one production execution path: GPU dynamic planning followed by
direct pack. The former host action-table/packed-layout execution path has
been removed; `dlb_scheduler` remains only as the readable CPU reference for
algorithm tests and benchmarks.

The same runtime supports real multi-node RDMA and an explicit same-host
`2xM` loopback test backend. The upper-layer DLB route and record protocol are
identical, while the backend is selected by `transport_backend`; loopback is
never selected implicitly in production.

## Build

Use the compute capability returned by PyTorch. On the current customized
H200 environment, `torch.cuda.get_device_capability()` returns `(9, 0)`.

```bash
DLB_VENV=/path/to/python-env
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${DLB_VENV}/lib/python3.11/site-packages/torch/share/cmake" \
  -DNVSHMEM_ROOT="${DLB_VENV}/lib/python3.11/site-packages/nvidia/nvshmem" \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DBUILD_DLB_TORCH_EXTENSION=ON \
  -DBUILD_DLB_SCHEDULER_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The extension is written to `libdlb_nvshmem.so` beside
`dlb_nvshmem_utils.py`.

For the CPU-only scheduler test:

```bash
cmake -S . -B build-host \
  -DBUILD_DLB_ALLTOALL_RUNTIME=OFF \
  -DBUILD_DLB_NVSHMEM_RUNTIME=OFF \
  -DBUILD_DLB_NVLINK_RUNTIME=OFF \
  -DBUILD_DLB_SCHEDULER_TESTS=ON
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

## MoE API

`DLBBuffer` hides demand matrices, plans, byte records, and transport buffers
from model code:

```python
from dlb import DLBBuffer

buffer = DLBBuffer(
    dist.group.WORLD,
    gpus_per_server=4,
    num_max_tokens_per_rank=max_tokens,
    hidden=hidden,
    num_experts=num_experts,
    num_topk=num_topk,
    num_comm_sms=24,
)

ticket = buffer.post_dispatch(x, topk_idx, topk_weights, expert_alignment=128)

# Run work that does not read this batch's recv_x here.
run_independent_compute()

recv_x, recv_expert_idx, recv_weights, handle, event = ticket.finish()
event.current_stream_wait()

expert_output = grouped_expert_gemm(
    recv_x, handle.num_recv_tokens_per_expert_list,
)
combined_x, combined_weights, event = buffer.combine(expert_output, handle)
event.current_stream_wait()
```

`post_dispatch` performs GPU route counting, server-local demand collection,
Rail planning, direct packing, and transport launch without a host wait.
`finish` waits for the posted epoch, repairs received records into the
expert-major layout, and performs the small count readback needed for exact
PyTorch output allocation.  A buffer accepts one outstanding ticket, so the
application must finish it before the next dispatch or combine.

The handle carries the original router tensors, per-expert aligned counts,
source rank/token/top-k metadata, and the padding mask needed by `combine`.

Python is only the model-facing API and test harness. The production
`post_dispatch_moe`, `finish_dispatch_moe`, and `combine_moe` extension calls
keep route counting, server-local demand collection, DLB planning, record
packing, receive repair, expert grouping, and reverse token accumulation in
C++/CUDA. No payload, expanded route, demand matrix, or action table crosses
into Python. The small per-expert count vector is copied to the host once so
PyTorch can allocate the exact aligned expert output shape.

`num_comm_sms` controls the fixed communication-CTA budget and must be a
positive even number. The runtime creates `num_comm_sms / 2` Rail channels;
one sender CTA and one receiver CTA own each channel. The default `24` is the
balanced setting. A communication-bound deployment can tune it independently
of the DLB scheduling algorithm (for example, `48` favors bandwidth at the
cost of more SM capacity).

## Default EP8 neural-network regression

The default regression constructs a deterministic `Linear` router and 256
two-layer MLP experts. Eight ranks each execute their 32 local experts after a
Top-8 DLB dispatch, combine the results, and compare every output element with
a communication-free oracle. The default workload uses global batch 16 and
sequence length 2048, giving 32768 global tokens and 4096 source tokens per
rank. Hidden states use 2048 BF16 elements. Dispatch groups all Top-K
selections of a token that reside on the same destination rank into one
4224-byte wire record (128-byte header plus 4096-byte hidden vector), then
expands it into expert rows at the receiver. Combine uses the saved group
mapping to perform a local FP32 weighted sum on the expert rank and returns
one hidden vector per `(source rank, token, expert rank)`. The result JSON
reports expert routes, actual rank-deduplicated dispatch/combine wire records,
wire bandwidth, semantic hidden-data throughput, and optional CUDA stage
profiles separately. It also records per-Rail and per-channel record counts,
so an apparently balanced Rail total cannot hide a hot communication channel.

```bash
DLB_PYTHON=/path/to/python-env/bin/python \
  ./scripts/run_dlb_moe_network_test.sh
```

Optional test arguments are forwarded directly:

```bash
DLB_PYTHON=/path/to/python-env/bin/python \
  ./scripts/run_dlb_moe_network_test.sh \
  --global-batch-size 8 --sequence-length 4096 --steps 4
```

Each invocation creates `log/YYYYMMDD-HHMMSS-xxxx/` with:

- `command.txt`: exact command;
- `environment.txt`: host, Python, PyTorch, CUDA, and GPU capability;
- `stdout.log`: complete output from all ranks;
- `result.json`: correctness, timing, memory, and Rail traffic;
- `status.txt`: pass/fail and exit code.

## Split-phase overlap benchmark

The following benchmark compares complete sequential MoE microbatches against
the split-phase schedule that posts microbatch `n + 1` before evaluating the
local expert MLP for microbatch `n`.  It validates each pipelined result both
against the sequential result and a communication-free MoE reference.

```bash
DLB_PYTHON=/path/to/python-env/bin/python \
  ./scripts/run_dlb_moe_overlap_benchmark.sh \
  --warmup-steps 10 --steps 20
```

It reports P50/P90 end-to-end microbatch time, tokens per second, the expert
kernel time, and the steady-state split-phase speedup.  Router computation and
reference validation are deliberately outside the timed interval.

To measure DLB itself without router, expert, or combine computation, run:

```bash
torchrun --standalone --nproc-per-node=8 benchmarks/bench_dlb_fused_prepare.py
./build-bench/bench_dlb_scheduler --iterations 50000
```

`bench_dlb_fused_prepare.py` measures the GPU-resident
count/collect/plan/direct-pack path while excluding Rail transport.

## Diagnostic tests

Use the narrower tests when the neural-network regression fails:

```bash
# CPU algorithm invariants.
ctest --test-dir build --output-on-failure

# Distributed input validation: one rank injects an invalid expert ID.
torchrun --standalone --nproc-per-node=8 tests/test_dlb_invalid_expert.py
```

The single-host topology treats GPUs `0..3` and `4..7` as two logical EP4
servers. Matching logical Rails are `0<->4`, `1<->5`, `2<->6`, and `3<->7`.
NVSHMEM uses node-local P2P/NVLink for this test and RDMA for peers on another
host.

Control-plane benchmarks:

```bash
cmake -S . -B build-bench \
  -DBUILD_DLB_ALLTOALL_RUNTIME=OFF \
  -DBUILD_DLB_NVSHMEM_RUNTIME=OFF \
  -DBUILD_DLB_NVLINK_RUNTIME=OFF \
  -DBUILD_DLB_BENCHMARKS=ON
cmake --build build-bench --target bench_dlb_scheduler -j
./build-bench/bench_dlb_scheduler --iterations 10000

```
