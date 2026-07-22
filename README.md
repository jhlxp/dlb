# DLB

DLB is an algorithm for source-server-local scheduling on Rail-based GPU clusters. Its goal is to balance each server's outgoing all-to-all traffic without requiring a cluster-wide demand matrix or a global communication schedule.

The DLB algorithm utilizes a conclusion derived from RailS: under the Rail architecture, if each server transmits traffic in a perfectly load-balanced manner, then the aggregate traffic received by the servers is also perfectly load-balanced.

## Repository Layout

- `simulation/dlb/`: DLB reference scheduler and evaluation benchmark. Its CSV and figures are written to `simulation/dlb/outputs/`.
- `nvidia-dlb/`: standalone DLB CUDA/NVSHMEM runtime, Python API, tests, and logs.
- `nvidia-dlb/src/dlb_alltoall/`: DLB source-local planning and GPU communication runtime.
- `nvidia-fast/`: isolated FAST comparison implementation and its own build.

## DLB Design

DLB schedules communication independently at every source server. With `S`
servers and `M` GPUs/NICs per server, a controller consumes only its local
logical-demand rows of shape `M x (S*M)`:

1. Split the local demand into one `M x M` tile for each remote destination
   server.
2. For each tile, distribute its records over the `M` local Rail NICs as
   evenly as possible. Every NIC receives either `floor(total / M)` or
   `ceil(total / M)` records for that tile.
3. Preserve the complete route for every record:
   `source GPU -> selected source Rail -> destination Rail -> real destination GPU`.
   Selecting a Rail changes only the physical path, not the logical target.
4. Materialize the result on GPU as direct-pack offsets, inter-server Rail
   transfers, and destination-side repair metadata. Each source server can
   update this plan concurrently with every other server.

DLB therefore neither collects a global `S x S` server matrix nor performs a global decomposition. It trades a globally coordinated schedule for local Rail balancing plus destination-side repair.
