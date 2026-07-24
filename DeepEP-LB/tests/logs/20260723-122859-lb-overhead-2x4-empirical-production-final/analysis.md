# Empirical TopK=8 LB Performance Comparison

## Input

The benchmark uses the byte-identical empirical distribution copied from
`eth-htsim-ocs-eps`:

```text
SHA-256:
55cdb477e3952fb9a14ac44735936b4f6675dd69310e5cdc9737c10ab8b34e4d
```

| Item | Value |
|---|---:|
| Global batch | 16 |
| Sequence length | 2048 |
| Global tokens | 32768 |
| Tokens per EP Rank | 4096 |
| Experts | 256 |
| TopK | 8 |
| Expert assignments | 262144 |
| Hidden | 2048 |
| EP | 8 |
| Logical topology | 2 servers × 4 Rails |

Every token selects eight distinct experts according to the empirical expert
hotness. The realized assignment distribution is strongly skewed:

| Statistic | Assignments per expert |
|---|---:|
| Minimum | 6 |
| P50 | 387 |
| P95 | 6295 |
| Maximum | 11823 |
| Mean | 1024 |

After DeepEP token-to-Rank deduplication, the 262,144 expert assignments become
171,741 token/Rank messages.

## Rail Result

The empirical expert skew does not create a large source-Rail skew in this
EP8 topology because almost every TopK=8 token reaches both logical servers.

| Tile | LB OFF | LB cached |
|---|---|---|
| server 0 → server 1 | `[4090, 4087, 4091, 4093]` | `[4090, 4090, 4090, 4091]` |
| server 1 → server 0 | `[4089, 4088, 4090, 4089]` | `[4089, 4089, 4089, 4089]` |

The cached plan moves only four of 65,415 wire messages. It therefore measures
mostly the cost of enabling LB rather than a communication gain.

## BF16

| Path | Dispatch P50 | Dispatch P95 | Data plane P50 | Effective GB/s | Combine P50 |
|---|---:|---:|---:|---:|---:|
| LB OFF | 1.809 ms | 1.977 ms | 1.395 ms | 151.57 | 2.292 ms |
| LB cold | 2.960 ms | 3.484 ms | 1.325 ms | 92.63 | 2.140 ms |
| LB cached | 2.191 ms | 2.383 ms | 1.452 ms | 125.17 | 2.406 ms |

Cached LB relative to OFF:

- Dispatch P50: `+0.382 ms` (`+21.1%`)
- Data-plane P50: `+0.057 ms` (`+4.1%`)
- Combine P50: `+0.114 ms` (`+5.0%`)
- End-to-end effective throughput: `-17.4%`

## FP8

| Path | Dispatch P50 | Dispatch P95 | Data plane P50 | Effective GB/s |
|---|---:|---:|---:|---:|
| LB OFF | 1.705 ms | 2.014 ms | 1.277 ms | 84.74 |
| LB cold | 3.142 ms | 3.455 ms | 1.300 ms | 45.97 |
| LB cached | 2.100 ms | 2.390 ms | 1.332 ms | 68.78 |

Cached LB relative to OFF:

- Dispatch P50: `+0.395 ms` (`+23.2%`)
- Data-plane P50: `+0.055 ms` (`+4.3%`)
- End-to-end effective throughput: `-18.8%`

## Control-Plane Cost

| Operation | BF16 P50 | FP8 P50 |
|---|---:|---:|
| Cold planner CUDA kernels | 0.142 ms | 0.140 ms |
| Cached plan D2D restore | 0.016 ms | 0.015 ms |
| Cached control host wall | 0.416 ms | 0.430 ms |

The CUDA cost of reusing a cached plan is approximately 15–16 microseconds.
The larger host-wall number is caused by the test-only loopback benchmark
synchronizing every phase explicitly. The cold path additionally contains
Gloo process barriers. These host-wall values must not be interpreted as
production IBGDA/RDMA performance.

## Result

- BF16 and FP8 dispatch correctness passed for OFF, cold, and cached LB.
- BF16 combine correctness passed for all three paths.
- Token count, TopK metadata, hidden payload, final Rank, and expert routes are
  unchanged by LB.
- Every remote tile satisfies `max(rail_load) - min(rail_load) <= 1`.
- The empirical case is already nearly source-Rail balanced, so the measured
  difference is the LB no-benefit overhead case.

