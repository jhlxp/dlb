# Exact Rail Load: Single-Run Performance

## Configuration

```text
EP8
2 logical servers × 4 Rails
8000 tokens per Rank
64000 global tokens
hidden = 2048
experts = 256
TopK = 8
5 warmups
1 measured execution per case
```

This is a single measured execution, not a percentile benchmark.

## Rail Loads

Both directions use the same controlled demand:

| Path | Rail 0 | Rail 1 | Rail 2 | Rail 3 | Total |
|---|---:|---:|---:|---:|---:|
| LB OFF | 5000 | 6000 | 7000 | 8000 | 26000 |
| LB cold | 6500 | 6500 | 6500 | 6500 | 26000 |
| LB cached | 6500 | 6500 | 6500 | 6500 | 26000 |

DLB moves the minimum 2,000 messages per direction:

```text
Rail 3 surplus: 1500
Rail 2 surplus:  500
Rail 1 deficit:  500
Rail 0 deficit: 1500
```

The two directions therefore contain 4,000 source-forwarded messages.

## BF16

All latency values are milliseconds.

| Phase | LB OFF | LB cold | LB cached |
|---|---:|---:|---:|
| Control plane | 0.000 | 1.781 | 0.570 |
| Reset | 0.272 | 0.398 | 0.379 |
| Source stage | 0.365 | 0.394 | 0.406 |
| Rail transfer | 0.387 | 0.506 | 0.526 |
| Destination forward | 0.789 | 0.882 | 0.907 |
| Dispatch data plane | 1.541 | 1.782 | 1.839 |
| Dispatch total | 1.812 | 3.961 | 2.787 |
| Combine total | 2.887 | 3.543 | 3.167 |

Cached LB relative to OFF:

| Metric | Delta | Change |
|---|---:|---:|
| Source stage | +0.041 ms | +11.2% |
| Rail transfer | +0.140 ms | +36.2% |
| Destination forward | +0.118 ms | +14.9% |
| Dispatch data plane | +0.298 ms | +19.4% |
| Dispatch total | +0.975 ms | +53.8% |
| Combine total | +0.280 ms | +9.7% |

## FP8

All latency values are milliseconds.

| Phase | LB OFF | LB cold | LB cached |
|---|---:|---:|---:|
| Control plane | 0.000 | 1.748 | 0.417 |
| Reset | 0.751 | 0.393 | 0.360 |
| Source stage | 0.392 | 0.408 | 0.381 |
| Rail transfer | 0.249 | 0.483 | 0.452 |
| Destination forward | 0.252 | 0.710 | 0.681 |
| Dispatch data plane | 0.880 | 1.601 | 1.513 |
| Dispatch total | 1.631 | 3.742 | 2.291 |

Cached LB relative to OFF:

| Metric | Delta | Change |
|---|---:|---:|
| Source stage | -0.011 ms | -2.8% |
| Rail transfer | +0.203 ms | +81.6% |
| Destination forward | +0.429 ms | +170.5% |
| Dispatch data plane | +0.633 ms | +72.0% |
| Dispatch total | +0.659 ms | +40.4% |

## Planner

| Operation | BF16 | FP8 |
|---|---:|---:|
| Cold planner CUDA | 0.239 ms | 0.230 ms |
| Cached plan D2D restore | 0.018 ms | 0.015 ms |

The larger control-plane wall time includes explicit loopback phase
synchronization and Gloo barriers. The transport is CUDA IPC/P2P over the
single H200 host, not RDMA/HCA.

## Validation

- BF16 and FP8 dispatch passed for OFF, cold, and cached.
- BF16 combine passed for OFF, cold, and cached.
- OFF loads were exactly `[5000, 6000, 7000, 8000]` in both directions.
- LB loads were exactly `[6500, 6500, 6500, 6500]` in both directions.
- Token count, hidden payload, TopK metadata, final Rank, and expert routing
  were unchanged.

