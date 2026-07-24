# Model-Routed TopK=8 Result

## Workload

- Global batch: 16
- Sequence length: 2048
- Global tokens: 32,768
- Tokens per Rank: 4,096
- Hidden: 2,048
- Experts: 256
- TopK: 8
- EP: 8
- Logical topology: 2 servers × 4 Rails

This run does not assign eight random expert IDs directly. It creates a fixed
router matrix, calculates the 256 expert logits for every token, and applies
`torch.topk(logits, k=8)`. The router seed and input are fixed so OFF, cold,
and cached runs use exactly the same routes and the result is reproducible.
Router computation is outside the communication timing.

## Routing Result

The generated routes were already source-Rail balanced:

```text
server 0 → server 1: [4096, 4096, 4096, 4096]
server 1 → server 0: [4096, 4096, 4096, 4096]
```

DLB correctly preserved this mapping:

- Direct messages: 65,536
- Source-forwarded messages: 0
- Source-and-destination-forwarded messages: 0
- Maximum remote Rail load difference after LB: 0

TopK=8 frequently selects experts on both logical servers. DeepEP keeps its
token-to-Rank deduplication semantics, while DLB balances one token/server
message rather than copying the hidden state once per selected expert.

## Performance

All values are the slowest-Rank P50 unless otherwise stated.

| Dtype | Path | Dispatch P50 | Dispatch P95 | Data Plane P50 | Planner CUDA | Cached Restore CUDA |
|---|---|---:|---:|---:|---:|---:|
| BF16 | OFF | 1.792 ms | 1.899 ms | 1.351 ms | 0 | 0 |
| BF16 | cold | 2.999 ms | 3.330 ms | 1.317 ms | 0.144 ms | 0 |
| BF16 | cached | 2.157 ms | 2.438 ms | 1.374 ms | 0 | 0.015 ms |
| FP8 | OFF | 1.619 ms | 1.865 ms | 1.230 ms | 0 | 0 |
| FP8 | cold | 2.946 ms | 3.197 ms | 1.214 ms | 0.143 ms | 0 |
| FP8 | cached | 1.979 ms | 2.085 ms | 1.247 ms | 0 | 0.016 ms |

BF16 combine P50 was `1.191 ms` with LB disabled and `1.175 ms` with the
cached plan, which is within normal run variation for a plan that moves no
messages.

Because this route is already balanced, it provides no communication benefit
that can offset control-plane work:

- BF16 cached dispatch P50: `+0.365 ms` (`+20.4%`)
- FP8 cached dispatch P50: `+0.360 ms` (`+22.2%`)
- BF16 cached data plane: `+1.7%`
- FP8 cached data plane: `+1.3%`

The cached plan's actual Device-to-Device CUDA copy is only about `15 µs`.
The larger `0.38–0.43 ms` host wall contribution comes from the loopback
benchmark's explicit phase synchronization. Likewise, the cold control path
contains Gloo process barriers and is not a measurement of production
GPU/NVLink barriers or RDMA/HCA transport.

## Interpretation

This result verifies the required fast behavior for an already balanced
model-generated route: DLB performs no source forwarding and leaves the data
plane effectively unchanged. It also shows why production policy should skip
or reuse DLB planning when the route is already balanced.

The controlled skewed and extreme workloads remain necessary to measure the
benefit side of DLB. A uniformly initialized router tends to distribute
experts evenly and therefore does not reproduce the learned expert hot spots
that DLB is intended to correct.

