# 第 07 章：接收布局与 Combine

Dispatch 的目标是把 records 按目标 expert 聚起来；Combine 的目标是先在 expert 所在 Rank 内聚合属于同一 source token 的输出，再把每个目标 Rank 的局部和送回原 source rank。

两者通过 dispatch 保存的 record header 连接。

## 1. Dispatch 输出布局

四机 EP32、256 routed experts 时，每个 rank 持有 8 个 local experts：

```text
recv_x =
  local expert slot 0 的有效 records + padding
  local expert slot 1 的有效 records + padding
  ...
  local expert slot 7 的有效 records + padding
```

辅助数据：

```text
recv_counts[e]      实际有效 record 数
aligned_counts[e]   加 padding 后的长度
expert_offsets[e]   expert 分段起点
valid_mask[i]       该位置是否是有效 record
record_headers[i]   原 source/token/topk/expert/destination/epoch
recv_weights[i]     原 router weight
group_output_indices[g, 8]  一条 dispatch group 展开后对应的 expert rows
```

padding 位置不能送回 source，也不能参与有效 token 计数。

## 2. 本地专家计算接口

DLB 本身不实现具体 MLP。业务可以使用 grouped GEMM、Triton kernel 或其他 expert runtime：

```text
expert_y = local_expert_forward(
    recv_x,
    recv_counts / expert_offsets,
    expert_weights
)
```

要求是保持输出行与输入 record 行对应。expert 内不必按 source token 排序。

生产部署中，该接口连接 grouped GEMM 或专家 MLP runtime。DLB 提供 expert 连续分段、计数、padding mask 和回程 metadata，不约束专家 kernel 实现。`ticket.finish()` 返回的 `EventOverlap` 是读取 `recv_x` 的 CUDA 依赖；host 已从小型 count D2H 得到分段长度，不代表 scatter 已同步完成。

## 3. Header 与返回寻址

每条有效输出仍关联 dispatch header：

```text
source_rank
source_token_index
topk_slot
expert_id
epoch
```

Combine 不依赖 record 的到达 Rail；返回目标由 `source_rank` 确定，累加位置由 `source_token_index` 确定。`topk_slot` 与 weight 仍保留在 grouped header 中，用于恢复 router weights 和验证 route 守恒。

## 4. Combine 的生产调用链

入口：[`combine_moe()`](../dlb_nvshmem_binding.cpp)

```text
expert_y + dispatch headers + group_output_indices
  -> 按 dispatch group 计数 return routes
  -> local-team fcollect
  -> dynamic DLB Rail plan
  -> 本 Rank FP32 weighted sum + pack_combine_direct
  -> NVSHMEM transport/repair
  -> 按 destination Rank 的局部和 accumulate 到原 token
  -> cast to activation dtype
```

返回路径重新执行 source-server-local DLB 调度，不复用 dispatch 的物理路径。

## 5. return route count

`launch_dlb_count_combine_routes()` 遍历 `dispatch_group_output_indices`：

- 每个 group 对应一个 `(source_rank, source_token, current_rank)`；
- 一个 group 内最多包含 8 个有效 expert rows，padding 不会成为独立 route；
- 读取 group header 中的 `source_rank` 作为回程 destination rank；
- 每个 group 只给 destination demand 增加 1。

随后与 dispatch 相同，在本服务器 local team 内 fcollect，并运行 GPU 动态 Rail planner。

## 6. `pack_combine_direct`

Combine pack 以一个 CUDA block 处理一个 dispatch group，读取：

```text
group 对应的 1..8 个 expert_y rows
record headers
router weights
```

同一 block 先将这些 expert 输出按 router weight 做 FP32 局部求和，然后只生成一条 hidden record，并把 `source_rank` 当作真实 destination。若 source rank 在同一台物理服务器，直接写其 repair buffer；否则按新的 DLB plan 选择 Rail。

返回 record 仍使用同一固定大小布局，这让 transport/runtime 无需区分 dispatch payload 和 combine payload。

## 7. 返回后的累加

source rank 收齐 repair records 后，`launch_dlb_accumulate_combined_records()` 验证：

```text
epoch
destination_rank == current rank
token index 范围
group 中 selection count 和 topk slot 范围
```

随后：

```text
combined_fp32[token, h] += grouped_weighted_sum[record, h]
returned_weight[token, each_topk_slot] = each_topk_weight
```

group 内局部求和和 source 端跨目标 Rank 累加都使用 FP32。所有 records 完成后再统一 cast 回 activation dtype，减少逐次 BF16 累加误差。

`DLBBuffer.combine(..., apply_router_weights=True)` 默认在 expert rank 内执行这一步 router-weighted 局部求和。若显式设为 `False`，payload 改为未加权局部和；header 中传给本次 `combine()` 的 weight 仍会回填到返回的 `combined_weights`，调用方可在外部实现自己的加权规则。

## 8. FP32 atomic accumulation

一个 token 的 `topk` 条专家结果会先在各目标 Rank 内局部聚合，但不同目标 Rank 的 grouped results 仍可能经过不同 Rails/transfers，返回先后不确定：

```text
token t <- rank 1 的本地 expert 加权和
        <- rank 7 的本地 expert 加权和
        <- rank 12 的本地 expert 加权和
        ...
```

多个 CUDA blocks 可能同时更新 `combined_fp32[token]`，所以仍需要 atomic。但 atomic 更新次数已从“每 expert 一次”减少到“每命中的目标 Rank 一次”。

## 9. 数值确定性

逻辑结果与到达顺序无关，但浮点加法不满足严格结合律。不同执行时序可能造成 FP32 末位差异，因此测试应使用数值容差，而不是要求逐 bit 一致。

当前正确性语义是：

```text
每条有效 route 恰好在某个 grouped return 中贡献一次
每条 route 使用正确 router weight
每个 token 收到其 top-k 专家结果之和
```

## 10. Dispatch 与 Combine 的对称关系

| Dispatch | Combine |
|---|---|
| destination 来自 `expert_id` | destination 来自 header `source_rank` |
| 输入按 token 排列 | 输入按 local expert 排列 |
| 输出按 local expert 分组 | 输出恢复为 source token 顺序 |
| scatter 到 expert offsets | atomic accumulate 到 token index |
| 保存 header/weight 和 group-to-expert mapping | 按 group 局部聚合后消费 header/weight |

两者共享同一 DLB planner、Rail buffer、NVSHMEM transport 和 repair 协议，但 pack/scatter 语义不同。

## 11. 业务侧必须保证什么

- expert output 第一维与 dispatch 输出布局一致；
- 不删除、重排 records，除非同步重排 handle 中对应 headers/mask/weights；
- padding 行不能标成有效；
- handle 不能跨 communicator 或错误 epoch 使用；
- combine 前不能提前复用仍属于该 pipeline epoch 的业务 buffer。

`DLBHandle` 保存该批 expert records 的回程路由状态，以及 `dispatch_group_output_indices`，用它把 expert-expanded 输出重新归并到 rank-deduplicated return records。

最后一条是业务契约，也是当前实现仍需由调用方遵守的边界：`combine()` 校验 `num_experts`、Top-K shape、device、dtype 和 tensor shape，但 handle 中尚未编码 communicator identity/epoch 并做完整 fail-fast 检查。生产封装层应把 handle 视为不可跨 batch、不可跨 `DLBBuffer` 的线性资源。
