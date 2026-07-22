# 第 06 章：Dispatch 生产调用链

本章沿实际调用链说明 `x/topk_idx/topk_weights` 如何到达目标专家 GPU，并形成按 expert 分组的 tensor。

入口链：

```text
DLBBuffer.dispatch()                         dlb.py
  -> nvshmem_comm_t.dispatch_moe()           dlb_nvshmem_binding.cpp
     -> prepare_moe_direct()
     -> launch_dlb_nvshmem_rdma()
     -> wait_dlb_nvshmem_rdma_epoch()
     -> count received experts
     -> allocate exact PyTorch outputs
     -> scatter received records
```

## 1. Python 层：校验与所有权

[`DLBBuffer.dispatch()`](../dlb.py) 检查 device、shape、dtype、contiguous、专家数和容量，然后调用 custom class。

Python 不做：

- 遍历 tokens；
- 展开 records；
- 生成 Rail plan；
- pack payload；
- 逐 record 通信。

返回后 Python 只把 C++ tensors 封装成 `DLBHandle/EventOverlap`。

## 2. route count：从 top-k 得到需求

[`prepare_moe_direct()`](../dlb_nvshmem_binding.cpp) 等待待复用 pipeline slot 可用，并清零 `local_demand_row`。

随后启动：

```text
count_moe_routes_kernel
```

每个 `(token, topk_slot)` 是一条逻辑 route，但 demand 只统计该 token 对某
`destination_rank` 的第一次命中：

```text
expert_id        = topk_idx[token, slot]
destination_rank = expert_id / num_local_experts
if first selection of this token for destination_rank:
    atomicAdd(demand[destination_rank], 1)
```

因此：

```text
wire_record_count
  = sum_token(unique destination ranks selected by this token)
  <= num_tokens * topk
```

该 kernel 对越界 expert ID 执行 `atomicAdd(invalid_route_count, 1)`，不把错误
route 计入 demand。binding 在已有的小型计数 D2H 边界一并读取该计数，
若非零则通过 `TORCH_CHECK` 报告越界数量和合法区间。这避免了
错误 route 被静默丢弃；业务仍应保证 `0 <= topk_idx < num_experts`。

## 3. local fcollect：形成源服务器需求矩阵

```cpp
nvshmemx_uint64_fcollect_on_stream(local_team,
                                   local_demand_matrix,
                                   local_demand_row,
                                   world_size,
                                   stream);
```

结果 shape：

```text
[rails_per_server, world_size]
```

所有同服务器 ranks 拿到相同矩阵，所以能独立生成一致的 tile 分配。没有 cluster-wide demand matrix。

## 4. GPU 动态 DLB plan

`launch_dlb_build_dynamic_rail_plan()` 对每个远端服务器执行 floor/ceil 均衡，生成：

```text
flow_rail_counts
rail_transfers
```

设 `C = num_comm_sms / 2`，本 rank 只负责自身 `local_rail` 的
`(S-1)*M*C` 个 channel transfers，但完整 flow map 在所有本服务器 ranks 上一致。

该 kernel 将 DLB 调度规则物化为 GPU-resident flow counts 和 transfer descriptors。

## 5. direct pack

`launch_dlb_pack_moe_direct()` 当前为每个 source token 启动一个 128-thread block。
block 在 Top-K 中遍历唯一 destination Rank，每个目标只写一条 record：

```text
读取 token activation
  -> 收集该目标 Rank 上的 expert/slot/weight
  -> 写 128-byte grouped header
  -> 向量化复制 hidden payload
```

远端 group record 根据 `flow_rail_counts` 选择 Rail，直接写对应 GPU 的
symmetric Rail slot；同服务器 group record 直接写真实目标 GPU 的 repair slot。

record 中保存 `source_rank/token/destination/epoch` 以及最多 8 组
`expert/topk_slot/weight`，所以物理发送顺序不是语义顺序。

## 6. transport 与 repair

`launch_dlb_nvshmem_rdma()` 接手已 pack 的 buffers：

```text
producer stage epoch signals
  -> NVSHMEM put+signal
  -> destination Rail receive
  -> CUDA IPC repair 到真实目标 GPU
  -> completion signals
```

`wait_dlb_nvshmem_rdma_epoch()` 让调用 stream 等待 `repair_ready` event，而不是在 CPU 上轮询所有网络操作。

## 7. 接收 record 计数

repair 完成后，目标 rank 的 repair buffer 中按 source rank 分区保存原始 records。`launch_dlb_count_received_experts()` 扫描 headers，并验证：

```text
record.epoch == current_epoch
record.destination_rank == local_rank
expert_id 属于本 rank 的 local expert 范围
```

合法 record 对本地 expert counter 做 atomic add。

## 8. Host 侧输出 shape 确定

PyTorch tensor 创建需要在 host 侧知道输出维度。当前 binding：

1. 将 `num_local_experts` 个计数和少量 Rail 计数复制到 pinned host memory；
2. 对当前 stream 做一次 synchronize；
3. 在 CPU 上计算 aligned counts 与 prefix offsets；
4. 按真实 record 数创建输出 tensors；
5. 把 offsets 上传 GPU。

回传数据为几十或数百个整数，不包含 records 或 activation。该步骤产生一次 host 同步；最大容量预分配或 device-side shape 管理可消除此边界。

## 9. expert-aligned 输出布局

对每个 local expert：

```text
raw_count[e]     实际收到多少 records
aligned_count[e] align_up(raw_count[e], alignment)
offset[e]        aligned_count 的 prefix sum
```

输出 buffer 按 expert 连续：

```text
[expert 0 records + padding]
[expert 1 records + padding]
...
```

alignment 用于让下游 grouped GEMM/专家 kernel 获得更友好的 tile 边界。padding 不是有效 token，必须结合 `valid_mask` 或 raw counts 处理。

## 10. scatter

binding 创建并清理：

```text
recv_x          [total_aligned_records, hidden]
record_headers  [total_aligned_records, 6]
recv_weights    [total_aligned_records]
valid_mask      [total_aligned_records]
expert offsets/cursors
```

`launch_dlb_scatter_received_experts()` 再扫 repair partitions。每条有效 record：

```text
local_expert = expert_id - local_expert_begin
position = expert_offset[local_expert]
         + atomicAdd(expert_cursor[local_expert], 1)
```

然后复制 header、weight 和 activation，并设 `valid_mask[position] = true`。

不同 source/Rail 的 record 可能以不同顺序占据同一 expert 区间，这是预期行为；header 保留了 combine 恢复关系。

## 11. 返回值的语义

返回值语义：

```text
recv_x           按本地 expert 分组的 activation
recv_expert_idx  与 padded records 对齐的 expert ID view
recv_weights     与 padded records 对齐的 router weight
handle           counts、mask、headers、回程映射与 traffic 统计
event            CUDA completion dependency
```

本地 expert kernel 以 expert offsets/counts 作为输入，无需感知 Rail、source server 或 repair。

## 12. 关于“cached dispatch”

`DLBHandle` 会缓存 combine 必需的路由 tensor/headers，但当前 C++ `dispatch_moe()` 每次仍重新执行：

```text
route count -> local fcollect -> dynamic plan -> direct pack
```

因此 handle 是 combine 所需的回程元数据，不是 dispatch layout cache。即使路由输入保持不变，当前实现仍会重新执行需求计数、服务器内收集、Rail 规划和 direct pack。

## 13. 热路径执行位置

当前阶段映射为：

| 阶段 | 执行位置 |
|---|---|
| 输入校验/API 封装 | Python/C++ |
| route count | CUDA |
| 源服务器需求收集 | NVSHMEM on CUDA stream |
| DLB 动态规划 | CUDA |
| record pack | CUDA |
| Rail transport/repair | CUDA + NVSHMEM |
| expert count/scatter | CUDA |
| 精确 shape 分配 | 少量 pinned D2H + C++/PyTorch |

性能分段应使用 CUDA events 或 profiler，覆盖 CUDA kernels 与 transport；Python wall-clock 不具备 kernel 级归因能力。
