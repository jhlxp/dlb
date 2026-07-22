# 第 05 章：生产 API 与初始化

当前面向业务的入口是 [`dlb.py`](../dlb.py) 中的 `DLBBuffer`。Python 负责 API 契约、对象生命周期和结果封装；逐 record 的 route 计数、Rail 规划、pack、通信、scatter/combine 均在 C++/CUDA/NVSHMEM 热路径中执行。本章默认部署是四台物理服务器、每台八张 GPU 的 EP32 communicator。

## 1. 对外对象

### 1.1 `DLBBufferSizeHint`

该对象由最大 MoE shape 推导，包含：

```text
record_bytes
receive_slot_bytes
repair_slot_bytes
rail_send_capacity_bytes
estimated_runtime_bytes
```

它用于推导 record、pipeline slot、Rail buffer 和 repair buffer 的最坏空间。

### 1.2 `DLBBuffer`

业务持有的 communicator/runtime 对象，主要方法：

```text
post_dispatch(...)
DLBDispatchTicket.finish()
combine(...)
close()
```

不存在 `DLBBuffer.dispatch()` 兼容接口。若业务不做 overlap，仍应显式写成 `ticket = post_dispatch(...); ticket.finish()`；这只是立即完成同一套 split-phase 数据面，不是第二套底层实现。

对象内部保存 C++ custom class communicator、调用方声明的 rank/world/逻辑服务器配置和生命周期状态。

### 1.3 `DLBDispatchTicket`

`post_dispatch()` 返回的短生命周期对象。它持有本轮 `x/topk_idx/topk_weights`，防止 post 阶段已经入队的 GPU work 在 `finish()` 前看到被释放的输入。一个 `DLBBuffer` 只允许一个未 finish 的 ticket；重复 post、在 ticket 未 finish 时 combine，或 close 都会 fail-fast。

### 1.4 `DLBHandle`

一次 dispatch 的返回句柄，保存 combine 恢复原 token 顺序所需信息，例如：

```text
recv_headers / valid_mask
recv_source_rank / recv_source_token
recv_expert_idx / recv_topk_slot / recv_topk_weights
原始 topk_idx / topk_weights
expert counts 和 Rail traffic counts
```

业务不应自行构造或修改 handle；它在语义上属于产生它的 `DLBBuffer` 和对应 dispatch。当前 `combine()` 只检查专家数、Top-K shape 与 CUDA tensor 形状，尚未在 handle 内保存并 fail-fast 校验 communicator identity 或 dispatch epoch；生产调用方不得跨 buffer、跨 rank 或跨批次混用 handle。

### 1.5 `EventOverlap`

包装 CUDA event，用于让调用方把后续 GPU 工作挂到通信完成依赖上。它表达 CUDA 执行依赖，不是 CPU 侧全局 barrier。

## 2. Python 到共享库的连接

[`dlb_nvshmem_utils.py`](../dlb_nvshmem_utils.py) 仅负责以下运行时装载与辅助功能：

1. 定位并加载 `libdlb_nvshmem.so`；
2. 获取 `torch.classes.dlb_classes.nvshmem_comm_t`；
3. 暴露 NVSHMEM unique-id 与环境 probe 辅助函数。

调度、pack 和通信不在 Python 中执行。

C++ 注册位于 [`dlb_nvshmem_binding.cpp`](../dlb_nvshmem_binding.cpp) 末尾：

```text
torch.classes.dlb_classes.nvshmem_comm_t
  .post_dispatch_moe(...)
  .finish_dispatch_moe(...)
  .combine_moe(...)
  .close()
```

custom class 还注册了 `benchmark_prepare_moe_device()`，仅供非传输热路径基准使用，不是模型业务 API。

## 3. 多进程初始化

每个 rank 通常对应一个进程和一张 GPU。当前 `DLBBuffer` 由 EP group rank 0 创建 128-byte NVSHMEM unique-ID CPU tensor，再通过 `torch.distributed.broadcast()` 发给其余 ranks。

随后每个进程构造 communicator：

```text
rank / world_size
rails_per_server = M
server_count = world_size / M
source_server = rank / M
local_rail = rank % M
```

这是软件约定而不是硬件发现。`gpus_per_server=M` 同时决定 local NVSHMEM team、CUDA IPC peer table 和 DLB Rail 索引；部署层必须保证连续 `M` 个 process-group ranks 确实位于同一物理服务器，且 local rank 与 GPU/NIC Rail 的绑定符合约定。

[`dlb_nvshmem_comm_t`](../dlb_nvshmem_binding.cpp) 用 unique ID 初始化 NVSHMEM，并通过 strided team split 为每台物理服务器建立只包含本机 `M` 个 ranks 的 local team。

## 4. local NVSHMEM team 的作用

DLB 只需要收集本服务器的源端需求：

```text
M 个 local demand rows
  -> local-team fcollect
  -> M × world_size local demand matrix
```

world team all-gather 会收集全集群需求矩阵，不满足 source-server-local 调度的输入范围。

## 5. communicator 初始化了什么

C++ 构造函数主要建立四类状态。

### 5.1 调度状态

```text
local_demand_row      [world_size]
local_demand_matrix   [M * world_size]
flow_rail_counts      [(S-1) * M * M * M]
destination_cursors
rail_record_counts    [M]
channel_record_counts [num_comm_sms / 2]
invalid_route_count   [1]
```

### 5.2 transfer descriptors

设 `C = num_comm_sms / 2`，每个 rank 每轮固定物化：

```text
(S - 1) * M * C
```

个 Rail transfers。每个 descriptor 对应一个
`(remote_server, final_destination_local, channel)` slice；零记录 slice 也保留 descriptor，
使接收端可以完成该 arrival 的 metadata/credit 状态转换。

### 5.3 runtime buffers

```text
按 pipeline slot 分区的 symmetric Rail send buffer
symmetric receive buffer
repair buffers
metadata/progress/credit
stage/completion signals
CUDA streams/events
```

### 5.4 CPU 小型回传区

为基于实际接收量创建精确 shape 的 PyTorch 输出 tensor，runtime 使用 pinned host
小型回传区读取 expert、Rail、channel、group 和 invalid-route 计数。该边界不承载
record 或 activation payload；aligned counts 和 expert offsets 随后在 CPU 上计算。

## 6. CUDA IPC 指针表

同一台物理服务器内，每个进程各自创建本 GPU 的：

```text
NVSHMEM symmetric rail_send_buffer[slot]
repair_buffer
stage/completion signal
```

Rail send buffer 通过 `nvshmem_ptr()` 构建同机 peer pointer table；repair buffer
和 stage/completion signal 则交换 CUDA IPC handles。这样一个 GPU kernel 可以
直接写本机另一进程所拥有的 GPU buffer。

该指针表支持 direct pack、same-server direct write 和 destination repair，且不经过 CPU memcpy。

## 7. Buffer 容量公式

[`DLBBuffer`](../dlb.py) 按下式计算：

```text
record_bytes = align_up(128 + hidden * dtype_size, 16)
slot_records = num_max_tokens_per_rank * num_topk
slot_bytes   = align_up(slot_records * record_bytes, 16)
```

再根据 `server_count`、pipeline depth、send/receive/repair 分区估算 runtime bytes。

运行时不在热路径自动扩容。`num_tokens * topk` 必须不超过初始化 hint，并应按训练或推理的最大 micro-batch 配置容量。

构造参数还有以下硬约束：

- `1 <= num_topk <= 8`；
- `pipeline_depth > 0` 且 `chunk_bytes > 0`；
- `num_comm_sms` 必须是正偶数且不能超过当前 GPU 的 SM 数，channel 数为其一半；
- `world_size` 必须能被 `gpus_per_server` 整除，当前 GPU planner 支持
  `gpus_per_server <= 8`；
- 生产跨服务器路径使用 `transport_backend="nvshmem"`；`"loopback"` 只用于显式
  数据面验证，不应被生产配置隐式选中。

## 8. `post_dispatch()` / `finish()` 的输入契约

核心输入为：

```text
x            [num_tokens, hidden]，CUDA，连续
topk_idx     [num_tokens, num_topk]，专家 ID
topk_weights [num_tokens, num_topk]，转为 FP32
num_experts  来自 DLBBuffer 初始化配置
expert_alignment > 0
```

`post_dispatch()` 还接受可选的 `handle=`。传入时必须省略 `topk_idx`，Python 会复用该 handle 保存的 `topk_idx`，并复用或覆盖其 `topk_weights`。这是**复用 router 决策**的便利接口，不会复用上次的 demand matrix、Rail plan、record offsets 或 transport slot；当前调用仍会重新执行 count、local fcollect、动态规划和 direct pack。

当前设计假定专家均匀分布：

```text
num_local_experts = num_experts / world_size
destination_rank  = expert_id / num_local_experts
```

因此 `num_experts` 必须能被 `world_size` 整除。expert ID 的范围在 GPU route-count kernel 中统计，并在**当前 rank** 的 `finish()` 小型计数 D2H 边界 fail-fast；`post_dispatch()` 本身可以先返回，不能把它当作 expert-ID 已验证完成的同步检查点。该计数没有跨 rank 汇总，生产训练框架仍应负责把一个 rank 的异常升级为整组失败/退出。

当前生产调用还应保证每个 Rank 的 `num_tokens > 0`，并避免形成完全为空的本地
combine 输入。底层部分 launcher 会把空 tensor 的空指针或
`output_record_count == 0` 视为非法；完整的零 token/零接收 fast path 尚未实现。

## 9. `combine()` 的输入契约

业务先对 `ticket.finish()` 输出的 local expert records 执行专家计算，再调用：

```text
combine(expert_output, handle)
```

combine 根据 handle 中保存的 group-to-expert mapping 和每条 record 的原始：

```text
source_rank
source_token_index
topk_slot
topk_weight
```

先在当前 expert Rank 内对同一 source token 的 expert 结果做 FP32 加权和，
每个 `(source_rank, source_token, current_rank)` 只回传一条 hidden。source rank
再累加不同 expert Ranks 的局部和，恢复原 token 顺序。

`apply_router_weights=True` 是默认语义。设为 `False` 时，expert rank 内的局部和改为不乘 router weight，但返回的 `combined_weights` 仍从 record header 回填传给本次 `combine()` 的 weight，供上层自行处理。

## 10. 生命周期与 `close()`

关闭 communicator 涉及 NVSHMEM team、对称内存和跨 rank 状态，是 collective 生命周期动作。业务应让所有 ranks 在不再有未 finish 的 ticket、且不再依赖已提交 combine 输出后显式调用 `close()`。C++ close 会执行 device synchronize 后释放 runtime 资源，不能把它放入正常层间热路径。

析构函数不偷偷执行可能阻塞的 collective finalize，避免 Python GC 时某个 rank 单独析构导致死锁。

## 11. 生产调用骨架

```python
# 每个 rank 已绑定自己的 CUDA device，torch.distributed 已初始化
buffer = DLBBuffer(
    dist.group.WORLD,
    gpus_per_server=8,
    num_max_tokens_per_rank=max_tokens,
    hidden=7168,
    num_experts=256,
    num_topk=8,
    dtype=torch.bfloat16,
    num_comm_sms=24,
)

ticket = buffer.post_dispatch(x, topk_idx, topk_weights, expert_alignment=128)

# 可以在这里执行不读取本批 recv_x 的独立计算。
recv_x, recv_expert_idx, recv_weights, handle, event = ticket.finish()
event.current_stream_wait()

# counts 位于 handle，按 expert 分段执行本地计算
expert_y = local_moe(recv_x, handle.num_recv_tokens_per_expert_list)

combined_x, combined_weights, event = buffer.combine(expert_y, handle)
event.current_stream_wait()

buffer.close()
```

`finish()` 为精确 shape 会有一次小型 D2H 同步，但 scatter 仍异步排在 caller stream；因此 `event.current_stream_wait()` 仍是 expert kernel 读取 `recv_x` 前的正确依赖表达。

第 06 章按 `post_dispatch() -> ticket.finish()` 调用链说明这些操作对应的 kernels。
