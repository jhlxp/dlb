# 第 04 章：GPU 运行时与同步协议

本章覆盖以下实现文件：

- [`dlb_alltoall_nvshmem.cu`](../src/dlb_alltoall/dlb_alltoall_nvshmem.cu)：runtime 初始化、stream/event 编排、epoch 启动与等待。
- [`dlb_nvlink_runtime.cu`](../src/dlb_alltoall/dlb_nvlink_runtime.cu)：CUDA-IPC epoch signal 的发布与等待。
- [`dlb_rail_transport.cu`](../src/dlb_alltoall/dlb_rail_transport.cu)：Rail put、metadata/progress、receive/repair、credit。

## 1. 三条运行时 stream

runtime 内部创建三条 CUDA stream：

```text
stage stream      等待 producer 数据就绪，发布本机 stage epoch
transport stream  matching Rail 收齐本机 producer 后执行 NVSHMEM 发送
repair stream     接收远端 arrivals、repair，并等待本机所有 producer 完成
```

调用者自己的 PyTorch stream 不被替换。runtime 通过 CUDA event 建立依赖，最终再把 `repair_ready` event 接回调用者 stream。

## 2. direct pack：从路由记录到通信 Buffer

`pack_moe_direct_kernel` 直接写 selected Rail 的 NVSHMEM symmetric peer
buffer，或同服务器真实目标 GPU 的 CUDA-IPC repair buffer。它读取 GPU planner
生成的 `flow_rail_counts`，为每条 group record 选择已分配的 Rail slot。

`stage_stream` 等待 caller stream 的 direct pack 完成，然后发布跨进程 stage epoch；[`dlb_nvlink_runtime.cu`](../src/dlb_alltoall/dlb_nvlink_runtime.cu) 提供 CUDA-IPC epoch 的 publish/wait kernels。

## 3. 本机 stage signal

一条 Rail buffer 可能由本服务器 `M` 个 GPU 进程共同写入：

```text
GPU 0 producer --\
GPU 1 producer ---+--> Rail q 的 CUDA-IPC buffer
...                |
GPU 7 producer --/
```

CUDA event 的作用域限于创建它的进程，不能表示其余 `M - 1` 个本机进程的 pack 已完成。

每个 producer 完成写入后，通过 system-scope atomic 向所有本机 Rail 发布当前 slot epoch。Rail `q` 收齐 `M` 个 producer signal 后读取完整 Rail buffer。

对应 [`dlb_nvlink_runtime.cu`](../src/dlb_alltoall/dlb_nvlink_runtime.cu) 中的 publish/wait kernels。

## 4. symmetric send buffer

每个 Rail 的发送区直接分配在 NVSHMEM symmetric memory 中。初始化阶段用
`nvshmem_ptr()` 取得同服务器 peers 的可访问地址并构建设备指针表，producer
因此可以直接 pack 到 selected Rail 的合法 RMA 源 buffer：

```text
source producer direct pack
  -> selected Rail symmetric send buffer
  -> remote symmetric receive buffer
```

原来的 private Rail staging 到 symmetric buffer 的 D2D copy 已删除。matching
Rail 收齐 stage signals 后，直接通过同号 Rail 的 RNIC/RDMA transport 发送。

## 5. 发送 kernel：固定 channel CTA，warp 并行处理 group

每个 `(remote server, real destination)` group 被连续切成 `C` 个 channel slice。
默认 `num_comm_sms=24`，所以 `C=12`。每个 slice 都有自己的 transfer descriptor、
metadata/progress/credit，不共享完成标志。

[`dlb_internode_rail_kernel()`](../src/dlb_alltoall/dlb_rail_transport.cu) 只启动
`C` 个 sender CTAs，而不是按 descriptor 数启动 block。CTA 固定拥有一个 channel，
其 640 threads 组成 20 个 warps，不同 warp 并行处理该 channel 下的不同 group：

```text
wait credit
  -> publish metadata(offset, bytes)
  -> 分 chunk 执行 blocking putmem_signal_warp
```

生产动态计划中，一个 Rail rank 的 descriptor 数为：

```text
(server_count - 1) * rails_per_server * channel_count
```

四机、每机八条 Rail、12 channels 时，每个 Rail rank 每轮有
`(4 - 1) × 8 × 12 = 288` 个 descriptors，但仍然只有 12 个 sender CTAs。
每个 warp 按 `20 × channel_count` 的步长继续领取后续 descriptor，因此拓扑扩大不会
扩大 CTA 预算。

每个 descriptor 是同一远端服务器、同一 real destination local GPU、同一 channel
的连续 records slice。

## 6. metadata、progress 与 payload

三个对象承担不同职责：

```text
metadata：本轮 payload 在 receive slot 中的 offset 和总 bytes
payload：真正的 records
progress：已有多少连续 bytes 对接收端可见
```

发送使用 blocking fused put+signal：

```cpp
nvshmemx_putmem_signal_warp(...,
                            progress,
                            chunk_bytes,
                            NVSHMEM_SIGNAL_ADD,
                            ...)
```

它表达的因果关系是：接收端观察到 progress 已跨过某个 chunk 末尾时，该 chunk payload 已远端可见。

metadata 与 progress 分离后，接收端可在各 chunk 可见时执行 repair，无需等待整个 transfer 完成。

## 7. 为什么当前发送路径不再调用 `quiet()`

若采用 NBI put，通常需要最后调用 `nvshmem_quiet()`，把一个发送者此前发起的通信
统一收口。当前实现使用 blocking `nvshmemx_putmem_signal_warp()`：每个 chunk
返回时，该 warp 对这个 chunk 的 put+signal 已完成，所以接收端看到的 progress 天然
是准确连续前缀，不再需要尾部的全局 quiet。

这并不改变 quiet 本身的定义。若其他代码使用 NBI 操作，quiet 仍表示等待该 PE 此前
发起的通信完成；其语义不包括：

- 所有 GPU 的 barrier；
- 对端专家计算已经完成；
- CPU 必须参与 polling；
- 与 `fence` 等价。

当前 DLB 的 payload-before-progress 与 chunk 完成边界都由 blocking fused
put+signal 直接定义。

## 8. 接收与 destination repair

[`dlb_receive_and_repair_chunks_kernel`](../src/dlb_alltoall/dlb_rail_transport.cu)
同样只启动 `C` 个 receiver CTAs。CTA 固定拥有一个 channel，不同 warp 处理该
channel 下的不同 arrival descriptor：

```text
等待 metadata
  -> 解码 receive offset / bytes
  -> 对每个 chunk 等待 progress >= chunk_end
  -> 从 symmetric receive buffer copy
  -> 真实目标 GPU 的 repair_buffer[source_rank partition]
```

因此默认配置总共是 12 个 sender CTAs 加 12 个 receiver CTAs。repair 不要求额外
的独立 kernel；channel-owned receive kernel 在 chunk 可见后立即执行目标侧 copy。

完整 slice repair 后执行 system fence，清除 metadata/progress，并把下一次可复用 epoch 的 credit 返回源端。credit 按完整 descriptor/slot 生命周期归还，不是每修复一个 chunk 就归还一次。

## 9. credit 防止覆盖旧数据

pipeline buffer 会循环使用。仅知道“本轮发送完成”还不够，因为接收端可能仍在读取旧 slot。

协议是：

```text
sender:   credit >= arrival_epoch 才能写本次 transfer
receiver: payload repair 完成后，credit = arrival_epoch + 1
```

因此 sender 不会提前覆盖 receiver 尚未消费完的 receive slot。

## 10. pipeline slot 与 slot epoch

默认 pipeline depth 为 2：

```text
epoch 1 -> slot 1, slot_epoch 1
epoch 2 -> slot 0, slot_epoch 1
epoch 3 -> slot 1, slot_epoch 2
epoch 4 -> slot 0, slot_epoch 2
```

slot 决定使用哪份 buffer；单调增长的 slot epoch 区分同一 slot 的不同生命周期，避免上一轮残留 signal 被当成本轮完成。

## 11. 本机 completion signal

每个 producer 可向目标服务器内多张真实 destination GPU 的 repair buffer 写入数据。目标 GPU 必须等待全部 `M` 个 producer 的 completion signal，而非仅等待自身 matching Rail。

每个 producer/Rail 完成本 epoch 的相关本地直写和远端 repair 后，向所有目标 GPU 发布 completion signal。目标 GPU 收齐 `M` 个 producer 后，才记录 `repair_ready` event。

## 12. 调用者看到的异步语义

`wait_dlb_nvshmem_rdma_epoch()` 本身不做 CPU `cudaStreamSynchronize()`，而是让调用者 stream 等待 runtime 的 `repair_ready` event，所以 GPU 计算依赖可以继续排在同一 stream 上。

不过当前 [`dispatch_moe()`](../dlb_nvshmem_binding.cpp) 为了精确分配 PyTorch 输出 tensor，会把紧凑的 expert/rail 计数复制到 pinned host memory，并在这一小段执行一次 stream synchronize。

当前异步边界如下：

- payload 调度、pack、通信、repair、scatter 都在 GPU；
- runtime epoch 等待通过 CUDA event 表达；
- 当前 dispatch 仍有一个“小计数回主机”的同步边界，不是完全 host-asynchronous。

## 13. 完整正确性链

```text
producer 完成 direct pack
  -> system-scope stage signal
  -> matching Rail 收齐 M 个 producer
  -> metadata 发布
  -> blocking fused put 使 chunk payload 远端可见
  -> 同一操作增长该 channel 的 progress
  -> receiver repair chunk
  -> system fence
  -> return credit
  -> M 个 completion signals 到齐
  -> repair_ready event
  -> 调用者 stream 上的 count/scatter/computation
```

不同 Rail、不同 transfer 的先后顺序可以变化。正确性依赖独立 header、epoch、signal、credit 和最终计数，不依赖所有 records 全局有序。
