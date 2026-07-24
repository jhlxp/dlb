# DeepEP V1 与 DeepEP V1 + DLB 性能分析

## 结论

本轮验证全部通过。DLB 能在不改变 token 最终目标 Rank、专家和 TopK
语义的前提下，将每个跨服务器 tile 的四条 Rail 精确均衡到
`max(load) - min(load) = 0`。

生产路径应优先复用 cached plan：

- DLB planner 的纯 CUDA P50 为 `0.142–0.150 ms`。
- 测试后端的 cold control P50 为 `1.488–1.515 ms`，其中包含三个
  Gloo 跨进程 barrier，不能解释为生产环境的 GPU barrier 或 RDMA
  开销。
- cached plan 的 Device-to-Device 恢复耗时仅为 `0.015–0.036 ms`；
  当前测试程序逐阶段同步后观测到的 host wall time 为
  `0.324–0.401 ms`。
- 在不均衡流量下，DLB 将 dispatch 数据面缩短约 `5.6%–9.3%`；
  BF16 combine 缩短约 `7.4%–13.9%`。
- 将测试控制开销也计入后，cached LB 的 dispatch 总 P50 在主要
  不均衡负载下增加 `8.5%–14.0%`。真实 IBGDA/RDMA 收益需要在能够
  打开 HCA 的多机环境中继续验证。

## 测试范围

| 项目 | 配置 |
|---|---:|
| EP | 8 |
| 逻辑服务器 | 2 |
| 每服务器 GPU / Rail | 4 |
| Rail 对 | `0↔4, 1↔5, 2↔6, 3↔7` |
| Global batch | 16 |
| Sequence length | 2048 |
| Global tokens | 32768 |
| 每 Rank tokens | 4096 |
| Hidden | 2048 |
| Experts | 256 |
| TopK | 8 |
| Dtype | BF16 dispatch/combine、FP8 dispatch |
| 路由 | skewed、balanced、extreme |
| 采样 | 每组 20 次预热 × 5 轮，100 次计时 × 5 轮 |

一共有 18 个配置，记录了 9,000 个 dispatch 样本和 4,500 个 BF16
combine 样本。每个样本报告该次迭代中最慢 Rank 的时间。

本轮使用 tests 目录中的 CUDA IPC/P2P loopback Rail backend。GPU
之间实际走 H200 NVLink P2P，但它不是 RDMA/HCA 性能测试。正式
`csrc/` 中的 DLB planner 与生产 DeepEP V1 集成代码被直接复用，
`2 × 4` 拓扑和 loopback transport 没有进入生产源码。

## 三个对照路径

| 名称 | 行为 |
|---|---|
| `lb_off` | 使用原始 source Rail，不执行 DLB planner |
| `lb_cold` | 每次 dispatch 执行 count、build、finalize |
| `lb_cached` | 复用已生成的 plan，每次仅恢复 plan metadata |

`dispatch_total` 的定义为：

```text
control plane
  + reset
  + source stage
  + Rail transfer
  + destination forward
```

`dispatch_data_plane` 不含 control plane 和 reset。表中的数据面吞吐为：

```text
聚合 wire payload bytes / 最慢 Rank 的数据面时间
```

它用于比较同一测试中的实现变化，不等价于单条 NVLink 或 HCA 的物理
利用率。

## Rail 均衡结果

以下数字均为双向聚合后的 plan 统计。所有模式的跨服务器 wire message
总数都保持为 32,768。

| 路由 | LB | 单方向四条 Rail 负载 | 直传 | 源侧转发 | 源侧和目标侧都转发 |
|---|---|---|---:|---:|---:|
| skewed | OFF | `[4096, 3072, 2048, 1024]` | 32768 | 0 | 0 |
| skewed | ON | `[2560, 2560, 2560, 2560]` | 28672 | 4096 | 4096 |
| balanced | OFF/ON | `[4096, 4096, 4096, 4096]` | 32768 | 0 | 0 |
| extreme | OFF | `[4096, 0, 0, 0]` | 32768 | 0 | 0 |
| extreme | ON | `[1024, 1024, 1024, 1024]` | 26624 | 6144 | 6144 |

already-balanced 输入不会产生无意义的源侧转发。skewed 输入只移动
4,096 个 message，extreme 输入只移动达到精确均衡所需的 6,144 个
message，符合“优先保留原 Rail”的设计。

## BF16 性能

单位：延迟为 ms，吞吐为 GB/s。

| 路由 | 路径 | Dispatch P50 | Dispatch P95 | 数据面 P50 | 数据面吞吐 | Combine P50 |
|---|---|---:|---:|---:|---:|---:|
| skewed | OFF | 1.847 | 1.941 | 1.422 | 96.58 | 2.389 |
| skewed | cold | 3.177 | 3.394 | 1.324 | 103.75 | 2.213 |
| skewed | cached | 2.004 | 2.154 | 1.291 | 106.43 | 2.158 |
| balanced | OFF | 1.838 | 1.922 | 1.408 | 97.56 | 2.407 |
| balanced | cold | 3.245 | 3.453 | 1.397 | 98.32 | 2.394 |
| balanced | cached | 2.115 | 2.398 | 1.380 | 99.51 | 2.349 |
| extreme | OFF | 1.798 | 2.000 | 1.381 | 99.46 | 2.355 |
| extreme | cold | 3.107 | 3.258 | 1.258 | 109.15 | 2.038 |
| extreme | cached | 1.982 | 2.119 | 1.257 | 109.30 | 2.027 |

cached LB 相对 OFF：

| 路由 | Dispatch 总延迟 | 数据面延迟 | 数据面吞吐 | Combine |
|---|---:|---:|---:|---:|
| skewed | `+8.5%` | `-9.3%` | `+10.2%` | `-9.7%` |
| balanced | `+15.0%` | `-2.0%` | `+2.0%` | `-2.4%` |
| extreme | `+10.2%` | `-9.0%` | `+9.9%` | `-13.9%` |

balanced 是隔离纯 LB 开销的基准，因为它不需要转发任何 token。这里
数据面基本不变，总延迟差主要来自测试中同步执行的 cached plan 恢复。

## FP8 性能

FP8 每次 dispatch 的聚合 wire payload 为 69 MiB；BF16 为 131 MiB。

| 路由 | 路径 | Dispatch P50 | Dispatch P95 | 数据面 P50 | 数据面吞吐 |
|---|---|---:|---:|---:|---:|
| skewed | OFF | 1.707 | 1.860 | 1.282 | 56.45 |
| skewed | cold | 3.021 | 3.289 | 1.203 | 60.13 |
| skewed | cached | 1.945 | 2.096 | 1.210 | 59.81 |
| balanced | OFF | 1.736 | 1.808 | 1.304 | 55.48 |
| balanced | cold | 3.086 | 3.298 | 1.257 | 57.58 |
| balanced | cached | 1.686 | 2.179 | 1.080 | 66.97 |
| extreme | OFF | 1.717 | 1.829 | 1.287 | 56.24 |
| extreme | cold | 2.998 | 3.343 | 1.174 | 61.65 |
| extreme | cached | 1.951 | 2.067 | 1.213 | 59.65 |

cached LB 在 skewed 和 extreme 下的数据面分别缩短 `5.6%` 和
`5.7%`，总 P50 分别增加 `14.0%` 和 `13.6%`。balanced FP8 的 P50
出现 `-2.8%`，但 P95 从 `1.808 ms` 上升到 `2.179 ms`，应视为阶段
同步和运行波动，不能解释为 LB 对均衡流量有加速作用。

## Planner 开销拆分

cold planner 的三个 CUDA 阶段 P50：

| 阶段 | P50 范围 |
|---|---:|
| count | `0.021–0.023 ms` |
| build | `0.091–0.096 ms` |
| finalize | `0.032–0.033 ms` |
| 合计 | `0.142–0.150 ms` |

planner 的真实 GPU 工作约为 0.15 ms。测试中 cold control 的
1.49–1.52 ms 还包括以下 loopback 协调：

```text
count
  → Gloo barrier
  → build
  → Gloo barrier
  → finalize
  → Gloo barrier
```

生产路径使用已有的 GPU/NVLink barrier，并在 DeepEP comm stream 上
异步执行，不能用 Gloo wall time 推断生产冷启动开销。cached plan 的
Device-to-Device 恢复 CUDA P50 为 `0.015–0.036 ms`，说明 plan
复用本身的 GPU 复制成本很低。

## 生产判断

1. DLB 算法正确，且只移动达到均衡所需的 token。
2. 开启 LB 后，DeepEP V1 的 token→Rank 去重语义、最终专家路由、
   BF16/FP8 dispatch 和 BF16 combine 均正确。
3. 不均衡流量的数据面已经受益；当前端到端增量主要来自测试后端的
   control-plane 同步，而不是 planner kernel 本身。
4. 生产默认策略应为“首次生成 plan，后续 cached dispatch/combine
   复用”，避免每次重新执行带 barrier 的 cold planner。
5. 下一阶段应在真实 HCA 环境测量 fused internode kernel、IBGDA
   Rail 吞吐和 GPU barrier，才能判断 DLB 在真实跨机拥塞下的端到端
   净收益。
