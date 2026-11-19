# alofa GPU 合批时代战报（账本一百三十四 → 一百四十二）

2026-11-13 收束。本阶段把 GPU 服务层从「每流独占前向」推向「行池统一合批」：
任意采样参数、流式/非流式统一调度、HTTP 层收益实测、稳定性验收。

## 一、九刀全景

| 账 | commit | 一句话 |
|---|---|---|
| 134 | `0796897` | 单 session 批行池 + tick 相位混合（row_base 参） |
| 135 | `feaf4fe` | 合批性能基线：per-row 成本平坦，t=8 摊薄 6.63× |
| 136 | `62c9119` | per-row 采样核：合批条件彻底消失（任意 temp/top_p 一批） |
| 137 | `ce47a24` | complete 进批行池：流式/非流式互斥消除 |
| 138 | `b32fec7` | 真前向计数器（n_forwards / n_forward_rows） |
| 139 | `249ed74` | HTTP 并发压测 + /health 五口径观测面 |
| 140 | `f52ad2d` | TCP_NODELAY：一行修复 HTTP 吞吐 +38-46% |
| 141 | `762de6b` | 25 分钟 soak：11016 请求 0 错误、无泄漏 |
| 142 | `ea38412` | 并发爬坡：HTTP 收益峰 conc8 1.66×，loop 是天花板 |

## 二、session 层摊薄曲线（`bench_batch135.mojo`，Qwen2.5-0.5B，q4/f16，GPU3）

| t（行） | 每步 | per-row | 摊薄效率 |
|---|---|---|---|
| 1 | 5.10-5.82 ms | — | 1.00× |
| 2 | 4.27 ms | 2.14 ms | 2.4-2.7× |
| 4 | 4.90 ms | 1.22 ms | 4.2-4.8× |
| 8 | 6.15 ms | 0.77 ms | 6.6-7.6× |
| 16 | 8.58 ms | 0.54 ms | **10.85×** |

**判读**：前向成本 t=1→16 只涨 ~65%——权重读主导贯穿全区间（与 119-123 的
B=16-32 拐点结论同源）。t=16 的 per-row 0.54ms 与 123 的 B=16 0.539ms 吻合：
**合批引擎跑在裸批前向的理论线上**，服务层调度零开销。

## 三、HTTP 层收益曲线（`pressure_139.py`，NODELAY + 帧积压守门，同 run 内取比）

| 并发 | 142（无守门） | 144（固定阈值） | **145（活跃标定阈值）** |
|---|---|---|---|
| solo | 1.00× | 1.00× | 1.00×（66.5 tok/s） |
| conc2 | 1.20× | 1.15× | 1.21× |
| conc4 | 1.51× | 1.26×（误伤） | **1.68×** |
| conc8 | 1.66× | 2.16× | **2.47×** |
| **conc16** | 1.57×（回落） | 3.26× | **3.31×**（219.8 tok/s） |

**曲线单调化**——142 的「conc8 峰、conc16 回落」被守门修正。

**破案与修法**（144-145）：分段口径显示 conc16 平均批宽 9.18 行但**平均活跃
仅 1.33 行/前向**——85% 的行·步在推 padding。根因链：prompt 相位的 next
连发 tick → 全池帧队列无界堆积 → 行「done 未收尾」久占行位 → 活跃度塌陷
→ 低批宽 tick 主导。142 判的「loop 轮转天花板」是此假象。

**守门**：`stream_pending(request)`（Service/Handler trait）——本行无帧可领
且全池待领帧 > 2×活跃行数 → True，loop 先问后领、True 即跳过本轮。恒不死锁
（有帧的行不 pending、积压必有主）；solo 永不触发。阈值从固定 n_rows 改为
2×活跃数（145）：固定阈值在低并发误伤（conc4 的 4 流积压贴着 16 的阈值，
1.51×→1.26×），活跃标定后 conc4 反超基线且高并发收益保留。

**conc16 细节**：forwards 2677→1210（冗余前向 -55%）、row-steps/tok 12.80→
6.94（padding 浪费 -46%）、批宽 11.0 行、lat 2.24s→1.07s。

**chunked prefill 进 tick**（147，`PREFILL_CHUNK_TICKS = 8`）：prompt 相位的
stream_next 至多驱动 8 次 tick，走满无帧回 waiting（上层回 SSE 注释帧）。
切片让 prompt 步与 gen 步在同一 tick 交错、批宽更饱满——协议零新增
（token.waiting + sse_comment 本就存在；complete 对 waiting 幂等；CPU 路不
返回 waiting）。

| 场景（SLO 口径） | 146 | 147 | Δ |
|---|---|---|---|
| **long-c16 TTFT p99** | 8275ms | **2627ms** | **-68%** |
| long-c16 TTFT p50 | 3122ms | 1348ms | -57% |
| **long-c16 吞吐** | 35.7 | **60.7 tok/s** | **+70%** |
| long-c8 TTFT p50 | 1324ms | 574ms | -57% |
| short-c4 / short-c8 吞吐 | — | — | +29-36% |

守门（背压）+ 切片（公平调度）是同一根因的两刀：背压挡「引擎领先过远」，
切片挡「一段 prompt 独占引擎」。

## 四、吞吐口径（/health 五字段）

```
rows / ticks / batched / forwards / row_steps
```

- `forwards` = 真 launch 次数（`prefill_step` 入口计数）——**吞吐口径用这个**
- `ticks` 混空转 tick：间歇负载下噪声可达 6.6×（138 实证）；持续满负荷下
  ticks ≈ forwards（141 soak 实证）
- `row_steps / forwards` = 平均批宽——摊薄证据的直接读数
- 门级断言：两流交替真前向 34 < 串行 sum（135 断言钉死）

## 五、稳定性验收（141 soak，25 分钟混合负载）

- 4 线程 ×（流式 temp0 / 流式 temp0.8 / 非流式 temp0.8 / 非法请求 4xx）
- **11016 成功 / 2754 个预期拒绝 / 0 错误**；每分钟完成数平稳无衰减
- GPU 显存恒定 3767MiB；进程 RSS 4.17GB 合静态预算；fd=54 无累积
- 错误路径自恢复；满负荷零空转 tick

## 六、运维坑清单

1. **serve_gpu nohup 必须带 `LD_LIBRARY_PATH`**——漏则 `CUDA_ERROR_INVALID_VALUE`
   于 GpuService 构造（报错点与真因无关）；boot 最小门（`test_serve_boot.mojo`）
   是此类问题的二分工具。
2. **TCP_NODELAY 必须由调用方设**——flare 明示留给 caller；SSE 小帧不设则
   每帧撞延迟 ACK（~40ms、随并发非线性放大）。
3. **门收集用 `_slot_of(rid)`**——「begin 返回实际行」的正确用法；写死
   `streams[1]` 会收到空闲行。
4. **`ALOFA_GPU_SESSIONS` 上限 64**（=GPU_BMAX）——134 重写后 N=行池容量，
   显存与 N 无关（pool=16 实测显存不变）。

## 七、遗留行动项

1. **KV 按需分配**——830MB 固定 → 按 pos 增长（平均利用率 ~20%）
2. ~~prompt 相位 TTFT~~——147 的 chunked prefill 落地，long-c16 p99 -68%，此项关闭
3. **紧致行压缩**——降级搁置（~30 核签名改动 vs 0.15ms/行边际），行边际
   大涨（长上下文）时再议
4. ~~loop 帧处理成本削减~~——144/145 的守门已把「loop 天花板」证伪并打破，
   此项关闭

## 八、版本

0.2.0：GPU 服务层合批时代。0.1.0 = 119-123 的可服务底座（单流）。
144-147（守门 + 活跃标定 + chunked prefill）为 0.2.0 内的引擎节奏突破：
HTTP 加速比曲线单调化（3.31×@conc16）、长 prompt TTFT p99 -68%。
