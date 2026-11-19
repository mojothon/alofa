# alofa

project version: 0.2.1。

## 0.2.1（2026-11-19，账本一百四十四 → 一百四十九）

引擎节奏突破：**帧积压守门**（tick 引擎背压；conc16 假天花板证伪）+ 守门
阈值活跃标定（低并发误伤消除，加速比曲线完全单调 1.21/1.68/2.47/3.31×）
+ **chunked prefill 进 tick**（prompt 步切片混合 gen）+ 切片参数扫定版
K=1——长 prompt c16 TTFT p50 **3.1s→0.38s（-88%）**、p99 8.3s→2.7s
（-67%）、吞吐 60.7→68.4 tok/s；守门后与切片后 soak 双复跑零错误。
战报：`docs/plan/batching-report.md`。

## 0.2.0（2026-11-13，账本一百三十四 → 一百四十二）

GPU 服务层合批时代：单 session 批行池 + tick 统一调度；per-row 采样核
（任意 temp/top_p 一批）；complete 与流式统一进池；HTTP 层实测收益峰
conc8 1.66×（session 层 t=16 摊薄 10.85×）；TCP_NODELAY 修复（+38-46%）；
25 分钟 soak 零失败无泄漏。
战报：`docs/plan/batching-report.md`。

## 0.1.0（2026-11-12，账本一百一十九 → 一百二十三）

GPU 推理可服务底座：tokenizer + prefill/decode + 采样 + session 池。
战报：`docs/plan/serving-report.md`。