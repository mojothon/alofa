# alofa

project version: 0.2.0。

## 0.2.0（2026-11-13，账本一百三十四 → 一百四十二）

GPU 服务层合批时代：单 session 批行池 + tick 统一调度；per-row 采样核
（任意 temp/top_p 一批）；complete 与流式统一进池；HTTP 层实测收益峰
conc8 1.66×（session 层 t=16 摊薄 10.85×）；TCP_NODELAY 修复（+38-46%）；
25 分钟 soak 零失败无泄漏。战报：`docs/plan/batching-report.md`。

## 0.1.0（2026-11-12，账本一百一十九 → 一百二十三）

GPU 推理可服务底座：tokenizer + prefill/decode + 采样 + session 池。
战报：`docs/plan/serving-report.md`。