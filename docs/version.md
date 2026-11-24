# alofa

project version: 0.3.0。

## 0.3.0（2026-11-24，账本一百五十九 → 一百七十六）

**投机解码 + KV 前缀复用时代**：174 投机解码（n-gram 草稿 + 批内验证 + 自适应
预算，贪心重复率全谱系 1.6-2.7×）；175 采样验证打通温度轴（per-row 采样核，
temp 0.7 × R 全谱系 1.19-2.29×，最差格仍为正——生产矩阵三轴 × 交叉格闭合）；
176 KV 前缀复用（KV 行位置决定 → 拷贝即复用：精确匹配 + chunk 链 LCP 部分前
缀，重复流量 spec 之上 +3.0× 合计 ~6.8×；并发坍缩救回 c=32 1.9× 吞吐触顶
~4100 tok/s；**TTFT p50 243ms → 22.3ms**）。GPU 门 23 个全绿；战报：
`docs/plan/capability-ledger.md` 174-176 各条。


## 0.2.2（2026-11-20，账本一百五十 → 一百五十八）

**发射剖析与减发射时代**：profile 基建（ALOFA_PROFILE 分段 10 槽位）+ 发射
开销剖析（tick=launch-bound：~760 发射 × 7.4µs 自洽，GPU 本体被流水线隐藏，
153 的 rmsnorm 30% 系段语义误读）+ **减发射两刀**——rmsnorm 直出 f16 镜像
（norm1/norm2 后独立 x_to_f16 删除）、VT2/dual_pk 尾写直出 f16（o/down 链
x_to_f16 条件删除），每层发射 27→23；conc32 **433.6→453.8 tok/s（+4.7%）**、
加速比 6.55× 保持、GPU 门十判据全绿。kv2 A/B 持平归档（cat2 每层重拼吃掉
收益）；CUDA graph probe 归档（AsyncRT 有图 C ABI 但 public 入口不可达 +
动态参数冲突）；profile=2 观测者效应 -15% 教训；soak 25 分钟复验零错误无
泄漏。战报：`docs/plan/batching-report.md`。

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