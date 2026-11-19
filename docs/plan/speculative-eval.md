# 批量 decode 路线评估：speculative decoding 在 alofa 的适用性（一百五十二）

结论先行：**合批高并发段（conc8+）净亏，不做；低并发段（bs=1-4）有甜点，
self-speculative（prompt-lookup）可做小规模试验。**

## 一、为什么经典 speculative 在本引擎高并发段净亏

speculative 的经典收益场景是**内存带宽瓶颈**（bs=1：每前向算一个 token，
GEMM 大量时间在等权重搬运）——draft 换来「一次前向出多个 token」。

本引擎在 151 已证 conc32 前向 **100% 打满**（5.6ms/前向，5.6 tok/前向，
批宽 86%）——**计算瓶颈**。此时 speculative 的账：

- 验证前向要并行验证 k 个 draft 位置：GEMM 按位置数线性（27 行 × k），
  attention 因序列长度 +k 几乎不变（O(n²) 中 n 微增）——每验证前向
  成本 ≈ 单 token 前向的 2-3 倍（k=4）
- 净收益 = 接受 token 数 ÷ 验证成本倍数。睡前故事类文本 self-speculative
  接受率 ~30-50% → k=4 接受 ~1.6 tok ÷ 2.5× = **0.64——净亏 36%**

文献共识与此一致：speculative 在大 batch 计算瓶颈下收益急剧缩水甚至为负。
**合批（每前向更多行）与 speculative（每前向更多 token/行）争夺同一资源
——前向时间**；前者已经赢了（6.55×），后者在高并发段没有剩余空间。

## 二、低并发段的甜点

bs=1-4 时前向远未打满（solo 批宽 1.00、每 token 15ms = 3 tick）——
**内存带宽瓶颈区**，正是 speculative 的主场：

- **self-speculative（prompt-lookup）**：从 prompt / 已生成文本查 n-gram
  作 draft——零额外模型、零 draft 前向、零 KV 回滚成本（查表即可）。
  对模板化/重复模式文本（代码、README、固定话术）接受率可达 60-80%
- 收益估算：bs=1 接受 70%（k=4）→ 2.8 tok / ~2× 成本 = **1.4× 单流加速**
  ——把 solo 段从 1.00× 抬到 ~1.4×，低并发段的曲线断点被填平
- 工程量：tick 引擎需支持**每行变长步进**（一次前向写 k 个位置的 KV、
  按接受数回滚 pos）——KV 布局、pos 管理、帧序号全要动——**中等偏重**
  （~2 个账本次的改造 + 门大改）

## 三、其它路线速评

| 路线 | 判定 |
|---|---|
| 独立 draft 模型 | ✗ Qwen2.5-0.5B 已是家族最小，无更小 draft；加载体积翻倍 |
| Medusa 式多头 | ✗ 需训练管线，无 |
| 多 token 步进（无验证的贪心外推） | ✗ 无验证 = 输出不再是模型分布，破坏逐位同的门 |
| prompt-lookup（self-speculative） | △ 仅低并发段，见第二节 |
| 连续批处理（continuous batching） | ✓ 已实现（tick 引擎天然逐 tick 重组批） |

## 四、决策

1. **高并发段（≥conc8）**：不做。前向打满，speculative 无空间。151 的
   饱和曲线就是终局——除非换更快的 kernel（前向 5.6ms 的 kernel 级优化
   是另一条路，与 speculative 正交）
2. **低并发段（bs≤4）**：可做 prompt-lookup 试验——但工程量中等偏重，
   收益面窄（单流/小并发场景），**优先级低于 KV 按需分配**（后者收益面
   覆盖全部部署形态）
3. 前向 kernel 优化（5.6ms 里 GEMM/attention 的占比剖析）是吞吐侧的
   下一个真实入口——值得先做一次 profile 而不是直接改

**下一步建议**：① profile 前向 kernel 构成（~1 次跑）；② KV 按需分配；
③ 低并发段 prompt-lookup 试验（若 ①② 仍有余力）。
