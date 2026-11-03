# 方案：KV cache 量化

状态：**未开工**（账本里没有这一行 —— 不是 `missing`，是**从未登记**）。本文只做设计与
判据，不含未测的性能数字。

## 1. 动机要说准：是容量，不是今天的 tok/s

先摆一个今天刚量出来的数，免得这个方案被当成熟度优化：**当前配置下 KV 的访存量小到可以
忽略**。`MAX_SEQ_TOKENS=64`、`kv_dim=128`、24 层 ⇒ 一个 decode step 读 KV
`64 × 128 × 2 × 24 × 4 B ≈ 1.57 MB`，而同一拍要读的权重是 345 MB ⇒ **KV 占 0.5%**。
把 KV 压成 8 bit，今天的 tok/s **不会有可测的变化**。

所以本方案的收益写在两处，且只有这两处：

1. **容量**：同样大的池子能装 **3.76×** 的 token（见 §3），长上下文才进得来；
2. **长上下文时的带宽**：序列拉长后 KV 流量线性涨，那时减半才有意义。

⚠️ 与 `q8_head` 那次是同一个教训的正面版本：**先算清楚要减的字节在总流量里占多少，再决定
动不动手**。q8_head 当时是"少读 5.7× 字节只快 3.2×"引起的追问，这次是反过来 —— 别为 0.5%
去付一份复杂度。

## 2. 现在的接缝（代码事实）

| 位置 | 是什么 |
|---|---|
| `src/alofa/engine/executor.mojo:117-127, 180-184, 301-310` | `block_view(base, block_size, kv_dim)` 建 **fp32** 连续视图 `[MAX_LAYERS*MAX_BLOCKS, block_size, kv_dim]`，步幅 `(block_size*kv_dim, kv_dim, 1)`；`BatchExecutor.kv_store` 持有 |
| `src/alofa/engine/executor.mojo:84-93` | `DEFAULT_BLOCK_SIZE=16`、`MAX_LAYERS=32` |
| `src/alofa/runtime/kv/__init__.mojo:55-76` | `MAX_BLOCKS=112`、`MAX_BLOCKS_PER_SEQ=16`、`MAX_REQUESTS=8`、`MAX_SEQ_TOKENS=64` |
| 当前模型（`src/alofa/cli.mojo:109-124`） | `n_kv_heads=2`、`head_dim=64` ⇒ `kv_dim=128` |
| `src/alofa/kernels/cpu/paged.mojo:208-246` | `paged_scatter(cache, table, first, src)` —— **写** |
| `src/alofa/kernels/cpu/paged.mojo:129-143` | `PagedTable.row_offset(j, kv_cols) = (block*block_size + slot) * kv_cols` —— 地址 |
| `src/alofa/kernels/cpu/paged.mojo:249-336` | `paged_attention[MAX_ENTRIES](...)` —— **读**，fp32 直读，`head//group` 做 GQA，三遍 softmax，Float64 归约；**没有 AVX2 版** |
| `src/alofa/model/arch/qwen.mojo:1113-1155, 1239-1249, 1522-1652` | 非分页那条路：每层每侧连续 `[max_tokens, kv_dim]` fp32，`copy_into` 写入 |
| `src/alofa/runtime/kv/pool.mojo:28-98` | `BlockPool`：refcount + LIFO 空闲栈 |
| `src/alofa/runtime/kv/radix.mojo:63-75, 226-271` | `RadixTree` 数组式，**split 只改元数据**，同一物理块可属两个节点 |

现存量级：单层单侧一块 = `16 × 2 × 64 × 4 = 8192 B`；每侧 29.36 MB，**K/V 合计 58.7 MB**。

## 3. 设计

### 3.1 分块粒度：每 (token, 32 通道) 一块

直接复用 `quant.mojo` 的 `Q8_BLOCK=32` / `Q8_BYTES=34`（2 字节 fp16 缩放 + 32 个 int8 补码）。
`kv_dim=128 = 4 × 32` ⇒ **每 token 每侧 4 块，136 字节**（fp32 是 512 字节）⇒ **3.76×**。
一个物理块（16 token）每侧 `16 × 136 = 2176 B`（原 8192 B），边界干净、无余数。

块内排布按 `[slot][通道组]`：`offset = block_base + slot*136 + g*34`。

### 3.2 硬约束：缩放因子随**物理块**，且**禁止跨 token**

这是本方案唯一真正困难的地方，来自前缀缓存的复用语义：

* `radix.mojo` 的 split 只改元数据 ⇒ 同一物理块会同时挂在两个节点上；
* 因此**块的量化字节与缩放因子只能由"写入它的那一次前向"决定，之后永不重标定**。

⇒ 分块**不得跨 token**（若一个块横跨两个 token 的通道，第二个 token 写入时就要改写共享的
缩放因子，而那一侧的请求此刻可能正在读）。§3.1 的粒度天然满足：一块只属于一个 token。

⚠️ 反例要写进测试：故意做"跨 token 分块"的实现，共享前缀那道门必须抓住它（数值会随请求
顺序变化，而正常情况下必须与顺序无关）。

### 3.3 写：新增 `paged_scatter_q8`

输入仍是 RoPE 之后的 **fp32** K/V（量化发生在写这一刻），按 32 个一组求 `amax`、
`d = amax/127`、四舍五入取整（**不许截断** —— 与 `quantize_q8_0` 同一条规矩，截断会让值
系统偏小半个台阶，带容差的判据放它过去）。

⚠️ 保留 fp32 版并加开关（如 `ALOFA_KV_Q8=0` 退回），两档都要能被门跑到 —— 参照 q4 mub
那次的教训：**开关没接到真入口时，数字会和基线一模一样，光看结果分不出来**。

### 3.4 读：`paged_attention_q8` 融合解量化

第一版**解量化回 fp32 再做点积**（q 侧不量化：当前只有一个 query token，量化收益为负）。

⚠️ 一条本仓库用血换来的规矩：**热路径 helper 只要 `raises` 就不会内联**（`block_scale` 那次
同一个算法 2.27× vs 1.07×）⇒ 解量化 helper 必须 `@always_inline`，并把错误检查挡在循环外。

远期可选：q 也量化成 int8，走 `vpmaddubsw`（修正项 `−128·Σq`，与 q4 mub 的 `−8·Σx` 同构）。
⚠️ 但 `vpmaddubsw` 是 uint8×int8→int16，**255×127×2 = 64770 会溢出 int16** ⇒ 需要把激活压到
7 bit 范围（±63）才安全，代价是激活精度减半。是否做，等第一版量出注意力占比再定 —— 分页
注意力现在**连 AVX2 版都没有**，先别谈整数点积。

### 3.5 容量怎么改

`executor.mojo` 里 KV 按 fp32 定容（`block_view` 的字节计算）⇒ 改成按 136 B/token 定容。
⚠️ `MAX_SEQ_TOKENS=64`、`MAX_BLOCKS_PER_SEQ=16` 是**编译期常量**，改它们会牵动所有定容容器
与"零堆分配"那两道源码门（`test_kv_pool.mojo` / `test_kv_room.mojo` 之外还有源码级扫描）。
⇒ 建议**先只改字节口径、不动容量常量**：先把"同样的池子能装更多"这件事做实，再单独议
扩容量。

## 4. 判据

参照物：`scripts/dump_kv_q8_reference.py`（照 `dump_kv_reference.py` / `dump_paged_reference.py`
的写法，Python **独立另写**，fp64 累加）。

新门 `tests/unit/test_kv_q8.mojo`：

1. **两条路径逐位相等**：分页量化版 vs 连续（非分页）量化版 —— 对齐的是"同一实现的两条寻址
   路径"，不是"量化 vs fp32"。
2. **对照参考的相对误差**：与 fp64 参考比，阈值**实测后定**（先看注意力输出的相对 L2 与
   logits 余弦，不许照抄 q4/q8 的 1% —— 那是权重量化的数，KV 是另一回事）。
3. **三条常驻负向对照**：
   * ① **跨 token 分块**：故意让缩放因子横跨两个 token ⇒ 共享前缀那道门必须红（数值随请求
     顺序变）；
   * ② **忘符号扩展**：q8_0 是**补码**，照 q4 的习惯读会让负 KV 整体跳到 +128 那侧 —— 与
     `test_q8_parity.mojo` 那条负向对照同构；
   * ③ **块步长 34 vs 32**：错步长读到的两字节可能不是合法 fp16 ⇒ 比**字节**，不是比浮点
     （同一条理由：那会崩溃，不是判红）。
4. **既有门必须继续绿**（这是最要紧的一条）：
   * `test_batch_forward.mojo` 的"批 1/2/4/8 与单请求**逐 token 完全相同**" —— 量化是确定性
     的，这条不许松；
   * `test_kv_pool.mojo` / `test_kv_room.mojo` 的 refcount 与复用语义；
   * `tests/unit/test_paged_attention.mojo` 现有的"分页 vs 连续**逐位相等**"会因量化而失效
     ⇒ 判据改为"各自通路内部一致"（fp32 版仍逐位相等，q8 版走第 1 条），**不许删门**。

端到端质量：KV 量化会引入新误差 ⇒ 必须重跑 `test_q4_mub_greedy`（现 0.8672 / 门 0.85）与
`test_q8_head_greedy`。**这两门是"质量还站不站得住"的唯一裁判**，它们绿了才算做完。

## 5. 步骤（有序）

1. `scripts/dump_kv_q8_reference.py`：生成 scatter / attention 两级参照（fp64）。
2. `kernels/cpu/paged.mojo`：加 `paged_scatter_q8` + `paged_attention_q8`，**不动** fp32 版。
3. `engine/executor.mojo`：加量化 KV 视图与开关，字节口径改 136 B/token（**暂不改容量常量**）。
4. `tests/unit/test_kv_q8.mojo`：§4 的 1–3 条 + 既有门回归。
5. 重跑两道端到端质量门（这一步不做完，前面都不算数）。
6. 账本：新增一行"KV cache 量化（q8_0，按 token×32 通道分块）"，状态按门的结果登记；
   在"分页注意力"那条下补一句"已有 fp32 / q8_0 两档"。
7. 非分页那条路（`QwenForward`）：**先不动**，但要写明它仍是 fp32 ⇒ 两条路的 KV 精度不同，
   端到端质量门用的是哪条要说清楚。

## 6. 风险与边界

* **今天量不出收益**（§1）：本方案的验收标准是**容量与质量门**，不是 tok/s。若哪天有人拿
  "KV 量化后 tok/s 没变"来质疑，这就是答案。
* **分页注意力是标量实现** ⇒ 解量化开销可能让它更慢；注意力在短序列下占比很小，但**这条
  要实测**，不许先断言。
* **两条路不一致**（引擎分页量化 / 非分页 fp32）是已知的债务，写进账本而不是藏起来。
* 若将来扩 `MAX_SEQ_TOKENS`：所有定容容器与零堆分配源码门要一起过，那是**另一个改动**，
  不混在本方案里。
