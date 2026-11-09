# 方案：DeviceContext 穿进 model 层（P5 接线）

状态：设计稿（2026-11-09，账本五十九之后）。本文回答一个问题：**已经过七项差分门的 GPU kernel，怎么接到 `QwenForward` 的执行路径上** —— 以及为什么不是把 `DeviceContext` 塞进模型结构体。

## 1. 动机

P5 已有（账本五十一~五十九）：七个 kernel/算子的 GPU 实现全部通过差分门；decode 关键路性能实测 matmul 1.12ms + attention 0.23ms ≈ 1.4ms/token（A100）。但这些 kernel 只活在 `tests/gpu/` 的基准与门里，`alofa serve` 一行都没用到。缺的是**接线**：所有权、数据驻留、调用面。

## 2. 现在的接缝（代码事实）

- **分发是编译期 Int 参数**：`comptime BACKEND_SCALAR=0 / BACKEND_AVX2=1`（`qwen.mojo:160`），各 `*_k[backend]` 包装函数用 `comptime if uses_vector_backend[backend]()` 切换（`qwen.mojo:194/:204/:273`）。后端是**方法的参数**而非实例属性（`QwenForward` 文档原话，`qwen.mojo:1550`）。
- **唯一前向路径**：`QwenForward.run[backend]`（`qwen.mojo:2101`），prefill/decode 共用（`step`/`prefill` 是薄包装）；层循环 `qwen.mojo:2179`，逐层 `rmsnorm_k` → `project`（`q/k/v` 三次，`q4` 走 `q4_matmul_k_shards`）→ `rope_k` → append store → `attention_k` → `project`（o_proj）→ 残差 → `rmsnorm_k` → gate/up → `swiglu_k` → down → 残差 → 终 norm + lm_head。
- **线程模型**：`QwenForward` "Not thread-safe and not reentrant"（`qwen.mojo:1546`）；并发在 runtime 层 —— N 条 engine 线程各持一份 handler，孪生体 `QwenForward(proto, max_tokens)` 只**借用**原型的只读字节（fp32 权重、q4/q8 块流、rope 表），KV/scratch 自持（`serve.mojo:400`、`qwen.mojo:1783`）。
- **q4 权重**：`q4_arena: SharedRegion` + `q4_blocks: List[RawPtr]`（`qwen.mojo:1597`），槽位 `layer * Q4_SLOTS_PER_LAYER + slot`，`project` 把块流 RawPtr 直接递给 kernel（`qwen.mojo:1969`）。激活缓冲是 `act_arena: Arena`（构造期一次性分配，`qwen.mojo:1570`）。
- **上下文穿层的两种既有形态**：编译期参数（`[backend]`）或构造期实例字段（`prof`/`shards`）。**没有运行期传入上下文对象的先例**。

## 3. 设计

### 3.1 被否决的两条路

- **`DeviceContext` 作为 `QwenForward` 字段**：违背"后端是方法参数"的哲学且技术上矛盾 —— 原型被多线程共享，而 device 资源（权重副本、act 缓冲）必须 per-会话；编译期参数也变不出条件字段。
- **`run[BACKEND_GPU=2]` 走同一 host 视图链**：`run` 的每层中间量都是 host `RawPtr` 视图（`q_ptr`/`normed_ptr`/…），GPU 分支硬套它意味着每层把激活搬上搬下 —— 每层两次 PCIe 往返（~10µs/次 × 28 层 × 2），把 132 GB/s 的 kernel 全部淹没。**数据驻留差异是本质的，不是实现偶然**。

### 3.2 正路：平行的 `GpuSession`

新 struct（放 `src/alofa/runtime/` 或 `src/alofa/gpu/`）：

```
struct GpuSession:
    var ctx: DeviceContext          # 每 session 一个（AsyncRT 线程安全性未验证，保守 1:1）
    var d_w: List[DeviceBuffer]     # q4 块流 + fp32 参数（norm/bias/rope 表）的 device 只读副本
    var d_act: ...                  # 激活常驻缓冲（h/normed/q/k/v/rot/attn/proj/ffn 中间量）
    var d_kv: ...                   # per-layer k/v store（device 侧 KV cache）
    var d_logits: DeviceBuffer      # 只回传这一块（~608KB for vocab=151936）
```

- **所有权**：`GpuSession` 由持有孪生体/handler 的那一方构造（`spawn_twin` 处），生命周期 ≤ 原型（device 权重副本在构造时从 `q4_blocks` 上传，要求原型已加载完）。**`QwenForward` 一行不改** —— 它保持纯 CPU 语义，gpu 路径只**读**它的权重与形状。
- **调用面**：自由函数 `gpu_step(mut session, model, token_ids) -> F32Ptr`，与 `run[backend]` 平行而非同链。内部仍保持 prefill/decode 一条路（`t = len(ids)`，与 `run` 同构），一次 token 进、一次 logits 回传，**中间零 host 往返**。
- 后端编号 `BACKEND_GPU=2` 仍登记（`uses_vector_backend` 对它返回 False），但 gpu 路径不进 `run` 的分发链 —— 编号只是文档与将来 kernel 分片的命名空间。

### 3.3 数据面

- **权重上传（一次性）**：q4 块流按槽位、fp32 参数（两个 layernorm 权重×28、bias、rope cos/sin 表、embedding 矩阵）逐块 `enqueue_copy` + 一次 `synchronize`。此后只读。
- **激活驻留**：全部中间量常驻 device，跨层复用 device 缓冲（形状构造期已知，照 `act_arena` 的分配逻辑换到 device）。
- **KV cache**：device 侧 per-layer store，append 用一个逐元素 copy kernel（替代 host 的 `copy_into`）。max_tokens 上限时 28 层×2×4096×128×4B ≈ 117MB，单卡放得下。
- **每 token 的 host↔device**：进 = token id（几个字节），出 = logits 一次 d2h。采样仍在 CPU（`runtime/sampler.mojo` 不动）。

### 3.4 kernel 补齐清单（都很小，逐个进门）

| kernel | 现状 | 判据 |
|---|---|---|
| rmsnorm_gpu / q4 matmul / attention(_par) | ✅ 已过门 | — |
| **rope_gpu** | 无：逐元素旋转，cos/sin 表 device 读 | 标量同序，零容差（纯乘加） |
| **swiglu_gpu** | 无：`x·sigmoid(x)·up`，f64 同序 | 标量同序，零容差（exp 已知 1 ulp ⇒ 相对容差） |
| **add_gpu**（残差） | 无：逐元素 | 零容差 |
| **gather_gpu**（embedding） | 无：行拷贝 | 零容差 |
| lm_head | 复用 q4 matmul（若权重是 q4）或 linear_gpu | 已覆盖 |
| **kv append** | 无：逐元素 copy | 零容差 |

## 4. 判据（解锁条件）

1. 清单中每个新 kernel 进七项差分门（门变八项+），全部 ≤ 1e-5 相对（能零则零）。
2. **端到端 logits 差分**：同一 token 序列，`gpu_step` 与 `run[BACKEND_AVX2]` 的 logits `max_abs_diff` ≤ 1e-5 相对（允许 expf 级 ulp 差存在）。
3. 单流 decode step ≤ **2ms/token**（五十八的 1.4ms 模型级估计 + 传输与 launch 余量）；超过则先查 sync/launch 次数再查 kernel。

## 5. 步骤（有序，每步独立可验证）

1. 补齐 §3.4 的小 kernel，逐个进门（纯增量，不碰 model 层）。
2. `GpuSession` 骨架：构造、权重上传、**单层**（rmsnorm → q_proj）在 device 跑通，与 `run` 的同层输出对数。
3. 全层 decode 单流：`gpu_step` 逐层跑通，端到端 logits 过判据 2。
4. 计时与稳态验证（判据 3），数字进账本。
5. prefill（同一条路，`t > 1`，attention 的 q_len 维已备好）。
6. 批量/多会话/流重叠（`rows=32` tiled GEMM、多 stream）—— 留白，等真实负载定形状。

## 6. 风险与边界

- **借用时序**：device 权重副本上传时原型必须已完成加载；`GpuSession` 生命周期必须 ≤ 原型（与孪生体同一约定，靠 runtime 层结构保证）。
- **异步性的错误暴露**：每层不 sync，kernel 错误只在 logits 回传时浮现 —— 差分门与逐层对数（步骤 2）是对冲，不能跳。
- **DeviceContext 线程安全**：未验证 ⇒ 1 session 1 ctx，session 与 engine 线程 1:1（与孪生体同构）；多 session 共享 ctx 是将来验证过再说的事。
- **q8 KV**：若 KV cache 量化上线（`05-kv-cache-quantization.md`），d_kv 的布局要跟着换 —— 本方案的 KV 先按 fp16/fp32 存，给量化留槽位。
- **远端环境**：ncu 不可用（账本五十五），性能归因继续走"变体分解 + 同机对照臂"。
