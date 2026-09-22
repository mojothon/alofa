# 路线图与验收门

> 每个阶段都有**硬性验收门（Gate）**。门不通过就不进入下一阶段 —— 这是 `01-ecosystem-analysis.md` 里 13 个失败项目的共同教训：它们全部跳过了门。
>
> 所有时间估算为**单人全职等效**，且假设 Mojo 无 async/网络/包管理（见 `02-architecture.md` §10）。

---

## 目标与非目标

**短期目标（P0–P3）**：在**本机 CPU（AVX2）× int4 × Qwen2.5-1.5B** 上跑通"端到端可服务"的闭环，并且**可被第三方复现**。

**中期目标（P4–P5）**：跨后端（WebGPU → CUDA）与算法创新（频率感知淘汰、投机解码）。

**验证环境**（2026-09-16 打通，改变了多处时间估算）：

| 环境 | 用途 | 约束 |
|---|---|---|
| 开发机（i7-9700K / AVX2 / Maxwell sm_52 GPU 不可用 / ~13 GB 可用内存） | CPU 路径的端到端验证 | GPU 不可用 |
| **A100 远程验证机**（6×A100 共 280 GB / 128 vCPU / 755 GB RAM） | **CUDA 正确性 + 性能基准** | 无外网（须 rsync 同步 `.pixi`）；**共享机，只能用空闲卡** |

**明确不在本路线图内**：训练、多机分布式、GUI、纯 Python/Rust 主体。

---

## P0 — 地基与自我约束机制（1–2 周）

| # | 交付 | 说明 |
|---|---|---|
| 0.1 | ✅ `core/`：dtype、tensor 视图、arena 分配器、对齐分配 | 张量只做视图，不拥有数据；分配走单一 arena。`evidence:tests/unit/test_core_dtype.mojo`（10/10）、`test_core_tensor.mojo`（11/11）、`test_core_memory.mojo`（9/9） |
| 0.2 | ✅ `core/ffi/`：epoll / socket / timerfd / eventfd / mmap 绑定 | 平台探测 5/5（`tests/capability/test_libc_ffi.mojo`）+ 本项目绑定 `evidence:tests/unit/test_core_ffi.mojo`（11/11） |
| 0.3 | ✅ `core/error.mojo` + `core/log.mojo` | 具名错误（禁止字符串错误码）；JSONL 日志可被 verify 消费。`evidence:test_core_error.mojo`（7/7）、`test_core_log.mojo`（8/8） |
| 0.4 | ✅ **账本校验器** `tests/capability/ledger.mojo` | 解析表格行 + `std.os.stat` 判文件存在；入口 `pixi run check-ledger` |
| 0.5 | ✅ **账本接入 CI**（`.github/workflows/ci.yml`） | 三类 `verified` 变体都必须给出真实凭证，见 Gate P0 第 2 条 |
| 0.6 | ✅ **基准骨架** `verify/roofline.mojo` | 输出带宽/算力利用率（整数千分比），不输出裸 tok/s；峰值由调用方传入，`evidence:tests/unit/test_verify_roofline.mojo`（13/13） |
| 0.7 | ✅ 只读文件映射 + 页缓存提示 `core/mmap.mojo` | `evidence:tests/unit/test_core_mmap.mojo`（7/7，与 `FileHandle` 逐字节对比） |

### Gate P0
1. ✅ **已通过（2026-09-16）**：`pixi run test` 全绿 —— 96 项，含 capability 套件（账本门 7/7、分层守门 4/4、平台探测 5/5、外部依赖 4/4）与 7 个基础层单元套件。
2. ✅ **已验证（2026-09-16）**：账本出现"`verified` 但测试文件不存在"时 CI **会失败**。
   - **红测**：向真账本插入一行 `verified` + `evidence:tests/this_file_is_a_lie.mojo`
     → `test_real_ledger_passes` 失败，退出码 **1**，并报出具体行号与缺失路径。
   - **绿测**：撤掉该行 → 16/16 通过，退出码 0。
   - **红测已固化为常驻自检**：`tests/fixtures/ledger_bad.md` 是**故意写坏的账本**，
     校验器必须拒绝它（4 处违规：文件不存在 / 缺 evidence / 缺 probe / 拼错标签）。
     这样门的有效性**每次 CI 都被重新验证**，而不是只在演示那一次成立 ——
     否则将来有人删掉存在性检查，真账本依然"通过"，门就悄悄失效了。
3. ✅ **已通过（2026-09-16）**：`tests/capability/test_layering.mojo`（4/4）断言 `core/` 中不存在上层概念（含注释与 import）。该门自带红测：故意违规的文本必须被判违规，否则门失效。Mojo 1.0 无目录遍历，故文件清单硬编码在测试内，并由"清单非空且每个文件可读"一条防止清单腐坏。

---

## P1 — CPU 端到端最小闭环（3–4 周）

> **这是整个项目风险最高的一段**：两个参考项目（`MojoStream`、`MOJO_STUFF`）都死在这里 —— 有前向传播但没有可用的 tokenizer，因此"能生成、不能服务"。

| # | 交付 | 说明 |
|---|---|---|
| 1.1 | `model/weights.mojo`：safetensors + GGUF 读取器 | **GGUF 读取不信任 type id，从张量偏移反推 block layout**（借鉴 `MOJO_STUFF`，绕开 llama.cpp 的历史 bug） |
| 1.2 | `tokenizer/`：预分词器 + BPE + `chat_template` 子集 | Mojo stdlib 无 regex。**默认手写 GPT-2 规则**；可选路径是 `mojo.core` 的 `regexp`（Go RE2 移植，**但对齐度仅 23%，必须先实测再依赖**）。vocab 来自 `tokenizer.json` / GGUF。<br>⚠️ **绝不复用本地 `fast-tokenizer` 的合并算法** —— 它用贪心 trie 而非 rank 优先合并，已有实测分歧（见 `02-architecture.md` §2.3） |
| 1.3 | `kernels/cpu/scalar` + `avx2`：RMSNorm、SwiGLU、RoPE、GQA 因果注意力、matmul（fp32 + q4 dequant） | `scalar` 是 oracle，`avx2` 是性能路径 |
| 1.4 | `model/arch/qwen.mojo`（先做 Qwen2.5-0.5B/1.5B） | 单一架构做深，不做多架构 |
| 1.5 | `verify/oracle.mojo`：与 HF transformers 差分 | logits 余弦 + argmax + top-k 集合；greedy 序列逐 token 相等 |
| 1.6 | `runtime/sampler.mojo` | top-k/top-p/min-p/温度/重复惩罚/logit bias |
| 1.7 | **CUDA 后端 bring-up（A100 验证机）** | 环境已打通（§11）。P1 末同步启动：先跑通 RMSNorm / RoPE / matmul 三个 kernel 与标量后端的**逐值差分**，不追求性能 |

### Gate P1（全部必须通过）
1. **数值门**：Qwen2.5-0.5B fp32，与 HF 的 logits 余弦 ≥ 0.999，argmax **完全一致**，greedy 生成 128 token **逐 token 相等**。
2. **量化门**：q4_0 权重下，greedy 生成 **≥ 90%** token 与 fp32 一致（int4 的合理退化范围，阈值需实测后固化）。
3. **tokenizer 门**：对 4560 条差分用例（encode→decode 往返 + 与 HF `tokenizer.json` 逐 id 一致），失败率 = 0。
4. **内存门**：加载后 RSS 增长 ≤ 权重文件大小的 1.15×（验证 mmap + 无意外拷贝）。
5. **CUDA 正确性门**（1.7）：RMSNorm / RoPE / matmul 三个 kernel 在 A100 上的输出与标量后端**逐值一致**（fp32 容差 1e-5，且记录实际最大偏差）。**只验正确性，不报性能。**
6. **诚实门**：若任一子项未达成，账本中对应条目必须是 `partial` 或 `missing`，**不得**在 README 中声称支持。

---

## P2 — KV、调度与并发（4–6 周）

| # | 交付 | 说明 |
|---|---|---|
| 2.1 | `runtime/kv/`：`BlockPool` + `PageTable` + `RadixNode` 统一寻址 | 创新点 1；先独立单测三个视图，再接 attention（避免 `A.E.S.I.R.` 的"建好不接"） |
| 2.2 | `kernels`：paged attention（block table 索引） | 必须先有 2.1 的单测，才允许接入 |
| 2.3 | `engine/scheduler.mojo`：token-budget 统一调度 + chunked prefill + 抢占重计算 | 纯函数、零分配 |
| 2.4 | `engine/trace.mojo`：trace 录制/重放 | 让调度边界可脱离模型测试 |
| 2.5 | `runtime/executor.mojo` + `engine/core.mojo`：忙循环 + 批张量池 | 预分配、零运行时分配 |
| 2.6 | 连续批处理基准 | 同机同 prompt 对比 llama.cpp |

### Gate P2
1. **正确性门**：批大小 1/2/4/8 的输出**与单请求逐 token 完全一致**（批处理不得改变数值结果）。
2. **重放门**：从 trace 重放，`Action` 序列逐字节一致；手工构造的 6 个极端场景（超长 prompt / 并发抢占风暴 / 预算耗尽 / 0 预算 / 取消竞态 / KV 水位临界）全部有断言覆盖。
3. **分配门**：稳态 decode 阶段，每 token 的堆分配次数 = 0（arena 复用）。
4. **吞吐门**：Qwen2.5-1.5B int4，8 核 CPU，并发 8 时解码吞吐 ≥ **llama.cpp 同机同 prompt 的 0.7×**。达不到就把差距与原因写进账本 —— **不调阈值，不粉饰**。
5. **内存门**：KV 池峰值使用率 > 95% 时无 OOM、无正确性退化（抢占路径生效）。

---

## P3 — 服务化（**1–2 周**，原估 3–4 周）

> **因 `flare` 而大幅缩短。** 原方案要自研 epoll 事件循环 + HTTP/1.1 解析器 + 连接状态机 + SO_REUSEPORT 多进程。`flare` 0.2.0 已提供 reactor / scheduler / **timer_wheel** / **watchdog** / mutex / **reuseport** / buffer_pool / **io_uring** 与 HTTP1-3 / QUIC / TLS。**省下的约 2 周全部转入 P2 的 KV/调度深度与 P4 的算法创新。**

| # | 交付 | 说明 |
|---|---|---|
| 3.1 | 接入 `flare` 依赖（锁 0.2.0，来自 `https://prefix.dev/mojo-force`） | A7：不重复造轮子。**先只取 `runtime/` 子系统**，HTTP 层视稳定性决定用 flare 还是自建 |
| 3.2 | ✅（一半）`srv/loop.mojo`：**reactor 事件循环，一条循环 N 条连接** | 见 `02-architecture.md` §6.1 与账本 §8。**已落地**：8 条连接并发 + 「发了一半就停住」的对端不再拖住别人（旧形态停满 5 s，实测现在 0–1 ms），`pixi run test-loop`；线上入口（单进程与多 worker）已换成它，旧循环留作负向对照。⚠️ **没改「一次只生成一条」**：前向仍内联占住循环 —— 生成的并行仍只来自多 worker（**3.2b 已把前向搬到独占线程**；这句记的是 3.2 当时的状态，并解释为什么 3.2 之后仍要有一条 3.2b） |
| 3.2b | ✅ `srv/engine_thread.mojo`：**engine 独占线程**（mailbox + `ThreadHandle`） | **已落地**：前向搬到独占线程，reactor 只做 I/O 与调度决策。`evidence:tests/unit/test_engine_thread_server.mojo`（重门 `pixi run test-engine`：真 socket + 真 fork + 真线程，内联版 `run` 是常驻负向对照）。判据是「槽位占满时，慢生成期间一条新连接仍然被接进来并立刻关掉」：探子从连上到收到 EOF，`run_threaded` 0 ms、`run` 4.5 s（对照把那 5 s 生成全等完了）。⚠️ 别拿「生成期间还有字节在走」当判据 —— 内核缓冲（实测 1.2–2.5 MB）比服务端能排队的字节（1 MiB）还大，两端一样快，门会安静地什么都不测。线上两条入口已换成 `run_threaded`。⚠️ **这一条只换来「生成不再占住这条循环」**：它起的是**一条** engine 线程，同时能生成的条数仍是 1 —— 并发生成是紧跟着的 **3.2c** |
| 3.2c | ✅ `srv/engine_thread.mojo`：**engine 线程池**（N 条线程，每条一份 handler） | **已落地**：`engines>1` 时 N 条 engine 线程各领一份 handler（`Twinable.spawn_twin`），同时能生成 **N** 条。`evidence:tests/unit/test_engine_thread_server.mojo`（`pixi run test-engine`，三模式：threaded=1 / inline 负向 / pooled=2）。代价 **N 条 = N 份权重**（`spawn_twin` 是**重新加载**一份，还没有「共享只读权重 + 各一份 KV」那一层）→ **默认 1**，由 `ALOFA_ENGINE_THREADS`（1–16）开；多 worker 下是 `workers × engines` 份。判据换过两次：①「第二条请求什么时候被答」—— 错，**派得早但生成串行**时它也很快，那是排队不是并行；② 最后是「**两条 5 s 生成的总墙钟**」：`engines=2` **5002 ms**、`engines=1` **10002 ms**（同一条代码路径上的常驻负向对照，它不慢就证明不了门盯住了什么）。三条约束各有安排：流的**相位亲和**靠槽位→线程的黏性 `owner`（第一次派活定归属后不变）+ 单槽位单在途；同连接**按序**同上；**邮箱上界不破**靠单队列按线程过滤（每线程一队列会把一个 `MAILBOX_CAP` 变成 N 个上界）。⚠️ 真权重下 `engines=2` 是两份权重（1.98 GB × 2）；**吞吐**一个数字都没测 |
| 3.3 | ✅ `srv/sse.mojo` + `srv/openai.mojo`：`/v1/chat/completions` 流式与非流式 | OpenAI 兼容是本阶段的对外契约。流式以 SSE 落地（无 `Content-Length`，靠关连接定界），且**文本与非流式逐字节相同**；⚠️ 未做 `stream_options.include_usage`，流式**不修并发生成**（3.2 修的是连接级并发：连接不互相挡路；生成仍串行 —— 那要 3.2b/3.2c） |
| 3.4 | ✅ `srv/master.mojo`：SO_REUSEPORT 多 worker + 健康探测 + 优雅退出 | 复用 `flare.runtime.reuseport`；SIGTERM → 停止 accept → 排空 → 退出。`evidence:tests/unit/test_workers.mojo`（3 worker × 24 响应零错误，abnormal=0 forced=0）；⚠️ fork 后 asyncrt 不可用 → 每 worker 单分片（账本 §8）
| 3.5 | ✅ 压测脚本与报告模板 | `scripts/stress_serve.py`（分钟级非流式口径，与 Gate P3 的 100 并发 SSE 差距显式记账）；systemd 模板 `scripts/deploy/alofa.service`

### Gate P3（**这是纯 Mojo 服务层的判据门**）
1. **压测门**：100 并发长连接 SSE，连续运行 1 小时：零错误、零 fd 泄漏、P99 无劣化趋势。
2. **兼容门**：官方 `openai` Python SDK 仅改 `base_url` 即可完成流式对话。
3. **背压门**：慢客户端不导致内存无界增长（发送队列有上限且触发正确降级）。
4. **扩展门**：8 worker 相对 1 worker，QPS 提升 ≥ 5×。
5. **判据**：若 1 通过 → 纯 Mojo 服务层成立；若 1 不通过且两轮重构无效 → **切换到 `02-architecture.md` §6.3 的 sidecar 形态**，并把该决定与证据写入账本。

---

## P4 — 加速与算法创新（持续）

| # | 交付 | 前置条件 |
|---|---|---|
| 4.1 | AVX2 kernel 调优（寄存器分块、FMA、消除边界分支） | 有 roofline 报告定位瓶颈 |
| 4.2 | KV cache 量化（q8_0 → q4） | 有分布门验证质量退化 |
| 4.3 | **频率感知淘汰**（创新点 2） | **必须先有 workload 回放并证明优于 LRU**，否则不上线 |
| 4.4 | 投机解码（**必须用拒绝采样，不是精确匹配**） | 有分布检验门（卡方/TVD） |
| 4.5 | 分块预填充与重叠调度（对标 SGLang 的零开销重叠） | 先测出当前 CPU 空闲占比 |

### Gate P4
- 每一项都必须附**前后对比数据**（roofline 利用率 + 命中率 + 分布检验），数据不达标则功能回滚并从账本移除。
- 禁止"为了创新而创新"：4.3 和 4.4 都可被否决。

---

## P5 — 跨后端（持续；**本机限制已被 A100 验证机解除**）

| 顺序 | 后端 | 依据 | 可验证性 |
|---|---|---|---|
| 1 | **CUDA**（sm_80） | 生态主流 | ✅ **可验证**（A100 远程机已实测打通，见 `02-architecture.md` §11）→ 可标 `verified` 并可出性能数字（须报 roofline 且标注共享环境） |
| 2 | **WebGPU**（复用 `wgpu-mojo`） | 一份代码覆盖 Vulkan/Metal/DX12；`wgpu-mojo` 已 335 提交、有 conda 包、活跃维护 | 编译与逻辑可验证；**性能不可测** → `designed-only` |
| 3 | Apple / NEON | 开发机生态 | 不可验证 → `designed-only` |

> **顺序调整说明**：CUDA 从"第 2 且不可验证"升为"第 1 且可验证"。这不是乐观，而是因为 `gpu-query` 与端到端 kernel 已在该机跑通（`c[999] = 2997.0` ✓）。**正确性可验证，性能数字须在共享环境标注下谨慎给出。**

### Gate P5
1. 每个后端必须通过 **P1 的同一条数值门**（与标量后端逐 token 一致）才可标记 `verified`。
2. 无硬件的后端**只能标记 `designed-only`**，README 与文档中不得出现任何性能数字。
3. 后端切换不得改变上层 `engine/` 的任何一行代码（用编译断言 + 接口测试保证）。

---

## 里程碑总览

| 阶段 | 周期 | 关键交付 | 一票否决的门 |
|---|---|---|---|
| P0 | 1–2 周 | 地基 + 账本机制 | ✅ **已完成（2026-09-16）**：基础层 6 项 + 分层守门 + roofline 骨架，`pixi run test` 96 项全绿 |
| P1 | 3–4 周 | CPU 端到端（含 tokenizer）+ CUDA bring-up | 与 HF 逐 token 一致；CUDA kernel 与标量后端逐值一致 |
| P2 | 4–6 周 | KV + 调度 + 并发 | 吞吐 ≥ llama.cpp 0.7× |
| P3 | **1–2 周** ↓ | 服务化（基于 `flare`）+ 多进程 | 100 并发 1 小时零错 |
| P4 | 持续 | 加速 + 算法创新 | 有回放数据才能上线 |
| P5 | 持续 | 跨后端（**CUDA 已可验证**） | 无硬件只能标 designed-only |

**累计**：P0–P3 约 **2.5–3.5 个月**可达到"单人可复现的、CPU 上可服务的生产级原型"（P3 因 `flare` 缩短约 2 周）。这个时间表比生态里任何一个项目的自述都保守 —— 这正是它可信的原因。

> **省下的 2 周去向**：全部转入 P2 的 KV/调度深度与 P4 的算法创新（频率感知淘汰、投机解码），**不用于提前宣称功能**。

---

## 每个阶段都要做的三件事

1. **更新能力账本**：把新证据写进去，把未达成的降级。
2. **跑回归基准**：与上一里程碑对比，记录偏差。
3. **写"我们不知道什么"**：每阶段文档末尾追加一节已知未知项（借鉴 `esper` 的正负结果全记录）。

---

**相关文档**
- 生态与竞品分析 → [`01-ecosystem-analysis.md`](01-ecosystem-analysis.md)
- 架构设计 → [`02-architecture.md`](02-architecture.md)
- 能力账本 → [`capability-ledger.md`](capability-ledger.md)
