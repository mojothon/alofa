# 能力账本（Capability Ledger）

> **本文件是 alofa 的一等交付物，不是文档附录。**
>
> 设计动机：在调研的 14 个 Mojo 项目中，`RuneForgeAI/A.E.S.I.R.` 是唯一诚实的项目 —— 它的 README 宣称"生产可用"，而它的 `CAPABILITY_LEDGER.md` 明写"无吞吐、无内存效率、无性能证明"。**账本比 README 可信，所以 alofa 让账本成为唯一事实来源，并由 CI 强制校验。**
>
> 规则（由 `tests/capability/ledger.mojo` 在 CI 中执行，入口 `pixi run check-ledger`）：
> - `verified` / `verified-remote` **必须**给出 `evidence:<path>`，且**该文件真实存在**；否则 CI 失败。
> - `verified-env` **必须**给出 `probe:<观测命令>` —— 环境事实没有测试文件，但必须留下可复现的观测方式。
> - `scaffold` 表示代码存在但未接入主路径 —— **禁止出现在 README 的功能列表中**。
> - `designed-only` 表示设计完成、编译通过，但本机无对应硬件可验证 —— **禁止给出任何性能数字**。
> - `missing` 表示未开始。
> - `hardware-blocked` 表示在当前开发机上物理不可能验证。
> - `target` 表示计划达到的等级，尚未验证。
>
> **门本身也被测试**：`tests/fixtures/ledger_bad.md` 是**故意写坏的账本**，校验器必须拒绝它。
> 这一点至关重要 —— 一个只会通过的门等于没有门。若有人删掉文件存在性检查，真账本依然"通过"，
> 但这个 fixture 会让 `test_ledger.mojo` 立刻失败，暴露门已失效。

**最后更新**：2026-09-16（第三次更新：**账本门落地并被 CI 强制**；标签拆为三类 `verified` 变体，无免检档）

---

## 证据分级说明

> 写成**列表而非表格**，是刻意的：校验器只扫 `|` 开头的表格行，而这一节是标签的
> **定义**，不是被校验的条目 —— 若写成表格，它会被自己定的规则误伤。

- **`verified`** —— 已由可复现的测试证明。必须给出 `evidence:<path>`，且该文件真实存在。README ✅ 可提及，并须附测试路径。
- **`verified-remote`** —— 同上，但**只能在 A100 远程验证机执行**。同样必须给出 `evidence:<path>`；CI 只校验"复现方式存在"，不执行（CI 上没有 A100）。README 可提及，须注明需远程机。
- **`verified-env`** —— **环境事实**（硬件规格、驱动版本、工具链版本）。这类东西天生没有测试文件，要求它必须有测试文件是不诚实的；但也不能因此免检，所以必须给出 `probe:<观测命令>` 作为可复现凭证。README 不可作为"能力"提及。
- **`partial`** —— 部分可用，有已知缺口。
- **`scaffold`** —— 代码存在但未接入主路径。README ❌ 不可以。
- **`designed-only`** —— 设计完成、编译通过，但无硬件可验证。仅可说明"已设计"。
- **`hardware-blocked`** —— 当前开发机物理上无法验证。
- **`target`** —— 计划达到的等级，**尚未验证**。用它取代"`verified` 目标"这类措辞，避免造成已验证的错觉。
- **`missing`** —— 未开始。

**三类 `verified` 变体都没有免检档** —— 这是设计的核心：若要求一律有测试文件，环境事实会全部失败，门就会永远红灯然后被人 `--no-verify` 掉。

---

## 0. 平台与工具链

| 能力 | 状态 | 证据 / 说明 |
|---|---|---|
| Mojo 1.0.0 工具链可用 | `verified-env` | `probe:pixi run mojo --version` → `Mojo 1.0.0 (ed45d567)` |
| libc FFI（`external_call`） | `verified` | `evidence:tests/capability/test_libc_ffi.mojo`（5/5 通过）<br>`pixi run mojo run tests/capability/test_libc_ffi.mojo` |
| `socket()` / `epoll_create1()` / `timerfd_create()` / `eventfd()` | `verified` | `evidence:tests/capability/test_libc_ffi.mojo`（实测返回有效 fd 12/13/14/15） |
| `setsockopt(SO_REUSEPORT)` | `verified` | `evidence:tests/capability/test_libc_ffi.mojo`（`test_reuseport_setsockopt`） |
| **`flare` / `json` 可导入**（含 `flare.runtime.Reactor`、`flare.http.HttpServer`） | `verified` | `evidence:tests/capability/test_deps.mojo`（4/4 通过）<br>`pixi run mojo run tests/capability/test_deps.mojo`<br>注：import 在模块顶层，**编译失败即断言失败** |
| **CUDA kernel 在 A100 上数值正确** | `verified-remote` | `evidence:tests/gpu/vecadd.mojo`（实测 `c[999] = 2997.0` ✓）<br>`./scripts/a100.sh run 4 tests/gpu/vecadd.mojo`<br>⚠️ 需 A100 远程机；本地跑必失败，故**不加入 `pixi run test`** |
| `fork()` 多进程 | `partial` | 在编译产物中可用；**在 `mojo run`（JIT）下会崩溃编译器** → 只能由 e2e 套件验证 |
| Mojo 标准库 async / 并发 | `missing` | 官方 roadmap Phase 2 **未开始** |
| Mojo 标准库网络 / socket | `missing` | roadmap 中**未列为可追踪任务** |
| Mojo 原生包管理 | `missing` | roadmap Phase 2 未开始；依赖分发仅 pixi/conda + `.mojoc` |
| Mojo 运行时多态（动态 trait / existentials） | `missing` | roadmap Phase 2 未开始 → 后端抽象只能用编译期 trait |
| Mojo 模式匹配 / ADT | `missing` | roadmap Phase 2 未开始 |
| 官方基准框架 | `partial` | roadmap 🚧 进行中 → alofa 自建 `verify/roofline.mojo`；**骨架已完成**（见 §10），但尚无真实测量 |

## 1. 硬件可用性（决定哪些能力可以升级为 `verified`）

### 1.1 开发机

| 资源 | 实测 | 状态 |
|---|---|---|
| CPU | i7-9700K，8 核，**AVX2（无 AVX-512）** | `target` |
| GPU | **GTX TITAN X，12 GB，Maxwell sm_52，驱动 535** | `hardware-blocked`（现代 CUDA 栈与 MAX 不支持） |
| 内存 | 62 GB 总计，**可用 ~13 GB** | 限制：本地跑 ≤ 4B int4 |
| 磁盘 | 916 GB，余 99 GB（已用 89%） | 模型缓存需管理 |

### 1.2 远程验证机（A100，**2026-09-16 实测打通**）

| 资源 | 实测 | 状态 |
|---|---|---|
| 主机 | `lcl@10.107.6.60`，SSH 免密；`/app/lcl/mojo-projects` | `verified-env` — `probe:./scripts/a100.sh gpu` |
| GPU | **6× A100**（1×80GB + 5×40GB = **280 GB**），`compute_capability 8.0`，驱动 `560.35.03`，CUDA 12.1 | `verified-env` — `probe:./scripts/a100.sh gpu` |
| `gpu-query` 识别设备 | 输出 A100 / CC 8.0 / api 12060 / max_threads_per_block 1024 | `verified-env` — `probe:./scripts/a100.sh probe`（需 `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas` 绕过驱动版本检查） |
| **端到端 GPU kernel 数值正确** | 向量加 kernel，`c[999] = 2997.0`（= 999 + 2×999）✓ | `verified-remote` — `evidence:tests/gpu/vecadd.mojo` |
| CPU / 内存 / 磁盘 | 2× Xeon Platinum 8358（128 vCPU）/ 755 GB（可用 ~521 GB）/ 可用 ~252 GB | `verified-env` — `probe:./scripts/a100.sh shell` |
| Mojo 运行时 | `/shared/lcl/mojo-projects/nova/.pixi/envs/default/bin/mojo`（1.0.0） | `verified-env` — `probe:./scripts/a100.sh shell`（需 `MODULAR_HOME=$PIXI_ENV/share/max`） |
| ⚠️ 网络 | **无外网**（无 DNS / ping / curl 均不通） | 依赖同步须 rsync 本地 `.pixi` |
| ⚠️ 共享 | **共享机**，GPU 0–3 常满载（92–99%） | 只能用空闲卡（当前 4、5）；性能数字须标注共享环境 |
| 模型库 | `/app/lcl/models/` 已有 44 个模型 | `verified-env` — `probe:ssh lcl@10.107.6.60 ls /app/lcl/models` |

## 2. 语言级 ISA 后端

| 后端 | 状态 | 证据 / 说明 |
|---|---|---|
| `scalar` (fp32) | `missing` | P1 交付；作为数值 oracle |
| `avx2` (8×fp32) | `missing` | P1 交付；**本机唯一可端到端验证的向量路径** |
| `avx512` (16×fp32) | `hardware-blocked` | 本机无此指令集 |
| `neon` (4×fp32) | `hardware-blocked` | 本机为 x86 |
| `cuda` (sm_80) | `partial` | 开发机 Maxwell sm_52 不可用；**但 A100 验证机已跑通端到端 kernel（§1.2）** → 不再是 `hardware-blocked`。正确性可验；**性能数字须标注共享环境并报 roofline 利用率** |
| `wgpu` | `missing` | P5；计划复用 `wgpu-mojo` |

## 1.3 外部依赖（纯 Mojo，直接依赖而非自研）

| 依赖 | 版本 / 来源 | 状态 | 说明 |
|---|---|---|---|
| `flare` | 0.2.0，`https://prefix.dev/mojo-force` | 可用（未接入） | 纯 Mojo 网络栈：reactor / scheduler / timer_wheel / watchdog / mutex / reuseport / buffer_pool / io_uring + HTTP1-3 / QUIC / TLS。**L5/L6 地基** |
| `json` | 0.1.2，同频道 | 可用（未接入） | 纯 Mojo SIMD 两遍解析 + tape Document + 编译期反射序列化 |
| `mojo.core` | `tamnd/mojo.core`，Apache-2.0，2030 符号 / 23.1% 对齐 | 未评估 | Go stdlib 移植（log / flag / time / os / sync / regexp / template）。**无 conda 包**；`regexp` 是否可用须先实测 |
| `mojo-mpi` | `BenWibking/mojo-mpi` | 不适用（短期） | ⚠️ **只有点对点，无 allreduce/allgather** → TP 需自研集合通信或走 NCCL FFI |
| `mojo-embree` | `lee101/mojo-embree` | 不依赖 | 领域专用；只借鉴技法（调用方拥有的固定容量工作栈 / `地址+计数` C ABI / 与 C 参考实现的差分验证组织法） |
| ~~`fast-tokenizer`~~（本地） | — | ❌ **明确不采用** | 其 BPE 用贪心 trie 而非 rank 优先合并，**已有实测分歧**；`decode` 混入 Python FFI、golden 测试为空。作反面教材 |

## 3. 基础层（L0）

| 能力 | 状态 | 证据 |
|---|---|---|
| dtype 描述与量化元数据（含 q4_0 / q4_k / q8_0 块布局编译期元数据） | `verified` | `evidence:tests/unit/test_core_dtype.mojo`（10/10 通过） |
| 张量视图（不拥有数据；形状 / 步幅 / 类型 / 偏移） | `verified` | `evidence:tests/unit/test_core_tensor.mojo`（11/11 通过） |
| arena 分配器（bump + 对齐 + 整块重置） | `verified` | `evidence:tests/unit/test_core_memory.mojo`（9/9 通过） |
| 只读文件映射 + 页缓存提示 | `verified` | `evidence:tests/unit/test_core_mmap.mojo`（7/7 通过；与 `FileHandle` 逐字节对比） |
| 具名错误类型（`AlofaError` + 错误码常量） | `verified` | `evidence:tests/unit/test_core_error.mojo`（7/7 通过） |
| 结构化日志（JSONL，`verify` 可直接解析） | `verified` | `evidence:tests/unit/test_core_log.mojo`（8/8 通过） |
| 平台原语绑定（socket / epoll / timerfd / eventfd / openat / mmap / madvise / mlock） | `verified` | `evidence:tests/unit/test_core_ffi.mojo`（11/11 通过）<br>注：`tests/capability/test_libc_ffi.mojo` 是不依赖本项目的**平台探测**，与之互补 |
| **分层守门**（L0 不得出现上层概念，含注释与 import） | `verified` | `evidence:tests/capability/test_layering.mojo`（4/4 通过）<br>`pixi run test` 每次执行；含**红测自检**：故意违规的文本必须被判违规，否则门本身失效 |

## 4. 算子层（L1）

| 能力 | 状态 | 证据 |
|---|---|---|
| 量化 dequant（q4_0 / q4_k / q8_0 / int8 / fp8） | `missing` | — |
| RMSNorm | `missing` | — |
| SwiGLU | `missing` | — |
| RoPE | `missing` | — |
| GQA 因果注意力（非分页） | `missing` | — |
| GQA 因果注意力（分页 / block table） | `missing` | 依赖 L3 KV 统一寻址 |
| matmul（fp32 / q4 dequant） | `missing` | — |
| MAX 内核复用（`linalg` / `layout` / `quantization`） | `missing` | 需隔离层；注意 MAX 导入路径有迁移风险 |

## 5. 模型层（L2）

| 能力 | 状态 | 证据 |
|---|---|---|
| HF `config.json` 解析 | `missing` | — |
| safetensors 读取（mmap、分片） | `missing` | — |
| GGUF 读取（**从张量偏移反推 block layout，不信任 type id**） | `missing` | 借鉴 `MOJO_STUFF` 的教训 |
| Qwen2.5 架构 | `missing` | P1 首个架构 |
| Llama / Mistral 架构 | `missing` | P1 之后 |
| 多模态 | `missing` | 明确 Non-goal（近期） |
| MoE | `missing` | Non-goal（近期） |

## 6. 运行时层（L3）

| 能力 | 状态 | 证据 |
|---|---|---|
| `BlockPool`（物理池 + 空闲链 + refcount） | `missing` | 创新点 1 的基础 |
| `PageTable`（请求 → 块序列） | `missing` | — |
| `RadixNode` 前缀树（token 粒度映射到 block 粒度） | `missing` | 创新点 1；**须先独立单测再接 attention** |
| 三视图共享 refcount 的一致性 | `missing` | 需专门的不变量测试 |
| 频率感知淘汰（2Q + 复合评分） | `missing` | 创新点 2；**须有回放数据才可上线** |
| 采样器（top-k / top-p / min-p / 温度 / 重复惩罚 / logit bias） | `missing` | — |
| 采样分布正确性（卡方 / TVD 检验） | `missing` | 直击 `llm-mojo` 的分布错误 |

## 7. 执行编排层（L4）

| 能力 | 状态 | 证据 |
|---|---|---|
| 纯函数调度器（token budget，零分配） | `missing` | 创新点 3 |
| chunked prefill | `missing` | — |
| 抢占（重计算） + 抢占计数指标 | `missing` | — |
| 批张量池（稳态零堆分配） | `missing` | — |
| 调度 trace 录制 | `missing` | — |
| 调度 trace 重放 + 极端场景断言 | `missing` | 让调度边界可脱离模型测试 |
| 延迟护栏（最大等待拍数） | `missing` | 防止 MAX `--max-batch-size` 式的"等批次"延迟 |

## 8. 服务层（L5 / L6）

| 能力 | 状态 | 证据 |
|---|---|---|
| **依赖 `flare` reactor 作为事件循环** | 已决策（未接入） | 原"自研 epoll"方案被推翻；`flare` 0.2.0 已提供 reactor/scheduler/timer_wheel/watchdog/reuseport/io_uring（§1.3） |
| reactor 线程 + engine 独占线程的双线程模型 | `missing` | 见 `02-architecture.md` §6.1；调度决策留 reactor 以保持可重放零锁 |
| libc 事件循环原语（自研**回退**路径） | `verified` | `evidence:tests/capability/test_libc_ffi.mojo`（5/5）：`socket`/`epoll_create1`/`timerfd_create`/`eventfd`/`SO_REUSEPORT` |
| HTTP/1.1 解析与连接状态机 | `missing` | 优先用 `flare.http`；不稳则自建 |
| 发送队列与背压 | `missing` | — |
| SSE 流式输出 | `missing` | — |
| OpenAI 兼容 API | `missing` | JSON 用 `json` 0.1.2 的编译期反射序列化 |
| SO_REUSEPORT 多 worker | `missing` | 复用 `flare.runtime.reuseport`；底层原语亦已验证 |
| master 健康探测 + 优雅退出 | `missing` | — |
| 100 并发 1 小时压测 | `missing` | **P3 判据门** |
| sidecar 回退形态（stdio RPC） | `missing` | 判据失败时的降级路径 |

## 9. Tokenizer

| 能力 | 状态 | 证据 |
|---|---|---|
| 预分词器（**无正则依赖**，手工实现 GPT-2 规则） | `missing` | 两个参考项目死在这里 |
| BPE | `missing` | — |
| Unigram | `missing` | — |
| WordPiece | `missing` | — |
| `tokenizer.json` 加载 | `missing` | — |
| GGUF vocab 加载 | `missing` | — |
| `chat_template` 渲染（Jinja 子集） | `missing` | 借鉴 `molla`：语义来自模型的 `chat_template`，不手写每族渲染器 |
| 4560 条差分用例（与 HF 逐 id 一致 + 往返） | `missing` | **P1 门** |

## 10. 验证体系（正交支柱）

| 能力 | 状态 | 证据 |
|---|---|---|
| oracle 差分框架（logits 余弦 / argmax / top-k 集合） | `missing` | — |
| greedy 逐 token 相等校验 | `missing` | — |
| 分布检验（卡方 / TVD） | `missing` | — |
| roofline **骨架**（采集 + 利用率报告接口，峰值由调用方传入） | `verified` | `evidence:tests/unit/test_verify_roofline.mojo`（13/13 通过）<br>**不内置任何机型常数**：峰值是构造参数，≤0 直接报错 → 杜绝"抄规格书当实测" |
| roofline 报告（真实 kernel / 真实硬件） | `missing` | 骨架已在上一行通过验证，但尚未接入任何算子 → **未产出任何性能数字** |
| 性能回归门（PR 级） | `missing` | — |
| 能力账本 CI 校验 | `verified` | `evidence:tests/capability/test_ledger.mojo`（7/7 通过）<br>`pixi run check-ledger`<br>**自证循环**：这一条的 evidence 正是校验账本本身的那个测试 —— 账本用自己声明的门来证明自己可信 |

## 11. 对外承诺（兑现能力）

| 能力 | 状态 | 说明 |
|---|---|---|
| 可在本机跑通的端到端服务 | `missing` | P1–P3 的目标 |
| 与 llama.cpp 的同机同 prompt 对比 | `missing` | **唯一被接受的性能参考基准** |
| "比 vLLM / MAX 更快" | ❌ **不做此声明** | 缺少公平比较条件（vLLM/MAX 在 A100 上的数字来自不同栈与环境） |
| GPU **正确性**结论 | ✅ **可做** | A100 验证机已打通（§1.2），CUDA kernel 可与标量后端做逐值差分 |
| GPU **性能**数字 | ⚠️ **有条件可做** | 必须：① 在 A100 机测得 ② **报 roofline 利用率而非裸 tok/s** ③ 标注"共享环境"与其他卡占用 ④ 记录驱动/CUDA 版本 |

---

## 变更日志

> 与「证据分级说明」同理，这里也写成**列表而非表格**：变更日志天然要提到标签名
> （例如"拆分为 `verified-env`"），写成表格会被校验器当成待校验条目。
> **改措辞去迁就工具是错的，改结构才对。**

- **2026-09-16** —— 初始化账本；libc FFI 相关 6 项标记为已验证。完成本机实测（`tests/capability/test_libc_ffi.mojo`，5/5 通过）；其余全部为 `missing`。
- **2026-09-16** —— **新增 §1.2 A100 远程验证机（6×A100，sm_80）为已验证**；`cuda` 从 `hardware-blocked` 改为 `partial`。实测打通：`gpu-query` 识别 A100（CC 8.0），端到端向量加 kernel 数值正确（`c[999]=2997.0`）。关键坑：需 `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas` 绕过 MAX 对驱动 ≥580 的要求。
- **2026-09-16** —— **新增 §1.3 外部依赖**；服务层从"自研 epoll"改为"依赖 `flare` 0.2.0"；`fast-tokenizer` 标记❌不采用。调研本地 `flare`（纯 Mojo 生产级网络栈，已发 conda 包）后推翻原方案（A7：不重复造轮子）；其 BPE 合并算法有已知实测分歧。
- **2026-09-16** —— P3 周期 3–4 周 → **1–2 周**；P5 中 CUDA 从第 2 升为第 1 且可验证。`flare` 提供 reactor/HTTP/reuseport，省下约 2 周；A100 机使 CUDA 可验证。
- **2026-09-16** —— **账本门落地**：新增 `tests/capability/ledger.mojo`（校验器）+ `test_ledger.mojo`（门，7/7）+ `tests/fixtures/ledger_bad.md`（故意写坏的自检抗原）；标签拆为三类变体；§10「能力账本 CI 校验」升为已验证。**红测已验证**：插入假记录（标 verified 但 evidence 文件不存在）→ 门失败、退出码 1；撤掉后 16/16 通过。红测固化为常驻自检，防止门日后被悄悄改弱。
- **2026-09-16** —— **为什么要把 `verified-env` 拆出来**：环境事实（"6×A100""驱动 560.35.03"）天生没有测试文件。若一律要求测试文件，§1.2 会全部失败 → 门永远红灯 → 迟早被绕过。拆开后三类各有凭证要求，**没有免检档**。
- **2026-09-16** —— **P0 基础层补齐**：§3 六项（`dtype` / 张量视图 / `arena` / 只读映射 / 具名错误 / 结构化日志）由 `missing` 升为 `verified`，另增"平台原语绑定"与"**分层守门**"两条；§10 增 roofline 骨架（`verified`）。`pixi run test` 现跑 96 项：新增 7 个单元套件 + 分层守门。
- **2026-09-16** —— **`open` 不能直接用 `external_call`**：stdlib 自己的文件 I/O 已声明同名符号，两处签名不一致会让整个模块 lower 失败，且报错指向调用点而非冲突本身。改用 `openat(AT_FDCWD, ...)` —— 同一次 open，换个名字。
- **2026-09-16** —— **Mojo 在"最后一次使用"处析构，不是作用域末尾**：`arena.alloc()` 返回的裸指针不带借用信息，于是 arena 会在最后一次 `alloc` 处就被 `munmap`，随后写入即段错误。解法是显式的 `keep_alive()`：它运行时什么也不做，作用是**声明 arena 的最后一次使用在哪**。借出裸指针的代价，记在 §3 与代码注释里。
- **2026-09-16** —— **roofline 利用率用整数千分比而非浮点**：测量值是要被比较、打印、断言的，整数三样都精确，浮点三样都要容差。瓶颈判定用交叉相乘（`flops×peak_bw` vs `bytes×peak_flops`），全程不除法。**峰值必须是调用方传入的构造参数**，内置机型常数等于替规格书背书。
