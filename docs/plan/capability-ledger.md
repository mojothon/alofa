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

**最后更新**：2026-09-18（**N/M 快照不再腐烂** —— 手写的测试条数全部升级为机器核验的 `evidence:…?count=N`，新增 `check-counts` 门与 `ledger-sync`；同时订正本轮暴露出的三处漂移）

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
| libc FFI（`external_call`） | `verified` | `evidence:tests/capability/test_libc_ffi.mojo?count=5`<br>`pixi run mojo run tests/capability/test_libc_ffi.mojo` |
| `socket()` / `epoll_create1()` / `timerfd_create()` / `eventfd()` | `verified` | `evidence:tests/capability/test_libc_ffi.mojo?count=5`（实测返回有效 fd 12/13/14/15） |
| `setsockopt(SO_REUSEPORT)` | `verified` | `evidence:tests/capability/test_libc_ffi.mojo?count=5`（`test_reuseport_setsockopt`） |
| **`flare` / `json` 可导入**（含 `flare.runtime.Reactor`、`flare.http.HttpServer`） | `verified` | `evidence:tests/capability/test_deps.mojo?count=4`<br>`pixi run mojo run tests/capability/test_deps.mojo`<br>注：import 在模块顶层，**编译失败即断言失败** |
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
| CPU | i7-9700K，8 核，**AVX2（无 AVX-512）** | `verified-env` — `probe:grep -o ' avx2 ' /proc/cpuinfo \| head -1`（向量门 `test_avx2_parity.mojo` 每次都读它，缺失即具名失败） |
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
| `scalar` (fp32) | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（逐算子对 Hugging Face）<br>`evidence:tests/unit/test_model_parity.mojo`（整网 4/4 通过，含 128 token greedy 逐 token 相等）<br>作为数值 oracle：只按公式顺序写，**不做任何重排**，供后续向量后端对照 |
| `avx2` (8×fp32 / 4×fp64) | `verified` | `evidence:tests/unit/test_avx2_parity.mojo?count=9`（已并入 `pixi run test`）：rmsnorm / q·k·v 投影 / o 投影 / 残差加 / 第二道 RMSNorm / swiglu，**与标量后端同一份 fixture、同一条容差**（`1e-5 × max(1, |参考|ₘₐₓ)`）<br>与标量后端**逐位相同**（五个算子最大绝对差实测 **0.0**）—— 这是量出来的，不是假设的：同一条比较在偏置写错时确实红过<br>累加仍在 **f64 通道**（4 道）：点积有抵消，换成 f32 累加时相对误差约 `√n · 2^-24`，抵消严重处能吃掉整个 1e-5 判据 —— 与标量同一理由，**这一层当判据不当最快路径**<br>⚠️ **本轮不宣称指令，也不宣称性能**：模块名说的是"按 8 通道 f32 / 4 通道 f64 写的向量 kernel"，是否真的降成 VEX 编码指令**没有做 objdump 核验**；性能数字要等 §10 roofline 接入后才有资格谈<br>负向对照专钉**尾巴**：fixture 里 896 / 4864 恰好都能被通道数整除，标量尾巴**从未被真正执行过**；故另写一个只跑主循环、丢掉尾巴的版本，喂长度 10（非 8 的倍数）的输入，必须判红（实测相差 10.0）<br>另有一条：主机若没有 AVX2 则**具名失败**，不静默跳过 —— 会跳过的门在 CI 上永远绿，而它绿的原因是没跑<br>**已接入整网（2026-09-17）**：`prefill` / `step` / `run` 带编译期参数 `backend`（`BACKEND_SCALAR` / `BACKEND_AVX2`），五个算子（rmsnorm / linear / linear_bias / add / swiglu）按它静态分发<br>`evidence:tests/unit/test_model_avx2_parity.mojo`（`pixi run test-model-avx2`，依赖 2 GB 权重，**不进 `pixi run test`**）：与标量整网门**同一批 prompt、同一段 fp32 参考、同一条判据**（logits 余弦 ≥ 0.999 且 argmax 全等、128 token greedy 逐 token 相等、逐 token 解码与整段 prefill 落点一致）<br>⚠️ **`rope` / `attention` 仍只有标量实现** —— 整网跑在 `BACKEND_AVX2` 上时它们是混跑的，不假装全覆盖。（量化通路不属于这个名单了：2026-09-20 起它随 `q4_matmul_k[backend]` 分后端，见 §「matmul（q4 dequant 融合）」行与当日变更日志）<br>⚠️ **"跑的确实是向量后端"这件事，数值上证明不了**：两个后端在这些形状上逐位相同（这是设计目标，也是上一轮量出来的），任何数值比较都区分不了它们。守着它的是两样东西：① `backend_label` 与算子分发**共用同一个判断**（要谎报得把同一个判断改两遍）；② `pixi run test-backend-guard` 这道**编译期红测** —— `tests/fixtures/bad_backend.mojo` 必须编译失败且原因是 `unknown cpu backend`，否则拼错的后端常量会静默退化成标量、整网向量门照绿<br>⚠️ 后端是**方法**参数不是结构体参数，不是设计偏好：Mojo 1.0.0（ed45d567）在"参数化结构体 + 会抛错误的构造函数"上会直接把编译器进程搞崩，最小复现写在 `tests/fixtures/bad_backend.mojo` 的注释里<br>⚠️ 仍然**不宣称指令、不宣称性能** |
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
| dtype 描述与量化元数据（含 q4_0 / q4_k / q8_0 块布局编译期元数据） | `verified` | `evidence:tests/unit/test_core_dtype.mojo?count=10` |
| 张量视图（不拥有数据；形状 / 步幅 / 类型 / 偏移） | `verified` | `evidence:tests/unit/test_core_tensor.mojo?count=15` |
| arena 分配器（bump + 对齐 + 整块重置） | `verified` | `evidence:tests/unit/test_core_memory.mojo?count=9` |
| 只读文件映射 + 页缓存提示 | `verified` | `evidence:tests/unit/test_core_mmap.mojo?count=7`（；与 `FileHandle` 逐字节对比） |
| 具名错误类型（`AlofaError` + 错误码常量） | `verified` | `evidence:tests/unit/test_core_error.mojo?count=7` |
| 结构化日志（JSONL，`verify` 可直接解析） | `verified` | `evidence:tests/unit/test_core_log.mojo?count=8` |
| 平台原语绑定（socket / epoll / timerfd / eventfd / openat / mmap / madvise / mlock） | `verified` | `evidence:tests/unit/test_core_ffi.mojo?count=11`<br>注：`tests/capability/test_libc_ffi.mojo` 是不依赖本项目的**平台探测**，与之互补 |
| **分层守门**（L0 不得出现上层概念，含注释与 import） | `verified` | `evidence:tests/capability/test_layering.mojo?count=4`<br>`pixi run test` 每次执行；含**红测自检**：故意违规的文本必须被判违规，否则门本身失效 |

## 4. 算子层（L1）

| 能力 | 状态 | 证据 |
|---|---|---|
| 量化 dequant（q4_0） | `verified` | `evidence:tests/unit/test_q4_parity.mojo?count=10`（layer 0 四个真实投影矩阵，q_w/o_w 各 25088 块）**零容差逐位比较**<br>能做到零容差是因为可以：结果只取决于 4 位 nibble 与 fp16 缩放因子，两者都是精确的（半精度能表示的数单精度都能精确表示，故 fp16→fp32 是**精确转换**）—— 于是"差 1 ulp"不是舍入，是**布局读错**<br>块布局（32 值 / 18 字节、fp16 缩放小端、低半字节在前、`v=(nibble-8)*d`）是 **GGML q4_0 的外部事实**，故解量化属格式一致性检验<br>⚠️ **fp32→q4_0 的量化步是我方的格式转换，没有外部参照**，算法显式写在 `scripts/dump_q4_reference.py` 里；不许笼统写成"与 llama.cpp 一致"<br>两条负向对照常驻：高低半字节装反必须被零容差断言抓住；解出的值必须**确实**落在 `{-8d…7d}` 台阶上（否则"逐位相等"可能只是抄了原值） |
| 量化 dequant（q4_k / q8_0 / int8 / fp8） | `missing` | — |
| **量化步（fp32 → q4_0 块流）** | `verified` | `evidence:tests/unit/test_q4_parity.mojo?count=10`（与 `scripts/dump_q4_reference.py` 的离线实现**逐字节相同**，1.03 MB 块流零容差）<br>取整规则是这里唯一值得一提的地方：`round` 取**最近、并列取偶**（IEEE 默认），改成截断会让每个值平均偏小半个台阶 —— **负向对照专门钉这一点**：测试里另写一个截断版量化器，它必须与参考相差若干字节（实测 641074 字节），否则"逐字节比较"根本没在比取整。这类错误最阴：输出照样流利，只是分布整体偏了一点点，任何带容差的判据都放它过去<br>**块内缩放因子改按 MSE 选（2026-09-17）**：仍是`一个 fp16 scale + 32 个 nibble` 的块布局，只是不再取 `amax/7` —— 先用 `amax/7` 起个头量化一次，再对固定的 nibble 取重建误差的最小二乘解`d* = Σ(x·s)/Σ(s²)`（`s = nibble - 8`），迭代两轮<br>为什么值得：朴素写法为了让最大值够到台阶顶，把台阶钉在分布最稀疏的地方；MSE 解允许最大值被裁掉一点点，把台阶挪到分布密集处。实测 q/k/v/o 四个矩阵（另加整网 3 个权重）：**相对 L2 误差 10.75% → 10.34%**（cos 0.9942 → 0.9947）<br>**第三条负向对照**：测试里另写一个只改缩放因子选法（`amax/7`）、取整规则保持一致的旧规则版本，它必须与参考不同（实测相差 121846 字节）。这条是必需的，因为**参照物（fixture）是同一次导出的产物**—— 把实现退回 `amax/7`，参照物会跟着一起退，逐字节比较照样绿<br>顺带纠正上一行的一个说法：Mojo 侧**会**量化（加载期 `enable_q4` 一次），只是**推理期不量化** |
| **整网 q4_0 前向通路**（24 层全部投影走块流） | `verified` | `evidence:tests/unit/test_q4_greedy.mojo`（1/1：4 条 prompt × 128 步教师强制贪心，与 fp32 参考一致 **418/512 = 0.8164**）<br>门是 0.75 的下限，**它判的是"通路在工作"不是"质量达标"**：高低半字节装反会得到 0.0，前向悄悄退回 fp32 会得到 1.0，两头都被抓住（后者另由 `model.q4_enabled` 直接断言）<br>⚠️ **路线图里"一致率 ≥ 0.90"那条没过**，见下一行；本轮的 0.75 是"算的东西是对的"的下限，不是把 0.90 改小 |
| **整网 q4_0 教师强制贪心一致率 ≥ 0.90** | `missing` | **实测 0.8164**（输出投影留在 fp32 时；连它一起量化是 0.77），仍未达门<br>缩放因子这一路**已经走到底了**：按 MSE 选（见上一行）把一致率从 **0.800 抬到 0.8164**（+1.6 个百分点），而权重相对 L2 误差只从 10.75% 降到 10.34% —— 另一个数据点同样说明尺度选择已经到顶：在 q_w 上把 `amax/7` 整体乘一个系数扫一遍，最优是 **×0.90（10.23%）**，而逐块 MSE 解是 10.34%，两者只差 1%，说明"每块一个 scale"这个自由度本身已经榨干<br>**真正的约束是格式，不是选法**：每 32 个元素共用一个 fp16 缩放因子（cos 0.9947），落到 151936 维 argmax 上就是约两成位置翻盘<br>**不放宽门，也不假装达标**。下一步（本轮没做，别当成已有）：格式级改动 —— q4_K（超级块内再给一层 scale）、逐通道/逐行 scale，或带激活重要性矩阵（imatrix）的 scale 选择；这些都是**换块布局**，要新写 dequant 与配套的门 |
| RMSNorm | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（对参考输入/输出对，最大偏差 1e-5 量级；平方和与倒数平方根在 `Float64` 中累加，以免 oracle 自身的舍入成为被怀疑对象） |
| SwiGLU（含 `silu`） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（`silu` 与 `swiglu` 各有一条；参照物是 HF `act_fn` 的**真实输出**与 `down_proj` 的**真实输入**，脚本不自己乘一遍） |
| RoPE | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（cos/sin 表由参考导出，Mojo 只做查表与旋转；**不复现 `inv_freq`** —— 复现它本身就是一类事故源） |
| GQA 因果注意力（非分页） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（14 头 / 2 KV 头；`q_len` 可与 `kv_len` 不同，于是 prefill 与单步 decode 是**同一段代码**）<br>`evidence:tests/unit/test_model_parity.mojo`（`test_incremental_decode_matches_full_prefill`：逐 token 解码与整段 prefill 落点一致） |
| GQA 因果注意力（分页 / block table） | `verified` | `evidence:tests/unit/test_paged_attention.mojo?count=11`<br>`src/alofa/kernels/cpu/paged.mojo`：**只改行的地址**（`j * kv_cols` → `table.row_offset(j, kv_cols)`），算术与顺序和连续 oracle 逐字相同 → 与 `scalar.attention` **逐位相等**（11 个用例、0 个元素不同）。公式本身另由 `scripts/dump_paged_reference.py` 这份**独立 Python 实现**按 1e-5 相对容差钉住，唯一跨语言差异来源是 `exp` 的最后一位<br>⚠️ 本轮走标量后端；GPU 路径见 §4（本机 sm_52 阻塞） |
| matmul（fp32） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（q/k/v/o 四条投影，含带 bias 与不带 bias 两条路径；权重按 `[out, in]` 行主序，与 HF 存储一致，故加载时**没有转置**这一步可忘） |
| matmul（q4 dequant 融合） | `verified` | `evidence:tests/unit/test_q4_parity.mojo?count=10`（解出的权重 × 真实激活，fp64 累加，容差沿用 layer0 的同一判据 1e-5 相对）<br>"融合"是被检验的那件事本身：nibble 读到寄存器里直接乘缩放与激活累加，**不物化解量化后的权重** —— 若先解量化再走通用 matmul，被测的就只是通用 matmul 了<br>累加用 `Float64`，与 `scalar.mojo` 的 `_gemm` 同一理由：这一层当判据不当最快路径<br>`evidence:tests/unit/test_q4_matmul_vec.mojo`（2026-09-20 新增，专用夹具 `tests/fixtures/.../q4vec/`，48 KB）：同一批期望值上的**结构性**差分门 —— 块/行取 1/3/5/7/11/17/28/152、行数取 1/2/3/5/7/13，**专门用来把 SIMD 尾巴露出来**<br>⚠️ 两条 evidence 互补，缺一不可：原 fixture 四个用例的 `cols` 全是 896（**28 块/行**，能被 1/2/4/7/14/28 整除），任何「按 N 块展开、余数走标量尾巴」的写法都会**全套绕过尾巴** —— 与 2026-09-17 在 avx2 上踩的坑同一类：fixture 的形状恰好避开了唯一没被走过的分支<br>三条常驻负向对照（省掉 `-8` / 高低半字节装反 / 丢掉最后一块）的「不可见用例」用**显式名单**约束而非降阈值：不可见的原因多是数学上不可观测（±交替正好相消、整行同值、近零块），名单之外多出一处不可见就红<br>⚠️ **已知弱点（留着，将来改判据时回头看）**：`1e-5 × max(1, |ref|)` 在 `|ref| ≪ 1` 时退化成**绝对** 1e-5 —— `const_row` 一例参考值约 1.8e-5，于是「少读一整块」（贡献约 5e-6）躲过了判据；故单列一条要求：两个真实用例（`real13` / `down_like`）必须判红<br><br>**向量实现（`avx2.mojo`，2026-09-20）**：同一 fixtures、同一期望值、同一判据，换一个核再跑一遍 `check_kernel_against_reference`；九条用例与标量版**逐位一致**（含两条真实数据用例）。切法是**按 `j % 8` 分通道**（块内第 j 个量化值固定配到第 `j%8` 条通道）而不是按连续段切，所以尾巴由块布局本身给定（每块的数据字节刚好两个 8 字节），`cols` 必须是 32 的倍数这一约束与标量版同源。`bench` 侧新增 `q4_0/avx2` 一栏（`scripts/bench_decode_roofline.mojo`）；本机 `down_proj` 4864×896：标量 3.59–3.71 周期/元素 → 向量 **1.19–1.25**（约 3×）。<br>⚠️ **累加仍在 f64**（与标量同一理由），但它作为一个**被测对象**存在：`_matmul_q4_f32acc` 走 f32 累加，实测九条用例最大相对偏差 **8.38e-08**（`real13`；门限 1e-5，低 120 倍），速度再快 1.4×。**没有**把它设为默认 —— 换的是判据层的算术，要先回答「相消最严重的那处还剩多少余量」，今天没人量过。 |
| **CUDA kernel 与标量后端逐值差分**（路线图 1.7：RMSNorm / RoPE / matmul 三个 kernel） | `hardware-blocked` | **阻塞原因**：A100 验证机 `10.107.6.60:3389` 于 2026-09-17 实测连接超时（`scripts/a100.sh` 不可用）；本机 GTX TITAN X 为 Maxwell sm_52，现代 CUDA 栈与 MAX 均不支持<br>**解锁条件**：① A100 可达 ② 环境变量 `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas`（绕过 MAX 对驱动 ≥580 的要求）③ 门为 fp32 容差 1e-5 的**逐值差分，只验正确性不报性能**<br>⚠️ 本轮**未写任何 CUDA 代码**：写完不验的 kernel 比没有更危险，它会被后来者当成可用<br>区分：§1.2 的"端到端 GPU kernel 数值正确"（向量加）已作为远端观测条目单独成立，本机可复现的**算子级** CUDA 差分仍是本行状态 |
| MAX 内核复用（`linalg` / `layout` / `quantization`） | `missing` | 需隔离层；注意 MAX 导入路径有迁移风险 |

## 5. 模型层（L2）

| 能力 | 状态 | 证据 |
|---|---|---|
| HF `config.json` 解析 | `missing` | 当前由 `scripts/dump_model_reference.py` 落成 `config.tsv`；**Mojo 侧不解析 JSON**（为此不去写一个 JSON 解析器，见 §1.3 的 `json` 依赖：接入前不引入） |
| **权重常驻内存门**（加载 + 生成 32 token 后，峰值 RSS ≤ 1.15 × 权重字节） | `verified` | `evidence:tests/unit/test_memory_gate.mojo`（1/1 通过；实测 **1.006×**，预算 1.15×；权重 1.98 GB，峰值 1.99 GB）<br>**这个门能拦住的**：权重被多留了一份（拷贝 / 转置后留着原副本 / 整网常驻解量化成 fp32）—— 那些会让常驻量奔向 2×，门在 1.15× 就红<br>**拦不住的**（不许拿它去宣称）：权重是零拷贝的。mmap 触碰过的文件页与 `read()` 到堆上**都算进 RSS**，两者的常驻量都是约 1× —— **RSS 这个量本身区分不了它们**；零拷贝需要别的证据（缺页次数 / 映射区间），本轮没做<br>分母按**字节偏移去重**：绑定权重在索引里是两条**同偏移**的别名行，不去重会把 embedding 那 544 MB 数两遍，分母凭空大 27%，1.15× 的门实际成了 1.47×<br>**负向对照常驻**：额外 `mlock` 一整份权重 → 峰值涨到 3.97 GB（约 2.01×），门必须变红<br>用峰值（`VmHWM`）而非此刻值：内核随时可回收干净的文件页，"此刻常驻"会往下走，用它判上界等于看内核心情<br>⚠️ 该门**不进 `pixi run test`**（依赖 2 GB 本地导出），入口 `pixi run test-memory` |
| 参数清单加载（fp32 裸二进制 + TSV 索引，mmap 只读） | `verified` | `evidence:tests/unit/test_model_parity.mojo`（2.0 GB 参数以只读映射打开，构造时逐条校验参数名是否存在 —— 缺一个名字在构造期就失败，而不是在生成到第 100 个 token 时）<br>绑定权重（Qwen2.5-0.5B 的 `tie_word_embeddings`）写成**别名行**，指向同一偏移 → 加载侧无需知道 tie 的存在 |
| safetensors 读取（mmap、分片） | `missing` | — |
| GGUF 读取（**从张量偏移反推 block layout，不信任 type id**） | `missing` | 借鉴 `MOJO_STUFF` 的教训 |
| Qwen2.5 架构（0.5B，fp32，prefill + 单步 decode） | `verified` | `evidence:tests/unit/test_model_parity.mojo`（4/4 通过：4 条 prompt 的 logits 余弦 ≥0.999 且 argmax 相等；128 token greedy **逐 token 相等**；增量解码与整段 prefill 一致）<br>⚠️ 该门**不进 `pixi run test`**（标量后端跑完约 6 分钟且依赖 2 GB 本地导出），入口是 `pixi run test-model` |
| **投影分片并发（按输出切，默认关闭）** | `verified` | `evidence:tests/unit/test_parallel_shards.mojo?count=7`：把一次投影按输出切成 N 片并发跑（`std.runtime.asyncrt` 的 `TaskGroup`），判据是**与不切片那次逐位相等** —— 切分不改变任何一次浮点运算（fp32 通路每条输出一个 f64 累加器、q4_0 通路一个 f32 累加器，都在片内闭合），所以「差 1e-7」就是 bug，拿容差去比等于把「片起点算错」也放过去<br>形状表专门挑**除不尽**的：`out` 取 1/3/5/7/11/17/28、片数取 2/3/5/8。真实形状里的 896 与 151936 能被 2/4/7/8 整除，于是「余数没分给任何一片」这条支路**永远走不到**<br>decode（批 = 1）切**输出列**；prefill（批 > 1）因 `dst` 行主序、列不连续而改切**输出行**，偏置在切列时跟着列偏移，切行时整份共享<br>q4_0 通路另有一层：`blocks` 的片偏移是**字节**（每行 `cols/Q4_BLOCK` 块 × 18 字节），按元素算会静默读到别的行<br>**负向对照**：先把结果涂成哨兵值，再故意只算前 N-1 片（漏掉含余数的最后一片），必须被同一套比较抓出来。⚠️ 涂哨兵这步不能省 —— 不涂的话漏掉的那片保留上一次全量算出的**正确值**，正负两例逐位相同，门自己就是绿的<br>⚠️ **默认 `shards = 1`**：不加片时走的就是原来的直呼，语义与性能都不变；要看并发的收益必须由调用方 `set_shards(N)` **明说**（本机 8 核 runq 常年在 6–27，故不默认设成核数）<br>**端到端实测（2026-09-20，`scripts/bench_model_shards.mojo`，Qwen2.5-0.5B fp32 / avx2，32 个新 token，每 token 时间中位数，3 轮交错）**：`shards=1` **162–175 ms/token（5.73–6.16 tok/s）**；切成 2 片 **1.48–1.62×**、4 片 **1.49–1.81×**、8 片 **1.66–1.84×**（同轮配对比值，两个口径同向；非配对区间也不重合，8 片 1.69–1.90×）。**两次运行复现**<br>⚠️ 这份数**只针对批 = 1 的单流 decode**：批大于 1 时分片改切**输出行**，那条路**没量过**，不许拿这个倍数去说"服务吞吐也快这么多"<br>⚠️ 放大倍数是 **1.7×，而 llama.cpp 同机 `-t 1` 5.94 → `-t 8` 25.18 是 4.24×** —— 差的那一截就落在**没被切**的那几段上（注意力 / RMSNorm / RoPE / 残差）与每 token 169 × N 次协程调度，这是下一步该看的地方，不是"已经追平"<br>⚠️ Mojo 1.0.0 只有协程这一套并发设施（无 `thread` / `parallel`）：协程参数只能是**平凡值**（指针与整数），把 `TensorView` 直接传进协程实测会**静默写错地方**（参数槽在 `wait()` 之前失效，而每片算的又是同一个值，于是「对不对」看起来像随机的），故分片视图一律在协程内用 `shard_shape()` 现造；另：`out` 是参数传递约定关键字，不能作参数名 |
| Llama / Mistral 架构 | `missing` | P1 之后 |
| 多模态 | `missing` | 明确 Non-goal（近期） |
| MoE | `missing` | Non-goal（近期） |

## 6. 运行时层（L3）

| 能力 | 状态 | 证据 |
|---|---|---|
| `BlockPool`（物理池 + 空闲链 + refcount） | `verified` | `evidence:tests/unit/test_kv_pool.mojo?count=12`（7 个场景）<br>`src/alofa/runtime/kv/pool.mojo`：定容 `InlineArray` 空闲栈 + refcount；**摘要覆盖整个空间**（refcount 数组 + 空闲链顺序 + 全部请求 + 全部节点），把空闲链初始化倒过来，门**实测会红**（首处 `A=0,1` vs `A=111,110`） |
| `PageTable`（请求 → 块序列） | `verified` | `evidence:tests/unit/test_kv_pool.mojo?count=12`（`KvSpace` 的请求视图：块 id + 每块 token 长度 + 私有尾块标记；**连续追同一块合并成一项** —— refcount 按“视图条目”计，不按“有多少个节点贡献了它”计） |
| `RadixNode` 前缀树（token 粒度映射到 block 粒度） | `verified` | `evidence:tests/unit/test_kv_pool.mojo?count=12`（`src/alofa/runtime/kv/radix.mojo`：匹配是 **token 粒度**、持有是 **block 粒度**；匹配停在中间节点时按 `starts[p] + ntok[node]` 截断，不越读父节点的块数组）<br>✅ **已接入 attention**（2.2）：`runtime/kv/paging.mojo` 把请求页表拷成内核可读的 `PagedTable`（含**块内起始槽位**），`kernels/cpu/paged.mojo` 直接用 `(block, start, length)` 寻址；门断言"重放出的表"与夹具逐项一致，于是拼接这一环本身也被验过 |
| 三视图共享 refcount 的一致性 | `verified` | `evidence:tests/unit/test_kv_pool.mojo?count=12`（`check_invariants()` **从两个视图重算**每个块的持有数，再与 `pool.refcnt` 逐块比对；另加 `n_free + used == MAX_BLOCKS` 与空闲链成员精确匹配）<br>**负向对照常驻**：多一次 `retain`，不变量门必须报 >0（已实测报红） |
| 节点分裂（metadata-only，不新分配块） | `verified` | `evidence:tests/unit/test_kv_pool.mojo?count=12`（分裂前后 `pool.used` 不变、节点数 +1；父节点释放自己不再覆盖的尾块，子节点**沿用**原块 —— 前缀共享的收益来自元数据重排，不来自拷贝） |
| KV 寻址零堆分配（定容容器 + 源码门） | `verified` | `evidence:tests/unit/test_kv_pool.mojo?count=12`（`runtime/kv/` 下 4 个源文件全部扫构造点，不得出现 `List[` / `String(` / `Arena(` 等；常驻红测 `tests/fixtures/bad_kv_alloc.mojo` 必须被判违规）<br>⚠️ **边界同 §7 调度器那条**：拦得住“加一个会增长的容器”，拦不住 libc 小块分配，不等于进程 RSS 不动 |
| 频率感知淘汰（2Q + 复合评分） | `missing` | 创新点 2；**须有回放数据才可上线** |
| 采样器（top-k / top-p / min-p / 温度 / 重复惩罚 / logit bias） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo?count=11`（9 条逐阶段用例 × 32 次采样）<br>温度 / top-k / top-p / min-p / repetition penalty 用 **HF 4.41 的真实 warper 与 processor** 交叉核对**存活集合**（`scripts/dump_sampler_reference.py` 按其 `_get_logits_warper` 的自身顺序施加）<br>⚠️ **logit bias、frequency / presence penalty 在 HF 4.41 里没有对应 processor**，采用 OpenAI / vLLM 加法语义，属**语义自证**（由正负 bias、跨 top-k 边界、被 top-p 裁掉等边界用例钉住），**不是 HF 对齐**；两种重复惩罚语义字段名不共用<br>PRNG 自研 SplitMix64（`core/rng.mojo`），参考侧同一整数算法 → 采样出的 **token id 精确相等**（非容差相等） |
| 采样分布正确性（卡方 / TVD 检验） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo?count=11`（固定均匀序列 50000 次采样：TVD 0.00766 ≤ 0.05，卡方 23.05 ≤ 80）直击 `llm-mojo` 的分布错误<br>**负向对照常驻**：用温度减半的诱饵分布跑同一检验，TVD 0.4253 必须被判失败 —— 一个不会失败的拟合门等于没有门 |

## 7. 执行编排层（L4）

| 能力 | 状态 | 证据 |
|---|---|---|
| 纯函数调度器（单一 token 预算） | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=17`<br>`pixi run mojo run -O0 -I src tests/unit/test_scheduler.mojo`<br>无 I/O、无时钟、无模型、无权重依赖；`step(input) -> Action` 是唯一入口，于是调度边界可以脱离模型被测（创新点 3） |
| 调度器自身零堆分配（定容容器 + 源码门） | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=17`（类型层面：所有容器是编译期定长的 `InlineArray`；源码门扫描 `src/alofa/engine/scheduler.mojo` 不得出现 `List[` / `String(` / `Arena(` 等构造点，并有常驻红测 `tests/fixtures/bad_alloc.mojo` 必须被判违规）<br>⚠️ **这条证据的边界**：能拦住“给调度器加一个会增长的容器”，**拦不住** libc 里的小块分配，也**不等同**于进程级 RSS 不动 —— 账本就按这个口径写，不夸大成“进程零分配”。录 trace（`engine/trace.mojo`）**会**分配 String，所以录制是 `step` 之外的可选动作 |
| chunked prefill | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=17`（300 token 的 prompt 跨 19 拍切片：首尾相接、不重叠、每片不超过 `max_chunk`；同一拍不得给同一请求两个 chunk） |
| 抢占（重计算） + 抢占计数指标 | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=17`（并发抢占风暴 / KV 水位临界两个场景；被抢占者 KV 全作废并回到等待队列，累计抢占次数作为 `Action` 字段逐拍比对 —— 它是容量告警指标，不是调试字段）<br>⚠️ 只抢占 **RUNNING** 请求：抢占“半截 prefill”的请求会让 prefill 永远无法完成，那是伪装成策略的抖动<br>⚠️ 池子小到连一次 decode 增长都装不下时报 `capacity` 具名错误，**不静默丢 token**（有专门断言） |
| 调度 trace 录制 | `verified` | `evidence:src/alofa/engine/trace.mojo` + `tests/fixtures/scheduler/*.trace`（格式：整数 + 定长字段，**不含浮点**；水位用千分数而非比例 → 逐字节门不会退化成容差门） |
| 调度 trace 重放 + 极端场景断言 | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=17`（6 个场景与 `scripts/dump_scheduler_reference.py` 这份**独立 Python 实现**逐字节相同；三条常驻负向对照：改坏的 trace、换一种抢占顺序、给调度器加堆容器，三者都必须被判红）<br>6 个场景：超长 prompt、并发抢占风暴、预算耗尽、0 预算、取消竞态、KV 水位临界 |
| 延迟护栏（最大等待拍数） | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=17`（构造“队首长 prompt 每拍吃光预算”的最小复现：护栏生效时第 4 拍必须给短请求；把 `max_wait_ticks` 调到 99 该断言**实测会失败**）<br>护栏只在**等待者之间**插队，不越过 decode：它防的是“前面有个超长 prompt”，不是“预算被 decode 占满” —— 后者说明并发已饱和，插队只会把等待转嫁给已经占着 KV 的人 |
| 批张量池（稳态零堆分配） | `verified` | `evidence:tests/unit/test_batch_pool.mojo?count=12`<br>`src/alofa/engine/batch.mojo`：借用拿到的是**句柄**而不是指针（释放后同一个槽位会给别人）；池子不拥有内存，构造时收一个指针加容量；全部簿记是编译期定长的 `InlineArray`，**没有空闲链表** —— 每次放置都从“活着的借用”重导出可用空隙，于是不存在两套会互相漂移的账<br>7 个场景与 `scripts/dump_batch_reference.py` 这份**独立 Python 实现**逐字节相同：分配是“决定”不是数值，两种放置之间不存在“差一点点”<br>两条不依赖参照物的性质：**活着的借用不共享任何一个字节**（每个区间写自己的标记再逐字读回 —— 把放置往旁边挪一格，这条实测会红），**峰值不漂移**（20 个相同执行步之后 high-water 与第一步相同；否则稳态不稳，之后测的吞吐就是关于另一个池子的数字）<br>**拒绝必须是具名错误**：池子填满后隔一个释放，剩 131072 字节可用而最大空洞只有 16384 —— 借两块必须报 `capacity` 而不是绕回去或跨两个空洞凑；第 25 个活的借用同样是 `capacity`；重复释放是 `double_free`<br>3 条常驻负向对照：`alt.trace`（同一组操作改用 best-fit 放置，且**要求它至少挪动一处偏移**，否则那 6 个逐字节比对只是在验文件格式）、`bad.trace`（一处偏移挪一格）、给池子加堆容器的 `bad_batch_alloc.mojo`<br>⚠️ 证据的边界：拦得住“给池子加一个会增长的容器”，**拦不住** libc 的小块分配，也不等于进程级 RSS 不动；且这一层本身**不比对数值**（借到的字节里算得对不对由下面两行负责） |
| 批组装（每请求一段连续行） | `verified` | `evidence:tests/unit/test_batch_pool.mojo?count=12`<br>`BatchSlots` 给每个请求一段**连续**、且**不与别人重叠**的行：两个请求共用一行时，某一层会从别人的 token 上读出自己的激活，产出的每一个数都看起来合理 —— 与分页内核读错块是同一种失败<br>断言从外面重算：逐行统计所有者，**重复覆盖与未被覆盖都必须为 0**；同一请求被加两次是具名错误（一次 add 会让它拿到两段行，而某一层只会读其中一段）<br>⚠️ 只在**行归属**这一层成立；“注意力按请求分块”由 executor 接走（下一行） |
| 批执行器（注意力按请求分块） | `verified` | `evidence:tests/unit/test_batch_executor.mojo?count=12`<br>`src/alofa/engine/executor.mojo` 把 2.5 的两层接进前向：每个请求拿到一段**连续行**，每一行被交给注意力时都带上**自己的** K/V 基址、`upto`（能看到的最末一个 key）与 `pos`（rope 用）—— 于是“不同请求的行互相 attend”不是靠一张可能被丢掉的 mask 挡住的，而是**地址上不存在**：一行从来拿不到别人的地址<br>6 个场景与 `scripts/dump_batch_executor_reference.py` 这份**独立 Python 实现**逐字节相同（ADD/FEED/DROP/PLAN/ROW/FIN/NEXT 全序列 + 每步摘要 `d=`），摘要覆盖全部请求槽与全部行，不变量**每一步从外面重算**（缺陷计数不为 0 即红，而不是只在结尾查一次）<br>3 条常驻负向对照：`alt.trace`（换一种行分配策略，且**要求它真的不一样**，否则 6 个逐字节比对只是在验文件格式）、`bad.trace`（某一行的可见窗口挪一格）、给忙碌循环加堆容器的 `bad_executor_alloc.mojo`<br>每一步向 2.5 的池子借 15 块、步末全部归还：实测 `used==0`、`n_live==0`、峰值不漂移<br>⚠️ 边界：这一门**不比对数值**（注意力算得对不对由 §7 的分页门与下一行的批一致性门负责），它验的是行归属、每行的窗口与簿记 |
| 批一致性（批大小 1/2/4/8 与单请求逐 token 相同） | `verified` | `evidence:tests/unit/test_batch_forward.mojo`（4/4；重门，需要 2GB 权重，`pixi run test-batch-forward`，故意不进 `pixi run test`）<br>同一批 prompt 走两遍：**一批 N 条** 与 **一条一条跑**（`QwenForward.prefill`/`step`，也就是 `test_model_parity` 拿去和 Hugging Face 对过的那条路径），greedy 解码、逐 token **相等** —— greedy 让“第 3 个 token 不同”就是一个不同，而不是差一点点<br>N = 1 / 2 / 4 / 8：8 条请求的 prompt 合计 88 行 > 行块 64，所以这一门**真的把一次 prefill 切成两拍** —— 短的一拍也必须是对的一拍<br>2 条常驻负向对照：**不同 prompt 必须解出不同续写**（否则“批次与单请求一致”对任何实现都成立，包括不看输入的实现）；**交换两条请求的 prompt 必须被察觉**（交换后既**不等于**该槽位的基线、又**等于**它实际拿到的那条 prompt 的基线）—— 这才让逐字节比对成为“行归属”的证据<br>⚠️ 参照物是**本树的串行前向**，与批路径共享 kernel：这是刻意的，被比较的是**编排**（行归属、每行的 pos、每行的窗口、KV 区域），而串行路径只有一条请求、不可能在这些上出错；参照物本身对 Hugging Face 的一致性由 §5 的模型门负责 |
| 引擎循环（调度器 ↔ 批执行器接线） | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=16`（已进 `pixi run test`）<br>`src/alofa/engine/core.mojo` 把 2.0 的调度器与 2.5b 的批执行器接成一个忙碌循环：调度器出**决定**（谁 prefill、给 `[start, end)` 这一段、谁 decode、谁被抢占），执行器出**行**；被验的只有两者之间的**翻译** —— 切片喂给谁、prefill 结束那一拍白送的第一个 token 与之后 decode 出来的 token 怎么拼成同一份 transcript、抢占后重算要作废什么<br>argmax **由测试注入**（每槽一个整数，不跑模型）：贪心 argmax 是一行代码，这一门要验的是**时序**；也正因为期望值写成 `expected_token(请求, 第几个 token)` 而与「第几拍产生的」无关，同一份期望才能同时管住「抢占后被推回 prompt、重新生成一遍」的请求<br>3 条常驻负向对照：① 抢占场景**断言 `preempt_total > 0`** —— 声称「抢占安全」却从头到尾没抢占过的门，是穿着戏服的 happy path；② **把请求从执行器手里抽走再要一拍，必须报 `invalid_argument`**（静默服务一个空请求更省事，也更会藏 bug）；③ 给循环加堆容器的 `bad_executor_alloc.mojo`<br>每一拍都从两边重算「谁还活着」：引擎说谁 resident、执行器说它握着谁，二者不一致的那一拍**照样产出一串看起来是 token 的数**<br>⚠️ 边界：这一门**不比对数值**（续写得对不对由上一行的批一致性重门负责），也不意味着 KV 物理块池已接入 —— 执行器用的是自己构造时写死的 KV 区域（见下一行） |
| KV 物理块池（含 `freed_blocks` 这类外部释放） | `verified` | `evidence:tests/unit/test_kv_room.mojo?count=10`（已进 `pixi run test`）<br>`src/alofa/engine/kv_room.mojo` 是唯一做这层翻译的地方：调度器**数**块但不拥有块，`runtime/kv` 拥有块但只会说 radix 树操作。三件只活在这一层的事：① prefill 到达是**切片**，一次入场是多次 `append_tokens`，而「prompt 有多长」是另一件事实（它决定半个 prompt 不许进缓存）；② 完成的请求**换主人**（`commit` 发布到树），块变成「没有主人的占用」= 前缀缓存；③ 缓存的块只从**一扇门**回来 —— 驱逐是引擎的决定（只有引擎知道压力），还回多少**实测**（缓存块数前后之差）而不是记账<br>引擎侧接线（`src/alofa/engine/core.mojo`）：prefill 首片 `admit`、续片 `grow_to`、`settle` 后按「prompt + 已生成」对齐长度（给目标值不给增量）、完成 `publish` 再 `drop`、抢占与取消直接 `drop`；归还经 `SchedInput.freed_blocks` 回报调度器，两条规则都在调度器决定之前执行：**缓存让位给活着的请求**（缓存 ≤ 容量 − 持有）与**水位**（缓存顶高水位时先回收，免得调度器为还不了的块去抢占）<br>5 个场景与 `scripts/dump_kv_room_reference.py` 这份**独立 Python 实现**逐字节相同（`u/f/c/d`），且每拍从视图重算不变量、断言 `used + n_free == MAX_BLOCKS`；3 条常驻负向对照：`bad_room.trace`（改坏一拍）、`alt_room.trace`（换 prompt，必须判红 —— 否则只验了格式）、`bad_room_alloc.mojo`（给房间加堆容器）<br>不依赖参照物的性质：相同 prompt 的第二条 `last_matched == 32` 且 `last_fresh == 0`、池子用量不变；发布后 `used` 不降、回收后块真的回池且上报数等于实测；**未发布的 drop 必须真的回池**；`reclaim` 对活着的请求必须还回 0<br>⚠️ 边界：`runtime/kv` 并发上限 `MAX_REQUESTS`（8）、单序列上限 `MAX_SEQ_TOKENS`（64），第 9 条报 `capacity`、超长报 `out_of_range` —— 限制被**断言**而非绕过（悄悄少给几块会在几拍后变成「少一个答案」）。调度器的占用仍是**算术**的、房间的才是**物理**的，二者不要求相等（前缀共享让物理更少、节点粒度让物理可能更多），物理池满时房间具名拒绝。⚠️ 执行器的 KV 已从这张块表取地址（见后两行）；房间给了块之后，前向读到的位置由**表**决定，不是由算术决定 |
| 分页注意力接入引擎循环（前向从块表取地址） | `verified` | `evidence:tests/unit/test_paged_scatter.mojo?count=5`（已进 `pixi run test`）+ `tests/unit/test_batch_forward.mojo`（重门，真实 0.5B fp32 权重）<br>块池布局：一个块**持有每一层各一个槽**，所以请求的表在每层都叫同一批块号 —— `engine/executor.mojo` 把层号折进块 id（`layer * MAX_BLOCKS + block`），于是整个块池只有**一个**视图、在构造时建好，忙碌循环里不再构造形状（每步每层建一个 `List` 正是这一层的源码门要挡住的事）<br>`kernels/cpu/paged.mojo` 新增 `paged_scatter`：写**经过**表，与读经过同一张表。写按算术放（`j // block_size`）会把 token 放进「它若不共享前缀本会占用的块」，之后每一步都是从别人的历史里算出来的数；写越界报 `out_of_range` 而不是截断（截断是悄悄变短的上下文）<br>房间与执行器的交接只有一处：房间 `page_table()` 拷出表 → 引擎 `sync_page_table()` 在 `admit` / `grow_to` **之后立刻**交过去；表按 `hist + n` **裁剪**后再用 —— 房间可以为还没到的 token 预留整块（切片 prefill），而注意力不许读没人写过的位<br>批一致性重门跑在**乱序块号**下：块刻意不按连续区域的顺序排，所以「批与串行逐 token 相同」这句话是关于**页表**的 —— 前向若按算术取地址，数就不同<br>⚠️ 边界：`rows_view` 构造视图（形状是 `List`）在执行器 `forward` 里仍然存在 —— 块池的**账**是零分配的，视图构造不是 |
| 共享前缀只算未命中的那一段（前缀缓存省的是算术） | `verified` | `evidence:tests/unit/test_batch_forward.mojo`（6/6，真实 0.5B fp32 权重，重门）+ `tests/unit/test_batch_executor.mojo`（12/12，已进 `pixi run test`）<br>引擎层**端到端**已验（`tests/unit/test_engine_core.mojo` 11/11）：同一个 prompt 提交两次，第二次的第一拍**只跑一行**且 `history_of == 6` —— 此前这条链路只是「编译通过 + 单元绿」，房间 → 引擎 → 执行器这一段没人跑过；常驻对照是同一引擎里的冷 prompt：7 行、history 0。| KV 池高占用下的正确性 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=16`：四条请求同时在池（峰值 = 四条块数之和），全部跑完且**逐 token 等于逐条跑的基线**；每拍重算 `used + n_free == MAX_BLOCKS`、房间 `invariants()==0`、两本账 `defects()==0`<br>⚠️ **不声称 95%**：2026-09-18 撤回前一天记的「峰值 111/112」——那个数字是被下面那行的记账 bug 造出来的（房间白发整条 prompt 的块），修好后同一场景只到 99/112，边界见下面两行 |
| 分块 prefill 下的两本账 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=16`：**已修**：引擎原来在 `settle` 里把 KV 序列长度设成「整条 prompt + 已生成」，无视 prefill 只喂到第 16 个 token —— 房间因此白发整条 prompt 的块（实测第一拍：房间 15 块、调度器账 4 块；七条跑下来差 10 块），水位 950‰ **全程不触发**，池子只靠房间抛 `capacity` 兜住。现在按「已喂到的位置」grow，分块下两本账差 ≤ 1（`test_the_scheduler_and_the_room_count_the_same_blocks`；旧行为下该门差 10 块、红）|
| 物理池 >95% 且能跑完的场景 | `missing` | 在当前实现下**不可达**：`MAX_ROWS=64` 只能逐条 prefill，先完成的先释放，分块下峰值 98/112（87%）；要顶满就得让请求长驻留，而驻留总量一旦高过水位，抢占就在两条请求之间来回抢、谁也完不成（七条各生成 8 个 token：512 拍仍不空闲；容量预算 48 / 阈值 45 下四条同样活锁）。解锁条件：让抢占真正缓解而不是循环——受害者重算时应优先拿回块，或水位只在「有等待者需要块」时触发 |
| 缓存让位与缓存账的时序 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=16`：五条请求分一个装不下的预算（60），缓存必须让位，否则排队的请求永远拿不到块。修了两处：① **缓存占死预算**——yield 只按「已在跑的」算，缓存把预算吃满，四条请求在剩下的块里互相抢占，有一条一个 token 都没生成；现在按 `blocks_used + blocks_wanted()` 算，缓存只留别人用不到的。② **引擎交回调度器尚未记账的块**——上一拍发布的序列要等本拍 `step` 才进 `cached_blocks`，reclaim 却发生在 `step` 之前，于是下一拍的 `freed_blocks` 大于调度器认为的缓存，抛 `ERR_INVALID_ARGUMENT`；现在只交回调度器已记账的部分。负向对照：把 ① 改回旧算法，该门红在「512 拍从未空闲」 |
| 共享前缀下的缓存账 | `missing` | 调度器 `release(to_cache=True)` 按「每条已完成序列自己的块数」累加缓存，而房间的前缀树去重后只占一份 → 两本账不同源（调度器高估）。**首 token 归属那条已修**（见下），剩下的只有去重这一条。**回滚过一次**「让引擎把差额延后一拍用 `freed_blocks` 报出」：两本账当时对齐了，但随后撞 `id list overflow`，且回收与修正同拍叠加会超账。解锁：缓存占用数只能由房间报告，调度器不得自行推算——`SchedInput` 需要一个独立的「缓存增量」通道 |**第二次尝试也已回滚**（2026-09-18）：给 `SchedInput` 加绝对值通道 `cached_now`，房间每拍报真值覆盖调度器的和。失败原因不是实现细节：调度器的 `release` 比房间的 `publish` **晚一拍**（完成消息延后送达），于是「对齐到上一拍真值 + 本拍 release 整条」仍在叠加——实测对齐到 36 之后又加上两条的 24，得 60，而房间是 48；补 `cache_pending` 让引擎多跑一拍也没能把 60 降下来。真正的解锁是「调度器不再自己维护缓存账」，而这跟「调度器是纯整数函数、重放门不依赖房间」直接冲突（参考实现没有房间，报不出真值，那时调度器又必须能自己算）→ 属于架构取舍。
| 过载 + 长 prompt 的抢占活锁 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=16`（两条门：抢占确实发生，且受害者重算后仍跑完全程）<br>`evidence:tests/unit/test_scheduler.mojo?count=17`（trace 逐字节：抢占发生后拍末 blocks_used 回到阈值内、无一拍越过硬容量）<br>原探针（未固化为门）：容量 10 块、4 条请求各 20 prompt（16 行一拍 → 跨两片）+ 3 生成、块 4 字节 → 300 拍、**190 次抢占、0 个 token 产出**（⚠️ 该数字取自 `watermark_permille=8`，即水位 **0 块**的病态配置，不是默认——`SchedConfig` 第 5 个参数是水位千分比，早先误当成了预算）。**默认水位 950 重测仍不收敛**：容量 10、块 4、4 条 20 prompt + 3 → 400 拍、`outs=3 3 1 1`、抢占 258 → 活锁在默认水位下**依然成立**，只是程度较轻（两条能跑完）；前提是池子装不下在飞的序列（4×6=24 > 10），而准入与晋升都不做容量规划，于是「抢占归零 → 重喂」变成循环。**根因**（逐拍探针订正）：第 5 步「晋升」无条件把 `done >= prompt_len` 的请求全转成 `ST_RUNNING`，**不看池子能否容纳它们的 decode 增长** → 同拍多条一起晋升 → 第 6 步 decode 时 `ensure_room` 装不下 → 抢占（`scheduler.mojo:570-571` 把 `done`/`generated` 归零）→ 打回 `ST_WAITING` 从头再喂 → 循环。逐拍证据：60 拍内所有请求 `state` **始终为 1（WAITING）**，从未进入 RUNNING；某条 `done` 刚到 20，下一拍即 `done=0` 且 `preempt_count+1`。`max_wait_ticks` 越小 → 强制 prefill 越密集 → 同拍晋升越多 → 抢占越多，故默认 8 比 900 更糟。**不是记账问题**（两本账一致）。⚠️ 早先写的「喂一半被抢占」是**错的**：抢占只针对 `ST_RUNNING`（`:564`），部分喂的是 `ST_WAITING`，不会被抢占。修法方向：晋升加容量门槛（装得下整条序列才晋升）——属调度器预算类改动，但会改单拍决策 → scheduler 的 14 条 trace 需重导（Python 参考同步改） |
| 已发布序列的块数 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=16`：调度器按 `done + generated` 算一条已发布序列占多少块，而 `generated` 数的是 decode 拍——续写的**第一个 token 由「把 prompt 喂完的那一步」产出，不算一拍**，于是每条少记一个 token；跨块时少一整块（实测四条：44 对 48）。改按 `prompt_len + max_new` 记，并顺带补齐 `blocks_used`（它留着的是按拍算的旧数，否则池账比缓存账少同样多）。新门 `test_a_published_sequence_is_counted_whole`：37+8=45 token 是 4 字节块的 12 块，44 是 11——**块粒度 8 时两者都是 6 块，同一个 bug 会溜过去**（现有那两个门正是块粒度 8，当时全绿） |
| 引擎空闲判定与块释放 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=16`：批里最后一条请求完成后，消息要到下一拍才到调度器，`has_work` 却只看引擎自己的 state → 它宣布空闲，那条请求永不 `release`，实测 11 块永久占用（每批泄漏一次）。对称地，房间已回收的缓存块若没被下一拍带走，调度器会一直为它们记账（实测 32 块）。修：`has_work` 也认 `n_report` 与 `room.freed_pending`；新增 `room.take_freed_upto()`——回收按节点整块释放，可能多于请求量，多出的留到下一拍再报。空闲时两本账归零。负向对照：去掉这两个条件 → 该门红在「仍有块被持有」 |
房间 `admit` 时就知道重合多少（`last_matched`），引擎把这个数交给执行器 `add` 的 `matched`：命中的 token **入队但不建行** —— 历史从 `matched` 起算，队列里只剩没算过的那些。省下来的是**行**，行就是算术<br>**最后一行永远要算**：它的 logits 是第一个生成的 token。一个被完整命中的 prompt（`matched == n`）跑一行，不是零行 —— 零行就没有 logits，请求无从开口；`matched > n` 具名拒绝（`out_of_range`）<br>证据是端到端的：同一个 prompt 跑两遍，第二遍沿用第一遍的**同一批块**（`drop` 只忘地址、不清字节，这正是前缀缓存的定义）、`matched = n - 1`，跑一行，生成的 token 与串行**逐 token 相同**。配套常驻负向对照：谎报命中（`matched = n - 1` 但指向没人写过的块）必须产出**不同**的 token —— 否则上面那条可以因为「压根没读缓存位置」而白过<br>⚠️ 边界：命中的字节必须与本地计算**逐位相同**才成立（同机、同权重、同路径、同位置 —— 换 backend / 跨机未验）；执行器**不校验**块里真的是那段前缀，它信任房间 —— 谎报由上面的对照拦，不由类型拦 |
| paged attention（block table 索引） | `verified` | `evidence:tests/unit/test_paged_attention.mojo?count=11`<br>两类断言用两把尺子：**寻址用逐位相等**（`paged_gather` 与参照导出的连续行逐位一致；分页 kernel 与连续 oracle 逐位一致 —— 没有重排就没有"差一点点"的余地，差一个 ulp 就是地址算错），**公式用 1e-5 容差**（期望值来自 `scripts/dump_paged_reference.py`，与 Mojo 不共享任何代码）<br>**5 条常驻负向对照**：改坏一个元素的 `expected_bad.tsv`、用错 GQA 映射（`h % n_kv_heads`）的 `expected_hmap.tsv`、给 kernel 加堆容器的 `bad_paged_alloc.mojo`、把每段 run 的尾槽灌成垃圾值后输出必须逐位不变、共享同一块的两个请求必须读到同一段字节<br>⚠️ 夹具里的表是 `KvSpace` 对 2.1 真实操作序列重放出来的（8 例），另 3 例是 `syn_*` 块内偏移用例：当前树只从根共享、请求都从槽位 0 分配，**块内起始的 run 走 `KvSpace` 造不出来、走 `PagedTable` 造得出来**，内核就必须对它负责 |
| 垂直切片（一句真文本走完 tokenizer → model → engine → sampler） | `verified` | `evidence:tests/unit/test_vertical_slice.mojo`（4/4；重门，需 1.9GB 权重，`pixi run test-slice`，故意不进 `pixi run test`）+ `src/alofa/cli.mojo`（`pixi run generate`）<br>**这一行补的是一个真实存在的洞**：此前每一层都有自己的门且都是绿的，但**没有任何一处把它们串起来跑过** —— `test_engine_core.mojo` 的文件头自己写明 argmax 由测试注入。2026-09-17 预告过"各层各自绿、拼起来崩到 0/512"，这一行就是把那条路径固定成每天能走一遍的东西。<br>实测（真权重，scalar 后端）：`"The capital of France is"` → 贪心 `" Paris. It is the largest city in Europe and the second largest in the world"`；采样（温度 1.0、seed 固定）`":\nA: Paris B: not sure C: london D: BERLIN"`，两条路都说到 Paris。<br>**负向对照（本行的关键）**：温度 0.01 的采样必须**逐字等于**贪心 —— 若 `run_sampled` 悄悄退化成 argmax（logits 取错行、`build` 没被调用），"说出 Paris"照样全绿而采样路径一次都没生效过；温度趋零时分布塌到 argmax 上，两条独立路径必须给同一个答案。而温度 1.0 时两者**不同**（上面两段文本），一正一反才构成完整证据。<br>**本行真正的收获是两条契约，都不是猜测、都是撞出来的**：<br>① `tick` 内部硬编码 `model.argmax`，sampler 此前**没有任何介入点**。而采样循环**不能**加进 `core.mojo`：那个文件既是 `step_path_sources()` 之一（`test_core_tensor` 在其中查 `List[Int]()`），又是零分配门的 `SOURCE`（`test_engine_core` 在其中查 `List[`），而 `Sampler.build` 只收 `history: List[Int]`、**没有指针重载**。实测在那里加一个 `history_of` 会让**两个门同时变红** —— 门是对的，那是真承诺，于是循环内联在 `src/alofa/cli.mojo`，`core.mojo` 一字未改（236 项与全部 trace fixture 不受影响）。<br>② **同名常量两个取值**：`MAX_BATCH` 在 `engine/batch.mojo` 是 **8**、在 `engine/scheduler.mojo` 是 **32**；`core.mojo` 经 `executor` 拿到的是 **8**。垂直切片最初从 `scheduler` 导入，拿 32 去遍历只有 8 个槽的 `ex.live`，实测崩在 `Assert Error: index 8`。这不是类型能挡住的（`range()` 两端都是 `Int`），只能靠"从哪导入"这一行注释守住 —— 已写进 `src/alofa/cli.mojo`。<br>⚠️ 由此留下一个**未收回的边界**：这条采样路径**不在** §7 的零分配承诺内（每拍为每个活跃请求建一个历史列表），它只服务单请求 CLI；要进忙碌循环，必须先给 `Sampler` 一个指针版 `build`，那时这个循环才搬得进引擎。<br>⚠️ 边界：只验 **1 条请求、16 个 token、scalar 后端、单条序列 37 token**。不验 AVX2、不验批、不验流式输出（逐 token decode 的 UTF-8 边界未验），也不验生成质量 —— 0.5B 模型答得对不对不在这条门的职责内，它只负责"链路通" |

## 8. 服务层（L5 / L6）

| 能力 | 状态 | 证据 |
|---|---|---|
| **依赖 `flare` reactor 作为事件循环** | 已决策（未接入） | 原"自研 epoll"方案被推翻；`flare` 0.2.0 已提供 reactor/scheduler/timer_wheel/watchdog/reuseport/io_uring（§1.3） |
| reactor 线程 + engine 独占线程的双线程模型 | `missing` | 见 `02-architecture.md` §6.1；调度决策留 reactor 以保持可重放零锁 |
| libc 事件循环原语（自研**回退**路径） | `verified` | `evidence:tests/capability/test_libc_ffi.mojo?count=5`：`socket`/`epoll_create1`/`timerfd_create`/`eventfd`/`SO_REUSEPORT` |
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
| 预分词器（**无正则依赖**，手工实现 Qwen2 / GPT-2 规则） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=3`（4560 条参考用例逐 id 全等）<br>参考的两个项目死在这里；这里用**手写分支匹配器**替代正则引擎：收缩形式、字母串、数字、标点、换行、尾随空白六类按最左优先取最长，贪心与回溯由代码显式表达 |
| NFC 归一化（分解 / 规范排序 / 重组，含 Hangul 与多元分解） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=3`<br>**表是生成而非手写的**（`scripts/gen_unicode_tables.py`）；多元分解（如 U+1E5D）与组合类排序漏一条，整句就会落到不同的 merge 序列上 |
| BPE（**按 rank 合并**，同 rank 取最左） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=3`<br>贪心从左到右会挑错对；这里每轮全量扫描取 rank 最小者 |
| added token 切分（**最长匹配**） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=3`（语料含句中出现的 added token） |
| 词表 / 合并表加载（**离线 fixture，不启 Python**） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=3`<br>fixture 由 `scripts/dump_reference.py` 生成，但测试进程只读 TSV —— 差分门可在任何机器上重跑 |
| id → 文本还原（含非法 UTF-8 替换） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=3`（往返用例逐条对比**归一化后**的输入） |
| 4560 条差分用例（与 HF 逐 id 一致 + 往返） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=3`（0 处不一致）<br>**P1 门**：对不一致**零容忍**（0/4560），而不是容忍 0.1% —— 差分断言的是字节级等价 |
| Unigram | `missing` | — |
| WordPiece | `missing` | — |
| `tokenizer.json` 加载 | `missing` | 当前只读 off线 TSV fixture；直接读 HF `tokenizer.json` 尚未实现 |
| GGUF vocab 加载 | `missing` | — |
| `chat_template` 渲染（Jinja 子集） | `missing` | 借鉴 `molla`：语义来自模型的 `chat_template`，不手写每族渲染器 |

## 10. 验证体系（正交支柱）

| 能力 | 状态 | 证据 |
|---|---|---|
| oracle 差分框架（logits 余弦 / argmax） | `verified` | `evidence:tests/unit/test_model_parity.mojo`（余弦在 `Float64` 中累加：151936 维的 fp32 累加误差与被测间隙同量级时，相似度本身就不可信）<br>参照物是**导出的 fixture 而非实时调 Python** —— 被测代码变了，答案不会跟着变 |
| greedy 逐 token 相等校验 | `verified` | `evidence:tests/unit/test_model_parity.mojo`（4 条 prompt × 128 token，逐 token 相等）<br>这是三条判据里最强的一条：能通过余弦却在第 30 步分叉的实现，在这里一定失败 |
| 逐算子中间张量差分（定位用） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`<br>每个算子各自持有"参考实际看到的输入"与"参考实际产出的输出" → 失败时**只有该算子的测试红**，而不必在整网里二分 |
| top-k 集合比较 / 分布检验（卡方 / TVD） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo?count=11` 存活集合用 **FNV-1a 指纹做零容差比较**（只比数量会放过"对的个数、错的成员"）<br>并列取值的合成行**只比集合不比顺序**：`torch.sort` 在并列值上顺序未定义，逐元素比会 flaky；而真实 logits 几乎不并列 → 把 `>=` 写成 `>` 在真实数据上测不出来，在**量化后**一定会并列。该门已用变异测试验证过会红（改一个比较符 → 4 个测试失败） |
| 分布检验（卡方 / TVD） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo?count=11`（序列固定 → 断言确定性，**不会 flaky**；TVD / 卡方同时与参考值比对，不仅与阈值比对） |
| roofline **骨架**（采集 + 利用率报告接口，峰值由调用方传入） | `verified` | `evidence:tests/unit/test_verify_roofline.mojo?count=15`<br>**不内置任何机型常数**：峰值是构造参数，≤0 直接报错 → 杜绝"抄规格书当实测" |
| roofline 报告（真实 kernel / 真实硬件） | `missing` | 骨架已在上一行通过验证，但尚未接入任何算子 → **未产出任何性能数字** |
| 性能回归门（PR 级） | `missing` | — |
| 能力账本 CI 校验 | `verified` | `evidence:tests/capability/test_ledger.mojo?count=7`<br>`pixi run check-ledger`<br>**自证循环**：这一条的 evidence 正是校验账本本身的那个测试 —— 账本用自己声明的门来证明自己可信 |

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
- **2026-09-16** —— **§9 Tokenizer P1 门打通**：预分词器 / NFC / BPE / added token 切分 / fixture 加载 / id 还原 / 4560 条差分用例七项由 `missing` 升为 `verified`，证据统一指向 `tests/unit/test_tokenizer_parity.mojo`（3/3 通过，4560 条参考用例 0 处不一致、往返 0 处不一致）。三个曾经让这一切对不上的坑，均已固定在测试里：① 参考词表是**字节级字母表**，必须反查回真实字节（否则空格 token 落成 c4a0，单字节 token 全部缺失）；② 组合表必须支持**三元以上分解**（U+1E5D 之类），漏一条就 354 处不一致；③ 往返要用**归一化后**的输入比对，不是原始输入。**一个 Panic/断言也没放宽**：不一致容忍度是 0，不是 0.1%。
- **2026-09-17** —— **P1 数值门打通（端到端前向闭环）**：§2 `scalar (fp32)`、§4 的 RMSNorm / SwiGLU / RoPE / GQA 因果注意力 / fp32 matmul、§5 的 Qwen2.5 架构与参数清单加载、§10 的差分判据（logits 余弦 + argmax、greedy 逐 token 相等、逐算子中间张量）由 `missing` 升为 `verified`。实测：4 条 prompt 的末位 logits 余弦 ≥0.999 且 argmax 全等；**128 token greedy 逐 token 相等**（4 条 prompt 无一处分叉）；逐 token 解码与整段 prefill 落点一致。入口 `pixi run test-model`（约 6 分钟，依赖 `scripts/dump_model_reference.py --full` 的 2 GB 本地导出，**不进 `pixi run test`**）。
- **2026-09-17** —— **先做垂直切片再铺开，是这次唯一的顺序决策**：先只导出第 0 层的算子边界（9 MB fixture）→ 13 个算子测试全绿 → 再铺到 24 层整网（首次运行即 4/4 通过）。若一上来就跑整网，第一次失败要在 24 层 × 十几条张量里二分；切片让"哪个算子错"变成测试名本身。
- **2026-09-17** —— **差分参照物只能来自参考实现的真实输入/输出**：`apply_rotary_pos_emb` 不是 `nn.Module`（挂不上 hook），所以用一次函数替换来捕获它的输入输出 —— 但它是**模块级函数，24 层都会调它**，只录目标层那一层；其余张量全部是某个模块的真实输入或真实输出（含"注意力输出"= `o_proj` 的输入这一唯一干净边界）。脚本里**不做任何数学**：一旦脚本自己算了一步，参照物就不再独立于被测实现。
- **2026-09-17** —— **绑定权重用别名行而不是再存一份**：Qwen2.5-0.5B 的 `tie_word_embeddings` 让 `lm_head.weight` 与 `model.embed_tokens.weight` 是同一块存储（按 `data_ptr` 去重）。导出时给别名单独一行索引、指向同一偏移，于是加载侧不需要知道 tie 的存在 —— 也就没有"忘了 tie"这一类 bug。
- **2026-09-17** —— Mojo 1.0 三个语法事实（都撞过）：`out` 是关键字，**不能作参数名**（`def f(out: TensorView)` 直接解析失败）；`Pointer.bitcast` 已废弃 → `unsafe_bitcast`；`sqrt` / `exp` 在 `std.math`（不是 `math`）。另外 `FileHandle` 的错误是无类型的，会沿着调用链把 `raises` 都拖成无类型 —— 在 `core/text.mojo` 里就地转成具名 `AlofaError`，上层的 `except err: err.name()` 才成立。
- **2026-09-17** —— **同一份代码，`mojo run` 与 `mojo build` 差一个数量级**：逐算子套件 `mojo run` 约 10 s（含编译），整网门 `mojo run` 要 30 分钟以上，而 `mojo build -O2` 后 6 分钟跑完 4 条 prompt 的 prefill + 4×128 token greedy。标量 oracle 的**绝对耗时不构成任何性能结论**，故本账本不写 tok/s。
- **2026-09-17** —— **greedy 参考用 KV cache 生成，再用无 cache 全序列重放核对前 8 个 token**：两条路径在参考实现里就应当一致，这一步把"应当"变成"已核对"；若不核对，我们验证的 Mojo 增量路径就只是在跟一个有 cache 的黑箱比。
- **2026-09-17** —— **整网 q4_0 通路打通，但"一致率 ≥ 0.90"那条质量门**没过**，按实测记 `missing`**：实测教师强制贪心一致率 **0.80**（输出投影留在 fp32；连它一起量化是 0.77）。先说这个数是怎么来的才敢写下来：每 32 个元素共用一个 fp16 缩放因子，实测 q/k/v/o 权重相对 L2 误差约 10%，落到 151936 维 argmax 上约两成位置翻盘 —— 是**朴素 q4_0 本身**的精度，不是实现错。**不放宽门、也不假装达标**：0.90 那条保持 `missing` 并写明下一步（块内 scale 改按 MSE 选，格式不变）；本轮新增的 0.75 下限只回答"量化通路是不是真在算东西"（半字节装反 → 0.0，前向退回 fp32 → 1.0，两头都抓）。一个门只能证明一件事，把"能用"和"够好"混在一个阈值里，最后两个都证明不了。
- **2026-09-17** —— **算子级门全绿，整网却崩到 0/512**：q4 的算子门 7/7 通过，接进 24 层之后一致率却是 **0**。原因是**契约不匹配**：核做的是矩阵乘**向量**（一次给一个输出行），而模型要的是一整个 token 批次；第一版 `project` 把 `dst.shape.dims[0]`（token 数）当成了输出行数，于是每个 token 都只算了第一个输出单元。**算子级的 fixture 只按"一维 x"那一份契约写**，所以它一次都没红过 —— 测试覆盖到的契约，才是被测代码真正保证的契约；没被测到的那一半，就是 bug 的藏身处。修法是让模型侧逐行喂（而不是给核塞二维支持）：核的那份契约与它对着的 fixture 都是一维的，为省一个循环把两处契约一起改掉，换来的只是两边都说不清自己在算什么。
- **2026-09-17** —— **块内缩放因子改按 MSE 选，一致率 0.800 → 0.8164，那条 ≥0.90 的质量门仍然没过，按实测保持 `missing`**：格式（一个 fp16 scale + 32 个 nibble）一点没动，变的是台阶放哪儿（`d = amax/7` → 对固定 nibble 的最小二乘 `Σ(x·s)/Σ(s²)`，迭代两轮）。收获比预期小，但小在哪儿说清楚了才值得写：权重相对 L2 误差 10.75% → 10.34%，一致率 +1.6 个百分点；另在 q_w 上把 `amax/7` 整体乘系数扫一遍，最优是 **×0.90（10.23%）**，与逐块 MSE 解的 10.34% 只差 1% —— **"每块选一个 scale"这个自由度已经榨干**，再往下必须换块布局（q4_K / 逐行 scale / 带激活重要性的 scale）。配套补了**第三条负向对照**：测试里另写一个只改缩放因子选法、取整规则保持一致的旧规则版本，实测与参考相差 121846 字节 —— 这条是必需的，因为**参照物是同一次导出的产物**，实现退回 `amax/7` 时参照物会跟着一起退，逐字节比较照样绿。
- **2026-09-17** —— **avx2 接入整网（4/4 通过，156 s），代价是发现一个编译器崩溃**：后端做成 `prefill`/`step`/`run` 的编译期参数后，同一批 prompt、同一段 fp32 参考、同一条判据（余弦 ≥0.999 且 argmax 全等 / 128 token greedy 逐 token 相等 / 逐 token 解码与整段 prefill 一致）在向量后端上全绿。**但"跑的确实是向量后端"这件事数值上证明不了** —— 两个后端在这些形状上逐位相同，任何数值比较都区分不了。于是守它的是两样：① 后端名字与算子分发**共用同一个判断**（谎报要改两遍）；② 一道**编译期红测**（`pixi run test-backend-guard`）：`tests/fixtures/bad_backend.mojo` 必须编译失败，且失败原因是 `unknown cpu backend` —— 因为分发写成 `comptime if backend == BACKEND_AVX2`，拼错的常量会**静默退化成标量**，那是最坏的失败：它不红，它只是把要验的东西换掉了。后端之所以落在**方法**上而不是结构体上，是因为 Mojo 1.0.0（ed45d567）在"参数化结构体 + 会抛错误的构造函数"上会直接把编译器进程搞崩（`var m = S[0]()` 一行就够），最小复现写在那个 fixture 的注释里 —— 绕开之后语义反而更准：后端是**这一次前向**的属性，权重和 KV 缓存都不关心它。
- **2026-09-17** —— **向量后端（§2 `avx2` `missing` → `verified`），并抓到两个只有"端到端对照"才能暴露的 bug**：① **偏置播满四条通道** → 偏置被加了 4 倍。这个错法的杀伤力在于它的**选择性**：Qwen2 里正好 q/k/v 带偏置、o/gate/up/down 不带，于是不带偏置的投影全对，失败的只有前三个，现象看起来像"某个头的权重有问题"而不是"向量化写错了"。标量项进向量累加器只能占**一条**通道，其余置零 —— 播满是最自然的写法，也最错。② 契约层面：向量主循环之外必须有标量尾巴，而 fixture 的形状（896 / 4864）恰好都能被通道数整除，尾巴**从没被执行过** —— 这种"恰好没事"的依赖换个模型就塌，所以专门写了一条负向对照顶住（只跑主循环的版本喂长度 10 的输入，必须判红）。
- **2026-09-17** —— **内存门打通（§5 新增「权重常驻内存门」，`missing` → `verified`）**：证据 `tests/unit/test_memory_gate.mojo`（1/1；实测 1.006×，预算 1.15×）。**这个门的价值不在那个绿灯，而在它的负向对照**：额外锁一整份权重 → 峰值从 1.99 GB 涨到 3.97 GB，门必须红。账本里把"能拦住的"（多留一份）与"拦不住的"（零拷贝：mmap 与 `read()` 的 RSS 都是约 1×，**RSS 这个量区分不了它们**）分开写 —— 把能拦住的说成已经证明的，就会让后来者以为零拷贝这条已经被看过了。分母按**字节偏移去重**，因为绑定权重是两条同偏移的别名行：不去重分母大 27%，1.15× 的门实际成了 1.47×，门自己松了自己都不知道。
- **2026-09-17** —— **负向对照抓出的是测量本身的 bug，不是被测代码的 bug**：第一轮对照"多留一份"加进去，峰值 RSS 纹丝不动。查下去发现 Mojo 在**最后一次使用处**析构值 —— 生成循环一结束，`forward` 就到寿命，`MappedFile` 立刻 `munmap`，那 1.9 GB 当场消失（此刻 RSS 从 1.99 GB 掉到 10 MB）。于是"多留一份"只是把已空的坑重新填满，**峰值（历史最高点）根本不涨**。主断言从头到尾都是绿的，是负向对照把它揪出来的 —— 这正是对照存在的意义。ⓘ 另一个连带发现：`keep_alive()` 是**空函数**，内联之后可能被当成"没有使用"，光靠它钉不住寿命；末尾必须有一次**真正的使用**（这里是一次打印）把寿命撑到测量之后。
- **2026-09-17** —— **想让"占内存"这件事被观测到，就得用编译器穿透不了的手段**：手写"每页写一个字节再读回来"会被 store-to-load 转发优化掉（一片页都没真正触碰）；换成 `memset` 也照样没了 —— 它是 LLVM 认识的固有函数，后面跟着释放时被当死存储删掉。两种写法实测峰值 RSS 一模一样，对照看着做了其实没做。最终用 `mlock`：编译器不认识它的语义删不掉，而且它让内核把整段**锁进物理内存**，这一份必须常驻。
- **2026-09-17** —— **q4_0 解量化与融合 matmul 打通（§4 两行由 `missing` 升为 `verified`）**：证据 `tests/unit/test_q4_parity.mojo`（7/7，layer 0 四个真实投影矩阵）。要点是**这条门敢用零容差**：解出的值只取决于 4 位 nibble 与 fp16 缩放因子，两者都精确（半精度能表示的数单精度都能精确表示 → fp16→fp32 是精确转换，故 `quant.mojo` 手写位解码而不依赖任何 `Float16` 行为），于是"差 1 ulp"不是舍入，是**布局读错**。用容差比较等于给"高低半字节装反""缩放按大端读""块边界算错"这类错误发通行证 —— 它们每一个都会让输出仍然像权重。
- **2026-09-17** —— **区分"外部格式"与"我方转换"**：q4_0 的**块布局**是 GGML 的外部事实，所以解量化属格式一致性检验、可标 `verified`；而 **fp32 → q4_0 的量化步是我方离线的格式转换**（Mojo 侧永远不量化），没有外部参照，不许写成"与 llama.cpp 一致"。账本里这两件事分开成两行写。同理，§6 采样器里 `repetition_penalty` 有 HF processor 可对照，而 logit bias 与 frequency / presence 是自证 —— **同一行里混着两类证据就是含糊**。
- **2026-09-17** —— **"融合"必须是被测对象本身**：q4 matmul 把 nibble 读到寄存器里直接乘缩放与激活累加，**不物化解量化后的权重**。若先解量化再调通用 matmul，那被测的就只是通用 matmul，而"解量化融进 matmul"这条能力根本没被检验 —— 门会绿，但绿的是别的东西。
- **2026-09-17** —— **1.6 采样器门打通（§6 采样器 + 采样分布正确性、§10 top-k 集合比较 + 分布检验，共四行由 `missing` 升为 `verified`）**：证据统一指向 `tests/unit/test_sampler_parity.mojo`（11/11）。三层判据逐层收窄：① 9 条逐阶段用例的**概率向量逐元素相等**（容差 1e-6，约 16 个 fp32 存储步长）；② 自研 SplitMix64 与参考侧同一整数算法 → 采样出的 **token id 精确相等**（9 条用例 × 32 次）；③ 固定均匀序列 50000 次采样的经验分布 vs 理论分布（TVD 0.00766 ≤ 0.05，卡方 23.05 ≤ 80）。PRNG 的**原始 64 位输出**单独导出为 `draws.tsv`：若它对而采样错，错在逆 CDF；若它就错，错在 PRNG —— 与逐算子中间张量同一个思路，失败时只有该环节的断言红。
- **2026-09-17** —— **抽样正确性必须靠"逐 id 相等"而不是"分布相似"**：`llm-mojo` 那类用精确匹配替代拒绝采样的错误，产出的分布照样平滑、照样流利，任何"看起来合理"的判据都抓不住它。所以本轮不用 `std.random` —— 其算法无法在 Python 侧复现，做不到逐 id 相等；自研 SplitMix64 的全部代价就是"不能当密码学随机源用"，而这一点从来不在需求里（需求是同一个种子永远给出同一串输出）。
- **2026-09-17** —— **并列取值只比集合、不比顺序**：`torch.sort` 在并列值上的顺序未定义，逐元素断言会 flaky；但反过来，真实 logits 几乎不并列，于是把 `>=` 写成 `>` 在真实数据上**测不出来**，而量化后一定会并列 —— 故另设 4 条合成用例专门打这一点。**该门已用变异测试验证过会红**：改一个比较符 → 概率向量 / 存活集合 / 逐 id 采样 / 并列集合四个测试同时失败。一个没被验证过能失败的门，绿灯不说明任何事。
- **2026-09-17** —— **"语义自证"必须与"HF 对齐"分开标注**：`repetition_penalty` 走 HF 乘法语义并有真实 processor 可对照；`logit bias`、`frequency` / `presence` penalty 在 HF 4.41 里**没有对应 processor**，只能按 OpenAI / vLLM 加法语义实现并用边界用例钉住。把这两类混为一谈，等于让一个没有权威参照的算子顶着"已对齐 HF"的名头 —— 正是本账本要防的那种含糊。两种重复惩罚语义**字段名不共用**，避免调用方在两种不兼容的语义间无声切换。
- **2026-09-17** —— **1.7 CUDA 只核状态、不写代码**：A100 验证机 `10.107.6.60:3389` 本轮实测连接超时，本机 TITAN X 为 Maxwell sm_52 → §4 新增一行标 `hardware-blocked`，写明**阻塞原因**（机器不可达 + 本地 ISA 不支持）与**解锁条件**（可达 + `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas` + fp32 容差 1e-5 的逐值差分门）。写完不验的 kernel 比没有更危险：它会被后来者当成可用。§1.2 的"端到端 GPU kernel 数值正确"（向量加，`verified-remote`）与本行是两件事，不可互相顶替。
- **2026-09-16** —— **差分门的"慢"是编译不是运行**：`mojo run` 跑这个套件要 ~100 s，而同一份代码 `mojo build` 出的可执行文件跑完全部三条测试只要 **0.34 s** —— 瓶颈是生成的 Unicode 表（`src/alofa/tokenizer/unicode_data.mojo`，13806 行）在 -O3 下的编译。**结论：不要用 TestSuite 打印的耗时判断性能**（它报告 209 s，而进程墙钟只有 0.34 s，计时本身不可信）。要测速度先 `mojo build`；CI 可用 `-O0`（27.7 s）把编译降下来。
- **2026-09-17** —— **P2 调度器门打通（§7 七行由 `missing` 升为 `verified`）**：证据统一指向 `tests/unit/test_scheduler.mojo`（15/15）。调度器被做成“零分配、无 I/O、无时钟的纯状态机”，代价是策略里每一处“谁先谁后”都必须写下来（否则重放不可复现），好处是**抢占风暴、0 预算、取消撞车**这些边界不再需要 2GB 权重和 GPU 就能被测 —— 它们现在只是文本。
- **2026-09-17** —— **参照物必须是独立实现，否则门恒真**：`scripts/dump_scheduler_reference.py` 用 Python 把同一份策略重写一遍并导出 trace；若 fixture 由被测实现自己导出，实现改坏了 fixture 会跟着改坏。比对是**逐字节**而不是容差 —— 调度器输出的是**决定**（这一拍给谁多少 token、抢占谁），两个决定之间不存在“差一点点”，差一个 token 就是另一个决定。配套三条常驻负向对照：改坏一拍 OUT 的 `bad.trace`、换一种抢占顺序的 `alt.trace`、给调度器加堆容器的 `bad_alloc.mojo`，三者都必须被判红。
- **2026-09-17** —— **两个策略漏洞是被 fixture 逼出来的，不是想出来的**：① “取消先于到达”只说了一半 —— 取消名单还必须**挡住同拍的到达**，否则请求会先入队、再被服务，然后在下一拍消失（s05 第一版就抓到了它）；② 延迟护栏原定抢在 decode 之前，实测会让正在 decode 的请求一拍不进 —— 改为**只在等待者之间插队**：护栏防的是“队首有个超长 prompt 每拍吃光预算”，不是“预算被 decode 占满”，后者说明并发已饱和，插队只是把等待转嫁给已经占着 KV 的人。
- **2026-09-17** —— **一个只能填 0 的字段比没有字段更糟**：架构草图里 `SchedInput` 有个 `freed_blocks`（引擎侧释放的块）。但没有物理块池时每个块都归属于某个活跃请求，这个字段只能填 0 —— 写出来读起来像能力。本轮不实现它，并在 §7 单列一行 `missing`，写明解锁条件（P2.1 物理块池落地，出现“不属于任何请求的块”，例如前缀缓存条目）。同理，抢占**只针对 RUNNING 请求**：抢占“半截 prefill”的请求会让 prefill 永远无法完成，那是伪装成策略的抖动。
- **2026-09-17** —— **零分配这条证据必须写明边界**：所有容器是编译期定长 `InlineArray`，并用源码门（扫描 `List[` / `String(` / `Arena(` 等构造点，配常驻红测）钉住“没人把它改回会增长的样子”。但这条证据**拦不住** libc 里的小块分配，也**不等同**于进程级 RSS 不动，所以账本里就按这个口径写，不写成“进程零分配”。另外**录 trace 会分配 String** —— 因此录制是 `step` 之外的可选动作，稳态路径不碰它。
- **2026-09-17** —— **P2.1 统一 KV 寻址打通（§6 四行由 `missing` 升为 `verified`，另新增“节点分裂”“KV 零分配”两行）**：证据统一指向 `tests/unit/test_kv_pool.mojo`（12/12）。一个物理 `BlockPool` 上挂两个视图 —— 请求的 `PageTable` 与全局的 radix 前缀树 —— 二者看向同一批块，只共享一条 refcount 等式。7 个场景由 `scripts/dump_kv_reference.py` 这份**独立 Python 实现**导出 trace，逐字节比对；摘要 `d=` 覆盖**整个空间**（refcount 数组 + 空闲链顺序 + 全部请求 + 全部节点），不是只抽查请求的块序列。⚠️ 本轮只做**寻址与引用计数**，paged attention（§7 / 2.2）仍为 `missing` —— 建好不接是 `A.E.S.I.R.` 的教训，故账本明确标注未接入。
- **2026-09-17** —— **P2.2 分页注意力打通（§1 的"GQA 因果注意力（分页 / block table）"与 §7 的 paged attention 由 `missing` 升为 `verified`；§6 的 `RadixNode` 行由"尚未接入 attention"改为已接入）**：证据统一指向 `tests/unit/test_paged_attention.mojo`（11/11）。这一层的整个主张是"分页只改**行在哪里**，不改**算什么**"，所以判据必须与主张同形 —— **寻址用逐位相等，只有公式才用容差**：`paged_gather` 与参照物导出的连续行逐位一致，分页 kernel 与连续 oracle 逐位一致（11 个用例、0 个元素不同）。若这里用 1e-5，就是在替一个本不该存在的差异留余地：差一个 ulp 就说明地址算错了，而地址算错的后果是读到别人的 token —— 一个看起来完全合理的数。公式本身由 `scripts/dump_paged_reference.py` 按定义另写一遍（与 Mojo 零共享代码）导出期望值，1e-5 相对容差，唯一的跨语言差异来源是 `exp` 的最后一位。常驻负向对照 5 条：改坏一个元素的 `expected_bad.tsv`、把 GQA 映射写成 `h % n_kv_heads` 的 `expected_hmap.tsv`、给 kernel 加堆容器的 `bad_paged_alloc.mojo`、把每段 run 的尾部槽位灌成垃圾值后输出必须逐位不变、共享同一块的两条请求必须读到同一段字节。
- **2026-09-17** —— **页表少了"块内起始槽位"，2.1 必须回头补**：2.1 的 `PageTable` 只有 (块 id, 长度)，因为当时每个请求都从块首开始 —— 而 2.2 要读的是**共享前缀**，共享可以从块中间开始。若内核按"每段 run 都从块首读"寻址，它会读出别人的 token 并算出一个看似合理的数：这是分页内核唯一不能有的失败模式。修法是三段的 (block, start, length)：Mojo 的 `KvSpace` 与 Python 参照同步加 `bstart`，合并条目的条件从"同一个块"收紧为"`bstart + blen == start`"（**槽位连续**，不只是同块），`digest` 与 `check_invariants` 覆盖 `bstart`，2.1 的夹具随之重导出并重跑（仍 12/12）。另外，当前树策略只从根开始共享、请求一律从槽位 0 分配，所以**块内偏移在真实轨迹里不可达** —— 于是夹具里放了 3 条 `syn_*` 用例，并把"真实轨迹里 start 恒为 0"写成一条断言：策略哪天变了，这条会红，而不是悄悄少验一维。
- **2026-09-17** —— **refcount 计的是“视图条目”，不是“有多少个节点贡献了它”**：第一版让每个贡献块的树节点各 retain 一次，于是不变量在 s01 第三个请求处就崩（`counted[0]=5` vs `refcnt[0]=6`）—— 两个树节点指向同一块，而请求页表里只有一项，释放时会少放一次。**修法是让 `add_block` 返回“这一项是不是新条目”，只在为真时才 retain**；同理“commit 是 retain 而不是转移所有权”，请求与树各持一份，块只在两者都放手时才回池。这条如果只靠人眼读代码是发现不了的 —— 它只在“同一块被多条路径命中”时才现形，而 fixture 的价值就是把这种路径逼出来。
- **2026-09-17** —— **树的分裂必须是元数据重排，不能是新分配**：匹配停在节点中间时若整段复制 token，前缀共享的收益会被拷贝吃光。分裂只改 `offset/ntok/nblk/blocks`：子节点沿用父节点的块，父节点把自己不再覆盖的尾块放掉。门上是可测的 —— 断言分裂前后 `pool.used` **不变**且节点数 +1；若哪天有人改成了拷贝，这条会立刻红。
- **2026-09-17** —— **P2.5 批张量池落地（§7 的“批张量池（稳态零堆分配）”由 `missing` 升为 `verified`，另新增“批组装（每请求一段连续行）”一行）**：证据统一指向 `tests/unit/test_batch_pool.mojo`（12/12）。这一层要挡住的不是慢，是**无边界** —— 按当前批次需要什么就分配什么的修补会毁掉 P2 正在买的东西：忙碌循环里的一次分配没有上界，于是“比任何测过的批次大一个请求”的批次会在生产里失败。所以池子**构造一次、之后再不分配**：它不拥有内存（收一个指针加容量），全部簿记是编译期定长 `InlineArray`。7 个场景与 `scripts/dump_batch_reference.py` 这份独立 Python 实现**逐字节相同**（分配是决定，不是数值）。⚠️ **尚未与模型前向接线**：executor（`runtime/executor.mojo`）没写之前，“批大小 1/2/4/8 与单请求逐 token 一致”这条 P2 判据还验不了，账本就照这个口径写。
- **2026-09-17** —— **没有空闲链表，是因为两套账终会漂移**：多数分配器既维护空闲链表、又携带每借用的状态以便自查，两者会漂。这里“活着的借用”是唯一的一本账 —— 每次放置都从它重导出可用空隙，于是重叠在构造上不可能，也不存在 Release 之后还留在池子里的碎片。**代价**是放置 O(n)（n ≤ 24），比起会在这些字节里做的算术可以忽略。
- **2026-09-17** —— **“还有很多空间”时必须报 `capacity`，而不是绕回去或跨两个空洞凑**：s03 把池子填满后隔一个释放 —— 剩 131072 字节可用、最大空洞只有 16384 —— 借两块必须被拒。否则 caller 拿到的是**不相邻的两段**，而分页那一层的教训是：拿错位置的后果是一个看起来完全合理的数，不是一个错误。同理第 25 个活的借用也必须是 `capacity`（24 是上限不是起点）；重复释放是 `double_free`。
- **2026-09-17** —— **两条性质不靠参照物，因为它们说的就是实现本身**：① **活着的借用不共享任何一个字节** —— 每个区间通过 `f32_of` 写自己的标记再逐字读回；把 `place` 往旁边挪一格，这条实测会红（同一处变异还让另外两条测试的断言一起红）。② **峰值不漂移** —— 20 个相同的执行步之后 high-water 必须与第一步相同；若它在涨，稳态就谈不上稳态，此后测的吞吐是关于另一个池子的数字。
- **2026-09-17** —— **负向对照必须证明自己会失败**：`alt.trace` 用 best-fit 重跑同一组操作，除了“必须被拒”之外还要求它**至少挪动一处偏移** —— 否则那 6 个逐字节比对只是在验文件格式而不是在验分配器。这一点是写作时才发现 s02 的两个策略结果完全相同 —— 那种情况下这道对照等于没有。
- **2026-09-17** —— **KV 门的不变量必须“从视图重算”，不能只是“实现自己记得的数”**：`check_invariants()` 遍历所有活跃请求与所有存活节点，重新数一遍每块的持有数，再与 `pool.refcnt` 逐块比对 —— 于是即使 Mojo 与 Python 两侧**同时**漏了一次 retain，方程仍然会红（已用变异测试验证：多一次 retain → 7 个场景的逐字节比对全红）。加上另外两条常驻负向对照（改坏一拍摘要的 `bad.trace`、关掉分裂的 `alt.trace`、给 KV 加堆容器的 `bad_kv_alloc.mojo`）与两次变异实测（重复 retain / 空闲链倒序），这个门被证明**会失败**。
- **2026-09-17** —— **P2.5b 批执行器落地（§7 新增“批执行器（注意力按请求分块）”与“批一致性（批大小 1/2/4/8 与单请求逐 token 相同）”两行，均为 `verified`）**：证据分别是 `tests/unit/test_batch_executor.mojo`（10/10）与 `tests/unit/test_batch_forward.mojo`（4/4，重门）。“不同请求的行不能互相 attend”不写成一张 mask —— 一行被交给注意力时带着**自己的** K/V 基址、自己的 `upto` 与自己的 `pos`，于是别人的 key **在地址上不可达**：一条可以被裁掉或填错的 mask 只是把祈祷写进代码。
- **2026-09-17** —— **参照物是串行前向，且这是刻意的**：批一致性门（1/2/4/8 与单请求逐 token 相同）把同一批 prompt 走两遍 —— 一批 N 条 vs 一条一条跑 —— 参照物与本树的批路径共享 kernel。被比较的是**编排**（行归属、每行的 pos、每行的窗口、KV 区域），而串行路径只有一条请求、不可能在这些上出错；串行路径对 Hugging Face 的一致性另有 §5 的模型门负责。8 条 prompt 合计 88 行 > 行块 64，所以这一门**真的把一次 prefill 切成两拍**。
- **2026-09-17** —— **`unsafe_offset` 的单位是“元素”不是字节**：`Pointer[Int]` 上偏移 1 是 8 字节，`RawPtr`（UInt8）上才是 1 字节。写 `logits_of` 时按“字节”乘了 4，后果是**第 0 行完全正常、之后每一行都读越界** —— 批大小 1 全绿、批大小 2 的第 2 条请求输出一个看起来合理的错 token。这种错只有在“多行且取非首行”时才现形，所以批一致性门必须跑到 8。
- **2026-09-17** —— **prompt 没吃完的那一拍不许回馈 argmax**：一次 prefill 被切成两拍时，第一拍的 argmax 是在**半个 prompt** 上算出来的，把它 append 回队列就等于把生成的 token **插进 prompt 中间** —— 序列从此错了，而每一步的数看起来都对。修法是 `advance` 里 `left > 0` 时直接 continue：请求还没吃完 prompt 就不欠答案，**吃完 prompt 的那一拍才有权选择**。Mojo 与 Python 参照同步修改并重导出夹具（仍 10/10）。
- **2026-09-17** —— **KV 区域地址在构造时一次写死，步进循环只读不算**：第一版让 `forward` 用 `k_of[owner*MAX_LAYERS+layer]` 取基址，但那张表从来没被填过（只分配了一整块区域），于是 `copy_into` 往空指针里写 —— 段错误，且只在真的跑前向时才现形（plan 那一层的门全绿）。修法是在构造里把每个（请求, 层, 侧）的基址写进那张表：**步进循环只读指针，不算地址**，“读到一个从没被写过的指针”是这一层最该被结构性排除的失败。
- **2026-09-17** —— **P2 引擎循环落地（§7 新增「引擎循环（调度器 ↔ 批执行器接线）」一行，`verified`）**：`src/alofa/engine/core.mojo` + `tests/unit/test_engine_core.mojo`（10/10，已进 `pixi run test`）。调度器与执行器各自本来就有门（前者与独立 Python 实现逐字节，后者在真实权重上与串行前向逐 token 相等），**谁都抓不到接缝**：prefill 到达时是一片切片、decode 到达时只是一个「上一拍选的 token」、抢占要同时作废历史与已生成的 token —— 这三种错法留下的数都看起来合理。所以这一门把 argmax 注入、不跑模型，只验时序。
- **2026-09-17** —— **「队列空了」不等于「prompt 吃完了」**：调度器按 `max_chunk` 把 prompt 切片喂下去，于是**每个切片边界上队列都是空的**，而执行器原本正是用「队列空了」判断「prompt 吃完了、可以选 token」 —— 结果生成的 token 被**插进 prompt 中间**（实测：10 个 token 的 prompt 只跑了 7 行就宣称自己生成完了 2 个 token，而分块与不分块两条路径的拍数竟然相同）。修法是让执行器知道 prompt **有多长**（`add` 新增 `prompt_len`，默认等于本次给的切片），判定改成「队列空 **且** `hist >= n_prompt`」。它与 2.5b 那条「prompt 没吃完的那一拍不许回馈 argmax」是同一件事的两面：那一面管「一次 prefill 被行块切成两拍」，这一面管「被调度器切成 N 拍」。
- **2026-09-17** —— **花完预算的请求不许自己退场**：执行器原来的 `advance` 在 `budget <= 0` 时把 `live` 清零，而花掉最后一点预算的那个 token **正写在它自己的 `out` 里** —— 退场等于带走自己的 transcript，调用方下一句 `generated` 立刻报「请求不 resident」，最后一个 token 永远读不出来。改成**停止继续喂、但不退场**，退场交给调用方：只有调用方知道停止串、取消和「够了」。引擎侧「transcript 够长了就在 `settle` 里 drop」正是由此而来。
- **2026-09-17** —— **调度器数的是 decode 拍，执行器数的是 token，而第一个 token 是 prefill 结束那一拍白送的**：于是执行器比调度器早一拍用完预算；不回报的话，下一拍调度器会给一个已经不在执行器里的请求发 decode，引擎直接报 `invalid_argument` 挂掉。修法是引擎发现 `n_out >= max_new` 就 drop，并把它记进**下一拍**的 `inp.finished` —— 必须在那一拍决定任何事**之前**说出口，否则那一拍会为一个答案已经写完的请求再跑一次 decode。
- **2026-09-17** —— **KV 物理块池接入引擎（§7「KV 物理块池（含 `freed_blocks` 这类外部释放）」由 `missing` 升 `verified`）**：新增 `src/alofa/engine/kv_room.mojo` + `tests/unit/test_kv_room.mojo`（10/10，已进 `pixi run test`），参照物是独立 Python 实现 `scripts/dump_kv_room_reference.py`。调度器侧改账：完成即发布到前缀缓存（`release(slot, to_cache)` 把块从「持有」挪到「缓存」，`blocks_used` 不降），只有引擎的 `freed_blocks` 能把它们还回来；`blocks_used == blocks_held() + cached_blocks` 每拍重算。

- **2026-09-17** —— **缓存是没有主人的占用，`preempt_one` 永远动不了它**：水位被缓存顶高时，调度器既不能抢占（没有请求可抢）也不能自己归还（块不是它的），只能 break —— 那一刻看起来像「抢占失效」。所以回收放在引擎侧，且必须在调度器决定**之前**：「缓存让位给活着的请求」（缓存 ≤ 容量 − 持有）与「水位」两条规则。顺序错了就会在 `ensure_room` 里报 `capacity`，而报错的位置离真正的原因隔着一整拍。

- **2026-09-17** —— **房间要的是「序列现在有多长」，不是「这一拍长了多少」**：`grow_to(req, prompt + 已生成)` 传目标值。传增量的话，「一拍生成两个 token」与「一拍没生成」之后得自己记住数到哪了，而记住的计数第一次遇到前缀共享就会和池子分家。

- **2026-09-17** —— **`KvSpace.slot_of` 对未知请求是抛错，而引擎要问的是「它还在吗」**：第一版直接拿 `slot_of` 当查询用，`settle` 里对已经放掉的请求问一句就炸（`out_of_range: unknown request id`，两条用例红）。房间改成 `find_request` 查询、`slot_of` 取用（取不到才具名拒绝）—— 「不在这里」是答案，不是缺陷。

- **2026-09-17** —— **抢占场景的容量不能按「装不下」来设**：4 条请求各 19 个 token / 块 16 = 各 2 块，四条共 8 块；原来容量 4 的意思是「装不下三条」，而块变成物理的以后，完成即发布会让**缓存**先占满池子 —— 那时报 `capacity` 的真正原因是「缓存没还」，两件事被混成一件。场景改成容量 8 + 水位 900‰（只留 7 块）：第四条必然被挤掉一次，抢占与回收各验各的。
- **2026-09-17** —— **分页注意力接入引擎循环（§7 新增「分页注意力接入引擎循环（前向从块表取地址）」`verified`；KV 物理块池那行「执行器 KV 区域仍是写死的」边界消除）**：`paged_scatter` 写经过表、`_table_of` 按 `hist + n` 裁剪、房间 `page_table()` 一次拷贝交接、引擎 `sync_page_table()` 在 `admit`/`grow_to` 之后立刻同步。批一致性重门改在**乱序块号**下跑（4/4 绿）—— 于是「批 == 串行」成了关于页表的断言。

- **2026-09-17** —— **一个块持有每一层各一个槽**：执行器把层号折进块 id（`layer * MAX_BLOCKS + block`），整个块池只有一个视图、构造时建好。这是被「忙碌循环不许建 `List`」逼出来的设计 —— 每层建一次视图就会在每步每层建一个形状，零分配源码门当场判红（`executor.mojo` 出现 `List[`），所以 `view3` 放进了 `core/tensor.mojo`（形状是 `List`，L0 没有这条约束）。

- **2026-09-17** —— **写必须按 `hist + n` 裁剪，不能整张表用**：房间可以为还没到的 token 预留**整块**（切片 prefill 按块预留），整张表用会让注意力读到没人写过的位。所以执行器断言「表的容量不小于历史」再裁剪 —— 少了具名拒绝（`capacity`），多了悄悄截掉。

- **2026-09-17** —— **`maps` arena 少算了一次**：删掉 `row_k`/`row_v` 两张表时把它们的字节数一起减了，而 `out_rows`/`chosen` 两张 `MAX_BATCH` 表还留在原处 —— arena 不够**不报错**，它给出一段与下一张表重叠的地址，症状是「某一行指向了别人的请求」。执行器门 e01 当场判红，而红的位置离真正的原因隔着一次完全不相干的删除。

- **2026-09-17** —— **`Arena` 在最后一次使用处析构（老坑，又踩一次）**：scatter 门的用例里最后一次 `alloc` 之后还要读写那块内存 → 段错误。撑住寿命的 `keep_alive()` 必须放在**用完指针之后**，不是 alloc 之后。

- **2026-09-17** —— **越界用例要按「表的长度之和」来设**，不是按最后一块的大小：表覆盖 7 个位置时写第 6 个是合法的，写第 8 个才越界。第一版把 first 设成 5，门就红了 —— 这次是门对了、用例错了。
- **2026-09-17** —— **共享前缀只算未命中的那一段（§7 新增「共享前缀只算未命中的那一段（前缀缓存省的是算术）」`verified`；上一行「省的是块不是 prefill 计算」的边界随之消除）**：房间 `last_matched` → 引擎 → 执行器 `add` 的 `matched`，命中的 token 入队不建行。重门 6/6：第二遍沿用同一批块、只跑一行，token 与串行逐 token 相同；负向对照（谎报命中）必须产出不同的 token。同时补上上一轮漏写的「分页注意力接入引擎循环」表格行（那轮脚本只 append 了变更日志，主文没落盘 —— 记录与事实差了一行，账本门查不出来）。

- **2026-09-17** —— **最后一个 token 永远要算**：`matched == n` 时截到 `n - 1` 而不是 `n`。零行就没有 logits，而没有 logits 就没有第一个生成的 token —— 一个「完整命中」的请求会安静地卡住，而不是报错。

- **2026-09-17** —— **批量字符串替换会打到同名的另一处**：把 `ids[j]` 改成 `prompts[0][j]` 时改到了 `batched_tokens` 里（新用例与它有一行逐字相同），症状是「index 5 out of bounds」崩在一个压根没改过的用例里。门/夹具的替换必须带上下文锚，换完要 grep 命中数。

- **2026-09-17** —— **重门里 budget 给得比 `STEPS` 多是有意的**：`add(..., budget = steps + 8)` 配 `generate(steps + 4)` 会生成 `steps + 4` 个 token，断言写 `== STEPS` 就红。比较的是**前 `STEPS` 个**，所以断言该是 `>=`。

- **2026-09-17** —— **改账本要确认主文真的落盘**：一次「替换 + 插入 + append 变更日志」的脚本漏了写回主文（`open(p,'a')` 只追加），变更日志说「新增了某行」而表格里没有 —— `check-ledger` 7/7 照过，因为它只查格式。写完必须 grep 那一行。
- **2026-09-17** —— **引擎层端到端验上 `matched` 的接线（§7「共享前缀只算未命中的那一段」补引擎门证据）**：同一个 prompt 提交两次，第二次第一拍**只跑一行**、`history_of == 6`（证明房间 `last_matched` 真的交到了执行器），对照是同一引擎里的冷 prompt（7 行、history 0）。至此这条链路不再是「编译通过 + 单元绿」。`pixi run test` **230 项全绿**（25 套）。

- **2026-09-17** —— **引擎门的 `fill()` 按 req 生成 token id**：`req + i` 让两条请求的 prompt 天然不同 —— 写「共享前缀」的用例必须自己填同一串 id，否则房间永远匹配不到，而宽松的断言照样绿。
- **2026-09-17** —— **形状不再分配（一拍路径上的每视图分配被消除）**：`Shape.dims` 与 `TensorView.strides` 由 `List[Int]` 改为定长 `InlineArray[Int, MAX_RANK=8]`（rank 超限具名 `out_of_range`，不截断），新增 `shape2` / `shape3` / `rows_view`（后者从 `model/arch/qwen.mojo` **下沉**到 `core/tensor.mojo`）。动机不是洁癖：**一拍里每一行 scratch 都要建一个视图**，形状建在堆上就是「每行每拍一次分配」—— 而它藏在 core 层，引擎那条扫 `engine/*.mojo` 的零分配门**根本看不见**，门一直是绿的。所以新增源码门扫一拍路径 6 个文件（core/tensor、engine/core、engine/executor、kernels/cpu/{paged,scalar,segments}）不得出现 `List[Int]()` / `Shape(`，配红测 `tests/fixtures/bad_shape_alloc.mojo`（就是改回前的 `rows_view`）。`pixi run test` **234 项全绿**；重资产门 `test-batch-forward` 6/6、`test-model` 4/4 数值未变。

- **2026-09-17** —— **想给「零分配」配实测计数，这条路走不通**：`external_call` 的返回类型必须是 `RegisterPassable`，`mallinfo2()` 返回 80 字节结构体（sret），拿不到；`List` 分配-释放后再采样 `uordblks` 也看不出差异。所以分配门**仍是源码门**，账本继续按这个口径写（拦得住「加会增长的容器」，拦不住 libc 小块），不谎称有实测。另一个坑：`InlineArray` 是 Movable 而**非** ImplicitlyCopyable → `Shape` / `TensorView` 必须写 `__copyinit__` + `copy_dims()` 逐元素拷贝。
- **2026-09-17** —— **Gate P2.5（KV 池 >95% 的内存门）打通，账本新增一行**：`tests/unit/test_engine_core.mojo` **13/13**。七条 60-token 请求把 112 块的池顶到**实测峰值 111 块（99%）**，水位 950‰ 之下真的发生抢占，七条全部跑完且**逐 token 等于逐条单独跑的基线**。四条常驻对照：抢占次数 > 0、峰值实测（不是配置值）、七条全部完成、与基线逐 token 相同。`pixi run test` **236 项全绿**。

- **2026-09-17** —— **发现（账本新增 `missing` 一行）：分块 prefill 下水位事实上失效**。调度器按**本拍切片**记账、房间按**整条 prompt** 占块 —— 七条请求跑下来房间已达 107/112，调度器的账却只有 50，水位 950‰ 从头到尾没被触发，池子只靠房间抛 `capacity` 兜住，**不是抢占救的**。对照：不分块时两本账差 ≤ 1，所以这不是天然对不齐，是分块路径漏记。门因此只用不分块配置，分块下的水位**未验**，不粉饰；解锁条件写在 §7 那一行里。
- **2026-09-18** —— **修掉分块 prefill 的记账 bug（根因）**：`EngineCore.settle` 每一拍都 `grow_to(req, p_len + n)`，把 KV 序列长度说成整条 prompt，而 prefill 只喂到切片处 → 房间白发整条 prompt 的块（第一拍房间 15 块、调度器账 4 块；七条下来差 10 块），水位 950‰ **从未触发**，池子只靠房间抛 `capacity` 兜住。改为按「已喂到的位置」grow（新增 `fed` 字段）后，分块下两本账差 ≤ 1。新门 `test_the_scheduler_and_the_room_count_the_same_blocks`（旧行为下差 10 块、红）。`pixi run test` **25 套全绿**、`test_engine_core` **13/13**。

- **2026-09-18** —— **撤回前一天记的「KV 池峰值 111/112（99%）」**：那个数字是上面那个 bug 造出来的假象（房间白发整条 prompt 的块），修好后同场景只到 99/112。账本那行改为「高占用下的正确性（不声称 95%）」，并新增两条 `missing`：**①** 物理池 >95% 且能跑完的场景不可达——要顶满需长驻留，而驻留总量高过水位就抢占乒乓活锁（七条各生成 8 token：512 拍不空闲）；**②** 引擎 `reclaim` 缓存与调度器 `cached` 账不同步，抛 `engine freed more blocks than the scheduler holds cached`（既有缺陷，已用修复前的代码对照确认）。
- **2026-09-18** —— 活锁**修法尝试 1 失败并回滚**：`try_prefill` 加「整条序列（`prompt_len + max_new`）准入，池子空时才豁免」→ 抢占从 190 降到 **3**，但**仍不收敛**（300 拍、四条 `n_out` 全 0），且抢占门 `nothing was ever preempted` 转红。说明卡点不是晋升这一步：**部分喂的 `ST_WAITING` 请求仍持有已喂部分的块**，池子被它们占住，喂完的也升不动、decode 也拿不到块。**先定的契约是「prefill 分块期间，WAITING 请求是否继续持有块」**——在此定下之前，任何局部准入/晋升门槛都只是把活锁挪位置。代码已回滚。
- **2026-09-18** —— 活锁**修法尝试 2 失败并回滚**：改为「同一时刻只允许一个 mid-prefill
  请求」（`try_prefill` 里 `has_partial_prefill` 拦截）。无条件串行 → `s02_preempt_storm`
  结束后**仍有活跃请求**（跑不完）、latency guard 门转红 → 串行代价太大；再改「只在
  `blocks_used >= threshold_blocks()` 时才串行」→ 直接破坏记账不变量（`engine freed more
  blocks than the scheduler holds cached`，即此前 ③/④ 修过的同一类）。**代码已回滚**，
  scheduler 17 门恢复全绿。两次尝试（整条准入、串行 prefill）分别被「仍不收敛」和「记账
  被破坏」拦下 → 活锁确属**调度契约**问题，须先定契约（部分喂的 WAITING 请求是否持块、
  抢占是否保留 prefill 进度）再动代码，不要再试局部补丁。
- **2026-09-18** —— 上一条「待查 `kv_room.extend` 分配粒度」**已收窄**：`extend` 委托
  `space.append_tokens`（`runtime/kv/space.mojo:245`），后者在最后一块有空位时填充、否则
  `pool.alloc_one()`，**按 `block_size` 按需分配、不多分** → 差异不来自分配粒度，而来自
  `grow_to` 的 target：引擎传的是 `end`（`core.mojo:470`，prefill 片末）或 `fed[i] + n`
  （`:555`，已喂 + 本拍生成数）。故下一步应**直接比对房间的 `rq_ntok[slot]` 与调度器的
  `done[i]`**（差 1 个 token 即跨块时差 1 块）。未做。
- **2026-09-18** —— 上一条「比对 `rq_ntok` 与 `done`」**未取得数据**：探针
  `test_probe_ntok_vs_done` 直接 FAIL 且**无任何打印**（不是编译错，原因未查明，已恢复文件）。
  下一步别再新写探针：仓库里已有 `test_the_scheduler_and_the_room_count_the_same_blocks`
  （PASS），应**照它的断言写法把它扩成「每拍自洽」**，既复用可用的访问路径，也顺带堵住
  「早期分叉」这个漏检——它现在只验到末态。
- **2026-09-18** —— 活锁**根因第三次订正（真正根因）**：不是晋升、也不是 WAITING 持块，而是 **`admit`（`scheduler.mojo:483-490`）准入不足** —— 它只检查「单条 prompt 的静态块数 ≤ capacity」，**既不看并发累加、也不看 `max_new`**。于是四条各需约 6 块的请求，每一条单独都「装得下」，被全部接收；池子 10 块物理上装不下四条并发 → 调度器只能靠抢占循环。前两次定位都是这条的下游症状。**修法方向**：`admit` 做并发准入（活跃 slot 的 `prompt_len + max_new` 累加 + 本请求整序列 > capacity → 拒绝或排队）。⚠️ 但「装不下时**拒绝**（ERR_CAPACITY）还是**排队**等待」是引擎语义的设计决策，需先定契约，未动手。\r\n- **2026-09-18** —— **契约 A（串行 prefill）已证伪并回滚**：在 `try_prefill` 加「每拍只允许一个请求处于部分喂」守卫 → 300 拍仍不收敛、**133 次抢占**、四条中仅一条吐出 1 个 token。值得注意的是 **17 条 scheduler trace 仍全绿** → 现有 trace 里没有「多请求同时部分喂」的场景 → 若采纳 A，trace 重导成本其实很低，但**方案本身无效**。A 失败的原因：串行能保证一个请求喂完，但它晋升后 decode 仍要块，其余部分喂者还占着块 —— 池子物理容量不变，串行只是让抢占慢一点。\r\n- **2026-09-18（最终结论，推翻前四次订正）** —— 活锁机制已用数据锁定：**不是容量不够，也不是晋升/准入的单点 bug，而是「并发稳态需求 > watermark threshold」时抢占归零导致的不可恢复循环**。`scheduler.mojo:729-734` 第 9 步 `while blocks_used > threshold_blocks()`，`preempt_one`（`:553`）把受害者 `done`/`generated` 归零 → 受害者重喂 → 峰值再超 → 循环。**决定性对照**（同为 4 条 × 23 token，稳态需求 24 块）：`cap=40 + watermark=1000`（limit 40 ≥ 需求 24）→ **10 拍收敛、0 次抢占、四条各 3 token**；`cap=40 + watermark=8`（limit 0）→ 223 次抢占不收敛；`cap=10 + watermark=500`（limit 5）→ 190 次不收敛；`cap=24 + watermark=8` → 223 次不收敛。**修法判据（已验证方向）**：`admit`（`:483`）准入必须按 **`threshold_blocks()`** 而非 `capacity_blocks` —— 此前那次「整条序列准入」失败正因判据用了 capacity，四条全部通过准入却仍超 threshold。**两个采样教训**：① 探针配置顺序是 `(max_chunk, block_size, capacity_blocks, watermark_permille, max_wait_ticks)`，我一度把 watermark 传成 8（limit=0）而误判；② 拍末打印看到的是抢占**之后**的占用（曾见 `used=4/40`），据此推断「与容量无关」是错的。未动手修（改动会改 admit 语义，需定契约）。\r\n- **2026-09-18** —— **方案 1（准入 + 用缓存制造超限）实施到一半，主动回滚，留档。** 准入补丁本身成立且生效（`admit` 按 `threshold_blocks()` 而非 `capacity_blocks` 判并发：活跃 slot 的 `prompt_len + max_new` 累加 + 本请求整序列 > threshold → `ERR_CAPACITY`；首个请求豁免，否则过紧水位会一条都进不来）。配套同步了 Python 参照 `scripts/dump_scheduler_reference.py:admit`，并重导了 7 个 trace。\r\n
  - **实测代价远超预期**：准入让 **8 条测试变红**（不是原以为的 2 条），失败原因全部是 `concurrent sequences exceed the kv watermark`。重导 trace 后 6 条降到 3 条；`s02_preempt_storm` / `s06_kv_watermark` 两个场景按其原构造（多条并发顶破阈值）**在准入下无法建立**，只能重建。\r\n
  - **撞到的设计耦合**：`reclaim_tail`（`dump_scheduler_reference.py:353`）会**自动插入「引擎回收缓存」的拍** —— 队列一旦被缓存卡住就收干缓存。所以「让缓存顶高 `blocks_used` 越过水位」这条路会被脚本自动消解（实测新场景 `C=0` 全程不变）。\r\n
  - **下一步构造必须满足**：某一拍同拍「有请求完成并把整条序列发布进缓存」**且**「还有别的请求处于 RUNNING」—— 只有如此第 9 步才有可抢对象；并且要有 **≥2 个 RUNNING**，否则 `test_a_different_policy_is_detected` 这条抗原仍会失效（抢谁都一样 → alt 与默认策略同字节）。\r\n
  - 代码已回滚待下次开工（改动已在上述两处定位清楚），工作区保持干净。
- **2026-09-18** —— **抢钝活锁已修（准入）**，并订正本日早先那条「实施到一半、主动回滚」：准入已落地并通过全量 239 项。\r\n  - **准入**：`admit` 改按 `threshold_blocks()` 判「已提交序列的整条足迹 + 本请求整条序列」，而非硬容量；首个请求一律放行（否则过紧的水位一条都进不来）。缓存不进判据 —— 它不可抢占，也不是并发足迹。\r\n  - **认知订正（重要）**：trace 的 `C` 字段是 **抢占总次数**（参考 `act["C"] = preempt_total`），**不是**缓存量。早先据此断言「缓存没涨」是错的；真实故障是请求**活不过一拍** —— prefill 那一拍它仍是 `WAITING`（晋升发生在下一拍开头），第 9 步抢不到它，而旧的 `max_new=1` 又让它在超限的同一拍就跑完，于是永远没有可抢占对象。场景改用「多条需要多拍 decode 的请求」后抢占即发生（`preempt_total=54`）。\r\n  - **两条门的重建**：`s02_preempt_storm` / `s06_kv_watermark` 与引擎层两条门改由**前缀缓存**制造超限 —— 先让若干请求跑完并发布进缓存，再放入足迹自身合规的新请求。场景由搜索选出（约束：抢占 > 0、拍末 `blocks_used <= threshold`、全部请求最终完成、`newest` 与 `oldest` 抢占对象不同以保住抗原）。\r\n  - **判定边界（拦得住什么）**：拦得住「并发足迹自身越过水位」的组合 —— 它们在门口就被 `capacity` 具名错误拒绝。**拦不住**：缓存 + 并发足迹合计越过水位，这条路径**依然会抢占**；它靠引擎归还缓存（`room` 的 reclaim）收敛，而不是靠准入 —— 所以「抢占 + 受害者重算后仍跑完」这件事仍然被真实地验着，没有被准入旁路掉。\r\n
- **2026-09-18** —— **N/M 快照从「散文」升级为「机器可核验」**（新增 `evidence:…?count=N` 语法、`check-counts` 门、`ledger-sync` 刷新工具）。
  - **为什么必须做**：`check-ledger` 只校验 evidence **文件存在**，从来不看括号里写的数字。于是那些数字安静腐烂 —— 本轮实跑对照发现三处：① `test_engine_core.mojo` 在账本 §7 四个不同位置写着 **10/10 / 14/14 / 15/15 / 16/16**，实测只有 **16** 是对的；② `test_scheduler.mojo` 写 15，实测 **17**（本轮新增两道题后没同步）；③ `test_batch_executor.mojo` 写 10，实测 **12**。另有 §7「缓存让位」一行说预算是 **48**，而 `DEEP_TIGHT_CAP` 早已改成 **60** —— 同一类腐烂溢出到了常量上。
  - **修法**：数字从散文里拎出来变成结构化后缀 `?count=N`；`scripts/run_tests.sh` 在跑测试的过程中顺带产出 `target/test_counts.tsv`（各套件自报通过条数），`check-counts` 拿它逐项核对。**零额外时间** ——不需要再跑一遍测试。
  - **两个方向的都查**（见 `test_ledger_counts.mojo` 的负向对照）：写了却不一致 → 报错；**跑过却不登记 `?count=`** → 同样报错。后者是关键 —— 若只有「写了才查」，最省事的应对就变成「干脆一个都不写」，门会在最宽松的地方失效。清单缺失时门**失败而非跳过**。
  - **`?count=` 的可维护性**：手改 60+ 处不现实，配 `pixi run ledger-sync` 一条命令把数字刷回来（内部走行级替换 + keepends，不重排换行，不碰变更日志里的历史快照）。
  - 附带的清理：删除根目录 `core/rng.mojo`（一份从未入库、78 行、与 `src/alofa/core/rng.mojo` 只有注释措辞差异的旧草稿）与空转的 `tests/test_main.mojo`（只有一句 `print("test")`）；并将 `pixi run test` 的套件清单从 pixi.toml 的长命令搬到 `scripts/run_tests.sh` 单一维护。
  - **顺手写明一处「没被覆盖」的事实**：重资产门（`test_model_parity` / `test_batch_forward` / `test_q4_greedy` / `test_memory_gate`）不在 `pixi run test` 里，本机清单查不到它们 → 这几行的 N/M 仍是**手工记录**，不在自动核验范围内。这是已知边界，不是遗漏。
- **2026-09-18** —— **P1 第 4 步：垂直切片打通**（新增 `src/alofa/cli.mojo`、`tests/unit/test_vertical_slice.mojo`）。
  - **为什么必须做**：此前每层都有自己的门且都是绿的，但**没有任何一处把它们串起来跑过** —— `test_engine_core.mojo` 的文件头自己写明 argmax 由测试注入。2026-09-17 预告过「各层各自绿、拼起来崩到 0/512」，这一刀就是把那条路径固定成每天能走一遍的东西。
  - **实测**（真权重 1.9 GB、scalar 后端）：`"The capital of France is"` → 贪心 `" Paris. It is the largest city in Europe and the second largest in the world"`；采样（温度 1.0、seed 固定）`":\nA: Paris B: not sure C: london D: BERLIN"`。两条路都说到 Paris。
  - **新入口**：`pixi run generate`（跑一次，人读）+ `pixi run test-slice`（带断言的门）。后者是重资产门 —— 需 1.9 GB 权重，故意不进 `pixi run test`，与 `test-model` 同级。
  - **负向对照**：温度 0.01 的采样必须**逐字等于**贪心。若采样路径悄悄退化成 argmax（logits 取错行、`build` 没被调用），「说出 Paris」照样全绿而采样一次都没生效过；而温度 1.0 时两者**不同**（上面两段文本），一正一反才构成完整证据。
  - **撞出来的两条契约**（详见 §7 那一行的正文）：① 采样循环**进不了** `core.mojo` —— 那个文件既是 `step_path_sources()` 之一又是零分配门的 `SOURCE`，两门都查 `List[`，而 `Sampler.build`只收 `history: List[Int]`、无指针重载，实测在那里加一个 `history_of` 会让**两个门同时变红**（门是对的，那是真承诺）→ 循环内联在 `src/alofa/cli.mojo`，`core.mojo` 一字未改。⚠️ 由此留下未收回的边界：**这条采样路径不在 §7 的零分配承诺内**（每拍为每个活跃请求建一个历史列表），它只服务单请求 CLI；要进忙碌循环必须先给 `Sampler` 一个指针版 `build`。② `MAX_BATCH` **同名两个取值**：`engine/batch.mojo` 是 **8**、`engine/scheduler.mojo` 是 **32**，`core.mojo` 经 `executor` 拿到的是 8；从 scheduler 导入会拿 32 遍历只有 8 个槽的 `ex.live`，实测崩在 `Assert Error: index 8`。类型系统挡不住（`range()` 两端都是 `Int`），只能靠「从哪导入」那一行注释守住。
  - **第 5 步（config.json + safetensors 加载）尚未开始** —— 本机 `~/.cache/huggingface` 里`config.json` 与 `model.safetensors`（954 MB，bf16）都在，前提具备。
- **2026-09-18** —— **P2 第 7 项诊断：量化一致率 0.8164 到底伤在哪（定位完成，未修）**。
  - 新增 `scripts/diag_q4_proj.py`（须用 `/home/rontom/anaconda3/bin/python`，pixi 的没 torch）：**逐个投影单独量化**、其余保持 fp32，再跑与 `test_q4_greedy.mojo` **同构**的教师强制一致率。fp32 自洽 **1.0000**，故下面的数字可以互相比。
  - **结论：不是某个投影的量化误差特别大，而是「输出投影」对同等误差更敏感。**三个投影的相对 L2 几乎一样（q **10.07%** / o **9.56%** / down **9.98%**），一致率却单调变差 ——**q_proj 0.9297 → o_proj 0.9062 → down_proj 0.8594**。`o_proj` 的误差**比 q_proj 还小**却更伤一致率，这就把「误差大」和「位置敏感」分开了。
  - 机制：q/k/v 的误差要过 softmax 与 RMSNorm，被部分吸收；**`o_proj` / `down_proj` 直接写进残差流**，误差原样传给后面每一层；`down_proj` 还要先经 4864 维求和再投影回 896，故最差。所谓「量化输出投影反而更差」（门的文件头：留它 fp32 是 0.80、连它一起量化是 0.77）**不是它量得更差，而是它是每层最后写进残差的那一笔**。
  - 累积：单独量化 q_proj（512 步口径）0.9688、k_proj 0.9512，而**全部量化才 0.8164** ——总损失是 **24 层 × 7 个投影**的叠加，不是单个投影主导。故只优化某一个投影的 scale 选法（上一轮已做、已榨干）天花板很低；要 0.90 得同时降低**所有**投影的误差 —— 换块布局仍是主要路径。
  - ⚠️ 口径边界：上面 0.9297 / 0.9062 / 0.8594 是**单 prompt、128 步**快速档（只用于相对排序）；门的 512 步档本轮只跑完 q（0.9688）与 k（0.9512）两个，其余仍在跑。
  - **第 6 项（Q4_K）未交付**：`scripts/dump_q4k_reference.py` 已写（布局、量化、解量化），但后 4 个子块的 scale 与 min **共用 `scales[j-4]` 的高 2 位**，事后取 max 对齐会把 min 抬高（实测 37 → 53），相对 L2 反而从 q4_0 的 10% **恶化到 49%**，比不做还差。正解是 GGML 的**联合量化**（`make_qx_quants`）：先按误差最小解出 (scale, min)，压进 6 bit 时让高 2 位**自然**落进同一区间。本机无外网、无法核对 llama.cpp 逐位细节 → 文件头已明写**不声称这是真实的 GGUF Q4_K**，也不声称误差更低。
  - **第 8 项（共享前缀下的缓存账）未动手**：账本该行已**回滚两次**，根因是架构取舍 ——「缓存占用只能由房间报告」与「调度器是纯整数函数、重放门不依赖房间」直接冲突，不是实现细节问题。
- **2026-09-18** —— **P3 第 10 项：A100 恢复（根因是 SSH 端口，不是机器故障）**；第 11 项进行中。
  - `lcl@10.107.6.60` 的 SSH 监听 **3389**，22 / 2222 实测全关（ping 一直通，RTT 0.3 ms）。账本 2026-09-17 记的「连接超时」就是 `scripts/a100.sh` 一直敲 22 造成的。已修：新增 `REMOTE_PORT`（默认 3389），sync / run / shell / probe 全部带上。
  - 第二个坑（更难查）：pixi 环境**不可重定位** —— `.pixi/envs/default/share/max/modular.cfg`硬编码本机绝对路径（`package_root` / `cache_dir` / `path`），rsync 到远程后 mojo 报 `unable to locate module 'std'`（连 `print` / `range` 都找不到，**看着像语法错误，实为环境问题**）。`a100.sh sync` 结尾已自动 `sed` 重写。远程 `/home/rontom` 建不了（无权限），符号链接方案不可用。
  - 实测（GPU 4）：`gpu-query` → A100-PCIE-40GB / CC 8.0 / 驱动 560.35.03；`tests/gpu/vecadd.mojo` → `*** GPU VECADD PASS *** c[999]= 2997.0`，与 §1.2 记录一致 → §1.2 重新可复现。
  - ⚠️ **CUDA 差分仍未通过，§4 那一行不因此改判**：新增 `tests/gpu/test_cuda_diff.mojo`（RMSNorm，以 `kernels/cpu/scalar.rmsnorm` 为参照，fp32 **相对**容差 1e-5，与 AVX2 门同一个数，**只验正确性不报性能**）—— **编译通过，但运行到 `arena created` 后段错误**，尚未定位。按铁律不把没验过的 kernel 算作可用，故本行状态**维持不变**。该门故意不进 `pixi run test`（本地 sm_52 必失败）。
  - **第 11 项**：本机**有外网** → 下 tarball → rsync → 远程 cmake（远程无外网）。⚠️ `git clone` 会超时（curl 能通但 git 不行）→ 改用 v0.4.1 tarball（**无 submodule**，可放心用）；cmake 须显式 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`（否则 `CMAKE_CUDA_COMPILER-NOTFOUND`）+ `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80`。远程已有 gguf：`/app/lcl/models/Llama-3-Taiwan-70B-Q/`（Q4_0 / Q5_K_M / f16）。编译进行中，`llama-bench` 尚未产出 → 250 行「与 llama.cpp 同机同 prompt 对比」仍 `missing`。
- **2026-09-18** —— **P3 第 11 项：llama.cpp 基准通路打通，拿到首个同机同模型 CPU 对比数字。
  - llama.cpp v0.4.1 已在 A100 机编译完成（`llama-bench` / `llama-cli` / `llama-completion` 均在）：
    cmake 须 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80`
    （不给 nvcc 绝对路径会 `CMAKE_CUDA_COMPILER-NOTFOUND`）。源码走本机下载 tarball 再 rsync（远程无外网）。
  - **同机同模型**：远程只有 70B gguf，与 alofa 跑的 Qwen2.5-0.5B 对不上 → 本机把 HF 权重转成
    `qwen25-05b-f32.gguf`（**f32**，1.98 GB，290 tensors）rsync 过去，两边才是同一个模型。
  - **实测（A100 机 CPU，`-ngl 0`，fp32，greedy，同一 prompt）**：
    `llama-completion` decode **5.94 tok/s**（`-t 1`）/ **25.18 tok/s**（`-t 8`）；
    alofa **1.13 tok/s**（`mojo build -O2`；引擎墙钟 14198 ms / 16 new tokens，含 prefill）。
    alofa 侧新加了引擎墙钟输出（`monotonic_ns`，只包 prefill+decode，**不含**权重 mmap 与分词）。
  - ⚠️ **边界（不可省略）**：① alofa 走 **`BACKEND_SCALAR`** —— `rope` / `attention` 本轮只有标量实现，
    AVX2 尚未跑通整网，故 **1.13 是下界，不是 alofa 的水平**；② alofa 数字含 prefill，llama.cpp 的
    eval time 是**纯 decode**；③ 两者 prompt token 数不同（5 vs 13，tokenizer/BOS 差异）；
    ④ **共享机**（6 卡全忙、其他进程在跑）→ 有噪声，**不作吞吐门定论**。
  - 未做：AVX2 档、CUDA 档（后者要等第 10 项 CUDA 差分通过）。第 11 项**仍不算完成**：
    250 行「与 llama.cpp 同机同 prompt 对比」要有 alofa 自己的 GPU/AVX2 数字才算数。
- **2026-09-18** —— **补测 AVX2 档：CPU 基线从「下界」改成 alofa 的真实水平**（接上条）。
  - `cli.mojo` 的 `generate` 改成 `generate[backend: Int = BACKEND_SCALAR]`（后端是**编译期**参数），
    `main` 现在**标量与 AVX2 各跑一遍贪心、两段的文本都打印** —— 同一 prompt 下二者逐字相同
    才配叫基线；AVX2 覆盖 `rmsnorm` / `linear` / `linear_bias` / `add` / `swiglu` 五个算子，
    `rope` 与 `attention` 本轮仍只有标量实现。
  - **实测（A100 机 CPU，`-ngl 0`，fp32，greedy，同一 prompt，16 new tokens 含 prefill）**：
    标量 **1.15 tok/s**（13969 ms）→ AVX2 **2.82 tok/s**（5667 ms），**2.46×**；
    **两段文本逐字相同**（均为 " Paris. It is the largest city in Europe and the second largest in the world"）。
  - 对 llama.cpp 单线程 5.94 tok/s：alofa 由 0.19×（标量）升到 **0.48×**（AVX2）。
    剩余差距的主要来源是 `rope` / `attention` 仍走标量，以及 llama.cpp 可多线程。
  - ⚠️ 边界不变：alofa 数字含 prefill 而 llama.cpp 的 eval 是纯 decode；prompt token 数 5 vs 13；共享机有噪声。
  - 第 11 项**仍缺 CUDA 档**（要等 CUDA 差分通过才拿得到）。
- **2026-09-18** —— **CUDA 差分门的段错误已修，并首次通过**（接 AVX2 档那条）。
  - **根因不是 GPU**：`tests/gpu/test_cuda_diff.mojo` 少了 `arena.keep_alive()`。`Arena.__deinit__`
    会 `munmap`，而 `arena` 变量的**最后一次使用**是第三个 `scratch(...)`（第 91 行），比
    `fill(x, 1)`（第 92 行）还早 → Mojo 在第一个写之前就把整块映射释放了 → 段错误。
    ⚠️ **栈把它指到 `fill:84` 是误导**：GPU 那半边从头到尾没问题。
  - 定位方法（可复用）：逐步做 /tmp 探针 —— ①纯 CPU 复现（本地与远程都过）②保留 GPU import
    与 kernel 但只跑 CPU（过）③GPU 全流程内联在 `main`（过，`diff=0.0`）④与真实文件 `diff`，
    唯一实质差异就是那句 `keep_alive`。
  - ⚠️ `scripts/a100.sh run` **不 rsync**（只 ssh + `mojo run`）→ 改完源码必须先
    `./scripts/a100.sh sync`，否则一直在跑远端旧文件（本次为此白跑了一轮）。
  - **结果**：`*** RMSNORM GPU/SCALAR DIFF PASS *** diff= 0.0  tol= 5.0350770950317384e-05`
    （fp32 相对容差 1e-5，与 AVX2 门同一个数；A100 卡 0）。
  - ⚠️ **这不等于拿到了 GPU 档**：`src/alofa/kernels/` 只有 `cpu/`，产品代码里**没有**任何
    `DeviceContext` / `std.gpu` 引用 —— alofa 目前**没有端到端 GPU 推理路径**。本门只证明
    `rmsnorm` 这一个核与标量后端逐位一致。要定 250 行的 GPU 数字，还得先把前向搬到 GPU。
- **2026-09-18** —— **AVX2 补到 `rope`/`attention`：正确，但对 CPU 基线没有可测影响**（接 AVX2 档那条）。
  - 做法：`avx2.mojo` 新增向量版 `rope` / `attention`；`scalar.mojo` 里两者的**形状契约**抽成
    `rope_shapes` / `attention_shapes` 供两个后端共用 —— 契约抄两遍就会漂移，而漂移的那一侧会以
    「形状检查通过、却读到别人的内存」的形式出现，那比数值不一致难查得多。`qwen.mojo` 加
    `rope_k[backend]` / `attention_k[backend]` 并接到调用点。
  - **数值**：`rope` 与标量**逐位相同**（无累加，f32 8 通道）；`attention` 的值归约也逐位相同
    （沿通道切、每个通道各自**顺序**累加 `j`，没重排累加顺序）；只有打分点积是 4 通道 f64 归约，
    属文件头已声明允许的重排。实测向量 vs 标量**最大绝对差 0.0**。
  - 门：`tests/unit/test_avx2_parity.mojo` 7 → **9** 项（新增「对 HF 参考」一项，并把 rope/attention
    接进「与标量互比」）。另加**不整除形状的尾巴对照**：`head_dim=6` 让 `rope` 的向量主循环
    **一次都不执行**、全程走尾巴，输出先填哨兵 —— 尾巴若被丢掉就具名失败，而不是「恰好整除
    所以看起来没事」（fixture 的 head_dim=64 恰好整除，尾巴从未被跑到）。
  - ⚠️ **实测没有提速**：同一进程内 AVX2/标量 的比值三次为 2.43 / 2.88 / 2.69（绝对数 5345–6679 ms），
    与本轮之前的 2.47 **完全重合**，差异落在共享机噪声里。**不把这次改动记成提速。**
  - 原因是算术上的，不是没测准：本工作负载每 token 的矩阵乘约 152M 次乘加（含 `lm_head` 的
    896×151936），而 attention 约 14×21×64 ≈ 1.9 万次 —— **约 0.012%**。向量化一个占万分之
    一的算子，量级上就不可能在端到端计时里被测到。
  - 推论（**未验证**，是下一步的假设而非结论）：batch-1 decode 每 token 要流过约 2 GB fp32 权重，
    实测 5.3 s / 16 token ≈ 6 GB/s，与**内存带宽**的量级相符。若成立，下一步的杠杆是**减少权重
    流量**（Q4 已实现，但垂直切片没开），而不是继续向量化算力 —— 后者要验证得先量带宽。
- **2026-09-18** —— **batch-1 decode 确认是内存带宽受限**（先前「推论（未验证）」终结；工具
  `scripts/bench_decode_roofline.mojo`，新建）。
  - 判据不靠我自己的算术：三个数（**本机**可用读带宽、**本机** f64 FMA 吞吐、**当下 kernel**
    在权重常驻 cache 时能达到的 GFLOP/s）全由同一台机现场测出，再交给
    `src/alofa/verify/roofline.mojo` 的 `Roofline.bottleneck` 判 —— 它是交叉相乘比大小，**没有
    容差可调**，所以「把阈值放宽一点让它成立」这条路不存在。
  - **算术强度为何恒为 0.5 flops/byte**：批大小为 1 时每个 fp32 权重必须被读一次、也只被用
    一次，`2·N·K` 次浮点运算对 `4·N·K` 字节。没有任何分块技巧能绕开 —— 这是问题本身的
    性质，不是 `_gemm` 的写法问题。
  - **远程机（正是 CPU tok/s 基线所在那台）**：读带宽上限 9.25 GB/s；本机 f64 FMA 47.1
    GFLOP/s；当下 kernel 上限 7.27 GFLOP/s。平衡点 5.09（对机器）/ 0.79（对当下 kernel）
    flops/byte，两侧都高于 0.5 → **memory**。`lm_head`（151936×896）实测 6.86 GB/s = 读带宽
    上限的 **741‰**，纯读同样字节 ÷ 实测 = **0.741** —— 算术操作只让它比「把这块权重原样
    扫一遍」慢 35%。
  - **本机 i7-9700K 同结论**：读带宽 12.0 GB/s、机器 70.7 GFLOP/s、kernel 9.53 GFLOP/s，
    平衡点 5.91 / 0.80；`lm_head` 跑到 11.0 GB/s = **917‰** 上限，纯读/实测 0.917。（本机比
    远程更贴近上限，两台都判 memory；远程是共享机，最后那 26% 里可能有一部分是别人的。）
  - **端到端交叉校验**：每 token 要流的 fp32 权重是 2.108 GB（hidden 层 65.1 MB × 24 +
    `lm_head` 544.5 MB），按实测 6.86 GB/s 是 **307 ms**；而 ~334 ms/token（5345 ms / 16 token）
    里 **92% 就是这笔流量**。顺带校准了早先那个粗算的 ≈6 GB/s。
  - ⚠️ **因此算力侧的剩余空间是有界的**：`_gemm` 换成完美实现，这一档最多快
    **1/0.741 ≈ 1.35×**（本机只剩 9%）。**8× 级的杠杆只有减少每个权重的字节** —— Q4 已实现
    但垂直切片没开；开了之后 AI 从 0.5 升到约 4 flops/byte，仍在机器平衡点 5.09 之下，所以
    那时 kernel 自身的效率会成为新的限制。
  - 第二根杠杆：`_gemm` 只跑到机器 f64 FMA 吞吐的 **15%**（远程 7.27 / 47.1）—— 累加器只有
    一条链，`acc += xv*wv` 被 FMA 延迟按住。**但在字节降下来之前修它收不到多少，别反了顺序。**
  - 第三根杠杆是**批**：今天 `_gemm` 是行外层，多行照样把 w 重读一遍，批大小 M 拿不到任何
    复用。真想吃批的收益得先改循环次序。
- **2026-09-18** —— **`verify/roofline.mojo` 的 `bottleneck()` 在真实量级上整数溢出**（已修）。
  - 溢出有两处：`difference * 1000 <= TOLERANCE * larger` 的两侧，以及 `flops × 带宽峰值` /
    `bytes × 算力峰值` 这两个交叉乘积本身。真实数字下一次前向是 2.7e8 flops / 5.4e8 字节、
    峰值 1.5e10 B/s 与 7.2e10 FLOP/s → 乘积 3.9e19，越过 Int64 上界后绕回负数。
  - 症状**完全静默**：上面那条 `lm_head` 明明 AI=0.5 ≪ 平衡点 4.75，先被判 `balanced`，换一个
    更大的峰值后又被判 `compute`。溢出不是「总是判 balanced」，是**判什么都可能**。
  - 修法：两个峰值先按 `UNIT=1e6` 缩成「每微秒」再乘（两侧同缩，截断误差 ≤1e-6，离判
    balanced 用的 50‰ 容差还差五个数量级）；峰值小到除以 UNIT 会变 0 时退回更小的 UNIT。
  - ⚠️ **为什么原来 13 条测试全绿**：它们用的都是 4 MB 与 1e9 这种数，乘积到不了 1e18，永远
    碰不到这条路径 —— **小数的付讫责任在大数量级上是不成立的**。补了
    `tests/unit/test_verify_roofline.mojo` 两条真实量级用例（13 → **15** 项，含算力受限方向的
    对称一条），并**证明它敏感**：旧公式在同一批数上算出 `diff*1000=-8.76e18`、
    `tol*larger=-8.25e18`，`-8.76e18 <= -8.25e18` 成立 → 错判 balanced；新公式判否。
- **2026-09-18** —— **修正上一条**：两处数字没签名，一条建议被今天的实测推翻。
  - 上一条写的「实测 ~334 ms/token（5345 ms / 16 token）」**没有出处**。账本里唯一有出处的
    端到端数字是本体记者第 440 行：远程 AVX2 **5667 ms / 16 token = 354 ms/token**（2.82
    tok/s）。按远程带宽地板 307 ms 算，解释度是 **87%，不是 92%**。是我把数字记岔了。
  - ⚠️ 更要紧的一句是漏掉的限定：**只有 AVX2 档贴着带宽上限**。远程**标量**档是
    13969 ms / 16 = **873 ms/token**，地板只解释了 **35%** —— 标量侧从来就是**算力受限**，
    上一条把它笼统写成「batch-1 decode 是内存带宽受限」，少了一句 ` backend 限定`。
  - 上一条把「减少每个权重的字节（开 Q4）」列为**杠杆 ①**。今天按同一路径量过了：
    **这条建议是错的**，见下一条。错的根源是：量化通路虽然「已实现」，却从来没被**量过速度**，我把它
    当成了一根现成的杠杆。
- **2026-09-18** —— **Q4 通路首次端到端测速：今天不该开**（`cli.mojo` 现在自含答案）。
  - `cli.mojo` 的 `generate` 加了 `quantize` 参数，返回一趟的完整测量（文本 + 引擎耗时 +
    **量化耗时**），`main` 把四档打在同一张表里。量化耗时（~6 s 本机 / ~8 s 远程，每次
    加载付一次）单独记，**不进 tok/s** —— 混进去的话，胜负会随 `n_new` 漂移。
  - **同进程 A/B**（`mojo build -O2`，同一份权重、同一 prompt，16 new tokens 含 prefill）：
    | 档 | 本机 i7-9700K | A100 远程机 |
    | --- | --- | --- |
    | fp32/scalar | 9983–10694 ms → 1.50–1.60 tok/s | 14153–14336 ms → 1.12–1.13 tok/s |
    | **fp32/avx2** | **3780–3875 ms → 4.13–4.23 tok/s** | **5679–5718 ms → 2.80–2.82 tok/s** |
    | q4_0/scalar | 9397–9512 ms → 1.70 tok/s | 18016–19611 ms → 0.82–0.89 tok/s |
    | q4_0/avx2 | 7933–8166 ms → 1.96–2.02 tok/s | 15785–15996 ms → 1.00–1.01 tok/s |
    | **q4 / fp32（同档）** | 标量 1.06–1.14×；**avx2 0.47–0.49×** | 标量 0.73–0.79×；**avx2 0.36×** |
  - ⚠️ **第一次跑出来是 q4 快 6.5×**：那是远程 fp32/scalar 冷启动 121898 ms（正常 ~14000）
    造成的离群值 —— 同一进程里第三条标量 15410 ms 才是对的。**单次跑就是碰运气，每场 A/B 都要重复**：本机
    第一次也给出假的 1.50×（1.9 GB 权重冷页），重复三次才落到 1.06–1.14×。
  - **为什么字节少 7.1× 却拿不到 2.76×**（就地量 `down_proj` 4864×896，17.43 MB fp32 → 2.45 MB q4）：
    `fp32/avx2` 1.50 ms / 5.83 GFLOP/s / **自己的访存地板÷实测 = 0.847**；
    `fp32/scalar` 4.52 ms / 1.93 GFLOP/s / 0.280；
    **`q4_0/scalar` 4.18 ms / 2.09 GFLOP/s / 0.043** —— 它高出自己的访存地板 **23×**，是个纯算力活。
    且它跟 fp32 标量站在同一堵墙上：**每 flop 吞吐几乎一样（2.09 vs 1.93 GFLOP/s）**，因为
    两者的每权重开销都是标量指令，而量化省掉的是它们都不花的时间。
  - **结论**：「带宽受限」是 **AVX2 fp32 这条路的属性**（roofline 判 memory、达到读上限 748–851‰），
    不是一根能拉动的杠杆 —— **流量还没压下去之前先得让解量化跟上**。今天唯一的正确答案是
    **不开 Q4**：它把最好的那条路（fp32/avx2）拖慢 2.1–2.8×。<br>⚠️ **2026-09-20 复核：这条结论没有推翻，也还没被条件更好的数据顶掉。** 量化核向量化之后**核级**确实快了约 3×（当日 `down_proj` 4864×896 同进程 best-of-7：3.59–3.71 → **1.19–1.25 周期/元素**），但**端到端**同档 `q4 / fp32` 在六次跑里散成 **0.61–2.16×**，连符号都会变（第 5 次 Q4 反而慢）—— 这台上今天**量不出稳定结论**。**在它被认真重测之前，本条「不开 Q4」继续有效。**
  - 顺带交叉校验：远程 fp32/avx2 量出 2.80–2.82 tok/s，与账本已记的 2.82 自洽 —— 这条通路可信。
  - 质量代价（**必须跟着速度一起报**）：q4 档贪心文本整体跑偏（本机、远程都变成
    " a 1000-word essay on the history of the United States."，而 fp32 是 " Paris. It is..."）。
    与 `test_q4_greedy` 已记的逐步一致率 0.80 自洽：自由续写 16 步的存活率约 0.8^16 ≈ 3%。
  - **下一步**：把 `matmul_q4_f32` 向量化（SIMD 拆 nibble + 多条 f64 累加链），对着现有 q4 fixture
    做差分门；**达标线不是「比 fp32/scalar 快」，而是压到自己的访存地板附近**（那个「自己的访存地板÷实测」要接近 fp32/avx2 的
    0.847，而不是今天的 0.043）。在那之前 `cli.mojo` 默认不开 Q4，`test_q4_greedy` 也别动了。<br>    ⚠️ **2026-09-20 补记（这条达标线没达到，且很可能够不着）**：向量版实测该比值 **0.089–0.20**，离 0.847 差一个量级。更该说的是**这条达标线本身定错了**：它的分母「自己的访存地板」依赖同进程里同一次量出的读带宽上限，于是同一台机器三次跑出 0.51 / 1.44 / 0.65（`fp32/avx2` 这一栏自己就在抖）；换成不依赖当次测量的口径，达到 0.847 需要约 **0.2 周期/元素**，而光「4 位 → 浮点」的拆包与换算就要吃掉好几个周期/元素。**判据改成端到端 tok/s 上「值不值得开」**。2026-09-20 在这一项上做了三次，三条都是"量不出结论"：单次顺序跑 0.61–2.16×；重复交错 0.98–1.29×（区间重合）；只 decode + 每 token 中位数 + 预热 + 32 token 后，同一份二进制的两次运行分别给 1.05–1.10× 与 0.99–1.18×（**不可复现**）。**当晚把根因查清之后，答案换了形式**：核级（`down_proj` 4864×896，同进程相邻测量、best-of-3）`q4_0/avx2` = **1.539 ms** vs `fp32/avx2` = **1.485 ms** → **q4 慢 3.6%**；因为 fp32 已经跑在读带宽的 **78%**（访存受限），而 q4 把省下的 7.11× 字节**全花在解量化的算术上**（只用了 10.6% 的带宽）。所以端到端的"效应"本来就只有 **±5%**，小过本机噪声 **±10%** —— 不是"量不出来"，而是"当时确实没有收益"，且瓶颈在**可优化的解量化算术**、不在**硬天花板般的带宽**。**当晚把解量化改成半块切分（核级快 1.36–1.42×）之后，答案翻过来了**：端到端 avx2 档 **1.15–1.31×、整串 > 1.0，两次运行复现** → **开**；标量档 **0.88–0.95×** → **不开**。2026-09-18 那条「不开 Q4」据此**按档拆分** —— 它指的是标量核时代（详见当日变更日志）。
- **2026-09-20** —— **给 `matmul_q4_f32` 的向量化先铺夹具和负向对照（今天没动内核）**：
  新夹具 `tests/fixtures/.../q4vec/`（48 KB / 9 用例，由 `scripts/dump_q4_matmul_cases.py`
  导出），新门 `tests/unit/test_q4_matmul_vec.mojo`（**5/5**，已进 `pixi run test`）。
  - **为什么要另开一份夹具**：原 `q4/` 四个用例的 `cols` 全是 896 —— **每行正好 28 个块**，
    而 28 能被 1/2/4/7/14/28 整除，于是任何「按 N 块展开、余数走标量尾巴」的写法都能
    **全套绕过尾巴**；`rows` 全是 128/896，同样躲过行方向的尾部。这与 2026-09-17 在 avx2
    上踩的坑同一类 —— **fixture 的形状恰好避开了唯一没被走过的分支**。新夹具的块/行取
    1/3/5/7/11/17/28/152、行数取 1/2/3/5/7/13，并留两条真实用例：`real13`（真实 q_w ×
    真实 `norm_out`）与 `down_like`（down_proj 的真实长度 4864 × 真实 `swiglu_out`）。
  - 另覆盖三种数值情形：全零块（`d == 0`）、全部值踩在 ±amax（nibble 只剩 0/15，用来抓
    漏掉 `-8` 偏移）、一堆 1e-8 夹一个 1.0（累加顺序一变相对差就放大）。
  - **三条常驻负向对照**：省掉 `-8` / 高低半字节装反 / 丢掉最后一块。它们的「不可见用例」
    用**显式名单**约束而非降阈值 —— 名单外多一处不可见就红；并单列一条要求：两个真实用例
    必须判红。实测：`no_offset` 拒 7/8（看不见 `tail7`）、`swapped` 拒 4/8（看不见
    `odd_blocks`/`tail7`/`tail11`/`const_row`）、`dropped` 拒 7/8（看不见 `const_row`）。
  - ⚠️ **量出本项目统一判据的一处弱点（已知边界，留着）**：全项目统一的
    `1e-5 × max(1, |ref|)` 在 `|ref| ≪ 1` 时退化成**绝对** 1e-5 —— `const_row` 一例参考值
    约 1.8e-5，于是「少读一整块」（贡献约 5e-6）躲过了判据。**这不是变异无害，是判据在这里
    没牙**；将来改判据时回头看这一条。
  - ⚠️ **新发现的账本校验器边界（做实验证的，不是猜的）**：`extract_field(row,
    "evidence:")` **只取每行第一次出现** —— 同一行里的第二条 evidence 既不做文件存在校验、
    也不做 `?count=` 计数校验。证法：把行内**首条**路径写错 → `check-ledger` 红（6/7）；
    把**第二条**的 `?count=5` 改成 4 → `check-counts` 照样 9/9 绿。所以新门的 N/M **没有**
    以 `?count=` 形式写进能力行 —— 写一个没人核的数字比不写更差（这正是「三个互不相容的数字并存很久」那个坑要防的）。它的 5/5 只写在这里，
    改天按 `pixi run test` 的输出复核。修复校验器（改成核每一行里的全部 evidence）是一件
    独立的事，先记为已知边界。
  - `q4vec/blocks.bin` 已按 `q4/blocks.bin` 的先例加 `.gitignore` 例外，并逐个确认夹具里
    没有别的文件被 `*.bin` 连带忽略。
  - **还没做的事**：向量化内核一行没写，所以本行不升级、不宣称任何性能。下一步是把
    `matmul_q4_f32` 的 SIMD 版接进这个已经活着的门 —— 门现在的形态保证了尾巴一旦写错，
    它是**必然红**的（这条断言的资格是上面三条对照给的）。
- **2026-09-20** —— **`matmul_q4_f32` 的向量版本落地：接过上一轮的差分门、接进模型层分发；核级快约 3×，但端到端今天没测出稳定结论**：
  - **选择器先做好，这是上一轮的兑现**：同一份 9 用例夹具（`q4vec/`）、同一期望值、同一判据
    （`1e-5 × max(1, |ref|)`），只是把传进去的核换成向量版，两条用例摊的是同一条比较，
    于是「换个核就换一套阈值」这种事没有地方发生。门 **7/7**（6 条原有的 + 1 条下面要说的偏差测量），
    已进 `pixi run test`。
  - ⚠️ **门第一次就红了，这就是它的用处**：`odd_blocks` 偏差 13.28。根因与硬件无关 ——
    Mojo 的 `SIMD.fma(a, b)` 语义是 **`self * a + b`**（实测 `3.fma(5, 7) == 22`），不是 `a*b + self`；
    照直觉写会得到 `(a+b)*n` 型的结果，而在 `-O0` 下它只表现为「数不对」，看不出是约数问题。
    源码里已写成反驳式注释，防止有人把它「优化」回去。
  - **矢量的切法**：一个 q4_0 块 = 2 字节缩放 + 16 字节装 32 个 nibble（低半字节是第 j 个、
    高半字节是第 j+16 个）。拆成两半各 8 字节之后，**第 j 个量化值固定落在第 `j % 8` 条通道上**，
    每块的四次乘加各有一条累加链；`d` **折进 x**，整行只在行末归约一次。**这里不需要尾巴处理**：
    每块的数据字节刚好是两个完整的 8 字节，余数由块布局固定给出 —— 与上次那个坑的区别是，
    上次是「head_dim 恰好整除，于是尾巴从头到尾没被跑到」，这里是「换模型也不会变」。
  - **性能（本机，`down_proj` 4864×896，同一进程内 best-of-7）**：标量 **3.59–3.71 周期/元素**
    （4.34–4.52 ms）→ 向量 **1.19–1.25**（1.44–1.51 ms），约 **3×**。对照组：「只把这块 2.45 MB
    块流读完」只要 **40–51 µs** —— 差 30 倍，所以瓶颈**全在计算侧**，不是访存（`_gemm` 那套
    「按带宽算」的直觉在这里不成立，别拿过来用）。
  - ⚠️ **四种结构都试了，两种「对症」的猜测被自己的数据否掉**（同进程 best-of-7，周期/元素）：
    每块归约 + 单链 1.25 / 每块归约 + 四链 1.37 / 整行归约 1.25 / **整行归约 + `d` 折进 x 1.19**（采纳）。
    试过按 4 行分组以便跨行复用 x —— `InlineArray` 的下标走了栈，**反而慢 40%**，已回退并写进注释。
    「四条独立的链能把 4 周期乘加延迟盖住」这个推断是错的：**它是被嵌在 docs 里的一条建议，
    不是被量出来的结论**，今天第一次量就成了负数。
  - ⚠️ **顺产量了「累加换 f32 值不值」**：f32 累加 **0.83 周期/元素（≈1.01 ms）**，比 f64 版再快 **1.4×**，
    九条用例的最大相对偏差实测 **8.38e-08**（`real13`，门限 1e-5，**低 120 倍**），其中 `down_like` 8.98e-09。
    **没有**把它设为默认：换的是判据层的算术，而理由恰好写在 avx2 那行 —— 「点积有抵消，f32 累加在
    抵消严重处能吃掉整个 1e-5 判据」。要动它得先回答「相消最严重那处还剩多少余量」，今天没人量过；
    它以 `_matmul_q4_f32acc` 的形式**只作为被测对象**存在，不接任何调用路径。
  - **模型层分发**：新增 `q4_matmul_k[backend]` / `q4_matmul_bias_k[backend]`，读的是与 `backend_label`
    **同一个判断** `uses_vector_backend`。四处曾经写着「量化通路本轮只有标量实现」的注释
    （`qwen.mojo` 三处、`cli.mojo` 一处）已改成事实 —— 留着就是假地图，而这张图会被用来决定要不要开 Q4。
  - **端到端（本机，16 new tokens 含 prefill，`target/cli` 同进程四路）—— 今天**没测出结论**，以下是全部六次：

    | 次 | fp32/scalar | fp32/avx2 | q4_0/scalar | q4_0/avx2 | **q4 / fp32（avx2 档）** |
    |---|---|---|---|---|---|
    | 1 | 1.356 | 2.603 | 1.596 | 4.144 | 1.59× |
    | 2 | 0.882 | 1.634 | 1.381 | 3.529 | 2.16× |
    | 3 | 1.076 | 2.152 | 1.341 | 3.699 | 1.72× |
    | 4 | 1.621 | **4.512** | 1.637 | **4.573** | 1.01× |
    | 5 | 1.625 | **4.524** | 1.514 | **2.761** | **0.61×** ← Q4 更慢 |
    | 6 | 0.733 | 1.859 | 1.400 | 3.520 | 1.89× |

    比值散在 **0.61–2.16×**，而且**符号会变**（第 5 次 Q4 反而慢）。绝对 tok/s 本身也抖得离谱：
    同一份 `fp32/avx2` 六次给出 1.63–4.52 tok/s（**2.8×**），第 4/5 次连着两 ~4.5，又在第 6 次掉回 1.86。
    ⚠️ **原因没查明**（页缓存命中 / 调频 / 邻居负载都没排除），所以**不许拿它去解释任何别的数字** ——
    能写下的只有：`cli` 的四个 arm 每档只跑一遍、顺序执行，这个精度不够回答「该不该开 Q4」。
    对照：上面 micro 档是**交错 best-of-7**，它的 3× 才是我今天唯一敢报的速度结论。
    **下一步是把「每 arm 重复 + 交错」这套做法搬进 `cli.mojo`，而不是再多跑几遍同样的东西。**
  - 顺带在两个后端都等的 `q4_0` 上校对了一件事：标量/向量两档吐出的 16 个 token **逐位相同**
    （第 4 次：`[264, 220, 16, 15, 15, 15, 37328, 8895, 389, 279, 3840, 315, 279, 3639, 4180, 13]`，两档一致）；
    这是**一致性**而非性能门，也**不是**一道会因为回归而红的自动化门。
  - ⚠️ **质量代价没变，必须跟着速度一起报**：`q4_0` 的自由续写文本仍然与 fp32 不同（cli 打印
    `文本一致: False`），`test_q4_greedy` 的教师强制贪心一致率仍是 **418/512 = 0.81640625**（今日重跑，
    逐位等于已记基线 —— 顺带说明这道门在**默认（标量）后端**下跑，它**没有**走今天新写的向量核）。
    **速度快了不等于量化够好**：0.90 那条质量门仍是 `missing`。
  - **还没做的事**：① 「AVX2 + q4」**没有**端到端的数值门 —— 今天只有「两个后端各跑一遍、文本逐位相同」这种
    同一进程对照，它没有 fixture，也不是一道会因为坏掉而红的门（`backend_label` 那条只证明名字与分发共用一个判断）。
    ② f32 累加的余量（见上）。③ `lm_head` 仍未量化（这让 Q4 的天花板从 7.1× 掉到约 2.8×）。

- **2026-09-20** —— **端到端 A/B 改成「每 arm 重复 + 交错 + 轮间转顺序」（`cli.mojo` 新增 `bench_ab`），并据此订正上午那条「量不出结论」的归因**：
  - **做法**：`generate` 拆成 `run_slice`（模型由调用方给，可复用同一份权重跑很多趟）+ `generate`（加载一次、跑一趟）。A/B 现在把 fp32 与 q4 两份权重**各加载一次**，然后 `BENCH_REPS=4` 轮 × 四档，轮内四档全跑，第 `r` 轮从第 `r` 档起（`arm = (r + k) % 4`）—— 四轮下来每档恰好各占一次第一/二/三/四个位置，「排在前面所以快」这个偏差是被摊平的，而不是由某一次的顺序决定。模型复用要 `reset()`：KV 缓存是实例字段，上一趟的历史不清掉下一趟的 prefill 会带着旧位置一起算。
  - **判定只有一条**：比**区间重不重合**，不比均值。比值区间取最保守的两端（`q_min/f_max … q_max/f_min`），因为它假设两次测量的抖动方向相反 —— 这正是上午那六次里实际发生的事。重合 = 这份数据区分不出两者，照旧记「量不出结论」。
  - **本机实测（16 new tokens 含 prefill，`mojo build -O2 -I src`，同一进程、同一份权重、同一个 prompt）**：

    | 轮 | fp32/scalar | fp32/avx2 | q4_0/scalar | q4_0/avx2 |
    |---|---|---|---|---|
    | 1 | 1.60 | 3.64 | 1.61 | 4.52 |
    | 2 | 1.64 | 4.53 | 1.62 | 4.62 |
    | 3 | 1.63 | 4.09 | 1.62 | 4.68 |
    | 4 | 1.59 | 4.19 | 1.61 | 4.42 |

    `q4 / fp32（标量档）` = **0.98× … 1.02×**；`q4 / fp32（avx2 档）` = **0.98× … 1.29×** —— **两个都区间重合，判定仍然是「量不出结论」**（avx2 档重合的原因具体是：`fp32/avx2` 最好那次 4.53 高于 `q4_0/avx2` 最差那次 4.42）。
  - ⚠️ **上午那条 2.8× 抖动的归因要订正**：改用同进程交错重复之后，比值散度从 0.61–2.16× 收成 0.98–1.29×，绝对值也从 `fp32/avx2` 的 1.63–4.52 收成 3.64–4.53。所以那 2.8× **主要不是机器噪声，而是「四个 arm 各跑一遍、顺序执行」这个测量方式本身造成的** —— 上午写的「原因没查明（页缓存 / 调频 / 邻居负载都没排除）」那句过强了，真正没查明的只剩残余的约 ±12%。**「先改测量方式，再谈结论」这一步被自己的数据证实了。**
  - ⚠️ **不许反过来读**：区间重合 ≠ Q4 没变快。能写的是「今天这份数据不足以回答」，**不是**「Q4 与 fp32 一样快」。两点只作为**下一次往哪测**的线索记录：① 标量档 `q4` 1.61–1.62 对 `fp32` 1.59–1.64 完全重合（该档走的是标量核，今天新写的向量核不参与）；② avx2 档 `q4` 的下界 4.42 比 `fp32` 的下界 3.64 高约 1.21×，且 `q4_0/avx2` 自己更稳（4.42–4.68，±3% 对 ±12%）。
  - ⚠️ **第 1 轮天生吃亏**：权重是 mmap，惰性读页发生在第一轮前向里（`cli` 打的「权重加载 fp32: 0 ms」就是这个意思，别把它当加载成本）。这也是四个变体必须交错、不能各跑一遍的第二个理由 —— 顺序跑的话这份成本只落在第一个 arm 头上。
  - **下一步**：区间仍重合，剩下的障碍是 `fp32/avx2` 那一档自己的抖动（3.64–4.53，±12%）。要分离，要么降抖动（更长的生成、把 decode 与 prefill 分开量），要么加轮数 —— ⚠️ 加轮数只会让区间更宽、判定更保守（min 更低、max 更高），它**不是**让结论变好看的旋钮。

- **2026-09-20（第二次，当晚）** —— **继续降抖动：prefill/decode 分开 + 每 token 中位数 + 预热轮 + 32 新 token，并补「同轮配对比值」口径。最后仍是「量不出结论」，但这次的证据比前两次都强：同一份二进制的两次运行给出了相反的结论**：
  - **改了什么**（`cli.mojo`）：① `run_slice` **逐拍计时**，把 `prefill`（整段 prompt 的一次前向，固定成本）与 `decode`（每 token）分开记 —— 混在一起 tok/s 会随 `n_new` 漂移，且那一次性开销会稀释真正要测的差别；② 主统计量改成**每 token 时间的中位数**（一个被打断的 token 能把 16 个样本的均值推走 6%，推不动中位数），均值口径保留作对照；③ 开跑前加一轮**预热（2 个 token，结果丢弃）** —— 权重是 mmap 的，而**一个 decode step 会把全部权重读一遍**，所以两个 token 就够把每一页摸过；缺页量的是内核的页管理，不该由第 1 轮第一个跑的那档来付；④ 生成长度 16 → **32**（`MAX_GEN`=32 是引擎给一条请求的上限，没有更长的余地）；⑤ 新增**同轮配对比值**（第 i 轮 `q4 ÷ fp32`）作主口径 —— 交错本来就是为了让这个比值成立；各档自己的非配对区间保留作**保守对照**，它会被"这一轮整体快慢"这个共模因子撑宽。逐拍计时写在 `cli.mojo` 而不是引擎里，是因为 `engine/core.mojo` 有零分配源码门。
  - **抖动降下来过，但没站住**：第 1 次运行（`fp32/avx2`，中位数口径）`1.79 … 1.85` / `4.66 … 4.98` tok/s = **±3.4%**（前一版是 ±12%）；第 2 次运行又回到 **4.50 … 5.39（±18%）**，原因是第 3 轮**整轮**掉到 4.50。所以"抖动降到 ±3%"这个说法**不成立**，账本只写"降下来过，没站住"。

    | 运行 | 轮 | fp32/scalar | fp32/avx2 | q4_0/scalar | q4_0/avx2 |
    |---|---|---|---|---|---|
    | 第 1 次 | 1 | 1.80 | 4.66 | 1.82 | 4.99 |
    | 第 1 次 | 2 | 1.79 | 4.80 | 1.87 | 5.27 |
    | 第 1 次 | 3 | 1.83 | 4.88 | 1.88 | 5.18 |
    | 第 1 次 | 4 | 1.85 | 4.98 | 1.88 | 5.22 |
    | 第 2 次 | 1 | 1.92 | 5.39 | 1.90 | 5.35 |
    | 第 2 次 | 2 | 1.89 | 5.19 | 1.86 | 5.42 |
    | 第 2 次 | 3 | 1.68 | **4.50** | 1.87 | 5.30 |
    | 第 2 次 | 4 | 1.89 | 5.18 | 1.88 | 5.22 |

    （32 新 token，**只 decode**，每 token 时间的中位数，tok/s；同一进程、同一份权重、同一个 prompt）
  - ⚠️ **今天最有价值的新事实：抖动是 fp32 那一侧的，不是 q4 的**。两次运行共 8 个样本/档：`q4_0/avx2` = 4.99–5.27 与 5.22–5.42（**±1–2%**）、`q4_0/scalar` = 1.82–1.88 与 1.86–1.90；而 `fp32/avx2` = 4.66–4.98 与 **4.50–5.39**、`fp32/scalar` = 1.79–1.85 与 **1.68–1.92**。而且 fp32 的偏离**方向全是向下掉速**（从不向上），q4 八个样本里一次掉速都没有。→ 前面所有"量不出结论"的失败，**卡的是 fp32 这一侧，不是量化通路**。
  - ⚠️ **结论：仍是「量不出结论」—— 这是第三次独立确认，而这次的证据不是区间重合，是结论不可复现**：第 1 次运行的 `q4/fp32（avx2 档）` 配对比值 **1.05× … 1.10×（整串 > 1.0）**，非配对区间也不重合（4.99 > 4.98）—— 差一点就要写"q4 更快"；第 2 次同一份二进制给 **0.99× … 1.18×（含 1.0）**，当场把它推翻。→ **任何单次运行的结论都不可信，哪怕它做了重复与交错。**
  - 未做的判定：① 不因为"fp32 侧更抖"就反过来说 q4 更好 —— 抖是**测量**的性质，不是**通路**的性质；② 不拿第 1 次运行那个不重合的区间当结论；③ 也不把 fp32 的掉速算作 fp32 的"真实性能"（那会让 q4 显快，属于自己造结论）。2026-09-18 那条「不开 Q4」**继续有效**。
  - **下一步**：查 fp32 侧**整轮掉速**的根因。待验证的**假说**（不是结论）：fp32 是 2.1 GB / 4 KB 页 ≈ 52.5 万页，q4 只有 ≈ 13 万页 → TLB / 页表压力不成比例，且 fp32 每 token 要流 4× 的字节。在它查清之前，本机这条端到端通路给不出稳定的答案。

- **2026-09-20（第三次，深夜）** —— **查 fp32 侧整轮掉速的根因：三条假说，两条被证伪，剩下那条不在我们的代码里**：
  - **假说 A（页 / TLB / 缺页）→ 证伪**。`cli.mojo` 加了逐档的 `/proc/self/stat` 采样（Δminflt / Δmajflt / Δstime）与邻居 runq。两次运行共 **32 个样本，Δmajflt 全为 0**；Δminflt **恒定**（fp32 档 3568、q4 档 3610 —— 那是每趟 `run_slice` 新建 arena 的匿名页，与掉速无关）；Δstime 0–190 ms，且与掉速无关。→ fp32 的权重（文件映射，1.976 GB）**全程常驻**，没有被回收、没有读盘。
    - 顺带确认了结构上的不对称：fp32 档权重是**文件映射**，q4 档权重在**匿名** `q4_arena` 里。但这次量下来，这个差别**没有造成**掉速。
  - **假说 B（带宽争抢：fp32 贴着天花板所以更脆）→ 证伪**。正对照：在第 3 轮整轮期间跑一个已知强度的读带宽占用者（numpy 512 MB 反复求和，**7.9 GB/s × 55 s**）。结果是**四档同幅掉速**：`fp32/scalar` −14%、`fp32/avx2` −12%、`q4_0/scalar` −10%、`q4_0/avx2` −14%。若机制是带宽争抢，每 token 流 1.976 GB 的 `fp32/avx2` 必须远惨于只流 0.746 GB 的 `q4_0/avx2`；实测两者一样。
    - ⚠️ **顺带修正一个会误导判断的数**：账本里的"本机读带宽 12.0 GB/s"是**单线程**读带宽，**不是整机上限**（本机是双通道 DDR4）。这次同一脚本实测峰值 **15.03 GB/s** —— 也就是说**同一台机器不同时刻在 12.0–15.0 GB/s 之间漂**，又是一条"分母依赖当次测量"的实例（见测量规矩 ④）。"fp32 在 80% 天花板"说的是**单线程**天花板，不代表机器带宽见底；这也解释了为什么一个 7.9 GB/s 的邻居没能压垮它。
  - **假说 D（机器过载：邻居抢 CPU 时间 / 共享 L3）→ 与数据一致，但未证明**。四档同幅掉速正是"CPU 时间份额被稀释"的签名；本机 runq **6–27（8 核）**、loadavg 常年 10–12。⚠️ 写成"一致"而不是"证明"：我没法把 VS Code / Chrome 关掉来验证（那不是我的进程），所以只写到"**其余两条已排除，这条与数据一致**"。
  - ⚠️ **修正今天早些时候写进账本的一条**：「抖动是 fp32 侧，不是 q4 侧」**站不住**。那是从 2 次运行 8 个样本里的 **1 个异常轮**（第 2 次第 3 轮的 4.50）读出来的。第 4 次运行里 `fp32/avx2` 4.68–5.10（**±4.3%**）与 `q4_0/avx2` 5.02–5.42（**±3.8%**）**一样大**。
  - **本次最有价值的副产品 —— 量化收益的真相（先修两个算术错误，再做一次核级实测）**：
    - ⚠️ **"量化把每 token 字节降到 1/4"是错的**：q4_0 是 **32 个值 18 字节**（16 B nibble + 2 B fp16 scale）= 每值 0.5625 B → 相对 fp32 是 **7.11×**；而 `lm_head` **不量化**（与 `embed_tokens` 绑定共用，`tensors.tsv` 里两者 offset 都是 0，136.13M 参数 = 0.544 GB）。所以 q4 档每 token 仍要流 **0.746 GB** = fp32 的 **38%（2.65×）**，其中未量化的 `lm_head` 自己就占 **73%**。
    - **核级（`down_proj` 4864×896，同一进程相邻测量，best-of-3）**：`fp32/avx2` = **1.485 ms**（11.74 GB/s = 读峰值的 **78%**，访存受限）；`q4_0/avx2` = **1.539 ms**（1.59 GB/s = 自己访存地板的 **10.6%**，**算术受限**）→ **`q4_0/avx2` 比 `fp32/avx2` 慢 3.6%**。（此前那条"核级 ~3×"是拿 q4/**标量** 4.415 ms 当参照物比出来的 —— 对"该不该开量化"这个问题，参照物搞错了：该比的是 `fp32/avx2`。）
    - → **"该不该开量化"的答案不是"量不出结论"，而是"现在基本没有收益"**：按上面的每元素速度推算，端到端上限 ≈ **1.03×**（（357.85M×1.036 + 136.13M）÷ 494.1M），实测 0.98–1.15× 与之相符。也就是说**效应本身（±5%）比本机噪声（±10%）还小** —— 三番五次量不出来不是测量不够好，是真的没有可量的差别。
    - 但 **`q4_0/avx2` 只用了 10.6% 的带宽 → 它的限制是算术，而算术是可以优化的**（带宽是硬天花板，算术不是）。若把它做到 ~0.11 ns/元素（现在 0.353 ns/元素），端到端上限约 **1.9×**。这是"还值不值得往量化里投入"的判断依据，也是下一步唯一说得通的杠杆。
  - 门：`check-ledger` / `check-counts` 待跑；`mojo build -O2` 无 error。改动仍未提交。

- **2026-09-20（订正，紧接上一条）** —— ⚠️ 账本里"每 token 流 2.108 GB"是**旧的高估**，别再引用：它按 65.1 MB/层 × 24 + `lm_head` 0.5445 GB 得来，而每层实为 **59.6 MB**（q/k/v/o/gate/up/down = 14,909,440 参数）。**权威数是权重文件的字节数 1,976,393,216 B = 1.976 GB = 494.1M 参数**（= 24 层 357.85M + `lm_head` 136.13M；`lm_head` 与 `embed_tokens` 绑定共用一份字节）。已写进 `cli.mojo` 的 `BYTES_PER_TOKEN_FP32/Q4` 与文件头。
  - 连带影响两条：① "q4 端到端天花板 2.76×" 改为 **2.65×**（(1431/7.11 + 544.5) ÷ 1976）；② 那条"端到端交叉校验（2.108 GB ÷ 带宽 → 307 ms，实测 354 ms = 87%）"的分子要按 1.976 GB 重算（≈ **60%**）—— **它不再支持"AVX2 档已达带宽上限"这个说法**，待重测，在此之前别当作已验证结论往外引。

- **2026-09-20（第四次，深夜）** —— **杠杆 ④ 落地：q4 解量化改「半块切分」，端到端第一次量出结论**：
  - **改了什么**：`avx2.mojo` 新增 `_matmul_q4_halves` 并接成 `matmul_q4_f32` 的**默认**通路。与旧通路（`_matmul_q4`，按块内 `j%8` 切 f64 通道）的算术**完全一样多**，差别只在一个 q4_0 块怎么拆成向量：一次读满 16 个数据字节，低半字节对应值 0..15、高半字节对应 16..31 —— **两边各自配上 x 的一段连续区间**；旧通路按 8 字节读两趟，nibble 的提取与转换要做四遍而不是两遍。省的是**指令数**，而 q4 通路本来就是算术受限（只用到自己访存地板的 10.6%）。
  - **核级（`down_proj` 4864×896，best-of-3，三次运行）**：旧通路 1.552 / 1.538 / 1.566 ms，新通路 **1.121 / 1.130 / 1.101 ms** → **快 1.36–1.42×**，且新通路自身三次只差 ±1.3%。
  - **与 `fp32/avx2` 的比**：1.805 / 1.761 / 1.464 ms（⚠️ 这个数**随机器状态漂**，因为它是访存受限的）→ 新 q4 通路比它快 **1.61 / 1.56 / 1.33×**；而在此前一天的量里它是**慢 3.6%** —— 量化从"没有收益"变成"有收益"，卡点一直是解量化的算术，不是带宽。
  - **端到端 A/B（只 decode + 每 token 中位数 + 预热丢弃 + 32 token + 同轮配对比值）两次运行**：avx2 档 **1.15× … 1.24×** 与 **1.21× … 1.31×**，**整串 > 1.0**，且中位数（1.12–1.26×）与均值（1.13–1.26×）两个口径**同向** → **q4 更快**。标量档 **0.88× … 0.93×** 与 **0.92× … 0.95×** → q4 **更慢**（与"标量侧一直算力受限"的旧记录一致）。这是三次都「量不出结论」的那个数第一次有了结论 —— 不是统计口径终于对了，是**效应终于大到压过噪声**（本机端到端噪声 ±10%）。
  - **代价与判据**：累加从 f64 换成 f32，与标量版**不再逐位一致**。真实用例的最大相对偏差（`test_q4_matmul_vec.mojo` 逐例打印，该门现在 **8/8**：新增一条半块通路的偏差测量）`real13` **5.5e-08**、`down_like` **1.3e-08**，判据 1e-5；模型级 `test-q4`（teacher forcing 贪心一致率）通过（298 s）。旧通路以 `_matmul_q4` 保留，并在 `bench_decode_roofline.mojo` 里作为 `q4_0/f64-8w` 继续被量 —— 别让"与标量版逐位一致"这个性质悄悄消失。
  - ⚠️ **2026-09-18 那条「不开 Q4」按档拆分**：它指的是**标量核**时代；在 **avx2 档**上现在是 **1.15–1.31× 的收益**，标量档仍然是负收益（0.88–0.95×）。

- **2026-09-20（第五次）** —— **多线程的第一块砖：先把「N 个并发任务能把读带宽拉到多少」量成硬件事实**（`evidence:scripts/bench_thread_bandwidth.mojo`）：
  - **为什么先测这个，而不是直接改模型**：账本里 alofa 与 llama.cpp 的差距有一半写在明处 —— 同机 `-t 1` **5.94 tok/s** 对 `-t 8` **25.18 tok/s（4.24×）**。而我们所有的带宽数字（本机 12 GB/s、远程 9.25 GB/s）都是**单线程**量出来的。一个线程扫不满双通道 DDR4 是硬件常识，但**它是常识不代表在这台机上成立**：本机 8 核常年在 runq 6–27 之间，邻居不受我控制。所以在动一行模型代码之前，先把它量成一个数。
  - **Mojo 1.0.0 没有线程模块**：`thread` / `threading` / `concurrent` / `parallel` 逐个 import 试过，全是 `unable to locate module`。可用的只有 `std.runtime.asyncrt` 这一层协程 + `TaskGroup`（`parallelism_level()` 报 **8**，与 `nproc` 一致）与 `std.atomic`。⚠️ 因此这里量到的不只是「多线程的收益」，也**包含这套运行时的调度开销** —— 若开销把收益吃掉，那也是结论的一部分。
  - **实测（256 MB fp32 顺序读，交错 5 轮，min…max GB/s）**：T=1 **11.39–14.39**、T=2 **14.81–18.15**、T=4 **13.07–21.67**、T=6 **16.95–24.55**、T=8 **17.68–22.69**。
    - **T=1 与 T=8 的区间不重合** → 这差别是真的。保守加速比 = T=8 的 min ÷ T=1 的 max … T=8 的 max ÷ T=1 的 min = **1.23× … 1.99×**，两个端点都 > 1。
    - ⚠️ **不是单调的**：T=4 在第 3 轮只有 13.07、T=6 在第 5 轮却有 24.55。本机干扰照样在，只是**这次效应（≥1.23×）大过噪声**，所以量得出来 —— 这正是它比「量化那条 ±5% 的效应」好办的地方。
  - **粒度能有多细**：一次「建任务组 + 等它结束」（计时**含** `nop()` 的调用，协程帧是在调用点分配的）T=1 **1.77 µs**、T=2 **1.15**、T=4 **1.58**、T=6 **3.00**、T=8 **2.58 µs**。若**每个矩阵乘都起一组**（一次 decode 是 24 层 × 7 个投影 + `lm_head` = **169 次**）→ **0.19–0.51 ms/token**，对 ~300 ms/token 来说是 **0.06–0.17%**。→ **粒度可以细到单个矩阵乘**，不需要为了摊开销去做跨层流水。
  - 顺带：T=1 这一轮读到 **14.39 GB/s**，而账本记的「本机单线程 12.0 GB/s」是同一脚本另一时刻的值 —— 又漂了一次，再次印证测量规矩 ④（分母依赖当次测量，别当固定常数）。
  - **⚠️ 这条只回答了「值不值得做」，还没回答「做了能拿多少」**：聚合读带宽涨 1.23–1.99× 是**纯访存**的收益；`fp32/avx2` 核级已达单线程读峰值的 78%，它能吃到多少要看实测，而 `q4_0/avx2` 是算术受限、理论上更接近线性。下一步是**核级 A/B**（`down_proj` 4864×896，T=1 vs T=8，两条通路都量），不是直接接进模型。

  - **接着往下问了一步：核级 A/B（`evidence:scripts/bench_thread_matmul.mojo`）** —— 同一个 `down_proj`（896×4864）按输出列切成 T 份，两条通路都量，T=1 用**直呼**（今天线上那条路）当基线：
    - `fp32/avx2`：T=1 **1.523–1.698 ms**；T=2 0.965–1.107（1.38–1.76×）；**T=4 0.713–0.761（2.00–2.38×）**；T=8 0.478–1.431（1.06–3.55×，第 3 轮有一次 1.431 的离群）。
    - `q4_0/avx2`：T=1 **0.786–0.834 ms**；T=2 0.429–0.536（1.47–1.94×）；**T=4 0.232–0.357（2.20–3.59×）**；**T=8 0.145–0.213（3.69–5.76×）**。
    - **两端都 > 1 的那几格（T=4 的 fp32、T=4 与 T=8 的 q4）就是"有差别"**，且效应 2–4× ≫ 本机噪声 ±10% —— 这是这台吵机器上第一个**不用争**的结论。
    - 顺带：T=8 时 q4 每元素 **0.033 ns**、fp32 **0.110 ns**；q4 相对 fp32 的核级优势从 T=1 的 1.9× 拉到 T=8 的 **3.3×** —— 因为 fp32 越并行越撞上访存墙，q4 撞的是可优化的算术。
  - ⚠️ **踩到一个测量陷阱，第一版数字是假的**：`down_proj` 的权重只有 **17.4 MB**，本机 L3 **12 MB** —— 照 `bench_decode_roofline.mojo` 那套 best-of-3 反复跑同一块权重，第 2、3 趟里有相当一部分**没走 DRAM**，于是报出 T=8 **0.216 ms** = **80 GB/s**，比实测聚合峰值高 **3.5 倍**，物理上不可能。改成 **8 份互不相交的权重副本（139 MB）各算一趟取平均**（不再取最优）后才收敛。真实前向每 token 流 **1.976 GB**，一次都不会有这种复用 —— **任何小于 L3 的核级 bench 都得先问这一句**。
  - ⚠️ **连带修正带宽探针那个数**：`bench_thread_bandwidth.mojo` 的求和循环是「8 宽单累加器」，`add` 延迟 4 周期 → 每 4 周期才 8 个元素 = **25.6 GB/s 的天花板**，而 T=8 实测 22.69 GB/s 已经贴到它了。所以那份探针报的 **22.69 GB/s 是"累加器链"的顶，不是 DRAM 的顶**（`fp32/avx2` 8 个独立 f64 累加器时跑到 36.5 GB/s）。**比值**仍然可用，**绝对值**不要在别处引用。
- **2026-09-20** —— **端到端：把投影切成 N 片并发，一个 token 快 1.66–1.84×（`scripts/bench_model_shards.mojo` 新增）**：
  - 它接 `bench_thread_matmul.mojo` 往下走一站：那份是**核级**（fp32 T=4 2.00–2.38×），这里量的是**真的一个 token** —— 169 次投影之外，注意力 / RMSNorm / RoPE / 残差那几段**一次都没被切**，按 Amdahl 原样留在串行段里，天花板当然不是核级那个数。
  - 口径：Qwen2.5-0.5B **fp32 / avx2**，prefill 后贪心解码 32 个 token，取每 token 时间**中位数**；3 轮交错（`arm = (rep + k) % 4`），主口径 = **同轮配对比值**，非配对区间作保守对照，预热轮（2 token）丢弃；顺带读 `/proc/loadavg` 的 runq（实测 5–8）。**基线 `shards=1` 走直呼**，不是"换调度机制"的那条路。
  - 数：**162–175 ms/token（5.73–6.16 tok/s）** → 切 2 片 **1.48–1.62×**、4 片 **1.49–1.81×**、8 片 **1.66–1.84×**。两个口径（每 token 中位数 / decode 均值）**同向**，非配对区间也**不重合**（8 片 1.69–1.90×）。
  - **跨运行复现**：两次运行分别给 1.51–1.62 与 1.48–1.54（2 片）、1.49–1.81 与 1.64–1.75（4 片）、1.72–1.84 与 1.66–1.75（8 片）。效应 1.5–1.8× ≫ 本机噪声 ±10% —— 这是这台机器上第一个**端到端**测得出方向的结论（对照：q4 vs fp32 的 A/B 做了三次都没量出来）。
  - ⚠️ **只针对批 = 1 的单流 decode**：批大于 1 时分片改切**输出行**（`dst` 行主序、列不连续），那条路**没量过**。
  - ⚠️ **放大倍数是 1.7×，llama.cpp 同机是 4.24×**（`-t 1` 5.94 → `-t 8` 25.18 tok/s）。绝对 tok/s 从约 6 抬到约 10.5，对 llama.cpp `-t 8` 的 25.18 = **0.41–0.43×**，相对位置几乎没动。差的那一截落在**没被切的那几段**与**每 token 169 × N 次协程调度**上 —— 这是下一步该看的地方，不是"已经追平"。
