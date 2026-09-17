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
| `scalar` (fp32) | `verified` | `evidence:tests/unit/test_layer0_parity.mojo`（13/13 通过，逐算子对 Hugging Face）<br>`evidence:tests/unit/test_model_parity.mojo`（整网 4/4 通过，含 128 token greedy 逐 token 相等）<br>作为数值 oracle：只按公式顺序写，**不做任何重排**，供后续向量后端对照 |
| `avx2` (8×fp32 / 4×fp64) | `verified` | `evidence:tests/unit/test_avx2_parity.mojo`（7/7，已并入 `pixi run test`）：rmsnorm / q·k·v 投影 / o 投影 / 残差加 / 第二道 RMSNorm / swiglu，**与标量后端同一份 fixture、同一条容差**（`1e-5 × max(1, |参考|ₘₐₓ)`）<br>与标量后端**逐位相同**（五个算子最大绝对差实测 **0.0**）—— 这是量出来的，不是假设的：同一条比较在偏置写错时确实红过<br>累加仍在 **f64 通道**（4 道）：点积有抵消，换成 f32 累加时相对误差约 `√n · 2^-24`，抵消严重处能吃掉整个 1e-5 判据 —— 与标量同一理由，**这一层当判据不当最快路径**<br>⚠️ **本轮不宣称指令，也不宣称性能**：模块名说的是"按 8 通道 f32 / 4 通道 f64 写的向量 kernel"，是否真的降成 VEX 编码指令**没有做 objdump 核验**；性能数字要等 §10 roofline 接入后才有资格谈<br>负向对照专钉**尾巴**：fixture 里 896 / 4864 恰好都能被通道数整除，标量尾巴**从未被真正执行过**；故另写一个只跑主循环、丢掉尾巴的版本，喂长度 10（非 8 的倍数）的输入，必须判红（实测相差 10.0）<br>另有一条：主机若没有 AVX2 则**具名失败**，不静默跳过 —— 会跳过的门在 CI 上永远绿，而它绿的原因是没跑<br>**已接入整网（2026-09-17）**：`prefill` / `step` / `run` 带编译期参数 `backend`（`BACKEND_SCALAR` / `BACKEND_AVX2`），五个算子（rmsnorm / linear / linear_bias / add / swiglu）按它静态分发<br>`evidence:tests/unit/test_model_avx2_parity.mojo`（`pixi run test-model-avx2`，依赖 2 GB 权重，**不进 `pixi run test`**）：与标量整网门**同一批 prompt、同一段 fp32 参考、同一条判据**（logits 余弦 ≥ 0.999 且 argmax 全等、128 token greedy 逐 token 相等、逐 token 解码与整段 prefill 落点一致）<br>⚠️ **`rope` / `attention` 与量化通路仍只有标量实现** —— 整网跑在`BACKEND_AVX2` 上时它们是混跑的，不假装全覆盖<br>⚠️ **"跑的确实是向量后端"这件事，数值上证明不了**：两个后端在这些形状上逐位相同（这是设计目标，也是上一轮量出来的），任何数值比较都区分不了它们。守着它的是两样东西：① `backend_label` 与算子分发**共用同一个判断**（要谎报得把同一个判断改两遍）；② `pixi run test-backend-guard` 这道**编译期红测** —— `tests/fixtures/bad_backend.mojo` 必须编译失败且原因是 `unknown cpu backend`，否则拼错的后端常量会静默退化成标量、整网向量门照绿<br>⚠️ 后端是**方法**参数不是结构体参数，不是设计偏好：Mojo 1.0.0（ed45d567）在"参数化结构体 + 会抛错误的构造函数"上会直接把编译器进程搞崩，最小复现写在 `tests/fixtures/bad_backend.mojo` 的注释里<br>⚠️ 仍然**不宣称指令、不宣称性能** |
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
| 量化 dequant（q4_0） | `verified` | `evidence:tests/unit/test_q4_parity.mojo`（7/7，layer 0 四个真实投影矩阵，q_w/o_w 各 25088 块）**零容差逐位比较**<br>能做到零容差是因为可以：结果只取决于 4 位 nibble 与 fp16 缩放因子，两者都是精确的（半精度能表示的数单精度都能精确表示，故 fp16→fp32 是**精确转换**）—— 于是"差 1 ulp"不是舍入，是**布局读错**<br>块布局（32 值 / 18 字节、fp16 缩放小端、低半字节在前、`v=(nibble-8)*d`）是 **GGML q4_0 的外部事实**，故解量化属格式一致性检验<br>⚠️ **fp32→q4_0 的量化步是我方的格式转换，没有外部参照**，算法显式写在 `scripts/dump_q4_reference.py` 里；不许笼统写成"与 llama.cpp 一致"<br>两条负向对照常驻：高低半字节装反必须被零容差断言抓住；解出的值必须**确实**落在 `{-8d…7d}` 台阶上（否则"逐位相等"可能只是抄了原值） |
| 量化 dequant（q4_k / q8_0 / int8 / fp8） | `missing` | — |
| **量化步（fp32 → q4_0 块流）** | `verified` | `evidence:tests/unit/test_q4_parity.mojo`（10/10：与 `scripts/dump_q4_reference.py` 的离线实现**逐字节相同**，1.03 MB 块流零容差）<br>取整规则是这里唯一值得一提的地方：`round` 取**最近、并列取偶**（IEEE 默认），改成截断会让每个值平均偏小半个台阶 —— **负向对照专门钉这一点**：测试里另写一个截断版量化器，它必须与参考相差若干字节（实测 641074 字节），否则"逐字节比较"根本没在比取整。这类错误最阴：输出照样流利，只是分布整体偏了一点点，任何带容差的判据都放它过去<br>**块内缩放因子改按 MSE 选（2026-09-17）**：仍是`一个 fp16 scale + 32 个 nibble` 的块布局，只是不再取 `amax/7` —— 先用 `amax/7` 起个头量化一次，再对固定的 nibble 取重建误差的最小二乘解`d* = Σ(x·s)/Σ(s²)`（`s = nibble - 8`），迭代两轮<br>为什么值得：朴素写法为了让最大值够到台阶顶，把台阶钉在分布最稀疏的地方；MSE 解允许最大值被裁掉一点点，把台阶挪到分布密集处。实测 q/k/v/o 四个矩阵（另加整网 3 个权重）：**相对 L2 误差 10.75% → 10.34%**（cos 0.9942 → 0.9947）<br>**第三条负向对照**：测试里另写一个只改缩放因子选法（`amax/7`）、取整规则保持一致的旧规则版本，它必须与参考不同（实测相差 121846 字节）。这条是必需的，因为**参照物（fixture）是同一次导出的产物**—— 把实现退回 `amax/7`，参照物会跟着一起退，逐字节比较照样绿<br>顺带纠正上一行的一个说法：Mojo 侧**会**量化（加载期 `enable_q4` 一次），只是**推理期不量化** |
| **整网 q4_0 前向通路**（24 层全部投影走块流） | `verified` | `evidence:tests/unit/test_q4_greedy.mojo`（1/1：4 条 prompt × 128 步教师强制贪心，与 fp32 参考一致 **418/512 = 0.8164**）<br>门是 0.75 的下限，**它判的是"通路在工作"不是"质量达标"**：高低半字节装反会得到 0.0，前向悄悄退回 fp32 会得到 1.0，两头都被抓住（后者另由 `model.q4_enabled` 直接断言）<br>⚠️ **路线图里"一致率 ≥ 0.90"那条没过**，见下一行；本轮的 0.75 是"算的东西是对的"的下限，不是把 0.90 改小 |
| **整网 q4_0 教师强制贪心一致率 ≥ 0.90** | `missing` | **实测 0.8164**（输出投影留在 fp32 时；连它一起量化是 0.77），仍未达门<br>缩放因子这一路**已经走到底了**：按 MSE 选（见上一行）把一致率从 **0.800 抬到 0.8164**（+1.6 个百分点），而权重相对 L2 误差只从 10.75% 降到 10.34% —— 另一个数据点同样说明尺度选择已经到顶：在 q_w 上把 `amax/7` 整体乘一个系数扫一遍，最优是 **×0.90（10.23%）**，而逐块 MSE 解是 10.34%，两者只差 1%，说明"每块一个 scale"这个自由度本身已经榨干<br>**真正的约束是格式，不是选法**：每 32 个元素共用一个 fp16 缩放因子（cos 0.9947），落到 151936 维 argmax 上就是约两成位置翻盘<br>**不放宽门，也不假装达标**。下一步（本轮没做，别当成已有）：格式级改动 —— q4_K（超级块内再给一层 scale）、逐通道/逐行 scale，或带激活重要性矩阵（imatrix）的 scale 选择；这些都是**换块布局**，要新写 dequant 与配套的门 |
| RMSNorm | `verified` | `evidence:tests/unit/test_layer0_parity.mojo`（对参考输入/输出对，最大偏差 1e-5 量级；平方和与倒数平方根在 `Float64` 中累加，以免 oracle 自身的舍入成为被怀疑对象） |
| SwiGLU（含 `silu`） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo`（`silu` 与 `swiglu` 各有一条；参照物是 HF `act_fn` 的**真实输出**与 `down_proj` 的**真实输入**，脚本不自己乘一遍） |
| RoPE | `verified` | `evidence:tests/unit/test_layer0_parity.mojo`（cos/sin 表由参考导出，Mojo 只做查表与旋转；**不复现 `inv_freq`** —— 复现它本身就是一类事故源） |
| GQA 因果注意力（非分页） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo`（14 头 / 2 KV 头；`q_len` 可与 `kv_len` 不同，于是 prefill 与单步 decode 是**同一段代码**）<br>`evidence:tests/unit/test_model_parity.mojo`（`test_incremental_decode_matches_full_prefill`：逐 token 解码与整段 prefill 落点一致） |
| GQA 因果注意力（分页 / block table） | `missing` | 依赖 L3 KV 统一寻址 |
| matmul（fp32） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo`（q/k/v/o 四条投影，含带 bias 与不带 bias 两条路径；权重按 `[out, in]` 行主序，与 HF 存储一致，故加载时**没有转置**这一步可忘） |
| matmul（q4 dequant 融合） | `verified` | `evidence:tests/unit/test_q4_parity.mojo`（解出的权重 × 真实激活，fp64 累加，容差沿用 layer0 的同一判据 1e-5 相对）<br>"融合"是被检验的那件事本身：nibble 读到寄存器里直接乘缩放与激活累加，**不物化解量化后的权重** —— 若先解量化再走通用 matmul，被测的就只是通用 matmul 了<br>累加用 `Float64`，与 `scalar.mojo` 的 `_gemm` 同一理由：这一层当判据不当最快路径 |
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
| 采样器（top-k / top-p / min-p / 温度 / 重复惩罚 / logit bias） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo`（11/11，9 条逐阶段用例 × 32 次采样）<br>温度 / top-k / top-p / min-p / repetition penalty 用 **HF 4.41 的真实 warper 与 processor** 交叉核对**存活集合**（`scripts/dump_sampler_reference.py` 按其 `_get_logits_warper` 的自身顺序施加）<br>⚠️ **logit bias、frequency / presence penalty 在 HF 4.41 里没有对应 processor**，采用 OpenAI / vLLM 加法语义，属**语义自证**（由正负 bias、跨 top-k 边界、被 top-p 裁掉等边界用例钉住），**不是 HF 对齐**；两种重复惩罚语义字段名不共用<br>PRNG 自研 SplitMix64（`core/rng.mojo`），参考侧同一整数算法 → 采样出的 **token id 精确相等**（非容差相等） |
| 采样分布正确性（卡方 / TVD 检验） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo`（固定均匀序列 50000 次采样：TVD 0.00766 ≤ 0.05，卡方 23.05 ≤ 80）直击 `llm-mojo` 的分布错误<br>**负向对照常驻**：用温度减半的诱饵分布跑同一检验，TVD 0.4253 必须被判失败 —— 一个不会失败的拟合门等于没有门 |

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
| 预分词器（**无正则依赖**，手工实现 Qwen2 / GPT-2 规则） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo`（4560 条参考用例逐 id 全等）<br>参考的两个项目死在这里；这里用**手写分支匹配器**替代正则引擎：收缩形式、字母串、数字、标点、换行、尾随空白六类按最左优先取最长，贪心与回溯由代码显式表达 |
| NFC 归一化（分解 / 规范排序 / 重组，含 Hangul 与多元分解） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo`<br>**表是生成而非手写的**（`scripts/gen_unicode_tables.py`）；多元分解（如 U+1E5D）与组合类排序漏一条，整句就会落到不同的 merge 序列上 |
| BPE（**按 rank 合并**，同 rank 取最左） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo`<br>贪心从左到右会挑错对；这里每轮全量扫描取 rank 最小者 |
| added token 切分（**最长匹配**） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo`（语料含句中出现的 added token） |
| 词表 / 合并表加载（**离线 fixture，不启 Python**） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo`<br>fixture 由 `scripts/dump_reference.py` 生成，但测试进程只读 TSV —— 差分门可在任何机器上重跑 |
| id → 文本还原（含非法 UTF-8 替换） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo`（往返用例逐条对比**归一化后**的输入） |
| 4560 条差分用例（与 HF 逐 id 一致 + 往返） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo`（3/3 通过，0 处不一致）<br>**P1 门**：对不一致**零容忍**（0/4560），而不是容忍 0.1% —— 差分断言的是字节级等价 |
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
| 逐算子中间张量差分（定位用） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo`（13/13）<br>每个算子各自持有"参考实际看到的输入"与"参考实际产出的输出" → 失败时**只有该算子的测试红**，而不必在整网里二分 |
| top-k 集合比较 / 分布检验（卡方 / TVD） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo` 存活集合用 **FNV-1a 指纹做零容差比较**（只比数量会放过"对的个数、错的成员"）<br>并列取值的合成行**只比集合不比顺序**：`torch.sort` 在并列值上顺序未定义，逐元素比会 flaky；而真实 logits 几乎不并列 → 把 `>=` 写成 `>` 在真实数据上测不出来，在**量化后**一定会并列。该门已用变异测试验证过会红（改一个比较符 → 4 个测试失败） |
| 分布检验（卡方 / TVD） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo`（序列固定 → 断言确定性，**不会 flaky**；TVD / 卡方同时与参考值比对，不仅与阈值比对） |
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
