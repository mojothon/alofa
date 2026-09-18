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

**最后更新**：2026-09-17（第四次更新：**P2 调度器门打通** —— §7 执行编排层七行由 `missing` 升为 `verified`，另增两行写明本轮**刻意不做**的东西）

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
| GQA 因果注意力（分页 / block table） | `verified` | `evidence:tests/unit/test_paged_attention.mojo`（11/11）<br>`src/alofa/kernels/cpu/paged.mojo`：**只改行的地址**（`j * kv_cols` → `table.row_offset(j, kv_cols)`），算术与顺序和连续 oracle 逐字相同 → 与 `scalar.attention` **逐位相等**（11 个用例、0 个元素不同）。公式本身另由 `scripts/dump_paged_reference.py` 这份**独立 Python 实现**按 1e-5 相对容差钉住，唯一跨语言差异来源是 `exp` 的最后一位<br>⚠️ 本轮走标量后端；GPU 路径见 §4（本机 sm_52 阻塞） |
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
| `BlockPool`（物理池 + 空闲链 + refcount） | `verified` | `evidence:tests/unit/test_kv_pool.mojo`（12/12，7 个场景）<br>`src/alofa/runtime/kv/pool.mojo`：定容 `InlineArray` 空闲栈 + refcount；**摘要覆盖整个空间**（refcount 数组 + 空闲链顺序 + 全部请求 + 全部节点），把空闲链初始化倒过来，门**实测会红**（首处 `A=0,1` vs `A=111,110`） |
| `PageTable`（请求 → 块序列） | `verified` | `evidence:tests/unit/test_kv_pool.mojo`（`KvSpace` 的请求视图：块 id + 每块 token 长度 + 私有尾块标记；**连续追同一块合并成一项** —— refcount 按“视图条目”计，不按“有多少个节点贡献了它”计） |
| `RadixNode` 前缀树（token 粒度映射到 block 粒度） | `verified` | `evidence:tests/unit/test_kv_pool.mojo`（`src/alofa/runtime/kv/radix.mojo`：匹配是 **token 粒度**、持有是 **block 粒度**；匹配停在中间节点时按 `starts[p] + ntok[node]` 截断，不越读父节点的块数组）<br>✅ **已接入 attention**（2.2）：`runtime/kv/paging.mojo` 把请求页表拷成内核可读的 `PagedTable`（含**块内起始槽位**），`kernels/cpu/paged.mojo` 直接用 `(block, start, length)` 寻址；门断言"重放出的表"与夹具逐项一致，于是拼接这一环本身也被验过 |
| 三视图共享 refcount 的一致性 | `verified` | `evidence:tests/unit/test_kv_pool.mojo`（`check_invariants()` **从两个视图重算**每个块的持有数，再与 `pool.refcnt` 逐块比对；另加 `n_free + used == MAX_BLOCKS` 与空闲链成员精确匹配）<br>**负向对照常驻**：多一次 `retain`，不变量门必须报 >0（已实测报红） |
| 节点分裂（metadata-only，不新分配块） | `verified` | `evidence:tests/unit/test_kv_pool.mojo`（分裂前后 `pool.used` 不变、节点数 +1；父节点释放自己不再覆盖的尾块，子节点**沿用**原块 —— 前缀共享的收益来自元数据重排，不来自拷贝） |
| KV 寻址零堆分配（定容容器 + 源码门） | `verified` | `evidence:tests/unit/test_kv_pool.mojo`（`runtime/kv/` 下 4 个源文件全部扫构造点，不得出现 `List[` / `String(` / `Arena(` 等；常驻红测 `tests/fixtures/bad_kv_alloc.mojo` 必须被判违规）<br>⚠️ **边界同 §7 调度器那条**：拦得住“加一个会增长的容器”，拦不住 libc 小块分配，不等于进程 RSS 不动 |
| 频率感知淘汰（2Q + 复合评分） | `missing` | 创新点 2；**须有回放数据才可上线** |
| 采样器（top-k / top-p / min-p / 温度 / 重复惩罚 / logit bias） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo`（11/11，9 条逐阶段用例 × 32 次采样）<br>温度 / top-k / top-p / min-p / repetition penalty 用 **HF 4.41 的真实 warper 与 processor** 交叉核对**存活集合**（`scripts/dump_sampler_reference.py` 按其 `_get_logits_warper` 的自身顺序施加）<br>⚠️ **logit bias、frequency / presence penalty 在 HF 4.41 里没有对应 processor**，采用 OpenAI / vLLM 加法语义，属**语义自证**（由正负 bias、跨 top-k 边界、被 top-p 裁掉等边界用例钉住），**不是 HF 对齐**；两种重复惩罚语义字段名不共用<br>PRNG 自研 SplitMix64（`core/rng.mojo`），参考侧同一整数算法 → 采样出的 **token id 精确相等**（非容差相等） |
| 采样分布正确性（卡方 / TVD 检验） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo`（固定均匀序列 50000 次采样：TVD 0.00766 ≤ 0.05，卡方 23.05 ≤ 80）直击 `llm-mojo` 的分布错误<br>**负向对照常驻**：用温度减半的诱饵分布跑同一检验，TVD 0.4253 必须被判失败 —— 一个不会失败的拟合门等于没有门 |

## 7. 执行编排层（L4）

| 能力 | 状态 | 证据 |
|---|---|---|
| 纯函数调度器（单一 token 预算） | `verified` | `evidence:tests/unit/test_scheduler.mojo`（15/15 通过）<br>`pixi run mojo run -O0 -I src tests/unit/test_scheduler.mojo`<br>无 I/O、无时钟、无模型、无权重依赖；`step(input) -> Action` 是唯一入口，于是调度边界可以脱离模型被测（创新点 3） |
| 调度器自身零堆分配（定容容器 + 源码门） | `verified` | `evidence:tests/unit/test_scheduler.mojo`（类型层面：所有容器是编译期定长的 `InlineArray`；源码门扫描 `src/alofa/engine/scheduler.mojo` 不得出现 `List[` / `String(` / `Arena(` 等构造点，并有常驻红测 `tests/fixtures/bad_alloc.mojo` 必须被判违规）<br>⚠️ **这条证据的边界**：能拦住“给调度器加一个会增长的容器”，**拦不住** libc 里的小块分配，也**不等同**于进程级 RSS 不动 —— 账本就按这个口径写，不夸大成“进程零分配”。录 trace（`engine/trace.mojo`）**会**分配 String，所以录制是 `step` 之外的可选动作 |
| chunked prefill | `verified` | `evidence:tests/unit/test_scheduler.mojo`（300 token 的 prompt 跨 19 拍切片：首尾相接、不重叠、每片不超过 `max_chunk`；同一拍不得给同一请求两个 chunk） |
| 抢占（重计算） + 抢占计数指标 | `verified` | `evidence:tests/unit/test_scheduler.mojo`（并发抢占风暴 / KV 水位临界两个场景；被抢占者 KV 全作废并回到等待队列，累计抢占次数作为 `Action` 字段逐拍比对 —— 它是容量告警指标，不是调试字段）<br>⚠️ 只抢占 **RUNNING** 请求：抢占“半截 prefill”的请求会让 prefill 永远无法完成，那是伪装成策略的抖动<br>⚠️ 池子小到连一次 decode 增长都装不下时报 `capacity` 具名错误，**不静默丢 token**（有专门断言） |
| 调度 trace 录制 | `verified` | `evidence:src/alofa/engine/trace.mojo` + `tests/fixtures/scheduler/*.trace`（格式：整数 + 定长字段，**不含浮点**；水位用千分数而非比例 → 逐字节门不会退化成容差门） |
| 调度 trace 重放 + 极端场景断言 | `verified` | `evidence:tests/unit/test_scheduler.mojo`（6 个场景与 `scripts/dump_scheduler_reference.py` 这份**独立 Python 实现**逐字节相同；三条常驻负向对照：改坏的 trace、换一种抢占顺序、给调度器加堆容器，三者都必须被判红）<br>6 个场景：超长 prompt、并发抢占风暴、预算耗尽、0 预算、取消竞态、KV 水位临界 |
| 延迟护栏（最大等待拍数） | `verified` | `evidence:tests/unit/test_scheduler.mojo`（构造“队首长 prompt 每拍吃光预算”的最小复现：护栏生效时第 4 拍必须给短请求；把 `max_wait_ticks` 调到 99 该断言**实测会失败**）<br>护栏只在**等待者之间**插队，不越过 decode：它防的是“前面有个超长 prompt”，不是“预算被 decode 占满” —— 后者说明并发已饱和，插队只会把等待转嫁给已经占着 KV 的人 |
| 批张量池（稳态零堆分配） | `verified` | `evidence:tests/unit/test_batch_pool.mojo`（12/12）<br>`src/alofa/engine/batch.mojo`：借用拿到的是**句柄**而不是指针（释放后同一个槽位会给别人）；池子不拥有内存，构造时收一个指针加容量；全部簿记是编译期定长的 `InlineArray`，**没有空闲链表** —— 每次放置都从“活着的借用”重导出可用空隙，于是不存在两套会互相漂移的账<br>7 个场景与 `scripts/dump_batch_reference.py` 这份**独立 Python 实现**逐字节相同：分配是“决定”不是数值，两种放置之间不存在“差一点点”<br>两条不依赖参照物的性质：**活着的借用不共享任何一个字节**（每个区间写自己的标记再逐字读回 —— 把放置往旁边挪一格，这条实测会红），**峰值不漂移**（20 个相同执行步之后 high-water 与第一步相同；否则稳态不稳，之后测的吞吐就是关于另一个池子的数字）<br>**拒绝必须是具名错误**：池子填满后隔一个释放，剩 131072 字节可用而最大空洞只有 16384 —— 借两块必须报 `capacity` 而不是绕回去或跨两个空洞凑；第 25 个活的借用同样是 `capacity`；重复释放是 `double_free`<br>3 条常驻负向对照：`alt.trace`（同一组操作改用 best-fit 放置，且**要求它至少挪动一处偏移**，否则那 6 个逐字节比对只是在验文件格式）、`bad.trace`（一处偏移挪一格）、给池子加堆容器的 `bad_batch_alloc.mojo`<br>⚠️ 证据的边界：拦得住“给池子加一个会增长的容器”，**拦不住** libc 的小块分配，也不等于进程级 RSS 不动；且这一层本身**不比对数值**（借到的字节里算得对不对由下面两行负责） |
| 批组装（每请求一段连续行） | `verified` | `evidence:tests/unit/test_batch_pool.mojo`（12/12）<br>`BatchSlots` 给每个请求一段**连续**、且**不与别人重叠**的行：两个请求共用一行时，某一层会从别人的 token 上读出自己的激活，产出的每一个数都看起来合理 —— 与分页内核读错块是同一种失败<br>断言从外面重算：逐行统计所有者，**重复覆盖与未被覆盖都必须为 0**；同一请求被加两次是具名错误（一次 add 会让它拿到两段行，而某一层只会读其中一段）<br>⚠️ 只在**行归属**这一层成立；“注意力按请求分块”由 executor 接走（下一行） |
| 批执行器（注意力按请求分块） | `verified` | `evidence:tests/unit/test_batch_executor.mojo`（10/10）<br>`src/alofa/engine/executor.mojo` 把 2.5 的两层接进前向：每个请求拿到一段**连续行**，每一行被交给注意力时都带上**自己的** K/V 基址、`upto`（能看到的最末一个 key）与 `pos`（rope 用）—— 于是“不同请求的行互相 attend”不是靠一张可能被丢掉的 mask 挡住的，而是**地址上不存在**：一行从来拿不到别人的地址<br>6 个场景与 `scripts/dump_batch_executor_reference.py` 这份**独立 Python 实现**逐字节相同（ADD/FEED/DROP/PLAN/ROW/FIN/NEXT 全序列 + 每步摘要 `d=`），摘要覆盖全部请求槽与全部行，不变量**每一步从外面重算**（缺陷计数不为 0 即红，而不是只在结尾查一次）<br>3 条常驻负向对照：`alt.trace`（换一种行分配策略，且**要求它真的不一样**，否则 6 个逐字节比对只是在验文件格式）、`bad.trace`（某一行的可见窗口挪一格）、给忙碌循环加堆容器的 `bad_executor_alloc.mojo`<br>每一步向 2.5 的池子借 15 块、步末全部归还：实测 `used==0`、`n_live==0`、峰值不漂移<br>⚠️ 边界：这一门**不比对数值**（注意力算得对不对由 §7 的分页门与下一行的批一致性门负责），它验的是行归属、每行的窗口与簿记 |
| 批一致性（批大小 1/2/4/8 与单请求逐 token 相同） | `verified` | `evidence:tests/unit/test_batch_forward.mojo`（4/4；重门，需要 2GB 权重，`pixi run test-batch-forward`，故意不进 `pixi run test`）<br>同一批 prompt 走两遍：**一批 N 条** 与 **一条一条跑**（`QwenForward.prefill`/`step`，也就是 `test_model_parity` 拿去和 Hugging Face 对过的那条路径），greedy 解码、逐 token **相等** —— greedy 让“第 3 个 token 不同”就是一个不同，而不是差一点点<br>N = 1 / 2 / 4 / 8：8 条请求的 prompt 合计 88 行 > 行块 64，所以这一门**真的把一次 prefill 切成两拍** —— 短的一拍也必须是对的一拍<br>2 条常驻负向对照：**不同 prompt 必须解出不同续写**（否则“批次与单请求一致”对任何实现都成立，包括不看输入的实现）；**交换两条请求的 prompt 必须被察觉**（交换后既**不等于**该槽位的基线、又**等于**它实际拿到的那条 prompt 的基线）—— 这才让逐字节比对成为“行归属”的证据<br>⚠️ 参照物是**本树的串行前向**，与批路径共享 kernel：这是刻意的，被比较的是**编排**（行归属、每行的 pos、每行的窗口、KV 区域），而串行路径只有一条请求、不可能在这些上出错；参照物本身对 Hugging Face 的一致性由 §5 的模型门负责 |
| 引擎循环（调度器 ↔ 批执行器接线） | `verified` | `evidence:tests/unit/test_engine_core.mojo`（10/10，已进 `pixi run test`）<br>`src/alofa/engine/core.mojo` 把 2.0 的调度器与 2.5b 的批执行器接成一个忙碌循环：调度器出**决定**（谁 prefill、给 `[start, end)` 这一段、谁 decode、谁被抢占），执行器出**行**；被验的只有两者之间的**翻译** —— 切片喂给谁、prefill 结束那一拍白送的第一个 token 与之后 decode 出来的 token 怎么拼成同一份 transcript、抢占后重算要作废什么<br>argmax **由测试注入**（每槽一个整数，不跑模型）：贪心 argmax 是一行代码，这一门要验的是**时序**；也正因为期望值写成 `expected_token(请求, 第几个 token)` 而与「第几拍产生的」无关，同一份期望才能同时管住「抢占后被推回 prompt、重新生成一遍」的请求<br>3 条常驻负向对照：① 抢占场景**断言 `preempt_total > 0`** —— 声称「抢占安全」却从头到尾没抢占过的门，是穿着戏服的 happy path；② **把请求从执行器手里抽走再要一拍，必须报 `invalid_argument`**（静默服务一个空请求更省事，也更会藏 bug）；③ 给循环加堆容器的 `bad_executor_alloc.mojo`<br>每一拍都从两边重算「谁还活着」：引擎说谁 resident、执行器说它握着谁，二者不一致的那一拍**照样产出一串看起来是 token 的数**<br>⚠️ 边界：这一门**不比对数值**（续写得对不对由上一行的批一致性重门负责），也不意味着 KV 物理块池已接入 —— 执行器用的是自己构造时写死的 KV 区域（见下一行） |
| KV 物理块池（含 `freed_blocks` 这类外部释放） | `verified` | `evidence:tests/unit/test_kv_room.mojo`（10/10，已进 `pixi run test`）<br>`src/alofa/engine/kv_room.mojo` 是唯一做这层翻译的地方：调度器**数**块但不拥有块，`runtime/kv` 拥有块但只会说 radix 树操作。三件只活在这一层的事：① prefill 到达是**切片**，一次入场是多次 `append_tokens`，而「prompt 有多长」是另一件事实（它决定半个 prompt 不许进缓存）；② 完成的请求**换主人**（`commit` 发布到树），块变成「没有主人的占用」= 前缀缓存；③ 缓存的块只从**一扇门**回来 —— 驱逐是引擎的决定（只有引擎知道压力），还回多少**实测**（缓存块数前后之差）而不是记账<br>引擎侧接线（`src/alofa/engine/core.mojo`）：prefill 首片 `admit`、续片 `grow_to`、`settle` 后按「prompt + 已生成」对齐长度（给目标值不给增量）、完成 `publish` 再 `drop`、抢占与取消直接 `drop`；归还经 `SchedInput.freed_blocks` 回报调度器，两条规则都在调度器决定之前执行：**缓存让位给活着的请求**（缓存 ≤ 容量 − 持有）与**水位**（缓存顶高水位时先回收，免得调度器为还不了的块去抢占）<br>5 个场景与 `scripts/dump_kv_room_reference.py` 这份**独立 Python 实现**逐字节相同（`u/f/c/d`），且每拍从视图重算不变量、断言 `used + n_free == MAX_BLOCKS`；3 条常驻负向对照：`bad_room.trace`（改坏一拍）、`alt_room.trace`（换 prompt，必须判红 —— 否则只验了格式）、`bad_room_alloc.mojo`（给房间加堆容器）<br>不依赖参照物的性质：相同 prompt 的第二条 `last_matched == 32` 且 `last_fresh == 0`、池子用量不变；发布后 `used` 不降、回收后块真的回池且上报数等于实测；**未发布的 drop 必须真的回池**；`reclaim` 对活着的请求必须还回 0<br>⚠️ 边界：`runtime/kv` 并发上限 `MAX_REQUESTS`（8）、单序列上限 `MAX_SEQ_TOKENS`（64），第 9 条报 `capacity`、超长报 `out_of_range` —— 限制被**断言**而非绕过（悄悄少给几块会在几拍后变成「少一个答案」）。调度器的占用仍是**算术**的、房间的才是**物理**的，二者不要求相等（前缀共享让物理更少、节点粒度让物理可能更多），物理池满时房间具名拒绝。⚠️ 执行器的 KV 已从这张块表取地址（见后两行）；房间给了块之后，前向读到的位置由**表**决定，不是由算术决定 |
| 分页注意力接入引擎循环（前向从块表取地址） | `verified` | `evidence:tests/unit/test_paged_scatter.mojo`（5/5，已进 `pixi run test`）+ `tests/unit/test_batch_forward.mojo`（重门，真实 0.5B fp32 权重）<br>块池布局：一个块**持有每一层各一个槽**，所以请求的表在每层都叫同一批块号 —— `engine/executor.mojo` 把层号折进块 id（`layer * MAX_BLOCKS + block`），于是整个块池只有**一个**视图、在构造时建好，忙碌循环里不再构造形状（每步每层建一个 `List` 正是这一层的源码门要挡住的事）<br>`kernels/cpu/paged.mojo` 新增 `paged_scatter`：写**经过**表，与读经过同一张表。写按算术放（`j // block_size`）会把 token 放进「它若不共享前缀本会占用的块」，之后每一步都是从别人的历史里算出来的数；写越界报 `out_of_range` 而不是截断（截断是悄悄变短的上下文）<br>房间与执行器的交接只有一处：房间 `page_table()` 拷出表 → 引擎 `sync_page_table()` 在 `admit` / `grow_to` **之后立刻**交过去；表按 `hist + n` **裁剪**后再用 —— 房间可以为还没到的 token 预留整块（切片 prefill），而注意力不许读没人写过的位<br>批一致性重门跑在**乱序块号**下：块刻意不按连续区域的顺序排，所以「批与串行逐 token 相同」这句话是关于**页表**的 —— 前向若按算术取地址，数就不同<br>⚠️ 边界：`rows_view` 构造视图（形状是 `List`）在执行器 `forward` 里仍然存在 —— 块池的**账**是零分配的，视图构造不是 |
| 共享前缀只算未命中的那一段（前缀缓存省的是算术） | `verified` | `evidence:tests/unit/test_batch_forward.mojo`（6/6，真实 0.5B fp32 权重，重门）+ `tests/unit/test_batch_executor.mojo`（12/12，已进 `pixi run test`）<br>引擎层**端到端**已验（`tests/unit/test_engine_core.mojo` 11/11）：同一个 prompt 提交两次，第二次的第一拍**只跑一行**且 `history_of == 6` —— 此前这条链路只是「编译通过 + 单元绿」，房间 → 引擎 → 执行器这一段没人跑过；常驻对照是同一引擎里的冷 prompt：7 行、history 0。| KV 池高占用下的正确性 | `verified` | `evidence:tests/unit/test_engine_core.mojo`（13/13）：四条请求同时在池（峰值 = 四条块数之和），全部跑完且**逐 token 等于逐条跑的基线**；每拍重算 `used + n_free == MAX_BLOCKS`、房间 `invariants()==0`、两本账 `defects()==0`<br>⚠️ **不声称 95%**：2026-09-18 撤回前一天记的「峰值 111/112」——那个数字是被下面那行的记账 bug 造出来的（房间白发整条 prompt 的块），修好后同一场景只到 99/112，边界见下面两行 |
| 分块 prefill 下的两本账 | `verified` | `evidence:tests/unit/test_engine_core.mojo`：**已修**：引擎原来在 `settle` 里把 KV 序列长度设成「整条 prompt + 已生成」，无视 prefill 只喂到第 16 个 token —— 房间因此白发整条 prompt 的块（实测第一拍：房间 15 块、调度器账 4 块；七条跑下来差 10 块），水位 950‰ **全程不触发**，池子只靠房间抛 `capacity` 兜住。现在按「已喂到的位置」grow，分块下两本账差 ≤ 1（`test_the_scheduler_and_the_room_count_the_same_blocks`；旧行为下该门差 10 块、红）|
| 物理池 >95% 且能跑完的场景 | `missing` | 在当前实现下**不可达**：`MAX_ROWS=64` 只能逐条 prefill，先完成的先释放，分块下峰值 98/112（87%）；要顶满就得让请求长驻留，而驻留总量一旦高过水位，抢占就在两条请求之间来回抢、谁也完不成（七条各生成 8 个 token：512 拍仍不空闲；容量预算 48 / 阈值 45 下四条同样活锁）。解锁条件：让抢占真正缓解而不是循环——受害者重算时应优先拿回块，或水位只在「有等待者需要块」时触发 |
| 缓存让位与缓存账的时序 | `verified` | `evidence:tests/unit/test_engine_core.mojo`（14/14）：五条请求分一个装不下的预算（48），缓存必须让位，否则排队的请求永远拿不到块。修了两处：① **缓存占死预算**——yield 只按「已在跑的」算，缓存把预算吃满，四条请求在剩下的块里互相抢占，有一条一个 token 都没生成；现在按 `blocks_used + blocks_wanted()` 算，缓存只留别人用不到的。② **引擎交回调度器尚未记账的块**——上一拍发布的序列要等本拍 `step` 才进 `cached_blocks`，reclaim 却发生在 `step` 之前，于是下一拍的 `freed_blocks` 大于调度器认为的缓存，抛 `ERR_INVALID_ARGUMENT`；现在只交回调度器已记账的部分。负向对照：把 ① 改回旧算法，该门红在「512 拍从未空闲」 |
| 共享前缀下的缓存账 | `missing` | 调度器 `release(to_cache=True)` 按「每条已完成序列自己的块数」累加缓存，而房间的前缀树去重后只占一份 → 两本账不同源（调度器高估）。**首 token 归属那条已修**（见下），剩下的只有去重这一条。**回滚过一次**「让引擎把差额延后一拍用 `freed_blocks` 报出」：两本账当时对齐了，但随后撞 `id list overflow`，且回收与修正同拍叠加会超账。解锁：缓存占用数只能由房间报告，调度器不得自行推算——`SchedInput` 需要一个独立的「缓存增量」通道 |**第二次尝试也已回滚**（2026-09-18）：给 `SchedInput` 加绝对值通道 `cached_now`，房间每拍报真值覆盖调度器的和。失败原因不是实现细节：调度器的 `release` 比房间的 `publish` **晚一拍**（完成消息延后送达），于是「对齐到上一拍真值 + 本拍 release 整条」仍在叠加——实测对齐到 36 之后又加上两条的 24，得 60，而房间是 48；补 `cache_pending` 让引擎多跑一拍也没能把 60 降下来。真正的解锁是「调度器不再自己维护缓存账」，而这跟「调度器是纯整数函数、重放门不依赖房间」直接冲突（参考实现没有房间，报不出真值，那时调度器又必须能自己算）→ 属于架构取舍。
| 过载 + 长 prompt 的抢占活锁 | `verified` | `evidence:tests/unit/test_engine_core.mojo`（两条门：抢占确实发生，且受害者重算后仍跑完全程）<br>`evidence:tests/unit/test_scheduler.mojo`（trace 逐字节：抢占发生后拍末 blocks_used 回到阈值内、无一拍越过硬容量）<br>原探针（未固化为门）：容量 10 块、4 条请求各 20 prompt（16 行一拍 → 跨两片）+ 3 生成、块 4 字节 → 300 拍、**190 次抢占、0 个 token 产出**（⚠️ 该数字取自 `watermark_permille=8`，即水位 **0 块**的病态配置，不是默认——`SchedConfig` 第 5 个参数是水位千分比，早先误当成了预算）。**默认水位 950 重测仍不收敛**：容量 10、块 4、4 条 20 prompt + 3 → 400 拍、`outs=3 3 1 1`、抢占 258 → 活锁在默认水位下**依然成立**，只是程度较轻（两条能跑完）；前提是池子装不下在飞的序列（4×6=24 > 10），而准入与晋升都不做容量规划，于是「抢占归零 → 重喂」变成循环。**根因**（逐拍探针订正）：第 5 步「晋升」无条件把 `done >= prompt_len` 的请求全转成 `ST_RUNNING`，**不看池子能否容纳它们的 decode 增长** → 同拍多条一起晋升 → 第 6 步 decode 时 `ensure_room` 装不下 → 抢占（`scheduler.mojo:570-571` 把 `done`/`generated` 归零）→ 打回 `ST_WAITING` 从头再喂 → 循环。逐拍证据：60 拍内所有请求 `state` **始终为 1（WAITING）**，从未进入 RUNNING；某条 `done` 刚到 20，下一拍即 `done=0` 且 `preempt_count+1`。`max_wait_ticks` 越小 → 强制 prefill 越密集 → 同拍晋升越多 → 抢占越多，故默认 8 比 900 更糟。**不是记账问题**（两本账一致）。⚠️ 早先写的「喂一半被抢占」是**错的**：抢占只针对 `ST_RUNNING`（`:564`），部分喂的是 `ST_WAITING`，不会被抢占。修法方向：晋升加容量门槛（装得下整条序列才晋升）——属调度器预算类改动，但会改单拍决策 → scheduler 的 14 条 trace 需重导（Python 参考同步改） |
| 已发布序列的块数 | `verified` | `evidence:tests/unit/test_engine_core.mojo`（16/16）：调度器按 `done + generated` 算一条已发布序列占多少块，而 `generated` 数的是 decode 拍——续写的**第一个 token 由「把 prompt 喂完的那一步」产出，不算一拍**，于是每条少记一个 token；跨块时少一整块（实测四条：44 对 48）。改按 `prompt_len + max_new` 记，并顺带补齐 `blocks_used`（它留着的是按拍算的旧数，否则池账比缓存账少同样多）。新门 `test_a_published_sequence_is_counted_whole`：37+8=45 token 是 4 字节块的 12 块，44 是 11——**块粒度 8 时两者都是 6 块，同一个 bug 会溜过去**（现有那两个门正是块粒度 8，当时全绿） |
| 引擎空闲判定与块释放 | `verified` | `evidence:tests/unit/test_engine_core.mojo`（15/15）：批里最后一条请求完成后，消息要到下一拍才到调度器，`has_work` 却只看引擎自己的 state → 它宣布空闲，那条请求永不 `release`，实测 11 块永久占用（每批泄漏一次）。对称地，房间已回收的缓存块若没被下一拍带走，调度器会一直为它们记账（实测 32 块）。修：`has_work` 也认 `n_report` 与 `room.freed_pending`；新增 `room.take_freed_upto()`——回收按节点整块释放，可能多于请求量，多出的留到下一拍再报。空闲时两本账归零。负向对照：去掉这两个条件 → 该门红在「仍有块被持有」 |
房间 `admit` 时就知道重合多少（`last_matched`），引擎把这个数交给执行器 `add` 的 `matched`：命中的 token **入队但不建行** —— 历史从 `matched` 起算，队列里只剩没算过的那些。省下来的是**行**，行就是算术<br>**最后一行永远要算**：它的 logits 是第一个生成的 token。一个被完整命中的 prompt（`matched == n`）跑一行，不是零行 —— 零行就没有 logits，请求无从开口；`matched > n` 具名拒绝（`out_of_range`）<br>证据是端到端的：同一个 prompt 跑两遍，第二遍沿用第一遍的**同一批块**（`drop` 只忘地址、不清字节，这正是前缀缓存的定义）、`matched = n - 1`，跑一行，生成的 token 与串行**逐 token 相同**。配套常驻负向对照：谎报命中（`matched = n - 1` 但指向没人写过的块）必须产出**不同**的 token —— 否则上面那条可以因为「压根没读缓存位置」而白过<br>⚠️ 边界：命中的字节必须与本地计算**逐位相同**才成立（同机、同权重、同路径、同位置 —— 换 backend / 跨机未验）；执行器**不校验**块里真的是那段前缀，它信任房间 —— 谎报由上面的对照拦，不由类型拦 |
| paged attention（block table 索引） | `verified` | `evidence:tests/unit/test_paged_attention.mojo`（11/11）<br>两类断言用两把尺子：**寻址用逐位相等**（`paged_gather` 与参照导出的连续行逐位一致；分页 kernel 与连续 oracle 逐位一致 —— 没有重排就没有"差一点点"的余地，差一个 ulp 就是地址算错），**公式用 1e-5 容差**（期望值来自 `scripts/dump_paged_reference.py`，与 Mojo 不共享任何代码）<br>**5 条常驻负向对照**：改坏一个元素的 `expected_bad.tsv`、用错 GQA 映射（`h % n_kv_heads`）的 `expected_hmap.tsv`、给 kernel 加堆容器的 `bad_paged_alloc.mojo`、把每段 run 的尾槽灌成垃圾值后输出必须逐位不变、共享同一块的两个请求必须读到同一段字节<br>⚠️ 夹具里的表是 `KvSpace` 对 2.1 真实操作序列重放出来的（8 例），另 3 例是 `syn_*` 块内偏移用例：当前树只从根共享、请求都从槽位 0 分配，**块内起始的 run 走 `KvSpace` 造不出来、走 `PagedTable` 造得出来**，内核就必须对它负责 |

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