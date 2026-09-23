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
| **`flare` / `json` 可导入**（含 `flare.runtime.Reactor`、`flare.http.HttpServer`） | `verified` | `evidence:tests/capability/test_deps.mojo?count=7`<br>`pixi run mojo run tests/capability/test_deps.mojo`<br>注：import 在模块顶层，**编译失败即断言失败** |
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
| **整网 q4_0 前向通路**（24 层全部投影走块流） | `verified` | `evidence:tests/unit/test_q4_greedy.mojo`（1/1：4 条 prompt × 128 步教师强制贪心，与 fp32 参考一致 **445/512 = 0.8691**（2026-09-22 起；**之前是 418/512 = 0.8164，那时量化分支里有一个 4 倍错位的行偏移 bug**，见今天变更日志））<br>门是 0.75 的下限，**它判的是"通路在工作"不是"质量达标"**：高低半字节装反会得到 0.0，前向悄悄退回 fp32 会得到 1.0，两头都被抓住（后者另由 `model.q4_enabled` 直接断言）<br>⚠️ **路线图里"一致率 ≥ 0.90"那条没过**，见下一行；本轮的 0.75 是"算的东西是对的"的下限，不是把 0.90 改小；⚠️ 0.75 那道门**看不见**上面那个 bug（0.8164 也在它的绿区里），所以同一份测试里已补了一道 **0.85 的回归下限**（`Q4_REGRESSION_FLOOR`） |
| **整网 q4_0 教师强制贪心一致率 ≥ 0.90** | `missing` | **实测 0.8691**（输出投影留在 fp32 时），仍未达门<br>⚠️ 2026-09-22 更正：此前记的 **0.8164** 里有一部分**不是量化误差** —— 量化分支的行偏移多乘了一个 4，导致批 > 1 时第 1 行之后的 token 一行都没被写（并越界写）。去掉那个 `* 4` 后同一棵树 A/B：**0.8164 → 0.8691**（418/512 → 445/512）<br>缩放因子这一路**已经走到底了**：按 MSE 选（见上一行）把一致率从 **0.800 抬到 0.8164**（+1.6 个百分点），而权重相对 L2 误差只从 10.75% 降到 10.34% —— 另一个数据点同样说明尺度选择已经到顶：在 q_w 上把 `amax/7` 整体乘一个系数扫一遍，最优是 **×0.90（10.23%）**，而逐块 MSE 解是 10.34%，两者只差 1%，说明"每块一个 scale"这个自由度本身已经榨干<br>**真正的约束是格式，不是选法**：每 32 个元素共用一个 fp16 缩放因子（cos 0.9947），落到 151936 维 argmax 上就是约两成位置翻盘<br>**不放宽门，也不假装达标**。下一步（本轮没做，别当成已有）：格式级改动 —— q4_K（超级块内再给一层 scale）、逐通道/逐行 scale，或带激活重要性矩阵（imatrix）的 scale 选择；这些都是**换块布局**，要新写 dequant 与配套的门 |
| RMSNorm | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（对参考输入/输出对，最大偏差 1e-5 量级；平方和与倒数平方根在 `Float64` 中累加，以免 oracle 自身的舍入成为被怀疑对象） |
| SwiGLU（含 `silu`） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（`silu` 与 `swiglu` 各有一条；参照物是 HF `act_fn` 的**真实输出**与 `down_proj` 的**真实输入**，脚本不自己乘一遍） |
| RoPE | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（cos/sin 表由参考导出，Mojo 只做查表与旋转；**不复现 `inv_freq`** —— 复现它本身就是一类事故源） |
| GQA 因果注意力（非分页） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（14 头 / 2 KV 头；`q_len` 可与 `kv_len` 不同，于是 prefill 与单步 decode 是**同一段代码**）<br>`evidence:tests/unit/test_model_parity.mojo`（`test_incremental_decode_matches_full_prefill`：逐 token 解码与整段 prefill 落点一致） |
| GQA 因果注意力（分页 / block table） | `verified` | `evidence:tests/unit/test_paged_attention.mojo?count=11`<br>`src/alofa/kernels/cpu/paged.mojo`：**只改行的地址**（`j * kv_cols` → `table.row_offset(j, kv_cols)`），算术与顺序和连续 oracle 逐字相同 → 与 `scalar.attention` **逐位相等**（11 个用例、0 个元素不同）。公式本身另由 `scripts/dump_paged_reference.py` 这份**独立 Python 实现**按 1e-5 相对容差钉住，唯一跨语言差异来源是 `exp` 的最后一位<br>⚠️ 本轮走标量后端；GPU 路径见 §4（本机 sm_52 阻塞） |
| matmul（fp32） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`（q/k/v/o 四条投影，含带 bias 与不带 bias 两条路径；权重按 `[out, in]` 行主序，与 HF 存储一致，故加载时**没有转置**这一步可忘） |
| matmul（q4 dequant 融合） | `verified` | `evidence:tests/unit/test_q4_parity.mojo?count=10`（解出的权重 × 真实激活，fp64 累加，容差沿用 layer0 的同一判据 1e-5 相对）<br>"融合"是被检验的那件事本身：nibble 读到寄存器里直接乘缩放与激活累加，**不物化解量化后的权重** —— 若先解量化再走通用 matmul，被测的就只是通用 matmul 了<br>累加用 `Float64`，与 `scalar.mojo` 的 `_gemm` 同一理由：这一层当判据不当最快路径<br>`evidence:tests/unit/test_q4_matmul_vec.mojo?count=8`（2026-09-20 新增，专用夹具 `tests/fixtures/.../q4vec/`，48 KB）：同一批期望值上的**结构性**差分门 —— 块/行取 1/3/5/7/11/17/28/152、行数取 1/2/3/5/7/13，**专门用来把 SIMD 尾巴露出来**<br>⚠️ 两条 evidence 互补，缺一不可：原 fixture 四个用例的 `cols` 全是 896（**28 块/行**，能被 1/2/4/7/14/28 整除），任何「按 N 块展开、余数走标量尾巴」的写法都会**全套绕过尾巴** —— 与 2026-09-17 在 avx2 上踩的坑同一类：fixture 的形状恰好避开了唯一没被走过的分支<br>三条常驻负向对照（省掉 `-8` / 高低半字节装反 / 丢掉最后一块）的「不可见用例」用**显式名单**约束而非降阈值：不可见的原因多是数学上不可观测（±交替正好相消、整行同值、近零块），名单之外多出一处不可见就红<br>⚠️ **已知弱点（留着，将来改判据时回头看）**：`1e-5 × max(1, |ref|)` 在 `|ref| ≪ 1` 时退化成**绝对** 1e-5 —— `const_row` 一例参考值约 1.8e-5，于是「少读一整块」（贡献约 5e-6）躲过了判据；故单列一条要求：两个真实用例（`real13` / `down_like`）必须判红<br><br>**向量实现（`avx2.mojo`，2026-09-20）**：同一 fixtures、同一期望值、同一判据，换一个核再跑一遍 `check_kernel_against_reference`；九条用例与标量版**逐位一致**（含两条真实数据用例）。切法是**按 `j % 8` 分通道**（块内第 j 个量化值固定配到第 `j%8` 条通道）而不是按连续段切，所以尾巴由块布局本身给定（每块的数据字节刚好两个 8 字节），`cols` 必须是 32 的倍数这一约束与标量版同源。`bench` 侧新增 `q4_0/avx2` 一栏（`scripts/bench_decode_roofline.mojo`）；本机 `down_proj` 4864×896：标量 3.59–3.71 周期/元素 → 向量 **1.19–1.25**（约 3×）。<br>⚠️ **累加仍在 f64**（与标量同一理由），但它作为一个**被测对象**存在：`_matmul_q4_f32acc` 走 f32 累加，实测九条用例最大相对偏差 **8.38e-08**（`real13`；门限 1e-5，低 120 倍），速度再快 1.4×。**没有**把它设为默认 —— 换的是判据层的算术，要先回答「相消最严重的那处还剩多少余量」，今天没人量过。 |
| **CUDA kernel 与标量后端逐值差分**（路线图 1.7：RMSNorm / RoPE / matmul 三个 kernel） | `hardware-blocked` | **阻塞原因**：A100 验证机 `10.107.6.60:3389` 于 2026-09-17 实测连接超时（`scripts/a100.sh` 不可用）；本机 GTX TITAN X 为 Maxwell sm_52，现代 CUDA 栈与 MAX 均不支持<br>**解锁条件**：① A100 可达 ② 环境变量 `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas`（绕过 MAX 对驱动 ≥580 的要求）③ 门为 fp32 容差 1e-5 的**逐值差分，只验正确性不报性能**<br>⚠️ 本轮**未写任何 CUDA 代码**：写完不验的 kernel 比没有更危险，它会被后来者当成可用<br>区分：§1.2 的"端到端 GPU kernel 数值正确"（向量加）已作为远端观测条目单独成立，本机可复现的**算子级** CUDA 差分仍是本行状态 |
| MAX 内核复用（`linalg` / `layout` / `quantization`） | `missing` | 需隔离层；注意 MAX 导入路径有迁移风险 |

## 5. 模型层（L2）

| 能力 | 状态 | 证据 |
|---|---|---|
| HF `config.json` 解析 | `partial` | `src/alofa/model/config.mojo`：严格读取 Qwen 所需标量字段；`QwenConfig` 已同时接受项目 `config.tsv` 与 HF 字段名的 `config.json`。证据：`tests/unit/test_model_formats.mojo?count=8`。2026-09-21 补：`head_dim` 在真实 Qwen2.5 的 `config.json` 里**不存在**（HF 运行时自己从 `hidden / n_heads` 推导），故改为「有就读、没有就派生」，除不尽由 `validate` 兜住；新增 `json_has()` 专门问「有没有这个字段」。当前边界：只支持无转义标量，未知嵌套字段不解析；暂不宣称通用 JSON。 |
| **权重常驻内存门**（加载 + 生成 32 token 后，峰值 RSS ≤ 1.15 × 权重字节） | `verified` | `evidence:tests/unit/test_memory_gate.mojo`（1/1 通过；实测 **1.006×**，预算 1.15×；权重 1.98 GB，峰值 1.99 GB）<br>**这个门能拦住的**：权重被多留了一份（拷贝 / 转置后留着原副本 / 整网常驻解量化成 fp32）—— 那些会让常驻量奔向 2×，门在 1.15× 就红<br>**拦不住的**（不许拿它去宣称）：权重是零拷贝的。mmap 触碰过的文件页与 `read()` 到堆上**都算进 RSS**，两者的常驻量都是约 1× —— **RSS 这个量本身区分不了它们**；零拷贝需要别的证据（缺页次数 / 映射区间），本轮没做<br>分母按**字节偏移去重**：绑定权重在索引里是两条**同偏移**的别名行，不去重会把 embedding 那 544 MB 数两遍，分母凭空大 27%，1.15× 的门实际成了 1.47×<br>**负向对照常驻**：额外 `mlock` 一整份权重 → 峰值涨到 3.97 GB（约 2.01×），门必须变红<br>用峰值（`VmHWM`）而非此刻值：内核随时可回收干净的文件页，"此刻常驻"会往下走，用它判上界等于看内核心情<br>⚠️ 该门**不进 `pixi run test`**（依赖 2 GB 本地导出），入口 `pixi run test-memory` |
| 参数清单加载（fp32 裸二进制 + TSV 索引，mmap 只读） | `verified` | `evidence:tests/unit/test_model_parity.mojo`（2.0 GB 参数以只读映射打开，构造时逐条校验参数名是否存在 —— 缺一个名字在构造期就失败，而不是在生成到第 100 个 token 时）<br>绑定权重（Qwen2.5-0.5B 的 `tie_word_embeddings`）写成**别名行**，指向同一偏移 → 加载侧无需知道 tie 的存在 |
| **真实 HF 模型目录端到端**（`config.json` + `model.safetensors` + `tokenizer.json`，全程只读这一份） | `verified` | `evidence:tests/unit/test_real_model_dir.mojo`（**6/6**）。目录由 `scripts/build_real_model_dir.py` 用**符号链接**拼出：`model.safetensors` 指向 HF 缓存里那份 988 MB bf16（不往仓库复制第二个 GB），`tokenizer.json` 指向差分语料用的那份（两者不可能漂移）。这一门验的是**只有读真目录才会撞上**的三处接缝：**① 参数全是 BF16**（4.94 亿个，就地放宽）；**② 没有 `lm_head.weight`**（`tie_word_embeddings`：输出投影就是词表矩阵本身，要求这个名字就会拒绝所有 Qwen2）；**③ 没有 rope 表**（它是 `rope_theta` 的函数，参考实现也是现算的 —— 现算一份 `[max_tokens, head_dim]`）。判据沿用 §5 其他模型门：4 条 prompt 的末位 logits 余弦 ≥ 0.999 且 argmax 相等、**16 token 教师强制 greedy 逐 token 相等**；另两条守接缝本身 —— 真分词器解出的 id 与参考 prompt 的 id **全等**（否则后面都在跟另一道题的参考答案对），以及**现算的 rope 表与参考导出的那份相差 5.96e-08（1 个 fp32 ulp，256×64）**，门开在 1e-6（比一个 ulp 高一个半数量级，比「指数写错 / 半头没复制 / 按批内位置索引」的 O(1) 差距低六个数量级）。⚠️ 重资产门（放宽 4.94 亿个参数 + 真前向，约 90 s），入口 `pixi run test-model-dir`，**故意不进 `pixi run test`**。 |
| safetensors 读取（单文件；F32 直读、BF16 就地放宽） | `partial` | `src/alofa/model/safetensors.mojo` + `TensorFile` 兼容路径：校验 8-byte header length、tensor name、dtype、shape、offset、file bounds。2026-09-21：**BF16 在加载时就地放宽为 fp32**（bf16 就是 fp32 的高 16 位，故放宽**不带误差**，见 `bf16_to_f32`）；放宽结果由 `TensorFile` 自己的 arena 持有 —— mmap 是只读的，改不动。证据：`tests/unit/test_model_formats.mojo?count=8`（其中三条是放宽门：同一组数以 F32 与 BF16 各存一份，放宽回来必须**逐位相等**；`1/3` 落在 bf16 的 2^-9 邻域内；**FP16 仍按名拒绝** —— 它不是同一种截断，负向对照）。当前边界：**仍只支持单文件**，多分片 index 未做、FP16 明确报 `unsupported`。 |
| GGUF 读取（**从张量偏移反推 block layout，不信任 type id**） | `missing` | 借鉴 `MOJO_STUFF` 的教训 |
| Qwen2.5 架构（0.5B，fp32，prefill + 单步 decode） | `verified` | `evidence:tests/unit/test_model_parity.mojo`（4/4 通过：4 条 prompt 的 logits 余弦 ≥0.999 且 argmax 相等；128 token greedy **逐 token 相等**；增量解码与整段 prefill 一致）<br>⚠️ 该门**不进 `pixi run test`**（标量后端跑完约 6 分钟且依赖 2 GB 本地导出），入口是 `pixi run test-model` |
| **投影分片并发（按输出切，默认关闭）** | `verified` | `evidence:tests/unit/test_parallel_shards.mojo?count=8`：把一次投影按输出切成 N 片并发跑（`std.runtime.asyncrt` 的 `TaskGroup`），判据是**与不切片那次逐位相等** —— 切分不改变任何一次浮点运算（fp32 通路每条输出一个 f64 累加器、q4_0 通路一个 f32 累加器，都在片内闭合），所以「差 1e-7」就是 bug，拿容差去比等于把「片起点算错」也放过去<br>形状表专门挑**除不尽**的：`out` 取 1/3/5/7/11/17/28、片数取 2/3/5/8。真实形状里的 896 与 151936 能被 2/4/7/8 整除，于是「余数没分给任何一片」这条支路**永远走不到**<br>decode（批 = 1）切**输出列**；prefill（批 > 1）因 `dst` 行主序、列不连续而改切**输出行**，偏置在切列时跟着列偏移，切行时整份共享<br>q4_0 通路另有一层：`blocks` 的片偏移是**字节**（每行 `cols/Q4_BLOCK` 块 × 18 字节），按元素算会静默读到别的行<br>**负向对照**：先把结果涂成哨兵值，再故意只算前 N-1 片（漏掉含余数的最后一片），必须被同一套比较抓出来。⚠️ 涂哨兵这步不能省 —— 不涂的话漏掉的那片保留上一次全量算出的**正确值**，正负两例逐位相同，门自己就是绿的<br>**默认已启用**：构造时取核数，上限 `SHARDS_MEASURED = 8`（本机实测 `default_shards()` = 8）。超过 8 的片数**没量过**，所以上限取"量过的那一档"而不是"越多越好"。想退回不切就显式 `set_shards(1)` —— 那条路与加这个字段之前**逐位相同**。重门 `test-model`（4/4：4 条 prompt 的 128 token greedy 与参考**逐 token 相等**）与 `test-slice`（4/4）现在跑的就是这条分片路径，不是"另有一条路在跑"<br>**端到端实测（2026-09-20，`scripts/bench_model_shards.mojo`，Qwen2.5-0.5B fp32 / avx2，32 个新 token，每 token 时间中位数，3 轮交错）**：`shards=1` **162–175 ms/token（5.73–6.16 tok/s）**；切成 2 片 **1.48–1.62×**、4 片 **1.49–1.81×**、8 片 **1.66–1.84×**（同轮配对比值，两个口径同向；非配对区间也不重合，8 片 1.69–1.90×）。**两次运行复现**<br>⚠️ 这份数**只针对批 = 1 的单流 decode**：批大于 1 时分片改切**输出行**，那是**另一个旋钮**（见下），不许拿这个倍数去说「服务吞吐也快这么多」<br>**prefill 的片数现在是单独一档（2026-09-20，`prefill_shards()`）**：在量过的行数内（`rows ≤ PREFILL_ROWS_MEASURED = 48`）把片数压到实测最好的 `PREFILL_SHARDS_MEASURED = 4`，**超出就原样返回**（没量过 → 退回改动前的样子，与 `SHARDS_MEASURED` 同一条规矩：只启用量过的那一档）。机理：`_gemm_tile[RB]` 的权重复用**只在块内**发生，`rows ÷ shards` 决定每片行数 → 片数越多复用越碎，而片数越多并行度越高 —— 两个方向相反，只能量<br>**片数扫描（`scripts/bench_prefill.mojo`，Qwen2.5-0.5B fp32 / avx2，每 token ms，3 趟 min…max，轮外层片数内层交错，三次运行）**：n=8 → 1/2/4/8 片 = 35.5–49.8 / 24.0–28.9 / **19.8–25.3** / 32.3–67.6；n=16 → 33.2–39.1 / 17.8–20.5 / **13.7–17.1** / 22.8–37.8；n=32 → 32.3–38.0 / 17.6–38.2 / **10.4–16.0** / 14.2–20.7。→ **4 片在 n=16/32 上最好，且与 8 片区间不重合（13.37 < 14.15）**；1 片最差（并发仍然要）。即在这个 8 核机器上**复用比并行度更值钱**<br>**换手点在哪儿（同脚本，n = 32/48/64/96/128 × 片数 1/2/4/8，三次运行）**：按「权重被读的遍数 = 片数 × ceil(每片行数 / 8)」，n ≥ 48 时四档的遍数拉平（都是 8 遍），所以换手点该在 32 与 64 之间 —— 量出来是 **48**：n=48 → 4 片 10.78–11.62 / 11.60–12.01 / 11.04–11.37 对 8 片 11.58–11.77 / 15.88–21.23 / 11.58–15.61（**2/3 次不重合，第三次也同向、只差 0.04 ms 没分开**）；而 n=64/96/128 **区间全重合**（只有 n=64 一次 8 片不重合地更好 → 不跨运行复现）→ **量不出结论，那三档就不动**。<br>⚠️ 换手点**随机器负载挪**：n=32 这一档在负载低时（loadavg 5.6–7.0）三次里只有一次分开，在负载高时（上一轮 loadavg 8–12）三次里两次分开 —— 8 片在机器忙时掉得更多。**4 片从来没被量成更差**，这是选它的第二个理由<br>端到端 A/B（默认片数，交替 3 次）：n=48 改后 9.35–10.44 对改前 9.42–26.20 （2/3 次不重合）。⚠️ 同一份 A/B 里 n=32/64/96/128 两边走的是**同一条代码路径**（32 两边都是 4 片、≥64 两边都是 8 片），差值却是 8.76 对 16.01 ms/token —— **「默认片数」那一行是进程刚起来、权重刚 mmap 完时量的，噪声到 ±70%，不作数**；交错的片数扫描才是证据<br>**端到端 A/B（默认片数，新旧二进制交替 5 次）**：n=8 22.5–35.7 对 56.6–108.7、n=16 13.2–24.7 对 23.8–54.6、n=32 9.7–13.9 对 12.4–35.4 ms/token → **后 4 次区间不重合，1.07–2.50×**<br>⚠️ **第 1 次反常**：改后 n=32 是 25.9–26.7，反而慢于改前 12.4–14.3 —— 那一次两个二进制刚构建完、2 GB 权重的页缓存是冷的，片数少时缺页由更少的线程服务。冷启动那一轮不作数，**但它必须写在这里** —— 只留好看的那几轮就是把结论往自己那边掰<br>⚠️ `rows > 48` **量不出差别**（n = 64/96/128 区间全重合）→ 退回调用方给的片数，不假装量过。该策略只改**并行度**、不改任何一次浮点运算，故片数不同的 prefill 必须**逐位相等**（`bench_prefill.mojo` 的自检 + `test_parallel_shards.mojo` 的 prefill 档守这条）<br>⚠️ **作废一条**（2026-09-20 自查发现）：本行此前写的"放大 1.7×，llama.cpp 同机是 4.24×，对 llama.cpp `-t 8` = 0.41–0.43×"是**跨机器比较** —— llama.cpp 那组 5.94 / 25.18 tok/s 是在 **A100 远程机的 CPU**（`-ngl 0`，fp32 gguf）上量的（见变更日志 2026-09-18），而这里的端到端数是**本机**（i7-9700K）量的。两台机 DRAM 带宽不同，绝对值与比值**都不可用**。本机**没有** llama.cpp 基准 → 250 行「与 llama.cpp 同机同 prompt 对比」对本机**仍 `missing`<br>**离天花板还有多远（2026-09-20，同进程现测天花板，`bench_model_shards.mojo` 末段）**：每 token 实测带宽对"真实 `linear` 在 8 份互不相交权重上的天花板"的利用率，三次运行都在 **745‰–1132‰** 之间（区间跨过预先定死的 700‰ / 900‰ 阈值 → 按口径写"量不出结论"）。能说的是**下界**：没有任何一档低于 745‰ → **串行段（注意力 / RMSNorm / RoPE / 残差）+ 169 × N 次协程调度合起来最多也就 ~25%**，而且算术上它们只占全 token FLOPs 的 ~0.2%，所以 **"切串行段"不是杠杆**，别去切注意力 / norms。纯读天花板与 kernel 天花板的比值也是 ~90% → **kernel 自身的访存也没什么油水**<br>⚠️ Mojo 1.0.0 只有协程这一套并发设施（无 `thread` / `parallel`）：协程参数只能是**平凡值**（指针与整数），把 `TensorView` 直接传进协程实测会**静默写错地方**（参数槽在 `wait()` 之前失效，而每片算的又是同一个值，于是「对不对」看起来像随机的），故分片视图一律在协程内用 `shard_shape()` 现造；另：`out` 是参数传递约定关键字，不能作参数名 |
| **批 / 预填 GEMM 的权重复用（换循环次序）** | `verified` | `evidence:tests/unit/test_gemm_batch_layout.mojo?count=5`：`avx2._gemm` 从「行外层、列内层」改成「`RB` 行一块、列外层、`k` 中层、行内层」，判据是**一次算 `rows` 行 == 逐行算 `rows` 次，逐位相等**（换序不改变任何一次浮点运算：每个输出各自一个 f64×4 累加器、按同样的 `k` 次序累加 → 差 1 ulp 就是 bug，不是"精度损失"）<br>形状按**分块余数**挑：`rows` 取 1–9（3=2+1、5=4+1、7=4+2+1、9=8+1，`RB` = 8/4/2/1 四种分块连同余数块全走到）；`inner` 既有 4 的倍数（真形状 896 / 4864 都是）也有 7 / 13 / 17（只有后者才走 `k` 的标量尾巴 —— 真形状里这条支路**永远走不到**，不单独造形状就守不住它）。`linear` 与 `linear_bias` 两条通路都测<br>⚠️ **参照物分两类，各有各的盲区**（实测踩到）：前两条的参照物是"逐行调用**同一个 avx2 核**"，只能守住「换序不改变结果」—— 注入「`k` 尾巴初值少了 `reduce_add()`」这处两边共用的错时，那两条**仍然 PASS**。所以另挂第三条：与**没动过**的标量后端比（`1e-5 × max(1, |ref|)`），它是绝对参照物，能接住这一类。另注入「所有行写第 0 行的结果」验证前两条确实会红 —— 两处注入各由对应的一条接住<br>**自检**：`rows` 一次算完之前先把 `dst` 涂成 -999（漏掉任何一行都会留下哨兵）；另两条负向对照「少乘最后一个 `k`」必须被同一套判据拒（逐位判据与容差判据各一条）<br>**核级实测（2026-09-20，`scripts/bench_gemm_rows.mojo`，`-O2`，`down_proj` 896×4864，8 份互不相交权重 139 MB ≫ L3 12 MB，每份算一趟）**：换序前 `rows=8` 一趟 **23.31 ms**、`rows=1` 3.95 ms —— **每行耗时不随 `rows` 摊薄**（3.95 / 3.06 / 3.94 / 2.91 ms），"若每行各流一遍权重"的隐含带宽跨 `rows` 恒定（4.40–6.02 GB/s）→ 权重确实被读了 `rows` 遍。换序后 `rows=8` 一趟 **3.51 ms**、每 token 2.75 → **0.43 ms（6.4×）**<br>⚠️ **不报"换序前后 rows=1"的对比**：两次运行的同进程纯读带宽是 7.53 与 10.85 GB/s（机器状态不同），跨运行不可比。rows=1 走的是 `_gemm_tile[1]`，操作序列与老写法一致；四道重门 `test-model` 4/4、`test-model-avx2` 4/4、`test-slice` 4/4、`test-batch-forward` 6/6 **全绿**（含 prefill logits 与参考一致、4 条 prompt 的 128 token greedy **逐 token 相等**）<br>**端到端 prefill（2026-09-20，`scripts/bench_prefill.mojo`，`-O2`，Qwen2.5-0.5B fp32 / avx2，3 趟 min…max，预热一趟丢弃，新旧二进制**交替各跑 3 次**）**：`n=32` 每 token **35.65–39.77 → 8.25–13.29 ms**（旧的 min 30.28 > 新的 max 13.29 → 区间不重合，**2.68–3.66×**，三次运行复现）；`n=8` **27.2–36.1 对 27.3–30.1 ms**，区间**重合** → **量不出差别**<br>⚠️ `n=8` 没有差别不是"换序没生效"，是**分片策略**：`prefill` 按**输出行**分片，8 行 ÷ 8 片 = 每片 1 行 → `RB=1`，块里没有第二行可复用；换序只在同一块内摊薄。（这一步当天已在变更日志里落地：`prefill_shards()` 把片数压到量过的 4 片 —— **`RB=1` 那一档从此不许出现**，见当日变更日志）<br>**批大于 8 也量了（2026-09-22，同脚本把 `rows` 扩到 16 / 32，三次运行的 min…max）**：每行耗时 `rows`=1 **1.42–2.46**、2 **0.80–1.35**、4 **0.51–0.66**、8 **0.36–0.64**、16 **0.38–0.48**、32 **0.44–0.54** ms —— 8 以内按约 1/rows 掉，**8 以上不再掉**（8 / 16 / 32 三档区间互相重合）→ **复用被 `_gemm_tile` 的 `RB = 8` 封住**：`rows = 32` 时权重读 `ceil(32/8) = 4` 遍，每行摊到的字节与 `rows = 8` **完全相同**。所以「再多攒一批」不是这条路上的下一个杠杆。<br>**抬 `RB` 也试过并拒绝了（同一天，`scripts/bench_gemm_rb.mojo`，同进程配对、5 轮交错并轮换出场顺序）**：`RB=16` 对 `RB=8` 的**配对比值**在 `rows=16` 是 **0.78–0.81×**、`rows=32` 是 **0.79–0.80×** —— 权重流量**减半**、每行耗时却从 0.38 → **0.48 ms（慢约 24%）**，两档的结论一致且区间极紧（不到 3%）。机制就是写在那里的那条反方理由：`RB=16` 的热状态是 512 + 128 字节的累加器，AVX2 寄存器装不下，必然溢到栈。<br>**由此能推出的一件事**：`RB=8` 上 **DRAM 带宽已经不是瓶颈** —— 若是，流量减半至少得换来一点时间，而这里是**反向**的。所以这条核级通路的下一个杠杆**不在权重复用上**（既不是更大的批，也不是更大的 `RB`）<br>⚠️ 三轮里第 1 轮机器是忙的（同进程纯读带宽 T=1 只有 6.91 GB/s，第 2/3 轮是 11.22 / 10.26）→ 上面报的是**三轮合起来的 min…max**，单独引用某一轮会把结论往任意方向掰<br>⚠️ 换序**仍然只改了 `avx2._gemm`**：`scalar._gemm` 与 q4 通路（`_matmul_q4`）**没动**，批 > 1 时它们仍每行各流一遍权重 |
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
| 纯函数调度器（单一 token 预算） | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=19`<br>`pixi run mojo run -O0 -I src tests/unit/test_scheduler.mojo`<br>无 I/O、无时钟、无模型、无权重依赖；`step(input) -> Action` 是唯一入口，于是调度边界可以脱离模型被测（创新点 3） |
| 调度器自身零堆分配（定容容器 + 源码门） | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=19`（类型层面：所有容器是编译期定长的 `InlineArray`；源码门扫描 `src/alofa/engine/scheduler.mojo` 不得出现 `List[` / `String(` / `Arena(` 等构造点，并有常驻红测 `tests/fixtures/bad_alloc.mojo` 必须被判违规）<br>⚠️ **这条证据的边界**：能拦住“给调度器加一个会增长的容器”，**拦不住** libc 里的小块分配，也**不等同**于进程级 RSS 不动 —— 账本就按这个口径写，不夸大成“进程零分配”。录 trace（`engine/trace.mojo`）**会**分配 String，所以录制是 `step` 之外的可选动作 |
| chunked prefill | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=19`（300 token 的 prompt 跨 19 拍切片：首尾相接、不重叠、每片不超过 `max_chunk`；同一拍不得给同一请求两个 chunk） |
| 抢占（重计算） + 抢占计数指标 | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=19`（并发抢占风暴 / KV 水位临界两个场景；被抢占者 KV 全作废并回到等待队列，累计抢占次数作为 `Action` 字段逐拍比对 —— 它是容量告警指标，不是调试字段）<br>⚠️ 只抢占 **RUNNING** 请求：抢占“半截 prefill”的请求会让 prefill 永远无法完成，那是伪装成策略的抖动<br>⚠️ 池子小到连一次 decode 增长都装不下时报 `capacity` 具名错误，**不静默丢 token**（有专门断言） |
| 调度 trace 录制 | `verified` | `evidence:src/alofa/engine/trace.mojo` + `tests/fixtures/scheduler/*.trace`（格式：整数 + 定长字段，**不含浮点**；水位用千分数而非比例 → 逐字节门不会退化成容差门） |
| 调度 trace 重放 + 极端场景断言 | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=19`（6 个场景与 `scripts/dump_scheduler_reference.py` 这份**独立 Python 实现**逐字节相同；三条常驻负向对照：改坏的 trace、换一种抢占顺序、给调度器加堆容器，三者都必须被判红）<br>6 个场景：超长 prompt、并发抢占风暴、预算耗尽、0 预算、取消竞态、KV 水位临界 |
| 延迟护栏（最大等待拍数） | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=19`（构造“队首长 prompt 每拍吃光预算”的最小复现：护栏生效时第 4 拍必须给短请求；把 `max_wait_ticks` 调到 99 该断言**实测会失败**）<br>护栏只在**等待者之间**插队，不越过 decode：它防的是“前面有个超长 prompt”，不是“预算被 decode 占满” —— 后者说明并发已饱和，插队只会把等待转嫁给已经占着 KV 的人 |
| 批张量池（稳态零堆分配） | `verified` | `evidence:tests/unit/test_batch_pool.mojo?count=12`<br>`src/alofa/engine/batch.mojo`：借用拿到的是**句柄**而不是指针（释放后同一个槽位会给别人）；池子不拥有内存，构造时收一个指针加容量；全部簿记是编译期定长的 `InlineArray`，**没有空闲链表** —— 每次放置都从“活着的借用”重导出可用空隙，于是不存在两套会互相漂移的账<br>7 个场景与 `scripts/dump_batch_reference.py` 这份**独立 Python 实现**逐字节相同：分配是“决定”不是数值，两种放置之间不存在“差一点点”<br>两条不依赖参照物的性质：**活着的借用不共享任何一个字节**（每个区间写自己的标记再逐字读回 —— 把放置往旁边挪一格，这条实测会红），**峰值不漂移**（20 个相同执行步之后 high-water 与第一步相同；否则稳态不稳，之后测的吞吐就是关于另一个池子的数字）<br>**拒绝必须是具名错误**：池子填满后隔一个释放，剩 131072 字节可用而最大空洞只有 16384 —— 借两块必须报 `capacity` 而不是绕回去或跨两个空洞凑；第 25 个活的借用同样是 `capacity`；重复释放是 `double_free`<br>3 条常驻负向对照：`alt.trace`（同一组操作改用 best-fit 放置，且**要求它至少挪动一处偏移**，否则那 6 个逐字节比对只是在验文件格式）、`bad.trace`（一处偏移挪一格）、给池子加堆容器的 `bad_batch_alloc.mojo`<br>⚠️ 证据的边界：拦得住“给池子加一个会增长的容器”，**拦不住** libc 的小块分配，也不等于进程级 RSS 不动；且这一层本身**不比对数值**（借到的字节里算得对不对由下面两行负责） |
| 批组装（每请求一段连续行） | `verified` | `evidence:tests/unit/test_batch_pool.mojo?count=12`<br>`BatchSlots` 给每个请求一段**连续**、且**不与别人重叠**的行：两个请求共用一行时，某一层会从别人的 token 上读出自己的激活，产出的每一个数都看起来合理 —— 与分页内核读错块是同一种失败<br>断言从外面重算：逐行统计所有者，**重复覆盖与未被覆盖都必须为 0**；同一请求被加两次是具名错误（一次 add 会让它拿到两段行，而某一层只会读其中一段）<br>⚠️ 只在**行归属**这一层成立；“注意力按请求分块”由 executor 接走（下一行） |
| 批执行器（注意力按请求分块） | `verified` | `evidence:tests/unit/test_batch_executor.mojo?count=12`<br>`src/alofa/engine/executor.mojo` 把 2.5 的两层接进前向：每个请求拿到一段**连续行**，每一行被交给注意力时都带上**自己的** K/V 基址、`upto`（能看到的最末一个 key）与 `pos`（rope 用）—— 于是“不同请求的行互相 attend”不是靠一张可能被丢掉的 mask 挡住的，而是**地址上不存在**：一行从来拿不到别人的地址<br>6 个场景与 `scripts/dump_batch_executor_reference.py` 这份**独立 Python 实现**逐字节相同（ADD/FEED/DROP/PLAN/ROW/FIN/NEXT 全序列 + 每步摘要 `d=`），摘要覆盖全部请求槽与全部行，不变量**每一步从外面重算**（缺陷计数不为 0 即红，而不是只在结尾查一次）<br>3 条常驻负向对照：`alt.trace`（换一种行分配策略，且**要求它真的不一样**，否则 6 个逐字节比对只是在验文件格式）、`bad.trace`（某一行的可见窗口挪一格）、给忙碌循环加堆容器的 `bad_executor_alloc.mojo`<br>每一步向 2.5 的池子借 15 块、步末全部归还：实测 `used==0`、`n_live==0`、峰值不漂移<br>⚠️ 边界：这一门**不比对数值**（注意力算得对不对由 §7 的分页门与下一行的批一致性门负责），它验的是行归属、每行的窗口与簿记 |
| 批一致性（批大小 1/2/4/8 与单请求逐 token 相同） | `verified` | `evidence:tests/unit/test_batch_forward.mojo`（4/4；重门，需要 2GB 权重，`pixi run test-batch-forward`，故意不进 `pixi run test`）<br>同一批 prompt 走两遍：**一批 N 条** 与 **一条一条跑**（`QwenForward.prefill`/`step`，也就是 `test_model_parity` 拿去和 Hugging Face 对过的那条路径），greedy 解码、逐 token **相等** —— greedy 让“第 3 个 token 不同”就是一个不同，而不是差一点点<br>N = 1 / 2 / 4 / 8：8 条请求的 prompt 合计 88 行 > 行块 64，所以这一门**真的把一次 prefill 切成两拍** —— 短的一拍也必须是对的一拍<br>2 条常驻负向对照：**不同 prompt 必须解出不同续写**（否则“批次与单请求一致”对任何实现都成立，包括不看输入的实现）；**交换两条请求的 prompt 必须被察觉**（交换后既**不等于**该槽位的基线、又**等于**它实际拿到的那条 prompt 的基线）—— 这才让逐字节比对成为“行归属”的证据<br>⚠️ 参照物是**本树的串行前向**，与批路径共享 kernel：这是刻意的，被比较的是**编排**（行归属、每行的 pos、每行的窗口、KV 区域），而串行路径只有一条请求、不可能在这些上出错；参照物本身对 Hugging Face 的一致性由 §5 的模型门负责 |
| 引擎循环（调度器 ↔ 批执行器接线） | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=18`（已进 `pixi run test`）<br>`src/alofa/engine/core.mojo` 把 2.0 的调度器与 2.5b 的批执行器接成一个忙碌循环：调度器出**决定**（谁 prefill、给 `[start, end)` 这一段、谁 decode、谁被抢占），执行器出**行**；被验的只有两者之间的**翻译** —— 切片喂给谁、prefill 结束那一拍白送的第一个 token 与之后 decode 出来的 token 怎么拼成同一份 transcript、抢占后重算要作废什么<br>argmax **由测试注入**（每槽一个整数，不跑模型）：贪心 argmax 是一行代码，这一门要验的是**时序**；也正因为期望值写成 `expected_token(请求, 第几个 token)` 而与「第几拍产生的」无关，同一份期望才能同时管住「抢占后被推回 prompt、重新生成一遍」的请求<br>3 条常驻负向对照：① 抢占场景**断言 `preempt_total > 0`** —— 声称「抢占安全」却从头到尾没抢占过的门，是穿着戏服的 happy path；② **把请求从执行器手里抽走再要一拍，必须报 `invalid_argument`**（静默服务一个空请求更省事，也更会藏 bug）；③ 给循环加堆容器的 `bad_executor_alloc.mojo`<br>每一拍都从两边重算「谁还活着」：引擎说谁 resident、执行器说它握着谁，二者不一致的那一拍**照样产出一串看起来是 token 的数**<br>⚠️ 边界：这一门**不比对数值**（续写得对不对由上一行的批一致性重门负责），也不意味着 KV 物理块池已接入 —— 执行器用的是自己构造时写死的 KV 区域（见下一行） |
| KV 物理块池（含 `freed_blocks` 这类外部释放） | `verified` | `evidence:tests/unit/test_kv_room.mojo?count=10`（已进 `pixi run test`）<br>`src/alofa/engine/kv_room.mojo` 是唯一做这层翻译的地方：调度器**数**块但不拥有块，`runtime/kv` 拥有块但只会说 radix 树操作。三件只活在这一层的事：① prefill 到达是**切片**，一次入场是多次 `append_tokens`，而「prompt 有多长」是另一件事实（它决定半个 prompt 不许进缓存）；② 完成的请求**换主人**（`commit` 发布到树），块变成「没有主人的占用」= 前缀缓存；③ 缓存的块只从**一扇门**回来 —— 驱逐是引擎的决定（只有引擎知道压力），还回多少**实测**（缓存块数前后之差）而不是记账<br>引擎侧接线（`src/alofa/engine/core.mojo`）：prefill 首片 `admit`、续片 `grow_to`、`settle` 后按「prompt + 已生成」对齐长度（给目标值不给增量）、完成 `publish` 再 `drop`、抢占与取消直接 `drop`；归还经 `SchedInput.freed_blocks` 回报调度器，两条规则都在调度器决定之前执行：**缓存让位给活着的请求**（缓存 ≤ 容量 − 持有）与**水位**（缓存顶高水位时先回收，免得调度器为还不了的块去抢占）<br>5 个场景与 `scripts/dump_kv_room_reference.py` 这份**独立 Python 实现**逐字节相同（`u/f/c/d`），且每拍从视图重算不变量、断言 `used + n_free == MAX_BLOCKS`；3 条常驻负向对照：`bad_room.trace`（改坏一拍）、`alt_room.trace`（换 prompt，必须判红 —— 否则只验了格式）、`bad_room_alloc.mojo`（给房间加堆容器）<br>不依赖参照物的性质：相同 prompt 的第二条 `last_matched == 32` 且 `last_fresh == 0`、池子用量不变；发布后 `used` 不降、回收后块真的回池且上报数等于实测；**未发布的 drop 必须真的回池**；`reclaim` 对活着的请求必须还回 0<br>⚠️ 边界：`runtime/kv` 并发上限 `MAX_REQUESTS`（8）、单序列上限 `MAX_SEQ_TOKENS`（64），第 9 条报 `capacity`、超长报 `out_of_range` —— 限制被**断言**而非绕过（悄悄少给几块会在几拍后变成「少一个答案」）。调度器的占用仍是**算术**的、房间的才是**物理**的，二者不要求相等（前缀共享让物理更少、节点粒度让物理可能更多），物理池满时房间具名拒绝。⚠️ 执行器的 KV 已从这张块表取地址（见后两行）；房间给了块之后，前向读到的位置由**表**决定，不是由算术决定 |
| 分页注意力接入引擎循环（前向从块表取地址） | `verified` | `evidence:tests/unit/test_paged_scatter.mojo?count=5`（已进 `pixi run test`）+ `tests/unit/test_batch_forward.mojo`（重门，真实 0.5B fp32 权重）<br>块池布局：一个块**持有每一层各一个槽**，所以请求的表在每层都叫同一批块号 —— `engine/executor.mojo` 把层号折进块 id（`layer * MAX_BLOCKS + block`），于是整个块池只有**一个**视图、在构造时建好，忙碌循环里不再构造形状（每步每层建一个 `List` 正是这一层的源码门要挡住的事）<br>`kernels/cpu/paged.mojo` 新增 `paged_scatter`：写**经过**表，与读经过同一张表。写按算术放（`j // block_size`）会把 token 放进「它若不共享前缀本会占用的块」，之后每一步都是从别人的历史里算出来的数；写越界报 `out_of_range` 而不是截断（截断是悄悄变短的上下文）<br>房间与执行器的交接只有一处：房间 `page_table()` 拷出表 → 引擎 `sync_page_table()` 在 `admit` / `grow_to` **之后立刻**交过去；表按 `hist + n` **裁剪**后再用 —— 房间可以为还没到的 token 预留整块（切片 prefill），而注意力不许读没人写过的位<br>批一致性重门跑在**乱序块号**下：块刻意不按连续区域的顺序排，所以「批与串行逐 token 相同」这句话是关于**页表**的 —— 前向若按算术取地址，数就不同<br>⚠️ 边界：`rows_view` 构造视图（形状是 `List`）在执行器 `forward` 里仍然存在 —— 块池的**账**是零分配的，视图构造不是 |
| 共享前缀只算未命中的那一段（前缀缓存省的是算术） | `verified` | `evidence:tests/unit/test_batch_forward.mojo`（6/6，真实 0.5B fp32 权重，重门）+ `tests/unit/test_batch_executor.mojo`（12/12，已进 `pixi run test`）<br>引擎层**端到端**已验（`tests/unit/test_engine_core.mojo` 11/11）：同一个 prompt 提交两次，第二次的第一拍**只跑一行**且 `history_of == 6` —— 此前这条链路只是「编译通过 + 单元绿」，房间 → 引擎 → 执行器这一段没人跑过；常驻对照是同一引擎里的冷 prompt：7 行、history 0。| KV 池高占用下的正确性 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=18`：四条请求同时在池（峰值 = 四条块数之和），全部跑完且**逐 token 等于逐条跑的基线**；每拍重算 `used + n_free == MAX_BLOCKS`、房间 `invariants()==0`、两本账 `defects()==0`<br>⚠️ **不声称 95%**：2026-09-18 撤回前一天记的「峰值 111/112」——那个数字是被下面那行的记账 bug 造出来的（房间白发整条 prompt 的块），修好后同一场景只到 99/112，边界见下面两行 |
| 分块 prefill 下的两本账 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=18`：**已修**：引擎原来在 `settle` 里把 KV 序列长度设成「整条 prompt + 已生成」，无视 prefill 只喂到第 16 个 token —— 房间因此白发整条 prompt 的块（实测第一拍：房间 15 块、调度器账 4 块；七条跑下来差 10 块），水位 950‰ **全程不触发**，池子只靠房间抛 `capacity` 兜住。现在按「已喂到的位置」grow，分块下两本账差 ≤ 1（`test_the_scheduler_and_the_room_count_the_same_blocks`；旧行为下该门差 10 块、红）|
| 物理池 >95% 且能跑完的场景 | `missing` | 在当前实现下**不可达**：`MAX_ROWS=64` 只能逐条 prefill，先完成的先释放，分块下峰值 98/112（87%）；要顶满就得让请求长驻留，而驻留总量一旦高过水位，抢占就在两条请求之间来回抢、谁也完不成（七条各生成 8 个 token：512 拍仍不空闲；容量预算 48 / 阈值 45 下四条同样活锁）。解锁条件：让抢占真正缓解而不是循环——受害者重算时应优先拿回块，或水位只在「有等待者需要块」时触发 |
| 缓存让位与缓存账的时序 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=18`：五条请求分一个装不下的预算（60），缓存必须让位，否则排队的请求永远拿不到块。修了两处：① **缓存占死预算**——yield 只按「已在跑的」算，缓存把预算吃满，四条请求在剩下的块里互相抢占，有一条一个 token 都没生成；现在按 `blocks_used + blocks_wanted()` 算，缓存只留别人用不到的。② **引擎交回调度器尚未记账的块**——上一拍发布的序列要等本拍 `step` 才进 `cached_blocks`，reclaim 却发生在 `step` 之前，于是下一拍的 `freed_blocks` 大于调度器认为的缓存，抛 `ERR_INVALID_ARGUMENT`；现在只交回调度器已记账的部分。负向对照：把 ① 改回旧算法，该门红在「512 拍从未空闲」 |
| 共享前缀下的缓存账 | `missing` | 调度器 `release(to_cache=True)` 按「每条已完成序列自己的块数」累加缓存，而房间的前缀树去重后只占一份 → 两本账不同源（调度器高估）。**首 token 归属那条已修**（见下），剩下的只有去重这一条。**回滚过一次**「让引擎把差额延后一拍用 `freed_blocks` 报出」：两本账当时对齐了，但随后撞 `id list overflow`，且回收与修正同拍叠加会超账。解锁：缓存占用数只能由房间报告，调度器不得自行推算——`SchedInput` 需要一个独立的「缓存增量」通道 |**第二次尝试也已回滚**（2026-09-18）：给 `SchedInput` 加绝对值通道 `cached_now`，房间每拍报真值覆盖调度器的和。失败原因不是实现细节：调度器的 `release` 比房间的 `publish` **晚一拍**（完成消息延后送达），于是「对齐到上一拍真值 + 本拍 release 整条」仍在叠加——实测对齐到 36 之后又加上两条的 24，得 60，而房间是 48；补 `cache_pending` 让引擎多跑一拍也没能把 60 降下来。真正的解锁是「调度器不再自己维护缓存账」，而这跟「调度器是纯整数函数、重放门不依赖房间」直接冲突（参考实现没有房间，报不出真值，那时调度器又必须能自己算）→ 属于架构取舍。
| 过载 + 长 prompt 的抢占活锁 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=18`（两条门：抢占确实发生，且受害者重算后仍跑完全程）<br>`evidence:tests/unit/test_scheduler.mojo?count=19`（trace 逐字节：抢占发生后拍末 blocks_used 回到阈值内、无一拍越过硬容量）<br>原探针（未固化为门）：容量 10 块、4 条请求各 20 prompt（16 行一拍 → 跨两片）+ 3 生成、块 4 字节 → 300 拍、**190 次抢占、0 个 token 产出**（⚠️ 该数字取自 `watermark_permille=8`，即水位 **0 块**的病态配置，不是默认——`SchedConfig` 第 5 个参数是水位千分比，早先误当成了预算）。**该活锁已修复**：根因是 `admit` 只检查单条 prompt 的静态块数，未检查「已提交序列的完整稳态足迹 + 本请求完整序列」是否超过 `threshold_blocks()`；多条单独可容纳的请求合计超过水位后，才会在 decode 中反复抢占、重喂。现在 admission 对**首请求也执行水位检查**，超过水位立即抛具名 `capacity`，不再让请求进入必然重算的状态；同一拍多个 arrival 采用原子 admission，后到请求失败时回滚本拍已经接纳的请求。引擎层同步清理已经排队但被拒绝的 slot，`has_work()` 不会对同一个不可能请求无限重试。**保留的边界**：缓存 + 并发足迹合计超过水位的路径仍由引擎 reclaim/抢占处理，既有重算后完成门继续覆盖它；本次修复只拒绝并发足迹自身无法满足水位的组合。⚠️ 早先写的「喂一半被抢占」是**错的**：抢占只针对 `ST_RUNNING`（`:564`），部分喂的是 `ST_WAITING`，不会被抢占。已改为 admission 阶段按稳态足迹拒绝；不改变既有 scheduler trace 的正常路径 |
| 已发布序列的块数 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=18`：调度器按 `done + generated` 算一条已发布序列占多少块，而 `generated` 数的是 decode 拍——续写的**第一个 token 由「把 prompt 喂完的那一步」产出，不算一拍**，于是每条少记一个 token；跨块时少一整块（实测四条：44 对 48）。改按 `prompt_len + max_new` 记，并顺带补齐 `blocks_used`（它留着的是按拍算的旧数，否则池账比缓存账少同样多）。新门 `test_a_published_sequence_is_counted_whole`：37+8=45 token 是 4 字节块的 12 块，44 是 11——**块粒度 8 时两者都是 6 块，同一个 bug 会溜过去**（现有那两个门正是块粒度 8，当时全绿） |
| 引擎空闲判定与块释放 | `verified` | `evidence:tests/unit/test_engine_core.mojo?count=18`：批里最后一条请求完成后，消息要到下一拍才到调度器，`has_work` 却只看引擎自己的 state → 它宣布空闲，那条请求永不 `release`，实测 11 块永久占用（每批泄漏一次）。对称地，房间已回收的缓存块若没被下一拍带走，调度器会一直为它们记账（实测 32 块）。修：`has_work` 也认 `n_report` 与 `room.freed_pending`；新增 `room.take_freed_upto()`——回收按节点整块释放，可能多于请求量，多出的留到下一拍再报。空闲时两本账归零。负向对照：去掉这两个条件 → 该门红在「仍有块被持有」 |
房间 `admit` 时就知道重合多少（`last_matched`），引擎把这个数交给执行器 `add` 的 `matched`：命中的 token **入队但不建行** —— 历史从 `matched` 起算，队列里只剩没算过的那些。省下来的是**行**，行就是算术<br>**最后一行永远要算**：它的 logits 是第一个生成的 token。一个被完整命中的 prompt（`matched == n`）跑一行，不是零行 —— 零行就没有 logits，请求无从开口；`matched > n` 具名拒绝（`out_of_range`）<br>证据是端到端的：同一个 prompt 跑两遍，第二遍沿用第一遍的**同一批块**（`drop` 只忘地址、不清字节，这正是前缀缓存的定义）、`matched = n - 1`，跑一行，生成的 token 与串行**逐 token 相同**。配套常驻负向对照：谎报命中（`matched = n - 1` 但指向没人写过的块）必须产出**不同**的 token —— 否则上面那条可以因为「压根没读缓存位置」而白过<br>⚠️ 边界：命中的字节必须与本地计算**逐位相同**才成立（同机、同权重、同路径、同位置 —— 换 backend / 跨机未验）；执行器**不校验**块里真的是那段前缀，它信任房间 —— 谎报由上面的对照拦，不由类型拦 |
| paged attention（block table 索引） | `verified` | `evidence:tests/unit/test_paged_attention.mojo?count=11`<br>两类断言用两把尺子：**寻址用逐位相等**（`paged_gather` 与参照导出的连续行逐位一致；分页 kernel 与连续 oracle 逐位一致 —— 没有重排就没有"差一点点"的余地，差一个 ulp 就是地址算错），**公式用 1e-5 容差**（期望值来自 `scripts/dump_paged_reference.py`，与 Mojo 不共享任何代码）<br>**5 条常驻负向对照**：改坏一个元素的 `expected_bad.tsv`、用错 GQA 映射（`h % n_kv_heads`）的 `expected_hmap.tsv`、给 kernel 加堆容器的 `bad_paged_alloc.mojo`、把每段 run 的尾槽灌成垃圾值后输出必须逐位不变、共享同一块的两个请求必须读到同一段字节<br>⚠️ 夹具里的表是 `KvSpace` 对 2.1 真实操作序列重放出来的（8 例），另 3 例是 `syn_*` 块内偏移用例：当前树只从根共享、请求都从槽位 0 分配，**块内起始的 run 走 `KvSpace` 造不出来、走 `PagedTable` 造得出来**，内核就必须对它负责 |
| 垂直切片（一句真文本走完 tokenizer → model → engine → sampler） | `verified` | `evidence:tests/unit/test_vertical_slice.mojo`（4/4；重门，需 1.9GB 权重，`pixi run test-slice`，故意不进 `pixi run test`）+ `src/alofa/cli.mojo`（`pixi run generate`）<br>**这一行补的是一个真实存在的洞**：此前每一层都有自己的门且都是绿的，但**没有任何一处把它们串起来跑过** —— `test_engine_core.mojo` 的文件头自己写明 argmax 由测试注入。2026-09-17 预告过"各层各自绿、拼起来崩到 0/512"，这一行就是把那条路径固定成每天能走一遍的东西。<br>实测（真权重，scalar 后端）：`"The capital of France is"` → 贪心 `" Paris. It is the largest city in Europe and the second largest in the world"`；采样（温度 1.0、seed 固定）`":\nA: Paris B: not sure C: london D: BERLIN"`，两条路都说到 Paris。<br>**负向对照（本行的关键）**：温度 0.01 的采样必须**逐字等于**贪心 —— 若 `run_sampled` 悄悄退化成 argmax（logits 取错行、`build` 没被调用），"说出 Paris"照样全绿而采样路径一次都没生效过；温度趋零时分布塌到 argmax 上，两条独立路径必须给同一个答案。而温度 1.0 时两者**不同**（上面两段文本），一正一反才构成完整证据。<br>**本行真正的收获是两条契约，都不是猜测、都是撞出来的**：<br>① `tick` 内部硬编码 `model.argmax`，sampler 此前**没有任何介入点**。而采样循环**不能**加进 `core.mojo`：那个文件既是 `step_path_sources()` 之一（`test_core_tensor` 在其中查 `List[Int]()`），又是零分配门的 `SOURCE`（`test_engine_core` 在其中查 `List[`），而 `Sampler.build` 只收 `history: List[Int]`、**没有指针重载**。实测在那里加一个 `history_of` 会让**两个门同时变红** —— 门是对的，那是真承诺，于是循环内联在 `src/alofa/cli.mojo`，`core.mojo` 一字未改（236 项与全部 trace fixture 不受影响）。<br>② **同名常量两个取值**：`MAX_BATCH` 在 `engine/batch.mojo` 是 **8**、在 `engine/scheduler.mojo` 是 **32**；`core.mojo` 经 `executor` 拿到的是 **8**。垂直切片最初从 `scheduler` 导入，拿 32 去遍历只有 8 个槽的 `ex.live`，实测崩在 `Assert Error: index 8`。这不是类型能挡住的（`range()` 两端都是 `Int`），只能靠"从哪导入"这一行注释守住 —— 已写进 `src/alofa/cli.mojo`。<br>⚠️ 由此留下一个**未收回的边界**：这条采样路径**不在** §7 的零分配承诺内（每拍为每个活跃请求建一个历史列表），它只服务单请求 CLI；要进忙碌循环，必须先给 `Sampler` 一个指针版 `build`，那时这个循环才搬得进引擎。<br>⚠️ 边界：只验 **1 条请求、16 个 token、scalar 后端、单条序列 37 token**。不验 AVX2、不验批、不验流式输出（逐 token decode 的 UTF-8 边界未验），也不验生成质量 —— 0.5B 模型答得对不对不在这条门的职责内，它只负责"链路通" |

## 8. 服务层（L5 / L6）

| 能力 | 状态 | 证据 |
|---|---|---|
| **依赖 `flare` reactor 作为事件循环** | `verified` | `evidence:tests/capability/test_reactor.mojo?count=4`（非阻塞 accept 到 EAGAIN、poll 超时精度、跨线程 wakeup、一条循环同时看 8 条连接）+ `evidence:tests/unit/test_loop_server.mojo`（真服务跑在它上面）。原"自研 epoll"方案被推翻（§1.3）；**2026-09-22 已接入**：单进程与多 worker 两条入口都跑 `srv/loop.mojo`（`flare` 0.2.0 的 reactor / scheduler / timer_wheel / watchdog / reuseport / io_uring 里，本版只用了 reactor 与 reuseport） |
| reactor 线程 + engine 线程池（3.2b 独占线程 → 3.2c N 条并发生成） | `verified` | `evidence:tests/unit/test_engine_thread_server.mojo`（重门，`pixi run test-engine`：真 socket + 真 fork + **真线程**，故意不进 `pixi run test`）+ `evidence:tests/unit/test_conn.mojo?count=15`（`take_with_raw`：跨线程过的是原始字节）<br>**后半（engine 独占线程）2026-09-22 落地**：前向搬到一条独占线程（`ThreadHandle` + mailbox），reactor 上只剩 I/O —— 调度决策仍留在 reactor（可重放、零锁），箱体只有一个 `Mutex` 且临界区只有 append/pop（里面没有 I/O，也没有前向）。<br>**判据是「生成期间这条循环还在 accept」**：槽位被占满时新连接会被**接进来立刻关**（`MAX_CONNS` 那条规则：留在 backlog 里同样会空转，而「接了又关」至少让对端立刻知道），对端因此能靠 **EOF** 看见这件事 —— 探子从连上到收到 EOF，`run_threaded` **0 ms**，`run` **4.5 s**（那 5 s 的生成它全等完了；`run` 是**常驻负向对照**：它快了就说明这条门没盯住那条性质）。<br>⚠️ **为什么不用「生成期间还有字节在走」**（直觉上更直接，实测走了两遍都不成）：服务端一次能排队的字节被 `SEND_HIGH_WATER`（1 MiB）封住，而 loopback 上内核给一条 socket 的缓冲实测能吃下 **1.2–2.5 MB**（自动调优，随发送速率变）—— 服务端手上一字节不剩，客户端**不需要循环参与**就能把响应读完（实测两种模式一样快：内联那侧 38 ms 收完 800 KB）；读那一侧同样被挡住（`MAX_BODY_BYTES` 也是 1 MiB，而限速发送实测能被吸收 1.26 MB）。那个数字由**内核**决定，不由被改动的那一处决定 —— 而 accept 内核替不了应用做<br>⚠️ **它换来的不是并发生成**（这句是 3.2b 当时的状态）：engine 只有一条线程、一次一个 job，所以「慢生成期间另一条连接发一个请求」在两种模式下**都会**等（health 排在慢请求后面）。拿后者当判据的话，一条请求都没答的实现也能「看起来很快」。<br>**2026-09-22 3.2c 已把它扩成线程池**：`engines>1` 时 N 条线程各领一份 handler（`Twinable.spawn_twin`，**一份 = 一份权重**），同时能生成 **N** 条。判据是「**两条 5 s 生成的总墙钟**」—— `engines=2` **5002 ms**、`engines=1` **10002 ms**（后者是**同一条代码路径**上的常驻负向对照，它不慢就证明不了门盯住了什么）。代价 **N 条 = N 份权重**（尚无「共享只读权重 + 各一份 KV」那一层）→ **默认 1**，由 `ALOFA_ENGINE_THREADS`（1–16）开；多 worker 下是 `workers × engines` 份。<br>**三条约束各有安排**：流的**相位亲和**靠槽位→线程的**黏性 `owner`**（第一次派活定归属，之后不变）+ 单槽位单在途；同连接**按序**同上；**邮箱上界不破**靠**单队列按线程过滤**（每线程一队列会把一个 `MAILBOX_CAP` 悄悄变成 N 个上界）。<br>**跨线程只过一种形状的东西**：请求以**原始字节**过线程（`HttpRequest` 带着入站缓冲的借用关系，挪不进邮箱；“从结构体中间挪走一块”在这个编译器上也过不去），engine 侧重新解析一次；应答与流帧则是一段段字节加一个 kind。<br>**三条不能少的记账**：邮箱有上界（`MAILBOX_CAP`，**派之前先问 `has_room()`**：派了又失败等于把请求丢了）；每个槽位带 `gen`（结果回来时槽位已换主人就丢掉 —— 否则应答会发给**另一条连接**）；`inflight` 记着「这条连接在等生成」，空闲超时不许收它。<br>**流的头单独一个 kind**（`RES_STREAM_HEAD`）：一帧也是 `RES_BYTES`，合成一个的话循环不知道该开始喂帧 —— 症状是客户端只收到一个响应头，然后挂到空闲超时。<br>**生命周期**：放 mailbox 的那个值在 `join` **之后**必须再被碰一次（ASAP 析构会让它提前死，子线程就用上了已释放的锁 —— 症状是子进程 `dumped core`） |
| libc 事件循环原语（自研**回退**路径） | `verified` | `evidence:tests/capability/test_libc_ffi.mojo?count=5`：`socket`/`epoll_create1`/`timerfd_create`/`eventfd`/`SO_REUSEPORT` |
| 阻塞式 TCP 传输（`flare.tcp`） | `verified` | `evidence:tests/capability/test_deps.mojo?count=7`：socket / bind / listen / accept / TCP_NODELAY 全是 flare 的（A7：依赖而不自研）。<br>⚠️ 与上面 reactor 那一行不是一回事：这里用的是**阻塞**接口，没有事件循环 |
| reactor 事件循环（**一条循环 N 条连接**） | `verified` | `evidence:tests/unit/test_loop_server.mojo`（重门，`pixi run test-loop`：真 socket + 真 fork，**故意不进 `pixi run test`**）+ `evidence:tests/unit/test_conn.mojo?count=15`（连接机，无 socket，逐字节）+ `evidence:tests/capability/test_reactor.mojo?count=4`<br>**它替换的是「一 worker 一连接」**：一条连接上的 `send` / `recv` 不再挡住别人 —— 8 条连接同时开着同时收应答，一个「发了一半就停住」的对端让别人增加的延迟是 **0–1 ms**（旧形态会停满 5 s 接收超时，所以这条门带 2 s 上限，并备有把服务换回 `srv/server.mojo` 的负向对照：实测必红）。<br>**它没改「一次只生成一条」**（这句记的是 3.2 当时的状态）：请求的前向那时候在循环里**内联**执行，占住这条循环 —— 3.2b 已把前向搬到独占线程，3.2c 已把它扩成线程池（同时能生成 N 条，见上一行）。换来的不是吞吐，是「连接的 I/O 不再互相挡路」。<br>三条容量线各有判据：槽位上界 `MAX_CONNS`（满了就接了立刻关 —— 让对端马上知道，而不是等一个不来的应答）、出站队列上界（背压，见下行）、空闲连接 30 s 收（**只收真没事干的**：有完整请求在排队等生成的、还有字节没写出去的、正在写流的三条例外 —— 收错方向会让一个慢请求变成一次看起来随机的失败）。<br>⚠️ 边界：**一个坏对端只带走它自己**（畸形请求 → 丢该连接、循环继续，不会让服务退出）；不验 TLS；**engine 已独占线程**（见上一行）：循环上只剩 I/O，前向在另一条线程上 |
| HTTP/1.1 解析与连接状态机 | `verified` | `evidence:tests/unit/test_http.mojo?count=14`<br>线格式写成**纯函数**（字节进、结构出），于是能逐字节钉住：请求行必须正好三段、头值两侧的 OWS 要清掉、同名头取**第一个**（取最后一个等于让 `Content-Length` 变成可以被对端追加的头）、体**只**是 `Content-Length` 说的那么多字节（多出来的属于下一个请求）、`Content-Length` 数的是**字节**不是字符、头的结束只认 `CRLF CRLF`（认裸 LF 会把一个走私请求切成两个）。<br>⚠️ **传输层是 `flare.tcp` 的阻塞接口，线上格式是自研的**：`flare.http` 绑在它自己的 reactor 上（一次 `serve()` 按 `num_workers` 拉起线程），而这一版要的正是「一个进程、一条连接、阻塞」—— `srv/server.mojo` 的文件头写了这个取舍。要并发与背压时正确的动作是换成 `flare.http`，而不是把这里长大 |
| 单进程 HTTP 服务（非流式 + SSE 流式，真 socket、真 fork） | `verified` | `evidence:tests/unit/test_http_server.mojo`（重门，`pixi run test-http`，**故意不进 `pixi run test`**：它 fork，而 JIT 下 `fork` 会崩编译器）<br>钉的是**服务循环**的性质，纯函数单测钉不住：一条 keep-alive 连接上按序回答多个请求、且流水线里多出来的字节不丢；`Connection: close` **在响应里也写 close**（只在写完以后默默关，对端会往一个正在关闭的 socket 上发下一个请求 —— 那是一次看起来像「偶发连接重置」的失败）；一个发了一半的请求在接收超时后**只关它自己那条连接**、不能带走进程 —— 那是单进程形态最致命的失败方式。<br>⚠️ **这一行描述的是 `srv/server.mojo` 的旧形态**：阻塞 accept + 一次一条连接。自 P3.2（2026-09-22）起**线上入口已换成 reactor 循环**（见上面那行）；旧循环仍然在 —— 它是 reactor 门**常驻的负向对照**。无 TLS；并发与背压的现况见上面两行
| 发送队列与背压 | `verified` | `evidence:tests/unit/test_conn.mojo?count=15`<br>出站队列有**上界**（`SEND_HIGH_WATER`），写不出去就**停止读这条连接** —— 「背压」的意思不是「有队列」，而是队列满了要让上游停下来；只测「有字节排队」测不到这一点（队列可以无限长）。<br>一次 `send` 写一半是常态而不是错误：部分写按已写字节数**就地推进**，且**不许越过队列里有的**（越界的下标会把别人的内存发出去 —— 症状是一次「偶发的响应内容不对」）。要 `Connection: close` 的连接等队列**排空**再关（先关再排 = 对端收半个响应）。<br>⚠️ 边界：钉的是连接机（无 socket）；真 socket 上的取值与效果见 reactor 那一行 |
| SSE 流式输出 | `verified` | `evidence:tests/unit/test_sse.mojo?count=8`（帧格式，逐字节：头**不许**带 `Content-Length`（写第一帧时长度未知）、`CRLF CRLF` 恰好一处、**含裸换行的 JSON 必须被拒绝而不是被「修好」** —— 静默吃掉换行会把「内容里有个换行」变成「客户端少收一帧」，那是从外部查不出来的错）+ `evidence:tests/unit/test_http_server.mojo`（重门，真 socket：一条完整流的**线上字节** —— 头无长度、帧齐全、以 `data: [DONE]` 收尾、然后关连接；客户端中途断开只丢这一条连接、下一个请求照答）+ `evidence:tests/unit/test_workers.mojo`（多 worker：fork 出来的每个 worker 分帧与单进程一致，12 条连接每条一条完整流）。<br>定界方式：SSE 响应没有 `Content-Length`，靠**关连接**定界 —— 没有用 chunked（SSE 客户端按 `data:` 行切分，再套一层长度前缀只是多一处可以写错的地方）。<br>**每一条路都有终点**：`stream_begin` 的错（prompt 分词为空 / 过长）在响应头**之前**回 400（前向是**惰性**的，第一次 `stream_next` 才 prefill，校验才退得回去）；头发出去之后出错则给一帧 `error` 再 `[DONE]` —— 客户端得能区分「生成完了」与「生成坏了」，否则两种都表现为「收到若干帧然后连接关了」。客户端中途断开：写失败只丢这一条连接、进程继续（实测 `[srv] stream aborted: BrokenPipe` 之后下一个请求照答 ——这条同时是 SIGPIPE 的照妖镜，一个断开的客户端不该把 worker 带走）。<br>**文本与非流式逐字节相同**：每步用整段前缀解码再取新增的那一截，而不是把这个token 单独解码 —— 多字节字符可能跨两个 token，单独解码时前半截是不完整的 UTF-8，会被 `decode` 丢掉。实测（真权重、贪心）：英文与中文两种 prompt，流式拼出来的文本与非流式**逐字相等**。<br>官方 `openai` SDK（1.109.1，只改 `base_url`）流式对话跑通：14 帧 = 1 个 role 帧 + 12 个内容帧 + 1 个 finish 帧，`finish_reason` 与非流式一致（手动冒烟，不进门的证据）。<br>⚠️ 边界：同时能**流**的条数 = worker 数。P3.2 修的是**连接级**并发（连接不再互相挡路），但流共用一个 handler 状态、且生成仍串行 —— 要「同时生成 N 条」需要 engine 独占线程 + 批量调度，仍 `missing`；未验 `stream_options.include_usage`（未知键照旧跳过）、断线重连与代理缓冲；**没有任何性能数字进账本**（逐帧 `write_all`，~3 tok/s 量级下无瓶颈，未做批量合并） |
| OpenAI 兼容 API（非流式 + SSE 流式） | `verified` | `evidence:tests/unit/test_openai.mojo?count=30`：请求体的扫描、响应体的转义、路由（`/health`、404、405、`stream=true` → 流式，帧序列见下面 SSE 那一行）、`finish_reason` 的两条路。<br>**「半流式比明确拒绝更糟」这条纪律没变，只是换了个守它的地方**：要么从头到尾给完（以 `[DONE]` 收尾），要么在响应头之前就拒绝 —— 中途失败也必须给流一个终点，见下面 SSE 那一行。<br>JSON 是自扫的而没用 `json` 包：这里要的是**缺字段必须有名字**（`messages` 是空数组 → `invalid_request_error`，而不是「生成了空字符串」），而通用解析器会把缺字段变成运行期空值。转义必须还原成真字节（`u` 加四位十六进制 → UTF-8 字节），代理对与指数记法指名拒绝 —— 半个字符编不出 UTF-8，悄悄丢掉才是真的糟。<br>⚠️ **还没有 chat template**：`messages` 按出现顺序用换行拼起来，`role` 读掉但不参与；这件事随 `/health` 的 `notes` 一起返回，不读源码的人也看得见。⚠️ 边界：未验 `logprobs` / `tools` / 多候选 |
| SO_REUSEPORT 多 worker | `verified` | `evidence:tests/unit/test_workers.mojo`（重门，`pixi run test-workers`，fork → 先 build 再跑）：3 worker 各自`bind_reuseport` 同一端口，12 条连接 × 3 请求（健康检查 + 聊天 + 一条完整 SSE 流）全部 200 且分帧与单进程版**逐字段一致**；响应 id 从每个 worker 自己的 1 起计 →「chatcmpl-1 出现次数 = 接过活的 worker 数」（实测 3/3，内核四元组哈希分发）。⚠️ 权重在 fork **之前**加载 → 子进程 COW 共享：真权重 8 worker 冒烟（9 进程）PSS 合计 2874 MB ≈ 单进程 2874 MB（RSS 18.3 GB 是把 COW 页逐进程重复计入的记账假象）。⚠️ **fork 后 asyncrt 不可用**：worker 内分片 >1 时第一个前向**死等**到被 SIGKILL（shards=2/8 皆然）→ 多 worker 默认每 worker 单分片（`ALOFA_SHARDS` 可强行覆盖）；单进程不受影响 |
| master 健康探测 + 优雅退出 | `verified` | `evidence:tests/unit/test_workers.mojo`：master 不监听、只管进程（fork/回收/停止编排）；worker 把监听 fd `dup2` 到固定高位（900），SIGTERM/SIGINT 处理器只做一次 `close(2)`（异步信号安全表内、无锁无分配 —— Mojo 没有模块级可变全局，处理器带不了状态，状态由「fd 没了」本身承载），`serve_with_stop` 在 accept 前 / 每个响应后查标记。worker 现在跑的是 **reactor 循环**（不是阻塞 accept）：循环每轮（最多 50 ms 一次）都会回到标记检查处，标记自己就能让它退出；master 仍然发「唤醒连接」（连上即关；SO_REUSEPORT 按四元组哈希，一条只醒一个 → 发存活数 + 余量）—— 它们是给「等得更久」的形态留的兜底，多发几条没有代价（`pixi run test-workers` 仍绿：3 worker / 36 响应 / abnormal=0 forced=0）。宽限后仍不退的 SIGKILL 兜底，`forced`/`abnormal` 计数上报，任一非零 → master 非零退出（门会失败，不无声通过）。worker 运行期死亡 → master fail-fast 全停交给 systemd `Restart=` 接管（半套集群比快速重启难排障）。生产形态 systemd 模板 `scripts/deploy/alofa.service`（`TimeoutStopSec` 必须大于 `ALOFA_GRACE_MS`，否则 systemd 先 SIGKILL 整个组） |
| **批调度接进服务层**（一次前向推进多条流：贪心 + 采样 + 排队） | `verified` | `evidence:tests/unit/test_batch_stream.mojo?count=6`（重门，`pixi run test-batch-stream`，真权重、不并入 `test`）：批 N 条 vs 一条一条跑**逐 token 相等**；采样批 4 条 == 各自单跑；四条同 prompt 同种子的流必须互相一致（共用随机源必红）；10 条请求（> `MAX_BATCH`=8）不被拒绝而是**排队**且结果与单跑一致<br>`evidence:tests/unit/test_engine_core.mojo?count=18`（槽位用完要**显式** `release`：一整批跑完後第 9 条必须被接受）<br>⚠️ 边界：只收形状装得进引擎的请求（prompt ≤ 128、新 token ≤ 32），越界仍走单流老路；并发 ≤ 8 之外的请求在队列里等（上界 `MAX_CONNS`=128），不是拒绝 |
| 100 并发 1 小时压测 | `missing` | **P3 判据门**（100 并发长连接 **SSE** 1 小时，见 Gate P3）。已落地的是**分钟级、非流式**的压测观测（下一行）。**2026-09-22 补记**：并发数这一侧已经做到 100（真权重、单进程 reactor 循环、非流式、60 s）—— 0 connect_failed / 0 send_failed / 0 non_200 / 0 bad_body / 0 incomplete，fd 48→48（Δ0）。**仍然缺的是时长（1 小时）与 SSE**；另外 48 条 timeout 是**排到 120 s 还没轮到**的请求（生成串行，见 reactor 那一行），不是连接级失败。这轮压测抓出一个真 bug（空闲超时误收排队中的连接）并已修 |
| 部署配置（环境变量 → 运行配置） | `verified` | `evidence:tests/capability/test_deps.mojo?count=7`（`std.os.getenv` 契约：未设变量取默认值；编译产物里运行时读取已实测）。`srv/config.mojo` 全量变量：`ALOFA_PORT/HOST/WEIGHTS/TOKENIZER/WORKERS/SHARDS/MAX_TOKENS/MAX_PROMPT/RUN_SECONDS/GRACE_MS/MAX_REQUESTS/SEED/MODEL_NAME`，默认值与原编译期常量一字不差。为什么是环境变量：Mojo 1.0 的 `sys.argv` 在编译产物里是空的。⚠️ `ALOFA_MAX_TOKENS` 是**上下文窗口**（prompt + 生成的总槽位，`QwenForward` 拿它分配 KV），不是生成步数上限 —— 生成步数 = min(请求的 max_tokens, 窗口 − prompt) |
| 并发压测 harness（非流式 + **SSE 流式**，分钟级） | `verified` | `evidence:scripts/stress_serve.py`（`scripts/stress_serve.py`）（`pixi run stress`，纯客户端、只观测不参与判据）：多线程 keep-alive、QPS 与 p50/p90/p95/p99/max、错误**分类**计数（connect_failed/send_failed/timeout/non_200/bad_body/incomplete ——「零错误」是分类后的零）、服务端 fd / RSS / **PSS** 前后对比（RSS 把 COW 页逐进程重复计入，**PSS 才是证 fork 共享的正确口径**）、报告头强制带环境字段与「仅同轮相对比较」声明。⚠️ 本机长期过载（runq 6–27）→ 所有数字只能作同轮相对比较，同轮曲线：w1→w2→w4→w8 于并发 4 下 QPS 0.27→0.39→0.48→0.63，五轮全部零分类错误、fd 无泄漏、优雅退出零 forced。<br>**2026-09-22 同轮补充（单进程 + reactor 循环，并发 100、60 s、max_tokens=4）**：ok 103 / timeout 48 / 其余分类全 0 / fd Δ0；QPS 0.66，p50 19.2 s、p99 118 s。这一轮的结论不在 QPS，而在"100 条连接同时开着，没有一条被丢或被重置" —— 旧形态（一 worker 一连接）在这个并发数上根本连不上 |
| sidecar 回退形态（stdio RPC） | `missing` | 判据失败时的降级路径 |

## 9. Tokenizer

| 能力 | 状态 | 证据 |
|---|---|---|
| 预分词器（**无正则依赖**，手工实现 Qwen2 / GPT-2 规则） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=5`（4560 条参考用例逐 id 全等）<br>参考的两个项目死在这里；这里用**手写分支匹配器**替代正则引擎：收缩形式、字母串、数字、标点、换行、尾随空白六类按最左优先取最长，贪心与回溯由代码显式表达 |
| NFC 归一化（分解 / 规范排序 / 重组，含 Hangul 与多元分解） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=5`<br>**表是生成而非手写的**（`scripts/gen_unicode_tables.py`）；多元分解（如 U+1E5D）与组合类排序漏一条，整句就会落到不同的 merge 序列上 |
| BPE（**按 rank 合并**，同 rank 取最左） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=5`<br>贪心从左到右会挑错对；这里每轮全量扫描取 rank 最小者 |
| added token 切分（**最长匹配**） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=5`（语料含句中出现的 added token） |
| 词表 / 合并表加载（**离线 fixture，不启 Python**） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=5`<br>fixture 由 `scripts/dump_reference.py` 生成，但测试进程只读 TSV —— 差分门可在任何机器上重跑 |
| id → 文本还原（含非法 UTF-8 替换） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=5`（往返用例逐条对比**归一化后**的输入） |
| 4560 条差分用例（与 HF 逐 id 一致 + 往返） | `verified` | `evidence:tests/unit/test_tokenizer_parity.mojo?count=5`（0 处不一致）<br>**P1 门**：对不一致**零容忍**（0/4560），而不是容忍 0.1% —— 差分断言的是字节级等价 |
| Unigram | `missing` | — |
| WordPiece | `missing` | — |
| `tokenizer.json` 加载（**直接读 HF 文件，不经转换**） | `verified` | `evidence:tests/unit/test_tokenizer_json.mojo?count=8`（小真夹具：HF `tokenizers` 自己编码的 15 条 + 256 条逐字节用例 0 处不一致，外加 5 条拒绝路径）<br>另见 `tests/unit/test_tokenizer_parity.mojo?count=5`：**真实的 Qwen2.5-0.5B `tokenizer.json`（7.0 MB，原文未改）** 重编码同一份 4560 条语料 0 处不一致；并与 TSV 加载器**逐 token 比对 151665 条全等** —— 两个独立读取器对同一个分词器给出同样的字节。<br>**只信文件里能核对的东西**：`model.vocab` / `model.merges` / `added_tokens` 是读的，其余是**查的** —— `model.type` 非 BPE、`pre_tokenizer` 的 regex 不是 `pretokenize.mojo` 实现的那条、`normalizer` 非 NFC、`decoder` 非 ByteLevel、`add_prefix_space=true`、id 不稠密、JSON 截断，全部**指名拒绝**（`unsupported_tokenizer` / `bad_vocab` / `bad_json`）而不是近似：预分词器是给一条 regex 手写的匹配器，照另一条跑会给出一串看起来合理、含义不同的 id。<br>**撞出来的两个真实行为**（都不是猜的，是参考实现给出的答案）：① **added token 按字符串去重** —— 已存在的词表 token 保留词表 id（added token `é` 编码成 233，而不是文件写的 263），且它**仍按 added token 切出**（`xéy` → `[120,233,121]`，而非字节级的 `[120,195,169,121]`），所以比对必须在**字符串**层做，不能在字节层做；② 切分顺序是**先切 added token、再 NFC**：因此已合成的 `é` 走 added token（233），而 `e+U+0301` 走 NFC 合成后的字节级（195,169）—— 同一字符两条路。<br>⚠️ 边界：只验 **BPE + ByteLevel + NFC** 这一族；`tokenizers` ≥ 0.20 的 merges **pair 写法** `["A","B"]` 已支持，但本机装的 0.19.1 **读不了**这种文件，所以该形状的判据是"与字符串写法逐 id 相同"而非参考实现重跑。<br>⚠️ 边界：`tests/fixtures/qwen2.5-0.5b/tokenizer.json` 是 **7.0 MB 的原始 HF 文件**（故意入库，门要跑就得有它）；不验 WordPiece / Unigram / SentencePiece，不验 `chat_template` |
| GGUF vocab 加载 | `missing` | — |
| `chat_template` 渲染（Qwen2.5 ChatML 一族） | `partial` | `evidence:tests/unit/test_chat_template.mojo?count=6`（10 个词条，**逐字节比文本 + 逐个比 token id**）+ `evidence:tests/unit/test_openai.mojo?count=30`（接线那一侧：缺 `role` / 缺 `content` / 未知角色都得拒绝，同一段文字换个角色必须给出不同的 prompt）<br>原先按出现顺序用 `\n` 拼 content、`role` 读掉不参与，多轮与系统提示的语义是错的；现在由 `alofa.tokenizer.chat_template.render_chatml` 按模板排：先吐一段 system（无首条 `system` 时用模板自带那句），随后 `user` / `assistant` 各一段，最后补 `<|im_start|>assistant\n`。首条 system **不会**在循环里重复出现（`loop.first` 那条，用出现次数钉住：3 个 `<|im_start|>` / 2 个 `<|im_end|>`）<br>**答案不属于本仓库**：模板来自 HF 缓存里 `Qwen2.5-0.5B-Instruct/tokenizer_config.json:chat_template`，由 `scripts/dump_chat_template.py` 用 **transformers 自己的 Jinja 引擎**导出成夹具 `tests/fixtures/qwen2.5-0.5b/chat_template.tsv`（自检：`tok(text) == ids`，不过就不写夹具）。所以模板改一句话，这道门立刻红<br>⚠️ **它现在是 `partial` 而不是 `verified` 的理由（重要）**：这里是**一族的渲染器，不是模板引擎** —— 正确做法（「语义来自模型的 `chat_template`」，借鉴 `molla`）要求模板字符串来自模型，而本仓库夹具的 `tokenizer.json` **不带** `chat_template` 字段，`tokenizer_json.mojo` 遇到它也是静默跳过。于是**换族群要改代码，而不是改数据**。<br>⚠️ 明确未做：`tools` / `tool_calls` 与 `tool` 角色（模板里走 `<tool_response>` 那一段）；`role` 只认 system / user / assistant，**其余指名拒绝**（`invalid_request_error`）—— 拼出去的话不会报错，只会让模型把一段工具输出当成人话读 |

## 10. 验证体系（正交支柱）

| 能力 | 状态 | 证据 |
|---|---|---|
| oracle 差分框架（logits 余弦 / argmax） | `verified` | `evidence:tests/unit/test_model_parity.mojo`（余弦在 `Float64` 中累加：151936 维的 fp32 累加误差与被测间隙同量级时，相似度本身就不可信）<br>参照物是**导出的 fixture 而非实时调 Python** —— 被测代码变了，答案不会跟着变 |
| greedy 逐 token 相等校验 | `verified` | `evidence:tests/unit/test_model_parity.mojo`（4 条 prompt × 128 token，逐 token 相等）<br>这是三条判据里最强的一条：能通过余弦却在第 30 步分叉的实现，在这里一定失败 |
| 逐算子中间张量差分（定位用） | `verified` | `evidence:tests/unit/test_layer0_parity.mojo?count=13`<br>每个算子各自持有"参考实际看到的输入"与"参考实际产出的输出" → 失败时**只有该算子的测试红**，而不必在整网里二分 |
| top-k 集合比较 / 分布检验（卡方 / TVD） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo?count=11` 存活集合用 **FNV-1a 指纹做零容差比较**（只比数量会放过"对的个数、错的成员"）<br>并列取值的合成行**只比集合不比顺序**：`torch.sort` 在并列值上顺序未定义，逐元素比会 flaky；而真实 logits 几乎不并列 → 把 `>=` 写成 `>` 在真实数据上测不出来，在**量化后**一定会并列。该门已用变异测试验证过会红（改一个比较符 → 4 个测试失败） |
| 分布检验（卡方 / TVD） | `verified` | `evidence:tests/unit/test_sampler_parity.mojo?count=11`（序列固定 → 断言确定性，**不会 flaky**；TVD / 卡方同时与参考值比对，不仅与阈值比对） |
| roofline **骨架**（采集 + 利用率报告接口，峰值由调用方传入） | `verified` | `evidence:tests/unit/test_verify_roofline.mojo?count=15`<br>**不内置任何机型常数**：峰值是构造参数，≤0 直接报错 → 杜绝"抄规格书当实测" |
| roofline 报告（真实 kernel / 真实硬件） | `partial` | **已接进三档真实形状**（`scripts/bench_decode_roofline.mojo`，`-O2`，batch-1 decode：`hidden_proj` 896×896、`lm_head` 151936×896、`down_proj` 4864×896；fp32 与 q4_0 两种表示 × avx2 / scalar / f64-8w 三条通路），判定由 `Roofline.bottleneck` 的**交叉相乘**给出，不依赖任何容差<br>**能引用的数是「自己的访存地板 / 实测」这个同进程配对比值**（把这块权重原样纯读一遍 ÷ 实际跑完，1.0 = 算术完全被访存盖住；三次运行的 min…max，区间都紧）：`hidden_proj` **0.97–1.05**、`down_proj` fp32/avx2 **0.85–1.01**、fp32/scalar 0.37–0.48、**q4_0/avx2 0.24–0.29**、q4_0/f64-8w 0.15–0.22、q4_0/scalar 0.05–0.07<br>⇒ **fp32 decode 是带宽受限**（≈1.0，实测 6.64–7.15 GB/s，落在本机读带宽 7.02–8.79 GB/s 的下沿）；**q4_0 decode 不是**（0.24–0.29：实测只有 1.87–2.00 GB/s，却拿到 6.65–7.12 GFLOP/s ≈ 核峰值 5.42–6.80 的上沿）→ **q4 通路的下一个杠杆在算术/指令，不在访存**，与 2026-09-22 行复用（同样是改算术）拿到 2.38× 是同一个方向<br>⚠️ **利用率（‰ of peak）本机量不出结论**：三次运行里峰值探针自身就在漂（读带宽 7.02–8.79 GB/s、核峰值 5.42–6.80 GFLOP/s、机器峰值 51.32–66.23 GFLOP/s），于是利用率区间 `hidden_proj` **572–1446‰**、`lm_head` **572–840‰** —— 跨过 1000‰（比「峰值」还快，物理上不可能）。按本账本一贯口径，**跨阈值的区间写「量不出结论」，不挑一轮报数**<br>⚠️ **llama.cpp 同机对比仍 `missing` 且本次无法推进**：`github.com` / `huggingface.co` 均超时（无网络），取不到 llama.cpp 与 GGUF |
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
- **2026-09-20** —— **两处测量工具的完整性修复 + 一处跨机比较作废（自查发现，都没产新性能数）**：
  - ⚠️ **作废**：上一条里的"对 llama.cpp `-t 8` = 0.41–0.43×"是**跨机器**的 —— llama.cpp 那组数（5.94 / 25.18 tok/s）量自 **A100 远程机 CPU**（变更日志 2026-09-18），端到端数量自**本机**。已在本行与 §5 那行改写；**本机至今没有 llama.cpp 基准**，250 行对本机仍 `missing`。
  - `scripts/bench_thread_matmul.mojo` 的三处修复（**它报的核级数因此全部待重测**）：① 分片协程原来收 `TensorView` 作参数 —— 这正是账本里已记的"会静默写错地方"那个坑，改成只收指针 / 整数、视图在协程内现造；② 两条通路都加**哨兵 + 与 T=1 精确相等**的自检，没过就打印作废提示；③ **q4 那份只有 8 份 × 2.34 MB = 19 MB，比 L3（12 MB）大不到 2 倍**，后几趟有一部分落在 L3 里 → q4 区域数改 64 份 = 149 MB（复制同一份内容到不同地址即可，L3 看的是物理地址）。
  - 为什么判它"待重测"：它原来报的 `fp32/avx2 T=8` 折算 **36 GB/s**，而同进程现测的纯读天花板只有 **22 GB/s** —— 一个 GEMV 不可能比纯读还快。修完这一轮（机器很吵）fp32 T=1 是 8.7 GB/s、T=8 最好 17 GB/s，**没有越过天花板**，但同轮内 T=2 从 0.83 抖到 2.05 ms（2.5×），**这一轮的数不许进账本**，等安静的机器重测。
  - `scripts/bench_model_shards.mojo` 末段新增"离天花板还有多远"：同一进程现测 4 个天花板（纯读 B_1/B_8、真实 kernel K_1/K_8），**每轮都测、与 token 时间同一时间窗** —— 先跑完 3 轮再测天花板的话，机器变慢会让天花板偏低、利用率虚高（实测算出过 >1000‰ 这种不可能的数）。天花板同样带自检：不自检时它报出过 **56.56 GB/s**，而本机 DDR4 双通道理论上限只有 ~42 GB/s。
  - 结论（能说的部分）：**串行段 + 调度 ≤ ~25%，算术上更可能 ≈0**（非投影部分只占全 token FLOPs 的 ~0.2%）→ **不切注意力 / norms**；kernel 已达纯读天花板的 ~90% → **访存模式也没什么油水**。下一个有量级的杠杆不在"怎么算"，而在**每 token 要读多少字节**。
- **2026-09-20** —— **批 / 预填的 GEMM 换循环次序：权重不再每行各流一遍（端到端 prefill `n=32` 每 token 2.68–3.66×）**：
  - 起点是上一条的结论：下一个杠杆在**每 token 读多少字节**。查下去发现 `avx2._gemm` 是「行外层、列内层」—— `w` 的每一列在每一行上都被重读一遍，于是 `rows` 行的前向把整个权重矩阵读 `rows` 遍：**批 = 8 的 decode 和 32 token 的 prefill，摊到每个 token 上的字节数和批 = 1 是同一个数**（fp32 每 token 1.976 GB）。账本里那条「算术强度恒 0.5 flops/byte」不是 GEMM 天生如此，是**这个循环次序**造成的。
  - 先做判定性探针 `scripts/bench_gemm_rows.mojo`（新增，`-O2`，`down_proj` 896×4864，8 份互不相交权重 139 MB ≫ L3 12 MB，每份算一趟）：换序前 `rows=8` 一趟 **23.31 ms**，每行耗时 3.95 / 3.06 / 3.94 / 2.91 ms —— **不随 `rows` 摊薄**，且「若每行各流一遍权重」的隐含带宽跨 `rows` 恒定（4.40–6.02 GB/s）→ 机制成立，这是个真杠杆。
  - 改法：行按 8/4/2/1 分块交给 `_gemm_tile[RB]`，块内改成「列外层 / `k` 中层 / 行内层」，`w[col][:]` 每块**只读一遍**。`RB` 上限取 8 是寄存器数的限制（每行一个 f64×4 累加器占一个 ymm），不是调出来的。
  - 判据是**逐位相等**不用容差：换序不改变任何一次浮点运算 —— 每个输出各自一个 f64×4 累加器、按同样的 `k` 次序累加，连 `inner % 4` 的尾巴初值都必须是 `reduce_add()` 的结果（`0+t0+t1` 与 `r+t0+t1` 是两个数）。门 `tests/unit/test_gemm_batch_layout.mojo`（5/5，已登记进 `scripts/run_tests.sh`）：`rows` 取 1–9（`RB` = 8/4/2/1 四种分块连同余数块全走到），`inner` 既有 4 的倍数（真形状 896 / 4864 都是）也有 7 / 13 / 17 —— **真形状里 `k` 的标量尾巴永远走不到**，不单独造形状就守不住它。
  - ⚠️ **参照物分两类，各有各的盲区**（实测踩到）：前两条的参照物是"逐行调用**同一个 avx2 核**"，只能守住「换序不改变结果」—— 注入「尾巴初值少了 `reduce_add()`」这处两边共用的错时，那两条**仍然 PASS**。故另挂第三条：与**没动过**的标量后端比（绝对参照物）。另注入「所有行写第 0 行的结果」，由前两条接住 —— 两处注入各由对应的一条挡下，门不是"反正绿"。
  - 核级：换序后 `rows=8` 一趟 **23.31 → 3.51 ms**，每 token 2.75 → **0.43 ms（6.4×）**。
  - ⚠️ **不报"换序前后 rows=1"的对比**：两次运行的同进程纯读带宽是 7.53 与 10.85 GB/s（机器状态不同），跨运行不可比。rows=1 走 `_gemm_tile[1]`，操作序列与老写法一致；四道重门 `test-model` 4/4、`test-model-avx2` 4/4、`test-slice` 4/4、`test-batch-forward` 6/6 **全绿**。
  - 端到端 `scripts/bench_prefill.mojo`（新增，`-O2`，Qwen2.5-0.5B fp32 / avx2，3 趟 min…max，预热一趟丢弃，新旧二进制**交替各跑 3 次**）：`n=32` 每 token **35.65–39.77 → 8.25–13.29 ms**（旧 min 30.28 > 新 max 13.29 → **区间不重合，2.68–3.66×**，三次运行复现）；`n=8` **27.2–36.1 对 27.3–30.1 ms**，区间**重合** → **量不出差别**。端到端倍数小于核级那个，是因为注意力 / RMSNorm / RoPE / 残差不吃这份红利。
  - ⚠️ `n=8` 没差别不是"换序没生效"，是**分片策略**：`prefill` 按**输出行**分片，8 行 ÷ 8 片 = 每片 1 行 → `RB=1`，块里没有第二行可复用，换序只在同一块内摊薄。→ **下一步**：`prefill` 的片数要跟着 `rows` 走（片内行数 ≥ 2），否则短 prompt 一分钱红利都拿不到。
  - ⚠️ **只改了 `avx2._gemm`**：`scalar._gemm` 与 q4 通路（`_matmul_q4`）**没动**，批 > 1 时它们仍每行各流一遍权重。
- **2026-09-20** —— **prefill 的片数跟着复用走：默认 8 片改成量过的 4 片（端到端 1.07–2.50×）**：
  - 起因：上一轮把 `avx2._gemm` 换循环次序之后，批/预填的权重复用**只在 `_gemm_tile[RB]` 的块内**发生；而 `prefill` 是按**输出行**切片的，`rows ÷ shards` 决定每片行数 —— 于是 `n=8` 配 8 片时每片只剩 1 行、`RB=1`、复用为零，这就是上一条里「n=8 量不出差别」的真因。片数在这里是一个**真旋钮**，且两个方向相反：片数少 → 每片行数多 → 复用充分；片数多 → 线程并行度高。
  - 先量：把 `scripts/bench_prefill.mojo` 改成**片数 × prompt 长度**扫描（n = 8/16/32 × 片数 1/2/4/8，每 token 时间，3 趟 min…max，轮外层片数内层交错，**三次运行**）。结果 **4 片在 n=16/32 上最好，且与 8 片区间不重合**（n=32：10.4–16.0 对 14.2–20.7，最紧的一对是 13.37 < 14.15）；1 片最差（并发仍然要）。→ 这个 8 核机器上**复用比并行度更值钱**。
  - 落地 `prefill_shards()`：**量过的行数内**（`rows ≤ PREFILL_ROWS_MEASURED = 32`）压到 `PREFILL_SHARDS_MEASURED = 4`，**超出就原样返回** —— 没量过就不假装量过，与 `SHARDS_MEASURED` 是同一条规矩。
  - 自检：**片数不同 → prefill 必须逐位相等**（前向与怎么切无关：每个输出各自一个累加器，切分只决定「谁算哪几行」）。`test_parallel_shards.mojo` **8/8**：新增策略档，并把 prefill 的行数从 7 扩到 1/2/3/5/7/8/9/13/32/33（8/9/13 跨 `_gemm_tile[RB]` 的分块边界，**33 专门跨「未量过」那条分支**）；注入「不压片数」→ 策略档确实变红。
  - 端到端 A/B（**默认片数**，新旧二进制交替 **5 次**）：n=8 22.5–35.7 对 56.6–108.7、n=16 13.2–24.7 对 23.8–54.6、n=32 9.7–13.9 对 12.4–35.4 ms/token → **后 4 次区间不重合，1.07–2.50×**。⚠️ **第 1 次反常**（改后 n=32 25.9–26.7 反而慢于改前 12.4–14.3）：那次两个二进制刚构建完、2 GB 权重页缓存是冷的，片数少 → 缺页由更少的线程服务。冷启动那一轮不作数，但**必须写出来**（已写入账本该行），只留好看的那几轮就是把结论往自己那边掰。
  - 四道重门全绿：`test-model` 4/4、`test-model-avx2` 4/4、`test-slice` 4/4、`test-batch-forward` 6/6；`check-ledger` 7/7、`check-counts` 9/9。
  - ⚠️ `rows > 32` **没量过**；q4 通路的 prefill 是**逐行 matvec**（每片 `rows = 1`、按输出列切），不受这条影响。下一步：量更长的 prompt（`rows > 32`）到底该几片。
- **2026-09-20** —— **prefill 片数的换手点量出来了：48（不是 32）**：
  - 上一轮只量到 `n = 8/16/32` 就把 `PREFILL_ROWS_MEASURED` 定成 32，`n > 32` 一律退回调用方的 8 片。这一轮把扫描扩到 **n = 32/48/64/96/128 × 片数 1/2/4/8**（`max_tokens` 就是 128，这是 prefill 能到的上限）。
  - 先验：权重被读的遍数 = 片数 × ceil(每片行数 / 8) → n ≥ 48 时四档**遍数拉平**（都是 8 遍），所以换手点该在 32 与 64 之间。量出来确实是 **48**。
  - **n = 48：4 片确定更好**（2/3 次区间不重合，第三次同向、只差 0.04 ms 没分开）→ `PREFILL_ROWS_MEASURED` 32 → **48**。**n = 64/96/128 量不出结论**（区间全重合；只有 n=64 一次 8 片不重合地更好，不跨运行复现）→ 按口径**不改**，仍是调用方给的片数。
  - ⚠️ 换手点**随机器负载挪**：n=32 在负载低时三次只分开一次，在负载高时（上一轮）三次分开两次 —— 8 片在机器忙时掉得更多。这也是「4 片从来没被量成更差」的由来。
  - ⚠️ **「默认片数」那一行不作数**：它量在进程刚起来、2 GB 权重刚 mmap 完的时候，同一条代码路径（n=32 两边都是 4 片）能差出 8.76 对 16.01 ms/token（±70%）。交错扫描才是证据；那一行只用来确认改动接上了线（n=48：改后 9.35–10.44 对改前 9.42–26.20）。
  - 门：`test_parallel_shards.mojo` **8/8**（策略档的边界断言 33→**49**，prefill 行数扩到 …/48/49 —— 48 与 49 一对专门跨新的换手点）；注入「上界退回 32」→ 策略档变红。自检：五种长度下四种片数**逐位相等**。重门 `test-model` 4/4、`test-model-avx2` 4/4、`test-slice` 4/4、`test-batch-forward` 6/6；`check-ledger` 7/7、`check-counts` 9/9。
  - 下一步：n ≥ 64 **量不出差别本身就说明片数不再是杠杆**（遍数已拉平、4 片就够打满带宽）→ 该转向 ⑥ q4 解量化算术（上限 ~1.9×）。

- **2026-09-20** —— **scheduler 极端容量活锁修复**：根因是 admission 只检查单条 prompt 的静态块数，未检查并发序列的完整稳态足迹是否超过水位；现在首请求也执行 `committed + whole <= threshold_blocks()`，超限立即报具名 `capacity`，不进入必然「抢占归零 → 重喂」的循环。
  - 同一拍多个 arrival 改为原子 admission：后到请求失败时回滚本拍已经成功接纳的 request；`EngineCore.prepare()` 捕获 admission 失败后清理已经排队但未被 scheduler 接纳的 slot，避免 `has_work()` 对不可能请求无限重试。
  - 新增回归门：scheduler `test_a_single_request_over_watermark_fails_before_livelock`、`test_arrival_admission_is_atomic_on_a_late_rejection`；engine `test_a_watermark_rejection_does_not_leave_a_pending_request`。scheduler **19/19**、engine **17/17**；`pixi run test`、`check-ledger`、`check-counts`、`test-model` 均通过。
  - 判定边界：这次修复的是「并发足迹自身超过水位」；缓存 + 并发足迹超过水位的路径仍由既有 reclaim/抢占路径处理，不把更大的边界声称成已解决。
- **2026-09-20** —— **`tokenizer.json` 由 `missing` 升为 `verified`：直接读 Hugging Face 的文件，不再先转成自家 TSV**（`src/alofa/tokenizer/tokenizer_json.mojo` 新增）：从此一个 checkpoint 丢进来即可用，不需要转换步骤。
  - 证据是两条：**真实的 Qwen2.5-0.5B `tokenizer.json`（7.0 MB 原文入库）** 重编码既有的 4560 条差分语料 → **0 处不一致**；并与原 TSV 加载器**逐 token 比对 151665 条全等**（两个独立读取器、同一个分词器、同样的字节）。另有 `tests/unit/test_tokenizer_json.mojo`（**8/8**）：小真夹具的 15 条用例 + 256 条逐字节用例，id 由 HF `tokenizers` 自己编码产出；5 条拒绝路径（非 BPE / 异 regex / `add_prefix_space` / id 不稠密 / JSON 截断）必须指名报错。
  - **只信核对过的东西**：`model.vocab`、`model.merges`、`added_tokens` 是读的，其余是查的 —— `model.type`、`pre_tokenizer` 的 regex（与 `pretokenize.mojo` 实现的那条逐字节比）、`normalizer`、`decoder`、`add_prefix_space` 任一不符即 `unsupported_tokenizer`。预分词器是给**一条** regex 手写的匹配器，照另一条跑会给出一串看起来合理、含义不同的 id，所以拒绝是唯一诚实的回答。
  - **撞出来的两个真实行为**（都不是猜的，是参考实现给出的答案）：① added token 与已存在的词表 token **按字符串去重**（`é` 编码成 233 而非文件写的 263），且**仍按 added token 切出**（`xéy` → `[120,233,121]`，不是字节级的 `[120,195,169,121]`）—— 所以比对必须在字符串层做；② 顺序是**先切 added token、再 NFC**，于是已合成的 `é` 与 `e+U+0301` 走两条不同的路（233 vs 195,169）。
  - 边界：`tokenizers` ≥ 0.20 的 merges **pair 写法** `["A","B"]` 已支持，但本机 0.19.1 读不了这种文件，该形状的判据是"与字符串写法逐 id 相同"；只验 BPE + ByteLevel + NFC 一族，不验 WordPiece / Unigram / `chat_template`。

- **2026-09-21** —— **真实 Qwen 模型目录端到端：一个目录（`config.json` + `model.safetensors` + `tokenizer.json`）直接跑通，并与同一份 fp32 参考对齐**（`tests/unit/test_real_model_dir.mojo` 新增，6/6；入口 `pixi run test-model-dir`，重资产门，故意不进 `pixi run test`）：
  - **为什么另开一门**：既有模型门读的是**自家导出**（TSV 索引 + fp32 裸二进制 + 预先写好的 rope 表 + 写出来的 `lm_head.weight`），而用户手里的目录三处都不一样 —— 参数全是 bf16、没有 `lm_head`、没有 rope 表。读得懂自家导出 ≠ 读得懂 checkpoint，这一门专门守后者。
  - **三处接缝的落地**：① `TensorFile` 在加载时把 BF16 **就地放宽为 fp32**（bf16 就是 fp32 的高 16 位，放宽不带误差），结果由它自己持有的 arena 保存 —— mmap 是只读的，改不动；② 输出投影**按名解析**：有 `lm_head.weight` 就用它，没有就在 `tie_word_embeddings` 下用 `model.embed_tokens.weight`，两者都不是时报具名错；③ `rope_cos/rope_sin` 缺失时按 `rope_theta` 现算 `[max_tokens, head_dim]`，有则照读 —— 两条路都在用（自家导出走照读，真目录走现算）。
  - **判据沿用既有口径，没有为它放宽**：4 条 prompt 的末位 logits 余弦 ≥ 0.999 且 argmax 相等；**16 token 教师强制 greedy 逐 token 相等**；真分词器解出的 id 与参考 prompt 的 id **全等**（否则后面都在跟另一道题的参考答案对）；现算 rope 表 vs 参考导出的那份 **5.96e-08 = 1 个 fp32 ulp（256×64）**，门开在 1e-6。参考仍是导出的 fixture（Hugging Face 以 fp32 跑同一 checkpoint 的答案），不是实时调 Python。
  - **门的自我检查**：bf16 放宽那两条把同一组数以 F32 / BF16 各存一份，放宽回来必须**逐位相等**；把位移注入成 `<< 15` 后两条**立刻变红**（注入已还原）。FP16 仍按名拒绝，并单列一条负向对照 —— 实现的是 bf16 的放宽，把 fp16 当 bf16 读是静默损坏。
  - **顺带修掉两个「只有读真文件才会撞上」的 bug**：safetensors 的 `__metadata__` **只跳过了名字、没跳过对象**，于是 `{"format":"pt"}` 里的 `format` 被当成张量解析 → 真文件直接报 `unterminated safetensors tensor metadata`；`QwenConfig` 要求 `head_dim`，而真实 Qwen2.5 的 `config.json` **没有这个字段**（HF 运行时自己推导）→ 改为「有就读、没有就 `hidden // n_heads` 派生」，除不尽由 `validate` 兜住。
  - **登记与边界**：`tests/unit/test_model_formats.mojo` 此前**不在 `scripts/run_tests.sh` 的清单里**（门写了却没人跑），已登记 → `pixi run test` 现在跑到它（8/8）；`.gitignore` 给 `tests/fixtures/model-formats/*.safetensors` 开了例外（几十字节的夹具，CI 依赖），否则新门在干净克隆上会因缺 fixture 失败。真目录里 `model.safetensors` 是**符号链接**，本机有 HF 缓存才能跑 —— 与 `test-model` 依赖本地 1.9 GB 导出同级。
- **2026-09-21** —— **单进程非流式 HTTP：一个按 OpenAI 形状说话的服务起来了**（`pixi run serve`，绑 127.0.0.1:8000）。`src/alofa/srv/` 三个文件分工：`http.mojo` 是 HTTP/1.1 的**纯函数**线格式（字节进、结构出），`openai.mojo` 是`/v1/chat/completions` 的**非流式**请求/响应与路由，`server.mojo` 是「一个进程、一条连接、阻塞」的服务循环；入口 `src/alofa/serve.mojo`（权重只加载一次，每个请求先 `reset()` —— KV 是**这一条流**的状态）。
  - **传输层用 `flare`，线上格式自研**：`flare.tcp` 的阻塞 `TcpListener` / `TcpStream` 管 socket / accept / TCP_NODELAY（A7：依赖而不自研）；而 `flare.http` 绑在它自己的 reactor 上（一次 `serve()` 按 `num_workers` 拉起线程），这一版要的恰恰是「一个进程、一条连接、阻塞」—— 先在单进程阻塞形态上把并发、背压、SSE 一起加进来，等于同时调试四件事，而且每件坏了都表现为「卡住了」。并发与背压是 P3.3 换 `flare.http` 的理由，不是在这里长大的理由。
  - **`stream=true` 是 400，不是「也支持一下」**：半流式（先答应下来再吐一半）比明确拒绝更糟 —— 客户端会一直等一个不会来的 `data: [DONE]`。
  - **三个护栏都是因为「单进程」才这么致命**：接收超时（一个连上又不发数据的对端在单进程里等于让服务停摆）、头/体的上限、以及流水线里多出来的字节必须留到下一个请求（吃掉它，第二个请求就凭空消失）。端到端门逐条验：一条 keep-alive 连接上三个请求按序回答、`Connection: close` **在响应里也写 close**、发了一半的请求只关它自己那条连接而**服务还活着**（第 5 个请求照答）。
  - **撞出来并修掉的一个真 bug**：`Connection: close` 最初只在写完以后默默关socket，响应里仍写 `keep-alive` —— 对端会往一个正在关闭的连接上发下一个请求。端到端门就是靠这句断言抓到它的（纯函数单测看不到，因为响应对象是路由造的）。
  - **边界（写进 §8，不在 README 里提）**：一次只处理一条连接 —— 一个慢请求会让所有别的请求等着；无并发、无背压、无 TLS；**没有 chat template**（`messages` 按序用换行拼起来，这件事随 `/health` 的 `notes` 返回）；`created` 恒为 0，因为 Mojo 1.0 的 stdlib 里没有墙上时钟，填 0 而不是编一个。
  - **真实权重手动冒烟（不是门，不进本条的 `verified` 证据）**：`pixi run serve` 起来后 `/health` 返回 `"streaming":false` 与 notes；`{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8}` 得到 `" Paris. The United Nations headquarters is in"`、`finish_reason":"length"`，**连续第二次请求逐字相同**（每请求 `reset()` 生效 —— 不重置的话第二个请求会把第一个的 KV 当前缀，`prefill` 会报 `kv_len != 0` 而 `step` 不会，它会安静地把两段缝在一起）；`stream=true` → 400，未知路径 → 404。⚠️ 这次只证明**接线与正确性**，**没有测速**：本机长期过载，`curl` 的墙钟不能当吞吐数字用，故本条不附任何 tok/s。
  - **撞出来并修掉**：`QwenForward` 的 `params_dir` 要的是**权重文件**自己（它交给 `TensorFile` 去认格式），传目录会去找 `tensors.tsv` 然后报打不开 —— 入口原先传的是目录，**编译能过、一跑才炸**。

- **2026-09-21** —— **多 worker、部署配置与压测观测（P3.4 / P3.5）**：`ALOFA_WORKERS=N pixi run serve` 起 N 个 worker，各自 `bind(SO_REUSEPORT)` 同一端口、内核按四元组分发；master 不监听、只管进程。配置全部从环境变量读（`srv/config.mojo`），默认值与原编译期常量一字不差。
  - **权重共享是「fork 前加载」给的，PSS 实证**：真权重 8 worker（9 进程）PSS 合计 2874 MB ≈ 单进程 2874 MB —— RSS 18.3 GB 是把 COW 共享页逐进程重复计入的记账假象。前提是父进程 fork 前没跑过前向（KV/激活写时复制、各自私有）。
  - **撞出来并绕开：fork 后 asyncrt 死锁**：多 worker 下 worker 内分片 >1 时第一个前向**死等**到被 SIGKILL（shards=2/8 皆然，master 的 forced 计数与 fail-fast 非零退出正确触发）—— 父进程加载时 `default_shards()` 碰过并发运行时，子进程继承的线程池状态是坏的。绕法：多 worker 默认每 worker 单分片（`ALOFA_SHARDS` 可强行覆盖，后果自负）；单进程不受影响。**已知缺口**，没有悄悄跳过。
  - **优雅退出的记号是「一个被关掉的 fd」而不是管道**：Mojo 没有模块级可变全局，信号处理器带不了状态；`read`/`write` 又与 stdlib 撞符号 lowering（flare 为此专门绕道 dlopen）。worker 把监听 fd `dup2` 到固定高位 900，SIGTERM 处理器只做 `close(2)`，服务循环轮询「fd 还开着吗」；阻塞在 accept 的空闲 worker 靠 master 发「唤醒连接」叫醒，宽限后 SIGKILL 兜底。
  - **门先于硬做**：`pixi run test-workers`（3 worker × 12 连接 × 2 请求，分帧逐字段与单进程一致；master 退出码携带 abnormal×10+forced）—— asyncrt 死锁正是被这道门的「退出码非零 → 门失败」路径逮住的，SIGKILL 兜底没有无声通过。
  - **压测口径与 Gate P3 的差距显式记账**：已做的是非流式、分钟级、并发 ≤8（`scripts/stress_serve.py`：w1→w2→w4→w8 于并发 4 下 QPS 0.27→0.39→0.48→0.63，五轮零分类错误、fd 无泄漏）；Gate P3 要的是 100 并发 SSE 1 小时 —— 差的是 SSE（仍 `missing`），且本机过载数字只作同轮相对比较。
  - **修掉的静默 bug**：`ALOFA_MAX_TOKENS` 曾被当生成步数传给 `QwenForward`，而它其实是**上下文窗口**（KV 槽位）—— 症状是 `kv_len=9 max=8` 越界报错；现在生成步数 = min(请求 max_tokens, 窗口 − prompt)。
  - **未做的优化及原因（记账不硬做）**：每请求 `Sampler` 构造复用（worker 串行，分配 ≪ 前向，效应在本机 ±10% 噪声以下）、worker 内分片（asyncrt-fork 死锁，见上）。

- **2026-09-21** —— **SSE 流式：`stream=true` 从 400 变成逐 token 的真流式（`src/alofa/srv/sse.mojo` 新增）**：`/v1/chat/completions` 在 `stream=true` 时先回一个 `text/event-stream` 的头（**没有 `Content-Length`** —— 长度在写的那一刻未知），然后逐 token 吐 `chat.completion.chunk` 帧，以 `data: [DONE]` 收尾并关连接。关连接就是这条响应**唯一的定界方式**（没有用 chunked：SSE 客户端按 `data:` 行切分，再套一层长度前缀只是多一处可以写错的地方）。
  - **分帧是纯函数，I/O 只在服务循环**：`srv/sse.mojo` 只做「一段 JSON → 一帧字节」（`tests/unit/test_sse.mojo?count=8` 逐字节钉住，含两条负向对照）。增量从 handler 流到 socket 走的是**迭代器式**的 `stream_next()`（不是回调）：回调要把 socket 一路传进生成循环，那样「生成」与「写」就再也分不开，而分得开正是这一层能被无权重门钉住的原因。
  - **半流式仍然是最糟的失败形态，所以每条路都有终点**：`stream_begin` 的错（prompt 分词为空 / 过长）在响应头**之前**回 400 —— 前向刻意做成**惰性**（第一次 `stream_next` 才 prefill），否则校验就被挤到头发出去之后；头发出去之后前向出错，给一帧 `error` 再 `[DONE]`；客户端中途断开 → 写失败只丢这一条连接、进程继续（实测 `BrokenPipe` 之后下一个请求照答 —— 这条同时是 SIGPIPE 的照妖镜）。
  - **文本与非流式逐字节相同（不是「差不多」）**：每步用整段前缀解码再取新增的那一截，而不是把这个 token 单独解码 —— 多字节字符可能跨两个 token，单独解码时前半截是不完整的 UTF-8，会被 `decode` 丢掉，拼出来的文本就与非流式那份不同。实测（真权重、贪心）：英文与中文两种 prompt，流式与非流式**逐字相等**。代价是每步重解一遍（≤ 256 步，几十 KB）。
  - **官方 SDK 兼容（手动冒烟，不进门的证据）**：`openai` 1.109.1 只改 `base_url` 即可流式对话，14 帧 = 1 role + 12 内容 + 1 finish，`finish_reason` 与非流式一致。
  - **门**：`pixi run test`（纯函数，帧序列与两条出错路径）、`pixi run test-http`（真 socket：一条完整流的线上字节 + 客户端中途断开后服务存活）、`pixi run test-workers`（3 worker × 12 连接 × 3 请求，fork 出来的每个 worker 分帧与单进程一致）。⚠️ 边界：一 worker 一连接，流式**不修并发**（同时能流的条数 = worker 数）；未验 `stream_options.include_usage`、断线重连与代理缓冲；**没有任何性能数字进账本**。
- **2026-09-22** —— **P3.2 第一半：reactor 事件循环（一条循环 N 条连接）**
  - **换了什么**：服务入口（单进程 `serve.mojo` 与多 worker 里的每个 worker）从「阻塞 accept + 一次一条连接」换成 `srv/loop.mojo` 的 reactor 循环。旧形态 `srv/server.mojo` **保留**：它是这门**常驻的负向对照**（把 E2E 里的服务换回去 → 门必红，实测如此）。
  - **门在哪**：`pixi run test-loop`（真 socket + 真 fork：8 条并发连接 + 一个「发了一半就停住」的对端，延迟增量 ≤ 1 ms，带 2 s 上限）+ `tests/unit/test_conn.mojo`（15 条，已进 `pixi run test`）+ `tests/capability/test_reactor.mojo`（4 条能力门，含非阻塞 accept 的 EAGAIN 与跨线程 wakeup）。
  - **换来什么、没换来什么**：换来的是**连接的 I/O 不再互相挡路**（慢客户端 / 流水线 / keep-alive 的空闲连接都是）；**没换来并发生成** —— 前向仍内联占住循环，同时能生成的条数仍是 1，多 worker 仍是唯一的生成并行。
  - **压测抓到的真 bug**：100 并发压测出现 158 条 `incomplete` —— 空闲超时把**排着队等生成**的连接收掉了（生成慢不是一条连接该死的理由）。修：空闲超时只收「真没事干的」，为此给连接机加了 `peek()`（**只看不吃**，用来分开「排队等生成」与「发了一半」），三条例外（有完整请求 / 有字节没写出 / 正在写流）各有门钉住。修完同条件重跑：`incomplete` 0。
  - **100 并发观测**（真权重、单进程、非流式、60 s）：ok 103、timeout 48、其余分类全 0、fd Δ0。数字**只作同轮相对比较**；这一轮的结论是「100 条连接同时开着且没有被丢 / 被重置」，不是吞吐。
  - **仍 `missing`**（这句记的是 3.2 当时的状态）：engine 独占线程（双线程模型）—— 那才是「生成不再占住循环」的一步；100 并发 **1 小时 SSE** 长跑门（差的已是时长与 SSE，不再是并发数）。**前一半同日已由 3.2b / 3.2c 补上**（见下面两条），后一半仍在。

- **2026-09-22 —— 3.2b：engine 独占线程（`srv/engine_thread.mojo`）**
  - **换了什么**：前向搬到一条独占线程（`ThreadHandle` + mailbox），reactor 上只剩 I/O；线上两条入口（单进程 `serve.mojo`、多 worker 里的**每个** worker）都换成 `run_threaded`，`run` 留作**常驻负向对照**（`run` 那一次必须慢，否则门证明不了任何事）。
  - **门在哪**：`pixi run test-engine`（真 socket + 真 fork + **真线程**；它 fork，所以「先构建再跑」，故意不进 `pixi run test`）。另跑过 `pixi run test-loop` / `test-http` /`test-workers` / `pixi run test`（`test-workers` 顺带验了 fork 之后起线程这条风险路径）。
  - **换来什么、没换来什么**：换来「生成不再占住循环」—— 判据是「槽位占满时，慢生成期间一条新连接仍然被接进来并立刻关掉」：探子从连上到收到 EOF，`run_threaded` 0 ms、`run` 4.5 s（那 5 s 的生成它全等完了；数字只作**同轮相对比较**，本机长期过载）；**没换来并发生成**。
  - **踩到的坑**（都写进源码注释了）：① 装 mailbox 的那个值在 `join` **之后**必须再被碰一次 —— ASAP 析构会让它提前死，子线程于是用上了已释放的锁（症状：子进程 `dumped core`）；② 请求只能以**原始字节**过线程（`HttpRequest` 挪不进邮箱，engine 侧重新解析一次）；③ 流的头必须单独一个 kind（`RES_STREAM_HEAD`），否则循环不知道该开始喂帧 —— 症状是客户端只收到一个响应头然后挂到空闲超时；④ 一帧回来要把 `stream_inflight` 清掉，否则流在第一帧之后停住；⑤ **门的收尾必须 kill 自己的子进程**：监听 socket 是 SO_REUSEPORT 的，上一轮遗留的服务会和新的**一起**监听同一个端口，连接被内核随机分给其中一个 —— 症状是「请求偶尔没人答」「响应只有半截」，看起来完全是服务的间歇性 bug；⑥ **判据换过三次**：先拿「另一条连接的 health 答得快」（错 —— engine 只有一条线程，那条 health 本来就该等）；再拿「生成期间还有 800 KB 在往外走」（也错 —— 内核缓冲把字节全吞了，两端一样快）；最后是「生成期间还能 accept」（内核替不了应用做 accept）。前两次都是**门看起来在跑、其实没盯住被改动的那一处**。

- **2026-09-22 —— 3.2c：engine 线程池（并发生成，`srv/engine_thread.mojo` 的 `spawn_engines`）**
  - **换了什么**：engine 从「一条独占线程」扩成 **N 条**（`ALOFA_ENGINE_THREADS`，1–16，**默认 1**）。每条线程各领一份 handler：0 号用原型，>0 号 `handler.spawn_twin(index)`（新 trait `Twinable`）。`Job` 带 `thread`，`Mailbox.take_job(thread, job)` **扫单队列只取自己的** —— 每线程一队列会把一个 `MAILBOX_CAP` 变成 N 个上界，那是把上界悄悄放大 N 倍。Loop 侧按槽位记 `streaming` / **黏性 `owner`** / 单槽位单在途，于是**流有相位亲和**（一条流始终在同一条线程上）且**同一连接的请求按序**。
  - **代价（明文写在这里，不是疏漏）**：**N 条 = N 份权重**。`ModelService.spawn_twin` 是**重新加载**一份 `QwenForward`，还没有「共享只读权重 + 各一份 KV」那一层（那要动到 model，把"只读的权重"与"每条流的状态"分开）。所以**默认 1**，多 worker 下是 `workers × engines` 份；`engine_threads>1` 时启动打印会先把这句话说出来。
  - **门在哪**：`pixi run test-engine`（真 socket + 真 fork + **真线程**；三模式 `MODE_THREADED`=1 / `MODE_INLINE` 负向 / `MODE_POOLED`=2）。另跑过 `pixi run test`、`test-http`、`test-loop`、`test-workers`。
  - **换来什么**：同时能生成的条数 = `engines`。**判据换过两次**：①「第二条请求什么时候被答」—— 错，**派得早但生成串行**时它也会很快（那是排队，不是并行）；② 最后是「**两条 5 s 生成的总墙钟**」：`engines=2` **5002 ms**、`engines=1` **10002 ms**（阈值 7 s / 8 s；数字只作**同轮相对比较**，本机长期过载）。两条请求必须发在**两条连接**上（一个槽位一次只派一件在途的事，一条连接上的两个请求本来就是串行的 —— 拿它量并发量不到东西），且两台服务都**新起**（`engines` 是起线程那刻定下来的，「两条连接各归一条线程」靠 `next_engine` 从 0 开始轮转，复用一台派过活的服务，归属就不由这个门决定了）。
  - **顺手修掉的真 bug**：响应编号是**每一份 handler 自己**的计数器，两份都从 1 开始 → 两条并发生成交回同一个 `chatcmpl-N`，客户端按 id 去重会安静地丢一条。改法是把起点按 `ID_STRIDE`（100 万）**错开**，而不是共享一个计数器 —— 共享之后编号顺序变成「哪条线程先跑到那一行」，同一个请求在 `engines=1` 与 `engines=2` 下编号不同**而且是随机的**，门就没法拿它做逐字节比较了。
  - **逐字节比较只掩两处**：`chatcmpl-<编号>` 与跟着它变长的 `Content-Length`（它们**必然**随 engine 条数变）。状态行、头的顺序、JSON 字段的顺序与取值、SSE 的分帧与 `[DONE]` 收尾全部照比 —— 那正是「生成搬到另一条线程上」最容易漂的地方。
  - **踩到的坑**（都写进源码注释了）：① Mojo 1.0 的 **trait 方法不能返回 `Self`**、trait 也**不能带类型参数** → `spawn_twin` 只能交**堆地址**（`heap_place` / `heap_take` 一对助手；`heap_take` 取回后凭作用域析构归还那份权重），`ChatHandler` 因此要求 `S: Service & Deinitable & Movable & Twinable`，五个测试替身都要跟着实现；② 参数不可改（`engines` 要另起局部变量）；③ `queue_request(box, slot, self.gen[slot], self.owner_for(slot), …)` 报 provable aliases（不可变借用与可变借用同时活着）→ `gen` / `thread` 先拷成局部值；④ 不带 `var` 的函数参数是**借用**，`self.f = p^` 编译不过（要拷贝）—— 这一条在 `serve.mojo` 里躺了一版才发现，因为它只在 `-O2` 那条 `pixi run serve` 的路径上被编译。
  - **没换来什么**：并发生成 ≠ 吞吐。**批调度**（把多条生成合成一次前向）、只读权重共享、`engines` 与核数 / `shards` 的关系都还没做；`engines>1` 在真权重下的**内存**代价已写明并打印，**吞吐**数字一个都没测（本机过载，测了也不能进账本）。

- **2026-09-22 —— 批大于 8 那条曲线的尽头：`RB = 8` 是天花板（不是没量，是量到了）**
  - 起因：能力行「批 / 预填 GEMM 的权重复用」此前自己挂着一条边界 ——「只在批 = 8 / 预填 32 token 上量过，批大于 8 没量」。这条登记在案的空隙今天补掉：`_gemm` 按 8/4/2/1 **分块**，理论上 `rows > 8` 时权重读 `ceil(rows/8)` 遍、每行摊到的字节不再下降，但这从来没被量过 —— 没量过的东西不许进账本，也不许被当成「再攒一批还会更快」。
  - 改了什么：`scripts/bench_gemm_rows.mojo` 的 `MAX_ROWS` 8 → 32，`rows` 列表加 16 / 32。**`REGIONS = 8` 没有动** —— 它是「每趟换一份互不相交的权重」、防止同一份权重趟趟落在 L3 里（139 MB ≫ L3 12 MB），与一次算多少行无关，所以不跟着扩。
  - 量出来（`-O2`，`down_proj` 896×4864，三次运行的 min…max，口径是**每行**耗时）：`rows`=1 **1.42–2.46**、2 **0.80–1.35**、4 **0.51–0.66**、8 **0.36–0.64**、16 **0.38–0.48**、32 **0.44–0.54** ms → **8 以内约按 1/rows 掉，而 8 / 16 / 32 三档区间互相重合**。
  - 结论（只能这样写）：**8 以上没有新的复用红利**。与 `RB = 8` 的机理对得上 —— `rows=32` 时权重读 `ceil(32/8) = 4` 遍，每行摊到的字节与 `rows=8` **一模一样**。所以「再多攒一批」不是这条路上的下一个杠杆，**把 `RB` 抬上去才是**（那是拿累加器寄存器更强的 ILP，收益与代价都得另量，`RB` 已经是 8 条部分和的极限写法）。
  - ⚠️ **不许反着读**：区间重合 ≠「批大了没用」。批 8 相对批 1 依旧是约 6× 的每行摊薄（上面那行的核级实测 6.4×）；这里说的是**8 之后再加批次兑换不出更多**，以及排队延迟要另算。
  - ⚠️ 三轮里第 1 轮机器是忙的（同进程纯读带宽 T=1 只有 6.91 GB/s，第 2/3 轮是 11.22 / 10.26）→ 报的是**三轮合起来的 min…max**。单独引用第 1 轮会得出「`rows=32` 反而比 `rows=16` 慢」这种反向结论，它与后两轮不同向 —— 这就是那条「必须重复取区间」的规矩在这里具体救了一次。
  - 门的改动：**无**。这次只量了已有的核，`src/` 一行没改，`check-ledger` / `check-counts` 的数字不变。

- **2026-09-22 —— 抬 `RB` 到 16：假设被否掉（0.78–0.81×，不采用）**
  - 背景：同一天先量出「`RB = 8` 是批次复用的天花板」（每行耗时在 8/16/32 三档重合），当时把下一步写成「把 `RB` 抬上去才是」。这句话是一个**待证伪的假设**，不是结论 —— 所以第二天（同一会话内）就去否它，而不是带着它往下走。
  - 为什么它**可能输**，跑之前就写进脚本文件头了：`_gemm_tile[RB]` 的热状态是 `RB` 个 f64×4 累加器 + `RB` 个 f64 尾巴。`RB=16` 是 512 + 128 字节，而 AVX2 堆只有 16 × 32 字节、还得装 `wv`/`xv`/地址 → **必然溢到栈**。所以这不是「多复用一倍」那么简单，是拿**栈流量换 DRAM 流量**。
  - 怎么量的：`scripts/bench_gemm_rb.mojo`（新增）。**同进程配对 A/B**：两个 arm（全用 `_gemm_tile[8]` / 全用 `_gemm_tile[16]`）交错 **5 轮**，**逐轮轮换出场顺序**（不轮换则先跑的那个系统地吃亏）。主口径 = 每趟耗时，副口径 = 每轮的**配对比值**取 5 轮 min…max。**`src/` 一行没改** —— 探针直接越过 `_gemm` 的派发去调 `_gemm_tile[RB]`，赢了才回去改核。
  - 结果（`down_proj` 896×4864，8 份互不相交权重）：`rows=16` RB=8 **5.99–6.27** 对 RB=16 **7.68–7.71** ms/趟，配对比值 **0.78–0.81×**；`rows=32` **12.27–12.53** 对 **15.35–15.52**，配对比值 **0.79–0.80×**。→ **两档同向、且都远在 1.0 之下**，采用门槛（保守端 `RB8_min/RB16_max` 两档都 > 1.0）**没过** → **不采用**。
  - 自检（不通过就一个数都不报，全过了）：① 一块 `[16]` == 两块 `[8]`，**逐位**（`RB` 只决定谁和谁一起走，每个输出各自的累加器与 `k` 次序一次都没变）；② 两者都与**绝对参照物**「`rows` 次 `[1]`」逐位相等（防①两边共用一个错 —— 注入「所有行写第 0 行」时①会假绿）；③ 每次比前把 `dst` 涂 -999，**漏写一行就留下一整片哨兵**（不涂的话漏掉的行保留上一次的正确值，门自己就是绿的）。
  - 顺手学到的两件小事（都写进源码/脚本注释了）：① **`_gemm_tile` 可以从别处 import**（下划线不是访问控制）→ 探针能在不改核的前提下并行比较两个分块大小，这比「编两个二进制再交替跑」便宜得多；② arena 的坑今天踩了一次实的：`Arena` 在**最后一次使用处**析构，而 `alloc` 出来的指针要用到那之后 → 必须在 `main` 末尾 `arena.keep_alive()`，不写就是 SIGSEGV，且崩在**第一次写 `w`** 那一行（`-O0` 重建才敢确定，不是看起来可疑的那行）。
  - **别反着读**：这不是「复用不行」—— `RB=8` 相对 `RB=1` 依旧是 6.4×。它是说**在这个核的这个点上，再减 DRAM 流量已经换不到时间**，注意力该挪到算术那一侧。
  - ⚠️ 边界：只试了 `RB=16`（`RB=32` **没试**，那只是按同一机理的外推、**不是实测**），**没端到端数字**，`scalar` 通路与 q4 通路**没动**（它们批 > 1 时仍每行各流一遍权重）。

- **2026-09-22 —— 第二个杠也被否掉：「f32→f64 加宽转换」不是瓶颈（V1 = 0.73–1.06×，不采用）**
  - 上一个负结果留下的问题：`RB=8` 上 DRAM 带宽不是瓶颈，那时间花在哪？没有 `perf`（本机 `perf: command not found`，`perf_event_paranoid=4` 也不动）→ 不能用硬件计数看端口占用，于是改成**机制相反的实验**来判别。
  - 纸上的数（每 tile，`rows=8`，点的是新添的内层指令数）：`VMFMADD` 8.7 M / `VCVTPS2PD` 9.8 M（**比乘加本身还多**）/ load 9.8 M；观测 ~12 M 周期。⇒ **候选瓶颈 = 加宽转换**（纸算，不是实测）。
  - **V1**（`scripts/bench_gemm_cvt.mojo`，新增）：现在的循环是「列外层、k 中层、行内层」，于是**每一列都把整个 x tile 重新 load + 重新 cvt 一遍**。改成**一个 tile 只转一次**（预转进 `RB×inner` 的 f64 scratch，内层直接 load 宽操作数）→ cvt 从 9.8 M 掉到 ~1.09 M，**load 指令数不变**，但 **x 字节翻倍**（156 → 311 KB）。两种机理预测的方向相反，所以一个数能把它们分开。
  - 结果（同进程配对，5 轮交错 + 轮换出场顺序，RB ∈ {4,8} × rows ∈ {8,16} 四块）：配对比值 min…max 依次 **0.73–0.77 / 0.75–0.83 / 0.76–1.06 / 0.90–1.01×** → **四块的保守端全部远低于采用门槛 1.10** → **不采用**，加宽转换不是瓶颈。
  - 自检三道全过（否则一个数都不报）：① V1 与 baseline **逐位相等**（同一个 `Float64(px[..])`、`k` 次序一次没变）；② 两者都与绝对参照物「`rows` 次 `[1]`」逐位相等；③ 每次比较前涂 -999 哨兵。
  - **这条负结果把下一步指清楚了**（比"再来一发"有用）：x tile 每列重读一遍 ⇒ 每 tile 从缓存流出 **139 MB**，是权重本身（17 MB）的 **8 倍**；而 V1 只把 x 字节翻倍就慢了 25–30% ⇒ **真正的瓶颈大概率是 x 这条缓存流**（不是端口上的 cvt）。所以下一个候选是「让多个列共用一次 x」（见下一条）。
  - ⚠️ 边界：只 `down_proj` 一个形状、只 avx2 一条通路、**没有端到端数字**；这里的"端口 away 占用"是纸算，**没有硬件计数校**（本机没 `perf`）。

- **2026-09-22 —— 第三个杠杆（`cols` 也分块）：唯一看得见收入的那个，但没过门槛、也不推广 → 不采用**
  - 为什么试它：前两个负结果把矛头指到 **x 这条缓存流**上。现在的循环是「列外层、k 中层、行内层」⇒ 每列都要把整个 x tile 重读一遍 = 896 × 8 × 4864 × 4 B = **139 MB**，而权重本身一个 tile 只有 **17 MB**。之前所有注意力都在"权重怎么少读一遍"，其实最大的那股流动的是 x。
  - 怎么量：`scripts/bench_gemm_cblock.mojo`（新增）。**列也分块**：x 载入一次给 `C` 列共用 ⇒ x 流量 ÷ C。代价写在另一边 —— 累加器从 `RB` 个涨到 `RB×C` 个（每个输出各自一个 4 通道 f64），**必然溢出**。同进程配对，5 轮交错 + 轮换出场顺序。**采用门槛跑之前写死**：同一个 C 在 rows=8 与 16 两档的**保守端**都 > 1.10。
  - **逐位不变是硬约束**，做法是：每个输出仍各自一个 4 通道累加器（lane 由 `k % 4` 定）+ 一个尾巴标量；reduce 之后尾巴按 `k` 升序加。于是换遍历次序并不改变任何输出内部的加法顺序，结果必须与 `_gemm_tile[RB]` 逐位相同 —— 由三道自检验（含 `-999` 哨兵、绝对参照物 = `rows` 次 `[1]`）。
  - 结果（`down_proj` 896×4864，RB 固定 8，**两轮运行、每次每档 5 轮配对**）：
    | C | rows=8 | rows=16 | rows=32 |
    |---|---|---|---|
    | 2 | 0.73–0.84 | 0.75–0.83 | — |
    | 4 | 0.90–0.98 | 0.91–0.95 | — |
    | 8 | 1.01–1.15 | 0.98–1.08 | 1.03–1.18 |
    | **16** | **1.05–1.16** | **1.04–1.12** | **1.09–1.25** |
    | 32 | 1.01–1.07 | 1.00–1.07 | 1.01–1.09 |
    曲线**实测掉头**：峰值在 C=16（两轮、三档、共 30 次配对**全部**偏向 V2），因为到那时 x 流已降到 8.7 MB < 权重的 17 MB，再往下砍就没有红利、只剩溢出的账 —— **与机理一致**。但保守端 1.04–1.05 **没过事先定的 1.10 门槛** → **不采用**（喊票把门槛放松到 1.0 就属于"为了让结论过门而改口径"，账本里不许）。
  - **推广性检查（这一段才是决定性的）**：七个投影里**四个是 896×896**（q/k/v/o），那个形状里 x tile 只有 28 KB、**本来就装得进 L1**，没有那股 L2 流可砍。同一份探针（`ALOFA_PROBE_COLS` / `ALOFA_PROBE_INNER` 可换形状，形状现在是 **运行时**的）跑出来：C=16 在 rows=8 是 **0.54–1.96×**、rows=16 **0.78–1.50×**、rows=32 **0.88–1.18×**（两轮，loadavg 6.99）→ **方向判不了**，且质量重心偏 1 以下。这里的"没收益"与机理一致：可见的红利**只出现在 x tile 装不进 L1 的那几个投影上**。
  - **顺带第一次真的数了指令**（本机**没有 `perf`**，`perf_event_paranoid=4` 也不动，所以只能用 `objdump -d` 看静态产物；见 `src/alofa/kernels/cpu/avx2.mojo` 文件头那条 2026-09-22 注释）：`-O2` 编出来的 `_gemm_tile[8]` 内层**确实**是 VEX 编码的 `vfmadd231pd`（真融合乘加），且 **RB=8 的 8 个累加器留在寄存器里**（该函数体里 `%rsp` 相关的 `vmovupd` 是 **0 条**）；对照列分块版本，溢出后每次更新都变成「取栈 → FMA → 存栈」多两条 memory uop。每 `kk` 步 ≈ 9 load + 9 `vcvtps2pd` + 8 FMA，只推进 32 个乘加。
  - **今天这一串的整体结论**：三个杠杆（抬 `RB=16` / 去掉加宽转换 / 列分块）**都不是便宜的**，因为它们都是拿一种资源换另一种（寄存器 ↔ 缓存字节 ↔ DRAM 字节），三种都落在 ±25% 之内。最像机会的那个（C=16）**只在最大的那个形状上有可见收益，且低于噪声**。
  - ⚠️ 边界：**核级**数，avx2 一条通路，**没有端到端数字**；列分块版本不处理 `cols` 不能被 `C` 整除的余数列（形状都能整除，忘了处理的话哨兵自检会红）；`scalar._gemm` 与 q4 通路**没动**（它们批 > 1 时仍每行各流一遍权重 —— 那是**另一处已知的 6× 级产物**，与今天这些负结果无关）。

- **2026-09-22 —— 量化分支里一个真 bug（行偏移多乘了 4，批 > 1 时第 1 行之后的 token 一行都没被写 + 越界写）：已复现、已修、已补门**
  - 起因：今天第三个性能杠杆否掉之后，转向 q4 通路（`foreach` token 各自把整份量化权重重流一遍是已知的账），写探针时顺手核了 `project` 里那两行 row 偏移，看到了
    `var out_row = dst_p.unsafe_offset(r * out * 4)` / `var in_row = x_p.unsafe_offset(r * cols * 4)`。
  - **为什么它是错的**：`dst_p`、`x_p` 都是 `f32_data(...)` 出来的**类型化**指针，`unsafe_offset` 按**元素**走（实测：偏移 1 → 第 1 个元素，偏移 4 → 第 4 个）。而同一文件 410/423 行那一带的 `* 4` 是**对的** —— 那里的 `d0`/`x0` 是 `dst.data.unsafe_offset(dst.byte_offset)` 得来的 **RawPtr，按字节**走。看起来就是字节写法的习惯被抄到了类型化指针上。
  - **复现**（`unsafe_offset` 与守卫区两条判据都不依赖"有没有崩"，`out=32 / cols=64 / 批 4`）：按源码写法 → **第 1、2、3 行仍是哨兵（压根没被写）**，且 `dst` 之后的守卫区被改写 **64 个元素**；改成 `r * out` / `r * cols` → 哨兵行数为 0、守卫区干净。
  - **这个 bug 一直在被跑着**：`tests/unit/test_q4_greedy.mojo` 断言了 `model.q4_enabled` 之后调 `model.prefill(ids)` —— 整段 prompt 一批，正好是这个 `for r in range(t_rows)`。它之所以没让今天的绿三国破梦，是因为那道门只有 **0.75 的下限**，而 bug 状态是 0.8164 —— **落在绿区里**。
  - **同一棵树上的 A/B（证据，不是推断）**：带 `*4` → **418/512 = 0.81640625**（与账本里 2026-09-17 记的 0.8164 一字不差）；去掉 `*4` → **445/512 = 0.869140625**。也就是说：这 5.3 个百分点过去一直被算在"量化质量"头上（2026-09-18 那份诊断把低一致率归因于输出投影的敏感性，方向可能没错，但**其中的 5.3 pp 是 bug**，不是格式）。
  - **改了什么**：`qwen.mojo` 两处 `* 4` 去掉，并在原地写了注释说明"类型化指针按元素、上面分片函数那个 `* 4` 是对的因为那里是 RawPtr"。**`src/` 里没有别的同类写法**（已全仓扫过 `unsafe_offset(... * 4)`）。
  - **补的门**：同一份 q4 测试里新增 `Q4_REGRESSION_FLOOR = 0.85`（第二道断言），理由是 0.75 那道看不见这个差别 —— **退回这个 bug 会立刻红在这条上**。文件头那句"实测 0.80 / 0.77"也一并改成今天的实测值。
  - ⚠️ 边界：**余下的差距仍然是真的** —— 0.8691 < 0.90，那条质量门**照旧 `missing`**，没因为修好 bug 就宣布达标；"连输出投影一起量化会掉到 0.77"是**旧写法下**测的，**没重测**，所以那句话今天实质上已经站不住（列在这里提醒：下次动量化时先把它重测一遍，别引用）。
  - 未做的事：q4 的 `RB` 行复用（今天原本要量的那个杠杆）**还没量**，探针 `scripts/bench_q4_rows.mojo` 已写完但未跑通 —— 修 bug 插在了前面。

- **2026-09-22 —— 第四个杠杆（q4 的 `RB` 行复用）：**这是今天第一个**远超门槛**的结果，保守端 **2.38×**，但还没接进核**
  - 为什么轮到它：前面三个杠杆（抬 `RB=16`、去掉加宽转换、列分块）全都是"拿一种资源换另一种"，总 uop 数一点没减。q4 这条不一样 —— `_matmul_q4_halves` 的注释写着批=1 时它**算术受限**（只用到自己访存地板的 **10.6%**），而解量化那套算术（`raw` 读入、`&0x0F`、`>>4`、cast、减 8）**与输入行无关**。
  - **V3**（`scripts/bench_q4_rows.mojo`，新增）：一个块只为 `RB` 行解量化一次；只有后面的 x 载入、乘 `d`、两次乘加随行数增长。今天 `qwen.mojo` 是「每个 token 一行，逐行喂进一维的 matvec」⇒ 批=N 就把同样的权重流 N 趟。
  - 结果（`down_proj` 朝向 out=896 × cols=4864，每份块流 2 MB × 8 份 = 18 MB ≫ L3，**两次运行** × 每档 5 轮配对）：
    | RB | 批=8 | 批=16 | 批=32 | 两次合并的保守端 |
    |---|---|---|---|---|
    | 2 | 1.10–1.81 / 1.60–1.64 | 1.56–1.62 / 1.46–1.65 | — | ~1.46 |
    | 4 | 2.20–2.25 / 2.08–2.39 | 2.26–2.45 / 2.29–2.45 | — | ~2.08 |
    | **8** | **2.38–2.91** / **2.67–2.90** | **2.54–2.93** / **2.78–2.85** | **2.61–2.72** / **2.77–2.86** | **~2.38** |
    趋势单调往上、**没有像列分块那样掉头**（`RB` 越大共用得越彻底，什么时候掉头本轮没找到 —— 二号 RB=8 时累加器已 32 条 ymm，是寄存器堆的 2 倍）。**> 1.10 的门槛**在每一档、每一次配对上都满足（共约 70 次配对全部 > 1）。
  - 自检三道全过（否则一个数都不报）：① V3 与「逐行调 `matmul_q4_f32`」**逐位相等**（同一个半块切、同一个 f32 累加、`d` 同样先折进 x、块次序一次没变）；② 每次比较前涂 -999 哨兵；③ **负向对照针对自检本身**：漏最后一个块的 `MUTATE` 版必须被①判红。
  - ⚠️ **（这条记的是当天早上：当时还没接进核，且差下面四件事。当天晚些时候四项都补完了，证据见本日变更日志的下一条 —— 别停在这一段。）** 而且要接进去还差四件事（别把上面的数当成"已进账"）：① `t_rows % RB != 0` 的余下几行要有后备路径；② 这里只量了**无偏置**那条（down/gate/up），q/k/v 的带偏置包装是**另一个函数**（`matmul_q4_f32_bias` → `_matmul_q4`，累加结构不同），要单独做并单独的自检；③ `q4_matmul_k_shards` 现在是**每个输出分片、每个输入行**各调一次，行复用与分片这两个 slices 要一起排（直接换可能把多线程那份收益吃回去）；④ 缺少"与 N 次单行的参照物逐位相同"的**核级差分门**（现在只在探针里有）。
  - ⚠️ 边界：核级数、单线程、合成块流、**没有端到端数字**；`10.6%` 那条是账本里记的旧观测（批=1），本轮的结果说明批 > 1 时它早就不是算术受限了。

- **2026-09-22（续）—— q4 行复用已接进核与模型层：端到端 prefill **2.1–3.4×**，量化贪心一致率一个比特没动**
  - **核**（`src/alofa/kernels/cpu/avx2.mojo` 新增）：`_matmul_q4_halves_rows[RB]` —— 一个块只解量化一次、`RB` 行共用；外面两个包装：整批 `matmul_q4_f32_rows[RB=8]`，以及带 **`dst_stride`** 的 `matmul_q4_f32_rows_band[RB=8]`（这批在 `dst` 里不是紧挨着的时候用它）。`dst_stride < rows` 会 `raise` —— 那种错不会再长成"数不对"，而是直接写到别人的区域上。
  - **模型层**（`src/alofa/model/arch/qwen.mojo`）：新增 `q4_matmul_rows_k[backend]` 与 `q4_matmul_rows_k_shards`；分片顺序改成**每个输出行带一口气做完整个 token 批**（原来反过来：每个分片只做一个 token，于是整份权重被读 `n_tok` 遍）。`project` 里**无偏置**的三个投影（down/gate/up）走新路。
  - **门**（新增 `tests/unit/test_q4_rows.mojo`，3 tests，已登记进 `scripts/run_tests.sh`）：① **逐位等于**同一后端的单行通路（零容差，主判据）；② 数值等于 **标量**实现（另一套代码、f64 累加，判据 `1e-5 × max(1,|ref|)`）—— ① 是同源比较，半字节装反会两边一起反而照样绿，所以必须再挂这条；③ 哨兵 + 守卫区。形状故意**除不尽**（`cols` = 3/5 个块，批 = `RB-1` / `RB+1` / `2·RB+1`，行带 band 数除不尽）。
  - **三条负向对照**（缺一门就不算门）：N1 改一个元素 → 逐位比较必须红；N2 **把 2026-09-22 那个 bug 的旧步长（`r*rows*4`）原样复原** → 哨兵/守卫必须判红（实测：3 行没被写、守卫区被踩 32 个元素），这条同时是那个 bug 的常驻红测；N3 漏掉尾数行 → 哨兵必须红。另有一条针对新守卫：`dst_stride < rows` 必须报错。
  - **门本身被验过会红**：在核里注入"漏最后一个块"的变异 → `test_rows_match_single_row_calls` 变 FAIL，恢复后转绿。
  - **端到端 A/B**（真实权重 `qwen2.5-0.5b`，q4 全程、分片 8、`BACKEND_AVX2`；每长度预热 1 趟丢弃后取 5 趟 min…max）：
    | prefill 长度 | 逐 token（旧） | 整批（新） | 新（第二次运行） | 比值 |
    |---|---|---|---|---|
    | 16 | 827–1000 ms | **385–499 ms** | 416–479 ms | 2.1× |
    | 32 | 2101–2311 ms | **581–717 ms** | 689–804 ms | 3.4× |
    | 64 | 3224–4014 ms | **1086–1291 ms** | 1085–1480 ms | 3.1× |
    两次运行的区间重合，且与旧版**完全不重叠**（效应远大于本机 ±10% 噪声）。脚本 `scripts/bench_prefill_q4.mojo`。
  - **数值一分未动**：`test-q4` 仍是 **445/512 = 0.869140625**，与接线前逐位相同（Suite 里的 greedy 结果没变）。
  - ⚠️ **边界**：① **带偏置**的 q/k/v 今天**没接**（累加结构不是同一个函数，接之前要先给那条也开同样的差分门）；② **标量后端**拿不到收益 —— 分发里它走逐 token 兜底，这也是它保持逐位一致的理由；③ 只测了 prefill（批 > 1）；**decode 批 = 1 本来就没有"别的行"可复用**；④ 本机长期过载 → 上面的毫秒数只在同会话 A/B 里比，不与别处比；⑤ 这些数不来自 `pixi run test`（那里跑的是 `-O0` 的核级门）。

- **2026-09-22（续 2）—— 带偏置的 q/k/v 也接进去了：prefill 在上一档上再叠一层，`test-q4` 依然一分未动**
  - **另一条核**（`avx2.mojo` 新增 `_matmul_q4_wide_rows[wide, RB]` + 包装 `matmul_q4_f32_bias_rows_band[RB=2]`）：带偏置的通路的累加结构**不是**无偏置那条（这里是 f64 通道、按 `j%8` 切、`d` 折进 x、整行只归约一次；那边是 f32 半块），所以它是**另写一个**而不是改一下复用 —— 两条各自的每一行都得单独跟它原来那条通路对齐到逐位，这也是这两件事分开做、各自开一道门的原因。
  - **模型层**：新增 `q4_matmul_bias_rows_k` / `q4_matmul_bias_rows_k_shards`；`project` 现在**两条路都**走整批（旧的逐 token 循环已经被删掉，不是留着当后备）。
  - ⚠️ **接的时候踩到一个真陷阱**：行带的 API 里 `blocks` 由核自己按字节偏、`dst` 由调用方按元素偏，而 **`bias` 必须由调用方偏到这一带的起点**。漏偏的表现是"第一个带全对、后面每个带全不对"，而差分门只会把它说成"数不对"（第一次跑就是这几档红了：`RB=2` 批 2 / 批 10、`RB=4` 批 1，坏条目数正好等于"除第一个带以外的元素数"）。约定已写进 `matmul_q4_f32_bias_rows_band` 的文档。
  - **门**：`test_q4_rows.mojo` 加了 `check_bias_band_case`（逐位 vs `matmul_q4_f32_bias` + **标量**的 `quant.matmul_q4_f32_bias` 参照 + 哨兵/守卫，形状同样除不尽）。**另外往这条新核里注入"漏最后一个块"确认了它会红**（注入 → FAIL，恢复 → 绿）。
  - **端到端**（真实权重 0.5B、q4、shards=8、AVX2，预热 1 趟 + 5 趟 min…max）：在只接了无偏置那一档的基础上**再叠一层**
    | prefill 长度 | 逐 token（原始） | 只接无偏置 | **两条都接** |
    |---|---|---|---|
    | 16 | 827–1000 ms | 385–499 ms | **229–270 ms** |
    | 32 | 2101–2311 ms | 581–717 ms | **452–545 ms** |
    | 64 | 3224–4014 ms | 1086–1291 ms | **807–1065 ms** |
    对最初的逐 token 版本：**3.5× / 4.4× / 3.6×**。
  - ⚠️ **`RB` 取 2 不是量出来的**：`RB=2/4/8` 端到端三档跑出来（长度 16/32/64，ms）：2 **229–270 / 452–545 / 807–1065** 与 **294–346 / 538–642 / 962–1242**（两次）；4 **284–333 / 509–635 / 950–1179** 与 **261–353 / 497–611 / 965–1312**；8 **309–332 / 542–690 / 908–1104**。同一份代码的两次运行本身就能差 ~20% → **本机噪声把这个效应吃掉了，三者不可分辨**。取 2 的依据是累加器预算（这条每 RB 行 256 B，AVX2 只有 512 B 的 ymm），**这是个理由不是结论**，已经原样写在核的注释里。
  - **数值一分未动**：`test-q4` 仍是 **445/512 = 0.869140625**（这次连 q/k/v 的逐位结构也换成了整批版）。
  - ⚠️ **边界**：① 标量后端仍然拿不到收益（`q4_matmul_bias_rows_k` 的标量分支走逐 token 兜底，这也正是它逐位一致的理由）；② decode 批 = 1 无从复用；③ 上面的毫秒数只在同会话相邻两次里比。

- **2026-09-23 —— 第一批能进账本的性能数字：roofline 接进真实算子，并当场划掉一半**
  - 起因：能力行「roofline 报告（真实 kernel / 真实硬件）」此前挂 `missing`，而 `scripts/bench_decode_roofline.mojo` **早已写好却从未跑进账本**（同进程现测两个峰值、真实形状喂给 `Roofline`）。这次跑完，并把「哪些数能引用、哪些不能」分开记。
  - **能引用的**：同进程**配对**的「自己的访存地板 / 实测」（三次运行 min…max，区间都紧）：`hidden_proj` **0.97–1.05**、`down_proj` fp32/avx2 **0.85–1.01**、fp32/scalar 0.37–0.48、**q4_0/avx2 0.24–0.29**、q4_0/f64-8w 0.15–0.22、q4_0/scalar 0.05–0.07。
  - **由它得出的第一条性能事实**：fp32 decode **带宽受限**（比值 ≈1，实测 6.64–7.15 GB/s）；**q4_0 decode 不是**（0.24–0.29，实测只有 1.87–2.00 GB/s，却已拿到 6.65–7.12 GFLOP/s ≈ 核峰值上沿）。所以 q4 通路的下一个杠杆在**算术/指令**上 —— 与 2026-09-22 那个「解量化按 RB 行复用」（同样是改算术，核级 2.38×）是同一条路。这也解释了 q4 只比 fp32 快约 2.5× 而不是字节数之比 7.1×。
  - **不能引用的**：利用率（‰ of peak）。三次运行里峰值探针自身就在漂（读带宽 7.02–8.79 GB/s、核峰值 5.42–6.80 GFLOP/s、机器峰值 51.32–66.23 GFLOP/s），于是 `hidden_proj` 利用率区间 **572–1446‰**、`lm_head` **572–840‰** —— **跨过 1000‰**（比「峰值」还快，物理上不可能）。按本账本口径记成「量不出结论」，**不挑一轮报数**。
  - ⚠️ **llama.cpp 同机对比这次推不动**：`github.com` 与 `huggingface.co` 都超时（无网络），取不到 llama.cpp 也取不到 GGUF。能力行里那条仍是 `missing`，原因写在行内 —— 不让它变成「没做」。

- **2026-09-23 —— `messages` 不再是"按换行拼"：角色参与进来了**
  - 起因：对外契约上最实的一处缺口。原来的 `role` 被解析又被丢掉（`_read_messages` 里它走 `_skip_value`），多条 content 用 `\n` 接起来当 prompt —— 多轮与系统提示的语义因此是错的。
  - 现在 `render_chatml`（`src/alofa/tokenizer/chat_template.mojo`）按模板排：**先一段 system**（无首条 system 则用模板自带那句），随后 user / assistant 各一段，最后补 `<|im_start|>assistant\n`。
  - **答案不许自己写**：参照物是 `transformers.apply_chat_template` + HF 缓存里那份真正的 `chat_template`，导出成夹具后**逐字节比文本、逐个比 token id**（`scripts/dump_chat_template.py` → `tests/fixtures/qwen2.5-0.5b/chat_template.tsv`，10 词条含多轮、夹在中间的 system、只有 system、空 content、多行、中文 emoji、content 里字面写着 `<|im_end|>`）。两次全库门 30/30 与 6/6。
  - 服务层三处**以前没发过的错**现在会拒绝：缺 `role`、缺 `content`（以前是静默丢掉整条）、`tool` 这类不认得的角色。`/health` 的自白同步改成"按模板渲染、角色参与；它是一个族群的渲染器，不是模板引擎"。
  - 负向对照也抬到了 highest-stakes 那条：**同一段文字换个角色必须给出不同的 prompt** —— 若退回"按顺序拼"的老路，这条必红。
  - ⚠️ 能力行因此写的是 `partial` 不是 `verified`：模板字符串**不是**从模型读的（夹具的 `tokenizer.json` 不带这个字段），换族群要改代码而不是改数据 —— 这才是 roadmap 里"Jinja 子集"那件事没做完的部分。
- **2026-09-23** —— **批调度接进服务层（`ModelService` 持 `EngineCore`），并配上服务层那条批一致性门**：此前 P2 那套执行器只在 cli 与引擎门里跑，服务层一条 engine 线程一次仍只推一条流，批调度那笔收益一个字节也没落到线上。现在一次 `tick` 推进所有在批里的请求
  - 新增重门 `tests/unit/test_batch_stream.mojo`（**3/3，重资产：`pixi run test-batch-stream`，不并入 `test`**），判据与 `test-batch-forward` 同一个（一批 N 条 vs 一条一条跑，**逐 token 相等** —— 比的是 token id 这种整数，一个不同就是不同，没有"差不多"），但比的是**服务层那两条路**：批路（`EngineCore`）与单流老路（`model.prefill` / `step`）。三条：批 4 条（英文/中文/数字/散文，长度各不相同，所以 prefill 是真的凑成一批）、批 1 条（最小形状，与批 4 条分开跑是为了红的时候能直接看出是"批路本身错"还是"合批改变了结果"）、以及**常驻负向对照**「两条路真的是两条路」（批路必须进引擎、老路必须没进 —— 否则前两条就是同一条路和自己比，绿了也一个字节没验到）
  - 三条门**都注入变异验过会红**：① 老路少生成一个 token → 前两条 FAIL、对照仍 PASS；② 批路静默退回老路 → 对照 FAIL（批 4 条那条因老路"一次一条"的核对先崩，是崩溃不是优雅 FAIL —— 这也说明前两条**依赖**第三条才有意义）
  - 开关 `ALOFA_BATCH=0`（默认开）：它存在的理由就是这道门 —— 同一条 prompt 要能分别走两条路才对得上账；关掉即退回批调度之前那条路，所以它同时是"批调度出问题时"的退路（明着慢，不静默变慢）
  - ⚠️ **今天的边界（重要）**：批路只收**贪心**（`temperature <= 0`）+ prompt ≤ `MAX_PROMPT`(128) + 新 token ≤ `MAX_GEN`(32) + 并发 ≤ `MAX_BATCH`(8)，越界的请求退回老路；而**默认 `temperature` 是 1.0，所以默认请求走的是老路** —— 采样批化还没被逐 token 验过，验过之前不上
  - 顺带修掉两个接进来才撞上的真 bug：`engine/executor.mojo` 的输出投影**硬编码**按 `lm_head.weight` 查（绑定词表的检查点没有这一项，`QwenForward` 构造时就选好了名字 → 改用 `model.head`，且要先拷局部否则撞别名）；`Arena` 在指针最后一次使用处析构 → `submit` / `output` 用的 `IntPtr` 会段错误（补 `keep_alive()`）。改了 P2 模块，`test_engine_core` 17/17、`test_batch_executor` 12/12 都重跑过
- **2026-09-23（同日第二条）** —— **采样也搬上批路，上午那条"默认 `temperature=1.0` 走老路"的边界当天收回**：批路此前只收贪心，因为"选哪个 token"是**写死在引擎里**的（`EngineCore.tick` 内部逐槽 `argmax`），而采样要的两样东西引擎都不该持有 —— 每条请求自己的随机源与自己的历史
  - 引擎侧把"选谁"从一拍里拆出来：`decide[backend]()`（排一拍 + 跑前向，**不选**）+ `step_owes(i)` / `step_request(i)` / `step_logits(i)` / `step_choices()` + `settle(chosen)`。`tick` 不再是另一份实现，而是同一条接缝上"决策是一行 `argmax`"的那个特例 —— 所以两条路的"一拍"不可能走偏（一处改、另一处没改的那类错从结构上不成立）
  - 服务侧每条请求带三样自己的东西：`b_temp`（温度）、`b_rng`（随机源状态，存 `UInt64` 而不是 `Rng`，用它的那一步现造）、历史（`b_out` 的前 `b_n` 个 —— `sampler.build` 拿它做重复惩罚，拿错的话只在生成出重复词的那一段才分叉）
  - 门从 3 条扩到 **5 条**（`test_batch_stream`，5/5）：新增「采样批 4 条 == 各自单跑，逐 token」与「四条**同 prompt 同种子**的采样流一起批跑必须互相一致」。后一条不靠老路做参照：同 logits、同分布、同起点，四条流**必须**抽到同一个数；共用随机源时它们必然分叉（交错推进让每条抽到序列里不同位置的数），所以红了可直接归因到随机源
  - 注入变异验过会红：**把所有请求的随机源改成共用第 0 条的状态** → 两条采样门 FAIL、两条贪心门与路径对照仍 PASS（归因精确）
  - 顺带撞上一个真陷阱：`stream_end` 里"让 `cancel` 落地"的那一拍原本调 `engine.tick`（贪心），而它发生在**别人**断开连接的时候 —— 那样会给批里正在采样的请求塞一个 argmax 的 token，那条流的答案从此分叉而日志里什么都没有。已改成走同一个 `_batch_tick`
  - 回归：全库 14 个套件绿（含 `test_engine_core` 17/17、`test_batch_executor` 12/12 —— `tick` 重写行为不变）、引擎层批一致性重门 `test-batch-forward` 6/6
  - ⚠️ 仍存在的边界：批路只收**形状装得进引擎**的请求（prompt ≤ `MAX_PROMPT`(128)、新 token ≤ `MAX_GEN`(32)、并发 ≤ `MAX_BATCH`(8)），越界仍退回老路，而老路一次只握一条采样流（第二条**指名拒绝**）
- **2026-09-23（同日第三条）** —— **「第 9 条并发不是拒绝而是排队」，并在做它的过程中挖出一个更严重的既有 bug：引擎的槽位是**一次性**的**
  - ⚠️ **槽位泄漏（当天最大的收获）**：`EngineCore` 的槽位走完请求后只变成 `ST_DONE`，**没有任何地方**把它收回 `ST_FREE`（全文件 `state` 的赋值只有 5 处，唯一回到 FREE 的是"准入失败回滚"）。后果不是"第 9 条要等"，而是**这台服务一辈子只能服务 `MAX_BATCH` = 8 条请求** —— 第 9 条无论什么时候来都是 `capacity: the engine is full`，而它看起来就是"服务满了"。此前没暴露，是因为唯一长期跑批的调用方（cli）只跑一批就退出
  - 修法是**显式** `release(req)` 而不是引擎自己回收：槽位是输出数组的下标，什么时候能复用只有调用方知道；引擎自己猜就会在"调用方晚一拍才读"时静默覆盖掉上一条的 transcript。契约的另一半是**没走完的请求不许 release**（它的块还记在调度器账上）→ 要先 `cancel` + 一拍
  - 排队本身：槽位满了的请求进 **FIFO 等待队列**（`w_req` / `w_prompt` / `w_steps` / `w_temp` + 头指针，只 append 不删中间项），有空位就 `admit`；上界是 `MAX_CONNS`（一条连接最多一条在途流，所以队长不可能超过连接数 —— 是数的不是估的）。⚠️ **批表有空位 ≠ 引擎有空位**（槽位下一拍才 FREE），所以 `admit` 只吞 `capacity` 这一件事、队首不动下一轮再试，并空转一拍推进引擎状态机
  - "还在等"怎么表达：新增 `StreamToken.waiting` 与 `sse_comment`（SSE **注释帧**）。不能用空串 —— 空串是"流结束"，客户端会拿到一个连 `[DONE]` 都没有的空连接（文件头说的那种最糟形态）；也不能没有节流：注释帧的频率由批里真正的前向自然限定（一次前向一帧），所以不需要另加定时器
  - 门：服务层 `test_batch_stream` 5 → **6 条**（新增「10 条请求（> 8）全部跑完 + `waited` ≥ 2 + 与各自单跑逐 token 相等」—— 后半条是常驻对照："10 条都跑完"也可能只是"恰好没人超额"）；引擎层 `test_engine_core` 17 → **18 条**（新增「一整批跑完之后第 9 条必须被接受」+「未完成的请求不许 release」）
  - 变异一次验两层：把 `release` 里的 `ST_FREE` 改回 `ST_DONE` → 引擎层新门 FAIL（18 条中仅它）、服务层排队门 FAIL（第 9 条被拒）
  - 回归：全库 16 个套件绿、`test-batch-forward` 6/6
- **2026-09-23（同日第四条）** —— **压测 harness 支持 SSE，并首次用它撞出「流式 + 多连接」的崩溃（未修）**
  - `scripts/stress_serve.py` 加 `--stream`：请求体带 `stream: true`，读法改成**等 `data: [DONE]`**（SSE 头里**没有** `Content-Length` —— 写第一帧时长度还未知，所以等 EOF 会把"流走完了"和"连接被掐断"混成一件事）。新增两类观测：`stream_truncated`（200 也发了帧但没走到 `[DONE]` —— 客户端看到的是一条异常短的流而不是错误，所以不会进错误日志，**压测里最该盯的一类**）与 TTFT（首帧到达，SSE 才有这个时刻）
  - ⚠️ **新发现的硬阻塞（未修，与今天的批调度/排队无关）**：**流式 + 并发 > 8 会让服务进程崩溃**。判别实验两条：① 非流式 12 并发 —— **不崩**（15 成功 / 0 错误）；② 流式 12 并发且 `ALOFA_BATCH=0`（批路关闭、**根本不排队**）—— **照样崩**。所以崩的是"流式 + 多连接"这条既有路径，不是今天新加的排队
  - 它正是 **Gate P3（100 并发 SSE 1 小时）的硬阻塞**：在此之前流式从没被压到过 8 条并发以上（账本里已落地的是"分钟级、非流式、并发 ≤ 8"），所以这个缺陷一直没机会出现。栈指向 `Tokenizer::encode`（符号化被截断，未定位到行）
  - 顺带：8 并发流式下 8 个 token 要 13 s（门里同样 8 token 约 1.5 s）—— 慢了约一个并发数，怀疑流式下并没有真正合批，**未查**，与上面那个崩溃分开记
- **2026-09-23（同日第五条）** —— **流式并发崩溃的定位进展：位置抓到了，根因没抓到，一处推断被证伪后已回退**
  - 崩溃点有两个 gdb 栈（`gdb -batch -ex run -ex "bt 30" --args ./target/serve`，`-O2` 二进制 + 12 并发 SSE 复现）：
    ① `String::_add` ← `chunk_text_json` ← `ChatHandler::stream_next`；② 另一处 `List::_realloc` ← **`Loop::run_threaded`**（reactor 线程）。两次都是 SIGSEGV，都不是断言
  - 判别实验三条（都用同一份二进制）：非流式 12 并发**不崩**（15 ok / 0 err）；流式 12 并发 + `ALOFA_BATCH=0`（批路关闭、不排队）**照样崩**；流式 12 并发 + `ALOFA_SHARDS=1`（decode 不分片）**照样崩**。所以：**与批调度无关、与并行 decode 无关**，在"流式 + 多连接"这条路上
  - ❌ **一处推断被证伪，改动已回退**：我一度认为是"槽位还回去得太早 —— 调度器要到下一拍才把走完的请求移出活跃集合"（服务日志里有 `the scheduler asked to decode a request that is not resident`）。据此改了 `ModelService.stream_end`（`release` 之前无条件跑一拍 `_batch_tick`），并加了一道门（连开 12 条流、每条走完就收尾）。**变异验证门不红** —— 把改动退回原样，那道门照样 PASS。所以那个推断没有证据，改动与门**都已撤销**（不留没有证据的代码与门）
  - 剩下的可疑点在 `srv/engine_thread.mojo` 的 `Mailbox`：`jobs` / `results` 两个 `List` **是有 `Mutex` 护住的**（不是忘了加锁），所以要看的是 `Job` / `Result`（`Job` 里带了请求的原始字节 `raw`）**进出队列时的所有权** —— 含 `List` 的结构体在 push/pop 之间若被拷贝而非转移，堆会被写坏，而症状正是"某次 `List` 扩容时 SIGSEGV"。**未查**
  - ⚠️ **Gate P3 仍然被它堵着**：这不是"没跑过门"，是"一跑到量就崩"
- **2026-09-23（同日第六条）** —— **最小复现已经建起来，崩溃是「帧」推着走的，崩溃点精确到函数**
  - **最小场景**（`/tmp/mini_curl.sh`，curl + 后台进程，不用 `stress_serve.py` —— 它服务一死就产生几十万次重连，把要量的东西淹掉）：`并发=1`（**不需要并发**）× 顺序 40 条 × `max_tokens=1`。服务在第 **26 条 / 82 帧**处哑掉，随后进程消失
  - **帧驱动**（决定性判别，同一并发、只改每条流的帧密度）：`max_tokens=1` → 26 条 /**82 帧**；`max_tokens=8` → 8 条 /**77 帧**。请求数差 3 倍，**累计帧数几乎不变（82 / 77）**。所以它是被"每帧过一次队列"推着走的，阈值约 **80 帧**；与连接数无关（单连接顺序就能触发）
  - **崩溃点**（gdb 跑**最小场景**，最安静的条件下复现）：`List::_realloc` ← **`Conn::enqueue`** ← `Loop::run_threaded`（reactor 线程）。`enqueue` 往 `Conn.outbound`（`List[UInt8]`）逐字节 append，**每帧一次** —— 与上面的"帧驱动"对上了
  - **关于"修所有权还是换定长环形缓冲"的判别（已可作答）**：`Conn.enqueue` 的调用点**全在 `loop.mojo`（reactor 线程）**，engine 线程不碰 `Conn.outbound`（它把响应经 `Mailbox`（有 `Mutex`）交回来）。所以 `outbound` **不是竞争现场，是受害者** —— 堆是被别处写坏的，它只是第一个撞上的分配。**换环形缓冲不修根因**（只是把受害者从 `outbound` 挪到别处），要修的是堆损坏的**源头**
  - 一次判别失败已记录：`ALOFA_ENGINE_THREADS=0`（想让生成回到 reactor 线程那条路，以此断定源头在不在 engine 线程侧）**服务起不来**，这个取值不被支持，**没判成**
  - 下一步（按性价比）：`MALLOC_CHECK_=3 ./target/serve` 跑同一个最小场景 —— glibc 会在堆被写坏的**那一刻**就报错（`double free` / `invalid pointer`），比等到第 80 帧的 `realloc` 崩溃更接近源头
- **2026-09-23（同日第七条）** —— **范围收窄到「流式特有」，两条假设被实验证伪**
  - ✅ **非流式 120 条请求全部成功**（120/120 含 `finish_reason`，每条 254 B），服务存活。非流式每请求也要过一次同样的结果交接，所以**"结果交接次数"不是驱动量** —— 根因在**流式特有**的那条路上（每帧一次的 `enqueue` / SSE 帧序列化 / 每帧一次的引擎 `decide`+`settle`）
  - ❌ **假设一被证伪**：`Mailbox.take_result` 用 `self.results.pop(0)`，而 `Result` 里装着一个 `List[UInt8]` —— 一度怀疑"搬移含 `List` 的结构体时内部指针被复制而非转移"。写最小程序实测（`/tmp/pop_probe.mojo`）：4 条各 1000 字节、填充值 1/2/3/4，`pop(0)` 后**逐条求和全部正确**（1000/2000/3000/4000）。**`pop(0)` 的搬移是干净的**，不是它
  - ❌ **假设二被证伪（工具层面）**：`MALLOC_CHECK_=3` + `MALLOC_PERTURB_=165` 跑同一个最小场景，glibc **一个字都没报**。原因是崩溃栈的 `#0` 在 `libAsyncRTRuntimeGlobals.so` —— **Mojo 用的是自己的分配器**，glibc 的 malloc 钩子不在路径上。所以这一类环境变量（以及靠它吃饭的工具）在这台机器上对 Mojo 无效
  - 已排除的项（累计）：批路（`ALOFA_BATCH=0` 照样崩）、并行 decode（`ALOFA_SHARDS=1` 照样崩）、多连接（**并发=1 顺序就能崩**）、结果交接次数（非流式 120 次无恙）、`List` 搬移语义（实测干净）
  - 待查（流式特有且每帧一次的东西）：① `Conn.outbound` 的反复**部分写出**（`sent` 前进 + `pending == 0` 时 `clear()` 归零 —— 这条路非流式不走）；② 每帧的 SSE 帧序列化；③ 每帧一次引擎 `decide`/`settle` 里的分配（若有泄漏，流式跑的时候 RSS 会单调涨 —— 用 `/proc/<pid>/status` 采样就能看见，便宜）
- **2026-09-23（同日第八条）** —— **RSS 采样：没有泄漏；发送队列状态机是对的；顺手发现一个"没人跑"的门**
  - ✅ **没有泄漏**：流式跑时采样 `/proc/<pid>/status` 的 `VmRSS`（0.3 秒一点，989 点）：2947340 → 2951564 kB，一次跳变后**完全平**。所以"每帧一次引擎 `decide`/`settle` 里的分配泄漏"**排除**
  - ⚠️ **崩溃是概率性的，不是固定阈值**：同一最小场景这次只撑到 **8 条 / 27 帧**（上一次 26 条 / 82 帧）。之前"约 80 帧"的说法要修正为 —— 帧越多越容易撞上，但撞在哪一帧是随机的。这更像"堆元数据被写坏之后的随机崩"，而不是"数到某个值就坏"
  - ✅ **reactor 不碰引擎**：`Loop.dispatch` 只做 `mailbox.push_job` / `has_room`，引擎只被 engine 线程碰 —— "两个线程抢引擎"排除
  - ✅ **发送队列的状态机是对的**：补了一条**流式独有**的用例（写了一半 → 又来一帧要入队 → 接着写；非流式一次排完一次写完，不走这条路）。**变异验证门会红**（让新帧从头覆盖 → `FAIL`），还原后 16/16 绿。所以 `Conn.outbound` 的 `sent` / `clear()` **不是根因**
  - 🔧 **顺手修了一个缺口**：`tests/unit/test_conn.mojo` 一直存在，但 `pixi.toml` 里**没有跑它的任务** —— 是个没人跑的门。已挂上 `test-conn`
  - 剩下的方向：崩溃栈的 `#0` 在 `libAsyncRTRuntimeGlobals.so`（**Mojo 的异步运行时**，不是我们的代码）。下一步该问的是"每帧是否创建了一个异步对象/协程"，或者用 `-O0 -g` 在崩溃点直接打印 `outbound` 的 `len` 与 `sent`
- **2026-09-23（同日第九条）** —— **流式崩溃：只在 -O2 下出现，-O0 下不复现 —— 不是运气，是优化级别**
  - 同一份源码（含 `stream_end` 那处修复）、同一个最小场景（并发 1 × 40 条 × `max_tokens=1`）：
    - **`-O2`**：14 条 / 46 帧后死，第二轮 0 条 ✗
    - **`-O0` + gdb**：**40 条 / 120 帧全部通过，服务还活着** ✓
  - 所以之前"概率性"的说法要再修正一层：帧数决定的是**暴露机会**，而**崩不崩由优化级别决定**。这是 UB / 优化相关的特征，不是"数到某个值就坏"
  - ⚠️ 方案 1（`-O0 -g` 在崩溃点打印 `outbound` 的 `len`/`sent`）**没能执行到** —— 因为 `-O0` 下根本不崩，没有崩溃点可打印。这个"打不出来"本身就是结论
  - 结合崩溃栈 `#0` 在 `libAsyncRTRuntimeGlobals.so`（Mojo 运行时）+ `List::_realloc`：指向 **-O2 下某段 unsafe 代码被错误优化，或我们违反了别名/生命期规则**（后者在 -O2 才暴露）
  - 下一步候选（按便宜排序）：① 试 `-O1`，缩小到具体优化档；② 把流式每帧都要走的 `pending_view`（`unsafe_ptr().unsafe_offset(sent)` + `origin_of`）改成先拷一份，看 -O2 下是否还崩 —— 这条能直接验证假设，但它会改变零拷贝的设计，属于取舍，先不定
- **2026-09-23（同日第十条）** —— **两个判别都做完了：`-O1` 也崩；`pending_view` 不是根因**
  - **优化档**：`-O0` 不崩（40 条 / 120 帧全过）、**`-O1` 也崩**（6 条 / 18 帧）、`-O2` 崩（14 条 / 46 帧）。所以不是某个 `-O2` 特有 pass 的锅，是**一开优化就崩** —— 这反而更像我们能修的东西（我们的 UB 在优化下暴露），而不是编译器 bug
  - **`pending_view` 排除**：把它从"指向 `outbound` 内部缓冲的零拷贝视图"改成"先拷进 `scratch` 再给出去"（签名一起改成 `origin_of(self.scratch)`），`-O2` 下**照样崩**（7 条 / 24 帧；第二轮 14 条 / 42 帧）。实验改动已还原，零拷贝设计保持不变
  - 下一步：**查 `chunk_text_json`**。它是最早那次 gdb 栈里明确出现过的帧（`String::_add` ← `chunk_text_json` ← `ChatHandler::stream_next`），而且**每帧都走** —— 比继续猜别处更有依据。另一条栈是 `Conn::enqueue` ← `List::_realloc` ← Mojo 运行时，两个栈不同说明崩溃点是**随机落点**（堆损坏的典型表现），所以别再追"崩溃在哪一行"，要追"谁写坏了堆"
- **2026-09-23（同日第十一条）** —— **流式崩溃的根因假设：同一条流被两条线程同时推进（高置信度，尚未修）**
  - 证据链（都是代码，不是猜）：
    1. `Loop.progress`（reactor 主循环）：有流在途 → `step_stream` → `handler.stream_next(slot)` → 里面 `self.service.stream_next(request)` = **生成一步**，即**生成跑在 reactor 线程**
    2. `Loop.dispatch` 同时把 `JOB_STREAM_STEP`（每帧一帧）塞进 mailbox → **engine 线程也推进同一条流的帧**
    3. `dispatch` 推进前有 `if self.inflight[slot] == 1: continue`（一帧在途就不再派）；而 **`progress` → `step_stream` 完全没有这个检查**
  - → engine 线程正在推进 slot 那一帧（`inflight=1`）的同时，reactor 也在推进它：**两条线程同时对同一条流做一步生成**。共享的是 handler 的流状态（`stream_phase[slot]` / `stream_count` / `stream_id`…，作者在 `progress` 的注释里自己写了"handler 的流状态只有一份"）和引擎 —— 都没有锁
  - 与全部观测吻合：**每帧驱动**（每帧一次 `dispatch` + 一次 `progress`）、**崩溃点随机落点**（竞争窗口随机，所以一会儿 `chunk_text_json` 的 `String::_add`、一会儿 `Conn::enqueue` 的 `List::_realloc` —— 两处都是碰分配器的受害者）、**`-O0` 不崩而 `-O1/-O2` 崩**（优化下的重排/缓存放大竞争后果）、**非流式不崩**（非流式不走 `progress` 的流分支，也没有每帧的 `JOB_STREAM_STEP`）、**单连接也崩**（不需要多连接，reactor + 1 条 engine 线程就够）
  - ✅ 排除项（都已验证过）：RSS 无泄漏；`Conn` 的 `sent`/`clear()` 状态机正确（有门，且变异验证会红）；`pending_view` 零拷贝视图不是根因（改拷贝版照样崩）；`chunk_text_json` 是纯 `String` 拼接，自己写不出界；reactor 的 `dispatch` 不碰引擎
  - ⚠️ **还没修**：这是架构级决定（生成到底该在哪条线程上），三条路需定夺：① `progress` 推进前也看 `inflight`（让 engine 线程独占那一帧）；② 干掉 `dispatch` 里的 `JOB_STREAM_STEP`（若生成本就该在 reactor 同步跑）；③ 让 `step_stream` 只取结果、不触发生成
- **2026-09-23（同日第十二条）** —— **engine 线程那条路确认了：默认配置下它与 reactor 共用同一份 handler**
  - `_run_job`：`JOB_STREAM_STEP` → `_stream_step(handler, job)`（engine 线程跑一帧）
  - ⚠️ 而 `engine_thread.mojo:560` 的注释明写 —— **0 号线程用的是调用方那一份 handler**（"它归调用方管，而调用方的生命周期本来就盖到 join 之后"）。默认 `ALOFA_ENGINE_THREADS=1` → 只有 0 号 → **engine 线程 #0 与 reactor 共用同一份 handler/service，无锁**
  - 所以上一条的假设成立，但机制要修正一层：不是"所有 engine 线程都共享"，而是**默认的单线程配置下恰好 0 号复用了调用方那份**（1 号起才是 `spawn_twin` 的副本）
  - **设计意图有据**：`_answer` 的注释写着"流式只交头，帧由 `_stream_step` 一帧一帧给" → 帧本该**只**由 engine 线程给。reactor 侧 `progress` → `step_stream`（注释"一次一条流，handler 的流状态只有一份"）是早于 engine 线程的旧路径 —— 两条并存 = **双重推进**
  - → 修复应走③（reactor 不再触发生成，只把 engine 交回的字节发出去）。**动手前先确认 reactor 已有"把 `Result` 的字节排进发送队列"的路径**，否则摘掉 `step_stream` 的生成调用等于断流
- **2026-09-23（同日第十三条）** —— **⚠️ 第十条的"两条线程同时推进"假设被实验推翻了**
  - 做了什么：把 `Loop.progress` 里"有流在途就 `step_stream`"那段摘掉，让帧只由 engine 线程的 `JOB_STREAM_STEP` 给
  - 结果：服务**哑了** —— 第 1 轮只 1 条 / 4 帧，第 2、3 轮 0 条。流**停住**
  - → 实际运行中**只有 reactor 在推进流**；engine 线程的 `JOB_STREAM_STEP` 这条路**没有真正生效**。既然只有一条线程在推，**就不存在竞争** → 第十条那条假设不成立，崩溃不是数据竞争
  - ⚠️ 顺带发现一个真问题：`engine_thread.mojo` 的 `JOB_STREAM_STEP` / `_stream_step` 很可能是**没接上的死路径**（`loop.mojo:397` 那句"接下来由 `dispatch` 一帧一次往返地喂它"与实际不符）。留着它会误导下一个人 —— 它已经误导了我一次。要么接上，要么删掉
  - 改动已 `git checkout` 还原，服务恢复（改前 40 条能过若干条，改后 1 条就停）
  - 回到原点：现在是**单线程**内的 UB（`-O0` 不崩、`-O1`/`-O2` 崩、崩溃点随机落、每帧驱动）。该找的是"每帧一次的**越界写**"，不是竞争
