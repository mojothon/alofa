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

**最后更新**：2026-09-29（**AVX2 的 q4 通路换成整数点积并补两道门**；2026-09-18 那轮是 **N/M 快照不再腐烂** —— 手写的测试条数全部升级为机器核验的 `evidence:…?count=N`，新增 `check-counts` 门与 `ledger-sync`；同时订正本轮暴露出的三处漂移）

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
| **量化 q8_0（量化步 / 解量化 / 融合 matmul）** | `verified` | `evidence:tests/unit/test_q8_parity.mojo?count=9`（`lm_head.weight` 前 64 行真实权重，57344 值 / 1792 块；参照物由 `scripts/dump_q8_reference.py` 独立另写）<br>判据与 q4_0 那一行同一套：**解量化零容差**（8 位码与 fp16 缩放都精确，差一点就是读错字节）、**量化逐字节**、matmul 沿用 `1e-5 × max(1,\|ref\|)`（fp64 累加，期望值用**解量化后**的权重算，故只盯累加顺序）<br>三条常驻负向对照：① **忘符号扩展** —— q8_0 存的是**补码**，与 q4_0 的无符号 nibble 减 8 不同，照 q4 的习惯读会让负权重整体跳到 +128 那侧；② **取整改截断** —— 字节必须与参考不同；③ **块步长 32 vs 34** —— ⚠️ 这条**不能**走 `block_scale` 比浮点：错步长读到的两字节可能不是合法 fp16（实测 half_bits=32740 逼近 inf），会直接抛错，那是崩溃不是判红 ⇒ 比的是**字节**<br>**相对 L2 实测 0.5359%**（q4_0 是 10.3%）⇒ 门里断言 < 1%，作用是把 **q4_0 挡在输出投影外面**（`lm_head` 压到 q4_0 会让教师强制贪心一致率 0.8691 → 0.77）<br>**已接进 `lm_head`（2026-09-28 当日）**：`enable_q8_output()` + `q8_matmul_k_shards[backend]`（按输出行切 `shards` 片，与 fp32 通路**同一个**分片数，两档才是在比权重宽度）；⚠️ tied embedding ⇒ **embedding 查表也改读块流**（只改 matmul 一侧会让 prefill 把 q8 字节当 fp32 解释，且**只在 prefill 上错**、decode 看着完全正常）<br>**端到端实测（同机同 prompt，`shards=8`）**：q4 档 `13.72…15.38 → 17.70…20.23 tok/s`（**+30%**），对 llama.cpp 的 25.60 由 **0.60× 提到 0.79×**<br>⚠️ **第一次复测是负的**：标量 `matmul_q8_f32` 反而掉到 `7.12…9.36`（−22%）—— 379 MB/token 的省流量被 136M 次**标量**乘加吃回去；补了 AVX2 版 `_matmul_q8_vec` 才兑现。教训：**"减少每权重字节 ≠ 提速"的前提是计算足够快**<br>质量门 `evidence:tests/unit/test_q8_head_greedy.mojo`（重资产，`pixi run test-q8-head`，4 条）：教师强制贪心一致率 **0.8691 → 0.8672**（512 步只掉 2 步；对照：输出投影压 q4_0 是 → 0.77），标量 / 向量两条后端**逐位相同**<br>⚠️ 未尽：还差 27% 够到 25.60；q8 向量核尚未试按 `j%8` 切 f64 通道的写法 |
| **整网 q4_0 前向通路**（24 层全部投影走块流） | `verified` | `evidence:tests/unit/test_q4_greedy.mojo`（1/1：4 条 prompt × 128 步教师强制贪心，与 fp32 参考一致 **445/512 = 0.8691**（2026-09-22 起；**之前是 418/512 = 0.8164，那时量化分支里有一个 4 倍错位的行偏移 bug**，见今天变更日志））<br>门是 0.75 的下限，**它判的是"通路在工作"不是"质量达标"**：高低半字节装反会得到 0.0，前向悄悄退回 fp32 会得到 1.0，两头都被抓住（后者另由 `model.q4_enabled` 直接断言）<br>⚠️ **路线图里"一致率 ≥ 0.90"那条没过**，见下一行；本轮的 0.75 是"算的东西是对的"的下限，不是把 0.90 改小；⚠️ 0.75 那道门**看不见**上面那个 bug（0.8164 也在它的绿区里），所以同一份测试里已补了一道 **0.85 的回归下限**（`Q4_REGRESSION_FLOOR`） |
| **AVX2 q4 投影的整数点积通路（`vpmaddubsw`，AVX2 下的默认）** | `verified` | `evidence:tests/unit/test_q4_mub_greedy.mojo`（整网质量门，重门 `pixi run test-q4-mub`，不并进 `pixi run test`：4 条 prompt × 128 步教师强制贪心，与 fp32 参考一致 **444/512 = 0.8672**；默认 f32 展开是 445/512 = 0.8691 ⇒ 只差一个 token，0.85 回归下限过）<br>`evidence:tests/unit/test_q4_mub_rows.mojo?count=5`（核级差分门，已并入 `pixi run test`，不要权重：相对 L2 ≤ 1%（实测 0.33–0.38%）+ **两条通路不许逐位相同**（"开关没接到这条路上"就长这样）+ 守卫区及其红测）<br>⚠️ **只覆盖 AVX2**：标量后端没有整数版；`ALOFA_Q4_MUB=0` 退回 f32 展开<br>⚠️ 0.8672 是**逐 (行, 块) 缩放**之后的数；整批共用一个激活步长那版是 **0.8184**，红在 0.85 上<br>端到端（shards=1，`q4_0+q8_head`，同轮 A/B）：decode **10.78…12.95 → 13.79…15.61 tok/s**，prefill **281…381 → 155…238 ms** |
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
| reactor 线程 + engine 线程池（3.2b 独占线程 → 3.2c N 条并发生成） | `verified` | `evidence:tests/unit/test_engine_thread_server.mojo`（重门，`pixi run test-engine`：真 socket + 真 fork + **真线程**，故意不进 `pixi run test`）+ `evidence:tests/unit/test_conn.mojo?count=16`（`take_with_raw`：跨线程过的是原始字节）<br>**后半（engine 独占线程）2026-09-22 落地**：前向搬到一条独占线程（`ThreadHandle` + mailbox），reactor 上只剩 I/O —— 调度决策仍留在 reactor（可重放、零锁），箱体只有一个 `Mutex` 且临界区只有 append/pop（里面没有 I/O，也没有前向）。<br>**判据是「生成期间这条循环还在 accept」**：槽位被占满时新连接会被**接进来立刻关**（`MAX_CONNS` 那条规则：留在 backlog 里同样会空转，而「接了又关」至少让对端立刻知道），对端因此能靠 **EOF** 看见这件事 —— 探子从连上到收到 EOF，`run_threaded` **0 ms**，`run` **4.5 s**（那 5 s 的生成它全等完了；`run` 是**常驻负向对照**：它快了就说明这条门没盯住那条性质）。<br>⚠️ **为什么不用「生成期间还有字节在走」**（直觉上更直接，实测走了两遍都不成）：服务端一次能排队的字节被 `SEND_HIGH_WATER`（1 MiB）封住，而 loopback 上内核给一条 socket 的缓冲实测能吃下 **1.2–2.5 MB**（自动调优，随发送速率变）—— 服务端手上一字节不剩，客户端**不需要循环参与**就能把响应读完（实测两种模式一样快：内联那侧 38 ms 收完 800 KB）；读那一侧同样被挡住（`MAX_BODY_BYTES` 也是 1 MiB，而限速发送实测能被吸收 1.26 MB）。那个数字由**内核**决定，不由被改动的那一处决定 —— 而 accept 内核替不了应用做<br>⚠️ **它换来的不是并发生成**（这句是 3.2b 当时的状态）：engine 只有一条线程、一次一个 job，所以「慢生成期间另一条连接发一个请求」在两种模式下**都会**等（health 排在慢请求后面）。拿后者当判据的话，一条请求都没答的实现也能「看起来很快」。<br>**2026-09-22 3.2c 已把它扩成线程池**：`engines>1` 时 N 条线程各领一份 handler（`Twinable.spawn_twin`，**一份 = 一份权重**），同时能生成 **N** 条。判据是「**两条 5 s 生成的总墙钟**」—— `engines=2` **5002 ms**、`engines=1` **10002 ms**（后者是**同一条代码路径**上的常驻负向对照，它不慢就证明不了门盯住了什么）。代价 **N 条 = N 份权重**（尚无「共享只读权重 + 各一份 KV」那一层）→ **默认 0**（生成就在 reactor 循环上，串行），由 `ALOFA_ENGINE_THREADS`（0–16）开；多 worker 下是 `workers × engines` 份。<br>**三条约束各有安排**：流的**相位亲和**靠槽位→线程的**黏性 `owner`**（第一次派活定归属，之后不变）+ 单槽位单在途；同连接**按序**同上；**邮箱上界不破**靠**单队列按线程过滤**（每线程一队列会把一个 `MAILBOX_CAP` 悄悄变成 N 个上界）。<br>**跨线程只过一种形状的东西**：请求以**原始字节**过线程（`HttpRequest` 带着入站缓冲的借用关系，挪不进邮箱；“从结构体中间挪走一块”在这个编译器上也过不去），engine 侧重新解析一次；应答与流帧则是一段段字节加一个 kind。<br>**三条不能少的记账**：邮箱有上界（`MAILBOX_CAP`，**派之前先问 `has_room()`**：派了又失败等于把请求丢了）；每个槽位带 `gen`（结果回来时槽位已换主人就丢掉 —— 否则应答会发给**另一条连接**）；`inflight` 记着「这条连接在等生成」，空闲超时不许收它。<br>**流的头单独一个 kind**（`RES_STREAM_HEAD`）：一帧也是 `RES_BYTES`，合成一个的话循环不知道该开始喂帧 —— 症状是客户端只收到一个响应头，然后挂到空闲超时。<br>**生命周期**：放 mailbox 的那个值在 `join` **之后**必须再被碰一次（ASAP 析构会让它提前死，子线程就用上了已释放的锁 —— 症状是子进程 `dumped core`）<br>⚠️ **2026-09-28：这条路被标成实验，默认不开**。真模型 + **客户端中途断开**会把请求搞坏（被踩的是 `Job` 的缓冲头 8 字节，而那个 `Job` **从来没被析构过** ⇒ 是别人 free 了它）。八轮排查把业务代码里的机理**全证否**了（`Job`/`Result` 双重归属、`List` 扩容搬家、号复用、`stream_end`/`cancel` 链、`Arena` 越界 —— `Arena` 是 mmap 且 `alloc` 有边界检查，踩不到 glibc 的堆；Mojo 运行时的分配器本身也不犯案：3 线程 × 250 轮 × 每次 20 块分配、64 块常住在世，全干净），剩下的硬事实只有一条：**必须有客户端断开**。⇒ `>=1` 时服务启动会当场打 `[warn]`。<br>⚠️ 本行那扇门**测不到**这个错：它用的是不带权重的替身，而错在真模型那条生成路径上（替身版本的"中途断开"照样踩，但那是另一件事）。同一天给那扇门加了第 4 条性质：**三种模式各"中途断开 3 次"，之后探活必须有应答且答的是它自己的**（判据不是"进程还在" —— 哑掉的时候进程照样在）；探活耗时 1 engine ≈13 s / 2 engines ≈3.8 s，是"那些断开确实落在生成中"的证据（接近 0 就说明那段什么都没测） |
| libc 事件循环原语（自研**回退**路径） | `verified` | `evidence:tests/capability/test_libc_ffi.mojo?count=5`：`socket`/`epoll_create1`/`timerfd_create`/`eventfd`/`SO_REUSEPORT` |
| 阻塞式 TCP 传输（`flare.tcp`） | `verified` | `evidence:tests/capability/test_deps.mojo?count=7`：socket / bind / listen / accept / TCP_NODELAY 全是 flare 的（A7：依赖而不自研）。<br>⚠️ 与上面 reactor 那一行不是一回事：这里用的是**阻塞**接口，没有事件循环 |
| reactor 事件循环（**一条循环 N 条连接**） | `verified` | `evidence:tests/unit/test_loop_server.mojo`（重门，`pixi run test-loop`：真 socket + 真 fork，**故意不进 `pixi run test`**）+ `evidence:tests/unit/test_conn.mojo?count=16`（连接机，无 socket，逐字节）+ `evidence:tests/capability/test_reactor.mojo?count=4`<br>**它替换的是「一 worker 一连接」**：一条连接上的 `send` / `recv` 不再挡住别人 —— 8 条连接同时开着同时收应答，一个「发了一半就停住」的对端让别人增加的延迟是 **0–1 ms**（旧形态会停满 5 s 接收超时，所以这条门带 2 s 上限，并备有把服务换回 `srv/server.mojo` 的负向对照：实测必红）。<br>**它没改「一次只生成一条」**（这句记的是 3.2 当时的状态）：请求的前向那时候在循环里**内联**执行，占住这条循环 —— 3.2b 已把前向搬到独占线程，3.2c 已把它扩成线程池（同时能生成 N 条，见上一行）。换来的不是吞吐，是「连接的 I/O 不再互相挡路」。<br>三条容量线各有判据：槽位上界 `MAX_CONNS`（满了就接了立刻关 —— 让对端马上知道，而不是等一个不来的应答）、出站队列上界（背压，见下行）、空闲连接 30 s 收（**只收真没事干的**：有完整请求在排队等生成的、还有字节没写出去的、正在写流的三条例外 —— 收错方向会让一个慢请求变成一次看起来随机的失败）。<br>⚠️ 边界：**一个坏对端只带走它自己**（畸形请求 → 丢该连接、循环继续，不会让服务退出）；不验 TLS；**engine 已独占线程**（见上一行）：循环上只剩 I/O，前向在另一条线程上 |
| HTTP/1.1 解析与连接状态机 | `verified` | `evidence:tests/unit/test_http.mojo?count=14`<br>线格式写成**纯函数**（字节进、结构出），于是能逐字节钉住：请求行必须正好三段、头值两侧的 OWS 要清掉、同名头取**第一个**（取最后一个等于让 `Content-Length` 变成可以被对端追加的头）、体**只**是 `Content-Length` 说的那么多字节（多出来的属于下一个请求）、`Content-Length` 数的是**字节**不是字符、头的结束只认 `CRLF CRLF`（认裸 LF 会把一个走私请求切成两个）。<br>⚠️ **传输层是 `flare.tcp` 的阻塞接口，线上格式是自研的**：`flare.http` 绑在它自己的 reactor 上（一次 `serve()` 按 `num_workers` 拉起线程），而这一版要的正是「一个进程、一条连接、阻塞」—— `srv/server.mojo` 的文件头写了这个取舍。要并发与背压时正确的动作是换成 `flare.http`，而不是把这里长大 |
| 单进程 HTTP 服务（非流式 + SSE 流式，真 socket、真 fork） | `verified` | `evidence:tests/unit/test_http_server.mojo`（重门，`pixi run test-http`，**故意不进 `pixi run test`**：它 fork，而 JIT 下 `fork` 会崩编译器）<br>钉的是**服务循环**的性质，纯函数单测钉不住：一条 keep-alive 连接上按序回答多个请求、且流水线里多出来的字节不丢；`Connection: close` **在响应里也写 close**（只在写完以后默默关，对端会往一个正在关闭的 socket 上发下一个请求 —— 那是一次看起来像「偶发连接重置」的失败）；一个发了一半的请求在接收超时后**只关它自己那条连接**、不能带走进程 —— 那是单进程形态最致命的失败方式。<br>⚠️ **这一行描述的是 `srv/server.mojo` 的旧形态**：阻塞 accept + 一次一条连接。自 P3.2（2026-09-22）起**线上入口已换成 reactor 循环**（见上面那行）；旧循环仍然在 —— 它是 reactor 门**常驻的负向对照**。无 TLS；并发与背压的现况见上面两行
| 发送队列与背压 | `verified` | `evidence:tests/unit/test_conn.mojo?count=16`<br>出站队列有**上界**（`SEND_HIGH_WATER`），写不出去就**停止读这条连接** —— 「背压」的意思不是「有队列」，而是队列满了要让上游停下来；只测「有字节排队」测不到这一点（队列可以无限长）。<br>一次 `send` 写一半是常态而不是错误：部分写按已写字节数**就地推进**，且**不许越过队列里有的**（越界的下标会把别人的内存发出去 —— 症状是一次「偶发的响应内容不对」）。要 `Connection: close` 的连接等队列**排空**再关（先关再排 = 对端收半个响应）。<br>⚠️ 边界：钉的是连接机（无 socket）；真 socket 上的取值与效果见 reactor 那一行 |
| SSE 流式输出 | `verified` | `evidence:tests/unit/test_sse.mojo?count=8`（帧格式，逐字节：头**不许**带 `Content-Length`（写第一帧时长度未知）、`CRLF CRLF` 恰好一处、**含裸换行的 JSON 必须被拒绝而不是被「修好」** —— 静默吃掉换行会把「内容里有个换行」变成「客户端少收一帧」，那是从外部查不出来的错）+ `evidence:tests/unit/test_http_server.mojo`（重门，真 socket：一条完整流的**线上字节** —— 头无长度、帧齐全、以 `data: [DONE]` 收尾、然后关连接；客户端中途断开只丢这一条连接、下一个请求照答）+ `evidence:tests/unit/test_workers.mojo`（多 worker：fork 出来的每个 worker 分帧与单进程一致，12 条连接每条一条完整流）。<br>定界方式：SSE 响应没有 `Content-Length`，靠**关连接**定界 —— 没有用 chunked（SSE 客户端按 `data:` 行切分，再套一层长度前缀只是多一处可以写错的地方）。<br>**每一条路都有终点**：`stream_begin` 的错（prompt 分词为空 / 过长）在响应头**之前**回 400（前向是**惰性**的，第一次 `stream_next` 才 prefill，校验才退得回去）；头发出去之后出错则给一帧 `error` 再 `[DONE]` —— 客户端得能区分「生成完了」与「生成坏了」，否则两种都表现为「收到若干帧然后连接关了」。客户端中途断开：写失败只丢这一条连接、进程继续（实测 `[srv] stream aborted: BrokenPipe` 之后下一个请求照答 ——这条同时是 SIGPIPE 的照妖镜，一个断开的客户端不该把 worker 带走）。<br>**文本与非流式逐字节相同**：每步用整段前缀解码再取新增的那一截，而不是把这个token 单独解码 —— 多字节字符可能跨两个 token，单独解码时前半截是不完整的 UTF-8，会被 `decode` 丢掉。实测（真权重、贪心）：英文与中文两种 prompt，流式拼出来的文本与非流式**逐字相等**。<br>官方 `openai` SDK（1.109.1，只改 `base_url`）流式对话跑通：14 帧 = 1 个 role 帧 + 12 个内容帧 + 1 个 finish 帧，`finish_reason` 与非流式一致（手动冒烟，不进门的证据）。<br>⚠️ 边界：同时能**流**的条数 = worker 数。P3.2 修的是**连接级**并发（连接不再互相挡路），但流共用一个 handler 状态、且生成仍串行 —— 要「同时生成 N 条」需要 engine 独占线程 + 批量调度，仍 `missing`；未验 `stream_options.include_usage`（未知键照旧跳过）、断线重连与代理缓冲；**没有任何性能数字进账本**（逐帧 `write_all`，~3 tok/s 量级下无瓶颈，未做批量合并） |
| OpenAI 兼容 API（非流式 + SSE 流式） | `verified` | `evidence:tests/unit/test_openai.mojo?count=33`：请求体的扫描、响应体的转义、路由（`/health`、404、405、`stream=true` → 流式，帧序列见下面 SSE 那一行）、`finish_reason` 的两条路。<br>**「半流式比明确拒绝更糟」这条纪律没变，只是换了个守它的地方**：要么从头到尾给完（以 `[DONE]` 收尾），要么在响应头之前就拒绝 —— 中途失败也必须给流一个终点，见下面 SSE 那一行。<br>JSON 是自扫的而没用 `json` 包：这里要的是**缺字段必须有名字**（`messages` 是空数组 → `invalid_request_error`，而不是「生成了空字符串」），而通用解析器会把缺字段变成运行期空值。转义必须还原成真字节（`u` 加四位十六进制 → UTF-8 字节），代理对与指数记法指名拒绝 —— 半个字符编不出 UTF-8，悄悄丢掉才是真的糟。<br>⚠️ **还没有 chat template**：`messages` 按出现顺序用换行拼起来，`role` 读掉但不参与；这件事随 `/health` 的 `notes` 一起返回，不读源码的人也看得见。⚠️ 边界：未验 `logprobs` / `tools` / 多候选 |
| SO_REUSEPORT 多 worker | `verified` | `evidence:tests/unit/test_workers.mojo`（重门，`pixi run test-workers`，fork → 先 build 再跑）：3 worker 各自`bind_reuseport` 同一端口，12 条连接 × 3 请求（健康检查 + 聊天 + 一条完整 SSE 流）全部 200 且分帧与单进程版**逐字段一致**；响应 id 从每个 worker 自己的 1 起计 →「chatcmpl-1 出现次数 = 接过活的 worker 数」（实测 3/3，内核四元组哈希分发）。⚠️ 权重在 fork **之前**加载 → 子进程 COW 共享：真权重 8 worker 冒烟（9 进程）PSS 合计 2874 MB ≈ 单进程 2874 MB（RSS 18.3 GB 是把 COW 页逐进程重复计入的记账假象）。⚠️ **fork 后 asyncrt 不可用**：worker 内分片 >1 时第一个前向**死等**到被 SIGKILL（shards=2/8 皆然）→ 多 worker 默认每 worker 单分片（`ALOFA_SHARDS` 可强行覆盖）；单进程不受影响 |
| master 健康探测 + 优雅退出 | `verified` | `evidence:tests/unit/test_workers.mojo`：master 不监听、只管进程（fork/回收/停止编排）；worker 把监听 fd `dup2` 到固定高位（900），SIGTERM/SIGINT 处理器只做一次 `close(2)`（异步信号安全表内、无锁无分配 —— Mojo 没有模块级可变全局，处理器带不了状态，状态由「fd 没了」本身承载），`serve_with_stop` 在 accept 前 / 每个响应后查标记。worker 现在跑的是 **reactor 循环**（不是阻塞 accept）：循环每轮（最多 50 ms 一次）都会回到标记检查处，标记自己就能让它退出；master 仍然发「唤醒连接」（连上即关；SO_REUSEPORT 按四元组哈希，一条只醒一个 → 发存活数 + 余量）—— 它们是给「等得更久」的形态留的兜底，多发几条没有代价（`pixi run test-workers` 仍绿：3 worker / 36 响应 / abnormal=0 forced=0）。宽限后仍不退的 SIGKILL 兜底，`forced`/`abnormal` 计数上报，任一非零 → master 非零退出（门会失败，不无声通过）。worker 运行期死亡 → master fail-fast 全停交给 systemd `Restart=` 接管（半套集群比快速重启难排障）。生产形态 systemd 模板 `scripts/deploy/alofa.service`（`TimeoutStopSec` 必须大于 `ALOFA_GRACE_MS`，否则 systemd 先 SIGKILL 整个组） |
| **批调度接进服务层**（一次前向推进多条流：贪心 + 采样 + 排队） | `verified` | `evidence:tests/unit/test_batch_stream.mojo`（重门，`pixi run test-batch-stream`，真权重、不并入 `test`）：批 N 条 vs 一条一条跑**逐 token 相等**；采样批 4 条 == 各自单跑；四条同 prompt 同种子的流必须互相一致（共用随机源必红）；10 条请求（> `MAX_BATCH`=8）不被拒绝而是**排队**且结果与单跑一致<br>`evidence:tests/unit/test_engine_core.mojo?count=18`（槽位用完要**显式** `release`：一整批跑完後第 9 条必须被接受）<br>⚠️ 边界：只收形状装得进引擎的请求（prompt ≤ 128、新 token ≤ 32），越界仍走单流老路；并发 ≤ 8 之外的请求在队列里等（上界 `MAX_CONNS`=128），不是拒绝 |
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
| `chat_template` 渲染（Qwen2.5 ChatML 一族） | `partial` | `evidence:tests/unit/test_chat_template.mojo?count=6`（10 个词条，**逐字节比文本 + 逐个比 token id**）+ `evidence:tests/unit/test_openai.mojo?count=33`（接线那一侧：缺 `role` / 缺 `content` / 未知角色都得拒绝，同一段文字换个角色必须给出不同的 prompt）<br>原先按出现顺序用 `\n` 拼 content、`role` 读掉不参与，多轮与系统提示的语义是错的；现在由 `alofa.tokenizer.chat_template.render_chatml` 按模板排：先吐一段 system（无首条 `system` 时用模板自带那句），随后 `user` / `assistant` 各一段，最后补 `<|im_start|>assistant\n`。首条 system **不会**在循环里重复出现（`loop.first` 那条，用出现次数钉住：3 个 `<|im_start|>` / 2 个 `<|im_end|>`）<br>**答案不属于本仓库**：模板来自 HF 缓存里 `Qwen2.5-0.5B-Instruct/tokenizer_config.json:chat_template`，由 `scripts/dump_chat_template.py` 用 **transformers 自己的 Jinja 引擎**导出成夹具 `tests/fixtures/qwen2.5-0.5b/chat_template.tsv`（自检：`tok(text) == ids`，不过就不写夹具）。所以模板改一句话，这道门立刻红<br>⚠️ **它现在是 `partial` 而不是 `verified` 的理由（重要）**：这里是**一族的渲染器，不是模板引擎** —— 正确做法（「语义来自模型的 `chat_template`」，借鉴 `molla`）要求模板字符串来自模型，而本仓库夹具的 `tokenizer.json` **不带** `chat_template` 字段，`tokenizer_json.mojo` 遇到它也是静默跳过。于是**换族群要改代码，而不是改数据**。<br>⚠️ 明确未做：`tools` / `tool_calls` 与 `tool` 角色（模板里走 `<tool_response>` 那一段）；`role` 只认 system / user / assistant，**其余指名拒绝**（`invalid_request_error`）—— 拼出去的话不会报错，只会让模型把一段工具输出当成人话读 |

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
| 与 llama.cpp 的同机同 prompt 对比 | `verified` | **唯一被接受的性能参考基准** —— **2026-09-28 首次做成口径对齐**（上次 2026-09-18 那次三条边界没对齐，故一直没接受）<br>`evidence:scripts/bench_vs_llamacpp.mojo`（alofa 侧，`mojo build -O2 -I src`；**prefill 与 decode 分开报**、打印 token id 供对账）<br>**对齐项**：同机（i7-9700K 8C AVX2，非共享机）· 同模型（Qwen2.5-0.5B-Instruct）· **同 prompt（两边 tokenizer 逐 id 相同：`[785,6722,315,9625,374]`，5 token）** · 同生成 32 token · greedy · 权重同源（HF → f32 GGUF → `llama-quantize` 出 q4_0）<br>**decode tok/s —— fp32 档**：alofa `5.83…7.92`（shards=8）/ `3.95…5.22`（shards=1） vs llama.cpp `5.51…5.52`（`-t 1`）⇒ **alofa 是 llama.cpp 的 1.06…1.43×**<br>**decode tok/s —— q4_0 档**：alofa `13.83…15.37`（shards=8）/ `6.65…6.80`（shards=1） vs llama.cpp `25.60 ± 1.18`（`-t 1`）⇒ **alofa 是 llama.cpp 的 0.54…0.60×**<br>⚠️ **这个 0.54…0.60× 不可读成"q4 内核差"** —— **同流量对照**（当日变更日志）：把 llama.cpp 的**输出/嵌入张量也拖回 F32**（层仍 q4_0），它从 `25.60` 掉到 **`13.44 ± 0.82`**，与 alofa 的 `13.83…15.37` **几乎重合** ⇒ **相同流量下 alofa 反而 `1.03…1.14×` 快**；差距 100% 来自**输出投影的存储精度**（alofa `lm_head` 是独立 fp32 副本 = 544 MB；llama.cpp 因 tied embedding 只存一份且压成 **Q8_0** ≈165 MB ⇒ 每 token 流量 655 MB vs 276 MB）<br>⚠️ **边界（不可省略）**：① **本机长期过载**（本轮 runq 2–17）⇒ 绝对值只作**同机同轮**相对比较，不可跨机；② llama.cpp 在本机 **`-t 8` 反而慢于 `-t 1`**（fp32 `1.54…3.34` vs `5.52`；q4_0 `14.71 ± 6.88` vs `25.60`，且 `-t 8` 抖动大）⇒ **主口径取 `-t 1`**，这是本机环境的观测，不是 llama.cpp 的普遍结论；③ alofa 的 q4 是**加载期量化**（缩放按 MSE 选），llama.cpp 的 q4_0 由 `llama-quantize` 从**同一个** f32 GGUF 转出 —— 块布局相同、**缩放选法不同** ⇒ **速度可比、质量不可比**（alofa 整网 q4 一致率仍 `missing`，实测 0.8691）；④ prefill 只有 5 token，粒度太粗（alofa fp32 `195…457 ms`，同档抖动 2.3×）⇒ **只报不判**；⑤ llama.cpp 侧用 `llama-bench -p 5 -n 32 -r 5`，它的 prompt 是**随机 5 token**而非这段文本（已用 `llama-tokenize` 证两边 id 一致，prefill 时间只与长度/形状有关，故可比） |
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
- **2026-09-23（同日第十四条）** —— **逐处排除（一）**
  - ✅ `escape_json`（`openai.mojo:583`）**排除**：每帧对每个 token 都跑，是最像"固定缓冲写爆"的地方。但它全程 `List.append`（自动扩容），`capacity=len(raw)+8` 是初始容量不是上界 —— 全 `\u00XX` 转义会超出它，靠 `append` 扩容兜住。末尾 `String(unsafe_from_utf8=out)` 消费 `List`（有所有权），不是零拷贝。无裸写
  - ✅ `srv/sse.mojo` **排除**：全文件无 `unsafe` / `InlineArray` / `capacity=` —— 帧构造是纯字符串拼接
  - **已排除汇总**：RSS 无泄漏 · `Conn` 的 `sent`/`clear()`（有门且变异验证会红）· `pending_view` 零拷贝（拷贝版仍崩）· `chunk_text_json`（纯拼接）· `escape_json` · `sse.mojo` · reactor 的 `dispatch` 不碰引擎 · "双重推进"（已证伪，实际只有 reactor 推进）
  - **还没查**（流式每帧走）：① `ChatHandler.stream_next` 对 `stream_phase` / `stream_count` 等**按下标写**（有 `_check_request`，但要核对它比的是不是这些数组的上界）；② `service.stream_next`（引擎侧取 token 那一段）
- **2026-09-23（同日第十五条）** —— **逐处排除（二）**
  - ✅ `stream_next` 的按 `request` 下标写**排除**：`MAX_CONNS = MAX_STREAMS = 128`（两者相等，槽位号不会越出流数组），且 `_check_request`（`openai.mojo:980`）比的是 `MAX_STREAMS` —— 正确的那个上界
  - ✅ `_new_text` / `bytes_to_text` **排除**（**曾是最像的候选**）：`_new_text` 每帧把整段前缀重新解码，`full` 是**局部变量**，再用 `bytes_to_text(full, sent, len(full))` 取增量 —— 若那是零拷贝视图，返回的就是指向一个已死局部变量的 String（悬垂，且与"-O0 不崩 / -O1 -O2 崩"完美吻合）。查了：`bytes_to_text`（`srv/http.mojo:57`）是**纯拷贝**（新建 List + `append` + `String(unsafe_from_utf8=out)` 消费它），不悬垂
  - **下一个候选**：`tokenizer.decode`（`_new_text` 每帧调它）—— 流式每帧唯一还没看过的一大块
- **2026-09-23（同日第十六条）** —— **逐处排除（三）：`tokenizer.decode` 排除；每帧路径已全部排除，该换工具了**
  - `decode`（`tokenizer/tokenizer.mojo:156`）：id 有边界检查（`id >= self.vocab.count()` → raise）；三段全是 `List.append` + `String(unsafe_from_utf8=...)` 消费 List —— 无裸写、无零拷贝视图
  - 唯一可疑点：`utf8_decode_at(raw, index)` 对不完整 UTF-8 可能读 `index+1/+2` 而不检查（**读**越界）。但那是读，我们要找的是"谁**写**坏了堆"；读越界的后果是解出错误码点，而写进 `text` 仍是 `append`（安全）
  - ⚠️ **流式每帧路径上的候选至此全部排除**：`escape_json`、`sse.mojo`、`stream_next` 的下标写、`_new_text`/`bytes_to_text`（悬垂）、`pending_view`（零拷贝视图）、`chunk_text_json`、`Conn` 的 `sent`/`clear()`、`tokenizer.decode`。加上 RSS 无泄漏、`dispatch` 不碰引擎、"双重推进"已证伪
  - → **继续读代码的收益已经很低**，剩下两条路：① 引擎侧（`service.stream_next` → 引擎 step / KV 房间），但非流式也走引擎却不崩，所以要找的是**流式特有的引擎用法差异**；② **换工具**：用 ASan / valgrind 直接抓"谁写坏了堆"（`-O0` 二进制 + valgrind；或查 `mojo build` 是否支持 sanitizer）。② 比继续猜快得多
- **2026-09-23（同日第十七条）** —— **换工具：ASan 跑通了但零报错 ⇒ 崩溃不是堆越界写；新假设 = 未初始化读**
  - 工具：`mojo build` 支持 `--sanitize`（帮助里明写 "Dynamically link the address sanitizer runtime"）。编了 `-O1 -g --sanitize address` 版（`target/serve_asan`，约 255 秒）
  - ⚠️ 坑（下次直接用）：开箱即 abort 在**加载模型**阶段 —— 栈是 `List::_annotate_new` ← `List(capacity=N)` ← `SafeTensorFile.__init__`，报 `bad parameters to __sanitizer_annotate_contiguous_container`。这是 **Mojo 标准库与 ASan 的容器标注不兼容**，与流式无关。**解法：`ASAN_OPTIONS=detect_container_overflow=0`，不用重编**
  - 结果：流式 **80 条 / 240 帧全部通过，ASan 零报错**（同一份代码在 `-O1` 无 ASan 时 6 条就崩）
  - → **实质进展**：ASan 有 redzone，堆越界写是**必报**的。零报错 ⇒ **崩溃不是堆越界写**（也基本排除了 UAF）
  - → **新假设：未初始化读**。它最吻合"-O0 不崩 / -O1 -O2 崩"—— `-O0` 下栈槽与寄存器恰好是 0，`-O1`+ 是垃圾；一个垃圾长度/指针传给分配器，就崩在 `List::_realloc` 或 AsyncRT 的分配里，而崩溃点看起来随机。这也解释了为什么读代码找不到"写坏堆的人"：**根本没有人写坏，是有人读到了没写过的东西**
  - 下一步：查 `mojo build --sanitize memory`（MSan）是否可用；或人工找"可能未初始化就被读"的字段 —— 重点看引擎/服务层那些"先声明、后填"的 `var`（`serve.mojo` 与 `engine/core.mojo`）
- **2026-09-23（同日第十八条）** —— **MSan 不可用；转 TSan**
  - `mojo build --sanitize` 只接受 `address` 或 `thread`（报错原文：`invalid sanitizer 'memory', expected one of: address or thread`）→ **没有 MSan**，未初始化读没法靠 sanitizer 直接抓
  - → 转 `--sanitize thread`（TSan）。理由：虽然"双重推进"已被证伪，但 reactor 与 engine 线程**确实并存**，且默认配置下 0 号 engine 线程用的是调用方那一份 handler（`engine_thread.mojo:560` 注释明写）→ **共享是真实存在的**，值得让 TSan 直接测，而不是继续推理
  - 备用（若 TSan 也无收获）：① 人工找"先声明后填、可能在填之前被读"的 `var`（`serve.mojo` / `engine/core.mojo`）；② `-O2` + gdb 在崩溃点打印关键状态（当初方案 1 用 -O0 跑、没崩，所以没打成；该用 -O2 跑）
- **2026-09-23（同日第十九条）** —— **TSan 链接失败；sanitizer 路线盘点到此为止**
  - `--sanitize thread` 编译通过但**链接失败**：`collect2: error: ld returned 1 exit status` → `failed to link executable`。环境里有 ASan 运行时、没有 TSan 运行时；装它要 sudo（无密码，装不了）
  - **工具盘点**：ASan ✅（跑通，80 条 / 240 帧**零报错** ⇒ 排除了堆越界写与 UAF）· MSan ❌（`mojo build` 只支持 `address`/`thread`）· TSan ❌（链接失败）· valgrind ❌（未安装，装要 sudo）· gdb ✅（可用）
  - → 下一步二选一：① **`-O2` + gdb 在崩溃点打印状态** —— 当初方案 1 是用 `-O0` 跑的、没崩所以没打成；换成 `-O2` 二进制就能打到。它能直接回答"崩溃那一刻我们的状态是好的还是坏的"，最便宜（现成二进制 + gdb，不用重编）· ② 人工查"先声明后填、可能在填之前被读"的 `var`（`serve.mojo` / `engine/core.mojo`）
- **2026-09-24** —— **决定性对照：`-g` 是唯一开关；并锁定一个新候选（引擎调度器的定长 `InlineArray`）**
  - **对照**（同一份干净源码，只差编译选项）：`-O2 -g` → **40 条 / 120 帧全过**（累计 3 轮共 200 条，无一次崩）；`-O2` 无 `-g` → **12 条 / 40 帧后死**
  - **全量汇总**：`-O0 -g` 不崩 · `-O2 -g` 不崩 · ASan（`-O1 -g`）不崩 · `-O1` 无 g **崩** · `-O2` 无 g **崩** → **`-g` 是唯一开关，与优化级别无关**
  - 含义：代码里有 UB，只在"无调试信息的优化 codegen"下被优化成错误的代码；`-g` 改变了布局/变换，恰好让它正确。⚠️ **这不是修复**（生产构建就是 `-O2` 无 `-g`），但它是极强的诊断信号 —— 它把问题从"堆"缩小到了"栈 / codegen 假设"
  - 两类最像的解释：① **未初始化读**（ASan 抓不到，它只管越界）② **栈上越界写 / 栈对象的悬垂引用**（`-g` 改变栈帧布局 → 恰好写不到要害）
  - 🔎 **新候选（强）**：`src/alofa/engine/scheduler.mojo` 有一批 `InlineArray[Int, MAX_BATCH]`（`MAX_BATCH=8`；203–207、287 行等）—— 定长**栈**数组。写入越过 `MAX_BATCH` 就是栈越界写；且 `InlineArray` 是**值类型**，结构体被按值拷贝/移动后其地址即失效 → 悬垂。两者都与"-g 改变栈布局就不崩"高度吻合
  - 下一步：查 scheduler 里对这些数组的**每一次写入**有没有与 `MAX_BATCH` 对齐的边界检查；以及 scheduler 对象是否被**按值拷贝**过（拷贝后结构体内的 `InlineArray` 地址失效，而外部若仍持有旧地址就是悬垂）
- **2026-09-28（同日·基准）** —— **「与 llama.cpp 的同机同 prompt 对比」首次做成口径对齐**（§11 那一行 `missing` → `verified`）
  - **为什么上一次（2026-09-18）不算**：三条边界没对齐 —— alofa 报**含 prefill** 的墙钟而 llama.cpp 是纯 decode；两边 prompt token 数不同（5 vs 13，当时没对账 tokenizer）；跑在 A100 **共享机**上。这次三条逐条解决。
  - **踩到的坑（下一个做基准的人直接抄）**：
    - ① ⚠️ **llama.cpp 的 CMake 默认不开 AVX2**（`build/CMakeCache.txt` 里 `GGML_AVX2:BOOL=OFF` / `GGML_AVX:BOOL=OFF`）⇒ 按默认编出来的二进制跑的是**基线实现**，实测 `tg128 = 1.88 tok/s`，照它写会得出"alofa 快 3×"的**假结论**。必须显式 `-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON` 重编。
    - ② **别用 anaconda 的工具链**：`which cmake` 是 anaconda 的，编到 `ggml.c` 报 `PR_SET_PTRACER undeclared`（它定义在 `linux/prctl.h`，glibc 的 `sys/prctl.h` 不含），anaconda 的 libstdc++ 还报 `timespec_get has not been declared`。改用 `env -i PATH=/usr/bin:… CC=/usr/bin/gcc CXX=/usr/bin/g++` 并补 `-DCMAKE_C_FLAGS=-DPR_SET_PTRACER=0x59616d61`。
    - ③ **`llama-cli` 不打印 perf 行，且默认是交互模式**（生成完挂在那儿等输入 ⇒ 看着像卡死，实测 5 min 无输出、sys 时间虚高）。改用 **`llama-bench`**（`-p 5 -n 32 -r 5`，分别报 `pp5` / `tg32`）；若非用 cli 不可要加 `-st`（单轮退出）。
    - ④ **权重同源**：HF 缓存 → `convert_hf_to_gguf.py --outtype f32`（1.98 GB / 290 tensors）→ `llama-quantize … Q4_0`（336 MB）。alofa 侧用**同一份** fp32 fixture（它的 q4 是加载期量化，不需要另备 q4 权重）。
  - **对账（可比的前提）**：`llama-tokenize` 对同一段文本给出 `[785, 6722, 315, 9625, 374]`，与 alofa 的 `tok.encode` **逐 id 相同**、无 BOS 差异 ⇒ "5 vs 13" 那条边界消失。
  - **新脚本** `scripts/bench_vs_llamacpp.mojo`：prefill 单独占返回 List 的第 0 格（llama-bench 的 `pp` / `tg` 就是分开的，混在一起会让 tok/s 随生成长度漂移）；两档 shards（1 / 8）交错；两档权重（fp32 / q4_0）；打印 id 与生成文本供对账。⚠️ Mojo 1.0 **一个函数只允许一个 `out` 参数**，所以两个输出塞进一条 List 返回。
  - **实测（本机 i7-9700K 8C AVX2，prompt 5 token，生成 32 token，greedy）**：
    - fp32 decode：alofa `5.83…7.92`（shards=8）/ `3.95…5.22`（shards=1） vs llama.cpp `5.51…5.52`（`-t 1`）⇒ **1.06…1.43×**
    - q4_0 decode：alofa `13.83…15.37`（shards=8）/ `6.65…6.80`（shards=1） vs llama.cpp `25.60 ± 1.18`（`-t 1`）⇒ **0.54…0.60×**
    - ⚠️ llama.cpp `-t 8` 在本机**反而更慢且抖**（fp32 `1.54…3.34`、q4_0 `14.71 ± 6.88`）⇒ 主口径取 `-t 1`；本机本轮 runq 2–17，**过载是这台机器的常态**，绝对值不可跨机比较。
    - q4 一致性旁证：alofa q4 档生成的文本与 fp32 档**不同**（"…largest city in Europe. It is also…" vs "…largest city in Europe and the second largest…"）—— 与账本里"整网 q4 一致率实测 0.8691、未达 0.90 门"是同一件事的两个观测面。
  - **结论（决定下一个杠杆）**：**fp32 档已打平甚至略超**（llama.cpp 的 f32 权重在本机还吃不到 `CPU_REPACK`，它的优化重心不在这一档）；**差距集中在 q4_0 档** —— llama.cpp 的 q4_0 相对它自己的 fp32 提速 **4.64×**（5.52 → 25.60 tok/s），alofa 只做到 **1.94×**（7.92 → 15.37）。⇒ 下一个性能杠杆是 **q4 内核**（repack / gemv），**不是** fp32 gemm（后者已到天花板，见「性能」一节）。
  - **未做**：批（batch > 1）档、更长 prompt 档、`-t 8` 异常的机理（怀疑 ggml 线程 barrier 在过载下被放大，**未证**）、A100 机同口径复现。
- **2026-09-28（同日·对照实验：q4 档差距的归因）** —— **差距不在 q4 内核，在输出投影的存储精度**
  - **做法**：**不碰 alofa 一行代码**，只在 llama.cpp 侧做**单变量对照** —— 层恒为 q4_0，只改
    `--token-embedding-type`。⚠️ 两个前提：① Qwen2.5-0.5B 是 **tied embedding**，GGUF 里
    **没有** `output.weight`（输出投影用的就是 `token_embd.weight`）；② `llama-quantize` 的
    选项必须放在**位置参数之前**（放后面会被当成 `nthread`，报 `invalid nthread … (stoi)`）。
  - **实测**（`llama-bench -t 1 -p 5 -n 32 -r 5`，tg32，层恒 q4_0）：
    - 嵌入/输出 **Q8_0**（默认，336 MB）：**25.60 ± 1.18**
    - 嵌入/输出 **F16**（452 MB）：**17.77 ± 1.32**
    - 嵌入/输出 **F32**（712 MB）：**13.44 ± 0.82**
    - 全 F32 参考（1.84 GB）：5.52
  - **结论**：alofa 的 q4 档 `13.83…15.37` 与 llama.cpp「输出投影也拖回 F32」的 `13.44`
    **几乎重合** ⇒ **同流量下 alofa 反而 `1.03…1.14×` 快**。所以 §11 那个 `0.54…0.60×`
    **不是** q4 内核的水平，而是**输出投影字节数**的差：alofa 的 `lm_head` 是独立的 fp32
    副本（544 MB），llama.cpp 因 tied 只存一份且压成 Q8_0（≈165 MB）⇒ 每 token 流量
    **655 MB vs 276 MB**。
  - **为什么 alofa 的 lm_head 没量化**（`qwen.mojo` 的 `enable_q4` 注释，本轮之前量过）：
    连同它压成 **q4_0**，教师强制贪心一致率从 0.8691 掉到 **0.77** —— 词表那 151936 个
    logits 谁最大全看这一步，4 bit 在这儿省 76 MB 不值。**但 8 bit 是另一回事**：q8_0 的
    相对 L2 约 **0.4%**（q4_0 是 10.3%），精度损失比"掉到 0.77"小一个量级。
  - **下一个杠杆（已量化，未实现）**：把 `lm_head` 存成 **q8_0**（≈145 MB）而非 fp32
    （544 MB）⇒ 流量 655 → 256 MB。按 llama.cpp 同一对照的经验（流量降 2.1× 实拿 **1.90×**，
    效率约 90%），alofa q4 档预期 `15.37 → 约 29 tok/s`，**越过** llama.cpp 当前的 25.60；
    fp32 档同样受益（1432 → 1033 MB）。
  - **实现成本（未做）**：仓库**还没有 q8_0 通路**（`core/dtype.mojo` 只有枚举，无 kernel）
    ⇒ 要新写量化步 + dequant 融合 matmul（标量 / avx2）+ 接进 `lm_head` + 零容差门与常驻
    负向对照。q8_0 的 dequant 与 q4_0 同性质（8 bit 值 × fp16 scale，两者都精确）⇒ **可零容差**。
  - **未做 / 已否**：f16 方案（llama.cpp 实测只 **+32%**，不如 q8_0 的 **+90%**）；q8_0 用在
    embed 上（decode 只读一行，无收益；prefill 才有）。
- **2026-09-28（同日·q8_0 kernel 落地，未接进模型）** —— **为"输出投影降宽度"备好第一块料**
  - **做了什么**：`quant.mojo` 加 q8_0 的量化步 / 解量化 / **标量**融合 matmul；
    `scripts/dump_q8_reference.py` 独立另写离线参考并导出 fixture（`lm_head.weight`
    前 64 行真实权重，57344 值 / 1792 块）；`tests/unit/test_q8_parity.mojo` 8 条用例
    （解量化**零容差**、量化**逐字节**、matmul `1e-5 × max(1,|ref|)`、三条负向对照、
    相对 L2 < 1%）。
  - **实测相对 L2 = 0.5359%**（q4_0 是 10.3%）—— 门里那条 "< 1%" 的作用就是**把 q4_0
    挡在输出投影外面**：哪天有人为省 76 MB 把 `lm_head` 退回 q4_0，它立刻红。
  - **两个坑（下次直接抄）**：
    - ① q8_0 存的是**补码**，与 q4_0 的"无符号 nibble 减 8"不是一回事；照 q4 的习惯
      读，负权重整体跳到 +128 那侧，输出仍然像一堆合理的权重 ⇒ 负向对照专门钉它。
    - ② **负向对照不能在错步长上比浮点**：按 32 而非 34 读块会落到不是合法 fp16 的字
      节上（实测 `half_bits=32740`），`fp16_bits_to_f32` 直接**抛错** —— 那是崩溃不是
      判红。改为比**字节**。
  - ⚠️ **两处"写了门 ≠ 会跑"的漏子**（都当场修了）：① 新门必须加进
    `scripts/run_tests.sh`；② `tests/fixtures/.../q8/blocks.bin` 被顶层 `*.bin` 静默
    忽略（q4/ 与 q4vec/ 各有一条例外），不加例外就是**本地绿、干净检出红**（第一笔提交
    就漏了它，补了一笔）。
  - ⚠️ **`ledger-sync` 的一个副作用**（记下来免得下次白查）：它会给账本里**没登记**
    `?count=` 的行补数字，而 `tests/unit/test_batch_stream.mojo` 是**重资产门**
    （`pixi run test-batch-stream`，从不进 `target/test_counts.tsv`）⇒ 补上 `?count=6`
    反而让 `check-counts` **必然**红（它要求每个登记了 count 的文件出现在当次清单里）。
    **该处已回退**，保留纯 evidence 指针。
  - **未做**：把 q8_0 接进 `lm_head`（要动 `qwen.mojo` 的量化槽位与前向分派）⇒ **端到端
    尚未受益**，账本那一行明确不宣称提速；avx2 版 q8 kernel 也没有。下一步做完接线后，
    预期 q4 档 `15.37 → 约 29 tok/s`（按 llama.cpp 同对照"流量降 2.1× 实拿 1.90×"折算）。
- **2026-09-28（同日·q8_0 输出投影接进 `lm_head` 并复测）** —— **归因对了，这根杠杆也真的撬得动**
  - **接线**：`qwen.mojo` 加 `enable_q8_output()` + `q8_matmul_k_shards[backend]`（按输出行切
    `shards` 片，与 fp32 通路**同一个**分片数，两档才是在比权重宽度）。`run` 里改了**两处**：
    输出投影，以及 **embedding 查表** —— ⚠️ tied embedding 时它俩是**同一份字节**，只改 matmul
    一侧会让 prefill 把 q8 字节当 fp32 解释，而且**只在 prefill 上错**、decode 看着完全正常
    （所以这一侧单独开了一条用例盯着）。
  - **质量（新门 `tests/unit/test_q8_head_greedy.mojo`，4/4 全绿；重资产，`pixi run test-q8-head`）**：
    教师强制贪心一致率 **0.8691 → 0.8672**（512 步只掉 **2 步**，容忍差 0.03），且
    **标量后端与向量后端逐位相同**。对照：输出投影压 **q4_0** 是 0.8691 → **0.77** —— 这就是
    "为什么是 8 bit"的答案。另两条：logits 相对 L2 **0.77%**、按行解出的 embedding **0.57%**
    （与权重侧的 0.5359% 吻合）。
  - ⚠️ **第一次复测是负的，值得记**：接上标量 `matmul_q8_f32` 后，q4 档 `shards=8` 从
    `13.72…15.38` **掉到 7.12…9.36 tok/s（−22%）**。诊断不是"量化没用"，是省下的
    **379 MB/token 流量被 136M 次标量乘加吃了回去** —— fp32 那条通路是 AVX2 + 分片，等于让
    q8 瘸着一条腿去比赛。**补了 AVX2 版** `_matmul_q8_vec`（8 宽 f32 通道、`d` 折进 x、行末只
    归约一次；⚠️ 符号扩展交给**按 `Int8` 的 load**，不手写"读 u8 再判 ≥128 减 256"）⇒
    **17.70…20.23 tok/s（+30%）**。
  - **同机口径**：llama.cpp 25.60（它也是 Q8_0 输出投影）⇒ alofa 由 **0.60× 提到 0.79×**。
    与前两轮合起来看：差距的支配项就是这一个张量的宽度，改它一个地方就 +30%。
  - ⚠️ **还差 27% 才够到 25.60**，本轮没做：q8 向量核还没试 `_matmul_q4_wide` 那套按 `j%8`
    切 f64 通道的写法；层 kernel 与 llama.cpp 的差距也没重新量。
  - ⚠️ **本机 runq 5–11**（比上一轮更忙，q4 基线自己就从 15.37 掉到 15.38…13.72 区间）⇒ 上面
    所有 tok/s **只作同轮内部比较**，不跨轮比。
- **2026-09-28（同日·接着往下拆：离 25.60 还差的那 27% 在哪）** —— **不在输出投影，在 q4 层核的算术**
  - 新脚本 `scripts/bench_head_split.mojo`：一次 decode **当场切开**，不再反推（真实 0.5B、
    q4_0 层 + q8_0 输出投影、backend avx2、`shards=8`）⇒
    `step 88.18 ms = 输出投影 17.39 ms (19.7%) + 其余 70.79 ms`。输出投影只占**两成** ⇒
    继续抠 q8 核的天花板只有 `11.34 → 14.13 tok/s`。
  - **证否一个诱人的猜测**：新脚本 `scripts/bench_taskgroup.mojo` 量"一次 `TaskGroup` 空转"
    ⇒ `shards=8` 单次 **1.92 μs**，×169 次投影 = **0.32 ms/token** —— 分片 barrier **不是**
    那 70 ms。（`bench_model_shards.mojo` 同时确认分片**确实更快**：`shards=8` 同轮配对比
    **1.35…1.82×**、区间不重合，有效带宽 8.12 → **12.94…14.84 GB/s**。）
  - **真凶（`bench_decode_roofline.mojo`，现成脚本，单线程核级）** —— `down_proj` 4864×896：
    fp32/avx2 读 **17.4 MB** 跑 **11.47 GB/s**（自己的访存地板/实测 **0.883**，算术几乎被访存
    盖满）；**q4_0/avx2 读 2.45 MB**（八倍小于 L3，已常驻 cache）却只跑 **2.11 GB/s**（地板比
    **0.163**）。⇒ 块流在 **cache 里**跑都输给 fp32 从**内存**跑，它**不是带宽受限，是算术受限**。
  - ⚠️ 但 q4_0/avx2 已到 **7.51 GFLOP/s = 算力峰值 9.91 的 76%**（峰值是脚本现测的下限）⇒
    **f32 fma 这条路上只剩 1.3×**。参照：fp32 的 `lm_head` 能到 10.07 GB/s（774‰ 读峰值）。
  - ⇒ **剩下的 27% 要换算术，不是再抠核**：llama.cpp 的 q4_0 走**整数点积**（`vpmaddubsw`
    一类，一条指令 16 个 8×8 MAC），算力峰值比 f32 fma 高一个量级 —— 这才是它 501 MB/token
    能跑到 12.9 GB/s 的原因。本轮**没做**：要新增 x 的 int8 量化 + intrinsic 通路 + 精度门，
    是独立一笔活。
  - ⚠️ 边界：`bench_head_split` 那次的绝对值（88.18 ms）比 `bench_vs` 同档（49.4 ms）慢
    **1.6×**（本机当时更忙）⇒ 上面只用它的**占比**；拿 `bench_vs` 的绝对值配上这个占比，
    层那半约 39.7 ms 读 201 MB ⇒ **5.1 GB/s**，对照输出投影 17 GB/s、fp32 层 14 GB/s。
- **2026-09-28（同日·整数点积那条路：探通了，但只值 1.07× ⇒ 收手，不接线）**
  - **通路发现（可复用）**：Mojo 1.0 里 `llvm_intrinsic` **不在 `sys`**，在 **`std.sys`**
    （`from std.sys import llvm_intrinsic`）；写 `from sys import ...` 报"unable to locate
    module 'sys'"，写 `std.intrinsics` 也找不到。另一条 `__mlir_op.\`llvm.intr...\`` 走不
    通（报 `unrealized_conversion_cast` —— Mojo 的 `SIMD` 不是 LLVM vector，要转换）。
  - **能叫到的**：`llvm.x86.avx2.pmadd.wd`（`vpmaddwd`：16 个 int16 乘加 → 8 个 int32），
    数值已验（(3,4) → 24）。⚠️ **`vpmaddubsw` 拿不到**：`avx2.pmaddubsw` /
    `avx2.pmaddubs.w` / `avx2.pmaddub.sw` / `avx2.pmadd.ubsw` / `avx2.pmaddubsw.256` /
    `ssse3.pmaddubsw` 六种拼法本机全都"could not find LLVM intrinsic"。而 llama.cpp 的
    q4_0 **正是靠它**（无 VNNI 时用符号技巧 `ax=|x|`、`sy=sign(x)·y` 再走 `maddubs`）
    ⇒ 拿不到它，是本方案与 llama.cpp 之间**最实在的一处差距**。
  - **实测**（新脚本 `scripts/bench_q4_int8.mojo`，真形状 `down_proj` 4864×896，avx2，
    参照物是未量化 fp32 权重 × fp32 激活、fp64 累加，**独立另写**）：
    - 现在的 f32 解量化核：**2.28 GB/s**，相对 L2 **6.39%**
    - 整数点积（`pmadd.wd`）：**2.44 GB/s（1.07×）**，相对 L2 **6.40%**
    - 同上 + 激活预 widen 成 int16：**2.09 GB/s（0.96×，更慢）**
  - **精度站得住**：整数点积 6.40% 与 f32 核 6.39% 几乎相同 ⇒ 整数路径数学正确，误差
    只来自 q4 量化本身（合成权重取均匀分布，q4 误差本就是最大的一档；真实权重近正态）。
    参照 `test-q8-head` 那条门的 0.77% —— 那是词表输出投影，量级不同，不直接比。
  - **为什么只有 7%**：`pmadd.wd` 要 int16 输入 ⇒ 每块多付 4 条 `vpmovsxbw` 的 widen，
    把"一条指令 16 个乘加"的收益吃掉了。llama.cpp 直接吃 int8，没有这 4 条。试过把激活
    **预先** widen（省 2 条）⇒ 反而 0.96×：省下的指令抵不过激活体积翻倍带来的 L1 压力。
  - ⇒ **收手**：7% 配不上"改 q4 主通路"的风险（168 次投影的调用面 + 新精度门）。
    ⚠️ **记一个坑**：**`SIMD.cast` 是数值转换，不是位重解释** —— 补码字节（200 = −56）
    被 `cast[DType.int8]()` **饱和成 +127**，激活的负值全废，第一版就是这个 bug
    （相对 L2 **240%**）。修法是按 `Int8` 取指针，`unsafe_load` 即得符号扩展。
- **2026-09-28（同日·f16 权重：核级 1.09× ⇒ 端到端慢 3.68× ⇒ 收手；但量出一个真热点）**
  - 新脚本 `scripts/bench_f16_w.mojo`（真形状 `down_proj` 4864×896，avx2，F16C 有）：
    - q4_0（现在）：读 2394 KB，**2.85 GB/s**，相对 L2 6.39%
    - f16（朴素核，单累加器）：读 8512 KB，**11.03 GB/s**，相对 L2 **0.02%**
    - f16 四路展开：**6.67 GB/s（更慢）** —— 展开反而劣化，记一笔别再试
    - fp32（项目 `linear`）：读 17024 KB，**18.54 GB/s**
  - ⇒ 核级 f16 只快 **1.09×**；而 f16 每值 2 字节，是 q4 的 **4×** ⇒ 端到端的账
    ≈ 4 / 1.09 = **3.68×（f16 更慢）** ⇒ **收手，不接线**。判据与整数点积那条路同一把
    尺子：**核级要 >4× 才值得**（f16 是 1.09×，整数点积是 1.07×，两条都差得远）。
  - ⚠️ **但带宽对比暴露了一个更重要的事实**：f16 核几乎没什么算术就跑 **11.03 GB/s**，
    q4 核只有 **2.85** —— **差 3.9×，而两者的指令数只差约 1.25×** ⇒ q4 核慢的原因
    **不是** SIMD 算术太重，得上别处找。
  - ⇒ 顺手量了每块都要调一次的 `block_scale`（新脚本 `bench_block_scale.mojo`）：
    **5.4 ns/call**，折合 **38.6 条 f32 fma**；而一块 32 个值只需 **4 条 fma**（8 宽）
    ⇒ **手写 fp16 位运算占每块成本的 90.6%**。它是 `quant.mojo` 里那段标量位运算
    （指数/尾数/符号三段 + 次正规分支 + 一个 `raise`），其中**除法**最贵。
    （顺带验证：那个 `raise` 是**活代码** —— 我第一版随机填的字节撞上 exponent==31
    就被它抛出来了。）
  - ⇒ **下一个红包定位在这**：**不改格式、不增流量**，只把 `d` 的解码挪出热循环 ——
    加载时预解成 f32 表（+4 B/块 ≈ 12.5% 额外内存），或快路径走 F16C `vcvtph2ps`、
    慢路径（次正规/NaN）继续留手写。粗估 q4 核 2.85 → ~10 GB/s（**3.7×**），直接打在
    "离 llama.cpp 还差 27%"那笔账上。⚠️ **未验证、未接线**，下一步先做核级验证。
- **2026-09-28（同日·`block_scale` 的每块**调用开销**：已接线，层核 1.80×；端到端待复测）**
  - 症状（先量后改）：`block_scale` 一次 **5.4 ns ≈ 38.6 条 fma**，而一块 32 个值本身只需
    4 条 fma ⇒ 占每块成本 **90.6%**（`bench_block_scale.mojo`）。
  - ⚠️ **第一次改错了对象**：把手写 fp16 位运算换成 F16C `vcvtph2ps` 只快 **7%**
    （1157 → 1071 µs）。真凶是**每块一次的函数调用** —— 它是 `raises` 函数，编译器
    **不内联**；同样的写法**内联在核里**就快 2.27×。加上 `@always_inline` 才拿到
    **2.02×**（1157 → 572 µs，`bench_q4_predecode.mojo` 同场三档：① 572 / ② 360 / ③ 492）。
  - 接线：`block_scale` 加 `@always_inline` + 快路径走 F16C；慢路径（exponent==31 的
    inf/NaN、以及奇数偏移）仍走 `fp16_bits_to_f32` 的 `raise`。**调用面一处未动**。
  - 判据：与改动前**逐位相同**（fp16 能表示的数 fp32 都能精确表示，次正规硬件同样精确）
    —— `pixi run test` **389 用例 / 39 套件全绿**，`test-q4`（端到端 greedy）通过。
  - 层核（`bench_decode_roofline.mojo`，同形状）：q4_0/avx2 **2.29 → 4.12 GB/s**
    （1069 → 595 µs，**1.80×**）；地板比 0.137 → **0.449**；相对 fp32/avx2 由 1.34×
    变成 **2.43×**。
  - ⚠️ **端到端未确认，不要拿它下结论**：本机这轮 **runq 19–25**（严重过载），连
    **不受这个改动影响**的 fp32 档都只有 5.21 tok/s（此前 5.83…7.92）⇒ 跨轮数字不可用。
    q4_0+q8_head shards=8 这轮 14.52…18.85 tok/s。
  - ⚠️ 而且有个**结构性怀疑**：8 shards 并行后总带宽需求 ≈ 8 × 2.29 = 18 GB/s，已超本机
    内存峰值（~14）⇒ **可能已经撞上内存带宽墙**，那么"每核算术"省下的这些在端到端上
    会被吃掉 —— 单线程层核是算术受限，8 线程并行的端到端未必是。**待机器空闲时复测**。
- **2026-09-29（端到端复测：`q4_0+q8_head` 13.6…17.5 → 28.7…30.2 tok/s；与 llama.cpp 追平）**
  - 同机 A/B：`bench_vs_llamacpp.mojo` 在同一台机器上**相邻两轮**，各 3 趟取 min…max。
    - fp32 / shards=8（**控制档**）：改动前 6.53…7.15，改动后 6.69…6.99 ⇒ **重合**
      ⇒ 这两轮机器状态可比，下面的差可以认。
    - q4_0 / shards=1：6.57…6.82 → **9.22…9.49**；q4_0 / shards=8：13.16…14.91 → **16.57…17.37**
    - q4_0+q8_head / shards=1：7.43…7.57 → **12.40…12.59**
    - q4_0+q8_head / shards=8：13.63…17.49 → **28.70…30.24**（约 **1.8×**）
  - ⇒ **端到端确认提速**，不再是"待复测"。⚠️ 昨天那条"8 shards 可能撞内存带宽墙、
    每核算术收益会被吃掉"的怀疑**不成立** —— 收益确实转化到了端到端，而且是靠
    shards=8 这条多线程路拿到的。
  - llama.cpp 同轮对照（`llama-bench -p 5 -n 32 -r 3 -t 1`）：**tg32 27.05 ± 2.80 tok/s**。
    ⚠️ 已查 `build/CMakeCache.txt`：`GGML_AVX2/FMA/F16C` 均 **ON**（没开 AVX2 会得出
    "alofa 快 3×"的假结论，这个坑在本文件里记过一次）。
  - ⇒ 与 llama.cpp 的关系：**追平**。alofa 28.70…30.24 落在 llama 的误差带
    （24.25…29.85）**之内并有重叠** ⇒ 从"差 27%"到持平，但**不能**写成"超过"。
  - ⚠️ 单线程口径下 alofa 仍**慢 2.15×**（12.40…12.59 vs 27.05）：追平是靠 shards=8。
    llama.cpp 自己的 `-t 8` 在这台机器上崩到 **0.38 ± 0.11 tok/s**（已知：它 `-t 8` 慢于
    `-t 1`），所以它的强档就是 `-t 1` —— 这个比较对它是有利的。

- **2026-09-29（续）—— `vpmaddubsw` 行带通路：从"无增益"翻案到成为 AVX2 默认，并补上两道门**
  - ⚠️ **翻案**：此前"端到端无增益"与"greedy 顶得住"**都是空跑**。开关挂在 `q4_matmul_k`，
    而模型 `project` 走的是 `q4_matmul_rows_k` / `q4_matmul_bias_rows_k`（行带，**decode 也走它**，
    `n_tok=1` ⇒ 单行分支）⇒ mub 一次都没执行。证据是两样硬东西：bench 二进制里
    `vpmaddubsw` **0 条**，env 守卫的追踪 **0 行**；而直调 `q4_matmul_k[AVX2]` 的探针有 1 条。
    改挂行带入口后：追踪 **344400 行**，形状与 Qwen 对得上（896 / 4864 / 608 / 112 / 16）。
  - **真数字**（shards=1，`q4_0+q8_head`，同轮 A/B 两轮）：decode
    **默认 10.78…12.95 → 13.79…15.61 tok/s**；prefill **281…381 → 155…238 ms（约 1.7×）**；
    fp32 控制档（开关管不到它）在噪声内。离 llama.cpp 27.05：**2.15× → 约 1.8×**。
  - ⚠️ **第一版质量不过**：整批共用一个激活步长 ⇒ 核级相对 L2 **0.73–0.80%**（各行幅度差 20× 时
    **1.00%**），整网一致率 **419/512 = 0.8184**，红在 0.85 回归下限上。
  - **改成逐 (行, 块) 缩放**（每 32 个激活一个步长 = llama.cpp q8_0 激活口径；两趟走同一块顺手把
    `Σx` 带走；倒数提到块级免逐元素除法；量化改四舍五入）：相对 L2 **0.33–0.38%**（幅度差 20× 时
    0.34%），整网一致率 **444/512 = 0.8672**（默认 f32 展开 445/512 = 0.8691，只差一个 token）
    ⇒ **0.85 过**。速度没掉（MUB 档前后落在同一区间）。
    ⚠️ "每行一个步长"那一档**没有留**：内层每 (行,块) 都要一次 `d·s` 广播 ⇒ 与逐块**同成本**
    却仍被整行最大值支配 ⇒ 被严格支配。
  - **门**（在此之前这条通路**没有门能看见它**：`test-q4` 跑标量后端，mub 一行都不执行）：
    - `tests/unit/test_q4_mub_greedy.mojo`：整网，2 tests，重门 `pixi run test-q4-mub`
      （2 GB 权重 + 数分钟，故不并进 `pixi run test`）。
    - `tests/unit/test_q4_mub_rows.mojo`：核级，5 tests，已登记进 `scripts/run_tests.sh`。
      相对 L2 ≤ 1% + **两条通路不许逐位相同** + 守卫区 + "多乘 4 的旧步长必须被守卫抓住"的红测。
      **已验过它不是不会失败的门**：把 mub 换成 f32 再跑，两条"确实在算"全红（3 passed / 2 failed）。
  - ⇒ **翻默认**：`vpmaddubsw` 成为 AVX2 下的 q4 通路（`ALOFA_Q4_MUB=0` 退回 f32 展开）。
    原 `scripts/verify_q4_mub_greedy.mojo` 删除 —— 它"不是门、不登记"的定位已被上面的门取代。
  - ⚠️ **翻默认当天就发现它让 shards=8 慢 30%**（同轮、用 fp32 档归一化：mub 3.39× vs
    无 mub 4.85×，而 shards=1 是 +20%）。成因是**每次调用的固定税**：三个 scratch 缓冲用
    `fill=0` 构造 ⇒ 每次调用清 16 KB + 2 KB + 2 KB，与本次真正要读的权重流量无关；shards
    切分后每次调用只做 1/8 的输出行，**同一笔税被放大 8 倍**。改成 `uninitialized=True`
    （三个缓冲都是先写后读；单行核的 `Σx` 只有 lane 0 有意义，改用整向量 store 写满 8 lane）
    ⇒ shards=8 回到**齐平**（3.85× vs 3.87×），shards=1 的 +20% 保住。
    ⚠️ 记忆里"Mojo 1.0 没有未初始化构造"是**错的**：`unsafe_uninitialized=True` 不存在，
    `uninitialized=True` 存在且能编。
  - ⚠️ 本轮绝对吞吐被机器负载压低（runq 8–20）⇒ **只做同轮比较**，跨轮绝对数字不可比。
  - 已有重门里**唯一能看见这条通路的是 `test-q8-head`**（它两个后端都跑且开 q4）：翻默认后
    4/4 过（AVX2 档 plain 0.8672 / q8_0 0.8652，差 0.002 ≪ 0.03 容忍）。
    ⚠️ `test-batch-forward` 与 `test-model-avx2` 都是 **fp32**，**看不见**这条通路。
  - **同机同轮补跑与 llama.cpp 的对照（mub 默认生效后首次）**：alofa 侧 `q4_0+q8_head` 档；
    llama.cpp 侧 `llama-bench -p 5 -n 32 -r 5`（AVX2 构建，`GGML_AVX2=ON`），两个模型 ——
    `qwen25-05b-q4_0.gguf`（默认，嵌入/输出存 Q8_0）与 `qwen25-05b-q4_0-f32emb.gguf`（同流量
    对照档，2026-09-28 建的）。**交错 3 轮**，只取同轮比值。
    ⚠️ **修正一条基线口径**：本机 llama.cpp 的最佳线程档是 **`-t 4 = 40.41 ± 5.01`**，不是此前
    一直用的 `-t 1`（`-t 1 23.97…24.79`、`-t 2 32.85`、`-t 8 5.63`）⇒ 账本此前"主口径 `-t 1`"
    **低估 llama 约 1.7×**，之前"追平/略超"的结论是**跟一个被削弱的对手比出来的**，须按 `-t 4`
    重读。⚠️ `-t 8` 塌（5.63）是本机长期过载下的超订观测，不是 llama.cpp 的普遍结论 —— 与
    2026-09-28 记的边界② 同一件事；且 `-t 4` 抖动 ±5、只测了 1 次 ⇒ **要当基线得先复测**。
  - **同轮数字（decode tok/s）**：alofa shards=1 `15.20…15.76`、shards=8 `23.86…33.35`；llama
    默认档 `-t 1 23.97…24.79` / `-t 2 32.85` / `-t 4 40.41` / `-t 8 5.63`；llama F32 嵌入档
    `-t 1 14.69…14.97` / `-t 2 17.68…20.83` / `-t 4 19.72…20.69` / `-t 8 2.52`。
    ⇒ **按各自最佳配置**：alofa shards=8 是 llama `-t 4` 的 **0.66…0.83×**（8 核 vs 4 核）；
    **单核口径**：alofa shards=1 是 llama `-t 1` 的 **0.63…0.66×**。
  - ⚠️ **差距像在"每字节效率"，不在指令数**：F32 嵌入档的 llama **每 token 要读的 `lm_head` 是
    fp32（544 MB 量级，模型 711 MiB）**，比 alofa `q4_0+q8_head` 档（`lm_head` 已压到 q8_0，
    136 MB 量级）**重一倍多**，可它 `-t 1` 仍与 alofa shards=1 **同速**（14.69…14.97 vs
    15.20…15.76）⇒ alofa 少读一半流量却没更快。这与上面"mub 省下的算术在 shards=8 转不成吞吐"
    指向同一个病。⚠️ 这里的"重一倍"是**按张量字节数算的形状账，不是实测带宽** ⇒ 要把它变成
    判据，下一步得实测 GB/s（现在还不能写进 `verified` 结论）。
  - **同轮实测"达成带宽"（把上面那条形状账换成带字节数的账）**：每 token 字节数按**真模型形状**
    算（`config.json`：hidden 896 / 24 层 / 14 头 / 2 kv 头 / inter 4864 / vocab 151936 /
    **tied**；q4_0 = 18 B/32、q8_0 = 34 B/32）⇒ 非嵌入参数 357.83 M、嵌入 136.13 M ⇒ alofa 三档
    **fp32 1975.8 / `q4_0` 745.8 / `q4_0+q8_head` 345.9 MB/token**；llama.cpp 默认档 **345.9**
    （与该档 gguf 体积 `330.17 MiB` 自洽 —— tied 只存一份，每 token 正好把整个模型读一遍）、
    F32 嵌入档 745.8。**两边主战档是同一个数 ⇒ 今天的对比是严格同流量，不是"谁读得少"**。
  - ⚠️ **顺带修正第二个基线口径**：账本的"整机读带宽峰值 15.03 GB/s"**低估约 1.7×** —— 同轮用
    `scripts/bench_thread_bandwidth.mojo` 重标（256 MB 数组，min…max）：`T=1 12.00…15.89`、
    `T=2 17.92…23.28`、`T=4 18.70…26.50`、`T=6 18.06…23.21`、`T=8 19.47…22.71` ⇒ 15.03 其实是
    **单线程**量级的数，整机应取 T=4 那一档（≈26.5）。（"单线程 12.0"那条没错，是 T=1 的下限。）
  - **达成带宽 = 字节/token × 实测 tok/s**：alofa `q4_0+q8_head` shards=1 **5.33…5.48 GB/s**
    （单线程探针的 36…46%）、shards=8 **11.62…12.26**（T=8 探针的 51…63%）；llama.cpp 默认档
    `-t 1 8.17…9.79`（55…77%）、`-t 4 11.80…16.91`（T=4 探针的 63…90%）。
    ⇒ **同流量下 alofa 单核只到 llama 的 0.55…0.67×**；8 核到 llama 4 核的 **0.69…1.04×**
    （还**多用一倍核**）。⚠️ **两边都没吃满**（llama 最高 ~90%、alofa 最高 63%）。
  - ⚠️ **达成带宽跨轮不可比（2026-09-30 同轮重测）**：上面 shards=8 的 **11.62…12.26** 是
    **那一轮**的数，不是常数。同脚本 3 趟交错重测（本机 runq 6–25）得 `q4_0+q8_head`
    shards=1 **59.33…75.09 ms/token**、shards=8 **34.65…44.66** ⇒ 达成带宽
    **4.60…5.83 / 7.74…9.98 GB/s**（对 T=8 纯读探针 19.47…22.71 是 **34…51%**），
    加速比 **1.68…1.71×**。⇒ 两个数都"对"，但**引用时必须带上那一轮的 runq 与区间**；
    本机过载下**同轮**真值约 **8–10 GB/s**。由此作废一条旧推论：~~shards=8 已贴聚合带宽，
    故改核内指令收益 <5%~~ —— 34…51% 说明**没贴**，该判断要重估（核内弹性 0.55 才有意义）。
    ⚠️ 同时作废 ~~shards 1→8 只有 1.42×~~（那是另一轮：69 → 48.5 ms）。
  - ✅ **病定位到"q4/q8 通路自己"，不是流量不够**：alofa **自己的 fp32 档单核能吃到
    9.50…9.60 GB/s（63…80%）**，而 `q4_0` 档 7.56…7.72、`q4_0+q8_head` 档只有 5.33…5.48 ⇒
    **流量越小的档、达成带宽反而越低**：`q4_0+q8_head` 比 fp32 **少读 5.7× 字节，只快 3.2×**
    （4.81…4.86 → 15.42…15.84 tok/s）。⇒ 若 q4/q8 通路能吃到 llama `-t 1` 那一档（8.2…9.8 GB/s），
    单核应是 `345.9 / 8.2 ≈ 42 ms`（≈23.7 tok/s，现 15.4）⇒ **约 1.5× 的空间，且不需要再压流量**。
  - ⚠️ 边界：① 字节数是**按形状算出**的（自洽校验：两档 gguf 体积 == 算出的每 token 字节），
    **不是插桩实测**；② llama `-t 4` 三次 `34.11 / 40.41 / 48.87`，抖动很大 ⇒ 区间宽，**别挑一轮报**；
    ③ 带宽探针含 `TaskGroup` 调度开销，本机 runq 4–25 ⇒ **只作同轮参照，不是硬件白皮书数字**。
  - **核级复现（冷 cache，"越量化越不吃带宽"在核内就能看到）**：新探针
    `scripts/bench_q4_cold_bandwidth.mojo`（`mojo build -O2 -I src` 后跑）。⚠️ **不能**拿既有
    `bench_decode_roofline.mojo` 的 q4 那档下结论 —— 它量的是**普通核** `_matmul_q4` /
    `matmul_q4_f32`，而 decode 的真入口是**行带** `q4_matmul_rows_k`（`n_tok=1` ⇒ 单行分支），
    开关只挂在普通核上时**一次都没执行过**（早上那条坑）。这里量的是真入口；且**每次计时前扫
    256 MB 冲刷缓冲**（L3 12 MB，`down_proj` 的 q4 权重只有 2.45 MB，不冲就驻留在 cache 里，
    量到的是算力不是 DRAM）。
  - **核级达成带宽（decode 形状，best-of-5，同轮）**：`q/o_proj`(896×896) fp32 **12.83 GB/s** /
    q4 **4.32**；`gate/up/down`(4864×896) fp32 **11.97** / q4 **4.68**；`k/v_proj`(128×896)
    fp32 **11.42** / q4 **2.96**。⇒ **q4 只吃到 fp32 的 0.30…0.39×**；**字节少 7.11×，时间只快
    2.14…3.23×**。与端到端自洽：核级时间加总（24 层 ≈ 45 ms + `lm_head` q8_0）≈ 63 ms ≈ 15.9 tok/s，
    实测端到端 `15.4…15.8` ✓。
  - ✅ **病定位到核内，不是 DRAM**：同通路**热 cache**（权重留在 L3）对比冷 —— `q/o_proj`
    fp32 `250→183 µs`、q4 `105→84 µs`；`gate/up/down` fp32 `1456→1357`、q4 `524→502` ⇒ **热只快
    4…24%**。q4 在**权重已经在 L3 里**时仍只有 **5.4 GB/s**（fp32 同条件 17.5 GB/s）⇒ **连 L3 都
    喂不饱** ⇒ "吃不满 DRAM 带宽"不是访存问题，是**核内吞吐**问题 ⇒ **预取 / 访存模式不是杠杆**
    （DRAM 延迟只占 4%）。旁证：`vpmaddubsw` 峰值约 281 GMAC/s（2 条/周期 × 32 MAC × 4.4 GHz），
    `gate/up/down` 实测 **8.3 GMAC/s = 峰值的 3%** ⇒ 卡在**发射 / 依赖链**，不是算术峰值。
  - ⇒ **目标值（外推，非实测）**：若 q4 核能吃到 fp32 核的吞吐（~12 GB/s），单 token 应是
    `201.3 MB(层) + 144.6 MB(lm_head) ≈ 346 MB / 12 ≈ 29 ms`（≈34 tok/s，现 `15.4…15.8`）
    ⇒ **约 1.8× 的单核空间，且不用再压流量、不用动格式**。
  - ⚠️ 边界：① 探针里的"256 MB 扫一遍"是**标量读循环**（3.7 GB/s），只作冲刷、**不是**带宽标定
    （机器冷读带宽请看 `bench_thread_bandwidth.mojo` 的 12.00…15.89 / 18.70…26.50）；② 相邻档
    `ALOFA_Q4_MUB=0`（f32 展开）同形状是 2.53…3.88 GB/s ⇒ mub 快 **1.15…1.24×**（核级，与端到端
    +20% 同向）；③ 核级与端到端须**同轮**才可比（本机 runq 高）。
  - **按上面那条诊断改了 `_matmul_q4_mub_rows`：输出行从"编译器顺手展开 2 路"改成**显式
    `RU` 路一组**（`comptime MUB_ROW_ACC_TARGET = 8`）。**诊断链**（核级冷 cache + `objdump`）：
    内层每 (行,块) 约 **17 周期 / ~15 条指令 ≈ 0.85 IPC**，临界路径
    `vpmaddubsw(5) → vpmaddwd(5) → vcvtdq2ps(4) → vaddps(4) → vfmadd231ps(4)` ≈ 22 周期，而
    **只有 2 条独立链** ⇒ 卡在**发射 / 在飞访存数**（不是算术峰值、也不是 DRAM）。
    改法依据：`x32` / `Σx` / `scp` 都只取决于 `(rb, b)`、**与输出行无关** ⇒ 一块权重解量化
    一次喂 `RU` 个输出行**不花任何额外算术**，只换来 `RU` 条独立链 + `RU` 份在飞访存。
    ⚠️ `RU = 8 // RB`（**累加器组数恒定 ≈ 8**）：`RB=1`（decode）取 8、`RB=2`（prefill）取 4 ——
    固定 8 行会让 prefill 要 16 组，压到寄存器溢出。
  - **效果（本机同轮）**：核级冷 cache q4 **1.20…1.29×**（`gate/up/down` `524 → 383…430 µs`、
    `q/o_proj` `105 → 81…85`、`k/v` `21.8 → 17.5…18.2`）；端到端 `q4_0+q8_head` shards=1
    **`15.42…15.84 → 16.22…17.31` tok/s（+6…9%）**，`q4_0` 档 `9.59…10.35 → 10.34…11.43`；
    **prefill 没退**（`q4_0+q8_head` shards=1 `146.76…155.93 ms`，改前 `157…196`）。
    ⚠️ **shards=8 这轮机器太吵**（`29.26…37.55`，改前 `33.60…35.44`）⇒ **多核档未判定**，
    不能写"多核也快了"。
  - ✅ **数值逐位不变**（所以判据基线一条都没动）：每个输出行的**累加顺序一字未改**（仍是
    `b = 0…nb-1`、行末 `reduce_add`）⇒ 两道门照旧通过 —— `tests/unit/test_q4_mub_rows.mojo`
    **5/5**（核级，含"两条通路不许逐位相同"）与 `pixi run test-q4-mub` **一致率 0.8671875**
    （与改前 444/512 **完全相同**）。
- **2026-09-29（同日·`RU` 旋钮判到底：`RU=8` 就是最优点，`RU=16` 是**确定更慢**）**
  - 起因：跨进程 A/B 在本机噪声 **±25%**（同一个二进制相邻两轮 `gate` 393657 ↔ 581050 ns），
    而 `RU` 8→16 的效应 ≲10% ⇒ 五次交错配对**全部跨过 1.0、且方向随轮翻转**，只能写"判不了"。
  - 新探针 `scripts/bench_q4_rows_ru.mojo`（**同进程配对**）：把 `RU` 做成核的 **`RU_OVERRIDE`
    编译期参数**（生产路径不传 ⇒ 推导式与行为**一字不变**），于是**同一个二进制**里能实例化
    `RU = 4 / 8 / 16` 三档；每次计时前扫 256 MB 冲刷、三档**交错**出场、**逐轮轮换顺序**。
    ⚠️ 每轮只跑一遍时单样本比值散到 **0.58…2.48**（85 µs 的窗口会被调度打断，比跨进程还糟）
    ⇒ 改成**每轮每档 best-of-3** 才收得住。
  - 结果（每轮一个比值，**只有 min…max 全在 1.0 同侧才算分离**）：
    - `gate/up/down`（最大的形状）：`RU8/RU16 = 0.798…0.938`，**7/7 同侧** ⇒ `RU=16` **慢 6–20%**；
      `RU4/RU8 = 1.109…1.395`，7/7 同侧 ⇒ `RU=4` **慢 11–40%**。
    - `q_proj/o_proj`：`RU4/RU8 = 1.096…1.258`，7/7 同侧；`RU8/RU16 = 0.721…1.041`（6/7 偏
      `RU=8`，**未分离**）。
    - `k_proj/v_proj`（只有 13–17 µs）：两对都跨 1.0 ⇒ **这个形状判不了**。
  - ⇒ **`MUB_ROW_ACC_TARGET = 8` 保持不变**；`RU=16` 由"判不了"升级为**判否** —— 不是噪声问题，
    与 `objdump` 数到的"`RU=16` 整体栈流量更多"（含栈 `vmov` 220 → 398）同向。
  - 自证：三档输出**逐位相同**（`RU` 不动每行的累加顺序）⇒ 探针直接报 `逐位相同 896/896 ·
    4864/4864 · 128/128`，保证比值是"同一个东西的两种跑法"，不是两件不同的事。
  - ⚠️ **边界**：本机长期过载 ⇒ 绝对值只作同机同轮比较；分离与否只在**本轮这三个形状**上成立。
- **2026-09-30（`Σx` 修正挪 int32 域：指令少了、时间没少 ⇒ 判不了，不采用）**
  - 想动的那一处：`−8·Σx` 原来先转成 f32 再加（`vaddps`，4 周期），改成在 **int32 域** 补回
    （`vpaddd`，1 周期）⇒ 每 (行, 块) 依赖链 22 → ~18。
  - ✅ **指令层面确实改成了**（`objdump` 数 `RU=8` 那两臂）：`vaddps` **29 → 20**、`vpaddd`
    **0 → 9**；`vpmaddubsw` / `vcvtdq2ps` 各 9 不变，栈 `vmov` 90 / 91 持平。
  - ✅ **数值逐位相同**（不是"差不多"）：`q` 与 `Σx` 都是整数、|q + Σx| < 2^24 ⇒ 两条路在
    f32 里都精确；探针自证实测 **896/896 · 4864/4864 · 128/128** ⇒ 判据基线一条都不用动。
  - ❌ **时间上判不了**（同进程配对：256 MB 冲刷 + 每轮每臂 **best-of-5**、9 轮、逐轮轮换
    出场）：逐轮 `Σx f32/int32` = `q/o 0.931…1.098`、`gate 0.815…1.073`、`k/v 0.882…1.461`
    —— **三组都跨 1.0**，中位 ≈1.00；64 MB 冲刷 + best-of-3 那轮（中位 1.026 / 1.011）同一个
    读法。⇒ **不采用**，核里默认仍是 `SX_I32=0`（f32 那条）。
  - 读法：`RU=8` 给出 **8 条独立累加链**之后，**链长已经不是瓶颈** —— 真瓶颈是每行块 38.4
    条指令里那 **23% 的栈存取**。以后凡是"缩依赖链"的改动，先数有几条独立链。
  - ⚠️ **顺带否掉一个方法假设**：冲刷量试过 **64 MB**（≫ L3 12 MB，理论上够），结果**更散**
    —— `gate` 的 RU8/RU16 从 256 MB 那轮的 0.798…0.938（7/7 同侧）变成 0.776…1.111（6/9）
    ⇒ "一轮越短配对越紧"是错的，长冲刷把机器状态压得更一致 ⇒ **回到 256 MB**。
  - ✅ **`RU=16` 更慢这个结论扛住了两套口径**：本轮（256 MB + best-of-5、9 轮）`q/o` 的
    RU8/RU16 = **0.794…0.978（9/9 同侧）**、`gate` 0.816…1.060（8/9）⇒ `MUB_ROW_ACC_TARGET = 8`
    不变。
- **2026-09-30（砍栈存取第一发：`bias` 挪到块循环后读 ⇒ 判不了，已回退）**
  - 先量清楚 spill 是谁：`objdump` 看 `RU=8` 的内层循环，每行的 `raw`（16 B）与 `d`（块缩放，
    4 B）被 **spill 到栈再读回**（`vmovdqu 0x540(%rsp)` / `vmulss 0x130(%rsp)`）⇒ 8 行一轮
    ≈18 store + ≈18 load，每行块 ~10 次栈存取（26%）。
  - 想动的一处：那 `RB*RU` 个 f64 `accu` 在块循环**前**就把 `bias` 取好、循环后才加 ⇒ 要
    **跨整个块循环存活**，白占 8 个寄存器/栈槽。改成循环结束后再读 `bias` ⇒ **数值逐位相同**
    （`bias + Σ` 的相加顺序没动；探针 896/896 · 4864/4864 · 128/128）。
  - ❌ **换来 4 条、没换来时间**：栈 `vmov` 90 → 86（每行块 10.0 → 9.6）；同进程配对（256 MB
    冲刷 + 每轮每臂 best-of-5、11 轮、两臂轮换）逐轮 `A/B` = `q/o 0.939…1.028`、
    `gate 0.876…1.057`、`k/v 0.897…1.143`，**三组全跨 1.0**（中位 ≈1.00）⇒ 判不了 ⇒ 按纪律
    回退，不留没换来时间的旋钮。
  - ✅ **读法（这次真正的收获）**：spill 不是"寄存器不够被逼出来的"，是**调度器为了让 8 行数
    据同时在飞而主动 hoist 出来的** —— 白送它 8 个寄存器，只回收 4 次栈存取。⇒ 光靠"省寄存器"
    砍不动这 26%；要砍就得**减少同时在飞的值**：要么少在飞几行（与 `RU=8` 换来的 8 条独立
    链**直接冲突**），要么让每行的 `d` + `raw` 合成**一次**访存（在飞的值直接减半；代价是
    18 字节块的尾部要 16 B 填充，或最后一块走老路）。
  - ⚠️ **算不拢的账（下一步该先量的）**：`RU=8` 一轮（8 行 × 32 MAC）实测 ≈136 周期，而三个
    理论界都远低于它 —— 发射（≈89 条 ÷ 4 宽 ≈22）、栈端口（18 store ≈18）、DRAM（144 B ÷
    2.6 B/周期 ≈55）⇒ 还差 2.5×，指向**访存延迟没盖住**（MLP 不够）。⇒ 先做 **L2 常驻对照**
    （`rows` 小到权重能进 L2，如 64 行 = 28 KB）判定是不是延迟受限，再决定加预取还是改结构。
- **2026-09-30（L2 常驻对照：访存只占 ~30% ⇒ 那 2.5× 缺口主要在核内）**
  - 做法：新探针 `scripts/bench_q4_cache_resident.mojo`，同一个二进制**三臂**（冷 = 每次计时前
    扫 256 MB；温 = 扫 64 KB，只把 L1d 32 KB 挤掉、L2 256 KB 留着；热 = 不冲），权重 ladder
    `rows = 64/128/256/512/896/4864`（28 KB → 2.4 MB）、`inner=896`；每轮每臂 best-of-5、11 轮、
    逐轮轮换出场。本机 i7-9700K：L1d 32 KB / L2 256 KB(每核) / L3 12 MB。
  - 每 (行,块) 纳秒：冷 6.13 / 4.64 / 4.52 / 4.03 / 3.94 / 3.63，温 3.61 / 3.05 / 2.77 / 2.64 /
    2.59 / 2.55，热 3.89 / 3.33 / 3.04 / 2.92 / 2.88 / 2.85 ⇒ **冷/热 1.58 / 1.39 / 1.48 / 1.38 /
    1.37 / 1.27**。
  - ⚠️ ladder 小头有固定开销：拟合 `t = F + k·(行,块)` 得 **F ≈ 4.5 µs**（冷臂，含冷启动下的激活
    量化 + 核 prologue）⇒ `rows=64`（总 11 µs）时它占 **41%** ⇒ 只看大头的斜率：**冷 3.58 ns、
    温 2.54 ns**，渐近比值 **1.41**。
  - ✅ **判词**：把权重整个摆进 L1/L2 只快 **1.3–1.6×（渐近 1.41）** ⇒ **访存只占约 30%**；
    "≈17 周期/(行,块) vs 理论 ~2.7 周期"那 2.5× 缺口**主要不在访存** ⇒ **否掉"访存延迟没盖住"
    这个主假设**，别先去动预取 / 增加在飞访存数 ⇒ 主战场是核内那 **温 2.54 ns ≈ 11.7 周期**
    （4.6 GHz）。
  - ⚠️ 一个六点全同向的怪现象：温臂（冲刷 64 KB）反而比热臂（不冲）**快 7–10%** ⇒ 怀疑是无
    冲刷时上一遍的 store 还没排空就被计进下一遍。两个常驻臂都远快于冷臂 ⇒ **结论不依赖它**。
  - 下一步：既然在核内，就该量清楚那 11.7 周期卡在哪（发射？端口？依赖链？）⇒ 若 `perf` 可用，
    用 `perf stat` 取 IPC 与停顿原因；⚠️ 需要一个**只循环核、不冲刷**的小 harness，否则 256 MB
    扫描会把计数器占满。
- **2026-09-30（"在飞值/spill 值多少钱" ⇒ 不值钱；合成取数那发被工具链挡住）**
  - 想做的"小改动"：每行把 `d`(2 B) 与 `raw`(16 B) 合成**一次 32 字节取**（在飞值 16→8、spill
    减半）。❌ **做不了**：合成后 `raw` 落在 `base+2…base+17`，跨 16 字节 lane ⇒ 需要
    `palignr`（或 `extracti128` / `vperm2i128`）。把编译器里 `llvm.x86.avx2.*` 与
    `llvm.x86.ssse3.*` 的**全名单** dump 出来核对：**三个都没有**（avx2 的 shuffle 类只有
    `pshuf.b`/`permd`/`permps`/`pblendvb`；ssse3 没有 `palignr`）⇒ 只能拿
    `2×pshufb + por` 凑那 2 字节移位，`w32` 还要 `2×permd + blend` 重组 ⇒ **净多 4–5 条 p5
    指令/行块**。
  - 于是先量"这买卖值不值"：核加 `JUNK` 开关（每行每块多挂一个**被迫跨循环存活**的值：从
    本块开头再取 16 B —— 与 `raw`(+2 B)、`d`(2 B) 地址都不同 ⇒ 不会被 CSE）。
  - ⚠️ **第一发作废**：本想写成 `× 0.0` / `+ 0.0` 来保持逐位相同 ⇒ **Mojo 的浮点带 fast-math**，
    整段被折叠 ⇒ `objdump` 显示两臂是**同一份代码**（892 条指令 / 237 次栈存取，一模一样）
    ⇒ 改成"真加 0/1"（**故意不逐位相同**，只比时间）才留得住。
  - ✅ 结果（冷 cache 扫 256 MB + 同进程配对 + 每轮每臂 best-of-5 + 逐轮轮换，9 轮 × 3 档）：
    加料臂 **6…32% 更慢，27/27 轮同向**（`基线/加料` = 0.685…0.963）。但 `objdump` 同时显示：
    指令 892 → **945**（每块每行 ≈ +7 条），而**栈存取 237 → 237（一次 spill 都没多）**
    ⇒ 这一发量到的其实是"**多干活**"的代价，不是 spill 的代价。
  - ✅ **判词**：① 多 ~47% 指令只慢 ~20% ⇒ **弹性 ≈0.4** ⇒ 与前面"省 1–4 条指令测不出来"
    （1/18 ≈ 5% 的效应压在 ±10% 噪声下）**自洽**；② 之前 `defer bias` 省 4 次栈存取 ⇒ **0%**
    ⇒ **spill 基本不值钱，指令数才值钱** ⇒ 合成取数那发（−4 次栈存取、+4…5 条 p5 指令）
    按这个弹性算是**净亏 ~10%** ⇒ **不做**。
  - 副作用：配对探针 `bench_q4_rows_ru.mojo` 的 D 臂从「`Σx`∈int32」改成「`RU`=8 + `JUNK`=1」；
    核的 `JUNK` 默认 `0` ⇒ **生产路径一字未变**（`RU`=8/`SX`=0 那档仍 892 条指令）。
- **2026-09-30（复核"弹性" ⇒ 修正为 ≈0.55，剂量-响应近似线性；顺带量出噪声底）**
  - 复核办法：**剂量-响应** —— 同一段加料代码走 **0 / 1 / 2 份**（`JUNK` = 0/1/3）。四臂：
    A = B = 基线（`A/B` 白送**噪声底**）、C = `×1`、D = `×2`。冷 cache 256 MB + 同进程配对 +
    每轮每臂 best-of-5 + 逐轮轮换，9 轮 × 3 档。
  - 剂量（objdump）：基线 **892 条 / 栈 237**；`×2` **995 条 / 栈 255** ⇒ 每块每行 **+13 条**，
    且**栈存取第一次真的涨了**（+18）⇒ `×1` 约 +6 条。
  - ✅ 中位数：噪声底 `A/B` = **1.068 / 1.020 / 0.985**；`×1` 慢 **+30 / +26 / +21%**；
    `×2` 慢 **+49 / +46 / +60%**。
  - ✅ **弹性 ≈0.55**（基线按每块每行 ~14 条：×1 = +43% 指令 → +26% 时间 ⇒ 0.60；
    ×2 = +93% → +49% ⇒ 0.53）⇒ **近似线性**（剂量翻倍，变慢也差不多翻倍）。
    ⚠️ 之前那个 **0.4 偏低**，改成 **0.5…0.6**（区间 0.42…0.79，取决于基线每块每行按 11 还是 18 条）。
  - ⚠️ **白送的噪声底**：**同一份代码**在两个臂位上跑，中位能差 **7%**，单轮散 ±10%（小档
    `rows=128` 单轮最差 0.716…1.342）⇒ **<10% 判不了，20% 勉强，≥25% 才稳**。
    ⇒ 这才解释通了"为什么以前省指令全是 0%"：那些改动减的是**静态**条数（整函数 892 条里
    减 4 ≈ 每块每行 **0.5** 条 ≈ −3.6% 指令 ⇒ 预期仅 −2%）—— **不是弹性小，是剂量太小**。
  - ⚠️ **"逐位相同"推不出"代码没变"**：`×1` 那档输出**逐位相同**（加进去的字节恰好恒为 0）却
    **慢 26%** ⇒ 判断"加料留没留下"只能看 objdump / 时间，**不能看数值**；反过来，先前那档
    "轻加料"才是真没了（asm 里连 `vpmovzx` 都没有）。
  - 判词不变：合成取数（每块每行 −2 次栈存取、+4…5 条 p5 指令 ⇒ 净 **+2…3 条**）⇒ 按弹性
    0.55 算 **慢 ~8…12%** ⇒ **不做**。
- ❌ **预展开 nibble 已否（2026-09-30，`PRE` 开关 + `bench_q4_rows_ru.mojo` C/D 臂）**：把块从
  18 B（2 B 步长 + 16 B 半字节）重排成 34 B（2 B 步长 + 32 B 已拆成字节的 nibble，就是
  `lo.join(hi)` 那个排布）⇒ 内层省掉 `vpand`+`vpsrlw`+`vpand`+`vinserti128`（objdump：892 →
  **864** 条，静态 −28 条 ≈ 3.5 条/(行,块)），代价是**权重流量 ×1.89**。数值**逐位相同**
  （纯重排，自检 896/896 · 4864/4864 · 128/128）⇒ 这次**只能**靠 objdump 判它在不在二进制里。
  ✅ 结果：中位 `B/C` = **0.812 / 0.747 / 0.806**（逐轮 26/27 同侧）⇒ **预展开慢 19…25%**，
  而两条噪声底（`A/B` 原布局对原布局、`C/D` 预展开对预展开）中位都在 ±5% 内 ⇒ 效应为真。
- ⚠️ **"用字节换指令"在这条路上是亏的 ⇒ 字节弹性 ≈1.0**：按冷/温分解（每 (行,块) 冷 3.58 =
  核内 2.54 + 访存 1.04 ns）反解 ⇒ 访存 1.04 → **2.12 ns**（字节 ×1.89 ⇒ 时间 ×2.04）
  ⇒ **字节弹性 ≈1.0…1.1（近线性）**，而指令弹性只有 0.55 ⇒ 折成总时间：省 21% 指令值 **−8%**，
  多 89% 字节付 **+27%** ⇒ 净 +19%，与实测慢 19…25% 自洽。
  ⇒ **别再做"用权重字节换核内指令"的改动**（预展开、任何让块变大的排版）；反过来**能减权重
  字节**的改动按 ≈1.0 收钱，但访存只占总时间 ~29% ⇒ 绝对收益打三折。
- ⚠️ 事故（记住）：给核加参数时用 `sed` 补调用点，把 `PR_C=1` 插进了**错误的槽位**（直调点
  原来只传 3 个参数 ⇒ `1` 落到 `JUNK` 槽 ⇒ 那臂跑的是 `JUNK=1,PRE=0`）⇒ 白跑一轮。
  **判据抓到了它**（那臂不逐位相同），且 **mangled 名自带参数值**
  （…，RB=1,RU_OVERRIDE=8,SX_I32=0,**JUNK=1,PRE=0**）⇒ 一眼定位。以后加核参数：① 看 mangled
  名核对槽位；② 逐位相同自检必须**四臂全过**才认。
- **2026-09-30** —— **线程路流式 bug：复现到了，并读出完整症状链；同时确认它是"观测即掩盖"型（Heisenbug）**
  - **复现**（`ALOFA_ENGINE_THREADS=1`，真权重，流式 `max_tokens=8`，一条一条发）：第 1 轮 **6/12**（后 6 条 **0 字节**），第 2 轮 **0/12**，进程最终**崩溃**。默认串行路不受影响（与 09-28 第十轮一致）
  - **症状链（这次读全了）**：流式应答被**截短**（`f_1` 694 B、`f_2` 390 B，结尾是 `finish_reason:null`、**没有 `[DONE]`**；完整的那条 1540 B）⇒ 流停在中途 ⇒ 连接不关、槽位不释放 ⇒ 8 个槽位耗尽 ⇒ `has_room` 拒掉新连接（0 字节）⇒ 循环**高速空转**（日志里每 150–200 ms 一次 `arena.create capacity=320`，无任何报错）⇒ 最终 abort
  - **崩溃栈**：`#8 Loop::run_threaded` ← `#4–#7 libAsyncRTRuntimeGlobals.so`（Mojo 异步运行时）← `#3 libc`。⚠️ 与 09-25 记的"崩溃点确定性在 `Tokenizer::encode`"**不是同一处** ⇒ 崩溃点确实是随机落点，**继续追"崩在哪一行"没有意义**，要追"谁写坏了堆"
  - ⚠️ **观测即掩盖（本轮最要紧的一条）**：给 `collect`/`dispatch` 各加一行 `print`（只打 slot/kind/字节数）后，**6/6 全部完整 1540 字节、帧序列完全正常**。这与 `-g` / `-O0` / gdb / ASan 一加就不崩是**同一性质** ⇒ 任何插桩都改变时序并掩盖它。**下结论前必须先确认"尺子本身没有改变被测物"**
  - **静态排查也走到头了**：循环线程在 `run_threaded` 里只取 handler 地址（loop.mojo:306）、之后不碰 handler（"循环不碰 handler"是 `EngineBox` 的明确契约，678–680）⇒ 0 号 engine 线程**独占**原型 handler ⇒ **"两线程抢 handler"排除**（这条此前只是推理，这次读代码确认）。调度器的 `InlineArray[MAX_BATCH]` 写入**全都有边界检查**（`n_arrived >= MAX_BATCH` 即 raise）⇒ 09-24 锁定的那个候选**证否**
  - ⇒ 剩下的唯一结构性出路（09-28 收口建议里的同一条）：**拆掉"跨线程交接时让 `Job`/`Result` 各自持有一份独立堆分配字节"这一层** —— 改成循环侧缓冲池 + 只传下标，或定长内联。只要字节的归属不跨线程移动，UAF 那条腿就不存在
- **2026-09-30（同日第二条）** —— **流式 UAF 崩溃修掉了：跨线程不再有堆块跟着活走**
  - **改动**：`Job.raw`（`List[UInt8]`）与 `Result.bytes` 改成**定长内联** `InlineArray[UInt8, MAX_JOB_BYTES=8192]` / `MAX_RES_BYTES=16384`。构造签名不变（仍收 `List`，内部拷进内联），所以 7 个 `Result(…)` / 4 个 `Job(…)` 构造点一行没动。两个方向的字节改成**在各自线程内**拷出来（`_run_job` 拷给 `parse_request`、`collect` 拷给 `Conn.enqueue`）⇒ **没有堆块的归属跨线程移动**，UAF 那条腿被拆掉
  - **判据**（`ALOFA_ENGINE_THREADS=1`，真权重）：逐条 12 条流式 × **3 轮 = 36 条全部完整**（1540/1549 字节），进程存活、零崩溃、零 `damaged`。修复前同一场景是 **6/12 → 0/12 → 崩溃**
  - **断开复现**（8 次中途断开 + 探活）× **3 轮**：探活**每轮 263 字节**，服务存活。修复前这个场景**必死**（它是八轮排查里最强的放大器）
  - **没碰坏默认路**：`ALOFA_ENGINE_THREADS=0`（默认串行）并发 8 条流式 **5588 × 8 全完整**、非流式 12/12 ⇒ 改动对生产路径无影响
  - **门禁**：`pixi run test` **394 PASS / 0 FAIL**
  - ⚠️ **这条修复为什么此前被判过"证否"**：09-28 第七轮做过一版近似实验（`Job` 改**借用** + `Result` 抄一份 + 用过的活停进邮箱永不析构）⇒ **照样崩**。那次不是同一件事：**"借用"是跨线程传指针**，比传字节更危险；这次是字节归属**根本不跨线程**。判别实验与真设计的差别就在这里
  - ⚠️ **仍存在的边界（未解决，且不是这次引入的）**：线程路**并发**流式会截短（并发 2 有一条 1375 vs 完整 1540；并发 4 有 184/357/1035）。**原版同样有**（原版并发 4 也出现 1200）⇒ 是一个独立的老问题，这次没有引入也没有修掉
  - ⚠️ **正式化前必须处理**：`MAX_JOB_BYTES=8192` / `MAX_RES_BYTES=16384` 超限目前是**打印并截断**（因为 `raise` 会波及不容抛的调用链）。要么把界提到安全值并补一条门，要么按原计划做循环侧缓冲池 + 只传下标
- **2026-09-30（同日第三条）** —— **端到端剂量-响应跑出来了：那 2× 缺口，核内和访存两头都不是大头**
  - 工具 `scripts/bench_e2e_junk_pair.mojo`（**同进程两臂配对**，`JUNK=0` 对 `JUNK=3`，shards=8，9 轮 × 每臂 best-of-5，逐轮轮换出场）。自检：两臂生成的 32 个 token **逐位相同** ⇒ 加料数值中性 ✓
  - 结果：`JUNK=0` **16.83…24.67 ms**、`JUNK=3` **17.62…29.36 ms**；比值 **0.74…1.23，中位 1.10**
  - ⚠️ 读法（关键）：**同一份剂量在单核探针里是 +49 / +46 / +60%（中位 ≈+49%）**，端到端只有 **+10%**。按近似线性折算 ⇒ **核内指令只占端到端时间的 ~20%**
  - 与 09-30 另一条拼起来 ⇒ **访存 ~30% + 核内 ~20% ≈ 一半，剩下那一半在第三处**：既不是权重访存、也不是 GEMM 核内指令（候选：shards=8 的并行/同步与切分、KV、注意力、调度、非 GEMM 算子）
  - ⇒ **"继续砍每块每行那几条指令"这条路对端到端没有意义了**（就算把核内指令砍一半，端到端也只回来 ~10%）。下一步该做**分层计时**（`src/alofa/model/profiler.mojo` 已接线，`ALOFA_PROFILE` 编译期门），先把那一半**定位**到具体阶段再谈优化
  - 副作用（备用）：真路径的 `JUNK` 剂量开关已搬进 `avx2.mojo` / `qwen.mojo`（默认 0 ⇒ 生产二进制与加常量之前**逐位相同**）
- **2026-09-30（同日第四条）** —— **分层计时：端到端 ~98% 是 GEMM，非 GEMM 只有 2%；真杠杆是"带宽没吃满"**
  - 工具：`src/alofa/model/profiler.mojo`（`ALOFA_PROFILE` 编译期门，已接线到 `qwen.mojo`）+ `dump_profile()`。⚠️ 样本 64 个 token **含 prefill**，绝对值偏大，**比例才是要读的东西**
  - 每 token 各段（ms）：**投影核 96 次/tok = 29.11（303.21 µs/次）**；**输出投影 `lm_head` = 11.55**；rmsnorm×2/层 0.15 · rope+KVcopy+attn 0.35 · swiglu 0.34 · 残差 0.03 ⇒ **非投影串行部分合计仅 0.87**
  - ⇒ **GEMM 占 ~98%**（29.11 + 11.55），**非 GEMM 只 ~2%** ⇒ 与 09-22 那条"串行算子切分天花板 2%"**完全对上**：RMSNorm/RoPE/注意力/残差**彻底不值得再碰**
  - ⚠️ **与上一条怎么调和**（关键）：剂量-响应说"核内**指令**只占端到端 ~20%"，分层说"核占 ~98%" ⇒ **核的那 98% 里，指令不是瓶颈，等内存才是** —— 加料多出来的指令被访存等待**藏住了**。再砍每块每行那几条指令，对端到端**没有意义**
  - ⇒ **真杠杆是带宽利用率**：09-28 已量到 alofa `shards=8` 达成带宽只有 **11.62…12.26 GB/s**，而同机 `bench_thread_bandwidth.mojo` 的 **T=4 峰值 18.70…26.50** ⇒ **还有 ~2× 空间**，且 llama.cpp `-t 4`（11.80…16.91）也没吃满 ⇒ 这是双方共同的差距所在
  - 下一步：**量 shards = 1/2/4/8/16 的达成带宽与端到端 tok/s**，看曲线在哪饱和、以及它离线程带宽峰值有多远 ⇒ 定位"为什么 8 路只读到 12 GB/s"（候选：切分方式/预取/分配/NUMA/同步）
- **2026-09-30（同日第五条）** —— **shards 扫描 + 同进程天花板：单线程 decode 已经把 DRAM 带宽吃满 ⇒ 唯一杠杆是"减每 token 权重流量"**
  - 工具 `scripts/bench_model_shards.mojo`（交错 + 同轮配对 + 现测天花板）。⚠️ 它加载的是 **fp32 权重（`tensors.f32`，每 token 流 1.976 GB）**
  - **分片收益只有 1.3×**：`shards=2` 0.87…1.08×（**含 1.0 → 判不了**）；`shards=4` **1.07…1.29×**；`shards=8` **1.06…1.30×**（后两档两口径同向 ⇒ 分片确实更快，但**只有 ~1.3×**，而核级 T=8 是 3.69–5.76×）
  - **同进程现测天花板**（⚠️ 分母必须同进程：本机读带宽不同时刻 12.0–15.0 GB/s 会漂）：纯读 `B_1 = 9.62` / `B_8 = 16.85` GB/s；真实 `linear` 核 `K_1 = 9.09` / `K_8 = 11.00` GB/s（139 MB ≫ L3）
  - 🔑 **最关键的一行**：`shards=1` 实测 **8.57–10.21 GB/s**，对纯读天花板 **890‰ – 1061‰**、对 kernel 天花板 **942‰ – 1123‰** ⇒ **单线程 decode 已经贴到（甚至打平）DRAM 天花板** ⇒ 脚本自己的判词：**"串行段没油水，别去切注意力 / norms"**
  - ⇒ 三条证据现在**完全一致**（剂量-响应：核内指令只占 ~20%；分层：GEMM 占 98%、非 GEMM 2%；带宽：单线程已吃满）⇒ **decode 是彻底的带宽-bound**。**再砍核内指令、再切串行算子，都换不到端到端时间**
  - ⇒ **唯一真杠杆 = 减每 token 的权重流量**：fp32 **1975.8 MB** → `q4_0` **745.8** → `q4_0+q8_head` **345.9 MB**（**5.7×**）。这与 09-29 那条对上了：`q4_0+q8_head` 端到端 **13.6…17.5 → 28.7…30.2 tok/s**。再往下就是新开的两份文档（`docs/plan/04-gguf-reader.md`、`05-kv-cache-quantization.md`）
  - ⚠️ **待核实的口径疑点**：`bench_e2e_junk_pair` 也加载同一份 fp32 权重（日志 `tensors.f32 size=1976393216`）却报 **16.83…24.67 ms/步**，而按 1.976 GB/token ÷ 10 GB/s 应有 ~200 ms ⇒ **它那个"一步"很可能不是一整个 token**。**两个脚本的每 token 口径不能混用**，下一步先把这个对上再拿数字当结论
- **2026-09-30（同日第六条）** —— **口径对上了；主战档重标 ⇒ "带宽吃满"的结论被翻转：q4 档只吃到 ~55%**
  - ✅ **口径疑点解开**（上一条留的）：两个探针**都是整 token**（都量 `model.step`，返回 `[prefill, step0, …]`），差的 ~10× 来自**权重档 + 统计口径**：`bench_e2e_junk_pair` = **q4_0+q8_head**（`QwenForward` 第 4 参 `True` = 在线量化 + `enable_q8_output()`）且取 **best-of-5 的 min**；`bench_model_shards` 原为 **fp32**（3 参）且取**中位数**
  - **主战档重标**：`bench_model_shards.mojo` 已切到 `q4_0+q8_head`（`BYTES_PER_TOKEN` 改 345.9 MB + 第 4 参 `True` + `enable_q8_output()`）
  - 结果：分片收益 `shards=4` **1.24…1.43×**、`shards=8` **1.36…1.84×**（两口径同向，非配对区间 1.35…1.91× 不重合）⇒ **比 fp32 档（~1.3×）更明显**
  - 同进程天花板（本轮）：`B_1 = 9.20` / `B_8 = 15.57`；`K_1 = 9.41` / `K_8 = 12.77` GB/s
  - 🔑 **每 token 带宽：`shards=1` 只有 4.83–5.72 GB/s（对纯读 525‰–621‰、对 kernel 513‰–607‰）**；`shards=8` 7.76–9.27（对 kernel 607‰–725‰）
  - ⚠️ **翻转**：fp32 档那条"单线程吃满 890‰–1061‰"**不能外推到主战档** —— 主战档只吃 **~55%**。⇒ **"decode 已是带宽-bound、没油水"这个判断在主战档不成立，还有 ~1.8×**
  - ⚠️ **一处冲突别糊过去**：脚本自己的判词是"离 kernel 天花板 ≥30% ⇒ 串行段/调度有油水"，但分层计时（q4 档）说**非 GEMM 只有 2%** ⇒ 更可能是 **q4 档 GEMM 核自己没吃满带宽**（解包/依赖链上的真开销），**不是**串行段。判据：JUNK 加料加的是可被乱序藏住的"无关指令"（+10%），而解包是依赖链上的开销 —— 两者性质不同，不能用前者否认后者
  - 🧭 **决定（gguf reader vs KV 量化）**：**KV 量化先不做** —— 0.5B / 64 tok / bs=1 下 KV 流量 ≈ 5.5 MB vs 权重 345.9 MB ≈ **1.6%**，天花板就这么大（与"串行算子天花板 2%"同源）。**gguf reader 值得推**，但它的价值是**口径**：能用 llama.cpp 官方 gguf ⇒ 同格式对比、第三方可复现
  - ⇒ **顺序**：先花一步定位"q4 核为什么只吃 55% 带宽"（最大单一空间 ~1.8×，不需要新格式），**然后**再推 gguf reader
- **2026-09-30（同日第七条）** —— 🔑 **那 55% 缺口定性了：q4 真入口核只吃到 fp32 核带宽的一半（核内依赖链，不是 DRAM）**
  - 工具 `scripts/bench_q4_cold_bandwidth.mojo`（⚠️ 量的是**真入口** `q4_matmul_rows_k` 行带单行分支、mub 默认；256 MB 冲刷保证冷 cache；**同轮**含 fp32 对照）
  - **冷带宽（三个真投影形状）**：`q/o`(896×896) fp32 **11.95** / q4 **5.60**；`gate/up/down`(4864×896) fp32 **12.60** / q4 **6.13**；`k/v`(128×896) fp32 11.06 / q4 **3.22** GB/s
  - ⇒ **带宽比 `q4/fp32` ≈ 0.47 / 0.49 / 0.29**（字节比恒为 **7.11**，时间却只快 **3.33…3.46×**）⇒ 流量少了 7.11× 却只快 3.4× ⇒ **q4 核没吃满带宽**
  - 🔑 **判据**：两种核读的都是**顺序**权重，唯一差别在核内 ⇒ **差别就是核内开销**；而 fp32 核能吃到 ~12 GB/s ⇒ **DRAM 不是墙，q4 解包→点积的****依赖链**是墙**
  - ⚠️ **与"JUNK 加料只 +10%"不矛盾**：JUNK 加的是**可并行的无关加法**（累加进 `sink`，乱序能藏），不延长依赖链；q4 解包是 **load→unpack→madd→acc 的真依赖链**，藏不住 ⇒ **这不是"指令条数"问题，是"延迟/依赖链"问题** ⇒ 减指令条数这条路再次被否，方向该换成**缩短依赖链**
  - ⚠️ **强对照（同机同量化档）**：llama.cpp `q4_0` 单线程达成 **8.17…9.79 GB/s**（09-28 量），alofa q4 核只有 **5.60…6.13** ⇒ **llama 高 40–60%** ⇒ 这是**核实现的差距**，不是硬件上限 ⇒ 值得攻、也有证据可攻
  - ⇒ 那 ~1.8× 端到端空间就在这里：q4 核若吃到 fp32 核的 ~12 GB/s，大形状时间即可再快 ~2×
  - ⚠️ 待查：脚本末尾打印了 `unreachable -910.0`（三个形状都正常输出后才出现），暂不影响上述读数，但下次跑要顺手定位
- **2026-09-30（同日第八条）** —— **读了 llama.cpp 的 q4_0 GEMV 核（`4da6337`）：它比我们快的三处具体做法**
  - 位置：`ggml/src/ggml-cpu/arch/x86/repack.cpp` → `ggml_gemv_q4_0_8x8_q8_0`（AVX2 分支 `gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>`）。⚠️ 另有一条 `ggml_vec_dot_q4_0_q8_0`（`arch/x86/quants.c:701`，**非** repack 路径）
  - ① **权重 repack 成 8x8 交错布局**（`block_q4_0x8`），一次算 **4 行 × 8 列块**（`acc_rows[4]`，输出按 `(y*4+i)*bs + x*8`）⇒ 多行共享一次连续取数，ILP 与预取都更好
  - ② **`vpshufb` LUT 展开 nibble**：`signextendlut = (-1…-8, 7…0)`（一条 `vpshufb` 取代 `and`+`shift`+`or` 那一串）
  - ③ 🔑 **点积在 int32 域累加**：`mul_sum_i8_pairs_acc_int32x8` 四次嵌套盖满块内 32 值，**最后才** `cvtepi32_ps` + **每 32 值 1 次** `fmadd` ⇒ **浮点累加链极短**
  - **对比 alofa**：mub 通路把 `vpmaddubsw` 的结果在**浮点域**累加（`acc8u[…] = acc8u[…] + SIMD[f32](…)`，见 `avx2.mojo`）；行带 `RB` 默认 2（试过 16，落在 ±25% 内）
  - ⇒ **三条可搬的东西**：(a) **整个点积累加挪 int32 域** —— ⚠️ 注意 09-30 试过的只是「`Σx` 挪 int32」，**不是**「点积累加挪 int32」，这是另一件事、还没试过；(b) **多行交错布局**（repack，不只是"多行同时算"）；(c) `vpshufb` LUT 展开
  - ⚠️ 待确认：本机 `llama-bench` 是否真走 repack 的 8x8 路径（repack 需在加载时转换）。若它走的是 `ggml_vec_dot_q4_0_q8_0`（每块一次 `fmadd` 到**单个** `acc`，同样有链），那 8.17–9.79 GB/s 的对照要换一句解读 ⇒ 下一步先 `gdb`/环境变量确认走的是哪个
- **2026-10-01** —— ✅ **第一步：确认本机 llama-bench 走的是 repack 的 8x8 路径（对照成立，地基稳了）**
  - 编译期（读 `build/CMakeCache.txt`）：`GGML_CPU_REPACK=ON`（CMakeLists 默认 ON）、`GGML_AVX2=ON`、`GGML_FMA=ON`
  - **实测**（不用推理）：`gdb -batch` 对 4 个符号打断点跑 `llama-bench -m qwen25-05b-q4_0.gguf -p 5 -n 8 -r 1 -t 1` ⇒ **只命中 `ggml_gemv_q4_0_8x8_q8_0`**（多次），`ggml_vec_dot_q4_0_q8_0` / `_4x4_` / `_4x8_` **一次都没命中**
  - ⇒ decode 走的是 `gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>`（AVX2）⇒ 09-28 量到的 **8.17…9.79 GB/s** 就是**这条**路跑出来的 ⇒ 上一条那三处做法（8x8 交错布局 / `vpshufb` LUT / int32 域累加）**确实是我们 5.60…6.13 的对手**，照搬的地基成立
  - 方法备注：判断"走哪条路"用 **gdb 断点看命中**（首次命中即 disable 自己，开销可忽略）比读源码推理硬得多 —— 这套模板有 4x4 / 4x8 / 8x8 三个特化，光看代码定不下来
- **2026-10-01（同日第二条）** —— ❌ **搬 ③（int32 域累加）在 alofa 上不可行：每块必须乘各自的 scale，f32 FMA 躲不掉**
  - 读 `avx2.mojo` mub 内层（`SX_I32` 那一段）：块内**已经是** int32 域 —— `p = vpmaddubsw(w32,x32)` → `q = vpmaddwd(p, ones)`，之后才 `cast[f32]`；`−8·Σx` 也已能在 int32 域补（`SX_I32=1`）
  - 🔑 **否掉的理由**：`dsx = d * scp[rb*nb + b]` 是**每块各自的 scale**（q4 的 `d` × 激活的 scale），块与块不同 ⇒ **每块都必须转 f32 乘一次 `dsx` 再累加** ⇒ 那条 f32 FMA **躲不掉**，无法把累加整体留在 int32 域
  - ⚠️ **llama 也是每块一次 `fmadd`**（`acc_rows[i] = fmadd(cvtepi32_ps(iacc_row_i)*col_scale*row_scale, acc_rows[i])`，在块循环内）⇒ **这不是两边的差异来源**，09-30 第八条把 ③ 列为"可搬"是我读代码时的误判，特此更正
  - ⇒ 那 40–60% 的差异**不在累加域**，只剩两个候选：**多行/布局**（llama 8x8 交错 + 一次 4 行，alofa `RB=2`）与 **nibble 展开方式**（llama `vpshufb` LUT）
  - ⚠️ **② 也存疑（先别盲做）**：AVX2 的 `vpshufb` 是 **128 位 lane 内**操作，32 个 nibble 要**两次**才展得开 ⇒ `and`+`shift`+`join` 那一串换 LUT 很可能**不减反增** ⇒ 先验再改
  - ⇒ **改用最便宜的判决性实验**：`RB`（行带宽度）是**现成旋钮**，改它无需动布局 ⇒ 用核级冷带宽探针扫 `RB = 2/4/8`，若 RB 变大就能逼近 llama 的 8.17…9.79 ⇒ **"多行并行"就是答案**，之后才值得投工程量做 8x8 repack
- **2026-10-01（同日第三条）** —— ❌ **"多行并行"单独不成立：`RB 2→8` 没把 q4 带宽推向 llama**
  - 做法：临时把 `qwen.mojo` 里 `matmul_q4_mub_*_rows_band_vec[2, ·]` 的 `RB` 改成 8（4 处），重编 `bench_q4_cold_bandwidth.mojo`（它调真入口 `q4_matmul_rows_k`）。**已还原**（`git diff` 干净）
  - 结果（冷带宽 GB/s）：`q/o` **4.75**（RB=2 时 5.60）、`gate/up/down` **5.35**（6.13）、`k/v` 3.34（3.22）⇒ **没有改善，大形状反而更慢**
  - ⚠️ **同轮噪声极大**：同一脚本的 **fp32 档从 12.60 掉到 9.20（−27%）** ⇒ 单次跨进程对比在这个机器上**不可靠**，读这种数只能看**同轮比值**（`q4/fp32`：RB=2 是 0.468/0.487，RB=8 是 0.451/0.582 ⇒ 仍在 0.45…0.58，没有跳到 1.0 附近）
  - ⇒ **结论**：llama 的 8x8 快，不是因为"一次算 8 行"这件事本身（alofa `RB` 就是这个旋钮，且 09-22 已记 `RB=16` 落在 ±25% 内）⇒ **剩下的真差异是"交错布局"本身**：llama 把 **8 行的同列块交错存**，一次加载覆盖 8 行的连续字节；alofa 是行主序，`RB` 变大只是循环里多走几行，**字节的组织没变**
  - 🧭 **四条的顺序据此重排**：① 的前半（多行数）已否 ⇒ ① 只剩"改布局（repack）"，它是**唯一还没被否掉的**候选，但工程量最大；② `vpshufb` 待验（推理存疑：AVX2 是 lane 内，32 nibble 要两次，可能不减反增）⇒ **先验再改**；③ 已否
  - ⚠️ **方法教训（给后面所有实验）**：这台机器上跨进程/跨时刻对比会漂 27% ⇒ 下一步若要判 ② 或 ①，必须走**同进程配对**（像 `bench_q4_rows_ru.mojo` 那样两臂 + 逐轮轮换 + best-of-5），不能拿两次运行的绝对值比
- **2026-10-01（同日第四条）** —— ⚠️ **B（`vpshufb` LUT）在数学上就测不出来 ⇒ 不做，直接转 A（布局）**
  - **指令账（源码层面）**：现在是 `lo = raw & 0x0F` / `hi = (raw >> 4) & 0x0F` / `w32 = lo.join(hi)`。`vpshufb` 在 AVX2 是 **128 位 lane 内**、且只用索引的**低 4 位** ⇒ 它能替掉低 nibble 那条 `and`（1 条），但**替不掉 `>> 4`** —— 高 nibble 不先右移就取不到 ⇒ **净省 ≈ 1 条指令/块**
  - **换成时间**：已量出的**弹性 ≈0.55**（09-30），每块每行 ~14 条指令 ⇒ 省 1 条 ≈ **−7% 指令 ≈ −4% 时间**
  - **对比噪声底**：本机 **<10% 判不了**（09-30 实测同一份代码两臂中位能差 7%、单轮散 ±10%；今天 fp32 档同脚本跨轮还漂了 27%）⇒ **−4% 无论怎么做都测不出来**
  - ⇒ ② **不是无效，是小到不可测** ⇒ 做它只会得到一次"判不了"，白花一轮。llama 用 LUT 也只是省 1 条（它顺带完成符号扩展），**不是它快 40–60% 的原因**
  - 🧭 **转 A**：四步里 ③ 已否（每块须乘各自 scale）、① 的前半（多行数/`RB`）已否（RB=8 无改善）、② 判不了 ⇒ **交错布局是唯一还没被否掉的候选**
  - ⚠️ **A 的验证不能只改 `RB`**：`RB` 变大只是循环里多走几行，**字节组织没变**；要验的是"8 行同列块**交错存**" ⇒ 必须动核的**取址逻辑**，需要写一个交错布局的核 + 一个**同进程配对**探针（本机噪声 27%，跨进程比绝对值不可靠）
- **2026-10-01（同日第五条）** —— ⚠️ **投 A（布局）之前先摆一个反证：行主序本身不是带宽障碍**
  - 反证来自同一份探针的**同轮数据**alofa **fp32 核是行主序、无交错，却吃到 11.95–12.60 GB/s**（≈ 纯读天花板）⇒ **"行主序读不满带宽"这个前提是错的**
  - ⇒ q4 只吃到 5.60–6.13 就不是"字节摆得不好"，而是**每字节的核内工作量**太大（fp32 每 4 字节 1 次乘加；q4 每 32 字节要做 32 次解包 + 点积）
  - ⚠️ 于是 A（交错布局）的收益假设**本身存疑**：交错布局能摊薄循环控制/地址计算，**但不减少每字节的解包工作量**
  - 🧭 **投工程量前先花一步做便宜的核对**（三选一，都不用改生产路径）：① 数 alofa q4 核**每块指令数**（objdump）并与 llama 的 8x8 核对比 —— 若 alofa 明显更胖，那"每字节工作量"就是真凶，布局改了也白改；② 查 alofa 每块额外取的 `scp[rb*nb+b]`（激活 scale）是不是带来了**额外流量**（块 18 B + 4 B ⇒ +22%）；③ 用**核级配对探针**直接比"同形状下行主序 vs 手工交错"的时间
  - ⚠️ 现状小结（四步走完三条半）：③ 已否（每块须乘各自 scale）· ① 前半（多行数）已否（RB=8 无改善）· ② 判不了（省 ~1 条 ≈ −4% < 噪声底 10%）· **A 的假设现被自己的数据反证** ⇒ **llama 那 40–60% 的真因还没找到，先别动布局**
- **2026-10-01（同日第六条）** —— 🔑 **投工程量前的核对①做完了：真凶是"每块指令数是 llama 的 ~4 倍"，不是布局**
  - 做法：`objdump -d --start-address` 取两边核，脚本自动找**最大的向后跳转**定内层循环体，按 mnemonic 分类计数
  - 对象：alofa `matmul_q4_mub_rows_band[RB=2,JUNK=0]`（`target/q4cold`，@0x14ef0）vs llama `gemv_q4_b32_8x8_q8_0_lut_avx<block<4,8>>`（`build/bin/libggml-cpu.so`，@0x104470）
  - **原始条数**：alofa 函数 2150 / **内层循环体 1360**；llama 函数 189 / **内层循环体 139**
  - 🔑 **归一到"每次 `vpmaddubsw`（= 一块 32 个权重）摊到多少条指令"**（这个口径对循环边界识别误差不敏感）：**alofa 1360/19 = 71.6 条；llama 139/8 = 17.4 条 ⇒ 4.1×**
  - **构成对比（症结所在）**：alofa **取/存 213**（≈16%，栈 spill，与 09-28 那条"每行块 38.4 条里 23% 栈存取"同源）、**f32加 64**、点积 ubsw/wd 各 19、int→f32 19、f32乘加 19、移位/跨通道/掩码各 14；llama 取/存仅 **14**、LUT(`vpshufb`) 9、点积各 8、**int32加 7**、**f32乘加仅 1**
  - ⇒ **alofa 每块要做 ~4 倍的家务，主要是 spill（栈存取）+ 一条 f32 累加链**；llama 块内在 int32 域加完、每块只留 1 次 f32 乘加
  - ⇒ **A（交错布局）可以放下了**：同一探针里 fp32 行主序已吃满 11.95–12.60 GB/s ⇒ 内存路径没问题；这 4× 是**核自己胖**，换布局摊不掉
  - ⚠️ **口径不确定处（别拿它当最终数）**：循环体是自动按"最大向后跳转"切的；两边每次迭代服务的**行数不同**（alofa `RB=2`、llama `acc_rows[4]`），且 llama 那 8 次 ubsw 是否同时服务 4 行**未逐条确认** ⇒ "71.6 vs 17.4" 是**每块**口径（对行数不敏感，可作主判据），但倍数请当作 ~3–4× 的量级而非精确值
  - 🧭 **下一步改打"寄存器压力/spill"，而不是布局**：候选 ① 降 `RU`（现在 8 个 f32 累加器 × `RB` = 寄存器吃紧的直接来源，09-29 记 `RU=8` 最优/`RU=16` 更慢，但那是**时间**口径，没看过**spill 条数**）；② 把累加器换成 **int32 域**（少寄存器、且能去掉 `int→f32` 19 条 + `f32加` 64 条中的大部分）——注意这与 09-30 否掉的 ③ 不同：那里否的是"**整个**累加留在 int32"，这里说的是**累加器本身**用 int32、每块末尾仍要乘 scale
- **2026-10-01（同日第七条）** —— 🔴 **扫 `RU` 的 spill 曲线：两条曲线反向交叉 ⇒ "降 RU 减 spill"这条思路是错的，顺带把早上那个 4.1× 修正成 ~2×**
  - 工具 `scripts/bench_q4_ru_spill.mojo`（新）：用核的 `RU_OVERRIDE` 在**同一个二进制**里实例化 `RU = 1/2/4/8/16`，`RB=1`（decode）⇒ **累加器组数 = RU** ⇒ 这条轴就是寄存器压力本身。冷 cache（每遍扫 256 MB）+ 9 轮 × 每臂 best-of-5 + 逐轮正/倒序交替。自证：五档输出**逐位相同**（4864/4864、128/128）✓
  - **时间这一臂（附带，如期判不了）**：`gate` 中位 RU=1 462 · **RU=2 386** · RU=4 450 · **RU=8 418** · RU=16 488 µs —— 区间大量重叠（`RU=2` 343…446 对 `RU=8` 390…460）⇒ 与 09-29/09-30 那两次一致，**判不了**
  - 🔑 **主产出：`objdump` 出来的两条曲线**（每档内层循环体，按"最大向后跳转"切）

    | `RU` | 循环体 | 取/存 | 取/存% | `ubsw` | **每块(体/ubsw)** | `vaddps` | `int→f32` |
    |---|---|---|---|---|---|---|---|
    | 1 | 149 | 33 | **22%** | 1 | **149.0** | 3 | 1 |
    | 2 | 371 | 102 | 27% | 3 | 123.7 | 10 | 3 |
    | 4 | 452 | 130 | 28% | 5 | 90.4 | 16 | 5 |
    | **8（真路径）** | 665 | 208 | **31%** | 9 | **73.9** | 29 | 9 |
    | 16 | 1104 | 383 | **34%** | 17 | 64.9 | 55 | 17 |

  - 🔑 **读法（关键）**：`RU` 是**真实的权衡轴**，两条曲线**严格反向** —— `RU↑` ⇒ 每块指令 **149 → 64.9**（解量化复用摊得更好）；`RU↑` ⇒ spill **22% → 34%**（33 → 383 条，寄存器压力）。⇒ **历次 `RU` 扫描"判不了"不是测量无能，是这条曲线在 RU≈2–8 交叉、时间天然平坦**
  - 🔴 **修正我早上提的方案①（"降 `RU` 减 spill"）：方向是错的**。降 `RU` 确实减 spill（31%→22%），但每块指令从 73.9 **暴涨到 149**（一次解量化只喂 1 行，复用全没了）⇒ 净效应抵消，这条轴**两头都比现在差**
  - 🔴 **修正早上那条"每块指令是 llama 的 4.1×"**：那是**没除行数**的口径。按**每块每行**：alofa `RU=8` = 73.9 ÷ 8 = **9.2 条**；llama = 139 ÷ 8 块 ÷ 4 行 = **4.35 条** ⇒ 真实差距 **≈2.1×，不是 4.1×**。⚠️ llama 那 8 次 `ubsw` 是否同时服务 4 行**仍未逐条确认** ⇒ 这个 2.1× 带着同样的不确定，请当量级看
  - ⇒ **剩下那 ~2× 不只是 spill**：`RU=8` 把 spill 全去掉也只到 9.2 × 0.69 = 6.3 条/块行，仍 > 4.35 ⇒ 另一半在**解量化方式 + 累加域**（`vaddps` 29 条 / `int→f32` 9 条 / 拆半字节的 `vpand`+`vpsrlw`；llama 用 `vpshufb` LUT + int32 域，每块只剩 1 次 f32 乘加）
  - 🧭 **下一步改打"解量化方式 + 累加域"，别再动 `RU`**（它已在权衡带上，两头都更差）。这正好是我早上第 2 条建议（累加器改 int32 域），现在证据更强：**注意它与 09-30 否掉的 ③ 不同** —— 那里否的是"**整个**累加留在 int32"（做不到，每块要乘各自 scale），这里说的是**累加器本身**留在 int32、每块末尾仍乘 scale
  - 副作用：探针里比值分母 `med[3]`（RU=8 那档）在算前三档时还没赋值 ⇒ 第一次跑打出 10766279‰ 那种垃圾值，已改成"先算完所有中位再打印"
- **2026-10-08** —— 🔴 **"解量化 LUT 化 + 累加器改 int32 域"动刀前的核算：有证据表明做完也量不出来 ⇒ 先不下刀；顺带发现 09-30 那条"单线程已吃满 DRAM"站不住**
  - **① "换累加域" 09-30 就已经做过了**：`SX_I32` 旋钮把 `−8·Σx` 那一项从 f32 域（`vaddps`）挪到 int32 域（`vpaddd`），`objdump` 数得 `vaddps` 29→20、`vpaddd` 0→9 ⇒ **指令总数持平**（只是换域）；同进程配对三组比值都跨 1.0（中位 ≈1.00）⇒ 判不了、不采用
  - **② 结构上没有"可累加的东西"**：`W_Q4 = 8` ⇒ 一次 `vpmaddubsw` 就消化掉一整个块的 32 个 nibble，`vpmaddwd` 顺带完成 4 组部分和 ⇒ 每块只剩 1 个 int32 向量，而**下一拍必须乘本块的 scale**（q4_0 每块一个 `d`）⇒ 无法跨块在 int32 域累加 ⇒ "累加器改 int32 域"**没有对象**
  - **③ LUT 化能省的就是 ① 那一类指令**：llama 那个核名字里的 `lut` 是 **`signextendlut`**（把 nibble 查表成带符号 int8，见 `ggml/src/ggml-cpu/arch/x86/repack.cpp` 567–575），**不是**拆 nibble —— llama 拆 nibble 也还是 `and 0x0F` + `srli` + `and`。⇒ 换它换来的是"省掉 `Σx` 那次 int 域加法"，代价是 **2 条 `vpshufb` + 块末多一项修正** ⇒ **净指令数不降反升**；而 ① 已证明省这类指令换不到时间
  - ⚠️ **更正我 10-01 的说法**：10-01 那条把 llama 的 `vpshufb` 写成"LUT 展开 nibble"，**错了**，它是符号扩展查表
  - **④ 就算净省几条也换不到端到端**：10-01 端到端剂量-响应 —— 单核探针加料 **+49%**，端到端只 **+10%** ⇒ 核内指令对端到端的弹性本来就很低
  - ⇒ 结论：**这两条不是不能做，是做了换不到数字**（本机噪声 ±25%，效应 ≲2–5%）⇒ 先不下刀，把工程量留给下面的复核
  - ⚠️ **顺带暴露一个更要紧的问题（要复查）**：为复核 09-30"单线程已吃满 DRAM"重跑 `bench_model_shards`（fp32 档）⇒ **这次 `B_1 = 7.30` < `K_1 = 9.95`** —— 纯读天花板比 kernel 天花板还低，**自相矛盾**（纯读不该慢于 kernel）⇒ `shards=1` 利用率从 09-30 的 **890–1061‰** 掉到 **637–735‰**，脚本自己的判词也从"串行段没油水"翻成"切串行段有油水"
  - ⇒ **该探针的"同进程现测天花板"会漂 1.4×**（脚本自己也警告跨运行能差 1.6×）⇒ **09-30 第五条"单线程 decode 已吃满 DRAM"这个结论站不住**，它是拿一个会漂的分母除出来的
  - 🧭 **下一步（按优先级）**：先把"到底吃没吃满"这个**根结论**坐实 —— 它是"继续优化核内"还是"改打访存/并行"的分水岭。做法：多次重跑 + 同时记录 `B_1`/`K_1` 的**配对比值**（而不是各跑一次取中位），并判 `B_1 < K_1` 这种自相矛盾是不是稳定出现（若是 ⇒ 天花板测法本身有 bug）
- **2026-10-08（同日第二条）** —— 🔑 **带宽复核：主战档 q4 只吃到天花板约 55% ⇒ 09-30 那条"已吃满 DRAM"是 fp32 档的结论，不能外推**
  - 工具：`bench_model_shards.mojo` 加了**每轮配对**输出（不再"分子取中位 ÷ 分母取中位"—— 那种算法正是 09-30 与 10-08 两次读数不同步能溜过去的地方）。同一轮内 `A_1`（shards=1 每 token 实测带宽）／`B_1`（纯读天花板）／`K_1`（kernel 天花板）配好再除
  - **主战档（`q4_0+q8_head`，每 token 345.9 MB）3 轮**：`A_1` 6.42 / 6.31 / 6.43 GB/s；`B_1` 11.35 / 12.49 / 11.55；`K_1` 11.25 / 9.57 / 11.03 ⇒ **对纯读 546–557‰、对 kernel 572–583‰**；`shards=8` 也只 9.69–10.38 GB/s ⇒ 511–548‰ / 592–634‰
  - ⇒ **主战档只吃到 ~55%，离天花板还有约 45%**。而 09-30 第五条量的是 **fp32 档**（1.976 GB/token）的 890–1061‰ ⇒ 两条**不是同一档**，不能互相外推
  - **关于 `B_1 < K_1` 那个自相矛盾**（10-08 第一条记的）：这次 **3 轮都没出现**（`B_1/K_1` = 1.00 / 1.30 / 1.04，纯读都 ≥ kernel）⇒ 那次是**机器干扰的个例，不是测法 bug**；09-30 与 10-08 的差异主要来自机器整体变慢（分子掉 46%、分母只掉 24%，不同步）⇒ **该探针的结论必须同轮配对读，单看跨运行的绝对值没意义**
  - ⚠️ **与 10-01 剂量-响应怎么合**：那次说"核内指令只占端到端 ~20%" ⇒ 现在带宽只占 ~55% ⇒ **还有约 25% 既不在核内指令、也不在带宽上**（候选：q4 块 18 B 的访存模式/解量化、TLB、调度、`shards` 并行效率）。这是**新出现的缺口**，之前被"fp32 档吃满"这条结论盖住了
  - 🧭 **下一步**：在**主战档**上定位那 25–45%（09-30 的分层计时与剂量-响应都是在别档/别口径量的）。最便宜的第一刀：判**访存模式** —— 主战档是 18 B/块 + 解量化，与 fp32 的顺序 4 B 不同；用同进程配对比"q4 真入口"与"同流量 fp32 顺序读"的达成带宽，看差距是访存模式还是别的
  - 副作用：修了新加输出的单位坑（`ceiling_read`/`ceiling_kernel` 返回**字节/秒**，中位那几行是 `gbps_or_fail` 转过的 ⇒ 自己除 1e9，否则打出 11348046163）
- **2026-11-03** —— 🔑 **主战档那 55% 找到出处了：指吞吐在极限（每块每行 83 条 vs llama 17.4 条 = 4.8×），推翻 10-01 的"2.1×"**
  - **先确证 llama 的循环结构**（`ggml/src/ggml-cpu/arch/x86/repack.cpp:522`，`gemv_q4_b32_8x8_q8_0_lut_avx`）：外层 `y` 走**输出行（一次 1 行）**，中层 `x` 走 `nc/8` 个 8-列块，内层 `b` 一次处理 **8 个 q4 块**（权重是 `block_q4_0x8`，8 块交错），累加器 `acc_row` **只有 1 个**
    ⇒ 内层迭代服务 **8 块 × 1 行 × 8 列** ⇒ **每块每行 = 139 ÷ 8 = 17.4 条**
  - **alofa**：`RU=8` 循环体 665 条，服务 **1 块 × 8 行 × 8 lane** ⇒ **每块每行 = 665 ÷ 8 = 83 条** ⇒ **4.8×**
  - ⚠️ **推翻 10-01 的"2.1×"**：那次我假设 llama 每块服务 4 行（`acc_rows[4]`），**错了** —— 那个 4 是**输出列**的展开，行永远一次 1 行 ⇒ 10-01 早上报的 4.1× 反而更接近真相。口径教训：**行和列必须分清**
  - 🔑 **指令预算（决定性）**：主战档每 token = 345.9 MB ÷ 18 B = **19.22 M 块**；×83 条 = **1.59 G 条/token**；@30.2 tok/s ⇒ **48 G 指令/s**；本机 i7-9700K **8 核 @3.6 GHz**（`nproc=8`、max 4.9）⇒ 需 **IPC 1.67/核** ⇒ **正在极限附近** ⇒ 这就是"只吃到 ~55% 带宽"的直接原因（CPU 喂不饱内存控制器）
  - **llama 对照**：17.4 × 19.22 M = 334 M 条/token；@40.4 tok/s ⇒ **10 G 指令/s** ⇒ 8 核只需 IPC 0.35 ⇒ **真带宽 bound** ✓（40.4 tok/s 需 14 GB/s，在 `B_8 = 18.9` 之下，做得到）
  - ⇒ **两家不是同一个瓶颈**：alofa 指吞吐受限、llama 带宽受限 ⇒ 那 ~35% 的差距**只能从"减每块指令"拿**，加并行/改访存都没用
  - 🔑 **差距的结构来源**：llama 用 `block_q4_0x8`（8 块交错）⇒ 一次 `loadu` 就拿到 8 个块的字节 ⇒ 加载与解量化指令被 **8 块摊薄**（每块约 0.7 条）；alofa 每块 18 B 连续 ⇒ **每块独立做一遍解量化**（`vpand`/`vpsrlw`/`join` 5–6 条）+ 每块 4 B 的 `Σx` 加载，摊不掉
  - 🔴 **更正 10-01 对 A（交错布局）的判决**：当时说"A 可以放下"，理由是"fp32 行主序已吃满带宽 ⇒ 内存路径没问题"。**访存那半句对，但结论错了** —— A 的真正价值是**把每块的加载/解量化摊给 8 个块**，即**减指令**，不是提带宽。现在指吞吐是瓶颈 ⇒ **A 回到台面上，且是最大的那一项**
  - ⚠️ 同时**spill 仍是最大单项**：`RU=8` 循环体里取/存 **208 条 = 26 条/块行**（占 31%），而 llama 只有 1.75 条/块行 ⇒ 减 spill 与交错布局要**一起**做才有量级
  - 🧭 **下一步（真工程，不是旋钮）**：把权重改成 **8 块交错的 `q4_0x8` 式布局** + 降寄存器压力，目标每块每行 83 → ~30–40 条 ⇒ 指吞吐不再是瓶颈 ⇒ 端到端 30 → 40+ tok/s（带宽侧 `shards=8` 的 `B_8 = 18.9` 撑得住 40.4 tok/s 所需的 14 GB/s）。代价：布局转换 + 量化器 + 核重写 + **逐位验证**（这是本仓库的硬门槛）
- **2026-11-03（同日第二条）** —— 🔴 **撤销"每块每行 83 条 / 4.8×"：静态条数被冷路径污染，时间算术给出硬上界 ≤ ~31 条**
  - 起点：`objdump` 数 `RU=8` 循环体得 665 条，除以 8 行得 83 条/块行 ⇒ 由此推出"4.8×、指吞吐在极限"。**这个推论作废**
  - **验伪一（结构性）**：那 665 条里有 **11 条 `call` + 11 条 `vzeroupper` + 12 条 `pop`**，而 `call` 的目标是 `fp16_bits_to_f32`（9 次）与 `_matmul_q4_wide_rows` 自身（尾行回落路径）。`block_scale` 本身**已经**是 `@always_inline`，那 9 次 `call` 是它内部"inf/NaN 或对齐异常"的**冷分支**（`if (half & 0x7C00) == 0x7C00 or block_base & 1 != 0`）⇒ **物理上落在循环地址区间里，但正常数据一步都不会执行**
  - **验伪二（时间算术，决定性）**：`RU=8` 那臂 396 µs 跑完 4864 行 × 28 块 = **136,192 个块行** ⇒ **2.91 ns/块行** ⇒ @3.6 GHz 只有 **10.5 个周期**（@4.9 GHz 14.3）⇒ 即便 IPC 打满 3，上限也只有 **~31 条/块行**（@4.9 GHz ~43）。83 条需要 IPC 7.9 —— **不可能**
  - ⇒ **真实执行量是 ~20–30 条/块行，不是 83**。同理 llama 的 139 条也含它自己的错误分支 ⇒ **"17.4" 同样是上界而非实数**，两边的静态条数**都不能再拿来相比**
  - ⇒ **"4.8×" 撤销；10-01 的 "4.1×" 与 71.6 也一并撤销** —— 两次都是同一个错误：拿"地址区间里的指令条数"当"执行条数"。**纪律：静态条数只能当上界，跨核比较必须用时间或硬件计数器**
  - 🔴 **由此推翻 11-03 第一条给的方向**："先做交错布局（收益最大）"的前提是"解量化占 83 条里的一大块"。真实分母只有 ~25 条，`and`/`shift`/`join` 那 5–6 条最多占 20–25%，而布局改动**要重写累加语义**（llama 的 int32 累加器 lane = 块，不是列），不是搬字节 ⇒ **收益上界 ~15–20%，成本是核重写 + 量化器 + 逐位验证，不划算**
  - 🔴 **主战档 55% 的解释也要改**：原来归因"指吞吐在极限"，现在看**单核每次块行只差 llama ~1.3 倍**（10.5 周期 vs llama `-t 1` 24.0 tok/s 折出的 ~7.8 周期，⚠️ llama 那个折算依赖它的 token 字节数，口径不硬）。⇒ 缺口更像是 **单核内存并行度（在飞 load 数）** + **多线程扩展**（alofa `shards=8` 只 1.56×，llama `-t 4/-t 1` = 1.68×），不是"指令太多"
  - 🧭 **下一步（便宜且不会被冷路径骗）**：只在**时间域**做假设检验 —— ① 给内层加软件预取（十几行、不改布局），看 6.4 GB/s 是否上抬，直接判"是否 MLP/延迟受限"；② 若预取无效，再查 `shards` 的扩展损失（切分粒度 vs 带宽竞争）。**布局留到这两条都排除之后再谈**
- **2026-11-03（同日第三条）** —— 🟡 **按 head 切注意力：正确性已证，速度**量不出来**（机器 load 15.56 / 8 核）**
  - 起因：`shards` 只切 7 个投影包装（`TaskGroup` 模式），`run()` 里的 attention / RMSNorm / RoPE / 残差 / SwiGLU / argmax **全串行**。按 Amdahl，串行段 55% ⇒ `shards=8` 的天花板只有 `1/(0.55+0.45/8)=1.65×`，与实测 1.57–1.98× 吻合 ⇒ **并行扩展的瓶颈是"串行段没被切"**
  - 为什么挑注意力：`avx2.attention` 是 `for head in range(n_heads)` 的**纯独立循环**，头与头无共享状态 ⇒ 按头切**不改任何一个头的算术**，是本项目里唯一"切了逐位不变"的并行。RMSNorm/RoPE/残差也可切，但都要跨线程归约或改 shape 契约
  - 改动（commit 见下）：① `avx2.attention` / `scalar.attention` 加 `head_lo..head_hi`（默认全量，老调用点语义不变）；② `qwen.attention_k` 透传；③ 新增 `_attention_shard` + `attention_k_shards`（`TaskGroup`，**只在 `q_len==1` 即 decode 启用**，prefill 仍串行）；④ `run()` 改走分片版
  - 两个设计点（都不是可选项）：**`scores` 必须每片一份**（它是共用 scratch，`q_len×kv_len` 覆盖后即弃；按 `k*q_len*kv_len` 个 float 错开，decode 下每片只要 `kv_len` 个，现有 `max_tokens²` 缓冲本来就够）；**余数均摊**（14 头 8 片 ⇒ 2,2,2,2,2,2,1,1；照 `_linear_shard` 全给最后一片会变成 1×7+7，并行度掉一半）
  - ✅ **正确性（这条是硬的）**：`bench_model_shards.mojo` 加了同进程守卫 —— 同权重、同 prompt、只改 `shards`，`shards=1` 与 `shards=8` 的贪心 id **逐个相同（9 个 id）**
  - ❌ **速度：量不出来，不是"没变快"**。两次运行：改前 `shards=8` 比 = 1.57–1.98×；改后 = 1.22–1.76×。但本机 `loadavg 15.56 / 8 核`（`runq` 17–25），**2 倍超订 ⇒ 8 线程没有空闲核可用**，两次数值都落在噪声里，**不能用来说好或坏**
  - ⚠️ 因此本轮**不下性能结论**。要判它，得在 `loadavg < 8` 的窗口重跑同一份 `target/shards3`（守卫先过，比值两个口径同向且排除 1.0 才算）
  - 🧭 下一步：① 等空闲窗口重测（零成本）；② 若仍无收益，嫌疑是**每层 1 次 TaskGroup 派发 × 24 层**的调度开销吃掉 2 个头/片的活（单头工作量仅 ~19 µs），届时应把 24 层的注意力**合并成一次派发**（按层×头展平），而不是每层开一次 group
- **2026-11-03（同日第四条）** —— ✅ **按 head 分片的正确性已双重证明；"合并 24 层派发"这个方案作废（结构上做不到）**
  - **方案作废**：上一条留的"把 24 层注意力合并成一次派发"**不可行** —— 注意力的输入依赖本层 `k_store` 追加与 RoPE、输出立刻喂给本层 `o_proj`，它夹在层循环中间，抽不出来批量做。且派发开销本就不是嫌疑：24 次 `TaskGroup` ≈ 0.1 ms，只占 ~50 ms/token 的 **0.2%**
  - ✅ **门一（核级）**：`tests/unit/test_avx2_parity.mojo` **9/9** 通过 —— 标量与向量后端仍逐位一致（含注意力），说明 `head_lo..head_hi` 的默认全量路径没有被改坏
  - ✅ **门二（整网，强）**：`pixi run test-model-avx2` **4/4** 通过，其中 `test_greedy_generation_matches_token_for_token`（128 token 对照 fp32 参考）与 `test_incremental_decode_matches_full_prefill` 都是**在默认 `shards=8` 下跑的** ⇒ **按头分片的路径确实被执行并通过了整网逐 token 门**，不是只跑了串行分支
  - ⚠️ 速度仍未判：本机 `loadavg 15.56 / 8 核`。**并行扩展这条线到此为止**（注意力之后，剩余串行段都被层循环的依赖钉死：q/k/v → attention → o_proj）
  - 🧭 结论：**"并行扩展"不是当前可用的杠杆** —— 切到注意力就到顶了，再往上要拆层循环（重写前向）。下一个杠杆要么等空闲窗口确认注意力这步的真实收益，要么回到单线程（负载无关）的注意力分支消除（10-03 记：注意力 30/44 是 data-dependent 分支）
- **2026-11-03（同日第五条）** —— 🧭 **"单线程瘦身注意力"的前置：仓库里没有注意力的尺子，先造尺子**
  - 事实：`grep` 全部 `scripts/bench_*.mojo`，**没有任何一个直接调 `attention` 核**（`bench_head_split` 只在正文里提到"注意力"三个字，它量的是输出投影）。所以"注意力 26.4%""30/44 是 data-dependent 分支"这两条**至今没有一把能复现的尺子**
  - 先做算术核对（这是我怀疑的地方）：decode 一个 token 的注意力算术量 = 24 层 × 14 头 × 1 query × 17 kv × 64 dim × 2（打分+值归约）≈ **0.73 MFLOP**。若真占 26.4% × 24.6 ms = **6.5 ms**，则有效算力只有 **0.11 GFLOP/s** —— 比同核向量能力低三个数量级 ⇒ **那 6.5 ms 不可能主要是算术**，要么桶里混了 `copy_into`/`rope`/`prof.now()` 的开销，要么是每调用一次的固定成本
  - ⇒ **下一步必须先量出"一次 `attention` 调用多少纳秒"**（按真实 decode 形状：`n_heads=14, n_kv_heads=2, head_dim=64, q_len=1, kv_len=17/33/65`），再谈优化。规格：
    ① 新脚本 `scripts/bench_attn_decode.mojo`，只 import `alofa.core.ffi.monotonic_ns` + `alofa.core.memory.Arena` + `alofa.kernels.cpu.avx2.attention`（**不 import 模型**，否则编译几分钟）
    ② 单线程、每档 `REPS` 次、取 **min**（过载机器上 min 比中位数更抗抢核）；报 ns/调用 + 折算成"占一个 token 的百分比"（÷ 24 层 ÷ 24.6 ms）
    ③ 同轮对照：`head_lo=0,head_hi=14` 与 `head_lo=0,head_hi=7` ×2 次，验证**按头切分没有引入回归**（这条负载无关）
  - ⚠️ 在拿到这把尺子之前，**不许再对注意力做任何"优化"** —— 本轮已经因为无尺子下了两次错结论（83 条 / 4.8×，均已撤销）
- **2026-11-03（同日第六条）** —— 🔴 **尺子造出来了：decode 的注意力只占 token 的 0.69–2.01%，账本那个 26.4% 不是 decode 的数**
  - 新工具：`scripts/bench_attn_decode.mojo`（仓库里**第一个**直接调 `attention` 核的探针；此前全部 `scripts/bench_*.mojo` 都没碰过它）
  - 口径：真实 decode 形状 `n_heads=14, n_kv_heads=2, head_dim=64, q_len=1`，`kv_len` = 17/33/49，单线程、7 轮 × 400 次取 **min**（过载机器上 min 比中位数抗抢核）
  - **实测（两次运行，同向）**：
      `kv_len=17` 7.1–8.1 µs/次 → 占 token **0.69–0.79%**
      `kv_len=33` 13.8–16.3 µs/次 → **1.34–1.59%**
      `kv_len=49` 20.6–21.7 µs/次 → **2.01–2.12%**
  - 两条读出：
    ① **随 `kv_len` 线性**（17→49 涨 2.9×）⇒ 是真实 `O(kv_len)` 算术，**不是**每次调用的固定成本；折算 ~3.6 GFLOP/s
    ② **按头切分本身的代价 ≈ 0**（4.11% / −0.10% / −1.06%，在噪声内）⇒ 上一轮加的 `head_lo..head_hi` 没有引入额外固定成本
  - 🔴 **"注意力占 26.4%"作废（对 decode）**：真的是 0.7–2.1%，差一个数量级以上。那个 26.4% 只能来自**非 decode 形状**（prefill `q_len` 大时注意力按 `t×kv` 增长，`q_len=128` 就 ×128）。09-29 那张"GEMM 53.3% / 注意力 26.4% / 归一化 20.3%"表**不能再用在 decode 上** —— 它把 prefill 形状的分解套到了单 token 上
  - ⇒ **上一轮"按 head 并行"的收益天花板就是 ~2%**（正确性仍然成立、代码无罪，但它不可能是主战档 55% 缺口的解释）。同时 11-03 第三条用 `1/(0.55+0.45/8)` 做的 Amdahl 推算也**一并作废**（那个 0.55 的前提没了）
  - 🔧 顺带记录的坑：`Arena.keep_alive()` 必须放在**最后一次使用之后**。它只是借用 `self`，而 Mojo 的 ASAP 销毁发生在最后一次使用之后 —— 放在 `alloc` 后面等于把寿命截断在那里，紧接着的填数就 SIGSEGV（本探针踩了两次才定位）
  - 🧭 下一步：用同一把尺子的写法**重新测 decode 的真实分段预算**（`q4_matmul_rows` 一次多少 ns × 每 token 168 次），别再引用 prefill 形状的表
- **2026-11-03（同日第七条）** —— 🔴 **decode 真实分段预算到手：投影 55% / lm_head 24–38% / 全部串行算子 1.3–1.7%**
  - 工具：`scripts/bench_decode_budget.mojo`（真实权重 `q4_0 + q8_head`，单进程，`model.set_profile` + `dump_profile`；`ALOFA_PROFILE` 临时置 `True`，跑完已改回 `False`）
  - **实测（shards=1，墙钟 61.4–61.8 ms/token）**：
      `核本身（q4，96 次/token）` **33.4–33.9 ms = 55%**
      `输出投影 lm_head` **23.8–24.0 ms = 38%**
      串行段合计 **0.82 ms = 1.3%**（rmsnorm 0.18 / rope+KVcopy+attn 0.30 / swiglu 0.31 / 残差 0.03）
  - **实测（shards=8，墙钟 43.4–46.0 ms/token）**：核 22.7–25.1 ms，lm_head 10.1–13.7 ms，串行段 0.77 ms
  - 🔴 **由此推翻"主战档 55% 落在核之外 / 串行段没被切"这个整体叙事**：串行算子实测只占 **1.3%**，即便全部并行到 0 也只能省 0.8 ms。之前用 `1/(0.55+0.45/8)` 推 Amdahl 的那个 0.55（串行占比）**量出来是 0.013**
  - 🔴 **按 head 分片注意力：短上下文下是净亏，已加阈值**（`comptime ATTN_SHARD_MIN_KV = 256`，`qwen.mojo`）
      证据：同一探针 `rope+KVcopy+attn` 桶 `shards=8` 时 **1.94 ms → 加阈值后 0.27–0.29 ms**（`shards=1` 基线 0.30 ms）
      折算：单层实活 ~13 µs（微探针 7 µs/次），每层开一个 `TaskGroup` 的固定开销 ~68 µs ⇒ `kv_len` 需 > ~165 才够本，取 256 留余量
      ⇒ **切分不改算术（逐位已验证），但切分不免费** —— 这是"切了必须量"的第二个实例
  - ⚠️ 插桩自身有开销（96 次 `prof.now()` ×2/token）⇒ 表中绝对值偏大，**只读占比**；`shards=1` 墙钟 61.8 ms 明显高于无插桩时的量级，不要拿它跟 tok/s 比
  - 🧭 **下一个杠杆已经明确**：`lm_head`（151936×896，q4 读 76.6 MB/token）+ 层投影的**单核带宽**。两者都是同一件事 —— 喂不饱内存。`shards=8` 把 lm_head 从 24.0 压到 11.4 ms（2.1×）、核从 33.4 压到 25.1 ms（1.33×），说明**核这条路的并行扩展效率明显低于输出头**
- **2026-11-03（同日第八条）** —— ✅ **层投影：MLP 不是瓶颈；加核已到顶（1.34× 天花板，实测 1.33×）；空间全在核里**
  - 工具：`scripts/bench_proj_mlp.mojo`（三张表；单线程跑 A/B，仅 C 用 `TaskGroup`；每档 7 轮取 min；240 MB 缓冲先写一遍再计时；q4 块 scale 统一填 fp16 `1.0` 以**避开 `block_scale` 的 inf/NaN 冷分支** —— 账本 11-03 第二条的教训）
  - **A 单核纯读（同一块 153 MB，K 条独立流）**：K=1 **12.46** / 2 12.87 / 4 11.61 / 8 13.23 / 16 12.17 / 32 12.29 GB/s
      ⇒ **MLP 曲线是平的（±8%，在噪声内）**：单核挂 1 条流和挂 32 条流一样快 ⇒ **"在飞 load 数不足"这个假设被否掉**，软件预取/多路流水不是病灶
  - **B 真实 q4 核（24 层 × 7 投影 = 一趟 decode 的全部层投影，191 MB）**：**45–56 ms ⇒ 3.6–4.5 GB/s**（同进程纯读 12.5）
      ⇒ 核只用到单核纯读的 **29–36%**；按每块折算：纯读 **5.3 周期/块**、核 **10.5 周期/块** ⇒ **卡在指令吞吐，不在访存**（与 11-03 第二条的弹性实测 IPC≈2.8 一致）
  - **C 多线程纯读（互不相交，201 MB 总量）**：1 线程 11.4 / 2 → 14.9 / 4 → 16.5 / **8 → 16.63** / 16 → 18.5 GB/s
      ⇒ **整机 DRAM 天花板 ≈ 16.6–18.5 GB/s**，而**单核一个就能吃 12.4 = 全机的 75%**
      ⇒ 加核的理论上限 `C(8)/A(1) = 16.63/12.46 = 1.34×`，而 profiler 实测层投影加 8 片是 **1.33×**（33.4→25.1 ms）⇒ **切分粒度这条路已经撞到硬件天花板，再切无收益**
      ⇒ 这同时解释了 lm_head 为什么能拿 2.1×（它单核被指令卡在 3.2 GB/s，离 12.4 更远，所以加核还有用），而层投影只有 1.33×（它离单核天花板近）
  - 🔴 **两条结论（先写判据、后看数字，判据是事先写进探针 docstring 的）**：
      ① **不改切分粒度** —— 1.34× 是硬上界，已用掉；
      ② **改核有量级空间** —— 单核 3.6–4.5 → 12.4 GB/s，算术上 **2.8–3.4×**
  - 🔴 **但必须同时记下真实上界，否则又会去追一个不存在的 2×**：整机 DRAM 只有 16.6 GB/s，
      每 token 权重 = 层投影 201 MB + lm_head 76.6 MB = **278 MB** ⇒ **这台机器的 decode 地板 = 278/16.6 = 16.7 ms/token（≈60 tok/s）**
      llama `-t 4` 43 tok/s = 23.3 ms ⇒ **llama 离地板 1.39×**；alofa 24.6 ms ⇒ **离地板 1.47×**
      ⇒ **改核的真正上界是 1.47×（24.6 → 16.7 ms），不是 2.8×**；而且要吃满这 1.47× 必须"改核 + 多核"同时成立
      ⇒ 也说明**主战档与 llama 的差距现在是 5%，不是 55%** —— 两边都已经贴在这台机器的内存墙边
  - 🧭 下一步（若继续）：核重写的目标定成"单核跑到 12 GB/s"，验收口径用 B/A ≥ 0.9；天花板提前认下 = 1.47×
- **2026-11-03（同日第九条）** —— 🔴 **撤销第八条的"核只用到纯读 29–36%"：那是 f32 展开核，decode 走的是 mub（整数点积），已在 50–67%**
  - 触发：第八条的 B 表量的是 `matmul_q4_f32`（`_matmul_q4_halves`，f32 展开）。但 decode 的 `project` 走 `q4_matmul_rows_k_shards` ⇒ `q4_mub_enabled()` **默认开** ⇒ **`vpmaddubsw` 整数点积**（09-29 已翻默认）。**量错了核**
  - **同轮对照**（`bench_proj_mlp` 扩成 B1/B2，loadavg 15，绝对值被压低，只读比例）：
      `B1 f32 展开` **3.32 GB/s** ／ **`B2 mub` 4.69 GB/s** ／ `B2/B1 = **1.41×**`
      `B2 ÷ 同轮纯读(8.96–11.4)` = **0.41–0.52**；空载口径配 `bench_model_shards` 的 `K_1 = 6.82–8.34` 与纯读 12.46 ⇒ **0.55–0.67**
      ⇒ **修正：生产核已用掉单核纯读的 50–67%，不是 29–36%**；第八条据此推的"改核 2.8–3.4×"**作废**
  - 🔴 **决定性结构事实**：全二进制只有 **1 条 `vpmaddubsw` + 1 条 `vpmaddwd`**，就在 `_matmul_q4_mub` 的循环体里（`objdump` 全符号计数）⇒ **整数点积这条路早就走着了，且每块只花 1 条**
      ⇒ "改成 llama 的整数点积能拿 2.8×"这个方向**本身就不存在**——已经是它了
  - 剩下的每块开销在 **f32 域**：`vpmaddubsw → vpmaddwd → vcvtdq2ps → vfmadd231ps`（账本 09-29 那条已记 22 周期），且 `SX_I32` 选项已把 `−8·Σx` 挪进 int32 域（`vpaddd` 替 `vaddps`），**这部分已经吃过一次**
  - 🔴 **修正后的空间**：核级上界 **1.5–1.65×**（从 0.55–0.67 收到 1.0），端到端仍被第八条那条 **16.7 ms DRAM 地板**夹住 ⇒ **e2e 上界 1.47× 不变**
  - 🧭 若继续，唯一具体可做的改动：**8 块一组的 int32 域累加**，每 8 块只做一次 `(d·d_x)` 的 f32 乘加，把每块那条 `vcvtdq2ps + vfmadd` 摊到 1/8
      验收：`B2` 提升 ≥1.3× + `test_q4_mub_rows` 相对 L2 ≤1%（现 0.33–0.38%）+ 整网 greedy ≥0.85（现 0.8672）
  - ⚠️ **测量窗口警告**：本轮 loadavg 15（前几轮 7–9），B 表绝对值比第八条低 30–40%。新变体的 A/B **必须同轮交替**，否则量到的是负载不是代码
- **2026-11-03（同日第十条）** —— 🔴 **核级"提不变量"实验失败：不逐位相同 + 更慢 ⇒ 核已经没有便宜可捡**
  - 定位（`objdump`，正确的那次）：`_matmul_q4_mub` 热循环 **29 条/块**
      `vpinsrw + vcvtph2ps` 2 条（读 fp16 scale）｜`movzbl + mov + not + test + jne` **5 条（inf/NaN 守卫）**｜
      `vmulss + vbroadcastss` 2 条（算 `d·sc`）｜地址/循环 8 条｜maddubs/maddwd/paddd/cvt/fma 5 条｜其余装载 7 条
  - 改动（`bench_proj_mlp.mojo` 里的 `matmul_q4_mub_hoist`）：每行先扫一遍 scale 算出 `dsc[b] = d_b·sc_b`，热循环只剩一次 `vbroadcastss`
  - **结果两条，都是负的**：
      ① **不逐位相同**：896 行全部不一致，最大绝对差 **12.61** ⇒ "纯重排"这个前提错。Mojo 带 **fast-math**，`x·(d·sc)` 与 `(x·d)·sc` 结合顺序不同 ⇒ **提前算 `d·sc` 会改数值**
      ② **更慢 0.64×**（3.25 vs 5.12 GB/s，同轮交替）⇒ 省掉 7 条，却多出一趟 `nb` 次标量预扫（每次还要过守卫），**净亏**
  - ⇒ 唯一还剩的核级微改是 `block_scale_fast`（去掉那 5 条守卫、保留 inline 转换），约 **17%**，核级预期 <1.1× ⇒ **不值得**
  - 🔴 **这条线索到此为止**：便宜的改动已经全部试完（f32→整数点积、`−8·Σx` 挪 int32、行带复用、提不变量）。剩下的只有"**int32 域跨 8 块累加**"，而它**必须配合 8 块交错布局**（8 个块的 scale 落进同一向量的 8 个 lane，才能一次 f32 乘加摊到 8 块）⇒ 要重写量化器 + 核，收益上界 1.5×，**e2e 仍被 16.7 ms 地板夹在 1.47×**
  - 🧭 结论：**在当前内存配置下，decode 的核级优化已经见底**。继续投入只能拿 ~1.2–1.4×，而且要先动量化器与布局
- **2026-11-03（同日第十一条）** —— ✅ **prefill 分段预算到手；RB 按 `n_tok` 分流 ⇒ 短 prompt prefill 1.8–2.0×，decode 不动**
  - 新尺子：`scripts/bench_prefill_budget.mojo`（真权重 `q4_0+q8_head`，`n ∈ {32,128} × shards ∈ {1,4,8}`，每档两趟取 min；同时打**遍数模型** `passes = shards × ceil(每片行数 / RB)` 与"隐含带宽"）
  - **分段预算（插桩，`n=128, shards=1`，墙钟 2820 ms）**：核 **2473 ms（88%）**｜`lm_head` 20.6｜`rope+KVcopy+attn` **95.7**｜`swiglu` 37.5｜`rmsnorm` 4.3｜残差 3.5
      对比 `n=32, shards=1`（墙钟 737 ms）：核 648（88%）｜`lm_head` 24.1｜`rope+attn` **6.3**｜`swiglu` 9.4
      ⇒ **prefill 与 decode 同构：投影核吃 88%，其余是零头**；唯一随 `n` 变重的是注意力桶（6.3 → 95.7 ms，因为要过 `n²` 的 KV）
  - 🔴 **隐含带宽只有 1.2–2.8 GB/s**（整机天花板 16.6）⇒ **prefill 在单核/少片下不是权重流量受限**，这条否掉了"遍数就是全部"的直觉
  - **遍数模型拟合**：`n=128 shards=1` 核 2473 ms；若 `RB=2` ⇒ 64 遍 = 12.2 GB ⇒ **4.9 GB/s**（与核级 mub 5.12 GB/s 吻合）；若 `RB=8` ⇒ 16 遍 ⇒ 1.24 GB/s（不成立）⇒ **prefill 实际就是 `RB=2`**
  - **核级 D 表**（`bench_proj_mlp.mojo`，单线程，`n_tok∈{16,32}`）：
      `RB=2 → 353.8 ms（8 遍，4.55 GB/s）｜RB=4 → 299.2（4 遍，2.69）｜RB=8 → 266.5（2 遍，1.51）`
      ⇒ **流量降 4×、时间只降 1.33×** ⇒ 单核 ALU 受限；但 `1.51 × 8 片 ≈ 12 GB/s` 才刚要贴天花板 ⇒ **RB 只在多线程下变现**
  - ✅ **模型级同轮 A/B（无插桩，改成 `comptime Q4_MUB_ROWS_RB` 后一次改动一次重编）**：
      `n=32 shards=4` **627.3 → 344.6 ms（1.82×）**｜`n=32 shards=8` **554.5 → 279.6 ms（1.98×）**
      `n=128 shards=1` 2771.6 → 2323.4（1.19×）｜`n=128 shards=8` 1198.4 → 987.6（1.21×）
      🔴 **decode 反向退化**：`shards=1` 56.4 → 60.7、`shards=8` 33.2 → 41.3 —— `n_tok=1` 没有可复用行，宽带白占 7 个 lane
  - ✅ **决策：按 `n_tok` 分流**（`Q4_MUB_ROWS_MIN_TOK = 4` ⇒ `≥4` 走 `RB=8`，否则 `RB=2`）。prefill 每片在 `n=32, shards=4/8` 时正好 8/4 行，都落在宽带一侧；decode 恒 1 行 ⇒ **走的是与改动前同一条 `RB=2` 实例化**，构造上不可能退化
      阈值版实测：`n=32 shards=4` 325.5｜`shards=8` 253.9–334.1｜`n=128 shards=8` 988–1057（vs 基线 1198）⇒ 收益保留
  - **门**：`test-model-avx2` **4/4**（含 128 步逐 token 贪心）；`test-q4-mub` **2/2**（`RB=8` 无条件版跑的，整网一致率门过）
  - ⚠️ **本机方差必须记下**：同一份 decode 二进制连跑三轮 `shards=1` **84.9 / 61.9 / 55.0 ms**（loadavg 8–10）⇒ **单轮 A/B 不可信，±30–50%**。本轮 prefill 的 1.8–2.0× 远大于噪声所以站得住；decode 那三个数的差别**不能**用来下结论，这也是为什么收尾选了"构造上不碰 decode"的方案而不是"看谁快留谁"
  - 🧭 **prefill 下一杠杆已经可见**：`n=128` 时 `rope+KVcopy+attn` 已占 **95.7–104 ms（约 10%）** 且随 `n` 平方增长；再往上是 `swiglu` 37.5。核本身 88% 但那已经贴 `RB` 能给的边了
- **2026-11-03（同日第十二条）** —— ✅ **注意力桶拆开：`n²` 全在 attn 核里；`RB` 到 8 就饱和（流量杠杆用尽）**
  - 打点从一桶 `K_ROPE_ATTN` 拆成两桶：**`K_ROPE`（`rope_k` + 两次 KV `copy_into`，线性 ∝ t）** 与
    `K_ROPE_ATTN`（`attention_k_shards` 的分数/softmax/AV，∝ n²）。合计口径不变，只是能分清了
  - 实测（插桩，shards=1，每 token）：

      | | n=32 | n=128 | 增长 |
      |---|---|---|---|
      | attn 核 | 6.28 ms | **113.74 ms** | ×18 |
      | rope+KVcopy | 0.73 | 3.09 | ×4.2 |
      | swiglu | 10.57 | 41.33 | ×3.9 |
      | rmsnorm ×2 | 1.13 | 4.71 | ×4.2 |
      | 残差 add ×2 | 0.92 | 3.79 | ×4.1 |
      | `lm_head` | 25.36 | 23.88 | ×0.94（只跑最后一个 token，正确） |

      ⇒ **除 attn 核外每一段都严格线性**（≈4× 对应 n 的 4×），只有 attn 核是 ×18（≈n² 加常数项）
      ⇒ `n=128` 时串行段合计 166.7 ms，**attn 核独占 113.7（68%）**；`rope+KVcopy` 只占 1.9%
      ⇒ **"KV 搬运太贵"这个猜测当场否掉**，长 prompt 的串行成本就是注意力本身
  - ⚠️ attn 核的达成算力只有 **≈12 GFLOP/s**（`14 头 × 128 长 × 64 维 × 2 × 2 × 24 层 × 128 token`），
    远低于标量核应有水平 ⇒ 它也是**指令受限**的朴素实现，不是带宽或算力受限
  - **`RB` 剂量-反应（D 表加第 16 臂，单线程，24 层 × 7 投影）**：

      | RB | 2 | 4 | 8 | 16 |
      |---|---|---|---|---|
      | 每 token（`n_tok=16`） | 22.45 ms | 20.37 | **18.06** | 18.27 |
      | 每 token（`n_tok=32`） | 22.46 | 20.43 | 18.37 | 17.82 |
      | 权重流量 | 8 遍 | 4 遍 | 2 遍 | 1 遍 |

      ⇒ **`RB=16` 把流量再砍一半，时间却持平（−1%…+3%，在噪声内）** ⇒ `RB=8` 已饱和，
      **"靠减权重流量提速"这条路在 mub 通路上彻底用尽**；标度上去的全是核内指令
      ⇒ 也解释了 `n=128/shards=8` 的隐含带宽只有 **3.1 GB/s**（远低于 16.6）却提不动的现象
  - **结论 / 下一步**：prefill 的 81%（投影核，per token 6.2 ms）与 decode 的 88% 都卡在**每 (行,块)
    指令数**上，`RB`/流量/预取全都不是杠杆；注意力桶（`n=128` 占 11.5%）是唯一"改得动且已定位"的
    次级项，但它自身也是指令受限的朴素实现 ⇒ Amdahl 上限 **1.13×**（`n` 更长时才更大）
  - **门**：`test-model-avx2` **4/4**（拆桶只动 `ALOFA_PROFILE` 编译期门内的打点，默认档二进制不变）
- **2026-11-03（同日第十三条）** —— ✅ **mub 核逐条清点：`RU=8` 与 13 条/行块都是下限；真正的下一杠杆是"批"**
  - **objdump 清点**（`RB=1` 档 = 单流 decode 的真实路径，4 块展开，`vmovdqu`/`vpsrlw`/
    `vinserti128`/`vpand` 逐个对过源码）：每 (行, 块) **约 13 条**——
    1 条取权重（16 B）、**3 条拆 nibble**（`vpand` + `vpsrlw` + `vinserti128`）、2 条取标量步长
    （`vmulss` + `vbroadcastss`）、2 条 Σx（`vbroadcastss` + `vaddps`）、5 条点积/累加
    （`vpmaddubsw` + `vpmaddwd` + `vcvtdq2ps` + `vfmadd231ps`，`ones` 的广播被塞进循环体）
  - 另有**2 条栈往返**（`vmovdqu %ymm0,0x…(%rsp)` 再 `vmovdqu 0x…(%rsp),%ymm3`）：4 块展开 × 8 行
    = 32 个权重向量放不进 16 个寄存器 ⇒ 溢出。**这是展开的代价，不是算法要求**
  - **`RU=4` 判死**（同进程配对，A/B = RU8 噪声底、C/D = RU4 噪声底，四臂交错 9 轮）：
    `RU8/RU4` 中位 **0.92**（896 行）/ **0.94**（4864 行）⇒ **RU4 慢 6…8%**，方向在两个规模上一致
    ⇒ 溢出不是"累加器太多"引起的；`MUB_ROW_ACC_TARGET=8` 确认最优，**这条也关闭**
  - **批处理是迄今最大的未记账红包**（D 表加 `n_tok=1` 与 `n_tok=8` 两档，同进程同表）：

      | 形状 | 每 token |
      |---|---|
      | `n_tok=1`（单流 decode，`RB=1`） | **44.39 ms** |
      | `n_tok=8`（并发 8 的 decode 一步，`RB=8`） | **18.83 ms** |

      ⇒ **2.36×**（同一步交付 8 个 token：53.1 tok/s vs 22.5 tok/s）。机制不是"少读权重"
      （`RB=8` 单核只有 1.34 GB/s，远低于带宽），而是**一次权重读取摊到 8 个 token 的 ALU 工作**
    - 引擎接线已核：`executor.forward` 的 `normed = rows_view(raw, t, h)`，`t` = 在批请求数 ⇒
      `project` 收到的 `n_tok` 就是并发数 ⇒ 今天开的 `n_tok ≥ 4 走宽带` 在服务层**自动生效**
    - ⚠️ `RB=16` 在 `n_tok=8` 上塌到 32.63 ms/token **不是核的问题**：`n_tok < RB` ⇒
      `limit = 0` ⇒ 全部走 `RB=1` 尾巴。**这不是回归，是分流的边界**，但说明"宽窄带"必须与 `n_tok`
      同域，别单独把 `RB` 调大
  - **结论**：核内已无量级空间（13 条里能砍的只有拆 nibble 那 3 条，而换成预展开要付 1.89× 流量，
    实测 −19…25%）；**下一杠杆是把并发做上去**，因为它同时改善每 token 的权重摊薄与 ALU 摊薄，
    且不需要再动核
  - **待量（未记账）**：服务层并发 8 的端到端 QPS —— 模型层的 2.36× 已量，HTTP 层还没量，
    别把 2.36 当成 QPS 收益
  - **门**：`test-model-avx2` 4/4（本轮只改探针与打点，生产路径未动）
- **2026-11-03（同日第十四条）** —— ✅ **服务层并发实测：模型层 2.36× 在端到端一分都没拿到；两个档位一个被入场挡死、一个直接停摆**
  - **口径**：`stress_serve.py`（SSE 流式、`max_tokens=32`、60 s、`ALOFA_WORKERS=1`、prompt≈5 token、
    runq 20–27 —— 老规矩**只作同轮相对比较**）；tok/s = ok × 32 / 实测墙钟（harness 已补这行输出）。
    服务 = `target/serve`（`-O2`，真权重 `tests/fixtures/qwen2.5-0.5b-hf`），两档各起一个实例
  - **`ALOFA_ENGINE_THREADS=0`（支持档，生成在收发循环上）**：

      | | 并发 1 | 并发 8 |
      |---|---|---|
      | tok/s | **3.18** | **3.20** |
      | 请求 ok / 错误 | 6 / 0 | 13 / 1 timeout |
      | TTFT p50 / p90 | 1.5 ms / 71 ms | **9.3 s / 103.7 s** |
      | 单请求 p50 延迟 | 10.6 s | 18.5 s |

      ⇒ **吞吐 1.01×：模型层的 2.36× 被服务层整段吃掉**。证据链分两层：
      ① **tick 级批处理是真的在干活** —— 并发 8 的单请求 p50 18.5 s / 32 token = 578 ms/token，
      而单流是 313 ms/token ⇒ 8 行转发只比 1 行贵 **1.85×**（与模型层 2.36× 同量级），
      已入批的请求确实共享同一次前向；
      ② **病在入场**：TTFT p90 103 s = 新连接要等 100 s 才吐第一帧 —— 生成占着循环，
      `accept`/prefill 进不来，批一直填不满 ⇒ 吞吐停在单流水平
  - **`ALOFA_ENGINE_THREADS=1`（实验档，生成挪引擎线程）**：头 2 个请求正常（TTFT 1.25 s、
    9.6 s/32 tok），之后**永久停摆** —— 复测 0 ok / 45 s 超时全灭，端口仍可 `connect` 但再无帧。
    进程现场：**主线程 `state=R` 持续自旋，引擎线程全部 `futex_wait`** ⇒ 是**活锁**（握手丢了：
    主线程忙等一个不会到的结果，引擎线程等一个不会派的新活），与启动警示"客户端中途断开会搞坏
    堆"是同一族、更重的一档。**该档位在当前状态下不可用于压测结论**
  - **结论 / 下一步**：端到端吞吐的下一个瓶颈**不在算力、在服务层握手**——要么修 `T=1` 的活锁
    （`srv/engine_thread.mojo` 的派活/回执握手），要么让 `T=0` 的入场不被生成挡住（生成切片/让出
    循环）。修好之前，"把并发做上去"到不了 QPS；修好之后模型层的 2.36× 才能兑现
  - **门**：本轮**未改生产代码**（只给 `stress_serve.py` 补了 tok/s 输出行）；`T=0` 并发 1 零错误
    （6/6），并发 8 有 1 个 `timeout`，`T=1` 有超时 —— 这两个失败都是本轮的**发现本身**，不是回归
- **2026-11-03（同日第十五条）** —— `ALOFA_ENGINE_THREADS=1` 的停摆查清并修掉**两个真 bug**；
  第三个（并发下的 KV 重号）已定量定位、未修
  - **怎么拿到现场**（方法本身可复用）：`ptrace_scope=1` ⇒ 不能 attach 非子进程。把服务放在
    `gdb` 下、命令走 FIFO（`tail -f /tmp/gdb_in | gdb --args ./target/serve`），复现后给被测
    进程发 `SIGINT` 让 gdb 接管再 `thread apply all bt`；配合 `/proc/<pid>/fdinfo/<epollfd>`
    （注册表与 token）与 `ss -tnp`（TCP 状态）。**这次的关键证据全都来自后两者。**
  - **bug 1 = EOF 自旋（活锁的实体）**：客户端一断，连接进 `CLOSE-WAIT`，`recv` 对它**永远返回
    0**，而 epoll 是 **level-triggered** ⇒ 事件永远报。`read_ready` 收到 EOF 只
    `mark_peer_closed()` 就返回（**不收连接**），而 `reap` 的四条例外（有请求排队 / 还有字节没写 /
    流在跑 / 活在途）把一条"对端已经没了"的连接全部豁免；更糟的是 `read_ready` 末尾无条件刷
    `last_seen`，**自旋亲手把空闲超时喂活**。现场：主线程 `state=R`、gdb 停在 `recv` 且
    `$rax=0`（返回值就是 0），6 条连接全是 `CLOSE-WAIT`，新客户端 `connect` 成功（内核 backlog）
    但一个字节也收不到。**修**：`Conn.peer_gone()` + `reap` 里 `peer_gone && drained` 立刻收
    （还有要写的仍留给 `flush` 的 EPIPE 去收，那是"优雅发完"与"对端没了"的合取）+ EOF 不再刷
    `last_seen`。**验证**：`T=1` 并发 1 从「2 ok + 120 s 超时」变成 **8/8 零错误**，TTFT p50
    1.25 s → **2.3 ms**，3.79 tok/s（`T=0` 同轮 3.18）。
  - **bug 2 = 执行器被一次异常永久毒死**：`decide` 里 `plan` 借走缓冲（`n_handles>0`），只有
    `settle` 会还；而 `forward` 在借完之后抛（页表不够就是这条），调用方走异常分支就**永远到不了
    `settle`** ⇒ 之后每一次 `plan` 都报 "the previous step was not finished"。实测**一次**失败
    之后 **2708 次连败**，服务看着还活着、每个请求都秒挂。**修**：`Executor.abandon()`（只还缓冲、
    不推进历史）+ `decide` 的 try/except 兜底 + `_batch_tick` 的"先 `settle`（全 `NO_TOKEN`）再
    抛"。**验证**：同样的并发下不再连败，服务全程存活。
  - **未修（下一刀）= 并发下的 KV 重号**：并发 8 仍然 **108/122 请求**在
    `capacity: fewer blocks than it has history` 断流（`tok/s` 5.55 是**假数**，不可引用）。
    错误路径打点给出 `hist=64 n=1 table=64 runs=4` —— 而合法上限是 **5 prompt + `MAX_GEN` 32 = 37**
    ⇒ 这个 64 只可能是**请求号复用留下的旧历史**（号 = 连接槽位，断开与重连撞在一起）；且
    `table == hist` 恰好卡在**块边界**（4×16）上没有余量，`settle` 里 `grow_to(fed + generated)`
    用的是"这次喂了多少 + 生成了多少"，不是执行器的历史长度。下一刀两件事：① `admit` 遇到重号
    必须**拒绝并报**，`stream_end` 必须**一定**把引擎侧那一份放掉；② `grow_to` 要预留下一步要写的
    那个位置（`hist + 1`），否则每跨一个块边界都要靠"表恰好有余量"侥幸。
  - **门**：`test-batch-forward` 6/6、`test-batch-stream` 6/6、`test-loop`、`test-engine` 过
    （engine 门本来就钉着"客户端中途离开之后还能答"——这次修的两处正对着它）。
    ⚠️ **当时把 `test-http` 记成"过"是错的**（只看了 `tail -3` 里的响应体）：它在 HEAD 上就是
    红的，原因见第十六条末尾（断言没跟上 chat 模板渲染）。
- **2026-11-03（同日第十六条）** —— 并发下 `capacity: fewer blocks than history` 的**真凶**：
  批路入场只判"prompt ≤128 / 新 token ≤32"，没判二者**加起来**是否装得进一条 KV 序列
  - **先记一条被证伪的假设**：第十五条曾判成"请求号复用留下旧历史"。打点给出的
    `hist=64 table=64 runs=4 out=31` 看着像，但补上 `prompt` 之后一句话就推翻了 ——
    `prompt=34`（chat 模板自带 ~29 token 的 system 段），**34 + 32 = 66 > 64**
    （`runtime/kv` 的 `MAX_SEQ_TOKENS`，引擎与 KV 房间共用同一个常数）。历史没被谁污染，
    它是**合法地**涨到 64 的，之后房间扩不出第 5 块（4×16 已满），`forward` 报的正是
    "表的覆盖 < hist + 本拍的行数"。教训：**打点要一次打全**（`hist/table/n` 三个数分不出
    "表少一格"与"历史长得不对"，加 `prompt/out/runs` 才能一句话定案）。
  - **为什么早没炸**：老路（单流）用的是模型自己的上下文窗口（`self.max_tokens`，≈256），
    比 KV 序列的 64 宽得多 ⇒ 同一形状的请求走老路不炸；只有进批路才撞 64。
  - **修（两层，都在"入场"而不是"炸了再救"）**：
    ① `serve.mojo` 批路门加第四条判据 —— `batch_steps = min(steps, MAX_SEQ_TOKENS - len(ids))`，
    `≤0` 就不走批路；装不下按上下文窗口的规矩**截短**（`finish_reason` 会说 `length`，
    与既有 `steps = min(max_tokens, 窗口 - prompt)` 同一条语义）；
    ② `EngineCore.submit` 钉住不变量：`n + max_new > MAX_SEQ_TOKENS` ⇒ **入场就指名拒绝**
    （否则越界会晚 30 个 token 才以"前向内部"的形状暴露）。
  - **验证（同轮、runq ≈13）**：并发 8、`max_tokens=32`，`[srv]` 里
    `capacity … fewer blocks` 由 **108/122 条 → 0 条**，服务端其余错误行也归零。
    16-token 请求的对照：并发 1 **4.89 tok/s**、并发 8 **6.61 tok/s**（1.35×，此前是 1.01×）。
  - **仍未定位**：并发 8 每轮还有 4–5 条 `stream_truncated`（200 + 若干帧、无 `[DONE]`）
    与 1 条 `timeout`，而服务端**一行日志都没有**（`[dbg] stalled` 也没打）⇒ 不是已知的
    guard/EOF/毒死三条。下一个入口写在 `_batch_next`：它把"`has_work()` 为假"直接当成
    **流结束**返回（`StreamToken("", True, False)`），而"这一拍没轮到它"与"它永远走不动"
    在那里是同一个分支 —— 与 `_batch_next` 的 `MAX_PROGRESS_TICKS` 判据正相反。
  - **门**：`test-batch-forward` 6/6、`test-batch-stream` 6/6、`test-engine`、`test-loop` 过。
    ⚠️ **`test-http` 是一条既有红门，本轮顺手查清并修好**：它在 HEAD（`7a2b935`，不含本轮
    改动）上同样红 —— `git stash` + 重建复现，所以**不是**本轮引入。成因是断言漂移：
    `cba53a8`（messages 交给渲染器）之后 handler 交给 service 的是**渲染过的**整段 prompt，
    而门里仍断言回显 `S:hi`（渲染前的原始用户文本）。修的是**断言**而不是门限：新断言同时
    钉渲染头（`S:<|im_start|>system`）与用户文本（`hi<|im_end|>`），其余性质（200、keep-alive、
    `finish_reason`、流以 `[DONE]` 收尾）一条没减 ⇒ 修后 `PASS http server: …`（exit=0）。
    ⚠️ 该门 fork 一个子进程占 18123：**前一次失败会留下子进程**，下一次会 `AddressInUse`
    并且客户端连到**旧的**那个子进程上（表面症状是"流头不是 200"）—— 重跑前先
    `pkill -f http_e2e`。
- **2026-11-03（同日第十七条）** —— 并发下的 `stream_truncated`（200 + 若干帧 + **没有** `[DONE]`）
  定位到"**一帧派出去了、应答没回来**"，并把掐断改成**补派**
  - **症状的形状**：流停在半路、连接**不关**（客户端一直等），约 1/3–1/2 的并发流中招；
    服务端**没有任何** `[srv]` 报错。`guard_stalls` 的 4 帧诊断随后把它收掉，客户端这才看到
    EOF —— 压测把它记成 `stream_truncated`。
  - **per-slot 对账定案**（这次加的 `sent`/`back` 两个计数，全局的 `jpush/jtake` 分不出是**哪个**
    槽位丢的）：卡住的槽位一律 `sent=2 back=1` —— 第 1 件是请求（`head` 回来了），第 2 件是
    **第一帧 STEP，派出去了就没回来**。而同一时刻：`inflight=1 streaming=1 mailbox=0 busy=0`
    —— 循环这边的账是**对的**，它确实只欠一件应答。
  - **引擎侧的计数器自相矛盾**：`rpush=162 > jtake=160`（只取走 160 件活，却回 162 截应答），
    而 `runs=end_job=162`、`rtake=162`。一件活只能回一截 ⇒ 这两个数**不可能同时为真** ⇒
    邮箱的"取活/回执"归账（或 `Result.slot` 的归属）不可靠。**根因仍未定位**，下一个入口正是
    这里（`Mailbox.push_result` / `take_job` 的计数器与 `Result.intact()`）。
  - **先止血（已做，可验证）**：`guard_stalls` 第 4 帧不再直接掐断，而是**补派**（清
    `inflight`，下一轮 `dispatch` 再要一帧），连续补派 `MAX_STALL_RETRIES=3` 次仍一样才收。
    实测 8 并发 × 6 轮：流完成数由 3–5/8 抬到 **7–8/8**，**24 次补派全部首轮自愈**（没有一次
    走到 2/3，也没有一次触发 `stalled` 掐断）；60 s 压测 25 ok / 2 `stream_truncated`、
    **5.91 tok/s**（并发 1 对照 4.89）。
  - ⚠️ **补派的边界（必须写明）**：handler 的游标已经往前走了（`b_n`、`b_emitted`），SSE 的增量
    没有回退接口 ⇒ **丢掉那一帧的文本补不回来**：流会继续、能拿到 `[DONE]`，但客户端拼出来的
    文本少一截（因此这一刻**不**满足"流式拼接 == 非流式"）。取舍是"少一帧 vs 整条流断在中间"。
    要连这一截也补回来，得把 `b_emitted` 的提交从"算出来"挪到"确认发出去"（ack）—— 未做。
  - **仍未定位的残余**：60 s / 27 条里还有 2 条截断，且出现过一次"连响应头都没收到"（客户端
    0 字节、42 s 后 EOF），形状与上面不同 —— 下一轮从邮箱归账入手。
  - **门**：`test-loop`、`test-engine`、`test-batch-stream`、`test-batch-forward`、`test-http`。
- **2026-11-03（第十八条）** —— 终止帧 `[DONE]` 本地补发 + 死槽位宽限回收；第十七条留下的"邮箱归账不可靠"又剥掉两层
  - **把"不可靠"往下剥了两层（都证伪）**：
    - 出参不是凶手：`read_counters`（十一个 `mut` 出参）与"锁内直接拼串"（`counters_text`）逐字一致 ⇒
      读数这条链是自洽的；
    - 锁与 `List` 不是凶手：两个最小复现 —— `scripts/mailbox_race_int.mojo`（pthread 锁 +
      `List[Int]`，双方各 20000 次）与 `scripts/mailbox_list_big.mojo`（**大结构体**、字段形状与
      `Job` 一致、内容校验 + 只取一次校验）—— 都**干净**。留在 scripts/ 里，下一个人不必再猜。
  - **但恒等式破坏仍在**：`runs(57) > jpush(56)`、退化阶段 `jtake(117927) > jpush(117926)` ——
    `end_job` 只跟着"取到活"走、`jobs_pushed` 只跟着"循环派活"走，用户层代码**不可能**造出这两个
    组合 ⇒ 状态在临界区之外被改。与启动警告（"a client leaving mid-generation is known to
    corrupt the heap"）同源。**根因仍未定位**，入口在 handler/service 的跨线程相位上。
  - **丢的那一帧定位到了**：被截断流的尾巴**都带** `finish_reason:"stop"` ⇒ 丢的是它后面的
    `data: [DONE]`（16 字节，一次**独立往返**）。补派救不了它：handler 交出 `[DONE]` 那一步已经
    `stream_end`，相位是 `STREAM_IDLE`，再问一次只会拿到空串 → 循环按"流结束"关连接，终止符永远缺。
  - **修（可验证）**：`ends_sse_stream`（找**带值的** `"finish_reason":"`，文本帧是 `null` 不能
    碰）把"收尾帧已发"记进 `ended`，`has_sse_done` 记"终止符已发"（`done_sent`，防补成两帧）；
    `RES_DONE` 到达而终止符没发过 ⇒ 循环本地造 `sse_done()` 补上。**只在内容已完整时补** ——
    收尾帧都没送到的流宁可见断流，不给它一个假的"正常结束"。
  - **实测（8 并发 × 12 轮 = 96 条，engine 线程路径）**：**94/96** 完整（改动前同载荷 41/48）；
    受支持路径 `ALOFA_ENGINE_THREADS=0` 24/24 全绿、零 `stalled`/零补派；`pixi run test` 全过、
    `test-http` PASS。
  - **槽位泄漏（同轮发现并修）**：死槽位等 `JOB_STREAM_END` 的回执，而回执走"孤儿分支"被丢、
    `release_pending` 永不清 ⇒ `adopt` 不再复用它 ⇒ `live` 只长不缩 ⇒ 连打 96 条连接后新连接
    **全部被拒**（0/8）。修法：死槽位等回执**超 8 s** 由 `guard_stalls` 回收。⚠️ 试过"见回执即清"
    （立刻复用）：engine 线程**崩**在 `chunk_finish_json`（堆写穿）⇒ 立刻复用不安全；宽限回收
    实测 12 轮无崩溃、`live` 有界（多为 8、峰值 14）。
  - **残余（写明）**：① 2/96 截断的尾巴**没有** finish ⇒ 是内容真缺一帧（第十七条"游标已前进"
    的已知边界，补不回）；② 连打 ~16 轮后出现过 `JOB_REQUEST → RES_DONE(n=0)` 的**活件风暴**
    （jpush 118k）伴随计数器再度破坏 ⇒ engine 线程路径维持 EXPERIMENTAL；受支持的
    `ENGINE_THREADS=0` 不受影响。
  - **门**：`pixi run test`（全绿，含 `check-ledger`）、`test-http`（PASS，跑前 `pkill -f http_e2e`）。
- **2026-11-03（第十九条）** —— "活件风暴"定位到**收尾件重派**；加一道闸，洪泛 500× 消失
  - **对账定案**：退化阶段循环侧 `sent_sum=13113` 而邮箱侧 `jpush=111173` —— 差的约 10 万件
    **不经过 `sent` 计数**，而派件路径里唯一不动 `sent` 的就是 dispatch 第 0 步的收尾件
    （`JOB_STREAM_END`）。两个佐证一次对齐：`JOB_STREAM_END=2` 与取活环（tj）里满屏的
    `kind=2` 吻合；`_stream_end` 交回的正是 `RES_DONE n=0` 与结果环（tr）满屏的形状吻合
    —— 风暴不是请求、也不是幻影件，是**收尾件被一轮一件地重派**。
  - **机制**：死槽位的收尾回执走"孤儿分支"被丢 —— 那里清了 `inflight`、却不清
    `release_pending`/`release_sent` ⇒ dispatch 的判据（只看 `inflight`）认为"该再派一件"
    ⇒ engine 每秒约 200 次对同一个**已死的请求号**调 `stream_end`（每次还带一次
    `_batch_tick`），直到 8 s 宽限回收才停。8 槽 × 每窗约 1600 件 ≈ 观测到的 `sent_sum`；
    jpush 的 11 万全来自这里。这个 200 Hz 的重派同时是堆损坏的**放大器**（引擎侧状态被
    无休止地搅动）。
  - **修**：第 0 步加一个闸 —— `release_sent == 1`（这份待办的回执已经在路上）就跳过。
    回执丢了不靠重派去"催"，由 8 s 宽限回收兜底（第十八条那条）。
  - **实测**：jpush 峰值 **111173 → 223**（约 500×），干净轮 `vio=0`（恒等式不再破）；
    8 并发 × 3 轮 23/24 完整；受支持路径 `ENGINE_THREADS=0` 24/24 全绿；`pixi run test`
    全过、`test-http` PASS。
  - **残余（写明，与闸无因果）**：engine 线程路径仍有**间歇性崩溃**，第 1 轮也会发生（第 1 轮
    不存在死槽位，闸的跳过条件从未成立 ⇒ 崩溃另有源头）。两副栈：① `stream_next →
    _batch_tick → EngineCore.decide → BatchExecutor.forward` 崩在 forward（批路径自身）；
    ② `_engine_main → parse_request → parse_head`（`Job.raw` 被写穿）。路径维持 EXPERIMENTAL，
    下一个入口 = 引擎侧批执行与 `Job.raw` 的写穿点。
  - **门**：`pixi run test`（全绿，含 `check-ledger`）、`test-http`（PASS，跑前
    `pkill -f http_e2e`）。
- **2026-11-03（第二十条）** —— 批从未拼起来：`progress` 一次只驱动一条流；外加"空帧当终点"的静默截短
  - **宽度打点**（`ALOFA_BATCH_STATS=1`，窗口 64 拍）才是判"批拼没拼起来"的数：改前
    `window_width_sum=64/64` = **每拍恒 1 行**，`running=1 waiting=0 queued=0`，`w_req`
    长度恒 0（队列只 append，长度不回落 ⇒ 八条请求都提交过）。八条流被排成一条队，
    所以 8 并发 16 token 的聚合吞吐只有单流的 1.07×（5.2 vs 5.1 tok/s）——tick 级探针
    的 2.36× 从来没到过端到端。
  - **根因一（`progress`）**：`ENGINE_THREADS=0` 那条路上 `progress` 开头是
    "有流在途就喂它、这一轮不再答新请求"——那是批调度之前"handler 流状态只有一份"的
    老约束。批路状态早已按槽位分开（`b_req` 一族），一拍前向推进所有在批请求。改成
    **逐条喂完所有在途流、再按原规矩一轮答一个新请求**：`running` 1→4、宽度 1→5.9
    （峰值 8）。
  - **根因二（`_batch_next`）**：`b_n` 记的是**调用次数**而引擎的 `n_out` 是 **token
    数**。两者一错开，某次调用就会拿到"缓冲里有 token、文本已在上一帧发完"的**空串**，
    而空串在 `step_stream` 那里是"流结束"的暗号 ⇒ 客户端拿到一条正文被截短、却带着
    `finish_reason` 与 `[DONE]` 的"正常"流（8 并发同 prompt 下 7/8 条短一截，温 0
    也不一致；丢的是尾巴，不是算错）。改成 `b_n = n`（与引擎同口径）且**空帧不许当
    终点**（再要一步）。
  - **顺带补的日志**：handler 的 `except err` 分支原来静默回一帧 error —— 客户端只看到
    "短了"。现在打印 `stream failed on slot=…: …`；正是它把下面那批 KV room 错误捞了出来。
  - **实测**（同轮 A/B，`ENGINE_THREADS=0`，16 token）：单流 184–223 ms/token（4.9–5.4
    tok/s）；8 并发 529–577 ms/token、128 token / 9.6 s ≈ **13.3 tok/s**（改前 856–927
    ms/token、墙钟 24.5–25.5 s、聚合 5.1 tok/s）⇒ **聚合 2.7×、墙钟 2.6×**。8 token 用例
    8 并发与单跑**逐字节一致 8/8**（温 0）。
  - **门**：`pixi run test`（全绿）、`test-http`（PASS）、`test-batch-stream`（服务层批一致
    性门，6/6 通过）。
  - **残余（新入口，引擎侧）**：8 并发 × 32 token（prompt 34 + 生成 30）撞 KV room：
    `capacity(5): radix tree is full`、`invalid_argument(1): duplicate request id (kv room)`、
    `the scheduler asked to decode a request that is not resident`、
    `the request has fewer blocks than it has history`。都是"请求号 = 连接槽位"复用与
    room 节点预算的交互；16 token 不触发。下一步从这里进。
- **2026-11-03（第二十一条）** —— 前缀树没有节点淘汰路径：8 并发长序列把 32 个节点填满后永久失败，并伪装成"请求号冲突"
  - **症状**（8 并发 × 32 token，`ALOFA_BATCH_STATS=1`）：四条不同的错同时出现 ——
    `capacity(5): radix tree is full`、`invalid_argument(1): duplicate request id (kv room)`、
    `the scheduler asked to decode a request that is not resident`、
    `the request has fewer blocks than it has history`。16 token 不触发，32 token（prompt 34
    + 生成 30）必触发。
  - **根因**：块有 `reclaim` 这条路，**节点没有**。`MAX_NODES = 32` 是死预算，而八条 64 token
    的序列每条要 ~5 个节点（`MAX_NODE_TOKENS=16`）⇒ 40 > 32，第一次批就填满，之后每次
    `publish` 都抛。更糟的是 `EngineCore.publish` 的顺序是 publish→drop：抛在中间就跳过
    `drop`，room 里那个请求槽位永久占着；`MAX_REQUESTS` 也是 8 ⇒ 复用同一号（request 号 =
    连接槽位）的下一条立刻收到 `duplicate request id (kv room)` —— 一个缓存预算问题被伪装成
    "请求号冲突"。另外两条是同一棵树的连锁：publish 抛 → executor 已 `drop` 而 scheduler 仍
    当它 RUNNING ⇒ `not resident` / `fewer blocks than history`。
  - **修法**（与块侧对称，不动常量）：`KvRoom._make_node_room(need)` —— `commit` 之前按
    `rq_ntok/MAX_NODE_TOKENS + 3` 预留节点，不够就淘汰 `coldest_node()`（`evict` 对还被引用
    的块只剪树、不还池，活请求有自己的页表，照常解码）；`EngineCore.publish` 的 `drop` 改成
    无条件执行，publish 失败只打一行日志（缓存掉了是优化掉了，room 里那份是账）。
  - **验证**：8 并发 24 token（prompt 34 + 24 = 58，进批路）**零告警**、`error` 帧 0，且温 0 下
    八条不同 prompt 与单跑**逐字节一致 8/8**（改前同一用例 3/8）；帧数从 29 回到 24/32。
    `pixi run test`（40 套件全绿）、`test-batch-stream`（6/6）、`test-http`（PASS）。
  - **残余（下一入口）**：`MAX_NODES=32` 结构上装不下 8 条满长序列（要 ~40），所以现在每次
    publish 都在淘汰 ⇒ 缓存命中率掉、`evict` 走子树有成本。干净解法是把预算从准入上限推出来
    （`MAX_BATCH × (MAX_SEQ_TOKENS/MAX_NODE_TOKENS + 1) + 余量` = 64），并用独立 Python 参考
    （`scripts/dump_kv_reference.py` 里钉着 `MAX_NODES=32`）重生成 42 个 KV/scheduler fixture。
    另：连续多轮压测出现 `running=0 waiting=1` 的空转窗口（一条请求卡在 WAITING 反复分块预填
    48 行/拍），受控单轮未复现 —— 单独开一条。
  - ⚠️ 本轮绝对时间不可用：`/proc/loadavg` 13.4（8 核）。
- **2026-11-03（第二十二条）** —— 节点预算从准入上限推出来（32→64），并修掉"报表容量"的 `id list overflow`
  - **为什么改常量**：第二十一条的淘汰让 8 并发不再报错，但 32 个节点结构上装不下八条满长序列
    （8 × (64/16 + 1) = 40），于是**每次 publish 都在淘汰冷节点** —— 不报错，但缓存命中率被
    这次换掉了。预算改成从准入上限推：`MAX_BATCH(8) × (MAX_SEQ_TOKENS(64)/MAX_NODE_TOKENS(16) + 1)
    = 40`，再留一半余量 ⇒ **64**（每个节点 ~216 B，64 个 ≈ 14 KB，可忽略）。
  - **fixture 重生成是安全的**：先验证幂等 —— 在 32 下用 `scripts/dump_kv_reference.py` 重生成
    `/tmp/fx32`，与 `tests/fixtures/kv` **逐字节相同**（参考实现与仓库同步），说明它能当 oracle
    重跑。改 64 后重生成 6 个 dumper 的全部产物，diff 只有两类变化：`kv` 的 `CFG ... nn=32→64`
    与所有 `d=` 摘要（`MAX_NODES` 进哈希），`u=`/`f=`/`c=` 计数一个没变。`pixi run test` 40 套件
    全绿 ⇒ Mojo 侧摘要与 Python 参考在 64 下仍然一致。
  - **连带修掉 `id list overflow`**：`IdList` 容量是 `MAX_BLOCKS_PER_SEQ`（16，单条序列的块数），
    而 `evict` 报告的是**整棵子树**还回池多少块 —— 树装得下八条长序列以后，一个子树跨 30+ 块
    是常态，于是淘汰一个冷节点就抛 `capacity(5): id list overflow`，把流打断（实测 8 并发
    32 token 有 2/8 条这样死）。容量改成按**池**的上界 `MAX_ID_LIST = MAX_BLOCKS`（112）：
    一次 op 能还回池的块最多就是整个池。
  - **验证**：8 并发（温 0、八条不同 prompt、与各自单跑比）**24 token 8/8 逐字节一致**、
    **32 token（prompt 34 + 30 顶满 64）8/8 一致**，`error` 帧 0，服务日志零告警。
    `pixi run test`（40 套件）、`test-batch-stream`（6/6）、`test-http`（PASS）。
  - **吞吐**（load ~7.6–9.0，同轮）：24 token 单流 150–193 ms/token；8 并发 506–614 ms/token、
    墙钟 12.9–15.5 s ⇒ 聚合 ~14.2 tok/s ≈ **2.5×** 单流。改 64 之前同一用例 591–765 ms/token、
    15.2–19.4 s。
  - **打点补全**：批窗口行加 `blocks=<used>+<wanted>/<cap> cached= preempts=`（`ALOFA_BATCH_STATS=1`）。
    一次多轮压测曾出现 `running=0 waiting=1` 且每拍 48 行的空转（一条请求卡 WAITING 反复分块
    预填），受控单轮与本次多轮都没复现，`blocks=68+0/256 cached=2 preempts=0` —— 留待复现再追。
- **2026-11-03（第二十三条）** —— 服务默认跑 fp32；接上量化档的同时修掉「批组合改数值」的 mub 分发缺陷
  - **起点**：量宽度曲线时发现服务每 token 读 **1975.8 MB**（fp32，arena 1.98 GB），而账本
    里 q4_0+q8_head 只有 345.9 MB —— decode 带宽受限，档位直接决定墙钟。但 `QwenForward`
    的 `quantize=False` 是默认，服务从没开过量化。
  - **接入**：`ServeConfig.weight_bits`（`ALOFA_WEIGHT_BITS`，0=fp32 默认 / 1=体 q4_0 /
    2=体 q4_0+头 q8_0），`load_weights()` 一处决定装载，`spawn_twin` 按同档重载。默认留 0
    （与参考实现逐字节的那一档）。
  - **缺陷（修了）**：q4 + 默认 mub 下，8 条**相同** prompt 并发出 **7 种输出**、2 条截短
    （fp32 同用例 1 种、8/8 一致；`ALOFA_Q4_MUB=0` 同样 8/8 ⇒ 定位到 mub）。根因在
    `q4_matmul_rows_k` 的分流：`down_proj`（cols=4864）在 `n_tok≥4` 时选 `RB_WIDE=8`，
    `8×4864 > MUB_ROWS_MAX_XQ(16384)` ⇒ 核内退回**精确通路**；`n_tok<4` 时 `RB_NARROW=2`
    走 **mub**。mub 与精确通路不是逐位一致（多一层激活量化）⇒ 同一请求的数值随「这一步
    批里有几行」漂移。**修法**：宽带装不下就降到窄带（仍 mub），不退精确；窄带也装不下
    （cols>8192，本项目没有）才退 —— 那时对所有 `n_tok` 恒退，同样稳定。
  - **新门（④）** `test_wide_batch_stays_on_the_mub_band`（进 `pixi run test`）：直接打
    分发器，cols=4864、n_tok=8 的结果与精确行带**不许逐位相同**（负向对照），与逐行
    `n_tok=1` 参照的相对 L2 ≤ 1%。**负向对照有效性验证过**：把分发器临时改回旧行为，
    这条 FAIL；修后 PASS。⚠️ 不要求宽批与逐行参照逐位相同 —— mub 有损，RB 不同的实例化
    有 1–2 ULP 漂移（实测 96 个元素 34 个不同，L2 千分量级）；要逐位稳定就用
    `ALOFA_Q4_MUB=0`。服务对 `weight_bits>0 且 mub 开` 的情况启动时打一行提示。
  - **数**（同轮，load 12.6–13.2，三档紧挨着跑；24 token）：
    宽 1 —— fp32 169–193 / q4(MUB=0) 83–109 / q4(mub) 77–99 ms/token；
    宽 8 —— fp32 571–598 / q4(MUB=0) 331–355 / q4(mub) 341–343 ms/token。
    ⇒ q4 档约 **1.7–1.9×**；**MUB=0 与 MUB=1 在噪声内打平** ⇒ 推荐 q4 档配
    `ALOFA_Q4_MUB=0`：逐位稳定不花钱。
  - **门**：`pixi run test`（40 套件，含新门）、`test-q4-mub`（2/2，整网一致率下限仍过 ——
    down_proj 在 prefill 也换到 mub 窄带后）、`test-q8-head`（4/4）、`test-batch-stream`
    （6/6）、`test-http`（PASS）、`test-q4`（1/1）。
  - **残余**：① `stalled(13) ticks=256` 间歇性复现（本轮宽 8 第 2 轮墙钟 94.55s，带打点
    复测 5 轮干净；`blocks=140+0/256 cached=2 preempts=0`，块账在累积）—— 单独开条。
    ② fp32 批路按**批行**切分（`w` 共享）⇒ 权重被读 shards 遍：4 片 × 1975.8 MB ÷
    10.7 GB/s ≈ 738 ms，与实测 691 ms 吻合；按**输出列**切才能回到 1× 流量（TensorView
    支持 strides，但 `linear_k` 假定连续）—— 下一入口。
- **2026-11-04（第二十四条）** —— fp32 批路改切输出列：权重从读 shards 遍回到 1×，宽 8 近 2×；顺带抓出一个存量缺陷（同 prompt 并发 + 热缓存）
  - **起点**：二十三条的余项 —— fp32 批路按批行切分（`linear_k_shards` rows>1 分支，`w` 共享），
    每片都要整读权重：4 片 × 1975.8 MB ÷ 10.7 GB/s ≈ 738 ms，与实测 691 ms 吻合。
  - **修法**：恒切**输出列**（与 decode 同构）。`dst` 的片是母矩阵 `[rows, out]` 的**列段**
    ⇒ 行距 = `out`：`_gemm`（两个后端）从 `dst.strides[0]` 读行距传给 `_gemm_tile`，
    **只改写地址、不动任何一次浮点运算**；`x` 整批共享（几十 KB，留在缓存里），`w` 每片
    只读自己的行段 ⇒ 权重合计 1×。
  - **新守卫 `f32_data_rows`**：`f32_data` 的连续性守卫是给平铺索引消费者设的；`_gemm`
    对 `dst` 是二维寻址（行距读 strides[0]）。宽松取数**只给 dst** —— 连 `linear` 的偏置
    占位符 `f32_data(dst)` 都得跟着换，这个坑是新门**当场抓住**的（否则列切片在服务里
    一跑就 `strides-mismatch`）。`x`/`w`/bias 仍走严格守卫，守卫本身没松。
  - **门**：`test_gemm_batch_layout.mojo` 加 ④（整块 vs 逐片列切**逐位相等**：带余数片、
    bias 两路、rows 2/5/8）+ ⑤（负向对照：行距用成片宽必须算出**不同的数** —— 证明判据
    对行距敏感；谁把 `_gemm` 退回"永远用 cols"，④ 当场红）。`prefill_shards` 生产路径
    退役（函数与测试保留，docstring 注明由 ④⑤ 接棒）。
  - **数**（跨轮：改前 load 6.4–8.5、改后 8.7–9.7，过载机 ±25%，只作方向参考）：
    fp32 每条 ms/token —— 宽 1: 184–208 → 175–184（不变，符合预期：单流本来切列）；
    宽 2: 343 → 186–201；宽 4: 563 → 226–260；宽 8: 691 → 353–372
    （聚合 11.5 → 21.5–22.6 tok/s）。**流量模型兑现**：权重 1× 之后，fp32 宽 8 与
    q4(MUB=0) 档（331–355）几乎持平 —— fp32 批路不再是 shards 倍流量。
  - **判据**：不同 prompt 24/32 token 并发 vs 各自单跑 **8/8** 逐字节 ✓；宽 2 热缓存同
    prompt 与单跑逐 token 一致 ✓。
  - 🔴 **新发现（存量，非本次引入）**：**同 prompt ×8、缓存已热**时，7/8 条在**首个 token**
    就分歧（`Assistant:` vs `Assistant`），批内彼此一致但与单跑不一致；把本次改动
    `git stash` 后用旧代码重编复跑：同样 2 种、6/8 分歧 ⇒ 与列切无关。全新服务（冷缓存）
    同 prompt ×8 是 8/8 —— 触发条件是「共享同一条缓存条目 + 批里 ≥7 行」。现有判据
    （8 条**不同** prompt）测不到它：不同 prompt 的并发命中各自缓存条目，与单跑同构。
    嫌疑：prefill rows>1 时对缓存前缀的 attention（或共享块的读写竞争）；宽 2 不触发
    ⇒ 与行数/分片阈值有关。单独开条，复现脚本：单跑一次 → 同 prompt ×8 并发。
- **2026-11-04（第二十五条）** —— 二十四的 🔴 降级：同 prompt 并发"分歧"是 **SSE 分块边界**，不是数值；判据口径修正为**拼接字节**
  - **排查**（打点版服务：准入 `matched`/`hist`、每步 `served`、每次 pick 的 token id）：
    - 8 条同 prompt 并发流的**选中序列完全一致**（`71703, 25, 22555, …`，引擎逐位正确）；
    - 流式**拼接文本 8/8 逐字节一致**；不一致的只是 `data:` 块序列（2/8 相同）：
      单跑 `[Assistant][:]` 两块，并发 `[Assistant:]` 一块 —— 内容相同，切法不同。
  - **机制**（代码里写着，`_new_text`）：每步**整段前缀解码**、发与已发字节的**差**——
    这是为了流式拼接 == 非流式（多字节字符跨 token 的防御）。于是块边界由**收获批量**
    决定：某流的首次轮询晚了一拍，一次解码 2 个 token，piece 就合成一块。边界是
    收获时序的产物，类似 TCP 分包 —— **不是契约的一部分**。
  - **顺带验证过的两处引擎行为**（都正确）：`step_owes` 对预填充中途的槽也返回 true、
    `_batch_tick` 会为它挑 token，但 `advance` 按「prompt 未完不欠答案」丢弃 —— 浪费
    一次 argmax，语义无害；分块预填充（实测 n=31 切片）与 `matched > n` 的夹紧
    （`cached = n-1`）路径内容正确，只是把已缓存的一行重算一遍。
  - **判据口径修正**：并发 vs 单跑比**拼接字节**（内容契约），不再比分块序列。
    不同 prompt 旧口径本就通过；同 prompt ×8（温缓存）拼接口径 **8/8**。
  - **遗留**：会话一出现过的 2 条 `stalled(13) ticks=256`（截断流）今日未复现，
    仍是已知开放项（见「未复现」清单）。
  - ⚠️ 过程教训（重犯）：后台编译**先删旧产物再等大小稳定** —— 今晚 `serve_dbg`
    拿了昨晚的陈旧产物跑出"env 校验变了 + 请求挂死"两个假象，白白烧掉一轮排查。
- **2026-11-04（第二十六条）** —— 🔴→✅ 间歇 `stalled`/`running=0 waiting=N` 空转：**块账两处漏账 ⇒ 水位活锁**（复现、定位、修复）
  - **复现**：宽 8 × 多轮不同 prompt。8~14 轮后引擎整体楔死：`running=0 waiting=2~5`、
    `preempts` 每 64 拍 +8~112、`window_width_sum` 冻结（零进度），之后所有流空返回
    （25 轮 200 条里 128 条空）。
  - **诊断**（按因记账 + 双账对照）：`blocks=229` vs **池子实占 23**；调度账的"缓存"
    ~220 vs 房间实际 8 —— 虚账 ≈2 块/条 × 100+ 条。
  - **漏账 ①**（`kv_room`）：`_make_node_room` 腾节点直接 `space.evict`，绕过 `reclaim`
    的 `freed_pending` 上报 —— 池子真把块还了，调度账还把它们记在缓存名下。修：淘汰
    前后测 `cached_blocks()` 差值进 `freed_pending`（与 `reclaim` 同一测量口径）。
  - **漏账 ②**（`scheduler`，主根）：命中准入的**垫付块**没人冲 —— 调度账按满长给
    预填充记块（`blocks_for(end)`），房间对命中的前缀**保留**已有块（`last_fresh` 才是
    真新拿的；token 级共享最多费一个部分块）。差额每条命中请求虚涨 ~2 块。修：
    `Scheduler.credit_shared(req, blocks)` —— 引擎在 `room.admit` 后按
    `blocks_for(end) - room.last_fresh` 冲回，每次准入至多一次；`blk_base` 进入
    发布/抢占/`blocks_held` 的记账（base=0 时与旧式逐位相同）。
  - **门**：`test_publish_evictions_report_their_freed_blocks`（kv_room）+
    `test_a_cache_hit_credit_keeps_the_books_honest`（scheduler）。scheduler 的
    fixture 逐字节门全绿 —— 信用只对 `matched>0` 生效，旧 fixture 走 base=0 支路，
    行为逐位不变。
  - **效果**：25 轮 × 8 = 200 条 **0 截断**，`preempts=0`（此前 3468），两本账随跑随对；
    墙钟 792s → 258s。
  - **关联**：会话一的孤立 `stalled(13)`（2 条截断流）大概率同源（块账虚涨把水位顶死），
    未证同一；若再复发，先看 `preempts` 是否又在涨、`blocks` 是否钉在水位线上。
- **2026-11-04（第二十七条）** —— ✅ q4 主战档批路余量（一）：批头走 q8 行带核
  - **起点**：宽 8 聚合 23.1 tok/s vs 单流 17.7 —— 批几乎没赚到。第一版宽度曲线
    （非流式）显示"步时随批宽线性涨"是**假象**：非流式 `/v1/chat/completions` 走
    老串行路、根本不进批调度；**量批路必须用流式**。
  - **profile 打点**（executor.forward 加 K_STEP/K_ROPE_ATTN/K_HEAD，编译期门，
    `ALOFA_PROFILE=1` 时随批统计行 dump）：宽 8 一步 301 ms 里 q4 体核 131.6 +
    **fp32 头 72.0**（批路经 `project(Q4_NO_SLOT)` 落 fp32 通路 —— q8 头块流明明
    加载了，批路没用）+ 标量注意力 57.6 + qkv bias 路 ~40（未打点）。
  - **修**：`matmul_q8_f32_rows_band[RB=8]`（一个 q8 块只 load/解量化一次、`RB` 个
    token 行共用；每 token 行的累加结构与单行 vec 核一字不差 ⇒ **逐位一致**）+
    `q8_matmul_rows_k_shards`（行带 × 整批，切法同 q4 行带）+
    `QwenForward.project_head_batch`（q8 优先、fp32 回退）+ executor 接线。
  - **门**：`test_q8_rows_band.mojo`（逐位等于单行核 + 哨兵/守卫区 + 两条负向对照）。
    TDD 过程里红出两个真 bug：带内写出漏 `rb * dst_stride` 和 `+ row` —— 96 行全写
    进同一格。q8/q4/引擎/批路/http/全量套件全绿。
  - **数**（load 5.8，与基线同水位）：宽 8 每步 346 → **251 ms**，聚合 23.1 →
    **31.8 tok/s（1.38×）**；单流不变（56 ms）；批摊销 6.2× → 4.5×；15 轮 × 8 =
    120 条 0 截断。
  - **余量清单（下一步入口）**：标量分页注意力 57.6 ms/步（向量化是核工程）；
    qkv bias 路 f64 累加 RB=2（~40 ms）；forward 之外 ~50 ms/步（pick/settle/
    scheduler）。llama.cpp q4_0 tg32 单流 40.7 tok/s（同机带载）仍是单流天花板参照。
- **2026-11-04（第二十八条）** —— ✅ q4 主战档批路余量（二）：批路注意力走向量分页核
  - **起点**：二十七后宽 8 一步 251 ms，profile 显示标量分页注意力 **57.6 ms/步**——
    纯 f64 标量三遍循环，且值归约里 `row_offset(j)` 每对 `(d, j)` 都重走 run 列表；
    单路却早已是 `avx2.attention`（打分 4 通道 f64 重排 1e-5 + 值归约按通道逐位）。
    两条路对同一请求给 ulp 级不同的数，"并发 vs 单跑逐字节一致"全靠 argmax 不翻转。
  - **修**：`avx2.paged_attention_vec` —— 打分点积与值归约**照抄** `avx2.attention`
    的通道结构与累加顺序，只把行地址从 `j * kv_cols` 换成页表 `(block, slot)`；
    行地址**按 run 推进**（基地址一次，run 内 `+ kv_cols`），j 的顺序不变 ⇒ 与
    单路核**逐位一致**，"批 == 单"钉死在核级。标量 oracle 原样保留当独立参照。
  - **门**：`test_paged_attention_vec.mojo` —— 11 个夹具用例对 `avx2.attention`
    **逐位**（含块内偏移/共享前缀/q_len>1 因果截断）+ 1e-5 对 python 参照 +
    尾槽毒化逐位不变 + 越界表具名错误 + 篡改必红。executor 按 `backend` 分发
    （AVX2 走向量版，标量后端留 oracle）。
  - **数**（load 6.4 vs 基线 5.8，同轮相对）：宽 8 每步 251 → **164 ms**，聚合
    31.8 → **48.8 tok/s**（较二十七前累计 **2.11×**）；单流 47 ms 不变级；10 轮 ×
    8 = 80 条 0 截断。profile 归因（64 样本）：attn 57.6 → **3.98 ms/步（14.5×）**、
    lm_head 21.4、q4 体核 70.4、整步打点内 115.8。
  - **余量清单（下一步入口）**：qkv bias 路 ~20 ms/步（f64 RB=2）；forward 之外
    ~50 ms/步（pick/settle/scheduler）；q4 体核 70.4 ms 是否到带宽墙待核基准复核。
    llama.cpp q4 tg32 单流 40.7 tok/s —— 批路聚合已越过，单流 21.1 仍是天花板参照。
- **2026-11-04（第二十九条）** —— ✅ q4 主战档批路余量（三）：qkv bias 路换半块累加
  - **起点**：二十八的 profile 没给带偏置投影打点（q/k/v 走 `has_bias` 分支，K_GEMM
    桶没记它），整段藏在"未入账 ~19 ms"里。补打点后同桶读数：K_GEMM 70.6 →
    98.9 ms，**bias 路 ≈ 28.3 ms/步**（72 次调用：q 896×896、k/v 128×896）。
  - **修**：镜像无偏置路当年的切换 —— per-token `matmul_q4_f32_bias` 与行带
    `matmul_q4_f32_bias_rows_band` 都从 f64 通道（`_matmul_q4_wide_rows`）换到
    **f32 半块**（`_matmul_q4_halves_rows` 加 `bias/has_bias` 形参），`RB` 随之
    2 → 8（原 RB=2 是 f64 累加器 256 B/行的寄存器预算逼出来的；halves 每行只有
    2 个 f32×16）。批里 qkv 的解量化遍数 4 → 1。MUB 行带核"装不下"的回退同步换
    halves —— 回退必须与精确通路逐位一致，否则数值随装不装得下漂移（二十三的
    教训）。
  - **判定**：同窗 A/B（新旧二进制交替各两轮，load 9.9–10.7）：旧 162/185、
    新 145/164 ms/步 —— 两轮全向好，**步时 −11%**，聚合中位 46.3 → 51.9 tok/s。
    单流 47 → **39 ms/步（25.8 tok/s）**。干净环境 dump：forward 打点内
    115.6 → 95.8 ms，K_GEMM（含两路）98.9 → 70.2 ms。
  - **门**：`test_q4_rows`（行带 vs per-token 逐位 + 对标量 oracle 1e-5，无偏置与
    带偏置两节都过）、`test_q4_matmul_vec`、`test_q4_parity`、`test_q4_greedy`
    —— **贪心一致率钉值 0.8691 未动**（qkv 累加换 f32 没翻转任何贪心 token）；
    mub 三门、批路/引擎/http 全绿（41+ 套件 0 失败）。
  - **余量清单**：forward 外 ~50 ms/步（pick/settle/scheduler/拷贝）；q4 体核
    （无偏置）~60-70 ms 是否到带宽墙；lm_head 20-26 ms（q8 带）。llama.cpp
    q4 tg32 单流 40.7 tok/s —— 批路聚合已稳定越过，单流 25.8 还差 37%。
- **2026-11-04（第三十条）** —— ✅ "forward 之外"证伪为客户端伪影 + `d` 折权重侧
  - **"forward 之外 ~50 ms/步"证伪**：给批步三段补打点（plan/pick/settle 新桶
    K_PLAN/K_PICK/K_SETTLE + 服务端步墙钟 K_WALL）。实测：plan 0.01、pick 0.85
    （8 条 argmax/采样，152k 词表）、settle 0.01、detok 0.01（整段重解码便宜，
    24 token 不构成 O(n²) 痛点）、`_batch_next` 交付重叠计量后净额 ~1 ms、
    **服务端相邻两拍间隔 111.9 ms ≈ forward 本身 117.45 ms** —— 服务端背靠背，
    无空转。早先"墙钟 − forward ≈ 50 ms"的缺口全在**客户端**（bench 是 Python
    8 线程 + urllib 收流，GIL 串行化）。⚠️ 方法论：跨进程墙钟差分必须问"墙钟
    是谁家的墙钟"，服务端优化不能替客户端背账。
  - **带宽墙复核**：体核（q4 无 bias）~86.5 ms ÷ ~188 MB/步 ≈ 2.2 GB/s、
    lm_head 23.5 ms ÷ 144 MB ≈ 6.1 GB/s（FMA 吞吐上限 ~8.5 GB/s 的 72%）——
    两条都远没到访存墙，仍是**指令受限**。
  - **修（d 折权重侧）**：q4 halves 核里 `d` 原先折进 x（每行 2 次乘），行带
    RB=8 下这是每块 16 次乘；折到权重侧（`lod=lo·d` 每块 2 次乘）省 14 次/块。
    per-token `_matmul_q4_halves` 与行带 `_matmul_q4_halves_rows` **同步换**，
    单路/批路仍逐位一致；与标量 oracle 的偏差类别不变（舍入序差 ~1e-7，判据
    1e-5）。MUB 回退走的同一函数，自动跟随。
  - **判定**：同窗 A/B（参照=HEAD、d折版交替各两轮）：ref 176/183 vs d折
    171/164 ms/步 —— 两轮全向好，**步时 −6.7%**，聚合中位 44.6 → 47.8 tok/s
    （load 9.8→15.7 漂移下仍同向）。
  - **门**：q4 全套（rows/matmul_vec/parity/greedy——0.8691 未动/mub 三门）+
    批路/引擎/http/paged 41 套件 0 失败。**累计**：宽 8 聚合从 23.1（二十七前）
    到 ~48-50 tok/s（2.1×）。
  - **余量清单**：体核指令数（下一档是 mub，被"批==单逐字节一致"门挡着 —— 解锁
    需要先证明 mub 窄带==宽带逐位，或给 mub 立单独的宽容门）；lm_head FMA 吞吐
    （6.1/8.5 GB/s，余量 ~25%）；单流 25.8 vs llama.cpp 40.7 tok/s。
- **2026-11-04（第三十一条）** —— ✅ mub 解锁：默认档恢复 mub=1（它本来就是默认）
  - **起因**：复核余量清单时发现 `q4_mub_enabled()` 的默认**一直是 1** —— 今天的
    mub=0 全是照二十三-era"推荐 0"的笔记显式设的。那条推荐的两条理由都被现势
    推翻：①"随批组合漂移"的祸根是宽带**退精确通路**的回退（二十三已改成降窄
    带）；②"与 MUB=0 噪声内打平"是行带核改进之前的旧账。
  - **判定（同一二进制 `serve_dfold`，只切 `ALOFA_Q4_MUB`）**：宽 8 聚合 54.3–56.2
    vs 46.7–48.8 tok/s（**+15%**，步时 142–147 vs 164–171 ms）；宽 4 39.6 vs 32.4；
    **单流 25.6 vs 23.4**（mub 连 n_tok=1 都赢 —— 激活量化税小于解量化节省）。
  - **"批==单"硬门在 mub=1 下成立**：温 0，8 条**相同** prompt 与 8 条**不同**
    prompt 并发 vs 各自单跑，拼接正文**两次 8/8 逐字节一致**。结构原因：mub 行带
    核每行累加链与 RB/RU 分组无关（每行仍按块序、行末一次归约），宽带==窄带
    逐位。mub 与 **fp32 参考**不逐位 —— 那是量化档的本性，与并发一致性是两件事
    （旧注释把这两件事混成了一件，已改）。
  - **文案修订**：`serve.mojo` 启动 note 与 `config.mojo` 权重档注释不再教人配
    `ALOFA_Q4_MUB=0`。**门**：mub 三门 + parity + 批路/引擎/http 41 套件 0 失败。
  - **主战档现行数字（同轮，load 7.5–9.9）**：宽 8 聚合 **54–56 tok/s**（步时
    ~142-147 ms）；单流 25.6 tok/s。**累计宽 8：23.1 → ~55（2.4×，两轮内）**。
  - **余量清单**：体核 mub 化后的下一步指令杠杆（mub 行带核 ~17 周期/(行,块) 的
    旧测量是 PRE=1 前的，重测）；lm_head FMA 吞吐（余量 ~25%）；单流 25.6 vs
    llama.cpp 40.7。
- **2026-11-04（第三十二条）** —— ✅ 解码体归因：**指令受限**（剂量-响应 3/3 裁决）+
  两个旋钮按探针原判收手
  - **单流归因更新**（mub=1，宽 1，~36 ms/步）：K_GEMM ~29 ms（体+qkv ≈ 200 MB，
    **~7 GB/s**）、lm_head ~9 ms（144 MB，**~16 GB/s**）、attn ~1。同机同刻头比体
    快一倍 ⇒ 体没到访存墙。
  - **剂量-响应裁决**（`ALOFA_JUNK_ARM` 接到端到端，同二进制两臂重启即切）：
    剂量 3（+~30% 指令、+~178% 流量、数值逐位相同）vs 剂量 0 交替三轮：
    35/38/36 vs 46/44/49 ms —— **3/3 同向，+28%**。时间跟指令走、不跟流量走 ⇒
    **解码体是核内指令受限**。上一轮 46/45 的"平"是 load 干扰（1-min avg 未落）。
  - **两个旋钮不采纳（尊重探针原判）**：① `SX_I32=1`（`vpaddd` 替 `vaddps`，
    依赖链 22→18）—— 09-30 探针已判"RU=8 后链长非瓶颈"，端到端不重试；
    ② mub 尾带 `RU=4`（RB=1 下 8 组累加器溢栈的猜测）—— 交替 A/B 宽 1
    37/37 vs 38/37 平，宽 8 纯噪声（尾带不在批路），不采纳。
  - **净变化**：`serve.mojo` 接 `ALOFA_JUNK_ARM`（默认 0 = 与原二进制逐位相同）；
    尾带注释补 RU=4 的否证记录。门：mub 两门 + 批路/引擎/http 41 套件 0 失败。
  - **解码提质的下一步**（都在指令侧）：mub 窄带每 (行,块) ~13 条指令里解量化
    4 条 + 折算 4 条 —— PRE=1（−4 指令/+89% 流量）与"块间共享解量化"（RB 行
    共用，需 n_tok≥2，单流不适用）是仅剩的两个方向，前者需重测（09-30 判的是
    prefill 形状）；lm_head 已 16 GB/s 不动。
- **2026-11-04（第三十三条）** —— ✅ PRE=1（预展开权重）端到端否掉了；一条设计教训进账
  - **假设**：账本三十二把解码体钉在指令受限（剂量-响应 3/3），而窄带每 (行,块) 的
    解量化是 4 条（`vpand`+`vpsrlw`+`vpand`+`vinserti128`）—— PRE=1 能把它们整段删掉，
    代价是权重 18 → 34 B/块（×1.89）。prefill 形状（09-30）量过，解码形状没量过。
  - **实现**：加载时给每个 q4 槽位建一份 34 B/块副本（`unpack_q4_blocks`，纯重排），
    band 包装加 `PRE` 档、模型按 `ALOFA_Q4_PRE=1` 选副本并把档位**跟指针一起**传给核。
  - **裁决（同二进制 env 切档，交替三轮，load ~10 外部干扰下只取同轮比）**：
    宽 1 档 0/1 = 39·35·37 vs 45·46·40 ms（**3/3 更慢**）；宽 8 = 194·171·180 vs
    196·191·183 ms（**3/3 更慢 ~5%**）。⇒ 4 条指令换不来 ×1.89 流量。
  - **数值那一半是全绿的**（两档都跑）：`test_q4_mub_greedy` 贪心钉值 0.8691、
    `test_batch_stream` 批==单逐字节、mub 两门 —— 41 套件 0 失败 × 两档 ⇒
    "纯重排 ⇒ 逐位相同"这条承诺**实测成立**，否的只是性能。
  - ⚠️ **设计教训（已写进 `avx2.mojo` 旋钮旁）**：**布局属于缓冲区，不属于二进制**。
    第一版把 PRE 做成全局编译期常量，`test_wide_batch_stays_on_the_mub_band` 当场抓红
    （测试自造的 18 B 块被按 34 B 读，"scale is infinity"）⇒ 档位必须与 `blocks` 指针
    同生同死地传下去。这条对以后任何"换排布换速度"的尝试都成立。
  - **净变化**：生产代码零改动（否决后整体回退）；否决记录写在 `avx2.mojo` 的 PRE
    注释块里，避免下次再从同一个假设出发。
- **2026-11-04（第三十四条）** —— ✅ 分片档位补量（关闭 `default_shards` 的自认空白）
  - **动机**：`default_shards` 注释自认「批 > 1 的行分片没量过」，而批路已是主战档；
    且 1.66–1.84× 的旧证据只在 fp32 路上，mub 核快 5×，同一调度开销的相对占比不同。
  - **做法**：`ALOFA_SHARDS` 现成 env，同二进制切 1/4/8 三档，宽 1+宽 8 各量，
    轮转顺序共 5 轮（load 7–10 外部干扰，只取轮内配对）。
  - **裁决**：宽 8 单片 391–459 ms/步 vs 4 片 217–283 vs **8 片 207–266（4/5 胜 4 片）**；
    宽 1 单片 66–82 vs 8 片中位 54（4/5 胜）。⇒ **默认（核数 ≤ 8）就是最优档**，
    生产零改动；批路行分片的并行收益真实存在（单片慢 ~1.7×），mub 快核没有把
    调度开销吃成负收益。
  - **净变化**：`default_shards` 注释把「没量过」改成量到的结论；性能总账不动。
- **2026-11-04（第三十五条）** —— 🔴 engine 线程崩溃：复现从「间歇」升级到「秒级必现」，触发面收窄到**并发流式批路**
  - **复现矩阵**（`stress_serve.py`，fp32，`ALOFA_ENGINE_THREADS=1`，`max_tokens=8`，30–90 s）：
    - c8 流式：**崩**（shards=8 与 shards=1 都崩；8 条流先全截断，秒级死）
    - c4 流式：**崩**（同形状，4 条全截断）
    - c2 流式：**活**（18 ok；但 `[dbg] result dropped as orphan` 泛滥，`live_n` 涨到 5）
    - c8 非流式 / c1 流式 / c1 非流式：**全活、零错误**
    ⇒ **写者住在「≥4 条并发流式」（批宽度 ≥4）的引擎侧路径**，与客户端断开**无关**
    （11-03 的「第 1 轮也会崩」从此有了稳定配方），与 TaskGroup/分片无关（s1 照崩）。
  - **两副崩溃栈（受害者）**：① `forward ← decide`（TaskGroup/asyncrt 帧内）；
    ② `serialize ← String::_add ← _alloc_bytes ← asyncrt/KGEN` —— **都是 engine 线程上的
    分配器/运行时 machinery**。⇒ 崩点是**受害者不是写者**：堆被别处写坏，下一次分配死。
    与「崩溃点随机落点」「插桩即愈」「-O0 不崩」全部自洽。
  - **gdb 下不崩**（Heisenbug 再证）但病灶照跑：`peeked=10 live_n=13 live=0`（孤儿丢弃 +
    槽位泄漏持续）。core dump 被 apport 接管（`core_pattern` 管道），无 root 改不了；
    gdb 附着被 `ptrace_scope` 挡 ⇒ 运行时自身的符号化栈是唯一免插桩观测。
  - **已排除**：TaskGroup（s1 照崩）、栈溢出（flare `ThreadHandle.spawn` 默认 8 MB 栈）、
    邮箱锁/List（第十八、十九条最小复现干净）、reactor 双重推进（现行 `run_threaded`
    零 handler 触点，09-30 读码结论在 11-03 重写后仍成立）、`Job.raw` 拷贝上界（8192
    截断有检查，magic 四点校验在邮箱两侧）。
  - **下一步**（按性价比排序）：① 用 c4 流式（秒崩）做**二分移除**——engine 侧逐段
    short-circuit（先砍 `_batch_tick` 的 forward、再砍 settle、再砍 collect 的 enqueue）
    定位哪个环节在场才崩；② 静态审 executor 宽批 scratch 的按宽度寻址（崩溃面与宽度
    相关 ⇒ 写者的写入量随宽度涨）；③ 找 root 把 `core_pattern` 改成文件拿故障地址。
- **2026-11-04（第三十六条）** —— 🔴→✅ **engine 线程崩溃破案：栈溢出，不是堆写坏**（八轮排查的总根因）
  - **二分刀**（全部 env 门控默认关，随本轮提交）：`ALOFA_ET_FAKE`（假 token，砍
    `service.stream_next`）活（390 ok）⇒ mailbox/collect/SSE 管道洗清；
    `ALOFA_ET_CUT_TICK`（砍整个 `_batch_tick`）活（508 ok）⇒ decide/forward/settle
    是嫌疑；`ALOFA_ET_CUT_FWD`（`decide` 里 prepare/borrow/pick/settle 照跑、**只有
    forward 缺席**）**活** ⇒ 写者（当时误称）在 `ex.forward`。
  - **破案**：同一份 forward 在支持路径（reactor）24/7 无恙，唯一差异是线程 ⇒
    栈。`flare` 的 `ThreadHandle.spawn` 走 `pthread_create`（NULL attr）⇒ 默认栈 =
    soft `RLIMIT_STACK` = **8 MiB**；批前向调用链（执行器/模型按值传递大结构体 +
    宽带核深调用链）在 8 MiB 上**栈溢出**。**零代码判定**：`ulimit -s 65536` 起
    服务，c4 流式活（30 ok）；默认 8 MiB 同二进制同载荷秒崩（4 条全截断）。
    c2 活 / c4 崩 = `MIN_TOK=4` 处宽带核（RB=8）调用链更深。**全部历史观测一次
    自洽**：崩点随机落点（谁先碰 guard page 谁死）、三副栈都是分配器受害者
    （溢出后第一次 malloc 内部帧踩 guard）、插桩即愈（帧布局变）、-O0 不崩、
    gdb 掩盖（栈布局变）、gdb 下「病灶」照跑（孤儿丢弃/槽位泄漏/恒等式破坏
    都是溢出踩坏的周边状态，不是独立 bug）。
  - **修**：`core/ffi/mem.mojo` 加 `raise_stack_limit`（getrlimit/setrlimit，抬
    soft 不需要特权，必须在线程 spawn 前调，只影响之后创建的线程）；`serve.main`
    开头按 `ALOFA_STACK_MB`（默认 64，0 = 关）调它并打一行日志。
  - **验证**（fp32，`ALOFA_ENGINE_THREADS=1`）：c4 流式 40 s 活（22 ok）、
    c8 流式 60 s 活（62 ok）、**断开风暴**（8 条流生成中途掐 ×3 轮）后探活有
    `[DONE]`、零 `damaged` —— 修复前全部秒级必死。门：test-memory/test-core-ffi/
    test-engine/test-http/test-batch-stream/test-batch-forward 41 套件 0 失败
    （分层门先抓 `mem.mojo` 注释提「engine」，改措辞后绿）。
  - **reinterpretation（诚实写明）**：`ALOFA_ET_CUT_DECIDE`（无 decide 的 settle
    自旋）当时判「崩」—— 按栈溢出重读，它的每帧 `MAX_PROGRESS_TICKS` 自旋 +
    `ERR_STALLED` raise 链是另一种深帧布局，与根因不矛盾；该刀的协议疑点
    （settle 必须跟在 decide 后）仍然成立。EXPERIMENTAL 警示语里的「corrupt the
    heap」从此改述：不是谁写坏了堆，是**栈太小**。
- **2026-11-05（第三十七条）** —— 🔴 **推翻「栈溢出」结论，崩溃收窄到运行时格式化/追踪内部；建立免插桩崩溃探针**
  - **账本三十六作废的部分**：`setrlimit` 修复**无效** —— glibc 在进程启动时缓存
    默认 pthread 栈大小（读一次 soft RLIMIT_STACK），之后 setrlimit 改不动它。
    实证：`pthread_getattr_np` 打在 `ALOFA_STACK_MB=64` 下仍是 **8388608**。
    并且 **bash `ulimit -s 65536` 今天同载荷也秒崩** ⇒ 栈溢出**不是**根因。
  - **刀法全面作废（重要教训）**：同条件交替多轮取样（每臂 5 轮 × 18 s）后，
    FAKE 5/5、CUT_TICK 3/5、CUT_FWD 2/5、全量 2–3/5、shards=1 2/5 —— 
    单轮存活全是**运气**。以前任何「砍 X 后活」的结论（含 09-30、11-03 两轮）
    都不能作为证据。**规矩**：能崩的题一律 5 轮 × 交替臂；单轮只算轶事。
  - **干净的对照**：`ENGINE_THREADS=0` 0/5 崩；纯并发分配 churn（两线程各 20 万
    次 String 建/毁，含 `print` 与异常抛出/捕获）**0/5** ⇒（a）线程化专属；
    （b）分配器双线程并发本身清白。崩溃率随**请求量**涨（FAKE 模式请求率最高、
    崩得最密）。
  - **新仪器：崩溃探针**（`ALOFA_SEGV_PROBE=1`，serve.mojo）：`@export("alofa_segv_probe")`
    + `abi("C")` 处理器 → `dlsym(RTLD_DEFAULT)` 取址（构建必须加
    `-Xlinker --export-dynamic`，否则符号不进 .dynsym）→ `sigaction`（结构体
    一页零 mmap：handler@0、SA_SIGINFO@136；oldact 存 `MAP_FIXED` 固定页
    `0x6a0000000000`）。处理器把 `si_addr`(siginfo+16)/`rsp`/`rip`/`rbp`
    (ucontext gregs[15]/[16]/[10]，即 +160/+168/+120) **原字节 write(2)**
    （tag `b"SGVADDR2"`）后**恢复运行时处理器并返回** —— 故障指令重执行，
    运行时打印完整符号化栈。两份信息同一次崩溃都拿到。
  - **崩溃画像（首次拿到完整证据）**：典型链
    `#8 Tokenizer::decode ← #9 _batch_text ← #10 _batch_next ← #11 stream_next`
    ← `#7/#6/#5 libAsyncRTRuntimeGlobals` ← `#4`（故障指令）。#4 所在函数是
    **snprintf 风格格式化器**：`{ptr,len,pos}` 结构（`mov (%rbx),%rdi; mov
    0x8(%rbx),%rsi` → 定长片段拷贝 → `add %rax,0x10(%rbx)`），上下文串取自
    `0x14a0(%r13/%r14)`，`r13` 从栈上取再 `mov 0x18(%r13),%r14` 解引用。
    历次 `si_addr`：`0x54534f50`（ASCII **"POST"**！）、`0`、`0x11817`
    （前者减 0x18 恰为圆整的 `0x11800`，像整数被当指针）。rip 收敛到同一
    函数（+0x70433/70449 两站点）。⇒ 损坏的是**运行时格式化/追踪路径的上下文
    或缓冲指针**，且值带着「请求字节/整数」的味道 —— 要么运行时外部线程的
    上下文未初始化，要么仍有把整数/请求字节写进指针槽的堆损坏。
  - **注意**：`#0–#3` 是运行时处理器自己的 backtrace/符号化帧（#0 就是
    `call backtrace@plt`），**历次日志里的"分配器受害者栈"全部读错** —— 
    真正的故障帧是 #4 起、受害者调用链是 #8 起。
  - **下一步**（按序）：① 独立最小复现器（flare 式裸 pthread + 真实字符串/格式化
    混合，不依赖整个服务）；② 用 `LD_PRELOAD`/`LD_BIND_NOW` 或 MOJO 环境变量
    尝试拿到运行时调试信息；③ 既然 8 MiB → 64 MiB 不影响，试**显式 attr 起线程**
    （64 MiB + 大 guard）看崩溃率是否变化 —— 这是区分"栈/线程布局"与"上下文"
    的廉价实验；④ 若确认运行时缺陷，向上游报 issue 并考虑「engine 线程不做
    String 格式化」的绕行（把 decode/serialize 挪回 reactor）。

- **2026-11-05（第三十八条）** —— 🟢 **触发面收窄到「engine 线程上的 sse 帧构造」，tcmalloc 分配采样嫌疑坐实一半；bisect 刀全套入列**
  - 本条所有 A/B 都是**同轮交替、每臂 5–6 轮**（三十七的规矩）。今天先做两组对照再动刀：
    **显式 attr 起线程**（`ALOFA_THREAD_STACK_MB`，自写 `spawn_with_stack`：`pthread_attr_setstacksize`，
    flare 的 spawn 写死 NULL attr）：default 4/5 / attr64 4/5 / attr8 3/5 ⇒ **线程栈大小与布局彻底无关**
    （"64 MiB 大栈修复"这条线正式埋葬）。
  - **bisect 链**（臂 → 崩率；FAKE 基线 4–5/5；全部同轮交替）：
    - `ALOFA_ET_MAILBOX_ONLY`（取活不碰 handler，回固定字节）→ **0/5**：mailbox 机制并发清白。
    - `ALOFA_ET_CUT_HANDLER`（不碰 handler，保留 parse+serialize）→ **0/5**：触发点在 handler 侧。
    - `ALOFA_ET_CUT_PARSE`（跳过 parse_request）→ 1/5：parse 清白。
    - `ALOFA_ET_CUT_CHAT`（跳过 JSON 体解析）→ 2/5：JSON 解析清白。
    - `ALOFA_ET_CUT_TEXT`（跳过 decode）→ 1/5：decode 清白（"挪回 reactor"绕行死刑）。
    - `ALOFA_ET_CUT_SVC`（跳过 service.stream_begin/end）→ 3/5：service 簿记清白。
    - **`ALOFA_ET_CUT_FRAME`（帧改字面量，保留表写入与相位机）→ 0–1/5：触发点 = sse 帧构造链**。
    - CUT_FRAME + escape_json 加回 → 0/5：escape_json 清白。
  - **FAKE 模式崩溃栈（探针链式捕获）**：`#8 parse_head ← #9 parse_request ← #10 _engine_main`
    → 运行时 `#7(+0x51303)` → 格式化器 `#4(+0x94433)` —— **与 decode 链同一个 #4**。
    parse_head 在崩路上只做 `bytes_to_text`（构造 String = 分配）⇒ 崩点函数是
    **由分配路径进入的运行时 per-thread 追踪/采样记录写入器**。
  - **tcmalloc 采样在场**：`_ZTH16tcmalloc_sampler`（TLS 采样器）、`allocation_sampling.h`、
    `profile_sampling_interval`。**关采样两形式都显著降但不除根**：
    `TCMALLOC_SAMPLE_PARAMETER=0` 4/6→2/6；`MALLOC_CONF=sample_parameter:0` 3/6→1/6。
    ⇒ 采样是**贡献者**（崩率随 engine 线程分配量涨），要么 env 没真正关掉内嵌版的采样，
    要么另有第二触发源。
  - **解释力**：采样按分配字节量 1-in-N 触发 ⇒ 完美解释"插桩即愈/时序敏感/请求量相关/
    刀法的剂量效应"（CUT_FRAME 砍掉的正是每帧十几个小分配）。si_addr 的 "POST"/0x11817
    = 采样记录写入器读到了被复用内存的垃圾指针。
  - **新刀（全部 env 门控默认关）**：`ALOFA_ET_MAILBOX_ONLY` / `ALOFA_ET_CUT_HANDLER` /
    `ALOFA_ET_CUT_PARSE` / `ALOFA_ET_CUT_CHAT` / `ALOFA_ET_CUT_TEXT` / `ALOFA_ET_CUT_FRAME` /
    `ALOFA_ET_CUT_SVC` / `ALOFA_THREAD_STACK_MB`。CUT_FRAME 内置 escape_json 反向二分子刀。
  - **下一步**：① 找准内嵌 tcmalloc 的采样开关（试 `TCMALLOC_SAMPLE_PARAMETER` 大值 2^30、
    `MALLOC_CONF=tcmalloc_sample_parameter:0`、或启动时 ` mallopt`/符号 interpose）——
    若能真关掉且 0/5，就得到一个可发布的一行缓解；② 工程绕行：**engine 线程不做小分配**——
    帧的 JSON 拼接挪回 reactor（engine 只交 token 与相位），或给 parse/帧路径做缓冲复用；
    ③ 带着 #4/#7 偏移与探针证据向 Modular 报 issue（外来 pthread 上的 per-thread
    采样/追踪状态未初始化）。

- **2026-11-06（第三十九条）** —— 🟢 **破案：flare `Mutex` 在双线程并发下腐坏堆；邮箱换 raw pthread 互斥体，engine 线程崩溃根除**
  - **符号化定案**（上条遗留的 #4-#7 偏移）：全部落在 `TCMallocInternalDeleteArraySizedAligned`
    —— sized-delete 的 CHECK/采样路径。1 GiB 采样间隔 A/B（5/6 vs 6/6）把"采样"嫌疑否掉：
    崩在 delete 路径是因为**堆账已坏**，不是因为采样。
  - **独立复现器**（`scripts/diag_repro_mailbox_burst.mojo`，约 200 行，不依赖服务器/模型）：
    两条 flare 线程 + 互斥邮箱（Job 8 KB/Result 16 KB 内联元素，`reserve` 到上界）。
    - 单线程同序列自环 **1736 万次 ops 存活** ⇒ `List` 大元素 push/pop 无辜。
    - 双线程（burst=0，无任何用户分配）：**崩率 5/5**；burst=32 崩率同样 5/5。
    - 现行捕获：互斥体的 pthread 结构被写坏后 `pthread_mutex_(un)lock` 返回
      **rc=22 (EINVAL)**（flare 对 rc 断言 ⇒ SIGILL）；堆坏后两个活块重叠（一个
      String 的缓冲被分配到互斥体存储上）⇒ 服务器上"si_addr=POST"（指针槽里是请求
      字节）与崩点函数同源。
  - **根因：flare `Mutex`**。对照刀（复现器内）：
    - 纯 lock/unlock 隔离（无 List）：flare Mutex 多轮内必现 rc=22；**raw pthread 4/4 存活**。
    - 全邮箱高速搬运：flare Mutex ~半数崩；**raw pthread 4/4 存活**。
    - 即 `Mutex._storage`（unsafe_alloc 堆块）在并发下被写坏 —— 与分配器交互的
      运行时级问题（待上游 issue，复现器可交）。
  - **修复**：`srv/engine_thread.mojo` 新增 `RawMutex`（inline 48 字节 pthread 互斥体，
    与 flare Mutex 同 `lock()/unlock()` API，36 个锁点零改动），邮箱 `mutex` 字段换型。
    服务器验证（同轮、8 并发、30 s）：
    - FAKE 6 轮：**0/6**（修复前 FAKE 基线 4–6/6）。
    - 真模型 `ALOFA_ENGINE_THREADS=1`：70/70 请求 0 错误、无 damaged、服务器存活
      （17.7 tok/s 宽 4，仅作存活口径不作性能口径）。
  - **状态改判**：⑥ "engine 线程崩溃"由 🔴 未破改判 **🟢 已根因修复**（flare Mutex 上游
    bug，本仓库侧已绕开）。`ALOFA_ENGINE_THREADS` 默认仍 0（保守，待更长 soak 再开默认）。
  - **遗留**：① 向 flare/Modular 报 Mutex 并发 bug（附复现器）；② 复现器里的
    `REPRO_MUTEX_ONLY`/`REPRO_RAW_MUTEX`/`REPRO_SINGLE`/burst 旋钮保留作回归刀；
    ③ 服务器全部诊断刀（CUT_* 系列）保留，默认关。

- **2026-11-06（第四十条）** —— 🟢 **soak 全绿 + 重门 `test-engine` 由红转绿：非流式应答被补 `[DONE]`（预存，非 RawMutex 回归）**
  - **soak**（`ALOFA_ENGINE_THREADS=1` + `RawMutex`）：FAKE 6 轮 × 60 s **0/6 崩**；
    真模型 3 轮 × 120 s **0/3 崩**（流式，4 并发，70/70 口径照旧）；异常日志零条。
    三十九"待 soak 再开默认"的前提已满足。
  - **重门 `test-engine` 首次复跑即红**：`the plain answer changed shape` —— 线程路
    非流式应答尾部多 16 字节 `data: [DONE]\r\n\r\n`。worktree 在三十九之前的
    `caa2094` 上跑同门**同形状同红** ⇒ **预存**，与 RawMutex 无关（这重门故意不进
    `pixi run test`，自 09-28 加"中途断开"性质后没人跑过）。
  - **根因**：`collect` 里 `ends_sse_stream(out)` 认 `"finish_reason":"`（非 null 值），
    而**非流式应答同样含 `"finish_reason":"stop"`** ⇒ 普通应答被误标"流结束"，
    `RES_DONE` 一支按 `ended[slot]==1 && !done_sent` 补了终止符。客户端按
    Content-Length 读不受影响，但 **keep-alive 上的下一个响应前面会多 16 字节垃圾**。
  - **修复**：标记与判重都加 `self.streaming[slot] == 1` 前置（由 `RES_STREAM_HEAD`
    置位；非流式请求没有头，永远到不了）。一行语义修复，`guard_stalls` 的同款
    补终止符分支随之干净。
  - **验证**：`test-engine` 红 → **绿**（三种模式逐字节形状 + 三次中途断开 + 探活
    答自己的，全过）。`test-http` 兜底重跑。
  - **状态**：⑥ 维持 🟢；线程路的两个历史已知病（堆腐坏 = flare Mutex、断开串味 =
    finish_reason 误判）都已根因修复。`ALOFA_ENGINE_THREADS` 默认仍 0 —— 翻默认
    是下一步（需要连带重跑全部重门，另一轮做）。

- **2026-11-06（第四十一条）** —— 🟢 **`ALOFA_ENGINE_THREADS` 默认翻转 0→1**
  - **波及面核对**：默认值只有生产 serve 二进制消费 —— test-engine 直连
    `Loop.run`/`run_threaded`（免疫）、test-serve-soak 显式 `engine_threads=1`
    （免疫）、没有任何测试读该 env。README:46 本就写着"默认 1"，这条翻转只是
    让代码追上文档。
  - **改动**：`srv/config.mojo` 默认 `0→1`；字段文档与 serve.mojo 文件头、README:113
    的旧口径（"实验路/断开会搞坏请求/§8"）一并翻新到三十九/四十的最终事实。
  - **验证**：serve_probe 重编后**默认 env** 冒烟（不设任何 ENGINE_THREADS）：
    启动口径 `[note] engine threads: 1`、非流式无 `[DONE]` 泄漏、流式 `[DONE]`
    完整、服务存活；`pixi run test`（41 套件）+ `test-batch-stream` +
    `test-http` 三重绿。test-engine 不走 config（上轮已绿）。
  - **状态**：线程化成为默认生成路（一条 5 s 生成不再占住 reactor 循环）；
    `=0` 仍是受支持串行退路。⑥ 至此彻底收官（破案 → 根治 → soak → 重门 → 默认开）。

- **2026-11-06（第四十二条）** —— 🟢 **上游 issue 材料定稿**（`docs/plan/upstream-issue-flare-mutex.md`，待提交 modular/modular）
  - **复现器 MONLY 臂补flare/raw 分刀**（此前 MONLY 两边都走 raw，头注释声称的
    "纯 flare lock/unlock 崩"没被代码覆盖）：默认打 flare `Mutex`，
    `REPRO_RAW_MUTEX=1` 才走 raw 对照。
  - **最小复现确认**：两线程只 lock/unlock 同一个 flare `Mutex` —— 新鲜四轮
    （8-10s/轮）：1× SIGILL（flare 自己 assert `mutex.mojo:114:41` 打出
    `pthread_mutex_unlock failed with rc=22`）/ 1× 挂死 / 2× 存活，非确定；
    raw pthread 对照存活。FULL 邮箱臂依旧每跑必崩（764,370 jobs 后 SIGSEGV，
    `String._add → alloc → libAsyncRTRuntimeGlobals.so+0x4f2c2`）。
  - **issue 内嵌最小片段独立验证过**：从草稿原样提取 → 编译通过 → 20 秒内
    复现（同款 start_thread 栈形）；初版片段漏 `raises`（`Mutex()` 会抛）已修。
  - 环境：Mojo 1.0.0 (ed45d567) / Ubuntu 22.04.5 / i7-9700K。

- **2026-11-06（第四十三条）** —— 🟢 **engine 孪生体共享只读权重**（roadmap 3.2c 的代价消掉）
  - **先量再改**：engines=1 的 PSS 2.94 GB，engines=2 是 4.90 GB —— 每多一条
    engine 线程 **+1.96 GB**。实体不是文件映射（`MappedFile` 是 PROT_READ 的
    MAP_PRIVATE，同一文件的页在进程内本就共享），而是 loader 里那份 **bf16→fp32
    加宽副本**（988 MB 文件 → 1.98 GB 私有 arena），外加 q4 档每 twin 再一份
    0.2 GB 量化块。注释里"一份 = 一份权重"说的就是它们。
  - **改法**：`TensorFile.arena` 与 qwen 的 `rope/q4/q8` 三个 arena 换成新的
    `SharedRegion`；`TensorFile(proto, ...)` 与 `QwenForward(proto, max_tokens)`
    两条借用构造；`ModelService.spawn_twin` 改走后者。KV / scratch / 流相位
    仍是逐 twin 新建（共享它们是"偶尔答错一个 token"那一类失败）。
  - **生命周期（本轮最贵的一课）**：先试了引用计数，失败 —— Mojo 按**字段最后
    一次使用**析构，而 `TensorView` 里是未跟踪的裸指针，借用计数会在临时量析构
    里被吃掉、字段又可能提前回收，实测权重在第一次前向之前就被 munmap（`rope`
    里读到已解映射地址）。最终改成**只进不出的借用标记**：映射头 64 字节，发生
    过一次 `borrowed` 就把标记置位，此后所有句柄都不回收；**没人借过的单实例
    照常 munmap**（这一条是被 `test-batch-stream` 抓出来的：纯泄漏版连跑六个
    测试攒到 ~13 GB，排队那条 stream 的 token 开始对不上）。
  - **量到的结果**：engines=2 的 PSS 增量 **+1.96 GB → +34 MB**（2.94 → 2.98 GB，
    余下是 tokenizer/act/KV/sampler）；engines=2 双并发贪心答案与 engines=1
    单跑**逐条相同**。
  - **门**：`test-model-dir` 新增
    `test_twin_borrows_weights_and_matches_logits`（同一加宽区地址、KV 地址不同、
    同 prompt 的 logits **逐位相同**）→ 7/7；`test`(41 套件)、`test-memory`、
    `test-engine`、`test-http`、`test-batch-stream` 全绿。
  - **边界**：借用过的权重区不回收（服务只在启动时加载一次，代价可忽略；测试里
    重复加载会累积，这是它唯一的代价）；N 条 engine = 1 份权重 + N 份 KV/scratch。
  - **旁注（未处理，与本条无关）**：裸 `main()` 程序里直接调 `prefill` 会崩
    （`libKGENCompilerRTShared` 帧），基线与改动后同样崩、`TestSuite` 里同一段
    代码正常 —— 属于既有的"程序形状"问题，没有追。

- **2026-11-06（第四十四条）** —— ✅ **单流天花板参照勘误：40.7 不复现，alofa 单流（主战档）已不弱于 llama.cpp**
  - **起因**：账本二十七起把"llama.cpp q4_0 tg32 单流 40.7 tok/s（同机带载）"当单流
    天花板参照挂在余量清单上（三十一/三十二/三十三连引四次），但翻遍账本**找不到
    它自己的测量行** —— 09-29 的实测是 27.05 ± 2.80。今天重测配对验证。
  - **配对方法**：同一 load 窗口交替跑 `llama-bench -m qwen25-05b-q4_0.gguf
    -p 5 -n 32 -r 1/3 -t 1` 与 `bench_vs_llamacpp`（q4_0+q8_head，shards=8，
    同 prompt 5 出 32；两边流量形状一致 —— gguf 默认档的嵌入/输出就是 Q8_0）。
  - **新鲜配对（load 5–11）**：llama.cpp tg32 = 22.95 ± 5.19（r3 轮），另三轮
    23–27；alofa shards=8 = 23.4…31.9（中位 ~28）。**alofa ≥ llama**（其 -t 8
    在带载下崩到 1.04，9-29 曾 0.38 —— 它的强档就是 -t 1，这个比较对它有利）。
    f32emb 变体 16.80 —— 更慢，也排除"40.7 出自 f32emb"。
  - **口径不变的部分**：纯单线程口径 alofa 仍慢 ~1.5×（shards=1 的 14–16 vs llama
    -t 1 的 23–27，带载下）；追平靠 shards=8。这条照旧记账，不因勘误翻案。
  - **处置**：余量清单里"单流 25.x vs 40.7"的债务**删除**（它把优化方向指向了
    一个不存在的差距）；单流性能记录改挂同口径配对数字。参照数字必须带自己的
    测量行 —— 这条教训和"每项优化附前后对比"是同一条。

- **2026-11-06（第四十五条）** —— 🔴→✅ **老路活流被顶掉（`max_tokens > MAX_GEN` 的并发全部截断）** + 4.5 前置：**批解码有 ~48% 结构性空闲**
  - **怎么撞见的**：给 4.5 量 CPU 空闲占比时，8 条 `max_tokens=64` 的流只用了 157% 核。
    核实 token 数才发现 —— **流 0 出 66 个 token，其余 7 条各出 1 个就 EOF**，服务
    日志是 `this service is already streaming another request [asked=0
    streaming=7]`。先前那组"80% 空闲"因此作废：真正在解码的只有一条。
  - **根因**：`max_tokens > MAX_GEN`(32) 的请求走**单流老路**（一次只握一条），而
    `stream_begin` 里"先收掉上一条没走完的流"是**无条件**的 —— 它连正在服务的流
    一起收，所有权被最后 begin 的那条抢走，前面几条下一拍就成了"拿别人的号来问
    这一步"而被 `stream_next` 打断。文档承诺的"第二条**指名拒绝**"被这把清除
    架空了：拒绝路径永远走不到。
  - **修**：收走的前提改成**静默超时**（`STREAM_TAKEOVER_NS` = 30 s，每次
    `stream_next` 刷新 `stream_touched`）—— 对端消失、循环没通知到的僵尸流照样
    能收；还活着的流一律由下面那段抛 `ERR_CAPACITY`。实测断连接管仍然成立（掐掉
    首流 0.8 s 后新流 200）。
  - **门**（先红后绿）：`test_batch_stream` 新增
    `test_a_live_old_path_stream_is_not_stolen` —— ① 第二条 begin 必须抛；② 第一条
    被打断后必须还能出 token；③ 第一条收尾后第二条必须能进。修前红在第①条
    （断言 300 行），修后 7/7。线级复验：8 条 40-token 流 = **1×200（42 delta +
    `[DONE]`）+ 7×400**，不再有一 token 截断。
  - **4.5 前置数字**（8 核，批路形状 32 token；进程忙 = Σ线程 CPU / Δ墙钟）：
    - 批解码 8×32：**109 ms/步**（聚合 73 tok/s）、**420% 核**、**结构性空闲 47.5%**；
      8 条 shard 线程**各 ~52%**、没有一条接近 100% ⇒ 空闲不是"某段串行"造成的，
      而是**每层一次任务派发的同步气泡**（24 层 × 唤醒延迟）。
    - 长 prefill 1×128：1.06 s（~810 tok/s）、355% 核、空闲 55.6%。
    - 单流 1×32：31.6 ms/token、500% 核（shards=8 的单 token 并行）、空闲 37.4%。
    - 对照：8 个 Python 自旋线程同窗口能拿满 **800%** ⇒ 这些空闲是**我们自己的**，
      不是被抢核；整机另有余量 7–18%（别的租户）。
    ⇒ **4.5 的判据成立**：批解码的 step 里有近一半时间所有 shard 线程同时在等，
    分块预填充有真实的填充空间；下一步要动的是"把 24 层的派发合并/流水化"。
  - **顺带量到（四十一的默认翻转值多少钱）**：同探针 `ALOFA_ENGINE_THREADS=0` vs
    `1`（批路形状）：批解码 5.58 vs **3.50 s**、长 prefill 2.01 vs **1.06**、单流
    1.99 vs **1.01** —— 线程化路快 **1.6–2×**，而且 CPU 占用更低（240% vs 420%）。
    ⚠️ 两轮之间 load 从 7 漂到 22，没有交替配对，只作方向性记录（四十一当时只验了
    正确性，这条是它缺的性能面）。

- **2026-11-09（第四十六条）** —— 🔬 **批解码气泡解剖：三个假设死两个，"47.5% 空闲"改判**（纯测量，零产品码改动）
  - **问题**（接四十五）：批解码 8×32 每步 109 ms、8 条 AsyncRT 工作线程各 ~52% ——
    空闲在哪？三个嫌疑逐个用测量杀。
  - **假设一：TaskGroup 派发延迟（死）**。微基准（`scripts/diag_taskgroup_cost.mojo` +
    `_thread.mojo`）：空 TG(1 任务) 主线程 0.6 µs / 裸 pthread 6.3 µs；8 微任务
    2.9–5.8 µs。每步 ~192 次派发 ≈ **~1 ms**，离 50 ms 差一个量级 —— 不是它。
  - **假设二：批路选错带宽档 RB=8（死，且险些误杀）**。核级扫描
    （`scripts/diag_ru_sweep.mojo`，RB/RU × n_tok∈{8,32}）：单次调用 RB=2/RU=4 比
    RB=8/RU=1 每遍快 2.2–3.2× —— 看起来生产档全面劣。**但** `band` 包装是按
    token 带循环、**每带重读全部权重**的：RB=2 在 n_tok=8 要 4 遍。按生产语义折算
    RB=8 反而 1.8× 快。模型级交替配对（`diag_step_shape.mojo` 两臂各 3 轮）钉死：
    **wide t=8 = 90.6–98.6 ms vs 全窄带 107.3–118.1（wide 稳定快 ~15%）**，t=1 持平。
    生产分流保持原样。⚠️ 权重重复遍吃的是 L2/L3（单投影 2.4 MB），二十七的成本
    模型里"遍数 × DRAM"的假设过时了，但结论碰巧还对。
  - **假设三：服务层加延迟（死）**。in-process t=8 前向 90–98 ms ≈ 服务端批步
    109 ms —— 服务层（mailbox/序列化/派发）占比 ~10%。
  - **改判**：批步 ≈ **90% 是核时间**。构成：q4 投影核 ~80–95 ms + **lm_head
    （q8，144.6 MB/步，批宽无关）~14–16 ms** + 串行算子 ~4 ms。线程账也能对平：
    8 工作线程 × ~52% + engine ~8% ≈ 核时间/墙钟 —— 工作线程的 ~48% 空闲是
    **每个 TG 里等最慢分片**的屏障空闲（每步 ~168 个 TG），不是"没活派"。
  - **对 4.5 的含义**：空闲是**屏障空闲**，架构上可被另一条前向的 TG 填充，但
    填充者会与解码核抢同一条权重流与同一条依赖链 —— chunked prefill 的期望收益
    要按"屏障空闲 × 可并发度"打折，不能再按"47.5% 空核"线性乐观。
  - **教训（再次确认三十八的纪律）**：**单次相邻对照会撒谎** —— 全窄带第一次
    A/B"快 16%"，交替三轮后反转成"慢 15%"。负载漂移（runq 9–24）下，<3 轮
    交替的 perf 结论一律不作数。
  - **仪器沉淀**：`diag_step_shape`（in-process 步时 + profile 桶）、
    `diag_ru_sweep`（RB/RU 扫描）、`diag_taskgroup_cost/_thread`（派发成本，
    主线程 vs 外来线程）。均为 micro-bench，不进 `pixi run test`。

- **2026-11-09（第四十七条）** —— ❌ **抬 `MUB_ROWS_MAX_XQ` 让 `down_proj` 走宽带：试过，不采纳**（负结果，零产品码改动）
  - **来由**（接四十六）：批路 `down_proj`（inner=4864）被 16 KB 的 `MUB_ROWS_MAX_XQ`
    顶进窄带 RB=2 ⇒ n_tok=8 要 4 遍权重；核级扫描（四十六）说生产语义下 RB=8 快
    1.8×。把上限抬到 40960（xq 缓冲 40 KB）应该白捡。
  - **实测**（两臂交替 3 对，`diag_step_shape` 加了 t=32 臂；load 28–33 重污染窗）：
    中位 t=8 = 155.6 vs 164.9 ms（B 快 6%）、t=32 = 471.6 vs 526.3（B 快 10%）、
    t=1 = 75.8 vs 69.0（**B 慢 10%，而 t=1 两臂同码、理应零差**）。
  - **判决**：t=1 的"差"就是**噪声标尺** —— t=8/t=32 的收益全部埋在地板下，可能
    是零到 10%，不够抵一次二进制翻动 + 数值门重跑的验证成本。**保持 16384**。
  - **机制假设**（没再深挖）：RB=8 × cols=4864 的 `xq` 缓冲 38 KB > L1(32KB)，
    8 条 shard 线程各写一份 ⇒ 核级扫描（单线程、整矩阵）里 1.8× 的 ILP 赢面被
    缓冲自身的 L1/L2 流量吃掉。**核级微基准的胜利不自动迁移到 8 路分片的模型级**
    —— 与四十六"带宽档"那条的教训同构，这是第二例。
  - **仪器增量**：`diag_step_shape` 加 t=32 臂（顺带量 prefill 形状）。

- **2026-11-09（第四十八条）** —— 🔍 **屏障均衡诊断：~48% "结构性空闲"改判为邻居抢核**（探针 + 等量对照 + 同时刻整机账）
  - **要回答的**（接四十六）：批解码步内 ~90% 是核时间，8 条 worker 在核时间里只忙
    ~57% —— 那 43% 是唤醒、掉队还是收尾？挂真探针量。
  - **探针**（`src/alofa/model/probe_barrier.mojo`，编译期开关，默认关）：每条 TG
    记 `dispatch → 各分片 start/end → wait` 五个戳，按 q4 无偏置/带偏置/q8 头三类
    聚合。挂在三条批路派发器 + 单流头上；fp32 通路与注意力没挂。
  - **账**（in-process t=8 一步，8 分片，load ≈ 12）：
    | 类别 | TG数 | 墙钟 | 唤醒 | 掉队 | 收尾 | 并行利用率 |
    |---|---|---|---|---|---|---|
    | q4 无偏置 | 96 | 515.8 | **0.8** | 80.1 | 62.2 | 40% |
    | q4 带偏置 | 72 | 231.0 | **0.8** | 55.6 | 97.0 | 19% |
    | q8 头 | 1 | 9983.8 | **0.7** | 3186.6 | 17.3 | 62% |
    （µs，每类 TG 的平均值）
    - **唤醒 ≈ 1 µs** —— 派发→开跑不是问题（这是第三次独立证据，接四十五/四十六）。
    - **掉队很大**：同一 TG 里行数完全相同的分片，耗时差到 **4×**（原始记录：
      `k0=[0.5,799.7] k1=[5.8,201.9] ... k7=[202.4,395.6]`，墙钟 808.6）。
    - 收尾（`wait()` 返回滞后）= 62–97 µs/TG，每步累计 ~11 ms（约 12%）—— 唯一
      坐实的"我们自己的"开销，量级远小于 43%。
  - **等量对照（决定性）**：一个 TG 里 8 个**等量纯计算**任务（寄存器内 FMA 链，
    不碰大内存、不经过模型代码）：n=1/2/4 墙钟基本相等（1.00/0.99/1.10×），
    **n=8 = 2.01×、n=16 = 3.00×** ⇒ 实际并发只有 **~4**。同样的 4× 分片差在纯
    计算里原样复现 ⇒ 与我们自己的屏障/内存/派发**无关**。
  - **同时刻整机账**（这是 45 缺的那一半）：8 个自旋线程能拿 756%，说明 OS 不
    欠我们核 —— 但那是 3 秒样本。用"整机 + 本进程同时采"重测：
    - 纯计算 TG 负载：整机 idle 2.1%（7.8 核忙），**我们 403%（4.0 核），邻居 3.8 核**；
    - 服务端批解码 8×32：整机 idle 1.9%，**我们 288%（2.9 核），邻居 5.0 核**。
    ⇒ 这台机器有 **~4 核被别的租户长期占着**（loadavg 12–15），我们 8 条 worker
    只能分到 ~4 核 ⇒ 每条 50% 左右、"8 线程各 52%"正是这个，不是屏障。
  - **改判**：四十五的"结构性空闲 47.5%"**不成立**（至少不全是）；四十六的
    "8×52% ⇒ 每 TG 等最慢分片"也随之作废 —— 最慢分片之所以慢，是**那条线程被
    挤下了 CPU**，不是我们让它等。3 秒自旋对照之所以误导，是因为邻居负载是
    阵发的（bursty），恰好在那 3 秒歇了。
  - **对 4.5 的含义**：chunked prefill 的前提是"有空核可填"，在这台共享机上
    **前提不成立**（空核被邻居拿走了）。要验 4.5 必须换安静/dedicated 机器，
    或者接受"这台机的数字只作方向性参考"。
  - **顺带确认**：AsyncRT 的 worker 池就是那 8 条 `🔥 ThreadN`（进程共 9 线程 =
    1 主 + 8 worker），与四十五的线程表对上。
  - **遗留**：`shards=4` vs `8` 的 A/B 已备好脚本（在 ~4 核可用时 8 路分片是超订，
    4 路可能不慢反快），本轮被就绪探测的 bug 挡住，下一轮补。
  - **仪器**：`diag_tg_spread`（等量计算分布）、`diag_tg_parallel`（并行度定标）、
    `diag_tg_threads`（长任务 + 外部采样）、`diag_step_shape` 的探针臂。
  - **P5 侧**：A100 验证机复验可达（6×A100，卡 0/2 空闲、7 GB 占用、0% 利用率），
    源码已 rsync 到 `/app/lcl/mojo-projects/alofa`（远端 `.pixi` 3.3 GB 在）。

- **2026-11-09（第四十九条）** —— ✅ **P5 CUDA：rmsnorm 差分门在 A100 通过**；`shards` A/B 本机无结论
  - **shards=8 vs 4 A/B**（四十八遗留，服务端 8×32 交替 4 轮）：8 路均值 41.0 tok/s、
    4 路 44.0 —— 差 7%，在本机噪声地板（±10%）之下，**无结论**。两臂都只拿到
    2.0–2.7 核（邻居 5.9 核，load 18+）：核都不够分，分几路都一样。要在安静机上
    才能定 4 路是否真的不亏。
  - **CUDA 差分门 PASS**（A100 验证机，卡 0 空闲实测 7GB/0%）：
    - `tests/gpu/vecadd.mojo` 冒烟 PASS（`c[999]=2997` ✓）；
    - `tests/gpu/test_cuda_diff.mojo`：rmsnorm GPU vs 标量同输入逐值比，
      **`diff=0.0`（逐位一致），容差 5e-5** —— 比判据还干净。解锁条件③
      （"GPU 算的数与 CPU 算的数是同一个"）成立，CUDA 后端从
      `verified-capable` 升为 **rmsnorm `verified`**。
  - **环境坑（新增两条）**：
    - a100.sh 文件头说 SSH 必须走 3389 —— **已过时**，22 现在直通（脚本待改）；
    - 远程 `mojo run` **必须带 `-I src`**，否则 `alofa.verify` 报 "does not refer
      to a nested package"（本地惯例一直带，容易忘）。
  - **下一步（P5）**：测试文件里的 `rmsnorm_gpu` 提为 `src/alofa/kernels/cuda/`
    正式内核（目录尚不存在，架构图 L1 里画了）→ 后端分发接上 → 差分门扩到
    更多 kernel（先 matmul 碎片，再量化反量化）。

- **2026-11-09（第五十条）** —— ✅ **P5 CUDA：`kernels/cuda/` 落地，反量化差分零容差通过**
  - **`src/alofa/kernels/cuda/` 正式建包**（`__init__` + `basic`，账本四十九的下一步）：
    包 docstring 写明纪律（本地不跑、每个 kernel 过差分、DevicePassable 约束）；
    `rmsnorm_gpu` 从测试文件提为产品代码，测试改为导入 —— 「门测的必须是产品
    代码」否则差分绿了也白绿。
  - **新增 `dequant_q4_0_gpu`**（一个块一个线程）+ 设备端 fp16 位解码
    `_fp16_bits_to_f32_device`：与 `kernels/cpu/quant.mojo` 逐句对应（指数/尾数
    整数移位 + 2 的整数次幂，全程精确）。设备不能 raise，inf/NaN 缩放因子走位
    拼接给 IEEE 语义（主机侧 `quantize_q4_0` 本来就会拒绝溢出，正常流到不了）。
  - **差分门扩展**（`tests/gpu/test_cuda_diff.mojo`）：块从**产品量化器**
    `quantize_q4_0` 生成 —— 顺带验了 GPU 读的块布局与量化器写的一致；参照是
    产品 `dequant_q4_0`。判据**零容差**（这层结果只由 nibble 与缩放因子决定，
    都是精确的，用容差是放水）。
  - **A100 实测（卡 0）**：`*** RMSNORM GPU/SCALAR DIFF PASS *** diff= 0.0`、
    `*** DEQUANT Q4_0 GPU/SCALAR DIFF PASS *** diff= 0.0` —— 两个 kernel 都与
    CPU 标量**逐位一致**。
  - **踩坑（Mojo 1.0）**：`ref` 是**关键字**（ref 绑定），`var ref = ...` 直接
    解析错（报 "unexpected token in expression" 还指在 `=` 上，极难读）—— 变量
    改名 `ref_out` 解决；comptime 别名做内联算术（`arena.alloc(Q4_N * 4)`）在
    某些位置也会触发同样的假错，绕法是先落成运行时 `Int` 变量。
  - **本地编译语义**：`kernels/cuda/` 本地能过解析与语义（import 无碍），卡在
    codegen（本地 ptxas 不认 sm_52）—— 与 tests/gpu 同一条纪律：本地不进套件。
  - **下一步**：GPU 端 `linear`/matmul 碎片差分（q4 核整行 Reduce，一行一线程
    版本先过判据）；之后才是块内规约的性能内核；后端分发接进模型要等
    `DeviceContext` 穿过 model 层的方案定稿。

- **2026-11-09（第五十一条）** —— ✅ **P5 CUDA：差分门扩到 4 kernel，全部逐位一致；transformer 层的核心算子凑齐**
  - **新增两个差分对象**（`kernels/cuda/basic.mojo` + 门各加一段）：
    - `dequant_q8_0_gpu`：与 q4 同一条零容差理由；**有符号码的符号扩展**（`>=128`
      减 256）是这层唯一的坑，漏了所有负权重整体跳到 +128 那侧，门立刻红；
    - `linear_gpu`（无偏置 GEMM，`(行, 输出列)` 一线程，grid 二维铺满）：与
      `_gemm` 同一个累加顺序（`Float64`、`k` 升序、末尾一次 `Float32`）。
  - **A100 实测（卡 0）**：rmsnorm / dequant_q4_0 / dequant_q8_0 / linear
    **四项全部 `diff=0.0`** —— linear 也逐位一致，说明 GPU 编译器没有把乘加
    折叠成改变舍入的 FMA（这是 linear 门最想排除的变量；若哪天真出现，门会把
    差值打出来而不是默默放行）。
  - **意义**：`dst = rmsnorm(x) → x @ wᵀ` 的两条核心路（归一化 + 投影）在 GPU 上
    与标量后端数值等价已被逐值证明；q4/q8 反量化这两个量化通路的字节级读取也
    被零容差钉死。**从「GPU 能跑」到「GPU 算的数与 CPU 是同一个」的链路完整了**，
    P5 解锁条件③对已覆盖 kernel 成立。
  - **规模说明**：差分门不追规模（linear 用 4×16×896），性能数字在门之外另测
    （架构纪律：正确性门与性能基准分离）。
  - **下一步**：块内规约的性能内核（共享内存 + 浮点累加顺序两个新变量，判据要
    从「逐位」换成「容差 + 与一行一线程版对照」）；DeviceContext 穿 model 层的
    所有权方案（后端分发的前置）；性能基准（A100 上 q4 matmul 的 GB/s 首测）。

- **2026-11-09（第五十二条）** —— 📊 **P5 CUDA：q4 matmul A100 首测 —— 一行一线程是延迟瓶颈，q4 的 7× 字节优势被串行化吃光**
  - **`kernels/cuda/matmul.mojo`**：`q4_matmul_gpu`（融合反量化 + 点积，`(行,输出列)`
    一线程）。参照用**产品 dequant + 产品 linear** 组合（不是 AVX2 —— 它按
    `j%8` 切 SIMD 通道、d 折进 x，累加顺序与标量不同，会把"顺序差"和"算错"
    搅在一起）。块流用**平铺量化**造：`inner` 是 32 的倍数 ⇒ 行主矩阵整块量化
    天然就是行主块流，块永不跨行。
  - **正确性**：两个真实投影形状（896→4864、4864→896）diff 都 = 0.0（相对容差
    1e-5）—— 融合 kernel 与产品反量化+linear 逐位一致。
  - **性能（rows=1 解码形状，best-of-3，500 iter/轮）**：
    | 形状 | q4 融合 | fp32 linear | q4/fp32 时间比 |
    |---|---|---|---|
    | 896→4864 | **23.6 GB/s**（105 µs/iter） | **195.6 GB/s**（89 µs/iter） | 1.18× 慢 |
    | 4864→896 | **7.2 GB/s**（344 µs/iter） | **37.2 GB/s**（469 µs/iter） | 1.36× 慢 |
    （A100 HBM 峰值 ~1.5-2 TB/s）
  - **解读**：一行一线程在这个形状下**远没到带宽瓶颈**，是**延迟/占用率瓶颈**
    —— 4864 线程只占 A100 ~22 万线程容量的 2%。q4 臂读的字节少 7×，但每线程
    串行做了 16 轮 nibble 解包 + f64 乘加，绝对墙钟反而比 fp32 臂**慢**
    18–36%。q4 的字节优势要在**每行多线程**（block 规约）下才能兑现。
  - **对模型的含义（量级账）**：0.5B 参数 ≈ 277 MB q4 权重，按 23.6 GB/s 单流
    解码一步要 ~12 ms 权重时间 —— 比当前 CPU AVX2 单流慢。GPU 后端现在只是
    "数值正确"，不是"更快"；下一版（block 规约）才是性能内核。
  - **fp32 臂 195.6 GB/s**（13% 峰值）说明占用率上去后带宽跟得很快 —— 优化
    方向明确：每行 128–256 线程 + 共享内存/寄存器规约，预计能把 q4 臂推到
    数百 GB/s。
  - **坑**：`best` 单位是 ms，打印标成 µs/iter 差了 1000 倍 —— 第一版数字
    "0.12 µs/iter = 20 TB/s"物理上不可能，靠量纲自检抓出来的。已修。
  - **下一步**：q4 block-规约内核（判据从"逐位"换"相对容差 + 与一行一线程版
    对照"，规约顺序是新增变量）；attention/QK 和 softmax 还没碰。

- **2026-11-09（第五十三条）** —— 📈 **P5 CUDA：block 规约版 q4 matmul —— down 形状 8.8×，仍卡字节级加载**
  - **`q4_matmul_block_gpu[THREADS]`**（`kernels/cuda/matmul.mojo`）：一个 (行,输出列)
    分一个 CUDA block，线程按 q4 块 stride 分工，`Float64` 部分和写共享内存做
    树规约。规约顺序与串行版不同 ⇒ 判据换成相对容差 —— 但实测对 CPU 参照
    仍然 **diff=0.0**（f64 精度下两种顺序舍入一致）。
  - **A100 实测（rows=1，best-of-3 × 500）**：
    | 臂 | 896→4864 | 4864→896 |
    |---|---|---|
    | q4_row（五十二） | 87.4 µs / 28.3 GB/s | 345.8 µs / 7.2 GB/s |
    | fp32_row | 89.3 µs / 195.5 GB/s | 469.1 µs / 37.2 GB/s |
    | **q4_blk32** | **41.6 µs / 59.4 GB/s** | **39.2 µs / 63.5 GB/s** |
    | q4_blk128 | 44.1 µs / 56.1 GB/s | 39.5 µs / 62.7 GB/s |
    | 空 kernel 地板 | 2.2 µs | 2.2 µs |
    四臂 diff 全 0.0。**down 形状 8.8×**（345.8→39.2µs）、gate/up 2.1×。
  - **发射地板 2.2µs** —— 40µs 是真实执行时间，不是调度开销；`THREADS=32`
    与 128 几乎无差（分工粒度不是瓶颈），两形状统一用 32 即可。
  - **下一个瓶颈（诊断）**：~40µs 对 2.45MB 还是慢 10×+。每个线程**逐字节**
    读自己的 18B 块 —— warp 内 32 线程的 byte load 各落 18B 远的地址，一条
    指令打 32 个 32B 扇区，**读放大 ~32×**。下一版：每线程认领一个块、
    16B 数据用一次向量加载（uint4）+ 2B scale，加载事务数降一个量级。
  - **模型级量级（更新）**：28 层 × (gate/up 41.6 + down 39.2)µs ≈ **2.3ms/token**
    权重时间（还没算 attention/lm_head，也还没算重叠）。GPU 单流解码从
    "比 CPU 慢"进入"与 CPU 同量级"。
  - **GPU API 踩坑（MAX 26.5 实测）**：裸 `gpu` 包不存在；块索引/线程索引/
    屏障在 `std.gpu`（`block_idx`/`thread_idx`）与 `max.gpu`（`barrier`/
    `AddressSpace`）两处；共享内存用 `std.memory.stack_allocation[…,
    address_space=AddressSpace.SHARED]`。mojoc 是编译产物、strings 抓不到
    符号 —— 靠逐个 import 试错 + 本地语义编译验证（本地能过语义、只卡 ptxas
    正好当 API 探针）。

- **2026-11-09（第五十四条）** —— 🔬 **P5 CUDA：向量加载实验被否决 —— "读放大"理论错了，L1 在兜底**
  - **动机**：五十三把 40µs 归因于"warp 内 byte load 32 路散射 ⇒ 读放大 32×"。
    写了 `q4_matmul_vec_gpu[32]`（每线程认领一整块，16B 数据一次 `load[16]`
    128-bit 读）验证。
  - **实测（rows=1）**：
    | 版本 | 896→4864 | 4864→896 |
    |---|---|---|
    | blk32（byte load） | **41.6 µs / 59.4 GB/s** | **39.0 µs / 63.5 GB/s** |
    | vec32 首版（动态索引） | 68.5 µs / 36.1 GB/s | 68.1 µs / 36.3 GB/s |
    | vec32 `@parameter for` 展开后 | 59.0 µs / 41.9 GB/s | 57.3 µs / 43.2 GB/s |
    四版本 diff 全 0.0。
  - **两个教训**：
    1. **byte load 的散射被 L1 吸收了** —— 一个 warp 的 32 条 byte load 落在
       邻接的 18B 块上，总共就 ~600B，第一笔之后全在 L1。"读放大"是字节
       计数上的假象，不是真正的内存系统瓶颈；
    2. **对 SIMD 寄存器做动态索引（`nib[j]`，j 是运行期循环变量）会被抖到
       local memory** —— 首版比 byte 版还慢 1.6×；`@parameter for` 强制展开、
       索引静态化后挽回 15–20%，但仍输给朴素的 byte load 版。向量化的
       收益 < 寄存器压力的代价。
  - **处置**：`q4_matmul_vec_gpu` 保留在代码里（正确、有完整注释的**反例**
    与数据），blk32 仍是冠军（~60 GB/s）。追查 40µs 的真凶改用**profiler**
    （ncu on A100），不再猜。
  - **API 笔记**：`Pointer.load[width](offset)` 是向量加载的正确姿势
    （`SIMD.load` 在这个版本不存在）；`@parameter for` 是循环体含动态索引
    SIMD 元素时的必需品。
  - **下一步**：ncu profile q4_blk32（若远端有 ncu）定位 40µs；或者接受
    ~60 GB/s 转进 attention 差分 / DeviceContext 穿 model 层 —— 60 GB/s
    已经把单流解码权重时间压到 ~4.6ms，接近"可用"。

- **2026-11-09（第五十五条）** —— 🚀 **P5 CUDA：真凶落网是 x 的散射读取；smem 版 132 GB/s，单流权重时间 12ms → 1.5ms**
  - **ncu 变通**：远端 ncu 2023.1 的注入库不转发 `cuFuncGetParamInfo`（MAX 运行时
    在 ncu 下起不来，直跑正常；驱动 560 无辜）。等不了升级 —— 改用**变体 kernel
    分解**（`tests/gpu/prof_q4_blk.mojo`）：同一分工逐个摘掉一个成分。
  - **分解结果**（rows=1，gate/up；本轮机器态整体偏慢，看相对值）：
    full 76.6µs ≈ f32_acc 77.0µs（**f64 无辜**）；loads_only 64.8µs（大头在加载
    路径）；barrier_only 12.3µs（规约骨架 16%）。
  - **真凶**：不是权重（邻接 byte load 被 L1 兜底），是 **x** —— 每个 CTA 重读
    整行 x，而线程读 `x[kb+j]` 时 warp 内地址间隔 128B，一条指令 32 个散射
    32B 扇区（**放大 ~8×**）。验证臂 `smem_x`：x 行协作载入共享内存（相邻线程
    相邻地址，真合并）后 **76.6 → 34.3µs（2.2×）**，假设成立。
  - **晋升产品**：`q4_matmul_smem_gpu[THREADS, INNER]`（`INNER` comptime 化
    smem 大小；4864×4B=19.5KB 远低于 A100 每 CTA 上限）。基准实测：
    | 形状 | q4_row | blk32 | **smem32** | fp32_row |
    |---|---|---|---|---|
    | 896→4864 | 99.6 µs / 24.8 GB/s | 41.6 / 59.5 | **18.7 µs / 132.0 GB/s** | 89.3 / 195.6 |
    | 4864→896 | 343.6 µs / 7.2 GB/s | 39.0 / 63.5 | **34.8 µs / 71.2 GB/s** | 469.1 / 37.2 |
    全部 diff=0.0。smem32 墙钟比 fp32_row 快 **4.8×**（gate/up）与 **13.5×**
    （down）—— fp32 的 GB/s 更高但它搬 7× 的字节。
  - **三级跳总账**：q4_row（24.8/7.2 GB/s）→ blk32（59.5/63.5）→ smem32
    （**132/71.2**）。模型级：28 层 × (18.7+34.8)µs ≈ **1.50ms/token** 权重
    时间（五十二的 12ms → 1.5ms，**8×**），还没算 attention 与流重叠。
  - **方法论**：profiler 被环境卡死时，"变体分解"（逐成分摘除 + 同构对照臂）
    是同样严格的替代 —— 每个结论都有同机同时段的对照臂背书，不是跨时段
    数字硬比。
  - **下一步**：THREADS=64/两块每线程的下探（down 形状 152 块/32 线程 =
    每线程 4.75 块，负载不均）；rows=32 批形状；attention/QK 差分；把
    DeviceContext 穿进 model 层让这些 kernel 真正接上服务路径。

- **2026-11-09（第五十六条）** —— 📈 **P5 CUDA：THREADS 扫描出形状相关最优（down 128 → 116.7 GB/s）；rows=32 朴素版证伪，批形状要 tiled GEMM**
  - **THREADS 扫描**（smem 版，rows=1，全 diff=0.0）：
    | 形状 | smem32 | smem64 | smem128 |
    |---|---|---|---|
    | 896→4864（28 块/线程组） | **18.8 µs / 131.9 GB/s** | 18.9 / 130.8 | 20.8 / 118.9 |
    | 4864→896（152 块/线程组） | 34.8 / 71.2 | 26.1 / 94.8 | **21.2 µs / 116.7 GB/s** |
    规律符合分工账：gate/up 每线程 28/32≈0.9 块（32 已铺满，128 空转）；down
    152/32=4.75 块（负载串行）→ 128 线程把每线程压到 1.19 块。**取形相关
    THREADS：inner/32 ≤ 32 用 32，否则用 128**（或直接 min(128, nblk 取整)）。
  - **模型级更新**：28 层 × (18.8 + 21.2)µs ≈ **1.12ms/token**（五十二的
    12ms → 1.12ms，**10.7×**）。
  - **rows=32 朴素版证伪**：419.9µs / 7.6 GB/s（gate/up）、577.7µs / 5.5 GB/s
    （down），diff=0.0（`block_idx.y` 维度首次被真实走到且正确）。慢的原因
    是结构性的：一个 CTA 仍只算一个 (row, o)，**权重行不跨行复用** ——
    32 行 = 78MB 权重 L2 流量（≈32× 线性放大，实测 419µs 与 32×18.7=600µs
    的差距就是 L2 在兜底）。**批形状的正确设计是 tiled GEMM**（out×row 分块
    ，权重块载 smem 后被行块内所有 CTA 复用）—— 这是一个独立的设计步，不是
    参数调优；解码的关键路 rows=1 已经够快，批路径留待接入服务层后按真实
    batch 分布再设计。
  - ** Mojo 笔记**：`@parameter for` 遍历 comptime 元组字面量不行
    （`Tuple` 无 `__iter__`），用 `range(lo, hi, step)`。
  - **下一步**：attention/QK 差分（softmax 的 exp/max，最后一块算子空白）；
    DeviceContext 穿 model 层（把 116 GB/s 的 kernel 接上服务路径）；
    tiled GEMM 批形状（按真实 batch 分布设计）。

- **2026-11-09（第五十七条）** —— 🎯 **P5 CUDA：attention 差分入门 —— 第一个非零差分，且恰好 1 ulp；最后一块算子空白补上**
  - **`kernels/cuda/attention.mojo`**：因果 GQA attention 串行版，一个线程算一个
    `(head, t)`，三遍循环（scores → max → exp+sum → V 累加 / total）与标量
    `attention` **逐句同构** —— 同一个 `Float64` 累加位宽、同一个因果掩码
    （`upto = kv_len - q_len + t`）、同一个 GQA 分组（`h // group`，不物化重复
    kv）。没有 `head_lo/head_hi` 分片参数：GPU 的并行单位就是 head×t。
  - **差分结果（A100）**：`diff = 1.1920928955078125e-07 = 恰好 1 个 fp32 ulp`
    （2⁻²³），门通过（容差 1e-5 相对）。**这是差分门第一次出现非零差分**，
    而且变量被完全隔离：kernel 与标量除 exp 外逐句一致，差值只能来自
    **expf 的实现差（libdevice vs libm）**。量级记录：1 ulp，远低于容差。
  - **判据设计生效**：门用"同算法"容差而非"同指令序列"零容差 —— 如果当初
    对 attention 也用零容差，这个门会永远红；而 1 ulp 的证据链（其余全逐位
    + 差值恰好 2⁻²³）反而比"碰巧 0.0"更有信息量。
  - **transformer 层算子齐了**：rmsnorm / linear / q4 / q8 / **attention**
    （softmax 的 exp/max、因果掩码、GQA 分组全在内）。GPU 后端在算子覆盖上
    与 CPU 后端等价；剩下的是性能版（attention 的块内并行规约，串行版就是
    对照臂）与接线。
  - **规模说明**：门用小形状（2 头 : 1 kv 头、head_dim=16、q_len=3、
    kv_len=5 —— q_len<kv_len 同时覆盖 decode 位置公式与掩码）。
  - **下一步**：attention 性能版（一线程一 (head,t) 在 decode 形状
    q_len=1、kv_len≥512 时每个线程串行几百次 exp+dot，需要块内并行）；lm_head
    就是 linear/q4 matmul 已覆盖；DeviceContext 穿 model 层设计笔记。

- **2026-11-09（第五十八条）** —— ⚡ **P5 CUDA：attention 并行版 + 一个真 bug（scores 共享行竞态）—— decode 形状 58×**
  - **`attention_par_gpu[THREADS]`**（`kernels/cuda/attention.mojo`）：一个
    `(head, t)` 一个 CUDA block。顺序变量的账：scores 点积单 j 内 d 升序
    （与标量同序 ⇒ 逐位一致）；max 规约无舍入、顺序自由；**total 树规约是
    唯一新增舍入变量**；V 累加每线程一个 d、j 升序同序（要求
    `head_dim ≤ THREADS`）。门实测与串行版同样恰好 **1 ulp** —— 树规约的
    total 在此数据下没有新增可见舍入。
  - **真 bug：scores 行被所有 head 共享**。标量串行复用同一行（其文档明说
    "a row of scores would do"—— 写→读两遍→下一个头覆盖），GPU 头并行时
    互相覆写。**小形状差分门没逮住**（每 CTA 工作量 ~1µs，CTA 近似串行，
    竞态窗口关着），bench 的 decode 形状（14 头 × kv_len=512）把它揪出来：
    par 版 diff=0.0516（17%）。修复：scores 改为按 (head, t) 私有
    （`n_heads × q_len × kv_len`，行偏移 `(head*q_len+t)*kv_len`）。
  - **教训（进门规）**：差分门的小形状验证不了**并行结构本身** —— 需要
    至少一个"每 CTA 工作量大到 CTA 真正重叠"的形状。decode 形状进了性能
    基准的差分检查，正好补了这个位。另一个诚实的注记：修复前串行版在
    gate 与 bench 都"通过"了同一竞态 —— 机制未查明（疑似 CTA 调度时序的
    良性巧合），修复后问题整体消失，但"竞态碰巧通过"本身就是教训：
    **通过不等于无竞态，只是竞态没被触发**。
  - **decode 性能（14:2 头，hd=64，q_len=1，diff 全部 1.49e-8）**：
    | kv_len | 串行 | par[64] | par[128] |
    |---|---|---|---|
    | 512 | 2387.8 µs | 76.0 µs（31×） | **60.4 µs（40×）** |
    | 2048 | 13219.6 µs | 291.2 µs（45×） | **226.1 µs（58×）** |
  - **模型级**：decode 一步 ≈ matmul 1.12ms + attention 0.23ms + rmsnorm 等
    ≈ **1.4ms/token**。attention 有效带宽只有 9.8 GB/s —— 14 个 block × 128
    线程 = 1792 线程（0.8% 占用率），**并行度被头数封顶**；kv 跨 block 拆分
    （两阶段规约）是下一步的性能杠杆，但 attention 目前只占步时 ~17%，
    不是瓶颈，先不动。
  - **下一步**：DeviceContext 穿 model 层设计笔记（把 kernel 接上服务路径）；
    q4 matmul 正式进差分门（从 bench 的检查升格为门的一项）。

- **2026-11-09（第五十九条）** —— 🚪 **P5 CUDA：差分门第七项 —— q4 matmul 冠军 kernel 升格进门，diff=0.0**
  - **`q4_matmul_diff_check`**：产品冠军 `q4_matmul_smem_gpu[32, 896]` 对
    **组合参照**（`quantize_q4_0` → `dequant_q4_0` 产品量化往返 → 标量
    `linear`）。不用 AVX2 当参照：SIMD 通道序影响舍入顺序，浮点逐位对不上；
    组合参照与 kernel 共享同一套 dequant 语义，差值只面对「融合 matmul 的
    累加」这一件事。bench 里此检查长期 diff=0.0，门沿用同一判据。
  - **首跑就红**（diff=ref_max=2601.8）—— 又是门抓 bug，这次抓的是**门自己
    的调用**：grid 写成 `(out, 1)` 而参照有 2 行，第 1 行从未被算，差分比到
    未初始化内存。补 `grid_dim=(out, rows, 1)` 后 **diff=0.0**。（讽刺但健康：
    五十八说"通过不等于无竞态"，本轮补一句"**失败不一定是 kernel 错，参照
    侧的调用也要过一遍同样的怀疑**"。）
  - **门现状（七项，A100 全绿）**：rmsnorm / linear / attention /
    attention_par / q4_matmul（相对容差 1e-5）+ dequant_q4_0 / dequant_q8_0
    （零容差）。两个非零差分都是 1 ulp（expf）。**GPU 后端的算子正确性记录
    从此有单一入口** —— 后续每加一个 kernel 先进这个门再谈性能。
  - **下一步**：DeviceContext 穿 model 层设计笔记（勘察已完成：分发是编译期
    `[backend]` 参数、QwenForward not thread-safe、并发在 runtime 层孪生体、
    q4 权重 SharedRegion + List[RawPtr] 槽位）。

- **2026-11-09（第六十条）** —— 📐 **P5 接线设计定稿：`docs/plan/06-device-context.md` —— 不是"DeviceContext 穿进 model 层"，是"平行的 GpuSession"**
  - **勘察结论**（code-explorer 全仓勘察 + 亲手核对）：分发是编译期 `[backend]` Int 参数 + `comptime if`（无 trait、无运行期上下文先例）；`run[backend]` 是唯一前向路径、每层中间量全是 host RawPtr 视图；`QwenForward` not thread-safe，并发在 runtime 层孪生体（借原型只读字节）；q4 权重 = SharedRegion + `List[RawPtr]` 槽位。
  - **两条被否决的路**（都有硬理由）：① DeviceContext 进 `QwenForward` 字段 —— 违背"后端是方法参数"哲学且技术上矛盾（原型共享 vs device 资源 per-会话）；② `run[BACKEND_GPU=2]` 硬套 host 视图链 —— 每层两次 PCIe 往返，把 132 GB/s 的 kernel 全淹没。**关键论断：GPU 与 CPU 的数据驻留差异是本质的，不是实现偶然** —— 硬塞同一条 run 链是架构级错误。
  - **设计核心**：`GpuSession`（ctx + device 只读权重副本 + device 常驻激活/KV/logits 缓冲），由 spawn_twin 方构造、生命周期 ≤ 原型；**`QwenForward` 一行不改**，gpu 路径只读它的权重与形状；自由函数 `gpu_step(session, model, ids)` 与 `run[backend]` 平行（不是同链），内部 prefill/decode 仍一条路；每 token 只有 token id 进、logits（~608KB）出，中间零 host 往返。
  - **判据与步骤**（有序、每步可独立验证）：小 kernel 补齐（rope/swiglu/add/gather/kv-append，逐个进门）→ 单层对数 → 全层 decode 端到端 logits 差分（vs AVX2，1e-5）→ 计时 ≤2ms/token → prefill → 批量留白。风险清单：借用时序、异步错误暴露（逐层对数是对冲）、AsyncRT 线程安全（1:1 保守）、q8 KV 留槽位。
  - **下一步（照 §5 步骤 1）**：rope/swiglu/add/gather/kv-append 五个小 kernel + 进门。

- **2026-11-09（第六十一条）** —— 🧩 **P5 接线步骤 1 完成：五个逐元素 kernel 进门，门十一项全绿；两个工程教训（共享 DeviceContext、keep_alive 生命周期标记）**
  - **`kernels/cuda/elementwise.mojo`**（全部一线程一元素平铺、与标量逐句同构）：
    `add_gpu`（残差）、`copy_gpu`（KV append，对应 `copy_into`）、`swiglu_gpu`
    （f64 中间量 + exp 同序）、`rope_gpu`（半旋转，一线程一个
    `(token, head, d)`，q/k 各启动一次，cos/sin 两半分开读）、`gather_gpu`
    （embedding 行拷贝，对应 `run` 里的行复制）。
  - **门十一项全绿（A100）**：rope **零容差站住** —— `a*c - b*s` 这个经典
    FMA 折叠模式**没有被折叠**（linear 的点积模式之外，第二个被验证的表达式
    形态）；swiglu diff=0.0（exp 路径在此数据上逐位一致 —— 与 attention 的
    1 ulp 并不矛盾：expf 的实现差是数据相关的，两个数据点都记录在案）。
  - **坑 1：每 check 各建一个 DeviceContext 会崩**（~10 个上下文处 AsyncRT
    crash，栈在 syscall 上）。整门改成共享一个 ctx（main 创建、check 签名收
    `mut ctx`）后消失。机制未查明，结论按经验规则记：**AsyncRT 上下文要复用，
    不要按检查粒度建**。
  - **坑 2（本场最有价值的）：`arena.keep_alive()` 是生命周期标记，不是
    防御调用**。它是 `pass` 空函数，唯一作用是当 arena 的"最后使用点" ——
    放在 arena 读取**之前**，之后的读取就是 use-after-free。症状极具误导性：
    段错误落在 AsyncRT 栈或比较代码里，两次把嫌疑引向"上下文数量"和"拷贝
    交错"（都是冤案）。定位靠 -debug-level=line-tables 符号化栈 + 指针分离
    探针（cpu0/gpu0 都能读、keep_alive 后才崩）。**进门规：keep_alive 必须
    是函数体里对 arena 的最后一次触碰**。
  - **P5 进度**：`06-device-context.md` §3.4 清单还剩 5/12 → kernel 侧全部
    就位（rmsnorm/linear/q4/attention/rope/swiglu/add/copy/gather/q4 matmul
    + 两个 dequant）。**下一步：GpuSession 骨架（设计 §5 步骤 2）** ——
    构造、权重上传、单层 rmsnorm→q_proj 在 device 跑通并与 `run` 对数。

- **2026-11-09（第六十二条）** —— 🔌 **P5 接线步骤 2 完成：`GpuSession` 骨架落地，单层 rmsnorm→q_proj 与产品通路对数通过（`61c7a06` 之后）**
  - **`src/alofa/gpu/session.mojo`（新模块）**：设计 §3.2 的落点 —— device 权重副本与 `DeviceContext` 的宿主，与孪生体同构（1 session 1 ctx、生命周期 ≤ 原型）。构造即上传（一次同步）：第 `layer` 层 rmsnorm 权重、q_proj 偏置（fp32，`params.view` mmap）+ q 槽位 q4 块流（`q4_blocks[layer*7+0]`，SharedRegion）。计算原语 `enqueue_rmsnorm`（t=1 一行一线程）与 `enqueue_qproj[inner]`（冠军 smem matmul → `add_gpu` 加 bias，与 `project` bias 路同语义）。**`QwenForward` 一行未改**，session 只读它的 `params`/`q4_blocks`/`cfg`。
  - **对数（A100，真实权重）**：模型 `quantize=True` 构造（q4 化 2GB 权重 ~20s）；输入 = 真实 embedding 表第 42 行；CPU 参照 = `scalar.rmsnorm` + **`model.project[BACKEND_SCALAR]`**（run 逐层调用的同一个函数，非自写算术）。结果：**normed diff=0.0（零容差）**；**q diff=9.5e-07，tol=7.9e-04**。非零差有明确出处：参照侧 `project` 走 shards 核（f64 通道累加，RB=2），GPU 走冠军 smem 核（f32 累加）—— **第一次对比两种不同累加序**，差值是合法重组，比门的组合参照（共享 dequant 语义）宽一个数量级。结论：接线正确，"与 run 对数"成立。
  - **Mojo 坑（新）**：结构体方法的 buffer 参数默认**借用**（mut=False），`unsafe_ptr()` 给出只读指针，而 kernel 声明 `Pointer[..., mut=True]` ⇒ 发射时约束失败（错误信息藏在实例化链最底层的 "constraint failed"）。**经 kernel 指针的 DeviceBuffer 参数必须标 `mut`**。另一个小坑：kernel 不能像普通函数一样直接调用，必须 `ctx.enqueue_function[kernel](...)`。
  - **下一步（设计 §5 步骤 3）**：全层 decode 单流 —— 补齐 session 的 k/v/o/gate/up/down 槽位上传与 rope/attention/swiglu/add 原语，`gpu_step` 逐层跑通，端到端 logits 对数（判据 1e-5）。

- **2026-11-09（第六十三条）** —— 🔌 **P5 接线步骤 3 完成：`gpu_step` 端到端，28 层 decode 与产品通路对数通过（1 步 diff=4.8e-6、3 步 1.5e-5，argmax 一致）；三个真 bug + 一条门级教训**
  - **`decode_step[inner, inner_ff]`**：session 全量上传（28×7 q4 槽位 + biases + norms + rope 表 + embedding 544MB + device KV store/激活常驻），t=1 完整前向逐层 rmsnorm→q/k/v→rope→append→attention_par→o→残差→gate/up→swiglu→down→残差→终 norm→lm_head。每 token 只有 token id 进、logits 出。**端到端判据**：对 `model.run`（BACKEND_SCALAR）logits ≤1e-5 相对 —— 1 步 diff=4.77e-6（相对 2.6e-7）、**3 步 1.48e-5（相对 1.2e-6，覆盖 pos>0 rope 行与 kv_len>1 attention）**，argmax 两侧一致。
  - **bug 1：down 块流截断**。上传字节数公式 `hidden × n_out` 对 down 错 —— 块数由**整个矩阵** out×in 决定，down 的 in 是 inter（4864）不是 hidden：只传了 1/5.4 字节，matmul 越界读块流，NaN 一路传进 logits。修：`n_in = inter if slot==DOWN else hidden`。
  - **bug 2：pinned-staging 竞态**。q4 上传循环复用同一块 host 暂存，`enqueue_copy` 异步、host 立刻回填下一槽 —— 在途拷贝读到被覆盖的字节（层 0 探针 S2 q 差 0.77/79 的根因）。修：每次拷贝后同步（一次性上传，代价可忽略）。**教训：复用 host 暂存 ⇒ 每 copy 后 sync**。
  - **bug 3（门级，最值钱）：`max_abs_diff` 的 NaN 假绿**。NaN 的比较恒为 False，`if d > worst` 永不触发 —— **全 NaN 的输出以 diff=0.0 "通过"差分检查**（bug 1 的 logits 全 NaN 曾被判 PASS）。修：`max_abs_diff` 遇 NaN 返回 +inf。**进门规：差分检查必须对 NaN 设防**；这条和五十八的"通过≠无竞态"、五十九的"失败要怀疑参照侧"并列。
  - **aliasing 规则的完整版（账本六十二的坑展开）**：① `mut self` 与 self 派生缓冲不能同时 mutable 传 ⇒ 私有发射函数只收**选择子索引**；② 方法**内部**发射 self 派生指针没问题，方法**外部**混用 `session.ctx`+字段必被拒 ⇒ 诊断原语（`debug_gather`/`debug_read`）长在 session 上；③ 同一 List 的两个下标算同源 ⇒ `d_kv` 拆 `d_k_store`/`d_v_store`；④ `__init__` 半初始化状态调不了 self 的方法 ⇒ 槽位宽度内联、`_upload_*` 改模块级自由函数收 `ctx`。
  - **探针转正**：`test_session_probe.mojo` = 层 0 重放常驻回归（15 个检查点每步对照产品通路）。判据教训：**链式重放的零容差必须算继承差** —— q 的重组差被 rope/store 原样继承（q_rot diff == q diff 恰是 rope 干净的证据）；探针自己的 copy grid 也漏了 ceildiv（半个 store 是垃圾）—— "重放代码与被测代码一样要过同样的怀疑"。
  - **下一步（设计 §5 步骤 4）**：decode step 计时（对照 1.4ms 模型级估计与 2ms 判据）；随后 prefill（步骤 5）。

- **2026-11-09（第六十四条）** —— ⏱️ **P5 步骤 4（decode 计时）：判据未过（5.1ms vs 2ms），但把"为什么"全部查清；顺手修掉一个 100µs 级 kernel 与一个构造期 arena bug**
  - **基准**：`tests/gpu/bench_decode.mojo`（WARMUP→3 轮×100 步取最小；口径 = decode_step 整步，含 token 上传、每步 2 次 sync、logits 回传 —— 设计内的步进成本）。**首轮 10.2ms/token，修完 norm 后 5.1ms**（196 tok/s）。
  - **发现 1（估计错误，最值钱）**：账本五十八的"1.4ms/token"只算了 matmul+attention——**norm 被假设可忽略、lm_head 与每步 2 次 sync/发射开销根本没进模型**。估计错不丢人，拿判据当真前不实测才丢人。
  - **发现 2（真 kernel bug）**：`rmsnorm_gpu` 是"一行一个线程"——basic.mojo 的 docstring 自己写着这是门时代的 deliberate 选择（"性能内核是之后的事"）。decode 时 grid(1)×block(32)，实际单线程串行读 896 个 f64 ≈ **100µs/次，57 次 norm = 5.8ms/token**。修：一行一个 block、256 线程分块 + 共享内存树形归约。门里 rmsnorm 检查本就是相对容差（1e-5），树形归约的合法重组差直接覆盖，判据不用动。**10.2 → 5.1ms**；端到端对数复验仍过（3 步 diff=1.48e-5，argmax 一致），门 11 项全绿。
  - **构造期 bug（bench 顺手抓的）**：`act_arena` 容量公式没算 `Arena.alloc` 的 64 字节对齐 padding——max_tokens 是 64 倍数时所有 alloc 天然对齐（padding=0）从不暴露；max_tokens=313 时 padding(180) 超容量 60 字节，**构造即崩**。两处构造分支各补 1024 字节余量。
  - **分段方法论**（`bench_decode_sections.mojo`，可复用）：A 整层链 / B lm_head / C token 路径（0.01ms）/ D logits d2h（0.03ms）/ E 单 kernel / F,G 大小分组。结论数字：大 5 matmul ≈74µs/层、小发射组 ≈38µs/层（修 norm 前 F 曾是 244µs）、**lm_head 1.14ms**（544MB 读 = 3× 内存下限）、发射开销 2.3µs（bench_launch）～4µs（异构交替/轮转，bench_launch_alt）。
  - **aliasing 新规（比六十三更细）**：循环外取的 `List` 元素内部引用（`unsafe_ptr()`），循环体内任何 `session` 可变借用都使其失效——**内部引用必须紧贴使用点提取**。
  - **到 2ms 的剩余路径**（下一步）：① 大 matmul 74µs/层 vs 内存下限 ~4µs（q4 块流 5.8MB/层 ÷ 1.5TB/s）——19× 差距是最大头；② lm_head 1.14ms → 向 0.4ms（向量化/f32 通道）；③ 发射数 20/层 → 融合（bias 进 matmul、rope q+k 合一、KV append 合一、残差进 matmul epilogue）≈ 2ms 里 launch 占 ~2ms。
  - **下一步（步骤 4 续）**：kernel 级优化三连，目标 2ms 判据。

- **2026-11-09（第六十五条）** —— ⚡ **P5 步骤 4 续（kernel 优化第一轮）：5.1 → 4.26ms/token（235 tok/s）；融合三件套进门（门 15 项）；rope2 抄错公式被差分抓出**
  - **THREADS 重扫非杠杆**：三真实形状 × {32,64,128,256}（`prof_mm_threads.mojo`），最优组合只快 ~5%（q T128 12.0µs、gate T32 31.9、down T128 22.8）——五十六的配置没过时，瓶颈在结构不在线程数。
  - **svec 负结果**：smem x + `load[16]` 权重的组合变体比冠军慢 1.1-2×（gate 39.6 vs 31.9µs；down 43.9 vs 21.1）——`@parameter` 展开 16 nibble 的寄存器压力压过了合并收益。撤销。负结果也记账。
  - **融合三件套（进门，门 11→15）**：① `q4_matmul_fused_gpu`（bias 挪进写回尾缀，`Float32(acc)+bias[o]` 与 CPU project 顺序逐句一致 → diff=0.0；省 3 发射/层）；② `rope2_gpu`（q/k 合一）；③ `copy2_gpu`（KV append 合一）。每层 21→17 发射。
  - **rope2 公式 bug（门抓的）**：初版第二半写回用 `table + d`——标量与 rope_gpu 都是**表的两半分开读**（`table + d + half`）。定位过程自身贡献了新教训：我的 standalone 隔离 harness **忘了把数据传进 host 缓冲**，全零输入下"写入全零"与"没写"不可区分，白绕两轮——**差分 harness 的输入必须非零**，与"全零输入让相对容差退化"是同一条纪律的另一半。
  - **`linear_block_gpu`（lm_head）**：块内合并读（warp 32 线程各读不同 w 行零合并 → 一行一 block 256 线程分条）。实测只拿到 1.14→1.02ms（533GB/s，远未到 0.4ms 下限）——合并读只解决一半，待查（TLB/行缓冲/f64 转换）。
  - **aliasing 新坑**：同一指针在同一次发射里出现两次（哪怕语义上是"不读的占位"）必被拒；占位得传一个真正不同的缓冲（d_ws）。
  - **复验全绿**：门 15/15；decode 对数 diff=1.48e-5、argmax 287 两侧一致（fused 尾缀位级等价的端到端证据）；layer0 normed 0.0 / q 9.5e-7；probe 全净。
  - **到 2ms 剩余**：大 matmul ~60µs/层 vs 内存下限 4µs（仍 15×，需要真正的 GEMV 重构）；lm_head 1.02ms（待查读放大）；attention 0.23ms。下一步：matmul 结构性重写或接受现状修订判据。

- **2026-11-09（第六十六条）** —— ⚡ **P5 步骤 4 续（kernel 优化第二轮）：4.26 → 3.95ms/token（253 tok/s）；fat-CTA 三处接入；同 List 双下标 aliasing 的新形态**
  - **fat-CTA 扫型**（`prof_fat.mojo`：OUTS 输出/CTA 共享 x smem，`[OUTS][THREADS]` 归约布局每轮一次 barrier）：gate **-14%**（T64/O4 23.0 vs 26.9µs）、down **-12%**（T128/O2 26.9）、lm_head **-8%**（T256/O8 0.459 vs 0.501ms）；q 反而变差（输出太少，CTA 数降太多伤占据率）—— 按形状择优，q/k/v 留冠军。
  - **gate/up dual-fat 融合**（新 kernel `q4_matmul_dual_fat_gpu`）：两者读同一 normed 行，一个 CTA 出 gate 4 个 + up 4 个输出，x 相位再摊一倍；每输出累加序与冠军逐位同构，门项 diff=0.0（门 15→16）。
  - **aliasing 新形态**：同一 List 的两个下标（d_q4[gate]、d_q4[up]）**提取局部变量也救不了** —— origin 记在 List 字段上，两指针同源被拒。解法：init 时把 gate/up 句柄零拷贝分流到两个独立 List（d_q4_gate/d_q4_up），不同字段不同 origin —— 与六十三 d_k_store/d_v_store 的可用形态同族。
  - **mut=False 临时量坑再现**：`self._src(2).unsafe_ptr()` 内联进发射参数是 mut=False（六十二同款），存 `var x = self._src(2)` 再取即过。
  - **复验全绿**：门 16/16；decode 对数 diff=1.478e-5 与改前**逐位一致**（argmax 287 两侧同）；layer0/probe 净。
  - **到 2ms 剩余**：大 matmul 仍 ~10× 内存下限（fat 只啃了 12-14%，结构性上限还在：18B 块的字节加载链）；lm_head ~0.46ms；attention 0.23ms。下一轮候选：q/k/v 三融合（同 List 问题同解法）、权重 f16 预反量化（大改，需重立数值判据）、或修订判据。

- **2026-11-09（第六十七条）** —— ⚡ **P5 步骤 4 续（kernel 优化第三轮）：3.95 → 3.73ms/token（268 tok/s）；q/k/v 三融合进门（门 17 项）；向量化 nibble 路线二次证死**
  - **al（对齐重排）负结果**：装载期把 q4 块重排为 32B（nibble 16B 对齐 + scale 跟后），kernel 用一次 `load[16]` 替代 18 次标量字节加载 —— **全面更慢**（gate 77µs vs 冠军 31.9；down 41.6 vs 24.2）。与 svec 合并看：**不是对齐问题，是 SIMD nibble 展开路径本身慢**（寄存器压力 + 提取开销；编译器对标量字节循环的调度本来就很好）。撤销，不留死代码。向量化 nibble 路线两次证死，此路封。
  - **q/k/v 三融合**（`q4_matmul_qkv_gpu`）：三者读同一 normed 行，一个 CTA 出 q/k/v 各一个输出（k/v 输出数小于 q 时尾部 CTA 部分激活），bias 尾缀 f32 加。三分支显式展开（不用指针三元）。门项 diff=0.0×3（门 16→17）；3 发射→1，层链省 ~0.22ms。
  - **aliasing 同族扩展**：`d_biases` 三个下标同病同治 —— 分流 d_bias_q/k/v 三个独立 List。规则至此完整：**一次发射里每个 origin 只能出现一次；同 List 多下标、提取局部变量都救不了，必须按字段分流**。
  - **复验全绿**：decode 对数 diff=1.478e-5 与改前逐位一致（argmax 287 两侧同）；layer0/probe 净。
  - **到 2ms 剩余**：大 matmul ~8× 内存下限（向量化证死后，剩余路径只有 f16 预反量化 —— 大改且需重立数值判据 —— 或修订判据）；lm_head 0.46ms（下限 0.36，余量 28%）；attention 0.23ms。

- **2026-11-09（第六十八条）** —— ⚡ **P5 步骤 4 续（kernel 优化第四轮）：3.73 → 3.258ms/token（306.9 tok/s，-12.6%）；f16 预反量化全链进门（门 19 项）；混合行尾静默失配教了一大课**
  - **f16 预反量化路线落地**：装载期一个 kernel（`q4_dequant_f16_gpu`）把显存里的 q4 块流直接反量化成 f16 副本（每层 7 槽 `d_w16_*` + lm_head 的 `d_embed16`），host 不再中转（host pinned 回传路径在第 15 次 copy 处 `CUDA_ERROR_INVALID_VALUE`，隔离探针无法复现，GPU 侧转换整体绕开）。`decode_step` 五处发射全切 f16：q/k/v 三融合 `f16_matmul_qkv_gpu`、gate/up dual `f16_matmul_dual_gpu`、o/down fat `f16_matmul_fat_gpu`、head `linear_fat_f16_gpu`。q4 本体保留 —— qproj/probe 探针路径维持 q4 精确对数，`_mm`/`_mm_bias` 因此留任。
  - **数值判据重立**（f16 舍入 rel err ≤ 2^-11 ≈ 4.9e-4，24 层链路放大）：kernel 门 `F16_REL=0.05`（实测单 matmul rel 3-8e-5，30 倍以上余量）；decode 端到端从 1e-5 级改为 `F16_DECODE_REL=0.003` 相对（实测 rel=6.99e-4，4.3 倍余量）+ **argmax 必须与 CPU 参照一致**的功能判据。门 17→19（f16_fat、f16 dual/tri/head 组合各一项）。decode diff=0.00868、argmax 287 两侧一致。
  - **本轮最大的坑：混合行尾 + 无 assert 的文本替换 = 静默失配**。文件里 \r\n 与 \n 区段并存，python `str.replace` 按 \n 匹配在 \r\n 区段静默不中（无 assert 无从知晓）。后果：终 norm 的 append 被误塞进层循环（每层多 append 一个 FINAL_NORM，后续层的 post-attn norm 全读到终 norm 权重），decode diff=12.37≈ref_max；而"修复"替换同样静默没打上，假象横跨 q4/f16 所有 kernel 组合，把二分误导了两轮（一度怀疑 o 槽 f16 与整个 f16 构建）。**进门规：python 批量编辑要么 assert 匹配数，要么行级操作；对文件做过行级手术的区域一律先 cat -A 验行尾**。
  - **其余坑在案**：f16 sweep 的初版数字全在 `k=tid` 覆盖 bug 下测的（门重写后重测定标）；down 发射 block_dim(128) 配 [256] 模板 → inf（模板与块维必须成对）；f16 组合门忘 `arena.keep_alive()` → head 段崩在 AsyncRT 栈（六十一同款，进门规已记：keep_alive 必须是最后触碰）。
  - **代码清理**：bisect 临时件（d_q4_gate_bisect/up_bisect）、六十七的 q4 句柄分流 List（d_q4_q/k/v/gate/up，f16 后 decode_step 不再需要）、f16 构建循环的 hq4 死重填，均已撤销；`prof_f16.mojo` 留作 sweep 证据。
  - **复验全绿**：门 19/19；decode PASS（argmax 287 两侧同）；layer0 normed 0.0 / q 9.5e-7；probe 全净。
  - **到 2ms 剩余**：3.258 → 2 差 1.26ms。f16 后 gate matmul 有效带宽 ~462GB/s（8.7MB/2.6µs 下限），离 3.3TB/s 峰值仍 7×——瓶颈已从解包算力转到占据率/发射结构；`bench_decode_sections` 的分段数字还是 q4 时代口径（未更新 f16），下轮先校准再定位。下一轮候选：f16 matmul 吞吐结构（OUTS/THREADS 重扫 + fused 尾缀）、attention 0.23ms、或 lm_head 单独压。

- **2026-11-09（第六十九条）** —— 📏 **P5 步骤 4 续（f16 口径校准轮）：bench_decode_sections 校准 + prof_f16 对称重扫；lm_head 假瓶颈证伪（1.02→0.34ms）；3.26ms 的预算表立起来**
  - **prof_f16 重写为 session 核口径**：六十八的探针不对称（f16 臂用单矩阵 fat、q4 基线用 dual），本轮全部换成 session 在用的核同窗对比。结果：gate/up **f16 dual T64/O4 = 35.5µs（492GB/s）** vs 同窗 q4 dual 49.3µs；down **f16 T128/O2 = 15.6µs（560GB/s）**（session 原 T256/O2 15.8 → 切换，模型级 ±0.8% 在噪声地板）；o 6.7µs（240GB/s）；**head T256/O8 = 334µs（817GB/s，全模型最高带宽）**。session 配置除 down 外已是冠军，无进一步调参空间。
  - **lm_head 0.46ms“余量 28%”是假议题**：bench_decode_sections 的 B 段一直用六十四的 `linear_gpu`（一线程一输出，533GB/s 时代的过渡品）——session 实际跑的 `linear_fat_f16_gpu` 实测 **0.344ms（817GB/s，与 prof 一致）**。旧口径的 1.02ms 假瓶颈悬案（账本六十五“2× 未解释，待查”）就此了结：不是 kernel 慢，是 bench 测错了核。
  - **3.26ms 预算表**（f16 口径首立）：层链 kernel 和 ~86µs/层 ×24 ≈ 2.06ms + head 0.344 + token/d2h 0.045 + sync ≈ **2.45ms 实做 vs 3.26 实测 ⇒ ~0.8ms 是 288 次/步发射的间隙开销**（12 发射/层 ×24 + 头部）。下一个杠杆是**发射数/派发结构**（fuse rope2+copy2、swiglu 进 dual 尾缀、残差进 matmul 尾缀），不是再调 kernel 形参。
  - **新假错形态在案**：一次发射的实例化失败（head kernel 传了 5 参，签名只收 4）让 main 里**所有** enqueue 以 "no matching method in call to 'enqueue_function'" 报错——真错误 "Wrong number of arguments to enqueue" 埋在 note 链最深处。与账本六十一的 keep_alive 段错误同族：**Mojo 编译器的错误归因会跨语句级联，第一个错误未必是第一个报错**。程序性防御：新文件先发一个最简 launch 编译过再加臂。
  - **bench_decode_sections 半更新**：B 段换 f16 并校准；A/E/F/G 仍为 q4 kernel 口径（E 已标注 "(q4)"，f16 单核数字以 prof_f16 为准）——完整镜像 decode_step 的 12 发射层链留待下轮与发射融合一起做。
  - **复验全绿**：门 19/19；decode PASS（argmax 287 两侧同）；layer0/probe 净；bench 3.283ms/token（与六十八的 3.258 同噪声带）。
  - **到 2ms 剩余**：~0.8ms 发射间隙 + attention 0.23ms（kv 长度增长会涨）+ dual 492GB/s 距峰值的结构性差。下一轮候选：发射融合三件套（目标 12→8 发射/层）或 CUDA graph 类批量派发。

- **2026-11-09（第七十条）** —— ⚡ **P5 步骤 4 续（发射融合轮）：3.258 → 3.046ms/token（328 tok/s，-6.5%）；层发射 12 → 8；判据零松动**
  - **融合三件套**（按六十九预算表打 ~0.8ms 发射间隙）：
    - **`rope_kv_gpu`**（rope2+copy2 合一）：decode 里 `d_k_rot` 只喂 copy2 —— rope 后的 k **直写本 token 的 KV store 行**，中转缓冲整个省掉；三段平铺（q rope / k rope / v 拷贝）输出互不重叠无竞态。零容差门（+1 → 门 20）。
    - **dual + swiglu 尾缀**（`HAS_ACT`）：tid==0 写回 gate/up 后顺带算 `act = silu(gate)*up`，与 `swiglu_gpu` 同序（f64 中间量、一次 Float32 收尾）—— 门实测 **rel=0.0 精确同序**。
    - **fat + ACC 累加模式**（`ACC: Bool = False` 默认参，旧调用点零改动）：o/down 直接累加进 `d_x`，省两次 `add_into` 发射；舍入顺序与"写 proj 再 add_into"逐位一致（decode diff 0.00868 与 argmax 287 端到端逐位不变是等价性的最终证据）。
  - **坑在案**：dual 加 `dst_act` 形参后，session 原 bias 占位 `d_act` 与 act 输出撞同一缓冲 —— aliasing 检查要求**占位指针也两两不同**，bias_b 换 `d_proj` 占位。默认 comptime 参（`ACC: Bool = False`）在设备核模板里工作正常，9 个旧调用点零改动。
  - **预算表兑现**：预期 ~96 发射 × ~2.8µs ≈ 0.27ms，实测 -0.21ms —— 发射间隙模型得到验证。剩余间隙 ~0.65ms（192 发射 × ~3.4µs）。8 发射/层 = norm、tri、rope_kv、attention、o+ACC、norm、dual+act、down+ACC。
  - **复验全绿**：门 20/20（+rope_kv 零容差；fat_acc rel 3.0e-5、dual_act rel 0.0）；decode PASS（argmax 287 两侧同）；layer0 normed 0.0；probe 净；prof/sections 健全。
  - **到 2ms 剩余**：3.046 → 2 差 1.05ms。构成：发射间隙 ~0.65ms（再融合空间：norm 进 dual/tri 尾缀？attention 后处理？）、attention ~0.23ms、kernel 本体 ~2.4ms（dual 492GB/s 距峰值仍 4×）。下一轮候选：norm 融合进 matmul 头部（读 x 前先归一化，省 2 发射/层）或 CUDA graph 类批量派发。

- **2026-11-09（第七十一条）** —— 🔄 **P5 步骤 4 续（norm 就地融合轮）：3.046 → 3.013ms/token（-1%）；head-NORM 融合实测净亏被对照实验否决**
  - **tri/dual + NORM 就地归一化**（`NORM: Bool = False` 默认参 + `rms_w/eps` 参数）：三个 smem matmul（tri/dual/head）反正每块都载整行 x —— 载入后就地 rmsnorm（f64 累加 + 树形归约同 `rmsnorm_gpu`，分块顺序随 THREADS 不同 = 合法重组差）。层链前两发直接吃残差流 `d_x`，2 norm 发射/层省掉（48/步）。
  - **head-NORM 融合被否决（对照实验）**：head 是 18688 块 × 256 线程，每块冗余归约 896 元素 + 9 道屏障 —— 实测比 norm 单发**更慢**（3.094 vs 3.013）。教训：**融合收益 = 省的发射间隙 − 冗余计算 × 块数**，块数多的 kernel 不适用。头部保持分离发射。
  - **收益小于模型的原因在案**：预算模型预测 48 发射 × ~3.4µs ≈ 0.16ms，实测净赚仅 ~0.03ms —— 融合核自身增加 ~2.7µs/发（屏障链 + 冗余 f64 归约），且被替掉的 norm 是 grid=1 块的小发射（间隙可能低于均值）。发射间隙不是常数，"每发 ~3µs" 是粗均值，别外推。
  - **数值等价的强证据**：decode parity diff **0.008679389953613281 与融合前逐位相同**、argmax 287 两侧同 —— 融合 norm 在 decode 全链路上 bit-exact（T32/T64 的树形归约与 rmsnorm 的 T256 恰好同序或差被 f32 收尾吸收）。dual-NORM 门 rel 2.3e-4（容差内）。
  - **复验全绿**：门 20/20（+dual_norm 判据）；decode PASS；layer0 normed 零容差仍保（session 原语路径未动）；probe 净；sections/prof 健全。
  - **到 2ms 剩余**：3.013 → 2 差 1.01ms。norm 融合这条路已收割完毕。下一轮候选：attention 0.23ms（kv 长度增长会涨）或发射间隙的根治（批量派发/graph）；dual 492GB/s 距峰值 4× 的结构性差仍在。

- **2026-11-09（第七十二条）** —— ⚠️ **P5 步骤 4 续（负结果轮）：占据率假设被链上实测证伪；微基准跨迭代流水假象现形；三个方案全回退，3.013 冠军配置不动**
  - **假设**：dual 492GB/s vs head 817GB/s 的结构性差是占据率（T64/O4 = 1216 块只装 1/3 线程槽）。扫 O2/O1：微基准里 T32/O1 "21.7µs/806GB/s"（1.67×）——**链上实测反而慢 0.3ms（3.01 → 3.31）**。
  - **假象机理在案**：微基准同一 kernel 连发 30 次互相无依赖，块调度器让相邻迭代的块重叠填满 SM —— 波量化（4864 块 = 1.4 波，A100 每波上限 32 块/SM × 108 SM = 3456 块，第二波只剩 1408 块）被完全掩盖。**选型判据必须在真链上（或带依赖的微基准）复测** —— 这是本轮最重要的方法论产出。
  - **scale 拆分方案也净亏**（3.126）：`rms_scale_gpu`（1 块 × 256 线程，归约序与 rmsnorm_gpu 全同 → 系数逐位同）+ matmul `HAS_SCALE` 读 f64 标量（与分离 rmsnorm 通路 bit-exact）——数学上干净，但多出的 49 次 grid=1 小发射（~2.5µs 间隙/次）恰好吃掉省下的块内冗余归一化。**又一次验证：发射间隙 ~2-3µs 是小 kernel 方案的硬税**。
  - **处置**：三方案全部回退（`git checkout`），session 保持七十一冠军配置（NORM 就地 tri/dual、dual T64/O4、head 分离 norm）；基线复验 3.0145ms/token；门 20/20、decode diff/argmax 逐位不变、layer0/probe/sections 全绿。prof 留存 O2/O1 臂并加"微基准流水污染"警示注释。
  - **顺带修了 HEAD 上的两处带伤**：prof 的三个 dual 旧臂缺参（helper 11 参臂传 10）+ head helper 两处发射传 4 参 —— **七十一的 "prof 16 us" 检查是 grep 松模式假阳性**（数了错误文本里的 "us"）。教训：冒烟检查必须数真实结果行（"us  bw"），不是松子串。
  - **到 2ms 剩余**：1.01ms 不变。下一轮候选：attention 0.23ms（读链上形，勿信微基准）或 CPU 派发结构。

- **2026-11-10（第七十三条）** —— ⚡ **P5 步骤 4 续（attention+rope 融合轮）：3.013 → 2.966ms/token（337 tok/s）；层发射 8 → 7；判据 +1 → 门 21**
  - **`attn_rope_gpu`**（rope_kv 并进 attention 头部，decode q_len=1 专用）：每块 rope 自己的 q；k/v 段**每块算自己 kv_head 的 smem 副本、仅组内首块（`head % group == 0`）写 store 行**。跨块竞态的解法在案：同一 launch 内读兄弟块刚写的 store 行没有可见性保证 —— 当前 token（`j == upto`）一律读 smem 副本（与 store 行值逐位相同，rope 同式同序），`j < upto` 读 store。
  - **数值**：rope 段零容差门（q_rot/store 写回逐位）；attention rel 0.0（L=5）/ 7.8e-8（L=40，跨 THREADS 步距覆盖循环补算路径）。decode diff 0.00868 与 argmax 287 **逐位不变** = 端到端 bit-exact 铁证。
  - **收益小于模型的原因在案**：预期 24 发射 × ~3.3µs ≈ 0.08ms，实测 -0.047ms —— 融合后的 attention 头部多了 rope 段（14 块 × 串行小相位），且被替掉的 rope_kv 是 64 线程小发射（间隙低于均值）。方向对、单价折扣。
  - **坑在案**：门的 store 读回 `enqueue_copy` 只拷了前 kv_cols 元素（第 0 行）却比对第 L-1 行 —— 读回尺寸必须覆盖要比对的行；HostBuffer 下标不支持 `unsafe_offset=`（DeviceBuffer/指针才支持）。
  - **复验全绿**：门 21/21（+ATTENTION_ROPE）；decode PASS（argmax 287 两侧同）；layer0 零容差 normed 仍保；probe 净；prof（21 结果行）/sections（61）健全。
  - **到 2ms 剩余**：0.97ms。构成：发射间隙 ~0.6ms（168 发射 × ~3.5µs，再融合空间见顶：rope_kv 已进 attention，norm 已进 tri/dual，ACC 已吃 residual）、attention 本体 ~0.2ms（14 块延迟主导）、kernel 本体 ~2.3ms（dual 480GB/s 是最大单点）。下一轮候选：attention 相位数（在线 softmax 5 相位→3）或 dual 的 ILP/访存结构。

- **2026-11-10（第七十四条）** —— ⚡ **P5 步骤 4 续（f64 链重排轮）：2.966 → 2.947ms/token（339 tok/s，-0.6%）；微基准假象第三次现形（反向）**
  - **结构分析**：dual 480GB/s 的真因不是占据率（七十二已排除）而是**访存延迟**——旧内层 8 条 (half,oo) 链顺序执行，每链 while 内同时只 1 个 16B load 在飞（~16B/线程），而 latency hiding 需要 ~11KB/SM，差 30×。
  - **刀法**：k 外层、行内层重排——每步 8 个 load 背靠背 + f64 链交错（InlineArray 累加器）。**每行链内求和序逐一同旧（k 升序 j 升序）⇒ bit-exact**（decode diff 0.00868 与 argmax 287 逐位不变，门 21/21 全绿）。
  - **四核分化实测**（微基准）：dual 36.4→27.2µs（641GB/s，+34%）；**down（fat INNER=4864）14.2→20.3µs 净亏**——长循环里 ptxas 本就能在单链内流水，重排的每步 prologue 反成税；o（同核 INNER=896）6.1→5.76 略好；head -7%。→ **保 dual+tri，fat/head 回退旧结构**。
  - **真链隔离（A/B）**：dual-only 2.959（-0.007），dual+tri 2.947（-0.015）。**微基准承诺 0.22ms、真链只给 0.015ms** —— 第三次微基准假象，这次反向：新结构给相邻迭代留了重叠余地，连发计时把跨迭代重叠算进了"kernel 变快"。七十二的教训再次强化：**只有真链 bench 是判据**。
  - **裁决**：-0.6% 小赢且 bit-exact、复杂度可控（dual/tri 两处），保留；down 的回归若留着会净亏 0.1ms+（已回退验证）。InlineArray 无需 import（Mojo prelude 内置）。
  - **复验全绿**：门 21/21、decode PASS、layer0 零容差、probe 净、prof（21 行）/sections（61）健全。
  - **到 2ms 剩余**：0.95ms。kernel 本体 ~2.3ms（dual 真链时间未变——微基准的 27µs 是假的，真链 dual 仍 ~36µs）、间隙 ~0.6、attention ~0.2。下一轮候选：真链分段时间基建（sections 的 f16 口径化）或 CPU 派发结构。

- **2026-11-10（第七十五条）** —— ⚠️ **P5 步骤 4 续（测量方法论轮）：零性能变化；一个"快 0.39ms 的冠军"被证伪；交错 A/B 基建落地**
  - **起因**：七十四再次暴露微基准盲区 → 建真链形状扫（bench_dual_shape，每形状独立会话同参）。扫型示 **T128/O8 = 2.572 vs 默认 T64/O4 = 2.964（-0.39ms，-13%）**，且两次扫型复现、同参对齐 bench_decode —— 看起来铁证。
  - **证伪**：切换默认后 standalone bench_decode 却是 2.974×3 纹丝不动。矛盾 → **逐 token 交错 A/B**（bench_dual_ab：同会话两形状交替调用，decode_step 内建每步 sync 使逐调用 monotonic 计时精确）—— **两形状逐位打平**（round 0: 2.9575 vs 2.9526；round 1 同漂 3.368/3.364）。顺序扫型的窗口落在本机（共享 GPU，他租户负载随时段波动，窗口间漂移 ±0.4ms+）的不同负载时段 —— 排名是漂移假象。**教训：共享 GPU 上，唯一可信的对比是同窗口交错；任何跨 run 的 Δ<0.4ms 不可采信**。
  - **波及审计**：七十四的 "-0.6%"（2.966→2.947）也是跨 run 顺序测量 —— **落在该噪声带内，性能声明撤回**；重排代码保留（bit-exact 无害，decode diff/argmax 逐位不变）。六十九以来的全部微基准选型数字同理只能当参考。
  - **顺带发现**：MAXT ≳ 400 的会话在 AsyncRT 崩溃（368 可跑、608/958 崩）—— 独立 bug 在案，未修。
  - **基建**：`bench_dual_ab.mojo`（交错 A/B，可复用的形状/变体裁决工具）；`bench_dual_shape.mojo` 留作方法论陷阱存证（头部警示）；`decode_step` 的 dual_t/dual_o comptime 参保留（默认回 64/4，调用点零改动）。
  - **复验**：门 21/21、argmax 287 逐位不变、bench_decode 2.961（与基线同窗漂移带内）。
  - **到 2ms 剩余**：~0.96ms 不变。下一轮候选：真 CPU 派发结构（发射间隙 ~0.6ms 是现在唯一没被诚实测量过的池子 —— 用交错法先量）或 attention 相位数（同法）。

- **2026-11-10（第七十六条）** —— 📊 **P5 步骤 4 续（链上分解轮）：零性能变化；发射间隙估算被砍半；tri 138GB/s 现形为最大杠杆**
  - **基建**：`decode_step` 加 `skip` 运行时掩码（7 个 SKIP_* 位，默认 0 = 与原路径逐指令相同）+ `noop_gpu`（elementwise，1 块 32 线程零工作）+ `bench_decode_span`（八模式逐 token 交错：full / 单跳×6 / skeleton）。七十五的交错方法论直接复用；总步数压在 MAXT<368 安全区。
  - **实测分解**（两轮×两run，同窗口交错差值）：skeleton 地板 **0.31ms**（123 发射 × ~2.3µs 主机 enqueue + 2 sync + H2D/D2H，两轮两次 run 稳定）；tri **0.44**（18.3µs/层）/ attn **0.4-0.66**（r1 的 no-attn 恒 2.13 疑落负载低谷，取区间）/ o **0.15**（5-7µs ✓ prof）/ dual **0.70**（27-33µs ✓）/ down **0.40**（14-20µs ✓ prof）/ head+norm **0.35**（✓ prof）。和 ≈ 2.9 ≈ full ✓ 自洽。
  - **修正**：七十三以来"发射间隙 ~0.6ms"的估算**砍半以上** —— 纯空闲间隙池只有 ~0.1ms 量级（0.31 地板大半与 kernel 时间重叠），**减发射/融合路线接近见顶**。剩余路线必须是 kernel 本体。
  - **BW 表（真链口径，本轮最重要产出）**：tri **138GB/s**（2.52MB/18.3µs，最低效大户——896 块 × 32 线程只装 13% 线程槽 + 3 链顺序，正是 dual 重排前的病）< o 268 < down ~512 < dual ~550 < head **755**（272MB/0.36ms，接近流读上限）。**tri 提到 ~500GB/s ≈ 省 0.3ms**，是下一个最大杠杆；head 余量仅 ~0.1ms。
  - **坑在案（两个回归网都接住了）**：① python 脚本批量 wrap 丢基准缩进 —— head 块滑进 `for layer` 体内，norm+head 每步跑 24 次（+8.7ms，bench_decode 2.96→10.4 当场抓住）；② `replace(count=1)` 命中文件里第一处同文 —— gather 包装落进 `debug_gather`（sections 编译错误 `skip` 未定义抓住）。**批量文本操作后必须全量回归，缩进敏感的编辑用精确工具**。
  - **复验全绿**：门 21/21、argmax 287 逐位不变、layer0 零容差、probe COMPLETE、prof 21 行、sections 61、bench_decode 2.960（基线带内）。
  - **到 2ms 剩余**：~0.96ms。构成修正：kernel 本体 ~2.6（tri 0.44 + dual 0.70 + down 0.40 + head 0.35 + attn 0.5 + o 0.15）+ 地板 0.31。下一轮：tri 真链形状/结构交错扫（decode_step 加 tri_t comptime 参同法）。

- **2026-11-10（第七十七条）** —— ⚡ **P5 步骤 4 续（tri 占据率轮）：2.960 → 2.778ms/token（360 tok/s，-0.18ms/-6.2%）；七十六的 BW 表直接兑现**
  - **刀法**：`tri_t` comptime 参（默认原 32）+ `bench_tri_ab` 交错 A/B（T32/T128/T256 三形状同会话逐 token 交替，PAIRS=38×ROUNDS=3 压在 MAXT<368 崩线下）。**三轮同序**：T128 一致省 0.18-0.20ms（T256 中间省 ~0.09）——同窗交错 + standalone 复跑（2.778/2.780）双确认，本次声明落在方法论内。
  - **机理**：七十六 BW 表命中——tri 896 块 × 32 线程 = 28672 线程只装 13% 线程槽（108 SM × 2048）；T128 → 每块 1024 线程、2 块/SM 打满 2048 槽，tri 18.3 → ~10.3µs/层 ≈ **245GB/s**。与 dual 重排轮"重排救不了"对照：tri 的病是占据率不是链内 MLP，**换 T 一刀就中**。
  - **数值**：tri 是 NORM 融合核，跨线程归约序随 T 变——门 21/21 仍全绿、decode argmax 287 逐位不变、layer0 零容差 normed diff=**0.0** 仍保（x 归约序实际未变）。
  - **坑在案**：bench_tri_ab 初版 PAIRS=60×3 形状×3 轮 = 552 步 > 368 崩线 —— **多模式交错 bench 的步数预算 = WARMUP + 模式数×PAIRS×ROUNDS**，模式翻倍步数翻倍。
  - **到 2ms 剩余**：0.78ms。更新后的池子：tri 0.25（245GB/s 仍低于 dual/down 的 500+，T256/双列结构可再试）+ dual 0.70（600GB/s，真链口径见顶）+ down 0.40（512GB/s）+ head 0.35（755GB/s 近饱和）+ attn 0.5（相位候选）+ o 0.15 + 地板 0.31。下一轮候选：attention 相位数（5→3，交错法量）或 down/dual 的 O 结构真链扫（交错法）。
  - **复验全绿**：门 21/21、argmax 287 逐位、layer0 normed 0.0、probe COMPLETE、prof 21 行、sections 61、bench_decode 2.778/2.780。

- **2026-11-10（第七十八条）** —— 📋 **P5 步骤 4 续（形状维度收官轮）：零性能变化；fat/dual 形状空间交错法全部确认最优**
  - **两轮交错 A/B**（bench_fat_ab 四模式 / bench_dual_o_ab 三模式，同会话逐 token 交替，步数按七十七公式压崩线下）：
    - **dual T128：三轮一致 +0.15-0.17 拒绝**；O2 +0.12-0.14、O8 +0.17-0.19 同拒 —— **T64/O4 保持**。
    - **down T256 / o T256：噪声内（0-0.06）无意义** —— **T128 保持**。
  - **结论（重要）**：tri 是唯一有占据率病的核（13% 槽，七十七已修）；fat/dual 在 26-35% 槽下已到 BW 平台（512-600GB/s）——**占据率不是它们的限制因素**（更下游：DRAM 队列/L2 分片），加大 T/O 不再兑现。**形状/块结构维度正式关闭**。
  - **沉淀**：decode_step 加 down_t/o_t comptime 参（默认 128，与原硬编码一致，调用点零改动）；两个 A/B bench 留作复裁工具。
  - **到 2ms 剩余**：0.78ms，全部押在**结构改造**上：attention 相位数（0.5ms 池，在线 softmax 5 相位→3）、tri 双列/多列结构（245GB/s vs dual 600，仍有 2× 空间）、head（755 近饱和，仅 ~0.1 余量）。下一轮：attention 相位（最大单池）。
  - **复验**：bench_decode 未动默认，无需重跑门（本轮零行为变化——参数默认值全部等于原硬编码）。

- **2026-11-10（第七十九条）** —— ⚡ **P5 步骤 4 续（attention 结构轮·上半）：2.778 → 2.759ms/token（362 tok/s）；T 扫证伪，VEC 小胜，主导项仍未定位**
  - **开局发现**：attention grid=n_heads=**14 块** —— 全卡线程槽 <1%，比"相位 5→3"更重的病（账本七十六的池子归因要修：attention 不是"相位多"是"块太少+访存结构差"）。
  - **T 扫证伪**（bench_attn_ab，T128/256/512 交错三轮无稳定序）：块内加线程不兑现 —— 与七十八 fat/dual 同款教训，占据率叙事不能乱套。
  - **VEC 小胜**：scores 相位 k 行 `load[8]` 16B 向量读（16 条替代 128 条标量 2B 读，相互独立可全在飞）。求和序 d 升序不变 ⇒ **scores 逐位同旧** —— decode diff `0.008679389953613281` 与基线逐位相同、argmax 287、layer0 normed 0.0、门 21/21（bit-exact 强验证）。交错三轮一致 **-0.017ms**，standalone 2.759 吻合。默认开（`attn_vec`）。
  - **归因更新（诚实）**：VEC 只解释 0.02/0.5 —— attention 20µs/层的主导项仍未定位。候选：① 14 块 SM 饥饿（结构性，j 分块跨块两趟归约才吃得动，块内加线程已被证伪）；② V 相位 128 线程 × kv_len 串行链；③ 2 次树归约 16 屏障；④ f64 算术 1:2 吞吐。下一轮先做**单 kernel 分相计时**（prof 口径）再动刀。
  - **坑在案**：comptime Bool 参 + `comptime if` 在 kernel 内分支干净可用（VEC/标量两版共存，零运行时成本）。
  - **复验全绿**：门 21/21、diff 逐位、argmax 287、layer0 0.0、bench 2.759。
  - **到 2ms 剩余**：0.76ms。

- **2026-11-10（第八十条）** —— ⚡ **P5 步骤 4 续（attention 分相轮）：2.759 → 2.523ms/token（396 tok/s，-0.24ms/-8.5%）；分相计时基建兑现 + 七十六探针谜案告破**
  - **谜案告破**：round 76 六个 mini 探针全灭的根因 = **直发 import kernel 的文件必须先 `from max.gpu.host import ...`** —— 无此行 kernel 约定推断成 `thin`，绑不上 enqueue_function 的 `capturing thin` 候选（Mojo 编译器级怪象，import 集合改变约定推断；session/sections 都带这行所以无恙）。
  - **分相计时基建**：`attn_rope_gpu` 加 `cut` 编译期相位裁剪参（1=scores 2=max 4=exp 8=sum 16=V），累计裁剪差值 = 各相成本（bench_attn_phase，kv=300 口径）：**V 相 16.3µs（35%，最大单相）**、scores ~12.7、max 1.4、sum 2.1、exp≈0。V 相只许 128 线程干活（`tid < head_dim`）—— 七十九 T 扫无效的真因（加的线程被 V 相白吃）。
  - **刀法 V2**：V 相按 j 分组并行（组 g 处理 j ≡ g mod groups，组内升序，smem 部分和按组序合并；T512 → 4 组）—— V 16.3 → 6.6µs（-60%），kernel 46.1 → 36.5µs。端到端交错三轮同序（默认 2.80/2.70/2.89 vs V2 2.79/2.55/2.60）。默认 T512+V2。
  - **数值（意外之喜）**：j 求和序 + sum 树序双重 f64 重结合，decode diff **仍与基线逐位相同**（0.008679389953613281）——良性分布下重结合未翻位。argmax 287、门 21/21、layer0 零容差全保。bit-exact 判据理论上已弃用，实际上仍然成立。
  - **attention 剩余**：scores ~12.7µs 成并列主导（每 j 单线程 128 维点积；d 分摊会破点积序）。下一刀候选：scores 的 warp 级协作点积、或跨块 j 分块（两趟归约）。
  - **复验全绿**：门 21/21、decode diff/argmax 逐位、layer0 normed 0.0、probe 1、prof 21、sections 61、bench_decode 2.523。
  - **到 2ms 剩余**：0.52ms。

- **2026-11-10（第八十一条）** —— 📊 **P5 步骤 4 续（scores 归因轮）：零性能变化（2.523 保持）；三个假设全证伪 + 两个硬事实，scores 结构性判决**
  - **标度诊断**（bench_attn_phase kv=100/300 两口径）：kernel floor（launch+rope+屏障）**恒 9.2µs 不随 kv 变**；scores 与 V 均随 kv 线性（scores 6.3@100 → 17.6@300）。
  - **三个变体全负**（`smode` 编译期参：1=q 行进 smem f64 / 2=+k 预取 / 4=2 线程/j 半维点积+smem 对合并）：36.40/37.35/39.15 vs 基线 37.35 —— **q 的 LDG 重读、合迹、浅预取都不是病根**（pair 还慢 1.8µs）。
  - **微核直接测**（bench_scores_kernel，scores 循环剥离，grid=14 复刻真链并行度）：**空核地板 8.4µs**（launch+sync 税，解释分相"only-scores"9.2µs 的几乎全部）；load[8] 循环 23.8 → 实体 **15.4µs**，即 2.2MB/15.4µs = 143GB/s、**每 SM 9.7GB/s —— 14 块 × 单块 per-SM 天花板**。整行预取（4×load[32]）爆慢 38.5（向量降级/溢出，坑在案）。
  - **判决**：scores/floor 都锁死在"只有 14 块"上——**跨块 j 分块（两趟/三趟归约）是唯一剩余杠杆**，不是循环微优化。V2 已把 V 压到 6.6µs；T1024（V2×8 组）分相 -2.2µs 但端到端交错打平（+0.03/-0.02/-0.05）——不确认，保持 T512。
  - **沉淀**：`smode` 参（默认 0）+ 两个新 bench（bench_scores_kernel 微核剥离法、bench_attn_t1024_ab）留作复裁；分相 bench 恢复 kv=300 口径。
  - **复验全绿**：门 21/21、decode diff 逐位同基线、argmax 287、layer0 PASS、bench_decode 2.534（基线带内）。
  - **到 2ms 剩余**：0.52ms。attention 下一步：j 分块跨块（grid=(14, TILES)，三趟：scores+tile max/sum → V 分块部分和 → 合并；预计 scores 15.4 → ~5µs，代价 +2 发射/层）。

- **2026-11-10（第八十二条）** —— ⚡ **P5 步骤 4 续（attention 跨块分块轮）：2.523 → 2.418ms/token（414 tok/s，-0.105ms/-4.2%）；八十一的结构性判决兑现**
  - **刀法**：三趟 kernel 替代单发 —— **K1** `attn_rs_gpu`（rope + scores 分块，grid=(14, TILES)，每块自算当前 token k 的 smem 副本免竞态，仅 tile 0 写 store 行）→ **K2** `attn_vt_gpu`（V 分块 + 块内冗余 softmax：gmax 无舍入序自由、gsum 树序与旧 total 同式 ⇒ 逐位同；exp 写回 scores 免二次 exp）→ **K3** `attn_comb_gpu`（tile 部分和合并 + 归一，微核）。SM 数 14 → 14×TILES。
  - **裁决**：交错五轮（324 步）关键证据 —— **单 kernel 均值随 kv 单调爬升（2.40@kv102 → 2.81@kv282），tiles=4/8 平稳（~2.44）**：分块治的正是"kv 越大越亏"的 per-SM 天花板病，高 kv 段 **-0.32ms**。tiles=8 对 4 四轮小胜，默认 8。
  - **数值**：scores 点积同 VEC 式同序、gmax 无舍入、gsum 树序同旧 ⇒ 前两趟逐位同；V 部分和按 tile×组重结合 ⇒ decode diff 0.008679390 → 0.008680582（第 6 位小数，门容差 0.037 内），argmax 287、layer0 normed 0.0、门 21/21。
  - **发射账**：+2 发射/层（48 步 × ~2µs）被 SM×8 的收益吞掉还有余 —— 八十一"发射税吃掉收益"的担忧没发生。
  - **坑在案**：struct 新缓冲区必须同时加字段声明 + import（`__init__` 赋值不够，编译期字段检查）。
  - **复验全绿**：门 21/21、argmax 287、layer0 0.0、probe 1、prof 21、sections 61、bench_decode 2.418/2.419 复现。
  - **到 2ms 剩余**：0.42ms。池子更新：attn 三趟后 ~0.3（K2 冗余 softmax 可去：gsum/gmax 存共享免逐块重算）+ dual 0.70（600GB/s 见顶）+ down 0.40 + head 0.35 + tri 0.25 + o 0.15 + 地板 0.31。下一轮候选：K2 冗余 softmax 消除（gsum 一次算三趟共享）或 down/dual 结构。

- **2026-11-10（第八十三条）** —— ⚡ **P5 步骤 4 续（K2 冗余 softmax 消除轮）：交错 A/B 双跑十轮全胜 -0.24ms（默认路径标称 ~2.18）；八十二候选之一兑现**
  - **病根定量**：八十二的 K2 让 8 tile × 14 头每块**全量重扫 scores**——8× exp + 8× scores 写回 + gsum 树（10 屏障/块）。112 块 × 300 j 的冗余实际比 V 分块本身还贵。
  - **刀法 K2'**：gmax 每块冗余**只读**（无 exp 无写回，max 无舍入序自由）；exp 只算本 tile 且 smem（e_s）缓存供 V 循环复用（全行恰 1×）；tile 局部和树规约写 tsum[head×TILES+tile]，gsum 推迟 K3' 合并（Σ tile 和后归一）——**kernel 数不加**（仍 3 发）。exp 值逐位同旧（同 s 同 gmax）；gsum 变重结合。
  - **裁决**：交错 A/B（tiles=8 vs -8，324 步 × 2 跑）**十轮全胜**，mean -0.24/-0.27/-0.24/-0.25/-0.26 与复跑 -0.25 级完全一致；best 口径 2.38-2.43 → 2.155-2.176。standalone bench 读数 2.45-2.47 受盒子热漂污染（八十二同口径 2.418 → 漂 +0.27），按 A/B 标称默认路径 ~2.18。
  - **数值**：gsum 重结合 ⇒ decode diff 0.008680582 → 0.008678675（第 6 位小数，容差内），argmax 287、layer0 零容差、门 21/21。
  - **坑在案**：① e 缓存与归约 scratch 不可共用 smem（树规约会毁缓存）；② j↔tid 映射写缓存后 V 循环读它需 barrier（跨线程）；③ session 分支交换用唯一串逐对替换（assert 锚点防错位）。
  - **复验全绿**：门 21/21、argmax 287、layer0 0.0、probe 1、prof 21、sections 61。
  - **到 2ms 剩余**：~0.18ms（按 A/B 标称）。池子更新：attn 三趟后 ~0.06（K2' 已榨）+ dual 0.70（见顶）+ down 0.40 + head 0.35 + tri 0.25 + o 0.15 + 地板 0.31。剩余候选：down/dual 结构、tri 多列、或地板（发射数再压）。

- **2026-11-10（第八十四条）** —— 📊 **P5 步骤 4 续（tri 多列证伪轮）：零性能变化（默认 tri_cols=0 保持）；一次结构误判被干净证伪**
  - **假设**：tri 245GB/s"有 2× 空间"——每块只算 1 输出、x 重载 896 次、latency 主导 ⇒ 多列（每块 COLS 输出）应摊薄。
  - **实现**：`f16_matmul_qkv_mc_gpu[THREADS, INNER, COLS, ...]`（k 序 tid*8 步 THREADS*8 与树规约组序同旧 ⇒ 逐位同； InlineArray 参数序 [T, size] 坑一次；COLS=16/T128 爆 48KB 静态 smem 上限 53760B——sacc 3·COLS·THREADS·8B 是预算大头）。
  - **证伪（单调签名）**：交错 A/B 四轮一致——旧 2.42-2.43 best；**c2t128 +0.05 / c4t256 +0.17 / c8t128 +0.38**，列越多块越少越慢。判决：**tri 的 896 块本身就是并行度**（~8 块/SM 驻留互相藏延迟），多列把 occupancy 砍到 1 块/SM 4 warp，延迟全暴露。245GB/s 不是带宽病是 latency-bound，"2× 空间"误读（七十八"占据率叙事不能乱套"教训的镜像：这次乱套的是我自己）。
  - **沉淀**：mc kernel + bench_tri_mc_ab 留作复裁工具；`tri_cols` 参（默认 0）留存。tri/o 这对小 matmul（同为 896 块×1 输出形状）的剩余杠杆只剩：K3 合并折进 o kernel（省 24 发射 ≈0.05ms，代价 o 读 vpart/tsum 的跨 kernel 依赖）。
  - **到 2ms 剩余**：~0.18（不变）。下一轮候选：K3→o 折叠、或 head/down 收尾勘察。

- **2026-11-10（第八十五条）** —— 📊 **P5 步骤 4 续（dual 深挖轮）：零性能变化（默认保持 ~2.18 标称）；dual 三条路线全证伪 + 池子重估**
  - **形状扫描证伪**（bench_dual_ab，端到端交错四轮）：t64o4（默认）best 2.40/2.44/2.44 最优；t64o8 +0.14、t128o4 +0.11、t128o8 +0.02~0.03 —— ILP/occupancy 旋钮不是病根。
  - **池子重估（prof_f16 重读）**：head T128O8 实测 **1031GB/s**（264µs 长核，稳态）证明盒子 >1TB/s 可达；但 dual/down 的独立测量含 ~8-10µs launch+sync 地板（八十一空核 8.4µs）——**dual 真实体 ≈800GB/s 而非 576**，池子里 dual 0.70 的"砍半"想象空间实际 ~0.1ms 级。
  - **norm 出核 + 无 smem 变体证伪**（`f16_matmul_dual_nx_gpu` + rmsnorm 单发写 d_xnorm，bench_dual_nx_ab 四轮）：nx4 +0.02、nx2 +0.04、nx4t128 +0.19 —— **44 warp/SM 也没换来带宽**（occupancy 假设第二次死），且 +1 norm 发射税吃掉 kernel 端收益。nx 路线留档（dot 序同旧，逐位同；d_xnorm 缓冲 + dual_nx 参留存）。
  - **判决**：dual/down 卡在 ~580-800GB/s 的统一平台上（f32 FMA 点积 + f16 读的指令路径），微结构路线（形状/ILP/occupancy/smem）全部走完。**剩余大杠杆只剩 tensor core（MMA）路径**——dual+down+head+tri+o ≈ 1.85ms 池子，MMA 潜在 2-4×，但为多轮工程。
  - **复验全绿**：门 21/21、decode diff 与八十三基线逐位同（0.008678674697875977）、argmax 287、layer0/probe/prof/sections 全过。
  - **到 2ms 剩余**：~0.18（微结构路线已尽，MMA 是下一阶段的主战场）。

- **2026-11-10（第八十六条）** —— 📊 **P5 步骤 4 续（skip 归因轮）：默认回滚 tiles=8 保持（2.417，diff 逐位同八十三）；换来管线真值归因 + 重大 bug 修复；RAWEXP 止损**
  - **skip 掩码归因（bench_skip_attr，管线真值）**：sections bench 是 q4 旧口径不可用；skip 差值 = 家族管线成本。四轮 best 稳定：**attention 0.472（#2，比池子估算高 0.17——三趟 48 发射税计入）**、dual 0.637、down 0.355、head 0.352、tri 0.228、o 0.155、gather ≈0；合计 2.204 → **真地板仅 ~0.22**（token 路径 0.011 + logits d2h 0.032 + 发射队列）。⚠️ probe MAXT > ~512 会炸（cos 表 fixture 上限）。
  - **重大 bug 修复（d_attot OOB）**：八十三起 d_attot 误配 `n_heads`（14 f64），K2' 写 tsum[head×8+tile] 需 112 —— **越界 784B 三轮未爆**（分配器 slack 掩护），diff 仅第 6 位微移的"好运气"其实是写坏邻接缓冲未发作。RAWEXP 改变布局后爆（device 端 tsum 行 assert）。已修 `n_heads * 8`。教训：**设备端越界写在本栈不 fault——需要缓冲金丝雀门**。
  - **RAWEXP（跳过 gmax 减稳）止损**：首轮 A/B -0.27ms 全胜是**假象**——RAWEXP 分支漏发 K1（rope/scores/kv store 全没跑，attn=0、tsum=陈旧 scores 的 exp≈1，-0.27 主要是"漏发整个 K1"）。补 K1 后 diff=inf；物理删 gmax 版（vt2r）+ 恒等直证 K2r 读到干净 scores（-3.5），exp 恢复后仍 inf——恒等/单行差异改变 K1 可观测输出，属编译器/缓存域，非我的代码域。**按止损纪律回滚**：默认 tiles=8，vt2 还原、vt2r 删除。gmax 相真实成本 ~2-4µs/层（非 11），候选保留但需另寻实现路径。
  - **复验全绿**：门 21/21、decode diff 逐位同八十三基线（0.008678674697875977）、argmax 287、layer0/probe 全过、bench 2.417。
  - **到 2ms 剩余**：~0.42（按管线真值重算：attention 0.472 是下一个主战场——但三趟结构已就位，K1/K2' 内部还有 ~5µs/层发射间隙；MMA 仍是 down/dual/head 的大杠杆）。

- **2026-11-10（第八十七条）** —— 🔬 **形状/结构穷尽轮：三连证伪 + HBM 上限锚定（1050GB/s）——f16 管线结构地板 ≈2.1-2.2ms/token**
  - **attn_tiles 交错重扫**：t8 四轮全胜（2.450 vs t16 2.485 / t32 2.654 / t4 2.456）——八十二结论经交错口径确认，gmax 冗余随 tiles 增长盖过块数收益。
  - **tri/o/down t512 重扫**：惨败（tri512 +0.29、dn-o512 +0.18）——t128 驻留形状确认。形状空间（tiles/threads/outs/norm 位置）至此**全部穷尽**。
  - **split-inner（down_si）证伪**：归约维切 4/8 份 + combine 微核，四轮全败（si4 +0.12、si8 +0.18）——down 448 块并非饥饿，额外发射+部分和往返盖过延迟收益。基建保留（`f16_matmul_fat_si_gpu` + `fat_si_comb_gpu`，`down_si` 参默认 0）。
  - **HBM 纯流式探针（`hbm_probe_f16_gpu`，新锚点）**：512MB f16 流式读——18992 块 965、4748/2374 块 **1052 平台**、过度订阅（76k 块）反掉到 362（barrier 树主导）。**盒子 f16 流式上限 ≈1050GB/s**。
  - **战略结论**：权值族对上限的占比 = head 76% / dual 67% / down 64%——理论带宽余量 ~0.4ms，但三轮结构攻击（形状/split/nx）均未能把 dual/down 推向 head 的 76%。**f16 权值口径的结构地板 ≈2.1-2.2ms/token**；破 2ms 需换字节口径（权重量化，回到 q4 精度代价）或 MMA 路径（对本带宽界无益，仅益算力界）。2ms 目标在本盒 f16 口径下判定为**不可达**，建议后续轮转向：a) dual/down 向 76% 靠拢的微优化（~0.15ms）；b) 端到端系统口径（batch/多请求吞吐）。
  - **复验全绿**：diff 逐位同八十三基线、argmax 287、门 21、2.423。

- **2026-11-10（第八十八条）** —— 🚀 **GPU batched prefill 落地（系统口径转折点）+ dual_o 证伪 + 三个测试侧陷阱**
  - **dual_o 摊销证伪**：O8 +0.17、O16 +0.7（块数塌方 608/304，每线程负载链拉长反杀摊销）。dual/down 五路参数空间（tiles/threads/outs/norm/split）**全部穷尽**。
  - **prefill_step（GpuSession）**：t 行批量前向。gather/rmsnorm 本就多行 capable；matmul 族 row=block_idx.y 直通 grid y=t；attention 三趟 batch 化（grid z=t，因果界 upto=pos0+b，scores 行=(b·heads+head)·kv_len，stride 取终态 kv_len）。`PREFILL_MAX_BATCH=128` 批缓冲 ~25MB；超长 prompt 调用方分块 = 天然 chunked prefill。
  - **竞态发现与修复**：行 b 的 scores 读行 pos0..pos0+b-1 的 store 行 —— 同 launch 跨 block 写读竞态（decode 无此问题）。拆两发：`kv_store_b_gpu`（rope+落位）+ `attn_rs_b_gpu`（纯 scores），跨 kernel 边界可见。t=1/2/4/8 逐级二分定位。
  - **数值门（test_session_prefill，双 PASS）**：row8 diff 0.0053、row9（decode 续接 prefill KV）diff 0.0061，argmax 一致（f16 门 0.003×ref_max）。
  - **TTFT 基线（bench_prefill）**：128 tokens prefill **131ms** vs 逐 token decode ×128 = 315ms → **2.4×**。未达理论值的原因明确：当前 batch = B 个独立 GEMV，权重每行重读（38GB 流量）；真 GEMM 分块（x-tile 进 smem、权重每 RB 行共享一次）理论 3-6ms（20-40×）—— **这是下一轮的主战场**。
  - **三个测试侧陷阱（记账防复发）**：① decode_step/prefill_step 共享 h_logits host 缓冲，第二次调用覆写第一次返回指针内容——先比对再步进；② CPU run 返回指针指向模型 arena，下一次 run 复用覆写——快照到自有序；③ CPU run 累积历史，row9 参照需独立模型单次跑 9 tokens。
  - **修复悬空 import**：八十六删了 vt2r kernel 但 session import 残留（HEAD 上编译需靠远端 stale attention.mojo）——已清。
  - **复验全绿**：门 21、decode diff 逐位同八十三、argmax 287、layer0/probe/prof 全过、2.416。

- **2026-11-10（第八十九条）** —— 🚀 **真 GEMM 分块落地（prefill 主战场第一刀）：同核 A/B 2×，双数值门 bit-exact PASS；共享卡噪声入档**
  - **三核落地（`f16_bmm_fat/dual/qkv_gpu`）**：每 block 计算 OUTS 输出 × RB 行——w 载入提出 rr 环（权重每 RB 行一次读，流量 t×矩阵 → ceil(t/RB)×矩阵），x tile 走 L2（5.2GB@4TB/s ≈ 1.3ms 非瓶颈）。norm 前置单发（rmsnorm_gpu 本就多行 capable）。每输出累加序与 fat/dual/qkv 同构 ⇒ **f64 档与八十八路径 bit-exact**（row8/row9 diff 逐位相同）。RB/AF 均 comptime 参（AF=f64 默认保 bit-exact，f32 备选）。
  - **同核 A/B（bench_prefill_rb，rb=1 即 GEMV 行为）**：rb=8 vs rb=1 ≈ **2×**（~130-166ms vs ~230-341ms，128 tokens）。理论（权重 4.8GB@1050 = 4.6ms）未达——**f64 FMA 吞吐/发射界**确诊：每 FMA 2 条 cvt 挤占发射口，FP64 有效利用率 ~11%；f32 档仅 -10%（cvt 假说只兑现一小部分，且在共享卡噪声边缘）。
  - **测试侧陷阱两个**：① bench 位序 kv 超卖（3 档 × 2 rep × 128 = 768 > max_tokens 512 → "kv store is full"——rb=16 的"崩溃"实为同一原因，非 smem）；② SIMD[DType.float64] 语法三连坑（f64 别名不存在、DType 需 DType.float64、DType 无 __is__ 用 ==、lane += 需 Scalar[AF] 包装）。
  - **共享卡噪声入档**：卡3 被邻居占满（decode 7.5ms vs 惯常 2.42）——跨轮对比必须换卡重锚定基线；卡2 也有 ±30% 波动（decode mean 3.64 vs best 2.48）——同核同进程 A/B 是唯一可信口径。
  - **复验全绿**：门 21、decode diff 逐位同八十三、argmax 287、layer0/probe/prof 全过、prefill 双门 bit-exact 同八十八。
  - **下一步**：TTFT 仍 ~150ms 距 4.6ms 带宽地板 30×——需 MMA/tensor core（FP16 进 f32 acc，312 TFLOPS 级）或 w 载入形态重构；共享卡噪声下建议先做长窗口多次采样（每组 ≥10 rep 取 min）再比。

- **2026-11-10（第九十条）** —— 🔬 **诊断战役：down-bmm 单核 1350µs vs 理论 132µs，五假说系统性证伪；两个测量口径陷阱入账**
  - **长窗口纪律建立（bench_prefill_rb v2）**：每 rep **新 GpuSession**（同会话复 rep 会把 128 token/rep 追加 KV，rep 间注意力工作量递增——八十八/八十九的 rep0/rep1 对比全部失真，此坑记账）+ warmup rep 不计钟（per-session 首发射成本 ~95ms 落钟内——另一个口径陷阱）。10 rep 取 min 是共享卡唯一稳定口径。
  - **单核剖析（prof_bmm，16 臂）**：down 形状（896×4864×128 行）实测 1350-3310µs vs 理论带宽墙 132µs——有效带宽仅 100-135 GB/s（HBM 的 1/10）。正面锚点：decode-fat 1 行 431 GB/s、bmm RB1 1 行 370、RB8 纯 w 流（n_rows=1）498；单因子（acc lanes 2→16、块数 448→7168）各只 1.3-1.5×，组合超线性 14×。
  - **五假说全灭**：① FMA 吞吐界（f32 acc 仅 -10%，19.5T 峰下应 2×）✗；② x L2 流量（OUTS 2→8，x 流量 ÷4，时间不动）✗；③ MLP 不足（while→@parameter 展开 5 个 w load，无效）✗；④ smem x-tiling（三种几何全劣且越 tile 越差：CH256 1767 / CH512-T64 2481 / CH1024 3310——barrier×chunk + smem 挤占 occupancy 是毒，CH 必须=THREADS×8 否则半线程闲）✗；⑤ SIMD lane 累加 lowering（acc 搬 smem RMW 同速 1321µs）✗。
  - **已落地（保持 bit-exact）**：f32 快路向量化（SIMD 乘 + reduce_add，AF 参透传 session）；f64 路径 k 迭代展开（累加序不变）。门全绿：prefill row8/row9 diff 逐位同八十八，decode 门/argmax/门 21 全过。
  - **无 profiler 盲调的边界**：ncu/nsys 不在共享箱——cycle 级推理多次不能闭合（在飞字节、占用率、发射口都对不上实测），微架构定位到达极限。
  - **九一攻击面**：SASS/PTX 阅读（dump 编译产物看 lane 操作、load 流水与发射序列）——唯一剩余的决定性工具；或重审 decode-fat 431 GB/s 与 bmm 满载 100 GB/s 的差异因子表（x smem 单行 × acc 2 lanes × 无 chunk barrier 的组合形态）。

- **2026-11-10（第九十一条）** —— 💥 **SASS/PTX 阅读破案：`load[8]` 全仓库标量化灾难（736 条 LDG vs 999 条 FP 指令）；`alignment` 参一行修复，GEMV 路径 -65%**
  - **破案手法**：本地无 ncu/nsys——包一层 ptxas（拦截脚本留档输入 PTX）+ `cuobjdump/nvdisasm` 反汇编。JIT 不落盘、二进制无设备码、std 源码不在盘，拦截是唯一口子。
  - **病灶**：`Pointer.load[8]` 生成 **8 条 `ld.global.b32`**（w 的 f16×8 甚至拆成 8 条 2 字节载入），向量载入（v2/v4）**全仓库为零**。bmm SASS：LDG 736 条 vs FP 指令 999 条——发射口 3/4 被 LDG 淹没，这就是 0.83 TFLOP（FP 峰的 4%）与有效带宽 100 GB/s 的共同来源；decode-fat 431 GB/s 也是标量载入下的"好成绩"。
  - **修复（一行）**：`load[8, alignment=16/32]`——`Pointer.load` 有 `alignment` 关键字参（从 max kernels 源码 `aligned_load` 实现里挖出：`ptr.load[width=w, alignment=alignof[SIMD]]`）。PTX 验证：34 条 `ld.global.v4.b32`，LDG 静态数 736→50。全仓库 11 处 load[8] 全部加对齐（x: 32B=alignof(SIMD<f32,8>) ✓ 行偏移核算全过；w: 16B）。
  - **收益（长窗口 min，卡被邻居占满按 1.4× 折算干净卡）**：rb=1 f64 306→~107ms（**-65%**，LDG 密集度最高的 GEMV 受益最大）；rb=8 f32 110→~78ms（-29%）；rb=8 f64 122→~110（f64 档瓶颈转移回 cvt+FMA 发射）。decode 核同步受益待干净卡复测。
  - **bit-exact 保持**：载入形式变化不改数值——prefill row8/row9 diff 逐位同八十八（0.005281…/0.006142…），decode diff 逐位同八十三，门 21/layer0/probe/prof 全绿。
  - **prof_bmm 归档**：16 臂假说排除表（九十）+ vec/vec2 探针核。标量 LDG 假说其实统一解释了九十的全部"矛盾"：单因子小、组合爆炸（LDG×acc lanes×块数的发射口挤兑）。
  - **九十二攻击面**：① 干净卡重锚定全部数字（decode 收益确认）；② MMA/tensor core（剩余 6× 的唯一解，f16 v4 载入已就位）；③ 876µs→理论 132µs 的缺口 = LDG 发射 + 依赖停顿，MMA 把 FMA:LDG 比从 1:1.1 翻到 64:1。

- **2026-11-10（第九十二条）** —— ⚖️ 干净卡重锚定：两旧结论修正 + 主战场重新定位（matmul 已非 prefill 瓶颈）
  - **卡 4 空闲重锚定（全部 bit-exact 门同步全绿）**：decode **2.243ms/token（历史最佳，向量载入 decode 也 -8%）**；rb=1 f64 **56.8ms**（八十九口径 306ms → **5.4× 累计**）；rb=8 f64 61.3ms；**rb=8 f32 45.1ms/128tok**。九十一的折算（1.4×）低估了实际污染。
  - **修正一（九十条"组合爆炸 14×"）**：干净卡上 RB8-n1-gridY1 = 16.9µs vs RB1-gridY16 = 16.8µs——**1×**！"超线性交互"是脏卡伪象（邻居负载对多块核打击更重）；唯一真毒就是标量 LDG（已修）。
  - **修正二（decode-fat x-smem 路线）**：干净卡 decode-fat 12.5µs (700 GB/s) **慢于** bmm RB1 全局 x 8.1µs (**1084 GB/s ≈ HBM 满速**)——对齐向量载入后 x 进 smem 反而是负资产。
  - **主战场重新定位**：down f32 单核 325µs (438 GB/s) → 全模型 prefill matmul 估 ~1-5ms，而 prefill_step 全步 45ms——**matmul 已降至个位数占比**，TTFT 瓶颈转移至 attention/kv/softmax/逐 token 核链路。MMA（九十一预案）对 TTFT 的收益上限从"10×"缩至"~2×"以内，优先级下降。
  - **tiled 系列维持全灭**（944-1571µs，含干净卡复核）；prof_bmm 全 16+ 臂数据已入档供九十三使用。
  - **九十三攻击面**：prefill_step 45ms 逐段分解（attention vs kv vs norm vs 发射开销）——先测出大头再动手；MMA 降级为 matmul 补完项（f32 档容差门下做）。

- **2026-11-10（第九十三条）** —— 🔪 prefill 45ms 逐段分解：CPU 发射 5.4ms、GPU ~40ms；**真瓶颈 = INNER=896 的 qkv/o 核 k 迭代数不足**
  - **插桩落地**：prefill_step 加 `profile: Bool = False` 运行时参（默认零开销、编译器裁剪），10 段计时（gather/tri/kvs/scores/vt2/comb/o/ff/down/tail）。门复验 bit-exact 逐位同八十八/八十三。
  - **t=1 对照（43 launch、GPU≈0）**：无 profile 总 5.38ms → **CPU 发射+启动 ≈125µs/launch**；profile 段和 7.0ms。
  - **t=128 对账闭合**：45.1ms ≈ CPU 5.4 + o~21 + qkv~13 + dual~6 + down~4 + attn/norm~2。**o+qkv 占 ~34ms（75%）**。
  - **o/qkv vs down/dual 5× 的结构因子**：THREADS×8=1024 元素/迭代，INNER=896 → **单 k 迭代**——每线程仅 1 个 w load + 8 个 x load，无流水可谈，内存延迟全暴露；down/dual（INNER=4864）5 迭代流水 → 快 5×。session-env 连发 down = 518µs/launch（孤立 325，仅 1.6× 差）证明 session 环境无罪——**核结构就是原罪**。
  - **九十四攻击面（按预期收益排序）**：① qkv/o 核 k 维重构——split-K（grid.z 拆 k-slice + 两阶段归约，f32 档容差门）或 THREADS 减半×块翻倍（总量不变但调度粒度细化）；② norm 融回 bmm 核（省 8 launch ≈ 1ms CPU + 一次 xnormb 读写）；③ CPU 发射 125µs/launch 的构成（参数打包 vs JIT 查找）——若可降至 ~50µs，43 launch 省 3ms。
  - **工具链沉淀**：bench_prefill_profile（分段/无 profile/t=1 三口径一体）；arm_env（session-env 连发实验模板）。

- **2026-11-10（第九十四条）** —— 🕳️ 测量体系崩塌与重建：n_layers=12（非 4/8）、sync 弱语义、发射 0.3ms/launch、GPU 真实 ~1ms；45ms 主体 = final sync 的 AsyncRT 神秘等待
  - **o 形状 T 扫描（prof_bmm 六臂）**：back-to-back o（INNER=896）= 135µs（比 down 325 还快！）、T64/T32 = 87/84µs——**九三"单 k 迭代延迟暴露"证伪**（那是 sync 分段环境的伪象）。
  - **同进程铁证链**：prefill_step hot = 49-85ms vs 手写重放（同 session 同 buffer 同参数逐字节 dump 证实）同序列 7.6ms/4层——3-5× 差。**launch 参数逐字节相同**（指针 dump 对照）。
  - **n_layers = 12**（index 越界暴露）——**九十二/九三全部"4 层"解读作废**：生产 123 launch/次；probe 段值重算 tri 0.68/层、o 1.25、ff 1.28。
  - **probe-copy 法（本轮发明）**：D2H copy 阻塞至 stream 前序完成 → 每段真实 GPU 25-28µs/层、tail 353µs——**GPU 全模型真实 ~1ms**。孤立单核全家 2-3ms（o 142µs、down 330-590µs、qkv 87µs、lm_head 310µs=876GB/s）。
  - **synchronize() 语义崩塌**：second-sync 恒 0.0016ms、final-sync 在 0.001-79.9ms 跳变、GPU 空闲（tail probe 353µs 完成）状态下 final sync 仍等 43-80ms——**AsyncRT_DeviceStream_synchronize 层的神秘等待**（源码确认强语义外壳、内部行为异常）。**全部历史基于 sync 的 bench 数字可信度存疑**。
  - **分组 host 时刻戳（lts 法）**：发射 0.43ms/层 × 12 = 12.4ms（~0.3ms/launch，AsyncRT worker 层）、tail 发射 3.3-3.9ms、GPU ~1ms——**生产 45ms ≈ 发射 12 + sync 神秘 30-70**。
  - **修复落地（门全绿）**：① kernel 特化提为方法内 alias（LMHEAD_K 等 10 个）；② prefill_step 加 `sync_tail: Bool = True` 运行时参——**流水化调用路径可跳过尾部 sync**。
  - **九十五攻击面**：① AsyncRT sync 神秘等待机制（升级 MAX/AsyncRT 版本对照 / MAX_WORKER 环境变量）；② 流水化调用语义改造（sync_tail=False 的正确性协议：读 logits 前的单次轻同步）；③ 发射 0.3ms/launch 的构成（kernel 参数 marshalling vs worker 队列）——launch 融合（norm 进 bmm）直接减少发射数；④ 重放 1.9ms/层 vs 生产 3.75ms/层 的残余 2× 差。

- **2026-11-10（第九十五条）** —— 🎯 45ms 终局解剖：host 发射 0.7ms、GPU 真实 43-51ms、probe-copy 法证伪、sync_tail 真正落地
  - **绝对时刻戳对账法（本轮方法论）**：bench 侧 tka/tkb 与函数内 lts[0]/lts[19] 打同一时钟绝对值——单次调用双口径矛盾（发射 44 vs 函数体 4.5）最终对齐。
  - **真凶落网**：prefill 尾部有一个九十三时代遗留的**无条件裸 synchronize()**（lts[18]→lts[19] 段 48.8ms）——sync_tail 参数此前根本没包住真正的 final sync。**修复：裸 sync 包进 `if sync_tail:`（门 bit-exact 全绿）**。
  - **修复后口径**：发射 0.69-0.74ms（123 launch × ~2.9µs，noop 标定 200 launch=0.58ms 实证消化速率）+ **sync 等待 43.4→45.8→48.7→51.4ms（随 kv_len 128→512 线性增长）**。
  - **probe-copy 法证伪**：D2H copy 走 AsyncRT 独立 memcpy stream，其返回≠kernel stream 完成——九十四"GPU ~1ms"结论作废，"GPU 空闲下 sync 等待 44ms"重新解读为**GPU 真实执行 43-51ms**。
  - **sync 语义恢复正常解读**：九十四的"sync 跳变"实为不同调用点 sync 时机不同；`asyncrt.await.spinning/sleeping` + `MODULAR_THREAD_BUSY_WAIT_US=10s` 实验无效——不是 worker 等待机制。
  - **未解残余（九十六）**：生产每层 GPU ~3.6ms vs 手写重放 1.9ms/层（同参数逐字节对照仅做过 o launch；qkv/dual/down/attention 未对照）——1.9× 差在 GPU 执行本身。sync-profile 段谱（12 层重算）：tri 0.68 + o 1.25 + ff 1.28 + down 0.67 ≈ 3.9/层——o/ff 回归为最大目标。
  - **工程红线**：session max_tokens ≤ 512（rope 表 = model.cos 仅 512 行，>512 段错误 crash）。

- **2026-11-10（第九十六条）** —— 🔬 nsys 突破：GPU 时间线到手——生产与重放 launch 序列逐字节一致、但存在均匀 1.8× 慢执行区
  - **nsys 可用**（/usr/local/cuda/bin/nsys，九十四时误判不可用）： + ERROR: No input file specified.。
  - **launch 全参数逐字节对照**（python 块级提取并排 diff）：生产 10 launch/层与重放完全一致（kernel 名、Int32 参数、grid、block、buffer 指针）——序列无罪。
  - **trace 铁证**：重放 12 层 GPU = 22.5ms（S21=3 rep×22.5）；生产 12 层 GPU = 23.1ms（S17 前 124 kernel，1.03×）——**但同一 trace 中存在 1.8× 慢区**：qkv 608 vs 330、dual 1149 vs 627、down 592 vs 331、o 250 vs 138、vt2 511 vs 330——**所有 kernel 同比慢 ~1.8×**（grid 相同）。
  - **慢区位置**：紧随新 session 的 544MB H2D 权重上传（35.263s）之后的首个 prefill（35.29 区层 3.54ms）；快区（35.51 区层 2.37ms）在同 trace 稍后。
  - **假说排除**：①冷权重预热无效（warm0/1 后 call 仍 48.65，同 session 第 3 次调用与第 1 次一样慢）；②launch 消化（noop 2.9µs/launch）；③BUSY_WAIT；④序列/参数差异。
  - **未解（九十七）**：慢 1.8× 区与快区的切换条件不明——嫌疑收敛到 GPU 执行上下文（频率/功耗状态/L2 全局状态/AsyncRT 队列深度对 kernel 调度的影响）。下一步：NVTX range 精确标注每个 prefill 调用与重放 rep 的边界（ usage: nsys [--version] [--help] <command> [<args>] [application] [<application args>]

 The most commonly used nsys commands are:
	profile       Run an application and capture its profile into a nsys-rep file.
	launch        Launch an application ready to be profiled.
	start         Start a profiling session.
	stop          Stop a profiling session and capture its profile into a nsys-rep file.
	cancel        Cancel a profiling session and discard any collected data.
	service       Launch the Nsight Systems data service.
	stats         Generate statistics from an existing nsys-rep or SQLite file.
	status        Provide current status of CLI or the collection environment.
	shutdown      Disconnect launched processes from the profiler and shutdown the profiler.
	sessions list List active sessions.
	export        Export nsys-rep file into another format.
	analyze       Identify optimization opportunities in a nsys-rep or SQLITE file.
	recipe        Run a recipe for multi-node analysis.
	nvprof        Translate nvprof switches to nsys switches and execute collection.

 Use 'nsys --help <command>' for more information about a specific command.

 To run a basic profiling session:   nsys profile ./my-application
 For more details see "Profiling from the CLI" at https://docs.nvidia.com/nsight-systems），一次 trace 内交替触发慢/快区，锁定切换条件；辅以 nvidia-smi 时钟采样。
  - **方法论**：trace 分段必须用强边界（sleep/usleep 插入）+ kernel 名+grid 双重对齐，仅靠时长模式分段会错位。

- **2026-11-11（第九十七条）** —— 🌡️ 慢区之谜破案：A100 功耗墙降频 + 层间功耗余量窗落地（1.47×）
  - **GPU 映射确认**（ctypes cuDeviceGetUuid）：CUDA_VISIBLE_DEVICES=4 → 物理 GPU 4（空闲无争用）——争用假说死亡，DVFS 复活。
  - **GEMM 时钟探针**（1500 发/12.5min）：短 burst（每 500ms 10 发）恒 254µs；重放连续流 138µs——同 kernel 同 grid 同 buffer，差异只在活动模式。
  - **间隙阶梯 U 型曲线**（probe_gap）：连续/0-2ms 间隙 250µs（慢）→ **5-100ms 间隙 136µs（全速）** → 500ms 254µs（深眠掉档）。765/1410MHz = 0.54，136×1.84 = 250——**功耗墙降频模型精确闭合**：连续满载 45ms 撞 250W cap → SM 降到 765MHz；段级节奏让功耗余量恢复 → 1410 全速。
  - **层间功耗余量窗**（probe_layergap，8 臂受控）：nest0 连续 40.6ms（=生产模式）→ 纯 sync 34.1 → sync+20us 30.3 → **sync+30us 27.7ms（最优）** → sync+50us 28.0 → sync+300us 44.9（开销反噬）。
  - **落地**：prefill_step 加 `layer_gap_us: Int = 0` 参数（层末 sync+usleep；默认 0 行为不变），生产推荐 30。门 bit-exact 全绿（prefill row8/decode row9/decode PARITY）。
  - **环境噪声教训**：bench_r97 ABCD 顺序对照在整机高载（其他 GPU 96%）下被趋势污染（A 先跑系统性偏慢）——受控 probe_layergap（同 session 内臂间对比）为准。今天绝对值整体 +30%。
  - **未竟（九十八）**：①29.0 vs 重放 22.5 的残余差（sync 往返 ~0.55ms/层）——段级更细粒度间隙或 gap 与发射重叠；②生产服务层启用 layer_gap_us=30（TTFT 43→~29ms）；③功耗墙根因——其他 GPU 满载时机箱散热，建议向运维要 GPU 4 独立供电/散热数据或 -lgc 权限。

- **2026-11-11（第九十八条）** —— 🌡️ 功耗墙是环境相关瓶颈：稀疏调度扫描 + layer_gap_every 落地
  - **调度扫描两轮**（每 N 层 sync+GAP µs，同轮相对比较）：热时段 (1,30)=41.6 → (2,30)=24.8 → (6,300)=23.7 → (4,300)=23.3（另一轮复验 23.3/23.3）；安静时段 **(0,0) 连续 = 22.3ms = 理论值**。
  - **重要反转**：安静环境下连续执行即最优——**功耗墙只在机箱热压（其他 GPU 满载）时生效**。间隙调度是热环境缓解手段（40.6→23.3），安静环境开启略亏（23.3 vs 22.3）。跨天矛盾（昨 nest30 赚 1.47×、今天基线漂 ±60%）由此统一。
  - **落地**：prefill_step 加 `layer_gap_every: Int = 1`（每 N 层末 sync+usleep；默认 1，配 layer_gap_us=0 等效关闭）。热环境推荐 (4,300) 或 (2,30)。门 bit-exact 全绿。
  - **方法论**：整机高载时环境漂移 ±60%——**跨轮绝对值不可比，同轮内臂间相对值才可比**；调度扫描必须含 (0,0) 连续对照臂才能发现墙是否在场。
  - **未竟（九十九）**：①攻击面回归 kernel 本身（22.3ms 中 o/dual/down 段谱细分——重放段谱 tri 0.34/attn 0.3/o 0.33/ff 0.64+down 0.34）；②服务层接入引擎时按热压自适应启用 gap；③运维项：GPU 4 散热/-lgc 权限。

- **2026-11-11（第九十九条）** —— ⚡ FF/o GEMM 寄存器重构：x tile ÷OUTS + w 寄存器组（kernel 1.6-1.9×，端到端 20.0ms 新纪录）
  - **瓶颈定位**：fat/dual 的循环是"每输出重读 x tile"——x 的 L2 流量 = out_cols×t×its×4KB，与 OUTS 无关；down 段 2.34GB/0.545ms = **4.3TB/s 贴住 A100 L2 带宽墙**。
  - **fat2/dual2 失败**（纯循环倒置）：w 重载 ×RB 抵消 x 收益——down 545→616µs、dual 943→1482µs；只有 o（its=1）赚 11%。
  - **fat3/dual3 成功**：w tile 先进寄存器组（InlineArray+S@parameter 展开），再 rr×oo FMA——x÷OUTS 且 w 流量持平。kernel 级交替 30 rep：**down 585→328µs（1.78×）、dual 677→370µs（1.83×）、o 154→97µs（1.60×）**，device 侧逐位对照 **maxdiff 全 0.0**。
  - **寄存器配比是关键**：dual3 OUTS=4（acc 64 f32 + 双 w 32 regs）spill 反而慢 46%；OUTS=2（acc 32）达成 1.83×。fat3 OUTS=4 恰在甜点。
  - **端到端严格 AB**（同进程双 session 交替、kv 与顺序双平衡）：NEW 一致更快 1.11-1.44×；安静态 **22.3→20.0ms（−10%）**、热态 40.9→36.4。
  - **落地**：session.mojo 的 O_K/DUAL_K/DOWN_K 换 fat3(OUTS=4)/dual3(OUTS=2)/fat3(OUTS=4)，grid 相应 ÷OUTS；fat3/dual3 补齐 AF=float64 精确路径（与旧核逐位同，门 diff 数值不变）。门全绿。
  - **方法论**：①先量化墙（流量/带宽）再动刀；②寄存器预算决定 OUTS 上限（f32 acc 每份 4B×RB）；③grid 必须随 OUTS 同步改（改漏=只算一半输出，diff 5.78 当场暴露）；④跨轮端到端比较不可靠，同进程交替+双平衡才干净。
  - **未竟（一百）**：①qkv kernel 同构可改（o 同 INNER=896，潜力 ~1.6×≈0.2ms/层）；②attn rs/vt2 段（kv 相关）；③FA 融合 norm 进 GEMM 省 24 launch；④decode_step 的 matmul 同款改造。

- **2026-11-11（第一百条）** —— ⚡ qkv 拆链寄存器分块：607→188µs（3.2×）——九十九方法论的延伸与超越
  - **结构**：旧 qkv 单核 OUTS=1（每输出重读 x tile，x 流量 528MB），且 o<128 的块做 q/k/v 三段工作（kv 块拖累整体）。拆成 q2（q 链 OUTS=4 寄存器分块，grid 224）+ kv2（k/v 双链 OUTS=2，grid 64）。
  - **战果**：kernel 级交替 30 rep **607→188µs（3.2×）**——超预期（1.6×）：拆核后 kv 小块不再浪费 q 大块的 grid；device 侧逐位对照 q/k/v **全 0.0**。
  - **落地**：session 的 QKV_K 拆 Q_K/KV_K 双发（grid ceildiv(q_out,4)/ceildiv(kv_dim,2)），f64 精确路径与 qkv 同序逐位同——门 diff 数值逐位不变（0.005281448.../0.006142139.../0.008678674...）全绿。
  - **端到端**（热态同轮结构对比）：C/D 臂 42.0/44.1→36.4/38.8、45.0/47.7→39.6/42.3（-12~13%）；安静态预期 20.0→~16ms（待环境红利复测）。
  - **方法论**：拆核也是优化——当链间工作量不均（q 896 vs kv 128）时，合核的 grid 按最大者铺开、小块空转；按链拆分让每核 OUTS 独立选优。
  - **未竟（一百零一）**：①attn rs/vt2 段（kv 相关，vt2 511µs 热态值）；②norm 融进 GEMM 省 24 launch/层；③decode_step matmul 同款；④安静态端到端复测（20.0→? ）。

- **2026-11-11（第一百零一条）** —— ⚡ attn TILES 8→1：消 gmax 冗余 + 线程对齐，端到端 -16%
  - **结构性低效**：rs/vt2 的 TILES=8 使 chunk=nsc/8≈16 行——**每块仅 16/512 线程活跃**，且 gmax 全行扫描 8× 冗余（每 tile 块都扫全 nsc）。16384 块大部分空转。
  - **扫描**（正反序两轮抗漂移，call2/3 对齐）：T8 39.6/42.3 → T4 36.3/39.1 → T2 34.9/37.7 → **T1 34.1/37.1（-14%）**。T1 时 chunk=nsc≤512 ⇒ tid<nsc 全线程用满、gmax 零冗余。
  - **数值**：TILES 变 tsum/vpart 分块 ⇒ 重组非逐位；独立门（test_session_prefill_t1）diff **0.005281567** vs 默认 0.005281448（差 1e-7 量级），PASS。
  - **前置条件（关键）**：vt2 的 phase2 每线程只处理 1 行（jt=t0+tid），要求 **nsc ≤ THREADS(512)**——由 max_tokens ≤ attn_t(512) 的 kv 检查保证；更长上下文须 attn_tiles ≥ ceil(nsc/THREADS)。已在 docstring 记录。
  - **落地**：prefill_step 的 attn_tiles 默认 8→1；双门全绿（prefill row8 0.005281567 / decode row9 0.006138324 / PARITY 0.008678674）。
  - **端到端**（同 bench 结构对比上轮）：C 36.4/38.8→30.4/33.2、D 39.6/42.3→34.1/36.8（**-16%**）；A 62.4/64.6→51.5/51.4。热态 C=30.4ms，安静态预期 ~13-15ms（待复测）。
  - **坑**：同进程二建 GpuSession 即崩（首 session 已消费 host 参数）——双 session 对比须各自独立进程或建在两个 session 之前；本门拆为独立文件（test_session_prefill_t1）。
  - **未竟（一百零二）**：①rs/vt2 融合成 flash-attn 单核（省 1 launch + scores f32 往返）；②norm 融进 GEMM；③decode 路径 matmul 同款；④安静态端到端复测。

- **2026-11-11（第一百零二条）** —— ❌ flash 融合（rs+vt2）负结果：数值逐位同但慢 10%——寄存器峰值分配教训
  - **实现**：attn_fused_b_gpu（TILES=1 限定）——scores 算进 smem + gmax/e_s/V 同核，省 1 launch 与 scores 显存往返（~5µs）。
  - **数值**：vpart/tsum 逐位对照 **全 0.0**（kv=128/512 双点）——scores f32 经 smem 无损、算序同构，融合不破逐位性（可复用的融合方法论）。
  - **性能（A/B 交替 30 rep，kv=512）**：fused **532µs vs 分离 482µs（-10%）**——点积循环（128 次标量 q_rot 载入/c8 循环）与 V 累加同函数 ⇒ 编译器按寄存器峰值分配，全 phase 占用率下降，远超省下的 5µs。
  - **结论**：保留 kernel+probe 作参照（docstring 已标注勿接入）；融合不是免费午餐——**只有当合并后各 phase 寄存器占用相近时才可能赚**。rs（0 smem 高占用）与 vt2（14KB smem）本就工作在不同占用点。
  - **环境**：GPU 0 又满载（77%/72°C）——安静态复测仍不可得（三项 kernel 优化的安静态绝对值待环境红利）。
  - **未竟（一百零三）**：①f16 GEMM 向 tensor core（WMMA）迁移——FF 段 0.7ms/层的 3-5× 空间，走容差门（数值重组大）；②decode 路径 GEMV；③安静态复测。

- **2026-11-11（第一百零三条）** —— 🧨 MMA bring-up：m16n8k16 fragment 映射验证 + down 形状 WMMA 1.34×（未接入）
  - **API**：`max.gpu.compute.mma.mma(mut d, a, b, c)`（d=a·b+c，warp 同步，SIMD 宽度定形状）。m16n8k16 f16×f16→f32 fragment：A 每 lane 8 half（(gid,tig·2)/(gid+8,·)/(+8 列)），B 4 half（(k=tig·2,n=gid)/(k=tig·2+1,·)/(+8)/(+9)），C/D 4 f32（(gid,tig·2)/(gid+8,·)）。单步 check **maxdiff 0.0**（4 warp 独立数据）。
  - **关键 bug（广播模式诊断）**：down kernel 首版 B fragment 用输出列 `col` 而非 tile 内列 `gid`——等价于把 B 第 0 列广播到全部 8 列 ⇒ **got(r,c) = want(r, warp 首列)** 逐点吻合（k 链累加本身正确，值是全和）。修复 `w[(col+gid)*INNER+...]` 后大尺寸稀疏 maxdiff **8.4e-6**（f32 MMA vs f64 参照正常舍入）。
  - **性能**：down 形状（4864×896, t=128）简单 WMMA **245µs vs fat3 328µs = 1.34×**（4.5 TFLOPS）。错误版反而 143µs（B 读命中列少）——修复后变慢是访存真实成本。距离 tensor core 潜力（100+ TFLOPS）尚远：w 每 lane 散乱全局读、x f32→f16 每 col-block 冗余转换、无流水。**升级路径**：双操作数 smem 驻留 + ldmatrix + k 分块流水。
  - **经验**：①`ref` 是 Mojo 保留字；`barrier` 在 max.gpu 不在 std.gpu；comptime if 块内声明不外泄作用域；②mma 分离 d/c 传参在单步与链式均正确（无别名 UB）；③runtime 边界 while 包 mma 无碍（@parameter 展开后 143→245µs 是 B 修复所致）；④文档无 fragment 布局——**必须自建单步 check kernel 对照**（本项目方法论的 kernel 级版）。
  - **未接入原因**：MMA 累加序≠标量路径 ⇒ 破坏 f64 精确锚（门 diff 数值会变，容差内但失去逐位基线）；且 1.34× 值不上 ldmatrix/流水的工程量。**一百零四**：smem 驻留双操作数 + ldmatrix + 流水（目标 FF 段 3-5×），接入时门策略 = 容差门 + 新基线快照。

- **2026-11-11（第一百零四条）** —— ⚡⚡ MMA tensor core 落地：down+dual 接入 f32 快路径，端到端热态 -15~19%
  - **扫描**（交替 12 rep，V0-V6）：x 预转 f16 + 全局 A（V0）310µs 反而慢于 smem 版 245µs——**A fragment 延迟是主因**；多输出 tile 摊 A 流量是最大杠杆：NT1→NT2 208µs→**NT4 138µs**（V4=smem x+NT4+U2 双链）；U4 无增益、NT8 网格太小变差（174µs）。
  - **落地结构**：`f16_mma_down_gpu[INNER, CH=608]`（块 16 行×128 列，4 warp×4 tile，x 分块进 smem 19KB，ACC 残加同 fat3）+ `f16_mma_dual_gpu[896, 608]`（每 warp 每矩阵 2 tile，块 64+64 列，尾融合 silu(gate)·up 公式同 dual3）。仅 `prefill_af==float32` 分支走 MMA（+2 个 x→f16 转换 launch）；**f64 精确路径仍走 fat3/dual3 ⇒ 门 diff 逐位不变**（0.005281567/0.006138324 全绿）。
  - **战果**：down 328→~140µs（**2.3×**）、dual 370→**263µs（1.4×）**；端到端热态 C 30.4/33.2→**24.55/27.79**、D 34.1/36.8→28.91/32.10（-15~19%）；每层省 ~295µs ×12 层 ≈ 3.5ms GPU。
  - **两个坑**：①down 收缩维是 inter=4864 ⇒ smem x tile 16×4864×2B=152KB 超限（0xc000）——**probe 里用 896 方阵测试没暴露**，分块 CH=608 修复；②分块填充误写成扁平拷贝（缺行跨距 INNER）⇒ maxdiff 爆到 7~37——修正 `(r0+i//CH)*INNER + ch*CH + i%CH`。教训：**probe 必须用生产形状**（896 方阵 pass ≠ 4864 收缩维 pass）。
  - **数值定位（重要）**：f32 快路径现在含 f16 x 舍入 + MMA f32 累加——kernel 级对 f64 参照 gate/up/up/act 4.8e-6/1.3e-5/4.8e-5、down ACC 0.0；**prefill f32 路径暂无端到端容差门**（decode PARITY 只盖 decode），一百零五应补 prefill f32-vs-f64 容差门。
  - **MMA 现状**：down 7.9 TFLOPS（tensor core 峰值 312 的 2.5%）——瓶颈是 w 全局 fragment 载入延迟；升级路径 = w smem 驻留 + ldmatrix + k 流水（一百零五）。
  - **未竟（一百零五）**：①w smem 驻留/ldmatrix/流水（down 目标 <80µs）；②o/q2/kv2 GEMM 的 MMA 化（q2 188µs、o 97µs）；③prefill f32 容差门；④安静态复测。
