# 方案：GGUF 读取

状态：**未开工**（账本里这一行是 `missing`）。本文只做设计与判据，不含实现结论，
也不含任何未测的性能数字（账本铁律：未验证的性能数字不许写进仓库）。

## 1. 为什么是它

与 `tamnd/molla` 对照后（见 `01-ecosystem-analysis.md` 与当日日记），alofa 在**引擎侧**
（分页 KV / 前缀缓存 / 调度 / 抢占 / 自研量化核 / 真读张量权重）比 molla 深，缺的是
**覆盖面**。其中最大一项就是 GGUF：现在 alofa 只能吃自己导出的 fp32 裸二进制 + TSV 索引，
别人的模型文件一律进不来。

顺带一个既成事实：本机早就在用 GGUF —— 与 llama.cpp 对齐那档，llama 侧的 q4_0 是
`llama-quantize` 从**同一个 f32 GGUF** 转出来的。也就是说 **fixture 与转换工具链已经在
机器上跑通过**，缺的只是 alofa 自己读它。

## 2. 现在的接缝（代码事实）

| 位置 | 是什么 | 对 GGUF 的意义 |
|---|---|---|
| `src/alofa/model/loader.mojo:86` | `struct TensorFile`：`__init__(dir_path)` 读 `tensors.tsv`，提供 `index_of / has / dims / numel / view(name) -> TensorView / ptr(name) -> F32Ptr / keep_alive()` | **接口模板**：GGUF reader 应当长出同一组方法，模型层才不必分叉 |
| `src/alofa/model/safetensors.mojo:147` | `struct SafeTensorFile`：`mapped / names / shapes / begins / ends / dtypes / data_base` | **结构模板**：GGUF 的张量目录就是这六个字段再外加 dtype id 与对齐 |
| `src/alofa/model/safetensors.mojo:21` | `_u64_le(file, offset)` 小端 u64，带边界检查 | 直接复用（现为文件私有，需提到 `core` 或在 `gguf.mojo` 另写一份并交叉验证） |
| `src/alofa/core/mmap.mojo:36` | `struct MappedFile`，`byte_at(i)`、`size` | 只读映射已有，GGUF 不必新增映射方式 |
| `src/alofa/core/dtype.mojo:32-115` | `QUANT_NONE=0 / QUANT_Q4_0=1 / QUANT_Q8_0=3`，`quant_block_size / quant_bytes_per_block / quant_scale_bytes` | GGML type id → 这几个编译期常量的**映射表**要新增 |
| `src/alofa/kernels/cpu/quant.mojo:408-410` | `Q8_BLOCK=32`、`Q8_BYTES=34`；`dequant_q8_0`(425) / `quantize_q8_0`(473) | GGUF 里的 q8_0 张量**块布局与自家一致**，可直通 |
| `src/alofa/model/arch/qwen.mojo:1359,1362` | `enable_q4` / `enable_q8_output`：加载期把 fp32 量化成块流，arena 持有 | 见 §4.4：直通会绕过这一步，是本次最大的设计选择 |
| `scripts/dump_q8_reference.py`、`scripts/build_real_model_dir.py` | 参照物由 Python **独立另写**；真目录用符号链接拼，不复制第二个 GB | 参照物与 fixture 的组织方式照抄即可 |

⚠️ `safetensors.mojo` 当前明确**按名拒绝 F16**（`bf16_to_f32` 只管 BF16）。账本把这点写成
负向对照（"F16 不是同一种截断"）。GGUF 世界里 F16 是主流之一 ⇒ 是否支持 F16 是**独立的
一个决定**，不混在本方案里（见 §6）。

## 3. 落点：分层

GGUF reader 放 **`src/alofa/model/gguf.mojo`**，与 `safetensors.mojo` 同级。理由：

* 它是"文件格式 → 张量目录"，与 safetensors 完全同职责，模型层（`arch/qwen.mojo`）消费它；
* 它**不许** import `kernels/`（解量化是核层的事，GGUF reader 只给出"这块是什么类型、
  在第几个字节"），否则分层守门会红（`tests/unit/test_layer0_parity.mojo` 那条单向依赖）；
* 它**不许**自己做放宽 / 解量化 —— 那是 `kernels/cpu/quant.mojo` 与 `TensorFile` 的活。

## 4. 设计

### 4.1 结构

```
struct GgufFile(Movable):
    var mapped: MappedFile
    var names: List[String]          # 张量名
    var shapes: List[String]         # 维数串（与 SafeTensorFile 同写法）
    var dtypes: List[Int]            # GGML type id（原始值，不改写）
    var begins: List[Int]            # 张量数据起点（绝对偏移，含对齐）
    var bytes: List[Int]             # 实际占用字节（由**下一个张量偏移**反推，见 4.3）
    var meta: ...                    # 只留我们认识的那几个键
```

外加 `has / index_of / dims / numel / metadata(key)`。`ptr(name)` **暂时不提供**：GGUF 的张量
可能是量化的，返回 `F32Ptr` 是撒谎。谁要 fp32，谁自己走 `TensorFile` 那条兼容路径（放宽 +
arena），与 safetensors 现在对 BF16 的处理保持一致。

### 4.2 头部解析

`magic("GGUF") → version(u32) → tensor_count(u64) → metadata_kv_count(u64) → [kv…] → [tensor_info…]`，
随后按对齐补齐到数据区。三点纪律：

1. **version 必须校验并打印**。不认识的版本直接 `ERR_UNSUPPORTED`，不许"试着往下解析"。
2. **对齐常量取文件说的，取不到就用 32**（`GENERAL_ALIGNMENT`）。⚠️ 对齐漏算是这类 parser 最
   经典的错误：症状不是崩溃，是第 0 个张量对、后面全错。
3. **metadata 只挑要的键，其余必须"跳过"**。跳过需要知道每个 value type 的字节长度
   （UINT8/INT8/U16/I16/U32/I32/F32/BOOL/STRING/ARRAY…），这是全文件最容易写错的一处；
   ARRAY 还要递归跳过元素。判据见 §5 的负向对照 ②。

### 4.3 类型：偏移与 type id **互为校验**，谁都不能单独信

账本原话是"从张量偏移反推 block layout，不信任 type id"。⚠️ **这一条在 2026-09-30 被部分
修正** —— 单向信任偏移是有反例的，见下。

```
declared = ggml_bytes_per_value[type_id]        # 文件声称的
span     = next_begin - begin                   # 偏移算出来的
```

三个分支，缺一不可：

* **span == declared × numel** ⇒ 采用（绝大多数张量）；
* **span != 声明值** ⇒ **`ERR_PARSE`，报出两个数**，不许静默取其一；
* ⚠️ **张量区间重叠**（`begin + size > next_begin`，即两个名字指向同一段字节）⇒ **这是合法的，
  不是错误**：GGUF 里存在共享/别名张量（例如 MTP 多头预测层与主干共享权重）。此时"用偏移差
  反推大小"**必然算错**。

最后一条不是推测：`will2469/disk-streaming-moe-engine`（DISMOEN）在真模型上撞到过 —— 它的
`M0`/后续里程碑里有一条"GGUF MTP Deduplication：第 0 层 MTP 张量重叠"，处理前后**余弦相似度
从 0.004 恢复到 0.9985+**。也就是说：把重叠张量按"偏移差"当作大小来读，模型不是"精度差一点"，
而是**完全错**（余弦 0.004 = 与随机同价）。

⇒ 修正后的口径：**主判据用 type id 声明的字节数，偏移只做校验**（与账本原话正好对调）：

1. 大小按 `declared × numel` 算；
2. 用偏移区间做**重叠检测**：相交的区间登记为别名，不参与"span 必须等于大小"的校验；
3. 对非别名的张量，`span` 与声明值不符 ⇒ `ERR_PARSE`。

这是"两个方向互为校验"，而不是"只信其一"。账本里那句"不信任 type id"仍成立 —— 它反的是
**盲信**，不是反"把它当作待校验的声明"。

### 4.4 量化张量：直通还是重新量化（**最大的设计选择，两案并列**）

* **A 案·直通**：GGUF 里的 `Q4_0 / Q8_0` 张量直接把块流指针交给现有通路，跳过
  `enable_q4` / `quantize_q8_0`。
  好处：不加载 2 GB fp32、不做加载期量化（省一次全量扫描）。
  代价：**缩放因子选法不是我们的**（llama.cpp 用 `amax/7` 一类，alofa 用 MSE 最小二乘），
  质量档位会变 ⇒ 整网一致率必须重跑（`test_q4_mub_greedy` 现 0.8672/门 0.85、
  `test_q8_head_greedy` 0.8672），而且**质量不可与自家量化档直接比较**（与 alofa 对
  llama.cpp 的既有口径一致：速度可比、质量不可比）。
* **B 案·重量化**：只把 GGUF 当容器，读到 fp32（F16/BF16 放宽后）再走自家量化。
  好处：质量口径与今天完全一致，门基线不动。
  代价：丢掉"直接吃别人量化模型"这个最大的动机。

建议：**先 B 后 A**。B 案改动小、门不用动，先把"能读 GGUF 并跑通一个真模型"落到
`verified`；A 案单独立一条账本行，配自己的质量门与"缩放因子选法不同"的说明。

### 4.5 dtype 覆盖矩阵（开工前定死，不许边写边加）

| GGML type | 本方案 | 说明 |
|---|---|---|
| F32 | 支持 | 直通 `TensorFile` 兼容路径 |
| BF16 | 支持 | 复用 `bf16_to_f32`（safetensors.mojo:123），**就地放宽到调用方的 arena** |
| F16 | **本方案不做** | 需新写 `f16_to_f32`，而账本现在把"拒绝 F16"当负向对照 ⇒ 单独立项 |
| Q8_0 | B 案支持（解量化）/ A 案直通 | 块布局与自家一致（`dequant_q8_0`） |
| Q4_0 | 同上 | 与自家 q4_0 一致 |
| Q4_K / Q5_K / Q6_K / Q2_K / Q3_K / Q8_1 / IQ 系列 | **明确拒绝并报名字** | `scripts/dump_q4k_reference.py` 已有参照物，但解量化在账本里仍是 `missing` ⇒ 不在本方案范围 |

"未支持"必须是**报错并点名**，不是静默跳过 —— 否则一个 Q4_K 模型会被当成空目录加载成功，
然后在第 100 个 token 上给出垃圾。

## 5. 判据

参照物：`scripts/dump_gguf_reference.py`（Python，**独立另写**，与 `dump_q8_reference.py` 同
构），用 `struct` 解析同一个 GGUF 文件，导出张量目录 TSV（name / shape / type id / offset /
numel / bytes_per_value）+ 若干张量的解量化值。

门（新 `tests/unit/test_gguf_dir.mojo`）：

1. **目录逐项相等**：名字集合、形状、type id、偏移、numel 与参照物全等。
2. **解量化零容差**：对 F32 / Q8_0 / Q4_0 张量，前若干值与解量化参考**逐位相同**
   （与 q4_0、q8_0 那两行账本同一套理由：码与 fp16 缩放都精确，差一点就是读错字节）。
3. **三条常驻负向对照**：
   * ① **对齐漏算**：构造一个（或取现成的）需要 padding 的文件，故意按"无对齐"算偏移 ⇒
     必须与参照物不符（否则第 1 条门根本没在比偏移）；
   * ② **metadata 跳过写错**：在文件头部放一个 ARRAY 类型的键，跳过逻辑算错字节数 ⇒
     下一个张量名必须解析错 ⇒ 门必须红；
   * ③ **type id 与偏移不符**：人工改一个张量的 type id（不动数据）⇒ 必须 `ERR_PARSE`，
     且**不许**照 type id 继续加载。
4. **未支持类型必须报错**：喂一个 Q4_K 张量 ⇒ 抛错且错误信息里点名该类型。
5. **重叠张量必须被识别为别名，不是报错**（§4.3 那条反例的门）：构造/取一个含 MTP 重叠的
   文件，重叠区间的两个名字都必须能取到**同一段字节且解出相同的值**；若按"偏移差当大小"
   读，余弦会掉到 0.004 量级 ⇒ 这条门专抓它。
6. **解析阶段不许碰 payload**（借自 DISMOEN 的 M0 硬门，见 §8）：`gguf.mojo` 的索引阶段
   **只允许读 `[0, data_base)`**；用 `pread` 的调用记录做硬门（越界 = FAIL）。
   ⚠️ `/proc/<pid>/io` 的 `read_bytes` **只作观测，不作判据** —— 内核 readahead 与 page cache
   让精确字节预算不可靠，那句理由 DISMOEN 写得很清楚，直接采用。
7. **fuzz fixture 集**（同样借自 DISMOEN 的 `fixtures/m0-fuzz/`）：空文件、截断、`huge_len`、
   `short_prefix`、非 JSON、`begin > end`、负数字段、长度溢出文件大小、`hole`、`overlap` …
   共十余个畸形输入，每个都必须给出**明确的错误码**而不是崩溃或静默成功。

端到端门（B 案做完后）：沿用 `test_real_model_dir.mojo` 那套（6/6）：prompt id 全等、
16 token 教师强制 greedy 逐 token 相等、末位 logits 余弦 ≥ 0.999 且 argmax 相等。

fixture：由同一份 f32 GGUF 经 `llama-quantize` 转出（本机已跑通过这条链）。⚠️ 不往仓库复制
第二个 GB —— 照 `build_real_model_dir.py` 用符号链接拼。

## 6. 步骤（有序）

1. `scripts/dump_gguf_reference.py` + 导出 f32 与 q8_0 两份小 fixture（小模型，
   仓库里放链接不是文件）。
2. `src/alofa/model/gguf.mojo`：头部 + metadata 跳过 + 张量目录（**不读张量数据**）。
3. `tests/unit/test_gguf_dir.mojo`：§5 的 1–3 条（4 条随类型支持推进）。
4. 张量读取：F32 直通 + BF16 放宽（走 `TensorFile` 兼容路径）。
5. 端到端：`QwenForward` 从一个 GGUF 目录加载并跑通（B 案）。
6. 账本：GGUF 读取 `missing` → `partial`（元数据 + F32/BF16）；A 案另起一行。
7. F16 与 Q4_K：**各自单独立项**，不并入本方案。

## 7. 风险与边界

* **GGUF 版本与对齐常量**会变 ⇒ 版本打印出来、对齐优先取文件声明的。
* **metadata 的 value type 表**是本方案最容易错的地方（ARRAY 递归、string 无终止符、
  UTF-8 边界）⇒ §5 负向对照 ② 必须为它服务。
* **不许因为"能解析"就宣称"能跑"**：解析通了只等于目录对，端到端门（步骤 5）才等于能跑。
* RSS：GGUF 走 mmap 只读，与现有权重加载一致；但**若走 A 案直通**，常驻量口径要重新过一遍
  `test_memory_gate.mojo`（1.15× 那道门）—— 那是"权重被多留一份"的守门，直通路径会绕过
  arena 量化，可能改变常驻构成。

## 8. 外部借鉴：本方案里哪些判据是抄来的

不抄代码，只抄判据。两处来源，都已在正文对应位置标注：

**`will2469/disk-streaming-moe-engine`（DISMOEN，Apache-2.0，Mojo 1.0.0，活跃）** —— 与 alofa
同构：纯 Mojo 从零写、**oracle-first**（逐层对 PyTorch fp32 参照）、里程碑式文档。抄它三条：

* **"解析阶段不许读 payload"** 这条硬门（它的 M0：`check-index` 只允许读 `[0, data_base)`，
  越界即 FAIL）；
* **"/proc 的 read_bytes 只作观测不作判据"** 这条边界（内核 readahead 让字节预算不可靠）；
* **fuzz fixture 集**（十余个畸形输入，每个都必须给明确错误码）。

⚠️ 它那条 **GGUF MTP 张量重叠**（余弦 0.004 → 0.9985）直接改掉了本方案 §4.3 的写法 —— 这是
本次调研唯一一处**推翻了既有设计**的外部证据。

**`phunck/MojoStream`（Apache-2.0，Mojo 0.26.2，2026-05 后停更）** —— 抄它两条**想法**，
不抄代码（它的语法是 Mojo 0.26 的 `alias` / `UnsafePointer` / `MutAnyOrigin`，与 1.0 不通）：

* **ShapeGuard / TensorGuard + checksum**：加载期对每个张量做形状守卫与校验和，真权重 12/12
  通过并给出 checksum。与 alofa"不信任 type id"是同一精神，但**多了一层值级校验**，
  值得加进本方案（尤其是 A 案直通：别人的块流更该校验）；
* **P95 benchmark harness + JSON 输出**：alofa 现在只报中位数与 min…max，而 PR 级性能回归门在
  账本里是 `missing` ⇒ P95 是更合适的统计量。

⚠️ 它的性能数字**不可采信**：声称 i7-7500U（2 核）4 线程上 `matmul_q4_bpack` 跑到
**128 GFLOPS**，而该 CPU 的 AVX2 FMA 理论峰值约 86 GFLOPS —— **超过峰值**。这正是 alofa 的
roofline 门要拦的那类数字（账本原话：峰值探针自身会漂，跨过 1000‰ 就写"量不出结论"）。
