# alofa

[![CI](https://github.com/mojothon/alofa/actions/workflows/ci.yml/badge.svg)](https://github.com/mojothon/alofa/actions/workflows/ci.yml)

an inference engine in pure Mojo.

> **alofa 不是一个"更快的推理引擎"，而是一个"敢于被人验证的推理引擎"。**
> 用编译期后端特化换可移植性，用事件循环绕开语言级 async 的缺失，用能力账本把"生产级"从形容词变成可执行的门。

---

## 当前状态

**到下面这些东西为止的能力，是被测试证明过的；没有列出来的，一律当作还没有。**

[`docs/plan/capability-ledger.md`](docs/plan/capability-ledger.md) 是本项目**唯一的事实来源**。
README 不得出现账本里未标 `verified` 的能力；账本的分级由 `pixi run check-ledger` 在 CI 里强制执行
（它自己也有一条指向自己的证据，账本用它自己的门来证明自己可信）。

### 已验证（`verified`，均可复现）

| 能力 | 证据 |
|---|---|
| Mojo 1.0.0 工具链 | `Mojo 1.0.0 (ed45d567)` |
| libc FFI、平台原语（socket / epoll / timerfd / eventfd / `SO_REUSEPORT`） | `tests/capability/test_libc_ffi.mojo` |
| `flare` / `json` 可导入（`flare.runtime.Reactor`、`flare.http.HttpServer`） | `tests/capability/test_deps.mojo` |
| **CUDA kernel 在 A100 上数值正确**（`c[999] = 2997.0`） | `tests/gpu/vecadd.mojo`（**仅远程验证机**） |
| 标量后端 `scalar` (fp32) / 向量后端 `avx2` | `tests/unit/test_layer0_parity.mojo`、`tests/unit/test_avx2_parity.mojo` |
| L0 地基：张量视图、arena、mmap、具名错误、结构化日志、分层守门 | `tests/unit/test_core_*.mojo`、`tests/capability/test_layering.mojo` |
| 算子：RMSNorm / SwiGLU / RoPE / GQA 注意力（非分页 + 分页）/ matmul（fp32、q4 融合） | `tests/unit/test_layer0_parity.mojo`、`tests/unit/test_paged_attention.mojo`、`tests/unit/test_q4_parity.mojo` |
| q4_0 量化：解量化、量化步、**整网 q4_0 前向通路**（24 层投影走块流） | `tests/unit/test_q4_parity.mojo`、`tests/unit/test_q4_greedy.mojo` |
| Qwen2.5 0.5B fp32 架构：prefill + greedy 逐 token 与 HuggingFace 一致 | `tests/unit/test_model_parity.mojo` |
| **权重常驻内存门**（加载 + 生成 32 token 后峰值 RSS ≤ 1.15× 权重；实测 1.006×） | `tests/unit/test_memory_gate.mojo` |
| KV：物理块池 + 分页视图 + 基数树前缀视图，三视图共享 refcount；前缀分裂零拷贝 | `tests/unit/test_kv_pool.mojo`、`tests/unit/test_kv_room.mojo` |
| 调度：纯函数 `step(state, input) -> Action`、chunked prefill、抢占、延迟护栏、trace 录制与重放 | `tests/unit/test_scheduler.mojo` |
| 批处理：批张量池、批执行器、**批大小 1/2/4/8 与单请求逐 token 相同** | `tests/unit/test_batch_pool.mojo`、`tests/unit/test_batch_executor.mojo`、`tests/unit/test_batch_forward.mojo` |
| 引擎循环：调度器 ↔ 批执行器接线、缓存让位、共享前缀只算未命中段、抢占活锁修复 | `tests/unit/test_engine_core.mojo` |
| 采样器：top-k / top-p / min-p / 温度 / 重复惩罚 / logit bias，分布经卡方与 TVD 检验 | `tests/unit/test_sampler_parity.mojo` |
| Tokenizer：预分词器（无正则依赖）、NFC 归一化、BPE、added token、id→文本还原，**4560 条差分与 HF 逐 id 一致** | `tests/unit/test_tokenizer_parity.mojo` |
| **垂直切片**：一句真文本走完 tokenizer → model → engine → sampler（真权重、端到端） | `tests/unit/test_vertical_slice.mojo`<br>`pixi run test-slice` / `pixi run generate` |

### 还没有（`missing` / `hardware-blocked`，请勿假设可用）

这些同样是事实的一部分，写在这里是为了避免"看起来已经能用"：

- **服务层：有 OpenAI 兼容 API 与 reactor 事件循环，但有几处明文边界。** HTTP + OpenAI 兼容 API（非流式 + SSE 流式）已有；入口跑 **reactor 事件循环**（一条循环 N 条连接 —— 慢客户端 / keep-alive 的空闲连接不再互相挡路，有背压队列与空闲回收），生成不在循环上（engine 线程）。⚠️ 同时能生成几条 = engine 线程数，**默认 1**（每多一条就多一份权重，`ALOFA_ENGINE_THREADS`）；无 TLS；无批调度（把多条生成合成一次前向 —— 那才是**不**多占权重的并发生成）。
- ⚠️ 垂直切片只验 **1 条请求 / 16 个 token / scalar 后端**：不验 AVX2、不验批、不验流式输出的 UTF-8 边界，也不验生成质量。它只负责"链路通"。
- **权重形态只认两种。** HF 的 `config.json` + safetensors（bf16 **就地放宽**成 fp32）可以直接读；本机导出的 fp32 裸二进制 + TSV 索引（`scripts/dump_model_reference.py`）也还在。GGUF 不支持。
- **只有一个模型架构**（Qwen2.5 0.5B，fp32）；Llama / Mistral 未开始。
- **量化只有 q4_0 一种格式**（q4_k / q8_0 / int8 / fp8 未开始），且整网 q4_0 的教师强制贪心一致率实测 **0.8164**，未达到自己设的 ≥0.90 质量门 → 这条在账本里按 `missing` 如实记录。
- **没有任何性能数字。** 唯一被接受的性能基准是同机同 prompt 对比 llama.cpp，尚未做。

### 两台验证机

| 环境 | 用途 | 约束 |
|---|---|---|
| 开发机（i7-9700K / AVX2 / Maxwell sm_52 GPU 不可用 / ~13 GB 可用内存） | CPU 路径的端到端验证 | 无 AVX-512，本地 GPU 不可用 |
| **A100 远程验证机**（6×A100 共 280 GB / 128 vCPU / 755 GB RAM） | CUDA 正确性与性能基准 | **无外网**（须 rsync 同步 `.pixi`）；**共享机，只能用空闲卡** |

```bash
./scripts/a100.sh gpu                            # 挑一张空闲卡（务必先看，0-3 常满载）
./scripts/a100.sh sync                           # 同步项目（含 .pixi）
./scripts/a100.sh run 4 tests/gpu/vecadd.mojo    # 在 GPU 4 上跑 kernel 冒烟测试
```

## 怎么用

```bash
pixi run test             # 全部套件（约 4.5 min），末尾自动跑账本门
pixi run check-ledger     # 只校验账本：每个 verified 都要有真实证据
pixi run check-counts     # 校验账本里的 ?count=N 是否等于本次实测条数（须在 test 之后）
pixi run ledger-sync      # 套件条数变了？一条命令把账本里的 ?count=N 刷新回来
```

重资产门（要 2 GB 权重并先 `mojo build -O2`，所以**不并进 `pixi run test`**）：

```bash
pixi run test-model           # 整网 Qwen2.5 0.5B 数值门（prefill + 128 token greedy）
pixi run test-model-avx2      # 同上，向量后端
pixi run test-q4              # 整网 q4_0 教师强制贪心一致率
pixi run test-memory          # 权重常驻内存门
pixi run test-batch-forward   # 批一致性（1/2/4/8 与单请求逐 token 相同）
pixi run test-backend-guard   # 编译期红测：拼错的后端常量必须编译失败
pixi run test-slice           # 垂直切片门（需 1.9 GB 权重）
```

### 跑一句真文本

```bash
pixi run generate
```

```
[greedy]  The capital of France is
          → " Paris. It is the largest city in Europe and the second largest in the world"
[sampled] → ": A: Paris B: not sure C: london D: BERLIN"      # 温度 1.0，seed 固定
```

两条路都说到 Paris。它们**必须**在温度趋零时逐字相等 —— 若 `run_sampled` 悄悄退化成
argmax，"说出 Paris"照样全绿而采样路径一次都没生效过，所以那条才是这道门的关键断言。

### 跑服务（OpenAI 兼容，非流式 + SSE 流式）

```bash
pixi run serve                                      # 单进程，127.0.0.1:8000
ALOFA_WORKERS=4 ALOFA_HOST=0.0.0.0 pixi run serve   # 4 worker，SO_REUSEPORT
curl -s localhost:8000/health                       # 就绪探针（含边界说明）
curl -s localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

- 配置走**环境变量**（Mojo 1.0 编译产物里 `sys.argv` 是空的），全量清单与默认值见
  `src/alofa/srv/config.mojo`。注意 `ALOFA_MAX_TOKENS` 是**上下文窗口**（prompt+生成
  总槽位），不是生成步数上限；`ALOFA_ENGINE_THREADS` 是同时能生成的条数（默认 1 ——
  每多一条就多一份权重，那是线程池这条路的代价）。
- 多 worker：master 加载一次权重后 fork，worker 靠写时复制共享（8 worker 的实际内存
  ≈ 单进程）；**worker 内分片请保持 1** —— fork 之后 asyncrt 不可用（账本 §8 有实测）。
- 流式：`stream=true` 走 SSE（`text/event-stream`，逐 token 一帧，以 `data: [DONE]`
  收尾并关连接 —— 没有 `Content-Length`，长度在写的那一刻未知）。官方 `openai`
  SDK 只改 `base_url` 即可流式对话；实测流式拼出来的文本与非流式**逐字相等**
  （多字节字符可能跨 token，所以每步是「整段前缀解码后取新增的那一截」）。
- 优雅退出：`SIGTERM` → 停止接新请求 → 在途答完 → 退出；systemd 部署模板见
  `scripts/deploy/alofa.service`（`TimeoutStopSec` 要大于 `ALOFA_GRACE_MS`）。
- 压测观测：`pixi run stress -- --url http://127.0.0.1:8000 --concurrency 8 --duration 60`
  （QPS/分位数/错误分类/fd 与内存；本机过载数字只作同轮相对比较）。

### 给账本加一条新能力

新增套件后：① `scripts/run_tests.sh` 里登记（含编译参数）② 若文件在 `core/` 下，同步
`tests/capability/test_layering.mojo` 的文件清单 ③ `pixi run ledger-sync` 刷新 `?count=`。

账本里的 N/M 快照（形如 `evidence:…mojo?count=17`）由 `check-counts` 逐项核对 —— **写数字的地方只有这一个**，
其余地方不再手写"（15/15 通过）"式的描述，因为那类描述会腐烂，而账本的职责是不腐烂。

## 规划文档

| 文档 | 内容 |
|---|---|
| [生态与竞品分析](docs/plan/01-ecosystem-analysis.md) | 14 个 Mojo 参考项目的评价、vLLM / SGLang / LM Studio 六维对比、MAX 的能力与缺陷、Mojo 1.0 能力边界实测 |
| [架构设计](docs/plan/02-architecture.md) | 分层架构、统一 KV 寻址、可重放调度器、编译期后端特化、事件循环服务层、验证体系 |
| [路线图与验收门](docs/plan/03-roadmap.md) | P0–P5 阶段划分与硬性验收门 |
| [能力账本](docs/plan/capability-ledger.md) | 每个能力的证据分级（`verified` / `scaffold` / `designed-only` / `hardware-blocked` / `missing`） |

### 核心设计取舍（一句话版）

- **不做 async，也不自研事件循环**：Mojo 1.0 无语言级 async；实测 libc FFI 全通（自研可行），但生态已有纯 Mojo 的 `flare` reactor → **直接依赖**，并把 reactor 线程与 GPU 执行线程分离。
- **KV 只有一套物理池**：分页视图与基数树视图共享同一 `BlockPool` 与 `refcount`，token 粒度前缀映射到 block 粒度存储（节点分裂零拷贝）。
- **调度器是纯函数**：`step(state, input) -> Action`，`Action` 可序列化 → 调度边界可脱离模型被测试。
- **后端在编译期特化**：一份 kernel 源码用 `comptime` 参数化 ISA / 量化格式 / 块形状，运行期只派发一次。
- **不重复造轮子**：`flare`（网络/运行时）、`json`（SIMD 解析 + 反射序列化）、`mojo.core`（stdlib 缺口）能解决的直接依赖；**稀缺工时只投在无人做过的引擎内核（L0–L4）**。
- **不产出未经验证的性能数字**：开发机 GPU 为 Maxwell（sm_52）不可用，但 A100 验证机已打通 → CUDA 可验正确性；性能数字须在该机测得、报 roofline 利用率并标注共享环境。

## package build

```bash
pixi run mojo precompile src/alofa/ -o alofa.mojoc
```
or 
```bash
make package
```

## test

```bash
pixi run test 
```

or 

```bash
make test
```

## add new package

```bash
pixi add alofa --platform linux-64
```

## publish package

```bash
make upload
```

## version

version 0.1.0
