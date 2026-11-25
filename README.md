# alofa

[![CI](https://github.com/mojothon/alofa/actions/workflows/ci.yml/badge.svg)](https://github.com/mojothon/alofa/actions/workflows/ci.yml)
[![Pages](https://github.com/mojothon/alofa/actions/workflows/pages.yml/badge.svg)](https://mojothon.github.io/alofa/)
[![Version](https://img.shields.io/badge/version-0.3.0-blue)](docs/version.md)
[![Mojo](https://img.shields.io/badge/Mojo-1.0.0-orange)](https://mojolang.org)

**an inference engine in pure Mojo.**

> **alofa 不是一个"更快的推理引擎"，而是一个"敢于被人验证的推理引擎"。**
> 用编译期后端特化换可移植性，用事件循环绕开语言级 async 的缺失，用能力账本把"生产级"从形容词变成可执行的门。

📕 **完整文档**：<https://mojothon.github.io/alofa/>（安装 / 使用 / 原理 / 架构 / 开发指南）

---

## 目录

- [特性一览](#特性一览) · [性能](#性能) · [快速开始](#快速开始) · [使用](#使用)
- [架构与设计取舍](#架构与设计取舍) · [已验证能力](#已验证能力) · [还没有](#还没有)
- [文档](#文档) · [开发](#开发) · [版本](#版本)

---

## 特性一览

| | |
|---|---|
| **模型** | Qwen2.5-0.5B，fp32 / q4_0 |
| **后端** | CPU：`scalar` / `avx2`（编译期特化）· GPU：CUDA via AsyncRT（A100 验证） |
| **服务** | OpenAI 兼容 API（JSON + SSE 流式），CPU 路与 GPU 路同一套 API |
| **并发** | GPU 单 session 64 槽批行池，批调度 + tick 引擎 |
| **吞吐三机制** | 投机解码（自适应草稿）× 采样验证（温度轴）× KV 前缀复用（精确 + chunk LCP） |
| **可信度** | 每个能力一条可复现门；[能力账本](docs/plan/capability-ledger.md) 是唯一事实来源，CI 强制校验 |

## 性能

> 全部为 **A100 共享机上的同轮紧邻对相对值**（不做跨窗口绝对比较，不当绝对吞吐外引）。

| 流量形态 | 结果 |
|---|---|
| 精确重复 × 贪心 × spec | 重复率全谱系 1.6–2.7× |
| temp 0.7 × R = 0 → 1.0 | 1.19× → 2.29×（接受率 = q(草稿)，与理论逐点吻合） |
| 精确重复 × spec + 前缀复用 | **~6.8×**（对照无 spec 无前缀基线） |
| 共享 preamble（全唯一问题） | 1.6–2.2×（部分前缀路） |
| c=32 高并发 | 坍缩救回（1.65× → 1.9×），吞吐触顶 ~4100 tok/s |
| **TTFT（首帧）** | c=8：243 ms → **22.3 ms**（p50 −91%）；c=32：581 ms → 102 ms |

最差交叉格 **1.19×**——三轴 × 交叉格全谱系为正，无惩罚格。完整曲线与判据见账本 174–176 各条。

## 快速开始

```bash
git clone https://github.com/mojothon/alofa && cd alofa
pixi run test        # 全部套件 + 账本门（~4.5 min）——绿 = 环境端到端正确
pixi run serve       # OpenAI 兼容服务，127.0.0.1:8000
```

```bash
curl -s localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

权重与 GPU 主机配置见 [`docs/install.md`](docs/install.md)。

## 使用

```bash
# CPU 路
pixi run serve                                      # 单进程，127.0.0.1:8000
ALOFA_WORKERS=4 ALOFA_HOST=0.0.0.0 pixi run serve   # 4 worker，SO_REUSEPORT（fork COW 共享权重）

# GPU 路（批调度 + spec + 前缀复用全开）
pixi run mojo run -O2 -I src src/alofa/serve_gpu.mojo

# 一句真文本（greedy + 采样，固定种子）
pixi run generate

# 压测（QPS / 延迟分位 / TTFT / 错误分类 / fd 与内存）
pixi run stress -- --url http://127.0.0.1:8000 --concurrency 8 --duration 60
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `ALOFA_WEIGHTS` | `tests/fixtures/qwen2.5-0.5b-hf` | 权重目录 |
| `ALOFA_PORT` / `ALOFA_HOST` | `8000` / `127.0.0.1` | 绑定地址 |
| `ALOFA_WORKERS` | `1` | master + fork worker（写时复制共享权重） |
| `ALOFA_ENGINE_THREADS` | `1` | CPU 路并发生成条数（每条一份权重） |
| `ALOFA_MAX_TOKENS` | — | **上下文窗口**（prompt + 生成总槽位），非生成步数上限 |
| `ALOFA_GPU_SESSIONS` | `32` | GPU 并发槽位（池 64 槽，实测 32 为甜点档） |
| `ALOFA_KV_PREFIX` / `ALOFA_SPEC_SAMPLE` | `1` / `1` | 前缀复用 / 采样验证开关 |

流式：`stream=true` 走 SSE（逐 token 一帧，`data: [DONE]` 收尾）；官方 `openai` SDK 只改 `base_url` 即可，实测流式与非流式**逐字相等**。优雅退出：`SIGTERM` → 停接新请求 → 在途答完 → 退出（systemd 模板 `scripts/deploy/alofa.service`）。配置全量清单见 `src/alofa/srv/config.mojo`（Mojo 编译产物里 `sys.argv` 是空的，配置走环境变量）。

## 架构与设计取舍

```
srv/          HTTP + OpenAI API + 事件循环 + 双服务路
engine/       纯函数调度器、批池/批执行器、KV 视图
gpu/          GpuSession（设备缓冲、流）、spec_draft
kernels/      comptime 特化：cpu/（scalar、avx2）、cuda/（AsyncRT）
model/        Qwen2.5 架构接线（prefill、decode）
tokenizer/    预分词器 + BPE（与 HF 逐 id 一致）
core/         张量视图、arena、mmap、具名错误、结构化日志
```

依赖单向：`core → kernels → model → engine → srv`，分层由门强制（`test_layering.mojo`）。核心取舍（一句话版）：

- **不做 async，也不自研事件循环**：生态已有纯 Mojo 的 `flare` reactor → 直接依赖，reactor 线程与执行线程分离。
- **KV 只有一套物理池**：分页视图与基数树视图共享同一 `BlockPool` 与 refcount，前缀分裂零拷贝。
- **调度器是纯函数**：`step(state, input) -> Action`，`Action` 可序列化 → 调度边界可脱离模型被测试。
- **后端在编译期特化**：一份 kernel 源码用 `comptime` 参数化 ISA / 量化格式 / 块形状，运行期只派发一次。
- **KV 行由位置决定**（GPU 路）：rope 相位 = 位置 → 同前缀 KV 行逐位相同，**拷贝即复用**。
- **不重复造轮子**：`flare` / `json` / `mojo.core` 能解决的直接依赖，稀缺工时只投在引擎内核（L0–L4）。

## 已验证能力

**到下面这些东西为止的能力，是被测试证明过的；没有列出来的，一律当作还没有。**

| 层 | 能力 | 证据 |
|---|---|---|
| 工具链 | Mojo 1.0.0、libc FFI（socket/epoll/timerfd/eventfd/SO_REUSEPORT）、flare/json、**CUDA kernel A100 数值正确** | `tests/capability/`、`tests/gpu/vecadd.mojo` |
| L0 地基 | 张量视图、arena、mmap、具名错误、结构化日志、分层守门 | `tests/unit/test_core_*.mojo` |
| 算子 | RMSNorm / SwiGLU / RoPE / GQA（非分页 + 分页）/ matmul（fp32、q4 融合） | `test_layer0_parity.mojo` 等 |
| 量化 | q4_0 解量化与整网前向通路（24 层投影走块流） | `test_q4_parity.mojo`、`test_q4_greedy.mojo` |
| 模型 | **Qwen2.5-0.5B prefill + greedy 与 HuggingFace 逐 token 一致**；权重常驻内存实测 1.006× | `test_model_parity.mojo`、`test_memory_gate.mojo` |
| 引擎 | 纯函数调度、chunked prefill、抢占、trace 重放；**批 1/2/4/8 与单请求逐 token 相同** | `test_scheduler.mojo`、`test_batch_forward.mojo` |
| 采样器 | top-k / top-p / min-p / 温度 / 重复惩罚 / logit bias（卡方与 TVD 检验） | `test_sampler_parity.mojo` |
| Tokenizer | **4560 条差分与 HF 逐 id 一致** | `test_tokenizer_parity.mojo` |
| 服务（CPU） | OpenAI 兼容 API（JSON + SSE）、reactor 事件循环、多 worker fork 共享 | HTTP 门 + soak |
| 服务（GPU） | 64 槽批行池、tick 引擎、**投机解码 + 采样验证 + KV 前缀复用** | 23 个 GPU 门（账本 128–176） |

## 还没有

这些同样是事实的一部分，写在这里是为了避免"看起来已经能用"：

- **CPU 服务路无批调度**（GPU 路才有）：同时能生成几条 = engine 线程数，默认 1；无 TLS。
- 垂直切片只验 **1 条请求 / 16 token / scalar 后端**——只负责"链路通"。
- 权重形态只认两种（HF safetensors 就地放宽 fp32、fp32 裸二进制 + TSV）；GGUF 不支持。
- 架构只有 Qwen2.5 一族：0.5B 全验证；**1.5B 冒烟已通**（无自动门）；Llama / Qwen3 被加载器三处硬阻断（qkv bias 必需、分片 safetensors、fp32 显存）。
- 量化只有 q4_0，整网教师强制贪心一致率实测 **0.8164**（未达自设 ≥0.90 门，账本按 `missing` 如实记录）。
- 性能数字只在 A100 远程验证机测得（共享机）；llama.cpp 对比基线尚未做。

## 文档

| 文档 | 内容 |
|---|---|
| 📕 **文档站** | <https://mojothon.github.io/alofa/> —— [安装](docs/install.md) · [使用](docs/usage.md) · [原理](docs/principles.md) · [架构](docs/architecture.md) · [开发指南](docs/developing.md) |
| [能力账本](docs/plan/capability-ledger.md) | **唯一事实来源**：每个能力的证据分级与完整实验曲线 |
| [生态与竞品分析](docs/plan/01-ecosystem-analysis.md) | 14 个 Mojo 参考项目、vLLM / SGLang / LM Studio 六维对比 |
| [架构设计](docs/plan/02-architecture.md) | 分层架构、统一 KV 寻址、可重放调度器、编译期后端特化 |
| [路线图与验收门](docs/plan/03-roadmap.md) | P0–P5 阶段划分与硬性验收门 |
| [版本历史](docs/version.md) | 0.1.0 → 0.3.0，每版绑定账本区间 |

## 开发

```bash
# 常规套件（账本纪律：数字只写在账本 evidence 快照一处，check-counts 逐项核对）
pixi run test             # 全部套件，末尾自动跑账本门
pixi run check-ledger     # 每个 verified 都要有真实证据
pixi run check-counts     # ?count=N 与实测条数核对（须在 test 之后）
pixi run ledger-sync      # 套件条数变了？刷新账本快照

# 重资产门（要 ~2 GB 权重，build -O2 后跑，不并入 test）
pixi run test-model           # 整网数值门（prefill + 128 token greedy）
pixi run test-model-avx2      # 向量后端整网门
pixi run test-q4              # q4_0 教师强制贪心一致率
pixi run test-memory          # 权重常驻内存门
pixi run test-batch-forward   # 批一致性门
pixi run test-backend-guard   # 编译期红测：拼错的后端必须编译失败
pixi run test-slice           # 垂直切片门

# 打包 / 发布
pixi run mojo precompile src/alofa/ -o alofa.mojoc   # 即 make package
pixi add alofa --platform linux-64                    # 以依赖方式安装
```

GPU 开发走共享 A100 主机（无外网，rsync 同步 `.pixi`，只用空闲卡）：

```bash
./scripts/a100.sh gpu                            # 挑一张空闲卡（0-3 常满载）
./scripts/a100.sh sync                           # 同步项目（含 .pixi）
./scripts/a100.sh run 3 tests/gpu/test_gpu_service.mojo
```

压测数字的纪律：共享机阵发租户下**只作同轮紧邻对相对比较**。门与账本的完整工作流见 [`docs/developing.md`](docs/developing.md)。

## 版本

**0.3.0**（2026-11-24）——投机解码 + KV 前缀复用时代：贪心重复率全谱系 1.6–2.7×；采样验证打通温度轴（1.19–2.29×）；前缀复用重复流量 ~6.8×、TTFT p50 243ms → 22.3ms。完整历史见 [`docs/version.md`](docs/version.md)。
