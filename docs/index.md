# alofa

an inference engine in pure Mojo.

> **alofa 不是一个"更快的推理引擎"，而是一个"敢于被人验证的推理引擎"。**
> 用编译期后端特化换可移植性，用事件循环绕开语言级 async 的缺失，用能力账本把"生产级"从形容词变成可执行的门。

## 当前状态

**Qwen2.5 0.5B 在 A100 上跑通完整服务栈：OpenAI 兼容 API + 批调度 + 投机解码 + KV 前缀复用，三机制生产口径全部实测为正，最差交叉格 1.19×，无惩罚格。**

[`plan/capability-ledger.md`](plan/capability-ledger.html) 是本项目**唯一的事实来源**：每个能力都有证据分级（`verified` / `scaffold` / `designed-only` / `missing`），CI 里的 `check-ledger` 门强制「账本里没有的，一律当作还没有」。

### 已验证的主干（均带可复现门）

| 层 | 能力 | 证据 |
|---|---|---|
| 工具链 | Mojo 1.0.0 / libc FFI / flare reactor / CUDA kernel | `tests/capability/`、`tests/gpu/vecadd.mojo` |
| L0-L2 | 张量视图、arena、算子（RMSNorm/SwiGLU/RoPE/GQA/分页注意力）、q4_0、采样器、tokenizer（4560 条 HF 差分逐 id 一致） | `tests/unit/` |
| 模型 | Qwen2.5 0.5B fp32：prefill + greedy 与 HuggingFace 逐 token 一致；权重常驻内存 1.006× | `test_model_parity.mojo`、`test_memory_gate.mojo` |
| 引擎 | 纯函数调度器、chunked prefill、抢占、批一致性（1/2/4/8 ≡ 单请求） | `test_scheduler.mojo`、`test_batch_forward.mojo` |
| 服务（CPU） | OpenAI 兼容 API（非流式 + SSE）、reactor 事件循环、多 worker fork 共享 | `test-engine`、soak |
| **服务（GPU）** | 单 session 64 槽批行池、CUDA 异步前向、23 个 GPU 门 | `tests/gpu/test_*.mojo` |
| **投机解码** | n-gram 草稿 + 批内验证 + 自适应预算；采样验证打通温度轴 | 账本 174-175 系列 |
| **KV 前缀复用** | 精确匹配 + chunk 链 LCP，槽间 D2D 拷贝 | 账本 176 系列 |

### 性能（A100，同轮紧邻对相对值）

| 流量形态 | 结果 |
|---|---|
| 精确重复 × 贪心 | spec 全谱系 1.6-2.7× |
| temp 0.7 × 全 R 谱系 | 1.19×（R=0）→ 2.29×（R=1.0），接受率 = q(草稿) 逐点吻合 |
| 精确重复 × c=8 × spec+prefix | 吞吐 ~6.8×（对照无 spec 无前缀基线） |
| 共享 preamble（全唯一问题） | 1.6-2.2×（精确匹配域零命中的流量首次吃到红利） |
| c=32 高并发 | 坍缩救回（1.65× → 1.9×），吞吐触顶 ~4100 tok/s，c=32 即天花板 |
| **TTFT（首帧）** | c=8：243ms → **22.3ms**（p50 −91%）；c=32：581ms → 102ms |

### 还没有

- 多架构（Llama / Mistral）、GGUF、q4 之外的量化格式、llama.cpp 对比基线
- GPU 服务路的持续 soak 与干净窗口绝对值重测（运营项）

## 文档

| 文档 | 内容 |
|---|---|
| [能力账本](plan/capability-ledger.html) | 每个能力的证据分级与完整实验曲线（唯一事实来源） |
| [生态与竞品分析](plan/01-ecosystem-analysis.html) | 14 个 Mojo 参考项目、vLLM / SGLang / LM Studio 对比、Mojo 1.0 边界实测 |
| [架构设计](plan/02-architecture.html) | 分层架构、统一 KV 寻址、可重放调度器、编译期后端特化 |
| [路线图与验收门](plan/03-roadmap.html) | P0-P5 阶段与硬性验收门 |
| [服务压测报告](plan/serving-report.html) | CPU 服务路压测口径与结论 |
| [批处理报告](plan/batching-report.html) | GPU 批行池设计报告 |
| [投机评测](plan/speculative-eval.html) | spec 机制的评测协议 |

## 仓库

<https://github.com/mojothon/alofa> —— 本站由仓库 `docs/` 自动构建（GitHub Actions Pages）。
