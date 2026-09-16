# alofa

an inference engine in pure Mojo.

> **alofa 不是一个"更快的推理引擎"，而是一个"敢于被人验证的推理引擎"。**
> 用编译期后端特化换可移植性，用事件循环绕开语言级 async 的缺失，用能力账本把"生产级"从形容词变成可执行的门。

---

## 当前状态（诚实声明）

**`v0.1.0` — 仅有地基与规划，没有任何推理能力。**

唯一被验证的东西是工具链假设。详见 [`docs/plan/capability-ledger.md`](docs/plan/capability-ledger.md) —— **该账本是本项目唯一的事实来源**，README 不得出现账本中非 `verified` 的能力。

| 已验证 | 证据 |
|---|---|
| Mojo 1.0.0 工具链可用 | `Mojo 1.0.0 (ed45d567)` |
| libc FFI（`external_call`）可用 | `tests/capability/test_libc_ffi.mojo` |
| `socket` / `epoll_create1` / `timerfd_create` / `eventfd` / `SO_REUSEPORT` | 同上，5/5 测试通过 |
| **`flare` / `json` 可导入**（含 `flare.runtime.Reactor`、`flare.http.HttpServer`） | `tests/capability/test_deps.mojo`，4/4 通过 |
| **CUDA kernel 在 A100 上数值正确** | `tests/gpu/vecadd.mojo`，实测 `c[999] = 2997.0` ✓（**仅远程验证机**） |
| **能力账本被 CI 强制** | `tests/capability/test_ledger.mojo`（7/7）；`pixi run check-ledger` |

> 未列出的一切能力（推理、KV cache、tokenizer、服务层等）**均为未实现**。请勿假设可用。

### 两台验证机

| 环境 | 用途 | 约束 |
|---|---|---|
| 开发机（i7-9700K / AVX2 / Maxwell sm_52 GPU 不可用 / ~13 GB 可用内存） | CPU 路径的端到端验证 | GPU 不可用 |
| **A100 远程验证机**（6×A100 共 280 GB / 128 vCPU / 755 GB RAM） | CUDA 正确性与性能基准 | **无外网**（须 rsync 同步 `.pixi`）；**共享机，只能用空闲卡** |

```bash
./scripts/a100.sh gpu                            # 挑一张空闲卡（务必先看，0-3 常满载）
./scripts/a100.sh sync                           # 同步项目（含 .pixi）
./scripts/a100.sh run 4 tests/gpu/vecadd.mojo    # 在 GPU 4 上跑 kernel 冒烟测试
```

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
