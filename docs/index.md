# alofa

an inference engine in pure Mojo.

> **alofa 不是一个"更快的推理引擎"，而是一个"敢于被人验证的推理引擎"。**

## 当前状态

**`v0.1.0` — 仅有地基与规划，没有任何推理能力。**

唯一被验证的是工具链假设。**能力账本（[capability-ledger](plan/capability-ledger.md)）是本项目唯一的事实来源。**

| 已验证 | 证据 |
|---|---|
| Mojo 1.0.0 工具链 | `Mojo 1.0.0 (ed45d567)` |
| libc FFI（`socket` / `epoll_create1` / `timerfd_create` / `eventfd` / `SO_REUSEPORT`） | `tests/capability/test_libc_ffi.mojo`，5/5 |
| `flare` / `json` 可导入（含 `flare.runtime.Reactor`、`flare.http.HttpServer`） | `tests/capability/test_deps.mojo`，4/4 |
| CUDA kernel 在 A100 上数值正确 | `tests/gpu/vecadd.mojo`，`c[999] = 2997.0` ✓（仅远程机） |

### 两台验证机

| 环境 | 用途 | 约束 |
|---|---|---|
| 开发机（i7-9700K / AVX2 / Maxwell sm_52 GPU 不可用） | CPU 端到端验证 | GPU 不可用 |
| **A100 远程验证机**（6×A100 280 GB / 128 vCPU / 755 GB RAM） | CUDA 正确性与性能基准 | 无外网（rsync 同步 `.pixi`）；**共享机，只用空闲卡** |

```bash
./scripts/a100.sh gpu                            # 挑空闲卡（0-3 常满载）
./scripts/a100.sh sync                           # 同步项目（含 .pixi）
./scripts/a100.sh run 4 tests/gpu/vecadd.mojo    # GPU kernel 冒烟测试
```

## 规划文档

| 文档 | 内容 |
|---|---|
| [生态与竞品分析](plan/01-ecosystem-analysis.md) | 14 个 Mojo 参考项目评价、vLLM / SGLang / LM Studio 对比、MAX 能力与缺陷、Mojo 1.0 边界实测 |
| [架构设计](plan/02-architecture.md) | 分层架构、统一 KV 寻址、可重放调度器、编译期后端特化、事件循环服务层 |
| [路线图与验收门](plan/03-roadmap.md) | P0–P5 阶段与硬性验收门 |
| [能力账本](plan/capability-ledger.md) | 证据分级与 CI 强制校验规则 |

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

## version

version 0.1.0
