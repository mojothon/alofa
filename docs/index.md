# alofa

`alofa` is an inference engine written in pure [Mojo](https://mojolang.org) —
no C, no Python in the hot path, no MAX runtime dependency in the kernel
layer.

> **alofa 不是一个"更快的推理引擎"，而是一个"敢于被人验证的推理引擎"。**
> 用编译期后端特化换可移植性，用事件循环绕开语言级 async 的缺失，用能力账本把
> "生产级"从形容词变成可执行的门。

It serves **Qwen2.5-0.5B** behind an OpenAI-compatible API, on CPU (scalar /
AVX2 backends) and on NVIDIA GPUs (CUDA via AsyncRT), with three throughput
mechanisms stacked on the GPU path: batched tick execution, speculative
decoding with sampled verification, and cross-slot KV prefix reuse.

## Status

The package is developed against **Mojo 1.0.0** and exercised on Linux
x86-64 (CPU path) and a shared 6×A100 host (GPU path). The
[capability ledger](plan/capability-ledger.md) is the single source of
truth: every capability carries an evidence grade (`verified` / `scaffold` /
`designed-only` / `missing`), and CI enforces that **anything not in the
ledger must be assumed absent**.

| Layer | Verified |
|---|---|
| Toolchain | Mojo 1.0.0, libc FFI, flare reactor, CUDA kernels on A100 |
| Ops | RMSNorm / SwiGLU / RoPE / GQA (paged + dense) / matmul (fp32, q4) |
| Tokenizer | BPE + pretokenizer, **4560-case differential, id-exact vs HuggingFace** |
| Model | Qwen2.5-0.5B fp32: prefill + greedy **token-exact vs HuggingFace** |
| Engine | Pure-function scheduler, chunked prefill, preemption, batching ≡ single-stream |
| Serving (CPU) | OpenAI-compatible API (JSON + SSE), reactor event loop, multi-worker fork |
| Serving (GPU) | 64-slot batch pool, async forward, 23 GPU gates |
| Speculative decoding | n-gram drafts + in-batch verification, adaptive budget, **sampled verification** |
| KV prefix reuse | exact match + chunk-chain LCP, cross-slot D2D copy |

## Performance highlights

All numbers are same-round adjacent-pair ratios on a shared A100 host (never
absolute numbers taken out of context):

| Traffic shape | Result |
|---|---|
| Exact repetition, greedy, spec on | 1.6–2.7× across the whole repetition spectrum |
| temp 0.7, R = 0 → 1.0 | 1.19× → 2.29× (acceptance = q(draft), matches theory) |
| Exact repetition, spec + prefix | **~6.8×** vs the no-spec no-prefix baseline |
| Shared preamble, all-unique questions | 1.6–2.2× (partial-prefix path) |
| c=32 high concurrency | collapse healed (1.65× → 1.9×), ceiling ~4100 tok/s |
| **TTFT (first frame)** | c=8: 243 ms → **22.3 ms** (p50 −91%) |

Worst measured cell anywhere in the matrix: **1.19× — no penalized cell.**

## Quick start

```bash
git clone https://github.com/mojothon/alofa && cd alofa
pixi run test        # full CPU suite + ledger gate, ~4.5 min
pixi run serve       # OpenAI-compatible server on 127.0.0.1:8000
```

```bash
curl -s localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

GPU serving, spec decoding and prefix reuse are one page each:
[Install](install.md) · [Usage](usage.md) · [Principles](principles.md).

## Pages

| Page | Content |
|---|---|
| [Install](install.md) | prerequisites, pixi, weights, remote A100 host |
| [Usage](usage.md) | CPU server, GPU server, OpenAI API, health, load testing |
| [Principles](principles.md) | how batching, speculative decoding and prefix reuse actually work |
| [Architecture](architecture.md) | layering, module map, KV addressing, scheduler, backends |
| [Developing](developing.md) | adding kernels, gates and ledger entries; the GPU dev loop |
| [Ledger](plan/capability-ledger.md) | every capability with its evidence (4200+ lines) |
| [Version](version.md) | release history tied to ledger ranges |
