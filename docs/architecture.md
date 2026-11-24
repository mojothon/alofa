# Architecture

## Layers

```
srv/          HTTP + OpenAI API + event loop + two serving paths
  serve.mojo       CPU path: scalar/AVX2 backends, engine threads
  serve_gpu.mojo   GPU path: single session, batch pool, ticks
engine/       pure-function scheduler, batch pool/executor, KV views
gpu/          GpuSession (device buffers, streams), spec_draft
model/        Qwen2.5 architecture wiring (prefill, decode)
kernels/      comptime-specialized: cpu/ (scalar, avx2), cuda/ (AsyncRT)
tokenizer/    pretokenizer + BPE, id-exact vs HuggingFace
runtime/      weights loading (TSV dump dir, HF safetensors)
core/         tensor views, arena, mmap, named errors, structured log
verify/       reference harnesses for differential testing
```

Dependencies point one way: `core → kernels → model → engine → srv`. The
layering is enforced by a gate (`test_layering.mojo`) — a file in `core/`
importing `srv/` fails CI.

## Backend specialization at compile time

One kernel source, many specializations: ISA (`scalar` / `avx2` / CUDA),
quantization format (fp32 / q4_0) and block shapes are `comptime`
parameters resolved once at build time. There is no runtime dispatch table
to get wrong — and a **red gate** proves a misspelled backend constant
fails compilation instead of silently falling back to scalar
(`scripts/check_bad_backend.sh`).

## KV addressing

- **CPU engine**: one physical block pool, three views on top — pool view
  (allocation), paged view (blocks per sequence), radix-tree view (prefix
  sharing). Views share refcounts; prefix splits are zero-copy.
- **GPU path**: per-layer stores `[B][max_tokens][kv_dim]` for K and V.
  Slot `s`, row `p` lives at `s*mt_stride + p*kv_dim`. Because KV rows are
  position-determined, cross-slot prefix reuse is a pure memcpy
  ([principles](principles.md#kv-prefix-reuse)).

## Scheduler and event loop

The scheduler is a pure function `step(state, input) -> Action`; actions
are serializable, so scheduling boundaries are testable without a model
(trace recording and replay). The serving layer never blocks the event
loop: a **flare reactor** owns connections (with backpressure and idle
reclaim), while generation runs on the engine thread (CPU) or the tick
engine (GPU). Graceful shutdown: `SIGTERM` → stop accepting → drain → exit.

## The two serving paths

| | CPU (`serve.mojo`) | GPU (`serve_gpu.mojo`) |
|---|---|---|
| Concurrency | engine threads (default 1) | 64-slot batch pool, `ALOFA_GPU_SESSIONS` |
| Throughput mechanisms | — | ticks + spec + prefix reuse |
| Workers | fork, COW weight sharing | single process |
| Verified by | HTTP gates, soak | 23 GPU gates |

Both speak the same OpenAI-compatible API; clients cannot tell them apart.
