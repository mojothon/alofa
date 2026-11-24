# Usage

Two serving paths share one OpenAI-compatible API: a **CPU server**
(`src/alofa/serve.mojo`, scalar/AVX2 backends) and a **GPU server**
(`src/alofa/serve_gpu.mojo`, single-session batch pool on CUDA).

## CPU server

```bash
pixi run serve                                      # builds + runs on 127.0.0.1:8000
ALOFA_WORKERS=4 ALOFA_HOST=0.0.0.0 pixi run serve   # 4 workers, SO_REUSEPORT
```

| Variable | Default | Meaning |
|---|---|---|
| `ALOFA_WEIGHTS` | `tests/fixtures/qwen2.5-0.5b-hf` | weights directory (see [install](install.md#weights)) |
| `ALOFA_HOST` / `ALOFA_PORT` | `127.0.0.1` / `8000` | bind address |
| `ALOFA_WORKERS` | `1` | master + forked workers (COW shares weights) |
| `ALOFA_ENGINE_THREADS` | `1` | concurrent generations on the CPU path |
| `ALOFA_MAX_TOKENS` | — | **context window** (prompt + generation slots), not a generation cap |
| `ALOFA_GRACE_MS` | — | graceful-shutdown window for `SIGTERM` |

## GPU server

```bash
pixi run mojo run -O2 -I src src/alofa/serve_gpu.mojo
```

| Variable | Default | Meaning |
|---|---|---|
| `ALOFA_PORT` | `8000` | bind port |
| `ALOFA_GPU_SESSIONS` | `32` | concurrent batch slots (the pool holds 64; 32 is the measured sweet spot) |
| `ALOFA_KV_PREFIX` | `1` | KV prefix reuse — exact match + chunk-chain LCP |
| `ALOFA_SPEC_SAMPLE` | `1` | sampled verification (temperature axis) |
| `ALOFA_RUN_SECONDS` | — | auto-shutdown after N seconds (test harness) |

This path stacks **batched ticks + speculative decoding + prefix reuse**;
see [principles](principles.md).

## OpenAI-compatible API

Non-streaming:

```bash
curl -s localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

Streaming (SSE, one token per `data:` frame, closed by `data: [DONE]`):

```bash
curl -sN localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

The official `openai` SDK works with just a `base_url` change; streamed and
non-streamed output are verified byte-identical.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "hi"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Health and telemetry

```bash
curl -s localhost:8000/health
```

The GPU server reports throughput-relevant counters: `ticks`,
`spec_ticks`, `spec_accepted`, `spec_drafted`, `prefix_hits`,
`prefix_rows`, `chunk_hits` — use them to confirm each mechanism is
actually engaged (the ledger gates do exactly this).

## One-shot generation and load testing

```bash
pixi run generate                 # greedy + sampled demo with fixed seed
pixi run stress -- --url http://127.0.0.1:8000 \
  --concurrency 8 --duration 60 --max-tokens 120 --stream --temperature 0.7
```

The stress report prints QPS, latency percentiles, **TTFT (first-frame)
p50/p90/p99**, error classes and fd/memory deltas. Numbers from a shared
host are only valid as same-round adjacent-pair comparisons.
