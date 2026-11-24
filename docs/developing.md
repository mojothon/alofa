# Developing

## Gates

The project's discipline is **gates**: every claimed capability has a
reproducible gate, and the [ledger](plan/capability-ledger.md) records the
evidence. Three tiers:

**CPU suite** (`pixi run test`, ~4.5 min) — unit + capability + HTTP gates,
closes with the ledger gate.

**Heavy gates** (need ~2 GB weights, built with `-O2`, run separately):

```bash
pixi run test-model            # whole-network parity (prefill + 128-token greedy)
pixi run test-model-avx2       # same, vector backend
pixi run test-q4               # q4_0 teacher-forced greedy agreement
pixi run test-batch-forward    # batching ≡ single stream
pixi run test-memory           # weights-resident memory gate (1.006×)
pixi run test-slice            # vertical slice (tokenizer → model → engine)
```

**GPU suite** (23 gates on the A100 host):

```bash
for g in test_session_prefill test_session_decode test_session_batch \
         test_multi_e2e test_gpu_service test_service_deep \
         test_spec_gen test_spec_sample test_spec_shared test_service_spec \
         test_service_prefix test_service_prefix_chunk; do
  ./scripts/a100.sh run 3 tests/gpu/$g.mojo
done
```

Each prints an explicit `PASS`; a gate without a PASS banner is failed —
the CI ledger gate treats claims without evidence as errors.

## Adding a gate / a ledger entry

1. Register the file in `scripts/run_tests.sh` (with its compile flags).
2. If the file lives under `core/`, add it to `test_layering.mojo`'s list.
3. `pixi run ledger-sync` refreshes the `?count=N` snapshots in the ledger.
4. Write the ledger entry: what was claimed, what evidence proves it, and
   the honest reading. Grades: `verified` / `scaffold` / `designed-only` /
   `missing`. `pixi run check-ledger` (CI) and `pixi run check-counts`
   enforce the bookkeeping.

The rule that keeps the ledger alive: **numbers are written in exactly one
place** — the ledger's evidence snapshots. Everything else links to it.

## Writing kernels

Kernels are `@gpu` functions launched through `enqueue_function`:

```mojo
self.ctx.enqueue_function[kv_rows_copy_gpu](
    ptr, Int32(kv_dim), Int32(stride),
    Int32(dst), Int32(src), Int32(nrows),
    grid_dim=(nrows, 1, 1), block_dim=(256, 1, 1),
)
```

Gotchas learned the hard way (each cost a compile round):

- `enqueue_function` captures variadic args by **mutable reference** — two
  same-origin pointer arguments trip the aliasing rejection. Design kernels
  with a **single pointer** and expand slot ids inside.
- `List[Int]` is not implicitly copyable — hash helpers take borrowed
  params; callers pass the name, never `^`.
- Tuples are indexed `r[0]`, not `r.0` (the dot form parses ambiguously).
- Move (`^`) exactly once; anything used after the move is a compile error.

## Service-layer lifecycle

Streams live in `gpu_service.mojo`: `stream_begin` (slot assignment,
prefix-reuse hit decision, hash registration) → tick loop (batch rows) →
`stream_end` (prefix/chunk entry registration). Hit decisions happen
**before** ids move into stream state, so a raise cannot leak a slot.
Telemetry counters go into the `/health` JSON — if a mechanism can be
disabled, its health must show it.

## The remote GPU loop

The A100 host has no outbound network:

```bash
scp src/alofa/srv/gpu_service.mojo lcl@10.107.6.60:/app/lcl/mojo-projects/alofa/src/alofa/srv/
ALOFA_A100_PORT=3389 ./scripts/a100.sh run 3 tests/gpu/test_gpu_service.mojo
```

For HTTP A/B: boot the server on an idle GPU, run
`scripts/stress_serve.py`, kill, boot the control arm, run again —
**tight adjacent pairs**, because the host is shared and bursty. Never
compare numbers across windows; ratios within a pair only.

## CI

`.github/workflows/ci.yml` runs the ledger gate **before** the test suite
(the ledger's credibility outranks whether code compiles), then the CPU
suite. [Pages](https://mojothon.github.io/alofa/) build from `docs/` on
every push to main.
