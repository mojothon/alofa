# Principles

Three throughput mechanisms live on the GPU serving path. Each owns a
distinct cost face, each is gated in CI, and each composes multiplicatively
with the others. Evidence lives in the
[ledger](plan/capability-ledger.md) series 128-176.

## Batched tick execution

The GPU server keeps a single model instance with a **64-slot batch pool**
(`[B][max_tokens][kv_dim]` per layer for K and V). All active streams share
one **tick**: one scheduled forward over the batch rows. A pure-function
tick engine decides each tick's row mix (prompt rows vs generation rows)
under budget guards (frame backlog, prefill slicing), so one slow client
never blocks another and long prompts cannot starve generation.

Measured: batch execution ≡ single-stream token-for-token
(`test_batch_forward`), and the tick engine alone peaked at ~6.5× speedup
at c=32.

## Speculative decoding

1. A **shared n-gram table** accumulates continuations across all streams —
   drafts are free (no second model).
2. Draft tokens are appended to a row and validated by **one real forward**:
   accepted tokens are kept, the first divergence is corrected.
3. The draft budget **k_eff adapts** per row to its acceptance history —
   heterogeneous traffic gets short drafts, repetitive traffic gets long
   ones.
4. **Sampled verification** (per-row sampling kernel) extends the mechanism
   to temperature > 0: instead of argmax comparison, each draft position is
   accepted with the standard speculative-sampling criterion.

Measured: greedy spectrum 1.6-2.7× by repetition rate; at temp 0.7 the
whole spectrum stays positive (2.29× at R=1.0 down to 1.19× at R=0, where
acceptance ≈ q(draft) = 0.62 as theory predicts).

## KV prefix reuse

The key insight: **KV rows are uniquely determined by position index**
(RoPE phase = position), so rows for the same prefix are bit-identical —
**copying is reuse**.

- **Exact path**: a dual-salt FNV-1a hash of the full prompt (effective
  128-bit check) + prompt length + slot-occupancy validation. On hit, a
  single-pointer CUDA kernel copies rows `[0, plen-1)` from the source slot
  and the stream starts at the last prompt token.
- **Chunk path**: multi-turn conversations *extend* earlier prompts, so
  exact equality never matches. At stream end, every 16-token boundary is
  registered through an incremental FNV chain (O(plen)). A request advances
  the same chain and takes the **highest valid boundary** = longest common
  prefix, chunk-aligned. Rows `[0, lcp)` are copied; the remainder prefills
  normally.
- **Safety**: the source slot's current or previous stream must actually be
  the registered prompt (occupancy validation) — stale entries pointing at
  overwritten rows are rejected. A miss costs two hash computations.

Measured: +3.0× on top of spec for repeated traffic (~6.8× combined);
1.6-2.2× for shared-preamble traffic that exact matching can never see;
concurrency collapse at c=32 largely healed (1.65× → 1.9×) because
eliminated prompt rows return their tick budget to speculative drafts;
**TTFT p50 243 ms → 22.3 ms at c=8**.

## Why no penalty cells

Every mechanism is guarded by an adaptive budget: drafts shrink when
acceptance drops, prefix lookups cost only two hashes when they miss, and
the tick engine rebalances rows every tick. The worst measured cell across
the whole traffic matrix is 1.19×.
