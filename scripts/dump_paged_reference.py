#!/usr/bin/env python3
"""Export fixtures for paged attention (2.2).

Two things are being pinned, and they need two different kinds of evidence, so
this file exports both:

1. **Addressing.** Paging changes *where a row is*, not *what is computed with
   it*. The Mojo paged kernel and the Mojo contiguous kernel must therefore
   agree **bit for bit**, and a tolerance would be the wrong instrument: there
   is no reassociation to allow for, so any difference at all is a bug. What
   this file exports is the (block, start, length) table plus the contiguous
   rows that table *should* gather, so a gather that reads the wrong slot is
   caught exactly, not approximately.

2. **Arithmetic.** Bit-exactness against itself proves nothing about whether the
   formula is the right formula. So the expected output here is computed by a
   reference written from the definition of grouped-query causal attention, with
   no shared code with either Mojo kernel, and is compared with a tolerance.

The tables are not invented. They are the page tables that `KvSpace` (2.1)
produces for real op sequences, replayed here with the same op lists that
`dump_kv_reference.py` writes into `tests/fixtures/kv/`. That is the point of
2.2: the address space and the kernel have to compose, and the only honest way
to show it is to feed the kernel tables that came out of the space.

Why the values are shaped the way they are
------------------------------------------

Every float is written with 9 decimal places and kept away from zero
(|v| in [0.5, 1.5]) so that the text round-trips through fp32 exactly. A
fixture that only round-trips approximately would make the bit-exact
comparison meaningless, because the two sides would be reading different
numbers before either did any arithmetic.

Run:
    /home/rontom/anaconda3/bin/python scripts/dump_paged_reference.py
"""

import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dump_kv_reference import MAX_REQUESTS, KvSpace, apply_op, scenarios  # noqa: E402

OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "tests", "fixtures", "paged"))

# One block size for the whole fixture: a cache tensor has one block stride, so
# every table in it has to agree with it. The bs=16 scenarios are the ones that
# can share it; s06_small_blocks (bs=4) would need a second cache.
BLOCK_SIZE = 16
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 8
KV_COLS = N_KV_HEADS * HEAD_DIM
Q_COLS = N_HEADS * HEAD_DIM

# (scenario, which COM to snapshot after, q_len, how many requests to emit)
SNAPSHOTS = [
    ("s01_prompt_sharing", 3, 1, 3),
    ("s02_split_divergence", 3, 2, 3),
    ("s03_long_prompt_chain", 2, 3, 2),
]

# Contexts that no KvSpace op sequence can currently produce, listed as
# `(name, q_len, runs)` with runs as `(block, start slot, length)`.
#
# Why they exist: the tree only ever shares *from the root*, and every request
# is allocated from slot 0, so a request's runs all begin at slot 0 — the
# mid-block case is unreachable through `KvSpace` today. It is not unreachable
# through `PagedTable`, so the kernel is tested on it anyway. A kernel that
# quietly ignored `start` and read every run from slot 0 would produce a
# plausible number made of somebody else's tokens, which is the one failure
# this whole layer cannot afford.
#
# `scenario == "-"` marks a case the Mojo test must not cross-check against a
# replayed trace: there is no op sequence behind it.
SYNTHETIC = [
    ("syn_midblock_scatter", 1, [(2, 3, 4), (3, 7, 5), (2, 11, 4)]),
    ("syn_midblock_shift", 3, [(1, 5, 9), (0, 0, 6)]),
    ("syn_midblock_tight", 2, [(2, 1, 6), (3, 9, 3), (1, 13, 2)]),
]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def fmt(v):
    return "%.9f" % v


class Rng:
    """A 64-bit LCG: deterministic, dependency-free, and obviously not shared
    with anything on the Mojo side."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next_u32(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.s >> 32) & 0xFFFFFFFF

    def unit(self):
        """In [-1.5, -0.5] or [0.5, 1.5]: never near zero, so 9 decimals
        round-trip through fp32."""
        v = 0.5 + (self.next_u32() / 4294967296.0)
        if self.next_u32() & 1:
            v = -v
        return f32(v)


def attention_ref(q, q_len, k, v, kv_len, n_heads, n_kv_heads, head_dim, head_map=None):
    """Grouped-query causal attention, written from the definition.

    `head_map` exists only to export a deliberately wrong mapping as a negative
    control: with the correct mapping, query head `h` reads KV head `h // group`.
    Nothing in the default path should depend on it.
    """
    group = n_heads // n_kv_heads
    if head_map is None:

        def head_map(h):
            return h // group

    q_cols = n_heads * head_dim
    kv_cols = n_kv_heads * head_dim
    scale = f32(1.0) / f32(math.sqrt(f32(head_dim)))
    out = [f32(0.0)] * (q_len * q_cols)
    for head in range(n_heads):
        kv_head = head_map(head)
        for t in range(q_len):
            upto = kv_len - q_len + t
            row = [f32(0.0)] * (upto + 1)
            best = None
            for j in range(upto + 1):
                acc = 0.0
                for d in range(head_dim):
                    acc += q[t * q_cols + head * head_dim + d] * k[
                        j * kv_cols + kv_head * head_dim + d
                    ]
                s = f32(acc) * scale
                row[j] = s
                if best is None or s > best:
                    best = s
            total = 0.0
            for j in range(upto + 1):
                e = f32(math.exp(f32(row[j] - best)))
                row[j] = e
                total += e
            for d in range(head_dim):
                acc = 0.0
                for j in range(upto + 1):
                    acc += row[j] * v[j * kv_cols + kv_head * head_dim + d]
                out[t * q_cols + head * head_dim + d] = f32(acc / total)
    return out


def gather_rows(cache, table, n_cols):
    """The table's runs, in order, as one contiguous run of rows."""
    out = []
    for block, start, length in table:
        base = (block * BLOCK_SIZE + start) * n_cols
        for i in range(length * n_cols):
            out.append(cache[base + i])
    return out


def build():
    by_name = {name: (desc, bs, ops) for (name, desc, bs, ops) in scenarios()}

    cases = []
    for scenario, com_ordinal, q_len, want in SNAPSHOTS:
        _desc, bs, ops = by_name[scenario]
        if bs != BLOCK_SIZE:
            raise SystemExit(
                "scenario %s has block_size=%d, fixture is %d" % (scenario, bs, BLOCK_SIZE)
            )
        # Snapshot after the Nth commit: that is where the tree has been
        # reshaped (and so where mid-block starts and shared blocks exist).
        seen = 0
        cut = len(ops)
        for i, op in enumerate(ops):
            if op[0] == "COM":
                seen += 1
                if seen == com_ordinal:
                    cut = i + 1
                    break
        if seen < com_ordinal:
            raise SystemExit("scenario %s has no commit #%d" % (scenario, com_ordinal))

        space = KvSpace(bs, "split")
        for op in ops[:cut]:
            apply_op(space, op)

        live = [s for s in range(MAX_REQUESTS) if space.rq_live[s] == 1]
        if not live:
            raise SystemExit("scenario %s has no live request at cut %d" % (scenario, cut))
        for slot in live[:want]:
            table = []
            for i in range(space.rq_nblk[slot]):
                table.append(
                    (
                        space.block_at(slot, i),
                        space.bstart_at(slot, i),
                        space.blen_at(slot, i),
                    )
                )
            cases.append(
                {
                    "name": "%s_c%d_r%d" % (scenario.split("_")[0], com_ordinal, space.rq_id[slot]),
                    "scenario": scenario,
                    "req": space.rq_id[slot],
                    "ops": cut,
                    "q_len": q_len,
                    "table": table,
                }
            )

    for name, q_len, table in SYNTHETIC:
        cases.append(
            {
                "name": name,
                "scenario": "-",
                "req": 0,
                "ops": 0,
                "q_len": q_len,
                "table": list(table),
            }
        )

    # Size the cache to the highest block any table names, so an out-of-range
    # block id is a real error and not just a big tensor.
    max_block = 0
    for c in cases:
        for block, _start, _length in c["table"]:
            max_block = max(max_block, block)
    n_blocks = max_block + 1

    rng = Rng(0x5EED)
    k_cache = [rng.unit() for _ in range(n_blocks * BLOCK_SIZE * KV_COLS)]
    v_cache = [rng.unit() for _ in range(n_blocks * BLOCK_SIZE * KV_COLS)]

    for c in cases:
        kv_len = sum(t[2] for t in c["table"])
        if c["q_len"] > kv_len:
            c["q_len"] = kv_len
        c["kv_len"] = kv_len
        c["k"] = gather_rows(k_cache, c["table"], KV_COLS)
        c["v"] = gather_rows(v_cache, c["table"], KV_COLS)
        c["q"] = [rng.unit() for _ in range(c["q_len"] * Q_COLS)]
        c["y"] = attention_ref(
            c["q"], c["q_len"], c["k"], c["v"], kv_len, N_HEADS, N_KV_HEADS, HEAD_DIM
        )
    return cases, k_cache, v_cache, n_blocks


def write_floats(path, values):
    with open(path, "w") as handle:
        for v in values:
            handle.write(fmt(v) + "\n")


def main():
    cases, k_cache, v_cache, n_blocks = build()

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    def out(name):
        return os.path.join(OUT_DIR, name)

    with open(out("config.tsv"), "w") as handle:
        handle.write("n_blocks\t%d\n" % n_blocks)
        handle.write("block_size\t%d\n" % BLOCK_SIZE)
        handle.write("kv_cols\t%d\n" % KV_COLS)
        handle.write("q_cols\t%d\n" % Q_COLS)
        handle.write("n_heads\t%d\n" % N_HEADS)
        handle.write("n_kv_heads\t%d\n" % N_KV_HEADS)
        handle.write("head_dim\t%d\n" % HEAD_DIM)
        handle.write("n_cases\t%d\n" % len(cases))

    with open(out("cases.tsv"), "w") as handle:
        handle.write("# name\tscenario\treq\tops\tq_len\tn_entries\tthen (block start length) * n_entries\n")
        for c in cases:
            fields = [
                c["name"],
                c["scenario"],
                str(c["req"]),
                str(c["ops"]),
                str(c["q_len"]),
                str(len(c["table"])),
            ]
            for block, start, length in c["table"]:
                fields += [str(block), str(start), str(length)]
            handle.write("\t".join(fields) + "\n")

    write_floats(out("k_cache.tsv"), k_cache)
    write_floats(out("v_cache.tsv"), v_cache)

    for c in cases:
        write_floats(out("%s.q.tsv" % c["name"]), c["q"])
        write_floats(out("%s.kv.tsv" % c["name"]), c["k"] + c["v"])
        write_floats(out("%s.y.tsv" % c["name"]), c["y"])

    # Negative controls. Both are expected to be *rejected*, so both are
    # exported from the first case rather than generated inside the test: a
    # control the test builds itself can drift into passing.
    first = cases[0]
    bad = list(first["y"])
    bad[0] = f32(bad[0] + 0.05)
    write_floats(out("expected_bad.tsv"), bad)

    wrong_map = attention_ref(
        first["q"],
        first["q_len"],
        first["k"],
        first["v"],
        first["kv_len"],
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        head_map=lambda h: h % N_KV_HEADS,
    )
    write_floats(out("expected_hmap.tsv"), wrong_map)

    nonzero = sum(1 for c in cases for (_b, s, _l) in c["table"] if s != 0)
    shared = 0
    seen_blocks = {}
    for c in cases:
        for (b, _s, _l) in c["table"]:
            seen_blocks[b] = seen_blocks.get(b, 0) + 1
    shared = sum(1 for b, n in seen_blocks.items() if n > 1)

    print("wrote %d cases to %s" % (len(cases), OUT_DIR))
    print("  n_blocks=%d block_size=%d" % (n_blocks, BLOCK_SIZE))
    print("  entries with a non-zero start: %d" % nonzero)
    print("  blocks named by more than one case: %d" % shared)
    if nonzero == 0:
        raise SystemExit(
            "no entry starts mid-block: the fixture would not exercise the "
            "addressing 2.2 exists for"
        )
    if shared == 0:
        raise SystemExit(
            "no block is shared between two cases: the fixture would not "
            "exercise prefix sharing"
        )


if __name__ == "__main__":
    main()
