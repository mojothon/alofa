#!/home/rontom/anaconda3/bin/python
"""Export fixtures for the batch tensor pool (2.5) and serve as its reference.

An independent implementation, not a mirror of the Mojo one
-----------------------------------------------------------

This file re-derives the allocator from the same *rules* and not from the same
code: first fit over gaps derived from the live set, halves rounded up to 64
bytes, lowest free slot wins. When the two agree byte for byte, the agreement is
evidence; when either side could produce the fixtures, the agreement is nothing.

It earns its keep the same way `dump_kv_reference.py` did: writing it surfaces the
question "what is 'used' after a release", and answering it here first means the
Mojo side never got to decide the answer alone.

The interesting scenarios are the ones where the allocator has to say no:

- **room but not contiguous.** Borrowing two anchor regions and freeing the
  smaller of them leaves plenty of free bytes and no usable gap. Saying "no" is
  the only honest answer; growing the pool, or wrapping around, would turn a
  capacity question into a wrong number.
- **handle pressure.** Twenty-four live borrows is a cap, and reaching it must be
  an error rather than an expansion.

Exported layout
---------------

    tests/fixtures/batch/s<N>_<name>.trace   one line per operation
    tests/fixtures/batch/alt.trace           same ops, best-fit placement
    tests/fixtures/batch/bad.trace           one digit wrong in one result

Line format follows the KV trace rules: integers, fixed field order read by
position, and `-` for nothing. An operation that fails carries `ERR=<name>`
instead of its value fields, because a scenario has to be able to say "this
sequence ends in a refusal" without ending the file.

Run:
    /home/rontom/anaconda3/bin/python scripts/dump_batch_reference.py
"""

from __future__ import annotations

import os

ALIGN = 64
MAX_HANDLES = 24
MAX_BATCH = 8
MOD = 2147483647
MUL = 1000003

CAPACITY = 262144

# Qwen2.5-0.5B, fp32: hidden 896, intermediate 4864, two KV heads of 64.
HIDDEN_ROW = 896 * 4
INTER_ROW = 4864 * 4
KV_ROW = 128 * 4

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "batch")


def ceil_bytes(n: int) -> int:
    return ((n + ALIGN - 1) // ALIGN) * ALIGN


class PoolError(Exception):
    """Mirror of alofa's named errors, carrying only the name."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


class BatchPool:
    """Borrow and return byte intervals of a buffer nobody named here."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise PoolError("invalid_argument")
        if capacity != ceil_bytes(capacity):
            raise PoolError("invalid_argument")
        self.capacity = capacity
        self.used = 0
        self.live = [0] * MAX_HANDLES
        self.begin = [0] * MAX_HANDLES
        self.size = [0] * MAX_HANDLES
        self.high_water = 0
        self.last_handle = -1

    def n_live(self) -> int:
        return sum(self.live)

    def free_slot(self) -> int:
        for h in range(MAX_HANDLES):
            if self.live[h] == 0:
                return h
        return -1

    def place(self, aligned: int, best: bool = False) -> int:
        """Lowest gap that fits (first fit), or the tightest gap (best fit).

        The policy parameter exists only so `alt.trace` can disagree: a fixture
        that every placement policy satisfies is a fixture that checks the file
        format and not the allocator.
        """
        taken = sorted((self.begin[h], self.begin[h] + self.size[h])
                       for h in range(MAX_HANDLES) if self.live[h])
        gaps = []
        pos = 0
        for lo, hi in taken:
            if lo - pos > 0:
                gaps.append((pos, lo - pos))
            pos = max(pos, hi)
        if self.capacity - pos > 0:
            gaps.append((pos, self.capacity - pos))
        fits = [(at, room) for at, room in gaps if room >= aligned]
        if not fits:
            return -1
        if best:
            fits.sort(key=lambda pair: (pair[1], pair[0]))
        return fits[0][0]

    def borrow(self, size: int, best: bool = False) -> int:
        if size <= 0:
            raise PoolError("invalid_argument")
        aligned = ceil_bytes(size)
        if aligned > self.capacity:
            raise PoolError("capacity")
        slot = self.free_slot()
        if slot < 0:
            raise PoolError("capacity")
        at = self.place(aligned, best)
        if at < 0:
            raise PoolError("capacity")
        self.begin[slot] = at
        self.size[slot] = aligned
        self.live[slot] = 1
        self.used += aligned
        self.high_water = max(self.high_water, at + aligned)
        self.last_handle = slot
        return slot

    def release(self, handle: int) -> int:
        if handle < 0 or handle >= MAX_HANDLES:
            raise PoolError("out_of_range")
        if self.live[handle] == 0:
            raise PoolError("double_free")
        self.live[handle] = 0
        self.used -= self.size[handle]
        return self.begin[handle]

    def reset(self) -> None:
        self.live = [0] * MAX_HANDLES
        self.used = 0

    def digest(self) -> int:
        value = 0
        for h in range(MAX_HANDLES):
            value = (value * MUL + self.begin[h] + self.size[h] * 7
                     + self.live[h] * 13) % MOD
        value = (value * MUL + self.used) % MOD
        value = (value * MUL + self.high_water) % MOD
        return value


class BatchSlots:
    """Row ranges of the packed token axis, one contiguous run per request."""

    def __init__(self):
        self.req = [-1] * MAX_BATCH
        self.off = [0] * MAX_BATCH
        self.cnt = [0] * MAX_BATCH
        self.live = [0] * MAX_BATCH

    def n_live(self) -> int:
        return sum(self.live)

    def total_tokens(self) -> int:
        return sum(self.cnt[i] for i in range(MAX_BATCH) if self.live[i])

    def slot_of(self, request: int) -> int:
        for i in range(MAX_BATCH):
            if self.live[i] and self.req[i] == request:
                return i
        return -1

    def place(self, count: int) -> int:
        taken = sorted((self.off[i], self.off[i] + self.cnt[i])
                       for i in range(MAX_BATCH) if self.live[i])
        pos = 0
        for lo, hi in taken:
            if lo - pos >= count:
                return pos
            pos = max(pos, hi)
        return pos

    def add(self, request: int, count: int) -> int:
        if count <= 0:
            raise PoolError("invalid_argument")
        if self.slot_of(request) >= 0:
            raise PoolError("invalid_argument")
        slot = -1
        for i in range(MAX_BATCH):
            if not self.live[i]:
                slot = i
                break
        if slot < 0:
            raise PoolError("capacity")
        at = self.place(count)
        self.req[slot] = request
        self.off[slot] = at
        self.cnt[slot] = count
        self.live[slot] = 1
        return at

    def remove(self, request: int) -> int:
        slot = self.slot_of(request)
        if slot < 0:
            raise PoolError("invalid_argument")
        self.live[slot] = 0
        self.req[slot] = -1
        return self.off[slot]

    def clear(self) -> None:
        self.req = [-1] * MAX_BATCH
        self.live = [0] * MAX_BATCH

    def digest(self) -> int:
        value = 0
        for i in range(MAX_BATCH):
            value = (value * MUL + self.req[i] + self.off[i] * 7
                     + self.cnt[i] * 11 + self.live[i] * 13) % MOD
        return value


def result_of(kind: str, pool: BatchPool, slots: BatchSlots, extra: int) -> str:
    """One `R=` field list. Nothing here is optional: absent means zero."""
    if kind == "get":
        return (f"o={extra} h={pool.last_handle} u={pool.used}"
                f" n={pool.n_live()} d={pool.digest()}")
    if kind == "rel":
        return f"o={extra} u={pool.used} n={pool.n_live()} d={pool.digest()}"
    if kind == "clr":
        return f"u={pool.used} n={pool.n_live()} d={pool.digest()}"
    if kind in ("add", "del"):
        return (f"t={extra} N={slots.total_tokens()} b={slots.n_live()}"
                f" d={slots.digest()}")
    raise AssertionError(f"unknown result kind {kind}")


def run(name: str, ops, best: bool = False):
    """Run one scenario and return its operation lines, header excluded.

    A refused operation still gets its line, carrying `ERR=<name>`. Otherwise a
    scenario could not say "this sequence ends in a refusal" without either
    truncating the file or hiding the refusal behind silence.
    """
    pool = BatchPool(CAPACITY)
    slots = BatchSlots()
    tags: dict[str, int] = {}
    lines = []
    for op in ops:
        label = ""
        try:
            kind = op[0]
            if kind == "add":
                label = f"ADD r={op[1]} n={op[2]}"
                extra = slots.add(op[1], op[2])
            elif kind == "del":
                label = f"DEL r={op[1]}"
                extra = slots.remove(op[1])
            elif kind == "get":
                label = f"GET s={op[1]}"
                handle = pool.borrow(op[1], best)
                if len(op) > 2:
                    tags[op[2]] = handle
                extra = pool.begin[handle]
            elif kind == "rel":
                handle = tags[op[1]] if isinstance(op[1], str) else op[1]
                label = f"REL h={handle}"
                extra = pool.release(handle)
            elif kind == "clr":
                label = "CLR"
                pool.reset()
                slots.clear()
                extra = 0
            else:
                raise AssertionError(f"unknown op {op}")
            rendered = result_of(kind, pool, slots, extra)
        except PoolError as exc:
            rendered = f"ERR={exc.name}"
        lines.append(f"{label} R={rendered}")
    return lines

STEPS = [
    ("add", 1, 3), ("add", 2, 5), ("add", 3, 2),
    ("get", HIDDEN_ROW * 10, "hid"),
    ("get", 10 * 10 * 4, "sc"),
    ("rel", "sc"), ("rel", "hid"),
    ("del", 2), ("add", 4, 5),
    ("get", HIDDEN_ROW * 10, "hid2"),
    ("get", 10 * 10 * 4, "sc2"),
    ("rel", "sc2"), ("rel", "hid2"),
]


SCENARIOS = [
    (
        "s01_executor_step",
        "two executor steps over a three-request batch with one swap in "
        "between: the packed batch is still ten rows, so the second step must "
        "land its scratch on the same addresses as the first",
        STEPS + [("del", 1), ("del", 3), ("del", 4), ("clr",)],
    ),
    (
        "s02_interleaved_lifetimes",
        "three live regions whose middle one is freed first, so the next borrow "
        "has to step into the gap instead of climbing over everything",
        [
            ("get", HIDDEN_ROW, "a"),
            ("get", INTER_ROW, "b"),
            ("get", KV_ROW, "c"),
            ("rel", "b"),
            ("get", INTER_ROW, "d"),
            ("rel", "a"),
            ("get", HIDDEN_ROW, "e"),
            ("rel", "c"), ("rel", "d"), ("rel", "e"),
        ],
    ),
    (
        "s03_room_but_not_contiguous",
        "the pool is filled by sixteen regions of 16384 and then every other "
        "one is given back: 131072 bytes are free, the largest hole is 16384, "
        "and there is no tail to fall back on, so borrowing two regions at "
        "once must be refused rather than split across two holes, while a "
        "single region must still take the first hole",
        [("get", 16384, f"r{i}") for i in range(16)]
        + [("rel", f"r{i}") for i in range(1, 16, 2)]
        + [("get", 32768, "wide"), ("get", 16384, "narrow"), ("clr",)],
    ),
    (
        "s04_batch_turnover",
        "four requests arrive, the middle one leaves, and a fifth takes the "
        "rows it had: batch rows are reused, not appended forever",
        [
            ("add", 7, 12), ("add", 8, 7), ("add", 9, 9), ("add", 10, 5),
            ("del", 8), ("add", 11, 7),
            ("del", 10), ("add", 12, 3),
            ("del", 7), ("del", 9), ("del", 11), ("del", 12),
        ],
    ),
    (
        "s05_two_holes",
        "releasing both large anchors leaves one huge hole low down and one "
        "small one high up; an 8192-byte borrow fits in either, so this "
        "scenario is what pins the placement policy down",
        [
            ("get", 65536, "a"),
            ("get", 65536, "b"),
            ("get", 8192, "c"),
            ("rel", "a"), ("rel", "b"),
            ("get", 8192, "d"),
            ("rel", "c"), ("rel", "d"),
            ("clr",),
        ],
    ),
    (
        "s06_refusals_are_named",
        "every refusal is a named error: releasing an unknown borrow, "
        "borrowing zero bytes, admitting one request twice, dropping a request "
        "that is not batched",
        [
            ("rel", 5),
            ("get", 0),
            ("add", 4, 1),
            ("add", 4, 1),
            ("del", 99),
        ],
    ),
    (
        "s07_handle_pressure",
        "twenty-four live borrows is the cap; the twenty-fifth must be "
        "refused rather than expanding the table",
        [("get", ALIGN, f"h{i}") for i in range(MAX_HANDLES)]
        + [("get", ALIGN, "toomany")]
        + [("rel", f"h0"), ("get", ALIGN, "after")],
    ),
]


def write(name: str, comment: str, lines) -> str:
    header = [
        "# alofa batch trace v1",
        f"# scenario: {name}",
        f"# {comment}",
        f"CFG c={CAPACITY} mh={MAX_HANDLES} mb={MAX_BATCH} al={ALIGN}",
    ]
    body = "\n".join(header + lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".trace")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
    return path


def main() -> None:
    written = []
    for name, comment, ops in SCENARIOS:
        written.append(write(name, comment, run(name, ops)))

    # Negative control one: the operations of s05 under a different placement
    # policy. s05 was built so the two policies disagree, and the suite checks
    # that they really do -- otherwise the byte comparison of the other six
    # scenarios could be checking the line format and not the allocator.
    holes = [s for s in SCENARIOS if s[0] == "s05_two_holes"][0]
    written.append(
        write("alt", holes[1] + " (policy: best fit)",
              run(holes[0], holes[2], best=True))
    )

    # Negative control two: one digit wrong in one result line.
    good = run(SCENARIOS[0][0], SCENARIOS[0][2])
    wrong = list(good)
    for i, line in enumerate(wrong):
        if line.startswith("GET") and " R=o=0 " in line:
            wrong[i] = line.replace(" R=o=0 ", " R=o=64 ", 1)
            break
    written.append(write("bad", "same as s01 with one wrong offset", wrong))

    for path in written:
        print(os.path.relpath(path, start=os.path.dirname(__file__)))


if __name__ == "__main__":
    main()
