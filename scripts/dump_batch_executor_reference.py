#!/usr/bin/env python3
"""Reference for the batch executor: row assignment and the step loop, in Python.

The Mojo executor decides two things every step — which request owns which rows,
and how far back each row may look — and both are *decisions*, not numbers. A
wrong row assignment does not produce a slightly different logit; it produces a
perfectly plausible one, because a layer reading request A's row at request B's
position is reading a real tensor. So the expected lines this script writes are
compared byte for byte, exactly as the scheduler and the KV pool are compared.

This is a second implementation of the same rules, written from the rules and not
from the Mojo: nothing is imported from the tree. Where the two disagree, the
trace is what decides, and the test says which scenario disagreed.

A refused operation still gets its line, carrying `ERR=<name>`: a scenario has to
be able to end in a refusal without truncating the file.

Run:
    /home/rontom/anaconda3/bin/python scripts/dump_batch_executor_reference.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "batchexec"
FORMAT_VERSION = "# alofa batch executor trace v1"

MAX_BATCH = 8
MAX_GEN = 32
MAX_ROWS = 16
MAX_POS = 24
NO_ROW = -1
NO_TOKEN = -1
NO_SLOT = -1
NO_REQUEST = -1
POOL_ALIGN = 64

# A toy model, so the reference can be run and read without loading weights.
HIDDEN = 8
INTER = 12
KV_DIM = 4
VOCAB = 16
N_LAYERS = 2
N_HEADS = 2
N_KV_HEADS = 1
HEAD_DIM = 4

MUL = 1000003
MOD = 2147483647


class Named(Exception):
    """An error with the name the Mojo side raises."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


def ceil_bytes(n: int) -> int:
    return ((n + POOL_ALIGN - 1) // POOL_ALIGN) * POOL_ALIGN


def scratch_bytes() -> int:
    total = 6 * MAX_ROWS * HIDDEN * 4
    total += 3 * MAX_ROWS * KV_DIM * 4
    total += 3 * MAX_ROWS * INTER * 4
    total += MAX_ROWS * MAX_POS * 4
    total += MAX_BATCH * HIDDEN * 4
    total += MAX_BATCH * VOCAB * 4
    return ceil_bytes(total + 16 * POOL_ALIGN)


class Slots:
    """Which request owns which run of rows, first fit by increasing row."""

    def __init__(self):
        self.req = [NO_REQUEST] * MAX_BATCH
        self.off = [0] * MAX_BATCH
        self.cnt = [0] * MAX_BATCH
        self.live = [0] * MAX_BATCH

    def n_live(self) -> int:
        return sum(self.live)

    def slot_of(self, request: int) -> int:
        for i in range(MAX_BATCH):
            if self.live[i] == 1 and self.req[i] == request:
                return i
        return NO_REQUEST

    def total_tokens(self) -> int:
        return sum(self.cnt[i] for i in range(MAX_BATCH) if self.live[i] == 1)

    def place(self, count: int) -> int:
        taken = sorted((self.off[i], self.off[i] + self.cnt[i])
                       for i in range(MAX_BATCH) if self.live[i] == 1)
        pos = 0
        for start, stop in taken:
            if start - pos >= count:
                return pos
            if stop > pos:
                pos = stop
        return pos

    def add(self, request: int, count: int) -> int:
        if count <= 0:
            raise Named("invalid_argument")
        if self.slot_of(request) != NO_REQUEST:
            raise Named("invalid_argument")
        slot = NO_SLOT
        for i in range(MAX_BATCH):
            if self.live[i] == 0:
                slot = i
                break
        if slot == NO_SLOT:
            raise Named("capacity")
        at = self.place(count)
        self.req[slot] = request
        self.off[slot] = at
        self.cnt[slot] = count
        self.live[slot] = 1
        return at

    def clear(self):
        for i in range(MAX_BATCH):
            self.live[i] = 0
            self.req[i] = NO_REQUEST

    def digest(self) -> int:
        value = 0
        for i in range(MAX_BATCH):
            value = (value * MUL + self.req[i] + self.off[i] * 7
                     + self.cnt[i] * 11 + self.live[i] * 13) % MOD
        return value


class Executor:
    """The rules: queue, plan rows, consume a step, retire finished requests."""

    def __init__(self):
        self.slots = Slots()
        self.req = [0] * MAX_BATCH
        self.hist = [0] * MAX_BATCH
        # How long the prompt *is*, as opposed to how much of it is queued.
        self.n_prompt = [0] * MAX_BATCH
        self.pending = [[] for _ in range(MAX_BATCH)]
        self.budget = [0] * MAX_BATCH
        self.served = [0] * MAX_BATCH
        self.live = [0] * MAX_BATCH
        self.out = [[] for _ in range(MAX_BATCH)]
        self.rows = 0
        self.row_req = []
        self.row_tok = []
        self.row_pos = []
        self.used = 0

    def slot_of(self, request: int) -> int:
        for i in range(MAX_BATCH):
            if self.live[i] == 1 and self.req[i] == request:
                return i
        return NO_SLOT

    def add(self, request: int, tokens, budget: int, prompt_len: int = -1) -> int:
        if not tokens:
            raise Named("invalid_argument")
        if len(tokens) > MAX_ROWS:
            raise Named("capacity")
        total = prompt_len if prompt_len > 0 else len(tokens)
        if len(tokens) > total:
            raise Named("out_of_range")
        if self.slot_of(request) >= 0:
            raise Named("invalid_argument")
        slot = NO_SLOT
        for i in range(MAX_BATCH):
            if self.live[i] == 0:
                slot = i
                break
        if slot < 0:
            raise Named("capacity")
        for token in tokens:
            if token < 0 or token >= VOCAB:
                raise Named("out_of_range")
        self.req[slot] = request
        self.live[slot] = 1
        self.hist[slot] = 0
        self.n_prompt[slot] = total
        self.pending[slot] = list(tokens)
        self.budget[slot] = budget
        self.served[slot] = 0
        self.out[slot] = []
        return slot

    def feed(self, request: int, token: int) -> int:
        slot = self.slot_of(request)
        if slot < 0:
            raise Named("invalid_argument")
        if token < 0 or token >= VOCAB:
            raise Named("out_of_range")
        if len(self.pending[slot]) >= MAX_ROWS:
            raise Named("capacity")
        if self.hist[slot] + len(self.pending[slot]) + 1 > self.n_prompt[slot]:
            raise Named("out_of_range")
        self.pending[slot].append(token)
        return len(self.pending[slot])

    def drop(self, request: int) -> int:
        slot = self.slot_of(request)
        if slot < 0:
            raise Named("invalid_argument")
        self.live[slot] = 0
        self.pending[slot] = []
        self.n_prompt[slot] = 0
        self.hist[slot] = 0
        self.served[slot] = 0
        self.budget[slot] = 0
        self.out[slot] = []
        self.req[slot] = 0
        return slot

    def plan(self, reverse: bool = False) -> int:
        self.slots.clear()
        self.rows = 0
        self.row_req = []
        self.row_tok = []
        self.row_pos = []
        self.used = 0
        order = list(range(MAX_BATCH))
        if reverse:
            order.reverse()
        seen = 0
        for i in order:
            self.served[i] = 0
            if self.live[i] != 1:
                continue
            room = MAX_ROWS - self.slots.total_tokens()
            n = min(len(self.pending[i]), room)
            if n <= 0:
                continue
            if self.hist[i] + n > MAX_POS:
                raise Named("capacity")
            at = self.slots.add(self.req[i], n)
            self.served[i] = n
            for j in range(n):
                self.row_req.append(self.req[i])
                self.row_tok.append(self.pending[i][j])
                self.row_pos.append(self.hist[i] + j)
            seen += 1
        self.rows = self.slots.total_tokens()
        if self.rows <= 0:
            return 0
        served = self.slots.n_live()
        for size in step_sizes(self.rows, served):
            self.used += ceil_bytes(size)
        return self.rows

    def advance(self, chosen) -> None:
        for i in range(MAX_BATCH):
            if self.live[i] != 1 or self.served[i] == 0:
                continue
            n = self.served[i]
            self.hist[i] += n
            self.pending[i] = self.pending[i][n:]
            # A request whose prompt is not finished owes no answer: the step
            # that ends the prompt is the step that gets to choose. Feeding the
            # argmax back here would splice a generated token into the middle of
            # a prompt that was split across two steps.
            # An empty queue is not the end of the prompt: a scheduler that
            # prefills in slices empties it at every slice boundary, and a token
            # chosen there would land in the middle of the prompt.
            if len(self.pending[i]) > 0 or self.hist[i] < self.n_prompt[i]:
                continue
            token = chosen[i]
            if token == NO_TOKEN:
                continue
            if len(self.out[i]) < MAX_GEN:
                self.out[i].append(token)
            self.budget[i] -= 1
            # Not evicted for spending its budget: the token that spent it is in
            # `out`, and a request that leaves on its own takes that transcript
            # with it. It stops being fed; the caller drops it.
            if self.budget[i] <= 0:
                continue
            if len(self.pending[i]) >= MAX_ROWS:
                raise Named("capacity")
            self.pending[i].append(token)

    def digest(self) -> int:
        # Reduced at every step: the Mojo side is a 64-bit integer and this one
        # is not, so an unreduced product would wrap on one side only and every
        # line would differ for a reason that has nothing to do with placement.
        acc = MUL * 7 + self.rows
        for i in range(MAX_BATCH):
            acc = (acc * MUL + self.req[i] + 1) % MOD
            acc = (acc * MUL + self.hist[i] + 1) % MOD
            acc = (acc * MUL + self.n_prompt[i] + 1) % MOD
            acc = (acc * MUL + len(self.pending[i]) + 1) % MOD
            acc = (acc * MUL + self.budget[i] + 2) % MOD
            acc = (acc * MUL + self.live[i] + 1) % MOD
        return (acc * MUL + self.slots.digest()) % MOD


def step_sizes(rows: int, served: int) -> list:
    """Every buffer one forward borrows, in the executor's order."""
    return [
        rows * HIDDEN * 4,
        rows * HIDDEN * 4,
        rows * HIDDEN * 4,
        rows * KV_DIM * 4,
        rows * KV_DIM * 4,
        rows * HIDDEN * 4,
        rows * KV_DIM * 4,
        rows * HIDDEN * 4,
        rows * HIDDEN * 4,
        rows * INTER * 4,
        rows * INTER * 4,
        rows * INTER * 4,
        rows * MAX_POS * 4,
        served * HIDDEN * 4,
        served * VOCAB * 4,
    ]


def render_next(ex: Executor) -> str:
    """Histories and queues of every resident request, in slot order."""
    parts = []
    queues = []
    for i in range(MAX_BATCH):
        if ex.live[i] != 1:
            continue
        parts.append(f"{ex.req[i]}:{ex.hist[i]}")
        queues.append(f"{ex.req[i]}:{len(ex.pending[i])}")
    return ("h=" + (",".join(parts) if parts else "-")
            + " p=" + (",".join(queues) if queues else "-"))


def run(ops, reverse: bool = False):
    """Turn operations into trace lines, header excluded."""
    ex = Executor()
    lines = []
    for op in ops:
        kind = op[0]
        try:
            if kind == "add":
                label = (f"ADD r={op[1]} b={op[3]} n={len(op[2])}"
                         f" k={','.join(str(t) for t in op[2])}")
                slot = ex.add(op[1], op[2], op[3])
                result = f"i={slot} n={len(ex.pending[slot])} d={ex.digest()}"
            elif kind == "feed":
                label = f"FEED r={op[1]} k={op[2]}"
                result = f"p={ex.feed(op[1], op[2])} d={ex.digest()}"
            elif kind == "drop":
                label = f"DROP r={op[1]}"
                result = f"i={ex.drop(op[1])} d={ex.digest()}"
            elif kind == "fin":
                label = "FIN"
                ex.used = 0
                result = f"u=0 d={ex.digest()}"
            elif kind == "plan":
                label = "PLAN"
                rows = ex.plan(reverse)
                result = (f"T={rows} n={ex.slots.n_live()} u={ex.used}"
                          f" d={ex.digest()}")
            elif kind == "next":
                chosen = [-1] * MAX_BATCH
                for slot, token in enumerate(op[1]):
                    chosen[slot] = token
                label = ("NEXT c="
                         + ",".join(str(t) if t >= 0 else "-" for t in chosen))
                ex.advance(chosen)
                result = f"{render_next(ex)} d={ex.digest()}"
            else:
                raise AssertionError(f"unknown op {op}")
        except Named as exc:
            result = f"ERR={exc.name}"
            if kind == "plan":
                # A step that cannot be planned has no rows to describe.
                ex.rows = 0
                ex.row_req = []
                ex.row_tok = []
                ex.row_pos = []
        lines.append(f"{label} R={result}")
        if kind == "plan" and ex.rows > 0:
            for i in range(ex.rows):
                lines.append(
                    f"ROW i={i} r={ex.row_req[i]} t={ex.row_tok[i]}"
                    f" p={ex.row_pos[i]} v={ex.row_pos[i]}"
                )
    return lines


HEADER = f"""{FORMAT_VERSION}
# Reference for the batch executor: row ownership and the step loop.
# Written by scripts/dump_batch_executor_reference.py -- do not edit by hand.
#
#   ADD  r=<id> b=<budget> n=<count> k=<tokens>   -> i=<slot> n=<queued>
#   FEED r=<id> k=<token>                         -> p=<queued>
#   DROP r=<id>                                   -> i=<slot>
#   PLAN                                          -> T=<rows> n=<served> u=<pool bytes>
#   FIN                                           -> u=0 after the step gives its buffers back
#   ROW  i=<row> r=<owner> t=<token> p=<position> v=<last visible key>
#   NEXT c=<token per slot>                       -> h=<id:history> p=<id:queued>
#
# A refused operation carries ERR=<name> instead of a result: a scenario has to
# be able to end in a refusal without truncating the file.
#
# ROW lines follow the PLAN they belong to and are compared in order: the row
# block is the concatenation of the requests' runs, so the owner sequence is the
# answer rather than a detail of how it was printed.
"""


def write(name: str, comment: str, lines) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = (f"# CFG mb={MAX_BATCH} mr={MAX_ROWS} mp={MAX_POS} h={HIDDEN}"
           f" i={INTER} k={KV_DIM} v={VOCAB} l={N_LAYERS} nh={N_HEADS}"
           f" nk={N_KV_HEADS} hd={HEAD_DIM} c={scratch_bytes()}")
    body = "\n".join([HEADER.rstrip("\n"), cfg, "# " + comment] + lines) + "\n"
    (OUT / f"{name}.trace").write_text(body, encoding="utf-8")
    return f"{name}.trace"


def seq(start: int, count: int, limit: int = VOCAB - 1) -> list:
    """Token ids that stay inside the toy vocabulary."""
    return [(start + i) % limit for i in range(count)]


SCENARIOS = [
    (
        "e01_two_requests",
        "two prompts of different length: the rows are request 7's run followed "
        "by request 9's, and each row's visible window is its own position, not "
        "its index in the block",
        [
            ("add", 7, seq(1, 3), 4),
            ("add", 9, seq(4, 2), 4),
            ("plan",), ("fin",),
            ("next", [6, 7]),
            ("plan",), ("fin",),
            ("next", [8, 9]),
            ("plan",), ("fin",),
            ("next", [10, 11]),
            ("plan",), ("fin",),
        ],
    ),
    (
        "e02_hole_after_drop",
        "three requests, then the middle one leaves: its run becomes a hole and "
        "the next request's rows must take it rather than grow the block",
        [
            ("add", 1, seq(1, 2), 4),
            ("add", 2, seq(3, 3), 4),
            ("add", 3, seq(6, 2), 4),
            ("plan",), ("fin",),
            ("next", [8, 9, 10]),
            ("drop", 2),
            ("add", 4, seq(11, 2), 4),
            ("plan",), ("fin",),
            ("next", [12, -1, 13, 14]),
            ("plan",), ("fin",),
        ],
    ),
    (
        "e03_more_than_one_step",
        "three prompts that together are longer than the row block: the last "
        "request waits and keeps its queue, so a short step is a step and not a "
        "silent truncation",
        [
            ("add", 1, seq(1, 10), 8),
            ("add", 2, seq(5, 10), 8),
            ("plan",), ("fin",),
            ("next", [11, 12]),
            ("plan",), ("fin",),
            ("next", [13, 14]),
            ("plan",), ("fin",),
            ("next", [15, 1]),
            ("plan",), ("fin",),
        ],
    ),
    (
        "e04_batch_full",
        "eight requests is the compiled limit: the ninth is refused by name "
        "instead of dropping someone already resident",
        [("add", 100 + i, seq(1 + i, 2), 4) for i in range(8)]
        + [("add", 999, seq(1, 2), 4), ("plan",), ("fin",)],
    ),
    (
        "e05_prompt_longer_than_a_step",
        "a prompt longer than the row block cannot be queued whole, and the "
        "refusal has to be a name rather than a partial prompt",
        [
            ("add", 5, seq(1, MAX_ROWS + 1), 4),
            ("plan",), ("fin",),
        ],
    ),
    (
        "e06_history_outgrows_its_region",
        "one request decoded until its history would pass the region it was "
        "given: the plan refuses rather than writing past the end",
        [("add", 7, seq(1, MAX_ROWS), 40)]
        + [("plan",), ("fin",), ("next", [1])] * 9
        + [("plan",), ("fin",)],
    ),
]


def main() -> None:
    written = []
    for name, comment, ops in SCENARIOS:
        written.append(write(name, comment, run(ops)))

    # Negative control one: the operations of e02 with the rows handed out in
    # reverse slot order. Same requests, same queues, different owner sequence --
    # which is what makes the byte comparison of the other scenarios a statement
    # about placement rather than about the line format.
    holes = [s for s in SCENARIOS if s[0] == "e02_hole_after_drop"][0]
    written.append(write("alt", holes[1] + " (policy: reverse slot order)",
                         run(holes[2], reverse=True)))

    # Negative control two: one row's window moved by one. Everything else is
    # byte-identical to e01, so this file is a statement about the `v=` field:
    # a row that may look one key further back is a row that can read a token
    # from the future or from another request.
    good = run(SCENARIOS[0][2])
    bad = list(good)
    for i, line in enumerate(bad):
        if line.startswith("ROW i=1 "):
            head, value = line.rsplit("v=", 1)
            bad[i] = head + "v=" + str(int(value) + 1)
            break
    written.append(write("bad", "one row's visible window moved by one key "
                                "-- must be refused", bad))

    print(f"wrote {len(written)} traces to {OUT}")
    for name in written:
        print("  " + name)


if __name__ == "__main__":
    main()
