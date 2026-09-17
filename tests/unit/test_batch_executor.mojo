"""The gate for the busy loop: a batch must be a set of independent sequences.

Two claims are checked here, and they are different in kind.

**The layout is a decision, so it is compared byte for byte.** Which request owns
which rows, and how far back each row may look, are not numbers that can be
nearly right: a layer reading request A's row at request B's slot is reading a
real tensor, so the failure is invisible in the output. The expected lines come
from `scripts/dump_batch_reference.py`'s sibling
`scripts/dump_batch_executor_reference.py`, written from the rules and sharing no
code with this tree. Six scenarios, compared exactly.

**Batching must not change arithmetic.** The second half needs no fixture: with
one sequence in the batch, `row_rope` must agree with `rope` and
`segmented_attention` must agree with `attention` **bit for bit** — same
`Float64` accumulation, same order, same three softmax passes. That is what makes
the later claim ("a batch of 8 generates the same tokens as 8 serial runs") a
statement about batching rather than about luck. Two sequences packed together
must then equal the two computed apart, which is the isolation property: if row
4 could see row 0's keys, this comparison is the one that notices.

The negative controls are permanent, and each one is required to be *capable* of
failing: `alt.trace` hands the rows out in reverse slot order and the suite checks
that this really changes the owner sequence (otherwise the five other byte
comparisons would only be checking that the file format round-trips), and
`bad.trace` moves one row's window by one key. The cross-request control
deliberately gives a row another request's base address and asserts that the
answer moves — a gate that cannot fail is not a gate.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_batch_executor.mojo
"""

from alofa.core.error import AlofaError
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import f32_data
from alofa.core.text import parse_int, read_text
from alofa.engine.batch import MAX_BATCH
from alofa.engine.executor import (
    BatchExecutor,
    MAX_ROWS,
    base_map,
    int_map,
)
from alofa.kernels.cpu.scalar import attention, rope
from alofa.kernels.cpu.segments import (
    PtrPtr,
    row_rope,
    segmented_attention,
    select_rows,
)
from alofa.model.arch.qwen import rows_view
from std.testing import TestSuite, assert_equal, assert_true

comptime FIXTURE = "tests/fixtures/batchexec/"
comptime SOURCE = "src/alofa/engine/executor.mojo"
comptime ANTIGEN = "tests/fixtures/bad_executor_alloc.mojo"
comptime FORMAT_VERSION = "# alofa batch executor trace v1"
comptime SCENARIOS = 6
comptime REPEATS = 5

# The toy model the reference was written against. If these move, the fixtures
# stop describing this build and the suite says so instead of agreeing.
comptime HIDDEN = 8
comptime INTER = 12
comptime KV_DIM = 4
comptime VOCAB = 16
comptime N_LAYERS = 2
comptime N_HEADS = 2
comptime N_KV_HEADS = 1
comptime HEAD_DIM = 4
comptime MAX_POS = 24
comptime TEST_ROWS = 16

# Kernel shapes for the bit-exact comparisons.
comptime K_HEADS = 2
comptime K_KV_HEADS = 1
comptime K_HEAD_DIM = 8
comptime K_ROWS = 4
comptime K_TABLE = 32


def lines_of(path: String) raises AlofaError -> InlineArray[String, 512]:
    var out = InlineArray[String, 512](fill="")
    var n = 0
    for line in read_text(path).split("\n"):
        if n >= 512:
            raise AlofaError(17, "fixture too long")
        out[n] = String(line)
        n += 1
    return out^


def split_fields(text: String) raises AlofaError -> InlineArray[String, 16]:
    var out = InlineArray[String, 16](fill="")
    var n = 0
    for part in text.split(" "):
        if n >= 16:
            raise AlofaError(17, "too many fields")
        out[n] = String(part)
        n += 1
    return out^


def value_after(fields: InlineArray[String, 16], key: String) -> String:
    for i in range(16):
        if fields[i].find(key + "=") == 0:
            return String(fields[i][byte=key.byte_length() + 1 :])
    return ""


def parts_of(text: String, sep: String) raises AlofaError -> InlineArray[String, 2]:
    var out = InlineArray[String, 2](fill="")
    var at = text.find(sep)
    if at < 0:
        out[0] = text
        return out^
    out[0] = String(text[byte=0:at])
    out[1] = String(text[byte=at + sep.byte_length() :])
    return out^


def ints_of(text: String) raises AlofaError -> InlineArray[Int, 64]:
    """`1,2,3` or `-` — the two shapes a per-slot list takes in a trace."""
    var out = InlineArray[Int, 64](fill=-1)
    var n = 0
    if text == "-":
        return out^
    for part in text.split(","):
        if n >= 64:
            raise AlofaError(17, "too many values")
        var field = String(part)
        if field == "-":
            out[n] = -1
        else:
            out[n] = parse_int(field)
        n += 1
    return out^


def scenario_name(index: Int) raises AlofaError -> String:
    if index == 0:
        return FIXTURE + "e01_two_requests.trace"
    if index == 1:
        return FIXTURE + "e02_hole_after_drop.trace"
    if index == 2:
        return FIXTURE + "e03_more_than_one_step.trace"
    if index == 3:
        return FIXTURE + "e04_batch_full.trace"
    if index == 4:
        return FIXTURE + "e05_prompt_longer_than_a_step.trace"
    if index == 5:
        return FIXTURE + "e06_history_outgrows_its_region.trace"
    raise AlofaError(17, "no such scenario")


def render_state(ex: BatchExecutor) raises AlofaError -> String:
    """`h=<id:history> p=<id:queued>` for every resident request, slot order."""
    var hist = ""
    var queue = ""
    for i in range(MAX_BATCH):
        if ex.live[i] != 1:
            continue
        if hist.byte_length() > 0:
            hist += ","
            queue += ","
        hist += String(ex.req[i]) + ":" + String(ex.hist[i])
        queue += String(ex.req[i]) + ":" + String(ex.pending_of(ex.req[i]))
    if hist.byte_length() == 0:
        hist = "-"
        queue = "-"
    return "h=" + hist + " p=" + queue


def check_cfg(line: String) raises:
    """The trace has to describe this build, not merely a plausible one."""
    var fields = split_fields(line)
    assert_equal(parse_int(value_after(fields, "mb")), MAX_BATCH)
    assert_equal(parse_int(value_after(fields, "mr")), TEST_ROWS)
    assert_equal(parse_int(value_after(fields, "mp")), MAX_POS)
    assert_equal(parse_int(value_after(fields, "h")), HIDDEN)
    assert_equal(parse_int(value_after(fields, "i")), INTER)
    assert_equal(parse_int(value_after(fields, "k")), KV_DIM)
    assert_equal(parse_int(value_after(fields, "v")), VOCAB)
    assert_equal(parse_int(value_after(fields, "l")), N_LAYERS)
    assert_equal(parse_int(value_after(fields, "nh")), N_HEADS)
    assert_equal(parse_int(value_after(fields, "nk")), N_KV_HEADS)
    assert_equal(parse_int(value_after(fields, "hd")), HEAD_DIM)
    assert_equal(parse_int(value_after(fields, "c")), scratch_capacity())


def cfg_line(path: String) raises AlofaError -> String:
    """The `# CFG` line, found by its marker rather than by its line number.

    Counting lines would make the suite agree with any file whose header is one
    comment longer, which is exactly the edit that would silently change what
    the fixtures describe.
    """
    for line in lines_of(path):
        if line.find("# CFG") == 0:
            return String(line[byte=2:])
    raise AlofaError(17, "trace has no CFG line")


def scratch_capacity() -> Int:
    var total = 6 * TEST_ROWS * HIDDEN * 4
    total += 3 * TEST_ROWS * KV_DIM * 4
    total += 3 * TEST_ROWS * INTER * 4
    total += TEST_ROWS * MAX_POS * 4
    total += MAX_BATCH * HIDDEN * 4
    total += MAX_BATCH * VOCAB * 4
    return ((total + 16 * 64 + 63) // 64) * 64


def build_executor() raises AlofaError -> BatchExecutor:
    return BatchExecutor(
        HIDDEN,
        INTER,
        KV_DIM,
        N_LAYERS,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        VOCAB,
        Float32(0.000001),
        MAX_POS,
        TEST_ROWS,
    )


def replay(
    mut ex: BatchExecutor, path: String, mut arena: Arena
) raises AlofaError -> Bool:
    """Replay one trace. Returns False on the first disagreement."""
    var tokens = int_map(arena.alloc(64 * 8))
    var chosen = int_map(arena.alloc(MAX_BATCH * 8))
    for line in lines_of(path):
        if line.find("#") == 0 or line.byte_length() == 0:
            continue
        var halves = parts_of(line, " R=")
        var op = halves[0]
        var expected = halves[1]
        var fields = split_fields(op)
        var observed = ""

        if op.find("ADD") == 0:
            var request = parse_int(value_after(fields, "r"))
            var budget = parse_int(value_after(fields, "b"))
            var n = parse_int(value_after(fields, "n"))
            var list = ints_of(value_after(fields, "k"))
            for i in range(n):
                tokens[unsafe_offset=i] = list[i]
            try:
                var slot = ex.slot_of(request)
                ex.add(request, tokens, n, budget)
                slot = ex.slot_of(request)
                observed = (
                    "i="
                    + String(slot)
                    + " n="
                    + String(ex.pending_of(request))
                    + " d="
                    + String(ex.digest())
                )
            except err:
                observed = "ERR=" + err.name()
        elif op.find("FEED") == 0:
            var request = parse_int(value_after(fields, "r"))
            var token = parse_int(value_after(fields, "k"))
            try:
                ex.feed(request, token)
                observed = (
                    "p=" + String(ex.pending_of(request)) + " d=" + String(ex.digest())
                )
            except err:
                observed = "ERR=" + err.name()
        elif op.find("DROP") == 0:
            var request = parse_int(value_after(fields, "r"))
            try:
                var slot = ex.slot_of(request)
                ex.drop(request)
                observed = "i=" + String(slot) + " d=" + String(ex.digest())
            except err:
                observed = "ERR=" + err.name()
        elif op.find("PLAN") == 0:
            try:
                var rows = ex.plan()
                observed = (
                    "T="
                    + String(rows)
                    + " n="
                    + String(ex.slots.n_live())
                    + " u="
                    + String(ex.pool.used)
                    + " d="
                    + String(ex.digest())
                )
            except err:
                observed = "ERR=" + err.name()
        elif op.find("FIN") == 0:
            ex.finish()
            observed = "u=0 d=" + String(ex.digest())
        elif op.find("NEXT") == 0:
            var list = ints_of(value_after(fields, "c"))
            for i in range(MAX_BATCH):
                chosen[unsafe_offset=i] = list[i]
            try:
                ex.advance(chosen)
                observed = render_state(ex) + " d=" + String(ex.digest())
            except err:
                observed = "ERR=" + err.name()
        elif op.find("ROW") == 0:
            var row = parse_int(value_after(fields, "i"))
            if row >= ex.rows:
                return False
            var owner = ex.req[ex.row_req[unsafe_offset=row]]
            observed = (
                "ROW i="
                + String(row)
                + " r="
                + String(owner)
                + " t="
                + String(ex.row_tok[unsafe_offset=row])
                + " p="
                + String(ex.row_pos[unsafe_offset=row])
                + " v="
                + String(ex.row_upto[unsafe_offset=row])
            )
            if observed != line:
                return False
            continue
        else:
            raise AlofaError(17, "unknown operation in trace")

        if observed != expected:
            return False
        # Re-derived after every line, not once at the end: a treaty broken for
        # one operation and repaired by the next is still broken.
        if ex.defects != 0:
            return False
        if ex.slots.defects() != 0:
            return False
    return True


def test_every_scenario_matches_the_reference_byte_for_byte() raises:
    for i in range(SCENARIOS):
        var path = scenario_name(i)
        assert_equal(lines_of(path)[0], FORMAT_VERSION)
        var arena = Arena(1 << 16)
        var ex = build_executor()
        check_cfg(cfg_line(path))
        # Named, because "a scenario disagrees" tells you nothing about which
        # rule broke, and the six scenarios are there to disagree in six ways.
        assert_true(
            replay(ex, path, arena),
            "scenario disagrees with the reference: " + path,
        )
        arena.keep_alive()


def test_the_alt_policy_is_rejected_and_really_differs() raises:
    var good = lines_of(FIXTURE + "e02_hole_after_drop.trace")
    var alt = lines_of(FIXTURE + "alt.trace")

    # It has to move something. A control that cannot fail is decoration, and
    # without this check the other five comparisons would only be evidence that
    # the file format round-trips.
    var moved = 0
    var i = 0
    while i < 512:
        if good[i].find("ROW") != 0:
            i += 1
            continue
        if good[i] != alt[i]:
            moved += 1
        i += 1
    assert_true(moved > 0, "the two placement policies agree everywhere")

    var arena = Arena(1 << 16)
    var ex = build_executor()
    assert_true(
        not replay(ex, FIXTURE + "alt.trace", arena),
        "the wrong row order was accepted",
    )
    arena.keep_alive()


def test_a_mutated_window_is_rejected() raises:
    var arena = Arena(1 << 16)
    var ex = build_executor()
    assert_true(
        not replay(ex, FIXTURE + "bad.trace", arena),
        "one row seeing one key too far was accepted",
    )
    arena.keep_alive()


def test_the_batched_kernels_agree_with_the_serial_ones() raises:
    """One sequence in the batch: the packed path must be the serial path.

    Bit for bit, not "close". The arithmetic was copied so that this holds, and
    the whole point of the copy is that the model-level comparison later is
    comparing batching against batching, with numerics taken out of the question.
    """
    var arena = Arena(1 << 20)
    var base = arena.alloc(1 << 19)
    var h = K_HEADS * K_HEAD_DIM
    var kv_dim = K_KV_HEADS * K_HEAD_DIM

    var q = rows_view(base.unsafe_offset(0), K_ROWS, h)
    var k = rows_view(base.unsafe_offset(4096), K_ROWS, kv_dim)
    var v = rows_view(base.unsafe_offset(8192), K_ROWS, kv_dim)
    var cos = rows_view(base.unsafe_offset(12288), K_TABLE, K_HEAD_DIM)
    var sin = rows_view(base.unsafe_offset(20480), K_TABLE, K_HEAD_DIM)
    var serial_q = rows_view(base.unsafe_offset(28672), K_ROWS, h)
    var serial_k = rows_view(base.unsafe_offset(32768), K_ROWS, kv_dim)
    var batched_q = rows_view(base.unsafe_offset(36864), K_ROWS, h)
    var batched_k = rows_view(base.unsafe_offset(40960), K_ROWS, kv_dim)
    var attn_serial = rows_view(base.unsafe_offset(45056), K_ROWS, h)
    var attn_batched = rows_view(base.unsafe_offset(49152), K_ROWS, h)
    var scores = rows_view(base.unsafe_offset(53248), K_ROWS, K_TABLE)

    for i in range(K_ROWS * h):
        f32_data(q)[unsafe_offset=i] = Float32(i % 13) * Float32(0.25)
    for i in range(K_ROWS * kv_dim):
        f32_data(k)[unsafe_offset=i] = Float32(i % 7) * Float32(0.5)
        f32_data(v)[unsafe_offset=i] = Float32(i % 11) * Float32(0.125)
    for i in range(K_TABLE * K_HEAD_DIM):
        f32_data(cos)[unsafe_offset=i] = Float32(i % 5) * Float32(0.1)
        f32_data(sin)[unsafe_offset=i] = Float32(i % 3) * Float32(0.05)

    # One sequence: every row is given the same base addresses the serial call
    # is given, and its own position. What is being compared is the batch
    # plumbing, so the per-row inputs have to say "one sequence" and nothing
    # else.
    var pos = int_map(arena.alloc(64 * 8))
    var upto = int_map(arena.alloc(64 * 8))
    var k_base = base_map(arena.alloc(64 * 8))
    var v_base = base_map(arena.alloc(64 * 8))
    for i in range(K_ROWS):
        pos[unsafe_offset=i] = i
        upto[unsafe_offset=i] = i
        k_base[unsafe_offset=i] = base.unsafe_offset(4096)
        v_base[unsafe_offset=i] = base.unsafe_offset(8192)

    rope(
        serial_q,
        serial_k,
        q,
        k,
        cos.slice_dim(0, 0, K_ROWS),
        sin.slice_dim(0, 0, K_ROWS),
        K_HEAD_DIM,
    )
    row_rope(batched_q, batched_k, q, k, cos, sin, K_HEAD_DIM, pos)

    var wrong_rope = 0
    for i in range(K_ROWS * h):
        if f32_data(serial_q)[unsafe_offset=i] != f32_data(batched_q)[unsafe_offset=i]:
            wrong_rope += 1
    for i in range(K_ROWS * kv_dim):
        if f32_data(serial_k)[unsafe_offset=i] != f32_data(batched_k)[unsafe_offset=i]:
            wrong_rope += 1
    assert_equal(wrong_rope, 0)

    attention(
        attn_serial,
        serial_q,
        k,
        v,
        rows_view(scores.data, K_ROWS, K_ROWS),
        K_HEADS,
        K_KV_HEADS,
        K_HEAD_DIM,
    )
    segmented_attention(
        attn_batched,
        batched_q,
        k_base,
        v_base,
        upto,
        scores,
        K_HEADS,
        K_KV_HEADS,
        K_HEAD_DIM,
    )
    var wrong_attn = 0
    for i in range(K_ROWS * h):
        if f32_data(attn_serial)[unsafe_offset=i] != f32_data(
            attn_batched
        )[unsafe_offset=i]:
            wrong_attn += 1
    assert_equal(wrong_attn, 0)
    arena.keep_alive()


def test_two_sequences_packed_are_two_sequences_apart() raises:
    """Isolation: a row must not reach another request's history.

    Two sequences share one matrix here. The packed result has to equal each
    sequence computed on its own, which is the property a shared row block would
    break silently — and the last assertion is the control: hand one row the
    other request's keys and the answer moves, so this comparison is not one that
    agrees no matter what.
    """
    var arena = Arena(1 << 20)
    var base = arena.alloc(1 << 19)
    var h = K_HEADS * K_HEAD_DIM
    var kv_dim = K_KV_HEADS * K_HEAD_DIM
    var rows_a = 3
    var rows_b = 2
    var rows = rows_a + rows_b

    # Sequence A: 3 keys. Sequence B: 2 keys, in its own region.
    var k_a = rows_view(base.unsafe_offset(0), rows_a, kv_dim)
    var v_a = rows_view(base.unsafe_offset(1024), rows_a, kv_dim)
    var k_b = rows_view(base.unsafe_offset(2048), rows_b, kv_dim)
    var v_b = rows_view(base.unsafe_offset(3072), rows_b, kv_dim)
    var q = rows_view(base.unsafe_offset(4096), rows, h)
    var attn_packed = rows_view(base.unsafe_offset(8192), rows, h)
    var attn_alone_a = rows_view(base.unsafe_offset(12288), rows_a, h)
    var attn_alone_b = rows_view(base.unsafe_offset(16384), rows_b, h)
    var scores = rows_view(base.unsafe_offset(20480), rows, K_TABLE)

    for i in range(rows * h):
        f32_data(q)[unsafe_offset=i] = Float32(i % 17) * Float32(0.2)
    for i in range(rows_a * kv_dim):
        f32_data(k_a)[unsafe_offset=i] = Float32(i % 5) * Float32(0.4)
        f32_data(v_a)[unsafe_offset=i] = Float32(i % 9) * Float32(0.3)
    for i in range(rows_b * kv_dim):
        f32_data(k_b)[unsafe_offset=i] = Float32(i % 6) * Float32(0.6)
        f32_data(v_b)[unsafe_offset=i] = Float32(i % 4) * Float32(0.7)

    var upto = int_map(arena.alloc(64 * 8))
    var k_base = base_map(arena.alloc(64 * 8))
    var v_base = base_map(arena.alloc(64 * 8))
    for i in range(rows_a):
        upto[unsafe_offset=i] = i
        k_base[unsafe_offset=i] = base.unsafe_offset(0)
        v_base[unsafe_offset=i] = base.unsafe_offset(1024)
    for i in range(rows_b):
        upto[unsafe_offset=rows_a + i] = i
        k_base[unsafe_offset=rows_a + i] = base.unsafe_offset(2048)
        v_base[unsafe_offset=rows_a + i] = base.unsafe_offset(3072)

    segmented_attention(
        attn_packed, q, k_base, v_base, upto, scores, K_HEADS, K_KV_HEADS, K_HEAD_DIM
    )
    attention(
        attn_alone_a,
        q.slice_dim(0, 0, rows_a),
        k_a,
        v_a,
        rows_view(base.unsafe_offset(20480), rows_a, rows_a),
        K_HEADS,
        K_KV_HEADS,
        K_HEAD_DIM,
    )
    attention(
        attn_alone_b,
        q.slice_dim(0, rows_a, rows_b),
        k_b,
        v_b,
        rows_view(base.unsafe_offset(24576), rows_b, rows_b),
        K_HEADS,
        K_KV_HEADS,
        K_HEAD_DIM,
    )

    var wrong = 0
    for i in range(rows_a * h):
        if f32_data(attn_packed)[unsafe_offset=i] != f32_data(
            attn_alone_a
        )[unsafe_offset=i]:
            wrong += 1
    for i in range(rows_b * h):
        if f32_data(attn_packed)[unsafe_offset=rows_a * h + i] != f32_data(
            attn_alone_b
        )[unsafe_offset=i]:
            wrong += 1
    assert_equal(wrong, 0)

    # Control: give B's rows A's keys and A's windows. If the answer were the
    # same, the comparison above would be agreeing regardless of isolation.
    var leak = rows_view(base.unsafe_offset(28672), rows, h)
    for i in range(rows_b):
        upto[unsafe_offset=rows_a + i] = rows_a + i
        k_base[unsafe_offset=rows_a + i] = base.unsafe_offset(0)
        v_base[unsafe_offset=rows_a + i] = base.unsafe_offset(1024)
    segmented_attention(
        leak, q, k_base, v_base, upto, scores, K_HEADS, K_KV_HEADS, K_HEAD_DIM
    )
    var moved = 0
    for i in range(rows_b * h):
        if f32_data(leak)[unsafe_offset=rows_a * h + i] != f32_data(
            attn_packed
        )[unsafe_offset=rows_a * h + i]:
            moved += 1
    assert_true(moved > 0, "cross-request attention changed nothing")
    arena.keep_alive()


def test_select_rows_gathers_exactly_the_named_rows() raises:
    var arena = Arena(1 << 18)
    var base = arena.alloc(1 << 17)
    var src = rows_view(base.unsafe_offset(0), 5, 4)
    var dst = rows_view(base.unsafe_offset(1024), 3, 4)
    for i in range(20):
        f32_data(src)[unsafe_offset=i] = Float32(i)
    var which = int_map(arena.alloc(64 * 8))
    which[unsafe_offset=0] = 4
    which[unsafe_offset=1] = 0
    which[unsafe_offset=2] = 2
    select_rows(dst, src, which, 3)

    var wrong = 0
    var wanted = InlineArray[Int, 3](fill=0)
    wanted[0] = 4
    wanted[1] = 0
    wanted[2] = 2
    for i in range(3):
        for c in range(4):
            if f32_data(dst)[unsafe_offset=i * 4 + c] != Float32(
                wanted[i] * 4 + c
            ):
                wrong += 1
    assert_equal(wrong, 0)
    arena.keep_alive()


def test_the_step_gives_every_byte_back() raises:
    """Steady state, as something that could fail.

    A step borrows and returns. If it forgot any part of what it took, the pool
    would slowly run out under a load that never changes — the failure that shows
    up as a capacity error hours into a run, on a batch that used to fit.
    """
    var arena = Arena(1 << 16)
    var tokens = int_map(arena.alloc(64 * 8))
    for i in range(3):
        tokens[unsafe_offset=i] = i + 1
    var ex = build_executor()
    ex.add(7, tokens, 3, 40)
    ex.add(9, tokens, 2, 40)

    var chosen = int_map(arena.alloc(MAX_BATCH * 8))
    chosen[unsafe_offset=0] = 1
    chosen[unsafe_offset=1] = 2

    var first = 0
    var served = 0
    for i in range(REPEATS):
        # The first step is the two prompts; every step after it serves one
        # token per request, because one fed-back token is all a request has
        # queued once its prompt has been consumed.
        var rows = ex.plan()
        assert_equal(rows, 5 if i == 0 else 2)
        if i == 0:
            first = ex.pool.high_water
        assert_equal(ex.pool.high_water, first)
        ex.finish()
        assert_equal(ex.pool.used, 0)
        assert_equal(ex.pool.n_live(), 0)
        assert_equal(ex.defects, 0)
        ex.advance(chosen)
        served += rows
    assert_equal(ex.history_of(7), 3 + (REPEATS - 1))
    assert_equal(ex.history_of(9), 2 + (REPEATS - 1))
    assert_equal(served, 5 + 2 * (REPEATS - 1))
    arena.keep_alive()


def test_refusals_are_named() raises:
    var arena = Arena(1 << 16)
    var tokens = int_map(arena.alloc(MAX_ROWS * 8 + 64))
    for i in range(MAX_ROWS + 1):
        tokens[unsafe_offset=i] = 1
    var ex = build_executor()

    var name = "none"
    try:
        ex.add(1, tokens, MAX_ROWS + 1, 4)
    except err:
        name = err.name()
    assert_equal(name, "capacity")

    for i in range(MAX_BATCH):
        ex.add(100 + i, tokens, 1, 4)
    name = "none"
    try:
        ex.add(999, tokens, 1, 4)
    except err:
        name = err.name()
    assert_equal(name, "capacity")

    name = "none"
    try:
        ex.add(100, tokens, 1, 4)
    except err:
        name = err.name()
    assert_equal(name, "invalid_argument")

    name = "none"
    try:
        ex.feed(321, 1)
    except err:
        name = err.name()
    assert_equal(name, "invalid_argument")

    # A token outside the vocabulary is refused here rather than read from
    # whatever row the embedding table happens to have there.
    name = "none"
    try:
        ex.feed(100, VOCAB)
    except err:
        name = err.name()
    assert_equal(name, "out_of_range")
    arena.keep_alive()


def test_the_busy_loop_source_keeps_no_growable_container() raises:
    """Read out of the source, not out of a profile.

    The boundary: this says no member is a growable container and none is
    constructed on any path. It does not say the loop never asks the allocator
    for anything — the weight names and the shape metadata inside `rows_view` do
    allocate, and they are bounded. What is bounded is what matters: the scratch
    is a pool sized once.
    """
    var text = read_text(SOURCE)
    for token in ["List[", "Dict[", "Set[", "DynamicVector", "InlinedFixedVector"]:
        assert_true(text.find(token) < 0, "executor source grows")


def test_the_zero_alloc_gate_can_fail() raises:
    var text = read_text(ANTIGEN)
    var hits = 0
    for token in ["List[", "Dict[", "Set[", "DynamicVector"]:
        if text.find(token) >= 0:
            hits += 1
    assert_true(hits >= 3, "the antigen no longer contains what the gate rejects")


def test_a_cached_prefix_is_history_the_step_never_runs() raises:
    """A matched prefix is queued but not computed.

    Two requests, one prompt. The second one's first `k` tokens already have
    their K/V — another request computed them and the room kept the blocks — so
    queueing them again would be a decision to spend the arithmetic twice. What
    a shared prefix is worth is not only the blocks: it is the rows, and rows
    are the arithmetic.

    The last token is the exception and it is kept: its logits are the first
    generated token, so a prompt that is entirely cached still runs one row. A
    request with no rows has nothing to speak from.
    """
    var arena = Arena(1 << 16)
    var tokens = int_map(arena.alloc(64 * 8))
    var n = 12
    for i in range(n):
        tokens[unsafe_offset=i] = i % VOCAB
    var ex = build_executor()

    ex.add(1, tokens, n, 8)
    assert_equal(ex.pending_of(1), n, "a prompt nobody cached was not queued whole")
    assert_equal(ex.history_of(1), 0, "a prompt nobody cached claims history")

    var k = 7
    ex.add(2, tokens, n, 8, n, k)
    assert_equal(
        ex.history_of(2), k, "a cached prefix is not where the history starts"
    )
    assert_equal(
        ex.pending_of(2), n - k, "a cached prefix was queued like any other token"
    )

    ex.add(3, tokens, n, 8, n, n)
    assert_equal(
        ex.history_of(3), n - 1, "a wholly cached prompt dropped its last token too"
    )
    assert_equal(
        ex.pending_of(3), 1, "a wholly cached prompt stopped running the last row"
    )

    var kind = String("")
    try:
        ex.add(4, tokens, n, 8, n, n + 1)
    except err:
        kind = err.name()
    assert_equal(
        kind,
        "out_of_range",
        "a prefix longer than the prompt was accepted, got " + kind,
    )
    arena.keep_alive()


def test_the_rows_a_cached_prefix_saves_are_not_built() raises:
    """The saving is rows, and it is asserted on the row block.

    `plan` is what turns a queue into rows. Four tokens cached out of twelve has
    to come out as eight rows, starting at position four — a step that built
    twelve rows would have saved the blocks and nothing else.
    """
    var arena = Arena(1 << 16)
    var tokens = int_map(arena.alloc(64 * 8))
    var n = 12
    for i in range(n):
        tokens[unsafe_offset=i] = i % VOCAB
    var ex = build_executor()
    ex.add(7, tokens, n, 8, n, 4)
    var rows = ex.plan()
    assert_equal(rows, n - 4, "the cached tokens were built anyway")
    var first_pos = ex.row_pos[unsafe_offset=0]
    assert_equal(first_pos, 4, "the first row did not start at the cached length")
    ex.finish()
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
