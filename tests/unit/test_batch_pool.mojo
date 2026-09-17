"""The gate for 2.5: what a batch may borrow, and what it may grow into.

Every layer of a forward wants scratch, and the tempting way to serve a batch is
to allocate whatever this batch happens to need. That repairs batching at the
cost of the thing P2 is buying: an allocation inside the busy loop is unbounded,
so a batch one request larger than any tested batch can fail in production. So
this suite holds two things down.

Hold down correctness with **byte-exact comparison against an independent
reference**. The expected lines are written by `scripts/dump_batch_reference.py`,
which re-derives the allocator from the same rules and shares no code with this
tree. Allocation decisions are what earlier layers called "decisions": two
placements are not nearly the same, one is right and the other is wrong, so there
is no tolerance here. The formula has not entered this layer yet.

Hold down **the refusal**. `s03` leaves 258560 free bytes and no usable gap; the
only honest answer is a named error. Wrapping around, or growing the pool, would
turn a capacity question into a plausible number. `s07` does the same for the
handle table: twenty-four live borrows is a cap, not a starting point.

Two properties are checked without any reference, because they are properties of
the implementation rather than of a trace:

- **no live borrow shares a byte with another.** Markers are written into real
  memory through `f32_of` and read back through every live handle. Overlap would
  corrupt at least one region, and unlike a wrong offset it would not need a
  reference to notice. This is the same failure as a paged kernel reading the
  wrong block: the result is a number, not a diagnosis.
- **the peak does not drift.** Twenty identical executor steps must leave the
  same high-water mark as the first. If it grows, the steady state is not steady
  and the pool is slowly leaking its own space.

Three negative controls stand permanently: `alt.trace` re-runs one scenario under
best-fit placement (and the suite checks that this really moves an offset, so the
other six comparisons are not just checking the line format), `bad.trace` moves
one offset by one alignment unit, and `bad_batch_alloc.mojo` is a source file the
zero-allocation gate must reject.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_batch_pool.mojo
"""

from alofa.core.error import AlofaError
from alofa.core.memory import Arena
from alofa.core.text import parse_int, read_text
from alofa.engine.batch import (
    MAX_BATCH,
    MAX_HANDLES,
    POOL_ALIGN,
    BatchPool,
    BatchSlots,
)
from std.testing import TestSuite, assert_equal, assert_true

comptime FIXTURE = "tests/fixtures/batch/"
comptime SOURCE = "src/alofa/engine/batch.mojo"
comptime ANTIGEN = "tests/fixtures/bad_batch_alloc.mojo"
comptime FORMAT_VERSION = "# alofa batch trace v1"
comptime SCENARIOS = 7
comptime STEPS = 20
comptime MARKERS = 12


def lines_of(path: String) raises -> InlineArray[String, 256]:
    """Fixture lines, so a path typo is a capacity error, not silent silence."""
    var out = InlineArray[String, 256](fill="")
    var n = 0
    for line in read_text(path).split("\n"):
        if n >= 256:
            raise AlofaError(17, "fixture too long")
        out[n] = String(line)
        n += 1
    return out^


def value_after(fields: InlineArray[String, 12], key: String) -> String:
    for i in range(12):
        if fields[i].find(key + "=") == 0:
            return String(fields[i][byte=key.byte_length() + 1 :])
    return ""


def split_fields(text: String) raises -> InlineArray[String, 12]:
    var out = InlineArray[String, 12](fill="")
    var n = 0
    for part in text.split(" "):
        if n >= 12:
            raise AlofaError(17, "too many fields")
        out[n] = String(part)
        n += 1
    return out^


def parts_of(text: String, sep: String) raises -> InlineArray[String, 2]:
    """Split on the first `sep`; the tail keeps whatever separators remain."""
    var out = InlineArray[String, 2](fill="")
    var at = text.find(sep)
    if at < 0:
        out[0] = text
        return out^
    out[0] = String(text[byte=0:at])
    out[1] = String(text[byte=at + sep.byte_length() :])
    return out^


def pool_fields(at: Int, handle: Int, used: Int, n: Int, d: Int) -> String:
    """One `R=` line, rendered here so both sides agree on nothing.

    Built from values passed in rather than from the pool, so this helper cannot
    accidentally see a field the reference does not have.
    """
    return (
        "o="
        + String(at)
        + " h="
        + String(handle)
        + " u="
        + String(used)
        + " n="
        + String(n)
        + " d="
        + String(d)
    )


def pool_result(at: Int, used: Int, n: Int, d: Int) -> String:
    return (
        "o="
        + String(at)
        + " u="
        + String(used)
        + " n="
        + String(n)
        + " d="
        + String(d)
    )


def pool_empty(used: Int, n: Int, d: Int) -> String:
    return "u=" + String(used) + " n=" + String(n) + " d=" + String(d)


def slot_fields(at: Int, total: Int, n: Int, d: Int) -> String:
    return (
        "t="
        + String(at)
        + " N="
        + String(total)
        + " b="
        + String(n)
        + " d="
        + String(d)
    )


def capacity_of(path: String) raises -> Int:
    """The `CFG` line of a trace; there is one, and it is the first non-comment."""
    var fields = split_fields(cfg_line(path))
    return parse_int(value_after(fields, "c"))


def scenario_name(index: Int) raises -> String:
    if index == 0:
        return FIXTURE + "s01_executor_step.trace"
    if index == 1:
        return FIXTURE + "s02_interleaved_lifetimes.trace"
    if index == 2:
        return FIXTURE + "s03_room_but_not_contiguous.trace"
    if index == 3:
        return FIXTURE + "s04_batch_turnover.trace"
    if index == 4:
        return FIXTURE + "s05_two_holes.trace"
    if index == 5:
        return FIXTURE + "s06_refusals_are_named.trace"
    if index == 6:
        return FIXTURE + "s07_handle_pressure.trace"
    raise AlofaError(17, "no such scenario")


def cfg_line(path: String) raises -> String:
    """The single non-comment line before the operations."""
    for line in lines_of(path):
        if line.find("#") == 0:
            continue
        if line.find("CFG") == 0:
            return line
        break
    raise AlofaError(17, "trace has no CFG line")


def check_cfg(path: String, capacity: Int) raises:
    """The trace has to be describing this build of the pool.

    Without this, moving `MAX_HANDLES` would silently change the numbers both
    sides compute and the comparison would still agree — the failure the KV suite
    guarded against with the same line.
    """
    var fields = split_fields(cfg_line(path))
    assert_equal(parse_int(value_after(fields, "mh")), MAX_HANDLES)
    assert_equal(parse_int(value_after(fields, "mb")), MAX_BATCH)
    assert_equal(parse_int(value_after(fields, "al")), POOL_ALIGN)
    assert_true(parse_int(value_after(fields, "c")) == capacity, "capacity disagrees")


def replay(
    mut pool: BatchPool,
    mut slots: BatchSlots,
    path: String,
) raises -> Bool:
    """Replay one trace against the reference's expected results.

    Returns False on the first disagreement rather than raising, so the negative
    controls can be run through the same path.
    """
    for line in lines_of(path):
        if line.find("#") == 0:
            continue
        if line.find("CFG") == 0:
            continue
        if line.byte_length() == 0:
            continue
        var halves = parts_of(line, " R=")
        var op = halves[0]
        var expected = halves[1]
        var fields = split_fields(op)
        var observed = ""

        if op.find("GET") == 0:
            var size = parse_int(value_after(fields, "s"))
            try:
                var handle = pool.borrow(size)
                var at = pool.offset_of(handle)
                observed = pool_fields(
                    at, handle, pool.used, pool.n_live(), pool.digest()
                )
            except err:
                observed = "ERR=" + err.name()
        elif op.find("REL") == 0:
            var handle = parse_int(value_after(fields, "h"))
            try:
                var at = pool.release(handle)
                observed = pool_result(at, pool.used, pool.n_live(), pool.digest())
            except err:
                observed = "ERR=" + err.name()
        elif op.find("CLR") == 0:
            pool.reset()
            slots.clear()
            observed = pool_empty(pool.used, pool.n_live(), pool.digest())
        elif op.find("ADD") == 0:
            var request = parse_int(value_after(fields, "r"))
            var count = parse_int(value_after(fields, "n"))
            try:
                var at = slots.add(request, count)
                observed = slot_fields(
                    at, slots.total_tokens(), slots.n_live(), slots.digest()
                )
            except err:
                observed = "ERR=" + err.name()
        elif op.find("DEL") == 0:
            var request = parse_int(value_after(fields, "r"))
            try:
                var at = slots.remove(request)
                observed = slot_fields(
                    at, slots.total_tokens(), slots.n_live(), slots.digest()
                )
            except err:
                observed = "ERR=" + err.name()
        else:
            raise AlofaError(17, "unknown operation in trace")

        if observed != expected:
            return False
        # Re-derived every line, not once at the end: a treaty broken for one
        # operation and repaired by the next is still broken.
        if pool.defects() != 0:
            return False
        if slots.defects() != 0:
            return False
    return True


def test_every_scenario_matches_the_reference_byte_for_byte() raises:
    for i in range(SCENARIOS):
        var path = scenario_name(i)
        var capacity = capacity_of(path)
        # Check the header before trusting a single number in the file.
        assert_equal(lines_of(path)[0], FORMAT_VERSION)
        var arena = Arena(1 << 22)
        var pool = BatchPool(arena.alloc(capacity), capacity)
        check_cfg(path, capacity)
        var slots = BatchSlots()
        assert_true(
            replay(pool, slots, path), "scenario disagrees with the reference"
        )
        arena.keep_alive()


def test_the_alt_policy_is_rejected_and_really_differs() raises:
    var good = lines_of(FIXTURE + "s05_two_holes.trace")
    var alt = lines_of(FIXTURE + "alt.trace")

    # It must actually move something: a control that cannot fail does not count
    # as a control, and the byte comparisons above would otherwise be evidence
    # that the file format round-trips and nothing else.
    var moved = 0
    for i in range(256):
        if good[i].find("#") == 0 or alt[i].find("#") == 0:
            continue
        if good[i].byte_length() == 0 and alt[i].byte_length() == 0:
            break
        if parts_of(good[i], " R=")[1] != parts_of(alt[i], " R=")[1]:
            moved += 1
    assert_true(moved > 0, "the two placement policies agree everywhere")
    assert_true(
        parts_of(good[9], " R=")[1].find("o=0 ") == 0,
        "first fit no longer puts this borrow at the bottom",
    )

    var capacity = capacity_of(FIXTURE + "alt.trace")
    var arena = Arena(1 << 22)
    var pool = BatchPool(arena.alloc(capacity), capacity)
    var slots = BatchSlots()
    assert_true(
        not replay(pool, slots, FIXTURE + "alt.trace"),
        "the wrong placement policy was accepted",
    )
    arena.keep_alive()


def test_a_mutated_result_is_rejected() raises:
    var capacity = capacity_of(FIXTURE + "bad.trace")
    var arena = Arena(1 << 22)
    var pool = BatchPool(arena.alloc(capacity), capacity)
    var slots = BatchSlots()
    assert_true(
        not replay(pool, slots, FIXTURE + "bad.trace"),
        "one wrong offset was accepted",
    )
    arena.keep_alive()


def test_the_capacity_is_refused_and_named() raises:
    var capacity = 64 * 1024
    var arena = Arena(1 << 20)
    var pool = BatchPool(arena.alloc(capacity), capacity)

    var whole = pool.borrow(capacity)
    assert_equal(pool.offset_of(whole), 0)
    assert_equal(pool.used, capacity)

    var name = "none"
    try:
        _ = pool.borrow(POOL_ALIGN)
    except err:
        name = err.name()
    assert_equal(name, "capacity")
    # Refusing must not have secretly made room, and must not have forgotten
    # what was already handed out.
    assert_equal(pool.used, capacity)
    assert_equal(pool.n_live(), 1)

    _ = pool.release(whole)
    assert_equal(pool.used, 0)
    var again = pool.borrow(capacity)
    assert_equal(pool.offset_of(again), 0)
    arena.keep_alive()


def test_there_is_room_but_not_contiguous() raises:
    """The refusal, without a reference to tell us it was coming.

    The pool is filled by sixteen regions, then every other one is given back:
    131072 bytes are free, no hole is bigger than one region, and there is no
    tail to fall back on. Splitting a double borrow across two holes is the
    repair a growable allocator would make, and it would hand the caller a pair
    of regions that are not adjacent.
    """
    var capacity = 262144
    var row = 16384
    var arena = Arena(1 << 22)
    var pool = BatchPool(arena.alloc(capacity), capacity)
    var handles = InlineArray[Int, 16](fill=0)
    for i in range(16):
        handles[i] = pool.borrow(row)
    for i in range(1, 16, 2):
        _ = pool.release(handles[i])

    var free = capacity - pool.used
    assert_true(free > 2 * row, "there should be plenty of room -- that is the point")

    var name = "none"
    try:
        _ = pool.borrow(2 * row)
    except err:
        name = err.name()
    assert_equal(name, "capacity")

    var fits = pool.borrow(row)
    assert_equal(pool.offset_of(fits), row)
    _ = pool.release(fits)
    for i in range(0, 16, 2):
        _ = pool.release(handles[i])
    assert_equal(pool.used, 0)
    assert_equal(pool.defects(), 0)
    arena.keep_alive()


def test_releasing_what_you_do_not_hold_is_a_named_error() raises:
    var capacity = 64 * 1024
    var arena = Arena(1 << 20)
    var pool = BatchPool(arena.alloc(capacity), capacity)
    var held = pool.borrow(1024)

    var name = "none"
    try:
        _ = pool.release(held + 1)
    except err:
        name = err.name()
    assert_equal(name, "double_free")

    _ = pool.release(held)
    name = "none"
    try:
        _ = pool.release(held)
    except err:
        name = err.name()
    assert_equal(name, "double_free")

    name = "none"
    try:
        _ = pool.release(MAX_HANDLES)
    except err:
        name = err.name()
    assert_equal(name, "out_of_range")

    name = "none"
    try:
        _ = pool.borrow(0)
    except err:
        name = err.name()
    assert_equal(name, "invalid_argument")
    arena.keep_alive()


def test_live_borrows_never_share_a_byte() raises:
    """The property that would be invisible in a number.

    Each live region is filled with a value derived from its own handle, then
    every region is read back. Sharing a single byte would corrupt at least one
    of them, and this needs no reference to know.
    """
    var capacity = 64 * 1024
    var arena = Arena(1 << 22)
    var pool = BatchPool(arena.alloc(capacity), capacity)
    var handles = InlineArray[Int, MARKERS](fill=0)
    var sizes = InlineArray[Int, MARKERS](fill=0)

    for i in range(MARKERS):
        # 64..1088 bytes, deliberately mixing sizes so gaps have to be reused.
        handles[i] = pool.borrow(64 + i * 64)
        sizes[i] = pool.length_of(handles[i])

    for i in range(MARKERS):
        if i % 3 != 0:
            continue
        _ = pool.release(handles[i])

    # Reuse the freed room; anything handed out now overlaps nothing by right,
    # only by accident if placement is wrong.
    for i in range(MARKERS):
        if i % 3 != 0:
            continue
        handles[i] = pool.borrow(64 + i * 64)
        sizes[i] = pool.length_of(handles[i])

    for i in range(MARKERS):
        var words = sizes[i] // 4
        var p = pool.f32_of(handles[i])
        for w in range(words):
            p[unsafe_offset=w] = Float32(i * 1000 + w)

    var wrong = 0
    for i in range(MARKERS):
        var words = sizes[i] // 4
        var p = pool.f32_of(handles[i])
        for w in range(words):
            var mine = Float32(i * 1000 + w)
            if p[unsafe_offset=w] != mine:
                wrong += 1
    assert_equal(wrong, 0)
    assert_equal(pool.defects(), 0)
    arena.keep_alive()


def test_the_peak_does_not_drift_over_steps() raises:
    """Steady state, stated as something that can go wrong.

    P2 gate five asks whether a steady decode loop touches the heap at all; this
    is the half of that claim a pool can make alone. If the high-water mark grew
    with each step the pool would be slowly eating its own space, and every
    throughput number taken after it began would be a lie about a different pool.
    """
    var capacity = 64 * 1024
    var arena = Arena(1 << 22)
    var pool = BatchPool(arena.alloc(capacity), capacity)
    var peak = 0

    for step in range(STEPS):
        var hidden = pool.borrow(896 * 8 * 4)
        var scores = pool.borrow(8 * 8 * 4)
        _ = pool.release(scores)
        _ = pool.release(hidden)
        if step == 0:
            peak = pool.high_water
        assert_equal(pool.high_water, peak)
        assert_equal(pool.used, 0)
        assert_equal(pool.defects(), 0)
    arena.keep_alive()


def test_batch_rows_are_disjoint_and_contiguous() raises:
    """A request owns one run of rows, and nobody else owns any of them.

    This recomputes the answer instead of asking the pool. It is the whole reason
    the rows exist: if two requests shared a row, a layer would read one
    request's activation out of another request's token, and every number it
    produced would look reasonable.
    """
    var slots = BatchSlots()
    var requests = InlineArray[Int, MAX_BATCH](fill=0)
    var counts = InlineArray[Int, MAX_BATCH](fill=0)

    for i in range(MAX_BATCH):
        var request = 100 + i * 3
        var count = 1 + i * 2
        requests[i] = request
        counts[i] = count
        var row = slots.add(request, count)
        var after = slots.total_tokens()
        assert_equal(row, after - count)

    # Turn one over: remove from the middle and put a different request there.
    assert_equal(slots.remove(requests[3]), 0 + counts[0] + counts[1] + counts[2])
    var heir = slots.add(777, counts[3])
    assert_equal(heir, counts[0] + counts[1] + counts[2])

    # Coverage recomputed from the outside: every row is owned exactly once.
    var total = slots.total_tokens()
    var owner = InlineArray[Int, 4096](fill=-1)
    for i in range(MAX_BATCH):
        var at = slots.slot_of(requests[i])
        if at < 0:
            continue
        for r in range(slots.off[at], slots.off[at] + slots.cnt[at]):
            owner[r] += 1
    var at2 = slots.slot_of(777)
    for r in range(slots.off[at2], slots.off[at2] + slots.cnt[at2]):
        owner[r] += 1

    var doubled = 0
    var uncovered = 0
    for r in range(total):
        if owner[r] < 0:
            uncovered += 1
        if owner[r] > 0:
            doubled += 1
    assert_equal(doubled, 0)
    assert_equal(uncovered, 0)
    assert_equal(slots.defects(), 0)


def test_a_request_is_never_counted_twice() raises:
    var slots = BatchSlots()
    var rows = slots.add(5, 4)
    assert_equal(rows, 0)

    var name = "none"
    try:
        _ = slots.add(5, 4)
    except err:
        name = err.name()
    assert_equal(name, "invalid_argument")
    # A refused add must not have reserved rows behind our back.
    assert_equal(slots.total_tokens(), 4)
    assert_equal(slots.n_live(), 1)

    name = "none"
    try:
        _ = slots.remove(6)
    except err:
        name = err.name()
    assert_equal(name, "invalid_argument")
    assert_equal(slots.n_live(), 1)


def test_the_pool_cannot_allocate_by_construction() raises:
    """Zero allocation read out of the source, not out of a profiling run.

    The boundary is worth stating plainly: this says no member is a growable
    container and no path constructs one. It does not say the kernel never asks
    libc for a small block, and it is not a claim about process RSS.
    """
    var text = read_text(SOURCE)
    for token in [
        "List[",
        "Dict[",
        "Set[",
        "String(",
        "Arena(",
        "InlinedFixedVector",
        "DynamicVector",
    ]:
        assert_true(text.find(token) < 0, "batch source allocates")


def test_the_zero_alloc_gate_can_fail() raises:
    """The source gate above must be capable of refusing a file."""
    var text = read_text(ANTIGEN)
    var hits = 0
    for token in ["List[", "Dict[", "Set[", "String("]:
        if text.find(token) >= 0:
            hits += 1
    assert_true(hits >= 3, "the antigen no longer contains what the gate rejects")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
