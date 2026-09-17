"""Differential test: a batch of requests against the same requests run alone.

This is the claim the batch layer exists for, and it is not the claim the unit
gate makes. `test_batch_executor.mojo` shows that rows are owned, that refusals
are named and that a step returns every byte it borrowed; none of that says the
arithmetic came out right. Here the same prompts are decoded twice — once as
`N` requests in one batch, once as `N` separate single-request runs through the
model's own serial path — and the two token sequences have to be **equal**, not
close. Greedy decoding is used on both sides so that nothing random stands
between the two: a difference of one token at position 3 is a difference.

**What the reference is, and what it is not.** The reference is this tree's
serial forward (`QwenForward.prefill` / `step`), the path `test_model_parity`
checks against Hugging Face token for token over 128 tokens. It shares the
kernels with the batched path, and that is deliberate: what is being compared is
orchestration — row ownership, per-row positions, per-row key windows, the KV
regions — against a path that has one request and therefore cannot get any of
those wrong. A disagreement is a batching bug or nothing.

**Why batch sizes 1, 2, 4 and 8 and not just 8.** A batch of one must equal a
single request trivially; a batch of two can pass while the two requests share a
row block by accident of layout; eight with prompts of four different lengths is
where a request that waits, or a prompt that is split across two steps, shows
up. The row block is 64 rows and eight prompts are 88, so this suite *does*
split a prefill: a short step has to be a correct step, which is the property
the unit gate can only state.

Negative controls are permanent. Two different prompts must produce different
continuations — otherwise "the batch agrees with the serial run" would be true
of any implementation, including one that ignores its input. And swapping two
requests' prompts must be *noticed*: the swapped batch has to follow the
prompt it was actually given, which is what makes the byte comparison evidence
about row ownership rather than about the model being insensitive.

Build and run — heavy, like the other whole-model gates (2 GB of weights, a few
minutes), and therefore not part of `pixi run test`:

    pixi run mojo build -O2 -I src tests/unit/test_batch_forward.mojo -o target/batch_forward
    ./target/batch_forward
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.memory import Arena
from alofa.core.text import parse_int, read_text
from alofa.engine.executor import (
    KvPageTable,
    MAX_GEN,
    BatchExecutor,
    IntPtr,
    int_map,
)
from alofa.model.arch.qwen import QwenForward
from alofa.model.loader import config_value

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime WEIGHTS_DIR = FIXTURE + "/weights"
comptime CONFIG = FIXTURE + "/config.tsv"
comptime PROMPTS = FIXTURE + "/prompts.tsv"

# Short, because what is being compared is batching: four greedy tokens per
# request is enough for one wrong token to be a difference, and the whole suite
# is a few hundred forwards of a 0.5B model on the scalar backend.
comptime STEPS = 4
comptime MAX_POS = 64
comptime MAX_ROWS = 64
comptime MAX_TOKENS = 64
comptime BASE_REQUEST = 1000
# Blocks reserved per request. Eight each keeps eight requests inside the pool,
# which holds `MAX_BLOCKS`.
comptime BLOCKS_PER_REQUEST = 8


def cfg_int(key: String) raises AlofaError -> Int:
    return parse_int(config_value(CONFIG, key))


def load_prompts() raises AlofaError -> List[List[Int]]:
    """Prompt token ids, one list per line of the fixture."""
    var text = read_text(PROMPTS)
    var prompts_out = List[List[Int]]()
    for line_span in text.split("\n"):
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        var ids_text = String(fields[1])
        var ids = List[Int]()
        for part_span in ids_text.split(","):
            var part = String(part_span)
            if part.byte_length() > 0:
                ids.append(parse_int(part))
        prompts_out.append(ids^)
    return prompts_out^


def build_model() raises AlofaError -> QwenForward:
    return QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)


def build_executor() raises AlofaError -> BatchExecutor:
    """An executor over the same shapes as the model, sized for this suite."""
    var n_kv_heads = cfg_int("n_kv_heads")
    var head_dim = cfg_int("head_dim")
    return BatchExecutor(
        cfg_int("hidden"),
        cfg_int("intermediate"),
        n_kv_heads * head_dim,
        cfg_int("n_layers"),
        cfg_int("n_heads"),
        n_kv_heads,
        head_dim,
        cfg_int("vocab"),
        0.000001,
        MAX_POS,
        MAX_ROWS,
    )


def serial_tokens(
    mut model: QwenForward, ids: List[Int], steps: Int, dest: IntPtr
) raises AlofaError -> Int:
    """Greedy continuation of one prompt, the way a single request is served."""
    model.reset()
    var token = model.argmax(model.prefill(ids))
    var n = 0
    while n < steps:
        dest[unsafe_offset=n] = token
        n += 1
        if n < steps:
            token = model.argmax(model.step(token))
    return n


def batched_tokens(
    mut model: QwenForward,
    mut ex: BatchExecutor,
    prompts: List[List[Int]],
    order: List[Int],
    steps: Int,
    dest: IntPtr,
) raises -> Int:
    """Greedy continuation of `order`'s prompts, all of them in one batch.

    Request `i` carries prompt `order[i]`. More iterations than `steps` are run
    because a request whose prompt was split across two steps spends an extra
    iteration on its prompt and produces nothing that iteration; each request is
    still expected to reach `steps` tokens, and the caller asserts that.
    """
    var arena = Arena(1 << 16)
    var tokens = int_map(arena.alloc(MAX_ROWS * 8 + 64))
    for i in range(len(order)):
        var ids = prompts[order[i]].copy()
        for j in range(len(ids)):
            tokens[unsafe_offset=j] = ids[j]
        # A generous budget, not because that many tokens are wanted but because
        # a request whose budget runs out leaves the batch, and a request that
        # has left cannot be asked what it generated.
        ex.add(BASE_REQUEST + i, tokens, len(ids), steps + 8)
        # Blocks handed over in *descending* order: they are deliberately not
        # where one contiguous region per request would have put them. A forward
        # that reached its KV by arithmetic — slot = position // block_size —
        # would compute a different number; only a forward that reads the table
        # can agree with the serial run. That is what makes this comparison a
        # claim about the page table, and not merely about the batch.
        var want = (len(ids) + steps + 8 + ex.block_size - 1) // ex.block_size
        if want > BLOCKS_PER_REQUEST:
            raise AlofaError(1, "the run needs more blocks than a request reserves")
        var table = KvPageTable(ex.block_size)
        for k in range(want):
            _ = table.push(i * BLOCKS_PER_REQUEST + (BLOCKS_PER_REQUEST - 1 - k), 0, ex.block_size)
        ex.set_page_table(BASE_REQUEST + i, table)
    var done = ex.generate(model, steps + 4)
    assert_true(done > 0, "the busy loop did not run")
    var least = steps
    for i in range(len(order)):
        # `unsafe_offset` counts elements, and `dest` is `Pointer[Int]`.
        var got = ex.generated(BASE_REQUEST + i, dest.unsafe_offset(i * MAX_GEN))
        if got < least:
            least = got
    # The pool is checked here rather than in a test of its own because this is
    # the loop the claim is about: a leak that only appears after a real
    # forward, on a batch that does not fit in one step, is the leak that
    # matters.
    assert_equal(ex.pool.used, 0)
    assert_equal(ex.pool.n_live(), 0)
    assert_equal(ex.defects, 0)
    arena.keep_alive()
    return least


def first_difference(a: IntPtr, b: IntPtr, n: Int) -> Int:
    """Index of the first token that differs, or -1 when they agree."""
    for i in range(n):
        if a[unsafe_offset=i] != b[unsafe_offset=i]:
            return i
    return -1


def baselines(
    mut model: QwenForward, prompts: List[List[Int]], dest: IntPtr
) raises -> Int:
    """One serial continuation per prompt, in prompt order."""
    var n_prompts = len(prompts)
    for p in range(n_prompts):
        var got = serial_tokens(
            model, prompts[p], STEPS, dest.unsafe_offset(p * MAX_GEN)
        )
        assert_equal(got, STEPS, "the serial run produced fewer tokens")
    return n_prompts


def test_one_request_at_a_time_agrees_with_the_serial_run() raises:
    """Batch size 1: the packed path must be the serial path, token for token."""
    var prompts = load_prompts()
    var model = build_model()
    var arena = Arena(1 << 16)
    var want = int_map(arena.alloc(MAX_GEN * 8))
    var got = int_map(arena.alloc(MAX_GEN * 8))

    for p in range(len(prompts)):
        var n = serial_tokens(model, prompts[p], STEPS, want)
        assert_equal(n, STEPS, "the serial run produced fewer tokens")

        var ex = build_executor()
        var order = List[Int]()
        order.append(p)
        var least = batched_tokens(model, ex, prompts, order, STEPS, got)
        assert_equal(least, STEPS, "the batch produced fewer tokens")
        var at = first_difference(want, got, STEPS)
        assert_equal(at, -1, "batch of one differs at token " + String(at))
    model.keep_alive()
    arena.keep_alive()


def test_two_four_and_eight_requests_agree_with_the_serial_run() raises:
    """Batch sizes 2, 4 and 8, against the same prompts decoded alone."""
    var prompts = load_prompts()
    var model = build_model()
    var arena = Arena(1 << 16)
    var want = int_map(arena.alloc(len(prompts) * MAX_GEN * 8 + 64))
    var got = int_map(arena.alloc(8 * MAX_GEN * 8))
    _ = baselines(model, prompts, want)

    for size in [2, 4, 8]:
        var order = List[Int]()
        for i in range(size):
            order.append(i % len(prompts))
        var ex = build_executor()
        var least = batched_tokens(model, ex, prompts, order, STEPS, got)
        assert_equal(least, STEPS, "batch of " + String(size) + " fell short")
        for i in range(size):
            var at = first_difference(
                want.unsafe_offset(order[i] * MAX_GEN),
                got.unsafe_offset(i * MAX_GEN),
                STEPS,
            )
            assert_equal(
                at,
                -1,
                "batch of "
                + String(size)
                + " differs for request "
                + String(i)
                + " at token "
                + String(at),
            )
    model.keep_alive()
    arena.keep_alive()


def test_two_different_prompts_do_not_agree() raises:
    """Control: the comparison above is not one that agrees regardless.

    If every prompt decoded to the same continuation, "the batch agrees with the
    serial run" would be true of an implementation that never read its input.
    """
    var prompts = load_prompts()
    assert_true(len(prompts) >= 2, "the fixture has fewer than two prompts")
    var model = build_model()
    var arena = Arena(1 << 16)
    var want = int_map(arena.alloc(len(prompts) * MAX_GEN * 8 + 64))
    _ = baselines(model, prompts, want)

    var differing = 0
    for p in range(1, len(prompts)):
        if first_difference(want, want.unsafe_offset(p * MAX_GEN), STEPS) >= 0:
            differing += 1
    assert_true(
        differing > 0,
        "every prompt decoded to the same continuation; the batch comparison "
        + "could not fail",
    )
    model.keep_alive()
    arena.keep_alive()


def test_swapped_prompts_are_not_accepted_as_the_originals() raises:
    """Control: a request must follow the prompt it was actually given.

    Two requests, prompts exchanged. The batch has to produce the continuation
    of the prompt each one carries — so this asserts both that the swapped
    result is *not* the baseline for the slot and that it *is* the baseline for
    the prompt. A harness that always compared request `i` against prompt `i`
    would pass the first and fail the second; one that read another request's
    rows would fail both.
    """
    var prompts = load_prompts()
    assert_true(len(prompts) >= 2, "the fixture has fewer than two prompts")
    var model = build_model()
    var arena = Arena(1 << 16)
    var want = int_map(arena.alloc(len(prompts) * MAX_GEN * 8 + 64))
    var got = int_map(arena.alloc(2 * MAX_GEN * 8))
    _ = baselines(model, prompts, want)

    var order = List[Int]()
    order.append(1)
    order.append(0)
    var ex = build_executor()
    var least = batched_tokens(model, ex, prompts, order, STEPS, got)
    assert_equal(least, STEPS, "the swapped batch produced fewer tokens")

    for i in range(2):
        assert_true(
            first_difference(
                want.unsafe_offset(i * MAX_GEN),
                got.unsafe_offset(i * MAX_GEN),
                STEPS,
            ) >= 0,
            "request " + String(i) + " decoded the prompt it was not given",
        )
        assert_equal(
            first_difference(
                want.unsafe_offset(order[i] * MAX_GEN),
                got.unsafe_offset(i * MAX_GEN),
                STEPS,
            ),
            -1,
            "request " + String(i) + " did not decode the prompt it was given",
        )
    model.keep_alive()
    arena.keep_alive()


def blocks_of(i: Int, want: Int, block_size: Int) raises -> KvPageTable:
    """`want` blocks for request `i`, handed over in descending order.

    Descending is deliberate and it is the same decision `batched_tokens` makes:
    a forward that reached its KV by arithmetic would compute something else, so
    agreeing with the serial run is a statement about the table.
    """
    var table = KvPageTable(block_size)
    for k in range(want):
        _ = table.push(
            i * BLOCKS_PER_REQUEST + (BLOCKS_PER_REQUEST - 1 - k), 0, block_size
        )
    return table^


def test_a_shared_prefix_is_computed_once() raises:
    """The same prompt twice: the second time only the missing tail runs.

    The first request fills the blocks and keeps them. The second one is handed
    the same blocks and told how much of the prompt they already hold, so it
    queues one row instead of the whole prompt — and what it generates still has
    to be the same tokens, because everything it reads at positions it never
    computed came out of a forward that was run on this model with this prompt.

    A prefix cache that saved the blocks but re-ran the arithmetic would pass
    every other test in this file. This is the one that would fail, and the
    reason it is here rather than in the unit gate is that "the same tokens"
    only means something on real weights.
    """
    var prompts = load_prompts()
    var model = build_model()
    var arena = Arena(1 << 16)
    var want = int_map(arena.alloc(MAX_GEN * 8))
    var got_a = int_map(arena.alloc(MAX_GEN * 8))
    var got_b = int_map(arena.alloc(MAX_GEN * 8))
    var n_prompt = len(prompts[0])

    var n_want = serial_tokens(model, prompts[0], STEPS, want)
    assert_equal(n_want, STEPS, "the serial run produced fewer tokens")

    var ex = build_executor()
    var tokens = int_map(arena.alloc(MAX_ROWS * 8 + 64))
    for j in range(n_prompt):
        tokens[unsafe_offset=j] = prompts[0][j]
    var n_blocks = (n_prompt + STEPS + 8 + ex.block_size - 1) // ex.block_size
    var table = blocks_of(0, n_blocks, ex.block_size)

    ex.add(BASE_REQUEST, tokens, n_prompt, STEPS + 8)
    ex.set_page_table(BASE_REQUEST, table)
    _ = ex.generate(model, STEPS + 4)
    var n_a = ex.generated(BASE_REQUEST, got_a)
    # The budget here is generous on purpose: what is compared is the first
    # `STEPS` tokens, and a request that ran out of budget would leave the batch
    # before the comparison could be made.
    assert_true(n_a >= STEPS, "the first request fell short")
    assert_equal(
        first_difference(want, got_a, STEPS),
        -1,
        "the first request differs from the serial run",
    )

    # Dropped, not forgotten: dropping hands the slot back and forgets the
    # address, but the bytes stay where they are, which is exactly what a prefix
    # cache is — blocks whose owner is gone and whose contents are still true.
    ex.drop(BASE_REQUEST)
    ex.add(
        BASE_REQUEST + 1, tokens, n_prompt, STEPS + 8, n_prompt, n_prompt - 1
    )
    ex.set_page_table(BASE_REQUEST + 1, table)
    _ = ex.generate(model, STEPS + 4)
    var n_b = ex.generated(BASE_REQUEST + 1, got_b)
    assert_true(n_b >= STEPS, "the shared-prefix request fell short")
    assert_equal(
        first_difference(want, got_b, STEPS),
        -1,
        "a shared prefix produced different tokens",
    )
    assert_equal(ex.pool.used, 0, "the shared-prefix run leaked")
    model.keep_alive()
    arena.keep_alive()


def test_a_prefix_that_was_never_computed_is_not_one() raises:
    """Control for the case above: claim the prefix, skip the arithmetic, but
    put nothing behind it.

    One row again, same prompt — only the blocks are fresh, and nobody wrote the
    first `n - 1` positions into them. The claim "these bytes are a prefix" is
    then a lie, and the generated tokens have to come out different.

    Without this, the shared-prefix case could pass for the wrong reason: a
    forward that never really read the cached positions would agree with the
    serial run there too, and the suite would have proved nothing.
    """
    var prompts = load_prompts()
    var model = build_model()
    var arena = Arena(1 << 16)
    var want = int_map(arena.alloc(MAX_GEN * 8))
    var got = int_map(arena.alloc(MAX_GEN * 8))
    var n_prompt = len(prompts[0])
    _ = serial_tokens(model, prompts[0], STEPS, want)

    var ex = build_executor()
    var tokens = int_map(arena.alloc(MAX_ROWS * 8 + 64))
    for j in range(n_prompt):
        tokens[unsafe_offset=j] = prompts[0][j]
    var n_blocks = (n_prompt + STEPS + 8 + ex.block_size - 1) // ex.block_size
    ex.add(BASE_REQUEST, tokens, n_prompt, STEPS + 8, n_prompt, n_prompt - 1)
    # Request 1's own blocks: reserved, never filled.
    ex.set_page_table(BASE_REQUEST, blocks_of(1, n_blocks, ex.block_size))
    _ = ex.generate(model, STEPS + 4)
    var n_got = ex.generated(BASE_REQUEST, got)
    assert_true(n_got >= STEPS, "the control request fell short")
    assert_true(
        first_difference(want, got, STEPS) >= 0,
        "a prefix nobody computed produced the same tokens",
    )
    model.keep_alive()
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
