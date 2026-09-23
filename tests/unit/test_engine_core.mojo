"""The gate for the engine loop: the scheduler decides, the executor runs, and
the translation between them is what is under test.

Both halves arrived with their own gates — `test_scheduler.mojo` compares the
scheduler's integers against a Python reference, `test_batch_forward.mojo`
compares a batch against a serial run on real weights. What neither of them can
catch is the join: a prefill that arrives as a slice `[start, end)`, a decode
that arrives as a bare id whose next token is whatever the last step chose, and a
preemption that must cost a request both its history and the tokens it already
generated. Every one of those mistakes leaves the numbers plausible. A prompt
that gets spliced still decodes; a request that gets decoded while absent still
produces a row. So this suite pins the *sequencing* down, and it does it without
a model: the argmax is supplied by the test, one token per slot, because greedy
argmax is one line and the loop is everything else.

The tokens the test supplies are a function of the request and of how many
tokens that request already has. That makes the expected output order-sensitive
**and** independent of when the tokens were produced — which is what lets the
same expectation hold across a preemption, where a request is deliberately thrown
back to its prompt and made to generate everything again.

Three negative controls stand permanently:

- **the preemption test asserts `preempt_total > 0`.** A gate that claims
  preemption is safe but never preempts is a gate about the happy path wearing a
  costume.
- **`test_a_decode_for_an_absent_request_is_refused`** pulls a request out from
  under the scheduler and requires a named error. Silently serving nothing would
  be the cheaper behaviour and the one that hides bugs.
- **`bad_executor_alloc.mojo`** is a source file the zero-allocation gate must
  reject, so the gate that forbids a growable container inside the loop is not
  itself a container-counting no-op.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_engine_core.mojo
"""

from alofa.core.error import AlofaError
from alofa.core.memory import Arena
from alofa.core.text import read_text
from alofa.engine.core import ST_DONE, ST_RESIDENT, EngineCore
from alofa.engine.executor import MAX_BATCH, MAX_GEN, IntPtr, int_map
from alofa.engine.scheduler import SchedConfig
from alofa.runtime.kv import MAX_BLOCKS
from std.testing import TestSuite, assert_equal, assert_true

comptime SOURCE = "src/alofa/engine/core.mojo"
comptime ANTIGEN = "tests/fixtures/bad_executor_alloc.mojo"

# Toy shapes, so the suite runs in milliseconds and never touches the weights.
comptime HIDDEN = 8
comptime INTER = 12
comptime KV_DIM = 4
comptime N_LAYERS = 2
comptime N_HEADS = 2
comptime N_KV_HEADS = 1
comptime HEAD_DIM = 4
comptime VOCAB = 512
comptime MAX_POS = 32
comptime ROWS = 16

comptime NO_TOKEN = -1
comptime MAX_TICKS = 64

# The water-mark scenario: eight requests, a 112-block pool, and a history
# long enough that the two cannot both be satisfied at once.
comptime DEEP_TICKS = 512
comptime DEEP_POS = 64
comptime DEEP_BLOCK = 4
comptime DEEP_PROMPT = 45
comptime DEEP_GENS = 3
comptime DEEP_REQS = 4
comptime DEEP_BASE = 100
# Rows per step has to be at least a whole prompt: see the note in
# `deep_core` about what chunked prefill does to the watermark.
comptime DEEP_ROWS = 64
# The same gate's chunked twin: a step too small for a whole prompt, so the
# prompt arrives in slices. This is the case the accounting bug lived in.
comptime DEEP_CHUNK_ROWS = 16
# What the scheduler is allowed to hand out. Not the pool's 112 physical
# blocks: see the gate's docstring for why the physical pool cannot be filled
# this far by requests that are also allowed to finish.
comptime DEEP_BLOCKS_PER_REQ = (DEEP_PROMPT + DEEP_GENS + DEEP_BLOCK - 1) // DEEP_BLOCK

# A budget the queue cannot fit in, with generations long enough that the
# requests pile up instead of finishing one by one. Everything the cache is
# tempted to keep, these five need.
comptime DEEP_TIGHT_CAP = 60
# 取 60 而不是更小的数：五条请求的稳态足迹合计 55 块，准入按水位的判据
# 只在它们的足迹自己就越线时才挡。要验的是「已经跑完的请求把缓存留下、
# 和队列抢同一份预算」，不是「门口把人劝回去」。
comptime DEEP_TIGHT_REQS = 5
comptime DEEP_TIGHT_PROMPT = 37
comptime DEEP_TIGHT_GENS = 5


def toy_core(cfg: SchedConfig) raises AlofaError -> EngineCore:
    return EngineCore(
        cfg,
        HIDDEN,
        INTER,
        KV_DIM,
        N_LAYERS,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        VOCAB,
        1e-6,
        MAX_POS,
        ROWS,
    )


def roomy() raises AlofaError -> SchedConfig:
    """A policy with room for everything: no preemption, no chunking."""
    return SchedConfig(ROWS, ROWS, 16, 1024, 900, 8)


def fill(tokens_ptr: IntPtr, req: Int, n: Int) -> Int:
    """Distinct, non-zero token ids so a mis-slice is visible in a row."""
    for i in range(n):
        tokens_ptr[unsafe_offset=i] = req + i
    return n


def expected_token(req: Int, which: Int) -> Int:
    """What the test feeds back, and therefore what it expects to read."""
    return req * 4 + which


def drive(mut core: EngineCore, mut arena: Arena) raises -> Int:
    """Tick until the engine is idle, checking the two books after every one.

    Returns the number of ticks. The tokens are the test's; the sequencing is
    the engine's, and it is the only thing being measured.
    """
    var chosen = int_map(arena.alloc(MAX_BATCH * 8))
    var ticks = 0
    while core.has_work() and ticks < MAX_TICKS:
        var rows = core.prepare()
        for i in range(MAX_BATCH):
            chosen[unsafe_offset=i] = NO_TOKEN
            if core.ex.live[i] != 1:
                continue
            var req = core.ex.req[i]
            chosen[unsafe_offset=i] = expected_token(req, core.n_output(req))
        core.settle(chosen)
        ticks += 1
        # Re-derived after every tick, not once at the end: the engine says who
        # is resident and the executor says who it holds, and a step where they
        # disagree still produces tokens that look like tokens.
        assert_equal(
            core.defects(),
            0,
            "the engine and the executor disagree about who is resident",
        )
        assert_true(rows <= ROWS, "a tick asked for more rows than a step owns")
    return ticks


def expect_output(
    mut core: EngineCore, req: Int, n: Int, what: String,
) raises:
    var arena = Arena(1 << 12)
    var dest = int_map(arena.alloc(MAX_GEN * 8))
    var got = core.output(req, dest)
    assert_equal(got, n, what + ": wrong number of tokens")
    for i in range(n):
        assert_equal(dest[unsafe_offset=i], expected_token(req, i), what)
    arena.keep_alive()


def test_one_request_arrives_prefills_and_decodes() raises:
    var arena = Arena(1 << 16)
    var core = toy_core(roomy())
    var toks = int_map(arena.alloc(256 * 8))
    core.submit(11, toks, fill(toks, 11, 7), 3)

    var ticks = drive(core, arena)
    # The step that ends the prompt also chooses the first token, so a three
    # token continuation costs one prefill and two decodes. The fourth tick
    # carries the completion to the scheduler: the executor is the first to know
    # a request is done, and until a tick says so the scheduler still holds its
    # blocks. Stopping at three used to look idle and was not.
    assert_equal(ticks, 4, "a request should prefill once, then decode")
    assert_equal(core.rows_run, 9, "seven prompt rows plus two decode rows")
    expect_output(core, 11, 3, "one request")
    assert_true(not core.has_work(), "the request was not released")
    arena.keep_alive()


def test_a_finished_request_gives_its_slot_back() raises:
    """A finished request must hand its slot back: the engine serves more than
    `MAX_BATCH` requests over its life, not `MAX_BATCH` ones and then no more.

    This is the negative control for the whole queueing story. If a finished
    slot stays `ST_DONE` forever, the `(MAX_BATCH + 1)`-th request is refused
    however long it waits — and it presents as "the server is full", not as a
    bug. Nothing inside one batch of eight ever notices it, which is why it
    needs a test that outlives a batch on purpose.
    """
    var arena = Arena(1 << 16)
    var core = toy_core(roomy())
    var toks = int_map(arena.alloc(256 * 8))

    for req in range(MAX_BATCH):
        core.submit(req + 1, toks, fill(toks, req + 1, 4), 2)
    _ = drive(core, arena)
    for req in range(MAX_BATCH):
        assert_true(core.is_done(req + 1), "a driven request should be finished")
        core.release(req + 1)

    # The ninth request, same engine. Refusing it here is the bug.
    core.submit(99, toks, fill(toks, 99, 4), 2)
    _ = drive(core, arena)
    expect_output(core, 99, 2, "the request after a full batch")

    # And the other half of the contract: a request that has **not** finished
    # must not be released. Its blocks are still counted by the scheduler, so
    # releasing it would put the two books out by exactly one request.
    var fresh = toy_core(roomy())
    fresh.submit(77, toks, fill(toks, 77, 4), 2)
    var caught = "no-error"
    try:
        fresh.release(77)
    except err:
        caught = String(err)
    assert_true(
        caught.find("cancelled") >= 0, "an unfinished request must be refused: " + caught
    )
    arena.keep_alive()


def test_a_prompt_longer_than_a_chunk_is_fed_in_slices() raises:
    """Chunking changes when a request finishes, never what it produces."""
    var arena = Arena(1 << 16)
    var toks = int_map(arena.alloc(256 * 8))

    var chunked = toy_core(SchedConfig(6, 3, 16, 1024, 900, 8))
    chunked.submit(21, toks, fill(toks, 21, 10), 2)
    var slow = drive(chunked, arena)

    var whole = toy_core(roomy())
    whole.submit(21, toks, fill(toks, 21, 10), 2)
    var fast = drive(whole, arena)

    assert_true(slow > fast, "chunking did not cost any extra ticks")
    assert_equal(chunked.rows_run, 11, "ten prompt rows plus one decode row")
    expect_output(chunked, 21, 2, "chunked prefill")
    expect_output(whole, 21, 2, "one-shot prefill")
    arena.keep_alive()


def test_two_requests_share_one_step() raises:
    var arena = Arena(1 << 16)
    var core = toy_core(roomy())
    var toks = int_map(arena.alloc(256 * 8))
    core.submit(31, toks, fill(toks, 31, 4), 2)
    core.submit(51, toks, fill(toks, 51, 5), 2)

    var chosen = int_map(arena.alloc(MAX_BATCH * 8))
    var rows = core.prepare()
    assert_equal(rows, 9, "both prompts fit in one step and should share it")
    # Real tokens, not NO_TOKEN: on the step that ends a prompt, the request's
    # next row *is* the token it just chose. Feeding nothing back would leave it
    # with an empty queue and nothing to decode.
    for i in range(MAX_BATCH):
        chosen[unsafe_offset=i] = NO_TOKEN
        if core.ex.live[i] != 1:
            continue
        var req = core.ex.req[i]
        chosen[unsafe_offset=i] = expected_token(req, core.n_output(req))
    core.settle(chosen)

    _ = drive(core, arena)
    expect_output(core, 31, 2, "first of two")
    expect_output(core, 51, 2, "second of two")
    arena.keep_alive()


def test_preemption_is_exercised_and_the_request_still_finishes() raises:
    """失效的成本落在缓存上 —— 而不是落在「四条并发把池子撑爆」上。

    四条请求分两批到达。前两条先跑完，把整条序列发布进前缀缓存；缓存没有
    可以被驱逐的请求，所以它不可抢占。后两条到达时，它们自己的稳态足迹
    （合计 4 块）仍在水位之内，准入放行；但叠上缓存之后 blocks_used 越线，
    第 9 步就在这里把正在跑的请求顶出去。

    旧写法是让四条并发把足迹堆过水位来挤出一次抢占 —— 那样的状态现在进不
    来：准入会在门口挡掉它，因为那样的状态本来谁也跑不完。    这条门要验的就是「抢占发生了，而且受害者重算之后仍然跑完」。
    """
    var arena = Arena(1 << 16)
    # 水位 5 块 = 容量 8 × 625‰。前两条跑完留下 4 块缓存；后两条的稳态足迹
    # 4 块（准入放行），叠加缓存后 8 块 —— 正好压住硬容量而不越过它。
    var core = toy_core(SchedConfig(ROWS, ROWS, 16, 8, 625, 8))

    # 第一批：跑完之后，整条序列以缓存的形式留在池子里。
    for i in range(2):
        var req = 40 + i * 20
        var own = int_map(arena.alloc(256 * 8))
        core.submit(req, own, fill(own, req, 16), 3)
    _ = drive(core, arena)

    # 第二批：缓存已经占了 4 块，这两条会让 blocks_used 越过水位。
    for i in range(2, 4):
        var req = 40 + i * 20
        var own = int_map(arena.alloc(256 * 8))
        core.submit(req, own, fill(own, req, 16), 3)
    _ = drive(core, arena)

    assert_true(core.preempt_total() > 0, "nothing was ever preempted")
    for i in range(4):
        var req = 40 + i * 20
        # Recompute throws the generated tokens away with the history, so the
        # expectation is not "at least three" — it is exactly three, in order.
        expect_output(core, req, 3, "preempted request")
    arena.keep_alive()


def deep_core(cfg: SchedConfig) raises AlofaError -> EngineCore:
    """The toy engine, with the batch pool sized by the caller.

    `ROWS` is what caps a chunk: a step too small to hold a whole prompt forces
    the prompt to arrive in slices, which is the case the accounting gate is
    about, and a step that holds one is what the water-mark gate uses.
    """
    return EngineCore(
        cfg,
        HIDDEN,
        INTER,
        KV_DIM,
        N_LAYERS,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        VOCAB,
        1e-6,
        DEEP_POS,
        DEEP_ROWS,
    )


def tick_deep(mut core: EngineCore, chosen: IntPtr) raises -> Int:
    """One tick, with the pool's books checked after it. Returns blocks used."""
    _ = core.prepare()
    for i in range(MAX_BATCH):
        chosen[unsafe_offset=i] = NO_TOKEN
        if core.ex.live[i] != 1:
            continue
        var req = core.ex.req[i]
        chosen[unsafe_offset=i] = expected_token(req, core.n_output(req))
    core.settle(chosen)

    assert_equal(
        core.room.used() + core.room.n_free(),
        MAX_BLOCKS,
        "a block left the pool without being freed",
    )
    assert_equal(core.room.invariants(), 0, "the KV views stopped agreeing")
    assert_equal(core.defects(), 0, "the two books disagree")
    return core.room.used()


def drive_deep(mut core: EngineCore, mut arena: Arena) raises -> Int:
    """Tick until idle. Returns the peak number of blocks out of the pool.

    The peak is *measured*, not configured: a gate that asserted the watermark
    was set to 950‰ would be green whether or not the pool ever got near full,
    and "95%" is a claim about blocks actually out of the pool.
    """
    var chosen = int_map(arena.alloc(MAX_BATCH * 8))
    var peak = 0
    var ticks = 0
    while core.has_work() and ticks < DEEP_TICKS:
        var used = tick_deep(core, chosen)
        if used > peak:
            peak = used
        ticks += 1
    assert_true(
        ticks < DEEP_TICKS,
        "an engine never went idle: " + String(ticks) + " ticks",
    )
    return peak


def test_a_crowded_pool_still_returns_the_same_tokens() raises:
    """Four requests sharing a pool, each answered as if it were alone.

    The pool is not asserted to be at 95% here, and the reason is worth more
    than the number: at this engine, a pool that full and a batch that finishes
    cannot be had at the same time. Filling it takes requests that stay resident
    long enough to pile up, and the moment their combined blocks sit above the
    watermark the scheduler starts preempting in circles — the victim refills
    what it lost, the watermark is breached again, and a run of seven requests
    with eight tokens each to generate is still not idle after 512 ticks. Keep
    them below the watermark instead and the pool never gets near full. Both
    halves are in the ledger; neither is claimed here.

    What this does assert is the part that has to hold at any occupancy:

    - **all four finish**, with all of their tokens — a request dropped rather
      than served is the quietest failure a crowded pool has.
    - **every token matches** the same prompt run alone, with the pool to
      itself — a crowded pool is where a history restarted from the wrong place
      still produces plausible tokens.
    - **the pool adds up after every tick**, and the room's own invariants
      hold — checked inside `drive_deep`, on every tick rather than at the end.
    - **the peak is the four requests' blocks together**, so the gate is not
      quietly measuring an engine that never had more than one request resident.
    """
    var arena = Arena(1 << 20)
    var toks = int_map(arena.alloc(256 * 8))

    var baseline = InlineArray[Int, DEEP_REQS * MAX_GEN](fill=0)
    for i in range(DEEP_REQS):
        var req = DEEP_BASE + i * 30
        var alone = deep_core(
            SchedConfig(DEEP_ROWS, DEEP_ROWS, DEEP_BLOCK, MAX_BLOCKS, 1000, 8)
        )
        alone.submit(req, toks, fill(toks, req, DEEP_PROMPT), DEEP_GENS)
        _ = drive_deep(alone, arena)
        var dest = int_map(arena.alloc(MAX_GEN * 8))
        var got = alone.output(req, dest)
        assert_equal(got, DEEP_GENS, "the baseline request did not finish")
        for j in range(got):
            baseline[i * MAX_GEN + j] = dest[unsafe_offset=j]

    var full = deep_core(
        SchedConfig(DEEP_ROWS, DEEP_ROWS, DEEP_BLOCK, MAX_BLOCKS, 1000, 8)
    )
    for i in range(DEEP_REQS):
        var req = DEEP_BASE + i * 30
        full.submit(req, toks, fill(toks, req, DEEP_PROMPT), DEEP_GENS)

    var peak = drive_deep(full, arena)
    assert_true(
        peak >= DEEP_REQS * DEEP_BLOCKS_PER_REQ,
        "the four were never in the pool together: peak "
        + String(peak)
        + " blocks, wanted "
        + String(DEEP_REQS * DEEP_BLOCKS_PER_REQ),
    )
    for i in range(DEEP_REQS):
        var req = DEEP_BASE + i * 30
        var dest = int_map(arena.alloc(MAX_GEN * 8))
        var got = full.output(req, dest)
        assert_equal(got, DEEP_GENS, "a request was lost in a crowded pool")
        for j in range(got):
            assert_equal(
                dest[unsafe_offset=j],
                baseline[i * MAX_GEN + j],
                "a crowded pool changed request " + String(req),
            )
    arena.keep_alive()


def test_an_idle_engine_holds_nothing_and_the_two_books_agree() raises:
    """Idle has to mean idle: nothing resident, and one number for the pool.

    The executor is the first to know a request is finished, and it says so on
    the *next* tick. An engine that declared itself idle as soon as its own
    transcript was complete stopped one tick too early: the scheduler never
    heard about the last request of the batch, never released it, and went on
    charging it for blocks nobody could hand out. Once per batch, forever.

    Cache blocks already given back are the same story from the other side.
    They are free in the room the moment they are reclaimed and only free in the
    scheduler's book once a tick has carried the number across.

    So this asserts the thing that used to be quietly false: after the last
    tick, nothing is held, and the scheduler and the room count the same
    blocks — the pool's occupancy and the cache's alike.
    """
    var arena = Arena(1 << 20)
    var core = deep_core(
        SchedConfig(DEEP_ROWS, DEEP_ROWS, DEEP_BLOCK, MAX_BLOCKS, 1000, 8)
    )
    for i in range(DEEP_TIGHT_REQS):
        var req = DEEP_BASE + i * 30
        # Its own buffer: one shared buffer leaves every request holding the
        # last one's tokens, which is a different experiment.
        var own = int_map(arena.alloc(DEEP_TIGHT_PROMPT * 8))
        core.submit(req, own, fill(own, req, DEEP_TIGHT_PROMPT), DEEP_TIGHT_GENS)
    _ = drive_deep(core, arena)

    assert_true(not core.has_work(), "the engine never went idle")
    assert_equal(
        core.sched.blocks_held(), 0, "a finished request still holds blocks"
    )
    assert_equal(
        core.sched.blocks_used,
        core.room.used(),
        "the two books disagree about the pool",
    )
    assert_equal(
        core.sched.cached_blocks,
        core.room.cached_blocks(),
        "the two books disagree about the cache",
    )
    for i in range(DEEP_TIGHT_REQS):
        expect_output(core, DEEP_BASE + i * 30, DEEP_TIGHT_GENS, "idle engine")
    arena.keep_alive()


def test_a_published_sequence_is_counted_whole() raises:
    """A published sequence is charged for the token the prompt's last step chose.

    The executor picks the first token of a continuation on the very step that
    finishes the prompt. The scheduler counts decode steps, which that step is
    not, so its own count of a finished sequence comes out one token short.
    Harmless, until the token is the one that crosses a block boundary: then
    every published sequence is a whole block under-counted in both books, and
    the pool is believed to have room it does not have.

    The numbers here put that token exactly on a boundary — 37 prompt tokens
    plus 8 generated is 45, which is twelve blocks of four, while 44 is eleven —
    so a scheduler that counts decode steps is off by one block per request,
    four blocks over four requests. Checking this at a coarser block size
    would let the same bug through: 45 and 44 are both six blocks of eight.
    """
    var arena = Arena(1 << 20)
    var core = EngineCore(
        SchedConfig(64, 64, 4, 1024, 1000, 8), 8, 12, 4, 2, 2, 1, 4, 512, 1e-6,
        64, 64,
    )
    var chosen = int_map(arena.alloc(MAX_BATCH * 8))
    var reqs = 4
    var prompt = 37
    var gens = 8
    for i in range(reqs):
        var req = 200 + i * 30
        # Its own buffer, so nothing is shared and every sequence is counted
        # on its own.
        var own = int_map(arena.alloc(prompt * 8))
        for j in range(prompt):
            own[unsafe_offset=j] = req + j
        core.submit(req, own, prompt, gens)
    for tick in range(64):
        if not core.has_work():
            break
        _ = core.prepare()
        for i in range(MAX_BATCH):
            chosen[unsafe_offset=i] = -1
            if core.ex.live[i] != 1:
                continue
            var req = core.ex.req[i]
            chosen[unsafe_offset=i] = req * 4 + core.n_output(req)
        core.settle(chosen)

    assert_true(not core.has_work(), "the engine never went idle")
    assert_equal(
        core.room.cached_blocks(),
        reqs * 12,
        "45 tokens is twelve blocks of four",
    )
    assert_equal(
        core.sched.cached_blocks,
        core.room.cached_blocks(),
        "the two books disagree about the cache",
    )
    assert_equal(
        core.sched.blocks_used,
        core.room.used(),
        "the two books disagree about the pool",
    )
    for i in range(reqs):
        expect_output(core, 200 + i * 30, gens, "published sequence")
    arena.keep_alive()


def test_the_cache_gives_blocks_back_before_a_request_starves() raises:
    """A budget too small for the queue, and a cache that would like to keep it.

    Five requests want more blocks than the scheduler may hand out, and every
    request that finishes leaves its sequence behind in the prefix cache — so
    the cache and the queue compete for the same budget, and the cache cannot
    be preempted: a cached block has no request to evict.

    Two ways this failed, neither of them announced:

    - **the cache kept the budget.** Measured against what was merely resident,
      a cache filling the budget looked like it was yielding, while the queued
      requests squeezed into what was left and preempted each other forever
      over blocks the cache was sitting on. One request never produced a token.
    - **the engine handed back blocks the scheduler had not counted yet.** A
      sequence published by the last tick joins the scheduler's cache book only
      when that completion is read; reclaiming it before then makes the next
      tick's `freed_blocks` larger than the cache the scheduler believes in,
      and the engine fails instead of degrading.

    What is asserted is the part that cannot be faked: all five finish, with
    every token equal to the same prompt run alone.
    """
    var arena = Arena(1 << 20)
    var toks = int_map(arena.alloc(256 * 8))

    var tight = deep_core(
        SchedConfig(
            DEEP_ROWS, DEEP_ROWS, DEEP_BLOCK, DEEP_TIGHT_CAP, 950, 8
        )
    )
    for i in range(DEEP_TIGHT_REQS):
        var req = DEEP_BASE + i * 30
        var own = int_map(arena.alloc(DEEP_TIGHT_PROMPT * 8))
        tight.submit(req, own, fill(own, req, DEEP_TIGHT_PROMPT), DEEP_TIGHT_GENS)
    _ = drive_deep(tight, arena)

    for i in range(DEEP_TIGHT_REQS):
        var req = DEEP_BASE + i * 30
        var dest = int_map(arena.alloc(MAX_GEN * 8))
        var got = tight.output(req, dest)
        assert_equal(
            got, DEEP_TIGHT_GENS, "a request starved while the cache held the pool"
        )
    arena.keep_alive()


def test_the_scheduler_and_the_room_count_the_same_blocks() raises:
    """The two books, under a prompt that arrives one slice at a time.

    This is the case the bug lived in. The scheduler is told how much of the
    prompt ran this step; the room is told how much of the sequence exists. If
    the engine answered the second question with "the whole prompt, always" —
    which it did — then the room held fifteen blocks for a request the
    scheduler believed was holding four, and the watermark, read off the
    scheduler's number, never came close to firing while the pool drained.

    Asserted as a bound, not an equality, only because a partially filled block
    is legitimately counted differently at the two ends: one block's worth of
    rounding, and no more.
    """
    var arena = Arena(1 << 20)
    var toks = int_map(arena.alloc(256 * 8))
    var chosen = int_map(arena.alloc(MAX_BATCH * 8))
    var core = deep_core(
        SchedConfig(
            DEEP_CHUNK_ROWS, DEEP_CHUNK_ROWS, DEEP_BLOCK, MAX_BLOCKS, 1000, 8
        )
    )
    core.submit(DEEP_BASE, toks, fill(toks, DEEP_BASE, DEEP_PROMPT), DEEP_GENS)

    var ticks = 0
    var worst = 0
    while core.has_work() and ticks < DEEP_TICKS:
        _ = core.prepare()
        for i in range(MAX_BATCH):
            chosen[unsafe_offset=i] = NO_TOKEN
            if core.ex.live[i] != 1:
                continue
            var req = core.ex.req[i]
            chosen[unsafe_offset=i] = expected_token(req, core.n_output(req))
        core.settle(chosen)
        ticks += 1
        var gap = core.room.used() - core.sched.blocks_used
        if gap < 0:
            gap = -gap
        if gap > worst:
            worst = gap
    assert_true(
        worst <= 1,
        "the scheduler and the room disagree by "
        + String(worst)
        + " blocks: one counts a slice, the other counts the prompt",
    )
    arena.keep_alive()


def test_a_request_that_reaches_max_new_is_released_and_kept() raises:
    var arena = Arena(1 << 16)
    var core = toy_core(roomy())
    var toks = int_map(arena.alloc(256 * 8))
    core.submit(51, toks, fill(toks, 51, 5), 2)

    _ = drive(core, arena)
    assert_equal(core.state[0], ST_DONE, "a finished request was not retired")
    assert_true(core.ex.slot_of(51) < 0, "a finished request still holds a slot")
    expect_output(core, 51, 2, "after release")
    arena.keep_alive()


def test_a_cancelled_request_never_runs() raises:
    var arena = Arena(1 << 16)
    var core = toy_core(roomy())
    var toks = int_map(arena.alloc(256 * 8))
    core.submit(61, toks, fill(toks, 61, 6), 2)
    core.submit(81, toks, fill(toks, 81, 6), 2)
    core.cancel(81)

    _ = drive(core, arena)
    expect_output(core, 61, 2, "the request that was kept")
    expect_output(core, 81, 0, "the request that was cancelled")
    arena.keep_alive()


def test_the_engine_refuses_what_it_cannot_hold() raises:
    var arena = Arena(1 << 16)
    var toks = int_map(arena.alloc(256 * 8))
    var core = toy_core(roomy())

    for i in range(MAX_BATCH):
        core.submit(70 + i, toks, fill(toks, 70 + i, 4), 2)
    var name = ""
    try:
        core.submit(99, toks, 4, 2)
    except err:
        name = err.name()
    assert_equal(name, "capacity", "a ninth request was admitted")

    name = ""
    try:
        core.submit(101, toks, 200, 2)
    except err:
        name = err.name()
    assert_equal(name, "capacity", "a prompt longer than the engine keeps")

    name = ""
    try:
        core.submit(102, toks, 4, MAX_GEN + 1)
    except err:
        name = err.name()
    assert_equal(name, "capacity", "max_new beyond what the engine records")

    name = ""
    try:
        core.submit(70, toks, 4, 2)
    except err:
        name = err.name()
    assert_equal(name, "invalid_argument", "a duplicate id was admitted")

    name = ""
    try:
        var bad = toy_core(SchedConfig(ROWS + 1, ROWS, 16, 1024, 900, 8))
    except err:
        name = err.name()
    assert_equal(name, "invalid_argument", "a token budget beyond one step's rows")

    name = ""
    try:
        var bad = toy_core(SchedConfig(ROWS, ROWS + 1, 16, 1024, 900, 8))
    except err:
        name = err.name()
    assert_equal(name, "invalid_argument", "a chunk larger than one step's rows")
    arena.keep_alive()


def test_a_watermark_rejection_does_not_leave_a_pending_request() raises:
    """A rejected admission must not keep the engine permanently busy.

    The scheduler rejects a sequence whose steady-state footprint exceeds the
    watermark. The engine has already queued the submission, so it must clear
    that arrival when prepare propagates the capacity error.
    """
    var arena = Arena(1 << 16)
    var core = toy_core(SchedConfig(16, 16, 16, 4, 500, 8))
    var toks = int_map(arena.alloc(256 * 8))
    core.submit(901, toks, fill(toks, 901, 32), 4)

    var name = ""
    try:
        _ = core.prepare()
    except err:
        name = err.name()
    assert_equal(name, "capacity", "超过 KV 水位的首请求必须被拒绝")
    assert_true(not core.has_work(), "被拒绝的 arrival 仍让 engine 保持 busy")
    arena.keep_alive()


def test_a_decode_for_an_absent_request_is_refused() raises:
    """Pull the request out from under the scheduler and ask for a tick.

    The cheap behaviour is to serve nothing and carry on; the honest behaviour is
    a named error, because a scheduler and an executor that disagree about who is
    resident will disagree again, and the second time it will be at scale.
    """
    var arena = Arena(1 << 16)
    var core = toy_core(roomy())
    var toks = int_map(arena.alloc(256 * 8))
    core.submit(81, toks, fill(toks, 81, 6), 3)

    var rows = core.prepare()
    assert_equal(rows, 6, "the prompt did not prefill")
    var chosen = int_map(arena.alloc(MAX_BATCH * 8))
    for i in range(MAX_BATCH):
        chosen[unsafe_offset=i] = NO_TOKEN
    core.settle(chosen)

    core.ex.drop(81)

    var name = ""
    try:
        _ = core.prepare()
    except err:
        name = err.name()
    assert_equal(
        name, "invalid_argument", "a decode for an absent request was served"
    )
    arena.keep_alive()


def test_the_busy_loop_source_keeps_no_growable_container() raises:
    """The loop is a loop: nothing in it may grow.

    A `List` in here costs nothing on the first tick and everything on the
    ten-thousandth. It is checked as source text because that is the only place
    the claim can be checked: a growable container that is never grown looks
    exactly like a fixed one at runtime.
    """
    var text = read_text(SOURCE)
    assert_true(
        text.find("List[") < 0, "the engine loop reaches for a List"
    )
    assert_true(text.find("Dict[") < 0, "the engine loop reaches for a Dict")
    assert_true(text.find("Set[") < 0, "the engine loop reaches for a Set")
    assert_true(
        text.find("DynamicVector") < 0,
        "the engine loop reaches for a DynamicVector",
    )
    assert_true(
        text.find("InlinedFixedVector") < 0,
        "the engine loop reaches for an InlinedFixedVector",
    )


def test_the_zero_alloc_gate_can_fail() raises:
    var text = read_text(ANTIGEN)
    var hits = 0
    if text.find("List[") >= 0:
        hits += 1
    if text.find("Dict[") >= 0:
        hits += 1
    if text.find("Set[") >= 0:
        hits += 1
    if text.find("DynamicVector") >= 0:
        hits += 1
    assert_true(hits >= 3, "the antigen no longer contains what the gate forbids")


def test_a_second_request_with_the_same_prompt_runs_one_row() raises:
    """The room's match reaches the executor: a shared prompt is one row.

    The first request publishes its sequence when it finishes, which is what
    `publish` is for. The second hands over the same tokens; the room says how
    much of them it already holds, and the engine passes that number down to the
    executor instead of computing the row again.

    What is asserted is rows, because rows are the arithmetic: the executor
    starts the request's history at the match, so the first step runs the one
    row that is left and not seven.

    The control at the end is the same engine, same shape, a prompt nobody
    cached — seven rows. Without it, "one row" could just as well mean the
    engine stopped planning rows at all.
    """
    var arena = Arena(1 << 16)
    var core = toy_core(roomy())
    var toks = int_map(arena.alloc(256 * 8))
    var chosen = int_map(arena.alloc(MAX_BATCH * 8))

    # Ids that do not depend on the request: `fill` keys them off it, which is
    # exactly what a shared prefix is not.
    for i in range(7):
        toks[unsafe_offset=i] = 100 + i
    core.submit(11, toks, 7, 2)
    _ = drive(core, arena)
    expect_output(core, 11, 2, "the request that filled the cache")
    assert_true(not core.has_work(), "the first request was not released")

    core.submit(12, toks, 7, 2)
    var rows = core.prepare()
    assert_equal(rows, 1, "a cached prompt was recomputed row by row")
    assert_equal(
        core.ex.history_of(12),
        6,
        "the executor did not start its history at the match",
    )
    for i in range(MAX_BATCH):
        chosen[unsafe_offset=i] = NO_TOKEN
        if core.ex.live[i] != 1:
            continue
        var req = core.ex.req[i]
        chosen[unsafe_offset=i] = expected_token(req, core.n_output(req))
    core.settle(chosen)
    _ = drive(core, arena)
    expect_output(core, 12, 2, "the request that hit the cache")

    for i in range(7):
        toks[unsafe_offset=i] = 200 + i
    core.submit(13, toks, 7, 2)
    var cold = core.prepare()
    assert_equal(cold, 7, "a cold prompt was skipped like a cached one")
    assert_equal(core.ex.history_of(13), 0, "a cold prompt claims history")
    for i in range(MAX_BATCH):
        chosen[unsafe_offset=i] = NO_TOKEN
        if core.ex.live[i] != 1:
            continue
        var req = core.ex.req[i]
        chosen[unsafe_offset=i] = expected_token(req, core.n_output(req))
    core.settle(chosen)
    _ = drive(core, arena)
    expect_output(core, 13, 2, "the request that missed the cache")
    assert_equal(core.defects(), 0, "the two books disagree at the end")
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
