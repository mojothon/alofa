"""Who gets to run this tick — as a pure function of what happened last tick.

Why it is shaped like this
--------------------------

The failure mode of a scheduler is not a crash. It is a scheduler that works on
every workload you try by hand and then, under a preemption storm, silently
starves a request for ten seconds. Nobody finds that by reading the code; and
nobody finds it by testing either, if testing requires a 2 GB checkpoint and a
GPU.

So the scheduler here is a state machine with **no I/O, no clock, no
allocation**: its inputs are events, its output is plain data. That makes every
edge reachable from a text file, and that is exactly what the trace fixture in
`tests/fixtures/scheduler/` is for.

Policy (this is the contract — the Python reference implements the same text)
----------------------------------------------------------------------------

1. **One token budget.** Prefill and decode are not separate pools: a chunk of
   prefill and a decode step spend the same currency. This is what lets chunked
   prefill, prefix reuse and (later) speculative decoding coexist without
   special cases.
2. **Order within a tick:** cancels, then engine-reported completions, then the
   wait-tick increment, then arrivals, then promotion, then decode, then
   *starved* prefill, then ordinary prefill, then watermark preemption.
   A cancel also **blocks an arrival with the same id on the same tick**: the
   request must never enter a queue. "Which one wins" has to be written down,
   or replay is not reproducible.
3. **At most one chunk per request per tick.** Giving one long prompt the whole
   budget is better for that prompt's TTFT and worse for everyone else, so the
   per-request cap is `max_chunk` and the rest of the budget goes to the next
   request.
4. **Running requests decode before anyone prefills.** A request that is already
   generating has a KV cache that will not fit forever; admitting new work ahead
   of it is how you turn a latency problem into an OOM.
5. **The latency guard jumps the head of the *waiting* queue, not the decode
   pass.** A request whose `wait_ticks` reached `max_wait_ticks` is prefilled
   before every other waiting request — that is the defence against a long
   prompt at the head of the queue eating the whole budget every tick. It
   deliberately does *not* steal from decode: if the budget is smaller than the
   number of running requests, the engine is saturated and the guard cannot fix
   that by moving the wait onto requests that already hold KV.
6. **Preemption is recompute, and only of running requests.** No KV is swapped
   out; a preempted request loses its cache (`done` and `generated` both go back
   to 0) and re-prefills from scratch. Only *running* requests are candidates:
   preempting a half-prefilled request would let a tight watermark stop prefill
   from ever completing, which is a livelock dressed up as a policy.
   Consequence, stated plainly: while every request is still prefilling, the
   watermark may be exceeded, because there is nothing to reclaim.
7. **The preemption count is an exposed metric, not a debug field.** It is the
   earliest honest signal that the KV pool is undersized.

Accounting
----------

KV occupancy is *derived* from sequence lengths: `ceil((done + generated) /
block_size)` per request, summed incrementally into `blocks_used`. The scheduler
counts blocks, it does not own them: the physical pool lives one layer down, in
`runtime/kv`, and is driven by the engine.

Two kinds of occupancy therefore have to be told apart, because they are freed
by different owners:

* **Per-request blocks** — freed by the scheduler itself, when it releases a
  request (preempted, cancelled).
* **Cache blocks** — a finished request's sequence is *published* to the prefix
  cache, so its blocks stay occupied after the request is gone. The scheduler
  keeps counting them under `cached_blocks`; it cannot free them, because
  eviction is the engine's call.

`freed_blocks` is the channel back: blocks the engine released from the cache
since the last tick. Without it the scheduler would go on counting blocks that
are already back in the pool, and would refuse work that fits — a leak dressed
up as back-pressure. Reporting more freed than the scheduler believes are
cached is a real inconsistency, so it is a named error rather than a clamp.

The invariant is `blocks_used == blocks_held() + cached_blocks`.

Allocation
----------

Every container here is an `InlineArray` of compile-time size `MAX_BATCH`, and
`step` does nothing but integer arithmetic and indexed writes. Nothing in this
file can grow. Note that serialising a trace (`engine/trace.mojo`) *does*
allocate a `String` per line — that is why recording is a separate function, not
part of `step`: the steady state stays allocation-free, and recording is opt-in.

Run:
    pixi run mojo run -I src tests/unit/test_scheduler.mojo
"""

from alofa.core.error import ERR_CAPACITY, ERR_INVALID_ARGUMENT, AlofaError

# Concurrency ceiling. It is a compile-time constant and not a knob because the
# whole point of the zero-allocation claim is that no container can grow; a
# runtime-sized scheduler would need a heap container to hold its slots.
comptime MAX_BATCH = 32

comptime ST_FREE = 0
comptime ST_WAITING = 1
comptime ST_RUNNING = 2

comptime DEFAULT_TOKEN_BUDGET = 256
comptime DEFAULT_MAX_CHUNK = 64
comptime DEFAULT_BLOCK_SIZE = 16
comptime DEFAULT_CAPACITY_BLOCKS = 1024
comptime DEFAULT_WATERMARK_PERMILLE = 900
comptime DEFAULT_MAX_WAIT_TICKS = 8

# Digest arithmetic stays inside Int64: MOD is 2^31-1, so h * MUL < 2^51.
comptime DIGEST_MOD = 2147483647
comptime DIGEST_MUL = 1000003


def blocks_for(seq: Int, block_size: Int) raises AlofaError -> Int:
    """How many KV blocks a sequence of `seq` tokens occupies.

    Zero tokens occupy zero blocks — a request that has been preempted owns
    nothing, which is what makes recompute-preemption free its cache.
    """
    if block_size <= 0:
        raise AlofaError(ERR_INVALID_ARGUMENT, "block_size must be positive")
    if seq <= 0:
        return 0
    return (seq + block_size - 1) // block_size


def min3(a: Int, b: Int, c: Int) -> Int:
    """Smallest of three, without importing a collection to do it."""
    var m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m


struct SchedConfig:
    """Everything that changes scheduling behaviour, and nothing else.

    `watermark_permille` is thousandths rather than a float so that the same
    policy produces the same decisions in Mojo and in the Python reference —
    a float threshold would turn a byte-exact gate into a tolerance gate.
    """

    var token_budget: Int
    var max_chunk: Int
    var block_size: Int
    var capacity_blocks: Int
    var watermark_permille: Int
    var max_wait_ticks: Int

    def __init__(out self):
        self.token_budget = DEFAULT_TOKEN_BUDGET
        self.max_chunk = DEFAULT_MAX_CHUNK
        self.block_size = DEFAULT_BLOCK_SIZE
        self.capacity_blocks = DEFAULT_CAPACITY_BLOCKS
        self.watermark_permille = DEFAULT_WATERMARK_PERMILLE
        self.max_wait_ticks = DEFAULT_MAX_WAIT_TICKS

    def __init__(
        out self,
        token_budget: Int,
        max_chunk: Int,
        block_size: Int,
        capacity_blocks: Int,
        watermark_permille: Int,
        max_wait_ticks: Int,
    ) raises AlofaError:
        if block_size <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "block_size must be positive")
        if capacity_blocks <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "capacity_blocks must be positive")
        if max_chunk <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "max_chunk must be positive")
        if token_budget < 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "token_budget must not be negative")
        if watermark_permille < 0 or watermark_permille > 1000:
            raise AlofaError(ERR_INVALID_ARGUMENT, "watermark_permille out of range")
        if max_wait_ticks < 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "max_wait_ticks must not be negative")
        self.token_budget = token_budget
        self.max_chunk = max_chunk
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self.watermark_permille = watermark_permille
        self.max_wait_ticks = max_wait_ticks

    def threshold_blocks(self) -> Int:
        """Watermark in blocks: above this, running requests get preempted."""
        return self.capacity_blocks * self.watermark_permille // 1000


struct SchedInput:
    """What the engine tells the scheduler about the tick that just ended.

    Three event kinds and nothing else. `finished` is *engine-reported*
    completion (EOS, a stop string); hitting `max_new_tokens` is detected by the
    scheduler itself, because only it knows how many tokens it handed out.
    """

    var arr_id: InlineArray[Int, MAX_BATCH]
    var arr_prompt: InlineArray[Int, MAX_BATCH]
    var arr_max_new: InlineArray[Int, MAX_BATCH]
    var cancelled: InlineArray[Int, MAX_BATCH]
    var finished: InlineArray[Int, MAX_BATCH]
    var n_arrived: Int
    var n_cancelled: Int
    var n_finished: Int
    # Blocks the engine returned to the pool since the last tick, that no live
    # request ever owned: prefix-cache eviction. See the module docstring.
    var freed_blocks: Int

    def __init__(out self):
        self.arr_id = InlineArray[Int, MAX_BATCH](fill=0)
        self.arr_prompt = InlineArray[Int, MAX_BATCH](fill=0)
        self.arr_max_new = InlineArray[Int, MAX_BATCH](fill=0)
        self.cancelled = InlineArray[Int, MAX_BATCH](fill=0)
        self.finished = InlineArray[Int, MAX_BATCH](fill=0)
        self.n_arrived = 0
        self.n_cancelled = 0
        self.n_finished = 0
        self.freed_blocks = 0

    def clear(mut self):
        """Reset for the next tick without letting go of the buffers."""
        self.n_arrived = 0
        self.n_cancelled = 0
        self.n_finished = 0
        self.freed_blocks = 0

    def add_freed_blocks(mut self, n: Int) raises AlofaError:
        """Report blocks the engine released from the prefix cache this tick."""
        if n < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "freed_blocks must not be negative"
            )
        self.freed_blocks += n

    def add_arrival(mut self, req: Int, prompt_len: Int, max_new: Int) raises AlofaError:
        if self.n_arrived >= MAX_BATCH:
            raise AlofaError(ERR_CAPACITY, "too many arrivals in one tick")
        self.arr_id[self.n_arrived] = req
        self.arr_prompt[self.n_arrived] = prompt_len
        self.arr_max_new[self.n_arrived] = max_new
        self.n_arrived += 1

    def add_cancel(mut self, req: Int) raises AlofaError:
        if self.n_cancelled >= MAX_BATCH:
            raise AlofaError(ERR_CAPACITY, "too many cancels in one tick")
        self.cancelled[self.n_cancelled] = req
        self.n_cancelled += 1

    def add_finished(mut self, req: Int) raises AlofaError:
        if self.n_finished >= MAX_BATCH:
            raise AlofaError(ERR_CAPACITY, "too many completions in one tick")
        self.finished[self.n_finished] = req
        self.n_finished += 1


struct Action:
    """What to run this tick. Pure data, fixed size, therefore replayable.

    `preempt_total` is cumulative on purpose: a caller watching capacity needs a
    counter that does not reset, not a per-tick list it has to integrate itself.
    `state_digest` is the scheduler state after the decision — replaying a trace
    that matches every action but not the digests would mean two different
    states happened to emit the same plan, and that is worth failing on.
    """

    var p_req: InlineArray[Int, MAX_BATCH]
    var p_start: InlineArray[Int, MAX_BATCH]
    var p_end: InlineArray[Int, MAX_BATCH]
    var decode: InlineArray[Int, MAX_BATCH]
    var preempted: InlineArray[Int, MAX_BATCH]
    var finished: InlineArray[Int, MAX_BATCH]
    var n_prefill: Int
    var n_decode: Int
    var n_preempted: Int
    var n_finished: Int
    var tick_seq: Int
    var preempt_total: Int
    var state_digest: Int

    def __init__(out self):
        self.p_req = InlineArray[Int, MAX_BATCH](fill=0)
        self.p_start = InlineArray[Int, MAX_BATCH](fill=0)
        self.p_end = InlineArray[Int, MAX_BATCH](fill=0)
        self.decode = InlineArray[Int, MAX_BATCH](fill=0)
        self.preempted = InlineArray[Int, MAX_BATCH](fill=0)
        self.finished = InlineArray[Int, MAX_BATCH](fill=0)
        self.n_prefill = 0
        self.n_decode = 0
        self.n_preempted = 0
        self.n_finished = 0
        self.tick_seq = 0
        self.preempt_total = 0
        self.state_digest = 0

    def add_prefill(mut self, req: Int, start: Int, end: Int) raises AlofaError:
        if self.n_prefill >= MAX_BATCH:
            raise AlofaError(ERR_CAPACITY, "too many prefill slices in one tick")
        self.p_req[self.n_prefill] = req
        self.p_start[self.n_prefill] = start
        self.p_end[self.n_prefill] = end
        self.n_prefill += 1

    def add_decode(mut self, req: Int) raises AlofaError:
        if self.n_decode >= MAX_BATCH:
            raise AlofaError(ERR_CAPACITY, "too many decode entries in one tick")
        self.decode[self.n_decode] = req
        self.n_decode += 1

    def add_preempted(mut self, req: Int) raises AlofaError:
        if self.n_preempted >= MAX_BATCH:
            raise AlofaError(ERR_CAPACITY, "too many preemptions in one tick")
        self.preempted[self.n_preempted] = req
        self.n_preempted += 1

    def add_finished(mut self, req: Int) raises AlofaError:
        if self.n_finished >= MAX_BATCH:
            raise AlofaError(ERR_CAPACITY, "too many completions in one tick")
        self.finished[self.n_finished] = req
        self.n_finished += 1

    def tokens(self) -> Int:
        """Tokens this action asks the model for — the budget invariant."""
        var total = self.n_decode
        for i in range(self.n_prefill):
            total += self.p_end[i] - self.p_start[i]
        return total


struct Scheduler:
    """The state machine. `step` is the only way in or out."""

    var cfg: SchedConfig
    var ids: InlineArray[Int, MAX_BATCH]
    var prompt_len: InlineArray[Int, MAX_BATCH]
    var done: InlineArray[Int, MAX_BATCH]
    var generated: InlineArray[Int, MAX_BATCH]
    var max_new: InlineArray[Int, MAX_BATCH]
    var state: InlineArray[Int, MAX_BATCH]
    var wait_ticks: InlineArray[Int, MAX_BATCH]
    var preempt_count: InlineArray[Int, MAX_BATCH]
    # Scratch: cleared at the top of every step, never read across ticks. It is
    # a field rather than a local so that `step` needs no per-tick buffer.
    var progressed: InlineArray[Int, MAX_BATCH]
    # Blocks the scheduler believes are occupied, in total: live requests' plus
    # the prefix cache's. Freed by two different owners — see the docstring.
    var blocks_used: Int
    var cached_blocks: Int
    var preempt_total: Int
    var tick_seq: Int

    def __init__(out self):
        self.cfg = SchedConfig()
        self.ids = InlineArray[Int, MAX_BATCH](fill=0)
        self.prompt_len = InlineArray[Int, MAX_BATCH](fill=0)
        self.done = InlineArray[Int, MAX_BATCH](fill=0)
        self.generated = InlineArray[Int, MAX_BATCH](fill=0)
        self.max_new = InlineArray[Int, MAX_BATCH](fill=0)
        self.state = InlineArray[Int, MAX_BATCH](fill=0)
        self.wait_ticks = InlineArray[Int, MAX_BATCH](fill=0)
        self.preempt_count = InlineArray[Int, MAX_BATCH](fill=0)
        self.progressed = InlineArray[Int, MAX_BATCH](fill=0)
        self.blocks_used = 0
        self.cached_blocks = 0
        self.preempt_total = 0
        self.tick_seq = 0

    def __init__(out self, cfg: SchedConfig) raises AlofaError:
        self.cfg = SchedConfig(
            cfg.token_budget,
            cfg.max_chunk,
            cfg.block_size,
            cfg.capacity_blocks,
            cfg.watermark_permille,
            cfg.max_wait_ticks,
        )
        self.ids = InlineArray[Int, MAX_BATCH](fill=0)
        self.prompt_len = InlineArray[Int, MAX_BATCH](fill=0)
        self.done = InlineArray[Int, MAX_BATCH](fill=0)
        self.generated = InlineArray[Int, MAX_BATCH](fill=0)
        self.max_new = InlineArray[Int, MAX_BATCH](fill=0)
        self.state = InlineArray[Int, MAX_BATCH](fill=0)
        self.wait_ticks = InlineArray[Int, MAX_BATCH](fill=0)
        self.preempt_count = InlineArray[Int, MAX_BATCH](fill=0)
        self.progressed = InlineArray[Int, MAX_BATCH](fill=0)
        self.blocks_used = 0
        self.cached_blocks = 0
        self.preempt_total = 0
        self.tick_seq = 0

    # --- inspection (for tests and for the engine's own metrics) ---

    def find(self, req: Int) -> Int:
        """Slot of a live request, or -1."""
        for i in range(MAX_BATCH):
            if self.state[i] != ST_FREE and self.ids[i] == req:
                return i
        return -1

    def n_live(self) -> Int:
        var n = 0
        for i in range(MAX_BATCH):
            if self.state[i] != ST_FREE:
                n += 1
        return n

    def n_waiting(self) -> Int:
        var n = 0
        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING:
                n += 1
        return n

    def n_running(self) -> Int:
        var n = 0
        for i in range(MAX_BATCH):
            if self.state[i] == ST_RUNNING:
                n += 1
        return n

    def wait_ticks_of(self, req: Int) -> Int:
        var slot = self.find(req)
        if slot < 0:
            return -1
        return self.wait_ticks[slot]

    def blocks_held(self) raises AlofaError -> Int:
        """Blocks implied by live sequence lengths.

        Together with `cached_blocks` this closes the accounting:
        `blocks_used == blocks_held() + cached_blocks`.
        """
        var total = 0
        for i in range(MAX_BATCH):
            if self.state[i] != ST_FREE:
                total += blocks_for(self.done[i] + self.generated[i], self.cfg.block_size)
        return total

    def blocks_wanted(self) raises AlofaError -> Int:
        """Blocks the queued requests still owe the pool.

        `blocks_held` counts what is already written; this counts what is still
        owed by requests that have been admitted but not served yet. Yielding to
        the cache has to leave room for both: a cache allowed to fill the whole
        budget leaves the next request nothing to start on, and the engine then
        sits on work it can never begin.
        """
        var total = 0
        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING:
                var want = blocks_for(
                    self.prompt_len[i], self.cfg.block_size
                ) - blocks_for(self.done[i], self.cfg.block_size)
                if want > 0:
                    total += want
        return total

    def digest(self) -> Int:
        """Integer rolling hash of the state, so a trace can be checked for more
        than just the emitted plan."""
        var h = 0
        for i in range(MAX_BATCH):
            h = (h * DIGEST_MUL + self.state[i]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.ids[i]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.prompt_len[i]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.done[i]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.generated[i]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.max_new[i]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.wait_ticks[i]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.preempt_count[i]) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.blocks_used) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.cached_blocks) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.preempt_total) % DIGEST_MOD
        return h

    # --- state transitions ---

    def admit(mut self, req: Int, prompt_len: Int, max_new: Int) raises AlofaError:
        if prompt_len <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "prompt_len must be positive")
        if max_new <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "max_new must be positive")
        if self.find(req) >= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "duplicate request id")
        var need = blocks_for(prompt_len, self.cfg.block_size)
        if need > self.cfg.capacity_blocks:
            raise AlofaError(ERR_CAPACITY, "prompt does not fit in the kv pool")
        # Admission is judged against the watermark, not the raw capacity:
        # once the steady-state footprint of the concurrent sequences passes
        # `threshold_blocks()`, step 9 preempts on every tick and a preempted
        # request restarts from zero, so nothing ever finishes. Refusing here
        # keeps that state unreachable, including when this is the first request.
        var committed = 0
        for i in range(MAX_BATCH):
            if self.state[i] != ST_FREE:
                committed += blocks_for(
                    self.prompt_len[i] + self.max_new[i], self.cfg.block_size
                )
        var whole = blocks_for(prompt_len + max_new, self.cfg.block_size)
        if committed + whole > self.cfg.threshold_blocks():
            raise AlofaError(
                ERR_CAPACITY, "concurrent sequences exceed the kv watermark"
            )
        var slot = -1
        var i = 0
        while slot < 0 and i < MAX_BATCH:
            if self.state[i] == ST_FREE:
                slot = i
            i += 1
        if slot < 0:
            raise AlofaError(ERR_CAPACITY, "scheduler is full")
        self.ids[slot] = req
        self.prompt_len[slot] = prompt_len
        self.max_new[slot] = max_new
        self.done[slot] = 0
        self.generated[slot] = 0
        self.wait_ticks[slot] = 0
        self.preempt_count[slot] = 0
        self.state[slot] = ST_WAITING

    def release(mut self, slot: Int, to_cache: Bool) raises AlofaError:
        """Drop a request's own view of its blocks.

        `to_cache` keeps the occupancy: a finished request's sequence is
        published to the prefix cache, so its blocks do not come back to the
        pool, they change owner. Only the engine can hand them back, and it says
        so through `freed_blocks`. Cancellation and preemption are the opposite:
        nothing was published, so the blocks really are free.
        """
        var held = blocks_for(
            self.done[slot] + self.generated[slot], self.cfg.block_size
        )
        if to_cache:
            # What is published is the whole sequence: every prompt token plus
            # every token asked for. A request is finished when it has written
            # that many, so `max_new` is exact here — and `generated` is not.
            # `generated` counts decode steps, and the first token of a
            # continuation is chosen by the step that finishes the prompt. It is
            # one short, every time, and a whole block short whenever that token
            # is the one that crosses a block boundary.
            held = blocks_for(
                self.prompt_len[slot] + self.max_new[slot], self.cfg.block_size
            )
            # The live request was charged one block at a time, and the last of
            # those charges was a token short. Nothing comes back to the pool —
            # the blocks change owner — so the pool book keeps its count and
            # picks up the difference, or it would go on believing the pool has
            # room it does not have.
            self.blocks_used += held - blocks_for(
                self.done[slot] + self.generated[slot], self.cfg.block_size
            )
            self.cached_blocks += held
        else:
            self.blocks_used -= held
        self.ids[slot] = 0
        self.prompt_len[slot] = 0
        self.done[slot] = 0
        self.generated[slot] = 0
        self.max_new[slot] = 0
        self.wait_ticks[slot] = 0
        self.preempt_count[slot] = 0
        self.state[slot] = ST_FREE

    def preempt_one(mut self, mut act: Action, protect: Int) raises AlofaError -> Bool:
        """Preempt the newest running request (highest slot = latest arrival).

        Newest-first is what keeps a burst from evicting work that is nearly
        done. `protect` is the slot currently being decoded: a request cannot
        preempt itself to make room for its own next token.
        """
        var k = 0
        while k < MAX_BATCH:
            var i = MAX_BATCH - 1 - k
            if i != protect:
                if self.state[i] == ST_RUNNING:
                    if self.done[i] + self.generated[i] > 0:
                        act.add_preempted(self.ids[i])
                        self.blocks_used -= blocks_for(
                            self.done[i] + self.generated[i], self.cfg.block_size
                        )
                        self.done[i] = 0
                        self.generated[i] = 0
                        self.wait_ticks[i] = 0
                        self.preempt_count[i] += 1
                        self.preempt_total += 1
                        self.state[i] = ST_WAITING
                        return True
            k += 1
        return False

    def ensure_room(
        mut self, need: Int, protect: Int, mut act: Action
    ) raises AlofaError:
        """Make `need` blocks available by preempting, or fail loudly.

        Failing is the honest outcome when the pool cannot hold one sequence:
        silently dropping a token would corrupt the request instead.
        """
        while self.blocks_used + need > self.cfg.capacity_blocks:
            if not self.preempt_one(act, protect):
                raise AlofaError(ERR_CAPACITY, "kv pool too small for this batch")

    def try_prefill(
        mut self, slot: Int, budget: Int, mut act: Action
    ) raises AlofaError -> Int:
        """One chunk for one waiting request. Returns the remaining budget."""
        if budget <= 0:
            return budget
        if self.state[slot] != ST_WAITING:
            return budget
        if self.progressed[slot] != 0:
            return budget
        var remaining = self.prompt_len[slot] - self.done[slot]
        if remaining <= 0:
            return budget
        var chunk = min3(self.cfg.max_chunk, remaining, budget)
        if chunk <= 0:
            return budget
        var new_done = self.done[slot] + chunk
        var need = blocks_for(new_done, self.cfg.block_size) - blocks_for(
            self.done[slot], self.cfg.block_size
        )
        # Hard capacity: leave the request for a later tick rather than
        # over-committing the pool. Preemption (below) handles the watermark.
        if self.blocks_used + need > self.cfg.capacity_blocks:
            return budget
        act.add_prefill(self.ids[slot], self.done[slot], new_done)
        self.blocks_used += need
        self.done[slot] = new_done
        self.progressed[slot] = 1
        self.wait_ticks[slot] = 0
        return budget - chunk

    def step(mut self, mut inp: SchedInput) raises AlofaError -> Action:
        """The only entry point. No I/O, no clock, no allocation."""
        var act = Action()
        self.tick_seq += 1
        act.tick_seq = self.tick_seq

        for i in range(MAX_BATCH):
            self.progressed[i] = 0

        # 0. Blocks the engine handed back from the prefix cache. This has to be
        #    first: it is last tick's news, and every decision below is made
        #    against the occupancy it produces.
        if inp.freed_blocks > 0:
            if inp.freed_blocks > self.cached_blocks:
                raise AlofaError(
                    ERR_INVALID_ARGUMENT,
                    "engine freed more blocks than the scheduler holds cached",
                )
            self.cached_blocks -= inp.freed_blocks
            self.blocks_used -= inp.freed_blocks

        # 1. Cancels first: a request that arrives and is cancelled in the same
        #    tick must never have been queued. Nothing was published, so the
        #    blocks really come back.
        for i in range(inp.n_cancelled):
            var slot = self.find(inp.cancelled[i])
            if slot >= 0:
                self.release(slot, False)

        # 2. Engine-reported completion (EOS / stop string). The sequence is
        #    published to the prefix cache, so its blocks stay occupied.
        for i in range(inp.n_finished):
            var slot = self.find(inp.finished[i])
            if slot >= 0:
                act.add_finished(self.ids[slot])
                self.release(slot, True)

        # 3. One more tick waited — before arrivals, so a new request starts at 0.
        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING:
                self.wait_ticks[i] += 1

        # 4. Arrivals — minus anything cancelled on this very tick. Cancelling
        #    first is not enough on its own: the cancel list has to also *block*
        #    the arrival, or a request that arrives and is cancelled in the same
        #    tick gets queued and then served. Admission is atomic for this
        #    batch: a later rejection must not leave an earlier arrival orphaned.
        var admitted = InlineArray[Int, MAX_BATCH](fill=0)
        var n_admitted = 0
        try:
            for i in range(inp.n_arrived):
                var blocked = False
                for j in range(inp.n_cancelled):
                    if inp.cancelled[j] == inp.arr_id[i]:
                        blocked = True
                if not blocked:
                    self.admit(
                        inp.arr_id[i], inp.arr_prompt[i], inp.arr_max_new[i]
                    )
                    admitted[n_admitted] = inp.arr_id[i]
                    n_admitted += 1
        except err:
            for i in range(n_admitted):
                var slot = self.find(admitted[i])
                if slot >= 0:
                    self.release(slot, False)
            raise err.copy()

        # 5. Promotion: a fully prefilled request starts decoding next. Costs no
        #    tokens — the prefill already happened on an earlier tick.
        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING:
                if self.done[i] >= self.prompt_len[i] and self.prompt_len[i] > 0:
                    self.state[i] = ST_RUNNING
                    self.wait_ticks[i] = 0

        var budget = self.cfg.token_budget

        # 6. Decode: running requests first, one token each. Their claim on the
        #    budget is bounded by the number of running requests, which is why
        #    they go before the latency guard rather than after it.
        var di = 0
        while di < MAX_BATCH and budget > 0:
            if self.state[di] == ST_RUNNING:
                var seq = self.done[di] + self.generated[di]
                var need = blocks_for(seq + 1, self.cfg.block_size) - blocks_for(
                    seq, self.cfg.block_size
                )
                if need > 0:
                    self.ensure_room(need, di, act)
                self.blocks_used += need
                self.generated[di] += 1
                act.add_decode(self.ids[di])
                budget -= 1
                if self.generated[di] >= self.max_new[di]:
                    act.add_finished(self.ids[di])
                    self.release(di, True)
            di += 1

        # 7. Latency guard: a request that has waited `max_wait_ticks` is
        #    prefilled ahead of every other *waiting* request. It does not jump
        #    ahead of decode: what the guard defends against is a long prompt at
        #    the head of the queue eating the budget every tick, not a saturated
        #    decode phase — stealing from decode would only move the wait onto
        #    somebody who already has a KV cache at stake.
        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING:
                if self.wait_ticks[i] >= self.cfg.max_wait_ticks:
                    budget = self.try_prefill(i, budget, act)

        # 8. Ordinary prefill, FCFS by arrival.
        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING:
                budget = self.try_prefill(i, budget, act)

        # 9. Watermark preemption: recompute, newest running request first.
        #    Preemption cannot always get under the limit, because cached blocks
        #    are not preemptable — that is the engine's call, and it is the
        #    engine's job to keep the cache small enough. Breaking out is the
        #    honest outcome: a scheduler that freed blocks it does not own would
        #    be writing a number that the pool is about to contradict.
        var limit = self.cfg.threshold_blocks()
        while self.blocks_used > limit:
            if not self.preempt_one(act, -1):
                break

        act.preempt_total = self.preempt_total
        act.state_digest = self.digest()
        return act^
