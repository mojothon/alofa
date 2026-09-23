"""The engine loop: the scheduler decides, the executor runs, and this file is
the only place that knows both.

The two halves were built apart on purpose — `scheduler.mojo` is a pure function
of integers, `executor.mojo` turns resident requests into one forward — and
joining them is where the interesting mistakes live. It is not the arithmetic
that goes wrong here; it is the *translation*. A prefill arrives as a slice
`[start, end)` of a prompt this file still owns; a decode arrives as a bare
request id whose next token is whatever the last step chose; a preemption
arrives as a request that must lose its history **and** the tokens it already
generated. Get any of those three wrong and every number stays plausible: the
model will happily continue a prompt that was spliced, or decode a request that
is no longer resident.

**Order inside a tick is part of the contract.**

1. Preemption is applied *before* the batch is assembled: a preempted request's
   rows must not be part of a step that has already decided to evict it.
2. Completion is applied *after* the step: the scheduler releases a request on
   the same tick it asks for the request's last token, so dropping it first
   would throw away the token the request was owed.
3. A request that is preempted and decoded in the same action loses the decode.
   The scheduler preempts newest-first and the decode loop has already passed
   some slots by then; the alternative is decoding a sequence whose KV was just
   declared dead.

**Where the blocks are.** The scheduler counts them and does not own them; this
file owns them, through `KvRoom`. Three facts live in that translation and
nowhere else:

* A prefill is a slice, so one admission arrives as several `append_tokens`;
* A finished request's sequence is *published* to the prefix cache, so its
  blocks change owner instead of coming back — that is what `cached_blocks`
  counts and what `SchedInput.freed_blocks` eventually returns;
* Eviction is the engine's call, because pressure is the engine's business.

The boundary is real and stated rather than papered over: `runtime/kv` caps
concurrent KV requests at `MAX_REQUESTS` (8) and a sequence at `MAX_SEQ_TOKENS`
(64). The ninth concurrent request is refused with `capacity` at its first
prefill, and a longer sequence with `out_of_range`. A request that quietly ran
without the blocks it is charged for would surface as a plausible continuation
of the wrong prompt.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_engine_core.mojo
"""

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    AlofaError,
)

from alofa.core.memory import Arena
from alofa.engine.executor import (
    F32Ptr,
    KvPageTable,
    MAX_BATCH,
    MAX_GEN,
    MAX_ROWS,
    BatchExecutor,
    IntPtr,
    int_map,
)
from alofa.engine.kv_room import KvRoom
from alofa.engine.scheduler import Action, SchedConfig, SchedInput, Scheduler
from alofa.model.arch.qwen import BACKEND_SCALAR, QwenForward
from alofa.runtime.kv import MAX_SEQ_TOKENS

comptime ST_FREE = 0
comptime ST_QUEUED = 1
comptime ST_RESIDENT = 2
comptime ST_DONE = 3

# A prompt is kept here, in full, rather than streamed: preemption recomputes
# from the beginning, which needs the beginning.
comptime MAX_PROMPT = 128

comptime NO_TOKEN = -1


struct EngineCore:
    """Requests in, tokens out, one scheduler tick at a time.

    The shapes are the executor's and the policy is the scheduler's; this struct
    holds neither. What it holds is the part that belongs to the engine and to
    nothing else: each request's prompt, each request's output, and the one
    number per tick that says how many rows ran.
    """

    var sched: Scheduler
    var ex: BatchExecutor
    # The blocks. The scheduler counts them, the executor never sees them, and
    # this is the only handle that can actually change who owns one.
    var room: KvRoom

    # Per request, indexed by an engine slot of its own. The scheduler has slots
    # too, and they are deliberately not the same slots: the scheduler's are a
    # scheduling decision and these are a promise to a caller.
    var ids: InlineArray[Int, MAX_BATCH]
    var state: InlineArray[Int, MAX_BATCH]
    # The prompts are kept in arena memory, not in a field: `add` and `feed` are
    # handed an untracked pointer to a slice, and a pointer to a struct field
    # carries an origin that will not reach that far.
    var prompts: IntPtr
    var p_len: InlineArray[Int, MAX_BATCH]
    var p_new: InlineArray[Int, MAX_BATCH]
    # How far into the prompt the executor has actually been fed. Not the
    # prompt's length: a prompt delivered in chunks has only been partly
    # written, and a room told the whole length hands out blocks nobody has
    # filled, while the scheduler — which counts what was fed — goes on reading
    # an occupancy smaller than the pool's by a whole prompt.
    var fed: InlineArray[Int, MAX_BATCH]
    var out: InlineArray[Int, MAX_BATCH * MAX_GEN]
    var n_out: InlineArray[Int, MAX_BATCH]

    # Events for the next tick, queued because a caller that submits three
    # requests expects them to arrive together, not one per tick.
    var arrivals: InlineArray[Int, MAX_BATCH]
    var n_arrivals: Int
    var cancels: InlineArray[Int, MAX_BATCH]
    var n_cancels: Int

    var inp: SchedInput
    var act: Action
    # Completions the executor is the first to know about, held for the next
    # tick's input. The scheduler counts decode steps; the executor counts
    # tokens, and the first token of a continuation is produced by the very step
    # that finished the prompt. Somebody has to reconcile those two, and it is
    # the side holding the transcript.
    var report: InlineArray[Int, MAX_BATCH]
    var n_report: Int
    # Requests evicted by the action now being applied. A decode entry for one
    # of them is a stale instruction, not work.
    var evicted: InlineArray[Int, MAX_BATCH]
    var n_evicted: Int

    var tmp: IntPtr
    var chosen: IntPtr
    var arena: Arena

    var ticks: Int
    var rows_run: Int
    var last_rows: Int

    def __init__(
        out self,
        cfg: SchedConfig,
        hidden: Int,
        inter: Int,
        kv_dim: Int,
        n_layers: Int,
        n_heads: Int,
        n_kv_heads: Int,
        head_dim: Int,
        vocab: Int,
        eps: Float32,
        max_pos: Int,
        max_rows: Int = MAX_ROWS,
    ) raises AlofaError:
        # The scheduler counts tokens; the executor counts rows. One tick that
        # asked for more rows than a step can hold would not fail in the
        # scheduler's arithmetic — it would fail in `plan`, three layers away
        # from the decision that caused it. Refusing here names the disagreement
        # where both numbers are visible.
        if cfg.token_budget > max_rows:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "token budget exceeds one step's rows", ""
            )
        if cfg.max_chunk > max_rows:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "max chunk exceeds one step's rows", ""
            )
        self.sched = Scheduler(cfg)
        self.room = KvRoom(cfg.block_size)
        self.ex = BatchExecutor(
            hidden,
            inter,
            kv_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab,
            eps,
            max_pos,
            max_rows,
        )
        self.ids = InlineArray[Int, MAX_BATCH](fill=0)
        self.state = InlineArray[Int, MAX_BATCH](fill=ST_FREE)
        self.p_len = InlineArray[Int, MAX_BATCH](fill=0)
        self.p_new = InlineArray[Int, MAX_BATCH](fill=0)
        self.fed = InlineArray[Int, MAX_BATCH](fill=0)
        self.out = InlineArray[Int, MAX_BATCH * MAX_GEN](fill=0)
        self.n_out = InlineArray[Int, MAX_BATCH](fill=0)
        self.arrivals = InlineArray[Int, MAX_BATCH](fill=0)
        self.n_arrivals = 0
        self.cancels = InlineArray[Int, MAX_BATCH](fill=0)
        self.n_cancels = 0
        self.inp = SchedInput()
        self.act = Action()
        self.report = InlineArray[Int, MAX_BATCH](fill=0)
        self.n_report = 0
        self.evicted = InlineArray[Int, MAX_BATCH](fill=0)
        self.n_evicted = 0
        self.arena = Arena(
            MAX_BATCH * MAX_PROMPT * 8 + MAX_GEN * 8 + MAX_BATCH * 8 + 64
        )
        self.prompts = int_map(self.arena.alloc(MAX_BATCH * MAX_PROMPT * 8))
        self.tmp = int_map(self.arena.alloc(MAX_GEN * 8))
        self.chosen = int_map(self.arena.alloc(MAX_BATCH * 8))
        self.ticks = 0
        self.rows_run = 0
        self.last_rows = 0

    def find(self, req: Int) -> Int:
        var slot = -1
        for i in range(MAX_BATCH):
            if self.state[i] != ST_FREE and self.ids[i] == req:
                slot = i
        return slot

    def prompt_ptr(self, i: Int, at: Int) -> IntPtr:
        """A pointer into one request's stored prompt, for `add` or `feed`."""
        return self.prompts.unsafe_offset(i * MAX_PROMPT + at)

    def kv_tokens(self, i: Int) raises AlofaError -> InlineArray[Int, MAX_SEQ_TOKENS]:
        """One request's prompt as a fixed array, for the KV room.

        `MAX_SEQ_TOKENS` is `runtime/kv`'s cap on a sequence and `MAX_PROMPT` is
        this file's cap on a prompt; they are different numbers and the smaller
        one wins. Refusing is the point: a request that ran while being charged
        for blocks it never got would produce a perfectly plausible continuation
        of the wrong prefix.
        """
        if self.p_len[i] > MAX_SEQ_TOKENS:
            raise AlofaError(
                ERR_OUT_OF_RANGE, "prompt is longer than a KV sequence", ""
            )
        var toks = InlineArray[Int, MAX_SEQ_TOKENS](fill=0)
        for j in range(self.p_len[i]):
            toks[j] = self.prompts[unsafe_offset=i * MAX_PROMPT + j]
        return toks^

    def in_action(self, n: Int, req: Int) -> Bool:
        """Whether the action already released `req`, so it is not told twice."""
        for k in range(n):
            if self.act.finished[k] == req:
                return True
        return False

    def was_evicted(self, req: Int) -> Bool:
        for i in range(self.n_evicted):
            if self.evicted[i] == req:
                return True
        return False

    def submit(
        mut self, req: Int, tokens: IntPtr, n: Int, max_new: Int
    ) raises AlofaError:
        """Hand the engine a request. It arrives on the next tick.

        The prompt is copied, which is the price of preemption: a request that
        loses its history is re-fed from position zero, and the engine cannot
        re-feed what it did not keep.
        """
        if n <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "a request needs a prompt", "")
        if n > MAX_PROMPT:
            raise AlofaError(
                ERR_CAPACITY, "prompt is longer than the engine keeps", ""
            )
        if max_new <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "max_new must be positive", "")
        if max_new > MAX_GEN:
            raise AlofaError(
                ERR_CAPACITY, "max_new is more than the engine records", ""
            )
        if self.find(req) >= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "duplicate request id", "")
        var slot = -1
        for i in range(MAX_BATCH):
            if self.state[i] == ST_FREE:
                slot = i
                break
        if slot < 0:
            raise AlofaError(ERR_CAPACITY, "the engine is full", "")
        for i in range(n):
            self.prompts[unsafe_offset=slot * MAX_PROMPT + i] = tokens[
                unsafe_offset=i
            ]
        self.ids[slot] = req
        self.p_len[slot] = n
        self.p_new[slot] = max_new
        self.n_out[slot] = 0
        self.fed[slot] = 0
        self.state[slot] = ST_QUEUED
        self.arrivals[self.n_arrivals] = slot
        self.n_arrivals += 1

    def cancel(mut self, req: Int) raises AlofaError:
        """Drop a request on the next tick, whether it has started or not."""
        if self.find(req) < 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "no such request", "")
        if self.n_cancels >= MAX_BATCH:
            raise AlofaError(ERR_CAPACITY, "too many cancels queued", "")
        self.cancels[self.n_cancels] = req
        self.n_cancels += 1

    def publish(mut self, req: Int) raises AlofaError:
        """Hand a finished request's blocks to the cache, then let go of them.

        Publish first, release second. The scheduler has just moved this
        request's blocks into `cached_blocks`; releasing without publishing
        would put the blocks back in the pool while both books still count them,
        and the disagreement would be exactly as large as the request.
        """
        # What the scheduler will credit the cache with when it reads this
        # completion: the finished sequence's own block count. It is a nominal
        # figure — two requests sharing a prefix are one copy in the tree, and a
        # block is not always filled to the brim — so it is recorded here, before
        # the room has said anything, and the engine owes the difference.
        if self.room.has(req) < 0:
            return
        _ = self.room.publish(req)
        _ = self.room.drop(req)

    def sync_page_table(mut self, req: Int) raises AlofaError:
        """Hand the executor the blocks the room just gave this request.

        Called right after the room moved something, and nowhere else: a table
        stale by one `grow_to` is a context shorter than the history the next
        step writes, which is the kind of mismatch that shows up as a wrong
        token three steps later.

        The room owns the blocks and the executor owns the bytes; this is the
        only place the two meet.
        """
        var table = KvPageTable(self.room.block_size)
        self.room.page_table(req, table)
        self.ex.set_page_table(req, table)

    def prepare(mut self) raises AlofaError -> Int:
        """Ask the scheduler, then assemble what it asked for.

        Returns the rows this step will run. Everything up to here is integers;
        nothing has touched the model.
        """
        self.inp.clear()
        # Blocks the engine handed back from the cache since the last tick. This
        # is the only door: the scheduler cannot free cache blocks and must not
        # pretend to.
        #
        # Only as many as the scheduler's own cache book can absorb. A node is
        # freed whole, so the room can hand back more than was asked for — more,
        # sometimes, than the scheduler ever counted as cached. What it cannot
        # be told now waits for a tick when its book is big enough to hear it.
        self.inp.add_freed_blocks(self.room.take_freed_upto(self.sched.cached_blocks))
        for i in range(self.n_report):
            self.inp.add_finished(self.report[i])
        self.n_report = 0
        for i in range(self.n_arrivals):
            var slot = self.arrivals[i]
            self.inp.add_arrival(self.ids[slot], self.p_len[slot], self.p_new[slot])
        for i in range(self.n_cancels):
            self.inp.add_cancel(self.cancels[i])

        # The cache stands between live requests and the pool, and there is no
        # scheduler decision that can move it: a cached block has no request to
        # preempt. Two rules, both applied before the scheduler decides anything,
        # because "the pool was full" is only a fair complaint if the engine had
        # already given back what it was holding for nobody.
        #
        # 1. Yield: the cache keeps what the rest of the budget is not using —
        #    not what is resident now, but what is resident *and* what the
        #    queued requests are still owed, with the cache's own occupancy
        #    counted in. That last term is the one that matters: a cache that
        #    measured itself only against what is resident will happily fill the
        #    budget, and the requests are then squeezed into what is left,
        #    preempting each other for a block that the cache is sitting on.
        # Both rules draw on one allowance, counted out once: reclaiming blocks
        # and correcting the cache's book are two withdrawals from the same
        # account, and the scheduler is handed their sum as a single number. An
        # allowance recomputed between the rules would let the two of them
        # together hand back more than the account holds.
        var budget = self.reclaimable()
        var held = self.sched.blocks_used + self.sched.blocks_wanted()
        var spare = self.sched.cfg.capacity_blocks - held
        var want = budget - spare
        # Never more than the cache can account for: `spare` goes negative once
        # the queued requests are owed more than the budget holds, and asking
        # the room for the difference would hand back blocks the scheduler has
        # not counted yet.
        if want > budget:
            want = budget
        if want > 0:
            budget -= self.room.reclaim(want)
        # 2. Watermark: above it, the scheduler would preempt to get under, and
        #    preemption cannot touch cache. Reclaim first so the preemption it
        #    performs is about requests, not about the engine's hoarding.
        var limit = self.sched.cfg.threshold_blocks()
        if self.sched.blocks_used > limit:
            if budget > 0:
                var want2 = self.sched.blocks_used - limit
                _ = self.room.reclaim(want2 if want2 < budget else budget)

        try:
            self.act = self.sched.step(self.inp)
        except err:
            # An admission can fail after submit() queued the engine slot. Keep
            # arrivals the scheduler accepted, but retire rejected slots so
            # has_work() cannot retry the same impossible request forever.
            for i in range(self.n_arrivals):
                var slot = self.arrivals[i]
                if self.sched.find(self.ids[slot]) < 0:
                    self.ids[slot] = 0
                    self.p_len[slot] = 0
                    self.p_new[slot] = 0
                    self.fed[slot] = 0
                    self.n_out[slot] = 0
                    self.state[slot] = ST_FREE
            self.n_arrivals = 0
            raise err.copy()
        self.n_evicted = 0

        # The scheduler drops a cancelled request, so the engine must drop it too:
        # otherwise it goes on holding a prompt for a request that no tick will
        # ever schedule again, and `has_work` never becomes false.
        for i in range(self.n_cancels):
            var req = self.cancels[i]
            var slot = self.find(req)
            if self.ex.slot_of(req) >= 0:
                self.ex.drop(req)
            # Nothing was published, so these blocks really do come back — and
            # the scheduler has already subtracted them itself, which is why they
            # are not reported through `freed_blocks`.
            if self.room.has(req) >= 0:
                _ = self.room.drop(req)
            if slot >= 0:
                self.state[slot] = ST_DONE
        self.n_cancels = 0
        self.n_arrivals = 0

        # 1. Eviction first: those rows are not allowed into this step.
        for k in range(self.act.n_preempted):
            var req = self.act.preempted[k]
            self.evicted[self.n_evicted] = req
            self.n_evicted += 1
            var i = self.find(req)
            if self.ex.slot_of(req) >= 0:
                self.ex.drop(req)
            # A preempted request was never published, so its blocks come back
            # and the scheduler has already accounted for that.
            if self.room.has(req) >= 0:
                _ = self.room.drop(req)
            if i >= 0:
                # What it generated goes with its history: the recomputation
                # will produce those tokens again, from the prompt. Nothing has
                # been fed either, so the room's target starts over at zero.
                self.n_out[i] = 0
                self.fed[i] = 0
                self.state[i] = ST_QUEUED

        # 2. Prefill: a slice of a prompt this file owns.
        for k in range(self.act.n_prefill):
            var req = self.act.p_req[k]
            var start = self.act.p_start[k]
            var end = self.act.p_end[k]
            var i = self.find(req)
            if i < 0:
                raise AlofaError(
                    ERR_INVALID_ARGUMENT,
                    "the scheduler scheduled a request the engine does not hold",
                    "",
                )
            if start < 0 or end > self.p_len[i] or end <= start:
                raise AlofaError(
                    ERR_OUT_OF_RANGE, "prefill slice is outside the prompt", ""
                )
            var n = end - start
            var at = self.prompt_ptr(i, start)
            if self.ex.slot_of(req) >= 0:
                for j in range(n):
                    self.ex.feed(req, at[unsafe_offset=j])
                # Another slice of a prompt already admitted: grow, do not admit.
                _ = self.room.grow_to(req, end)
                self.sync_page_table(req)
            else:
                if start != 0:
                    raise AlofaError(
                        ERR_OUT_OF_RANGE,
                        "a request cannot enter the KV room mid-prompt",
                        "",
                    )
                # The room decides *first*: it is the one that knows how much of
                # this prompt it has seen before, and that number has to exist
                # before the executor is told anything — the executor starts the
                # request's history there, and queues only what is left.
                #
                # The room is told the whole prompt's length and handed the first
                # slice: those are different numbers whenever the scheduler
                # slices, and the room needs both.
                _ = self.room.admit(req, self.kv_tokens(i), self.p_len[i], end)
                # The whole prompt's length, not the slice's: an executor told
                # only about this slice cannot tell a slice boundary from the
                # end of the prompt, and would choose a token in the middle.
                self.ex.add(
                    req,
                    at,
                    n,
                    self.p_new[i],
                    self.p_len[i],
                    self.room.last_matched,
                )
            # What the executor has now, which is what the room must be holding:
            # the slice ends at `end`, however long the prompt is going to be.
            self.fed[i] = end
            self.sync_page_table(req)
            self.state[i] = ST_RESIDENT

        # 3. Decode: nothing to queue — the last step fed the token back. What is
        #    checked is that the request is actually there, because a decode for
        #    an absent request is a scheduler bug the model would happily serve.
        for k in range(self.act.n_decode):
            var req = self.act.decode[k]
            if self.was_evicted(req):
                continue
            if self.find(req) < 0:
                raise AlofaError(
                    ERR_INVALID_ARGUMENT,
                    "the scheduler scheduled a request the engine does not hold",
                    "",
                )
            if self.ex.slot_of(req) < 0:
                raise AlofaError(
                    ERR_INVALID_ARGUMENT,
                    "the scheduler asked to decode a request that is not resident",
                    "",
                )

        self.ticks += 1
        self.last_rows = self.ex.plan()
        return self.last_rows

    def decide[backend: Int = BACKEND_SCALAR](
        mut self, mut model: QwenForward
    ) raises AlofaError -> Int:
        """Plan one step and run the forward — **without choosing anything**.

        Returns the rows it ran; 0 means nothing was queued, so no slot is owed
        an answer. After it returns, `step_owes(i)` says whether slot `i` is owed
        a token, `step_request(i)` says whose slot it is, `step_logits(i)` is
        where that request's logits are, and the caller writes one token per slot
        into `step_choices()` (default `NO_TOKEN`) before calling `settle`.

        Why this exists: greedy is one line (`argmax`) and it was fine for that
        line to live inside `tick`. Sampling is not — it needs a random source
        **per request** and that request's own history, and neither is the
        engine's to hold: a shared source would make what a request draws depend
        on how many steps *other* requests asked for, so the same prompt would
        answer differently in a batch than alone. Splitting the step is what lets
        the caller own the decision without owning the forward.
        """
        var rows = self.prepare()
        if rows > 0:
            self.ex.forward[backend](model)
        return rows

    def step_owes(self, i: Int) -> Bool:
        """Slot `i` is owed a token this step (it is live and was served)."""
        return self.ex.live[i] == 1 and self.ex.served[i] != 0

    def step_request(self, i: Int) -> Int:
        """Whose slot `i` is. The caller keeps its own per-request state and
        needs this to find it."""
        return self.ex.req[i]

    def step_logits(self, i: Int) raises AlofaError -> F32Ptr:
        """Where slot `i`'s logits are. Only meaningful when `step_owes(i)`."""
        return self.ex.logits_of(self.ex.req[i])

    def step_choices(self) -> IntPtr:
        """The buffer `settle` reads: one token per executor slot, `NO_TOKEN`
        for the slots that owe no answer."""
        return self.chosen

    def settle(mut self, chosen: IntPtr) raises AlofaError:
        """End the step: history grows, tokens are recorded, finished released.

        `chosen` is one token per executor slot — `NO_TOKEN` for the slots that
        owe no answer, which is every slot still inside its prompt.
        """
        self.ex.advance(chosen)
        self.ex.finish()
        # Harvested before any release: the tokens live in a slot that is about
        # to be given to somebody else, and a caller that reads them afterwards
        # reads them from here.
        for i in range(MAX_BATCH):
            if self.state[i] != ST_RESIDENT:
                continue
            var req = self.ids[i]
            if self.ex.slot_of(req) < 0:
                continue
            var n = self.ex.generated(req, self.tmp)
            for j in range(n):
                self.out[i * MAX_GEN + j] = self.tmp[unsafe_offset=j]
            self.n_out[i] = n
            # The KV sequence ends where the history ends: the prompt as far as
            # it has been fed, plus what this step generated. A step that
            # produced two tokens and one that produced none still land on the
            # same number, which is why this is a target rather than a delta —
            # but the target is the history's length, not the prompt's.
            _ = self.room.grow_to(req, self.fed[i] + n)
            self.sync_page_table(req)
            # The transcript is complete as soon as it is long enough, and the
            # executor stops generating at that point without evicting itself.
            # Saying so is left to the engine, and it is said on the *next* tick,
            # before that tick decides anything — otherwise the scheduler would
            # spend a decode on a request whose answer is already written down.
            if n >= self.p_new[i] and not self.in_action(self.act.n_finished, req):
                self.ex.drop(req)
                self.state[i] = ST_DONE
                self.publish(req)
                if self.n_report < MAX_BATCH:
                    self.report[self.n_report] = req
                    self.n_report += 1
        for k in range(self.act.n_finished):
            var req = self.act.finished[k]
            var i = self.find(req)
            if self.ex.slot_of(req) >= 0:
                self.ex.drop(req)
            self.publish(req)
            if i >= 0:
                self.state[i] = ST_DONE
        self.rows_run += self.last_rows

    def tick[backend: Int = BACKEND_SCALAR](
        mut self, mut model: QwenForward
    ) raises AlofaError -> Int:
        """One **greedy** tick, model included. Returns the rows it ran.

        Greedy is the case where the decision is one line, so it stays here —
        but it now goes through `decide` / `settle` like every other policy, so
        there is exactly one place where a step is planned and ended.
        """
        var rows = self.decide[backend](model)
        var chosen = self.chosen
        for i in range(MAX_BATCH):
            chosen[unsafe_offset=i] = NO_TOKEN
        if rows > 0:
            for i in range(MAX_BATCH):
                if self.step_owes(i):
                    chosen[unsafe_offset=i] = model.argmax(self.step_logits(i))
        self.settle(chosen)
        return rows

    def run[backend: Int = BACKEND_SCALAR](
        mut self, mut model: QwenForward, max_ticks: Int
    ) raises AlofaError -> Int:
        """Tick until nothing is queued or resident, or until `max_ticks`."""
        var n = 0
        while n < max_ticks and self.has_work():
            self.tick[backend](model)
            n += 1
        return n

    def has_work(self) -> Bool:
        # A completion the scheduler has not heard about yet is work. Its blocks
        # are still held in the scheduler's book, and only a tick that carries
        # the message will release them: the executor is the first to know a
        # request is done, and the scheduler learns it one tick later.
        #
        # Leaving that term out lets the last request of a batch be published,
        # finished, and never released — idle by every measure the engine keeps,
        # still charged for blocks nobody can hand out. It leaks, and it leaks
        # once per batch.
        if self.n_report > 0:
            return True
        # The same for blocks the cache has already given back: they are only
        # free in the scheduler's book once a tick has carried the number
        # across, and a pool that stops one tick early leaves the scheduler
        # paying for blocks the room has already returned.
        if self.room.freed_pending > 0:
            return True
        for i in range(MAX_BATCH):
            if self.state[i] == ST_QUEUED or self.state[i] == ST_RESIDENT:
                return True
        return False

    def n_output(self, req: Int) raises AlofaError -> Int:
        var slot = self.find(req)
        if slot < 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "no such request", "")
        return self.n_out[slot]

    def output(self, req: Int, dest: IntPtr) raises AlofaError -> Int:
        """Copy a request's generated tokens out; returns how many."""
        var slot = self.find(req)
        if slot < 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "no such request", "")
        var n = self.n_out[slot]
        for i in range(n):
            dest[unsafe_offset=i] = self.out[slot * MAX_GEN + i]
        return n

    def preempt_total(self) -> Int:
        return self.sched.preempt_total

    def reclaimable(self) raises AlofaError -> Int:
        """Cached blocks the engine may hand back *this* tick.

        Not every block in the cache. A sequence published by the last `settle`
        joins the scheduler's `cached_blocks` only when `step` reads that
        completion — which happens at the end of this very tick, after the
        reclaim above has run. Handing such blocks back now means reporting,
        next tick, more freed blocks than the scheduler ever counted as cached,
        and it is right to refuse that. One tick of patience for a freshly
        published sequence is what keeps the two books closeable.
        """
        var in_room = self.room.cached_blocks()
        var known = self.sched.cached_blocks
        return in_room if in_room < known else known

    def cached_blocks(self) raises AlofaError -> Int:
        """Blocks held by the prefix cache and by no request."""
        return self.room.cached_blocks()

    def used_blocks(self) raises AlofaError -> Int:
        """Blocks actually out of `runtime/kv`'s pool."""
        return self.room.used()

    def defects(self) raises AlofaError -> Int:
        """Re-derived from the two halves, not remembered.

        The engine says which requests are resident and the executor says which
        requests it holds; those are two books and they must agree after every
        tick. A mismatch is invisible in the output — the tokens still look like
        tokens — which is exactly why it is counted here.
        """
        var bad = 0
        for i in range(MAX_BATCH):
            var resident = self.ex.slot_of(self.ids[i]) >= 0
            if self.state[i] == ST_RESIDENT and not resident:
                bad += 1
            if self.state[i] != ST_RESIDENT and resident:
                bad += 1
            if self.state[i] == ST_FREE and self.ids[i] != 0:
                bad += 1
            if self.state[i] != ST_FREE and self.p_len[i] <= 0:
                bad += 1
        return bad + self.ex.defects
