"""The engine's KV room: engine requests ↔ `KvSpace`, and the blocks nobody owns.

The scheduler counts blocks, it does not own them. One layer down, `runtime/kv`
owns them, but it speaks in radix-tree operations and has never heard of "the
scheduler admitted request 7 with a 40-token prompt". This module is the only
place that translation happens, and it holds the three facts that live in
neither layer:

1. **A slice is not a prompt.** Prefill arrives as `[start, end)` slices, so one
   admission shows up as several `append_tokens` calls. How long the prompt *is*
   is a separate fact, and it is what keeps a half-delivered prompt out of the
   prefix cache.
2. **A finished request's blocks change owner.** `commit` publishes the sequence
   to the tree; from then on those blocks are referenced by the tree and by no
   request. They are *cached* blocks — occupancy without an owner.
3. **Cached blocks come back through one door.** Eviction is the engine's call,
   because the engine is the only thing that knows about pressure. The number it
   hands back is what `SchedInput.freed_blocks` carries, and it is recomputed
   from the views (cached blocks before minus cached blocks after) rather than
   remembered: a remembered count drifts from the pool the first time a block
   turns out to be shared.

Boundary, stated because it is a real limit and not a detail: `runtime/kv` caps
concurrent requests at `MAX_REQUESTS` (8) and a sequence at `MAX_SEQ_TOKENS`
(64). The room refuses the ninth request with `capacity` and a longer sequence
with `out_of_range`. A room that quietly dropped a request to stay silent would
surface three ticks later as a missing answer, which is the worst place for a
refusal to become visible.
"""

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    AlofaError,
)
from alofa.runtime.kv import (
    MAX_BLOCKS,
    MAX_NODES,
    MAX_NODE_TOKENS,
    MAX_REQUESTS,
    MAX_SEQ_TOKENS,
    OpResult,
)
from alofa.runtime.kv.paging import KvPageTable, fill_page_table
from alofa.runtime.kv.space import KvSpace


struct KvRoom:
    """The KV side of an engine tick: who owns which blocks, and what came back.

    All bookkeeping is fixed-size and indexed by the `KvSpace` slot, so the room
    cannot allocate — the busy loop calls it every tick.
    """

    var space: KvSpace
    # How many tokens a block holds. Kept here because the reader of a page
    # table needs it to interpret one: a table is a list of block ids, and an
    # id means nothing without the size of what it names.
    var block_size: Int
    # Per space slot: how long the prompt is (as opposed to how much of it has
    # been handed over), and whether the sequence has been published. Both are
    # facts about the *engine's* intent that the space itself does not carry.
    var n_prompt: InlineArray[Int, MAX_REQUESTS]
    var published: InlineArray[Int, MAX_REQUESTS]
    # Blocks handed back to the pool that no live request ever owned, waiting to
    # be reported to the scheduler. One door, one accumulator.
    var freed_pending: Int
    # What the last `admit` did, so a caller can ask "did that hit the cache?"
    # without the room growing a return struct per question.
    var last_matched: Int
    var last_fresh: Int

    def __init__(out self, block_size: Int) raises AlofaError:
        self.space = KvSpace(block_size)
        self.block_size = block_size
        self.n_prompt = InlineArray[Int, MAX_REQUESTS](fill=0)
        self.published = InlineArray[Int, MAX_REQUESTS](fill=0)
        self.freed_pending = 0
        self.last_matched = 0
        self.last_fresh = 0

    # --- the three numbers the engine and the scheduler argue about ---

    def used(self) raises AlofaError -> Int:
        return self.space.pool.used

    def n_free(self) raises AlofaError -> Int:
        return self.space.pool.n_free

    def digest(self) raises AlofaError -> Int:
        return self.space.digest()

    def invariants(self) raises AlofaError -> Int:
        """Broken conditions, recomputed from the views. Zero is the only answer."""
        return self.space.check_invariants()

    def cached_blocks(self) raises AlofaError -> Int:
        """Blocks referenced by the prefix tree and by no live request.

        Recomputed from both views every time: "the cache holds N blocks" is a
        claim about refcounts and page tables, and a remembered version of it is
        exactly the kind of number that keeps looking right after the pool stops
        agreeing with it.
        """
        var owned = InlineArray[Int, MAX_BLOCKS](fill=0)
        for s in range(MAX_REQUESTS):
            if self.space.rq_live[s] == 1:
                for i in range(self.space.rq_nblk[s]):
                    owned[self.space.block_at(s, i)] = 1
        var n = 0
        for b in range(MAX_BLOCKS):
            if self.space.pool.refcnt_of(b) > 0:
                if owned[b] == 0:
                    n += 1
        return n

    # --- admission, growth, publication, release ---

    def page_table(self, req: Int, mut table: KvPageTable) raises AlofaError:
        """Copy this request's block table out, for whoever reads the bytes.

        The room owns the blocks; the executor only reads where they are. This
        is the handover, and it is a copy: the room may move a block the moment
        it returns, and a reader holding a reference would be reading a decision
        that has already changed.

        A request the room does not hold has no table, and it says so instead of
        handing over an empty one — an empty table read as "no history" is one
        step away from a request attending over nothing and still producing a
        token.
        """
        var slot = self.has(req)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "the room holds no such request", ""
            )
        fill_page_table(self.space, slot, table)

    def admit(
        mut self,
        req: Int,
        tokens: InlineArray[Int, MAX_SEQ_TOKENS],
        prompt_len: Int,
        n_now: Int,
    ) raises AlofaError -> Int:
        """Admit a request with its first slice; returns blocks in use after.

        `prompt_len` is the whole prompt, `n_now` is what this tick hands over.
        They differ whenever the scheduler slices, and conflating them is how a
        half-delivered prompt ends up cached as if it were complete.
        """
        if prompt_len <= 0 or prompt_len > MAX_SEQ_TOKENS:
            raise AlofaError(ERR_OUT_OF_RANGE, "bad prompt length")
        if n_now <= 0 or n_now > prompt_len:
            raise AlofaError(ERR_OUT_OF_RANGE, "first slice exceeds the prompt")
        var res = self.space.new_request(req, tokens, n_now)
        var slot = self.slot_of(req)
        self.n_prompt[slot] = prompt_len
        self.published[slot] = 0
        self.last_matched = res.matched if res.matched > 0 else 0
        self.last_fresh = res.fresh.n
        return res.used

    def has(self, req: Int) raises AlofaError -> Int:
        """Whether `req` is currently in the room; `-1` if it is not."""
        return self.space.find_request(req)

    def slot_of(self, req: Int) raises AlofaError -> Int:
        """The room's slot for `req`; a named error if it has none.

        `KvSpace.slot_of` refuses an unknown id outright, which is right for the
        space and wrong here: the engine asks about requests it may already have
        let go of, and "not here" is an answer rather than a defect.
        """
        var slot = self.space.find_request(req)
        if slot < 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "no such request in the room")
        return slot

    def n_tok(self, req: Int) raises AlofaError -> Int:
        var slot = self.slot_of(req)
        if slot < 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "no such request in the room")
        return self.space.rq_ntok[slot]

    def grow_to(mut self, req: Int, target: Int) raises AlofaError -> Int:
        """Grow the sequence to `target` tokens; returns blocks in use after.

        A target, not a delta: the engine knows how long a sequence *is* (prompt
        plus generated), and a step that produced two tokens followed by one that
        produced none lands on the same number either way. A delta would have to
        be remembered, and a remembered delta is how a sequence ends up one
        token short of its own KV.
        """
        var slot = self.slot_of(req)
        if slot < 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "no such request in the room")
        if target > MAX_SEQ_TOKENS:
            raise AlofaError(
                ERR_OUT_OF_RANGE, "sequence is longer than a KV sequence"
            )
        var guard = 0
        while self.space.rq_ntok[slot] < target and guard < MAX_SEQ_TOKENS:
            _ = self.extend(req, target - self.space.rq_ntok[slot])
            guard += 1
        return self.space.pool.used

    def extend(mut self, req: Int, n: Int) raises AlofaError -> Int:
        """Grow a request by `n` tokens: one more slice, or one decode step.

        A published request must not grow — its blocks are the cache's now, and
        appending past a published tail would write a continuation into storage
        another request is already reading as a prefix.
        """
        var slot = self.slot_of(req)
        if self.published[slot] == 1:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "cannot grow a published request"
            )
        var res = self.space.append_tokens(slot, n)
        return res.used

    def prompt_done(self, req: Int) raises AlofaError -> Bool:
        var slot = self.slot_of(req)
        return self.space.rq_ntok[slot] >= self.n_prompt[slot]

    def _make_node_room(mut self, need: Int) raises AlofaError:
        """Evict cold nodes until `need` of them are free.

        块有 `reclaim` 这条路，节点却一直没有：32 个节点（`MAX_NODES`）全活着以后
        `commit` 直接抛 "radix tree is full"，八条并发长序列第一次就能把它填满
        （每条 prompt+生成 64 token ≈ 5 个节点）。更糟的是调用方的顺序 ——
        `EngineCore.publish` 先 `publish` 再 `drop`，抛在中间就让 room 里那个请求
        槽位永远占着；`MAX_REQUESTS` 也是 8，于是复用同一号的下一条请求收到
        "duplicate request id (kv room)" —— 一个缓存预算问题被伪装成"请求号冲突"。

        淘汰是块那边早就做过的同一种取舍：树丢掉一段缓存前缀，活着的请求有自己的
        页表，照常解码（`evict` 对还被引用的块只剪树、不还池）。
        """
        var guard = 0
        var before = self.cached_blocks()
        while MAX_NODES - self.space.tree.n_live < need and guard < MAX_NODES:
            var node = self.coldest_node()
            if node < 0:
                break
            _ = self.space.evict(node)
            guard += 1
        # 腾节点淘汰的块和 `reclaim` 的是同一笔账：被逐节点若还压着活请求，块不回池
        # （`cached_blocks` 本来就不数它们）；逐掉纯缓存块，块真回池。这个差值必须进
        # `freed_pending` 报给调度器 —— 少了这一步，池子把块还了，调度账却还把它们记
        # 在缓存名下：每条完成序列虚涨几块，几十条之后水位之下再无余量，准入即抢占，
        # 引擎活锁（实测 64 条后 `book=224` vs `cached=25`、`pool=31` vs 账 227，
        # `running=0`、`preempts` 每拍 +8、零进度 —— 账本二十六）。
        self.freed_pending += before - self.cached_blocks()

    def publish(mut self, req: Int) raises AlofaError -> Int:
        """Hand the sequence to the prefix tree; returns cached blocks after.

        The request keeps its blocks (a refcount, not a transfer), and the tree
        gains a second reference. From here on the occupancy belongs to the
        cache, and only `reclaim` can give it back.
        """
        var slot = self.slot_of(req)
        if self.space.rq_ntok[slot] < self.n_prompt[slot]:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "refusing to cache a prompt that was never fully seen",
            )
        # 一条序列最多占 rq_ntok/16 个新节点，再加一次拆分（+1）和余数（+1）；
        # +3 是留出来的余量，宁可多淘汰一个冷节点，也不要让 `commit` 抛。
        self._make_node_room(self.space.rq_ntok[slot] / MAX_NODE_TOKENS + 3)
        _ = self.space.commit(slot)
        self.published[slot] = 1
        return self.cached_blocks()

    def drop(mut self, req: Int) raises AlofaError -> Int:
        """Drop a request's own view; returns blocks in use after.

        Published blocks survive by refcount — that is the whole point of the
        cache. Unpublished ones come back to the pool, and the scheduler already
        accounted for those itself (preemption and cancellation are its own
        decisions), so they are deliberately *not* reported through
        `freed_blocks`: reporting them would double-count.
        """
        var slot = self.slot_of(req)
        var res = self.space.release(slot)
        self.n_prompt[slot] = 0
        self.published[slot] = 0
        return res.used

    # --- eviction ---

    def coldest_node(self) -> Int:
        """Which node to evict: lowest frequency, ties broken by smallest id.

        The policy lives here and only here — `KvSpace.evict` refuses to have
        one, so that "which node" and "how to free a node" stay separable. A tie
        break is not a detail: without one, two implementations of the same
        policy produce different bytes and the gate compares noise.
        """
        var best = -1
        var best_freq = 0
        for n in range(1, MAX_NODES):
            if self.space.tree.alive[n] == 1:
                if best < 0 or self.space.tree.freq[n] < best_freq:
                    best = n
                    best_freq = self.space.tree.freq[n]
        return best

    def reclaim(mut self, need: Int) raises AlofaError -> Int:
        """Hand back at least `need` cached blocks; returns how many came back.

        "At least", because a node is the unit of eviction and a node holds a
        whole number of blocks: asking for one block gives four back if that is
        the node's size. Asking is the engine's decision (it owns the cache
        budget); *this* is where the answer is measured.

        Only blocks no live request owns can actually come back: `release` on a
        block that a request still holds leaves it in place, so evicting a node
        that is being read right now prunes the tree without freeing anything.
        The number reported is therefore measured, not assumed — cached blocks
        before minus cached blocks after.
        """
        if need <= 0:
            return 0
        var before = self.cached_blocks()
        var guard = 0
        while before - self.cached_blocks() < need and guard < MAX_NODES:
            var node = self.coldest_node()
            if node < 0:
                break
            _ = self.space.evict(node)
            guard += 1
        var freed = before - self.cached_blocks()
        self.freed_pending += freed
        return freed

    def take_freed(mut self) -> Int:
        """Blocks handed back since the last call — one tick's `freed_blocks`."""
        var n = self.freed_pending
        self.freed_pending = 0
        return n

    def take_freed_upto(mut self, limit: Int) -> Int:
        """Hand back at most `limit` blocks, and keep the rest for a later tick.

        Reclaiming works on nodes, so a request for twelve blocks can release
        the sixteen of the node those blocks live in. The scheduler's cache book
        is written per request, not per node, and being told about more blocks
        than it ever counted as cached is not a difference of opinion it can
        absorb — it is an error. What it cannot be told this tick it is told
        later, when its own book has grown enough to hear it.
        """
        var cap = limit
        if cap < 0:
            cap = 0
        var n = self.freed_pending
        if n > cap:
            n = cap
        self.freed_pending -= n
        return n
