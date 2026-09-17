"""The unified space: one pool, two views, one consistency equation.

`KvSpace` is the object the engine will actually hold. It owns the pool and both
views, and every operation that changes one view changes the refcounts the other
one reads. The four operations that matter:

- `new_request` — match a prefix (view B), share its blocks, allocate for the
  rest (view A).
- `append_tokens` — grow a request's tail; in place when the last block is still
  private to it.
- `commit` — hand a request's tokens to the tree. The blocks are *retained*, not
  transferred: the request and the tree now point at the same blocks, and the
  block dies only when both let go.
- `release` / `evict` — drop one holder each; a block returns to the free list
  when the last holder does.

The invariant that ties it together is checked by `check_invariants` and is
independent of any fixture: it recomputes, from the views, what each refcount
*should* be, and compares. A bug that forgets to retain a shared block therefore
fails here even if the reference implementation made the same mistake.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_kv_pool.mojo
"""

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    AlofaError,
)
from alofa.runtime.kv import (
    DIGEST_MOD,
    DIGEST_MUL,
    INVALID,
    MAX_BLOCKS,
    MAX_BLOCKS_PER_SEQ,
    MAX_NODES,
    MAX_NODE_BLOCKS,
    MAX_NODE_TOKENS,
    MAX_REQUESTS,
    MAX_SEQ_TOKENS,
    ROOT,
    OpResult,
    ceil_div,
    imin,
)
from alofa.runtime.kv.pool import BlockPool
from alofa.runtime.kv.radix import Match, RadixTree


struct KvSpace:
    """Physical pool + per-request page tables + prefix tree, sharing refcounts."""

    var pool: BlockPool
    var tree: RadixTree
    # View A: request -> block sequence.
    var rq_id: InlineArray[Int, MAX_REQUESTS]
    var rq_live: InlineArray[Int, MAX_REQUESTS]
    var rq_ntok: InlineArray[Int, MAX_REQUESTS]
    var rq_off: InlineArray[Int, MAX_REQUESTS]
    var rq_nblk: InlineArray[Int, MAX_REQUESTS]
    var rq_priv: InlineArray[Int, MAX_REQUESTS]
    var rq_tok: InlineArray[Int, MAX_REQUESTS * MAX_SEQ_TOKENS]
    var rq_blk: InlineArray[Int, MAX_REQUESTS * MAX_BLOCKS_PER_SEQ]
    var rq_blen: InlineArray[Int, MAX_REQUESTS * MAX_BLOCKS_PER_SEQ]
    # Start slot *inside* the block for each entry. A page-table entry is
    # (block, start, length), not (block, length): a shared prefix can begin in
    # the middle of a block, and without the start a paged kernel would read
    # that block from slot 0 — somebody else's tokens. `rq_off` holds it for
    # the first entry only, and a request that matched a *chain* of nodes has
    # one such offset per node.
    var rq_bstart: InlineArray[Int, MAX_REQUESTS * MAX_BLOCKS_PER_SEQ]

    def __init__(out self, block_size: Int) raises AlofaError:
        self.pool = BlockPool(block_size)
        self.tree = RadixTree()
        self.rq_id = InlineArray[Int, MAX_REQUESTS](fill=0)
        self.rq_live = InlineArray[Int, MAX_REQUESTS](fill=0)
        self.rq_ntok = InlineArray[Int, MAX_REQUESTS](fill=0)
        self.rq_off = InlineArray[Int, MAX_REQUESTS](fill=0)
        self.rq_nblk = InlineArray[Int, MAX_REQUESTS](fill=0)
        self.rq_priv = InlineArray[Int, MAX_REQUESTS](fill=0)
        self.rq_tok = InlineArray[Int, MAX_REQUESTS * MAX_SEQ_TOKENS](fill=0)
        self.rq_blk = InlineArray[Int, MAX_REQUESTS * MAX_BLOCKS_PER_SEQ](fill=0)
        self.rq_blen = InlineArray[Int, MAX_REQUESTS * MAX_BLOCKS_PER_SEQ](fill=0)
        self.rq_bstart = InlineArray[Int, MAX_REQUESTS * MAX_BLOCKS_PER_SEQ](fill=0)

    # --- small accessors ---

    def block_at(self, slot: Int, i: Int) -> Int:
        return self.rq_blk[slot * MAX_BLOCKS_PER_SEQ + i]

    def blen_at(self, slot: Int, i: Int) -> Int:
        return self.rq_blen[slot * MAX_BLOCKS_PER_SEQ + i]

    def bstart_at(self, slot: Int, i: Int) -> Int:
        return self.rq_bstart[slot * MAX_BLOCKS_PER_SEQ + i]

    def token_at(self, slot: Int, i: Int) -> Int:
        return self.rq_tok[slot * MAX_SEQ_TOKENS + i]

    def find_request(self, req: Int) -> Int:
        var slot = INVALID
        var i = 0
        while slot < 0 and i < MAX_REQUESTS:
            if self.rq_live[i] == 1:
                if self.rq_id[i] == req:
                    slot = i
            i += 1
        return slot

    def free_slot(mut self) raises AlofaError -> Int:
        var slot = INVALID
        var i = 0
        while slot < 0 and i < MAX_REQUESTS:
            if self.rq_live[i] == 0:
                slot = i
            i += 1
        if slot < 0:
            raise AlofaError(ERR_CAPACITY, "no free request slot")
        return slot

    def slot_of(self, req: Int) raises AlofaError -> Int:
        var slot = self.find_request(req)
        if slot < 0:
            raise AlofaError(ERR_OUT_OF_RANGE, "unknown request id")
        return slot

    # --- view A growth ---

    def add_block(
        mut self, slot: Int, block: Int, start: Int, take: Int
    ) raises AlofaError -> Bool:
        """Append `take` tokens held in `block` from slot `start`; True if new entry.

        Consecutive tree nodes share the block their boundary lands in, so the
        same block id legitimately arrives twice in a row: that is one storage
        block holding a continuous run of tokens, not two references. This is
        why the return value matters — a refcount is one per *view entry*, and
        retaining once per contributing node would desynchronise the pool from
        the page table it is supposed to describe.

        The merge requires the run to be *slot-contiguous* (`start` picks up
        where the previous entry left off), not merely the same block. Two
        nodes can share a block without being adjacent inside it, and merging
        those would claim the request's tokens are contiguous in the block when
        they are not — exactly the claim a paged kernel reads.
        """
        var idx = self.rq_nblk[slot]
        if idx > 0:
            if self.block_at(slot, idx - 1) == block:
                if self.bstart_at(slot, idx - 1) + self.blen_at(slot, idx - 1) == start:
                    self.rq_blen[slot * MAX_BLOCKS_PER_SEQ + idx - 1] += take
                    return False
        if idx >= MAX_BLOCKS_PER_SEQ:
            raise AlofaError(ERR_CAPACITY, "request block table is full")
        self.rq_blk[slot * MAX_BLOCKS_PER_SEQ + idx] = block
        self.rq_blen[slot * MAX_BLOCKS_PER_SEQ + idx] = take
        self.rq_bstart[slot * MAX_BLOCKS_PER_SEQ + idx] = start
        self.rq_nblk[slot] = idx + 1
        return True

    # --- operations ---

    def new_request(
        mut self, req: Int, tokens: InlineArray[Int, MAX_SEQ_TOKENS], n: Int
    ) raises AlofaError -> OpResult:
        """Admit a request with `n` prompt tokens.

        The matched prefix's blocks are *retained*, not re-computed; the rest are
        freshly allocated. The request never writes into the tail of a shared
        block — that tail belongs to the cached continuation — so the cost of
        token-granularity sharing is at most one wasted partial block.
        """
        if n <= 0 or n > MAX_SEQ_TOKENS:
            raise AlofaError(ERR_INVALID_ARGUMENT, "bad token count")
        if self.find_request(req) >= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "duplicate request id")
        var slot = self.free_slot()
        var bs = self.pool.block_size
        var res = OpResult()

        for i in range(n):
            self.rq_tok[slot * MAX_SEQ_TOKENS + i] = tokens[i]

        var m = self.tree.match(tokens, n)
        res.matched = m.matched
        self.rq_off[slot] = 0

        if m.matched > 0:
            res.node = m.node
            res.offset = self.tree.offset[m.path[0]]
            self.rq_off[slot] = res.offset
            for p in range(m.n_path):
                var node = m.path[p]
                var off = self.tree.offset[node]
                # 只有路径上最后一个节点可能是"部分命中"：前面的都被完整命中，
                # 不按节点长度截断就会读到该节点块数组之外的陈旧 id。
                var stop_at = m.matched
                var node_end = m.starts[p] + self.tree.ntok[node]
                if node_end < stop_at:
                    stop_at = node_end
                var used_here = stop_at - m.starts[p]
                if used_here <= 0:
                    continue
                var need = ceil_div(off + used_here, bs)
                for j in range(need):
                    var block = self.tree.block_at(node, j)
                    var start = j * bs
                    if off > start:
                        start = off
                    var stop = (j + 1) * bs
                    if off + used_here < stop:
                        stop = off + used_here
                    var take = stop - start
                    if take <= 0:
                        continue
                    # `start` is in slot space; the block's own slot 0 is at
                    # `j * bs`. Only j == 0 can begin mid-block (that is the
                    # node's `offset`), every later block starts at 0.
                    var bstart = start - j * bs
                    if self.add_block(slot, block, bstart, take):
                        self.pool.retain(block)
                        res.shared.push_unique(block)

        var remaining = n - m.matched
        while remaining > 0:
            var block = self.pool.alloc_one()
            var take = imin(bs, remaining)
            _ = self.add_block(slot, block, 0, take)
            res.fresh.push(block)
            remaining -= take

        self.rq_ntok[slot] = n
        self.rq_id[slot] = req
        self.rq_live[slot] = 1
        # Only a block this request allocated itself may be extended in place;
        # a shared block's tail is somebody else's tokens.
        self.rq_priv[slot] = 1 if remaining_fresh(res) else 0
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res^

    def append_tokens(mut self, slot: Int, n: Int) raises AlofaError -> OpResult:
        """Add `n` tokens to a live request; returns the blocks it had to take."""
        if n <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "bad token count")
        var bs = self.pool.block_size
        var res = OpResult()
        var remaining = n

        if self.rq_priv[slot] == 1 and self.rq_nblk[slot] > 0:
            var last = self.rq_nblk[slot] - 1
            var room = bs - self.blen_at(slot, last)
            if room > 0:
                var take = imin(room, remaining)
                self.rq_blen[slot * MAX_BLOCKS_PER_SEQ + last] += take
                remaining -= take

        while remaining > 0:
            var block = self.pool.alloc_one()
            var take = imin(bs, remaining)
            _ = self.add_block(slot, block, 0, take)
            res.fresh.push(block)
            remaining -= take

        self.rq_ntok[slot] += n
        self.rq_priv[slot] = 1
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res^

    def commit(mut self, slot: Int) raises AlofaError -> OpResult:
        """Publish a request's tokens into the tree, sharing its blocks.

        Sharing, not handing over: the request keeps its references and the tree
        takes its own, so the block survives until both are gone. This is the one
        operation where "two views, one refcount" is visible in the numbers.
        """
        var bs = self.pool.block_size
        var res = OpResult()
        var n = self.rq_ntok[slot]
        var tokens = self.tokens_of(slot)
        var m = self.tree.match(tokens, n)
        res.matched = m.matched

        var parent = ROOT
        if m.matched > 0:
            var last = m.node
            var used_here = m.matched - m.starts[m.n_path - 1]
            if used_here < self.tree.ntok[last]:
                # The cached node covers more than we matched: split it so the
                # divergence has its own branch. Metadata only.
                var old_nblk = self.tree.nblk[last]
                var child = self.tree.split(last, used_here, bs)
                for i in range(self.tree.nblk[child]):
                    var block = self.tree.block_at(child, i)
                    self.pool.retain(block)
                var i = self.tree.nblk[last]
                while i < old_nblk:
                    _ = self.pool.release(self.tree.block_at(last, i))
                    i += 1
                res.nodes.push(child)
            parent = last
            res.node = last

        var k = m.matched
        var remaining = n - k
        # Locate the block holding token k. Shared blocks cover exactly the
        # matched tokens, so this lands on a boundary — but nothing below
        # assumes that.
        var cursor = 0
        var acc = 0
        while cursor < self.rq_nblk[slot]:
            if acc + self.blen_at(slot, cursor) > k:
                break
            acc += self.blen_at(slot, cursor)
            cursor += 1
        var inblk = k - acc
        # `k - acc` is the offset inside the run; the slot is the run's own
        # start plus that. The start is read from the table rather than assumed
        # to be 0, because "every run begins at slot 0" is a property of the
        # current tree policy — requests are allocated from slot 0 — and not of
        # the page table itself. Adding this does not change any number today;
        # it stops `commit` from silently disagreeing with the table if that
        # ever stops being true.
        if cursor < self.rq_nblk[slot]:
            inblk += self.bstart_at(slot, cursor)

        while remaining > 0:
            var cap = imin(MAX_NODE_TOKENS, MAX_NODE_BLOCKS * bs - inblk)
            var take = imin(cap, remaining)
            var nb = ceil_div(inblk + take, bs)
            if cursor + nb > self.rq_nblk[slot]:
                raise AlofaError(ERR_INVALID_ARGUMENT, "request blocks exhausted")
            var node = self.tree.alloc_node()
            self.tree.offset[node] = inblk
            self.tree.ntok[node] = take
            for i in range(take):
                self.tree.set_token(node, i, self.token_at(slot, k + i))
            self.tree.nblk[node] = nb
            for i in range(nb):
                var block = self.block_at(slot, cursor + i)
                self.tree.set_block(node, i, block)
                self.pool.retain(block)
                res.shared.push_unique(block)
            self.tree.add_child(parent, node)
            res.nodes.push(node)
            parent = node
            k += take
            remaining -= take
            var slots = inblk + take
            cursor += slots // bs
            inblk = slots % bs

        # Its blocks now belong to the tree as well: no more in-place growth.
        self.rq_priv[slot] = 0
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res^

    def match_only(
        mut self, tokens: InlineArray[Int, MAX_SEQ_TOKENS], n: Int
    ) raises AlofaError -> OpResult:
        """Prefix lookup with no side effects beyond the tree's frequency count.

        It exists so the fixture can ask "what would this hit" without changing
        anything: a read-only op is the only way to catch a lookup that mutates
        by accident.
        """
        var res = OpResult()
        var m = self.tree.match(tokens, n)
        res.matched = m.matched
        if m.matched > 0:
            res.node = m.node
            res.offset = self.tree.offset[m.path[0]]
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res^

    def release(mut self, slot: Int) raises AlofaError -> OpResult:
        var res = OpResult()
        for i in range(self.rq_nblk[slot]):
            if self.pool.release(self.block_at(slot, i)):
                res.fresh.push(self.block_at(slot, i))
        self.rq_live[slot] = 0
        self.rq_nblk[slot] = 0
        self.rq_ntok[slot] = 0
        self.rq_off[slot] = 0
        self.rq_priv[slot] = 0
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res^

    def evict(mut self, node: Int) raises AlofaError -> OpResult:
        """Free a node and its whole subtree.

        No policy here on purpose: which node to evict is a separate capability
        with its own replay gate, and a policy smuggled in here would be tested
        only by accident.
        """
        if node <= ROOT or node >= MAX_NODES or self.tree.alive[node] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "evict: bad node")
        var res = OpResult()
        self.tree.detach(node)
        var stack = InlineArray[Int, MAX_NODES](fill=0)
        var n = 1
        stack[0] = node
        while n > 0:
            n -= 1
            var cur = stack[n]
            var c = self.tree.child[cur]
            while c != INVALID:
                stack[n] = c
                n += 1
                c = self.tree.sibling[c]
            for i in range(self.tree.nblk[cur]):
                if self.pool.release(self.tree.block_at(cur, i)):
                    res.fresh.push(self.tree.block_at(cur, i))
            self.tree.kill(cur)
        res.node = node
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res^

    def tokens_of(self, slot: Int) -> InlineArray[Int, MAX_SEQ_TOKENS]:
        var out = InlineArray[Int, MAX_SEQ_TOKENS](fill=0)
        for i in range(self.rq_ntok[slot]):
            out[i] = self.rq_tok[slot * MAX_SEQ_TOKENS + i]
        return out^

    # --- the consistency equation ---

    def check_invariants(self) -> Int:
        """Number of violated conditions. Zero is the only acceptable answer.

        This is recomputed from the views and compared against the pool, so it
        catches a missing retain even when the reference implementation shares
        the bug.
        """
        var bad = 0
        var counted = InlineArray[Int, MAX_BLOCKS](fill=0)

        for slot in range(MAX_REQUESTS):
            if self.rq_live[slot] == 0:
                continue
            var total = 0
            for i in range(self.rq_nblk[slot]):
                var block = self.block_at(slot, i)
                if block < 0 or block >= MAX_BLOCKS:
                    bad += 1
                    continue
                counted[block] += 1
                total += self.blen_at(slot, i)
                if self.blen_at(slot, i) <= 0:
                    bad += 1
                if self.blen_at(slot, i) > self.pool.block_size:
                    bad += 1
                # The run must fit inside the block it claims to start in:
                # `start + len > block_size` is a table that reads the next
                # block's bytes through the back door.
                if self.bstart_at(slot, i) < 0:
                    bad += 1
                if (
                    self.bstart_at(slot, i) + self.blen_at(slot, i)
                    > self.pool.block_size
                ):
                    bad += 1
            if total != self.rq_ntok[slot]:
                bad += 1

        for node in range(MAX_NODES):
            if self.tree.alive[node] == 0:
                continue
            if node == ROOT:
                continue
            if self.tree.ntok[node] <= 0 or self.tree.ntok[node] > MAX_NODE_TOKENS:
                bad += 1
            if self.tree.nblk[node] != ceil_div(
                self.tree.offset[node] + self.tree.ntok[node], self.pool.block_size
            ):
                bad += 1
            if self.tree.nblk[node] > MAX_NODE_BLOCKS:
                bad += 1
            for i in range(self.tree.nblk[node]):
                counted[self.tree.block_at(node, i)] += 1

        for block in range(MAX_BLOCKS):
            if counted[block] != self.pool.refcnt[block]:
                bad += 1

        # The free list is exactly the set of blocks nobody holds.
        if self.pool.n_free + self.pool.used != MAX_BLOCKS:
            bad += 1
        var seen = InlineArray[Int, MAX_BLOCKS](fill=0)
        for i in range(self.pool.n_free):
            var block = self.pool.free_stack[i]
            if block < 0 or block >= MAX_BLOCKS:
                bad += 1
                continue
            seen[block] += 1
        for block in range(MAX_BLOCKS):
            var want = 1
            if self.pool.refcnt[block] > 0:
                want = 0
            if seen[block] != want:
                bad += 1
        return bad

    def digest(self) -> Int:
        var h = self.pool.digest()
        h = (h * DIGEST_MUL + self.tree.digest()) % DIGEST_MOD
        for slot in range(MAX_REQUESTS):
            h = (h * DIGEST_MUL + self.rq_live[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_id[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_ntok[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_off[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_nblk[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_priv[slot]) % DIGEST_MOD
            for i in range(self.rq_nblk[slot]):
                h = (h * DIGEST_MUL + self.block_at(slot, i)) % DIGEST_MOD
                h = (h * DIGEST_MUL + self.blen_at(slot, i)) % DIGEST_MOD
                h = (h * DIGEST_MUL + self.bstart_at(slot, i)) % DIGEST_MOD
        return h


def remaining_fresh(res: OpResult) -> Int:
    """1 when the operation allocated at least one block of its own."""
    if res.fresh.n > 0:
        return 1
    return 0
