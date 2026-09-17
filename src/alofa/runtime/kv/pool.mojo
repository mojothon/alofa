"""The physical block pool: the only place a block exists.

A block here is an integer. That is the point — the KV bytes are the model's
business, while *who is currently holding block 37* is everybody's business, and
it is the thing that goes wrong silently. So the pool is a refcount array plus a
free list, and every other view in this package is a set of references into it.

Allocation order is part of the contract, not an implementation detail: the free
list starts out as `0, 1, 2, …` and is popped from the tail, so a trace that says
"block 3" means block 3 in both implementations. Release pushes back to the tail
(LIFO), which is what makes a freed block come back immediately — a property the
exhaustion test relies on.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_kv_pool.mojo
"""

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_DOUBLE_FREE,
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    AlofaError,
)
from alofa.runtime.kv import DIGEST_MOD, DIGEST_MUL, MAX_BLOCKS


struct BlockPool:
    """Refcount per block, plus a free stack. Nothing else."""

    var refcnt: InlineArray[Int, MAX_BLOCKS]
    var free_stack: InlineArray[Int, MAX_BLOCKS]
    var n_free: Int
    var used: Int
    var block_size: Int

    def __init__(out self, block_size: Int) raises AlofaError:
        if block_size <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "block_size must be positive")
        self.refcnt = InlineArray[Int, MAX_BLOCKS](fill=0)
        self.free_stack = InlineArray[Int, MAX_BLOCKS](fill=0)
        # Highest id at the bottom of the stack, so the first allocations come
        # out as 0, 1, 2, ... — an order both implementations can agree on
        # without a shared convention beyond "pop the tail".
        for i in range(MAX_BLOCKS):
            self.free_stack[i] = MAX_BLOCKS - 1 - i
        self.n_free = MAX_BLOCKS
        self.used = 0
        self.block_size = block_size

    def alloc_one(mut self) raises AlofaError -> Int:
        """Take a block off the free list. Raises when the pool is empty.

        Failing loudly is the honest outcome: the alternative ("this tick just
        gets nothing") produces a request that looks like it is generating while
        going nowhere.
        """
        if self.n_free <= 0:
            raise AlofaError(ERR_CAPACITY, "kv block pool is exhausted")
        self.n_free -= 1
        var block = self.free_stack[self.n_free]
        if self.refcnt[block] != 0:
            raise AlofaError(ERR_DOUBLE_FREE, "free list holds a referenced block")
        self.refcnt[block] = 1
        self.used += 1
        return block

    def retain(mut self, block: Int) raises AlofaError:
        """One more holder of `block`.

        Retaining a block nobody holds is a bug in the caller, not a state the
        pool can repair: it would mean a view is claiming a block by id without
        having been given it.
        """
        if block < 0 or block >= MAX_BLOCKS:
            raise AlofaError(ERR_OUT_OF_RANGE, "block id out of range")
        if self.refcnt[block] <= 0:
            raise AlofaError(ERR_DOUBLE_FREE, "retain of a block nobody holds")
        self.refcnt[block] += 1

    def release(mut self, block: Int) raises AlofaError -> Bool:
        """Drop one holder. Returns True when the block became free."""
        if block < 0 or block >= MAX_BLOCKS:
            raise AlofaError(ERR_OUT_OF_RANGE, "block id out of range")
        if self.refcnt[block] <= 0:
            raise AlofaError(ERR_DOUBLE_FREE, "release of a block nobody holds")
        self.refcnt[block] -= 1
        if self.refcnt[block] > 0:
            return False
        self.free_stack[self.n_free] = block
        self.n_free += 1
        self.used -= 1
        return True

    def refcnt_of(self, block: Int) -> Int:
        if block < 0 or block >= MAX_BLOCKS:
            return 0
        return self.refcnt[block]

    def digest(self) -> Int:
        """Integer fingerprint: refcounts, the free list *in order*, and counts.

        The free list order is included because it determines the next
        allocation, and two states with the same `used` but different orders
        will diverge one operation later.
        """
        var h = 0
        for i in range(MAX_BLOCKS):
            h = (h * DIGEST_MUL + self.refcnt[i]) % DIGEST_MOD
        for i in range(self.n_free):
            h = (h * DIGEST_MUL + self.free_stack[i]) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.n_free) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.used) % DIGEST_MOD
        return h
