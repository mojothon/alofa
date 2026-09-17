"""Unified KV addressing: one physical pool, two index views, one refcount.

Why this exists
---------------

A KV cache has two things that want to own it: the **page table** (this request
owns these blocks, in this order) and the **prefix tree** (these blocks hold the
KV of this token sequence, whoever asked for it). Keeping them as two allocators
is how you get the failure vLLM and SGLang both live with: the tree says "this
prefix is evictable" while the pager says "this block is in use by a live
request", and the two answers are computed from different books.

Here both views are *indices into the same blocks*, and the block's `refcnt` is
the sum of what the views hold. That is the whole trick, and it makes the
consistency check a single equation:

    for every block b:  refcnt[b] == (requests holding b) + (nodes holding b)

Two consequences worth stating up front, because they are the load-bearing ones:

- **A block is never copied.** Prefix sharing, node splitting and node chaining
  all move (block id, offset, length) triples around. The only thing that ever
  moves bytes is the model writing its own KV.
- **Prefix matching is token-granular, storage is block-granular, and the seam
  costs one partial block.** A request that matches 5 tokens of a 16-token block
  shares that block and then allocates a *fresh* block for its own continuation:
  it must not write into the tail, because the tail belongs to the cached
  continuation. The alternative is copy-on-write; this round deliberately does
  not do it, because "wasted tail slots" is a cost you can measure, while
  "silently wrote over someone else's prefix" is not.

What is deliberately *not* here
-------------------------------

- **No eviction policy.** `evict` takes the node id and frees that subtree. The
  2Q / composite-score policy is a separate capability with its own replay gate;
  putting a policy here would let it hide behind this module's tests.
- **No node merging** after eviction. SGLang merges a parent that is left with a
  single child; that is an optimisation of the tree shape and has no effect on
  refcount consistency, so it waits for the eviction round.
- **No KV payload.** This module accounts for blocks. The bytes behind a block
  are the model's business (P2.2 paged attention), and keeping them out is what
  lets this be tested without a 2 GB checkpoint.

Capacities are compile-time constants on purpose: nothing here can grow, so
"steady state allocates nothing" is a property of the types rather than of
discipline.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_kv_pool.mojo
"""

from alofa.core.error import ERR_CAPACITY, AlofaError

# Physical pool. Deliberately smaller than MAX_REQUESTS * MAX_BLOCKS_PER_SEQ:
# over-subscription is the normal case for a KV cache, and it is what makes the
# "pool exhausted" path reachable (and therefore testable) at all.
comptime MAX_BLOCKS = 112
# Blocks referenced by one sequence (a request or a tree node).
comptime MAX_BLOCKS_PER_SEQ = 16
# Live requests at once.
comptime MAX_REQUESTS = 8
# Radix tree nodes.
comptime MAX_NODES = 32
# Tokens covered by one node. Longer runs become a chain of nodes.
comptime MAX_NODE_TOKENS = 16
# Blocks one node may reference (>= ceil((MAX_NODE_TOKENS + block_size) / block_size)).
comptime MAX_NODE_BLOCKS = 4
# Tokens in one operation.
comptime MAX_SEQ_TOKENS = 64
# Nodes visited while matching one sequence.
comptime MAX_PATH = 8

comptime ROOT = 0
comptime INVALID = -1
comptime DEFAULT_BLOCK_SIZE = 16

# Digest arithmetic stays in Int64: MOD is 2^31-1, so h * MUL < 2^51.
comptime DIGEST_MOD = 2147483647
comptime DIGEST_MUL = 1000003


def imin(a: Int, b: Int) -> Int:
    """Smaller of two, without importing a collection to do it."""
    var m = a
    if b < m:
        m = b
    return m


def imax(a: Int, b: Int) -> Int:
    var m = a
    if b > m:
        m = b
    return m


def ceil_div(a: Int, b: Int) -> Int:
    """Ceiling division for non-negative `a` and positive `b`."""
    return (a + b - 1) // b


struct IdList:
    """A short fixed-capacity list of ids (blocks or nodes).

    It exists so that operations can report *which* blocks they touched:
    "the pool released 3 blocks" is a claim, "it released 7, 8, 9" is evidence.
    """

    var v: InlineArray[Int, MAX_BLOCKS_PER_SEQ]
    var n: Int

    def __init__(out self):
        self.v = InlineArray[Int, MAX_BLOCKS_PER_SEQ](fill=0)
        self.n = 0

    def push(mut self, value: Int) raises AlofaError:
        if self.n >= MAX_BLOCKS_PER_SEQ:
            raise AlofaError(ERR_CAPACITY, "id list overflow")
        self.v[self.n] = value
        self.n += 1

    def push_unique(mut self, value: Int) raises AlofaError:
        """Push unless the *last* entry is already `value`.

        Adjacent duplicates happen legitimately: consecutive tree nodes share
        the block their boundary falls in.
        """
        if self.n > 0:
            if self.v[self.n - 1] == value:
                return
        self.push(value)

    def at(self, i: Int) -> Int:
        return self.v[i]


struct OpResult:
    """What one operation did. Every op fills the fields it uses; the rest stay
    at `INVALID` / empty, and the trace printer only writes the ones that apply.
    """

    var shared: IdList
    var fresh: IdList
    var nodes: IdList
    var matched: Int
    var offset: Int
    var node: Int
    var used: Int
    var n_free: Int

    def __init__(out self):
        self.shared = IdList()
        self.fresh = IdList()
        self.nodes = IdList()
        self.matched = INVALID
        self.offset = INVALID
        self.node = INVALID
        self.used = 0
        self.n_free = 0
