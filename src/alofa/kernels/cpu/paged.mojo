"""Paged attention: the oracle's arithmetic, read through a block table.

What "paged" changes, and what it must not
------------------------------------------

`scalar.attention` reads a contiguous `[kv_len, kv_cols]` matrix. A paged cache
is not that: it is `[n_blocks, block_size, kv_cols]`, and the context one
request attends over is a *list of runs* — `(block, start, length)` triples —
because a shared prefix can begin in the middle of a block. So the only thing
that changes here is how a row is addressed. The arithmetic is deliberately
identical to the oracle's, and
`tests/unit/test_paged_attention.mojo` asserts that identity **bit for bit**
rather than within a tolerance.

That identity *is* the claim. Paging is an addressing change, so if the paged
kernel disagrees with the contiguous one by even one ulp, the bug is in the
addressing: there is no reassociation to hide behind, because no operation was
reordered. A future fused or vectorised paged kernel *will* reorder, and will
then be held to a tolerance instead — against this module, which is what makes
this module worth writing in formula order.

Why the oracle's body is repeated here
--------------------------------------

The scalar oracle is meant to be frozen and readable top to bottom; a shared
core would make "which arithmetic does the oracle use" depend on a call graph.
So the body is repeated, and the bit-exact gate is what keeps the two copies in
step: if they drift apart, the gate goes red rather than quietly tolerating it.

The table is the contract
-------------------------

`PagedTable` carries `start` because a run need not begin at slot 0. Dropping
it — reading every run from the start of its block — produces a plausible
number computed from somebody else's tokens, which is the one failure mode a
paged kernel must not have. The kernel therefore *rejects* a table whose runs
are empty, negative, or run off the end of their block, instead of clamping:
a clamped table would turn that bug into a wrong answer.

**Nothing here allocates.** Every container is an `InlineArray` sized at compile
time, and there is no `String` anywhere — not even in an error message, because
the zero-allocation gate is textual and a message that allocates would have to
be carved out of it. Diagnostics are therefore the error kind plus a fixed
sentence; the caller knows which table it passed.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_paged_attention.mojo
"""

from std.math import exp, sqrt

from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    ERR_SHAPE_MISMATCH,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.tensor import TensorView, f32_data
from alofa.kernels.cpu.scalar import expect_matrix, rows_of

# The seed for a softmax row maximum: the most negative finite fp32 value, so
# the first unmasked score always wins and no row needs a special case.
comptime LOWEST_FP32 = Float32(-3.4028234663852886e38)

# Returned by `PagedTable.row_offset` for a position the table does not cover.
# The kernels validate first, so this is a bug detector, not a control path.
comptime BAD_OFFSET = -1


struct PagedTable[MAX_ENTRIES: Int = 64]:
    """A context as a list of runs: `(block, start slot, length)`.

    This is what `KvSpace` holds per request (see `runtime/kv/paging.mojo`) and
    what a paged kernel needs that a contiguous one does not. `start` is the
    slot inside the block where the run begins; it is non-zero exactly when the
    request shares a prefix that begins mid-block.

    Nothing here checks the runs — `check_table` does, once, against the cache
    the table indexes. A table on its own does not know how many blocks exist.
    """

    var ids: InlineArray[Int, Self.MAX_ENTRIES]
    var starts: InlineArray[Int, Self.MAX_ENTRIES]
    var lens: InlineArray[Int, Self.MAX_ENTRIES]
    var n: Int
    var block_size: Int

    def __init__(out self, block_size: Int):
        self.ids = InlineArray[Int, Self.MAX_ENTRIES](fill=0)
        self.starts = InlineArray[Int, Self.MAX_ENTRIES](fill=0)
        self.lens = InlineArray[Int, Self.MAX_ENTRIES](fill=0)
        self.n = 0
        self.block_size = block_size

    def push(mut self, block: Int, start: Int, length: Int) -> Bool:
        """Append a run; False when the table is full.

        Full is reported rather than raised so the capacity check stays where
        the caller can see it: the kernel sizes its table, and a silent
        overflow would drop the tail of someone's context.
        """
        if self.n >= Self.MAX_ENTRIES:
            return False
        self.ids[self.n] = block
        self.starts[self.n] = start
        self.lens[self.n] = length
        self.n += 1
        return True

    def clear(mut self):
        self.n = 0

    def block_at(self, i: Int) -> Int:
        return self.ids[i]

    def start_at(self, i: Int) -> Int:
        return self.starts[i]

    def length_at(self, i: Int) -> Int:
        return self.lens[i]

    def n_tokens(self) -> Int:
        var total = 0
        for i in range(self.n):
            total += self.lens[i]
        return total

    def row_offset(self, j: Int, kv_cols: Int) -> Int:
        """Element offset of context position `j`; `BAD_OFFSET` if out of range.

        The walk is linear in the number of runs. For the oracle backend that
        is the right trade: a per-token index would be another structure to
        keep in step with this one, and the point here is to have the
        addressing in one readable place.
        """
        var acc = 0
        for i in range(self.n):
            if j < acc + self.lens[i]:
                var slot = self.starts[i] + (j - acc)
                return (self.ids[i] * self.block_size + slot) * kv_cols
            acc += self.lens[i]
        return BAD_OFFSET


def check_table[MAX_ENTRIES: Int](
    table: PagedTable[MAX_ENTRIES], n_blocks: Int
) raises AlofaError -> Int:
    """Validate a table against the cache it indexes; return the context length.

    Every rejection here is a case where clamping would have produced a
    plausible number from the wrong bytes.
    """
    if table.block_size <= 0:
        raise AlofaError(ERR_INVALID_ARGUMENT, "block size must be positive")
    var total = 0
    for i in range(table.n):
        if table.starts[i] < 0:
            raise AlofaError(ERR_OUT_OF_RANGE, "page table run starts before its block")
        if table.lens[i] <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "page table run is empty")
        if table.starts[i] + table.lens[i] > table.block_size:
            raise AlofaError(ERR_OUT_OF_RANGE, "page table run leaves its block")
        if table.ids[i] < 0 or table.ids[i] >= n_blocks:
            raise AlofaError(ERR_OUT_OF_RANGE, "page table names an absent block")
        total += table.lens[i]
    return total


def cache_cols(cache: TensorView, table_block_size: Int) raises AlofaError -> Int:
    """Check `cache` is `[n_blocks, block_size, kv_cols]`; return `kv_cols`.

    Rank 3 rather than a flat matrix so that the block stride is stated by the
    shape instead of by a convention the caller has to remember.
    """
    if cache.rank() != 3:
        raise AlofaError(ERR_SHAPE_MISMATCH, "expected a 3-D [blocks, size, cols] cache")
    if cache.shape.dims[1] != table_block_size:
        raise AlofaError(ERR_SHAPE_MISMATCH, "cache block size disagrees with the table")
    return cache.shape.dims[2]


def paged_gather[MAX_ENTRIES: Int](
    dst: TensorView, cache: TensorView, table: PagedTable[MAX_ENTRIES]
) raises AlofaError:
    """Materialise the table's context as a contiguous `[kv_len, kv_cols]` matrix.

    This is what "paged attention without a paged kernel" looks like, which
    makes it two things at once: the obvious thing to check the paged kernel
    against, and the reason the paged kernel exists — it copies every row it
    reads, which is precisely the copy paged attention is there to avoid.
    """
    var kv_cols = cache_cols(cache, table.block_size)
    var kv_len = check_table(table, cache.shape.dims[0])
    expect_matrix(dst, kv_len, kv_cols, "dst")

    var pc = f32_data(cache)
    var pd = f32_data(dst)
    for j in range(kv_len):
        var src = table.row_offset(j, kv_cols)
        if src < 0:
            raise AlofaError(ERR_OUT_OF_RANGE, "page table does not cover a position")
        var out = j * kv_cols
        for c in range(kv_cols):
            pd[unsafe_offset=out + c] = pc[unsafe_offset=src + c]


def paged_scatter[MAX_ENTRIES: Int](
    cache: TensorView,
    table: PagedTable[MAX_ENTRIES],
    first: Int,
    src: TensorView,
) raises AlofaError:
    """Write `src`'s rows into the cache at positions `[first, first + n)`.

    The write that fills a block and the read that attends over it must ask the
    same table the same question, which is why this exists rather than a copy
    into `(block, slot)` computed by the caller: a write at `first + j` by
    arithmetic places a token where the request *would* have kept it had it not
    shared a prefix, and every number downstream is then computed from a
    history that is quietly somebody else's.

    Refuses a write past the table instead of clamping. A clamped write is a
    shorter context, and a shorter context surfaces as a missing token rather
    than as an error — the caller must hear about it here.

    Positions outside `[first, first + n)` are left exactly as they were, which
    is what makes a block shared by two requests safe to write into twice: the
    shared part is written with the same bytes both times.
    """
    var kv_cols = cache_cols(cache, table.block_size)
    var kv_len = check_table(table, cache.shape.dims[0])
    var n = rows_of(src, "src")
    expect_matrix(src, n, kv_cols, "src")
    if first < 0 or first + n > kv_len:
        raise AlofaError(ERR_OUT_OF_RANGE, "scatter would write past the table")

    var pc = f32_data(cache)
    var ps = f32_data(src)
    for j in range(n):
        var dst = table.row_offset(first + j, kv_cols)
        if dst < 0:
            raise AlofaError(ERR_OUT_OF_RANGE, "page table leaves a hole")
        var at = j * kv_cols
        for c in range(kv_cols):
            pc[unsafe_offset=dst + c] = ps[unsafe_offset=at + c]


def paged_attention[MAX_ENTRIES: Int](
    dst: TensorView,
    q: TensorView,
    k_cache: TensorView,
    v_cache: TensorView,
    table: PagedTable[MAX_ENTRIES],
    scores: TensorView,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
) raises AlofaError:
    """Causal grouped-query attention over a paged cache.

    The arithmetic is the oracle's, in the oracle's order: the dot product
    accumulates over `head_dim` in `Float64`, the scale is applied once, and
    the softmax is three passes over the row (maximum, exponentiated weights,
    value reduction). Only the row address changes — `j * kv_cols` becomes
    `table.row_offset(j, kv_cols)` — which is what makes the bit-exact
    comparison against `scalar.attention` meaningful rather than lucky.

    `scores` is caller-owned scratch of at least `q_len * kv_len` elements;
    this kernel allocates nothing.
    """
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise AlofaError(ERR_UNSUPPORTED, "n_kv_heads must divide n_heads")
    var kv_cols = cache_cols(k_cache, table.block_size)
    if cache_cols(v_cache, table.block_size) != kv_cols:
        raise AlofaError(ERR_SHAPE_MISMATCH, "k and v caches differ in column count")
    if k_cache.shape.dims[0] != v_cache.shape.dims[0]:
        raise AlofaError(ERR_SHAPE_MISMATCH, "k and v caches differ in block count")
    if kv_cols != n_kv_heads * head_dim:
        raise AlofaError(ERR_SHAPE_MISMATCH, "cache columns disagree with head counts")

    var kv_len = check_table(table, k_cache.shape.dims[0])
    var q_len = rows_of(q, "q")
    if q_len > kv_len:
        raise AlofaError(ERR_SHAPE_MISMATCH, "more queries than cached positions")
    expect_matrix(q, q_len, n_heads * head_dim, "q")
    expect_matrix(dst, q_len, n_heads * head_dim, "dst")
    if scores.numel() < q_len * kv_len:
        raise AlofaError(ERR_SHAPE_MISMATCH, "score scratch is too small")

    var pq = f32_data(q)
    var pk = f32_data(k_cache)
    var pv = f32_data(v_cache)
    var ps = f32_data(scores)
    var po = f32_data(dst)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var group = n_heads // n_kv_heads
    var q_cols = n_heads * head_dim

    for head in range(n_heads):
        var kv_head = head // group
        var q_head_base = head * head_dim
        var kv_head_base = kv_head * head_dim
        for t in range(q_len):
            # The last `q_len` positions of a `kv_len`-long context.
            var upto = kv_len - q_len + t
            var row = t * kv_len
            var best = LOWEST_FP32
            for j in range(upto + 1):
                var kbase = table.row_offset(j, kv_cols)
                if kbase < 0:
                    raise AlofaError(ERR_OUT_OF_RANGE, "page table leaves a hole")
                var acc = Float64(0)
                for d in range(head_dim):
                    acc += Float64(pq[unsafe_offset=t * q_cols + q_head_base + d]) * Float64(
                        pk[unsafe_offset=kbase + kv_head_base + d]
                    )
                var s = Float32(acc) * scale
                ps[unsafe_offset=row + j] = s
                if s > best:
                    best = s
            var total = Float64(0)
            for j in range(upto + 1):
                var e = Float64(exp(ps[unsafe_offset=row + j] - best))
                ps[unsafe_offset=row + j] = Float32(e)
                total += e
            for d in range(head_dim):
                var acc = Float64(0)
                for j in range(upto + 1):
                    var vbase = table.row_offset(j, kv_cols)
                    acc += Float64(ps[unsafe_offset=row + j]) * Float64(
                        pv[unsafe_offset=vbase + kv_head_base + d]
                    )
                po[unsafe_offset=t * q_cols + q_head_base + d] = Float32(
                    acc / total
                )
