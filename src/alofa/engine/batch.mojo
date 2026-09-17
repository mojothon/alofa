"""Batch assembly and the batch tensor pool.

What this is for
----------------

Every layer of a forward wants scratch: hidden states for this step's tokens, the
attention score matrix, the three MLP intermediates. Today that scratch is carved
out once when a model is loaded, sized for `max_tokens` of *one* sequence. A batch
breaks that assumption — several sequences, different lengths, arriving and
leaving at different steps — and the tempting repair is to allocate whatever the
current batch happens to need.

That repair costs exactly the thing P2 is trying to buy. A heap allocation inside
the busy loop is not just slow: it is unbounded, so a batch that happens to be
larger than the last one can fail in production having passed every smaller test.
So the rule here is that the pool is allocated **once**, before any request is
served, and once it exists it never allocates again.

What "never allocates" is made of
---------------------------------

Two pieces, and neither is a discipline:

- The pool does not own its memory. It is handed a pointer and a capacity, so
  "where does the backing store come from" is somebody else's question, and this
  module can be tested on a small mapping.
- Every piece of bookkeeping is a compile-time-sized `InlineArray`. There is no
  free list to disagree with its own contents — see below.

The missing free list is deliberate
-----------------------------------

Most allocators keep a free list and derive nothing; then they *also* carry enough
per-allocation state to catch themselves, and the two books drift. Here the live
intervals are the only book there is. Placement re-derives the gaps from the live
set every time, first fit by increasing address. Consequences worth naming:

- Overlap is impossible by construction, so it cannot be introduced later by an
  edit to the free list.
- There is no fragmentation that survives a release: a gap exists only while
  nothing is using those bytes.
- Placement is O(n) in the number of live borrows with n capped at 32, which is
  irrelevant next to the arithmetic that will be done in those bytes.

The one thing the caller must still remember is that a borrow is a **handle**, not
a pointer: the pool may hand slot 3 to something else once you release it. Holding
a pointer past its release is a use-after-free wearing a plausible number.

This module is in L4 and depends on `core` only. It knows nothing about KV blocks,
page tables or model weights — 2.1 and 2.2 are L3/L1 concerns, and a batch pool
that could not be tested without them would repeat the mistake their design was
made to get out of.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_batch_pool.mojo
"""

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_DOUBLE_FREE,
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    AlofaError,
)
from alofa.core.ffi.mem import RawPtr

# Live borrows at once. One forward borrows a handful per layer, so 24 is roomy;
# it is a cap rather than a target, and hitting it is a named error rather than a
# silent grow.
comptime MAX_HANDLES = 24
# Requests assembled into one forward. Same value MAX_REQUESTS uses in the KV
# space, because a batch that cannot be described to the pool is a batch that
# cannot be served.
comptime MAX_BATCH = 8
# Bytes. Chosen for cache-line sanity, and because a borrowed region that shared
# a line with a neighbour would let a stray write look like someone else's bug.
comptime POOL_ALIGN = 64
# Returned by internal searches that found nothing.
comptime NO_SLOT = -1
# Returned by `find` when a request is not in the batch.
comptime NO_REQUEST = -1

comptime BATCH_MUL = 1000003
comptime BATCH_MOD = 2147483647

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]


def ceil_bytes(size: Int) -> Int:
    """Round `size` up to `POOL_ALIGN`, so neighbours never share a line."""
    return (size + POOL_ALIGN - 1) // POOL_ALIGN * POOL_ALIGN


struct BatchPool:
    """Borrow and give back byte intervals of one buffer it does not own."""

    var base: RawPtr
    var capacity: Int
    var used: Int
    var live: InlineArray[Int, MAX_HANDLES]
    var begin: InlineArray[Int, MAX_HANDLES]
    var size: InlineArray[Int, MAX_HANDLES]
    var high_water: Int

    def __init__(out self, base: RawPtr, capacity: Int) raises AlofaError:
        if capacity <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "pool capacity must be positive"
            )
        if capacity != ceil_bytes(capacity):
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "pool capacity must be a multiple of the alignment"
            )
        self.base = base
        self.capacity = capacity
        self.used = 0
        self.live = InlineArray[Int, MAX_HANDLES](fill=0)
        self.begin = InlineArray[Int, MAX_HANDLES](fill=0)
        self.size = InlineArray[Int, MAX_HANDLES](fill=0)
        self.high_water = 0

    def n_live(self) -> Int:
        """Live borrows. Reported because "the pool freed two" is a claim."""
        var n = 0
        for h in range(MAX_HANDLES):
            if self.live[h] == 1:
                n += 1
        return n

    def offset_of(self, handle: Int) -> Int:
        """Where this borrow starts. Valid after release, which is the point."""
        return self.begin[handle]

    def length_of(self, handle: Int) -> Int:
        return self.size[handle]

    def covers(self, handle: Int, cap: Int) -> Bool:
        """Whether `handle` is a live borrow that fits inside `cap`."""
        if self.live[handle] != 1:
            return False
        var end = self.begin[handle] + self.size[handle]
        return end <= cap

    def f32_of(self, handle: Int) raises AlofaError -> F32Ptr:
        """Typed access to a live borrow.

        The convenient answer would be to hand this out on borrow and let it
        dangle past release. Handing it out per use means the only way to reach
        stale bytes is to lie about which handle you still hold.
        """
        if handle < 0 or handle >= MAX_HANDLES:
            raise AlofaError(ERR_OUT_OF_RANGE, "borrow handle out of range")
        if self.live[handle] == 0:
            raise AlofaError(ERR_DOUBLE_FREE, "borrow is not live")
        return self.base.unsafe_offset(self.begin[handle]).unsafe_bitcast[
            Float32
        ]()

    def free_slot(self) -> Int:
        """Lowest unused handle slot, so placement is reproducible."""
        for h in range(MAX_HANDLES):
            if self.live[h] == 0:
                return h
        return NO_SLOT

    def place(self, aligned: Int) -> Int:
        """Lowest address at which `aligned` bytes of nothing-live begins.

        Gaps are derived from the live set rather than kept in a second
        structure: there is then nothing that can disagree with the answer.
        """
        var lo = InlineArray[Int, MAX_HANDLES](fill=0)
        var hi = InlineArray[Int, MAX_HANDLES](fill=0)
        var n = 0
        for h in range(MAX_HANDLES):
            if self.live[h] != 1:
                continue
            var start = self.begin[h]
            var stop = start + self.size[h]
            var i = n
            while i > 0 and lo[i - 1] > start:
                lo[i] = lo[i - 1]
                hi[i] = hi[i - 1]
                i -= 1
            lo[i] = start
            hi[i] = stop
            n += 1
        var pos = 0
        var k = 0
        while k < n:
            if lo[k] - pos >= aligned:
                return pos
            if hi[k] > pos:
                pos = hi[k]
            k += 1
        if self.capacity - pos >= aligned:
            return pos
        return NO_SLOT

    def borrow(mut self, size: Int) raises AlofaError -> Int:
        """Reserve `size` bytes and return the handle that owns them."""
        if size <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "borrow size must be positive"
            )
        var aligned = ceil_bytes(size)
        if aligned > self.capacity:
            raise AlofaError(
                ERR_CAPACITY, "pool is too small for this borrow even when empty"
            )
        var handle = self.free_slot()
        if handle == NO_SLOT:
            raise AlofaError(ERR_CAPACITY, "too many live borrows at once")
        var at = self.place(aligned)
        if at == NO_SLOT:
            # Refusing is the whole point. Wrapping around, or scribbling past
            # the end, would produce a number instead of a diagnosis.
            raise AlofaError(ERR_CAPACITY, "pool has no contiguous space left")
        self.begin[handle] = at
        self.size[handle] = aligned
        self.live[handle] = 1
        self.used += aligned
        var top = at + aligned
        if top > self.high_water:
            self.high_water = top
        return handle

    def release(mut self, handle: Int) raises AlofaError -> Int:
        """Give a borrow back; returns the offset that was released."""
        if handle < 0 or handle >= MAX_HANDLES:
            raise AlofaError(ERR_OUT_OF_RANGE, "borrow handle out of range")
        if self.live[handle] == 0:
            raise AlofaError(ERR_DOUBLE_FREE, "borrow was already released")
        self.live[handle] = 0
        self.used -= self.size[handle]
        return self.begin[handle]

    def reset(mut self):
        """Reclaim everything. This is the per-step boundary of an executor."""
        for h in range(MAX_HANDLES):
            self.live[h] = 0
        self.used = 0

    def defects(self) -> Int:
        """Violations found by re-deriving everything from the borrow records.

        Deliberately independent of anything the allocator believes: both this
        and `place` are computed from `live` / `begin` / `size`, but they check
        different things, and neither is asked to remember a running total.
        """
        var bad = 0
        for h in range(MAX_HANDLES):
            if self.live[h] != 1:
                continue
            if self.begin[h] < 0:
                bad += 1
            if self.begin[h] != ceil_bytes(self.begin[h]):
                bad += 1
            if self.size[h] <= 0:
                bad += 1
            if self.begin[h] + self.size[h] > self.capacity:
                bad += 1
        for i in range(MAX_HANDLES):
            if self.live[i] != 1:
                continue
            for j in range(i + 1, MAX_HANDLES):
                if self.live[j] != 1:
                    continue
                var ai = self.begin[i]
                var bi = ai + self.size[i]
                var aj = self.begin[j]
                var bj = aj + self.size[j]
                if ai < bj and aj < bi:
                    bad += 1
        var sum_bytes = 0
        for h in range(MAX_HANDLES):
            if self.live[h] == 1:
                sum_bytes += self.size[h]
        if sum_bytes != self.used:
            bad += 1
        return bad

    def digest(self) -> Int:
        """One number over the whole pool, including the released slots.

        Empty slots are part of the answer: two histories that reached the same
        live set through different paths differ, and the reference must be able
        to tell them apart rather than agreeing on every reachable state.
        """
        var value = 0
        for h in range(MAX_HANDLES):
            value = (
                value * BATCH_MUL
                + self.begin[h]
                + self.size[h] * 7
                + self.live[h] * 13
            ) % BATCH_MOD
        value = (value * BATCH_MUL + self.used) % BATCH_MOD
        value = (value * BATCH_MUL + self.high_water) % BATCH_MOD
        return value


struct BatchSlots:
    """Which rows of the packed token axis belong to which request.

    A batch is fed to kernels as one matrix whose rows are the concatenation of
    the participating requests. That only works if every request owns a run of
    rows that nothing else owns — otherwise two requests share an activation and
    one of them silently reads the other's value. It is the same failure as a
    paged kernel reading the wrong block, with the same property: the result is a
    number, not an error.
    """

    var req: InlineArray[Int, MAX_BATCH]
    var off: InlineArray[Int, MAX_BATCH]
    var cnt: InlineArray[Int, MAX_BATCH]
    var live: InlineArray[Int, MAX_BATCH]

    def __init__(out self):
        self.req = InlineArray[Int, MAX_BATCH](fill=NO_REQUEST)
        self.off = InlineArray[Int, MAX_BATCH](fill=0)
        self.cnt = InlineArray[Int, MAX_BATCH](fill=0)
        self.live = InlineArray[Int, MAX_BATCH](fill=0)

    def n_live(self) -> Int:
        var n = 0
        for i in range(MAX_BATCH):
            if self.live[i] == 1:
                n += 1
        return n

    def slot_of(self, request: Int) -> Int:
        """Slot holding `request`, or NO_REQUEST when it is not batched."""
        for i in range(MAX_BATCH):
            if self.live[i] == 1 and self.req[i] == request:
                return i
        return NO_REQUEST

    def total_tokens(self) -> Int:
        """Rows somebody owns. A layer only needs this many rows of scratch."""
        var total = 0
        for i in range(MAX_BATCH):
            if self.live[i] == 1:
                total += self.cnt[i]
        return total

    def place(self, count: Int) -> Int:
        """Lowest row nothing owns, first fit by increasing row."""
        var rows = InlineArray[Int, MAX_BATCH](fill=0)
        var ends = InlineArray[Int, MAX_BATCH](fill=0)
        var n = 0
        for i in range(MAX_BATCH):
            if self.live[i] != 1:
                continue
            var start = self.off[i]
            var stop = start + self.cnt[i]
            var k = n
            while k > 0 and rows[k - 1] > start:
                rows[k] = rows[k - 1]
                ends[k] = ends[k - 1]
                k -= 1
            rows[k] = start
            ends[k] = stop
            n += 1
        var pos = 0
        var j = 0
        while j < n:
            if rows[j] - pos >= count:
                return pos
            if ends[j] > pos:
                pos = ends[j]
            j += 1
        return pos

    def add(mut self, request: Int, count: Int) raises AlofaError -> Int:
        """Reserve `count` contiguous rows for `request`; returns its row."""
        if count <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "a request must bring at least one token"
            )
        if self.slot_of(request) != NO_REQUEST:
            # Counting one request twice would hand it two runs of rows, and a
            # layer that reads one of them would read a partially written batch.
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is already in the batch"
            )
        var slot = NO_SLOT
        for i in range(MAX_BATCH):
            if self.live[i] == 0:
                slot = i
                break
        if slot == NO_SLOT:
            raise AlofaError(ERR_CAPACITY, "batch is full")
        var at = self.place(count)
        self.req[slot] = request
        self.off[slot] = at
        self.cnt[slot] = count
        self.live[slot] = 1
        return at

    def remove(mut self, request: Int) raises AlofaError -> Int:
        """Drop a request; returns the row it had, so callers can name it."""
        var slot = self.slot_of(request)
        if slot == NO_REQUEST:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is not in the batch"
            )
        self.live[slot] = 0
        self.req[slot] = NO_REQUEST
        return self.off[slot]

    def clear(mut self):
        for i in range(MAX_BATCH):
            self.live[i] = 0
            self.req[i] = NO_REQUEST

    def defects(self) -> Int:
        """Violations re-derived from the slot records: overlap, and dupes."""
        var bad = 0
        for i in range(MAX_BATCH):
            if self.live[i] != 1:
                continue
            if self.cnt[i] <= 0:
                bad += 1
            if self.off[i] < 0:
                bad += 1
            for j in range(i + 1, MAX_BATCH):
                if self.live[j] != 1:
                    continue
                var ai = self.off[i]
                var bi = ai + self.cnt[i]
                var aj = self.off[j]
                var bj = aj + self.cnt[j]
                if ai < bj and aj < bi:
                    bad += 1
        return bad

    def digest(self) -> Int:
        var value = 0
        for i in range(MAX_BATCH):
            value = (
                value * BATCH_MUL
                + self.req[i]
                + self.off[i] * 7
                + self.cnt[i] * 11
                + self.live[i] * 13
            ) % BATCH_MOD
        return value
