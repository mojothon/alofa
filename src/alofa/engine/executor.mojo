"""The busy loop: turn a set of requests into one forward, over and over.

Two things live here, and the reason both are in one file is that they are one
decision. The row layout says which request owns which rows; the forward then
reads that layout and nothing else. Split them and the forward would need a
convention ("the answer for request r is at row n_r") that no caller has heard
of, and a wrong convention there is invisible: every row is a plausible tensor.

**What batching means here.** Rows of different requests sit side by side in one
activation matrix, so the projections — the bulk of the arithmetic — run once
over all of them. Attention is the operation that reads across rows, and it is
the one that must not: every row is given the base address of its **own**
sequence's keys and values and the index of its own last visible key. A row
therefore has no address for another request's history at all. That is what
stops request 3 from attending to request 0 — not a mask that could be dropped,
but an address it never receives.

**Why the scratch is a `BatchPool` and not a `List`.** Everything a step needs is
borrowed from a pool whose capacity was fixed when this executor was built, and
given back before the step ends. The alternative — allocate what this batch
happens to need — is unbounded by construction: a batch one request larger than
any batch tested would then fail in production, which is the failure P2 exists
to prevent. When a batch does not fit, `plan` says so by name.

**What this file does not claim.** The forward is not allocation-free in the
sense of never calling the allocator: `rows_view` builds its own small shape
metadata, and the kernels take views by value. What is claimed is that no
allocation here *grows with the batch* — every buffer is a borrow from a pool
that was sized once — and that this module constructs no container of its own.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_batch_executor.mojo
"""

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    ERR_SHAPE_MISMATCH,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import TensorView, f32_data, rows_view, view3
from alofa.engine.batch import (
    MAX_BATCH,
    MAX_HANDLES,
    POOL_ALIGN,
    BatchPool,
    BatchSlots,
    ceil_bytes,
)
from alofa.kernels.cpu.paged import PagedTable, paged_attention, paged_scatter
from alofa.kernels.cpu.segments import PtrPtr, row_rope, select_rows
from alofa.runtime.kv import (
    MAX_BLOCKS,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ_TOKENS,
)
from alofa.model.arch.qwen import (
    BACKEND_SCALAR,
    EMBED,
    FINAL_NORM,
    OUTPUT,
    Q4_SLOTS_PER_LAYER,
    Q4_SLOT_DOWN,
    Q4_SLOT_GATE,
    Q4_SLOT_K,
    Q4_SLOT_O,
    Q4_SLOT_Q,
    Q4_SLOT_UP,
    Q4_SLOT_V,
    Q4_NO_SLOT,
    QwenForward,
    add_k,

    rmsnorm_k,
    swiglu_k,
)

comptime BATCH_MUL = 1000003
comptime BATCH_MOD = 2147483647
comptime MAX_ROWS = 64
comptime MAX_LAYERS = 32
comptime MAX_GEN = 32
comptime NO_ROW = -1
comptime NO_TOKEN = -1

# A block's token count. The pool, the room and this executor have to agree on
# it, because a page table is a list of block ids — a disagreement would put a
# token in a slot nobody reads.
comptime DEFAULT_BLOCK_SIZE = 16

comptime IntPtr = Pointer[Int, MutUntrackedOrigin]
comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]

# One page table per request, in the shape the paged kernels read.
comptime KvPageTable = PagedTable[MAX_BLOCKS_PER_SEQ]


def int_map(raw: RawPtr) -> IntPtr:
    """Read a `RawPtr` as `Int` elements — the one cast a caller has to make.

    Row maps live in an arena rather than in an `InlineArray` because the kernels
    take untracked pointers: an array's pointer carries the array's origin, and
    the origin of an arena allocation is already nobody's.
    """
    return raw.unsafe_bitcast[Int]()


def base_map(raw: RawPtr) -> PtrPtr:
    """Read a `RawPtr` as base addresses, one per row."""
    return raw.unsafe_bitcast[RawPtr]()


def block_view(base: RawPtr, block_size: Int, kv_dim: Int) raises AlofaError -> TensorView:
    """Every layer's blocks as one `[MAX_LAYERS * MAX_BLOCKS, size, kv_dim]` view.

    One view for all the layers, with the layer folded into the block id, is
    what a block *is*: it holds a slot of K/V per layer, so a request's block
    table names the same blocks in every layer. It also means the view is built
    once, at construction — the busy loop never builds a shape, and a list
    built per step would be an allocation in the one place this layer promised
    not to allocate.
    """
    return view3(base, MAX_LAYERS * MAX_BLOCKS, block_size, kv_dim)


def scratch_bytes(
    max_rows: Int,
    max_pos: Int,
    hidden: Int,
    inter: Int,
    kv_dim: Int,
    vocab: Int,
) -> Int:
    """Bytes one step can ever need, sized once so the loop never grows."""
    var total = 6 * max_rows * hidden * 4
    total += 3 * max_rows * kv_dim * 4
    total += 3 * max_rows * inter * 4
    total += max_rows * max_pos * 4
    total += MAX_BATCH * hidden * 4
    total += MAX_BATCH * vocab * 4
    return ceil_bytes(total + 16 * POOL_ALIGN)


struct BatchExecutor:
    """A fixed set of resident requests, stepped together.

    Nothing here owns weights: the shapes are copied in at construction and the
    model is passed to each forward. That keeps this layer honest about what it
    is — orchestration — and lets a test build one with toy dimensions and check
    the row arithmetic without loading anything.
    """

    # Shape. Copied, never owned.
    var hidden: Int
    var inter: Int
    var kv_dim: Int
    var block_size: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var vocab: Int
    var eps: Float32
    var max_pos: Int
    var max_rows: Int

    # Assembly.
    var slots: BatchSlots
    var pool: BatchPool

    # Storage, allocated once and never grown.
    var scratch: Arena
    var kv_store: Arena
    var maps: Arena

    # One block pool per layer, per side: `[MAX_BLOCKS, block_size, kv_dim]`.
    # A request's history is a list of runs into it, not a region of its own,
    # which is what lets two requests read the same prefix.
    var k_cache: TensorView
    var v_cache: TensorView
    var row_pos: IntPtr
    var row_upto: IntPtr
    var row_tok: IntPtr
    var row_req: IntPtr
    var row_start: IntPtr
    var out_rows: IntPtr
    var chosen: IntPtr

    # Per request state, indexed by slot.
    var req: InlineArray[Int, MAX_BATCH]
    var hist: InlineArray[Int, MAX_BATCH]
    # How long the prompt *is*, as opposed to how much of it is queued. The two
    # differ whenever a scheduler hands the prompt over in slices, and an empty
    # queue at a slice boundary looks exactly like an empty queue at the end of a
    # prompt — which is how a generated token ends up spliced into the middle of
    # one.
    var n_prompt: InlineArray[Int, MAX_BATCH]
    var n_pending: InlineArray[Int, MAX_BATCH]
    var pending: InlineArray[Int, MAX_BATCH * MAX_ROWS]
    var budget: InlineArray[Int, MAX_BATCH]
    var served: InlineArray[Int, MAX_BATCH]
    var live: InlineArray[Int, MAX_BATCH]
    var out: InlineArray[Int, MAX_BATCH * MAX_GEN]
    var n_out: InlineArray[Int, MAX_BATCH]

    # Page tables, one per request: which blocks hold its history, and from
    # which slot inside each. The room decides whose blocks they are and hands
    # them over once; a step only reads them. A request with an empty table has
    # no blocks, and asking the cache for its history anyway is what refuses
    # the step rather than reading uninitialised memory.
    var page_ids: InlineArray[Int, MAX_BATCH * MAX_BLOCKS_PER_SEQ]
    var page_starts: InlineArray[Int, MAX_BATCH * MAX_BLOCKS_PER_SEQ]
    var page_lens: InlineArray[Int, MAX_BATCH * MAX_BLOCKS_PER_SEQ]
    var page_n: InlineArray[Int, MAX_BATCH]

    # Step state.
    var rows: Int
    var handles: InlineArray[Int, MAX_HANDLES]
    var n_handles: Int
    var n_steps: Int
    var defects: Int

    def __init__(
        out self,
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
        block_size: Int = DEFAULT_BLOCK_SIZE,
    ) raises AlofaError:
        if hidden <= 0 or inter <= 0 or vocab <= 0 or max_pos <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "executor needs positive dimensions", ""
            )
        if n_layers <= 0 or n_layers > MAX_LAYERS:
            raise AlofaError(
                ERR_UNSUPPORTED, "layer count is outside what was compiled in", ""
            )
        if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            raise AlofaError(
                ERR_UNSUPPORTED, "n_kv_heads must divide n_heads", ""
            )
        if kv_dim != n_kv_heads * head_dim:
            raise AlofaError(
                ERR_SHAPE_MISMATCH, "kv_dim is not n_kv_heads * head_dim", ""
            )
        if max_rows <= 0 or max_rows > MAX_ROWS:
            raise AlofaError(
                ERR_UNSUPPORTED, "row count is outside what was compiled in", ""
            )

        self.hidden = hidden
        self.inter = inter
        self.kv_dim = kv_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.vocab = vocab
        self.eps = eps
        self.max_pos = max_pos
        self.max_rows = max_rows
        if block_size <= 0 or block_size > MAX_SEQ_TOKENS:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "block size must fit inside a sequence", ""
            )
        self.block_size = block_size

        self.scratch = Arena(
            scratch_bytes(max_rows, max_pos, hidden, inter, kv_dim, vocab)
        )
        self.pool = BatchPool(
            self.scratch.alloc(
                scratch_bytes(max_rows, max_pos, hidden, inter, kv_dim, vocab)
            ),
            scratch_bytes(max_rows, max_pos, hidden, inter, kv_dim, vocab),
        )

        # The KV cache is a pool of blocks, one per layer and side, and a
        # request's history is a list of runs into it. Not one region per
        # request: a shared prefix is the same bytes read twice, and a region
        # per request cannot express that.
        #
        # Whose blocks they are is decided elsewhere (`engine/kv_room.mojo`);
        # this layer is handed a table and reads it. The addresses are recorded
        # once, here. A step only ever *reads* one of these pointers; it never
        # computes one and never allocates one — and the paged kernels are what
        # turn a block id and a slot into an address, so the arithmetic that
        # could be wrong about a shared prefix lives in exactly one module.
        var block_bytes = MAX_LAYERS * MAX_BLOCKS * block_size * kv_dim * 4
        self.kv_store = Arena(2 * block_bytes + 16 * POOL_ALIGN)
        # Four row maps, three per-request maps. Sized exactly: an arena that is
        # short does not fail loudly, it hands out an address that overlaps the
        # next map, and the symptom is a row pointing at another request.
        self.maps = Arena(
            MAX_ROWS * 8 * 4 + MAX_BATCH * 8 * 3 + 16 * POOL_ALIGN
        )
        self.k_cache = block_view(self.kv_store.alloc(block_bytes), block_size, kv_dim)
        self.v_cache = block_view(self.kv_store.alloc(block_bytes), block_size, kv_dim)
        self.row_pos = int_map(self.maps.alloc(MAX_ROWS * 8))
        self.row_upto = int_map(self.maps.alloc(MAX_ROWS * 8))
        self.row_tok = int_map(self.maps.alloc(MAX_ROWS * 8))
        self.row_req = int_map(self.maps.alloc(MAX_ROWS * 8))
        self.row_start = int_map(self.maps.alloc(MAX_BATCH * 8))
        self.out_rows = int_map(self.maps.alloc(MAX_BATCH * 8))
        self.chosen = int_map(self.maps.alloc(MAX_BATCH * 8))

        self.req = InlineArray[Int, MAX_BATCH](fill=0)
        self.hist = InlineArray[Int, MAX_BATCH](fill=0)
        self.n_prompt = InlineArray[Int, MAX_BATCH](fill=0)
        self.n_pending = InlineArray[Int, MAX_BATCH](fill=0)
        self.pending = InlineArray[Int, MAX_BATCH * MAX_ROWS](fill=0)
        self.budget = InlineArray[Int, MAX_BATCH](fill=0)
        self.served = InlineArray[Int, MAX_BATCH](fill=0)
        self.live = InlineArray[Int, MAX_BATCH](fill=0)
        self.out = InlineArray[Int, MAX_BATCH * MAX_GEN](fill=0)
        self.n_out = InlineArray[Int, MAX_BATCH](fill=0)
        self.page_ids = InlineArray[Int, MAX_BATCH * MAX_BLOCKS_PER_SEQ](fill=0)
        self.page_starts = InlineArray[Int, MAX_BATCH * MAX_BLOCKS_PER_SEQ](fill=0)
        self.page_lens = InlineArray[Int, MAX_BATCH * MAX_BLOCKS_PER_SEQ](fill=0)
        self.page_n = InlineArray[Int, MAX_BATCH](fill=0)

        self.rows = 0
        self.handles = InlineArray[Int, MAX_HANDLES](fill=0)
        self.n_handles = 0
        self.n_steps = 0
        self.defects = 0
        self.slots = BatchSlots()
        for i in range(MAX_BATCH):
            self.row_start[unsafe_offset=i] = NO_ROW
            self.chosen[unsafe_offset=i] = NO_TOKEN

    def slot_of(self, request: Int) -> Int:
        """Slot holding `request`, or `NO_REQUEST` if it is not resident."""
        for i in range(MAX_BATCH):
            if self.live[i] == 1 and self.req[i] == request:
                return i
        return -1

    def set_page_table(
        mut self, request: Int, read table: KvPageTable
    ) raises AlofaError:
        """Hand a resident request the blocks its history lives in.

        Copied, not referenced: the room that chose those blocks is free to
        move them afterwards, and this layer must not be able to notice. A
        request that is not resident has no slot to copy into — handing blocks
        to a request this layer has never heard of would be how a step ends up
        reading blocks nobody filled.
        """
        var slot = self.slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "no resident request has that id", ""
            )
        if table.n > MAX_BLOCKS_PER_SEQ:
            raise AlofaError(ERR_CAPACITY, "a sequence holds fewer blocks", "")
        var base = slot * MAX_BLOCKS_PER_SEQ
        for i in range(table.n):
            self.page_ids[base + i] = table.block_at(i)
            self.page_starts[base + i] = table.start_at(i)
            self.page_lens[base + i] = table.length_at(i)
        self.page_n[slot] = table.n

    def n_page_tokens(self, slot: Int) -> Int:
        """How many positions this request's table covers.

        Deliberately a sum over the runs rather than a remembered count: the
        two drift apart the moment a block is shared, and the drift is exactly
        what the comparison against `hist` in `forward` is there to catch.
        """
        var total = 0
        var base = slot * MAX_BLOCKS_PER_SEQ
        for i in range(self.page_n[slot]):
            total += self.page_lens[base + i]
        return total

    def _table_of(
        self, slot: Int, layer: Int, upto: Int
    ) raises AlofaError -> KvPageTable:
        """This request's table, cut down to `upto` positions — no allocation.

        Cut, not used whole: the room is free to hand over a block it reserved
        for tokens that have not arrived yet (a sliced prompt arrives in pieces,
        and a block is the unit it reserves in), and attention over a longer
        table than the history would read positions nobody has written. The
        caller has already established that the table covers `upto`.
        """
        var table = KvPageTable(self.block_size)
        var base = slot * MAX_BLOCKS_PER_SEQ
        var left = upto
        for i in range(self.page_n[slot]):
            if left <= 0:
                break
            var length = self.page_lens[base + i]
            if length > left:
                length = left
            _ = table.push(
                layer * MAX_BLOCKS + self.page_ids[base + i],
                self.page_starts[base + i],
                length,
            )
            left -= length
        return table^

    def add(
        mut self,
        request: Int,
        tokens: IntPtr,
        n: Int,
        budget: Int,
        prompt_len: Int = -1,
        matched: Int = 0,
    ) raises AlofaError:
        """Take a request: some of its prompt is queued, nothing is planned yet.

        `prompt_len` is how long the prompt is in total, which is not always how
        much of it is being handed over now. A scheduler that prefills in slices
        hands over `[0, k)` first and the rest later, and the executor has to
        know the difference: an empty queue means "the prompt is finished" only
        if the prompt really was finished.

        `matched` is how much of the prompt already has its K/V: another request
        computed the same prefix, the room kept the blocks, and the bytes are
        already in the cache under this request's own page table. Those tokens
        are queued but never computed — a shared prefix is worth the blocks *and*
        the arithmetic, and only the second of those is a saving you can time.

        At least one token is always computed: the last one. Its logits are the
        first generated token, so a prompt that is *entirely* cached still runs
        one row — a request with no rows has nothing to speak from.
        """
        if n <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "a request needs a prompt", "")
        if n > self.max_rows:
            raise AlofaError(
                ERR_CAPACITY, "a prompt longer than one step cannot be queued", ""
            )
        var total = prompt_len if prompt_len > 0 else n
        if n > total:
            raise AlofaError(
                ERR_OUT_OF_RANGE, "more prompt queued than the request declared", ""
            )
        if self.slot_of(request) >= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is already resident", ""
            )
        # A cached prefix longer than the prompt is nonsense, and a cached prefix
        # as long as the prompt is refused *down* to one row rather than taken at
        # face value: no rows means no logits means no first token.
        if matched < 0 or matched > n:
            raise AlofaError(
                ERR_OUT_OF_RANGE, "a cached prefix cannot be longer than the prompt", ""
            )
        var cached = matched if matched < n else n - 1
        var slot = -1
        for i in range(MAX_BATCH):
            if self.live[i] == 0:
                slot = i
                break
        if slot < 0:
            raise AlofaError(ERR_CAPACITY, "the batch is full", "")
        self.req[slot] = request
        self.live[slot] = 1
        # History starts at the cached length, not at zero: the positions before
        # it are occupied by bytes somebody else computed, which is the whole
        # point of having matched them.
        self.hist[slot] = cached
        self.n_prompt[slot] = total
        self.n_pending[slot] = n - cached
        self.budget[slot] = budget
        self.served[slot] = 0
        self.n_out[slot] = 0
        for i in range(n):
            var token = tokens[unsafe_offset=i]
            if token < 0 or token >= self.vocab:
                raise AlofaError(
                    ERR_OUT_OF_RANGE, "token id is outside the vocabulary", ""
                )
            if i < cached:
                continue
            self.pending[slot * MAX_ROWS + (i - cached)] = token

    def feed(mut self, request: Int, token: Int) raises AlofaError:
        """Queue one more token for a resident request."""
        var slot = self.slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is not resident", ""
            )
        if token < 0 or token >= self.vocab:
            raise AlofaError(
                ERR_OUT_OF_RANGE, "token id is outside the vocabulary", ""
            )
        if self.n_pending[slot] >= self.max_rows:
            raise AlofaError(ERR_CAPACITY, "queue is full", "")
        # A request cannot be handed more prompt than it declared. Feeding past
        # the end is how a mis-slice reaches the model as a plausible row: the
        # token is valid, the position is valid, and the sequence is not.
        if self.hist[slot] + self.n_pending[slot] + 1 > self.n_prompt[slot]:
            raise AlofaError(
                ERR_OUT_OF_RANGE, "more prompt queued than the request declared", ""
            )
        self.pending[slot * MAX_ROWS + self.n_pending[slot]] = token
        self.n_pending[slot] += 1

    def drop(mut self, request: Int) raises AlofaError:
        """Give a slot back. Its rows and its history go with it."""
        var slot = self.slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is not resident", ""
            )
        self.live[slot] = 0
        self.n_prompt[slot] = 0
        self.n_pending[slot] = 0
        self.hist[slot] = 0
        self.served[slot] = 0
        self.budget[slot] = 0
        self.n_out[slot] = 0
        self.req[slot] = 0
        # The blocks go too: a table left behind would be a context this layer
        # still thinks it can read, belonging to a request that no longer
        # exists. Whose blocks they are is the room's to decide — this only
        # forgets the address.
        self.page_n[slot] = 0

    def pending_of(self, request: Int) raises AlofaError -> Int:
        var slot = self.slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is not resident", ""
            )
        return self.n_pending[slot]

    def history_of(self, request: Int) raises AlofaError -> Int:
        var slot = self.slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is not resident", ""
            )
        return self.hist[slot]

    def plan(mut self) raises AlofaError -> Int:
        """Decide this step's rows, and borrow the memory to run them.

        Rows are handed out in slot order and each request gets one run, so a
        request's rows are always consecutive: that is what lets its keys and
        values be copied in one piece and lets attention be given a single base
        address. A request whose queued tokens do not fit simply waits — a
        shorter step now is not a wrong step.
        """
        if self.n_handles != 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "the previous step was not finished", ""
            )
        self.slots.clear()
        self.rows = 0
        var seen = 0
        for i in range(MAX_BATCH):
            self.served[i] = 0
            self.row_start[unsafe_offset=i] = NO_ROW
            if self.live[i] != 1:
                continue
            var room = self.max_rows - self.slots.total_tokens()
            var n = self.n_pending[i]
            if n > room:
                n = room
            if n <= 0:
                continue
            if self.hist[i] + n > self.max_pos:
                raise AlofaError(
                    ERR_CAPACITY, "history would outgrow its region", ""
                )
            var at = self.slots.add(self.req[i], n)
            self.served[i] = n
            self.row_start[unsafe_offset=i] = at
            # The answer rows are compacted: request k's logits are row k of the
            # output, whatever hole its history left in the row block.
            self.out_rows[unsafe_offset=seen] = at + n - 1
            seen += 1
            for j in range(n):
                var row = at + j
                self.row_req[unsafe_offset=row] = i
                self.row_tok[unsafe_offset=row] = self.pending[i * MAX_ROWS + j]
                var pos = self.hist[i] + j
                self.row_pos[unsafe_offset=row] = pos
                # The key base is filled per layer, because it is the address of
                # a request's history *in that layer*. Leaving it stale here is
                # not an option the forward can take by accident: it is written
                # before every attention call and read nowhere else.
                self.row_upto[unsafe_offset=row] = pos

        self.rows = self.slots.total_tokens()
        if self.rows <= 0:
            return 0
        self._borrow(self.rows, self.slots.n_live())
        return self.rows

    def _borrow(mut self, rows: Int, out_rows: Int) raises AlofaError:
        """Take every buffer the forward needs, in one place.

        The sizes are what `rows` needs now, not what `max_rows` would need, so a
        small batch touches less of the pool. The borrows are recorded because
        `finish` has to give back exactly these handles.
        """
        var h = self.hidden
        var kv_dim = self.kv_dim
        self._take(rows * h * 4)
        self._take(rows * h * 4)
        self._take(rows * h * 4)
        self._take(rows * kv_dim * 4)
        self._take(rows * kv_dim * 4)
        self._take(rows * h * 4)
        self._take(rows * kv_dim * 4)
        self._take(rows * h * 4)
        self._take(rows * h * 4)
        self._take(rows * self.inter * 4)
        self._take(rows * self.inter * 4)
        self._take(rows * self.inter * 4)
        self._take(rows * self.max_pos * 4)
        self._take(out_rows * h * 4)
        self._take(out_rows * self.vocab * 4)

    def _take(mut self, n_bytes: Int) raises AlofaError:
        var handle = self.pool.borrow(ceil_bytes(n_bytes))
        self.handles[self.n_handles] = handle
        self.n_handles += 1

    def raw_of(self, index: Int) raises AlofaError -> RawPtr:
        """The bytes behind a borrow recorded by `_borrow`."""
        if index < 0 or index >= self.n_handles:
            raise AlofaError(
                ERR_OUT_OF_RANGE, "no buffer was borrowed at that index", ""
            )
        return self.pool.base.unsafe_offset(
            self.pool.offset_of(self.handles[index])
        )

    def finish(mut self) raises AlofaError:
        """Give the step's buffers back.

        Called in the same place whether the forward succeeded or not: a step
        that ends in an error must not leave the pool short of bytes it will
        need next time.
        """
        for i in range(self.n_handles):
            _ = self.pool.release(self.handles[i])
        self.n_handles = 0
        self.n_steps += 1
        if self.pool.n_live() != 0:
            self.defects += 1

    def forward[backend: Int = BACKEND_SCALAR](
        mut self, mut model: QwenForward
    ) raises AlofaError -> Int:
        """One pass over the planned rows, producing one logits row per request.

        Everything except attention is a per-row kernel, so it runs over the whole
        packed matrix unchanged. Attention gets, per row, the address of that
        row's own history and the index of its own last visible key; rotary
        embedding gets that row's own position. Those three inputs are the whole
        difference between a batch and a single sequence, and this method has no
        other knowledge of requests.
        """
        var t = self.rows
        if t <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "nothing was planned", "")
        if self.n_handles != 15:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "the step was not borrowed for", ""
            )
        var h = self.hidden
        var kv_dim = self.kv_dim
        var hidden = rows_view(self.raw_of(0), t, h)
        var normed = rows_view(self.raw_of(1), t, h)
        var q = rows_view(self.raw_of(2), t, h)
        var k = rows_view(self.raw_of(3), t, kv_dim)
        var v = rows_view(self.raw_of(4), t, kv_dim)
        var q_rot = rows_view(self.raw_of(5), t, h)
        var k_rot = rows_view(self.raw_of(6), t, kv_dim)
        var attn = rows_view(self.raw_of(7), t, h)
        var proj = rows_view(self.raw_of(8), t, h)
        var gate = rows_view(self.raw_of(9), t, self.inter)
        var up = rows_view(self.raw_of(10), t, self.inter)
        var act = rows_view(self.raw_of(11), t, self.inter)
        var scores = rows_view(self.raw_of(12), t, self.max_pos)

        # Embedding gather, one row per token.
        var embed = model.params.view(EMBED)
        var pe = f32_data(embed)
        var ph = f32_data(hidden)
        for row in range(t):
            var token = self.row_tok[unsafe_offset=row]
            if token < 0 or token >= self.vocab:
                raise AlofaError(
                    ERR_OUT_OF_RANGE, "token id is outside the vocabulary", ""
                )
            var src = token * h
            var dst = row * h
            for j in range(h):
                ph[unsafe_offset=dst + j] = pe[unsafe_offset=src + j]

        for layer in range(self.n_layers):
            var p = "model.layers." + String(layer) + "."

            rmsnorm_k[backend](
                normed, hidden, model.params.view(p + "input_layernorm.weight"), self.eps
            )
            model.project[backend](
                q,
                normed,
                p + "self_attn.q_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_Q,
                model.params.view(p + "self_attn.q_proj.bias"),
                True,
            )
            model.project[backend](
                k,
                normed,
                p + "self_attn.k_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_K,
                model.params.view(p + "self_attn.k_proj.bias"),
                True,
            )
            model.project[backend](
                v,
                normed,
                p + "self_attn.v_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_V,
                model.params.view(p + "self_attn.v_proj.bias"),
                True,
            )
            row_rope(q_rot, k_rot, q, k, model.cos, model.sin, self.head_dim, self.row_pos)

            # Append this step's keys and values *through the page table*: the
            # write and the read ask the same table the same question, which is
            # what stops a request from filling the block it would have kept had
            # it not shared a prefix.
            #
            # The table has to cover exactly `hist + n`. Fewer and the write is
            # refused; more and attention would read positions nobody wrote —
            # and "more" is not a hypothetical, it is what a room that over-
            # allocated for a sliced prompt looks like.
            for i in range(MAX_BATCH):
                if self.served[i] == 0:
                    continue
                var n = self.served[i]
                var at = self.row_start[unsafe_offset=i]
                var upto = self.hist[i] + n
                if self.n_page_tokens(i) < upto:
                    raise AlofaError(
                        ERR_CAPACITY,
                        "the request has fewer blocks than it has history",
                        "",
                    )
                var table = self._table_of(i, layer, upto)
                paged_scatter(
                    self.k_cache, table, self.hist[i], k_rot.slice_dim(0, at, n)
                )
                paged_scatter(
                    self.v_cache, table, self.hist[i], v.slice_dim(0, at, n)
                )

            # One call per request, each with its own table: a row attends to
            # its own history because it is given that history's address, not
            # because a mask says it may not look further.
            for i in range(MAX_BATCH):
                if self.served[i] == 0:
                    continue
                var n = self.served[i]
                var at = self.row_start[unsafe_offset=i]
                var table = self._table_of(i, layer, self.hist[i] + n)
                paged_attention(
                    attn.slice_dim(0, at, n),
                    q_rot.slice_dim(0, at, n),
                    self.k_cache,
                    self.v_cache,
                    table,
                    scores,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                )
            model.project[backend](
                proj,
                attn,
                p + "self_attn.o_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_O,
                proj,
                False,
            )
            add_k[backend](hidden, hidden, proj)

            rmsnorm_k[backend](
                normed,
                hidden,
                model.params.view(p + "post_attention_layernorm.weight"),
                self.eps,
            )
            model.project[backend](
                gate,
                normed,
                p + "mlp.gate_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_GATE,
                gate,
                False,
            )
            model.project[backend](
                up,
                normed,
                p + "mlp.up_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_UP,
                up,
                False,
            )
            swiglu_k[backend](act, gate, up)
            model.project[backend](
                proj,
                act,
                p + "mlp.down_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_DOWN,
                proj,
                False,
            )
            add_k[backend](hidden, hidden, proj)

        # One answer per request, and those rows are not adjacent: gather them,
        # then the final norm and the output projection run over the small matrix.
        var served = self.slots.n_live()
        var which = self.out_rows
        var tail = rows_view(self.raw_of(13), served, h)
        select_rows(tail, hidden, which, served)
        var tail_normed = rows_view(self.raw_of(1), served, h)
        rmsnorm_k[backend](
            tail_normed, tail, model.params.view(FINAL_NORM), self.eps
        )
        var logits = rows_view(self.raw_of(14), served, self.vocab)
        model.project[backend](logits, tail_normed, OUTPUT, Q4_NO_SLOT, logits, False)
        return served

    def logits_of(self, request: Int) raises AlofaError -> F32Ptr:
        """A request's logits row, valid until `finish`.

        Valid until `finish` and not longer: the row lives in a buffer the pool
        will hand to someone else, and a pointer that outlives the step is the
        failure this whole layer is arranged to avoid.
        """
        var slot = self.slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is not resident", ""
            )
        var row = NO_ROW
        var seen = 0
        for i in range(MAX_BATCH):
            if self.served[i] == 0:
                continue
            if i == slot:
                row = seen
            seen += 1
        if row < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request was not served this step", ""
            )
        # `unsafe_offset` counts elements of the pointee, not bytes: this pointer
        # is `Float32`, so a row is `vocab` floats and nothing else. Getting the
        # unit wrong here is silent for row 0 and out of bounds after it.
        var base = self.raw_of(14)
        return base.unsafe_bitcast[Float32]().unsafe_offset(row * self.vocab)

    def advance(mut self, chosen: IntPtr) raises AlofaError:
        """Consume this step: histories grow, budgets fall, finished leave."""
        for i in range(MAX_BATCH):
            if self.live[i] != 1 or self.served[i] == 0:
                continue
            var n = self.served[i]
            self.hist[i] += n
            var left = self.n_pending[i] - n
            for j in range(left):
                self.pending[i * MAX_ROWS + j] = self.pending[
                    i * MAX_ROWS + n + j
                ]
            self.n_pending[i] = left
            # A request whose prompt is not finished owes no answer: the step
            # that ends the prompt is the step that gets to choose. Feeding the
            # argmax back here would splice a generated token into the middle of
            # a prompt that was split across two steps. "Not finished" means the
            # queue is empty *and* the whole prompt has been seen — a scheduler
            # that prefills in slices empties the queue at every boundary, and an
            # empty queue alone cannot tell the two apart.
            if left > 0 or self.hist[i] < self.n_prompt[i]:
                continue
            var token = chosen[unsafe_offset=i]
            if token == NO_TOKEN:
                continue
            if self.n_out[i] < MAX_GEN:
                self.out[i * MAX_GEN + self.n_out[i]] = token
                self.n_out[i] += 1
            self.budget[i] -= 1
            # A request is *not* evicted for spending its budget. The token that
            # spent it is already written into `out`, and a request that leaves
            # on its own takes its own transcript with it — the caller would be
            # left asking a slot that no longer exists for the last token it was
            # owed. It stops being fed instead, and dropping it is left to the
            # caller, which is the only thing that knows about stop strings,
            # cancellation and "enough".
            if self.budget[i] <= 0:
                continue
            if self.n_pending[i] >= self.max_rows:
                raise AlofaError(ERR_CAPACITY, "queue is full", "")
            self.pending[i * MAX_ROWS + self.n_pending[i]] = token
            self.n_pending[i] += 1

    def generated(self, request: Int, dest: IntPtr) raises AlofaError -> Int:
        """Copy a request's generated tokens out; returns how many."""
        var slot = self.slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "request is not resident", ""
            )
        var n = self.n_out[slot]
        for i in range(n):
            dest[unsafe_offset=i] = self.out[slot * MAX_GEN + i]
        return n

    def generate[backend: Int = BACKEND_SCALAR](
        mut self, mut model: QwenForward, steps: Int
    ) raises AlofaError -> Int:
        """The busy loop: plan, forward, take the argmax, feed it back.

        Greedy decoding, because the claim being tested is that a batch and a
        serial run choose the same token — sampling would put a random number
        between the two and hide the difference.
        """
        var chosen = self.chosen
        var done = 0
        for step in range(steps):
            var rows = self.plan()
            if rows == 0:
                break
            var served = self.forward[backend](model)
            for i in range(MAX_BATCH):
                chosen[unsafe_offset=i] = NO_TOKEN
                if self.live[i] != 1 or self.served[i] == 0:
                    continue
                chosen[unsafe_offset=i] = model.argmax(
                    self.logits_of(self.req[i])
                )
            self.advance(chosen)
            self.finish()
            done += 1
        return done

    def live_count(self) -> Int:
        var n = 0
        for i in range(MAX_BATCH):
            n += self.live[i]
        return n

    def digest(self) -> Int:
        """A number this layer re-derives, not one it remembers.

        Mixed over the per-request state and the two lower structures' own
        digests, so a fixture can ask for the state without being told which
        field to trust.
        """
        # Reduced at every step, because this number is compared against a
        # reference that cannot overflow: a product left unreduced would wrap
        # here and not there, and the first line of every fixture would differ
        # for a reason that has nothing to do with placement.
        var acc = BATCH_MUL * 7 + self.rows
        for i in range(MAX_BATCH):
            acc = (acc * BATCH_MUL + self.req[i] + 1) % BATCH_MOD
            acc = (acc * BATCH_MUL + self.hist[i] + 1) % BATCH_MOD
            acc = (acc * BATCH_MUL + self.n_prompt[i] + 1) % BATCH_MOD
            acc = (acc * BATCH_MUL + self.n_pending[i] + 1) % BATCH_MOD
            acc = (acc * BATCH_MUL + self.budget[i] + 2) % BATCH_MOD
            acc = (acc * BATCH_MUL + self.live[i] + 1) % BATCH_MOD
        return (acc * BATCH_MUL + self.slots.digest()) % BATCH_MOD
