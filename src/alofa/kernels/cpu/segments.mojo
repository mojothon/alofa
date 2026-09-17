"""Row-wise kernels for a batch that has been packed into one matrix.

Packing a batch means the token axis is no longer one sequence: row 17 might be
request 3's ninth token while row 18 is request 0's first. Every other kernel in
this directory is per-row already and needs nothing new — `rmsnorm` normalises a
row, `linear` maps a row through a matrix, `swiglu` combines two rows — which is
exactly why those are safe to run over a packed batch unchanged.

Attention is the one operation that reads **across** rows, and rotary embedding
is the one operation that reads **position**. Those two are what stop being
correct when rows from different requests are neighbours:

- `row_rope` takes a position per row instead of assuming row `t` is at position
  `t`. Failing to do this is not a small numerical drift: the wrong rotary phase
  changes a query enough to change the token, and it changes nothing that any
  shape check would notice.
- `segmented_attention` gives every row its own key window. Row `t` sees rows
  `0..upto[t]` **of its own sequence** — the K/V base pointer is per row, so a
  row has no address for another request's history at all. Two requests sharing
  a row would let one read the other's activation, which is the same failure as
  a paged kernel reading the wrong block: it produces a number, not an error.

"Per row" here means "taken from a pointer", not "taken from a list": these
kernels never see a request, only an index and a base address, so the module
below them decides what a request is.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_batch_pool.mojo
"""

from alofa.core.error import (
    ERR_OUT_OF_RANGE,
    ERR_SHAPE_MISMATCH,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.ffi.mem import RawPtr
from alofa.core.tensor import TensorView, f32_data
from alofa.kernels.cpu.scalar import cols_of, expect_matrix, rows_of
from std.math import exp, sqrt

comptime LOWEST_FP32 = Float32(-3.4028234663852886e38)

comptime IntPtr = Pointer[Int, MutUntrackedOrigin]
comptime PtrPtr = Pointer[RawPtr, MutUntrackedOrigin]


def row_rope(
    out_q: TensorView,
    out_k: TensorView,
    q: TensorView,
    k: TensorView,
    cos: TensorView,
    sin: TensorView,
    head_dim: Int,
    positions: IntPtr,
) raises AlofaError:
    """`scalar.rope` with one absolute position per row instead of `t`.

    The arithmetic is copied from the scalar kernel expression for expression,
    including the separate accumulation order of the two halves, because this
    kernel's whole job is to agree with that one bit for bit when every row
    happens to belong to the same request. Nothing here is "equivalent to"
    the scalar result — either both write the same bits or this call fails a
    comparison the caller can see.
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise AlofaError(
            ERR_UNSUPPORTED, "head dimension must be positive and even"
        )
    var tokens = rows_of(q, "q")
    var q_cols = cols_of(q, "q")
    var k_cols = cols_of(k, "k")
    if rows_of(k, "k") != tokens:
        raise AlofaError(
            ERR_SHAPE_MISMATCH, "q and k must have the same token count"
        )
    if q_cols % head_dim != 0 or k_cols % head_dim != 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH, "channel count is not a whole number of heads"
        )
    expect_matrix(out_q, tokens, q_cols, "out_q")
    expect_matrix(out_k, tokens, k_cols, "out_k")
    var cos_rows = rows_of(cos, "cos")
    if cols_of(cos, "cos") != head_dim:
        raise AlofaError(
            ERR_SHAPE_MISMATCH, "cos table must be [tokens, head_dim]"
        )
    expect_matrix(sin, cos_rows, head_dim, "sin")

    var pq = f32_data(q)
    var pk = f32_data(k)
    var pc = f32_data(cos)
    var ps = f32_data(sin)
    var poq = f32_data(out_q)
    var pok = f32_data(out_k)
    var half = head_dim // 2

    for t in range(tokens):
        # The position is per row, and is looked up rather than assumed. A row
        # whose position is past the table is refused: a stale position would
        # otherwise read whatever follows the table and stay plausible.
        var at = positions[unsafe_offset=t]
        if at < 0 or at >= cos_rows:
            raise AlofaError(
                ERR_OUT_OF_RANGE, "rotary position is outside the table"
            )
        var table = at * head_dim
        for head in range(q_cols // head_dim):
            var base = t * q_cols + head * head_dim
            for d in range(half):
                var a = pq[unsafe_offset=base + d]
                var b = pq[unsafe_offset=base + d + half]
                poq[unsafe_offset=base + d] = a * pc[
                    unsafe_offset=table + d
                ] - b * ps[unsafe_offset=table + d]
                poq[unsafe_offset=base + d + half] = b * pc[
                    unsafe_offset=table + d + half
                ] + a * ps[unsafe_offset=table + d + half]
        for head in range(k_cols // head_dim):
            var base = t * k_cols + head * head_dim
            for d in range(half):
                var a = pk[unsafe_offset=base + d]
                var b = pk[unsafe_offset=base + d + half]
                pok[unsafe_offset=base + d] = a * pc[
                    unsafe_offset=table + d
                ] - b * ps[unsafe_offset=table + d]
                pok[unsafe_offset=base + d + half] = b * pc[
                    unsafe_offset=table + d + half
                ] + a * ps[unsafe_offset=table + d + half]


def segmented_attention(
    dst: TensorView,
    q: TensorView,
    k_bases: PtrPtr,
    v_bases: PtrPtr,
    upto: IntPtr,
    scores: TensorView,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
) raises AlofaError:
    """Causal grouped-query attention, one visible window per row.

    Row `t` attends to rows `0..upto[t]` starting from `k_bases[t]`. Nothing else
    in the cache is reachable from this call: the address of another request's
    history is simply not passed in, so "requests do not attend to each other" is
    true here by construction rather than by a mask that could be dropped.

    The arithmetic is `scalar.attention`'s, in its order — `Float64` dot products,
    one scale, three softmax passes — so with one request in the batch the two
    kernels produce the same bits. That is what makes comparing a batch against
    a serial run a statement about batching rather than about luck.

    `scores` is caller-owned scratch of at least `rows * pitch` floats, where
    `pitch` is derived from the scratch itself; this kernel allocates nothing.
    """
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise AlofaError(ERR_UNSUPPORTED, "n_kv_heads must divide n_heads")
    var rows = rows_of(q, "q")
    var kv_cols = n_kv_heads * head_dim
    var q_cols = n_heads * head_dim
    expect_matrix(q, rows, q_cols, "q")
    expect_matrix(dst, rows, q_cols, "dst")
    if rows <= 0:
        raise AlofaError(ERR_SHAPE_MISMATCH, "attention needs at least one row")
    var pitch = scores.numel() // rows
    if pitch <= 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH, "score scratch has no room for one row"
        )

    var pq = f32_data(q)
    var ps = f32_data(scores)
    var po = f32_data(dst)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var group = n_heads // n_kv_heads

    for head in range(n_heads):
        var kv_head = head // group
        var q_head_base = head * head_dim
        var kv_head_base = kv_head * head_dim
        for t in range(rows):
            var stop = upto[unsafe_offset=t]
            if stop < 0:
                raise AlofaError(
                    ERR_OUT_OF_RANGE, "a row has no visible keys"
                )
            if stop + 1 > pitch:
                raise AlofaError(
                    ERR_OUT_OF_RANGE, "score scratch row is too short"
                )
            var pk = k_bases[unsafe_offset=t].unsafe_bitcast[Float32]()
            var pv = v_bases[unsafe_offset=t].unsafe_bitcast[Float32]()
            var row = t * pitch
            var best = LOWEST_FP32
            for j in range(stop + 1):
                var acc = Float64(0)
                for d in range(head_dim):
                    acc += Float64(
                        pq[unsafe_offset=t * q_cols + q_head_base + d]
                    ) * Float64(
                        pk[unsafe_offset=j * kv_cols + kv_head_base + d]
                    )
                var s = Float32(acc) * scale
                ps[unsafe_offset=row + j] = s
                if s > best:
                    best = s
            var total = Float64(0)
            for j in range(stop + 1):
                var e = Float64(exp(ps[unsafe_offset=row + j] - best))
                ps[unsafe_offset=row + j] = Float32(e)
                total += e
            for d in range(head_dim):
                var acc = Float64(0)
                for j in range(stop + 1):
                    acc += Float64(ps[unsafe_offset=row + j]) * Float64(
                        pv[unsafe_offset=j * kv_cols + kv_head_base + d]
                    )
                po[unsafe_offset=t * q_cols + q_head_base + d] = Float32(
                    acc / total
                )


def select_rows(
    dst: TensorView,
    src: TensorView,
    which: IntPtr,
    rows: Int,
) raises AlofaError:
    """Copy `rows` named rows of `src` into consecutive rows of `dst`.

    A batch's answer lives in one row per request, and those rows are not
    adjacent unless the batch happens to be assembled that way. Gathering them
    costs one copy of `rows * cols` floats; not gathering them would mean every
    consumer learned that "the result of request r is at row n_r", which is a
    convention nobody else has heard of.
    """
    var dst_rows = rows_of(dst, "dst")
    var cols = cols_of(dst, "dst")
    var src_rows = rows_of(src, "src")
    if cols_of(src, "src") != cols:
        raise AlofaError(
            ERR_SHAPE_MISMATCH, "selected rows must keep their width"
        )
    if rows <= 0 or dst_rows < rows:
        raise AlofaError(
            ERR_SHAPE_MISMATCH, "nothing selected, or nowhere to put it"
        )
    var pd = f32_data(dst)
    var psrc = f32_data(src)
    for i in range(rows):
        var at = which[unsafe_offset=i]
        if at < 0 or at >= src_rows:
            raise AlofaError(ERR_OUT_OF_RANGE, "selected row is not there")
        for c in range(cols):
            pd[unsafe_offset=i * cols + c] = psrc[unsafe_offset=at * cols + c]
