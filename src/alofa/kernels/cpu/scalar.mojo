"""The scalar CPU backend: one element at a time, in formula order.

This is the oracle backend. It is read more often than it is run, so it is
written the way the mathematics is written — a dot product accumulates over `k`
in order, a norm computes its sum of squares before it divides — and it does no
tiling, no unrolling and no reassociation. A faster backend is allowed to
reorder any of that; it is then validated against this one, and against the
Hugging Face reference, by `tests/unit/test_layer0_parity.mojo`.

Layouts are fixed and stated per function:

- activations are row-major `[tokens, channels]`;
- projection weights are row-major `[dst, in]`, so a projection is
  `x @ wᵀ` — the same orientation Hugging Face stores them in, which keeps
  the fixture a plain dump of `nn.Linear.weight` with no transpose step that
  could be forgotten;
- per-head tensors (q, k, v, and the rotary tables) keep heads packed in the
  channel dimension: index `h * head_dim + d`.

Errors are named and raised, never silent: a shape mismatch that produced a
plausible-looking tensor would cost more to find than it costs to check.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_layer0_parity.mojo
"""

from std.math import exp, sqrt

from alofa.core.error import ERR_SHAPE_MISMATCH, ERR_UNSUPPORTED, AlofaError
from alofa.core.tensor import F32Ptr, TensorView, f32_data

# The seed for a softmax row maximum: the most negative finite fp32 value.
# Every unmasked score beats it, so the first comparison always wins and no
# row needs a special case for "no scores yet".
comptime LOWEST_FP32 = Float32(-3.4028234663852886e38)


def rows_of(view: TensorView, name: String) raises AlofaError -> Int:
    """Row count of a 2-D view, or a named shape error."""
    if view.rank() != 2:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "expected a 2-D view",
            "name=" + name + " rank=" + String(view.rank()),
        )
    return view.shape.dims[0]


def cols_of(view: TensorView, name: String) raises AlofaError -> Int:
    """Column count of a 2-D view, or a named shape error."""
    if view.rank() != 2:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "expected a 2-D view",
            "name=" + name + " rank=" + String(view.rank()),
        )
    return view.shape.dims[1]


def expect_matrix(
    view: TensorView, rows: Int, cols: Int, name: String
) raises AlofaError:
    """Assert a view is `[rows, cols]`; a wrong shape is a bug, not a case."""
    var got_rows = rows_of(view, name)
    var got_cols = cols_of(view, name)
    if got_rows != rows or got_cols != cols:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "view has the wrong shape",
            "name="
            + name
            + " got="
            + String(got_rows)
            + "x"
            + String(got_cols)
            + " want="
            + String(rows)
            + "x"
            + String(cols),
        )


def expect_vector(view: TensorView, n: Int, name: String) raises AlofaError:
    """Assert a view holds exactly `n` elements."""
    if view.numel() != n:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "view has the wrong number of elements",
            "name=" + name + " got=" + String(view.numel()) + " want=" + String(n),
        )


def rmsnorm(
    dst: TensorView, x: TensorView, w: TensorView, eps: Float32
) raises AlofaError:
    """`dst = x / sqrt(mean(x²) + eps) * w`, per row.

    The reciprocal square root is computed once per row and then multiplied,
    rather than dividing twice: the two agree to within a rounding, but the
    multiply is what every reference implementation does, and matching the
    reference's rounding is the point of this backend.

    The sum of squares accumulates in `Float64`. A row is 896 elements wide and
    a fp32 accumulator's error grows with the number of additions, so the
    double accumulator is what keeps this backend's answer from being the thing
    under suspicion when a faster one disagrees.
    """
    var rows = rows_of(x, "x")
    var cols = cols_of(x, "x")
    expect_vector(w, cols, "w")
    expect_matrix(dst, rows, cols, "dst")

    var px = f32_data(x)
    var pw = f32_data(w)
    var po = f32_data(dst)

    for row in range(rows):
        var base = row * cols
        var sum_sq = Float64(0)
        for col in range(cols):
            var v = Float64(px[unsafe_offset=base + col])
            sum_sq += v * v
        var scale = Float64(1.0) / sqrt(sum_sq / Float64(cols) + Float64(eps))
        for col in range(cols):
            po[unsafe_offset=base + col] = Float32(
                Float64(px[unsafe_offset=base + col])
                * scale
                * Float64(pw[unsafe_offset=col])
            )


def _gemm(
    dst: TensorView,
    x: TensorView,
    w: TensorView,
    bias: F32Ptr,
    has_bias: Bool,
) raises AlofaError:
    """`dst[m, n] = bias[n] + Σ_k x[m, k] * w[n, k]`.

    `w` is stored `[dst, in]`, so the reduction walks `k` in both rows at once
    — the layout that makes a row of `w` contiguous, which is what a later
    SIMD backend wants and what Hugging Face already stores.

    The accumulator is `Float64` for the same reason as in `rmsnorm`: `k` is
    896 for a Qwen2 projection, and this backend's job is to be the answer
    other backends are judged against.
    """
    var rows = rows_of(x, "x")
    var inner = cols_of(x, "x")
    var cols = rows_of(w, "w")
    if cols_of(w, "w") != inner:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "weight columns do not match input columns",
            "w_cols=" + String(cols_of(w, "w")) + " x_cols=" + String(inner),
        )
    expect_matrix(dst, rows, cols, "dst")

    var px = f32_data(x)
    var pw = f32_data(w)
    var po = f32_data(dst)

    for row in range(rows):
        var x_base = row * inner
        for col in range(cols):
            var w_base = col * inner
            var acc = Float64(0)
            if has_bias:
                acc = Float64(bias[unsafe_offset=col])
            for k in range(inner):
                acc += Float64(px[unsafe_offset=x_base + k]) * Float64(
                    pw[unsafe_offset=w_base + k]
                )
            po[unsafe_offset=row * cols + col] = Float32(acc)


def linear(dst: TensorView, x: TensorView, w: TensorView) raises AlofaError:
    """`dst = x @ wᵀ`; see `_gemm` for the layout.

    `dst`'s own pointer stands in for the unused bias: `_gemm` reads it only
    when `has_bias` is set, and passing a real pointer keeps the signature free
    of an optional that every call site would have to unwrap.
    """
    _gemm(dst, x, w, f32_data(dst), False)


def linear_bias(
    dst: TensorView, x: TensorView, w: TensorView, bias: TensorView
) raises AlofaError:
    """`dst = x @ wᵀ + bias`; Qwen2's q/k/v projections all carry a bias."""
    expect_vector(bias, rows_of(w, "w"), "bias")
    _gemm(dst, x, w, f32_data(bias), True)


def add(dst: TensorView, a: TensorView, b: TensorView) raises AlofaError:
    """Element-wise `dst = a + b` — the residual path."""
    var n = a.numel()
    expect_vector(b, n, "b")
    expect_vector(dst, n, "dst")
    var pa = f32_data(a)
    var pb = f32_data(b)
    var po = f32_data(dst)
    for i in range(n):
        po[unsafe_offset=i] = pa[unsafe_offset=i] + pb[unsafe_offset=i]


def silu(dst: TensorView, x: TensorView) raises AlofaError:
    """`dst = x * sigmoid(x)`, computed as `x / (1 + exp(-x))`."""
    var n = x.numel()
    expect_vector(dst, n, "dst")
    var px = f32_data(x)
    var po = f32_data(dst)
    for i in range(n):
        var v = Float64(px[unsafe_offset=i])
        po[unsafe_offset=i] = Float32(v / (Float64(1) + exp(-v)))


def swiglu(dst: TensorView, gate: TensorView, up: TensorView) raises AlofaError:
    """`dst = silu(gate) * up` — the feed-forward block's activation."""
    var n = gate.numel()
    expect_vector(up, n, "up")
    expect_vector(dst, n, "dst")
    var pg = f32_data(gate)
    var pu = f32_data(up)
    var po = f32_data(dst)
    for i in range(n):
        var v = Float64(pg[unsafe_offset=i])
        po[unsafe_offset=i] = Float32(
            v / (Float64(1) + exp(-v)) * Float64(pu[unsafe_offset=i])
        )


def rope_shapes(
    out_q: TensorView,
    out_k: TensorView,
    q: TensorView,
    k: TensorView,
    cos: TensorView,
    sin: TensorView,
    head_dim: Int,
) raises AlofaError -> Int:
    """`rope` 的形状契约；返回 token 数，其余维度调用方自己从 view 上读。

    两个后端共用这一份检查，而不是各抄一遍：形状契约抄两遍就会漂移，而漂移的
    那一侧会以「形状检查通过、但读到了别人的内存」的形式出现 —— 那是比数值不
    一致难查得多的失败。

    这里只判契约、不碰算术，所以把标量版改成调它不会改变算出的任何一个数。
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise AlofaError(
            ERR_UNSUPPORTED,
            "head dimension must be positive and even",
            "head_dim=" + String(head_dim),
        )
    var tokens = rows_of(q, "q")
    var q_cols = cols_of(q, "q")
    var k_cols = cols_of(k, "k")
    if rows_of(k, "k") != tokens:
        raise AlofaError(
            ERR_SHAPE_MISMATCH, "q and k must have the same token count", ""
        )
    if q_cols % head_dim != 0 or k_cols % head_dim != 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "channel count is not a whole number of heads",
            "q_cols=" + String(q_cols) + " k_cols=" + String(k_cols),
        )
    expect_matrix(out_q, tokens, q_cols, "out_q")
    expect_matrix(out_k, tokens, k_cols, "out_k")
    # The tables are per token over one head's channels, `[tokens, head_dim]`.
    if rows_of(cos, "cos") < tokens or cols_of(cos, "cos") != head_dim:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "cos table must be [tokens, head_dim]",
            "rows=" + String(rows_of(cos, "cos")) + " cols=" + String(cols_of(cos, "cos")),
        )
    expect_matrix(sin, rows_of(cos, "cos"), head_dim, "sin")
    return tokens


def rope(
    out_q: TensorView,
    out_k: TensorView,
    q: TensorView,
    k: TensorView,
    cos: TensorView,
    sin: TensorView,
    head_dim: Int,
) raises AlofaError:
    """Rotary position embedding, applied the way the reference applies it.

    For each head, the channel dimension is split in half and the two halves
    are rotated into each other:

        dst[d]         =  q[d] * cos[d]          - q[d + h] * sin[d]
        dst[d + h]     =  q[d + h] * cos[d + h]  + q[d] * sin[d + h]

    where `h = head_dim / 2`. Both halves of `cos`/`sin` are stored and indexed
    separately even though the reference's table repeats itself across them:
    the repetition is a property of how the table is built, not of this
    function, and hard-coding it here would break the day a table is not
    symmetric.

    The tables are indexed by token, so a caller decoding from a KV cache passes
    the row for the token's absolute position, not for its index in the batch.
    """
    var tokens = rope_shapes(out_q, out_k, q, k, cos, sin, head_dim)
    var q_cols = cols_of(q, "q")
    var k_cols = cols_of(k, "k")
    var half = head_dim // 2

    var pq = f32_data(q)
    var pk = f32_data(k)
    var pc = f32_data(cos)
    var ps = f32_data(sin)
    var poq = f32_data(out_q)
    var pok = f32_data(out_k)

    for t in range(tokens):
        var table = t * head_dim
        for head in range(q_cols // head_dim):
            var base = t * q_cols + head * head_dim
            for d in range(half):
                var a = pq[unsafe_offset=base + d]
                var b = pq[unsafe_offset=base + d + half]
                poq[unsafe_offset=base + d] = a * pc[unsafe_offset=table + d] - b * ps[
                    unsafe_offset=table + d
                ]
                poq[unsafe_offset=base + d + half] = b * pc[
                    unsafe_offset=table + d + half
                ] + a * ps[unsafe_offset=table + d + half]
        for head in range(k_cols // head_dim):
            var base = t * k_cols + head * head_dim
            for d in range(half):
                var a = pk[unsafe_offset=base + d]
                var b = pk[unsafe_offset=base + d + half]
                pok[unsafe_offset=base + d] = a * pc[unsafe_offset=table + d] - b * ps[
                    unsafe_offset=table + d
                ]
                pok[unsafe_offset=base + d + half] = b * pc[
                    unsafe_offset=table + d + half
                ] + a * ps[unsafe_offset=table + d + half]


def attention_shapes(
    dst: TensorView,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    scores: TensorView,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
) raises AlofaError -> Int:
    """`attention` 的形状契约；返回 query 条数，key 条数调用方自己从 `k` 上读。

    与 `rope_shapes` 同理：两后端共用一份，只判契约不碰算术。
    """
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise AlofaError(
            ERR_UNSUPPORTED,
            "head counts must be positive with n_kv_heads dividing n_heads",
            "n_heads=" + String(n_heads) + " n_kv_heads=" + String(n_kv_heads),
        )
    var q_len = rows_of(q, "q")
    var kv_len = rows_of(k, "k")
    if rows_of(v, "v") != kv_len:
        raise AlofaError(
            ERR_SHAPE_MISMATCH, "k and v must have the same token count", ""
        )
    if q_len > kv_len:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "more queries than keys: a query cannot attend to the future",
            "q_len=" + String(q_len) + " kv_len=" + String(kv_len),
        )
    expect_matrix(q, q_len, n_heads * head_dim, "q")
    expect_matrix(dst, q_len, n_heads * head_dim, "dst")
    expect_matrix(k, kv_len, n_kv_heads * head_dim, "k")
    expect_matrix(v, kv_len, n_kv_heads * head_dim, "v")
    if scores.numel() < q_len * kv_len:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "score scratch is too small",
            "got=" + String(scores.numel()) + " want=" + String(q_len * kv_len),
        )
    return q_len


def attention(
    dst: TensorView,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    scores: TensorView,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
) raises AlofaError:
    """Causal grouped-query attention: `softmax(q kᵀ / √d, masked) · v`.

    Grouped-query attention is handled by mapping query head `h` to key head
    `h // (n_heads / n_kv_heads)`, which is exactly the reshape the reference
    does when it repeats a key head — but without materializing the repeated
    tensor, so nothing is copied per token.

    The query length and the key length may differ. A query row `t` is at
    absolute position `kv_len - q_len + t`, so the same call covers a whole
    prefill (`q_len == kv_len`, position `t`) and one decode step (`q_len` is
    1, position `kv_len - 1`) — one code path, tested once, used twice. `k`
    and `v` here are whatever the caller wants attention over, which in a
    decode is the key-value store, not a projection output.

    `scores` is caller-owned scratch of at least `q_len * kv_len` elements:
    this backend allocates nothing, and a row of scores is written, read twice
    and then overwritten, so a single row would do if the caller had one.

    The softmax is done in three passes over the row — maximum, then
    exponentiated weights, then the value reduction — rather than fused, so
    that a disagreement with a fused backend can be localized to the pass that
    differs.
    """
    var q_len = attention_shapes(dst, q, k, v, scores, n_heads, n_kv_heads, head_dim)
    var kv_len = rows_of(k, "k")

    var pq = f32_data(q)
    var pk = f32_data(k)
    var pv = f32_data(v)
    var ps = f32_data(scores)
    var po = f32_data(dst)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var group = n_heads // n_kv_heads
    var q_cols = n_heads * head_dim
    var kv_cols = n_kv_heads * head_dim

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
                var acc = Float64(0)
                for d in range(head_dim):
                    acc += Float64(pq[unsafe_offset=t * q_cols + q_head_base + d]) * Float64(
                        pk[unsafe_offset=j * kv_cols + kv_head_base + d]
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
                    acc += Float64(ps[unsafe_offset=row + j]) * Float64(
                        pv[unsafe_offset=j * kv_cols + kv_head_base + d]
                    )
                po[unsafe_offset=t * q_cols + q_head_base + d] = Float32(
                    acc / total
                )
