"""向量版 kernel：与 `scalar.mojo` 同一套签名、同一套形状契约。

这个模块本轮只承诺一件事：**算出的数和标量后端一样**。它不承诺快，也不承诺真
的降成了某条指令 —— 见下面的说明。

**为什么累加还在 `Float64`。** 向量化最常见的"提速"是把累加从 f64 换成 f32 多
累加器，那样确实快，但点积是有抵消的：896 项里正负相消之后，结果的量级可能远
小于各项绝对值之和，f32 累加的相对误差（约 `√n · 2^-24`）在抵消严重时能吃掉
整个 1e-5 的判据。本模块与标量共用同一条判据，而这条判据要判的是"向量化有没有
算错"，不是"向量化有多快" —— 掺进一个"顺便放宽了精度"的变化，这两件事就都
说不清了。所以：乘加在 f64 的通道里做，宽度 4（AVX2 上 f64 就是 4 通道）。

**宽度是 `comptime` 常量，不是"这台机器的宽度"。** 本机是 i7-9700K，只有 AVX2；
写成常量的意思是"这段 kernel 按 4/8 通道写"，换机器时改这里，而不是让人以为
编译器会自己挑。尾巴（不足一个向量的部分）走标量，**不许**假设形状能整除：
今天 hidden=896、intermediate=4864 恰好都能被 4 和 8 整除，明天换个模型就不是
了，而"恰好整除所以没测到尾巴"正是最难查的那类 bug。

**不宣称指令。** 这个模块叫 avx2，说的是"按 8 通道 f32 / 4 通道 f64 写的向量
kernel"；它是否真的被编译成 VEX 编码的 `vmulpd`，取决于编译器与目标特性，本轮
**不测、也不写进账本**。要宣称就得拿 `objdump` 数指令，没数过就不许说。

为什么 q4 这条融合通路用 **8 条 f64 通道**（而不是本文件别处惯用的 4 条）
--------------------------------------------------------------------
`_gemm` 用 4 通道是因为它沿 `inner` 这条连续内存切通道，每条通道累加自己那一段。
q4 的块是 **32 个值 / 18 字节**，一个 nibble 不是一个通道宽度能对齐的东西，于是这里
的切法是按 "j % 8" 分：一个块拆成两半（每半 8 字节），第 j 个量化值固定落在第 `j%8`
条通道上。这样切之后，整块的 `Σ (n-8)·x` 是**一次**通道归并就得到的，而它正好能
被后来的 `* d` 一次乘掉 —— 缩放因子是**整块共享**的，没必要每个元素乘一遍。
"""

from std.math import exp, sqrt

from alofa.core.dtype import (
    QUANT_Q4_0,
    quant_block_size,
    quant_bytes_per_block,
    quant_scale_bytes,
)
from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_SHAPE_MISMATCH,
    AlofaError,
)
from alofa.core.tensor import F32Ptr, TensorView, f32_data
from alofa.kernels.cpu.quant import block_scale
from alofa.kernels.cpu.scalar import (
    LOWEST_FP32,
    attention_shapes,
    cols_of,
    expect_matrix,
    expect_vector,
    rope_shapes,
    rows_of,
)

# 通道数。f32 取 8（AVX2 一条 ymm 装 8 个单精度），f64 取 4（双精度减半）。
comptime W_F32 = 8
comptime W_F64 = 4

# q4 块流用字节寻址（`quant.mojo` 用的是同一套定义）而不是 `TensorView`：量化的元素
# 是"32 个一组共享一个 fp16 缩放"，没有固定字节步长可言。
comptime Q4_U8 = Pointer[UInt8, MutUntrackedOrigin]
comptime Q4_BLOCK = quant_block_size[QUANT_Q4_0]()
comptime Q4_BYTES = quant_bytes_per_block[QUANT_Q4_0]()
comptime Q4_SCALE_BYTES = quant_scale_bytes[QUANT_Q4_0]()
# 一次吃 8 个字节 = 16 个量化值 = 半块。
comptime W_Q4 = 8
# 半块切分：一个 q4_0 块的 16 个数据字节一次读满，低半字节对应值 0..15、高半字节
# 对应 16..31 —— 两边各自配上 x 的一段连续区间，这就是宽度取 16 的理由。
comptime W_Q4_HALF = 16


def rmsnorm(
    dst: TensorView, x: TensorView, w: TensorView, eps: Float32
) raises AlofaError:
    """`dst = x / sqrt(mean(x²) + eps) * w`，逐行。

    与标量版唯一的算术差别是求和的**结合顺序**：标量的 896 次加法是顺序的，这里
    是 4 条通道各自累加再归并。两者都在 f64 里，差别在 1e-16 量级 —— 远在 1e-5
    判据之下，所以这个差别是**被允许的**，而换精度不是。
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
        var acc = SIMD[DType.float64, W_F64](Float64(0))
        var col = 0
        while col + W_F64 <= cols:
            var v = px.unsafe_load[width=W_F64](base + col).cast[DType.float64]()
            acc += v * v
            col += W_F64
        var sum_sq = acc.reduce_add()
        while col < cols:
            var v = Float64(px[unsafe_offset=base + col])
            sum_sq += v * v
            col += 1

        var scale = Float64(1.0) / sqrt(sum_sq / Float64(cols) + Float64(eps))
        col = 0
        while col + W_F64 <= cols:
            var xv = px.unsafe_load[width=W_F64](base + col).cast[DType.float64]()
            var wv = pw.unsafe_load[width=W_F64](col).cast[DType.float64]()
            var scaled = (xv * scale) * wv
            po.unsafe_store[width=W_F64](base + col, scaled.cast[DType.float32]())
            col += W_F64
        while col < cols:
            po[unsafe_offset=base + col] = Float32(
                Float64(px[unsafe_offset=base + col])
                * scale
                * Float64(pw[unsafe_offset=col])
            )
            col += 1


def _gemm_tile[RB: Int](
    po: F32Ptr,
    px: F32Ptr,
    pw: F32Ptr,
    bias: F32Ptr,
    has_bias: Bool,
    row0: Int,
    cols: Int,
    inner: Int,
) -> None:
    """`RB` 行 × 全部 `cols` 列：列外层、`k` 中层、行内层。

    循环次序是**唯一**被改动的东西：老写法是「行外层、列内层」，于是 `w` 的
    每一列在每一行上都被重读一遍 —— 一个 `rows` 行的前向把整个权重矩阵读了
    `rows` 遍。批 = 8 的 decode 和 32 token 的 prefill，摊到每个 token 上的
    字节数和批 = 1 一样（fp32 每 token 1.976 GB）。

    这里把 `RB` 行捏成一块，一块走完一遍 `w`：`w[col][:]` 从 DRAM 进来一次，
    被 `RB` 行复用。于是权重总流量除以约 `RB`（严格是 `rows / ceil(rows/RB)`）。
    量出来的证据在 `scripts/bench_gemm_rows.mojo`。

    ⚠️ **逐位兼容是硬约束，不是"在容差内"**：每个输出各自一个 f64×4 累加器，
    按**同样的 `k` 次序**累加，求和顺序与老写法一步一步对应 ——
    ① 偏置进 0 号通道（只加一次，见 `_gemm` 的注释）；② 4 个一组做乘加；
    ③ `reduce_add()`；④ `inner % 4` 的尾巴**顺序**加在 reduce 之后（所以
    `tail[r]` 的初值是 `reduce_add()` 的结果，不是 0 —— 浮点加法不满足结合律，
    `0+t0+t1` 与 `r+t0+t1` 是两个数）。切分/换序不改变任何一次浮点运算，
    差 1 ulp 就是 bug。

    `RB` 取 1/2/4/8 由 `_gemm` 按剩余行数派发：`RB` 行累加器要占 `RB` 个
    ymm 寄存器，取 16 会溢出到栈而失去意义。
    """
    var acc = InlineArray[SIMD[DType.float64, W_F64], RB](
        fill=SIMD[DType.float64, W_F64](Float64(0))
    )
    var tail = InlineArray[Float64, RB](fill=Float64(0))
    for col in range(cols):
        var w_base = col * inner

        comptime for r in range(RB):
            acc[r] = SIMD[DType.float64, W_F64](Float64(0))
            if has_bias:
                acc[r][0] = Float64(bias[unsafe_offset=col])
        var k = 0
        while k + W_F64 <= inner:
            var wv = pw.unsafe_load[width=W_F64](w_base + k).cast[
                DType.float64
            ]()

            comptime for r in range(RB):
                var xv = px.unsafe_load[width=W_F64](
                    (row0 + r) * inner + k
                ).cast[DType.float64]()
                acc[r] += xv * wv
            k += W_F64

        comptime for r in range(RB):
            tail[r] = acc[r].reduce_add()
        while k < inner:
            var wk = Float64(pw[unsafe_offset=w_base + k])

            comptime for r in range(RB):
                tail[r] += Float64(
                    px[unsafe_offset=(row0 + r) * inner + k]
                ) * wk
            k += 1

        comptime for r in range(RB):
            po[unsafe_offset=(row0 + r) * cols + col] = Float32(tail[r])


def _gemm(
    dst: TensorView,
    x: TensorView,
    w: TensorView,
    bias: F32Ptr,
    has_bias: Bool,
) raises AlofaError:
    """`dst[m, n] = bias[n] + Σ_k x[m, k] * w[n, k]`；与标量 `_gemm` 同布局。

    内层在 f64 通道里跑，理由写在文件头。偏置先播进累加器（而不是最后再加），
    与标量版一致 —— 浮点加法不满足结合律，"先加"和"后加"是两个数。

    行按 8/4/2/1 分块交给 `_gemm_tile`，块内权重只读一遍：
    **批和预填的每 token 字节数因此除以约 `RB`**，这是唯一被改的东西
    （`rows == 1` 的那条 decode 通路形状与老写法一致）。
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

    var row = 0
    while row < rows:
        var left = rows - row
        if left >= 8:
            _gemm_tile[8](po, px, pw, bias, has_bias, row, cols, inner)
            row += 8
        elif left >= 4:
            _gemm_tile[4](po, px, pw, bias, has_bias, row, cols, inner)
            row += 4
        elif left >= 2:
            _gemm_tile[2](po, px, pw, bias, has_bias, row, cols, inner)
            row += 2
        else:
            _gemm_tile[1](po, px, pw, bias, has_bias, row, cols, inner)
            row += 1


def linear(dst: TensorView, x: TensorView, w: TensorView) raises AlofaError:
    """`dst = x @ wᵀ`，向量版；未用的偏置位由 `dst` 自己的指针占位。"""
    _gemm(dst, x, w, f32_data(dst), False)


def linear_bias(
    dst: TensorView, x: TensorView, w: TensorView, bias: TensorView
) raises AlofaError:
    """`dst = x @ wᵀ + bias`；Qwen2 的 q/k/v 投影都带偏置。"""
    expect_vector(bias, rows_of(w, "w"), "bias")
    _gemm(dst, x, w, f32_data(bias), True)


def add(dst: TensorView, a: TensorView, b: TensorView) raises AlofaError:
    """逐元素 `dst = a + b`，残差通路。

    这里用 f32 的 8 通道 —— 加法没有累加，也就不存在误差随长度增长的问题，
    4 通道的 f64 只会让通路白白窄一半。
    """
    var n = a.numel()
    expect_vector(b, n, "b")
    expect_vector(dst, n, "dst")
    var pa = f32_data(a)
    var pb = f32_data(b)
    var po = f32_data(dst)
    var i = 0
    while i + W_F32 <= n:
        po.unsafe_store[width=W_F32](
            i, pa.unsafe_load[width=W_F32](i) + pb.unsafe_load[width=W_F32](i)
        )
        i += W_F32
    while i < n:
        po[unsafe_offset=i] = pa[unsafe_offset=i] + pb[unsafe_offset=i]
        i += 1


def swiglu(dst: TensorView, gate: TensorView, up: TensorView) raises AlofaError:
    """`dst = silu(gate) * up`，激活函数。

    `exp` 在 f64 通道里求，与标量版（`Float64` 里算完再舍回 f32）逐步对应：`silu`
    在负半轴是 `x·sigmoid(x)`，两个都接近 0 的量相乘，精度全靠中间那步在 f64。
    """
    var n = gate.numel()
    expect_vector(up, n, "up")
    expect_vector(dst, n, "dst")
    var pg = f32_data(gate)
    var pu = f32_data(up)
    var po = f32_data(dst)
    var i = 0
    while i + W_F64 <= n:
        var v = pg.unsafe_load[width=W_F64](i).cast[DType.float64]()
        var u = pu.unsafe_load[width=W_F64](i).cast[DType.float64]()
        var activated = v / (SIMD[DType.float64, W_F64](Float64(1)) + exp(-v))
        po.unsafe_store[width=W_F64](i, (activated * u).cast[DType.float32]())
        i += W_F64
    while i < n:
        var v = Float64(pg[unsafe_offset=i])
        po[unsafe_offset=i] = Float32(
            v / (Float64(1) + exp(-v)) * Float64(pu[unsafe_offset=i])
        )
        i += 1


def _rope_half(
    po: F32Ptr,
    px: F32Ptr,
    pc: F32Ptr,
    ps: F32Ptr,
    base: Int,
    table: Int,
    half: Int,
) -> None:
    """一个头的一半：沿 `half` 这条连续内存做 8 通道 f32，尾巴走标量。"""
    var d = 0
    while d + W_F32 <= half:
        var a = px.unsafe_load[width=W_F32](base + d)
        var b = px.unsafe_load[width=W_F32](base + d + half)
        po.unsafe_store[width=W_F32](
            base + d,
            a * pc.unsafe_load[width=W_F32](table + d)
            - b * ps.unsafe_load[width=W_F32](table + d),
        )
        po.unsafe_store[width=W_F32](
            base + d + half,
            b * pc.unsafe_load[width=W_F32](table + d + half)
            + a * ps.unsafe_load[width=W_F32](table + d + half),
        )
        d += W_F32
    while d < half:
        var a = px[unsafe_offset=base + d]
        var b = px[unsafe_offset=base + d + half]
        po[unsafe_offset=base + d] = a * pc[unsafe_offset=table + d] - b * ps[
            unsafe_offset=table + d
        ]
        po[unsafe_offset=base + d + half] = b * pc[
            unsafe_offset=table + d + half
        ] + a * ps[unsafe_offset=table + d + half]
        d += 1


def rope(
    out_q: TensorView,
    out_k: TensorView,
    q: TensorView,
    k: TensorView,
    cos: TensorView,
    sin: TensorView,
    head_dim: Int,
) raises AlofaError:
    """旋转位置编码，向量版；形状契约与标量版**共用** `rope_shapes`。

    这里用 f32 的 8 通道而不是 f64：`rope` 是**没有累加**的逐元素运算，两项相乘
    就写回，误差不随长度增长 —— 走 f64 通道只是让通路白白窄一半。也正因为没有累
    加、没有重排，本实现与标量版是**逐位相同**的，不只是「在容差内」。

    向量化沿 `half`（半个头的通道数）展开，它是一段连续内存；`cos`/`sin` 的同一
    半同样连续。不足 8 个通道的尾巴走标量：head_dim=64 恰好整除，但「恰好整除所
    以尾巴没被跑到」正是换个模型就塌的那类依赖。
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
            _rope_half(poq, pq, pc, ps, t * q_cols + head * head_dim, table, half)
        for head in range(k_cols // head_dim):
            _rope_half(pok, pk, pc, ps, t * k_cols + head * head_dim, table, half)


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
    """因果分组注意力，向量版；形状契约与标量版**共用** `attention_shapes`。

    向量化只落在两处 `O(kv_len × head_dim)` 的工作上 —— 打分点积与值归约，
    因为那才是这段的耗时主体；softmax 的取最大 / 求指数 / 求和仍是标量的，于是
    `scores` 行里的数与标量版**逐位相同**，一旦对不上就能把问题定位到「点积」
    而不是「softmax」。

    **打分点积**沿 `head_dim` 做 4 通道 f64，4 条通道各自累加最后归并 —— 这是
    一次**重排结合**，与文件头写的理由一致（都在 f64 里，差别在 1e-16 量级，远
    在 1e-5 判据之下）。尾巴的通道单独在标量 f64 里累加，再与向量部分的归约
    **相加之后**才乘 `scale`：先乘后加与先加后乘是两个数，不许拆开。

    **值归约**反过来是**逐位相同**的：向量沿 `head_dim` 切通道，每个通道各自
    **顺序**累加 `j`，与标量版同一条顺序 —— 向量化的是通道，不是累加顺序。
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
    var d_end = (head_dim // W_F64) * W_F64

    for head in range(n_heads):
        var kv_head = head // group
        var q_head_base = head * head_dim
        var kv_head_base = kv_head * head_dim
        for t in range(q_len):
            # The last `q_len` positions of a `kv_len`-long context.
            var upto = kv_len - q_len + t
            var row = t * kv_len
            var qb = t * q_cols + q_head_base
            var best = LOWEST_FP32
            for j in range(upto + 1):
                var kb = j * kv_cols + kv_head_base
                var acc = SIMD[DType.float64, W_F64](Float64(0))
                var d = 0
                while d < d_end:
                    var qv = pq.unsafe_load[width=W_F64](qb + d).cast[DType.float64]()
                    var kvv = pk.unsafe_load[width=W_F64](kb + d).cast[DType.float64]()
                    acc += qv * kvv
                    d += W_F64
                var tail = Float64(0)
                while d < head_dim:
                    tail += Float64(pq[unsafe_offset=qb + d]) * Float64(
                        pk[unsafe_offset=kb + d]
                    )
                    d += 1
                var s = Float32(acc.reduce_add() + tail) * scale
                ps[unsafe_offset=row + j] = s
                if s > best:
                    best = s
            var total = Float64(0)
            for j in range(upto + 1):
                var e = Float64(exp(ps[unsafe_offset=row + j] - best))
                ps[unsafe_offset=row + j] = Float32(e)
                total += e

            var ob = t * q_cols + q_head_base
            var d = 0
            while d + W_F64 <= head_dim:
                var acc = SIMD[DType.float64, W_F64](Float64(0))
                for j in range(upto + 1):
                    var w = Float64(ps[unsafe_offset=row + j])
                    var vv = pv.unsafe_load[width=W_F64](
                        j * kv_cols + kv_head_base + d
                    ).cast[DType.float64]()
                    acc += w * vv
                po.unsafe_store[width=W_F64](
                    ob + d,
                    (acc / SIMD[DType.float64, W_F64](total)).cast[DType.float32](),
                )
                d += W_F64
            while d < head_dim:
                var acc = Float64(0)
                for j in range(upto + 1):
                    acc += Float64(ps[unsafe_offset=row + j]) * Float64(
                        pv[unsafe_offset=j * kv_cols + kv_head_base + d]
                    )
                po[unsafe_offset=ob + d] = Float32(acc / total)
                d += 1


# ---------------------------------------------------------------------------
# q4_0 融合 matmul（向量版）
# ---------------------------------------------------------------------------


def _matmul_q4_wide[
    wide: Bool
](
    dst: F32Ptr,
    x: F32Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    bias: F32Ptr,
    has_bias: Bool,
) raises AlofaError:
    """`dst[r] = Σ_c w[r,c] · x[c]`，`w` 为 q4_0 块流；`wide` 选累加类型。

    **按 `j % 8` 切通道，而不是按连续段切。** 一个 q4_0 块是 18 字节：2 字节缩放、
    16 字节装 32 个 nibble，低半字节是第 j 个、高半字节是第 `j+16` 个。把 16 个字节
    拆成两半各 8 字节之后，第 j 个量化值固定配到第 `j % 8` 条通道上：

        前半低半字节 → 第 0..7 个值 → x[0..7]      前半高半字节 → 第 16..23 个值
        后半低半字节 → 第 8..15 个值                后半高半字节 → 第 24..31 个值

    这一点决定了这里**不需要"尾巴"处理**：每个块的 16 个数据字节刚好是两个完整的
    8 字节，而 `cols` 必须是 32 的倍数（标量版同样约束，写在入口）。真正的余数在
    `rows` 与 `blocks_per_row` 上，而它们一律是块粒度、不受通道宽度影响 —— 这也是它
    和上一次那个坑的区别：那里是"head_dim 恰好整除，于是 4 通道版本的尾巴从头到尾
    没被跑到"，这里踩的是块布局**固定**给的余数，换模型也不会变。

    **`d` 折进 x、整行只归约一次。** 标量版每个元素都要乘一次 `d`；这里把 `d` 乘到 x
    上（整块共享，一次乘 8 个），于是八个部分和可以一路加**到行末**再归约一次，省掉
    每块一次的跨通道归约 —— 那玩意在向量单元上是三次 shuffle 加三次加，谁也绕不开。
    四种结构在本机同一进程里 A/B 过（best-of-7，`down_proj` 4864×896）：

        每块归约 + 单条链          1.25 周期/元素
        每块归约 + 四条链          1.37
        整行归约 + `d` 单独乘      1.25
        整行归约 + `d` 折进 x      1.19   ← 本文采纳

    **累加类型由 `wide` 选**：`True` 用 f64（与标量版逐位一致，也是默认），`False` 用
    f32（快约 1.4×，但会差出约 1e-7 量级 —— 要不要换是另一个决定，见账本）。

    `bias` 只在 `has_bias` 为真时读，约定同标量版。
    """
    if rows <= 0 or cols <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "matrix dimensions must be positive",
            "rows=" + String(rows) + " cols=" + String(cols),
        )
    if cols % Q4_BLOCK != 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "inner dimension must be a whole number of blocks",
            "cols=" + String(cols) + " block=" + String(Q4_BLOCK),
        )
    var blocks_per_row = cols // Q4_BLOCK
    var low_mask = SIMD[DType.uint8, W_Q4](0x0F)

    for row in range(rows):
        var acc = Float64(0)
        if has_bias:
            acc = Float64(bias[unsafe_offset=row])
        if wide:
            var eight = SIMD[DType.float64, W_Q4](8.0)
            var s0 = SIMD[DType.float64, W_Q4](0)
            var s1 = SIMD[DType.float64, W_Q4](0)
            var s2 = SIMD[DType.float64, W_Q4](0)
            var s3 = SIMD[DType.float64, W_Q4](0)
            for b in range(blocks_per_row):
                var base = (row * blocks_per_row + b) * Q4_BYTES
                var d = Float64(block_scale(blocks, base))
                var xb = b * Q4_BLOCK
                var raw0 = blocks.unsafe_load[width=W_Q4](base + Q4_SCALE_BYTES)
                var raw1 = blocks.unsafe_load[width=W_Q4](
                    base + Q4_SCALE_BYTES + W_Q4
                )
                var lo0 = (raw0 & low_mask).cast[DType.float64]() - eight
                var hi0 = (raw0 >> 4).cast[DType.float64]() - eight
                var lo1 = (raw1 & low_mask).cast[DType.float64]() - eight
                var hi1 = (raw1 >> 4).cast[DType.float64]() - eight
                var x0 = x.unsafe_load[width=W_Q4](xb).cast[DType.float64]() * d
                var x1 = x.unsafe_load[width=W_Q4](xb + 16).cast[
                    DType.float64
                ]() * d
                var x2 = x.unsafe_load[width=W_Q4](xb + W_Q4).cast[
                    DType.float64
                ]() * d
                var x3 = x.unsafe_load[width=W_Q4](xb + 24).cast[
                    DType.float64
                ]() * d
                # ⚠️ 不许写成 `s0.fma(lo0, x0)`：Mojo 的 `SIMD.fma(a, b)` 是
                # `self * a + b`（实测 3.fma(5, 7) == 22 == 3*5+7），照 `a*b+self`
                # 的直觉写会得到 `(a+b)*n` 型结果，且 -O0 下只表现为"数不对"。
                s0 += lo0 * x0
                s1 += hi0 * x1
                s2 += lo1 * x2
                s3 += hi1 * x3
            acc += ((s0 + s1) + (s2 + s3)).reduce_add()
        else:
            var eight = SIMD[DType.float32, W_Q4](8.0)
            var s0 = SIMD[DType.float32, W_Q4](0)
            var s1 = SIMD[DType.float32, W_Q4](0)
            var s2 = SIMD[DType.float32, W_Q4](0)
            var s3 = SIMD[DType.float32, W_Q4](0)
            for b in range(blocks_per_row):
                var base = (row * blocks_per_row + b) * Q4_BYTES
                var d = block_scale(blocks, base)
                var xb = b * Q4_BLOCK
                var raw0 = blocks.unsafe_load[width=W_Q4](base + Q4_SCALE_BYTES)
                var raw1 = blocks.unsafe_load[width=W_Q4](
                    base + Q4_SCALE_BYTES + W_Q4
                )
                var lo0 = (raw0 & low_mask).cast[DType.float32]() - eight
                var hi0 = (raw0 >> 4).cast[DType.float32]() - eight
                var lo1 = (raw1 & low_mask).cast[DType.float32]() - eight
                var hi1 = (raw1 >> 4).cast[DType.float32]() - eight
                var x0 = x.unsafe_load[width=W_Q4](xb) * d
                var x1 = x.unsafe_load[width=W_Q4](xb + 16) * d
                var x2 = x.unsafe_load[width=W_Q4](xb + W_Q4) * d
                var x3 = x.unsafe_load[width=W_Q4](xb + 24) * d
                s0 += lo0 * x0
                s1 += hi0 * x1
                s2 += lo1 * x2
                s3 += hi1 * x3
            acc += Float64(((s0 + s1) + (s2 + s3)).reduce_add())
        dst[unsafe_offset=row] = Float32(acc)


def _matmul_q4_halves(
    dst: F32Ptr,
    x: F32Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    bias: F32Ptr,
    has_bias: Bool,
) raises AlofaError:
    """`dst[r] = Σ_c w[r,c] · x[c]`；**按半块切**（一次 16 个值），累加 f32。

    与 `_matmul_q4_wide` 的算术完全一样多，差别只在一个 q4_0 块的 16 个数据字节
    怎么拆成向量：

        本文    一次读满 16 字节 → 低半字节是第 0..15 个值、高半字节是第 16..31 个
                值，**两边各自配上一段连续的 x**（x[0..16] 与 x[16..32]）
        `_wide` 按 8 字节读两趟 → 四段分别配 x[0..8]、x[16..24]、x[8..16]、
                x[24..32]，于是 nibble 的提取与转换要做四遍而不是两遍

    批=1 的 matmul 每权重读一次用一次、**算术受限**（账本：q4 通路只用到它自己
    访存地板的 10.6%），所以这里唯一能省的是**指令数** —— 没有任何访存技巧可言。
    半块切分省掉的正是每块重复两遍的解量化指令。

    ⚠️ 累加是 f32，不是默认通路的 f64（差约 1e-7 量级）。够不够格替掉默认通路，
    取决于 `test_q4_matmul_vec.mojo` 在**真实**用例（`real13` / `down_like`）上
    量出来的偏差 —— 在那之前它只是**被测量对象**，不该接到模型层。

    `bias` 只在 `has_bias` 为真时读，约定同标量版。
    """
    if rows <= 0 or cols <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "matrix dimensions must be positive",
            "rows=" + String(rows) + " cols=" + String(cols),
        )
    if cols % Q4_BLOCK != 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "inner dimension must be a whole number of blocks",
            "cols=" + String(cols) + " block=" + String(Q4_BLOCK),
        )
    var blocks_per_row = cols // Q4_BLOCK
    var low_mask = SIMD[DType.uint8, W_Q4_HALF](0x0F)
    var eight = SIMD[DType.float32, W_Q4_HALF](8.0)
    for row in range(rows):
        var acc = Float64(0)
        if has_bias:
            acc = Float64(bias[unsafe_offset=row])
        var s0 = SIMD[DType.float32, W_Q4_HALF](0)
        var s1 = SIMD[DType.float32, W_Q4_HALF](0)
        for b in range(blocks_per_row):
            var base = (row * blocks_per_row + b) * Q4_BYTES
            var d = block_scale(blocks, base)
            var xb = b * Q4_BLOCK
            var raw = blocks.unsafe_load[width=W_Q4_HALF](base + Q4_SCALE_BYTES)
            # `>> 4` 之后再与一次 0x0F：字节移位在本项目用过的两种宽度（8 / 16）上
            # 都得是**同一个**语义，多这一次掩码是为了不把这个正确性押在编译器
            # 怎么把 u8 移位降成 vpsrlw 上 —— 它一条指令，而装反半字节的 bug 在
            # 差分门上长得跟"数不对"一模一样。
            var lo = (raw & low_mask).cast[DType.float32]() - eight
            var hi = ((raw >> 4) & low_mask).cast[DType.float32]() - eight
            var x0 = x.unsafe_load[width=W_Q4_HALF](xb) * d
            var x1 = x.unsafe_load[width=W_Q4_HALF](xb + W_Q4_HALF) * d
            s0 += lo * x0
            s1 += hi * x1
        acc += Float64((s0 + s1).reduce_add())
        dst[unsafe_offset=row] = Float32(acc)


def _matmul_q4(
    dst: F32Ptr,
    x: F32Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    bias: F32Ptr,
    has_bias: Bool,
) raises AlofaError:
    """同上，累加用 f64 —— 默认通路，与标量版逐位一致。"""
    _matmul_q4_wide[True](dst, x, blocks, rows, cols, bias, has_bias)


def _matmul_q4_f32acc(
    dst: F32Ptr, x: F32Ptr, blocks: Q4_U8, rows: Int, cols: Int
) raises AlofaError:
    """累加用 f32 的实验通路：快约 1.4×，但与标量版差约 1e-7。

    它目前只是**被测量对象**，不是给调用方用的：够不够格替掉默认通路，取决于它的偏差
    在真实用例上被量到多少（`test_q4_matmul_vec.mojo` 里有一条专测这个）。在那之前，
    任何人都不该把它接到模型层。
    """
    _matmul_q4_wide[False](dst, x, blocks, rows, cols, dst, False)


def matmul_q4_f32(
    dst: F32Ptr, x: F32Ptr, blocks: Q4_U8, rows: Int, cols: Int
) raises AlofaError:
    """`dst = W · x`，`W` 为 q4_0 块流，无偏置；向量版。

    默认走**半块切分 + f32 累加**（`_matmul_q4_halves`）：核级比按块内 `j%8` 切 f64
    通道的那条（`_matmul_q4`）**快 1.36×**，并在 `down_proj` 4864×896 上首次比
    `fp32/avx2` 快（1.56×，此前是慢 3.6%）—— 量化能不能带来端到端收益，就卡在
    这个"解量化比省下的字节更贵"的算术上。付的代价是累加精度：与标量版不再逐位
    一致，真实用例（`real13` / `down_like`）上的最大相对偏差 ~5e-8，判据 1e-5。

    ⚠️ 换默认通路是**判据层**的事，不是性能层的事：偏差被
    `test_q4_matmul_vec.mojo` 逐例量出来并打印，别拿"看着差不多"换过来。
    """
    _matmul_q4_halves(dst, x, blocks, rows, cols, dst, False)


def matmul_q4_f32_bias(
    dst: F32Ptr, x: F32Ptr, blocks: Q4_U8, rows: Int, cols: Int, bias: F32Ptr
) raises AlofaError:
    """`dst = W · x + bias`，向量版；Qwen2 的 q/k/v 投影都带偏置。"""
    _matmul_q4(dst, x, blocks, rows, cols, bias, True)
