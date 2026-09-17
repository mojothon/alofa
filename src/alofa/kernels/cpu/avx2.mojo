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
"""

from std.math import exp, sqrt

from alofa.core.error import ERR_SHAPE_MISMATCH, AlofaError
from alofa.core.tensor import F32Ptr, TensorView, f32_data
from alofa.kernels.cpu.scalar import (
    cols_of,
    expect_matrix,
    expect_vector,
    rows_of,
)

# 通道数。f32 取 8（AVX2 一条 ymm 装 8 个单精度），f64 取 4（双精度减半）。
comptime W_F32 = 8
comptime W_F64 = 4


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
            var acc = SIMD[DType.float64, W_F64](Float64(0))
            if has_bias:
                # 偏置只加**一次**：放在 0 号通道里，其余通道是 0。播满四条通道
                # 会让偏置被加四遍 —— 那不是"差一点"，是错四倍，而且只在带偏置
                # 的投影上错（Qwen2 的 q/k/v 恰好都是），不带的那些全对，于是
                # 很容易被当成"某个 head 的问题"。
                acc[0] = Float64(bias[unsafe_offset=col])
            var k = 0
            while k + W_F64 <= inner:
                var xv = px.unsafe_load[width=W_F64](x_base + k).cast[
                    DType.float64
                ]()
                var wv = pw.unsafe_load[width=W_F64](w_base + k).cast[
                    DType.float64
                ]()
                acc += xv * wv
                k += W_F64
            var total = acc.reduce_add()
            while k < inner:
                total += Float64(px[unsafe_offset=x_base + k]) * Float64(
                    pw[unsafe_offset=w_base + k]
                )
                k += 1
            po[unsafe_offset=row * cols + col] = Float32(total)


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
