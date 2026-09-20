"""批 / 预填 GEMM 的**换序**差分门：一次算 `rows` 行 == 逐行算 `rows` 次。

为什么需要这道门
----------------
`avx2._gemm` 原来是「行外层、列内层」，`w` 的每一列在每一行上都被重读一遍 ——
一个 `rows` 行的前向把整个权重矩阵读了 `rows` 遍。`scripts/bench_gemm_rows.mojo`
量到了这件事（`rows=8` 一趟 23.31 ms vs `rows=1` 3.95 ms，每行耗时不摊薄）。

改成「`RB` 行一块、列外层、`k` 中层、行内层」之后，权重每块只读一遍 ——
但**换序本身会改变浮点求和的组合方式**，所以这里守的是：

    换序**不得**改变任何一个输出的那一条加法链。

判据因此用**精确相等**，不用容差：每个输出各自一个 f64×4 累加器、按同样的
`k` 次序累加，切分与换序不改变任何一次浮点运算 —— 差 1 ulp 就是 bug，
不是"精度损失"。

形状为什么这么挑（换序最容易漏的是分块余数）
--------------------------------------------
`_gemm` 按剩余行数派发 `RB` = 8/4/2/1，所以 `rows` 取 1、2、3、5、7、8、9：
3 = 2+1、5 = 4+1、7 = 4+2+1、9 = 8+1 —— **四种分块都被走到**，且都带余数块。
`inner` 取 4 的倍数（896 这种真形状）与**非** 4 的倍数（7、13、17）两种：
后者才会走 `k` 的标量尾巴，而尾巴在换序里是单独一条路径（`tail[r]` 的初值是
`reduce_add()` 的结果，不是 0 —— `0+t0+t1` 与 `r+t0+t1` 是两个数）。

两条通路都测：`linear`（无偏置）与 `linear_bias`（Qwen2 的 q/k/v 都带偏置，
偏置进的是 0 号**通道**，换序后仍必须只加一次）。

⚠️ 上面两条的参照物是"逐行调用**同一个 avx2 核**"，所以它守得住的只有
「换序不改变结果」—— 一处两个路径共用的错（比如 `k` 尾巴的初值少了
`reduce_add()`）两边一起错，逐位比较照样全绿（2026-09-20 实测：注入这处错，
那两条**仍然 PASS**）。所以文件里还挂了第三条：与**没动过**的标量后端比，
它是绝对参照物，能接住这一类。

负向对照（一个不会失败的门等于没门）
------------------------------------
- **少乘最后一个 `k`**：走同一套判据、同一批形状，必须**被拒**。它证明判据
  真的在看全部 `inner` —— 否则"换序时把尾巴丢了"也会是绿的。
- **哨兵**：`rows` 一次算完之前先把 `dst` 涂成 -999。不涂的话，"某一行没算"
  会保留上一次全量算出的**正确值**，正负两例逐位相同，门自己就是绿的
  （2026-09-20 在分片门上踩过）。

Run:
    pixi run mojo run -O0 -I src tests/unit/test_gemm_batch_layout.mojo
"""

from std.testing import TestSuite, assert_true

from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import linear, linear_bias
from alofa.kernels.cpu.scalar import linear as linear_scalar

# rows：把 RB = 8/4/2/1 四种分块连同余数块都走到（3=2+1、5=4+1、7=4+2+1、9=8+1）。
comptime ROW_CASES = 9

# cols / inner：真形状（896、4864 都是 4 的倍数）加上非 4 的倍数（走 k 尾巴）。
comptime SHAPES = 4

comptime SENTINEL = Float32(-999)

# 全项目统一判据：`1e-5 × max(1, |ref|)`。只在"与标量后端比"那两条上用。
comptime REL_TOL = Float64(1e-5)


def shape_cases() -> List[Int]:
    """`[cols, inner]` 交替排：`0/1` 是 4 的倍数，`2/3` 不是。"""
    var v = List[Int]()
    v.append(32)  # cols
    v.append(64)  # inner
    v.append(17)  # cols
    v.append(7)  # inner
    v.append(5)  # cols
    v.append(13)  # inner
    v.append(8)  # cols
    v.append(17)  # inner
    return v^


def fill(p: F32Ptr, n: Int, seed: Int) -> None:
    """确定性的伪随机，够把每一格区分开；不依赖任何外部夹具。"""
    var i = 0
    while i < n:
        var v = (i * 37 + seed * 11) % 19
        p[unsafe_offset=i] = Float32(v) - Float32(9)
        i += 1


def paint(p: F32Ptr, n: Int) -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = SENTINEL
        i += 1


def same(p: F32Ptr, q: F32Ptr, n: Int) -> Bool:
    var i = 0
    while i < n:
        if p[unsafe_offset=i] != q[unsafe_offset=i]:
            return False
        i += 1
    return True


def batch_layout_case(
    rows: Int, cols: Int, inner: Int, with_bias: Bool
) raises -> Bool:
    """一次算 `rows` 行 vs 逐行算 `rows` 次，逐位比较。"""
    # ⚠️ `Arena.alloc` 按 64 字节对齐**累加**，容量算短会静默错位 → 每项都算上、
    # 再留 512 字节给对齐填充。
    var arena = Arena(
        (rows * inner + cols * inner + 2 * rows * cols + cols) * 4 + 512
    )
    var x_raw = arena.alloc(rows * inner * 4)
    var w_raw = arena.alloc(cols * inner * 4)
    var d_raw = arena.alloc(rows * cols * 4)
    var r_raw = arena.alloc(rows * cols * 4)
    var b_raw = arena.alloc(cols * 4)

    var x = TensorView(x_raw, shape2(rows, inner), DT_FP32)
    var w = TensorView(w_raw, shape2(cols, inner), DT_FP32)
    var d = TensorView(d_raw, shape2(rows, cols), DT_FP32)
    var r = TensorView(r_raw, shape2(rows, cols), DT_FP32)
    var b = TensorView(b_raw, shape2(1, cols), DT_FP32)

    fill(f32_data(x), rows * inner, 3)
    fill(f32_data(w), cols * inner, 5)
    fill(f32_data(b), cols, 7)

    # ① 参考：`rows` 次 `rows=1`。
    paint(f32_data(r), rows * cols)
    for i in range(rows):
        var di = TensorView(
            r_raw.unsafe_offset(i * cols * 4), shape2(1, cols), DT_FP32
        )
        var xi = TensorView(
            x_raw.unsafe_offset(i * inner * 4), shape2(1, inner), DT_FP32
        )
        if with_bias:
            linear_bias(di, xi, w, b)
        else:
            linear(di, xi, w)

    # ② 被测：一次算 `rows` 行。涂哨兵在前 —— 漏掉任何一行都会留下 -999。
    paint(f32_data(d), rows * cols)
    if with_bias:
        linear_bias(d, x, w, b)
    else:
        linear(d, x, w)

    var ok = same(f32_data(d), f32_data(r), rows * cols)
    arena.keep_alive()
    return ok


def short_by_one(rows: Int, cols: Int, inner: Int) raises -> Bool:
    """负向对照：`inner` 少乘最后一个 `k`，必须**不等于**参考。

    走的是同一套判据（逐位相等），改的只是求和的项数 —— 所以它证明判据真的
    在看全部 `inner`：换序时把尾巴丢了，这里会当场红。
    """
    var arena = Arena(
        (rows * inner + cols * inner + 2 * rows * cols) * 4 + 512
    )
    var x_raw = arena.alloc(rows * inner * 4)
    var w_raw = arena.alloc(cols * inner * 4)
    var d_raw = arena.alloc(rows * cols * 4)
    var r_raw = arena.alloc(rows * cols * 4)

    var x = TensorView(x_raw, shape2(rows, inner), DT_FP32)
    var w = TensorView(w_raw, shape2(cols, inner), DT_FP32)
    var d = TensorView(d_raw, shape2(rows, cols), DT_FP32)
    var r = TensorView(r_raw, shape2(rows, cols), DT_FP32)
    fill(f32_data(x), rows * inner, 3)
    fill(f32_data(w), cols * inner, 5)

    linear(d, x, w)
    # 参考：同一个形状，但 `inner` 少一格 —— 权重最后一列被丢掉。
    var ws = TensorView(w_raw, shape2(cols, inner - 1), DT_FP32)
    var xs = TensorView(x_raw, shape2(rows, inner - 1), DT_FP32)
    var ds = TensorView(d_raw, shape2(rows, cols), DT_FP32)
    linear(r, xs, ws)
    _ = ds
    var differs = not same(f32_data(d), f32_data(r), rows * cols)
    arena.keep_alive()
    return differs


def test_batch_gemm_is_bit_identical_to_row_by_row() raises:
    """一次算 `rows` 行必须与逐行算**逐位相等**（换序不改变加法链）。"""
    var shapes = shape_cases()
    for s in range(SHAPES):
        var cols = shapes[2 * s]
        var inner = shapes[2 * s + 1]
        for rows in range(1, ROW_CASES + 1):
            assert_true(
                batch_layout_case(rows, cols, inner, False),
                "rows=" + String(rows) + " cols=" + String(cols) + " inner=" + String(inner),
            )


def test_batch_gemm_with_bias_is_bit_identical() raises:
    """同一条性质，走带偏置的 `linear_bias`（偏置进 0 号通道，只加一次）。"""
    var shapes = shape_cases()
    for s in range(SHAPES):
        var cols = shapes[2 * s]
        var inner = shapes[2 * s + 1]
        for rows in range(1, ROW_CASES + 1):
            assert_true(
                batch_layout_case(rows, cols, inner, True),
                "bias rows="
                + String(rows)
                + " cols="
                + String(cols)
                + " inner="
                + String(inner),
            )


def agrees_with_scalar(
    rows: Int, cols: Int, inner: Int, drop_last: Bool
) raises -> Bool:
    """与**未改动**的标量后端比（全项目统一判据 `1e-5 × max(1, |ref|)`）。

    ⚠️ 上面两条的参照物是"逐行调用同一个 avx2 核"，它只能守住**换序不改变结果**
    —— 一处两个路径共用的错（比如尾巴的初值少了 `reduce_add()`）两边一起错，
    逐位比较照样全绿。所以这里再挂一个**绝对**参照物：标量后端这次没动，
    且它的 `k` 是逐格累加的，不共享 avx2 的 4 通道分组。
    """
    var arena = Arena((rows * inner + cols * inner + 2 * rows * cols) * 4 + 512)
    var x_raw = arena.alloc(rows * inner * 4)
    var w_raw = arena.alloc(cols * inner * 4)
    var a_raw = arena.alloc(rows * cols * 4)
    var s_raw = arena.alloc(rows * cols * 4)
    var x = TensorView(x_raw, shape2(rows, inner), DT_FP32)
    var w = TensorView(w_raw, shape2(cols, inner), DT_FP32)
    var a = TensorView(a_raw, shape2(rows, cols), DT_FP32)
    var s = TensorView(s_raw, shape2(rows, cols), DT_FP32)
    fill(f32_data(x), rows * inner, 3)
    fill(f32_data(w), cols * inner, 5)
    if drop_last:
        # 负向对照：avx2 侧少乘最后一个 `k`，标量侧是完整的。
        linear(
            a,
            TensorView(x_raw, shape2(rows, inner - 1), DT_FP32),
            TensorView(w_raw, shape2(cols, inner - 1), DT_FP32),
        )
    else:
        linear(a, x, w)
    linear_scalar(s, x, w)
    var pa = f32_data(a)
    var ps = f32_data(s)
    var i = 0
    while i < rows * cols:
        # ⚠️ `ref` 是参数约定关键字，不能当变量名（`expected argument name`）。
        var want = Float64(ps[unsafe_offset=i])
        var got = Float64(pa[unsafe_offset=i])
        var unit = Float64(1)
        if want > unit:
            unit = want
        elif want < -unit:
            unit = -want
        if (got - want) * (got - want) > (REL_TOL * unit) * (REL_TOL * unit):
            arena.keep_alive()
            return False
        i += 1
    arena.keep_alive()
    return True


def test_non_four_inner_agrees_with_the_scalar_backend() raises:
    """`inner` 不是 4 的倍数时（会走 `k` 的标量尾巴）与标量后端一致。

    真形状（896 / 4864）都是 4 的倍数 → **尾巴在真实模型里一次都不会被跑到**，
    于是"尾巴写错了"在这里是唯一能被抓住的地方 —— 这正是 2026-09-20 那个
    「夹具形状替 bug 开后门」的同一类坑。
    """
    var shapes = shape_cases()
    for s in range(SHAPES):
        var cols = shapes[2 * s]
        var inner = shapes[2 * s + 1]
        for rows in range(1, ROW_CASES + 1):
            assert_true(
                agrees_with_scalar(rows, cols, inner, False),
                "scalar rows="
                + String(rows)
                + " cols="
                + String(cols)
                + " inner="
                + String(inner),
            )


def test_the_scalar_agreement_rejects_a_dropped_k() raises:
    """负向对照：少乘最后一个 `k` 在**同一个容差判据**下必须被拒。

    它证明上一条不是"两边都错所以一样" —— 判据真的在看全部 `inner`。
    """
    var shapes = shape_cases()
    for s in range(SHAPES):
        var cols = shapes[2 * s]
        var inner = shapes[2 * s + 1]
        for rows in range(1, ROW_CASES + 1):
            assert_true(
                not agrees_with_scalar(rows, cols, inner, True),
                "负向对照失效 rows="
                + String(rows)
                + " cols="
                + String(cols)
                + " inner="
                + String(inner),
            )


def test_dropping_the_last_k_is_rejected() raises:
    """负向对照：少乘最后一个 `k` 必须被拒 —— 否则判据是瞎的。"""
    var shapes = shape_cases()
    for s in range(SHAPES):
        var cols = shapes[2 * s]
        var inner = shapes[2 * s + 1]
        for rows in range(1, ROW_CASES + 1):
            assert_true(
                short_by_one(rows, cols, inner),
                "负向对照失效 rows="
                + String(rows)
                + " cols="
                + String(cols)
                + " inner="
                + String(inner),
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
