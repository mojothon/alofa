"""让多个列共用一次 x（去砍那股最大的缓存流）

上一个负结果把下一步指出来了
----------------------------
「f32→f64 加宽转换不是瓶颈」（`scripts/bench_gemm_cvt.mojo`）：把每 tile 的 cvt
从 9.8 M 砍到 1.09 M，结果**慢了 25–30%**。那次改动**顺带把 x 的字节翻倍**了，
而时间跟着变差 ⇒ 瓶颈大概率是 **x 这条缓存流**。因为现在的循环是「列外层、
k 中层、行内层」：

    每列都要把**整个 x tile 重新读一遍** = 896 列 × 8 行 × 4864 × 4 B = **139 MB**

而权重本身一个 tile 只有 **17 MB** —— **x 这条流是权重的 8 倍**。之前所有注意力
都在「权重怎么少读一遍」，其实最大的那股流动的是 x。

这个实验（`V2`）
----------------
**列也分块**：一次性把 `RB` 行的 x 载进来，给 `C` 列共用 ⇒ **x 流量 ÷ C**（C=8 时
139 MB → 17 MB）。代价写在另一边：`RB×C` 个输出各要一个自己的 4 通道 f64 累加器
= RB×C×32 字节（8×8 → 2 KB），**必然溢到 L1/栈** —— 这正是上一个背面 RB=16
学过的那道题（那里换 ⋯ DRAM 流量减半、换来的还是慢 24%）。这里不一样的地方是
**胜算更大**：x 是 139 MB 而不是 17 MB。

**逐位不变是硬约束**：每个输出仍然各自一个 4 通道累加器，lane 由 `k % 4` 定、
尾巴单独收集并在 reduce 之后按 k 升序加 —— 换遍历次序不改变任何一个输出内部的
加法顺序，所以结果必须与 `_gemm_tile[RB]` **逐位相同**（由下面三道自检验）。

判据（口径先定，再看数）
------------------------
  * **同进程配对**：两 arm 交错 **5 轮**，**逐轮轮换出场顺序**；主口径每趟耗时，
    每轮取**配对比值** baseline/V2，报 5 轮 min…max。
  * 矩阵：C ∈ {2, 4, 8} × rows ∈ {8, 16}（RB 固定 8 —— 今天 `_gemm` 派发的档），
    六块独立交错。
  * **采用门槛**：同一个 C 在 **rows=8 与 rows=16 两档**的**保守端**（配对比值
    min）**都 > 1.10** 才考虑写进核；只过一档就照实报，那就是噪声级别，不追。
  * **`src/` 一行没改**（`_gemm_tile` 可以直接 import）。

自检（不通过就一个数都不报）
----------------------------
① V2 与 baseline **逐位相等**；② 两者都与绝对参照物「`rows` 次 `_gemm_tile[1]`」
逐位相等（防①两边共用一个错 —— 注入「所有行写第 0 行」时①会假绿）；③ 每次比较前
把 `dst` 涂 -999 哨兵，**漏写任何一行都会留下一整片哨兵**（不涂的话漏掉的行保留
上一次算出的**正确值**，正负两例逐位相同，门自己就是绿的）。

⚠️ 余数列：这里只跑 `C` 能整除 `cols` 的形状（`down_proj` 896 = 2/4/8 都行）。
真核要处理余数列 —— 忘处理的话哨兵自检会红，这是它存在的理由之一。别据此宣称
"支持任意形状"。

    pixi run mojo run -O2 -I src scripts/bench_gemm_cblock.mojo
"""

from std.os import getenv

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.core.text import parse_int
from alofa.kernels.cpu.avx2 import _gemm_tile

comptime W_F64 = 4

# 形状是**运行时**的（默认 `down_proj` 896×4864；`ALOFA_PROBE_COLS` /
# `ALOFA_PROBE_INNER` 可换）。为什么非要能换：七个投影里只有三个是 896×4864，
# **四个是 896×896**（q/k/v/o），而那个形状里 x tile 只有 28 KB、本来就装得进
# L1 —— 只看一个形状就说「C=16 该写进核」是不允许的。
comptime MAX_ROWS = 32

# 权重副本份数：每趟算一份新的，8 份 = 133 MB ≫ L3（12 MB）。
comptime REGIONS = 8

# 交错轮数。本机 ±10% 噪声，靠**轮数 + 配对**而不是靠挑一轮。
comptime ROUNDS = 5

struct Shape(Copyable, Movable):
    """一个投影的形状：`dst[rows, cols] = x[rows, inner] @ w[cols, inner]ᵀ`。"""

    var cols: Int
    var inner: Int

    def __init__(out self, cols: Int, inner: Int):
        self.cols = cols
        self.inner = inner


def f2(v: Float64) -> String:
    """定点两位；这台机器上更多位数是假精度。"""
    var hundredths = Int(v * Float64(100))
    var frac = hundredths % 100
    var tail = String(frac) if frac >= 10 else "0" + String(frac)
    return String(hundredths // 100) + "." + tail


def interval(ts: List[Float64]) -> String:
    """min…max（`Float64`）。两个边界都要看着 —— 只看下界就是把结论往自己那边掰。"""
    var lo = ts[0]
    var hi = ts[0]
    for i in range(len(ts)):
        if ts[i] < lo:
            lo = ts[i]
        if ts[i] > hi:
            hi = ts[i]
    return f2(lo) + "–" + f2(hi)


def min_max(ns_list: List[Int]) -> String:
    var lo = ns_list[0]
    var hi = ns_list[0]
    for i in range(len(ns_list)):
        if ns_list[i] < lo:
            lo = ns_list[i]
        if ns_list[i] > hi:
            hi = ns_list[i]
    var unit = Float64(1000000)
    return f2(Float64(lo) / unit) + "–" + f2(Float64(hi) / unit) + " ms"


def _gemm_tile_cblock[RB: Int, C: Int](
    po: F32Ptr,
    px: F32Ptr,
    pw: F32Ptr,
    bias: F32Ptr,
    has_bias: Bool,
    row0: Int,
    cols: Int,
    inner: Int,
) -> None:
    """`RB` 行 × `C` 列一块：x 载一次给 `C` 列共用（x 流量 ÷ C）。

    与 `_gemm_tile[RB]` **逐位相同** —— 变的是"谁和谁在同一个时刻一起走"，不是
    任何一个输出内部的加法顺序：每个输出仍有一个 4 通道 f64 累加器（lane 由
    `k % 4` 定）+ 一个尾巴标量；reduce 之后尾巴按 `k` 升序加。

    ⚠️ `cols` 必须是 `C` 的倍数（余数列这里不处理 —— 漏写会留下 -999 哨兵，
    自检会红）。
    """
    var acc = InlineArray[SIMD[DType.float64, W_F64], RB * C](
        fill=SIMD[DType.float64, W_F64](Float64(0))
    )
    var tail = InlineArray[Float64, RB * C](fill=Float64(0))
    var xvv = InlineArray[SIMD[DType.float64, W_F64], RB](
        fill=SIMD[DType.float64, W_F64](Float64(0))
    )
    var cb = 0
    while cb + C <= cols:
        comptime for r in range(RB):
            var ci = 0
            while ci < C:
                acc[r * C + ci] = SIMD[DType.float64, W_F64](Float64(0))
                if has_bias:
                    acc[r * C + ci][0] = Float64(bias[unsafe_offset=cb + ci])
                tail[r * C + ci] = Float64(0)
                ci += 1

        var k = 0
        while k + W_F64 <= inner:
            # x 一次载进来，给下面的 C 列共用 —— 这就是本轮唯一被改的东西。
            comptime for r in range(RB):
                xvv[r] = px.unsafe_load[width=W_F64](
                    (row0 + r) * inner + k
                ).cast[DType.float64]()
            var ci = 0
            while ci < C:
                var wv = pw.unsafe_load[width=W_F64](
                    (cb + ci) * inner + k
                ).cast[DType.float64]()

                comptime for r in range(RB):
                    acc[r * C + ci] += xvv[r] * wv
                ci += 1
            k += W_F64

        comptime for r in range(RB):
            var cm = 0
            while cm < C:
                tail[r * C + cm] = acc[r * C + cm].reduce_add()
                cm += 1
        while k < inner:
            var ct = 0
            while ct < C:
                var wk = Float64(pw[unsafe_offset=(cb + ct) * inner + k])

                comptime for r in range(RB):
                    tail[r * C + ct] += Float64(
                        px[unsafe_offset=(row0 + r) * inner + k]
                    ) * wk
                ct += 1
            k += 1

        comptime for r in range(RB):
            var cs = 0
            while cs < C:
                po[unsafe_offset=(row0 + r) * cols + cb + cs] = Float32(
                    tail[r * C + cs]
                )
                cs += 1
        cb += C


def run_arm[RB: Int, C: Int, USE_CB: Bool](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, sh: Shape
) raises -> None:
    """`rows` 行整个算完；`USE_CB` 为假走 baseline `_gemm_tile[RB]`。

    ⚠️ `C` 恒 ≥ 1 —— 用另一个参数选 arm，是为了让不被选中那条分支不要去实例化
    `InlineArray[…, RB * 0]`（`size 0`）。公道地说这就是搬 Hybridizer 的回填。
    """
    var po = f32_data(TensorView(d_raw, shape2(rows, sh.cols), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(rows, sh.inner), DT_FP32))
    var pw = f32_data(TensorView(w_raw, shape2(sh.cols, sh.inner), DT_FP32))
    var b = 0
    while b < rows:
        if USE_CB:
            _gemm_tile_cblock[RB, C](po, px, pw, po, False, b, sh.cols, sh.inner)
        else:
            _gemm_tile[RB](po, px, pw, po, False, b, sh.cols, sh.inner)
        b += RB


def time_arm[RB: Int, C: Int, USE_CB: Bool](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, sh: Shape
) raises -> Int:
    """在 `REGIONS` 份互不相交的权重上各算一趟，返回**单趟**的纳秒。

    每趟换一份权重 —— 同份权重连着跑会让后几趟落在 L3 里，`bench_thread_matmul`
    第一版就是这么报出「80 GB/s」的（本机上限 ~42 GB/s）。
    """
    var po = f32_data(TensorView(d_raw, shape2(rows, sh.cols), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(rows, sh.inner), DT_FP32))
    var t0 = monotonic_ns()
    for r in range(REGIONS):
        var pw = f32_data(
            TensorView(
                w_raw.unsafe_offset(r * sh.cols * sh.inner * 4), shape2(sh.cols, sh.inner), DT_FP32
            )
        )
        var b = 0
        while b < rows:
            if USE_CB:
                _gemm_tile_cblock[RB, C](po, px, pw, po, False, b, sh.cols, sh.inner)
            else:
                _gemm_tile[RB](po, px, pw, po, False, b, sh.cols, sh.inner)
            b += RB
    var t1 = monotonic_ns()
    _ = po[unsafe_offset=0]
    return (t1 - t0) // REGIONS


def snapshot(rows: Int, d_raw: RawPtr, sh: Shape) raises -> List[Float32]:
    var got = List[Float32]()
    var pd = f32_data(TensorView(d_raw, shape2(rows, sh.cols), DT_FP32))
    for i in range(rows * sh.cols):
        got.append(pd[unsafe_offset=i])
    return got^


def paint(rows: Int, d_raw: RawPtr, sh: Shape) raises -> None:
    """把 `dst` 涂成哨兵 —— 漏写一行 / 漏写余数列会留下一整片 -999。"""
    var pd = f32_data(TensorView(d_raw, shape2(rows, sh.cols), DT_FP32))
    for i in range(rows * sh.cols):
        pd[unsafe_offset=i] = Float32(-999)


def check_parity[RB: Int, C: Int, USE_CB: Bool](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, sh: Shape
) raises -> Bool:
    """① baseline == V2；② 两者 == `rows` 次 `[1]`（绝对参照物）；③ 无哨兵残留。"""
    paint(rows, d_raw, sh)
    run_arm[RB, 1, False](rows, x_raw, w_raw, d_raw, sh)
    var gold = snapshot(rows, d_raw, sh)

    paint(rows, d_raw, sh)
    run_arm[RB, C, USE_CB](rows, x_raw, w_raw, d_raw, sh)
    var got = snapshot(rows, d_raw, sh)
    if len(got) != len(gold):
        return False
    for i in range(len(gold)):
        if got[i] != gold[i]:
            return False
        if got[i] == Float32(-999):
            return False  # 哨兵残留 = 有行（或余数列）没被写

    paint(rows, d_raw, sh)
    var pd = f32_data(TensorView(d_raw, shape2(rows, sh.cols), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(rows, sh.inner), DT_FP32))
    var pw = f32_data(TensorView(w_raw, shape2(sh.cols, sh.inner), DT_FP32))
    for r in range(rows):
        _gemm_tile[1](pd, px, pw, pd, False, r, sh.cols, sh.inner)
    for i in range(len(gold)):
        if pd[unsafe_offset=i] != gold[i]:
            return False
    return True


def bench_pair[C: Int](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, sh: Shape
) raises -> None:
    var ago = List[Int]()
    var new = List[Int]()
    var ratios = List[Float64]()
    print(
        "  C =",
        C,
        " rows =",
        rows,
        "  x tile 被读",
        rows // 8 * (sh.cols // C),
        "遍 vs baseline",
        rows // 8 * sh.cols,
        "遍   累加器状态",
        8 * C * 32,
        "B（寄存器堆 512 B）",
    )
    for rd in range(ROUNDS):
        var old_first = (rd % 2) == 0
        if old_first:
            ago.append(time_arm[8, 1, False](rows, x_raw, w_raw, d_raw, sh))
            new.append(time_arm[8, C, True](rows, x_raw, w_raw, d_raw, sh))
        else:
            new.append(time_arm[8, C, True](rows, x_raw, w_raw, d_raw, sh))
            ago.append(time_arm[8, 1, False](rows, x_raw, w_raw, d_raw, sh))
        var a = ago[len(ago) - 1]
        var b = new[len(new) - 1]
        ratios.append(Float64(a) / Float64(b))
        print(
            "    轮",
            rd + 1,
            "顺序",
            "旧→V2" if old_first else "V2→旧",
            "  旧",
            f2(Float64(a) / 1000000.0),
            "ms   V2",
            f2(Float64(b) / 1000000.0),
            "ms   配对比值",
            f2(Float64(a) / Float64(b)),
            "×",
        )
    var per_a = Float64(0)
    var per_b = Float64(0)
    for k in range(len(ago)):
        per_a += Float64(ago[k]) / Float64(rows) / Float64(len(ago))
        per_b += Float64(new[k]) / Float64(rows) / Float64(len(new))
    print("    每趟 min…max   旧", min_max(ago), "  V2", min_max(new))
    print(
        "    每行（均值）   旧",
        f2(per_a / 1000000.0),
        "ms   V2",
        f2(per_b / 1000000.0),
        "ms",
    )
    print("    配对比值 min…max", interval(ratios), "×（>1 = V2 更快）")
    print("")


def read_shape_env(name: String, fallback: Int) raises -> Int:
    """从环境变量读整数；没设就用默认（`parse_int` 会把空串和 `1e3` 都判错）。"""
    var raw = getenv(name, "")
    if raw == "":
        return fallback
    return parse_int(raw)


def main() raises:
    var sh = Shape(
        read_shape_env("ALOFA_PROBE_COLS", 896),
        read_shape_env("ALOFA_PROBE_INNER", 4864),
    )
    var n_w = sh.cols * sh.inner
    var w_bytes = n_w * 4
    # `C` 必须整除 `cols`（余数列这里不处理 —— 漏写会留下 -999，自检会红）。
    var arena = Arena(
        REGIONS * n_w * 4
        + MAX_ROWS * sh.inner * 4
        + MAX_ROWS * sh.cols * 4
        + 4096
    )
    var w_raw = arena.alloc(REGIONS * n_w * 4)
    var x_raw = arena.alloc(MAX_ROWS * sh.inner * 4)
    var d_raw = arena.alloc(MAX_ROWS * sh.cols * 4)

    var pw = f32_data(
        TensorView(w_raw, shape2(REGIONS * sh.cols, sh.inner), DT_FP32)
    )
    var px = f32_data(TensorView(x_raw, shape2(MAX_ROWS, sh.inner), DT_FP32))
    var i = 0
    while i < REGIONS * n_w:
        pw[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1
    i = 0
    while i < MAX_ROWS * sh.inner:
        px[unsafe_offset=i] = Float32(i % 13) - Float32(6)
        i += 1

    print("=== 让多个列共用一次 x（x 那条缓存流能不能砍掉？） ===")
    print(
        "形状：cols",
        sh.cols,
        " inner",
        sh.inner,
        "  每份权重",
        w_bytes // (1024 * 1024),
        "MB ×",
        REGIONS,
        "份 =",
        REGIONS * w_bytes // (1024 * 1024),
        "MB（≫ L3 12 MB）  交错",
        ROUNDS,
        "轮  RB 固定 8",
    )
    print(
        "baseline 每 tile 从缓存流 x：",
        8 * sh.inner * sh.cols * 4 // (1024 * 1024),
        "MB；权重",
        w_bytes // (1024 * 1024),
        "MB；x tile 本身 8×inner =",
        8 * sh.inner * 4 // 1024,
        "KB（L1 32 KB）",
    )
    print("")

    # 预热 + 自检。任何一处不逐位相等 → 一个数都不报。
    var ok = True
    if not check_parity[8, 2, True](8, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 2 rows = 8")
    if not check_parity[8, 2, True](16, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 2 rows = 16")
    if not check_parity[8, 2, True](32, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 2 rows = 32")
    if not check_parity[8, 4, True](8, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 4 rows = 8")
    if not check_parity[8, 4, True](16, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 4 rows = 16")
    if not check_parity[8, 4, True](32, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 4 rows = 32")
    if not check_parity[8, 8, True](8, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 8 rows = 8")
    if not check_parity[8, 8, True](16, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 8 rows = 16")
    if not check_parity[8, 8, True](32, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 8 rows = 32")
    if not check_parity[8, 16, True](8, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 16 rows = 8")
    if not check_parity[8, 16, True](16, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 16 rows = 16")
    if not check_parity[8, 16, True](32, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 16 rows = 32")
    if not check_parity[8, 32, True](8, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 32 rows = 8")
    if not check_parity[8, 32, True](16, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 32 rows = 16")
    if not check_parity[8, 32, True](32, x_raw, w_raw, d_raw, sh):
        ok = False
        print("  ⚠️ 自检失败：C = 32 rows = 32")
    if not ok:
        print("⚠️ 自检没过 → 下面的时间和比值一律不许引用。")
        arena.keep_alive()
        return

    print("--- C = 2（累加器 16 条，理论上还能贴着寄存器堆）---")
    bench_pair[2](8, x_raw, w_raw, d_raw, sh)
    bench_pair[2](16, x_raw, w_raw, d_raw, sh)
    print("--- C = 4（累加器 32 条 = 1 KB，开始溢到 L1）---")
    bench_pair[4](8, x_raw, w_raw, d_raw, sh)
    bench_pair[4](16, x_raw, w_raw, d_raw, sh)
    print("--- C = 8（累加器 64 条 = 2 KB；x 流量 ÷ 8）---")
    bench_pair[8](8, x_raw, w_raw, d_raw, sh)
    bench_pair[8](16, x_raw, w_raw, d_raw, sh)
    bench_pair[8](32, x_raw, w_raw, d_raw, sh)
    print("--- C = 16（累加器 128 条 = 4 KB；x 流量 ÷ 16）---")
    bench_pair[16](8, x_raw, w_raw, d_raw, sh)
    bench_pair[16](16, x_raw, w_raw, d_raw, sh)
    bench_pair[16](32, x_raw, w_raw, d_raw, sh)
    print("--- C = 32（累加器 256 条 = 8 KB；x 流量 ÷ 32，已比权重小了）---")
    bench_pair[32](8, x_raw, w_raw, d_raw, sh)
    bench_pair[32](16, x_raw, w_raw, d_raw, sh)
    bench_pair[32](32, x_raw, w_raw, d_raw, sh)

    print(
        "判据（跑之前就定死的）：同一个 C 在 rows=8 与 16 **两档**的保守端都 > 1.10 才考虑写进核。"
    )
    print(
        "⚠️ 这是**核级**数（avx2 一条通路、**只有这里这一个形状**），没有端到端数字。"
    )

    # ⚠️ arena 在**最后一次使用处**析构 —— 而分配出来的指针要用到这里之后。
    arena.keep_alive()
