"""RB 抬到 16 值不值？（同进程配对 A/B，判定性探针）

问的是什么
----------
`avx2._gemm` 按 `_gemm_tile[RB]` 分块（`RB` = 8/4/2/1），块内权重只读一遍。于是
一个 `rows` 行的前向把权重读 `ceil(rows / RB)` 遍 —— 2026-09-22 把这条曲线量到
尽头：每行耗时 8 以内约按 1/rows 掉，**8 / 16 / 32 三档区间重合**（机理不是"没
摊薄"，是 `rows=32` 被切成 4 块、权重读 4 遍，每行摊到的字节与 `rows=8` 相同）。

待证伪的假设：把 `RB` 抬到 16 → `rows=32` 只读 2 遍 → **每行字节再减半**。

反方的理由（同样要先写下来，免得事后挑有利的那一边）：`_gemm_tile` 的热状态是
`RB` 个 f64×4 累加器 + `RB` 个 f64 尾巴。`RB=16` 是 512 + 128 字节，而 AVX2 的
寄存器堆只有 16 × 32 = 512 字节且还得装 `wv` / `xv` / 地址 → **必然溢到栈**。
所以这不是"多复用一倍"那么简单，是拿**栈流量换 DRAM 流量**：DRAM 是百纳秒级、
L1 是几个周期，看起来划算，但溢出后每步乘加都要 reload/store 累加器，算术那条
链会被拉长（本机这条本来就落在算术上，见账本）。**谁赢只能量。**

判据（口径先定，再看数）
------------------------
  * **同进程配对**，两个 arm 交错，5 轮，**逐轮轮换出场顺序**（不轮换的话，先跑的
    那个系统地吃亏：冷的页缓存、冷的分支预测都在它那一次）；同一轮里这两个数是
    配对的，机器状态同时作用在两个 arm 上 → 取**每轮的比值**，再取 5 轮的 min…max。
  * 主口径 = **每行耗时**（一趟 ÷ rows），另一个 arm 也一样。
  * **采用 RB=16 的门槛**（先定死）：在 `rows` = 16 与 32 **两档**上，
    **保守端** `RB8_min / RB16_max` **都 > 1.0**（用最不利于 RB=16 的那一端，
    不是取好看的那一端）。任一一档不过 → **不采用**，理由写回账本。
  * 只要 evidence 在这里 RB 假设赢了，下一步才轮到改 `src/alofa/kernels/cpu/avx2.mojo`
    的分块派发 + 给 `test_gemm_batch_layout.mojo` 补 ≥16 的形状。**本脚本不改核。**

自检（不通过就一个数都不报）
----------------------------
① 一块 `_gemm_tile[16]` 必须与两块 `_gemm_tile[8]` **逐位相等** —— 每个输出各自
   一个累加器、按同样的 `k` 次序累加，`RB` 只决定"谁和谁一起走"，浮动运算本身
   一次都没变（差 1 ulp 就是 bug，不是"精度损失"）。
② 两者还要与**绝对参照物**逐位相等：`rows` 次 `_gemm_tile[1]`（防①两边共用同一
   个错 —— 注入一个"所有行写第 0 行"的错，①的两条都会绿）。
③ 每次比较前把 `dst` 涂成 -999：**漏写任何一行都会留下哨兵**；不涂的话漏掉的行
   保留上一次算出的**正确值**，正负两例逐位相同，门自己就是绿的。

边界（别越解释）
----------------
`rows` 只取 16 / 32（都能被 8 与 16 整除 → 没有余数块，纯粹在比 RB）；只看
avx2 通路；`down_proj` 896×4864 一个形状。**这里没有端到端数字**。

    pixi run mojo run -O2 -I src scripts/bench_gemm_rb.mojo
"""

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import _gemm_tile

# `down_proj`：24 层里最大的一个投影。
comptime COLS = 896
comptime INNER = 4864
comptime MAX_ROWS = 32

# 权重副本份数：每趟算一份新的，8 份 = 139 MB ≫ L3（12 MB）。
comptime REGIONS = 8

# 交错轮数。本机 ±10% 噪声，效应若接近就靠**轮数 + 配对**而不是靠挑一轮。
comptime ROUNDS = 5

comptime W_BYTES = COLS * INNER * 4
comptime N_W = COLS * INNER


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


def run_arm[RB: Int](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr
) raises -> None:
    """`rows` 行整个算完，块大小 `RB`（`rows` 必须是 `RB` 的倍数）。"""
    var po = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(rows, INNER), DT_FP32))
    var b = 0
    while b < rows:
        _gemm_tile[RB](po, px, f32_data(
            TensorView(w_raw, shape2(COLS, INNER), DT_FP32)
        ), po, False, b, COLS, INNER)
        b += RB


def time_arm[RB: Int](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr
) raises -> Int:
    """在 `REGIONS` 份互不相交的权重上各算一趟，返回**单趟**的纳秒。

    每趟换一份权重 —— 同份权重连着跑会让后几趟落在 L3 里，`bench_thread_matmul`
    第一版就是这么报出「80 GB/s」的（本机上限 ~42 GB/s）。
    """
    var po = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(rows, INNER), DT_FP32))
    var t0 = monotonic_ns()
    for r in range(REGIONS):
        var pw = f32_data(
            TensorView(
                w_raw.unsafe_offset(r * N_W * 4), shape2(COLS, INNER), DT_FP32
            )
        )
        var b = 0
        while b < rows:
            _gemm_tile[RB](po, px, pw, po, False, b, COLS, INNER)
            b += RB
    var t1 = monotonic_ns()
    _ = po[unsafe_offset=0]
    return (t1 - t0) // REGIONS


def snapshot(rows: Int, d_raw: RawPtr) raises -> List[Float32]:
    var got = List[Float32]()
    var pd = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))
    for i in range(rows * COLS):
        got.append(pd[unsafe_offset=i])
    return got^


def paint(rows: Int, d_raw: RawPtr) raises -> None:
    """把 `dst` 涂成哨兵 —— 漏写一行会留下一整片 -999。"""
    var pd = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))
    for i in range(rows * COLS):
        pd[unsafe_offset=i] = Float32(-999)


def check_exact(
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr
) raises -> Bool:
    """① `[16]` 一块 == `[8]` 两块；② 两者 == `rows` 次 `[1]`；③ 无哨兵残留。"""
    paint(rows, d_raw)
    run_arm[8](rows, x_raw, w_raw, d_raw)
    var ref8 = snapshot(rows, d_raw)

    paint(rows, d_raw)
    run_arm[16](rows, x_raw, w_raw, d_raw)
    var got16 = snapshot(rows, d_raw)
    if len(got16) != len(ref8):
        return False
    for i in range(len(ref8)):
        if got16[i] != ref8[i]:
            return False
        if got16[i] == Float32(-999):
            return False  # 哨兵残留 = 有行没被写

    paint(rows, d_raw)
    var pd = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(rows, INNER), DT_FP32))
    var pw = f32_data(TensorView(w_raw, shape2(COLS, INNER), DT_FP32))
    for r in range(rows):
        _gemm_tile[1](pd, px, pw, pd, False, r, COLS, INNER)
    for i in range(len(ref8)):
        if pd[unsafe_offset=i] != ref8[i]:
            return False
    return True


def main() raises:
    var arena = Arena(
        REGIONS * N_W * 4 + MAX_ROWS * INNER * 4 + MAX_ROWS * COLS * 4 + 4096
    )
    var w_raw = arena.alloc(REGIONS * N_W * 4)
    var x_raw = arena.alloc(MAX_ROWS * INNER * 4)
    var d_raw = arena.alloc(MAX_ROWS * COLS * 4)

    var pw = f32_data(TensorView(w_raw, shape2(REGIONS * COLS, INNER), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(MAX_ROWS, INNER), DT_FP32))
    var i = 0
    while i < REGIONS * N_W:
        pw[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1
    i = 0
    while i < MAX_ROWS * INNER:
        px[unsafe_offset=i] = Float32(i % 13) - Float32(6)
        i += 1

    print("=== RB 抬到 16：多复用一倍能不能赢过累加器溢出？ ===")
    print(
        "形状 down_proj",
        COLS,
        "×",
        INNER,
        "  每份权重",
        W_BYTES // (1024 * 1024),
        "MB ×",
        REGIONS,
        "份 =",
        REGIONS * W_BYTES // (1024 * 1024),
        "MB（≫ L3 12 MB）  交错",
        ROUNDS,
        "轮",
    )
    print("")

    var rows_list = List[Int]()
    rows_list.append(16)
    rows_list.append(32)

    # 预热 + 自检。任何一处不逐位相等 → 一个数都不报。
    var ok = True
    for ri in range(len(rows_list)):
        var rows = rows_list[ri]
        if not check_exact(rows, x_raw, w_raw, d_raw):
            ok = False
            print("  rows =", rows, "  ⚠️ 自检失败：不逐位相等 / 有哨兵残留")
    if not ok:
        print("⚠️ 自检没过 → 下面的时间和比值一律不许引用。")
        return

    for ri in range(len(rows_list)):
        var rows = rows_list[ri]
        var t8 = List[Int]()
        var t16 = List[Int]()
        var ratios = List[Float64]()
        print("  rows =", rows, "（权重被读的遍数：RB=8 →", rows // 8, "遍 / RB=16 →", rows // 16, "遍）")
        for rd in range(ROUNDS):
            var eight_first = (rd % 2) == 0
            if eight_first:
                t8.append(time_arm[8](rows, x_raw, w_raw, d_raw))
                t16.append(time_arm[16](rows, x_raw, w_raw, d_raw))
            else:
                t16.append(time_arm[16](rows, x_raw, w_raw, d_raw))
                t8.append(time_arm[8](rows, x_raw, w_raw, d_raw))
            var a8 = t8[len(t8) - 1]
            var a16 = t16[len(t16) - 1]
            ratios.append(Float64(a8) / Float64(a16))
            print(
                "    轮",
                rd + 1,
                "顺序",
                "8→16" if eight_first else "16→8",
                "  RB=8 一趟",
                f2(Float64(a8) / 1000000.0),
                "ms   RB=16 一趟",
                f2(Float64(a16) / 1000000.0),
                "ms   配对比值",
                f2(Float64(a8) / Float64(a16)),
                "×",
            )
        var per8 = Float64(0)
        var per16 = Float64(0)
        for k in range(len(t8)):
            per8 += Float64(t8[k]) / Float64(rows) / Float64(len(t8))
            per16 += Float64(t16[k]) / Float64(rows) / Float64(len(t16))
        print("    每趟 min…max   RB=8", min_max(t8), "  RB=16", min_max(t16))
        print("    每行（均值）   RB=8", f2(per8 / 1000000.0), "ms   RB=16", f2(per16 / 1000000.0), "ms")
        print("    配对比值 min…max", interval(ratios), "×（>1 = RB=16 更快）")
        print("")

    print("判据（跑之前就定死的）：rows=16 与 32 **两档**的保守端 RB8_min/RB16_max 都 > 1.0 才采用 RB=16。")
    print("⚠️ 这是**核级**数（`down_proj` 一个形状、avx2 一条通路），没有端到端数字。")

    # ⚠️ arena 在**最后一次使用处**析构 —— 而分配出来的指针要用到这里之后。
    #    不写这行，`arena` 的最后一个使用点落在最后一次 `alloc` 上，写到一半
    #    mmap 就被解映射（症状是 SIGSEGV，落在第一次写 `w` 的那一行）。
    arena.keep_alive()
