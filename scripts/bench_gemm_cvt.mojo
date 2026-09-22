"""瓶颈是不是「f32 → f64 的加宽转换」？（能判别的差异实验）

问的是什么
----------
上一轮把 `RB` 抬到 16 的假设否掉了，并得到一条推论：`RB=8` 上 **DRAM 带宽已经
不是瓶颈**（权重流量减半反而慢 24%）。那时间花在哪？读 `_gemm_tile[RB]` 的内层，
按 `down_proj` 896×4864、`rows=8` 一个 tile 点一下**每 tile 的指令数**：

    VMFMADD（256b f64 乘加）896 × 1216 × RB   = 8.7 M
    VCVTPS2PD（f32→f64 加宽）896 × 1216 ×(RB+1) = 9.8 M   ← **比乘加本身还多**
    load（w 一遍 + x 每个 r 各一遍）            = 9.8 M   → 9/组 ÷ 2 口 = 4.9 M 周期

观测是 ~12 M 周期/tile。三条候选瓶颈里，只有 cvt 这一条的量级对得上（≈80%）——
但**这是纸上的数**。本机**没有 `perf`**（`perf: command not found`），`perf_event_paranoid=4`
也不打算动，所以不能用硬件计数直接看端口占用，只能做**机制相反的实验**。

这个实验为什么能判别（`V1`）
----------------------------
内层现在是「列外层、k 中层、行内层」，于是**每一列都要把整个 x tile 重新 load
并重新 cvt 一遍**。改成：**一个 tile 只把 x 转一次**（预算进 RB×inner 的 f64
scratch），内层直接 `load` 宽操作数 —— cvt 从 9.8 M 掉到 ~1.09 M（只剩 `w` 那一
条），**load 指令数不变**（ whence x 的字节翻倍：4 → 8 字节/元素）。

两边方向的 predictions 相反，所以一个数能把两种机理分开：
  * **若是 cvt 受限** → `V1` 明显更快（估算上限 ~2×，实际会被别的口接住）；
  * **若是缓存带宽受限**（x tile 在 RB=8 时是 156 KB > L1 32 KB，`V1` 变 311 KB >
    L2 256 KB）→ `V1` **更慢**，而且 RB 越小越接近打平（tile 小了字节翻倍痛得少）；
  * **两者都受限** → 中间来回，那就报出来，不去挑好看的那一边。

反方理由（同样先写下来）：`V1` 换来字节翻倍 + 多一趟写 scratch。RB=8 时 x tile 从
L2 能装（156 KB）变成**装不下**（311 KB > 256 KB）→ 大概率掉到 L3。**预计最好看的
结果出现在小的 RB 上**，这也是为什么这次把 RB=4 也一起量了（顺带得出 `V1` 自己
的 RB 曲线）。

判据（口径先定，再看数）
------------------------
  * **同进程配对**：两个 arm 交错 **5 轮**，**逐轮轮换出场顺序**（不轮换则先跑的
    那个系统地吃亏）。主口径 = 每趟耗时；每轮取**配对比值** baseline/V1，报 5 轮
    的 min…max。
  * 矩阵：RB ∈ {4, 8} × rows ∈ {8, 16}，共 4 块，每块独立交错。
  * **采用 V1 的门槛**：4 块的**保守端**（配对比值的 min）**全部 > 1.10**。
    只有一部分过时 → **照实报**，只对稳定的那一档考虑动 `src/`（且要带背书门）。
  * **`src/` 一行没改**（`_gemm_tile` 下划线不是访问控制，可以直接 import）。
    赢了才是下一步去改 `src/alofa/kernels/cpu/avx2.mojo` + 给形状 ≥16 的差分门补张量。

自检（不通过就一个数都不报）
----------------------------
① `V1` 与 baseline **逐位相等**（`Float64(px[..])` 和预转出来的 f64 是同一个数、
`k` 次序一次都没变 —— 递归相加的顺序完全相同，差 1 ulp 就是 bug）。这条能抓住
比如「scratch 行偏移算错」这类 bug。
② 两者都要与**绝对参照物**逐位相等：`rows` 次 `_gemm_tile[1]`。这条防①两边共用
同一个错（注入「所有行写第 0 行」时①会假绿）。
③ 每次比较前把 `dst` 涂成 -999：**漏写任何一行都会留下哨兵**；不涂的话漏掉的行
保留上一次算出的**正确值**，正负两例逐位相同，门自己就是绿的。

边界（别越解释）
----------------
只 `down_proj` 一个形状；只 avx2 这一条通路（那条不进位/`_gemm` 的 scalar 通路没
动，q4 通路没动）；**这里没有端到端数字**，也没有 flops/toks 的换算。

    pixi run mojo run -O2 -I src scripts/bench_gemm_cvt.mojo
"""

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import _gemm_tile

comptime W_F64 = 4
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]

# `down_proj`：24 层里最大的一个投影。
comptime COLS = 896
comptime INNER = 4864
comptime MAX_ROWS = 16

# 权重副本份数：每趟算一份新的，8 份 = 133 MB ≫ L3（12 MB）。
comptime REGIONS = 8

# 交错轮数。本机 ±10% 噪声，靠**轮数 + 配对**而不是靠挑一轮。
comptime ROUNDS = 5

comptime N_W = COLS * INNER
comptime W_BYTES = N_W * 4


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


def _gemm_tile_precvt[RB: Int](
    po: F32Ptr,
    px: F32Ptr,
    pw: F32Ptr,
    bias: F32Ptr,
    has_bias: Bool,
    row0: Int,
    cols: Int,
    inner: Int,
    xf: F64Ptr,
) -> None:
    """与 `_gemm_tile[RB]` **逐位相同**，但 x 只转一次 f64。

    差别就一处：baseline 在**每一列**里都把 `x` 重新 `load` + `VCVTPS2PD` 一遍，
    这里改成进循环前把 `RB × inner` 的 tile 转一次（内容写进调用者的 `xf`，大小
    `RB × inner` 个 f64）。`k` 的次序、每个输出各自的累加器、尾巴的顺序全都没动 →
    结果必须逐位相等，这一点由 `check_parity` 拿着 `-999` 哨兵验。
    """
    # 一个 tile 一次：39 K 次标量 store（in lines ~ M 级的内层里可忽略）。
    comptime for r in range(RB):
        var base = (row0 + r) * inner
        var qq = 0
        while qq < inner:
            xf[unsafe_offset=r * inner + qq] = Float64(
                px[unsafe_offset=base + qq]
            )
            qq += 1

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
                acc[r] += xf.unsafe_load[width=W_F64](r * inner + k) * wv
            k += W_F64

        comptime for r in range(RB):
            tail[r] = acc[r].reduce_add()
        while k < inner:
            var wk = Float64(pw[unsafe_offset=w_base + k])

            comptime for r in range(RB):
                tail[r] += xf[unsafe_offset=r * inner + k] * wk
            k += 1

        comptime for r in range(RB):
            po[unsafe_offset=(row0 + r) * cols + col] = Float32(tail[r])


def run_arm[RB: Int, PREC: Bool](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, xf: F64Ptr
) raises -> None:
    """`rows` 行整个算完，块大小 `RB`（`rows` 必须是 `RB` 的倍数）。"""
    var po = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(rows, INNER), DT_FP32))
    var pw = f32_data(TensorView(w_raw, shape2(COLS, INNER), DT_FP32))
    var b = 0
    while b < rows:
        if PREC:
            _gemm_tile_precvt[RB](po, px, pw, po, False, b, COLS, INNER, xf)
        else:
            _gemm_tile[RB](po, px, pw, po, False, b, COLS, INNER)
        b += RB


def time_arm[RB: Int, PREC: Bool](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, xf: F64Ptr
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
            if PREC:
                _gemm_tile_precvt[RB](po, px, pw, po, False, b, COLS, INNER, xf)
            else:
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


def check_parity[RB: Int](
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, xf: F64Ptr
) raises -> Bool:
    """① baseline == `V1`；② 两者 == `rows` 次 `[1]`（绝对参照物）；③ 无哨兵残留。"""
    paint(rows, d_raw)
    run_arm[RB, False](rows, x_raw, w_raw, d_raw, xf)
    var gold = snapshot(rows, d_raw)

    paint(rows, d_raw)
    run_arm[RB, True](rows, x_raw, w_raw, d_raw, xf)
    var got = snapshot(rows, d_raw)
    if len(got) != len(gold):
        return False
    for i in range(len(gold)):
        if got[i] != gold[i]:
            return False
        if got[i] == Float32(-999):
            return False  # 哨兵残留 = 有行没被写

    paint(rows, d_raw)
    var pd = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(rows, INNER), DT_FP32))
    var pw = f32_data(TensorView(w_raw, shape2(COLS, INNER), DT_FP32))
    for r in range(rows):
        _gemm_tile[1](pd, px, pw, pd, False, r, COLS, INNER)
    for i in range(len(gold)):
        if pd[unsafe_offset=i] != gold[i]:
            return False
    return True


def bench_pair[RB: Int](
    rows: Int,
    x_raw: RawPtr,
    w_raw: RawPtr,
    d_raw: RawPtr,
    xf: F64Ptr,
) raises -> None:
    var ago = List[Int]()
    var prec = List[Int]()
    var ratios = List[Float64]()
    print(
        "  RB =",
        RB,
        " rows =",
        rows,
        "  tiles =",
        rows // RB,
        "  x tile：baseline",
        RB * INNER * 4 // 1024,
        "KB / V1",
        RB * INNER * 8 // 1024,
        "KB（L1 32 KB、L2 256 KB）",
    )
    for rd in range(ROUNDS):
        var old_first = (rd % 2) == 0
        if old_first:
            ago.append(time_arm[RB, False](rows, x_raw, w_raw, d_raw, xf))
            prec.append(time_arm[RB, True](rows, x_raw, w_raw, d_raw, xf))
        else:
            prec.append(time_arm[RB, True](rows, x_raw, w_raw, d_raw, xf))
            ago.append(time_arm[RB, False](rows, x_raw, w_raw, d_raw, xf))
        var a = ago[len(ago) - 1]
        var b = prec[len(prec) - 1]
        ratios.append(Float64(a) / Float64(b))
        print(
            "    轮",
            rd + 1,
            "顺序",
            "旧→V1" if old_first else "V1→旧",
            "  旧",
            f2(Float64(a) / 1000000.0),
            "ms   V1",
            f2(Float64(b) / 1000000.0),
            "ms   配对比值",
            f2(Float64(a) / Float64(b)),
            "×",
        )
    var per_a = Float64(0)
    var per_b = Float64(0)
    for k in range(len(ago)):
        per_a += Float64(ago[k]) / Float64(rows) / Float64(len(ago))
        per_b += Float64(prec[k]) / Float64(rows) / Float64(len(prec))
    print("    每趟 min…max   旧", min_max(ago), "  V1", min_max(prec))
    print(
        "    每行（均值）   旧",
        f2(per_a / 1000000.0),
        "ms   V1",
        f2(per_b / 1000000.0),
        "ms",
    )
    print("    配对比值 min…max", interval(ratios), "×（>1 = V1 更快）")
    print("")


def main() raises:
    var arena = Arena(
        REGIONS * N_W * 4
        + MAX_ROWS * INNER * 4
        + MAX_ROWS * INNER * 8
        + MAX_ROWS * COLS * 4
        + 4096
    )
    var w_raw = arena.alloc(REGIONS * N_W * 4)
    var x_raw = arena.alloc(MAX_ROWS * INNER * 4)
    var xf_raw = arena.alloc(MAX_ROWS * INNER * 8)
    var d_raw = arena.alloc(MAX_ROWS * COLS * 4)
    var xf = xf_raw.unsafe_bitcast[Float64]()

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

    print("=== x 的加宽转换是不是瓶颈？（同进程配对 A/B） ===")
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
    print("纸上估算（每 tile，rows=8）：VMFMADD 8.7 M / VCVTPS2PD 9.8 M / load 9.8 M；观测 ~12 M 周期")
    print("")

    # 预热 + 自检。任何一处不逐位相等 → 一个数都不报。
    var ok = True
    if not check_parity[8](8, x_raw, w_raw, d_raw, xf):
        ok = False
        print("  ⚠️ 自检失败 RB=8 rows=8")
    if not check_parity[8](16, x_raw, w_raw, d_raw, xf):
        ok = False
        print("  ⚠️ 自检失败 RB=8 rows=16")
    if not check_parity[4](8, x_raw, w_raw, d_raw, xf):
        ok = False
        print("  ⚠️ 自检失败 RB=4 rows=8")
    if not check_parity[4](16, x_raw, w_raw, d_raw, xf):
        ok = False
        print("  ⚠️ 自检失败 RB=4 rows=16")
    if not ok:
        print("⚠️ 自检没过 → 下面的时间和比值一律不许引用。")
        arena.keep_alive()
        return

    print("--- RB = 8（今天 `_gemm` 派发的档）---")
    bench_pair[8](8, x_raw, w_raw, d_raw, xf)
    bench_pair[8](16, x_raw, w_raw, d_raw, xf)
    print("--- RB = 4（x tile 小一半，字节翻倍疼得也少一半）---")
    bench_pair[4](8, x_raw, w_raw, d_raw, xf)
    bench_pair[4](16, x_raw, w_raw, d_raw, xf)

    print("判据（跑之前就定死的）：4 块的保守端（配对比值 min）**全部 > 1.10** 才考虑把 V1 写进核。")
    print("⚠️ 这是**核级**数（`down_proj` 一个形状、avx2 一条通路），没有端到端数字。")

    # ⚠️ arena 在**最后一次使用处**析构 —— 而分配出来的指针要用到这里之后。
    #    不写这行，`arena` 的最后一个使用点落在最后一次 `alloc` 上，写到一半
    #    mmap 就被解映射（症状是 SIGSEGV，落在第一次写 `w` 的那一行）。
    arena.keep_alive()
