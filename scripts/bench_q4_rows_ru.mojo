"""同进程配对：`RU`（一次解量化喂几个输出行）三档在**同一个二进制**里交错计时。

为什么需要它
------------
本机跨进程 A/B 噪声 ±25%（2026-09-29：同一个二进制相邻两轮 `gate` 393657 ↔ 581050 ns），
而 `RU` 8→16 的效应 ≲10% ⇒ 五次交错配对**全部跨过 1.0、且方向随轮翻转**，判不了。
同进程配对把系统漂移变成两臂共有：每次计时前**冲刷一遍 256 MB**（与
`bench_q4_cold_bandwidth.mojo` 同一套口径），两臂**交错**出场，并**逐轮轮换顺序**
（先跑的那臂不吃亏，也不占便宜）—— `bench_gemm_rb.mojo` 的先例能把比值区间收到
0.78–0.81 那种紧度。

量的是哪一层
------------
直接调核 `_matmul_q4_mub_rows[RB=1, RU]`（decode 档），**不是**模型层的 `q4_matmul_rows_k`：
后者的分发在编译期定死一个 `RU`，一个二进制里只能有一档。核外剩下的是**两臂完全相同**
的激活量化（RB=1、cols=896 时约 1 µs，占 `gate` 的 0.3%、占 `k/v` 的 6%）⇒ 它只把比值
往 1.0 拉一点，不会造出方向。

怎么读
------
三档 `RU = 4 / 8 / 16`，每轮各一次，报两条比值：`8/16`（>1 表示 16 更快）与 `4/8`
（>1 表示 8 更快）。判据与别处一致：**一轮一个比值，min…max 全在 1.0 同侧才叫分离**；
跨过 1.0 就是"判不了"。

另有一道自证：三档的输出**必须逐位相同**（`RU` 不改每行的累加顺序）。逐位相同数 ≠ rows
就说明两臂算的不是同一件事，比值无意义。

跑法（`-O2`；`fork()` 在 JIT 下会崩，所以先 build）：

    pixi run mojo build -O2 -I src scripts/bench_q4_rows_ru.mojo -o target/q4rowsru
    ./target/q4rowsru
"""

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import Q4_U8, _matmul_q4_mub_rows
from alofa.kernels.cpu.quant import Q4_BLOCK, Q4_BYTES, quantize_q4_0

# Qwen2.5-0.5B 的三个真投影形状（`tests/fixtures/qwen2.5-0.5b/config.tsv`）。
comptime HIDDEN = 896
comptime INTER = 4864
comptime KV = 128
comptime FLUSH_FLOATS = 64 * 1024 * 1024  # 256 MB ≫ L3 12 MB
comptime REPS = 7
# ⚠️ 每轮每档取 **best-of-3**：单次计时只有 85…400 µs，本机 runq 6–27 时这一个窗口
# 完全可能被调度打断（实测单样本比值散到 **0.58…2.48**，比跨进程还糟）⇒ 每遍都先
# 冲刷、跑 `INNER` 遍取最快，再用这个"最快"去做配对比值。
comptime INNER = 3
comptime RU_LO = 4
comptime RU_MID = 8
comptime RU_HI = 16


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, shape2(rows, cols), DT_FP32)


def fill(p: F32Ptr, n: Int) -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1


def read_sum(p: F32Ptr, n: Int) -> Float64:
    """扫一遍求和 —— 只用来**读**（把权重挤出 cache）。

    结果必须被用掉，否则整段会被优化掉；调用方累加进 sink 并在最后打印。
    """
    var acc = Float64(0)
    var i = 0
    while i < n:
        acc += Float64(p[unsafe_offset=i])
        i += 1
    return acc


def copy_f32(dst: F32Ptr, src: F32Ptr, n: Int) -> None:
    var i = 0
    while i < n:
        dst[unsafe_offset=i] = src[unsafe_offset=i]
        i += 1


def same_f32(a: F32Ptr, b: F32Ptr, n: Int) -> Int:
    var hits = 0
    var i = 0
    while i < n:
        if a[unsafe_offset=i] == b[unsafe_offset=i]:
            hits += 1
        i += 1
    return hits


def time_arm[RU: Int](
    pd: F32Ptr,
    px: F32Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    flush_p: F32Ptr,
    mut sink: Float64,
) raises -> Int:
    """`INNER` 遍里最快的那一遍：每遍都先冲刷（保证冷 cache），再跑 `RU` 那一档。"""
    var best = Int(0)
    for attempt in range(INNER):
        sink += read_sum(flush_p, FLUSH_FLOATS)
        var t0 = monotonic_ns()
        _matmul_q4_mub_rows[1, RU](pd, px, blocks, rows, cols, pd, False, rows)
        var t1 = monotonic_ns()
        if best == 0 or t1 - t0 < best:
            best = t1 - t0
    return best


def bench_shape(
    name: String, rows: Int, inner: Int, flush_p: F32Ptr, mut sink: Float64
) raises -> None:
    var n = rows * inner
    var q4_bytes = n // Q4_BLOCK * Q4_BYTES
    var arena = Arena((n + inner + 2 * rows) * 4 + q4_bytes + 64)
    var w = scratch(arena, rows, inner)
    var x = scratch(arena, 1, inner)
    var dst = scratch(arena, 1, rows)
    var refv = scratch(arena, 1, rows)
    fill(f32_data(w), n)
    fill(f32_data(x), inner)
    var blocks = arena.alloc(q4_bytes).unsafe_bitcast[UInt8]()
    quantize_q4_0(blocks, f32_data(w), n)

    var pd = f32_data(dst)
    var pr = f32_data(refv)
    var px = f32_data(x)

    # 预热（不计入）：页表、分支预测。
    _matmul_q4_mub_rows[1, RU_LO](pd, px, blocks, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU_MID](pd, px, blocks, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU_HI](pd, px, blocks, rows, inner, pd, False, rows)

    # 自证 1：三档必须算出**同一个**结果（`RU` 不动每行的累加顺序）。
    _matmul_q4_mub_rows[1, RU_LO](pd, px, blocks, rows, inner, pd, False, rows)
    copy_f32(pr, pd, rows)
    _matmul_q4_mub_rows[1, RU_MID](pd, px, blocks, rows, inner, pd, False, rows)
    var same_mid = same_f32(pd, pr, rows)
    _matmul_q4_mub_rows[1, RU_HI](pd, px, blocks, rows, inner, pd, False, rows)
    var same_hi = same_f32(pd, pr, rows)

    var tlo = InlineArray[Int, REPS](uninitialized=True)
    var tmid = InlineArray[Int, REPS](uninitialized=True)
    var thi = InlineArray[Int, REPS](uninitialized=True)
    for rep in range(REPS):
        var a = Int(0)
        var b = Int(0)
        var c = Int(0)
        # ⚠️ 轮换出场顺序：让"先跑"这件事在三档之间轮着来。
        var phase = rep % 3
        if phase == 0:
            a = time_arm[RU_LO](pd, px, blocks, rows, inner, flush_p, sink)
            b = time_arm[RU_MID](pd, px, blocks, rows, inner, flush_p, sink)
            c = time_arm[RU_HI](pd, px, blocks, rows, inner, flush_p, sink)
        elif phase == 1:
            b = time_arm[RU_MID](pd, px, blocks, rows, inner, flush_p, sink)
            c = time_arm[RU_HI](pd, px, blocks, rows, inner, flush_p, sink)
            a = time_arm[RU_LO](pd, px, blocks, rows, inner, flush_p, sink)
        else:
            c = time_arm[RU_HI](pd, px, blocks, rows, inner, flush_p, sink)
            a = time_arm[RU_LO](pd, px, blocks, rows, inner, flush_p, sink)
            b = time_arm[RU_MID](pd, px, blocks, rows, inner, flush_p, sink)
        tlo[rep] = a
        tmid[rep] = b
        thi[rep] = c

    var mlo = tlo[0]
    var mmid = tmid[0]
    var mhi = thi[0]
    var r_mid_hi = InlineArray[Float64, REPS](uninitialized=True)
    var r_lo_mid = InlineArray[Float64, REPS](uninitialized=True)
    for rep in range(REPS):
        if tlo[rep] < mlo:
            mlo = tlo[rep]
        if tmid[rep] < mmid:
            mmid = tmid[rep]
        if thi[rep] < mhi:
            mhi = thi[rep]
        r_mid_hi[rep] = Float64(tmid[rep]) / Float64(thi[rep])
        r_lo_mid[rep] = Float64(tlo[rep]) / Float64(tmid[rep])

    var lo1 = r_mid_hi[0]
    var hi1 = r_mid_hi[0]
    var lo2 = r_lo_mid[0]
    var hi2 = r_lo_mid[0]
    var s1 = ""
    var s2 = ""
    for rep in range(REPS):
        s1 = s1 + " " + String(r_mid_hi[rep])
        s2 = s2 + " " + String(r_lo_mid[rep])
        if r_mid_hi[rep] < lo1:
            lo1 = r_mid_hi[rep]
        if r_mid_hi[rep] > hi1:
            hi1 = r_mid_hi[rep]
        if r_lo_mid[rep] < lo2:
            lo2 = r_lo_mid[rep]
        if r_lo_mid[rep] > hi2:
            hi2 = r_lo_mid[rep]

    print(
        name
        + " rows=" + String(rows) + " inner=" + String(inner)
        + " | RU=4 最快 " + String(mlo) + " ns   RU=8 " + String(mmid)
        + " ns   RU=16 " + String(mhi) + " ns"
        + "\n      逐位相同 vs RU=4：RU=8 " + String(same_mid) + "/" + String(rows)
        + "  RU=16 " + String(same_hi) + "/" + String(rows)
        + "\n      逐轮 RU8/RU16 = " + String(lo1) + "…" + String(hi1)
        + "   （>1 表示 16 更快）\n        " + s1
        + "\n      逐轮 RU4/RU8  = " + String(lo2) + "…" + String(hi2)
        + "   （>1 表示 8 更快）\n        " + s2
    )
    arena.keep_alive()


def main() raises:
    print("冷 cache（每次计时前扫 256 MB）+ 同进程配对 + 逐轮轮换出场顺序")
    print("")
    var farena = Arena(FLUSH_FLOATS * 4 + 64)
    var flushv = scratch(farena, FLUSH_FLOATS, 1)
    var flush_p = f32_data(flushv)
    fill(flush_p, FLUSH_FLOATS)

    var sink = Float64(0)
    bench_shape("q_proj/o_proj ", HIDDEN, HIDDEN, flush_p, sink)
    bench_shape("gate/up/down  ", INTER, HIDDEN, flush_p, sink)
    bench_shape("k_proj/v_proj ", KV, HIDDEN, flush_p, sink)

    print("")
    print("unreachable " + String(sink))
    farena.keep_alive()
