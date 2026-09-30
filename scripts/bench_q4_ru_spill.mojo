"""扫 `RU`（= 累加器组数 = **寄存器压力旋钮**）：五臂同进程交错。

为什么做它（2026-10-01）
----------------------
`objdump` 核对发现 alofa 每块指令数是 llama 的 ~4 倍，构成里 **取/存 213 条
≈ 16% 是栈 spill**（与 09-28 那条"每行块 38.4 条里 23% 栈存取"同源）。而 `RU`
在真路径是**推导值**（`RU = MUB_ROW_ACC_TARGET // RB`，decode `RB=1` ⇒ `RU=8`），
本项目**从没看过"RU 变化时 spill 条数怎么变"** —— 09-29 / 09-30 那两次只量了
**时间**，效应 ≲10% 落在噪声里（本机跨进程 ±25%）判不了。

怎么扫
------
用核的第 2 个 comptime 参（`RU_OVERRIDE`）在**同一个二进制**里实例化
`RU = 1/2/4/8/16`，`RB` 固定 1（decode 档）⇒ **累加器组数 = `RU`** ⇒ 这条轴
就是寄存器压力本身。同进程交错，每轮各臂 best-of-5，并**逐轮正/倒序交替**
（先跑的那臂不吃亏也不占便宜）。每次计时前扫 256 MB 冲刷（冷 cache，与
`bench_q4_cold_bandwidth.mojo` 同一套口径）。

⚠️ **主产出不是下面这些时间**，而是配合 `objdump` 数出来的 **spill 条数曲线**
—— 五个实例在同一个二进制里，可以逐个 dump 内层循环体数指令。按 09-30 的经验，
时间这一臂大概率仍然判不了，只作附带记录。

自证：五臂必须**逐位相同**（`RU` 不动每行累加顺序：仍是 `b = 0…nb-1`、行末
`reduce_add`）。

跑法（`-O2`；`fork()` 在 JIT 下会崩，所以先 build）
--------------------------------------------------
    pixi run mojo build -O2 -I src scripts/bench_q4_ru_spill.mojo -o target/ru_spill
    ./target/ru_spill
"""

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import Q4_U8, _matmul_q4_mub_rows
from alofa.kernels.cpu.quant import Q4_BLOCK, Q4_BYTES, quantize_q4_0

# Qwen2.5-0.5B 的真投影形状（`tests/fixtures/qwen2.5-0.5b/config.tsv`）。
comptime HIDDEN = 896
comptime INTER = 4864
comptime KV = 128
comptime FLUSH_FLOATS = 64 * 1024 * 1024  # 256 MB ≫ L3 12 MB
comptime REPS = 9
comptime INNER = 5
comptime NARMS = 5

# 五档 `RU`。`RB=1` ⇒ 累加器组数 = RU ⇒ 这就是寄存器压力的扫描轴。
# ⚠️ 真路径是 `RU=8`（`MUB_ROW_ACC_TARGET // RB`），即这里第 4 档 ⇒ 比值以它为 1。
comptime RU1 = 1
comptime RU2 = 2
comptime RU3 = 4
comptime RU4 = 8
comptime RU5 = 16


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
    """`INNER` 遍里最快的那一遍：每遍都先冲刷（保证冷 cache），再跑这一臂。"""
    var best = Int(0)
    for attempt in range(INNER):
        sink += read_sum(flush_p, FLUSH_FLOATS)
        var t0 = monotonic_ns()
        _matmul_q4_mub_rows[1, RU, 0, 0, 0](
            pd, px, blocks, rows, cols, pd, False, rows
        )
        var t1 = monotonic_ns()
        if best == 0 or t1 - t0 < best:
            best = t1 - t0
    return best


def ru_label(arm: Int) -> Int:
    if arm == 0:
        return RU1
    if arm == 1:
        return RU2
    if arm == 2:
        return RU3
    if arm == 3:
        return RU4
    return RU5


def median9(mut a: InlineArray[Int, REPS]) -> Int:
    """插入排序后取中位（`REPS=9` ⇒ 第 5 个）。"""
    var i = 1
    while i < REPS:
        var v = a[i]
        var j = i - 1
        while j >= 0 and a[j] > v:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = v
        i += 1
    return a[REPS // 2]


def bench_shape(
    name: String, rows: Int, inner: Int, flush_p: F32Ptr, mut sink: Float64
) raises -> None:
    var n = rows * inner
    var nblocks = n // Q4_BLOCK
    var q4_bytes = nblocks * Q4_BYTES
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
    _matmul_q4_mub_rows[1, RU1, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU2, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU3, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU4, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU5, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)

    # 自证：五臂必须算出**同一个**结果（`RU` 不动每行的累加顺序）。
    _matmul_q4_mub_rows[1, RU1, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    copy_f32(pr, pd, rows)
    _matmul_q4_mub_rows[1, RU2, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    var s2 = same_f32(pd, pr, rows)
    _matmul_q4_mub_rows[1, RU3, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    var s3 = same_f32(pd, pr, rows)
    _matmul_q4_mub_rows[1, RU4, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    var s4 = same_f32(pd, pr, rows)
    _matmul_q4_mub_rows[1, RU5, 0, 0, 0](pd, px, blocks, rows, inner, pd, False, rows)
    var s5 = same_f32(pd, pr, rows)

    # 9 轮，每轮各臂一次；**逐轮正/倒序交替**。
    var m = InlineArray[Int, NARMS * REPS](uninitialized=True)
    for rep in range(REPS):
        if rep % 2 == 0:
            m[0 * REPS + rep] = time_arm[RU1](
                pd, px, blocks, rows, inner, flush_p, sink
            )
            m[1 * REPS + rep] = time_arm[RU2](
                pd, px, blocks, rows, inner, flush_p, sink
            )
            m[2 * REPS + rep] = time_arm[RU3](
                pd, px, blocks, rows, inner, flush_p, sink
            )
            m[3 * REPS + rep] = time_arm[RU4](
                pd, px, blocks, rows, inner, flush_p, sink
            )
            m[4 * REPS + rep] = time_arm[RU5](
                pd, px, blocks, rows, inner, flush_p, sink
            )
        else:
            m[4 * REPS + rep] = time_arm[RU5](
                pd, px, blocks, rows, inner, flush_p, sink
            )
            m[3 * REPS + rep] = time_arm[RU4](
                pd, px, blocks, rows, inner, flush_p, sink
            )
            m[2 * REPS + rep] = time_arm[RU3](
                pd, px, blocks, rows, inner, flush_p, sink
            )
            m[1 * REPS + rep] = time_arm[RU2](
                pd, px, blocks, rows, inner, flush_p, sink
            )
            m[0 * REPS + rep] = time_arm[RU1](
                pd, px, blocks, rows, inner, flush_p, sink
            )

    print(
        name
        + " rows="
        + String(rows)
        + " inner="
        + String(inner)
        + "\n      逐位相同（对 RU=1 那档）：RU2 "
        + String(s2)
        + "   RU4 "
        + String(s3)
        + "   RU8 "
        + String(s4)
        + "   RU16 "
        + String(s5)
        + "   （都该 = rows ⇒ RU 不动累加顺序）"
    )
    # ⚠️ 比值要**先算完所有档的中位**再打印：边算边打会拿 `med[3]`（RU=8）当分母，
    # 而前三档算它时 med[3] 还是未初始化的垃圾值（2026-10-01 第一次跑就踩了这个坑，
    # 打出 10766279‰ 那种数）。
    var med = InlineArray[Int, NARMS](uninitialized=True)
    var lo_a = InlineArray[Int, NARMS](uninitialized=True)
    var hi_a = InlineArray[Int, NARMS](uninitialized=True)
    for arm in range(NARMS):
        var one = InlineArray[Int, REPS](uninitialized=True)
        for rep in range(REPS):
            one[rep] = m[arm * REPS + rep]
        var lo = one[0]
        var hi = one[0]
        for rep in range(REPS):
            if one[rep] < lo:
                lo = one[rep]
            if one[rep] > hi:
                hi = one[rep]
        med[arm] = median9(one)
        lo_a[arm] = lo
        hi_a[arm] = hi
    for arm in range(NARMS):
        print(
            "      RU="
            + String(ru_label(arm))
            + "  中位 "
            + String(med[arm] // 1000)
            + " µs   区间 "
            + String(lo_a[arm] // 1000)
            + "…"
            + String(hi_a[arm] // 1000)
            + " µs   相对 RU=8 = "
            + String(med[arm] * 1000 // med[3])
            + "‰"
        )
    arena.keep_alive()


def main() raises:
    var arena = Arena(FLUSH_FLOATS * 4 + 64)
    var flush = scratch(arena, 1, FLUSH_FLOATS)
    fill(f32_data(flush), FLUSH_FLOATS)
    var sink = Float64(0)
    print(
        "冷 cache（每次计时前扫 "
        + String(FLUSH_FLOATS * 4 // (1024 * 1024))
        + " MB），"
        + String(REPS)
        + " 轮 × 每臂 best-of-"
        + String(INNER)
        + "，逐轮正/倒序交替；主产出是 objdump 的 spill 条数，这里的时间只作附带"
    )
    bench_shape("gate/up/down", INTER, HIDDEN, f32_data(flush), sink)
    bench_shape("k/v", KV, HIDDEN, f32_data(flush), sink)
    print("sink=" + String(sink))
    arena.keep_alive()
