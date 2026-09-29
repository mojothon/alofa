"""q4 核是**访存延迟没盖住**还是**核内堵着**？—— 用「权重常驻在哪级 cache」当自变量。

为什么做这个
------------
核级实测每 (行,块) ≈17 周期，而三个理论界都远低于它：

| 界 | 估算 |
|----|------|
| 发射（一轮 ≈89 条 ÷ 4 宽） | ≈22 周期/轮 ≈2.7 周期/(行,块) |
| 栈端口（18 store，1/cycle） | ≈18 周期/轮 |
| DRAM（144 B/轮 ÷ 2.6 B/周期） | ≈55 周期/轮 ≈7 周期/(行,块) |

差 2.5× ⇒ 要么**访存延迟没盖住**（MLP 不够，核在等数），要么核内还有没算到的堵点。把权重
**摆到不同级 cache** 就能分开这两种：延迟/带宽受限时，权重越靠近核应该越快；核内堵着时，
摆到 L1 也不该变快（因为根本没在等数）。

三臂（同一个二进制，交错 + 逐轮轮换出场）
------------------------------------------
| 臂 | 每次计时前 | 权重在哪 |
|----|-----------|----------|
| 冷 | 扫 256 MB | **DRAM**（L3 12 MB 也被冲掉了） |
| 温 | 扫 64 KB  | **L2**：64 KB > L1d 32 KB ⇒ L1 被挤掉；256 KB < L2 ⇒ 权重还在 |
| 热 | 不冲      | **L1d/L2**（28 KB 时 L1d 32 KB 也装得下） |

权重 ladder（`inner=896`，q4 一字节两个权重 ⇒ `rows*448` 字节）
--------------------------------------------------------------
| `rows` | 权重 | 装得下它的是 |
|--------|------|--------------|
| 64     | 28 KB | L1d(32 KB) / L2 |
| 128    | 56 KB | L2 |
| 256    | 112 KB | L2 |
| 512    | 224 KB | L2（贴着 256 KB 上限） |
| 896    | 401 KB | 只有 L3 |
| 4864   | 2.18 MB | 只有 L3 |

本机 i7-9700K：L1d 32 KB、L2 256 KB（每核）、L3 12 MB（共享）。

怎么读
------
- **冷/热 ≈ 1** ⇒ 权重摆到 L1 也不快 ⇒ 核**没在等数** ⇒ 那 2.5× 在核内（发射/端口/依赖），
  访存侧没得捞 ⇒ 别去动预取和结构。
- **冷/热 ≫ 1**（比如 2–3×）⇒ 核在等 DRAM ⇒ **延迟没盖住** ⇒ 该上预取或增加在飞访存数。
- 顺带看 ladder：冷臂的 ns/(行,块) 若随 `rows` 增大而上升 ⇒ 有容量/带宽效应。

⚠️ 口径
--------
- 每轮每臂 **best-of-5**（单样本会被调度打断，本机 runq 6–27）。
- 温/热臂的 5 遍**连着跑不冲刷**，取最快 ⇒ 第 1 遍可能还是冷的，最快那遍一定是常驻的。
- 核里含激活量化（`inner=896` 约 1 µs，不随 `rows` 变）⇒ `rows=64`（≈6 µs）时它占 ~15%，
  往大 `rows` 走它占比降到 ~4%。所以 **ladder 的大头（`rows ≥ 256`）才是干净的**。
- 计时只包住 `_matmul_q4_mub_rows` 这一句；冲刷在窗口外。

跑法（`-O2`；`fork()` 在 JIT 下会崩，所以先 build）：

    pixi run mojo build -O2 -I src scripts/bench_q4_cache_resident.mojo -o target/q4l2
    ./target/q4l2
"""

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import Q4_U8, _matmul_q4_mub_rows
from alofa.kernels.cpu.quant import Q4_BLOCK, Q4_BYTES, quantize_q4_0

comptime INNER = 896
comptime FLUSH_COLD = 64 * 1024 * 1024  # 256 MB ≫ L3 12 MB
comptime FLUSH_L2 = 16 * 1024  # 64 KB > L1d 32 KB，< L2 256 KB
comptime REPS = 11
comptime PASSES = 5




def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, shape2(rows, cols), DT_FP32)


def fill(p: F32Ptr, n: Int) -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1


def read_sum(p: F32Ptr, n: Int) -> Float64:
    """扫一遍求和 —— 只用来**读**（把别的数据挤出 cache），结果必须被用掉。"""
    var acc = Float64(0)
    var i = 0
    while i < n:
        acc += Float64(p[unsafe_offset=i])
        i += 1
    return acc


def time_arm[FLUSH: Int](
    pd: F32Ptr,
    px: F32Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    flush_p: F32Ptr,
    mut sink: Float64,
) raises -> Int:
    """`PASSES` 遍里最快的那一遍。`FLUSH=0` ⇒ 不冲刷（常驻臂）。"""
    var best = Int(0)
    for attempt in range(PASSES):
        if FLUSH > 0:
            sink += read_sum(flush_p, FLUSH)
        var t0 = monotonic_ns()
        _matmul_q4_mub_rows[1](pd, px, blocks, rows, cols, pd, False, rows)
        var t1 = monotonic_ns()
        # 读一遍结果：核的写不能因为"没人看"被整段删掉。
        sink += read_sum(pd, rows)
        if best == 0 or t1 - t0 < best:
            best = t1 - t0
    return best


def bench_shape(
    rows: Int, cold_p: F32Ptr, l2_p: F32Ptr, mut sink: Float64
) raises -> None:
    var n = rows * INNER
    var q4_bytes = n // Q4_BLOCK * Q4_BYTES
    var arena = Arena((n + INNER + 2 * rows) * 4 + q4_bytes + 64)
    var w = scratch(arena, rows, INNER)
    var x = scratch(arena, 1, INNER)
    var dst = scratch(arena, 1, rows)
    fill(f32_data(w), n)
    fill(f32_data(x), INNER)
    var blocks = arena.alloc(q4_bytes).unsafe_bitcast[UInt8]()
    quantize_q4_0(blocks, f32_data(w), n)
    var pd = f32_data(dst)
    var px = f32_data(x)

    # 预热（不计入）：页表、分支预测。
    _matmul_q4_mub_rows[1](pd, px, blocks, rows, INNER, pd, False, rows)

    var tc = InlineArray[Int, REPS](uninitialized=True)
    var tw = InlineArray[Int, REPS](uninitialized=True)
    var th = InlineArray[Int, REPS](uninitialized=True)
    for rep in range(REPS):
        var c = Int(0)
        var l = Int(0)
        var h = Int(0)
        # ⚠️ 轮换出场顺序：让"刚被冲刷过"这件事在三臂之间轮着来。
        var phase = rep % 3
        if phase == 0:
            c = time_arm[FLUSH_COLD](pd, px, blocks, rows, INNER, cold_p, sink)
            l = time_arm[FLUSH_L2](pd, px, blocks, rows, INNER, l2_p, sink)
            h = time_arm[0](pd, px, blocks, rows, INNER, l2_p, sink)
        elif phase == 1:
            l = time_arm[FLUSH_L2](pd, px, blocks, rows, INNER, l2_p, sink)
            h = time_arm[0](pd, px, blocks, rows, INNER, l2_p, sink)
            c = time_arm[FLUSH_COLD](pd, px, blocks, rows, INNER, cold_p, sink)
        else:
            h = time_arm[0](pd, px, blocks, rows, INNER, l2_p, sink)
            c = time_arm[FLUSH_COLD](pd, px, blocks, rows, INNER, cold_p, sink)
            l = time_arm[FLUSH_L2](pd, px, blocks, rows, INNER, l2_p, sink)
        tc[rep] = c
        tw[rep] = l
        th[rep] = h

    var mc = tc[0]
    var mw = tw[0]
    var mh = th[0]
    for rep in range(REPS):
        if tc[rep] < mc:
            mc = tc[rep]
        if tw[rep] < mw:
            mw = tw[rep]
        if th[rep] < mh:
            mh = th[rep]

    var nb = INNER // Q4_BLOCK
    var rb = rows * nb
    print(
        "rows="
        + String(rows)
        + "  权重 " + String(q4_bytes // 1024) + " KB"
        + "   冷(DRAM) " + String(mc) + " ns"
        + "   温(L2) " + String(mw) + " ns"
        + "   热(L1/L2) " + String(mh) + " ns"
    )
    print(
        "        每(行,块)：冷 " + String(Float64(mc) / Float64(rb))
        + " ns  →  温 " + String(Float64(mw) / Float64(rb))
        + "  →  热 " + String(Float64(mh) / Float64(rb))
        + " ns     冷/热 = "
        + String(Float64(mc) / Float64(mh))
    )
    arena.keep_alive()


def main() raises:
    print(
        "权重常驻在哪级 cache（冷：扫 " + String(FLUSH_COLD * 4 // (1024 * 1024))
        + " MB；温：扫 " + String(FLUSH_L2 * 4 // 1024)
        + " KB；热：不冲）+ 同进程配对 + 逐轮轮换（每轮每臂 best-of-"
        + String(PASSES) + "，" + String(REPS) + " 轮）"
    )
    print("")
    var carena = Arena(FLUSH_COLD * 4 + 64)
    var larena = Arena(FLUSH_L2 * 4 + 64)
    var cp = f32_data(scratch(carena, FLUSH_COLD, 1))
    var lp = f32_data(scratch(larena, FLUSH_L2, 1))
    fill(cp, FLUSH_COLD)
    fill(lp, FLUSH_L2)

    var sink = Float64(0)
    bench_shape(64, cp, lp, sink)  # 28 KB —— L1d(32 KB) 也装得下
    bench_shape(128, cp, lp, sink)  # 56 KB —— L2
    bench_shape(256, cp, lp, sink)  # 112 KB —— L2
    bench_shape(512, cp, lp, sink)  # 224 KB —— L2（贴着 256 KB 上限）
    bench_shape(896, cp, lp, sink)  # 401 KB —— 只有 L3
    bench_shape(4864, cp, lp, sink)  # 2.18 MB —— 只有 L3

    print("")
    print("unreachable " + String(sink))
    carena.keep_alive()
    larena.keep_alive()
