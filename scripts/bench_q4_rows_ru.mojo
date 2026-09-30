"""同进程配对：`RU` / `Σx` 域两个旋钮，**四臂在同一个二进制**里交错计时。

为什么需要它
------------
本机跨进程 A/B 噪声 ±25%（2026-09-29：同一个二进制相邻两轮 `gate` 393657 ↔ 581050 ns），
而这两个旋钮的效应 ≲10–20% ⇒ 五次交错配对**全部跨过 1.0、且方向随轮翻转**，判不了。
同进程配对把系统漂移变成各臂共有：每次计时前**冲刷一遍 256 MB**（与
`bench_q4_cold_bandwidth.mojo` 同一套口径），各臂**交错**出场，并**逐轮轮换顺序**
（先跑的那臂不吃亏，也不占便宜）—— `bench_gemm_rb.mojo` 的先例能把比值区间收到
0.78–0.81 那种紧度。

四臂（2026-09-30 起改判**预展开权重**；`RU`/`Σx` 那两问已判完）
----
| 臂 | 权重布局 | 作用 |
|----|----------|------|
| A  | 原 18 B/块 | 与 B 同档 ⇒ `A/B` = **噪声底 1** |
| B  | 原 18 B/块 | **基准**（与 C 比才是要判的那一比）|
| C  | **预展开 34 B/块** | 内层省 4 条拆半字节指令，代价流量 ×1.89 |
| D  | **预展开 34 B/块** | 与 C 同档 ⇒ `C/D` = **噪声底 2** |

量的是哪一层
------------
直接调核 `_matmul_q4_mub_rows[RB=1, RU, SX, JK](`（decode 档），**不是**模型层的
`q4_matmul_rows_k`：后者的分发在编译期定死，一个二进制里只能有一档。核外剩下的是**各臂
完全相同**的激活量化（RB=1、cols=896 时约 1 µs，占 `gate` 的 0.3%、占 `k/v` 的 6%）
⇒ 它只把比值往 1.0 拉一点，不会造出方向。

怎么读
------
每轮一个比值，**min…max 全在 1.0 同侧**才叫分离；跨过 1.0 就是"判不了"。
- `RU8/RU16` > 1 表示 `RU=16` 更快；`RU4/RU8` > 1 表示 `RU=8` 更快。
- `JUNK/基线` > 1 表示**加料那一档更快**（spill 不值钱）。

另有一道自证：四臂的输出**必须逐位相同**（`RU` 不动每行的累加顺序；`Σx` 挪 int32 域时
`q` 与 `Σx` 都是整数、|q + Σx| < 2^24 ⇒ 两条路在 f32 里都精确）。逐位相同数 ≠ rows 就说明
两臂算的不是同一件事，比值无意义。

跑法（`-O2`；`fork()` 在 JIT 下会崩，所以先 build）：

    pixi run mojo build -O2 -I src scripts/bench_q4_rows_ru.mojo -o target/q4rowsru
    ./target/q4rowsru
"""

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import Q4_U8, _matmul_q4_mub_rows
from alofa.kernels.cpu.quant import (
    Q4_BLOCK,
    Q4_BYTES,
    Q4_SCALE_BYTES,
    quantize_q4_0,
)

# 预展开布局的块字节数 = 2 B 步长 + 32 B（每个权重一个字节）⇒ 34 B（核里 `Q4_PRE_BYTES`）。
comptime Q4_PRE_BYTES = Q4_SCALE_BYTES + Q4_BLOCK

# Qwen2.5-0.5B 的三个真投影形状（`tests/fixtures/qwen2.5-0.5b/config.tsv`）。
comptime HIDDEN = 896
comptime INTER = 4864
comptime KV = 128
# ⚠️ 冲刷量试过 **64 MB**（≫ L3 12 MB，理论上够），结果**更散**：`gate` 的 RU8/RU16 从
# 256 MB 那轮的 0.798…0.938（7/7 同侧）变成 0.776…1.111（6/9）⇒ "一轮越短配对越紧"是错的，
# 长冲刷似乎能把机器状态（频率/内存子系统）压得更一致。故**回到 256 MB**。
comptime FLUSH_FLOATS = 64 * 1024 * 1024  # 256 MB ≫ L3 12 MB
comptime REPS = 9
# ⚠️ 每轮每臂取 **best-of-5**：单次计时只有 85…400 µs，本机 runq 6–27 时这一个窗口
# 完全可能被调度打断（实测单样本比值散到 **0.58…2.48**，比跨进程还糟）⇒ 每遍都先
# 冲刷、跑 `INNER` 遍取最快，再用这个"最快"去做配对比值。3 遍时仍有 ±10% 的轮间散度，
# 提到 5 遍（冲刷一次 70 ms，全程仍 <1 min）。
comptime INNER = 5

# 四臂：A/B/C 仍是 `RU` = 4/8/16；**D 改成 `RU`=8 + `JUNK`=1** —— 每行每块多挂一个被迫
# 跨循环存活的值 ⇒ 与 B（基准）只差"在飞值"这一项 ⇒ 配对比出它的价钱。
comptime RU_A = 8
comptime SX_A = 0
comptime RU_B = 8
comptime SX_B = 0
comptime RU_C = 8
comptime SX_C = 0
comptime RU_D = 8
comptime SX_D = 0
# ⚠️ `JK` = 每行每块再挂一个被迫跨循环存活的值（核里 `JUNK`）：数值不
# 变，只加在飞值与 spill。`B` 是基准，`D` 是加料，两者
# 同进程配对 ⇒ 量“在飞值值多少钱”。
comptime JK_A = 0
comptime JK_B = 0
comptime JK_C = 0
comptime JK_D = 0
# ⚠️ `PR` = **预展开权重**（核里 `PRE`）：块 18 B → 34 B（权重流量 **×1.89**），换来内层
# 省掉拆半字节那 4 条指令（`vpand`+`vpsrlw`+`vpand`+`vinserti128`）。A/B 用原布局、
# C/D 用预展开 ⇒ `A/B` 与 `C/D` 各是一条**噪声底**，`B/C` 才是要判的那一比
# （**>1 = 预展开更快**）。⚠️ 数值**逐位相同**（纯重排）⇒ "两臂是不是同一份代码"
# 只能看 `objdump`，看数值是看不出来的。
comptime PR_A = 0
comptime PR_B = 0
comptime PR_C = 1
comptime PR_D = 1


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, shape2(rows, cols), DT_FP32)


def fill(p: F32Ptr, n: Int) -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1


def expand_q4_0(dst: Q4_U8, src: Q4_U8, nblocks: Int) -> None:
    """把 18 B/块（2 B 步长 + 16 B 半字节）重排成 34 B/块（2 B 步长 + 32 B 字节）。

    排布必须与核里 `lo.join(hi)` **逐字节一致**：q4_0 的 16 个原始字节里，**低**半字节
    是权重 0…15、**高**半字节是权重 16…31 ⇒ 展开后是 `[W0..W15][W16..W31]`。
    ⚠️ 这是**纯重排**，不碰算术 ⇒ 输出必须**逐位相同**（探针拿它当自证之一）。
    """
    var i = 0
    while i < nblocks:
        var sb = i * Q4_BYTES
        var db = i * Q4_PRE_BYTES
        dst[unsafe_offset=db] = src[unsafe_offset=sb]
        dst[unsafe_offset=db + 1] = src[unsafe_offset=sb + 1]
        var j = 0
        while j < Q4_BLOCK // 2:
            var raw = src[unsafe_offset=sb + Q4_SCALE_BYTES + j]
            dst[unsafe_offset=db + Q4_SCALE_BYTES + j] = raw & UInt8(0x0F)
            dst[unsafe_offset=db + Q4_SCALE_BYTES + Q4_BLOCK // 2 + j] = (
                (raw >> UInt8(4)) & UInt8(0x0F)
            )
            j += 1
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


def time_arm[RU: Int, SX: Int, JK: Int, PRE: Int = 0](
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
        _matmul_q4_mub_rows[1, RU, SX, JK, PRE](
            pd, px, blocks, rows, cols, pd, False, rows
        )
        var t1 = monotonic_ns()
        if best == 0 or t1 - t0 < best:
            best = t1 - t0
    return best


def bench_shape(
    name: String, rows: Int, inner: Int, flush_p: F32Ptr, mut sink: Float64
) raises -> None:
    var n = rows * inner
    var nblocks = n // Q4_BLOCK
    var q4_bytes = nblocks * Q4_BYTES
    var q4_pre_bytes = nblocks * Q4_PRE_BYTES
    var arena = Arena(
        (n + inner + 2 * rows) * 4 + q4_bytes + q4_pre_bytes + 64
    )
    var w = scratch(arena, rows, inner)
    var x = scratch(arena, 1, inner)
    var dst = scratch(arena, 1, rows)
    var refv = scratch(arena, 1, rows)
    fill(f32_data(w), n)
    fill(f32_data(x), inner)
    var blocks = arena.alloc(q4_bytes).unsafe_bitcast[UInt8]()
    quantize_q4_0(blocks, f32_data(w), n)
    var blocks_pre = arena.alloc(q4_pre_bytes).unsafe_bitcast[UInt8]()
    expand_q4_0(blocks_pre, blocks, nblocks)

    var pd = f32_data(dst)
    var pr = f32_data(refv)
    var px = f32_data(x)

    # 预热（不计入）：页表、分支预测。
    _matmul_q4_mub_rows[1, RU_A, SX_A, JK_A, PR_A](pd, px, blocks, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU_B, SX_B, JK_B, PR_B](pd, px, blocks, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU_C, SX_C, JK_C, PR_C](pd, px, blocks_pre, rows, inner, pd, False, rows)
    _matmul_q4_mub_rows[1, RU_D, SX_D, JK_D, PR_D](pd, px, blocks_pre, rows, inner, pd, False, rows)

    # 自证：四臂必须算出**同一个**结果（`RU` 不动每行的累加顺序；`Σx` 挪 int32 域时
    # 两个操作数都是整数且和 < 2^24 ⇒ 两条路在 f32 里都精确 ⇒ 逐位相同）。
    _matmul_q4_mub_rows[1, RU_A, SX_A, JK_A, PR_A](pd, px, blocks, rows, inner, pd, False, rows)
    copy_f32(pr, pd, rows)
    _matmul_q4_mub_rows[1, RU_B, SX_B, JK_B, PR_B](pd, px, blocks, rows, inner, pd, False, rows)
    var same_b = same_f32(pd, pr, rows)
    _matmul_q4_mub_rows[1, RU_C, SX_C, JK_C, PR_C](pd, px, blocks_pre, rows, inner, pd, False, rows)
    var same_c = same_f32(pd, pr, rows)
    _matmul_q4_mub_rows[1, RU_D, SX_D, JK_D, PR_D](pd, px, blocks_pre, rows, inner, pd, False, rows)
    var same_d = same_f32(pd, pr, rows)

    var ta = InlineArray[Int, REPS](uninitialized=True)
    var tb = InlineArray[Int, REPS](uninitialized=True)
    var tc = InlineArray[Int, REPS](uninitialized=True)
    var td = InlineArray[Int, REPS](uninitialized=True)
    for rep in range(REPS):
        var a = Int(0)
        var b = Int(0)
        var c = Int(0)
        var d = Int(0)
        # ⚠️ 轮换出场顺序：让"先跑"这件事在四臂之间轮着来。
        var phase = rep % 4
        if phase == 0:
            a = time_arm[RU_A, SX_A, JK_A, PR_A](pd, px, blocks, rows, inner, flush_p, sink)
            b = time_arm[RU_B, SX_B, JK_B, PR_B](pd, px, blocks, rows, inner, flush_p, sink)
            c = time_arm[RU_C, SX_C, JK_C, PR_C](pd, px, blocks_pre, rows, inner, flush_p, sink)
            d = time_arm[RU_D, SX_D, JK_D, PR_D](pd, px, blocks_pre, rows, inner, flush_p, sink)
        elif phase == 1:
            b = time_arm[RU_B, SX_B, JK_B, PR_B](pd, px, blocks, rows, inner, flush_p, sink)
            c = time_arm[RU_C, SX_C, JK_C, PR_C](pd, px, blocks_pre, rows, inner, flush_p, sink)
            d = time_arm[RU_D, SX_D, JK_D, PR_D](pd, px, blocks_pre, rows, inner, flush_p, sink)
            a = time_arm[RU_A, SX_A, JK_A, PR_A](pd, px, blocks, rows, inner, flush_p, sink)
        elif phase == 2:
            c = time_arm[RU_C, SX_C, JK_C, PR_C](pd, px, blocks_pre, rows, inner, flush_p, sink)
            d = time_arm[RU_D, SX_D, JK_D, PR_D](pd, px, blocks_pre, rows, inner, flush_p, sink)
            a = time_arm[RU_A, SX_A, JK_A, PR_A](pd, px, blocks, rows, inner, flush_p, sink)
            b = time_arm[RU_B, SX_B, JK_B, PR_B](pd, px, blocks, rows, inner, flush_p, sink)
        else:
            d = time_arm[RU_D, SX_D, JK_D, PR_D](pd, px, blocks_pre, rows, inner, flush_p, sink)
            a = time_arm[RU_A, SX_A, JK_A, PR_A](pd, px, blocks, rows, inner, flush_p, sink)
            b = time_arm[RU_B, SX_B, JK_B, PR_B](pd, px, blocks, rows, inner, flush_p, sink)
            c = time_arm[RU_C, SX_C, JK_C, PR_C](pd, px, blocks_pre, rows, inner, flush_p, sink)
        ta[rep] = a
        tb[rep] = b
        tc[rep] = c
        td[rep] = d

    var ma = ta[0]
    var mb = tb[0]
    var mc = tc[0]
    var md = td[0]
    var r_sx = InlineArray[Float64, REPS](uninitialized=True)
    var r_hi = InlineArray[Float64, REPS](uninitialized=True)
    var r_lo = InlineArray[Float64, REPS](uninitialized=True)
    for rep in range(REPS):
        if ta[rep] < ma:
            ma = ta[rep]
        if tb[rep] < mb:
            mb = tb[rep]
        if tc[rep] < mc:
            mc = tc[rep]
        if td[rep] < md:
            md = td[rep]
        # 要判的那一比：**原布局 / 预展开**（>1 = 预展开更快）
        r_sx[rep] = Float64(tb[rep]) / Float64(tc[rep])
        # 噪声底 1：原布局 / 原布局
        r_hi[rep] = Float64(ta[rep]) / Float64(tb[rep])
        # 噪声底 2：预展开 / 预展开
        r_lo[rep] = Float64(tc[rep]) / Float64(td[rep])

    var s_sx = ""
    var s_hi = ""
    var s_lo = ""
    var lo1 = r_sx[0]
    var hi1 = r_sx[0]
    var lo2 = r_hi[0]
    var hi2 = r_hi[0]
    var lo3 = r_lo[0]
    var hi3 = r_lo[0]
    for rep in range(REPS):
        s_sx = s_sx + " " + String(r_sx[rep])
        s_hi = s_hi + " " + String(r_hi[rep])
        s_lo = s_lo + " " + String(r_lo[rep])
        if r_sx[rep] < lo1:
            lo1 = r_sx[rep]
        if r_sx[rep] > hi1:
            hi1 = r_sx[rep]
        if r_hi[rep] < lo2:
            lo2 = r_hi[rep]
        if r_hi[rep] > hi2:
            hi2 = r_hi[rep]
        if r_lo[rep] < lo3:
            lo3 = r_lo[rep]
        if r_lo[rep] > hi3:
            hi3 = r_lo[rep]

    print(
        name
        + " rows=" + String(rows) + " inner=" + String(inner)
        + "\n      A 原布局(18B) " + String(ma) + " ns   B 原布局 " + String(mb)
        + " ns   C 预展开(34B) " + String(mc) + " ns   D 预展开 " + String(md)
        + " ns"
        + "\n      逐位相同（对 A 那档）：B " + String(same_b)
        + "/" + String(rows) + "   C " + String(same_c) + "/" + String(rows)
        + "   D " + String(same_d) + "/" + String(rows)
        + "   （四臂都该 = rows：预展开是纯重排）"
        + "\n      逐轮 B/C 原布局/预展开 = " + String(lo1) + "…" + String(hi1)
        + "   （**>1 = 预展开更快**）\n        " + s_sx
        + "\n      逐轮 A/B 原布局/原布局 = " + String(lo2) + "…" + String(hi2)
        + "   （噪声底 1）\n        " + s_hi
        + "\n      逐轮 C/D 预展开/预展开 = " + String(lo3) + "…" + String(hi3)
        + "   （噪声底 2）\n        " + s_lo
    )
    arena.keep_alive()


def main() raises:
    print(
        "冷 cache（每次计时前扫 "
        + String(FLUSH_FLOATS * 4 // (1024 * 1024))
        + " MB）+ 同进程配对 + 逐轮轮换出场顺序（每轮每臂 best-of-"
        + String(INNER) + "）"
    )
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
