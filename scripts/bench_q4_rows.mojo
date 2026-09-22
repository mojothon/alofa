"""q4 通路：token 批次能不能别再每行各流一遍权重？（EB=RB 行复用一份块的解量化）

问的是什么
----------
`qwen.mojo` 里量化这条路是这么写的（源码原注释）：

> 量化核做的是**矩阵乘向量**（一次给一个输出行），这里要的是一整个 token 批次，
> 于是逐行喂进去。

也就是说：**prefill / 任意批 > 1 时，每个 token 都把整份量化权重重流一遍**。
今天在 fp32 那条路上连着否掉了三个杠杆（`RB=16`、去掉加宽转换、列分块），共同
教训是「它们都只是拿一种资源去换另一种，**总 uop 数一点没减**」。而 q4 这条路的
情况与 fp32 **不一样**，而且 `_matmul_q4_halves` 的注释已经把话说死了：

> 批=1 的 matmul 每权重读一次用一次、**算术受限**（账本：q4 通路只用到它自己
> 访存地板的 **10.6%**），所以这里唯一能省的是**指令数**。

既然受限的是**解量化的算术**，那么把它摊掉就是直击要害：一个 q4_0 块的
`raw` 读入、`& 0x0F`、`>> 4`、cast、减 8 —— 这些 **与输入行无关**，RB 行可以共用；
只有后面的 2 次 x 载入、乘 `d`、2 次乘加是随行数增长的。按 uop 粗算：

    现在（每行独立）：≈ 8 条（解量化，共享部分）+ 12 条（该行专用）= 20 条/块/行
    RB 行共用一次块：  8 条 + RB × 12 条            ⇒ RB=2 → ÷2 得 1.25×，
                                                    RB=4 → 1.43×，RB=8 → 1.54×

（上限 optimism 会被累加器溢出吃掉：`s0`/`s1` 各是 f32×16 = 2 条 ymm，**每行 4 条
ymm** ⇒ RB=8 要 32 条，寄存器堆只有 16。所以 RB 能开到多大是这次要量的。）

顺带还会把权重字节流也砍成 1/RB —— 但按上面那条 10.6%，这一项不是重点。

怎么量（`V3`）
--------------
同一份 `down_proj` 形状（out=896 × cols=4864 的 q4_0 块流，2.45 MB/份 × 8 份 ≫
L3 12 MB），同进程配对、5 轮交错、逐轮轮换出场顺序：
  * arm 旧：`N` 次 `matmul_q4_f32`（= `_matmul_q4_halves`，今天 `qwen.mojo` 的写法）；
  * arm V3：`_matmul_q4_rows[RB]`（本文件里新写的），一次带上 `RB` 个输入行，
    走 `N / RB` 趟权重。

判据（口径先定，再看数）
------------------------
  * **采用门槛**：同一个 RB 在 **N=8 与 N=16 两档**的保守端（配对比值 min）
    **都 > 1.10** 才算数；只过一档照实报。
  * **`src/` 一行没动**（`matmul_q4_f32` 可以直接 import）。
  * 边界：这里只量**无偏置**那条（`down` / `gate` / `up`）；
    q/k/v 带偏置，要走同一套处理得单独补，别从这里的数外推。

自检（不通过就一个数都不报）
----------------------------
① V3 与「旧写法」对每个输入行**逐位相等**（同一个 f32 累加路径、块的次序一次没变）；
② **负向对照针对自检本身**：带 `MUTATE` 编出的版本会漏掉最后一个块，①必须 **判
   红** —— 否则①只是在比较两个都得懒 PATH殃及池鱼的东西，门自己就是绿的；
③ 每次比较前把 `dst` 涂 -999 哨兵：**漏写任何一行都会留下一整片哨兵**（不涂的话
   漏掉的行保留上一次算出的**正确值**，正负两例逐位相同）。

    pixi run mojo run -O2 -I src scripts/bench_q4_rows.mojo
"""

from alofa.core.dtype import DT_FP32
from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_SHAPE_MISMATCH,
    AlofaError,
)
from alofa.core.ffi import monotonic_ns
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import (
    Q4_BLOCK,
    Q4_BYTES,
    Q4_SCALE_BYTES,
    W_Q4_HALF,
    block_scale,
    matmul_q4_f32,
)

comptime Q4_U8 = Pointer[UInt8, MutUntrackedOrigin]

# `down_proj` 朝向：权重 [out=896, cols=4864]，输入的 `cols` 就是 hidden dim。
comptime OUT = 896
comptime COLS = 4864
comptime MAX_ROWS = 32

comptime BLOCKS_PER_ROW = COLS // Q4_BLOCK
comptime TOT_BLOCKS = OUT * BLOCKS_PER_ROW
comptime TOT_BYTES = TOT_BLOCKS * Q4_BYTES

# 8 份 = 19.6 MB ≫ L3（12 MB）：每趟换一份，防止后几趟落在缓存里。
comptime REGIONS = 8

comptime ROUNDS = 5


def f2(v: Float64) -> String:
    """定点两位；这台机器上更多位数是假精度。"""
    var hundredths = Int(v * Float64(100))
    var frac = hundredths % 100
    var tail = String(frac) if frac >= 10 else "0" + String(frac)
    return String(hundredths // 100) + "." + tail


def interval(ts: List[Float64]) -> String:
    """min…max。两个边界都要看着 —— 只看下界就是把结论往自己那边掰。"""
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


def _matmul_q4_rows[RB: Int, MUTATE: Bool](
    dst: F32Ptr,
    x: F32Ptr,
    blocks: Q4_U8,
    out_rows: Int,
    cols: Int,
) raises AlofaError:
    """`RB` 个输入行一起过一遍权重（`dst[rb, o] = Σ_c w[o,c]·x[rb,c]`）。

    算术与 `_matmul_q4_halves`（今天的默认向量通路）**一步一步相同**：同样按半块
    切、同样 f32 累加、`d` 同样先折进 x、同样整行归约一次（四类结构的 A/B 见那个
    函数的注释）。唯一改的是**一个块只为 RB 行解量化一次**，而不是每行一遍。

    `MUTATE` 是给自检用的负向对照：为真时漏掉最后一个块 —— ①那条逐位比较必须
    因此判红。
    """
    if out_rows <= 0 or cols <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "matrix dimensions must be positive",
            "rows=" + String(out_rows) + " cols=" + String(cols),
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
    var acc = InlineArray[Float64, RB](fill=Float64(0))
    var s0 = InlineArray[SIMD[DType.float32, W_Q4_HALF], RB](
        fill=SIMD[DType.float32, W_Q4_HALF](0)
    )
    var s1 = InlineArray[SIMD[DType.float32, W_Q4_HALF], RB](
        fill=SIMD[DType.float32, W_Q4_HALF](0)
    )

    for row in range(out_rows):
        comptime for rb in range(RB):
            acc[rb] = Float64(0)
            s0[rb] = SIMD[DType.float32, W_Q4_HALF](0)
            s1[rb] = SIMD[DType.float32, W_Q4_HALF](0)

        # 负向对照的开关在这里：MUTATE 时最后一个块不参与（必须被①抓住）。
        var limit = blocks_per_row - 1 if MUTATE else blocks_per_row
        var b = 0
        while b < limit:
            var base = (row * blocks_per_row + b) * Q4_BYTES
            var d = block_scale(blocks, base)
            var xb = b * Q4_BLOCK
            var raw = blocks.unsafe_load[width=W_Q4_HALF](
                base + Q4_SCALE_BYTES
            )
            var lo = (raw & low_mask).cast[DType.float32]() - eight
            var hi = ((raw >> 4) & low_mask).cast[DType.float32]() - eight

            comptime for rb in range(RB):
                var xr = rb * cols
                s0[rb] += lo * (
                    x.unsafe_load[width=W_Q4_HALF](xr + xb) * d
                )
                s1[rb] += hi * (
                    x.unsafe_load[width=W_Q4_HALF](xr + xb + W_Q4_HALF) * d
                )
            b += 1

        comptime for rb in range(RB):
            acc[rb] += Float64((s0[rb] + s1[rb]).reduce_add())
            dst[unsafe_offset=rb * out_rows + row] = Float32(acc[rb])


def run_rows_old(
    n_rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, region: Int
) raises -> None:
    """今天 `qwen.mojo` 的写法：**每个输入行**把整份权重过一遍。"""
    var pd = f32_data(TensorView(d_raw, shape2(n_rows, OUT), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(n_rows, COLS), DT_FP32))
    var blocks = w_raw.unsafe_offset(region * TOT_BYTES).unsafe_bitcast[
        UInt8
    ]()
    for r in range(n_rows):
        matmul_q4_f32(
            pd.unsafe_offset(r * OUT),
            px.unsafe_offset(r * COLS),
            blocks,
            OUT,
            COLS,
        )


def run_rows_new[RB: Int](
    n_rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr, region: Int
) raises -> None:
    """V3：`RB` 行一组，一组共用一趟权重的解量化。"""
    var pd = f32_data(TensorView(d_raw, shape2(n_rows, OUT), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(n_rows, COLS), DT_FP32))
    var blocks = w_raw.unsafe_offset(region * TOT_BYTES).unsafe_bitcast[
        UInt8
    ]()
    var b = 0
    while b < n_rows:
        _matmul_q4_rows[RB, False](
            pd.unsafe_offset(b * OUT),
            px.unsafe_offset(b * COLS),
            blocks,
            OUT,
            COLS,
        )
        b += RB


def time_old(
    n_rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr
) raises -> Int:
    var t0 = monotonic_ns()
    for r in range(REGIONS):
        run_rows_old(n_rows, x_raw, w_raw, d_raw, r)
    var t1 = monotonic_ns()
    var pd = f32_data(TensorView(d_raw, shape2(n_rows, OUT), DT_FP32))
    _ = pd[unsafe_offset=0]
    return (t1 - t0) // REGIONS


def time_new[RB: Int](
    n_rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr
) raises -> Int:
    var t0 = monotonic_ns()
    for r in range(REGIONS):
        run_rows_new[RB](n_rows, x_raw, w_raw, d_raw, r)
    var t1 = monotonic_ns()
    var pd = f32_data(TensorView(d_raw, shape2(n_rows, OUT), DT_FP32))
    _ = pd[unsafe_offset=0]
    return (t1 - t0) // REGIONS


def paint(n_rows: Int, d_raw: RawPtr) raises -> None:
    var pd = f32_data(TensorView(d_raw, shape2(n_rows, OUT), DT_FP32))
    for i in range(n_rows * OUT):
        pd[unsafe_offset=i] = Float32(-999)


def check_parity[RB: Int](n_rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr) raises -> Bool:
    """① V3 == 旧写法（逐位）；② 负向对照必须被抓红；③ 无哨兵残留。"""
    paint(n_rows, d_raw)
    run_rows_old(n_rows, x_raw, w_raw, d_raw, 0)
    var gold = List[Float32]()
    var pd0 = f32_data(TensorView(d_raw, shape2(n_rows, OUT), DT_FP32))
    for i in range(n_rows * OUT):
        gold.append(pd0[unsafe_offset=i])

    paint(n_rows, d_raw)
    run_rows_new[RB](n_rows, x_raw, w_raw, d_raw, 0)
    var pd1 = f32_data(TensorView(d_raw, shape2(n_rows, OUT), DT_FP32))
    for i in range(n_rows * OUT):
        if pd1[unsafe_offset=i] != gold[i]:
            return False
        if pd1[unsafe_offset=i] == Float32(-999):
            return False  # 哨兵残留 = 有行没被写

    # ② 负向对照：漏掉最后一个块的版本必须被上面那条判据抓住。
    paint(n_rows, d_raw)
    var blocks = w_raw.unsafe_bitcast[UInt8]()
    var px = f32_data(TensorView(x_raw, shape2(n_rows, COLS), DT_FP32))
    var pd2 = f32_data(TensorView(d_raw, shape2(n_rows, OUT), DT_FP32))
    var b = 0
    while b < n_rows:
        _matmul_q4_rows[RB, True](
            pd2.unsafe_offset(b * OUT),
            px.unsafe_offset(b * COLS),
            blocks,
            OUT,
            COLS,
        )
        b += RB
    var caught = False
    for i in range(n_rows * OUT):
        if pd2[unsafe_offset=i] != gold[i]:
            caught = True
    return caught


def bench_pair[RB: Int](
    n_rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr
) raises -> None:
    var ago = List[Int]()
    var new = List[Int]()
    var ratios = List[Float64]()
    print(
        "  RB =",
        RB,
        " 批 =",
        n_rows,
        " 行   权重被读：旧",
        n_rows,
        "趟 / V3",
        n_rows // RB,
        "趟   累加器",
        RB * 4,
        "条 ymm（寄存器堆 16 条）",
    )
    for rd in range(ROUNDS):
        var old_first = (rd % 2) == 0
        if old_first:
            ago.append(time_old(n_rows, x_raw, w_raw, d_raw))
            new.append(time_new[RB](n_rows, x_raw, w_raw, d_raw))
        else:
            new.append(time_new[RB](n_rows, x_raw, w_raw, d_raw))
            ago.append(time_old(n_rows, x_raw, w_raw, d_raw))
        print(
            "    轮",
            rd + 1,
            "顺序",
            "旧→V3" if old_first else "V3→旧",
            "  旧",
            f2(Float64(ago[len(ago) - 1]) / 1000000.0),
            "ms   V3",
            f2(Float64(new[len(new) - 1]) / 1000000.0),
            "ms   配对比值",
            f2(Float64(ago[len(ago) - 1]) / Float64(new[len(new) - 1])),
            "×",
        )
        ratios.append(
            Float64(ago[len(ago) - 1]) / Float64(new[len(new) - 1])
        )
    print("    每批 min…max   旧", min_max(ago), "  V3", min_max(new))
    print("    配对比值 min…max", interval(ratios), "×（>1 = V3 更快）")
    print("")


def main() raises:
    var arena = Arena(
        REGIONS * TOT_BYTES
        + MAX_ROWS * COLS * 4
        + MAX_ROWS * OUT * 4
        + 4096
    )
    var w_raw = arena.alloc(REGIONS * TOT_BYTES)
    var x_raw = arena.alloc(MAX_ROWS * COLS * 4)
    var d_raw = arena.alloc(MAX_ROWS * OUT * 4)

    # q4_0 块：2 字节缩放 + 16 字节（32 个 nibble）。缩放用 1.0 的 fp16 位型。
    var blocks = w_raw.unsafe_bitcast[UInt8]()
    for cp in range(REGIONS):
        var base = cp * TOT_BYTES
        for bi in range(TOT_BLOCKS):
            var b0 = base + bi * Q4_BYTES
            # fp16(1.0) = 0x3C00，小端
            blocks[unsafe_offset=b0] = UInt8(0x00)
            blocks[unsafe_offset=b0 + 1] = UInt8(0x3C)
            var j = 0
            while j < Q4_BLOCK // 2:
                blocks[unsafe_offset=b0 + Q4_SCALE_BYTES + j] = UInt8(
                    (bi * 13 + j * 7) % 251
                )
                j += 1
    var px = f32_data(TensorView(x_raw, shape2(MAX_ROWS, COLS), DT_FP32))
    var i = 0
    while i < MAX_ROWS * COLS:
        px[unsafe_offset=i] = Float32(i % 29) - Float32(14)
        i += 1

    print("=== q4：token 批次别再每行各流一遍权重 ===")
    print(
        "形状 out",
        OUT,
        "× cols",
        COLS,
        "  每份块流",
        TOT_BYTES // (1024 * 1024),
        "MB ×",
        REGIONS,
        "份 =",
        REGIONS * TOT_BYTES // (1024 * 1024),
        "MB（≫ L3 12 MB）  交错",
        ROUNDS,
        "轮",
    )
    print("今天 `qwen.mojo` 是『每个 token 一行，逐行喂进一维的 matvec』：批=8 就把同样的权重流 8 趟。")
    print("")

    var ok = True
    if not check_parity[2](8, x_raw, w_raw, d_raw):
        ok = False
        print("  ⚠️ 自检失败：RB=2 批=8")
    if not check_parity[4](8, x_raw, w_raw, d_raw):
        ok = False
        print("  ⚠️ 自检失败：RB=4 批=8")
    if not check_parity[8](8, x_raw, w_raw, d_raw):
        ok = False
        print("  ⚠️ 自检失败：RB=8 批=8")
    if not check_parity[8](16, x_raw, w_raw, d_raw):
        ok = False
        print("  ⚠️ 自检失败：RB=8 批=16")
    if not ok:
        print("⚠️ 自检没过（含负向对照没被抓红）→ 下面的时间和比值一律不许引用。")
        arena.keep_alive()
        return

    print("--- RB = 2（累加器 8 条 ymm，稳稳装得下）---")
    bench_pair[2](8, x_raw, w_raw, d_raw)
    bench_pair[2](16, x_raw, w_raw, d_raw)
    print("--- RB = 4（累加器 16 条 ymm，正好等于寄存器堆）---")
    bench_pair[4](8, x_raw, w_raw, d_raw)
    bench_pair[4](16, x_raw, w_raw, d_raw)
    print("--- RB = 8（累加器 32 条 ymm，必然溢到栈）---")
    bench_pair[8](8, x_raw, w_raw, d_raw)
    bench_pair[8](16, x_raw, w_raw, d_raw)
    bench_pair[8](32, x_raw, w_raw, d_raw)

    print("判据（跑之前就定死的）：同一个 RB 在批=8 与 16 **两档**的保守端都 > 1.10 才算数。")
    print("⚠️ 只量**无偏置**那条投影（down/gate/up）；q/k/v 带偏置，要另行补。")

    # ⚠️ arena 在**最后一次使用处**析构 —— 而分配出来的指针要用到这里之后。
    arena.keep_alive()
