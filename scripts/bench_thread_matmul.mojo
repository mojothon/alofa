"""核级 A/B：同一个真形状按输出行切成 T 份并发跑，能快多少。

它接着 `bench_thread_bandwidth.mojo` 的那个结论往下问
----------------------------------------------------
那份探针量的是**纯访存**：聚合读带宽 T=1 11.39–14.39 → T=8 17.68–22.69 GB/s，
区间不重合，保守加速 **1.23×…1.99×**。但它只回答「多线程值不值得做」，没回答
「做了能拿多少」 —— 因为前向不是纯访存：

  * `fp32/avx2` 在 `down_proj` 上已经跑到**单线程**读峰值的 78%（访存受限）；
    它能吃到的加速**不可能**超过「新峰值 ÷ 0.78×旧峰值」。
  * `q4_0/avx2` 只用了自己访存地板的 10.6%（**算术受限**）→ 理论上更接近线性，
    但那要看运行时的调度开销吃掉多少。

所以这里把**同一个真形状**（`down_proj`，896 × 4864）按输出列切成 T 份，两条
通路都量，T=1 用**直呼**（就是今天线上那条通路）当基线。

为什么切输出列而不是切 `inner`
------------------------------
批大小为 1 时 `x` 只有一行，输出是 896 个标量，每个标量是 `w` 的一整行（4864
个权重）与 `x` 的点积 —— 896 个点积**互不依赖**，这就是全部的并行度。切 `inner`
会引入跨片的归约（要么原子加要么再一趟合并），那是**额外的**同步，不是这次要
量的东西。

⚠️ 为什么**不用** best-of-3 反复跑同一块权重
--------------------------------------------
第一版就是这么写的，然后它报出了 **T=8 时 0.216 ms** —— 那是 17.4 MB ÷ 0.216 ms
= **80 GB/s**，比这台机的聚合 DRAM 峰值（T=8 实测 22.69 GB/s）高 **3.5 倍**，
物理上不可能。原因：`down_proj` 的权重只有 **17.4 MB**，本机 L3 是 **12 MB** ——
同一块权重连跑 3 遍，第 2、3 遍里有相当一部分**根本没走 DRAM**。而真实前向每
token 要流 **1.976 GB**，一次都不会有这种复用。

→ 改成：**8 份互不相交的权重副本（139 MB）依次算一遍，取平均**。每一趟读的都
是一块全新的 17.4 MB，L3 复用被压到可忽略；同时也就不再"挑最好那一次"。

测量口径
--------
  * **必须 `-O2`**。
  * 每格 = 8 趟的平均（不取最优），**3 轮**，报 **min…max** 区间；加速比给
    「基线 max ÷ 自己 min」与「基线 min ÷ 自己 max」两端 —— **两端都 > 1 才算
    有差别**。
  * **交错**：外层轮次、内层任务数，两条通路在同一轮里交替量。
  * T=1 走**直呼**，不走 `TaskGroup` —— 基线必须是「今天线上跑的那条路」，
    否则量出来的是「换调度机制的收益」，不是「并行的收益」。

跑法：

    pixi run mojo run -O2 -I src scripts/bench_thread_matmul.mojo
"""

from std.runtime.asyncrt import TaskGroup

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import Q4_BLOCK, Q4_BYTES, Q4_U8, linear, matmul_q4_f32
from alofa.kernels.cpu.quant import quantize_q4_0

# `down_proj`：896 个输出 × 4864 个输入。它是 24 层里最大的一个投影。
comptime COLS = 896
comptime INNER = 4864
comptime ROUNDS = 3

# 权重副本的份数：每趟算一份新的，8 份 = 139 MB ≫ L3（12 MB）。份数再多只是
# 拉长单次测量、让它在噪声里漂，压 L3 复用 8 份已经够。
comptime REGIONS = 8

comptime W_BYTES = COLS * INNER * 4
comptime BYTES_PER_Q4_ROW = (INNER // Q4_BLOCK) * Q4_BYTES
comptime Q4_BYTES_PER_REGION = COLS * BYTES_PER_Q4_ROW


async def run_linear(dst: TensorView, x: TensorView, w: TensorView):
    """一片输出列上的 `linear`。

    ⚠️ `linear` 是 `raises` 的，而 `TaskGroup.create_task` 只收**不抛错**的协程
    （`RaisingCoroutine` 传不进去）。形状由调用方在进协程**之前**用同一次直呼验
    过，所以这里的 `except` 不是"忽略错误"，是"形状契约已经在别处守着"。
    """
    try:
        linear(dst, x, w)
    except err:
        _ = err


async def run_q4(dst: F32Ptr, x: F32Ptr, blocks: Q4_U8, rows: Int, cols: Int):
    """一片输出行上的 q4_0 融合投影；同上，`raises` 由调用方先验证。"""
    try:
        matmul_q4_f32(dst, x, blocks, rows, cols)
    except err:
        _ = err


def ns_per_elem(ns: Float64) -> Float64:
    return ns / Float64(COLS * INNER)


def bench_fp32(
    t: Int, x: TensorView, w_raw: RawPtr, d_raw: RawPtr
) raises -> Float64:
    """把 896 个输出列切成 `t` 片，在 8 份互不相交的权重上各算一趟。

    返回**单趟**的平均纳秒（总时间 ÷ `REGIONS`）。不取最优 —— 取最优正是上面
    那个 80 GB/s 的来历。
    """
    var per = COLS // t
    var rem = COLS - per * t
    var t0 = monotonic_ns()
    for r in range(REGIONS):
        var wbase = w_raw.unsafe_offset(r * W_BYTES)
        if t == 1:
            linear(
                TensorView(d_raw, shape2(1, COLS), DT_FP32),
                x,
                TensorView(wbase, shape2(COLS, INNER), DT_FP32),
            )
        else:
            var tg = TaskGroup()
            var c0 = 0
            for k in range(t):
                var n = per + (rem if k == t - 1 else 0)
                # 分片视图是**栈上的局部量**：协程按值捕获（`TensorView` 是
                # Copyable），所以这里不需要任何堆容器 —— 那正是它将来能直接
                # 进产品代码的前提（一次 decode 有 169 次矩阵乘）。
                tg.create_task(
                    run_linear(
                        TensorView(d_raw.unsafe_offset(c0 * 4), shape2(1, n), DT_FP32),
                        x,
                        TensorView(
                            wbase.unsafe_offset(c0 * INNER * 4),
                            shape2(n, INNER),
                            DT_FP32,
                        ),
                    )
                )
                c0 += n
            tg.wait()
    var t1 = monotonic_ns()
    var out = f32_data(TensorView(d_raw, shape2(1, COLS), DT_FP32))
    # 读一点结果：重复的计算不该被当成可消除的死代码。
    _ = out[unsafe_offset=0]
    return Float64(t1 - t0) / Float64(REGIONS)


def bench_q4(t: Int, x: F32Ptr, blocks: Q4_U8, d: F32Ptr) raises -> Float64:
    """同上，q4_0 通路：每行的块数是 `INNER / Q4_BLOCK`，按字节偏移切。"""
    var per = COLS // t
    var rem = COLS - per * t
    var t0 = monotonic_ns()
    for r in range(REGIONS):
        var bbase = blocks.unsafe_offset(r * Q4_BYTES_PER_REGION)
        if t == 1:
            matmul_q4_f32(d, x, bbase, COLS, INNER)
        else:
            var tg = TaskGroup()
            var c0 = 0
            for k in range(t):
                var n = per + (rem if k == t - 1 else 0)
                tg.create_task(
                    run_q4(
                        d.unsafe_offset(c0),
                        x,
                        bbase.unsafe_offset(c0 * BYTES_PER_Q4_ROW),
                        n,
                        INNER,
                    )
                )
                c0 += n
            tg.wait()
    var t1 = monotonic_ns()
    _ = d[unsafe_offset=0]
    return Float64(t1 - t0) / Float64(REGIONS)


def main() raises:
    var n_w = COLS * INNER
    var q4_bytes = n_w // Q4_BLOCK * Q4_BYTES
    var arena = Arena(
        REGIONS * n_w * 4 + INNER * 4 + COLS * 4 + REGIONS * q4_bytes + 4096
    )
    var w_raw = arena.alloc(REGIONS * n_w * 4)
    var x_raw = arena.alloc(INNER * 4)
    var d_raw = arena.alloc(COLS * 4)
    var blocks = arena.alloc(REGIONS * q4_bytes)

    var w = TensorView(w_raw, shape2(COLS, INNER), DT_FP32)
    var x = TensorView(x_raw, shape2(1, INNER), DT_FP32)
    var d = TensorView(d_raw, shape2(1, COLS), DT_FP32)
    var pw = f32_data(w)
    var px = f32_data(x)
    var i = 0
    while i < REGIONS * n_w:
        pw[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1
    i = 0
    while i < INNER:
        px[unsafe_offset=i] = Float32(i % 13) - Float32(6)
        i += 1
    for r in range(REGIONS):
        quantize_q4_0(
            blocks.unsafe_offset(r * Q4_BYTES_PER_REGION),
            pw.unsafe_offset(r * n_w),
            n_w,
        )

    var ts = List[Int]()
    ts.append(1)
    ts.append(2)
    ts.append(4)
    ts.append(8)

    # 预热 + 形状契约：这两次直呼会把任何形状错误**抛出来**，而不是让它在协程
    # 里被 `except` 吃掉。
    linear(d, x, w)
    matmul_q4_f32(f32_data(d), px, blocks, COLS, INNER)

    print("=== 核级：down_proj", COLS, "×", INNER, " 按输出切成 T 份 ===")
    print("fp32 每趟读", n_w * 4, "B；q4_0 每趟读", q4_bytes, "B")

    var f_lo = List[Float64]()
    var f_hi = List[Float64]()
    var q_lo = List[Float64]()
    var q_hi = List[Float64]()
    for _ in range(len(ts)):
        f_lo.append(Float64(0))
        f_hi.append(Float64(0))
        q_lo.append(Float64(0))
        q_hi.append(Float64(0))

    for r in range(ROUNDS):
        print("-- 第", r + 1, "轮 --")
        for ti in range(len(ts)):
            var t = ts[ti]
            var fns = bench_fp32(t, x, w_raw, d_raw)
            var qn = bench_q4(t, px, blocks, f32_data(d))
            if r == 0 or fns < f_lo[ti]:
                f_lo[ti] = fns
            if r == 0 or fns > f_hi[ti]:
                f_hi[ti] = fns
            if r == 0 or qn < q_lo[ti]:
                q_lo[ti] = qn
            if r == 0 or qn > q_hi[ti]:
                q_hi[ti] = qn
            print(
                "  T =",
                t,
                "  fp32/avx2",
                fns / 1e6,
                "ms (",
                ns_per_elem(fns),
                "ns/元素 )   q4_0/avx2",
                qn / 1e6,
                "ms (",
                ns_per_elem(qn),
                "ns/元素 )",
            )

    print("=== 区间（min…max）与相对各通路 T=1 的加速 ===")
    for ti in range(len(ts)):
        var t = ts[ti]
        print(
            "  T =",
            t,
            "  fp32 ",
            f_lo[ti] / 1e6,
            "…",
            f_hi[ti] / 1e6,
            "ms   加速 ",
            f_hi[0] / f_lo[ti],
            "…",
            f_lo[0] / f_hi[ti],
            "×",
        )
        print(
            "         q4    ",
            q_lo[ti] / 1e6,
            "…",
            q_hi[ti] / 1e6,
            "ms   加速 ",
            q_hi[0] / q_lo[ti],
            "…",
            q_lo[0] / q_hi[ti],
            "×",
        )

    arena.keep_alive()
