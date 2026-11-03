"""层投影的单核带宽 vs 在飞 load 数 —— 决定"改核"还是"改切分粒度"。

为什么要问这个
--------------
`scripts/bench_decode_budget.mojo`（2026-11-03）量出 decode 的分段预算：

    核本身（q4，96 次/token）  33.4–33.9 ms = 55%   ← 201 MB / token
    输出投影 lm_head          23.8–24.0 ms = 38%   ←  76.6 MB / token
    全部串行算子               0.82 ms     = 1.3%

而 `shards=8` 对两者的扩展差得很远：`lm_head` 拿到 **2.1×**，层投影只有 **1.33×**。
同样是从内存搬字节，输出头接近线性、层投影不行 —— 这说明问题出在**层投影这条路
喂不饱内存**，而不是"并行没切到"（串行段实测只有 1.3%，已经没有东西可切了）。

"喂不饱"有两种病因，修法完全不同：

* **病一：核没有足够的访存在飞量（MLP）** —— 单线程一次只挂几个 cache miss，
  DRAM 延迟（~80 ns）没被流水线盖住 ⇒ 修法是**改核**（软件流水、多路独立累加、
  一次挂 8–16 个流）。
* **病二：单核本来就只能到这么多** —— 那就只能靠**更多核**，也就是改切分粒度 /
  线程数；改核收益为零。

在量出"单核纯读能到多少 GB/s"之前，选哪条都是猜。

三张表
------
* **A 单核纯读 vs 在飞流数**（1/2/4/8/16/32 条独立流，同一块 201 MB）：硬件 MLP 上限
* **B 真实 q4 核**：一趟 decode 的全部层投影（24 层 × 7 投影 = 201 MB），实测 GB/s
* **C 多线程纯读**（1/2/4/8 线程，各读互不相交的一段）：DRAM 总带宽天花板

判据（先写下来，免得看到数字再编解释）：

    B/A_单核  ≈ 1  ⇒ 核已经把单核吃满 ⇒ **改核无收益**，只能加核（改切分粒度）
    B/A_单核 << 1  ⇒ 核没吃满单核 ⇒ **改核**有量级空间，先别碰切分
    C_8核/A_单核 接近 1 ⇒ DRAM 本来就窄，谁上都一样（本机是共享机器，很可能如此）

口径
----
* **单线程**跑 A 与 B；C 才是多线程，且 C 只用来夹逼天花板。
* 每档 `REPS` 轮取 **min**：本机长期过载（loadavg 7–9），min 最接近"这核自己有多快"。
* 缓冲**先写一遍**再计时：201 MB 的匿名映射第一次碰页会全 fault，不预热量到的是缺页。
* q4 块的 scale 字节统一填 fp16 `1.0`（`0x3C00`），nibble 全 0 —— **避开 inf/NaN**：
  `block_scale` 的 inf/NaN 是**冷分支**（账本 11-03 第二条踩过，那次把冷路径的
  `call` 数进了循环体，得出过"每块每行 83 条"的错数），这里不能让它混进热路径。
* ⚠️ micro-bench，不是门，不进 `pixi run test`。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_proj_mlp.mojo -o target/bench_proj_mlp
    ./target/bench_proj_mlp
"""

from std.runtime.asyncrt import TaskGroup

from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, RawPtr
from std.sys import llvm_intrinsic

from alofa.kernels.cpu.avx2 import (
    Q4_U8,
    matmul_q4_f32,
    matmul_q4_mub,
    matmul_q4_mub_rows_band,
)
from alofa.kernels.cpu.quant import block_scale

# ---- 真实形状（Qwen2.5-0.5B，decode = 每次 1 个 token）----
comptime HID = 896
comptime INTER = 4864
comptime KV_DIM = 128
comptime LAYERS = 24

comptime BLOCK = 32
comptime BB = 18  # q4_0：2 字节 fp16 scale + 16 字节 nibble

comptime REPS = 7

# 层投影每 token 的权重字节：24 × (q 896 + k 128 + v 128 + o 896 + gate 4864 + up 4864 + down 4864 行)
comptime PROJ_ROWS = HID + KV_DIM + KV_DIM + HID + INTER + INTER + HID
comptime PROJ_BYTES = LAYERS * PROJ_ROWS * HID // BLOCK * BB

# 与 201 MB 同量级的纯读缓冲（多留一点给对齐与 C 表切片）
comptime BUF = 240 * 1024 * 1024


def min_ns(mut v: List[Int]) -> Int:
    var m = v[0]
    for i in range(1, len(v)):
        if v[i] < m:
            m = v[i]
    return m


def f2(x: Float64) -> String:
    """两位小数（本机没有 sprintf）。"""
    var neg = x < Float64(0)
    var v = -x if neg else x
    var whole = Int(v)
    var frac = Int((v - Float64(whole)) * Float64(100) + Float64(0.5))
    if frac >= 100:
        whole += 1
        frac -= 100
    var s = String(whole) + "."
    if frac < 10:
        s = s + "0"
    s = s + String(frac)
    return "-" + s if neg else s


def read_multi[K: Int](base: RawPtr, nbytes: Int) -> UInt64:
    """单线程、`K` 条互不相交的流各读 `nbytes/K` 字节，返回一个无意义的和。

    `K` 是核心自变量：**只有 K 条流各自独立**，它们的 cache miss 才能同时在飞。
    `@parameter for` 让内层完全展开，LLVM 才不会把相邻的流合并成一个顺序扫描
    （合并了就退化成 K=1，这条探针就白跑了）。
    """
    var per = nbytes // K
    per = (per // 64) * 64
    var n8 = per // 8
    var p = base.bitcast[UInt64]()
    var acc = SIMD[DType.uint64, 8]()
    var i = 0
    while i < n8:
        @parameter
        for s in range(K):
            acc += (p + s * n8 + i).load[width=8]()
        i += 8
    return acc[0] + acc[7]


async def _read_task(base: RawPtr, nbytes: Int, sink: Pointer[UInt64, MutUntrackedOrigin]):
    """一条线程读一段。`sink` 只为防优化，不参与测量。"""
    var per = (nbytes // 64) * 64
    var p = base.bitcast[UInt64]()
    var acc = SIMD[DType.uint64, 8]()
    var n8 = per // 8
    var i = 0
    while i < n8:
        acc += (p + i).load[width=8]()
        i += 8
    sink[0] = acc[0]


def fill_q4_scales(blocks: Q4_U8, nbytes: Int):
    """整块写一遍（顺便把页 fault 掉），再把每个 18 字节块的 scale 设为 fp16 1.0。

    scale 必须是**有限值**：`block_scale` 对 inf/NaN 走冷分支（带 `raises` 的
    路径），一旦那些冷分支被真的执行，量到的就不是热路径了。
    """
    var p = blocks.bitcast[UInt64]()
    var n8 = nbytes // 8
    var z = UInt64(0)
    for i in range(n8):
        p[i] = z
    var nb = nbytes // BB
    for b in range(nb):
        var off = b * BB
        blocks[unsafe_offset=off] = UInt8(0)
        blocks[unsafe_offset=off + 1] = UInt8(0x3C)


comptime MUBSW = "llvm.x86.avx2.pmadd.ub.sw"
comptime MADDWD = "llvm.x86.avx2.pmadd.wd"
comptime MAX_BLOCKS = INTER // BLOCK


def matmul_q4_mub_hoist(
    dst: F32Ptr, x: F32Ptr, blocks: Q4_U8, rows: Int, cols: Int
) raises:
    """🔴 **失败实验（留作反面证据，别照抄）**：把 `d·sc` 的小预扫提到行外。

    **实测结果（2026-11-03，同轮交替）**：
      * **不逐位相同** —— 896 行里 896 行不一致，最大绝对差 12.61。所谓"纯重排"
        这个前提本身就是错的：Mojo 带 fast-math，`x·(d·sc)` 与 `(x·d)·sc` 的结合
        顺序不同 ⇒ **提前算 `d·sc` 会改数值**。
      * **更慢 0.64×**（B3 3.25 vs B2 5.12 GB/s）—— 省掉热循环里 7 条，却多出一趟
        `nb` 次标量预扫（每次还要过 `block_scale` 的守卫），**净亏**。

    ⇒ 想省那 5 条守卫只能做 `block_scale_fast`（去掉 inf/NaN 判断、保留 inline
    转换），约省 5/29 = 17%，核级预期 <1.1×，**不值得**。这个函数留着是因为
    "把不变量提出循环"在这类数值核里看起来天经地义，而它在 fast-math 下会静默
    改数 —— 比慢更危险。

    原文（未改）：生产 mub 核的**纯重排**版：把 `d·sc` 的小预扫提到行外。

    为什么这么改（`objdump` 实测，2026-11-03）：热循环 **29 条/块**，其中

        vpinsrw + vcvtph2ps            2 条   读 fp16 scale
        movzbl + mov + not + test + jne 5 条   inf/NaN 守卫（`block_scale` 内部）
        vmulss + vbroadcastss          2 条   算 `d * sc`

    **7 条（24%）只是把每块的 scale 取出来**，而它每块都要重算一遍、其中 5 条还是
    一个**永不成立的冷分支判断**。这里改成：每个输出行开始前，先用一趟只读 2 字节
    scale 的小循环把 `d_b * sc_b` 算进一个 `nb` 长的缓冲（`nb` = 28 或 152），
    点积循环里只剩一次 `vbroadcastss`。

    同一行的权重一共 `nb*18` 字节 = 504 B（`cols=896`），预扫与点积两趟贴着跑，
    第二趟全在 L1 —— **不增加 DRAM 流量**。

    ⚠️ 这是**纯重排**，不是新算术：`q + sh`、`cast`、`* dsc`、`+ acc` 一条不动，
    只是把同一个 `d * sc` 挪到循环外算 ⇒ 输出必须**逐位相同**（下面 `main` 里验）。
    """
    var xq_buf = InlineArray[Int8, INTER](uninitialized=True)
    var sumx_buf = InlineArray[Int32, MAX_BLOCKS * 8](uninitialized=True)
    var scale_buf = InlineArray[Float32, MAX_BLOCKS](uninitialized=True)
    var dsc_buf = InlineArray[Float32, MAX_BLOCKS](uninitialized=True)
    var xqp = xq_buf.unsafe_ptr().unsafe_bitcast[Int8]()
    var sump = sumx_buf.unsafe_ptr().unsafe_bitcast[Int32]()
    var scp = scale_buf.unsafe_ptr().unsafe_bitcast[Float32]()
    var dscp = dsc_buf.unsafe_ptr().unsafe_bitcast[Float32]()

    var nb = cols // BLOCK
    # --- 激活量化（与生产核一字不差地照抄，只做一次）---
    for b in range(nb):
        var xb = b * BLOCK
        var amax = Float32(0)
        for j in range(BLOCK):
            var v = x[unsafe_offset=xb + j]
            if v < Float32(0):
                v = -v
            if v > amax:
                amax = v
        var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
        scp[unsafe_offset=b] = amax / Float32(127.0)
        var s = Int(0)
        for j in range(BLOCK):
            var t = x[unsafe_offset=xb + j] * inv
            var qv = Int(t + Float32(0.5)) if t >= Float32(0) else Int(t - Float32(0.5))
            if qv > 127:
                qv = 127
            if qv < -128:
                qv = -128
            xqp[unsafe_offset=xb + j] = Int8(qv)
            s += qv
        var sv = SIMD[DType.int32, 8](0)
        sv[0] = Int32(-8 * s)
        sump.unsafe_store[width=8](b * 8, sv)

    var low_mask = SIMD[DType.uint8, 16](0x0F)
    var four = SIMD[DType.uint8, 16](4)
    var ones = SIMD[DType.int16, 16](1)
    for row in range(rows):
        # --- 预扫：只取每块的 scale，算出 d*sc（守卫在这里，不在热循环里）---
        var rbase = row * nb * BB
        for b in range(nb):
            dscp[unsafe_offset=b] = block_scale(blocks, rbase + b * BB) * scp[unsafe_offset=b]
        var acc8 = SIMD[DType.float32, 8](0)
        var base = rbase + BB
        for b in range(nb):
            var raw = blocks.unsafe_load[width=16](base)
            var lo = raw & low_mask
            var hi = (raw >> four) & low_mask
            var w32 = lo.join(hi)
            var x32 = xqp.unsafe_load[width=32](b * BLOCK)
            var p = llvm_intrinsic[MUBSW, SIMD[DType.int16, 16]](w32, x32)
            var q = llvm_intrinsic[MADDWD, SIMD[DType.int32, 8]](p, ones)
            var sh = sump.unsafe_load[width=8](b * 8)
            acc8 = (q + sh).cast[DType.float32]() * SIMD[DType.float32, 8](
                dscp[unsafe_offset=b]
            ) + acc8
            base += BB
        dst[unsafe_offset=row] = Float32(acc8.reduce_add())


comptime NTOK_MAX = 32


def time_band[RB: Int](buf: RawPtr, dst: F32Ptr, x: F32Ptr, n_tok: Int) raises -> Int:
    """`n_tok` 个 token 一批，走 `matmul_q4_mub_rows_band[RB]`（prefill 的实际通路）。"""
    var best = 0
    for _ in range(REPS):
        var t0 = monotonic_ns()
        for layer in range(LAYERS):
            var w = buf.unsafe_offset(layer * PROJ_ROWS * HID // BLOCK * BB)
            matmul_q4_mub_rows_band[RB](dst, x, w, n_tok, HID, HID, HID)
            w = w.unsafe_offset(HID * HID // BLOCK * BB)
            matmul_q4_mub_rows_band[RB](dst, x, w, n_tok, KV_DIM, HID, KV_DIM)
            w = w.unsafe_offset(KV_DIM * HID // BLOCK * BB)
            matmul_q4_mub_rows_band[RB](dst, x, w, n_tok, KV_DIM, HID, KV_DIM)
            w = w.unsafe_offset(KV_DIM * HID // BLOCK * BB)
            matmul_q4_mub_rows_band[RB](dst, x, w, n_tok, HID, HID, HID)
            w = w.unsafe_offset(HID * HID // BLOCK * BB)
            matmul_q4_mub_rows_band[RB](dst, x, w, n_tok, INTER, HID, INTER)
            w = w.unsafe_offset(INTER * HID // BLOCK * BB)
            matmul_q4_mub_rows_band[RB](dst, x, w, n_tok, INTER, HID, INTER)
            w = w.unsafe_offset(INTER * HID // BLOCK * BB)
            matmul_q4_mub_rows_band[RB](dst, x, w, n_tok, HID, INTER, HID)
        var el = monotonic_ns() - t0
        if best == 0 or el < best:
            best = el
    return best


def report[RB: Int](ns: Int, n_tok: Int, bytes_b: Int):
    """按 `遍数 = ceil(n_tok/RB)` 折算权重流量，打印隐含带宽与每 token 时间。"""
    var passes = (n_tok + RB - 1) // RB
    var tb = passes * bytes_b
    print(
        "     RB=" + String(RB) + "  n_tok=" + String(n_tok)
        + "   " + f2(Float64(ns) / 1e6) + " ms"
        + "   遍数 " + String(passes) + " ⇒ " + f2(Float64(tb) / 1048576.0) + " MB"
        + " ⇒ " + f2(Float64(tb) / Float64(ns)) + " GB/s"
        + " ⇒ 每 token " + f2(Float64(ns) / Float64(n_tok) / 1e6) + " ms"
    )


def _mm[mode: Int](dst: F32Ptr, x: F32Ptr, w: Q4_U8, r: Int, c: Int) raises:
    """`mode` 0 = f32 展开，1 = mub（生产），2 = mub 纯重排版。"""
    comptime if mode == 0:
        matmul_q4_f32(dst, x, w, r, c)
    elif mode == 1:
        matmul_q4_mub(dst, x, w, r, c)
    else:
        matmul_q4_mub_hoist(dst, x, w, r, c)


def time_layers[mode: Int](buf: RawPtr, dst: F32Ptr, x: F32Ptr) raises -> Int:
    """一趟 decode 的全部层投影（24 层 x 7 投影 = 191 MB），返回最小 ns。"""
    var best = 0
    for _ in range(REPS):
        var t0 = monotonic_ns()
        # 24 层各一段互不相交的权重（真 decode 就是每层读自己那份）
        for layer in range(LAYERS):
            var w = buf.unsafe_offset(layer * PROJ_ROWS * HID // BLOCK * BB)
            # q / k / v / o / gate / up / down
            _mm[mode](dst, x, w, HID, HID)
            w = w.unsafe_offset(HID * HID // BLOCK * BB)
            _mm[mode](dst, x, w, KV_DIM, HID)
            w = w.unsafe_offset(KV_DIM * HID // BLOCK * BB)
            _mm[mode](dst, x, w, KV_DIM, HID)
            w = w.unsafe_offset(KV_DIM * HID // BLOCK * BB)
            _mm[mode](dst, x, w, HID, HID)
            w = w.unsafe_offset(HID * HID // BLOCK * BB)
            _mm[mode](dst, x, w, INTER, HID)
            w = w.unsafe_offset(INTER * HID // BLOCK * BB)
            _mm[mode](dst, x, w, INTER, HID)
            w = w.unsafe_offset(INTER * HID // BLOCK * BB)
            _mm[mode](dst, x, w, HID, INTER)
        var el = monotonic_ns() - t0
        if best == 0 or el < best:
            best = el
    return best


def main() raises:
    print("")
    print("== 层投影：单核带宽 vs 在飞 load 数 ==")
    print("   机器 8 核；层投影每 token 过 " + String(PROJ_BYTES) + " 字节权重")

    var arena = Arena(BUF + 8 * 1024 * 1024)
    var buf = arena.alloc(BUF)
    print("   [a] 缓冲 " + String(BUF // (1024 * 1024)) + " MB 就绪，正在写页")
    fill_q4_scales(buf, BUF)
    print("   [b] 页已 fault，scale 已置 fp16 1.0")

    # ---------------- A：单核纯读 vs 在飞流数 ----------------
    print("")
    print("   A 单核纯读（同一块 201 MB，K 条独立流）")
    print("     K     GB/s")
    var a1 = 0
    var a8 = 0
    for kk in range(6):
        var t = List[Int]()
        var sink = UInt64(0)
        for _ in range(REPS):
            var t0 = monotonic_ns()
            if kk == 0:
                sink += read_multi[1](buf, PROJ_BYTES)
            elif kk == 1:
                sink += read_multi[2](buf, PROJ_BYTES)
            elif kk == 2:
                sink += read_multi[4](buf, PROJ_BYTES)
            elif kk == 3:
                sink += read_multi[8](buf, PROJ_BYTES)
            elif kk == 4:
                sink += read_multi[16](buf, PROJ_BYTES)
            else:
                sink += read_multi[32](buf, PROJ_BYTES)
            t.append(monotonic_ns() - t0)
        var ns = min_ns(t)
        var gbs = Float64(PROJ_BYTES) / Float64(ns)
        print("     " + String(1 << kk) + "     " + f2(gbs))
        if kk == 0:
            a1 = ns
        if kk == 3:
            a8 = ns
        if sink == UInt64(0xDEADBEEF):
            print("     (unreachable)")

    # ---------------- B：真实 q4 核 ----------------
    print("")
    print("   B 真实 q4 核（24 层 × 7 投影 = 一趟 decode 的全部层投影）")
    var dst = arena.alloc(INTER * 4).bitcast[Float32]()
    var x = arena.alloc(HID * 4).bitcast[Float32]()
    for i in range(HID):
        x[i] = Float32(0.01) * Float32(i % 7)
    # 总量按真实 7 个投影算：q/k/v/o 用 HID 列，down 用 INTER 列
    var b_layer = (HID + KV_DIM + KV_DIM + HID + INTER + INTER) * HID // BLOCK * BB + HID * INTER // BLOCK * BB
    var bytes_b = b_layer * LAYERS

    # --- 先验"纯重排"必须逐位相同 ---
    var dst2 = arena.alloc(INTER * 4).bitcast[Float32]()
    matmul_q4_mub(dst, x, buf, HID, HID)
    matmul_q4_mub_hoist(dst2, x, buf, HID, HID)
    var bad = 0
    var maxd = Float64(0)
    for i in range(HID):
        if dst[i] != dst2[i]:
            bad += 1
            var e = Float64(dst[i] - dst2[i])
            if e < Float64(0):
                e = -e
            if e > maxd:
                maxd = e
    print(
        "     逐位校验（mub vs 重排版，同输入 "
        + String(HID) + " 行）：不一致 " + String(bad) + "，最大绝对差 " + f2(maxd)
    )

    var nsb = time_layers[0](buf, dst, x)
    var gbs_b = Float64(bytes_b) / Float64(nsb)
    print("     B1 f32 展开核     " + f2(Float64(nsb) / 1e6) + " ms  ⇒ " + f2(gbs_b) + " GB/s")

    var nsm = time_layers[1](buf, dst, x)
    var gbs_m = Float64(bytes_b) / Float64(nsm)
    print("     B2 mub 整数点积   " + f2(Float64(nsm) / 1e6) + " ms  ⇒ " + f2(gbs_m) + " GB/s  ← 生产走这条")

    var nsh = time_layers[2](buf, dst, x)
    var gbs_h = Float64(bytes_b) / Float64(nsh)
    print("     B3 mub 重排版     " + f2(Float64(nsh) / 1e6) + " ms  ⇒ " + f2(gbs_h) + " GB/s")
    print("     B3/B2 = " + f2(Float64(nsm) / Float64(nsh)) + "×（同轮交替，只读比例）")
    print("     对比：profiler 在真机 decode 上量到 33.4–33.9 ms / 201 MB = 5.9–6.0 GB/s")

    # ---------------- C：多线程纯读 ----------------
    print("")
    print("   C 多线程纯读（各读互不相交的一段，201 MB 总量）")
    print("     线程   GB/s")
    var cc = 0
    var tc = List[Int]()
    var sink3 = arena.alloc(64).bitcast[UInt64]()
    for nt in range(5):
        var n = 1 if nt == 0 else (2 if nt == 1 else (4 if nt == 2 else (8 if nt == 3 else 16)))
        var best = 0
        var dummy = List[Int]()
        for _ in range(REPS):
            var t0 = monotonic_ns()
            var tg = TaskGroup()
            # ⚠️ 必须向下取整到 64 字节：`read_multi` 用 64 字节 SIMD load，
            # 切点不对齐会直接 SIGSEGV（本探针第一版就这么崩过）。
            var per = (PROJ_BYTES // n) // 64 * 64
            for s in range(n):
                tg.create_task(_read_task(buf.unsafe_offset(s * per), per, sink3))
            tg.wait()
            var el = monotonic_ns() - t0
            if best == 0 or el < best:
                best = el
        print("     " + String(n) + "      " + f2(Float64(PROJ_BYTES) / Float64(best)))
        if cc == 0:
            _ = tc.append(best)
        cc += 1

    print("")
    print("   判读（三条，缺一条就不能下结论）：")
    print("   ① B ÷ A(K=1)：若 ≈1，核已吃满单核 ⇒ 改核无收益，只能加核")
    print("   ② A(K=8..16) ÷ A(K=1)：MLP 还剩多少 —— 这是改核能拿到的上界")
    print("   ③ C(8) ÷ A(K=1)：DRAM 总带宽是不是本来就窄；若 C(8) ≈ A(K=8)，")
    print("      说明单核只要挂够流就能吃满 DRAM，加核的边际收益就很小")
    print("")
    print("   ⚠️ 本机 loadavg 7–9（共享机器），绝对值会被抢核抬高；只看同一轮内的比例。")

    # ---------------- D：prefill 行带的 RB 到底是几 ----------------
    #
    # 账本 11-03 第十一条用遍数模型 `passes = shards × ceil(每片行数 / RB)` 反推
    # prefill 的隐含带宽，拟合结果是 **RB=2**（单核 4.9 GB/s、4 片 9.5 GB/s 都自洽）。
    # 这张表直接在核级别上量 RB=2/4/8 —— 如果 RB 提上去流量真能按比例降，那 prefill
    # 的最大杠杆就是改 RB（一行默认值），而不是重写 GEMM。
    print("")
    print("   D prefill 行带 RB（24 层 x 7 投影 = 一整个 prompt 的层投影）")
    var xb = arena.alloc(NTOK_MAX * INTER * 4).bitcast[Float32]()
    var db = arena.alloc(NTOK_MAX * INTER * 4).bitcast[Float32]()
    for i in range(NTOK_MAX * INTER):
        xb[i] = Float32(0.01) * Float32(i % 7)
    var toks = List[Int]()
    toks.append(16)
    toks.append(32)
    for ti in range(2):
        var nt = toks[ti]
        report[2](time_band[2](buf, db, xb, nt), nt, bytes_b)
        report[4](time_band[4](buf, db, xb, nt), nt, bytes_b)
        report[8](time_band[8](buf, db, xb, nt), nt, bytes_b)
        report[16](time_band[16](buf, db, xb, nt), nt, bytes_b)

    # ⚠️ 必须是 `main` 的最后一条语句：这是 `arena` 的**最后一次使用**，Mojo 的
    # ASAP 销毁正好发生在这里。写在上面的 `alloc` 之后等于把寿命截断在那里，
    # 紧接着的 C 表就会读进已 munmap 的地址（本探针第一、二版都崩在这）。
    arena.keep_alive()
