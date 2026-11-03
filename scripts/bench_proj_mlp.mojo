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
from alofa.kernels.cpu.avx2 import Q4_U8, matmul_q4_f32

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
    var t = List[Int]()
    var rows = List[Int]()
    rows.append(HID)
    rows.append(HID)
    rows.append(KV_DIM)
    rows.append(KV_DIM)
    rows.append(KV_DIM)
    rows.append(HID)
    var sink2 = Float32(0)
    for _ in range(REPS):
        var t0 = monotonic_ns()
        # 24 层各一段互不相交的权重（真 decode 就是每层读自己那份）
        for layer in range(LAYERS):
            var w = buf.unsafe_offset(layer * PROJ_ROWS * HID // BLOCK * BB)
            var o = 0
            # q / k / v / o / gate / up / down
            matmul_q4_f32(dst, x, w, HID, HID)
            sink2 += dst[0]
            o += HID
            w = w.unsafe_offset(HID * HID // BLOCK * BB)
            matmul_q4_f32(dst, x, w, KV_DIM, HID)
            sink2 += dst[0]
            w = w.unsafe_offset(KV_DIM * HID // BLOCK * BB)
            matmul_q4_f32(dst, x, w, KV_DIM, HID)
            sink2 += dst[0]
            w = w.unsafe_offset(KV_DIM * HID // BLOCK * BB)
            matmul_q4_f32(dst, x, w, HID, HID)
            sink2 += dst[0]
            w = w.unsafe_offset(HID * HID // BLOCK * BB)
            matmul_q4_f32(dst, x, w, INTER, HID)
            sink2 += dst[0]
            w = w.unsafe_offset(INTER * HID // BLOCK * BB)
            matmul_q4_f32(dst, x, w, INTER, HID)
            sink2 += dst[0]
            w = w.unsafe_offset(INTER * HID // BLOCK * BB)
            matmul_q4_f32(dst, x, w, HID, INTER)
            sink2 += dst[0]
        t.append(monotonic_ns() - t0)
    var nsb = min_ns(t)
    # 总量按真实 7 个投影算：q/k/v/o 用 HID 列，down 用 INTER 列
    var b_layer = (HID + KV_DIM + KV_DIM + HID + INTER + INTER) * HID // BLOCK * BB + HID * INTER // BLOCK * BB
    var bytes_b = b_layer * LAYERS
    var gbs_b = Float64(bytes_b) / Float64(nsb)
    print("     总量 " + String(bytes_b // (1024 * 1024)) + " MB，" + f2(Float64(nsb) / 1e6) + " ms  ⇒ " + f2(gbs_b) + " GB/s")
    print("     对比：profiler 在真机 decode 上量到 33.4–33.9 ms / 201 MB = 5.9–6.0 GB/s")
    if sink2 == Float32(1234.5):
        print("     (unreachable)")

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

    # ⚠️ 必须是 `main` 的最后一条语句：这是 `arena` 的**最后一次使用**，Mojo 的
    # ASAP 销毁正好发生在这里。写在上面的 `alloc` 之后等于把寿命截断在那里，
    # 紧接着的 C 表就会读进已 munmap 的地址（本探针第一、二版都崩在这）。
    arena.keep_alive()
