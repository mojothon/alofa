"""批 / 预填的 GEMM 现在每行各把权重流一遍吗？（判定性探针）

问的是什么
----------
`avx2._gemm`（`scalar._gemm` 同构）的循环是

    for row in range(rows):          # 行外层
        for col in range(cols):
            ... 读 w[col][0..inner] ...   # 权重的一整行流进来

于是一个 `rows` 行的前向会把**整个权重矩阵读 `rows` 遍**：批 = 8 的 decode、
32 token 的 prefill，摊到每个 token 上的字节数和批 = 1 **一模一样**
（fp32 每 token 1.976 GB）。账本里那条「算术强度恒 0.5 flops/byte」不是
"GEMM 天生如此"，是**这个循环次序**造成的。

换成「列外层 / `inner` 中层 / 行内层」，`w[col][:]` 就只从 DRAM 读一次、
在 L1 里被 `rows` 行复用，每 token 字节数除以 `rows`。改之前先确认它值多少。

判据（口径先定，再看数）
------------------------
  * 主口径 = **每行耗时**（总时间 ÷ rows）。
    - 随 `rows` **不变** → 每行各流一遍（机制成立，有杠杆）；
    - 按 1/rows **掉** → 字节已经被摊薄了（那就没有这个杠杆）。
  * 辅助口径 = `rows × 17.4 MB ÷ 耗时` 这个「若每行各流一遍」的隐含带宽：
    跨 `rows` **基本恒定**且落在单线程读带宽附近 → 机制成立。
  * 效应量级 2–8× ≫ 本机噪声 ±10% → **一轮就够判定**，但结论**必须能复现**。
  * `rows > 8` 另有一档预期（2026-09-22 扩到 16 / 32）：`_gemm` 按 8/4/2/1
    **分块**，块内才复用，所以 `rows > 8` 时权重读 `ceil(rows / 8)` 遍 —— 每行
    耗时应按 `ceil(rows/8) / rows` 掉（1→8 摊薄 8×，而 16→32 只剩 2×）。把 8
    以上一并量出来，是为了让「批大于 8 还是不是同一条曲线」有数，而不是停在
    白嫖的这一段。

自检（不通过就一个数都不报）
----------------------------
`rows` 一次算完必须与 `rows` 次 `rows=1` **逐位相等**。这条不只为这次探针，
它是**改循环次序时的判据**：换序不改变任何一次浮点运算 —— 每个输出各自一个
f64×4 累加器、按同样的 `k` 次序累加 —— 差 1 ulp 就是 bug。漏算一行会让时间
偏快、把结论引向**相反**方向。

    pixi run mojo run -O2 -I src scripts/bench_gemm_rows.mojo
"""

from std.runtime.asyncrt import TaskGroup

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import linear

# `down_proj`：896 个输出 × 4864 个输入，24 层里最大的一个投影。
comptime COLS = 896
comptime INNER = 4864
comptime MAX_ROWS = 32

# 权重副本份数：每趟算一份新的，8 份 = 139 MB ≫ L3（12 MB）。同块权重反复跑
# 会让后几趟落在 L3 里 —— `bench_thread_matmul.mojo` 第一版就是这么报出
# 「80 GB/s」的（本机理论上限 ~42 GB/s）。
comptime REGIONS = 8

# 读带宽天花板：256 MB，8 个**独立** f64 累加器（单个累加器会撞上 add 延迟
# 4 周期的链顶 ≈25.6 GB/s，那是"累加器链"的顶，不是 DRAM 的顶）。
comptime CEIL_BYTES = 256 * 1024 * 1024
comptime CEIL_ELEMS = CEIL_BYTES // 4
comptime CEIL_ROUNDS = 3

comptime W_BYTES = COLS * INNER * 4


def f2(v: Float64) -> String:
    """定点两位；这台机器上更多位数是假精度。"""
    var hundredths = Int(v * Float64(100))
    var frac = hundredths % 100
    var tail = String(frac) if frac >= 10 else "0" + String(frac)
    return String(hundredths // 100) + "." + tail


def read_seq(p: RawPtr, n: Int, sink: RawPtr, slot: Int) raises -> None:
    """扫一遍 `[p, p+n)`，8 个**独立**的 f64 累加器（理由见文件头）。"""
    var pd = f32_data(TensorView(p, shape2(1, n), DT_FP32))
    var sd = f32_data(TensorView(sink, shape2(1, 16), DT_FP32))
    var a0: Float64 = 0
    var a1: Float64 = 0
    var a2: Float64 = 0
    var a3: Float64 = 0
    var a4: Float64 = 0
    var a5: Float64 = 0
    var a6: Float64 = 0
    var a7: Float64 = 0
    var i = 0
    while i + 8 <= n:
        a0 += Float64(pd[unsafe_offset=i])
        a1 += Float64(pd[unsafe_offset=i + 1])
        a2 += Float64(pd[unsafe_offset=i + 2])
        a3 += Float64(pd[unsafe_offset=i + 3])
        a4 += Float64(pd[unsafe_offset=i + 4])
        a5 += Float64(pd[unsafe_offset=i + 5])
        a6 += Float64(pd[unsafe_offset=i + 6])
        a7 += Float64(pd[unsafe_offset=i + 7])
        i += 8
    while i < n:
        a0 += Float64(pd[unsafe_offset=i])
        i += 1
    # 写一个槽位：这段读不能被当成可消除的死代码。
    sd[unsafe_offset=slot] = Float32(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)


async def read_slice(p: RawPtr, n: Int, sink: RawPtr, slot: Int) -> None:
    """`read_seq` 的协程包装。

    ⚠️ 协程参数只能是平凡值（指针 / 整数）；`TensorView` 直接当协程参数实测会
    **静默写错地方**，视图一律在协程内现造。
    """
    try:
        read_seq(p, n, sink, slot)
    except err:
        _ = err


def slice_len(t: Int, k: Int, per: Int, total: Int) -> Int:
    if k == t - 1:
        return total - per * (t - 1)
    return per


def sweep_read(t: Int, base: RawPtr, sink: RawPtr) raises -> Int:
    """把 256 MB 切成 `t` 段扫一遍，返回纳秒；**自检失败返回 -1**。

    ⚠️ 数组填全 1，所以第 k 片累加出来必须**精确等于**该片元素数。少算任何一片
    （或协程参数被写坏、那片压根没跑）这里就红 —— 没有这一步，"天花板"能报出
    56 GB/s 这种超过 DDR4 双通道上限的数而不自知。
    """
    var sd = f32_data(TensorView(sink, shape2(1, 16), DT_FP32))
    for k in range(16):
        sd[unsafe_offset=k] = Float32(0)
    var t0 = monotonic_ns()
    if t == 1:
        read_seq(base, CEIL_ELEMS, sink, 0)
        var dt = monotonic_ns() - t0
        if sd[unsafe_offset=0] != Float32(CEIL_ELEMS):
            return -1
        return dt
    var per = (CEIL_ELEMS // t) // 8 * 8
    var tg = TaskGroup()
    for k in range(t):
        tg.create_task(
            read_slice(
                base.unsafe_offset(k * per * 4),
                slice_len(t, k, per, CEIL_ELEMS),
                sink,
                k,
            )
        )
    tg.wait()
    var dt = monotonic_ns() - t0
    for k in range(t):
        if sd[unsafe_offset=k] != Float32(slice_len(t, k, per, CEIL_ELEMS)):
            return -1
    return dt


def median_ns(v: List[Int]) -> Int:
    var s = List[Int]()
    for i in range(len(v)):
        s.append(v[i])
    for i in range(1, len(s)):
        var key = s[i]
        var j = i - 1
        while j >= 0 and s[j] > key:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = key
    return s[len(s) // 2]


def ceiling_read(t: Int) raises -> Float64:
    """T 个并发任务纯读能到多少 B/s（3 趟取**中位数**，不取最优）。

    自检失败返回 **-1**：那个数不许进判定，也**不许**被引用。
    """
    var arena = Arena(CEIL_BYTES + 64)
    var base = arena.alloc(CEIL_BYTES)
    var sink = arena.alloc(64)
    var pv = f32_data(TensorView(base, shape2(1, CEIL_ELEMS), DT_FP32))
    for i in range(CEIL_ELEMS):
        pv[unsafe_offset=i] = Float32(1)
    # 预热：第一趟要付匿名页的缺页，那不是带宽。
    _ = sweep_read(t, base, sink)
    var times = List[Int]()
    var ok = True
    for _ in range(CEIL_ROUNDS):
        var dt = sweep_read(t, base, sink)
        if dt < 0:
            ok = False
        else:
            times.append(dt)
    _ = f32_data(TensorView(sink, shape2(1, 16), DT_FP32))[unsafe_offset=0]
    arena.keep_alive()
    if not ok or len(times) == 0:
        return -1.0
    return Float64(CEIL_BYTES) * 1000000000.0 / Float64(median_ns(times))


def call_linear(rows: Int, d: RawPtr, x: RawPtr, w: RawPtr) raises -> None:
    linear(
        TensorView(d, shape2(rows, COLS), DT_FP32),
        TensorView(x, shape2(rows, INNER), DT_FP32),
        TensorView(w, shape2(COLS, INNER), DT_FP32),
    )


def bench_rows(rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr) raises -> Int:
    """在 `REGIONS` 份互不相交的权重上各算一趟，返回**单趟**的纳秒。"""
    var t0 = monotonic_ns()
    for r in range(REGIONS):
        call_linear(
            rows, d_raw, x_raw, w_raw.unsafe_offset(r * COLS * INNER * 4)
        )
    var t1 = monotonic_ns()
    _ = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))[
        unsafe_offset=0
    ]
    return (t1 - t0) // REGIONS


def check_exact(
    rows: Int, x_raw: RawPtr, w_raw: RawPtr, d_raw: RawPtr
) raises -> Bool:
    """`rows` 一次算完 == `rows` 次 `rows=1`，**逐位**。

    这是改循环次序时的判据，不只是这次探针的自检：换序不改变任何一次浮点
    运算（每个输出各自一个 f64×4 累加器、按同样的 `k` 次序累加）。
    """
    var dv = f32_data(TensorView(d_raw, shape2(rows, COLS), DT_FP32))
    call_linear(rows, d_raw, x_raw, w_raw)
    var got = List[Float32]()
    for i in range(rows * COLS):
        got.append(dv[unsafe_offset=i])
    for i in range(rows * COLS):
        dv[unsafe_offset=i] = Float32(-999)
    for r in range(rows):
        call_linear(
            1,
            d_raw.unsafe_offset(r * COLS * 4),
            x_raw.unsafe_offset(r * INNER * 4),
            w_raw,
        )
    for i in range(rows * COLS):
        if dv[unsafe_offset=i] != got[i]:
            return False
    return True


def main() raises:
    var n_w = COLS * INNER
    var arena = Arena(
        REGIONS * n_w * 4
        + MAX_ROWS * INNER * 4
        + MAX_ROWS * COLS * 4
        + 4096
    )
    var w_raw = arena.alloc(REGIONS * n_w * 4)
    var x_raw = arena.alloc(MAX_ROWS * INNER * 4)
    var d_raw = arena.alloc(MAX_ROWS * COLS * 4)

    var pw = f32_data(TensorView(w_raw, shape2(REGIONS * COLS, INNER), DT_FP32))
    var px = f32_data(TensorView(x_raw, shape2(MAX_ROWS, INNER), DT_FP32))
    var i = 0
    while i < REGIONS * n_w:
        pw[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1
    i = 0
    while i < MAX_ROWS * INNER:
        px[unsafe_offset=i] = Float32(i % 13) - Float32(6)
        i += 1

    # 预热 + 形状契约：这次直呼会把任何形状错误抛出来，而不是被吃掉。
    call_linear(1, d_raw, x_raw, w_raw)

    print("=== 批 / 预填的 GEMM：每行各把权重流一遍吗？ ===")
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
        "MB（≫ L3 12 MB）",
    )

    var b1 = ceiling_read(1)
    var b8 = ceiling_read(8)
    print("同进程现测读带宽：T=1", f2(b1 / 1e9), "GB/s   T=8", f2(b8 / 1e9), "GB/s")
    if b1 <= Float64(0) or b8 <= Float64(0):
        print("⚠️ 读带宽自检失败 → 下面的「差多少倍」一律不许引用。")
    print("")

    var rows_list = List[Int]()
    rows_list.append(1)
    rows_list.append(2)
    rows_list.append(4)
    rows_list.append(8)
    rows_list.append(16)
    rows_list.append(32)

    var ok = True
    for ri in range(len(rows_list)):
        var rows = rows_list[ri]
        if not check_exact(rows, x_raw, w_raw, d_raw):
            ok = False
            print("  rows =", rows, "  ⚠️ 自检失败：与逐行算不逐位相等")
            continue
        var ns = bench_rows(rows, x_raw, w_raw, d_raw)
        var per_row = Float64(ns) / Float64(rows)
        # 「若每行各流一遍权重」的隐含带宽。跨 rows 恒定 → 机制成立。
        var implied = Float64(rows * W_BYTES) * 1000000000.0 / Float64(ns)
        # 换序后的访存地板：整份权重只读一次。
        var floor1 = Float64(0)
        var floor8 = Float64(0)
        if b1 > Float64(0):
            floor1 = Float64(W_BYTES) * 1000000000.0 / b1
        if b8 > Float64(0):
            floor8 = Float64(W_BYTES) * 1000000000.0 / b8
        print(
            "  rows =",
            rows,
            "  一趟",
            f2(Float64(ns) / 1e6),
            "ms   每行",
            f2(per_row / 1e6),
            "ms   隐含",
            f2(implied / 1e9),
            "GB/s",
            "   换序后地板 T=1",
            f2(floor1 / 1e6),
            "ms / T=8",
            f2(floor8 / 1e6),
            "ms",
        )
        if floor1 > Float64(0):
            print(
                "            实测 ÷ 地板 =",
                f2(Float64(ns) / floor1),
                "×（T=1）",
                f2(Float64(ns) / floor8),
                "×（T=8）  ← 这是这一格最多能拿到的倍数",
            )

    if not ok:
        print("")
        print("⚠️ 有档位没通过自检 → 上面的数一律不许引用。")
    arena.keep_alive()
