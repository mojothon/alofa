"""N 个并发任务能把读带宽拉到多少 —— 一个**硬件事实**探针，不是性能结论。

为什么先测这个
------------
账本里 alofa 与 llama.cpp 的差距，一半写在明处：同机 `-t 1` 5.94 tok/s 对
`-t 8` **25.18 tok/s（4.24×）**。而我们所有的带宽数字（本机 12 GB/s、远程
9.25 GB/s）都是**单线程**量出来的 —— 一个线程扫不满双通道 DDR4 的 DRAM，
这是硬件常识，但它是"常识"不代表在这台机上成立：**本机 8 核常年在 runq 6–27
之间，邻居不受我控制**。所以在动模型代码之前，先把这件事量成一个数：

  并发任务数 T 从 1 涨到 8，顺序读同一块 256 MB 数组能拿到多少聚合带宽？

它决定两件事：

  1. **多线程值不值得做**。若 T=8 只比 T=1 快 1.2×，那"并行化前向"就是在
     给一台已经饱和的机器加调度开销，做了也白做 —— 这个数会直接说"不值得"。
  2. **并行粒度能有多细**。前向一次 decode 有 24 层 × 7 个投影 + `lm_head`
     = **169 次矩阵乘**，最省事的接法是每次矩阵乘起一组任务。所以这里同时量
     **一次 `TaskGroup` 建 + 等的开销**，拿它乘 169 去看能不能接受。

为什么用 `std.runtime.asyncrt.TaskGroup`
---------------------------------------
Mojo 1.0.0 的标准库**没有** `thread` / `threading` / `concurrent`（逐个 import
试过，全是 `unable to locate module`），只有 `runtime.asyncrt` 这一层协程 +
任务组。`parallelism_level()` 报 8，与 `nproc` 一致 —— 它有线程池，且默认按
核数开。⚠️ 因此这里测出来的不只是"多线程的收益"，也包含**这套运行时的调度
开销**；如果调度开销把收益吃掉了，那也是结论的一部分，照样记下来。

测量口径（与本项目其它 bench 同一套）
------------------------------------
  * **必须 `-O2`**（`-O0` 下毫秒级 timing 不谈）。
  * **交错**：外层是轮次、内层是任务数，同一轮里把 1/2/4/6/8 都跑一遍。
    这样"这一轮机器整体快慢"是**共模**因子，不会只砸在某一個 T 上。
  * **不取最优**：每格给 5 轮的 max…min 区间。只报最好的那一次会让"有收益"
    这个结论变得不可证伪 —— 加轮数不是让结论变好看的旋钮。
  * 数组 256 MB ≫ L3（本机 12 MB），每轮每个字节都必须真的走 DRAM。

跑法：

    pixi run mojo run -O2 -I src scripts/bench_thread_bandwidth.mojo
"""

from std.runtime.asyncrt import TaskGroup, parallelism_level

from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import DT_FP32, F32Ptr, TensorView, f32_data, shape2

comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]

# 256 MB of fp32 —— 与 `bench_decode_roofline.mojo` 量带宽峰值时同一个尺寸，
# 两个文件报的数字因此可以直接对读。
comptime ELEMS = 64 * 1024 * 1024
comptime ROUNDS = 5
comptime BARRIER_ITERS = 2000

# 一次 decode 的矩阵乘次数：24 层 × 7 个投影 + lm_head。用来把「每次任务组
# 建 + 等」的开销换算成「每 token 付多少」。
comptime MATMULS_PER_TOKEN = 24 * 7 + 1


async def shard(p: F32Ptr, off: Int, n: Int, dst: F64Ptr, k: Int):
    """读 `[off, off+n)` 求和，结果写进 `dst[k]`。

    求和只是为了让编译器不能把读删掉 —— 每个元素只用一次，形式与 GEMV 读
    权重一致。用 8 宽 SIMD 是复用现有 bench 的写法：这里要量的是带宽，不是
    标量循环有多慢。
    """
    var acc = SIMD[DType.float32, 8](Float32(0))
    var i = off
    var end = off + n
    while i < end:
        acc += p.unsafe_load[width=8](i)
        i += 8
    dst[unsafe_offset=k] = Float64(acc.reduce_add())


async def nop():
    """空任务：用来单独量「建一组任务 + 等它结束」的开销。"""
    pass


def rate_per_s(count: Float64, ns: Float64) -> Float64:
    """`count` 个东西在 `ns` 纳秒里完成 → 每秒多少个。"""
    return count * 1_000_000_000.0 / ns


def gb_per_s(bytes_: Int, ns: Int) -> Float64:
    return rate_per_s(Float64(bytes_), Float64(ns)) / 1_000_000_000.0


def bench_barrier(t: Int) raises -> Float64:
    """建一组 `t` 个空任务并等它结束，平均每次多少纳秒。

    ⚠️ 计时包含 `nop()` 的调用（协程帧是在调用点分配的）—— 只量 `wait()`
    会把最贵的那一半漏掉，那正是"粒度能不能细"这个问题的答案所在。
    """
    var total = Int(0)
    for _ in range(BARRIER_ITERS):
        var tg = TaskGroup()
        var t0 = monotonic_ns()
        for _ in range(t):
            tg.create_task(nop())
        tg.wait()
        var t1 = monotonic_ns()
        total += t1 - t0
    return Float64(total) / Float64(BARRIER_ITERS)


def main() raises:
    var arena = Arena(ELEMS * 4 + 512)
    var p = f32_data(TensorView(arena.alloc(ELEMS * 4), shape2(1, ELEMS), DT_FP32))
    var dst = arena.alloc(512).unsafe_bitcast[Float64]()
    for i in range(ELEMS):
        p[unsafe_offset=i] = Float32(i % 17) - Float32(8)

    # 1 是基线：它也走 TaskGroup（1 个任务），所以「多线程的收益」里不含
    # 「换了一种调度机制」这部分。
    var ts = List[Int]()
    ts.append(1)
    ts.append(2)
    ts.append(4)
    ts.append(6)
    ts.append(8)

    print("=== 聚合读带宽 vs 并发任务数（256 MB fp32，交错，"
        + String(ROUNDS) + " 轮）===")
    print("parallelism_level =", parallelism_level(), " nproc = 8")

    # 每个 T 保留 5 轮的极值，最后报区间而不是报最好的一次。
    var lo = List[Float64]()
    var hi = List[Float64]()
    for _ in range(len(ts)):
        lo.append(Float64(0))
        hi.append(Float64(0))

    for r in range(ROUNDS):
        print("-- 第", r + 1, "轮 --")
        for ti in range(len(ts)):
            var t = ts[ti]
            var tg = TaskGroup()
            var per = ELEMS // t
            var t0 = monotonic_ns()
            for k in range(t):
                tg.create_task(shard(p, k * per, per, dst, k))
            tg.wait()
            var t1 = monotonic_ns()
            var bw = gb_per_s(ELEMS * 4, t1 - t0)
            if r == 0 or bw < lo[ti]:
                lo[ti] = bw
            if r == 0 or bw > hi[ti]:
                hi[ti] = bw
            print("  T =", t, " ", bw, " GB/s")

    print("=== 区间（min…max，GB/s）与相对 T=1 的加速 ===")
    var base_hi = hi[0]
    var base_lo = lo[0]
    for ti in range(len(ts)):
        var t = ts[ti]
        var a = lo[ti]
        var b = hi[ti]
        # 加速比用「自己的 min ÷ 基线的 max」算保守下界、「自己的 max ÷
        # 基线的 min」算保守上界 —— 两个区间若不重合，才谈得上有差别。
        var slow = a / base_hi
        var fast = b / base_lo
        print("  T =", t, " ", a, "…", b, " GB/s   加速 ", slow, "…", fast, "×")

    print("=== 一次「建任务组 + 等」的开销 ===")
    for ti in range(len(ts)):
        var t = ts[ti]
        var ns = bench_barrier(t)
        print("  T =", t, " ", ns, " ns/次  →",
            Float64(MATMULS_PER_TOKEN) * ns / 1_000_000.0,
            " ms/token（若每个矩阵乘都起一组）")

    arena.keep_alive()
