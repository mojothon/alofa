"""TaskGroup 真实并行度定标：1/2/4/8 个等量任务的墙钟。

四十八：8 个等量纯计算任务的 TG 墙钟约为单任务 2–4 倍（探针记录的"错峰起跑"
模式），而同窗口 8 条 OS 线程自旋能拿 756% —— 操作系统给核没问题。本基准去掉
探针与内存，只量墙钟：

    并行度 k  ⇒  8 任务墙钟 ≈ ceil(8/k) × 单任务墙钟

1 任务当基尺，各宽度取多轮最小值（本机长期过载，min 最接近真实能力）。
"""
from alofa.core.ffi.posix import monotonic_ns
from std.runtime.asyncrt import TaskGroup, parallelism_level

alias WORK = 3_000_000
alias ROUNDS = 12


def _spin(reps: Int) -> Float64:
    var x = Float64(1.0000001)
    var acc = Float64(0.0)
    for _ in range(reps):
        x = x * 1.0000001 + 0.0000001
        acc += x
    return acc


async def _task(reps: Int):
    var r = _spin(reps)
    if r == 0.0:
        print("unreachable")


def _wall(n: Int) -> Float64:
    var best = 1.0e18
    for _ in range(ROUNDS):
        var tg = TaskGroup()
        var t0 = monotonic_ns()
        for _ in range(n):
            tg.create_task(_task(WORK))
        tg.wait()
        var us = Float64(monotonic_ns() - t0) / 1e3
        if us < best:
            best = us
    return best


def main():
    print("parallelism_level=", String(parallelism_level()))
    var w1 = _wall(1)
    var w2 = _wall(2)
    var w4 = _wall(4)
    var w8 = _wall(8)
    var w16 = _wall(16)
    print("单任务墙钟 ", String(Int(w1)), " (0.1us)")
    print("n=2  ", String(Int(w2)), "  倍数 ", String(Int(w2 / w1 * 100)), "%")
    print("n=4  ", String(Int(w4)), "  倍数 ", String(Int(w4 / w1 * 100)), "%")
    print("n=8  ", String(Int(w8)), "  倍数 ", String(Int(w8 / w1 * 100)), "%")
    print("n=16 ", String(Int(w16)), "  倍数 ", String(Int(w16 / w1 * 100)), "%")
    print("反推并行度：", String(Int(Float64(8) * w1 / w8 * 100)), "%")
