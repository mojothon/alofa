"""TaskGroup 长任务：把 8×300ms 的 TG 连续跑 6 秒，供外部采样器看线程账。

四十八续：`n=8` 墙钟是单任务的 201%（≈4 并发）。本程序只负责"持续制造
TaskGroup 负载"，逐线程 CPU 由外部 Python 采样 `/proc/<pid>/task/*/stat`
（线程名/占用见 `/tmp/tg_threads_probe.py`）。
"""
from alofa.core.ffi.posix import monotonic_ns
from std.runtime.asyncrt import TaskGroup, parallelism_level

alias WORK = 90_000_000
alias ROUNDS = 20


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


def main():
    print("parallelism_level=", String(parallelism_level()))
    print("pid=", String(Int(external_pid())))
    var best = 1.0e18
    var t_start = monotonic_ns()
    for _ in range(ROUNDS):
        var tg = TaskGroup()
        var t0 = monotonic_ns()
        for _ in range(8):
            tg.create_task(_task(WORK))
        tg.wait()
        var ms = (monotonic_ns() - t0) // 1_000_000
        print("轮 ", String(Int(ms)), " ms")
        if Float64(ms) < best:
            best = Float64(ms)
    var total = (monotonic_ns() - t_start) // 1_000_000
    print("最短轮 ", String(Int(best)), " ms ；总 ", String(Int(total)), " ms")


def external_pid() -> Int:
    """getpid()：采样器要知道我们是谁。"""
    from std.ffi import external_call

    return Int(external_call["getpid", Int32]())
