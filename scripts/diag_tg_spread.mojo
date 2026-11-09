"""对照：一个 TG 里 8 个**等量纯计算**任务的耗时分布。

四十八的原始记录里，同一 TG 的 8 个分片做着**行数完全相同**的活，耗时却差到
4×（典型：一个 799 µs、其余 195 µs）。两种可能：

    a. 调度/抢核：线程被别的负载挤下 CPU ⇒ 等量纯计算也会差 4×；
    b. 内存运气：权重流/缓存冷热在我方核里就能造成 4×。

本对照去掉内存因子：8 个任务各做固定迭代的 FMA 链（寄存器内，不碰大内存），
用同一套探针戳时。若分布仍然差几倍 ⇒ 是 (a)，四十六的 ~48% 空闲里有一部分
其实是"线程被挤"，不是我们自己的屏障。
"""
from alofa.core.ffi.mem import RawPtr
from alofa.core.ffi.posix import monotonic_ns
from alofa.model.probe_barrier import (
    PROBE_KIND_ROWS,
    probe_alloc,
    probe_dispatch,
    probe_dump,
    probe_mark_end,
    probe_mark_start,
    probe_reset,
    probe_wait_done,
    probe_trace,
)
from std.runtime.asyncrt import TaskGroup, parallelism_level

alias ROUNDS = 200


def _spin(reps: Int) -> Float64:
    var x = Float64(1.0000001)
    var acc = Float64(0.0)
    for i in range(reps):
        x = x * 1.0000001 + 0.0000001
        acc += x
    return acc


async def _equal_task(probe: RawPtr, rec: Int, k: Int, reps: Int):
    probe_mark_start(probe, rec, k)
    var r = _spin(reps)
    if r == 0.0:  # 防优化
        print("unreachable")
    probe_mark_end(probe, rec, k)


def main() raises:
    print("parallelism_level=", String(parallelism_level()))
    var probe = probe_alloc()
    # 热身一轮（让 JIT/页表/频率稳定）
    var warm = TaskGroup()
    for k in range(8):
        warm.create_task(_equal_task(probe, -1, k, 100_000))
    warm.wait()

    for _ in range(ROUNDS):
        var tg = TaskGroup()
        var rec = probe_dispatch(probe, PROBE_KIND_ROWS, 8)
        for k in range(8):
            tg.create_task(_equal_task(probe, rec, k, 2_000_000))
        tg.wait()
        probe_wait_done(probe, rec)
    probe_dump(probe)
    probe_trace(probe, 0, 4)
