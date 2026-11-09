"""量 TaskGroup 的派发/join 往返成本（四十五：批解码 47.5% 空闲的嫌疑）。

四组各 N 轮（任务不共享状态，只计时间）：
  A 空 TaskGroup：create 1 个空任务再 wait —— 纯派发开销
  B 8 个微任务 —— 模拟"单 token 线性层的片"的形状（工作量≈0）
  C 8 个真活任务（每片 10 万次 FMA）—— 有实质工作时派发开销占多少
  C' 同样的活串行做 —— 对照
"""
from std.runtime.asyncrt import TaskGroup
from alofa.core.ffi.posix import monotonic_ns


async def _empty():
    pass


async def _tiny(k: Int):
    var x = k * 2
    _ = x


async def _work(k: Int, reps: Int):
    var x = Float32(k) + 1.0
    for _ in range(reps):
        x = x * 1.0000001 + 0.0000001
    _ = x


def _work_sync(k: Int, reps: Int):
    var x = Float32(k) + 1.0
    for _ in range(reps):
        x = x * 1.0000001 + 0.0000001
    _ = x


def main():
    var rounds = 0
    var t0 = 0

    # A：1 个空任务
    rounds = 2000
    for _ in range(3):
        var tg = TaskGroup()
        tg.create_task(_empty())
        tg.wait()
    t0 = monotonic_ns()
    for _ in range(rounds):
        var tg = TaskGroup()
        tg.create_task(_empty())
        tg.wait()
    print("A 空TG(1任务)   ", (monotonic_ns() - t0) // rounds, "ns/轮")

    # B：8 个微任务
    rounds = 2000
    for _ in range(3):
        var tg = TaskGroup()
        for k in range(8):
            tg.create_task(_tiny(k))
        tg.wait()
    t0 = monotonic_ns()
    for _ in range(rounds):
        var tg = TaskGroup()
        for k in range(8):
            tg.create_task(_tiny(k))
        tg.wait()
    print("B 空TG(8微任务) ", (monotonic_ns() - t0) // rounds, "ns/轮")

    # C：8 个真活任务 vs 串行
    rounds = 300
    for _ in range(3):
        var tg = TaskGroup()
        for k in range(8):
            tg.create_task(_work(k, 100_000))
        tg.wait()
    t0 = monotonic_ns()
    for _ in range(rounds):
        var tg = TaskGroup()
        for k in range(8):
            tg.create_task(_work(k, 100_000))
        tg.wait()
    print("C 真活TG(8x100k)", (monotonic_ns() - t0) // rounds, "ns/轮")
    t0 = monotonic_ns()
    for _ in range(rounds):
        for k in range(8):
            _work_sync(k, 100_000)
    print("C 真活 串行     ", (monotonic_ns() - t0) // rounds, "ns/轮")
