"""TaskGroup 派发成本：主线程 vs 裸 pthread 线程（flare/foreign 线程）。

四十五的疑点：engine 线程是 pthread（flare 起），前向里的 TaskGroup 是
AsyncRT 的。若"外来线程 → AsyncRT 派发"有 ~0.5 ms 级的跨运行时跳跃，
24 层 × ~5 次派发的链就能吃掉批解码一半的墙钟 —— 与实测的 47.5% 空闲对上。
主线程上同样的派发只要 ~2-6 µs，所以要分臂对比。
"""
from std.runtime.asyncrt import TaskGroup
from alofa.core.ffi.posix import monotonic_ns
from std.ffi import external_call
from std.memory import Pointer

comptime OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]


struct Result(Movable):
    var a_ns: Int
    var b_ns: Int
    var c_ns: Int


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


def _bench_a(rounds: Int) -> Int:
    for _ in range(5):
        var tg = TaskGroup()
        tg.create_task(_empty())
        tg.wait()
    var t0 = monotonic_ns()
    for _ in range(rounds):
        var tg = TaskGroup()
        tg.create_task(_empty())
        tg.wait()
    return (monotonic_ns() - t0) // rounds


def _bench_b(rounds: Int) -> Int:
    for _ in range(5):
        var tg = TaskGroup()
        for k in range(8):
            tg.create_task(_tiny(k))
        tg.wait()
    var t0 = monotonic_ns()
    for _ in range(rounds):
        var tg = TaskGroup()
        for k in range(8):
            tg.create_task(_tiny(k))
        tg.wait()
    return (monotonic_ns() - t0) // rounds


def _bench_c(rounds: Int) -> Int:
    for _ in range(3):
        var tg = TaskGroup()
        for k in range(8):
            tg.create_task(_work(k, 100_000))
        tg.wait()
    var t0 = monotonic_ns()
    for _ in range(rounds):
        var tg = TaskGroup()
        for k in range(8):
            tg.create_task(_work(k, 100_000))
        tg.wait()
    return (monotonic_ns() - t0) // rounds


def opaque_of(imm addr: Int) -> OpaquePtr:
    return OpaquePtr(unsafe_from_address=addr)


def _thread_main(arg: OpaquePtr) -> OpaquePtr:
    var res = Pointer[Result, MutUntrackedOrigin](unsafe_from_address=Int(arg))
    res[0].a_ns = _bench_a(300)
    res[0].b_ns = _bench_b(300)
    res[0].c_ns = _bench_c(100)
    var zero = 0
    return OpaquePtr(unsafe_from_address=zero)


def main():
    var raw = external_call["malloc", Int](Int(64))
    var res = Pointer[Result, MutUntrackedOrigin](unsafe_from_address=raw)
    res[0].a_ns = 0
    res[0].b_ns = 0
    res[0].c_ns = 0
    print("== 主线程 ==")
    print("A 空TG(1任务)   ", _bench_a(1000), "ns/轮")
    print("B 空TG(8微任务) ", _bench_b(1000), "ns/轮")
    print("C 真活TG(8x100k)", _bench_c(200), "ns/轮")

    var tid = UInt64(0)
    var tid_ptr = Pointer[UInt64, _](to=tid)
    var rc = external_call["pthread_create", Int32](
        tid_ptr, OpaquePtr(unsafe_from_address=Int(0)), _thread_main, opaque_of(raw)
    )
    if rc != 0:
        print("pthread_create 失败 rc=", rc)
        return
    _ = external_call["pthread_join", Int32](tid, OpaquePtr(unsafe_from_address=Int(0)))
    print("== 裸 pthread 线程 ==")
    print("A 空TG(1任务)   ", res[0].a_ns, "ns/轮")
    print("B 空TG(8微任务) ", res[0].b_ns, "ns/轮")
    print("C 真活TG(8x100k)", res[0].c_ns, "ns/轮")
