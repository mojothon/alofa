"""一次性最小复现：一个 pthread Mutex + 一个 List + 两个 Int 计数。

形状与 `srv/engine_thread.mojo` 的 `Mailbox` 完全一样：
- 生产者线程：`lock` → `jobs.append(v)` → `pushed += 1` → `unlock`
- 消费者线程：`lock` → 若非空则 `jobs.pop(0)` → `taken += 1` → `unlock`

服务里观测到的矛盾是 `results_pushed > jobs_taken`（每个活只回一截结果，所以这两个
数必须相等）。这把锁是 `pthread_mutex`，按说 RMW 不会丢 —— 所以要么"锁没锁住"，
要么"`List` 在锁内也会丢元素"。这个文件用 `pixi run mojo run` 就能跑，不需要重编
整个服务（那条要十几分钟）。

用法: pixi run mojo run -I src scripts/tmp_mailbox_race.mojo [每边条数]
"""

from std.collections import List
from std.ffi import external_call
from std.memory import Pointer

from flare.runtime._thread import ThreadHandle
from flare.runtime.mutex import Mutex

comptime N_ITEMS = 20000
comptime OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]


struct Box(Movable):
    var mutex: Mutex
    var items: List[Int]
    var pushed: Int
    var taken: Int
    var sum: Int

    def __init__(out self) raises:
        self.mutex = Mutex()
        self.items = List[Int]()
        self.pushed = 0
        self.taken = 0
        self.sum = 0


def _producer(arg: OpaquePtr) -> OpaquePtr:
    var box = arg.unsafe_bitcast[Box]()
    var i = 0
    while i < N_ITEMS:
        box[].mutex.lock()
        box[].items.append(i)
        box[].pushed += 1
        box[].mutex.unlock()
        i += 1
    var zero = 0
    return OpaquePtr(unsafe_from_address=zero)


def _consumer(arg: OpaquePtr) -> OpaquePtr:
    var box = arg.unsafe_bitcast[Box]()
    var got = 0
    var idle = 0
    while got < N_ITEMS:
        var took = False
        var v = -1
        box[].mutex.lock()
        if len(box[].items) > 0:
            v = box[].items.pop(0)
            box[].taken += 1
            took = True
        box[].mutex.unlock()
        if took:
            got += 1
            box[].sum += v
            idle = 0
        else:
            idle += 1
            if idle > 1000000:
                # 生产者已经停下了（或条数不够）：别死转。
                break
            _ = external_call["usleep", Int32](Int32(50))
    var zero = 0
    return OpaquePtr(unsafe_from_address=zero)



def main() raises:
    var box = Box()
    var addr = Int(Pointer(to=box))
    var ph = ThreadHandle.spawn[_producer](OpaquePtr(unsafe_from_address=addr))
    var ch = ThreadHandle.spawn[_consumer](OpaquePtr(unsafe_from_address=addr))
    var a = ph.join()
    var b = ch.join()
    var want = (N_ITEMS - 1) * N_ITEMS // 2
    print(
        "pushed="
        + String(box.pushed)
        + " taken="
        + String(box.taken)
        + " left="
        + String(len(box.items))
        + " sum="
        + String(box.sum)
        + " want_sum="
        + String(want)
    )
    if box.pushed != N_ITEMS or box.taken != N_ITEMS or box.sum != want:
        print("RACE: 计数或元素丢了")
    else:
        print("干净：这一层没有丢更新")
    _ = a
    _ = b
