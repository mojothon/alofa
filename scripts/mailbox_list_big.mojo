"""一次性最小复现（第二版）：把 `List[Int]` 换成"和 `Job` 形状一样"的大结构体。

第一版（`List[Int]`，2 万条）是干净的。服务里对不上的是**件数**：
`runs=57` 却 `jpush=56` —— engine 跑了 57 件，而循环只派出 56 件，且邮箱最后是空的。
"多出来一件"只能是 `List` 自己给出的：要么 `pop` 把同一个元素给了两次，要么
`append` 一次进了两个。

这里复刻 `Job` 的字段形状（含 `InlineArray[UInt8, 4096]`），生产/消费各 2 万次，
消费侧校验：
- 每个元素恰好被取出一次（用一张"见过"位图，位图本身只由消费者线程写）；
- 元素内容完整（`magic`/`slot` 都对）。

用法: pixi run mojo run -O2 -I src scripts/tmp_list_big.mojo
"""

from std.collections import List
from std.ffi import external_call
from std.memory import Pointer

from flare.runtime._thread import ThreadHandle
from flare.runtime.mutex import Mutex

comptime N_ITEMS = 20000
comptime PAYLOAD = 4096
comptime OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime MAGIC = 0x4A4F4253


struct Big(Movable):
    var magic: Int
    var slot: Int
    var gen: Int
    var kind: Int
    var thread: Int
    var raw: InlineArray[UInt8, PAYLOAD]
    var n_raw: Int

    def __init__(out self, slot: Int):
        self.magic = MAGIC
        self.slot = slot
        self.gen = 7
        self.kind = 1
        self.thread = 0
        self.raw = InlineArray[UInt8, PAYLOAD](fill=0)
        self.raw[0] = UInt8(slot % 256)
        self.raw[PAYLOAD - 1] = UInt8(slot % 251)
        self.n_raw = PAYLOAD


struct Box(Movable):
    var mutex: Mutex
    var items: List[Big]
    var pushed: Int
    var taken: Int
    var bad: Int
    var seen: List[UInt8]

    def __init__(out self) raises:
        self.mutex = Mutex()
        self.items = List[Big]()
        self.pushed = 0
        self.taken = 0
        self.bad = 0
        self.seen = List[UInt8]()
        var i = 0
        while i < N_ITEMS:
            self.seen.append(0)
            i += 1


def _producer(arg: OpaquePtr) -> OpaquePtr:
    var box = arg.unsafe_bitcast[Box]()
    var i = 0
    while i < N_ITEMS:
        box[].mutex.lock()
        box[].items.append(Big(i))
        box[].pushed += 1
        box[].mutex.unlock()
        i += 1
    var zero = 0
    return OpaquePtr(unsafe_from_address=zero)


def _consumer(arg: OpaquePtr) -> OpaquePtr:
    var box = arg.unsafe_bitcast[Box]()
    var idle = 0
    while box[].taken < N_ITEMS:
        var took = False
        var b = Big(0)
        box[].mutex.lock()
        if len(box[].items) > 0:
            b = box[].items.pop(0)
            box[].taken += 1
            took = True
        box[].mutex.unlock()
        if not took:
            idle += 1
            if idle > 2000000:
                break
            _ = external_call["usleep", Int32](Int32(50))
            continue
        idle = 0
        # 内容与"只取一次"都校验（`seen` 只有这一条线程写，不用锁）。
        if (
            b.magic != MAGIC
            or b.n_raw != PAYLOAD
            or Int(b.raw[0]) != b.slot % 256
            or Int(b.raw[PAYLOAD - 1]) != b.slot % 251
        ):
            box[].bad += 1
        elif b.slot >= 0 and b.slot < N_ITEMS:
            if box[].seen[b.slot] == 1:
                box[].bad += 1
            box[].seen[b.slot] = 1
    var zero = 0
    return OpaquePtr(unsafe_from_address=zero)


def main() raises:
    var box = Box()
    var addr = Int(Pointer(to=box))
    var ph = ThreadHandle.spawn[_producer](OpaquePtr(unsafe_from_address=addr))
    var ch = ThreadHandle.spawn[_consumer](OpaquePtr(unsafe_from_address=addr))
    var a = ph.join()
    var b = ch.join()
    var distinct = 0
    for v in box.seen:
        distinct += Int(v)
    print(
        "pushed="
        + String(box.pushed)
        + " taken="
        + String(box.taken)
        + " left="
        + String(len(box.items))
        + " distinct="
        + String(distinct)
        + " bad="
        + String(box.bad)
    )
    if box.pushed != N_ITEMS or box.taken != N_ITEMS or distinct != N_ITEMS or box.bad > 0:
        print("RACE: 大结构体在锁内丢了/重了元素")
    else:
        print("干净：大结构体也没有问题")
    _ = a
    _ = b
