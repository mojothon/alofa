# 独立复现器（账本三十九）：**flare `Mutex` 在双线程并发下腐坏自己的堆存储**。
#
# 最终触发条件（逐刀切出来的，每刀 3-6 轮交替）：
#   1. 两条线程对同一个 flare `Mutex` **只做 lock/unlock**（REPRO_MUTEX_ONLY=1，
#      不碰任何 List、零用户分配）—— 8-10 秒一轮：rc=22 (EINVAL，flare 自己的
#      assert 在 mutex.mojo:114 打出)/SIGILL/挂死 三种收场都见过（非确定，
#      2026-11-06 四轮：1 崩 / 1 挂 / 2 存活）；
#   2. 或者：一条外来线程分配/丢弃小字符串（burst）× 主线程与之双向搬运
#      大内联元素邮箱（Job 8 KB / Result 16 KB，REPRO_BURST 默认 32）
#      —— 同样的堆腐坏，崩点散落在 tcmalloc sized-delete 的 CHECK 格式化
#      路径（libAsyncRTRuntimeGlobals.so +0x94433 = TCMallocInternalDeleteArraySizedAligned）。
#   排除项（都有对照数据）：
#   - 单线程同序列自环 1736 万次 ops 存活 ⇒ `List` 大元素 push/pop 无辜；
#   - 换 raw pthread 互斥体（REPRO_RAW_MUTEX=1 / REPRO_MUTEX_ONLY+raw）：
#     隔离 4/4 存活、全邮箱 4/4 存活 ⇒ 腐坏源是 flare Mutex 本身，不是
#     mailbox/分配模式；
#   - 堆坏后两个活块重叠：String 的缓冲被 malloc 到互斥体存储上
#     （互斥体字节对账器抓到的现行）。
#   服务器侧由此根因修复：邮箱换 RawMutex（inline 48B pthread 互斥体）。
#
# 用法：
#   pixi run mojo build -O2 -I src scripts/diag_repro_mailbox_burst.mojo -o target/repro_burst
#   REPRO_SECONDS=15 REPRO_BURST=32 ./target/repro_burst          # 全邮箱（默认崩）
#   REPRO_SECONDS=10 REPRO_MUTEX_ONLY=1 REPRO_BURST=0 ./target/repro_burst   # 纯 Mutex 隔离
#   REPRO_RAW_MUTEX=1 REPRO_SECONDS=15 ./target/repro_burst       # raw pthread 对照（应存活）
#   REPRO_SINGLE=1 REPRO_SECONDS=15 ./target/repro_burst          # 单线程自环（应存活）
# 判据：进程跑满秒数并打印 survived = 未复现；SIGSEGV/SIGILL = 复现。

from std.collections import List
from std.ffi import external_call
from std.memory import Pointer
from std.time import sleep

from alofa.core.ffi.posix import monotonic_ms
from alofa.srv.config import env_int, env_str

comptime OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]

from flare.runtime._thread import ThreadHandle
from flare.runtime.mutex import Mutex

alias MAX_JOB_BYTES = 8192
alias MAX_RES_BYTES = 16384
alias MAILBOX_CAP = 64
alias MAGIC_JOB = 0xA11FA11F
alias MAGIC_RES = 0xB22FB22F


struct Job(Movable):
    var kind: Int
    var slot: Int
    var gen: Int
    var n_raw: Int
    var magic: Int
    var raw: InlineArray[UInt8, MAX_JOB_BYTES]

    def __init__(out self):
        self.kind = 0
        self.slot = 0
        self.gen = 0
        self.n_raw = 0
        self.magic = MAGIC_JOB
        self.raw = InlineArray[UInt8, MAX_JOB_BYTES](fill=0)
        self.raw[0] = 0


struct Result(Movable):
    var slot: Int
    var gen: Int
    var kind: Int
    var done: Bool
    var magic: Int
    var raw: InlineArray[UInt8, MAX_RES_BYTES]

    def __init__(out self):
        self.slot = 0
        self.gen = 0
        self.kind = 0
        self.done = False
        self.magic = MAGIC_RES
        self.raw = InlineArray[UInt8, MAX_RES_BYTES](fill=0)
        self.raw[0] = 0


struct Mailbox(Movable):
    var mutex: Mutex
    # raw pthread 互斥体（48 字节）：REPRO_RAW_MUTEX=1 时全方法走它（对照刀）。
    var raw: InlineArray[UInt8, 48]
    var use_raw: Int
    var jobs: List[Job]
    var results: List[Result]

    def __init__(out self) raises:
        self.mutex = Mutex()
        self.raw = InlineArray[UInt8, 48](fill=0)
        self.use_raw = 0
        self.jobs = List[Job]()
        self.results = List[Result]()
        self.jobs.reserve(MAILBOX_CAP)
        self.results.reserve(MAILBOX_CAP)

    def rlock(self) -> Int:
        return Int(external_call[
            "pthread_mutex_lock", Int32, Pointer[UInt8, MutUntrackedOrigin]
        ](Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=self.raw)))))

    def runlock(self) -> Int:
        return Int(external_call[
            "pthread_mutex_unlock", Int32, Pointer[UInt8, MutUntrackedOrigin]
        ](Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=self.raw)))))

    def push_job(mut self, var job: Job) raises -> Bool:
        if self.use_raw == 1:
            _ = self.rlock()
            if len(self.jobs) >= MAILBOX_CAP:
                _ = self.runlock()
                return False
            self.jobs.append(job^)
            _ = self.runlock()
            return True
        self.mutex.lock()
        if len(self.jobs) >= MAILBOX_CAP:
            self.mutex.unlock()
            return False
        self.jobs.append(job^)
        self.mutex.unlock()
        return True

    def take_job(mut self, mut job: Job) raises -> Bool:
        if self.use_raw == 1:
            _ = self.rlock()
            if len(self.jobs) == 0:
                _ = self.runlock()
                return False
            job = self.jobs.pop(0)
            _ = self.runlock()
            return True
        self.mutex.lock()
        if len(self.jobs) == 0:
            self.mutex.unlock()
            return False
        job = self.jobs.pop(0)
        self.mutex.unlock()
        return True

    def push_result(mut self, var res: Result) raises -> Bool:
        if self.use_raw == 1:
            _ = self.rlock()
            if len(self.results) >= MAILBOX_CAP:
                _ = self.runlock()
                return False
            self.results.append(res^)
            _ = self.runlock()
            return True
        self.mutex.lock()
        if len(self.results) >= MAILBOX_CAP:
            self.mutex.unlock()
            return False
        self.results.append(res^)
        self.mutex.unlock()
        return True

    def take_result(mut self, mut res: Result) raises -> Bool:
        if self.use_raw == 1:
            _ = self.rlock()
            if len(self.results) == 0:
                _ = self.runlock()
                return False
            res = self.results.pop(0)
            _ = self.runlock()
            return True
        self.mutex.lock()
        if len(self.results) == 0:
            self.mutex.unlock()
            return False
        res = self.results.pop(0)
        self.mutex.unlock()
        return True


struct Box(Movable):
    var mb: Mailbox
    var burst: Int
    var stop: Int
    # raw pthread 互斥体（48 字节，零初始化）—— 绕开 flare Mutex 的对照刀。
    var raw_mtx: InlineArray[UInt8, 48]

    def __init__(out self, burst: Int) raises:
        self.mb = Mailbox()
        self.burst = burst
        self.stop = 0
        self.raw_mtx = InlineArray[UInt8, 48](fill=0)


def engine_entry(arg: OpaquePtr) -> OpaquePtr:
    # 外来线程侧：取活 → burst（分配/丢弃小字符串）→ 交结果。
    # 不许让异常跑出去（pthread 没有异常通道，见服务器同款注释）。
    var box = arg.unsafe_bitcast[Box]()
    # 变体：一次性抬高 Mailbox 指针，不经 `box[].mb` 代理链逐次调用。
    var mbp = Pointer(to=box[].mb)
    var done_jobs = 0
    if box[].burst == 0 and env_str("REPRO_MUTEX_ONLY", "0") == "1":
        # 纯 Mutex 隔离：两线程只 lock/unlock **同一个**互斥体，不碰任何 List。
        # 默认打 flare `Mutex`；REPRO_RAW_MUTEX=1 换 raw pthread（对照）。
        var n = 0
        var mptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=box[].raw_mtx)))
        var use_raw = env_str("REPRO_RAW_MUTEX", "0") == "1"
        try:
            while box[].stop == 0:
                if use_raw:
                    _ = external_call[
                        "pthread_mutex_lock", Int32, Pointer[UInt8, MutUntrackedOrigin]
                    ](mptr)
                    _ = external_call[
                        "pthread_mutex_unlock", Int32, Pointer[UInt8, MutUntrackedOrigin]
                    ](mptr)
                else:
                    mbp[].mutex.lock()
                    mbp[].mutex.unlock()
                n += 1
        except:
            pass
        print(
            "[repro] mutex-only engine side: raw="
            + String(Int(use_raw))
            + " locks="
            + String(n)
        )
        var zero1 = Int(0)
        return OpaquePtr(unsafe_from_address=zero1)
    try:
        while box[].stop == 0:
            var job = Job()
            var res = Result()
            if mbp[].take_job(job):
                # burst：与服务器 FAKE 路同形状的小字符串分配/丢弃。
                var k = 0
                while k < box[].burst:
                    var s = (
                        '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":'
                        + String(1234567890)
                        + ',"choices":[{"index":0,"delta":{"content":"x"}}]}'
                    )
                    _ = s.byte_length()
                    k += 1
                res.slot = job.slot
                res.gen = job.gen
                _ = mbp[].push_result(res^)
                done_jobs += 1
            # 空转：满速取（服务器里 reactor 的 epoll 间隙也是这个形状）。
            pass
    except:
        pass
    print("[repro] engine side done: jobs=" + String(done_jobs))
    var zero = Int(0)
    return OpaquePtr(unsafe_from_address=zero)


def main() raises:
    # 参数走环境变量（编译后 sys.argv 是空的，见项目惯例）。
    var seconds = env_int("REPRO_SECONDS", 10, 1, 600)
    var burst = env_int("REPRO_BURST", 32, 0, 4096)
    print(
        "[repro] mailbox-burst reproducer: seconds="
        + String(seconds)
        + " burst="
        + String(burst)
    )

    if env_str("REPRO_SINGLE", "0") == "1":
        # 终极判定：**单线程**同序列自环（push→take→push_result→take_result），
        # 满速。崩 ⇒ List[8KB元素] 的 push/pop 自身有 bug，与线程无关。
        var box1 = Box(0)
        var j1 = Job()
        var r1 = Result()
        var ops = 0
        var deadline1 = monotonic_ms() + seconds * 1000
        while monotonic_ms() < deadline1:
            j1 = Job()
            j1.slot = ops % 8
            j1.gen = ops
            j1.n_raw = 64
            var i1 = 0
            while i1 < 64:
                j1.raw[i1] = UInt8((i1 + ops) % 251)
                i1 += 1
            if box1.mb.push_job(j1^):
                var j2 = Job()
                if box1.mb.take_job(j2):
                    var r2 = Result()
                    r2.slot = j2.slot
                    if box1.mb.push_result(r2^):
                        var r3 = Result()
                        if box1.mb.take_result(r3):
                            ops += 1
        print("[repro] single-thread survived: ops=" + String(ops))
        return

    var box = Box(burst)
    if env_str("REPRO_RAW_MUTEX", "0") == "1":
        box.mb.use_raw = 1
        print("[repro] using raw pthread mutex")
    _ = external_call[
        "pthread_mutex_init",
        Int32,
        Pointer[UInt8, MutUntrackedOrigin],
        Pointer[UInt8, MutUntrackedOrigin],
    ](
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(Pointer(to=box.raw_mtx))
        ),
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(0)),
    )

    # 互斥体自监控（账本三十九）：flare 的 Mutex._storage 是 unsafe_alloc 堆块。
    # pthread 结构（40 字节）在**没被加锁时**绝不许变；变 = 有谁写穿了它的堆邻居。
    # 快照 + 每轮对账，抓到就打印偏移与前后字节。
    var ms = box.mb.mutex._storage
    print("[repro] mutex storage at 0x" + hex(Int(ms)))
    # 只对账 16-39：0-3 是 futex 字（lock 本来就写），8-11 owner —— 16 起
    # 是 __kind 与链表指针，活着的普通互斥体绝不该变。
    var snap = List[UInt8](capacity=64)
    var k0 = 16
    while k0 < 40:
        snap.append(ms[k0])
        k0 += 1

    var t = ThreadHandle.spawn[engine_entry](
        OpaquePtr(unsafe_from_address=Int(Pointer(to=box)))
    )

    var sent = 0
    var got = 0
    var deadline = monotonic_ms() + seconds * 1000
    while monotonic_ms() < deadline:
        var job = Job()
        var res = Result()
        # 主线程侧：填一条像样的请求行，推过去，等结果。
        job.slot = sent % 8
        job.gen = sent
        # 主线程这一侧**零分配**：raw 填确定性图案（无 String/临时对象）。
        job.n_raw = 64
        var i = 0
        while i < 64:
            job.raw[i] = UInt8((i + sent) % 251)
            i += 1
        # 对账互斥体结构：任何字节变了 = 腐坏现行。
        var k1 = 16
        while k1 < 40:
            if ms[k1] != snap[k1 - 16]:
                print(
                    "[repro] MUTEX SMASHED at offset "
                    + String(k1)
                    + ": was "
                    + String(Int(snap[k1 - 16]))
                    + " now "
                    + String(Int(ms[k1]))
                )
                var k2 = 0
                while k2 < 48:
                    print(
                        "  [" + String(k2) + "] "
                        + String(Int(snap[k2]))
                        + " -> "
                        + String(Int(ms[k2]))
                    )
                    k2 += 1
                return
            k1 += 1
        if env_str("REPRO_MUTEX_ONLY", "0") == "1":
            if env_str("REPRO_RAW_MUTEX", "0") == "1":
                _ = external_call[
                    "pthread_mutex_lock", Int32, Pointer[UInt8, MutUntrackedOrigin]
                ](Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=box.raw_mtx))))
                _ = external_call[
                    "pthread_mutex_unlock", Int32, Pointer[UInt8, MutUntrackedOrigin]
                ](Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=box.raw_mtx))))
            else:
                box.mb.mutex.lock()
                box.mb.mutex.unlock()
        if box.mb.push_job(job^):
            sent += 1
        if box.mb.take_result(res):
            got += 1

    box.stop = 1
    t.join()
    print(
        "[repro] survived: sent="
        + String(sent)
        + " got="
        + String(got)
        + " (未复现)"
    )
