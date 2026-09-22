"""Capability gate: flare 的事件循环原语（reactor / 非阻塞 accept / 跨线程唤醒）。

`srv/loop.mojo`（roadmap 3.2）不自己写 epoll：epoll 由 `flare.runtime.Reactor`
提供，线程由 flare 的 pthread 出口提供（Mojo 1.0 没有语言级 async，也没有
stdlib 网络栈）。这个门回答的是"这两块在本仓库真的能用"，并且把三条**容易假绿**
的性质钉住 —— 它们都属于"看起来能跑、其实只是碰巧没出错"那一类：

1. **EAGAIN 必须报成错误**：非阻塞 accept 取空之后再取一次必须报错，而不是阻塞
   —— reactor 一个可读事件"accept 到空"的终止条件就是它。
2. **poll 的超时必须真的返回**：空闲超时与调度 tick 全靠它。它不返回的话，循环
   就退化成"等到有事件才动"，慢客户端永远不会超时。
3. **跨线程 wakeup 必须能在超时之前把 poll 叫醒**：engine 线程交回一个 token
   走的就是这条路；不通的话"非阻塞"退化成定时轮询，延迟等于 poll 超时。

这三条都是**前提**，不是"服务层能用"的证据 —— 后者由 `pixi run test-loop`
（真 socket 的端到端门）给。这个门只回答：底下这块地基在不在。

Run:
    pixi run mojo run -I src tests/capability/test_reactor.mojo
"""

from std.collections import List
from std.ffi import c_int, external_call
from std.memory import Pointer
from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.ffi import monotonic_ms
from alofa.core.ffi.posix import set_nonblocking
from flare.net import SocketAddr
from flare.runtime import INTEREST_READ, Event, Reactor
from flare.runtime._thread import ThreadHandle
from flare.runtime.reuseport import bind_reuseport
from flare.tcp import TcpStream, accept_fd

comptime PORT = UInt16(18371)
comptime LISTENER_TOKEN = UInt64(1)
# 连接用 100+i：与 `LISTENER_TOKEN` 分开一段，是为了让"拿到的 token 是不是监听
# 那个"在断言里一眼可读 —— reactor 只给数字，说不清的数字等于没有。
comptime CONN_TOKEN_BASE = UInt64(100)


def test_nonblocking_accept_reports_eagain() raises:
    """取空之后再 accept 必须报错，而不是阻塞在那里。

    这是 reactor 里 accept 循环的终止条件：**一个可读事件要 accept 到空**（内核
    可能把几条连接压成一个事件）。如果 EAGAIN 不报错而是等，循环就会停在第二次
    调用上 —— 症状是"服务 accept 完第一条之后不再接新的"，而不是崩溃。
    """
    # ⚠️ `listener` 必须活到这条门结束：Mojo 是**用完即析构**（ASAP），如果
    # `as_raw_fd()` 是它的最后一次使用，监听 socket 会在下一句之前就被关掉 ——
    # 症状是 `fcntl` 报 EBADF、客户端连接被拒，看起来像"flare 的 bind 坏了"。
    # 所以这里把 `listener.close()` 放在最后（那是它真正的最后一次使用），
    # `lfd` 才在整个循环期间有效。`srv/loop.mojo` 同理：监听 fd 的生命周期
    # 必须由循环自己持有，不能只留一个数字。
    var listener = bind_reuseport(SocketAddr.localhost(PORT))
    var lfd = listener.as_raw_fd()
    var rc = set_nonblocking(lfd)
    assert_true(rc == 0, "fcntl(O_NONBLOCK) failed: " + String(Int(rc)))

    var first = TcpStream.connect(SocketAddr.localhost(PORT))
    var second = TcpStream.connect(SocketAddr.localhost(PORT))

    var reactor = Reactor()
    reactor.register(lfd, LISTENER_TOKEN, INTEREST_READ)
    var events = List[Event]()
    var n = reactor.poll(300, events)
    assert_true(n >= 1, "the reactor did not report the listener readable")

    var got = 0
    var drained = False
    while got < 8:
        try:
            var conn = accept_fd(lfd)
            conn.close()
            got += 1
        except:
            # accept(2) 失败（EAGAIN）= 队列空了。这才是循环该停的地方。
            drained = True
            break
    assert_equal(got, 2, "the two pending connections should both be accepted")
    assert_true(drained, "accept must report EAGAIN once drained, not block")
    first.close()
    second.close()
    listener.close()


def test_poll_returns_on_timeout() raises:
    """带超时地等一个不会来的事件：必须**返回**，而不是一直等。

    空闲超时（fd 不泄漏）与调度 tick（给流发下一步）都挂在这个超时上。
    """
    var reactor = Reactor()
    var events = List[Event]()
    var began = monotonic_ms()
    var n = reactor.poll(80, events)
    var elapsed = monotonic_ms() - began
    assert_equal(n, 0, "nothing is registered, so nothing can be ready")
    assert_true(elapsed >= 60, "the poll returned early: " + String(elapsed) + "ms")
    assert_true(elapsed < 3000, "the poll overran its timeout: " + String(elapsed) + "ms")


comptime _OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]


struct WakerCtx(Copyable, Movable):
    """线程入口的实参。`pthread_create` 的启动函数是 C ABI 的，抓不到任何东西，
    所以跨线程只能传**一个地址**：reactor 的地址（叫醒它）与一个平凡计数器。"""

    var reactor_addr: Int
    var woke: Int

    def __init__(out self, reactor_addr: Int):
        self.reactor_addr = reactor_addr
        self.woke = 0


def _waker_entry(arg: _OpaquePtr) -> _OpaquePtr:
    """睡 100 ms 再叫醒 reactor。pthread 的启动函数不许抛，错误就地吞掉。"""
    var raw = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(arg))
    var ctx = raw.unsafe_bitcast[WakerCtx]()
    _ = external_call["usleep", Int32](Int32(100000))
    var reactor = Pointer[Reactor, MutUntrackedOrigin](
        unsafe_from_address=ctx[].reactor_addr
    )
    try:
        reactor[].wakeup()
    except:
        pass
    ctx[].woke = 1
    var null_addr = 0
    return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=null_addr)


def test_wakeup_from_another_thread() raises:
    """另一个线程能在 poll 的超时**之前**把它叫醒。

    engine 线程交回一个 token 走的就是这条路（roadmap 3.2：engine 独占线程）。
    它不通的话，reactor 只能靠 poll 超时"顺带"发现结果，每条流的延迟就等于
    poll 超时 —— 而那正是"事件驱动"与"定时轮询"的分界线。
    """
    var reactor = Reactor()
    var ctx = WakerCtx(Int(Pointer(to=reactor)))
    var arg = Pointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=Int(Pointer(to=ctx))
    )
    var handle = ThreadHandle.spawn[_waker_entry](arg)
    var events = List[Event]()
    var began = monotonic_ms()
    var n = reactor.poll(5000, events)
    var elapsed = monotonic_ms() - began
    _ = handle.join()
    assert_true(n >= 1, "the wakeup never arrived")
    assert_true(events[0].is_wakeup(), "the first event should be the wakeup")
    assert_true(
        elapsed < 2000, "the poll slept out instead of waking: " + String(elapsed) + "ms"
    )
    assert_equal(ctx.woke, 1, "the waker thread never ran")


def bytes_to_text(imm b: List[UInt8], start: Int, end: Int) -> String:
    """这个门里唯一需要的一小段：把读到的字节变成可断言的文本。

    `srv/http.mojo` 里那个同名的函数要带 `raises`，这里不需要 —— 能力门尽量少
    依赖被测层，免得"被测层坏了"把"平台有没有这个原语"这个问题一起带偏。
    """
    var out = List[UInt8]()
    var i = start
    while i < end:
        out.append(b[i])
        i += 1
    return String(unsafe_from_utf8=out)


def test_one_loop_watches_many_connections() raises:
    """一个 reactor 同时盯住 8 条连接，每条各自报可读 —— 连接并发的物理前提。

    一条一 worker 时"同时能有几条连接"等于 worker 数；reactor 化之后它等于
    `MAX_CONNS`。这条门钉的是后者成立：8 条连接各写一句，一次事件循环里 8 个
    都要被报出来（漏一个就说明 token 或注册出了问题，而漏报在客户端那边表现为
    "某条连接永远没人答"）。
    """
    comptime N = 8
    var listener = bind_reuseport(SocketAddr.localhost(PORT))
    var lfd = listener.as_raw_fd()
    assert_true(set_nonblocking(lfd) == 0, "fcntl(O_NONBLOCK) failed")

    var clients = List[TcpStream]()
    for _ in range(N):
        clients.append(TcpStream.connect(SocketAddr.localhost(PORT)))
    for i in range(N):
        clients[i].write_all(("ping" + String(i)).as_bytes())

    var reactor = Reactor()
    var conns = List[TcpStream]()
    var drained = False
    while len(conns) < 64:
        try:
            conns.append(accept_fd(lfd))
        except:
            drained = True
            break
    assert_true(drained, "accept never reached EAGAIN")
    assert_equal(len(conns), N, "every pending connection must be accepted")

    for i in range(N):
        reactor.register(
            c_int(conns[i].raw_fd()), UInt64(CONN_TOKEN_BASE) + UInt64(i), INTEREST_READ
        )
    assert_equal(reactor.registered_count(), N, "every connection must be registered")

    var events = List[Event]()
    _ = reactor.poll(500, events)
    var readable = 0
    for i in range(len(events)):
        if events[i].is_readable():
            readable += 1
    assert_true(
        readable >= N, "every connection must be reported readable: " + String(readable)
    )

    for i in range(N):
        var buf = List[UInt8](capacity=64)
        buf.resize(64, 0)
        var n = conns[i].read(buf.unsafe_ptr(), 64)
        assert_true(n > 0, "connection " + String(i) + " had nothing to read")
        assert_true(
            bytes_to_text(buf, 0, n).find("ping") >= 0,
            "unexpected bytes on connection " + String(i),
        )
        clients[i].close()
        conns[i].close()
    # 同上：`listener` 的最后一次使用在循环之后，监听 fd 才不会中途被关。
    listener.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
