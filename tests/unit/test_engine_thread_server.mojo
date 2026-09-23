"""**engine 独占线程**的端到端门（roadmap 3.2b + 3.2c）：真 socket、真 fork、**真线程**。

这条门盯的两条性质
------------------
**一、一次生成不再占住这条循环（3.2b）**。生成期间这条循环必须还在跑 —— 还在
accept、还在读、还在写。判据是**它还能接受一条新连接**：

槽位被占满时，新连接会被"接进来立刻关"（`MAX_CONNS` 那条规则：留在 backlog 里同样会空
转，而"接了又关"至少让对端立刻知道）。对端因此能靠 **EOF** 看见这件事。所以这个门量的是
**从发起连到收到 EOF** 用了多久：

- `run_threaded`：几百毫秒 —— 循环正在跑，只是槽位满了，于是立刻关。
- `run`：**等完那次生成**（5 秒）—— 那段时间循环在 `handler.handle` 里，accept 根本
  没有被调用，那条连接一直躺在 backlog 里没人接。

**二、同时能生成的条数 = engine 线程数（3.2c）**。判据是**两条慢生成加起来用了多久**：

- `engines=2`：≈ 一次 `SLOW_MS`（5 秒）—— 两条**同时在跑**。
- `engines=1`：≈ 两次 `SLOW_MS`（10 秒）—— 一条一条来。

为什么量"两条加起来"而不是"第二条什么时候被答"：前者是**两条生成的总机时**，只有
"真的并行"才能把它压到一次 SLOW_MS；后者在"派得早但生成串行"下也会很快（那是排队，
不是并行 —— 3.2b 那一节里那条 health 的坑就是这么踩的）。

为什么拿"accept"当判据（而不是"另一条连接的请求被答得快"）
----------------------------------------------------------
3.2b 换的是**生成不再占住 I/O**，不是并发生成：engine 只有一条线程、一次一个 job，所以
"慢生成期间另一条连接发一个 health"在两种模式下**都会**等那次生成（health 排在慢请求后
面）。拿它当判据的话，一条 health 都没答的实现也能"看起来很快"，而真正被改到的那一处
（循环还在跑）根本没被测到。

为什么不拿"生成期间还有字节在走"当判据
--------------------------------------
直觉上它更直接（"那 800 KB 还在往外走"），但这条路**测不成**，实测走通过：

服务端一次能排队的字节被 `SEND_HIGH_WATER`（1 MiB）封住，而 loopback 上内核给一条
socket 的缓冲实测能吃下 **1.2–2.5 MB**（自动调优，随发送速率变）。也就是说服务端手上
一字节都不剩，客户端**不需要循环参与**就能把响应读完 —— 两种模式一样快（内联那一侧实测
38 ms 收完 800 KB），判据退化成"什么都不测"。那条数字由内核决定，不由被改动的那一处决
定；读那一侧同样被它挡住（`MAX_BODY_BYTES` 也是 1 MiB，而限速发送也能被吸收 1.26 MB）。
"accept"不受这个影响：内核替不了应用做 accept。

为什么三种模式都跑（常驻负向对照）
----------------------------------
只看"threaded 下那条 health 很快"是不够的：一个**把请求丢了**的实现也能让它很快。
所以同一个序列在 `run` 下再跑一遍，断言它在那里**必须慢** —— 这既是"这条门真的盯到
了那条性质"，也是"哪天 threaded 退化回内联，这个门会红"。

并发那一半同样带负向对照，而且对照是**同一条代码路径**（都是 `run_threaded`，只差
`engines`）：`engines=1` 那一次必须串行（≈ 10 秒）。没有它，"两条 5 秒的生成 5 秒
答完"可能只是"第二条根本没生成"。

另外两条（都是这一版才有可能会错的）
------------------------------------
- **逐字节一致**：三种模式下同一个请求的响应必须**逐字节相等**。生成搬到别的线程
  上，最容易漂的是流的分帧与收尾（一段一段交回来，和一次交回来，形状必须一样）。
  唯一被掩掉的是 `chatcmpl-<编号>`（以及跟着它变长的 `Content-Length`）：编号是
  **每一份 handler 自己**的计数器，而 `engines=2` 下同一个请求落在哪条线程上不由
  这个门决定（见 `normalize`）。
- **槽位不串味**：一条连接在生成期间断开，它的应答**不许**发给接下来复用这个槽位的
  连接。这个错只在"客户端恰好中途断开"时出现，所以必须专门造一次 —— 而且要让新连接
  **确实复用那个槽位**（所以这一小段之前先把别的连接全关掉，只剩那一条空位）。

为什么不在 `pixi run test` 里
------------------------------
它 fork 出一个子进程跑服务（JIT 下的 `fork` 会崩编译器），所以必须**先构建再跑**：
`pixi run test-engine`。

替身 `SlowStub` 不带权重：这一层与模型无关。它唯一的额外能力是"prompt 里有 `slow`
就睡 1.5 秒" —— 那是一个不需要权重就能造出来的"生成"。
"""

from std.collections import List
from std.ffi import external_call
from std.testing import assert_equal, assert_true

from flare.net import SocketAddr
from flare.tcp import TcpStream

from alofa.core.ffi.posix import monotonic_ms
from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import bytes_of_text, bytes_to_text
from alofa.srv.loop import MAX_CONNS, Loop
from alofa.srv.openai import NO_REQUEST, ChatHandler, Completion, Service
from alofa.srv.sse import StreamToken

comptime PORT_THREADED = UInt16(18130)
comptime PORT_INLINE = UInt16(18131)
comptime PORT_POOLED = UInt16(18132)
# 并发那一半的两个端口。它要**各自一台新服务**（`engines` 是起线程时定下来的，
# 而"两条连接各归一条线程"靠的是 `Loop.next_engine` 从 0 开始轮转 —— 复用一台
# 已经派过活的服务，那两条连接落在哪条线程上就不由这个门决定了）。
comptime PORT_CONC_PARALLEL = UInt16(18133)
comptime PORT_CONC_SERIAL = UInt16(18134)
comptime READ_CHUNK = 4096
# 服务最多答多少条就退出。给得**宽一点**：中途断开那条在两种模式下不一定是同一条 ——
# `run_threaded` 下它的结果因为"槽位换过主人"被丢掉（`served` 不增），`run` 下它照答
# （`served` 增）。数得刚刚好的话，其中一种模式会在最后一条请求之前就退出并把连接全
# 关掉 —— 症状是"最后一条请求连接被重置"。收尾靠 `kill_child`，不靠这个数。
# slowA(1) + plain(1) + stream(1) + slowmark(可能 1) + second(1) ≤ 5。
comptime WANT_REQUESTS = 8
# 替身"生成"一次的时长。它要**明显长于**一次往返，也要**明显长于**本机过载带来的
# 抖动 —— 5 秒是这两个下界之间离两边都远的那个数（两个判据分别在它的两侧，间隔以
# 秒计，本机 ±10% 的噪声吞不掉）。
comptime SLOW_MS = 5000
# `run_threaded` 下探子收到 EOF 的上限：它不该等任何生成。
comptime PROBE_FAST_MS = 2500
# `run` 下探子收到 EOF 的下限：它**必须**差不多等完那次生成（这就是负向对照）。
comptime PROBE_SLOW_MS = 4000
# 客户端的接收超时：服务没答就**报错**，而不是把这个门挂住。
comptime CLIENT_TIMEOUT_MS = 30000
# 派出去之后等多久再放探子：这段时间内 engine 必须已经**在生成**（量早了，测的是
# "还没开始生成"那一段，两种模式都会很快）。
comptime DISPATCH_WAIT_MS = 500
# 并发那一半：`engines` 的条数。`2` 是"能看出并行"的最小那个数（1 是负向对照）。
comptime ENGINES_POOLED = 2
# 两条 `SLOW_MS` 的生成并行时**不该超过**这个数（1.4 × SLOW_MS：留 2 秒给往返与
# 本机过载），串行时**不该低于**这个数（1.6 × SLOW_MS）。两个判据隔着 1 秒的空隙，
# 本机 ±10% 的噪声吞不掉它。
comptime CONC_PARALLEL_MS = 7000
comptime CONC_SERIAL_MS = 8000

comptime MODE_THREADED = 0
comptime MODE_INLINE = 1
comptime MODE_POOLED = 2


def chat_body(imm content: String, max_tokens: Int, stream: Bool) -> String:
    """请求体。**长度由体算出来**，不手填 —— 手填的长度是这条门最该抓的那类错，
    而它一旦出现在**测试**里，门就变成了"两个错互相印证"。"""
    var body = (
        "{\"messages\":[{\"role\":\"user\",\"content\":\""
        + content
        + "\"}],\"max_tokens\":"
        + String(max_tokens)
        + ",\"stream\":"
    )
    if stream:
        body += "true}"
    else:
        body += "false}"
    return body^


def post_request(imm body: String, close: Bool) -> String:
    """一个 chat 请求。`close=True` 时服务端答完就关连接 —— 客户端因此能靠 **EOF**
    知道"响应收完了"，而不是去数 `Content-Length`（数错了会让这条门安静地测错对象，
    而数错恰恰是这类测试最容易犯的错）。"""
    var head = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        + "Host: localhost\r\n"
        + "Content-Length: "
        + String(body.byte_length())
        + "\r\n"
    )
    if close:
        head += "Connection: close\r\n"
    return head + "\r\n" + body


struct SlowStub(Service, Twinable):
    """一个"生成"会睡的替身：prompt 里带 `slow` 就睡 `SLOW_MS`。

    为什么要让**内容**决定睡不睡：这条门的时序是"先派一个慢的、再派一个快的"，而派
    什么只能由请求内容区分 —— 靠顺序区分的话，一旦哪天循环换了扫描方向，这条门就会
    安静地测错对象。
    """

    var stream_step: Int
    var stream_max: Int
    # `request` 是不是真的穿到了这一层：替身把它记下来，门去读它。
    var last_request: Int
    var ends: Int

    def __init__(out self):
        self.stream_step = 0
        self.stream_max = 0
        self.last_request = NO_REQUEST
        self.ends = 0

    def complete(
        mut self, prompt: String, max_tokens: Int, temperature: Float64
    ) raises -> Completion:
        if prompt.find("slow") >= 0:
            _ = external_call["usleep", Int32](Int32(SLOW_MS * 1000))
        return Completion("S:" + prompt, 7, max_tokens)

    def spawn_twin(self, index: Int) raises -> Int:
        """engine 线程池要的"再给我一份"（零权重，所以这里是真的零成本）。

        ⚠️ 为什么交地址而不是值：Mojo 1.0 的 trait 方法不能把 `Self` 当返回类型
        （见 `srv/engine_thread.mojo` 里 `Twinable` 的注释）。
        """
        return heap_place(SlowStub()^)

    def stream_begin(
        mut self,
        prompt: String,
        max_tokens: Int,
        temperature: Float64,
        request: Int,
    ) raises:
        self.stream_max = max_tokens
        self.stream_step = 0
        self.last_request = request

    def stream_end(mut self, request: Int) raises:
        """幂等：号对不上就什么也不做（契约见 `srv/openai.mojo` 的 `Service`）。

        替身在这里数的 `ends` 是给门看的：一条流**必须**被还回来一次，没有回收
        路径的批调度会把"新请求被拒"留到线上才现形。
        """
        if self.last_request != request:
            return
        self.ends += 1
        self.last_request = NO_REQUEST

    def stream_next(mut self, request: Int) raises -> StreamToken:
        # 每步睡 5 ms：真模型一步几十毫秒，而这一段要的是"服务还在流里"的时候另一条
        # 连接也能被答 —— 替身瞬间交完所有帧的话，那条性质就没有被测到。
        _ = external_call["usleep", Int32](Int32(5000))
        if self.stream_step >= self.stream_max:
            return StreamToken("", True, False)
        var text = "s" + String(self.stream_step)
        self.stream_step += 1
        return StreamToken(text, False, False)


def child_serve(mode: Int, port: UInt16) raises:
    """子进程：起一个服务，答完 `WANT_REQUESTS` 条就退出。

    `mode` 是 fork **之前**父进程里那个变量的值（fork 会复制内存，所以子进程看到的
    就是那一刻的值）—— 这是在没有模块级可变全局的语言里给子进程传参数的办法。
    """
    var loop = Loop.bind(port)
    var handler = ChatHandler(SlowStub(), "stub")
    if mode == MODE_INLINE:
        _ = loop.run(handler, WANT_REQUESTS, Int32(-1))
    elif mode == MODE_POOLED:
        _ = loop.run_threaded(handler, WANT_REQUESTS, Int32(-1), ENGINES_POOLED)
    else:
        _ = loop.run_threaded(handler, WANT_REQUESTS, Int32(-1))
    loop.close()


def spawn_server(mode: Int, port: UInt16) raises -> Int32:
    """**fork** 出子进程跑服务，然后等它的 listen 起来。返回子进程 pid。"""
    var pid = external_call["fork", Int32]()
    if pid == 0:
        child_serve(mode, port)
        # 直接 `_exit`：子进程不再跑一遍父进程的析构（fork 出来的 Mojo 运行时在退出
        # 路径上会打一段栈，那是噪音，不是这个门要盯的东西）。
        _ = external_call["_exit", Int32](Int32(0))
        return Int32(0)
    _ = external_call["usleep", Int32](Int32(300000))
    return pid


def kill_child(pid: Int32) raises:
    """把服务子进程收掉（SIGKILL）。

    ⚠️ 不收的后果不是"多一个进程"那么轻：这些 socket 是 **SO_REUSEPORT** 的（多
    worker 靠它让内核分发连接），所以下一轮新 fork 出来的服务会和上一轮遗留的那个
    **一起**监听同一个端口，连接被内核随机分给其中一个 —— 症状是"请求偶尔没人答"、
    "响应只有半截"，看起来完全像服务有间歇性 bug。所以**失败的路径上也要收**（见
    `main` 里的 try/except）。
    """
    _ = external_call["kill", Int32](pid, Int32(9))
    _ = external_call["usleep", Int32](Int32(50000))


def split_responses(imm buf: List[UInt8]) -> List[String]:
    """按 `HTTP/1.1 ` 把收到的字节切成一条条响应（替身的文本里不会出现这串字节）。"""
    var marker = bytes_of_text("HTTP/1.1 ")
    var out = List[String]()
    var start = 0
    var i = 0
    while i + len(marker) <= len(buf):
        var hit = True
        for j in range(len(marker)):
            if buf[i + j] != marker[j]:
                hit = False
        if hit and i > 0:
            out.append(bytes_to_text(buf, start, i))
            start = i
        i += 1
    if start < len(buf):
        out.append(bytes_to_text(buf, start, len(buf)))
    return out^


def starts_at(imm buf: List[UInt8], at: Int, imm marker: List[UInt8]) -> Bool:
    if at + len(marker) > len(buf):
        return False
    for j in range(len(marker)):
        if buf[at + j] != marker[j]:
            return False
    return True


def normalize(imm text: String) -> String:
    """把响应里**两处必然随 engine 条数变**的数字换成 `#`。

    - `chatcmpl-<编号>`：编号是**每一份 handler 自己**的计数器（N 条 engine 线程
      = N 份 handler，起点按 `ID_STRIDE` 错开，见 `srv/openai.mojo`），所以
      `engines=2` 下同一个请求落在哪条线程上，编号就不同 —— 那是设计，不是串味。
    - `Content-Length: <数字>`：它是**体**的长度，而体里就装着那个编号。掩了编号
      不掩它，比较的就变成"编号有几位"，而不是响应的形状。

    除了这两处，三种模式的响应必须**逐字节相等**：状态行、头的顺序、JSON 字段的
    顺序与取值、SSE 的分帧与 `[DONE]` 收尾全在里面 —— 那正是"生成搬到另一条线程
    上"最容易漂的地方。

    ⚠️ 掩的是数字而不是整段（`"chatcmpl-#"` 而不是 `"#"`）：连前缀一起掩掉的话，
    哪天某条响应的 id 换了前缀，这个比较会安静地什么都不比。
    """
    var buf = bytes_of_text(text)
    var id_marker = bytes_of_text("chatcmpl-")
    var len_marker = bytes_of_text("Content-Length: ")
    var i = 0
    while i < len(buf):
        var m = -1
        if starts_at(buf, i, id_marker):
            m = len(id_marker)
        elif starts_at(buf, i, len_marker):
            m = len(len_marker)
        if m < 0:
            i += 1
            continue
        var k = i + m
        var digits = 0
        while k < len(buf) and buf[k] >= 48 and buf[k] <= 57:
            digits += 1
            k += 1
        if digits == 0:
            i += 1
            continue
        # `digits` 位数字换成一位 `#`：多出来的 digits-1 位删掉。
        buf[i + m] = 35
        var d = 0
        while d < digits - 1:
            _ = buf.pop(i + m + 1)
            d += 1
        i = i + m + 1
    return bytes_to_text(buf, 0, len(buf))


def read_more(mut conn: TcpStream, mut buf: List[UInt8]) raises -> Int:
    var base = len(buf)
    buf.resize(base + READ_CHUNK, 0)
    var got = conn.read(buf.unsafe_ptr().unsafe_offset(base), READ_CHUNK)
    buf.resize(base + got, 0)
    return got


def await_responses(
    mut conn: TcpStream, mut buf: List[UInt8], want: Int
) raises -> List[String]:
    """读到至少 `want` 条响应，或者对端关掉连接为止。"""
    var got = split_responses(buf)
    while len(got) < want:
        if read_more(conn, buf) == 0:
            break
        got = split_responses(buf)
    return got^


def drain(mut conn: TcpStream, mut buf: List[UInt8]) raises:
    """一直读到 EOF。服务没有按时关掉的话，这里会撞上接收超时并抛错。"""
    while read_more(conn, buf) > 0:
        pass


def connect(port: UInt16) raises -> TcpStream:
    var conn = TcpStream.connect(SocketAddr.localhost(port))
    conn.set_recv_timeout(CLIENT_TIMEOUT_MS)
    return conn^


def fill_slots(port: UInt16, want: Int) raises -> List[TcpStream]:
    """接上 `want` 条**什么都不发**的连接，把服务端的槽位占住。

    为什么要占满：槽位没满时新连接会被**收下并保持**，对端什么都收不到 —— 判据就没
    了。所以下面还有一条自检（"现在再来一条必须被立刻关"），那一条不快就说明槽位没
    占满，而这个门会安静地什么都不测。
    """
    var conns = List[TcpStream]()
    for _ in range(want):
        var conn = connect(port)
        conns.append(conn^)
    return conns^


def await_eof_ms(mut conn: TcpStream) raises -> Int:
    """等到对端关掉这条连接，返回等了多少毫秒。

    服务没关的话这里会撞上接收超时并**抛错** —— 那正是"槽位满了却没人来接"该有的失
    败方式（挂住比报错糟：挂住的门看起来像"还在跑"）。
    """
    var began = monotonic_ms()
    var buf = List[UInt8]()
    while read_more(conn, buf) > 0:
        pass
    return monotonic_ms() - began


def send_text(mut conn: TcpStream, imm text: String) raises:
    var sent = conn.write(text.as_bytes())
    assert_equal(sent, text.byte_length(), "the write was short")


def run_scenario(mode: Int, port: UInt16) raises -> List[String]:
    """跑一遍同一个序列，返回"逐字节一致"那两条要用的响应文本。

    时序是刻意排的：**先把槽位占满、再派一个慢的** —— 慢的生成期间，循环必须还能
    accept（槽位满了，所以它只能"接了立刻关"，那正是探子能看见的 EOF）。
    """
    print("  [gate] scenario: mode=" + String(mode) + " port=" + String(Int(port)))
    # 1) 一条连接先占住一个槽位 —— 慢生成就派在它上面。
    var slow_conn = connect(port)
    var slow_buf = List[UInt8]()
    # 2) 把**剩下的**槽位占满（`MAX_CONNS` 是服务端的上界，从 `loop.mojo` 导入，不
    #    在门里另写一份 —— 另写一份的话，那边改了这边会安静地测错对象）。
    var idle = fill_slots(port, MAX_CONNS - 1)

    # 3) 自检：槽位确实满了 —— 现在再来一条，必须被"接了立刻关"。这一条不快，说明槽
    #    位没占满，下面那个判据就是空的（探子会被收下并保持，永远等不到 EOF）。
    var pre = connect(port)
    var pre_ms = await_eof_ms(pre)
    assert_true(
        pre_ms < PROBE_FAST_MS,
        "the slots were not full: the extra client waited "
        + String(pre_ms)
        + "ms for the close, so the probe below would prove nothing",
    )

    # 4) 派一次 `SLOW_MS` 的生成；等它确实**在生成**了，再放探子。
    var began = monotonic_ms()
    send_text(slow_conn, post_request(chat_body("slowA", 1, False), False))
    _ = external_call["usleep", Int32](Int32(DISPATCH_WAIT_MS * 1000))
    var probe = connect(port)
    var eof_ms = await_eof_ms(probe)
    print("  [gate] probe: eof_ms=" + String(eof_ms) + " (pre=" + String(pre_ms) + ")")

    if mode == MODE_INLINE:
        # 负向对照：内联模式下这条连接**必须**等到生成结束才有人来接。它快了，说明这
        # 个门没有真的盯住那条性质。
        assert_true(
            eof_ms >= PROBE_SLOW_MS,
            "the inline loop accepted in "
            + String(eof_ms)
            + "ms — the control case did not block, so the gate proves nothing",
        )
    else:
        # `MODE_THREADED` 与 `MODE_POOLED` 都必须快：这一条性质（循环还能 accept）
        # 与 engine 有几条线程无关 —— 线程数只影响"同时能生成几条"。
        assert_true(
            eof_ms < PROBE_FAST_MS,
            "a generation held the loop for "
            + String(eof_ms)
            + "ms — the loop did not accept while generating",
        )

    # 5) 那条慢请求自己还是要被答完（engine 不能在退出时把它丢下）。
    var slow_got = await_responses(slow_conn, slow_buf, 1)
    var slow_ms = monotonic_ms() - began
    assert_true(len(slow_got) >= 1, "the slow request was never answered")
    assert_true(slow_got[0].find("S:slowA") >= 0, slow_got[0])
    assert_true(
        slow_ms >= PROBE_SLOW_MS,
        "the slow request came back in " + String(slow_ms) + "ms — it did not generate",
    )

    # 6) 把占位的连接全放掉：后面几条请求需要**空槽位**（槽位还满着的话，它们会被
    #    "接了立刻关"，症状是"连接被重置" —— 那看起来像服务坏了，其实是这条门自己
    #    占着不放）。
    for i in range(len(idle)):
        idle[i].close()
    _ = external_call["usleep", Int32](Int32(200000))

    # 7) 逐字节一致要用的两条：一条非流式 + 一条完整的流。
    var plain_conn = connect(port)
    var plain_buf = List[UInt8]()
    send_text(plain_conn, post_request(chat_body("plain", 2, False), False))
    var plain = await_responses(plain_conn, plain_buf, 1)
    assert_true(len(plain) >= 1, "the plain request was never answered")

    var stream_conn = connect(port)
    var stream_buf = List[UInt8]()
    send_text(stream_conn, post_request(chat_body("plain", 3, True), True))
    drain(stream_conn, stream_buf)
    var streamed = bytes_to_text(stream_buf, 0, len(stream_buf))
    print(
        "  [gate] slow_ms="
        + String(slow_ms)
        + " stream bytes="
        + String(len(stream_buf))
    )
    assert_true(
        streamed.find("Content-Type: text/event-stream") >= 0,
        "the stream did not start: " + streamed,
    )
    assert_true(streamed.find("data: [DONE]") >= 0, "the stream did not end: " + streamed)

    # 8) 槽位不串味：把前面的连接**全关掉**，只留下一条空槽位，好让下一条新连接
    #    必然复用它 —— 不复用的话，这一小段就是一个安静的假绿。
    slow_conn.close()
    plain_conn.close()
    stream_conn.close()
    _ = external_call["usleep", Int32](Int32(100000))

    var gone = connect(port)
    send_text(gone, post_request(chat_body("slowmark", 1, False), False))
    # 等到它确实被派给 engine（派出去之前断开，测的就不是"在途结果"了）。
    _ = external_call["usleep", Int32](Int32(200000))
    gone.close()
    # 再等服务端把这条断掉的连接**真的收掉**（槽位还回去）：下一个客户端必须拿到同一个
    # 槽位号，"串味"才可能发生、才可能被测到。
    _ = external_call["usleep", Int32](Int32(200000))

    var next = connect(port)
    var next_buf = List[UInt8]()
    send_text(next, post_request(chat_body("second", 1, False), False))
    var next_got = await_responses(next, next_buf, 1)
    assert_true(len(next_got) >= 1, "the second client was never answered")
    assert_true(
        next_got[0].find("S:second") >= 0,
        "the second client did not get its own answer: " + next_got[0],
    )
    assert_true(
        next_got[0].find("slowmark") < 0,
        "an answer meant for a departed client was delivered to the next one: "
        + next_got[0],
    )
    next.close()

    var out = List[String]()
    out.append(plain[0])
    out.append(streamed)
    return out^


def run_concurrency(mode: Int, port: UInt16) raises -> Int:
    """两条 `SLOW_MS` 的生成**同时**派出去，量"两条都答完"用了多久。

    为什么量这个（而不是"第二条什么时候被答"）：见文件头"二" —— 后者在"派得早但
    生成串行"下也会很快，那是排队，不是并行。

    两条请求必须发在**两条连接**上：一个槽位上一次只派一件在途的事（`inflight`），
    所以一条连接上的两个请求本来就是串行的 —— 拿它量并发，量不到东西。

    两台服务都是**新起的**：`engines` 是起线程那一刻定下来的，而"这两条连接各归
    一条线程"靠的是 `Loop.next_engine` 从 0 开始轮转 —— 复用一台已经派过活的
    服务，它们落在哪条线程上就不由这个门决定了。
    """
    var pid = spawn_server(mode, port)
    var ms = -1
    var err = ""
    try:
        var a = connect(port)
        var b = connect(port)
        var buf_a = List[UInt8]()
        var buf_b = List[UInt8]()
        var began = monotonic_ms()
        send_text(a, post_request(chat_body("slowX", 1, False), False))
        send_text(b, post_request(chat_body("slowY", 1, False), False))
        # 先读 A 再读 B：墙钟于是等于"两条都好"的那一刻，而不是"第一条好的那一刻"。
        var got_a = await_responses(a, buf_a, 1)
        var got_b = await_responses(b, buf_b, 1)
        ms = monotonic_ms() - began
        assert_true(len(got_a) >= 1, "the first slow request was never answered")
        assert_true(len(got_b) >= 1, "the second slow request was never answered")
        assert_true(got_a[0].find("S:slowX") >= 0, got_a[0])
        assert_true(got_b[0].find("S:slowY") >= 0, got_b[0])
    except e:
        err = String(e)
    kill_child(pid)
    assert_true(err == "", "the concurrency scenario failed: " + err)
    return ms


def main() raises:
    var threaded = List[String]()
    var inline = List[String]()
    var pooled = List[String]()
    # 子进程**必须**在两条路（成功与失败）上都收掉：留一个下来，下一轮就会有两个服务
    # 在同一个 SO_REUSEPORT 端口上分连接（见 `kill_child`）。
    var pid_threaded = spawn_server(MODE_THREADED, PORT_THREADED)
    var threaded_err = ""
    try:
        threaded = run_scenario(MODE_THREADED, PORT_THREADED)
    except err:
        threaded_err = String(err)
    kill_child(pid_threaded)
    assert_true(threaded_err == "", "the threaded scenario failed: " + threaded_err)

    # ⚠️ 换端口再跑一遍内联版：同一台机器上 TIME_WAIT 会让第二次 bind 失败，那看起
    # 来像"服务坏了"，其实是端口还没放开。
    var pid_inline = spawn_server(MODE_INLINE, PORT_INLINE)
    var inline_err = ""
    try:
        inline = run_scenario(MODE_INLINE, PORT_INLINE)
    except err:
        inline_err = String(err)
    kill_child(pid_inline)
    assert_true(inline_err == "", "the inline scenario failed: " + inline_err)

    # ⚠️ 第三个端口：`engines=2` 那一版也要跑同一个序列（多一条线程 = 多一处"流
    # 的相位可能串"），而且它同样参与下面的逐字节比较。
    var pid_pooled = spawn_server(MODE_POOLED, PORT_POOLED)
    var pooled_err = ""
    try:
        pooled = run_scenario(MODE_POOLED, PORT_POOLED)
    except err:
        pooled_err = String(err)
    kill_child(pid_pooled)
    assert_true(pooled_err == "", "the pooled scenario failed: " + pooled_err)

    # 逐字节：生成搬到别的线程上，响应的**形状**不许变（三种模式必须一模一样）。
    assert_equal(threaded[0], inline[0], "the plain answer changed shape")
    assert_equal(threaded[1], inline[1], "the streamed answer changed shape")
    assert_equal(
        normalize(threaded[0]),
        normalize(pooled[0]),
        "the plain answer changed shape at engines=2",
    )
    assert_equal(
        normalize(threaded[1]),
        normalize(pooled[1]),
        "the streamed answer changed shape at engines=2",
    )

    # 并发那一半：两条 `SLOW_MS` 的生成，`engines=2` 必须并行、`engines=1` 必须串行。
    var parallel_ms = run_concurrency(MODE_POOLED, PORT_CONC_PARALLEL)
    print("  [gate] two slow generations, engines=2: " + String(parallel_ms) + "ms")
    assert_true(
        parallel_ms < CONC_PARALLEL_MS,
        "two "
        + String(SLOW_MS)
        + "ms generations took "
        + String(parallel_ms)
        + "ms with 2 engines — they did not run at the same time",
    )
    var serial_ms = run_concurrency(MODE_THREADED, PORT_CONC_SERIAL)
    print("  [gate] two slow generations, engines=1: " + String(serial_ms) + "ms")
    assert_true(
        serial_ms >= CONC_SERIAL_MS,
        "two "
        + String(SLOW_MS)
        + "ms generations took "
        + String(serial_ms)
        + "ms with 1 engine — the control case did not serialize, so the gate"
        + " proves nothing",
    )

    print(
        "PASS engine thread: the loop still accepts while a generation runs"
        + " (control case waits for the generation), two generations run at the"
        + " same time with 2 engines (control case serializes at 1), answers are"
        + " byte-identical across all three loops, and a departed client's answer"
        + " is not delivered to the next one"
    )
