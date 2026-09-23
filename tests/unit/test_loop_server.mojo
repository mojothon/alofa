"""Reactor 循环的端到端门：真 socket、真 fork、**并发**连接。

与 `tests/unit/test_http_server.mojo` 的差别是它盯的那条性质
--------------------------------------------------------------
那边的循环是"accept 一条 → 在这条上读到有完整请求为止"。这条门盯的是这条循环**不再
成立**的地方：一条连接上的等待不再连累别人。具体是两条：

1. **并发**：N 条连接同时开着、同时发请求、同时收应答。在旧形态下第二条连接要等第一
   条被服务完 —— 而"等"在旧形态里意味着"accept 都没轮到"。
2. **慢对端不连累别人**：一条连接发了一半就停在那儿（永远等不到剩下的体），另一条连
   接的请求**必须照常按时**。旧形态下这一条会让整个服务停 `RECV_TIMEOUT_MS`（5 秒）
   —— 所以这条门是**带时间上的断言**的：超过 2 秒就算失败。这是这条门里唯一会随时间
   环境波动的判据，也是唯一能直接测出"一条连接的 I/O 不再阻塞全体"的那个数。

为什么不在 `pixi run test` 里
------------------------------
它 fork 出一个子进程跑服务（JIT 下的 `fork` 会崩编译器），所以必须**先构建再跑**：
`pixi run test-loop`。

替身 `Stub` 不带权重：这一层与模型无关。
"""

from std.collections import List
from std.ffi import external_call
from std.testing import assert_equal, assert_true

from flare.net import SocketAddr
from flare.tcp import TcpStream

from alofa.core.ffi.posix import monotonic_ms
from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import bytes_of_text, bytes_to_text
from alofa.srv.loop import Loop
from alofa.srv.openai import NO_REQUEST, ChatHandler, Completion, Service
from alofa.srv.sse import StreamToken

comptime PORT = UInt16(18124)
comptime READ_CHUNK = 4096
# 并发门的连接数。它是**有意的小**：这条门要的是"第二条连接不等第一条"，8 条足够让
# "一条连接独占循环"现形，又不至于让 fork 出来的子进程在过载的本机上排队。
comptime CONCURRENT = 8
# 服务要回答的请求数：答完就退出，这个门才不会挂着一个不结束的进程。
# 8（并发）+ 1（慢对端那条之后的新连接）+ 2（keep-alive）+ 1（一条完整流）
# + 1（中断之后重新开始的那条流）= 13。发了一半的那个请求不算（它没被回答）。
comptime WANT_REQUESTS = 13
# "一条连接卡住，别的连接必须照常"的时间上限。旧形态下这个数是 5 秒（接收超时），
# 所以 2 秒这个界既能容纳本机过载，又不会放过"又变回阻塞了"。
comptime SLOW_PEER_BUDGET_MS = 2000
# 客户端的接收超时：服务没答就**报错**，而不是把这个门挂住。
comptime CLIENT_TIMEOUT_MS = 5000

comptime HEALTH = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
comptime STREAM_BODY = (
    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,"
    + "\"stream\":true}"
)
# 一条很长的流：客户端只想要头，剩下的帧会撞上"对端已经关了"。
comptime LONG_STREAM_BODY = (
    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":64,"
    + "\"stream\":true}"
)
# Content-Length 说 100，实际只发 5 个字节：一个"发了一半"的对端。
comptime HALF_SENT = (
    "POST /v1/chat/completions HTTP/1.1\r\n"
    + "Host: localhost\r\n"
    + "Content-Length: 100\r\n"
    + "\r\n"
    + "short"
)


struct Stub(Service, Twinable):
    var give: Int
    var stream_step: Int
    var stream_max: Int
    # `request` 是不是真的穿到了这一层：替身把它记下来，门去读它。
    var last_request: Int
    var ends: Int

    def __init__(out self, give: Int = -1):
        self.give = give
        self.stream_step = 0
        self.stream_max = 0
        self.last_request = NO_REQUEST
        self.ends = 0

    def complete(
        mut self, prompt: String, max_tokens: Int, temperature: Float64
    ) raises -> Completion:
        var n = max_tokens
        if self.give >= 0:
            n = self.give
        return Completion("S:" + prompt, 7, n)

    def spawn_twin(self, index: Int) raises -> Int:
        """engine 线程池要的"再给我一份"（零权重，所以这里是真的零成本）。"""
        return heap_place(Stub(self.give)^)

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
        # 每步睡 5 ms：真模型一步几十毫秒，而"客户端中途断开"这条门要的是**服务还在
        # 流里**的时候对端消失 —— 替身瞬间交完 64 帧的话，写永远成功，这条门就变成
        # 了没测到（安静地假绿）。
        _ = external_call["usleep", Int32](Int32(5000))
        if self.stream_step >= self.stream_max:
            return StreamToken("", True)
        var text = "s" + String(self.stream_step)
        self.stream_step += 1
        return StreamToken(text, False)


def child_serve() raises:
    var loop = Loop.bind(PORT)
    var handler = ChatHandler(Stub(), "stub")
    _ = loop.run(handler, WANT_REQUESTS, Int32(-1))
    loop.close()


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


def drain(mut conn: TcpStream, mut buf: List[UInt8]) raises -> Bool:
    """一直读到 EOF。服务没有按时关掉的话，这里会撞上接收超时并抛错。"""
    while read_more(conn, buf) > 0:
        pass
    return True


def tail_of(imm text: String, start: Int) -> String:
    var raw = bytes_of_text(text)
    return bytes_to_text(raw, start, len(raw))


def assert_framed(imm res: String, imm connection: String) raises:
    """钉住响应的分帧：头体之间有空行、`Content-Length` 等于体的字节数、`Connection`
    是期望的那个。只比"体里有没有那句话"会放过长度算错。"""
    var at = res.find("\r\n\r\n")
    assert_true(at > 0, "the response has no header terminator: " + res)
    var body = tail_of(res, at + 4)
    assert_true(
        res.find("Content-Length: " + String(body.byte_length()) + "\r\n") > 0,
        "Content-Length does not match the body: " + res,
    )
    assert_true(
        res.find("Connection: " + connection + "\r\n") > 0,
        "expected Connection: " + connection + " in: " + res,
    )


def sse_frames_of(imm res: String) -> List[String]:
    """把一条流式响应切成帧（不含头）。帧里没有空行（`sse_frame` 会拒绝裸换行），
    所以按空行切在这里是安全的。"""
    var raw = bytes_of_text(res)
    var out = List[String]()
    var head = res.find("\r\n\r\n")
    if head < 0:
        return out^
    var pos = head + 4
    while pos < len(raw):
        var end = res.find("\r\n\r\n", pos)
        if end < 0:
            out.append(bytes_to_text(raw, pos, len(raw)))
            pos = len(raw)
        else:
            out.append(bytes_to_text(raw, pos, end + 4))
            pos = end + 4
    return out^


def assert_stream(imm res: String) raises:
    """钉住一条流式响应的**线上形状**：头无长度、帧齐全、`[DONE]` 收尾。"""
    assert_true(res.find("HTTP/1.1 200 OK\r\n") == 0, "not a 200 stream head: " + res)
    assert_true(
        res.find("Content-Type: text/event-stream\r\n") > 0,
        "the head must announce SSE: " + res,
    )
    assert_true(res.find("Content-Length") < 0, "a stream must not claim a length: " + res)
    var frames = sse_frames_of(res)
    # 首帧（role）+ 3 个内容帧 + finish + [DONE]
    assert_equal(len(frames), 6, "frames: " + String(len(frames)))
    assert_true(frames[0].find("\"delta\":{\"role\":\"assistant\"") >= 0, frames[0])
    assert_true(frames[1].find("\"content\":\"s0\"") >= 0, frames[1])
    assert_true(frames[3].find("\"content\":\"s2\"") >= 0, frames[3])
    assert_equal(frames[5], "data: [DONE]\r\n\r\n")


def read_one_response(mut conn: TcpStream, mut buf: List[UInt8]) raises -> String:
    """发完请求后一直读到 EOF：流式响应靠关连接定界，所以 EOF 就是它的结尾。"""
    _ = drain(conn, buf)
    return bytes_to_text(buf, 0, len(buf))


def send_text(mut conn: TcpStream, imm text: String) raises:
    var sent = conn.write(text.as_bytes())
    assert_equal(sent, text.byte_length(), "the write was short")


def post_request(imm body: String) -> String:
    """`Content-Length` 由体算出来，不手填 —— 手填的长度是这条门最该抓的那类错，
    而它一旦出现在**测试**里，门就变成了"两个错互相印证"。"""
    return (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        + "Host: localhost\r\n"
        + "Content-Length: "
        + String(body.byte_length())
        + "\r\n\r\n"
        + body
    )


def connect(port: UInt16) raises -> TcpStream:
    """连上服务，并给客户端自己设一个接收超时：服务没答就报错，而不是把门挂住。"""
    var conn = TcpStream.connect(SocketAddr.localhost(port))
    conn.set_recv_timeout(CLIENT_TIMEOUT_MS)
    return conn^


def main() raises:
    var pid = external_call["fork", Int32]()
    if pid == 0:
        child_serve()
        # 直接 `_exit`：子进程不再跑一遍父进程的析构（fork 出来的 Mojo 运行时在退出
        # 路径上会打一段栈，那是噪音，不是这个门要盯的东西）。
        _ = external_call["_exit", Int32](Int32(0))
        return

    # 等子进程的 listen 起来（绑不上就连接被拒，而那看起来像"服务坏了"）。
    _ = external_call["usleep", Int32](Int32(300000))

    # 1) 并发：8 条连接**同时**开着、同时发请求、同时收应答。
    #    先全部连上再发、发完再统一收 —— 这样"第二条连接要等第一条被服务完"那种
    #    实现会现形：它会把 8 条连接变成 8 次串行。
    var conns = List[TcpStream]()
    var bufs = List[List[UInt8]]()
    for _ in range(CONCURRENT):
        conns.append(connect(PORT))
        bufs.append(List[UInt8]())
    for i in range(CONCURRENT):
        send_text(conns[i], HEALTH)
    for i in range(CONCURRENT):
        var got = await_responses(conns[i], bufs[i], 1)
        assert_true(len(got) >= 1, "connection " + String(i) + " got no response")
        assert_true(
            got[0].find("HTTP/1.1 200 OK\r\n") == 0,
            "connection " + String(i) + " not a 200: " + got[0],
        )
        assert_true(got[0].find("\"status\":\"ok\"") >= 0, got[0])
        assert_framed(got[0], "keep-alive")

    # 2) 一个发了一半就停在那儿的对端：**不能连累别人**。
    #    旧形态在这里会停 5 秒（接收超时）才去 accept 下一条连接 —— 所以这条断言
    #    带一个时间上限，它是"这条循环真的不阻塞"唯一的直接证据。
    var slow = connect(PORT)
    send_text(slow, HALF_SENT)

    var began = monotonic_ms()
    var other = connect(PORT)
    var other_buf = List[UInt8]()
    send_text(other, HEALTH)
    var answered = await_responses(other, other_buf, 1)
    var elapsed = monotonic_ms() - began
    assert_true(len(answered) >= 1, "a stalled peer must not swallow other requests")
    assert_true(
        answered[0].find("HTTP/1.1 200 OK\r\n") == 0, "not a 200: " + answered[0]
    )
    assert_true(
        elapsed < SLOW_PEER_BUDGET_MS,
        "a stalled peer delayed another connection by "
        + String(elapsed)
        + "ms — the loop is blocking again",
    )
    # 卡住的那条连接自己也没有被答（它没有完整的请求），服务只是没被它拖住。
    other.close()
    slow.close()

    # 3) keep-alive：一条连接上连着两个请求。
    var keep = connect(PORT)
    var keep_buf = List[UInt8]()
    send_text(keep, HEALTH)
    var first = await_responses(keep, keep_buf, 1)
    assert_true(len(first) >= 1, "no response to the first keep-alive request")
    assert_framed(first[0], "keep-alive")
    send_text(keep, HEALTH)
    var second = await_responses(keep, keep_buf, 2)
    assert_true(len(second) >= 2, "no second response on the same connection")
    assert_framed(second[1], "keep-alive")
    keep.close()

    # 4) 一条完整的流（并发连接里也能跑）。
    var stream = connect(PORT)
    var stream_buf = List[UInt8]()
    send_text(stream, post_request(STREAM_BODY))
    var streamed = read_one_response(stream, stream_buf)
    assert_stream(streamed)
    stream.close()

    # 5) 客户端在流中途离开：服务放弃这条流、关掉这条连接，然后继续活着。
    #    这一条同时是 SIGPIPE 的照妖镜 —— 一个断开的客户端把进程带走的话，下一条
    #    请求就没人答了。
    var gone = connect(PORT)
    var gone_buf = List[UInt8]()
    send_text(gone, post_request(LONG_STREAM_BODY))
    var head_only = await_responses(gone, gone_buf, 1)
    assert_true(len(head_only) >= 1, "no stream head after the request")
    assert_true(
        head_only[0].find("Content-Type: text/event-stream") >= 0,
        "the long stream did not start: " + head_only[0],
    )
    gone.close()
    # 给服务一点时间撞上"对端已经关了"。
    _ = external_call["usleep", Int32](Int32(200000))

    # 6) 服务还活着，而且下一条流**从头开始**（`begin_stream` 重置了相位与计数）：
    #    第一个内容帧必须是 s0，不能接着上一条流的计数往下走。
    var again = connect(PORT)
    var again_buf = List[UInt8]()
    send_text(again, post_request(STREAM_BODY))
    var restarted = read_one_response(again, again_buf)
    assert_stream(restarted)
    again.close()

    for i in range(CONCURRENT):
        conns[i].close()

    print(
        "PASS reactor loop: "
        + String(CONCURRENT)
        + " concurrent connections, a stalled peer delayed nobody"
        + " ("
        + String(elapsed)
        + "ms), keep-alive, and SSE streaming"
    )
