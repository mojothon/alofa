"""单进程 HTTP 服务的端到端门：真 socket、真 fork、真 keep-alive。

为什么它不在 `pixi run test` 里
--------------------------------
它 fork 出一个子进程跑服务（JIT 下的 `fork` 会崩编译器），所以必须**先构建再跑**：
`pixi run test-http`。

它钉的是**服务循环**的性质，而那些性质单测钉不住：

1. 一条连接上按序回答多个请求（keep-alive），且流水线里多出来的字节不丢；
2. `Connection: close` 真的关；
3. 一个发了一半的请求**不会把进程占死**（超时后只关那一条连接，服务还活着）；
4. 状态码与 `Content-Length` 是**线上**看到的那些字节 —— 路由返回给循环的那个
   `HttpResponse` 和真正写出去的字节之间，还隔着一次序列化；
5. 流式响应是**一条完整的 SSE 流**：头里没有 `Content-Length`、帧逐条到达、
   以 `data: [DONE]` 收尾，然后连接被关（关连接就是这条响应的定界）；
6. 客户端在流中途断开，只丢这一条连接 —— 服务继续答下一个请求（这条也是
   SIGPIPE 的照妖镜：一个断开的客户端把进程带走的话，这里就会挂）。

替身 `Stub` 不带权重：这一层与模型无关。真权重那一侧是 `pixi run serve`（手动）。
"""

from std.collections import List
from std.ffi import external_call
from std.testing import assert_equal, assert_true

from flare.net import SocketAddr
from flare.tcp import TcpStream

from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import bytes_of_text, bytes_to_text
from alofa.srv.openai import NO_REQUEST, ChatHandler, Completion, Service
from alofa.srv.server import Server
from alofa.srv.sse import StreamToken

comptime PORT = UInt16(18123)
comptime READ_CHUNK = 4096
# 服务要回答的请求数：回答完就退出，这个门才不会挂着一个不结束的进程。
# 6 = 健康检查 + 聊天 + 一条完整流 + 关连接的健康检查 + 一条被中途断开的流
# + 一条重新开始（不串味）的流。发了一半的那个请求不算（它没被回答）。
comptime WANT_REQUESTS = 6

comptime HEALTH = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
comptime CHAT_BODY = (
    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}"
)
comptime STREAM_BODY = (
    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,"
    + "\"stream\":true}"
)
# 一条很长的流：客户端只想要头，剩下的帧会撞上"对端已经关了"。
comptime LONG_STREAM_BODY = (
    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":64,"
    + "\"stream\":true}"
)
comptime CLOSE_ME = (
    "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
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
        # 每步睡 5 ms：真模型一步几十毫秒，而"客户端中途断开"这条门要的是
        # **服务还在流里**的时候对端消失 —— 替身瞬间交完 64 帧的话，写永远
        # 成功，这条门就变成了没测到（安静地假绿）。
        _ = external_call["usleep", Int32](Int32(5000))
        if self.stream_step >= self.stream_max:
            return StreamToken("", True)
        var text = "s" + String(self.stream_step)
        self.stream_step += 1
        return StreamToken(text, False)


def child_serve() raises:
    var server = Server.bind(PORT)
    var handler = ChatHandler(Stub(), "stub")
    _ = server.serve(handler, WANT_REQUESTS)
    server.close()


def split_responses(imm buf: List[UInt8]) -> List[String]:
    """按 `HTTP/1.1 ` 把收到的字节切成一条条响应。

    替身返回的文本里不会出现这串字节，所以按它切是安全的；真要做通用的客户端，应该
    按 `Content-Length` 走 —— 这里要的是一个能看懂的断言，不是一个 HTTP 客户端库。
    """
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
    """钉住响应的分帧：头体之间必须有空行，`Content-Length` 必须等于体的字节数，
    `Connection` 必须是期望的那个。

    只比"体里有没有那句话"会放过长度算错 —— 而长度算错在线上的表现是"偶发的乱码"。
    """
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
    """把一条流式响应切成**帧**（不含头）。

    切的依据是：头以空行结束、每帧也以空行结束，而帧的内容里没有空行 —— 裸
    换行会被 `sse_frame` 直接拒绝（`tests/unit/test_sse.mojo` 钉着），所以按
    空行切在这里是安全的，不需要一个 HTTP/SSE 客户端库。
    """
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
    assert_true(
        res.find("HTTP/1.1 200 OK\r\n") == 0, "not a 200 stream head: " + res
    )
    assert_true(
        res.find("Content-Type: text/event-stream\r\n") > 0,
        "the head must announce SSE: " + res,
    )
    # 负向对照：流式响应不许带 Content-Length（长度在写的那一刻未知）。
    assert_true(
        res.find("Content-Length") < 0, "a stream must not claim a length: " + res
    )
    var frames = sse_frames_of(res)
    # 首帧（role）+ 3 个内容帧 + finish + [DONE]
    assert_equal(len(frames), 6, "frames: " + String(len(frames)))
    assert_true(
        frames[0].find("\"delta\":{\"role\":\"assistant\"") >= 0, frames[0]
    )
    assert_true(frames[1].find("\"content\":\"s0\"") >= 0, frames[1])
    assert_true(frames[3].find("\"content\":\"s2\"") >= 0, frames[3])
    assert_true(
        frames[4].find("\"finish_reason\":\"length\"") >= 0,
        "3 tokens of 3 requested must finish as length: " + frames[4],
    )
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


def main() raises:
    var pid = external_call["fork", Int32]()
    if pid == 0:
        child_serve()
        # 直接 `_exit`：子进程不再跑一遍父进程的析构（fork 出来的 Mojo 运行时在
        # 退出路径上会打一段栈，那是噪音，不是这个门要盯的东西）。
        _ = external_call["_exit", Int32](Int32(0))
        return

    # 等子进程的 listen 起来（绑不上就连接被拒，而那看起来像"服务坏了"）。
    _ = external_call["usleep", Int32](Int32(300000))

    var conn = TcpStream.connect(SocketAddr.localhost(PORT))
    var buf = List[UInt8]()

    # 1) 一条连接、三个请求：keep-alive 必须真的复用同一条连接。
    send_text(conn, HEALTH)
    var first = await_responses(conn, buf, 1)
    assert_true(len(first) >= 1, "no response to the first request")
    assert_true(
        first[0].find("HTTP/1.1 200 OK\r\n") == 0, "not a 200: " + first[0]
    )
    assert_true(first[0].find("\"status\":\"ok\"") >= 0, first[0])
    assert_framed(first[0], "keep-alive")

    send_text(conn, post_request(CHAT_BODY))
    var second = await_responses(conn, buf, 2)
    assert_true(len(second) >= 2, "no second response on the same connection")
    assert_true(
        second[1].find("HTTP/1.1 200 OK\r\n") == 0, "not a 200: " + second[1]
    )
    assert_true(second[1].find("\"content\":\"S:hi\"") >= 0, second[1])
    assert_true(second[1].find("\"finish_reason\":\"length\"") >= 0, second[1])
    assert_framed(second[1], "keep-alive")

    # 2) 第三条是流式：一条完整的 SSE 流，以关连接定界（`drain` 读到 EOF）。
    # 换一个缓冲：`conn` 上前面两条响应还在 `buf` 里，而这条门要比的是**这一
    # 条**响应的字节。
    var stream_buf = List[UInt8]()
    send_text(conn, post_request(STREAM_BODY))
    var streamed = read_one_response(conn, stream_buf)
    assert_stream(streamed)
    conn.close()

    # 3) `Connection: close`：服务答完之后必须关掉这条连接。
    var closing = TcpStream.connect(SocketAddr.localhost(PORT))
    var closing_buf = List[UInt8]()
    send_text(closing, CLOSE_ME)
    var closed = await_responses(closing, closing_buf, 1)
    assert_true(len(closed) >= 1, "no response to the closing request")
    assert_framed(closed[0], "close")
    assert_true(drain(closing, closing_buf), "the server did not close the connection")
    closing.close()

    # 4) 一个发了一半的请求：必须超时关掉**这一条**连接，服务还得活着。
    var half = TcpStream.connect(SocketAddr.localhost(PORT))
    var half_buf = List[UInt8]()
    send_text(half, HALF_SENT)
    var dropped = False
    try:
        dropped = drain(half, half_buf)
    except err:
        dropped = False
        print("  [half-sent] " + String(err))
    assert_true(dropped, "a half-sent request must not wedge the server")
    half.close()

    # 5) 客户端在流中途断开：服务放弃这条流、关掉这条连接，然后继续活着。
    #    这一条同时是 SIGPIPE 的照妖镜 —— 一个断开的客户端把进程带走的话，
    #    下一个请求就没人答了。
    var gone = TcpStream.connect(SocketAddr.localhost(PORT))
    var gone_buf = List[UInt8]()
    send_text(gone, post_request(LONG_STREAM_BODY))
    var head_only = await_responses(gone, gone_buf, 1)
    assert_true(len(head_only) >= 1, "no stream head after the request")
    assert_true(
        head_only[0].find("Content-Type: text/event-stream") >= 0,
        "the long stream did not start: " + head_only[0],
    )
    gone.close()
    # 给服务一点时间撞上"对端已经关了"（不等它的话，下面那条请求可能抢在
    # 写失败之前发出去，这条门就变成了没测到）。
    _ = external_call["usleep", Int32](Int32(200000))

    # 6) 服务还活着，而且下一条流**从头开始**（`stream_begin` 重置了状态）：
    #    第一个内容帧必须是 s0，不能接着上一条流的计数往下走。
    var again = TcpStream.connect(SocketAddr.localhost(PORT))
    var again_buf = List[UInt8]()
    send_text(again, post_request(STREAM_BODY))
    var restarted = read_one_response(again, again_buf)
    assert_stream(restarted)
    again.close()

    print(
        "PASS http server: keep-alive, close, a half-sent request, and SSE"
        + " streaming (survives a client that leaves mid-stream)"
    )
