"""多 worker 门的端到端测试：真 fork ×3、真 SO_REUSEPORT、真优雅退出。

为什么它不在 `pixi run test` 里
--------------------------------
与 `test_http_server.mojo` 同因：fork 在 JIT 下崩编译器，走 `pixi run
test-workers`（先 build 再跑）。

它钉的是「多了一层进程编排」（`srv/master.mojo`）之后才存在的性质：

1. 三个 worker 各自 bind 同一端口（SO_REUSEPORT），连接走内核分发 —— 每个
   响应的分帧、状态码、体都必须与单进程版**完全一致**：进程数不该改变
   HTTP 契约；
2. 响应 id 从每个 worker 自己的 1 开始计：`"id":"chatcmpl-1"` 在不同响应里
   出现的次数 = 真正接过活的 worker 数（内核按四元组哈希分发，只打印当分布
   证据、不作断言 —— 哈希把全部连接派给同一个 worker 在内核侧是合法的）；
3. master 到点发 SIGTERM：空闲 worker 被唤醒连接叫醒、看到停止标记、以退出
   码 0 干净退出；master 的退出码 0 要求 abnormal=0 **且 forced=0** ——
   forced>0 意味着优雅路径没走通、靠 SIGKILL 兜的底，那不该无声通过；
4. master 能退 0 就说明所有子进程都被 waitpid 回收过（没有孤儿占着端口）。

真权重那一侧（fork 后 asyncrt 的取舍）由 `pixi run serve` 的手动冒烟看守，
证据在账本 —— 替身 `Stub` 不带权重，这一层与模型无关。
"""

from std.collections import List
from std.ffi import external_call
from std.testing import assert_equal, assert_true

from flare.net import SocketAddr
from flare.tcp import TcpStream

from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import bytes_of_text, bytes_to_text
from alofa.srv.master import run_workers
from alofa.srv.openai import NO_REQUEST, ChatHandler, Completion, Service
from alofa.srv.sse import StreamToken, sse_done

comptime PORT = UInt16(18124)
comptime WORKERS = 3
comptime READ_CHUNK = 4096
# master 到点自己优雅收工：客户端必须在这几秒里干完活。
comptime RUN_SECONDS = 6
comptime GRACE_MS = 8000
# 连接数：keep-alive 每条发「健康检查 + 聊天」两个请求。
comptime CONNS = 12

comptime HEALTH = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
comptime CHAT_BODY = (
    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}"
)
comptime STREAM_BODY = (
    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,"
    + "\"stream\":true}"
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
        if self.stream_step >= self.stream_max:
            return StreamToken("", True, False)
        var text = "s" + String(self.stream_step)
        self.stream_step += 1
        return StreamToken(text, False, False)


def master_child() raises:
    var handler = ChatHandler(Stub(), "stub")
    var report = run_workers(
        handler,
        SocketAddr.localhost(PORT),
        PORT,
        WORKERS,
        0,
        RUN_SECONDS,
        GRACE_MS,
    )
    # 退出码把两笔账都带上：非零 = abnormal 或 forced 有一个不为零。
    _ = external_call["_exit", Int32](Int32(report.abnormal * 10 + report.forced))


def split_responses(imm buf: List[UInt8]) -> List[String]:
    """按 `HTTP/1.1 ` 把收到的字节切成一条条响应（同 test_http_server）。"""
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
    var got = split_responses(buf)
    while len(got) < want:
        if read_more(conn, buf) == 0:
            break
        got = split_responses(buf)
    return got^


def drain(mut conn: TcpStream, mut buf: List[UInt8]) raises -> Bool:
    """一直读到 EOF。流式响应靠关连接定界，所以 EOF 就是它的结尾。"""
    while True:
        if read_more(conn, buf) == 0:
            return True


def count_frames(imm res: String) -> Int:
    """数一条流里有几帧（每帧以 `data: ` 起头）。

    不切成列表再比：这条门要的是**多 worker 下帧的数量与收尾和单进程一致**
    （逐字节的比对在 `test_http_server` 里），数清楚就够了 —— 少一帧意味着某个
    worker 的流被截断了，而这正是多进程这条路径特有的失败形态。
    """
    var n = 0
    var at = res.find("data: ")
    while at >= 0:
        n += 1
        at = res.find("data: ", at + 1)
    return n


def tail_of(imm text: String, start: Int) -> String:
    var raw = bytes_of_text(text)
    return bytes_to_text(raw, start, len(raw))


def assert_framed(imm res: String, imm connection: String) raises:
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


def send_text(mut conn: TcpStream, imm text: String) raises:
    var sent = conn.write(text.as_bytes())
    assert_equal(sent, text.byte_length(), "the write was short")


def post_request(imm body: String) -> String:
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
        master_child()
        # fork 出来的子进程不走父进程的退出路径（析构噪音）。
        _ = external_call["_exit", Int32](Int32(0))
        return

    # 等 3 个 worker 都把 listen 立起来。
    _ = external_call["usleep", Int32](Int32(800000))

    # 1) 12 条连接，每条上「健康检查 + 聊天 + 一条流」三个请求（keep-alive
    #    复用前两个，流以关连接收尾）。
    var served_chat = 0
    var streamed_count = 0
    for _ in range(CONNS):
        var conn = TcpStream.connect(SocketAddr.localhost(PORT))
        var buf = List[UInt8]()

        send_text(conn, HEALTH)
        var health = await_responses(conn, buf, 1)
        assert_true(len(health) >= 1, "no response to the health check")
        assert_true(
            health[0].find("HTTP/1.1 200 OK\r\n") == 0,
            "not a 200: " + health[0],
        )
        assert_true(health[0].find("\"status\":\"ok\"") >= 0, health[0])
        assert_framed(health[0], "keep-alive")

        send_text(conn, post_request(CHAT_BODY))
        var both = await_responses(conn, buf, 2)
        assert_true(len(both) >= 2, "no chat response on the same connection")
        var chat = both[1]
        assert_true(
            chat.find("HTTP/1.1 200 OK\r\n") == 0, "not a 200: " + chat
        )
        assert_true(chat.find("\"content\":\"S:hi\"") >= 0, chat)
        assert_true(chat.find("\"finish_reason\":\"length\"") >= 0, chat)
        assert_framed(chat, "keep-alive")

        # 1c) 同一条连接上再来一条流。多 worker 这条路径对流式是**新的**：
        #     每个 worker 是 fork 出来的独立进程，分帧必须与单进程一致 ——
        #     少一帧（尤其是少了 `[DONE]`）在工厂化的客户端上表现为"永远等
        #     着"，而不是报错。
        var stream_buf = List[UInt8]()
        send_text(conn, post_request(STREAM_BODY))
        _ = drain(conn, stream_buf)
        var streamed = bytes_to_text(stream_buf, 0, len(stream_buf))
        assert_true(
            streamed.find("HTTP/1.1 200 OK\r\n") == 0, "not a 200 stream: " + streamed
        )
        assert_true(
            streamed.find("Content-Type: text/event-stream") >= 0,
            "the stream did not announce SSE: " + streamed,
        )
        assert_true(
            streamed.find("Content-Length") < 0,
            "a stream must not claim a length: " + streamed,
        )
        assert_equal(count_frames(streamed), 6, "frames: " + streamed)
        # 收尾帧从 `sse_done()` 取，不在这里抄一份字面量：抄的话，改了帧格式
        # 而忘了改这条门，门就会继续"通过"。
        var done_frame = sse_done()
        assert_equal(
            tail_of(streamed, streamed.byte_length() - done_frame.byte_length()),
            done_frame,
        )
        streamed_count += 1

        # 2) 每个 worker 的计数器从 1 开始：第一次应答的 worker 贡献一个
        # chatcmpl-1。出现次数 = 接过活的 worker 数。
        if chat.find("\"id\":\"chatcmpl-1\"") >= 0:
            served_chat += 1
        conn.close()

    # 3) 等 master 按 RUN_SECONDS 收工并回收全部子进程（阻塞 waitpid）。
    var status = Int32(0)
    _ = external_call["wait4", Int32](
        pid, Pointer(to=status), Int32(0), Int32(0)
    )
    assert_true(
        (status & Int32(127)) == Int32(0),
        "the master itself was killed by a signal",
    )
    assert_equal(
        Int((status >> 8) & Int32(255)),
        0,
        "the master reported abnormal or forced workers (exit code = abnormal*10+forced)",
    )

    assert_equal(streamed_count, CONNS, "every connection must get a full stream")
    print(
        "PASS workers: " + String(WORKERS) + " processes on one port, "
        + String(CONNS * 3) + " responses, graceful stop (abnormal=0, forced=0)"
    )
    print(
        "  distribution evidence: " + String(served_chat)
        + " worker(s) answered at least one chat request"
    )
