"""一条连接上阻塞 accept、一次一个请求：HTTP 服务循环的最小形态。

为什么先做这个形态
------------------
P3 要的是"能对外服务"，但"能对外服务"最早的一环不是并发，而是**这条链路通不通**：
一个请求进来 → 分词 → 前向 → 采样 → JSON → 字节写回 socket。并发、背压、SSE 都是
在链路通了之后才有意义的东西；先在单进程阻塞形态上把它们一起加进来，等于同时调试四件
事，而且任何一件坏了都表现为"卡住了"。

这个形态的代价是明文写在这儿的：**一个 worker 同时只处理一条连接**。这不是没考虑
到的疏漏，而是这一版的取舍 —— 多 worker（`srv/master.mojo`，P3.4）修的就是并发：
N 个进程各跑一份这个循环，内核按四元组把连接分给它们。

SSE 在这个形态上**是成立的**：流式只需要"一条连接上按顺序写"，不需要并发。而且
阻塞写本身就是背压 —— 写不出去时生成就停在那儿，不会攒出一个无界的发送队列。
仍然不在其中的是**并发连接**（那是 reactor 化，roadmap 3.2 的事）。

传输层是 `flare.tcp`
--------------------
socket / bind / listen / accept / TCP_NODELAY 全是 flare 的阻塞接口（A7：依赖而
不自研）。我们只在这之上做两件事：把字节攒成一个完整的请求（`srv/http.mojo` 决定
"完整"是什么意思），以及把响应写回去。

三个护栏（都是因为单进程才会这么致命）
--------------------------------------
1. **接收超时**（`RECV_TIMEOUT_MS`）：一个连上又不发数据的对端，在单进程里等于让整个
   服务停摆。超时之后关掉连接，下一个请求还有机会。
2. **头/体的上限**（`MAX_HEADER_BYTES` / `MAX_BODY_BYTES`）：同上，一个永远发不完的
   请求必须有终点。
3. **流水线剩余字节不丢**（`pending`）：一次 `read` 可能带回两个请求。把多出来的字节
   吞掉会让第二个请求凭空消失；留到下一轮，才谈得上"一条连接上按顺序处理"。
"""

from std.collections import List

from flare.net import SocketAddr
from flare.tcp import TcpListener, TcpStream

from alofa.core.error import ERR_CAPACITY, ERR_PARSE, AlofaError
from alofa.core.ffi.posix import fd_is_open
from alofa.srv.http import (
    CRLF,
    MAX_BODY_BYTES,
    MAX_HEADER_BYTES,
    SSE_CONTENT_TYPE,
    HttpRequest,
    HttpResponse,
    bytes_of_text,
    json_response,
    parse_head,
    parse_request,
    serialize,
    serialize_stream_head,
    try_head_end,
)

comptime RECV_TIMEOUT_MS = 5000
comptime CHUNK = 4096
comptime MAX_TOTAL_BYTES = MAX_HEADER_BYTES + MAX_BODY_BYTES

# 串行那条路（`serve_with_stop`）给 handler 的 request 号。它没有槽位概念 ——
# 一次只握着一条连接，accept 下一条之前这条已经走完了 —— 所以恒用一个号。
# 写成常量而不是就地写 0：这个值是要被读代码的人**核对**的（"为什么是 0"），
# 而不是一个随手填的坑位。
comptime SOLO_REQUEST = 0


trait Handler:
    """一个请求进、一个响应出；流式则是一次 `handle` 加若干次 `stream_next`。

    服务循环不知道路由，路由不知道 socket：流式也是按这条线切的 —— handler 交
    出的是**帧的文本**（`srv/sse.mojo` 的纯函数拼的），"往 socket 上写"仍然只
    有这里做。这样流式那一半能被不需要权重的门逐字节钉住。

    `request` 参数：**这条连接这一次的请求号**，由调用方（服务循环）给出，同一条
    连接上 `handle` → `stream_next` × N → `stream_end` 必须给同一个值。Reactor 那
    个循环给的是**连接槽位**（`[0, MAX_CONNS)`），所以两条连接天然不同号 —— 这是
    "一条 handler 同时给多条连接生成"能成立的前提（在这之前，handler 只能靠"当前
    只有一条流"这个**未写进类型**的假设活着，见 `ChatHandler` 里那句 ⚠️）。
    串行的 `serve_with_stop` 一次只有一条连接，没有槽位概念，它传
    `SOLO_REQUEST`（= 0）。
    """

    def handle(mut self, request: Int, req: HttpRequest) raises -> HttpResponse:
        ...

    def stream_next(mut self, request: Int) raises -> String:
        """流的下一帧；空串 = 流结束。

        只在 `handle` 返回一个 `text/event-stream` 的响应之后被调用。返回空串
        是**唯一**的正常结束方式 —— 少了它这条流就没有终点。
        """
        ...

    def stream_end(mut self, request: Int) raises:
        """一条流**不是**靠走完而结束（对端断了 / 连接被丢 / 服务在收尾）。

        正常走完的那条路由 `stream_next` 自己收尾；这一条是给"没走完"的那条一个
        回收的机会，所以**必须幂等**：同一个 `request` 调两次（第二次可能已经没有
        这条流了）不能报错。

        为什么要有它：不带它，"结束"就只有一条路（走完），于是每一次提前收尾都在
        漏 —— 单条状态看不出漏（下一条 `begin` 会覆盖），按 request 索引之后漏的
        是**表里的一个位置**，漏到一定条数就变成"新请求被拒"。
        """
        ...


struct TakenRequest(Movable):
    """从一条连接上取到的东西。

    `complete=False` 表示**对端关了连接**且没有留下半个请求 —— 那是 keep-alive 的正常
    结束方式，不是错误；错误用 `raises` 表示（`srv/http.mojo` 里那几种）。
    """

    var request: HttpRequest
    var complete: Bool

    def __init__(out self):
        self.request = HttpRequest()
        self.complete = False


struct Server(Movable):
    """一个 bound + listening 的 socket，加上"一次处理一个连接"的循环。"""

    var listener: TcpListener
    var port: UInt16

    def __init__(out self, var listener: TcpListener, port: UInt16):
        self.listener = listener^
        self.port = port

    @staticmethod
    def bind(port: UInt16) raises -> Server:
        """绑到 127.0.0.1 —— 这一版不对外暴露（`SO_REUSEADDR` 由 flare 设置）。"""
        var listener = TcpListener.bind(SocketAddr.localhost(port))
        return Server(listener^, port)

    def serve[H: Handler](mut self, mut handler: H, max_requests: Int) raises -> Int:
        """接受连接并回答请求，直到答完 `max_requests` 个（`max_requests <= 0`
        表示一直跑）。返回实际回答的条数 —— 门靠这个数停下一个本来不会停的循环。

        ⚠️ 单进程、串行：一条连接上的所有请求按顺序处理，处理完才 accept 下一条。
        """
        return self.serve_with_stop(handler, max_requests, Int32(-1))

    def serve_with_stop[H: Handler](
        mut self, mut handler: H, max_requests: Int, stop_fd: Int32
    ) raises -> Int:
        """同 `serve`，外加一个停止标记：`stop_fd` 指着的描述符一旦被关掉，
        循环就退出 —— 答完「当前连接上正在处理的那条」，不再开始新的。

        标记的来历见 `master.mojo` 的文件头（一个被信号处理器 close 掉的高位
        fd）。检查点两处：accept 之前（不在 accept 前查，标记就永远没人看 ——
        accept 会一直阻塞）；每个响应写完之后（在途的答完，下一条不再开始）。
        `stop_fd < 0` 表示没有标记，行为与老 `serve` 一致。
        """
        var served = 0
        while max_requests <= 0 or served < max_requests:
            if stop_requested(stop_fd):
                break
            var conn = self.listener.accept()
            conn.set_recv_timeout(RECV_TIMEOUT_MS)
            var pending = List[UInt8]()
            var open = True
            while open and (max_requests <= 0 or served < max_requests):
                var taken: TakenRequest
                try:
                    taken = take_request(pending, conn)
                except err:
                    # 一个坏对端只能带走它自己那条连接。让异常冒出 `serve` 意味着
                    # "一个畸形请求 = 整个进程退出"，而单进程服务的全部意义就是
                    # 这一条循环不能停。
                    print("  [srv] connection dropped: " + String(err))
                    open = False
                    continue
                if not taken.complete:
                    open = False
                    continue
                var response: HttpResponse
                try:
                    response = handler.handle(SOLO_REQUEST, taken.request)
                except err:
                    # 500 且不带细节进响应：内部原因属于日志，而这一版还没有日志
                    # 分级，所以只在这里打印。消息本身仍然要打出来 —— 一个没有原因的
                    # 500 只能靠猜。
                    print("  [srv] request failed: " + String(err))
                    response = json_response(
                        500,
                        "{\"error\":{\"message\":\"the request could not be"
                        + " answered\",\"type\":\"server_error\",\"code\":"
                        + "\"internal\"}}",
                        True,
                    )
                # 对端说要关，就必须**在响应里也说要关**：只在写完以后默默关掉，对端
                # 会以为这条连接还能用，于是下一个请求发到一个正在被关的 socket 上 ——
                # 那是一次看起来像"偶发连接重置"的失败。
                if not taken.request.keep_alive():
                    response.close = True
                if response.content_type == SSE_CONTENT_TYPE:
                    write_stream(conn, handler, response.status, stop_fd, SOLO_REQUEST)
                else:
                    write_response(conn, response)
                served += 1
                if response.close or stop_requested(stop_fd):
                    open = False
            conn.close()
        return served

    def close(mut self):
        self.listener.close()


def stop_requested(stop_fd: Int32) -> Bool:
    """停止标记被关掉了吗？`stop_fd < 0` 表示没有标记（老行为，永远 False）。"""
    if stop_fd < 0:
        return False
    return not fd_is_open(stop_fd)


def read_chunk(mut conn: TcpStream, mut buf: List[UInt8]) raises -> Int:
    """往 `buf` 后面追加一次 `read` 的字节；返回读到的个数（0 = 对端关了）。"""
    var base = len(buf)
    if base + CHUNK > MAX_TOTAL_BYTES:
        raise AlofaError(
            ERR_CAPACITY,
            "the request is larger than this server reads",
            "bytes=" + String(base + CHUNK),
        )
    buf.resize(base + CHUNK, 0)
    var got = conn.read(buf.unsafe_ptr().unsafe_offset(base), CHUNK)
    buf.resize(base + got, 0)
    return got


def take_request(
    mut pending: List[UInt8], mut conn: TcpStream
) raises -> TakenRequest:
    """从连接上取**一个**完整请求，多出来的字节留在 `pending` 里给下一个请求。"""
    var head = try_head_end(pending)
    while head < 0:
        var got = read_chunk(conn, pending)
        if got == 0:
            var nothing = TakenRequest()
            return nothing^
        head = try_head_end(pending)

    var partial = parse_head(pending, head)
    var want = partial.content_length()
    if want > MAX_BODY_BYTES:
        raise AlofaError(
            ERR_CAPACITY,
            "the body is larger than this server reads",
            "bytes=" + String(want),
        )
    # `Expect: 100-continue`：curl 在体超过 1 KB 时会先等这个应答。不发它会让
    # 客户端白等一秒 —— 那一秒看起来像"服务慢"，其实是协议没接上。
    if partial.header("Expect") == "100-continue":
        conn.write_all(bytes_of_text("HTTP/1.1 100 Continue" + CRLF + CRLF))
    while len(pending) - head < want:
        var got = read_chunk(conn, pending)
        if got == 0:
            raise AlofaError(
                ERR_PARSE,
                "the client closed before the body was complete",
                "want=" + String(want) + " have=" + String(len(pending) - head),
            )

    var request = parse_request(pending)
    var total = head + want
    var rest = List[UInt8](capacity=len(pending) - total)
    var i = total
    while i < len(pending):
        rest.append(pending[i])
        i += 1
    pending = rest^

    var taken = TakenRequest()
    taken.request = request^
    taken.complete = True
    return taken^


def write_response(mut conn: TcpStream, imm res: HttpResponse) raises:
    conn.write_all(serialize(res).as_bytes())


def write_stream[H: Handler](
    mut conn: TcpStream,
    mut handler: H,
    status: Int,
    stop_fd: Int32,
    request: Int,
) raises:
    """写一条 SSE 流：先头，再一帧一帧，**最后关连接** —— 关连接就是这条响应
    的定界（`srv/http.mojo` 的 `serialize_stream_head` 没有 `Content-Length`）。

    三条提前结束的路，每一条都必须走到：

    1. **停止标记**（优雅退出）：帧之间查，于是"在途这条"能被走完，但下一个
       请求不再开始 —— 与 `serve_with_stop` 的非流式路径同一条语义。
    2. **写失败**（客户端中途断了）：只丢这一条连接，进程继续。`return` 而不是
       让异常冒出去：一个断开的客户端不该等于"服务挂了"。
    3. **取帧出错**（前向炸了）：同上，丢这条连接。handler 那边已经把错误变成
       了一帧（`srv/openai.mojo`），所以这里只负责不再往下问。
    """
    conn.write_all(serialize_stream_head(status).as_bytes())
    while True:
        if stop_requested(stop_fd):
            break
        var frame: String
        try:
            frame = handler.stream_next(request)
        except err:
            print("  [srv] stream failed: " + String(err))
            break
        if frame.byte_length() == 0:
            break
        try:
            conn.write_all(frame.as_bytes())
        except err:
            print("  [srv] stream aborted: " + String(err))
            break
    # 有借有还：无论从哪条路出的循环（走完 / 停止标记 / 对端断了 / 取帧出错），
    # 这条流都要给 handler 一次回收的机会。`stream_end` 幂等 —— 正常走完的那条
    # 在 `stream_next` 里已经收过一次了。
    try:
        handler.stream_end(request)
    except err:
        print("  [srv] stream could not be released: " + String(err))
