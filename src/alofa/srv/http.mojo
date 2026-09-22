"""HTTP/1.1 的请求解析与响应组帧：字节进、字节出，中间没有任何 I/O。

为什么这块是 alofa 写的，而不是 `flare.http`
---------------------------------------------
`flare` 是**依赖而不是自研**（A7），而且它的 `HttpServer` 确实能跑 —— 但它绑在
自己的 reactor 上：一次 `serve()` 按 `num_workers` 拉起 reactor 线程，连接状态机、
超时、背压都在那套循环里。而 P3 的第一步要的是**单进程、一次只处理一个连接、阻塞**
（`srv/server.mojo` 的文件头写了这个形态为什么本身就是第一步）。一个只有一个执行
线程的进程，不该为了收一个请求先引入一个事件循环 —— 事件循环解决的是"很多连接同时在
等"，而这一步的连接数是 1。

所以这一版的分工是：**传输层用 flare**（`flare.tcp` 的阻塞 `TcpListener` /
`TcpStream`：socket / bind / listen / accept / TCP_NODELAY 全是它的），**线上格式
是我们的**：请求行 + 头 + `Content-Length` 体的解析，以及响应组帧（带
`Content-Length` 的整段响应，与不带长度的流式响应头）。这块是纯函数（字节进、
结构出），所以能被**逐字节**的门钉住 —— 见 `tests/unit/test_http.mojo` 与
`tests/unit/test_sse.mojo`。

等 P3.3 要的是并发连接与背压时，正确的动作是换成 `flare.http`，而不是把这里长大。
届时要删的是这两个文件，不是要改的是它们。

严格到什么程度
--------------
- 请求行必须正好是三段（`GET /path HTTP/1.1`），多一个空格就是错 —— 宽松的解析器
  会把"多了一个空格"变成"路径里有一个空格"，而那类 bug 只在有人发畸形请求时才现形。
- 头字段的结束只认 `CRLF CRLF`。只认 LF 的解析器会把一个含裸 LF 的请求当成两个请求，
  那是请求走私的形状。
- `Transfer-Encoding: chunked` 指名拒绝（**请求**方向）：一次把整段读完再解析，
  长度先知道，才谈得上"整段做完再回"。响应方向另有一条**没有** `Content-Length`
  的路 —— 流式（见 `serialize_stream_head`），那是"长度未知"，不是"长度分块"。
"""

from std.collections import List

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_PARSE,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.text import parse_int

comptime CRLF = "\r\n"

# SSE 的响应类型。`srv/server.mojo` 靠它认出"这个响应是要逐帧写的"，所以它是
# 一个**标记**而不只是一个头的值：流式响应没有 `Content-Length`（长度未知），
# 走的是另一条写路径 —— 见 `serialize_stream_head`。
comptime SSE_CONTENT_TYPE = "text/event-stream"

# 一次请求的头最多读这么多字节。这不是性能旋钮，是"单进程"这个形态的护栏：一个
# 永远不发空行的对端会把整个进程占住，而单进程占住就等于所有请求都停了。
comptime MAX_HEADER_BYTES = 8192
# 一次请求的体最多读这么多字节（1 MiB）。一个 chat 请求只需要几 KB。
comptime MAX_BODY_BYTES = 1048576


def bytes_to_text(imm raw: List[UInt8], start: Int, stop: Int) -> String:
    """把 `raw[start:stop]` 当作 UTF-8 取出来。

    HTTP 的头与 JSON 的体在这个服务里都是 UTF-8；越界下标在调用点就被挡住了，这里是
    纯拷贝。
    """
    var out = List[UInt8]()
    var i = start
    while i < stop:
        out.append(raw[i])
        i += 1
    return String(unsafe_from_utf8=out)


def lower_ascii(imm text: String) -> String:
    """只折叠 A–Z。HTTP 的头名与几个取值是大小写不敏感的，而路径与体不是。"""
    var raw = text.as_bytes()
    var out = List[UInt8]()
    for i in range(len(raw)):
        var b = raw[i]
        if b >= 65 and b <= 90:
            out.append(b + 32)
        else:
            out.append(b)
    return String(unsafe_from_utf8=out)


def equals_ignore_case(imm a: String, imm b: String) -> Bool:
    """比较两个 ASCII 串，忽略大小写。长度不同就不需要逐字节了。"""
    if a.byte_length() != b.byte_length():
        return False
    var x = a.as_bytes()
    var y = b.as_bytes()
    for i in range(len(x)):
        var p = x[i]
        var q = y[i]
        if p >= 65 and p <= 90:
            p += 32
        if q >= 65 and q <= 90:
            q += 32
        if p != q:
            return False
    return True


def trim_ows(imm text: String) -> String:
    """去掉两端的空格与水平制表符（RFC 7230 的 OWS）。

    头值两边的空格是允许的，而 `Content-Length : 12` 里的空格如果不去掉，
    `parse_int` 会把它判成"不是数字" —— 那是一条正确的解析规则用在了没清干净的输入上。
    """
    var raw = text.as_bytes()
    var start = 0
    var stop = len(raw)
    while start < stop and (raw[start] == 32 or raw[start] == 9):
        start += 1
    while stop > start and (raw[stop - 1] == 32 or raw[stop - 1] == 9):
        stop -= 1
    var out = List[UInt8](capacity=stop - start)
    var i = start
    while i < stop:
        out.append(raw[i])
        i += 1
    return String(unsafe_from_utf8=out)


def bytes_of_text(imm text: String) -> List[UInt8]:
    """把 `String` 拷成 `List[UInt8]`。

    这一层的解析函数收 `List[UInt8]`（`tokenizer_json` 里也是这个约定），而
    `as_bytes()` 给的是不带所有权的 `Span`，拷一份才能跨函数传。
    """
    var raw = text.as_bytes()
    var out = List[UInt8](capacity=len(raw))
    for i in range(len(raw)):
        out.append(raw[i])
    return out^


struct Header(Copyable, Movable):
    var name: String
    var value: String

    def __init__(out self, name: String, value: String):
        self.name = name
        self.value = value


struct HttpRequest(Movable):
    """一个已经完整收到的请求。

    `body` 只在 `parse_request` 里被填上：`Content-Length` 说有多少字节，就只有那么多
    字节属于这个请求 —— 多出来的属于**下一个**请求（流水线），由 `srv/server.mojo`
    的 `pending` 缓冲接着，不能在这里吞掉。
    """

    var method: String
    var target: String
    var version: String
    var headers: List[Header]
    var body: String

    def __init__(out self):
        self.method = ""
        self.target = ""
        self.version = ""
        self.headers = List[Header]()
        self.body = ""

    def header(self, name: String) -> String:
        """第一个同名头的值，没有就返回空串。

        头名大小写不敏感（`Content-Length` 与 `content-length` 是同一个头），而取值
        原样返回 —— 大小写只影响查找，不影响内容。
        """
        for i in range(len(self.headers)):
            if equals_ignore_case(self.headers[i].name, name):
                return self.headers[i].value
        return ""

    def content_length(self) raises -> Int:
        """声明的体长；缺头就是 0（`GET` 通常没有体）。"""
        var raw = self.header("Content-Length")
        if raw.byte_length() == 0:
            return 0
        return parse_int(trim_ows(raw))

    def path(self) -> String:
        """target 去掉 `?` 与 `#` 之后的部分 —— 路由只认路径。"""
        var raw = self.target.as_bytes()
        var stop = len(raw)
        var i = 0
        while i < len(raw):
            if raw[i] == 63 or raw[i] == 35:  # '?' '#'
                stop = i
                break
            i += 1
        var out = List[UInt8](capacity=stop)
        var k = 0
        while k < stop:
            out.append(raw[k])
            k += 1
        return String(unsafe_from_utf8=out)

    def keep_alive(self) -> Bool:
        """这个请求之后连接还留着吗。

        HTTP/1.1 默认保持，HTTP/1.0 默认关闭，`Connection` 头可以反过来。`Connection`
        是一个逗号分隔的列表，所以两边都按"出现即算"处理：含 `close` 就关，否则含
        `keep-alive` 就留，都没有就回到版本的默认值。
        """
        var raw = lower_ascii(self.header("Connection"))
        if raw.find("close") >= 0:
            return False
        if raw.find("keep-alive") >= 0:
            return True
        return self.version == "HTTP/1.1"


def try_head_end(imm raw: List[UInt8]) raises -> Int:
    """头结束（`CRLF CRLF`）之后的那个下标；还没收到完整头则返回 -1。

    只认 `CRLF CRLF`：把裸 LF 也当成结束，一个夹带裸 LF 的请求就能被切成两个 —— 那是
    请求走私的形状，而单进程服务里"两个请求"意味着第二个请求用的是别人的权限。

    超过 `MAX_HEADER_BYTES` 直接报 `capacity`：对端不发结束标记时，这是唯一能把这个
    进程从"等下去"里救出来的东西。
    """
    var limit = len(raw) - 3
    var i = 0
    while i < limit:
        if raw[i] == 13 and raw[i + 1] == 10 and raw[i + 2] == 13 and raw[i + 3] == 10:
            return i + 4
        i += 1
    if len(raw) > MAX_HEADER_BYTES:
        raise AlofaError(
            ERR_CAPACITY,
            "the request headers are larger than this server reads",
            "bytes=" + String(len(raw)),
        )
    return -1


def parse_head(imm raw: List[UInt8], end: Int) raises -> HttpRequest:
    """解析 `[0, end)` 里的请求行与头，不碰体。

    分两步（`parse_head` + 体）而不是一次 `parse_request`，是因为**体长只有头说了才算**：
    服务循环必须先知道 `Content-Length` 才知道还要从 socket 上读多少字节。
    """
    var request = HttpRequest()

    var line_stop = 0
    while line_stop + 1 < end and not (
        raw[line_stop] == 13 and raw[line_stop + 1] == 10
    ):
        line_stop += 1
    if line_stop + 1 >= end:
        raise AlofaError(ERR_PARSE, "the request line is not terminated", "")

    # 请求行：正好三段，按单个空格切。多了少了都是错 —— 见文件头"严格到什么程度"。
    var parts = List[String]()
    var start = 0
    var i = 0
    while i <= line_stop:
        if i == line_stop or raw[i] == 32:
            if i > start:
                parts.append(bytes_to_text(raw, start, i))
            start = i + 1
        i += 1
    if len(parts) != 3:
        raise AlofaError(
            ERR_PARSE,
            "the request line does not have three fields",
            "line=" + bytes_to_text(raw, 0, line_stop),
        )
    request.method = parts[0]
    request.target = parts[1]
    request.version = parts[2]
    if request.method.byte_length() == 0 or request.target.byte_length() == 0:
        raise AlofaError(ERR_PARSE, "the request line has an empty field", "")
    if request.version != "HTTP/1.1" and request.version != "HTTP/1.0":
        raise AlofaError(
            ERR_UNSUPPORTED,
            "this server speaks HTTP/1.0 and HTTP/1.1 only",
            "version=" + request.version,
        )

    var at = line_stop + 2
    while at + 1 < end:
        var stop = at
        while stop + 1 < end and not (raw[stop] == 13 and raw[stop + 1] == 10):
            stop += 1
        if stop + 1 >= end:
            raise AlofaError(ERR_PARSE, "a header line is not terminated", "")
        if stop == at:
            # 空行：头的结束（正常不会走到这里，`end` 已经跳过了它）。
            break
        var colon = at
        while colon < stop and raw[colon] != 58:  # ':'
            colon += 1
        if colon >= stop:
            raise AlofaError(
                ERR_PARSE,
                "a header line has no colon",
                "line=" + bytes_to_text(raw, at, stop),
            )
        var name = bytes_to_text(raw, at, colon)
        # 头名里不允许空格（`Host : x`）：RFC 7230 明说这是必须拒绝的，而放过它会让
        # 两个解析器对"这个头叫什么"给出两个答案。
        var name_bytes = name.as_bytes()
        for k in range(len(name_bytes)):
            if name_bytes[k] == 32 or name_bytes[k] == 9:
                raise AlofaError(
                    ERR_PARSE,
                    "whitespace before the colon in a header name",
                    "name=" + name,
                )
        request.headers.append(
            Header(name, trim_ows(bytes_to_text(raw, colon + 1, stop)))
        )
        at = stop + 2

    return request^


def parse_request(imm raw: List[UInt8]) raises -> HttpRequest:
    """解析一个完整的请求：头 + `Content-Length` 个字节的体。"""
    var end = try_head_end(raw)
    if end < 0:
        raise AlofaError(ERR_PARSE, "the request headers are not terminated", "")

    var request = parse_head(raw, end)

    var transfer = lower_ascii(request.header("Transfer-Encoding"))
    if transfer.byte_length() > 0:
        # 见文件头：这一版只做非流式，长度先知道才谈得上整段做完再回。
        raise AlofaError(
            ERR_UNSUPPORTED,
            "this server does not accept chunked bodies",
            "transfer-encoding=" + transfer,
        )

    var want = request.content_length()
    if want < 0:
        raise AlofaError(
            ERR_PARSE, "Content-Length is negative", "value=" + String(want)
        )
    if want > MAX_BODY_BYTES:
        raise AlofaError(
            ERR_CAPACITY,
            "the body is larger than this server reads",
            "bytes=" + String(want),
        )
    if len(raw) - end < want:
        raise AlofaError(
            ERR_PARSE,
            "the body is shorter than Content-Length",
            "want=" + String(want) + " have=" + String(len(raw) - end),
        )
    request.body = bytes_to_text(raw, end, end + want)
    return request^


struct HttpResponse(Copyable, Movable):
    """一个待发送的响应。`Content-Length` 不在这里 —— 它由体的字节数算出来。

    手填长度会发生长度与体不一致，而那是 HTTP 里最难受的一类错：对端按长度切，切出来
    的字节既不像一个响应也不像一个错误。所以长度只有一个来源，就是体本身。
    """

    var status: Int
    var body: String
    var content_type: String
    var close: Bool

    def __init__(out self, status: Int, body: String, content_type: String, close: Bool):
        self.status = status
        self.body = body
        self.content_type = content_type
        self.close = False
        if close:
            self.close = True


def reason_phrase(status: Int) raises -> String:
    """状态码的原因短语。没有名字的状态码是错 —— 说明有人发了一个没想过的状态。"""
    if status == 200:
        return "OK"
    if status == 400:
        return "Bad Request"
    if status == 404:
        return "Not Found"
    if status == 405:
        return "Method Not Allowed"
    if status == 408:
        return "Request Timeout"
    if status == 413:
        return "Payload Too Large"
    if status == 415:
        return "Unsupported Media Type"
    if status == 500:
        return "Internal Server Error"
    raise AlofaError(
        ERR_UNSUPPORTED, "no reason phrase for this status", "status=" + String(status)
    )


def json_response(status: Int, body: String, close: Bool) -> HttpResponse:
    return HttpResponse(status, body, "application/json", close)


def text_response(status: Int, body: String, close: Bool) -> HttpResponse:
    return HttpResponse(status, body, "text/plain; charset=utf-8", close)


def serialize(imm res: HttpResponse) raises -> String:
    """把一个响应写成线上的字节。

    头只有四个：状态行、`Content-Type`、`Content-Length`、`Connection`。没有 `Date`
    是故意的 —— 一是 stdlib 里没有墙上时钟，二是带上它这条"逐字节相等"的门就变成
    只能比结构了，而结构相等放过了长度算错。
    """
    var out = (
        "HTTP/1.1 "
        + String(res.status)
        + " "
        + reason_phrase(res.status)
        + CRLF
    )
    out += "Content-Type: " + res.content_type + CRLF
    out += "Content-Length: " + String(res.body.byte_length()) + CRLF
    if res.close:
        out += "Connection: close" + CRLF
    else:
        out += "Connection: keep-alive" + CRLF
    out += CRLF
    out += res.body
    return out


def serialize_stream_head(status: Int) raises -> String:
    """一个流式响应的**头**：没有 `Content-Length`，靠关连接定界。

    为什么没有长度：写第一帧的时候还不知道一共会有多少字节，而"先算出来再
    写"等于把流式退回非流式（先整段生成完）。为什么不用 chunked：见
    `srv/sse.mojo` 的文件头 —— SSE 客户端按 `data:` 行切分，chunked 的长度
    前缀是多余的一层。

    `Cache-Control: no-cache` 是 SSE 的惯例（代理不得缓冲，否则"边生成边推"
    变成"生成完了才一起到"）。`Connection: close` 不是建议而是这条响应唯一
    的定界方式 —— 请求头说 keep-alive 也不改（`srv/server.mojo` 在流之后
    无条件关连接）。
    """
    return (
        "HTTP/1.1 "
        + String(status)
        + " "
        + reason_phrase(status)
        + CRLF
        + "Content-Type: "
        + SSE_CONTENT_TYPE
        + CRLF
        + "Cache-Control: no-cache"
        + CRLF
        + "Connection: close"
        + CRLF
        + CRLF
    )
