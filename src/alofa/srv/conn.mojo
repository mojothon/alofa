"""一条连接的字节状态机：入站攒字节 → 切出一个完整请求 → 出站排队（带上限）。

为什么这块没有 socket
--------------------
`srv/server.mojo` 的循环是"读 → 解析 → 写"串在一起的：一次 `read` 拿不到完整请求就
**阻塞在那里读**。一条 worker 一条连接时那是对的 —— 阻塞只影响那一条连接。并到一条
事件循环上之后（roadmap 3.2），同一个阻塞会让**所有**连接一起等，于是"读不到完整请求"
必须变成"这条连接暂时没活干，去照顾别的"，而不是"停在这儿"。

所以这里把状态从 I/O 里剥出来：`Conn` 只有字节与状态，一次 `feed` 是"刚读到的字节"、
一次 `take` 是"能切出一个完整请求了吗"、一次 `advance` 是"刚才写出去了这么多"。它不看
fd、不调 syscall，因此能被 `tests/unit/test_conn.mojo` **逐字节**钉住 —— 而事件循环
（`srv/loop.mojo`）只剩"什么时候读、什么时候写"这一个问题。

这条边界上最容易写错的四件事，都由这层的行为直接决定：

1. **多出来的字节不能丢**：一次 `read` 可能带回两个请求（流水线）。第二个请求被吞掉
   的表现是"客户端偶尔收不到应答"，而不是报错。
2. **体不够就是不够**：`Content-Length` 说 100 字节而只到了 40，不能"先按 40 处理"。
3. **出站必须有上限**：写不出去的时候（慢客户端），生成侧如果继续往队列里追加，内存
   就按客户端的数量涨 —— 那正是"背压"这个词要防的东西。满了要**报错**，不是静默增长。
4. **写出多少只能由 syscall 说了算**：`advance(n)` 的 `n` 来自 `send(2)` 的返回值。
   自己假设"写就写完"会在对端慢的时候把字节发重或发漏。
"""

from std.collections import List

from alofa.core.error import ERR_CAPACITY, ERR_INVALID_ARGUMENT, ERR_PARSE, AlofaError
from alofa.srv.http import (
    CRLF,
    MAX_BODY_BYTES,
    MAX_HEADER_BYTES,
    HttpRequest,
    bytes_of_text,
    parse_head,
    parse_request,
    try_head_end,
)

# 入站缓冲的上限：头 + 体，与 `srv/server.mojo` 的 `MAX_TOTAL_BYTES` 同一个数。
# 一个永远发不完的请求必须有终点 —— 否则"再等一字节"就是无界内存。
comptime MAX_INBOUND_BYTES = MAX_HEADER_BYTES + MAX_BODY_BYTES

# 出站的**高水位**：一条连接上最多攒这么多没写出去的字节。到顶之后循环必须停止读这
# 条连接（也停止给它生成下一帧），直到队列降下来 —— 这是背压唯一的形状。1 MiB 够放
# 一个完整的 chat 响应（几 KB）再加上几十条 SSE 帧，又不会让一个慢客户端吃掉进程。
comptime SEND_HIGH_WATER = 1048576

# 一次 `send(2)` 最多交这么多字节。不是性能旋钮：一次把 1 MiB 塞给内核等于让一条连接
# 在循环里独占一次写，而循环是**所有**连接共用的。
comptime WRITE_CHUNK = 65536

# `take()` 的三种结果。用整数而不是 Bool，是因为"还没凑齐"与"对端关了"是**两件不同的
# 事**：前者要继续等字节，后者要收尾。合成一个 `Bool` 的话，其中一个会被当成另一个处理。
comptime TAKEN_NEED_MORE = 0
comptime TAKEN_REQUEST = 1
comptime TAKEN_CLOSED = 2


struct Taken(Movable):
    """一次 `take` 的结果：`state` 之外的字段只在 `TAKEN_REQUEST` 时有意义。

    为什么请求是**值**而不是引用：`inbound` 里的字节在返回之前就被消费掉了（多出来的
    留给下一个请求），留下一个指向缓冲区的视图会在下一次 `feed` 时失效。
    """

    var state: Int
    var request: HttpRequest

    def __init__(out self, state: Int):
        self.state = state
        self.request = HttpRequest()

    def __init__(out self, var request: HttpRequest):
        self.state = TAKEN_REQUEST
        self.request = request^


struct Conn(Movable):
    """一条连接上"已经收到但还没处理"与"要发但还没发出去"的字节。

    生命周期由事件循环拿着：`feed` 来自 `read`，`advance` 来自 `send`，`take` 在两者
    之间。它不持有 fd —— 那是为了让这层可以被不需要 socket 的门整层钉住。
    """

    var inbound: List[UInt8]
    var outbound: List[UInt8]
    # `outbound` 里已经写出去的前缀长度。留着它（而不是每次从头部删）是因为一次
    # `send` 通常写不完：删头部等于每写一截就搬一次剩下的字节。
    var sent: Int
    # 对端关了连接（`read` 返回 0）。这是 keep-alive 的正常结束方式，不是错误。
    var peer_closed: Bool
    # 队列写空之后就关掉这条连接（响应说了 `close`、或者这是一条流式响应 —— 流靠关
    # 连接定界）。
    var closing: Bool

    def __init__(out self):
        self.inbound = List[UInt8]()
        self.outbound = List[UInt8]()
        self.sent = 0
        self.peer_closed = False
        self.closing = False

    # ── 入站 ────────────────────────────────────────────────────────────────

    def feed(mut self, imm bytes: List[UInt8]) raises:
        """追加刚从 socket 上读到的字节。

        超过 `MAX_INBOUND_BYTES` 报 `capacity`：一个只发不结束的对端到这儿就该停了。
        """
        if len(self.inbound) + len(bytes) > MAX_INBOUND_BYTES:
            raise AlofaError(
                ERR_CAPACITY,
                "the request is larger than this server reads",
                "bytes=" + String(len(self.inbound) + len(bytes)),
            )
        for i in range(len(bytes)):
            self.inbound.append(bytes[i])

    def mark_peer_closed(mut self):
        """`read` 返回 0：对端不会再发字节了。

        这不是错误 —— keep-alive 的连接就是这么结束的。但"关了 + 缓冲里还有半个请求"
        是错误：那是一个发了一半的请求，静默丢掉它等于让客户端收不到任何应答还不知道
        为什么。
        """
        self.peer_closed = True

    def inbound_len(self) -> Int:
        """还没被切成请求的字节数。门用它钉住"多出来的字节没被吞掉"。"""
        return len(self.inbound)

    def take(mut self) raises -> Taken:
        """试着切出**一个**完整请求；凑不齐就报 `TAKEN_NEED_MORE`。

        与 `srv/server.mojo` 里那个同名函数唯一的差别是"不够的时候怎么办"：那里阻塞读，
        这里返回 —— 因为在这条循环上阻塞会连累所有连接。
        """
        var total = 0
        var state = self._prepare(total)
        if state != TAKEN_REQUEST:
            return Taken(state)
        var request = parse_request(self.inbound)
        self._consume(total)
        return Taken(request^)

    def take_with_raw(mut self, mut raw: List[UInt8]) raises -> Taken:
        """与 `take` 同一件事，另外把这一个请求的**原始字节**写进 `raw`。

        为什么原始字节是**出参**而不是 `Taken` 的一个字段：请求要过线程时
        （`srv/engine_thread.mojo`）带的正是这一块字节，而"从一个结构体中间把一块
        `List` 挪走"在这个编译器上过不去（`field destroyed out of the middle of a
        value`）。出参是调用方自己的变量，把它整个挪走没有歧义。

        `take()`（生成就在循环上那条路）不要这份拷贝：它自己拿 `request` 就够了 ——
        一次请求多拷几百字节在"每条连接都排队等一条循环"的那条路上不值当。
        """
        var total = 0
        var state = self._prepare(total)
        if state != TAKEN_REQUEST:
            return Taken(state)
        var request = parse_request(self.inbound)
        # 这一个请求的原始字节正好是前 `total` 个：多出来的是下一个请求，留给它。
        var k = 0
        while k < total:
            raw.append(self.inbound[k])
            k += 1
        self._consume(total)
        return Taken(request^)

    def _prepare(mut self, mut total: Int) raises -> Int:
        """这一个请求凑齐了吗 —— 凑齐了就把它的字节数写进 `total` 并返回
        `TAKEN_REQUEST`，否则返回 `TAKEN_NEED_MORE` / `TAKEN_CLOSED`。

        切请求这件事被拆成"定位"（这里）与"取走"（`take` / `take_with_raw`）两步，
        是因为取走有两种形状（只要 `request`，或者还要一份原始字节），而定位只有
        一种 —— 拆开之后两条路共用同一套"够不够 / 对端是不是关了"的判断，不会出现
        "其中一条漏了一个分支"。
        """
        var head = try_head_end(self.inbound)
        if head < 0:
            if self.peer_closed:
                if len(self.inbound) == 0:
                    return TAKEN_CLOSED
                raise AlofaError(
                    ERR_PARSE,
                    "the client closed before the request was complete",
                    "bytes=" + String(len(self.inbound)),
                )
            return TAKEN_NEED_MORE

        var partial = parse_head(self.inbound, head)
        var want = partial.content_length()
        if want > MAX_BODY_BYTES:
            raise AlofaError(
                ERR_CAPACITY,
                "the body is larger than this server reads",
                "bytes=" + String(want),
            )
        if len(self.inbound) - head < want:
            if self.peer_closed:
                raise AlofaError(
                    ERR_PARSE,
                    "the client closed before the body was complete",
                    "want=" + String(want) + " have=" + String(len(self.inbound) - head),
                )
            return TAKEN_NEED_MORE

        # `Expect: 100-continue`：curl 在体超过 1 KB 时会先等这个应答再发体。不发它，
        # 客户端会白等一秒 —— 那一秒看起来像"服务慢"，其实是协议没接上。它排在真正
        # 的响应**之前**，所以进的是同一个出站队列而不是另一条写路径。
        if partial.header("Expect") == "100-continue":
            self.enqueue(bytes_of_text("HTTP/1.1 100 Continue" + CRLF + CRLF))
        total = head + want
        return TAKEN_REQUEST

    def peek(self) -> Int:
        """缓冲里已经凑够一个完整请求了吗 —— **只是看，不吃**。

        为什么要有它：空闲超时必须能区分"排着队等生成"（不能因为生成慢就把它丢掉）
        与"发了一半就不发了"（那是要被收掉的）。这条线画错的方向很要紧 —— 前者画错
        会让一个慢一点的请求变成一次连接重置，后者画错会让几个半截请求把槽位占满。
        """
        var head: Int
        try:
            head = try_head_end(self.inbound)
        except:
            return TAKEN_NEED_MORE
        if head < 0:
            return TAKEN_NEED_MORE
        var partial: HttpRequest
        var want = 0
        try:
            partial = parse_head(self.inbound, head)
            want = partial.content_length()
        except:
            return TAKEN_NEED_MORE
        if want > MAX_BODY_BYTES:
            # 一个超大的请求不是"凑齐了"，但也不是"还要字节" —— 交给 `take` 去报错，
            # 这里只回答"能不能现在就处理"这个问题。
            return TAKEN_NEED_MORE
        if len(self.inbound) - head < want:
            return TAKEN_NEED_MORE
        return TAKEN_REQUEST

    def _consume(mut self, n: Int):
        """丢掉 `inbound` 的前 `n` 个字节（那是一个请求），剩下的留给下一个。"""
        var rest = len(self.inbound) - n
        if rest <= 0:
            self.inbound.clear()
            return
        var out = List[UInt8](capacity=rest)
        var i = 0
        while i < rest:
            out.append(self.inbound[n + i])
            i += 1
        self.inbound = out^

    # ── 出站 ────────────────────────────────────────────────────────────────

    def enqueue(mut self, imm bytes: List[UInt8]) raises:
        """把一段要发的字节排到队尾。

        队列满了报 `capacity` —— **不是**静默扩容。背压的全部意义就是让"生产得快、
        消费得慢"在某一个地方变成一次可见的拒绝；无限队列会把慢客户端的代价从"这一条
        连接变慢"变成"进程的内存涨上去"。
        """
        var pending = self.pending_len()
        if pending + len(bytes) > SEND_HIGH_WATER:
            raise AlofaError(
                ERR_CAPACITY,
                "the send queue for this connection is full",
                "pending=" + String(pending) + " more=" + String(len(bytes)),
            )
        if pending == 0 and self.sent > 0:
            # 上一批已经写完：把缓冲归零，免得一个 1 MiB 的缓冲跟着连接活一整天。
            self.outbound.clear()
            self.sent = 0
        for i in range(len(bytes)):
            self.outbound.append(bytes[i])

    def pending_len(self) -> Int:
        """还没写出去的字节数。"""
        return len(self.outbound) - self.sent

    def pending_view(mut self, n: Int) raises -> Span[UInt8, origin_of(self.outbound)]:
        """还没写出去的前 `n` 个字节 —— 直接给 `send(2)` 的一段视图（不拷贝）。

        `n` 由调用方给（一次 `send` 交多少是循环的事），但不能超过队列里有的：越界
        的下标会把别人的内存发出去，而那是一次"偶发的响应内容不对"。
        """
        if n < 0 or n > self.pending_len():
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "cannot send more than the queue holds",
                "n=" + String(n) + " pending=" + String(self.pending_len()),
            )
        return Span[UInt8, origin_of(self.outbound)](
            unsafe_ptr=self.outbound.unsafe_ptr().unsafe_offset(self.sent), length=n
        )

    def advance(mut self, n: Int) raises:
        """`send(2)` 报告写出去了 `n` 个字节。

        `n` 只能来自 syscall 的返回值：自己假设"写就写完了"会在对端慢的时候把字节发重
        （重发已写的那截）或发漏（跳过没写的那截），两种都是对端收到一段拼不起来的
        响应 —— 而那种错看起来像"偶发的解析失败"。
        """
        var pending = self.pending_len()
        if n < 0 or n > pending:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "cannot advance past what is queued",
                "n=" + String(n) + " pending=" + String(pending),
            )
        self.sent += n
        if self.sent == len(self.outbound):
            self.outbound.clear()
            self.sent = 0

    def drained(self) -> Bool:
        """队列空了吗 —— `closing` 的连接要等它为真才可以关。"""
        return self.pending_len() == 0

    def backpressure(self) -> Bool:
        """这条连接该停止读（也停止生成下一帧）了吗。

        判据是**队列满了**，不是"还有字节没写" —— 后者是常态（写完之前总有没写的），
        按它停读等于把并发退回成轮流等待。
        """
        return self.pending_len() >= SEND_HIGH_WATER

    # ── 收尾 ────────────────────────────────────────────────────────────────

    def close_when_drained(mut self):
        """队列写空之后关掉这条连接（响应说了 `close`，或者这是一条流）。

        不立刻关：立刻关会丢掉还没写出去的字节，而那些字节正是客户端在等的应答。
        """
        self.closing = True

    def should_close(self) -> Bool:
        """现在可以关了吗 —— 要关，且该发的都发了。"""
        return self.closing and self.drained()
