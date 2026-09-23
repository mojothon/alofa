"""连接机的门：入站怎么攒、请求怎么切、出站怎么排队。

`srv/conn.mojo` 是**没有 socket** 的一层，所以这里能把它整层钉住 —— 事件循环里那些
最难复现的错（一次 read 带回两个请求、慢客户端把发送队列撑爆、写出一半）在字节层面都
是确定的，而在"跑一个真服务"层面全是偶发。

每条门都带一个**会失败的写法**在旁边：门要钉的是"只有正确写法能通过"，不是"能跑通"。

- 流水线与体里的 `CRLF CRLF`：只按"第一次出现 CRLFCRLF 的地方"切的实现，会把一个
  体里带空行的请求切成两个 —— 那正是请求走私的形状，所以这里有一个专门的负向对照。
- 出站上限：满了必须**报错**。静默扩容的实现在这条门上过不去，而它的后果是慢客户端
  把进程内存吃掉。
- `advance`：只能按 `send(2)` 的返回值走。这里钉住"写一半"之后剩下的字节还是原来那
  截 —— 多发与漏发都表现为对端收到一段拼不起来的响应。
"""

from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from alofa.srv.conn import (
    SEND_HIGH_WATER,
    TAKEN_CLOSED,
    TAKEN_NEED_MORE,
    TAKEN_REQUEST,
    Conn,
)
from alofa.srv.http import bytes_of_text

comptime GET = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
comptime POST_HEAD = "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 5\r\n\r\n"


def is_error(imm text: String, imm name: String) -> Bool:
    """`String(err)` 是不是以 `name(` 开头 —— 与 `tests/unit/test_http.mojo` 同一个
    约定：比名字不比消息。"""
    return text.find(name + "(") == 0


def pending_text(mut conn: Conn) -> String:
    """还没写出去的那截字节（`sent` 之后的）。门要读它，所以从字段里拷出来。"""
    var out = List[UInt8]()
    var i = conn.sent
    while i < len(conn.outbound):
        out.append(conn.outbound[i])
        i += 1
    return String(unsafe_from_utf8=out)


def test_peek_does_not_consume() raises:
    """`peek` 只看不吃：它存在的意义是让空闲超时能分开"排队等生成"与"发了一半"。

    吃掉了的话，那条排队中的请求会在被服务之前先被看没 —— 症状是一个**偶发的空
    响应**，而它只在"生成慢到超过空闲超时"的时候才出现。
    """
    var conn = Conn()
    conn.feed(bytes_of_text(POST_HEAD + "hello"))
    assert_equal(conn.peek(), TAKEN_REQUEST)
    assert_equal(conn.peek(), TAKEN_REQUEST, "peek must not consume")
    assert_equal(conn.inbound_len(), len(bytes_of_text(POST_HEAD + "hello")))
    var taken = conn.take()
    assert_equal(taken.state, TAKEN_REQUEST, "the request must still be there")
    assert_equal(taken.request.body, "hello")


def test_peek_needs_more_for_a_half_sent_request() raises:
    """发了一半：`peek` 必须是 NEED_MORE —— 空闲超时正是靠它把这种连接收掉。"""
    var conn = Conn()
    conn.feed(bytes_of_text(POST_HEAD + "hel"))
    assert_equal(conn.peek(), TAKEN_NEED_MORE)
    conn.mark_peer_closed()


def test_one_request_in_one_feed() raises:
    """一次 feed 一个完整 GET：切出来，缓冲清空。"""
    var conn = Conn()
    conn.feed(bytes_of_text(GET))
    var taken = conn.take()
    assert_equal(taken.state, TAKEN_REQUEST, "a complete request must be taken")
    assert_equal(taken.request.method, "GET")
    assert_equal(taken.request.path(), "/health")
    assert_equal(conn.inbound_len(), 0, "nothing may be left over")


def test_head_split_across_feeds_is_need_more() raises:
    """头被拆成两次 feed：第一次必须是"还要字节"，不是"半个请求"。

    事件循环靠这个返回值决定"这条连接本轮没事干"。把它写成"半个请求"的话，循环要么
    空转要么把半截当完整 —— 两种都不会崩，只是客户端收不到应答。
    """
    var conn = Conn()
    var all = bytes_of_text(GET)
    var cut = 12
    var head_bytes = List[UInt8]()
    for i in range(cut):
        head_bytes.append(all[i])
    conn.feed(head_bytes)
    var first = conn.take()
    assert_equal(first.state, TAKEN_NEED_MORE, "half a head is not a request")
    assert_equal(conn.inbound_len(), cut, "the bytes must still be there")

    var rest = List[UInt8]()
    for i in range(cut, len(all)):
        rest.append(all[i])
    conn.feed(rest)
    var second = conn.take()
    assert_equal(second.state, TAKEN_REQUEST, "the completed head must be taken")
    assert_equal(second.request.path(), "/health")


def test_body_shorter_than_content_length_is_need_more() raises:
    """头齐了、体没齐：还是"还要字节"。按已有的体先处理会拿到一个缺尾的 JSON。"""
    var conn = Conn()
    conn.feed(bytes_of_text(POST_HEAD + "hel"))
    var first = conn.take()
    assert_equal(first.state, TAKEN_NEED_MORE, "a partial body is not a request")
    conn.feed(bytes_of_text("lo"))
    var second = conn.take()
    assert_equal(second.state, TAKEN_REQUEST, "the completed body must be taken")
    assert_equal(second.request.body, "hello")


def test_pipelined_requests_keep_the_rest() raises:
    """一次 read 带回两个请求：第二个不能被吞掉。

    这是"多出来的字节不能丢"那条性质唯一的观测点 —— 丢了的症状是客户端偶尔少一个
    应答，而不是报错。
    """
    var conn = Conn()
    conn.feed(bytes_of_text(POST_HEAD + "hello" + POST_HEAD + "world"))
    var first = conn.take()
    assert_equal(first.state, TAKEN_REQUEST)
    assert_equal(first.request.body, "hello")
    var second = conn.take()
    assert_equal(second.state, TAKEN_REQUEST, "the pipelined request must survive")
    assert_equal(second.request.body, "world", "the second body must be its own")
    assert_equal(conn.inbound_len(), 0)


def test_crlfcrlf_inside_the_body_is_not_a_split() raises:
    """负向对照：体里带 `CRLF CRLF` 仍然**只有一个**请求。

    "从头找第一个空行就切"的实现在这里会切出第二个请求，而那正是请求走私的形状：
    第二个"请求"用的是别人的连接与别人的权限。这条门是这条边界上最该有的那条。
    """
    var conn = Conn()
    var body = "a\r\n\r\nb"
    conn.feed(
        bytes_of_text(
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: "
            + String(body.byte_length())
            + "\r\n\r\n"
            + body
        )
    )
    var first = conn.take()
    assert_equal(first.state, TAKEN_REQUEST)
    assert_equal(first.request.body, body, "the body must be taken as a whole")
    assert_equal(conn.inbound_len(), 0, "the rest of the body is not a second request")
    var second = conn.take()
    assert_equal(second.state, TAKEN_NEED_MORE, "there is no second request")


def test_peer_closed_with_nothing_pending_is_closed() raises:
    """对端关了、缓冲干净：`TAKEN_CLOSED`（keep-alive 的正常结束），不是错误。"""
    var conn = Conn()
    conn.mark_peer_closed()
    assert_equal(conn.take().state, TAKEN_CLOSED)


def test_peer_closed_mid_request_is_an_error() raises:
    """对端关了但发了一半：必须报错，不能静默丢。

    静默丢的表现是"客户端什么也没收到，服务端也不觉得自己做错了什么"。
    """
    var conn = Conn()
    conn.feed(bytes_of_text(POST_HEAD + "hel"))
    conn.mark_peer_closed()
    var got = ""
    try:
        _ = conn.take()
    except err:
        got = String(err)
    assert_true(is_error(got, "parse"), "a half-sent request must be a parse error: " + got)


def test_expect_continue_is_queued_before_the_response() raises:
    """`Expect: 100-continue` 的 interim 应答进的是**同一个**出站队列。

    走另一条写路径的话它会插到真正的响应后面 —— 客户端等到超时才发体，看起来像
    "服务慢了一秒"。
    """
    var conn = Conn()
    conn.feed(
        bytes_of_text(
            "POST /v1/chat/completions HTTP/1.1\r\nExpect: 100-continue\r\n"
            "Content-Length: 5\r\n\r\nhello"
        )
    )
    var taken = conn.take()
    assert_equal(taken.state, TAKEN_REQUEST)
    assert_true(
        pending_text(conn).find("HTTP/1.1 100 Continue") == 0,
        "the interim response must be queued first: " + pending_text(conn),
    )


def test_send_queue_refuses_to_grow_past_the_high_water() raises:
    """出站队列有上限，满了报错 —— 不是静默扩容。

    这条门是背压唯一的观测点：允许无限增长的实现，在慢客户端面前会把"这一条连接变慢"
    变成"进程内存涨上去"，而那种故障在压测里才现形、在单测里从不现形。
    """
    var conn = Conn()
    var big = List[UInt8](capacity=SEND_HIGH_WATER)
    big.resize(SEND_HIGH_WATER, 65)
    conn.enqueue(big)
    assert_equal(conn.pending_len(), SEND_HIGH_WATER, "the queue must reach its ceiling")
    assert_true(conn.backpressure(), "a full queue must say so")

    var got = ""
    try:
        conn.enqueue(bytes_of_text("x"))
    except err:
        got = String(err)
    assert_true(
        is_error(got, "capacity"), "the ceiling must be a capacity error: " + got
    )
    assert_equal(conn.pending_len(), SEND_HIGH_WATER, "nothing may be appended past it")


def test_backpressure_is_not_just_having_bytes_queued() raises:
    """还有字节没写 ≠ 该停读。按后者停读等于把并发退回成"一条一条来"。

    这是背压这条线上最容易写反的一处：写反了不会错，只是并发消失，而并发消失在
    低负载下看不出来。
    """
    var conn = Conn()
    conn.enqueue(bytes_of_text("hello"))
    assert_true(
        not conn.backpressure(), "a few queued bytes must not stop the connection"
    )


def test_a_partial_write_advances_in_place() raises:
    """`send` 只写了一半：剩下的还是原来那截字节，不多也不少。

    钉住它，是因为"假设写就写完"的实现在慢客户端上会把字节发重或发漏 —— 对端收到一段
    拼不起来的响应，而那种错看起来像偶发的解析失败。
    """
    var conn = Conn()
    conn.enqueue(bytes_of_text("hello world"))
    assert_equal(conn.pending_len(), 11)
    conn.advance(5)
    assert_equal(conn.pending_len(), 6, "only what was sent may be marked sent")
    assert_equal(pending_text(conn), " world", "the unsent tail must be unchanged")
    conn.advance(6)
    assert_true(conn.drained(), "the queue must be empty once everything is sent")


def test_advance_past_what_is_queued_is_refused() raises:
    """`advance` 比队列长：那是调用方数错了，必须拒绝而不是把下标推过去。"""
    var conn = Conn()
    conn.enqueue(bytes_of_text("hi"))
    var got = ""
    try:
        conn.advance(3)
    except err:
        got = String(err)
    assert_true(
        is_error(got, "invalid_argument"),
        "over-advancing must be an invalid argument: " + got,
    )
    assert_equal(conn.pending_len(), 2, "the queue must be untouched")


def test_closing_waits_for_the_queue_to_drain() raises:
    """说了要关，也得等队列写完 —— 立刻关会丢掉客户端正在等的那些字节。"""
    var conn = Conn()
    conn.enqueue(bytes_of_text("bye"))
    conn.close_when_drained()
    assert_true(not conn.should_close(), "closing must wait for the queued bytes")
    conn.advance(3)
    assert_true(conn.should_close(), "once drained the connection may be closed")


def test_a_frame_queued_behind_a_partial_write_keeps_its_place() raises:
    """写了一半，又来一帧 —— 新字节必须接在**还没发出去的那截后面**。

    这条序列是**流式独有**的：非流式一次把整段响应排进队列、一次写完，不会在"还有
    没发出去的字节"的时候再入队一帧；流式是每帧一次，所以**每帧都走这条路**。

    钉住它，是因为这里有两条容易写错的路，而它们的后果都不像 bug：把新帧从头覆盖
    （对端收到缺开头的响应），或者把已经发出去的那截又发一遍（对端收到重复的一截，
    JSON 还能解析，只是内容不对）。
    """
    var conn = Conn()
    conn.enqueue(bytes_of_text("aaaaaaaaaa"))
    conn.advance(4)
    assert_equal(conn.pending_len(), 6, "only what was sent may be marked sent")
    conn.enqueue(bytes_of_text("bbbb"))
    assert_equal(conn.pending_len(), 10, "the new frame lands behind the unsent tail")
    assert_equal(
        pending_text(conn),
        "aaaaaabbbb",
        "the unsent tail must come first, then the new frame",
    )
    conn.advance(10)
    assert_true(conn.drained(), "everything is out after the last write")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
