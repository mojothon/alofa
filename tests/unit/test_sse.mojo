"""SSE 分帧的门：逐字节，不需要权重，也不需要 socket。

这里钉的是**格式**，而格式最容易被"看起来对"蒙过去：

- 流式头**不能有** `Content-Length`（长度在开始写的时候未知；带上一个算错的
  长度，客户端要么切短要么一直等）—— 这是负向对照，不是"顺便看一下"。
- 一帧必须以 `CRLF CRLF` 结束，且只有一处 —— 少了不结束，多了是两帧被粘住。
- 含裸换行的 JSON 必须**被拒绝**而不是被"修好"：静默改掉换行会让"内容里有
  个换行"变成"客户端少收一帧"。

`StreamToken` 的两个字段分开测，是因为它们表达的是两件不同的事：这一步有没
有文本，以及生成是不是结束了。
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.srv.http import SSE_CONTENT_TYPE, serialize_stream_head
from alofa.srv.sse import StreamToken, sse_done, sse_frame


def test_stream_head_is_exact() raises:
    """头的字节钉住：状态行、三个头、空行结束，没有体。"""
    assert_equal(
        serialize_stream_head(200),
        "HTTP/1.1 200 OK\r\n"
        + "Content-Type: text/event-stream\r\n"
        + "Cache-Control: no-cache\r\n"
        + "Connection: close\r\n"
        + "\r\n",
    )


def test_stream_head_has_no_content_length() raises:
    """负向对照：流式响应长度未知，所以**不许**出现 `Content-Length`。

    少一个长度头只是让客户端改用"读到连接关"；多一个**算错**的长度头会让客
    户端从流里切出一段既不像帧也不像错误的东西 —— 后者才是这条门要挡的。
    """
    var head = serialize_stream_head(200)
    assert_true(
        head.find("Content-Length") < 0,
        "a streaming head must not claim a length: " + head,
    )


def test_stream_head_marks_the_content_type() raises:
    """服务循环靠 `Content-Type` 认出流式响应，所以头里的值必须**就是**那个
    常量 —— 两边各写一个字面量的话，改了一处会静默地让流式退化成非流式。"""
    var head = serialize_stream_head(200)
    assert_true(
        head.find("Content-Type: " + SSE_CONTENT_TYPE + "\r\n") > 0,
        "the head must carry the SSE content type: " + head,
    )
    assert_equal(SSE_CONTENT_TYPE, "text/event-stream")


def test_frame_is_exact() raises:
    assert_equal(sse_frame("{\"a\":1}"), "data: {\"a\":1}\r\n\r\n")


def test_frame_ends_with_one_blank_line() raises:
    """`CRLF CRLF` 恰好一处：多处意味着帧被粘住，少处意味着帧永不结束。"""
    var frame = sse_frame("{}")
    var first = frame.find("\r\n\r\n")
    assert_true(first > 0, "the frame has no terminator: " + frame)
    assert_equal(frame.byte_length(), first + 4, "the frame has trailing bytes")
    assert_true(frame.find("\r\n\r\n", first + 4) < 0, "two frames got glued: " + frame)


def test_done_frame_is_exact() raises:
    """`[DONE]` 是协议要求的收尾：客户端靠它区分"生成完了"与"连接断了"。"""
    assert_equal(sse_done(), "data: [DONE]\r\n\r\n")


def test_frame_refuses_a_raw_newline() raises:
    """负向对照：裸换行必须报错，不许被静默改掉（见文件头）。"""
    var got = ""
    try:
        _ = sse_frame("{\"a\":1}\n")
    except err:
        got = String(err)
    assert_true(
        got.find("invalid_argument(") == 0,
        "a raw newline must be refused, not repaired: " + got,
    )


def test_token_carries_text_and_done_separately() raises:
    var token = StreamToken("hi", False)
    assert_equal(token.text, "hi")
    assert_true(not token.done, "a token with text is not the end")

    # 结束的那一帧可以没有文本：两者不是一回事，所以不能用一个字段表示。
    var last = StreamToken("", True)
    assert_equal(last.text, "")
    assert_true(last.done, "the final token must say it is done")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
