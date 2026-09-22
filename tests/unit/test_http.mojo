"""HTTP/1.1 线上格式的门：请求怎么被拆开，响应怎么被拼回去。

这里的断言全是**逐字节**的。HTTP 的错大多不是"取错了值"，而是"边界找错了地方"：
`Content-Length` 数成字符数、多出来的字节被当成体、只认 LF 的解析器把走私请求切成两个。
这些错在"结构相等"下多半看不出来，在字节相等下无处可藏。

负向对照都在同一个文件里：能解析的用例证明代码**会**干活，被拒的用例证明它**只在**
该干的时候干。
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.srv.http import (
    bytes_of_text,
    json_response,
    parse_request,
    serialize,
    text_response,
    try_head_end,
)


def is_error(imm text: String, imm name: String) -> Bool:
    """`String(err)` 是不是以 `name(` 开头。

    比名字，不比消息：消息是给人看的，名字才是程序能据以分支的东西。而这个门要钉的是
    "拒绝的是不是这一类错" —— 一个把 `unsupported` 报成 `parse` 的实现，会让调用方
    把"我们不收"当成"你写错了"。
    """
    return text.find(name + "(") == 0


def test_request_line_and_headers() raises:
    """一个 GET：三段请求行、两个头、头值两边的空格要被清掉。"""
    var req = parse_request(
        bytes_of_text(
            "GET /v1/models HTTP/1.1\r\nHost: localhost:8000\r\nX-A:  7 \r\n\r\n"
        )
    )
    assert_equal(req.method, "GET")
    assert_equal(req.target, "/v1/models")
    assert_equal(req.version, "HTTP/1.1")
    assert_equal(req.header("host"), "localhost:8000")
    # 头值两边的 OWS 必须清掉，否则 `parse_int` 会把 " 7 " 判成不是数字。
    assert_equal(req.header("X-A"), "7")
    assert_equal(req.content_length(), 0)
    assert_equal(req.body, "")


def test_path_drops_the_query() raises:
    """路由只认路径：`?` 与 `#` 之后都不是路径的一部分。"""
    var req = parse_request(
        bytes_of_text("GET /v1/chat/completions?pretty=1#x HTTP/1.1\r\n\r\n")
    )
    assert_equal(req.target, "/v1/chat/completions?pretty=1#x")
    assert_equal(req.path(), "/v1/chat/completions")


def test_header_lookup_ignores_case_and_takes_the_first() raises:
    """头名大小写不敏感；同名头取第一个 —— 取最后一个会让 `Content-Length` 变成
    一个可以被对端追加的头，那是走私的形状。"""
    var req = parse_request(
        bytes_of_text(
            "GET / HTTP/1.1\r\nContent-Length: 3\r\ncontent-length: 9\r\n\r\nabc"
        )
    )
    assert_equal(req.content_length(), 3)
    assert_equal(req.body, "abc")


def test_keep_alive_defaults_and_connection_close() raises:
    """1.1 默认留着，1.0 默认关掉，`Connection` 可以把两者反过来。"""
    var keep = parse_request(bytes_of_text("GET / HTTP/1.1\r\n\r\n"))
    assert_true(keep.keep_alive(), "HTTP/1.1 must default to keep-alive")

    var old = parse_request(bytes_of_text("GET / HTTP/1.0\r\n\r\n"))
    assert_true(not old.keep_alive(), "HTTP/1.0 must default to close")

    var closed = parse_request(
        bytes_of_text("GET / HTTP/1.1\r\nConnection: close\r\n\r\n")
    )
    assert_true(not closed.keep_alive(), "Connection: close must close")

    var reopened = parse_request(
        bytes_of_text("GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
    )
    assert_true(reopened.keep_alive(), "Connection: keep-alive must keep alive")


def test_body_is_exactly_content_length_bytes() raises:
    """体**只**是 `Content-Length` 说的那么多字节。

    这条是负向对照：多出来的 `GET /second` 属于**下一个**请求（流水线）。把它吃进这个
    请求的体里，第二个请求就凭空消失了 —— 而单进程服务里"消失"表现为"没响应"。
    """
    var req = parse_request(
        bytes_of_text(
            "POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello"
            + "GET /second HTTP/1.1\r\n\r\n"
        )
    )
    assert_equal(req.body, "hello")


def test_short_body_is_refused() raises:
    """体比 `Content-Length` 短：不是一个"半截请求"，是解析失败。

    若当成半截请求继续读，这个连接就会一直等一个永远不来的字节 —— 而在单进程里那等于
    整个服务停摆。
    """
    var got = ""
    try:
        _ = parse_request(
            bytes_of_text("POST /x HTTP/1.1\r\nContent-Length: 50\r\n\r\nshort")
        )
    except err:
        got = String(err)
    assert_true(is_error(got, "parse"), "a short body must be a parse error: " + got)


def test_chunked_is_refused_by_name() raises:
    """`Transfer-Encoding` 指名拒绝，而且必须是 `unsupported` 而不是 `parse`：
    "我们不收"和"你写错了"对调用方是两件事。"""
    var got = ""
    try:
        _ = parse_request(
            bytes_of_text(
                "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
            )
        )
    except err:
        got = String(err)
    assert_true(
        is_error(got, "unsupported"), "chunked must be refused as unsupported: " + got
    )


def test_request_line_with_wrong_field_count() raises:
    """请求行必须正好三段。宽松的解析器会把"多一个空格"变成"路径里有个空格"。"""
    var few = ""
    try:
        _ = parse_request(bytes_of_text("GET /\r\n\r\n"))
    except err:
        few = String(err)

    var many = ""
    try:
        _ = parse_request(bytes_of_text("GET /a b HTTP/1.1\r\n\r\n"))
    except err:
        many = String(err)

    assert_true(is_error(few, "parse"), "two fields must not parse: " + few)
    assert_true(is_error(many, "parse"), "four fields must not parse: " + many)


def test_bare_lf_is_not_a_header_end() raises:
    """只认 `CRLF CRLF`。

    这条是负向对照：把裸 LF 也当结束，一个夹带裸 LF 的请求就能被切成两个。而 `-1`
    是"还没收完"的意思，不是"解析失败" —— 两者在服务循环里走的是两条路。
    """
    assert_equal(try_head_end(bytes_of_text("GET / HTTP/1.1\n\n")), -1)
    assert_equal(try_head_end(bytes_of_text("GET / HTTP/1.1\r\n\r\n")), 18)


def test_oversized_headers_are_refused() raises:
    """永远不发空行的对端必须有个终点（`capacity`），否则它占住的是**整个进程**。"""
    var raw = bytes_of_text("GET / HTTP/1.1\r\n")
    while len(raw) <= 8192:
        raw.append(88)  # 'X'
        raw.append(13)
        raw.append(10)
    var got = ""
    try:
        _ = try_head_end(raw)
    except err:
        got = String(err)
    assert_true(
        is_error(got, "capacity"), "oversized headers must be refused: " + got
    )


def test_response_bytes_are_exact() raises:
    """响应的**每一个字节**都钉住：状态行、头序、`Content-Length`、空行。

    没有 `Date` 是故意的（见 `serialize` 的注释），所以这条能逐字节比 —— 带上了就
    只能比结构，而结构相等放过了长度算错。
    """
    var res = json_response(200, "{\"ok\":true}", False)
    assert_equal(
        serialize(res),
        "HTTP/1.1 200 OK\r\n"
        + "Content-Type: application/json\r\n"
        + "Content-Length: 11\r\n"
        + "Connection: keep-alive\r\n"
        + "\r\n"
        + "{\"ok\":true}",
    )


def test_close_response_says_close() raises:
    """`close=True` 必须落在 `Connection` 上，而不只是"写完就关" —— 对端要靠这个头
    决定还能不能再发一个请求。"""
    var res = text_response(404, "no", True)
    assert_equal(
        serialize(res),
        "HTTP/1.1 404 Not Found\r\n"
        + "Content-Type: text/plain; charset=utf-8\r\n"
        + "Content-Length: 2\r\n"
        + "Connection: close\r\n"
        + "\r\n"
        + "no",
    )


def test_content_length_counts_bytes_not_characters() raises:
    """`Content-Length` 是**字节**数。

    这条是负向对照：`é` 是两个字节一个字符。数成字符数，对端就会少读一个字节，然后把
    下一个响应的第一个字节当成这个体的最后一个字节 —— 之后的一切都错位，而且看起来
    像"偶发的乱码"。
    """
    var res = json_response(200, "é", False)
    assert_equal(res.body.byte_length(), 2)
    assert_true(
        serialize(res).find("Content-Length: 2\r\n") >= 0,
        "Content-Length must count bytes: " + serialize(res),
    )


def test_unknown_status_has_no_reason() raises:
    """没有名字的状态码必须报错，而不是编一个原因短语出来。"""
    var got = ""
    try:
        _ = serialize(json_response(299, "{}", False))
    except err:
        got = String(err)
    assert_true(
        is_error(got, "unsupported"), "an unnamed status must be refused: " + got
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
