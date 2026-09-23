"""`/v1/chat/completions` 非流式一半的门：请求体怎么读，响应体怎么写，路由怎么走。

这里用的替身（`Stub`）**不需要权重**：HTTP 与 JSON 这一层的对错与模型无关，让它们
靠加载 1.9 GB 权重才能测，等于把"改一个头字段"的反馈周期变成"加载一次模型"。真权重
那一侧由 `src/alofa/serve.mojo` 手动跑（`pixi run serve`）。

负向对照两条思路：
- **会拒绝的必须拒绝**（`stream=true`、代理对、指数、空 `messages`、越界的
  `max_tokens`、尾部多余字节）—— 一个把 `stream=true` 静默忽略的服务，客户端会一直
  等一个不会来的 SSE 事件。
- **会保留的必须保留**（转义、多字节 UTF-8、多个 `messages` 的顺序）—— 一个把
  `Content-Length` 或引号处理错的实现，出来的 JSON 仍然"看起来像 JSON"。
"""

from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import ERR_CAPACITY, ERR_INVALID_ARGUMENT, AlofaError
from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import SSE_CONTENT_TYPE, HttpRequest
from alofa.srv.openai import (
    ChatHandler,
    Completion,
    Service,
    chunk_text_json,
    completion_json,
    error_json,
    escape_json,
    parse_chat_request,
)
from alofa.srv.sse import StreamToken, sse_frame
from alofa.tokenizer.chat_template import (
    GENERATION_PROMPT,
    IM_END,
    IM_START,
    QWEN25_DEFAULT_SYSTEM,
)


def is_error(imm text: String, imm name: String) -> Bool:
    """`String(err)` 是不是以 `name(` 开头 —— 拒绝对不对，看的是名字。"""
    return text.find(name + "(") == 0


struct Stub(Service, Twinable):
    """一个不需要权重的生成端。

    `give` 控制"声称生成了多少个 token"：`-1` 是要多少给多少（于是 `finish_reason`
    应当是 `length`），正数则用来造出 `stop` 那条路。流式沿用同一个数：要多少
    就真的交多少个 token（`s0`、`s1`……），所以 `finish_reason` 的两条路在流式
    这一侧也都能走到。

    `fail` / `refuse` 造的是两条**出错**的路：`refuse` 在 `stream_begin`（响应头
    之前，应当回 400），`fail` 在 `stream_next`（头已经发出去了，只剩下的退路是
    一帧 error —— 见 `ChatHandler.stream_next`）。
    """

    var give: Int
    var calls: Int
    var begins: Int
    var stream_step: Int
    var stream_max: Int
    var fail: Bool
    var refuse: Bool

    def __init__(out self, give: Int = -1, fail: Bool = False, refuse: Bool = False):
        self.give = give
        self.calls = 0
        self.begins = 0
        self.stream_step = 0
        self.stream_max = 0
        self.fail = fail
        self.refuse = refuse

    def complete(
        mut self, prompt: String, max_tokens: Int, temperature: Float64
    ) raises -> Completion:
        self.calls += 1
        var n = max_tokens
        if self.give >= 0:
            n = self.give
        return Completion("S:" + prompt, 7, n)

    def spawn_twin(self, index: Int) raises -> Int:
        """engine 线程池要的"再给我一份"（零权重，所以这里是真的零成本）。

        `give` / `fail` / `refuse` 都要跟着走：它们决定这个替身答什么，而"两条
        线程上的替身行为不一致"会让并发那条门安静地测错对象。
        """
        return heap_place(Stub(self.give, self.fail, self.refuse)^)

    def stream_begin(
        mut self, prompt: String, max_tokens: Int, temperature: Float64
    ) raises:
        self.begins += 1
        if self.refuse:
            # 与非流式同一个错：prompt 分词为空，属于**请求**的错。
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "the prompt tokenized to nothing", prompt
            )
        self.stream_max = max_tokens
        self.stream_step = 0

    def stream_next(mut self) raises -> StreamToken:
        if self.fail:
            raise AlofaError(
                ERR_CAPACITY, "the stream broke", "step=" + String(self.stream_step)
            )
        var total = self.stream_max
        if self.give >= 0:
            total = self.give
        # 交完之后**继续调用仍然返回 done**：路由可能多问一次（它才知道要了
        # 多少），重新开始或报错都会让流失去终点。
        if self.stream_step >= total:
            return StreamToken("", True)
        var text = "s" + String(self.stream_step)
        self.stream_step += 1
        return StreamToken(text, False)


def make_request(method: String, target: String, body: String) -> HttpRequest:
    var req = HttpRequest()
    req.method = method
    req.target = target
    req.version = "HTTP/1.1"
    req.body = body
    return req^


def test_request_fields() raises:
    var chat = parse_chat_request(
        "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        + "\"max_tokens\":8,\"temperature\":0.5}"
    )
    assert_equal(chat.model, "qwen")
    assert_equal(chat.prompt, chatml(turn("user", "hi")))
    assert_equal(chat.max_tokens, 8)
    assert_equal(chat.temperature, Float64(0.5))
    assert_true(not chat.stream, "stream must default to false")


def test_defaults_when_the_request_omits_them() raises:
    """`max_tokens` 与 `temperature` 缺了就用默认值，而不是 0 —— `temperature=0`
    是贪心，`temperature` 缺失是"按默认来"，两者不是一回事。"""
    var chat = parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
    )
    assert_equal(chat.max_tokens, 32)
    assert_equal(chat.temperature, Float64(1.0))


def chatml(imm body: String) -> String:
    """没有首条 system 时的标准形状：模板自带那句 + 这一段 body + 补笔。

    期望是**在这座 Note 里手拼的**（不是再调一次 `render_chatml`）—— 调回渲染器比
    的话，这扇门就只是在跟自己对答案。渲染器那侧求真是在
    `tests/unit/test_chat_template.mojo`，它比的是 transformers 导出来的夹具。
    """
    return (
        IM_START
        + "system\n"
        + QWEN25_DEFAULT_SYSTEM
        + IM_END
        + "\n"
        + body
        + GENERATION_PROMPT
    )


def turn(imm role: String, imm content: String) -> String:
    """一条非首条的消息段。"""
    return IM_START + role + "\n" + content + IM_END + "\n"


def test_messages_are_rendered_with_roles() raises:
    """首条 `system` 进 system 段，`user` 按序各占一段 —— 角色**参与**了。

    上一版的期望是 `"a\\nb\\nc"`（按顺序拼起来），那时角色被读掉不参与。这条钉的是
    "同一份输入现在会长成什么样"，以及那条最容易被写错的判据：首条 system **不在**
    循环里重复出现。
    """
    var chat = parse_chat_request(
        "{\"messages\":[{\"role\":\"system\",\"content\":\"a\"},"
        + "{\"role\":\"user\",\"content\":\"b\"},{\"role\":\"user\",\"content\":\"c\"}]}"
    )
    var expected = (
        IM_START
        + "system\n"
        + "a"
        + IM_END
        + "\n"
        + turn("user", "b")
        + turn("user", "c")
        + GENERATION_PROMPT
    )
    assert_equal(chat.prompt, expected)


def test_unknown_keys_are_skipped() raises:
    """不认识的键跳过而不是报错：OpenAI 的客户端会带一堆别的键。

    这段同时是 `skip_value` 的负向对照：`"top_p":0.9` 后面跟着**对象**与**数组**，
    一个"找到下一个逗号就停"的跳过器会在这里把结构切坏。
    """
    var chat = parse_chat_request(
        "{\"top_p\":0.9,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        + "\"extra\":{\"a\":[1,2],\"b\":\"}\"},\"user\":\"abc\",\"list\":[1,2,3],"
        + "\"max_tokens\":4}"
    )
    assert_equal(chat.max_tokens, 4)
    # prompt 现在是渲染过的一段 ChatML（不是 "hi" 本身），而对不对的答案在
    # `test_chat_template.mojo` 那边的夹具里；这里只保证辅助 key 被跳过了之后
    # **`messages` 照旧被读到了**。
    assert_equal(chat.prompt, chatml(turn("user", "hi")))


def test_escapes_and_unicode() raises:
    """转义必须还原成真的字节；`\\u00e9` 要变成两个字节的 `é`。

    这条是负向对照：原样保留 `\\u00e9` 六个字符的实现，出来的 JSON 仍然合法，但
    prompt 已经不是用户写的那个 prompt 了 —— 而分词器会因此分出完全不同的 token。
    """
    var chat = parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\\\"b\\\\c\\nd\\u00e9\"}]}"
    )
    # 转义要还原成真的字节 —— 只是现在它还被包在 ChatML 里。这条照样是负向对照：
    # 原样保留 `\u00e9` 的实现出来的仍然是一段合法的 JSON/prompt，但 token 全变了。
    assert_equal(chat.prompt, chatml(turn("user", "a\"b\\c\ndé")))


def test_surrogate_half_is_refused() raises:
    """半个代理对没法编成 UTF-8，所以指名拒绝 —— 悄悄丢掉才是真的糟。"""
    var got = ""
    try:
        _ = parse_chat_request(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"\\ud83d\\ude00\"}]}"
        )
    except err:
        got = String(err)
    assert_true(is_error(got, "unsupported"), "surrogates must be refused: " + got)


def test_exponent_notation_is_refused() raises:
    """`parse_float64` 不收指数，所以必须显式拒绝，而不是读一半当成 1。"""
    var got = ""
    try:
        _ = parse_chat_request(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
            + "\"temperature\":1e-5}"
        )
    except err:
        got = String(err)
    assert_true(is_error(got, "unsupported"), "exponents must be refused: " + got)


def test_no_messages_is_refused() raises:
    """没有 `messages` 是**对端**的错（`invalid_argument`）：静默生成空字符串会让
    客户端以为模型"回答了，回答是空的"。"""
    var empty_array = ""
    try:
        _ = parse_chat_request("{\"messages\":[]}")
    except err:
        empty_array = String(err)

    var missing = ""
    try:
        _ = parse_chat_request("{\"max_tokens\":4}")
    except err:
        missing = String(err)

    assert_true(
        is_error(empty_array, "invalid_argument"),
        "an empty messages array must be refused: " + empty_array,
    )
    assert_true(
        is_error(missing, "invalid_argument"),
        "a missing messages field must be refused: " + missing,
    )


def test_max_tokens_bounds() raises:
    """0 与负值是对端的错；超过这条流的上限是容量问题 —— 两者名字不同，因为修法
    不同（改请求 vs 换服务）。"""
    var zero = ""
    try:
        _ = parse_chat_request(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":0}"
        )
    except err:
        zero = String(err)

    var huge = ""
    try:
        _ = parse_chat_request(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":257}"
        )
    except err:
        huge = String(err)

    assert_true(is_error(zero, "invalid_argument"), "max_tokens=0: " + zero)
    assert_true(is_error(huge, "capacity"), "max_tokens=257: " + huge)


def test_trailing_bytes_are_refused() raises:
    """对象结束后还有字节就是错：那通常是两个请求被拼在了一个体里。"""
    var got = ""
    try:
        _ = parse_chat_request(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]} {}"
        )
    except err:
        got = String(err)
    assert_true(is_error(got, "parse"), "trailing bytes must be refused: " + got)


def test_completion_json_is_exact() raises:
    """响应体的字节钉住：字段名、顺序、`usage` 的三数一致。

    `total_tokens` 必须是另外两个数之和 —— 它由这两个数算出来，不是一个可以自己填的
    字段（手填就会有人填错）。
    """
    assert_equal(
        completion_json("chatcmpl-1", "m", "hi", 2, 3, "length"),
        "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"created\":0,"
        + "\"model\":\"m\",\"choices\":[{\"index\":0,\"message\":{\"role\":"
        + "\"assistant\",\"content\":\"hi\"},\"finish_reason\":\"length\"}],"
        + "\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,"
        + "\"total_tokens\":5}}",
    )


def test_completion_json_escapes_the_content() raises:
    """生成的文本里可以有引号、反斜杠与换行 —— 它们必须被转义，否则出来的不是
    JSON（而"看起来像 JSON"正是这类 bug 的样子）。"""
    var body = completion_json("i", "m", "a\"b\\c\nd", 1, 1, "stop")
    assert_true(
        body.find("\"content\":\"a\\\"b\\\\c\\nd\"") >= 0,
        "the content must be escaped: " + body,
    )


def test_error_json_escapes_the_message() raises:
    """错误消息常常来自请求本身（"no route for /x?a=b"），所以它也要转义。"""
    var body = error_json("no route for \"x\"", "invalid_request_error", "not_found")
    assert_equal(
        body,
        "{\"error\":{\"message\":\"no route for \\\"x\\\"\","
        + "\"type\":\"invalid_request_error\",\"code\":\"not_found\"}}",
    )


def test_handler_answers_health() raises:
    var handler = ChatHandler(Stub(), "stub-model")
    var res = handler.handle(make_request("GET", "/health", ""))
    assert_equal(res.status, 200)
    assert_true(res.body.find("\"status\":\"ok\"") >= 0, res.body)
    assert_true(res.body.find("\"model\":\"stub-model\"") >= 0, res.body)
    # "还没有 chat template"这件事必须能被不读源码的人看见。
    assert_true(res.body.find("\"streaming\":true") >= 0, res.body)
    # "它有 chat template 了、但那只是一个族群的渲染器"这件事，也必須能被不读源码
    # 的人看见 —— 所以 `PROMPT_NOTE` 里同时写着两件事。
    assert_true(res.body.find("roles") >= 0, res.body)
    assert_true(
        res.body.find("not a template engine") >= 0, res.body
    )


def test_handler_answers_a_completion() raises:
    var handler = ChatHandler(Stub(), "stub-model")
    var res = handler.handle(
        make_request(
            "POST",
            "/v1/chat/completions",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}",
        )
    )
    assert_equal(res.status, 200)
    assert_true(res.body.find(escape_json("S:" + chatml(turn("user", "hi")))) >= 0, res.body)
    # 替身要多少给多少 → 撞上了上限 → `length`，不是 `stop`。
    assert_true(res.body.find("\"finish_reason\":\"length\"") >= 0, res.body)
    assert_true(res.body.find("\"id\":\"chatcmpl-1\"") >= 0, res.body)


def test_finish_reason_is_stop_when_it_did_not_hit_the_limit() raises:
    """`finish_reason` 由"要了多少、给了多少"决定 —— 两条路都要走到，否则其中一条
    永远是没被执行过的代码。"""
    var handler = ChatHandler(Stub(1), "stub-model")
    var res = handler.handle(
        make_request(
            "POST",
            "/v1/chat/completions",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}",
        )
    )
    assert_true(res.body.find("\"finish_reason\":\"stop\"") >= 0, res.body)


def test_handler_lets_the_request_override_the_model_name() raises:
    var handler = ChatHandler(Stub(), "stub-model")
    var res = handler.handle(
        make_request(
            "POST",
            "/v1/chat/completions",
            "{\"model\":\"asked-for\",\"messages\":[{\"role\":\"user\","
            + "\"content\":\"hi\"}],\"max_tokens\":2}",
        )
    )
    assert_true(res.body.find("\"model\":\"asked-for\"") >= 0, res.body)


def stream_request(imm max_tokens: Int) -> HttpRequest:
    return make_request(
        "POST",
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":"
        + String(max_tokens)
        + ",\"stream\":true}",
    )


def collect_frames[S: Service & Deinitable & Movable & Twinable](
    mut handler: ChatHandler[S]
) raises -> List[String]:
    """把一条流从头问到尾：**收到空串才算收尾**。

    上限 64 帧是防"这条流没有终点"把自己挂住 —— 那正是这些门要抓的失败形态，
    挂住的话门就变成了"测试跑不完"，而不是"测试失败"。
    """
    var out = List[String]()
    for _ in range(64):
        var frame = handler.stream_next()
        if frame.byte_length() == 0:
            return out^
        out.append(frame)
    raise AlofaError(ERR_CAPACITY, "the stream never ended", "frames=" + String(len(out)))


def test_handler_answers_a_stream_with_a_head() raises:
    """`stream=true` 给的是**一个头**：体为空、类型是 SSE、连接必须关。

    关连接不是"顺便关一下"，而是这条响应唯一的定界方式（没有 `Content-Length`，
    见 `srv/http.mojo`）。非流式那半的帧由 `stream_next` 逐条吐，不在这里。
    """
    var handler = ChatHandler(Stub(), "stub-model")
    var res = handler.handle(stream_request(8))
    assert_equal(res.status, 200)
    assert_equal(res.content_type, SSE_CONTENT_TYPE)
    assert_equal(res.body, "", "a stream head must not carry a body")
    assert_true(res.close, "a stream must delimit itself by closing")
    assert_equal(handler.service.begins, 1, "the stream must begin with the request")
    assert_equal(
        handler.service.calls, 0, "a stream must not run the whole completion at once"
    )


def test_stream_frames_are_exact() raises:
    """帧的字节钉住：首帧只有 role、内容帧带 delta、末帧带 finish_reason、
    然后是 `data: [DONE]`。

    逐字节而不是"包含某句话"：SSE 客户端按 `data:` 行与空行切分，字段多一个
    少一个在"能解析"这一层看不出来，但在客户端那边是"少收一帧"。
    """
    var handler = ChatHandler(Stub(3), "stub-model")
    _ = handler.handle(stream_request(8))
    var frames = collect_frames(handler)
    assert_equal(len(frames), 6, "role + 3 content + finish + done: " + String(len(frames)))
    assert_equal(
        frames[0],
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
        + "\"created\":0,\"model\":\"stub-model\",\"choices\":[{\"index\":0,"
        + "\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
        + "\"finish_reason\":null}]}\r\n\r\n",
    )
    assert_equal(
        frames[1],
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
        + "\"created\":0,\"model\":\"stub-model\",\"choices\":[{\"index\":0,"
        + "\"delta\":{\"content\":\"s0\"},\"finish_reason\":null}]}\r\n\r\n",
    )
    assert_equal(
        frames[3],
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
        + "\"created\":0,\"model\":\"stub-model\",\"choices\":[{\"index\":0,"
        + "\"delta\":{\"content\":\"s2\"},\"finish_reason\":null}]}\r\n\r\n",
    )
    assert_equal(
        frames[4],
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
        + "\"created\":0,\"model\":\"stub-model\",\"choices\":[{\"index\":0,"
        + "\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n",
    )
    assert_equal(frames[5], "data: [DONE]\r\n\r\n")


def test_stream_finish_reason_follows_the_same_rule() raises:
    """`finish_reason` 的两条路在流式这一侧也必须都走到：撞上上限是 `length`，
    没撞上是 `stop`（与非流式同一条规则：取决于"要了多少、给了多少"）。"""
    var limited = ChatHandler(Stub(), "stub-model")
    _ = limited.handle(stream_request(8))
    var hit = collect_frames(limited)
    assert_true(
        hit[len(hit) - 2].find("\"finish_reason\":\"length\"") >= 0,
        "generating the requested 8 tokens must finish as length: " + hit[len(hit) - 2],
    )

    var early = ChatHandler(Stub(1), "stub-model")
    _ = early.handle(stream_request(8))
    var stopped = collect_frames(early)
    assert_equal(len(stopped), 4, "role + 1 content + finish + done")
    assert_true(
        stopped[2].find("\"finish_reason\":\"stop\"") >= 0,
        "stopping early must finish as stop: " + stopped[2],
    )


def test_stream_with_no_tokens_still_ends() raises:
    """一步都没走（prompt 就占满了窗口）也要有完整的收尾：首帧、finish、
    `[DONE]`。少任何一帧，客户端都在等一个不会来的东西。"""
    var handler = ChatHandler(Stub(0), "stub-model")
    _ = handler.handle(stream_request(8))
    var frames = collect_frames(handler)
    assert_equal(len(frames), 3, "role + finish + done: " + String(len(frames)))
    assert_true(frames[1].find("\"delta\":{}") >= 0, frames[1])
    assert_equal(frames[2], "data: [DONE]\r\n\r\n")


def test_stream_reports_the_failure_as_a_frame() raises:
    """生成中途出错：头已经发出去了，退路只剩下"给流一个终点"。

    错误帧 + `[DONE]` 而不是默默关掉 —— 客户端得能区分"生成结束了"与"生成
    坏了"，否则这两种情况在它看来都是"收到了若干帧然后连接关了"。
    """
    var handler = ChatHandler(Stub(-1, True), "stub-model")
    _ = handler.handle(stream_request(8))
    var frames = collect_frames(handler)
    assert_equal(len(frames), 3, "role + error + done: " + String(len(frames)))
    assert_true(frames[1].find("\"code\":\"stream_failed\"") >= 0, frames[1])
    assert_equal(frames[2], "data: [DONE]\r\n\r\n")


def test_stream_begin_failure_is_400_before_any_frame() raises:
    """`stream_begin` 的错（prompt 分词为空/过长）发生在**头之前**，所以还能回
    400 —— 而且**不关连接**（客户端可以改个 prompt 重试），也不能开始一条流。"""
    var handler = ChatHandler(Stub(-1, False, True), "stub-model")
    var res = handler.handle(stream_request(8))
    assert_equal(res.status, 400)
    assert_equal(res.content_type, "application/json")
    assert_true(not res.close, "a refused stream must not close the connection")
    assert_equal(handler.service.begins, 1, "the request must have been checked")
    assert_equal(handler.stream_next(), "", "a refused request must not start a stream")


def test_stream_chunk_escapes_the_text() raises:
    """一个 token 里可以有引号和换行：不转义的话，换行会把一帧切成两帧（而
    `sse_frame` 会直接拒绝 —— 那条负向对照在 `test_sse.mojo`）。"""
    var frame = sse_frame(chunk_text_json("i", "m", "a\"b\nc"))
    assert_equal(
        frame,
        "data: {\"id\":\"i\",\"object\":\"chat.completion.chunk\",\"created\":0,"
        + "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":"
        + "\"a\\\"b\\nc\"},\"finish_reason\":null}]}\r\n\r\n",
    )


def test_handler_refuses_unknown_paths_and_methods() raises:
    var handler = ChatHandler(Stub(), "stub-model")
    var missing = handler.handle(make_request("GET", "/nope", ""))
    assert_equal(missing.status, 404)
    assert_true(missing.body.find("\"code\":\"not_found\"") >= 0, missing.body)

    var wrong_method = handler.handle(make_request("GET", "/v1/chat/completions", ""))
    assert_equal(wrong_method.status, 405)
    assert_true(
        wrong_method.body.find("\"code\":\"method_not_allowed\"") >= 0,
        wrong_method.body,
    )


def test_handler_reports_a_broken_body_as_400() raises:
    """请求体的错是对端的错（400），不是服务器的错（500）。"""
    var handler = ChatHandler(Stub(), "stub-model")
    var res = handler.handle(make_request("POST", "/v1/chat/completions", "{oops"))
    assert_equal(res.status, 400)
    assert_true(res.body.find("\"code\":\"invalid_body\"") >= 0, res.body)


def test_a_message_without_a_role_is_refused() raises:
    """缺 `role` 不许被当成"无角色的消息"照样渲染 —— 那正是这次修掉的 bug。

    所以回的是 `invalid_argument`（对端的错），而不是 500：请求写错了要对端知道。
    """
    var got = ""
    try:
        _ = parse_chat_request("{\"messages\":[{\"content\":\"hi\"}]}")
    except err:
        got = String(err)
    assert_true(is_error(got, "invalid_argument"), "no role must be refused: " + got)


def test_an_unknown_role_is_refused() raises:
    """`tool` 这类角色不许被拼成 `<|im_start|>tool\n...`。

    它在模板里走的完全是另一段（`<tool_response>`，包在 `<|im_start|>user` 里），
    拼错了不会有错 —— 只会让模型把一段工具输出当成人话读，安静地答得不一样。
    """
    var got = ""
    try:
        _ = parse_chat_request(
            "{\"messages\":[{\"role\":\"tool\",\"content\":\"hi\"}]}"
        )
    except err:
        got = String(err)
    assert_true(is_error(got, "invalid_argument"), "unknown role: " + got)


def test_a_message_without_content_is_refused() raises:
    """只有 `role` 的消息不许被静默丢掉 —— 上一版正是这么做的（`has_content` 为假
    就整条跳过），那样一条消息会消失得毫无痕迹，而 prompt 看起来照样正常。"""
    var got = ""
    try:
        _ = parse_chat_request("{\"messages\":[{\"role\":\"user\"}]}")
    except err:
        got = String(err)
    assert_true(
        is_error(got, "invalid_argument"), "no content must be refused: " + got
    )


def test_roles_change_the_prompt() raises:
    """同一段文字换个角色，prompt 必须不一样 —— 否则"角色参与了"只是一句注释。

    这条同时是前面那些"按序拼"的负向对照：若还停在拼 content 的老路上，两条请求会
    出一模一样的 prompt。
    """
    var as_user = parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
    )
    var as_assistant = parse_chat_request(
        "{\"messages\":[{\"role\":\"assistant\",\"content\":\"hello\"}]}"
    )
    assert_true(
        as_user.prompt != as_assistant.prompt,
        "the same text under another role must not give the same prompt",
    )
    assert_equal(as_user.prompt, chatml(turn("user", "hello")))
    assert_equal(as_assistant.prompt, chatml(turn("assistant", "hello")))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
