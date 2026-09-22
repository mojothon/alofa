"""`/v1/chat/completions`：请求怎么读，响应怎么写（非流式与流式两半）。

请求整段进来之后有两条路：

- **非流式**：生成整段做完，**一次**带着 `Content-Length` 回去。
- **流式**（`stream=true`）：先回一个 `text/event-stream` 的头（**没有**
  `Content-Length` —— 那时候还不知道有多长），然后逐 token 吐 SSE 帧，以
  `data: [DONE]` 收尾并关连接。

分帧（`data:` 行、 `[DONE]`）在 `srv/sse.mojo` 里，是纯函数；这里只拼帧里的
JSON。流式**中途出错**也要给流一个终点（一帧错误 + `[DONE]` 或至少关连接）——
半流式（答应下来再吐一半，客户端永远等不到 `[DONE]`）是比明确拒绝更糟的失败
形态。

JSON 为什么自己扫
-----------------
`json` 包是好东西，但它解析出的是**通用**结构：缺字段变成运行期的空值，多字段被安静
丢掉。而对外契约要的恰恰相反 —— **缺字段必须有名字**：`messages` 是空数组要回
`invalid_request_error` 而不是"生成了空字符串"；`stream=true` 要回 400 而不是被忽略。
所以这里扫的是**一个**形状，超出这个形状的部分按如下规则处理：

- 不认识的键：跳过（`skip_value`），不报错 —— OpenAI 的客户端会带一堆别的键。
- 认识但格式不对：报错，并把位置写进 detail。
- 字符串里的转义（`u` 加四位十六进制）：解成 UTF-8；代理对（surrogate pair）指名
  拒绝 —— 半个字符
  没法编成 UTF-8，悄悄丢掉才真的糟。
- 数字里的指数（`1e-5`）：指名拒绝，因为 `parse_float64` 不收指数（见
  `core/text.mojo`），而"不收"必须显式说出来。

还没有的东西
------------
**没有 chat template。** `messages` 的内容按出现顺序用换行拼起来当 prompt，角色
（`role`）被读掉但不参与 —— 真正的 chat template（`chat_template.mojo`）还没做，在
这之前假装角色起了作用是更糟的那种假。这一点写在响应之外：见文件末尾的
`PROMPT_NOTE`（会随 `/health` 一起返回）。
"""

from std.collections import List

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    ERR_PARSE,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.text import parse_float64, parse_int
from alofa.srv.http import (
    SSE_CONTENT_TYPE,
    HttpRequest,
    HttpResponse,
    bytes_of_text,
    json_response,
)
from alofa.srv.engine_thread import Twinable, heap_place, heap_take
from alofa.srv.server import Handler
from alofa.srv.sse import StreamToken, sse_done, sse_frame

comptime DEFAULT_MAX_TOKENS = 32
comptime DEFAULT_TEMPERATURE = Float64(1.0)
comptime MAX_MESSAGES = 64
comptime MAX_MAX_TOKENS = 256

comptime PROMPT_NOTE = "messages are joined with newlines; there is no chat template yet"

# `created` 字段：OpenAI 要求它是秒级时间戳，而 Mojo 1.0 的 stdlib 里没有墙上时钟
# （`core/ffi` 只有单调时钟）。填 0 而不是编一个：编出来的时间戳会让"这个响应是
# 什么时候生成的"变成一个假的答案。
comptime CREATED = 0

# 第 `index` 份 handler 的 id 从 `1 + index * ID_STRIDE` 开始（见
# `ChatHandler.spawn_twin`）。它要**够大**：一份 handler 数过 ID_STRIDE 条请求
# 之后就会撞上下一份的号 —— 一百万条之后才可能撞，而"撞了"的后果只是两个 id 重
# 复，不是崩溃。上界由 `ALOFA_ENGINE_THREADS` 的 16 兜着（最大号 1600 万）。
comptime ID_STRIDE = 1_000_000

# 一条流走到哪一步了。用**一个**整数而不是几个布尔：几个布尔能表达的状态比
# 合法的帧序列多得多，多出来的那些是"两个来源的真值打架"的状态 —— 那种状态
# 不该存在，也就不该能表示。
comptime STREAM_IDLE = 0  # 没有流在途：`stream_next` 返回空串
comptime STREAM_ROLE = 1  # 下一帧是只有 role 的首帧
comptime STREAM_TEXT = 2  # 下一帧由 service 的下一步决定
comptime STREAM_DONE = 3  # 下一帧是 `data: [DONE]`，之后回到 IDLE


struct Completion(Copyable, Movable):
    """一次生成的结果。`finish_reason` 不在这里 —— 它取决于"要了多少"和"给了多少"，
    只有路由那一层同时知道两个数。"""

    var text: String
    var prompt_tokens: Int
    var completion_tokens: Int

    def __init__(out self, text: String, prompt_tokens: Int, completion_tokens: Int):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


trait Service:
    """生成一段文本的东西。路由只依赖这个，所以门里可以塞一个不需要权重的替身。

    两条路：`complete` 一次给整段；`stream_begin` + `stream_next` 一次给一步。
    流式是**迭代器式**的（不是回调）：回调要把 socket 一路传进生成循环里，那
    样"生成"和"写"就再也分不开，而分得开正是这一层能被无权重门钉住的原因。
    """

    def complete(
        mut self, prompt: String, max_tokens: Int, temperature: Float64
    ) raises -> Completion:
        ...

    def stream_begin(
        mut self, prompt: String, max_tokens: Int, temperature: Float64
    ) raises:
        """开始一次流式生成：**只做校验与准备，不走前向**。

        前向留在第一次 `stream_next` 里（惰性），因为这一步是唯一能在"响应头
        已经发出去之前"把请求的问题回成 400 的地方 —— 头发出去之后再报错，
        客户端拿到的就是一个没有 `[DONE]` 的流（见文件头）。
        """
        ...

    def stream_next(mut self) raises -> StreamToken:
        """走一步，返回这一步的文本；`done=True` 表示生成结束。

        结束之后**继续调用**必须仍然返回 `done=True`（而不是报错或重新开始）：
        路由什么时候停由它自己决定（它同时知道要了多少），它可能会多问一次。
        """
        ...


struct ChatRequest(Movable):
    var model: String
    var prompt: String
    var max_tokens: Int
    var temperature: Float64
    var stream: Bool

    def __init__(out self):
        self.model = ""
        self.prompt = ""
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.stream = False


def _skip_ws(imm raw: List[UInt8], mut at: Int):
    while at < len(raw) and (
        raw[at] == 32 or raw[at] == 9 or raw[at] == 10 or raw[at] == 13
    ):
        at += 1


def _expect(imm raw: List[UInt8], mut at: Int, expected: Int) raises:
    if at >= len(raw) or Int(raw[at]) != expected:
        raise AlofaError(
            ERR_PARSE,
            "unexpected byte in the JSON body",
            "at=" + String(at) + " want=" + String(expected),
        )
    at += 1


def _hex_value(byte: UInt8) -> Int:
    var v = Int(byte)
    if v >= 48 and v <= 57:
        return v - 48
    if v >= 97 and v <= 102:
        return v - 87
    if v >= 65 and v <= 70:
        return v - 55
    return -1


def _read_hex4(imm raw: List[UInt8], mut at: Int) raises -> Int:
    if at + 4 > len(raw):
        raise AlofaError(ERR_PARSE, "truncated \\u escape", "at=" + String(at))
    var value = 0
    for _ in range(4):
        var digit = _hex_value(raw[at])
        if digit < 0:
            raise AlofaError(
                ERR_PARSE, "not a hex digit in a \\u escape", "at=" + String(at)
            )
        value = value * 16 + digit
        at += 1
    return value


def _append_utf8(mut sink: List[UInt8], cp: Int) raises:
    if cp < 0x80:
        sink.append(UInt8(cp))
    elif cp < 0x800:
        sink.append(UInt8(0xC0 | (cp >> 6)))
        sink.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        sink.append(UInt8(0xE0 | (cp >> 12)))
        sink.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        sink.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        raise AlofaError(
            ERR_UNSUPPORTED,
            "code points outside the BMP are not accepted",
            "cp=" + String(cp),
        )


def _read_string(imm raw: List[UInt8], mut at: Int, mut sink: List[UInt8]) raises:
    """读一个 JSON 字符串（含转义），填进 `sink`。"""
    _expect(raw, at, 34)  # '"'
    while True:
        if at >= len(raw):
            raise AlofaError(ERR_PARSE, "unterminated string in the JSON body", "")
        var b = raw[at]
        at += 1
        if b == 34:
            return
        if b == 92:  # '\'
            if at >= len(raw):
                raise AlofaError(ERR_PARSE, "truncated escape in the JSON body", "")
            var esc = raw[at]
            at += 1
            if esc == 34:
                sink.append(UInt8(34))
            elif esc == 92:
                sink.append(UInt8(92))
            elif esc == 47:
                sink.append(UInt8(47))
            elif esc == 98:
                sink.append(UInt8(8))
            elif esc == 102:
                sink.append(UInt8(12))
            elif esc == 110:
                sink.append(UInt8(10))
            elif esc == 114:
                sink.append(UInt8(13))
            elif esc == 116:
                sink.append(UInt8(9))
            elif esc == 117:  # 'u'
                var cp = _read_hex4(raw, at)
                if cp >= 0xD800 and cp <= 0xDFFF:
                    raise AlofaError(
                        ERR_UNSUPPORTED,
                        "surrogate halves are not accepted",
                        "cp=" + String(cp),
                    )
                _append_utf8(sink, cp)
            else:
                raise AlofaError(
                    ERR_PARSE,
                    "unknown escape in the JSON body",
                    "escape=" + String(Int(esc)),
                )
        elif b < 32:
            raise AlofaError(
                ERR_PARSE,
                "control character inside a JSON string",
                "byte=" + String(Int(b)),
            )
        else:
            sink.append(b)


def _read_number_text(imm raw: List[UInt8], mut at: Int, mut sink: List[UInt8]) raises:
    """读一个定长写法的数字的**原文**（`-1.5`），交给 `parse_int` / `parse_float64`。"""
    var start = at
    if at < len(raw) and (raw[at] == 45 or raw[at] == 43):  # '-' '+'
        at += 1
    while at < len(raw) and (
        (raw[at] >= 48 and raw[at] <= 57) or raw[at] == 46
    ):
        at += 1
    if at == start:
        raise AlofaError(ERR_PARSE, "not a number in the JSON body", "at=" + String(at))
    for i in range(start, at):
        sink.append(raw[i])
    if at < len(raw) and (raw[at] == 101 or raw[at] == 69):  # 'e' 'E'
        raise AlofaError(
            ERR_UNSUPPORTED,
            "numbers in exponent notation are not accepted",
            "at=" + String(at),
        )


def _read_bool(imm raw: List[UInt8], mut at: Int) raises -> Bool:
    if at + 4 <= len(raw) and (
        raw[at] == 116
        and raw[at + 1] == 114
        and raw[at + 2] == 117
        and raw[at + 3] == 101
    ):
        at += 4
        return True
    if at + 5 <= len(raw) and (
        raw[at] == 102
        and raw[at + 1] == 97
        and raw[at + 2] == 108
        and raw[at + 3] == 115
        and raw[at + 4] == 101
    ):
        at += 5
        return False
    raise AlofaError(ERR_PARSE, "not a boolean in the JSON body", "at=" + String(at))


def _skip_string(imm raw: List[UInt8], mut at: Int) raises:
    _expect(raw, at, 34)
    while at < len(raw):
        var b = raw[at]
        at += 1
        if b == 92:
            at += 1
        elif b == 34:
            return
    raise AlofaError(ERR_PARSE, "unterminated string in the JSON body", "")


def _skip_value(imm raw: List[UInt8], mut at: Int) raises:
    """跳过一个我们不认识的键的值：对象、数组、字符串、数字或字面量。

    必须真的按结构跳（而不是找到下一个逗号就停）：`"a":{"b":1},"c":2` 里的第一个逗号
    在对象里面。
    """
    _skip_ws(raw, at)
    if at >= len(raw):
        raise AlofaError(ERR_PARSE, "the JSON body ends inside a value", "")
    var b = raw[at]
    if b == 34:
        _skip_string(raw, at)
        return
    if b == 123 or b == 91:  # '{' '['
        var depth = 0
        while at < len(raw):
            var c = raw[at]
            if c == 34:
                _skip_string(raw, at)
                continue
            if c == 123 or c == 91:
                depth += 1
            elif c == 125 or c == 93:  # '}' ']'
                depth -= 1
                at += 1
                if depth == 0:
                    return
                continue
            at += 1
        raise AlofaError(ERR_PARSE, "unterminated container in the JSON body", "")
    # 数字或字面量：读到底层分隔符。
    while at < len(raw) and (
        raw[at] != 44 and raw[at] != 125 and raw[at] != 93 and raw[at] != 32
    ):
        at += 1


def _read_messages(imm raw: List[UInt8], mut at: Int, mut sink: List[String]) raises:
    """读 `messages` 数组，把每条消息的 `content` 按序收进 `sink`。"""
    _expect(raw, at, 91)  # '['
    _skip_ws(raw, at)
    if at < len(raw) and raw[at] == 93:  # ']'
        at += 1
        return
    while True:
        _skip_ws(raw, at)
        _expect(raw, at, 123)  # '{'
        var content = List[UInt8]()
        var has_content = False
        while True:
            _skip_ws(raw, at)
            var key = List[UInt8]()
            _read_string(raw, at, key)
            _skip_ws(raw, at)
            _expect(raw, at, 58)  # ':'
            _skip_ws(raw, at)
            var name = String(unsafe_from_utf8=key)
            if name == "content":
                _read_string(raw, at, content)
                has_content = True
            else:
                # `role` 也走这里：这一版没有 chat template（见文件头）。
                _skip_value(raw, at)
            _skip_ws(raw, at)
            if at >= len(raw):
                raise AlofaError(ERR_PARSE, "the JSON body ends inside a message", "")
            if raw[at] == 44:  # ','
                at += 1
                continue
            if raw[at] == 125:  # '}'
                at += 1
                break
            raise AlofaError(
                ERR_PARSE, "unexpected byte inside a message", "at=" + String(at)
            )
        if has_content:
            if len(sink) >= MAX_MESSAGES:
                raise AlofaError(
                    ERR_CAPACITY,
                    "too many messages in one request",
                    "n=" + String(len(sink)),
                )
            sink.append(String(unsafe_from_utf8=content))
        _skip_ws(raw, at)
        if at >= len(raw):
            raise AlofaError(ERR_PARSE, "the JSON body ends inside messages", "")
        if raw[at] == 44:  # ','
            at += 1
            continue
        if raw[at] == 93:  # ']'
            at += 1
            return
        raise AlofaError(
            ERR_PARSE, "unexpected byte inside messages", "at=" + String(at)
        )


def parse_chat_request(imm body: String) raises -> ChatRequest:
    """扫一个 `/v1/chat/completions` 的请求体。缺 `messages` 或一条都没有就是错。"""
    var raw = bytes_of_text(body)
    var at = 0
    _skip_ws(raw, at)
    _expect(raw, at, 123)  # '{'

    var request = ChatRequest()
    var messages = List[String]()

    while True:
        _skip_ws(raw, at)
        if at >= len(raw):
            raise AlofaError(ERR_PARSE, "the JSON body ends inside the object", "")
        if raw[at] == 125:  # '}'
            at += 1
            break
        var key = List[UInt8]()
        _read_string(raw, at, key)
        _skip_ws(raw, at)
        _expect(raw, at, 58)  # ':'
        _skip_ws(raw, at)
        var name = String(unsafe_from_utf8=key)
        if name == "model":
            var value = List[UInt8]()
            _read_string(raw, at, value)
            request.model = String(unsafe_from_utf8=value)
        elif name == "messages":
            _read_messages(raw, at, messages)
        elif name == "max_tokens":
            var digits = List[UInt8]()
            _read_number_text(raw, at, digits)
            request.max_tokens = parse_int(String(unsafe_from_utf8=digits))
        elif name == "temperature":
            var digits = List[UInt8]()
            _read_number_text(raw, at, digits)
            request.temperature = parse_float64(String(unsafe_from_utf8=digits))
        elif name == "stream":
            request.stream = _read_bool(raw, at)
        else:
            _skip_value(raw, at)
        _skip_ws(raw, at)
        if at >= len(raw):
            raise AlofaError(ERR_PARSE, "the JSON body ends inside the object", "")
        if raw[at] == 44:  # ','
            at += 1
            continue
        if raw[at] == 125:  # '}'
            at += 1
            break
        raise AlofaError(
            ERR_PARSE, "unexpected byte inside the JSON object", "at=" + String(at)
        )

    _skip_ws(raw, at)
    if at != len(raw):
        raise AlofaError(
            ERR_PARSE, "trailing bytes after the JSON object", "at=" + String(at)
        )

    if len(messages) == 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT, "the request carries no message content", ""
        )
    if request.max_tokens <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "max_tokens must be positive",
            "value=" + String(request.max_tokens),
        )
    if request.max_tokens > MAX_MAX_TOKENS:
        raise AlofaError(
            ERR_CAPACITY,
            "max_tokens is larger than this server generates",
            "value=" + String(request.max_tokens),
        )
    if request.temperature < 0.0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "temperature must not be negative",
            "value=" + String(request.temperature),
        )

    var prompt = ""
    for i in range(len(messages)):
        if i > 0:
            prompt += "\n"
        prompt += messages[i]
    request.prompt = prompt
    return request^


def escape_json(imm text: String) -> String:
    """按 JSON 转义一个字符串值。非 ASCII 的 UTF-8 原样输出（JSON 允许）。"""
    var raw = text.as_bytes()
    var out = List[UInt8](capacity=len(raw) + 8)
    for i in range(len(raw)):
        var b = raw[i]
        if b == 34:
            out.append(UInt8(92))
            out.append(UInt8(34))
        elif b == 92:
            out.append(UInt8(92))
            out.append(UInt8(92))
        elif b == 10:
            out.append(UInt8(92))
            out.append(UInt8(110))
        elif b == 13:
            out.append(UInt8(92))
            out.append(UInt8(114))
        elif b == 9:
            out.append(UInt8(92))
            out.append(UInt8(116))
        elif b == 8:
            out.append(UInt8(92))
            out.append(UInt8(98))
        elif b == 12:
            out.append(UInt8(92))
            out.append(UInt8(102))
        elif b < 32:
            out.append(UInt8(92))
            out.append(UInt8(117))
            out.append(UInt8(48))
            out.append(UInt8(48))
            out.append(_hex_digit(Int(b) // 16))
            out.append(_hex_digit(Int(b) % 16))
        else:
            out.append(b)
    return String(unsafe_from_utf8=out)


def _hex_digit(value: Int) -> UInt8:
    if value >= 10:
        return UInt8(87 + value)
    return UInt8(48 + value)


def completion_json(
    imm id: String,
    imm model: String,
    imm text: String,
    prompt_tokens: Int,
    completion_tokens: Int,
    imm finish_reason: String,
) -> String:
    """一个非流式chat completion 的响应体。

    `finish_reason` 由调用方给：它取决于"要了多少、给了多少"，只有那一层同时知道。
    """
    return (
        "{"
        + "\"id\":\"" + id + "\","
        + "\"object\":\"chat.completion\","
        + "\"created\":" + String(CREATED) + ","
        + "\"model\":\"" + escape_json(model) + "\","
        + "\"choices\":[{"
        + "\"index\":0,"
        + "\"message\":{\"role\":\"assistant\",\"content\":\""
        + escape_json(text)
        + "\"},"
        + "\"finish_reason\":\"" + finish_reason + "\""
        + "}],"
        + "\"usage\":{"
        + "\"prompt_tokens\":" + String(prompt_tokens) + ","
        + "\"completion_tokens\":" + String(completion_tokens) + ","
        + "\"total_tokens\":" + String(prompt_tokens + completion_tokens)
        + "}"
        + "}"
    )


def chunk_role_json(imm id: String, imm model: String) -> String:
    """流式第一帧：一个只有 `role` 的 delta。

    OpenAI 的第一帧不带文本，只说"assistant 要说话了"。少了它，客户端要么自己
    补一个角色（各家补法不同），要么把第一帧的文本当成一个新消息。
    """
    return (
        "{"
        + "\"id\":\"" + id + "\","
        + "\"object\":\"chat.completion.chunk\","
        + "\"created\":" + String(CREATED) + ","
        + "\"model\":\"" + escape_json(model) + "\","
        + "\"choices\":[{"
        + "\"index\":0,"
        + "\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
        + "\"finish_reason\":null"
        + "}]"
        + "}"
    )


def chunk_text_json(imm id: String, imm model: String, imm text: String) -> String:
    """流式的内容帧。文本必须转义：一个 token 里完全可以有引号或换行
    （`"、`\\n`），而裸换行会把一帧切成两帧（见 `srv/sse.mojo`）。"""
    return (
        "{"
        + "\"id\":\"" + id + "\","
        + "\"object\":\"chat.completion.chunk\","
        + "\"created\":" + String(CREATED) + ","
        + "\"model\":\"" + escape_json(model) + "\","
        + "\"choices\":[{"
        + "\"index\":0,"
        + "\"delta\":{\"content\":\"" + escape_json(text) + "\"},"
        + "\"finish_reason\":null"
        + "}]"
        + "}"
    )


def chunk_finish_json(
    imm id: String, imm model: String, imm finish_reason: String
) -> String:
    """流式最后一帧：空 delta + `finish_reason`。

    `finish_reason` 由调用方给（与非流式同一条规则：它取决于"要了多少、给了
    多少"，只有那一层同时知道两个数）。
    """
    return (
        "{"
        + "\"id\":\"" + id + "\","
        + "\"object\":\"chat.completion.chunk\","
        + "\"created\":" + String(CREATED) + ","
        + "\"model\":\"" + escape_json(model) + "\","
        + "\"choices\":[{"
        + "\"index\":0,"
        + "\"delta\":{},"
        + "\"finish_reason\":\"" + finish_reason + "\""
        + "}]"
        + "}"
    )


def error_json(imm message: String, imm kind: String, imm code: String) -> String:
    """OpenAI 形状的错误体。消息进 JSON 之前必须转义：它可能来自请求本身。"""
    return (
        "{\"error\":{"
        + "\"message\":\"" + escape_json(message) + "\","
        + "\"type\":\"" + escape_json(kind) + "\","
        + "\"code\":\"" + escape_json(code) + "\""
        + "}}"
    )


def health_json(imm model: String, max_tokens: Int) -> String:
    """`/health` 的响应体。`notes` 里那句是"还没有 chat template"的自白 —— 它必须
    能被不读源码的人看见。"""
    return (
        "{\"status\":\"ok\","
        + "\"model\":\"" + escape_json(model) + "\","
        + "\"max_tokens\":" + String(max_tokens) + ","
        + "\"streaming\":true,"
        + "\"notes\":\"" + PROMPT_NOTE + "\"}"
    )


struct ChatHandler[S: Service & Deinitable & Movable & Twinable](Handler, Twinable):
    """把 `Service` 接到 HTTP 上：`/health` 与 `/v1/chat/completions`（非流式）。

    参数化在 `Service` 上而不是直接持有模型，是为了让门能塞一个**不需要 1.9 GB 权重**
    的替身进去 —— HTTP 的机械部分（分帧、keep-alive、状态码）不该靠加载真模型才能测。
    """

    var service: Self.S
    var model: String
    var next_id: Int
    # 一次流式请求的在途状态。**一份 handler 一次只处理一条**连接（engine 线程池里
    # 是"每条线程一份 handler"，所以每条线程一次一条流），所以不需要按连接分开存
    # —— 这个前提是明写的：等哪天一份 handler 同时持有多条流，这里必须改成按连接
    # 索引，而那时改不动的代价是"两条流互相覆盖对方的 id"。
    var stream_id: String
    var stream_model: String
    var stream_max: Int
    var stream_count: Int
    var stream_phase: Int

    def __init__(out self, var service: Self.S, model: String):
        self.service = service^
        self.model = model
        self.next_id = 1
        self.stream_id = ""
        self.stream_model = ""
        self.stream_max = 0
        self.stream_count = 0
        self.stream_phase = STREAM_IDLE

    def spawn_twin(self, index: Int) raises -> Int:
        """再造一份（engine 线程池：每条线程一份，见 `srv/engine_thread.mojo`）。

        `Service` 也得能造一份：生成发生在**它**里面，而两条线程共用一份
        `Service` 就是共用一份 KV —— 症状只是"偶尔答错一次"，而那是最难查的一类。

        ⚠️ `next_id` 必须**错开**（`ID_STRIDE`）：它是这一份 handler 自己的计数器，
        两份都从 1 开始 = 两条并发生成交回同一个 `chatcmpl-N`，而客户端按 id 去重
        就会安静地丢一条。它**不能**改成"共享一个计数器"：那样编号的顺序变成了
        "哪条线程先跑到那一行"，于是同一个请求在 `engines=1` 与 `engines=2` 下
        编号不同 —— 而且是随机的，门就没法拿它做逐字节比较了。
        """
        var svc = heap_take[Self.S](self.service.spawn_twin(index))
        var twin = ChatHandler(svc^, self.model)
        twin.next_id = 1 + index * ID_STRIDE
        return heap_place(twin^)

    def handle(mut self, req: HttpRequest) raises -> HttpResponse:
        var path = req.path()
        if path == "/health":
            return json_response(200, health_json(self.model, MAX_MAX_TOKENS), False)

        if path != "/v1/chat/completions":
            return json_response(
                404,
                error_json(
                    "no route for " + path, "invalid_request_error", "not_found"
                ),
                False,
            )
        if req.method != "POST":
            return json_response(
                405,
                error_json(
                    "use POST for " + path,
                    "invalid_request_error",
                    "method_not_allowed",
                ),
                False,
            )

        var chat: ChatRequest
        try:
            chat = parse_chat_request(req.body)
        except err:
            # 请求体的错是**对端**的错，所以是 400 而不是 500；把消息带回去是因为
            # "哪个字节不对"正是请求方唯一能据此修的东西。
            return json_response(
                400,
                error_json(String(err), "invalid_request_error", "invalid_body"),
                False,
            )

        if chat.stream:
            return self.begin_stream(chat)

        var completion = self.service.complete(
            chat.prompt, chat.max_tokens, chat.temperature
        )
        var model = self.model
        if chat.model.byte_length() > 0:
            model = chat.model
        var finish = "stop"
        if completion.completion_tokens >= chat.max_tokens:
            finish = "length"
        var id = "chatcmpl-" + String(self.next_id)
        self.next_id += 1
        return json_response(
            200,
            completion_json(
                id,
                model,
                completion.text,
                completion.prompt_tokens,
                completion.completion_tokens,
                finish,
            ),
            False,
        )

    def begin_stream(mut self, imm chat: ChatRequest) raises -> HttpResponse:
        """`stream=true` 的响应：**一个头**，帧由 `stream_next` 逐条吐。

        为什么 `stream_begin` 在这里调、前向却不在：这一步只做校验与准备，是
        唯一还能把请求的问题回成 400 的地方（响应头还没发出去）。前向留到第一
        次 `stream_next` —— 头发出去之后除了"给流一个终点"没有别的退路。
        """
        try:
            self.service.stream_begin(chat.prompt, chat.max_tokens, chat.temperature)
        except err:
            # `stream_begin` 只做校验，所以这里的错都是**请求**的错（prompt 分词
            # 为空 / 过长）—— 而不是"服务坏了"。回 400 且**不关连接**：客户端可以
            # 立刻改一个 prompt 重试。
            return json_response(
                400,
                error_json(String(err), "invalid_request_error", "invalid_body"),
                False,
            )

        var model = self.model
        if chat.model.byte_length() > 0:
            model = chat.model
        self.stream_model = model
        self.stream_id = "chatcmpl-" + String(self.next_id)
        self.next_id += 1
        self.stream_max = chat.max_tokens
        self.stream_count = 0
        self.stream_phase = STREAM_ROLE
        # 体是空的：帧不在响应里。流式响应没有 `Content-Length`（长度未知），
        # 靠关连接定界，所以 `close=True` 是这条响应的分帧方式，不是建议。
        return HttpResponse(200, "", SSE_CONTENT_TYPE, True)

    def stream_next(mut self) raises -> String:
        """流的下一帧；空串 = 流结束（服务循环据此停手）。

        每一条路都保证**有终点**：生成出错给一帧 `error` 再 `[DONE]`；service
        交了空文本却不说结束，也当作结束 —— 否则一条流会一直问下去，而客户端
        那边就表现为"永远等不到 `[DONE]`"（见文件头：这是最糟的失败形态）。
        """
        if self.stream_phase == STREAM_IDLE:
            return ""
        if self.stream_phase == STREAM_ROLE:
            self.stream_phase = STREAM_TEXT
            return sse_frame(chunk_role_json(self.stream_id, self.stream_model))
        if self.stream_phase == STREAM_TEXT:
            var token: StreamToken
            try:
                token = self.service.stream_next()
            except err:
                self.stream_phase = STREAM_DONE
                return sse_frame(
                    error_json(String(err), "server_error", "stream_failed")
                )
            # 上限是路由在守（它知道要了多少）：service 不说结束也不能没完。
            var hit_limit = self.stream_count >= self.stream_max
            if token.text.byte_length() > 0 and not token.done:
                self.stream_count += 1
                return sse_frame(
                    chunk_text_json(self.stream_id, self.stream_model, token.text)
                )
            var finish = "stop"
            if hit_limit:
                finish = "length"
            self.stream_phase = STREAM_DONE
            return sse_frame(
                chunk_finish_json(self.stream_id, self.stream_model, finish)
            )
        self.stream_phase = STREAM_IDLE
        return sse_done()
