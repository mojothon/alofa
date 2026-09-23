"""SSE（Server-Sent Events）的帧格式：一段 JSON 进、一段字节出，中间没有 I/O。

为什么单独一个文件
------------------
流式与非流式的差别只在**分帧**：`chat.completion.chunk` 的 JSON 由
`srv/openai.mojo` 拼（JSON 的转义规则本来就在那里），而"一段 JSON 怎么变成
线上的一帧"是纯格式的事，与 OpenAI 的形状无关 —— 它可以被逐字节钉住，既不
需要权重，也不需要 socket。

为什么没有 `Content-Length`
--------------------------
流在**开始写的那一刻**长度未知。HTTP/1.1 只有三种定界方式：长度、chunked、
关连接。这里用关连接（`srv/http.mojo` 的 `serialize_stream_head`）——
chunked 也行，但它要求每帧再套一层长度前缀，而 SSE 客户端本来就按 `data:`
行切分，多一层就多一处可以写错的地方。

行结束只认 `CRLF`
-----------------
SSE 规范允许 `\\n`、`\\r`、`\\r\\n` 三种行结束；这里只用 `\\r\\n`（与这个项目
里其它 HTTP 字节一致）。混用行结束符的响应在不同客户端的 SSE 解码器上行为
不一致，而"我的客户端能解析"正是这类 bug 的样子。

半帧比拒绝更糟
--------------
`sse_frame` 对含裸换行的 JSON 直接报错而不是"修好它"：静默把换行吃掉会让
"内容里有个换行"变成"客户端少收一帧"，那是从外部查不出来的错。转义是拼
JSON 那一侧的职责（`escape_json`）。
"""

from alofa.core.error import ERR_INVALID_ARGUMENT, AlofaError
from alofa.srv.http import CRLF


struct StreamToken(Copyable, Movable):
    """生成器交出来的一个增量。

    `done=True` 表示**生成结束**：这一帧的 `text` 可以是空（最后一步没有
    产出文本，或者一步都没走）。`done` 与 `text` 分开，是因为"这一步有没有
    文本"和"生成是不是结束了"是不同的两件事 —— 合成一个字段就得用空串去
    表示"结束"，于是"生成了一个空文本的 token"没法表达。

    `waiting=True` 表示**还没轮到这一条**：它已经排上队了，但引擎里的槽位全
    占着，所以它这一步**还没有**增量 —— 这与"结束了"和"产出了一个空文本的
    token"是第三件事，混进前两个里就会变成：要么把等待当成结束（客户端拿到
    一个没有 `[DONE]` 的空连接），要么把等待当成空 token（一路空转到有人
    让出槽位）。它必须单独是一个字段，因为**这三种状态在客户端看来完全不
    同**。
    """

    var text: String
    var done: Bool
    var waiting: Bool

    def __init__(out self, text: String, done: Bool, waiting: Bool):
        self.text = text
        self.done = done
        self.waiting = waiting


def sse_frame(imm json: String) raises -> String:
    """把一段 JSON 包成一个 SSE 事件。

    `json` 里不允许裸 `CR` 或 `LF`：一个跨了两行的 `data:` 要么是两帧被粘在
    一起，要么是一帧被切成两帧 —— 无论哪种，客户端都收不到它以为收到的东西。
    """
    if json.find("\n") >= 0 or json.find("\r") >= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "an SSE frame must not carry a raw newline",
            "json=" + json,
        )
    return "data: " + json + CRLF + CRLF


def sse_comment(imm text: String) raises -> String:
    """一帧**注释**：客户端必须忽略它，但它仍然是一帧。

    它是"这条流还活着，只是这一步没有内容"的表达。服务循环只把**空串**当作
    流结束，所以等待不能靠空串说 —— 那样会被当成结束，客户端拿到一个连
    `[DONE]` 都没有的空连接。注释帧是 SSE 协议里专门留给"什么都不说"的那一格：
    有它，等待就既不是结束也不是空转。

    ⚠️ 与 `sse_frame` 同一个约束：注释行不能含裸 `CR`/`LF`。
    """
    if text.find("\n") >= 0 or text.find("\r") >= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "an SSE comment must not carry a raw newline",
            "text=" + text,
        )
    return ": " + text + CRLF + CRLF


def sse_done() raises -> String:
    """流的最后一帧。

    它是**协议**要求的收尾，不是装饰：客户端靠它区分"生成完了"和"连接断了"。
    少了它，一个正常结束的流和一个中途死掉的流在客户端看来长得一样 —— 那是
    比不流式更糟的形态，因为客户端以为自己一直在收。
    """
    return "data: [DONE]" + CRLF + CRLF
