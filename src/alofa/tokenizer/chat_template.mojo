"""把一串 `{role, content}` 渲染成模型认得的那一段 —— **照着 HF 那份模板描的**。

这段排版**不是本仓库的偏好**，是 Hugging Face 缓存里
`Qwen2.5-0.5B-Instruct/tokenizer_config.json:chat_template` 那一大段 Jinja 说的。
没有 tools、没有 tool_calls 时它等于下面这些话：

  * **永远先吐一段 system**：首条消息是 `system` 就用它的 content，否则用模板自带的
    那句默认语（`QWEN25_DEFAULT_SYSTEM`）。
  * 然后逐条处理消息：`user` / `assistant` 各吐一段 `<|im_start|>role\\ncontent<|im_end|>\\n`；
    `system` 只有**不是首条**时才在循环里再吐一段（`loop.first` 那条判据）—— 所以
    首条 system **不会**被吐两遍，而夹在中间的 system 会老老实实出现。
  * 最后按 `add_generation_prompt` 补一小段 `<|im_start|>assistant\\n`，把笔交给模型。
    服务端永远要补，这里默认就是 True。

为什么写成"一个族群写死"，而不是先从模型读模板
--------------------------------------------
那种做法（`chat_template.mojo` 里做个 Jinja 子集）是 roadmap 的正文，它值得做，
但门槛是「模板字符串从哪来」：本仓库的夹具 `tokenizer.json` **没有** `chat_template`
字段（真正的那份在 HF 的 `tokenizer_config.json` 里），于是这条路今天没有可验证的输入。
写到这里为止的差别必须说清：**这里是 Qwen ChatML 的一个渲染器，不是 Jinja 引擎。**
所以默认 system 那句、`role` 的取舍都明写在常量和注释里，换族群要改这里 —— 而不是
换一个"看起来也行"的模板然后假装没差别。

判据从哪来
----------
夹具 `tests/fixtures/qwen2.5-0.5b/chat_template.tsv` 里的每一行，都是
`transformers.apply_chat_template`（它自己的 Jinja 引擎 + HF 那份模板，见
`scripts/dump_chat_template.py`）吐出来的：渲染后文本 + 同一段文本切出来的 id。
同一次调用给了两个写法（文本 + id），这样"渲染错了"和"切分错了"能被分开看见。
关键点是：它不是我们自己写的答案 —— 模板改一句话，这条门会立刻红，那就对了。
"""

from std.collections import List

from alofa.core.error import ERR_INVALID_ARGUMENT, AlofaError

comptime ROLE_SYSTEM = "system"
comptime ROLE_USER = "user"
comptime ROLE_ASSISTANT = "assistant"

comptime IM_START = "<|im_start|>"
comptime IM_END = "<|im_end|>"

# 模板自带的默认 system 语。它是**那个模板的一部分**，不是我们挑的开场白：改它等于
# 改了所有没有首条 system 的请求的实际 prompt，所以名字里写死它是哪一族的。
comptime QWEN25_DEFAULT_SYSTEM = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)

# `add_generation_prompt` 补的那一小段：把最后一支笔交给模型。
comptime GENERATION_PROMPT = "<|im_start|>assistant\n"


def known_role(imm role: String) -> Bool:
    """这个渲染器认得的角色。

    反过来即是策略：**不认得的角色拒绝渲染**，而不是按 `<|im_start|>bogus` 拼出去。
    后者看起来"更宽容"，实际是把一个契约错误变成一串看起来正常的 token —— 而
    `tool` 这种角色在模板里走的完全是另一段（`<tool_response>`），拼错了不会有错，
    只会静静地答得不一样。
    """
    if role == ROLE_SYSTEM:
        return True
    if role == ROLE_USER:
        return True
    if role == ROLE_ASSISTANT:
        return True
    return False


def render_chatml(
    imm roles: List[String],
    imm contents: List[String],
    add_generation_prompt: Bool = True,
) raises -> String:
    """渲染一串消息。`roles` 与 `contents` 平行且等长。

    * 一条消息都没有 → 报错。模板里是 `messages[0]`，它取空表的第 0 条会炸在 Jinja
      那一侧；这里要炸在我们的 API 边缘上，并且**有名字**。
    * 角色不在 {system, user, assistant} 里 → 报错，错误里带上它是第几条、值是
      什么。安静地把 `<|im_start|>` 拼给一个不认识的角色是"没有 chat template"那条
      老路改头换面回来。
    * `roles` 与 `contents` 不等长 → 报错（这是内部的形状错，不是"请求里缺了个字段"，
      两者不该共用一个名字）。

    `add_generation_prompt`：服务端永远要 True（生成从 `<|im_start|>assistant\\n`
    之后开始）；留成参数是为了让"补的那一小段"这件事在测试里能被单独关照。
    """
    if len(roles) == 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT, "there is no message to render", ""
        )
    if len(roles) != len(contents):
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "roles and contents do not line up",
            "roles=" + String(len(roles)) + " contents=" + String(len(contents)),
        )

    var out = ""

    var i = 0
    while i < len(roles):
        if not known_role(roles[i]):
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "this renderer does not know how to address that role",
                "at=" + String(i) + " role=" + roles[i],
            )
        i += 1

    # 第一段：system。首条消息若就是 system，用它的 content；否则用模板自带那句。
    out += IM_START
    out += ROLE_SYSTEM
    out += "\n"
    if roles[0] == ROLE_SYSTEM:
        out += contents[0]
    else:
        out += QWEN25_DEFAULT_SYSTEM
    out += IM_END
    out += "\n"

    # 循环：`user` / `assistant` 一律吐；`system` 只在**不是首条**时吐（`loop.first`）。
    var k = 0
    while k < len(roles):
        var role = roles[k]
        var skip = False
        if role == ROLE_SYSTEM and k == 0:
            skip = True
        if role != ROLE_SYSTEM and role != ROLE_USER and role != ROLE_ASSISTANT:
            skip = True
        if not skip:
            out += IM_START
            out += role
            out += "\n"
            out += contents[k]
            out += IM_END
            out += "\n"
        k += 1

    if add_generation_prompt:
        out += GENERATION_PROMPT
    return out
