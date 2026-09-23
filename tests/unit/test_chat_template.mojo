"""chat template 的差分门：** ours 与 transformers 自己的 Jinja 引擎**。

每一行的参考答案都由 `scripts/dump_chat_template.py` 导出 —— 它拿 Hugging Face 缓存里
那份**真正的** `chat_template`（不是我们复述的那一版）跑 `apply_chat_template`，于是这
扇门不可能变成"自己跟自己对"：

角色要不要出现、出现几次、拼接顺序、默认 system 那段的插入时机，全是那个模板说的。

参考答案每一词条两列，两侧也就分两件事对：

1. **渲染**：我们拼出来的文本必须**逐字节**等于参考答案；
2. **切分**：这段文本喂给**本仓库自己的 tokenizer**，切出的 id 必须**逐个**等于
   参考答案的 id。

少了第 2 条，"渲染对了"和"渲染对了而且能喂进去"就没区别 —— 例如 `<|im_start|>`
这两个 marker 如果没被当 added token 切出来，第 1 条照样绿。

为什么负面例子要写到这么具体（N1–N5）
----------------------------------
这块的失败方式极其安静：它不会崩，也不会报错，只会让模型答得不一样一点点。所以
每一条都配一个能被 FN (false negative) 触发的对照。

跑法：

    pixi run mojo run -O0 -I src tests/unit/test_chat_template.mojo
"""

from std.io import FileHandle
from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import ERR_PARSE, AlofaError
from alofa.core.text import parse_int
from alofa.tokenizer.chat_template import (
    GENERATION_PROMPT,
    IM_END,
    IM_START,
    QWEN25_DEFAULT_SYSTEM,
    known_role,
    render_chatml,
)
from alofa.tokenizer.tokenizer import load_tokenizer

comptime FIXTURE_DIR = "tests/fixtures/qwen2.5-0.5b"
comptime CASES_PATH = FIXTURE_DIR + "/chat_template.tsv"
comptime TOKENIZER_DIR = FIXTURE_DIR

# 夹具里至少要有这么多词条。生成少了必须红 —— 而不是悄悄把门槛降下来。
comptime MIN_CASES = 10

comptime LF = 0x0A
comptime TAB = 0x09


def read_text(imm path: String) raises -> String:
    var handle = FileHandle(path, "r")
    var content = handle.read()
    handle.close()
    return content^


def bytes_of(imm source: String) -> List[UInt8]:
    """一次物化。`String.as_bytes()` 每次调用都重建，直接按字节索引是 O(n²)。"""
    var raw = source.as_bytes()
    var out = List[UInt8](capacity=len(raw))
    for i in range(len(raw)):
        out.append(raw[i])
    return out^


def next_field(imm raw: List[UInt8], mut at: Int) -> String:
    """读一个字段（到 TAB 或 LF 为止）。

    不认转义 —— 夹具里的文本一律 hex 编码，字面换行与制表符进不了文本之内，于是
    这块不需要一个"逃逸规则怎么协商"的约定（那种约定本身就是一类 bug）。
    """
    var out = List[UInt8]()
    while at < len(raw) and raw[at] != TAB and raw[at] != LF:
        out.append(raw[at])
        at += 1
    at += 1
    return String(unsafe_from_utf8=out)


def hex_value(byte: UInt8) raises -> Int:
    var v = Int(byte)
    if v >= 48 and v <= 57:
        return v - 48
    if v >= 97 and v <= 102:
        return v - 87
    raise AlofaError(
        ERR_PARSE, "a fixture byte is not a hex digit", "byte=" + String(v)
    )


def decode_hex(imm hexed: String, mut sink: List[UInt8]) raises:
    var raw = hexed.as_bytes()
    if len(raw) % 2 != 0:
        raise AlofaError(
            ERR_PARSE, "a hex field has an odd number of digits", "field=" + hexed
        )
    var i = 0
    while i < len(raw):
        sink.append(UInt8(hex_value(raw[i]) * 16 + hex_value(raw[i + 1])))
        i += 2


def parse_ids(imm text: String) raises -> List[Int]:
    """把 `151644,8948,198` 读成 id 列表。"""
    var raw = text.as_bytes()
    var out = List[Int]()
    var digits = List[UInt8]()
    var i = 0
    while i < len(raw):
        if raw[i] == 44:  # ','
            out.append(parse_int(String(unsafe_from_utf8=digits^)))
            digits = List[UInt8]()
        else:
            digits.append(raw[i])
        i += 1
    if len(digits) > 0:
        out.append(parse_int(String(unsafe_from_utf8=digits^)))
    return out^


def count_needle(imm hay: String, imm needle: String) -> Int:
    """数 needle 在 hay 里出现了几次（不重叠）。

    自己写而不是 `find` 循环用是因为 `find` 有没有"从某处开始"的重载这里不能假设 ——
    而"数了几次"这件事本身就是 N4 那条对照的全部。
    """
    var h = hay.as_bytes()
    var p = needle.as_bytes()
    if len(p) == 0:
        return 0
    var count = 0
    var i = 0
    while i + len(p) <= len(h):
        var same = True
        var k = 0
        while k < len(p):
            if h[i + k] != p[k]:
                same = False
                break
            k += 1
        if same:
            count += 1
            i += len(p)
        else:
            i += 1
    return count


def refused(imm roles: List[String], imm contents: List[String]) -> Bool:
    """render 没得商量地抛错 = True。任何别的结果（包括"拼出来了"）都算它没有通过 ——
    那是这扇门最不该放行的事情。"""
    try:
        _ = render_chatml(roles, contents)
    except err:
        _ = String(err)
        return True
    return False


def test_every_case_matches_the_reference_engine() raises:
    """十个词条逐个对：文本逐字节、token id 逐个。**任何一处不等就红**。"""
    var content = read_text(CASES_PATH)
    var raw = bytes_of(content)
    var tokenizer = load_tokenizer(TOKENIZER_DIR)
    var at = 0
    var cases = 0
    while at < len(raw):
        var name = next_field(raw, at)
        if name.byte_length() == 0:
            break
        var count = parse_int(next_field(raw, at))
        var roles = List[String]()
        var contents = List[String]()
        var i = 0
        while i < count:
            roles.append(next_field(raw, at))
            var content_bytes = List[UInt8]()
            decode_hex(next_field(raw, at), content_bytes)
            contents.append(String(unsafe_from_utf8=content_bytes^))
            i += 1
        var expected_text = List[UInt8]()
        decode_hex(next_field(raw, at), expected_text)
        var expected_ids = parse_ids(next_field(raw, at))

        var rendered = render_chatml(roles, contents)
        assert_true(
            rendered == String(unsafe_from_utf8=expected_text^),
            "the rendered text differs in case " + name + ": " + rendered,
        )

        var got = tokenizer.encode(rendered)
        assert_true(
            len(got) == len(expected_ids),
            "the token count differs in case "
            + name
            + ": got "
            + String(len(got))
            + " want "
            + String(len(expected_ids)),
        )
        var j = 0
        while j < len(got):
            assert_equal(got[j], expected_ids[j])
            j += 1
        cases += 1
    assert_true(cases >= MIN_CASES, "cases=" + String(cases))


def test_an_unknown_role_is_refused() raises:
    """N1：`tool` 这类角色必须被拒绝，不许拼成 `<|im_start|>tool`。

    拼出去的话错得极其安静：模型照常回答，只是把一段 tool 输出当成人话读。
    """
    assert_true(known_role("user"), "user must be known")
    assert_true(known_role("assistant"), "assistant must be known")
    assert_true(known_role("system"), "system must be known")
    assert_true(not known_role("tool"), "tool must not be accepted yet")
    var roles = List[String]()
    roles.append("user")
    roles.append("tool")
    var contents = List[String]()
    contents.append("hi")
    contents.append("{}")
    assert_true(refused(roles, contents), "an unknown role must be refused")


def test_no_message_at_all_is_refused() raises:
    """N2：空列表必须报错，而不是产出"只有 system 那一段"的半截 prompt。

    半截 prompt 会像一个正常的请求一样被答完 —— 而它没有 user 内容。
    """
    var roles = List[String]()
    var contents = List[String]()
    assert_true(refused(roles, contents), "nothing to render must be refused")


def test_mismatched_lengths_are_refused() raises:
    """N3：roles/contents 不等长是**内部**形状错，不许伪装成请求缺字段。"""
    var roles = List[String]()
    roles.append("user")
    roles.append("user")
    var contents = List[String]()
    contents.append("only one")
    assert_true(refused(roles, contents), "a shape error must be refused")


def test_the_leading_system_is_not_emitted_twice() raises:
    """N4：模板里 `loop.first` 那条判据 —— 首条 system 只在开头那段出现。

    照着多吐一遍是最容易犯的错（看起来"更完整"），而且Ids 会多出一段、模型
    看起来答得也不错。这一条把它钉成具体数字。
    """
    var roles = List[String]()
    roles.append("system")
    roles.append("user")
    var contents = List[String]()
    contents.append("Be brief.")
    contents.append("Hello")
    var rendered = render_chatml(roles, contents)
    # system + user + assistant = 3 段开头；`<|im_end|>` 只在前两段末尾 = 2 次。
    assert_equal(count_needle(rendered, IM_START), 3)
    assert_equal(count_needle(rendered, IM_END), 2)
    assert_equal(count_needle(rendered, "Be brief."), 1)
    # 首条若不是 system，系统那一段要用模板自带的那句，位置不变。
    var without_system_roles = List[String]()
    without_system_roles.append("user")
    var without_system_contents = List[String]()
    without_system_contents.append("Hello")
    var without_system = render_chatml(without_system_roles, without_system_contents)
    assert_equal(count_needle(without_system, QWEN25_DEFAULT_SYSTEM), 1)
    assert_equal(count_needle(without_system, IM_START), 3)


def test_the_generation_prompt_is_the_tail() raises:
    """N5：最后补的那一小段是"尾巴"，且只有尾巴不同。

    服务端永远要它（生成从它之后开始）；这一条同时保证「关掉」真的少了它，而不是
    少了别的东西。
    """
    var roles = List[String]()
    roles.append("user")
    var contents = List[String]()
    contents.append("Hello")
    var with_tail = render_chatml(roles, contents, True)
    var without_tail = render_chatml(roles, contents, False)
    assert_true(
        with_tail.byte_length()
        == without_tail.byte_length() + GENERATION_PROMPT.byte_length(),
        "the tail must be exactly one generation prompt long",
    )
    assert_equal(count_needle(with_tail, GENERATION_PROMPT), 1)
    assert_equal(count_needle(without_tail, GENERATION_PROMPT), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
