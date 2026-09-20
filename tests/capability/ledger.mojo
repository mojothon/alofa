"""能力账本校验器 —— 把"承诺"变成可执行断言。

为什么需要它
------------
账本（`docs/plan/capability-ledger.md`）是 alofa 唯一的事实来源。但如果"事实来源"
只能靠人自觉维护，它就会退化成另一份会过期的 README —— 这正是我们在生态调研中
反复看到的失败模式（宣传 >> 实现）。所以账本**必须能被机器校验**。

校验什么
--------
三类 `verified` 变体都必须给出凭证，否则 CI 失败：

  `verified`         必须 `evidence:<path>`，且**文件必须真实存在**
                     → 能力已由可复现的测试证明

  `verified-remote`  必须 `evidence:<path>`，文件必须存在
                     → 同上，但**只能在 A100 远程验证机执行**；CI 只校验
                       "复现方式存在"，不执行（CI 上没有 A100）

  `verified-env`     必须 `probe:<command>`
                     → **环境事实**（硬件规格、驱动版本、工具链版本）。这类东西
                       天生没有测试文件，要求它必须有测试文件是不诚实的 —— 但
                       也不能因此就免检，所以要求留下**可复现的观测命令**。

其余标签（`partial` / `scaffold` / `designed-only` / `hardware-blocked` /
`missing` / `target`）不要求凭证。

为什么把 `verified-env` 从 `verified` 里拆出来
---------------------------------------------
若要求所有 `verified` 都有测试文件，§1.2 的硬件条目（"6×A100 280GB""驱动
560.35.03"）会全部失败 → 门永远红 → 迟早被 `--no-verify` 掉 → 门等于不存在。
拆开后三类各有各的凭证要求，**没有免检档**。

拼写防护
--------
形如 `` `verifed` `` 的拼错标签不会被任何规则命中，从而静默绕过检查。因此对含
`verif` 但不在已知标签集合内的反引号词单独报错。

标签必须写在反引号里（`` `verified` ``）—— 这既让人工可读，也让匹配天然避免
前缀歧义（`` `verified-remote` `` 不会被 `` `verified` `` 命中）。

N/M 快照为什么也要被机器校验
----------------------------
早期账本在"证据 / 说明"列里手写过一堆 `` test_engine_core.mojo（14/14） `` 这样的
描述。它们长得很像事实，记录的却只是**当时那一瞬间**的状态。问题在于数字会腐烂：
同一份 `test_engine_core.mojo` 在账本里曾同时写着 10/10、14/14、15/15 三个互不相容
的取值，而 CI 只看"文件在不在"，于是它们并存了很久也没被发现 —— 这正是"看似有证据、
实则已失真"的那一类话。

修法是把数字从散文里拎出来，变成**可校验的结构化后缀**：

    `evidence:tests/unit/test_scheduler.mojo?count=17`

`?count=` 是可选的，但**一旦写了就必须与实测一致**：`pixi run test` 会顺带产出
`target/test_counts.tsv`（每份套件自报的通过条数），`pixi run check-counts` 拿它去
核对。反过来也一样 —— **跑过的套件若在账本里没登记 `?count=`，同样报错**；否则
"干脆不写 count" 就成了绕过这个门的最短路径。
"""

from std.collections import List
from std.io import FileHandle
from std.os import stat


def file_exists(path: String) -> Bool:
    """用 `stat` 判断路径是否存在（不存在时会 raise，转为 False）。"""
    try:
        _ = stat(path)
        return True
    except:
        return False


def is_known_label(word: String) -> Bool:
    """已知证据标签集合。用字符串包含实现，避免依赖集合类型。"""
    var known = " verified verified-remote verified-env partial scaffold designed-only hardware-blocked missing target "
    return known.find(" " + word + " ") >= 0


def read_text(path: String) raises -> String:
    var handle = FileHandle(path, "r")
    return handle.read()


def _cut_before(s: String, sep: String) -> String:
    """截取 `s` 中第一个 `sep` 之前的部分；没有 `sep` 则原样返回。

    拆成独立函数是因为 Mojo 不允许 `String(s.split(x)[0])` 这种"既当参数又当
    构造结果"的写法（aliasing 冲突），必须经临时变量中转。
    """
    if s.find(sep) < 0:
        return s
    var parts = s.split(sep)
    var first = parts[0]
    return String(first.strip())


def extract_field(row: String, key: String, cut_space: Bool) -> String:
    """从表格行中提取 `key:<value>`。

    `cut_space=True`  → 用于 `evidence:`：路径不含空格，遇空格即止。
    `cut_space=False` → 用于 `probe:`：观测命令要能带参数
                        （如 `./scripts/a100.sh gpu`），不能按空格截断。

    约定写法是把整个 token 放进反引号：`` `evidence:tests/foo.mojo` ``
    """
    if row.find(key) < 0:
        return ""
    var parts = row.split(key)
    var rest = parts[1]
    var s = String(rest.strip())
    s = _cut_before(s, "`")
    s = _cut_before(s, "<")
    s = _cut_before(s, "|")
    s = _cut_before(s, "，")
    s = _cut_before(s, "（")
    if cut_space:
        s = _cut_before(s, " ")
    return s


def parse_count(s: String) -> Int:
    """解析十进制数字前缀；开头不是数字时返回 -1。

    不用 builtin 的 `int(String)` —— 本项目统一走这类手写解析（见 `alofa.core.text`）。
    这里允许数字后面紧跟其他字符（例如收尾的反引号），扫到第一个非数字即止。

    Mojo 的 `String` 是 UTF-8，`len(s)` 与 `s[i]` 都被语言明确禁掉了（长度与位置在
    字节 / 码点 / 字素三种口径下各不同），所以这里按**码点**遍历；遇到多字节字符
    时它不是数字，一样会在那里停住。
    """
    var digits = "0123456789"
    var value = 0
    var seen = 0
    for cp in s.codepoints():
        var d = digits.find(String(cp))
        if d < 0:
            break
        value = value * 10 + d
        seen += 1
    if seen == 0:
        return -1
    return value


def evidence_path(token: String) -> String:
    """剥掉 `?count=N` 后缀，得到真正的路径（供存在性检查使用）。"""
    if token.find("?count=") < 0:
        return token
    return _cut_before(token, "?count=")


def evidence_count(token: String) -> Int:
    """取 `?count=` 后的数字；没有该后缀返回 -1。"""
    var marker = "?count="
    if token.find(marker) < 0:
        return -1
    var parts = token.split(marker)
    var rest = parts[1]
    return parse_count(String(rest.strip()))


def load_counts(text: String) -> List[String]:
    """把 "path<TAB>count" 形式的清单切成条目表；空行跳过。"""
    var entries = List[String]()
    var lines = text.split("\n")
    for line in lines:
        var row = String(line.strip())
        if row.byte_length() == 0:
            continue
        entries.append(row)
    return entries^


def lookup_count(entries: List[String], path: String) -> Int:
    """查 path 在清单中的实测条数；本次没跑到则返回 -1。"""
    var prefix = path + "\t"
    for entry in entries:
        if entry.find(prefix) == 0:
            var parts = entry.split("\t")
            var raw = parts[1]
            return parse_count(String(raw.strip()))
    return -1


def looks_like_path(s: String) -> Bool:
    """最小合理性检查：拒绝 `evidence:待补` 这类占位符蒙混过关。"""
    if s.find(" ") >= 0:
        return False
    if s.find("/") < 0 and s.find(".") < 0:
        return False
    return True


def declared_label(row: String) -> String:
    """该行声明的 `verified` 类标签；空串表示没有声明。

    顺序无关：反引号精确匹配使 `` `verified-remote` `` 不会被 `` `verified` `` 命中。
    """
    if row.find("`verified-remote`") >= 0:
        return "verified-remote"
    if row.find("`verified-env`") >= 0:
        return "verified-env"
    if row.find("`verified`") >= 0:
        return "verified"
    return ""


def check_typos(row: String, source: String, line_no: Int) -> Int:
    """捕获形如 `` `verifed` `` 的拼写错误 —— 它不会被任何规则命中，会静默绕过。"""
    var errors = 0
    var parts = row.split("`")
    var idx = 0
    for part in parts:
        if idx % 2 == 1:
            var ws = part.strip()
            var word = String(ws)
            # 含 "/" 的是文件路径而非标签 —— `verify/roofline.mojo` 也会命中 "verif"
            if (
                word.find("/") < 0
                and word.find("verif") >= 0
                and not is_known_label(word)
            ):
                errors += 1
                print(
                    "[ledger] "
                    + source
                    + ":"
                    + String(line_no)
                    + " 未知/拼错的证据标签 `"
                    + word
                    + "` —— 它不会被任何校验规则命中"
                )
        idx += 1
    return errors


def check_stated_count(
    source: String,
    line_no: Int,
    path: String,
    wanted: Int,
    entries: List[String],
) -> Int:
    """核对账本声明的 `?count=N` 与实测条数，返回错误条数。

    两个方向都要查，缺一不可：

      wanted >= 0 → 既然声明了条数，就必须能在清单里找到且相等
      wanted < 0  → 跑过的套件却没登记，同样报错

    第二条是防绕过：若只有"写了才查"，最省事的做法就变成"干脆一个都不写"。
    """
    var got = lookup_count(entries, path)

    if got < 0:
        if wanted < 0:
            return 0
        print(
            "[ledger] "
            + source
            + ":"
            + String(line_no)
            + " 声明了 count="
            + String(wanted)
            + "，但 "
            + path
            + " 没出现在本次测试清单里 —— 它可能没被执行，或账本里写错了路径"
        )
        return 1

    if wanted < 0:
        print(
            "[ledger] "
            + source
            + ":"
            + String(line_no)
            + " "
            + path
            + " 本次跑了 "
            + String(got)
            + " 项，账本却没登记 `?count=` —— 不登记会让未来的漂移再次静默通过"
        )
        return 1

    if wanted != got:
        print(
            "[ledger] "
            + source
            + ":"
            + String(line_no)
            + " N/M 快照漂移："
            + path
            + " 账本写 "
            + String(wanted)
            + "，实测 "
            + String(got)
            + " → 跑 `pixi run ledger-sync` 刷新，或如实说明它为何变了"
        )
        return 1

    return 0


def validate_impl(
    text: String,
    source: String,
    verbose: Bool,
    counts: List[String],
) -> Int:
    """校验账本文本，返回错误条数（0 = 通过）。

    `counts` 为空   → 只做静态校验（凭证存在 + 标签拼写）。
    `counts` 非空   → 额外核对 `?count=N` 与本次实测条数；见文件顶部。
    """
    var errors = 0
    var line_no = 0
    var lines = text.split("\n")
    for line in lines:
        line_no += 1
        var stripped = line.strip()
        var row = String(stripped)
        if row.find("|") != 0:
            continue
        if row.find("---") >= 0:
            continue

        var label = declared_label(row)
        if label == "":
            errors += check_typos(row, source, line_no)
            continue

        if label == "verified-env":
            var probe = extract_field(row, "probe:", False)
            if probe == "":
                errors += 1
                print(
                    "[ledger] "
                    + source
                    + ":"
                    + String(line_no)
                    + " `verified-env` 缺少 `probe:<观测命令>`"
                )
            elif verbose:
                print("[ledger] ok (env)     " + probe)
            continue

        var evidence = extract_field(row, "evidence:", True)
        if evidence == "":
            errors += 1
            print(
                "[ledger] "
                + source
                + ":"
                + String(line_no)
                + " `"
                + label
                + "` 缺少 `evidence:<path>`"
            )
            continue

        var path = evidence_path(evidence)
        if not looks_like_path(path):
            errors += 1
            print(
                "[ledger] "
                + source
                + ":"
                + String(line_no)
                + " evidence 不是有效路径: "
                + path
            )
        elif not file_exists(path):
            errors += 1
            print(
                "[ledger] "
                + source
                + ":"
                + String(line_no)
                + " `"
                + label
                + "` 的 evidence 文件不存在: "
                + path
            )
        elif len(counts) > 0 and label != "verified-remote":
            # verified-remote 的套件只在 A100 验证机上执行，本机清单里不会有它
            errors += check_stated_count(
                source, line_no, path, evidence_count(evidence), counts
            )
        elif verbose:
            print("[ledger] ok (" + label + ") " + path)
    return errors


def validate_text(text: String, source: String, verbose: Bool) -> Int:
    """静态校验账本文本，返回错误条数（0 = 通过）。不核对 N/M 快照。"""
    return validate_impl(text, source, verbose, List[String]())


def validate_text_with_counts(
    text: String,
    source: String,
    verbose: Bool,
    counts_text: String,
) -> Int:
    """在静态校验之上，用 "path<TAB>count" 清单核对 `?count=N`。

    这里接收清单**内容**而不是路径，好让门测试能内联构造清单，不必生成真实文件。
    """
    return validate_impl(text, source, verbose, load_counts(counts_text))


def validate_ledger(path: String, verbose: Bool) raises -> Int:
    """校验指定账本文件，返回错误条数（0 = 通过）。"""
    return validate_text(read_text(path), path, verbose)


def validate_ledger_with_counts(
    path: String,
    counts_text: String,
    verbose: Bool,
) raises -> Int:
    """同上，并额外核对 `?count=N` 与实测条数是否一致。"""
    return validate_text_with_counts(read_text(path), path, verbose, counts_text)
