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
"""

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


def validate_text(text: String, source: String, verbose: Bool) -> Int:
    """校验账本文本，返回错误条数（0 = 通过）。"""
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
        elif not looks_like_path(evidence):
            errors += 1
            print(
                "[ledger] "
                + source
                + ":"
                + String(line_no)
                + " evidence 不是有效路径: "
                + evidence
            )
        elif not file_exists(evidence):
            errors += 1
            print(
                "[ledger] "
                + source
                + ":"
                + String(line_no)
                + " `"
                + label
                + "` 的 evidence 文件不存在: "
                + evidence
            )
        elif verbose:
            print("[ledger] ok (" + label + ") " + evidence)
    return errors


def validate_ledger(path: String, verbose: Bool) raises -> Int:
    """校验指定账本文件，返回错误条数（0 = 通过）。"""
    return validate_text(read_text(path), path, verbose)
