#!/usr/bin/env python
"""把账本里手写的 N/M 快照刷新成当场实测的结构化 `?count=N`。

用法
----
    pixi run test                      # 先落下 target/test_counts.tsv
    pixi run python scripts/ledger_sync.py --dry-run   # 先看会改什么
    pixi run python scripts/ledger_sync.py             # 真的改
    git diff docs/plan/capability-ledger.md            # 看看什么漂了

为什么需要这一步
----------------
`?count=` 一旦被 `check-counts` 核验，任何一次套件条数变化都会让 CI 变红。如果
每次都要人去数 20+ 处数字，这个门很快就会被 `--skip` 掉。它的职责就是把
"会腐烂的数字"变成"一条命令就能刷新的数字"。

它做什么
--------
1. `evidence:<path>` → `evidence:<path>?count=N`（N 取自本次实测清单）。
2. 删掉紧跟其后的 `（N/M 通过）` 之类手写描述 —— 数字现在有了唯一归处，留在散文里
   只会再次腐烂。
3. **不碰**清单里没有的套件（如 `test_model_parity.mojo` 这类重资产门，它们不在
   `pixi run test` 里），也不碰 `verified-remote`（只在 A100 上执行）。

换行：账本目前是混合换行（表格行 CRLF），所以这里按 **二进制 / keepends** 处理，
整份文件不重新 join，避免把 CRLF 洗成 LF 污染 diff。
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COUNTS = ROOT / "target" / "test_counts.tsv"
LEDGER = ROOT / "docs" / "plan" / "capability-ledger.md"

# evidence token —— 字符集必须与 ledger.mojo 的 extract_field 截断集保持一致
# （反引号 / < / | / 全角逗号 / 全角左括号 / 空格 都会终止 token）
TOKEN = re.compile(r"evidence:([^\s`<|，（]+)")

# 紧跟 evidence 的括号里的 N/M 前缀：`（13/13 通过，`、`（7/7，`、`（10/10：`。
# 开头的 named group 收掉可能出现的**闭合反引号** —— 账本约定把整个 token 包在
# 反引号里（`` `evidence:…?count=5`（5/5 通过） ``），所以括号之前其实是反引号。
NM_HEAD = re.compile(r"^(?P<quote>`*)（\d+/\d+(?:\s*通过)?[：:]?[，,]?\s*")


def load_counts(path: Path) -> dict:
    """读 "path<TAB>count" 清单；count 为 -1（没数出来）的条目忽略。"""
    counts = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        row = line.strip()
        if not row or "\t" not in row:
            continue
        file_part, _, n = row.partition("\t")
        try:
            value = int(n.strip())
        except ValueError:
            continue
        if value >= 0:
            counts[file_part.strip()] = value
    return counts


def strip_stale_snapshot(rest: str) -> str:
    """删掉 rest 开头的 `（N/M…` 前缀；若括号里只剩它，则连括号一起删。"""
    m = NM_HEAD.match(rest)
    if not m:
        return rest
    prefix = m.group("quote") or ""
    tail = rest[m.end() :]
    if tail.startswith("）"):
        # 括号里原本只有 N/M → 整组删掉，保留括号后面的文字
        return prefix + tail[1:]
    return prefix + "（" + tail


def rewrite_line(line: str, counts: dict) -> tuple:
    """把一行里的 evidence token 升级为带 count 的形式。返回 (新行, 改动数)。"""
    out = line
    changed = 0
    # 从后往前替换：已替换的部分总在当前位置之后，前身索引不会失效
    for m in reversed(list(TOKEN.finditer(line))):
        token = m.group(1)
        path = token.split("?count=")[0]
        if path not in counts:
            continue
        wanted = counts[path]
        if token.endswith("?count=" + str(wanted)):
            continue
        replacement = "evidence:{}?count={}".format(path, wanted)
        out = out[: m.start()] + replacement + strip_stale_snapshot(out[m.end() :])
        changed += 1
    return out, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印改动，不写文件")
    args = ap.parse_args()

    if not COUNTS.exists():
        print("error: 缺少 {} —— 请先跑 `pixi run test`".format(COUNTS))
        return 1

    counts = load_counts(COUNTS)
    if not counts:
        print("error: {} 里没有可用条目".format(COUNTS))
        return 1

    # newline="" + keepends：原样保住每一行自带的行尾
    raw = LEDGER.read_text(encoding="utf-8", newline="")
    lines = raw.splitlines(keepends=True)

    new_lines = []
    total = 0
    for line in lines:
        new, n = rewrite_line(line, counts)
        total += n
        new_lines.append(new)

    if total == 0:
        print("没有需要刷新的地方 —— 账本里的 count 与本次实测一致")
        return 0

    result = "".join(new_lines)
    print("改动 {} 处（涉及 {} 份套件）".format(total, len(counts)))
    for line in new_lines:
        if "evidence:" in line and "?count=" in line:
            print("  " + line.strip()[:120])

    if args.dry_run:
        print("\n[dry-run] 未写入 {}".format(LEDGER))
        return 0

    LEDGER.write_text(result, encoding="utf-8", newline="")
    print("\n已写入 {}".format(LEDGER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
