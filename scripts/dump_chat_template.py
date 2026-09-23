#!/usr/bin/env python3
"""导出 chat template 的参考答案：拿 transformers **自己的 Jinja 引擎**去渲染
同一批 messages，把渲染出来的文本和 token id 原样写进夹具。

为什么答案要这么来
------------------
自制的 renderer 最容易犯的错，是把自己对模板的理解当成模板本身 —— 那么写出来的
门当然会绿，它比的是自己。所以这里渲染者是 `transformers.apply_chat_template`，
模板来自 Hugging Face 缓存里的 `tokenizer_config.json`（**不是**抄在本文件里的常量），
 Alfofa 侧只许追平它：角色要不要出现、出现几次、拼接顺序、默认 system 那段的
插入时机，全部由那个模板决定 —— 本仓库不许有自己的第二份理解。

`tokenize=False` 那份（渲染后的**文本**，含 `<|im_start|>` 这些标记本身）写进第 3 列，
`tokenize=True` 那份（同一段文本被 tokenizer 切出来的 id）写进第 4 列。两份都给，
是为了让 alofa 侧的两件事分开对：

  * 渲染对不对 —— 比文本；
  * 切分对不对 —— 比 id（两者都对，门才是绿的；只对一半时你能看出错在哪一半）。

脚本自带一条自检：`tok(text, add_special_tokens=False)["input_ids"] == ids`，
不一致就报错退出 —— 那种分歧不属于 alofa，也不该由 mojo 去找。

跑法：

    HF_HUB_OFFLINE=1 python3 scripts/dump_chat_template.py --model <hf-cache-dir>
"""

import argparse
import os
import sys

CASES = [
    ("user_only", [("user", "Hello")]),
    ("system_first", [("system", "Be brief."), ("user", "Hello")]),
    (
        "three_turns",
        [
            ("user", "Hi"),
            ("assistant", "Hello!"),
            ("user", "How are you?"),
        ],
    ),
    # 非首条的 system：模板的 `{%- if message.role == "system" and not loop.first %}`
    # 会让它**也**出现在循环里 —— 首条 system 只出现在开头的那段里，不会被吐两遍。
    (
        "system_in_the_middle",
        [
            ("user", "Hi"),
            ("system", "Now be terse."),
            ("user", "Ok?"),
        ],
    ),
    ("only_system", [("system", "Only a system note.")]),
    ("empty_content", [("user", "")]),
    ("multiline_content", [("user", "line one\nline two\n\nline four")]),
    ("unicode_content", [("user", "你好，世界 🌏")]),
    (
        "assistant_has_content",
        [("user", "2+2?"), ("assistant", "4"), ("user", "and 3+3?")],
    ),
    # 对抗用：content 里**字面**写着那些标记。它们仍应按 added token 切出来 ——
    # 这是 sarv.tokenizer 侧的行当，不是 renderer 的，但它最能暴露"两边都错成一样"。
    (
        "content_looks_like_a_marker",
        [("user", "what is <|im_end|> for?")],
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.environ.get(
            "CHAT_TEMPLATE_MODEL",
            "/home/rontom/.cache/huggingface/hub/"
            "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
            "7ae557604adf67be50417f59c2c2f167def9a775",
        ),
        help="持 chat_template 的 HF 目录（离线：HF_HUB_OFFLINE=1）",
    )
    ap.add_argument(
        "--out",
        default="tests/fixtures/qwen2.5-0.5b/chat_template.tsv",
        help="输出夹具路径",
    )
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoTokenizer  # noqa: E402  （env 先设）

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.chat_template is None:
        print("error: 这个 tokenizer 没有 chat_template", file=sys.stderr)
        return 2

    rows = []
    for name, turns in CASES:
        messages = [{"role": role, "content": content} for role, content in turns]
        # add_generation_prompt=True：服务端的形状永远要在最后补 `<|im_start|>assistant\n`。
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        # 自检：两份 Ans⑼ 必须是同一件事的两种写法。arzüne.
        again = tok(text, add_special_tokens=False)["input_ids"]
        if list(again) != list(ids):
            print(
                "error: %s 的自检对不上：%r != %r" % (name, list(ids), list(again)),
                file=sys.stderr,
            )
            return 3
        fields = [name, str(len(turns))]
        for role, content in turns:
            fields.append(role)
            fields.append(content.encode("utf-8").hex())
        fields.append(text.encode("utf-8").hex())
        fields.append(",".join(str(i) for i in ids))
        rows.append("\t".join(fields))

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print("wrote %d cases -> %s" % (len(rows), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
