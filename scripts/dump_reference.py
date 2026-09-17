#!/usr/bin/env python3
"""把 HuggingFace 的 Qwen2.5-0.5B 变成 alofa 的**参考基准（oracle）**。

这份脚本是 P1 数值门与分词门的地基：它把"权威答案"从 Python 侧一次性导出成
纯文本 / 裸浮点文件，之后 Mojo 侧的实现只与这些产物对比，**不再依赖
Python 运行时**。

为什么这样做，而不是在测试里直接调 transformers：

- 差分测试需要一个**不会随被测代码一起变**的参照物。让它实时调用 Python，
  等于让参照物依赖另一套栈；导出成文件后，参照物是数据，可以进版本库、
  可以被第三方核对。
- 分词门要求 4560 条用例。用 Python 现算会让"门"的耗时与 torch 启动时间
  绑定；导出后 Mojo 侧可以在毫秒级跑完，于是它可以进 CI 每次都跑。

导出物（默认落在 `tests/fixtures/qwen2.5-0.5b/`）：

| 文件 | 内容 | 用途 |
|---|---|---|
| `vocab.tsv` | `id <tab> <token 字节的 hex>` | 词表（byte-level，hex 以免歧义） |
| `merges.tsv` | `rank <tab> left_id <tab> right_id` | BPE 合并优先级（rank 越小越先合） |
| `special.tsv` | `id <tab> <内容的 hex>` | 需要整串匹配的特殊词元 |
| `tokenizer_cases.tsv` | `<文本 hex> <tab> <id,id,...>` | 分词差分用例 |
| `prompts.json` | 提示词与其参考 ids | 数值门的输入 |
| `greedy.json` | 每道题 greedy 生成的 128 个 id | 逐 token 相等门 |
| `logits_last.f32` | 每题**末位** logits，fp32 裸数据 | 余弦 / argmax 门 |
| `manifest.json` | 形状、来源、版本 | 让产物可被第三方核对 |

用法：

    python3 scripts/dump_reference.py                    # 全量
    python3 scripts/dump_reference.py --cases-only       # 只导出分词相关（快）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import unicodedata
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B"


def _byte_level_alphabet() -> dict[int, str]:
    """GPT-2's byte -> character alphabet (OpenAI 的 byte-level 编码)。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = _byte_level_alphabet()
_BYTE_DECODER = {ch: byte for byte, ch in _BYTE_ENCODER.items()}


def token_to_bytes(token: str) -> bytes:
    """词元 -> 真实字节。

    `get_vocab()` 给的是 byte-level 的 unicode 串；直接 `token.encode()` 会得到
    "Ġ 的 UTF-8"（c4a0）而不是空格本身（20）。用错了这一步，词表里的"空格"
    会变成一个两字节的东西，所有含空格的文本都对不上。
    """
    return bytes(_BYTE_DECODER[ch] for ch in token)


# 数值门的提示词。覆盖四种真实形态：英文续写、代码、中文、ChatML 模板。
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "你好，请用一句话介绍你自己。",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]

GREEDY_TOKENS = 128
TARGET_CASES = 4560


# --------------------------------------------------------------------------
# 分词差分语料
# --------------------------------------------------------------------------

# 手工构造的对抗样本：正则预分词器最容易在这里与权威实现分叉。
ADVERSARIAL = [
    "Hello, world!",
    " leading space",
    "trailing space ",
    "multiple   spaces",
    "tab\there",
    "crlf\r\nline",
    "I'm sure it's fine — we've seen y'all'd've done it.",
    "It's 3 o'clock; they're here, aren't they?",
    "numbers: 0 1 42 007 1234567890 3.14159 1e10 0x1F",
    "punct: ... !!! ?! ,, ;; :: (( )) [[ ]] -- __",
    "url: https://example.com/path?a=1&b=2#frag",
    "email: someone@example.co.uk",
    "path: /usr/local/bin/../lib/libfoo.so",
    "code: for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }",
    "emoji: 👍 👍🏽 👨‍👩‍👧‍👦 🇨🇳 🚀",
    "cjk: 你好世界，这是一个测试。",
    "kana: こんにちは、世界！",
    "hangul: 안녕하세요 세계",
    "cyr: Привет, мир! Как дела?",
    "greek: Καλημέρα κόσμε",
    "hebrew: שלום עולם",
    "arabic: مرحبا بالعالم",
    "thai: สวัสดีชาวโลก",
    "devanagari: नमस्ते दुनिया",
    "fullwidth: ＡＢＣ１２３，。！",
    # NFC 用例：分解序列应被合成为合成形式后再分词
    "nfc: é à ñ Ámos",
    "nfc2: ç vs ç and é vs é",
    "hangul-jamo: 각 vs 각",
    "mixed: 你好Hello世界123!@#مرحبا",
    "repeat: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "longword: supercalifragilisticexpialidocious" * 2,
    "upper: SCREAMING_SNAKE_CASE and camelCase and kebab-case",
    "newlines: a\n\nb\n\n\nc",
    "trailing-newlines: end\n\n\n",
    "only-spaces:     ",
    "one-char: a",
    "one-space: ",
    "unicode-space: a b c　d",
    "quotes: \"double\" 'single' `backtick` “smart” ‘single’",
    "math: ∑_{i=1}^{n} x_i^2 ≤ y² and α ≠ β",
    "brackets: 【】（）《》〈〉「」『』",
    "arrows: → ← ↑ ↓ ↔ ⇒ ⇐",
    "currency: $1,234.56 €789 ¥100 £2 ₽3 ₹4",
    "units: 10km/h 5m/s² 20°C 100%",
    "json: {\"key\": \"value\", \"n\": [1, 2, 3]}",
    "sql: SELECT * FROM users WHERE id = 1 ORDER BY name;",
    "shell: ls -la | grep 'foo' && echo $HOME",
    "markdown: # Title\n\n- item **bold** `code`\n\n> quote",
    "zero-width: a​b​c",
    "bom: ﻿hello",
    "nbsp: a b",
    "soft-hyphen: long­word",
    "surrogate-ish: 𝕳𝖊𝖑𝖑𝖔 𝕎𝖔𝖗𝖑𝖉",
    "astral: 𠀋 𐍈 𤭢",
    "special: <|im_start|>user\nhi<|im_end|>",
    "special2: <|endoftext|>middle<|im_start|>",
    "special3: no special here <|not_a_token|>",
]

# 各脚本的随机采样区间，用于覆盖预分词器的 \p{L} / \p{N} 分支。
SCRIPT_RANGES = [
    (0x0020, 0x007E),  # Basic Latin
    (0x00A0, 0x00FF),  # Latin-1 Supplement
    (0x0100, 0x017F),  # Latin Extended-A
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x04FF),  # Cyrillic
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x1E00, 0x1EFF),  # Latin Extended Additional
    (0x2010, 0x206F),  # General Punctuation
    (0x2190, 0x21FF),  # Arrows
    (0x3000, 0x303F),  # CJK Symbols
    (0x3040, 0x30FF),  # Kana
    (0x4E00, 0x4FFF),  # CJK (片段)
    (0xAC00, 0xAFFF),  # Hangul (片段)
    (0xFE30, 0xFE4F),  # CJK Compatibility Forms
    (0xFF00, 0xFFEF),  # Halfwidth / Fullwidth
    (0x1F300, 0x1F5FF),  # 表情符号
]


def _repo_text_lines(root: Path, cap: int = 800) -> list[str]:
    """仓库文档里的真实文本行 —— 比合成语料更接近实际分布。"""
    out: list[str] = []
    seen: set[str] = set()
    for path in sorted(root.glob("docs/**/*.md")):
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or len(line) > 220:
                continue
            if line in seen:
                continue
            seen.add(line)
            out.append(line)
            if len(out) >= cap:
                return out
    return out


def _random_text(rng: random.Random, length: int) -> str:
    chars = []
    for _ in range(length):
        lo, hi = SCRIPT_RANGES[rng.randrange(len(SCRIPT_RANGES))]
        code = rng.randint(lo, hi)
        # 避开代理区：它们不是合法标量值。
        if 0xD800 <= code <= 0xDFFF:
            code = 0x4E00 + (code - 0xD800) % 0x100
        ch = chr(code)
        # 控制字符与未分配码位会让用例含义不清，跳过。
        if unicodedata.category(ch) in ("Cc", "Cn", "Cs"):
            continue
        chars.append(ch)
    return "".join(chars)


def build_corpus(root: Path, rng: random.Random, tokenizer) -> list[str]:
    cases: list[str] = []

    cases.extend(ADVERSARIAL)
    cases.extend(_repo_text_lines(root))

    # 从词表里抽样再解码回文本 —— 这是对 BPE 合并表覆盖最直接的语料。
    vocab_size = tokenizer.vocab_size
    for _ in range(3400):
        n = rng.randint(1, 6)
        ids = [rng.randrange(256, min(vocab_size, 151000)) for _ in range(n)]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if text:
            cases.append(text)

    # 随机脚本混排
    for _ in range(900):
        cases.append(_random_text(rng, rng.randint(1, 24)))

    # 合成噪声：在合法文本里插入随机的空白 / 标点 / 换行
    for _ in range(600):
        base = _random_text(rng, rng.randint(2, 16))
        noise = rng.choice(["  ", "\t", "\n", "\r\n", " ", "!", ",", "。", "123", "ab"])
        pos = rng.randint(0, len(base))
        cases.append(base[:pos] + noise + base[pos:])

    # 去重并保持确定性顺序
    seen: set[str] = set()
    unique: list[str] = []
    for c in cases:
        if c in seen:
            continue
        seen.add(c)
        unique.append(c)
    return unique


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------


def dump_tokenizer(tokenizer, out: Path) -> int:
    vocab = tokenizer.get_vocab()  # token(字节级字符串) -> id
    inv = {i: t for t, i in vocab.items()}

    # 词表：token 是 byte-level 的 unicode 串，转回真实字节用 hex 存，避免歧义。
    with (out / "vocab.tsv").open("w", encoding="utf-8") as f:
        for i in range(len(inv)):
            token = inv[i]
            raw = token_to_bytes(token)
            f.write("%d\t%s\n" % (i, raw.hex()))

    # 合并表：HF 的 merges 是 "A B" 的字符串对，按出现顺序即 rank。
    # `tokenizers` 的 BPE 对象不暴露 merges（它存的是 trie），序列化成 JSON 才有。
    merges = json.loads(tokenizer.backend_tokenizer.to_str())["model"]["merges"]
    with (out / "merges.tsv").open("w", encoding="utf-8") as f:
        for rank, m in enumerate(merges):
            if isinstance(m, (list, tuple)):
                left, right = m[0], m[1]
            else:
                left, right = m.split(" ", 1)
            if left not in vocab or right not in vocab:
                raise SystemExit("merge 的某一侧不在词表中: %r %r" % (left, right))
            f.write("%d\t%d\t%d\n" % (rank, vocab[left], vocab[right]))

    with (out / "special.tsv").open("w", encoding="utf-8") as f:
        for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
            if getattr(tokenizer, "all_special_tokens", None) and tok in set(
                tokenizer.all_special_tokens
            ):
                f.write("%d\t%s\n" % (idx, token_to_bytes(tok).hex()))

    return len(inv)


def dump_cases(tokenizer, cases: list[str], out: Path) -> int:
    written = 0
    with (out / "tokenizer_cases.tsv").open("w", encoding="utf-8") as f:
        for text in cases:
            ids = tokenizer.encode(text, add_special_tokens=False)
            f.write("%s\t%s\n" % (text.encode("utf-8").hex(), ",".join(str(i) for i in ids)))
            written += 1
    return written


def dump_numeric(model, tokenizer, out: Path, greedy_tokens: int) -> dict:
    import torch

    prompts = []
    for text in PROMPTS:
        prompts.append({"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)})

    (out / "prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    greedy = []
    logits_rows = []
    with torch.no_grad():
        for item in prompts:
            input_ids = torch.tensor([item["ids"]], dtype=torch.long)
            last = model(input_ids).logits[0, -1]
            logits_rows.append(last.to(torch.float32).numpy())

            cur = list(item["ids"])
            generated = []
            for _ in range(greedy_tokens):
                x = torch.tensor([cur], dtype=torch.long)
                nxt = int(model(x).logits[0, -1].argmax())
                generated.append(nxt)
                cur = cur + [nxt]
            greedy.append(generated)

    vocab_size = int(logits_rows[0].shape[0])
    with (out / "logits_last.f32").open("wb") as f:
        for row in logits_rows:
            f.write(struct.pack("<%df" % vocab_size, *row.tolist()))

    (out / "greedy.json").write_text(json.dumps(greedy), encoding="utf-8")

    return {"n_prompts": len(prompts), "vocab_size": vocab_size, "greedy_tokens": greedy_tokens}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default="tests/fixtures/qwen2.5-0.5b")
    ap.add_argument("--greedy", type=int, default=GREEDY_TOKENS)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--cases-only", action="store_true", help="只导出分词相关产物")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.is_fast:
        raise SystemExit("需要 fast tokenizer（tokenizers 实现），否则无法读取 merges")

    n_vocab = dump_tokenizer(tokenizer, out)

    rng = random.Random(args.seed)
    cases = build_corpus(root, rng, tokenizer)
    if len(cases) < TARGET_CASES:
        raise SystemExit("语料不足：%d < %d，请扩充 ADVERSARIAL 或采样量" % (len(cases), TARGET_CASES))
    n_cases = dump_cases(tokenizer, cases, out)

    manifest = {
        "model": args.model,
        "vocab_size": n_vocab,
        "n_cases": n_cases,
        "python": sys.version.split()[0],
        "generated_with": "scripts/dump_reference.py",
    }

    if not args.cases_only:
        import torch
        import transformers

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, device_map="cpu"
        )
        model.eval()
        manifest.update(dump_numeric(model, tokenizer, out, args.greedy))
        manifest["torch"] = torch.__version__
        manifest["transformers"] = transformers.__version__

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    size = sum(p.stat().st_size for p in out.iterdir())
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("产物总大小: %.1f MB -> %s" % (size / 1e6, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
