r"""Generate the small `tests/fixtures/tokenizer-json/` fixtures.

Why these exist: the real gate reads Qwen's 7 MB `tokenizer.json` against a
4560-case corpus, which is the honest proof that the loader works — but it can
only ever contain what that one file happens to contain. The shapes the loader
has to survive and does not meet there are:

- `\uXXXX` escapes (Hugging Face writes UTF-8, so escapes never appear in the
  Qwen file, yet they are legal JSON and a `u`-escape bug would be invisible);
- merges written as a pair `["A", "B"]` instead of a string `"A B"` — the shape
  `tokenizers` >= 0.20 writes;
- the refusals: a model that is not BPE, a regex that is not the one this
  pre-tokenizer implements, sparse ids, `add_prefix_space`, and broken JSON.

So this script writes a **real** `tokenizer.json` — same schema, same structure,
small enough to commit — and then lets Hugging Face's own `tokenizers` load it
and encode the corpus: the expected ids in `cases.tsv` are the reference's, not
ours.

Run (needs the reference stack, which has `tokenizers`):

    /home/rontom/anaconda3/bin/python scripts/gen_tokenizer_json_fixtures.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "tokenizer-json"

# The pattern `src/alofa/tokenizer/pretokenize.mojo` implements. It is written to
# the fixture and checked against `TOKENIZER_PATTERN` at load time: a mismatch is
# a refusal, not a silent difference.
PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

MERGES = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("l", "l"), ("he", "ll"), ("hell", "o")]

CASES = [
    "",
    "hello world",
    " hello the ",
    "Hello, world!",
    "<|mini|>hi<|mini|>",
    "é",
    "café",
    "café",
    "áb",
    "  spaced   out  ",
    "123 4567 0",
    "tab\tsep",
    "xéy",
    "HELLO hello Hello",
    "don't 42x",
]


def byte_level_alphabet() -> dict[int, int]:
    """GPT-2's bytes_to_unicode: one code point per byte."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping: dict[int, int] = {}
    extra = 0
    for byte in range(256):
        if byte in printable:
            mapping[byte] = byte
        else:
            mapping[byte] = 256 + extra
            extra += 1
    return mapping


def build_vocab() -> dict[str, int]:
    """All 256 byte tokens, plus one entry per merge result."""
    alphabet = byte_level_alphabet()
    vocab: dict[str, int] = {}
    for byte in range(256):
        vocab[chr(alphabet[byte])] = byte
    for index, (left, right) in enumerate(MERGES):
        vocab[left + right] = 256 + index
    return vocab


def document(
    vocab: dict[str, int],
    merges: list,
    *,
    model_type: str = "BPE",
    pattern: str = PATTERN,
    prefix_space: bool = False,
) -> dict:
    return {
        "version": "1.0",
        "added_tokens": [
            {
                "id": 262,
                "content": "<|mini|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 263,
                # Deliberately not a `special` token: the reference splits text
                # on added tokens whether or not they are special.
                "content": "é",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": False,
            },
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": pattern},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": prefix_space,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": model_type,
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


def main() -> int:
    try:
        from tokenizers import Tokenizer
    except ImportError:  # pragma: no cover - the reference stack has it
        print("需要 tokenizers：用 /home/rontom/anaconda3/bin/python 跑本脚本", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab()

    # `ensure_ascii=True` is the point: every non-ASCII token becomes a
    # `\uXXXX` escape, a shape the Qwen file never contains.
    string_merges = [" ".join(pair) for pair in MERGES]
    escaped = json.dumps(
        document(vocab, string_merges), ensure_ascii=True, indent=1
    )
    (OUT / "mini.json").write_text(escaped, encoding="utf-8")

    # Same tokenizer, UTF-8 instead of escapes, merges as pairs: two
    # serializations that must tokenize identically.
    pair_merges = [[left, right] for left, right in MERGES]
    (OUT / "mini_pairs.json").write_text(
        json.dumps(document(vocab, pair_merges), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # ---- the refusals -----------------------------------------------------
    (OUT / "bad_model_type.json").write_text(
        json.dumps(document(vocab, string_merges, model_type="Unigram"), indent=1),
        encoding="utf-8",
    )
    (OUT / "bad_regex.json").write_text(
        json.dumps(document(vocab, string_merges, pattern=r"\w+"), indent=1),
        encoding="utf-8",
    )
    (OUT / "bad_prefix_space.json").write_text(
        json.dumps(document(vocab, string_merges, prefix_space=True), indent=1),
        encoding="utf-8",
    )
    sparse = {token: id_ for token, id_ in vocab.items() if id_ != 100}
    (OUT / "bad_vocab.json").write_text(
        json.dumps(document(sparse, string_merges), indent=1), encoding="utf-8"
    )
    truncated = escaped[: len(escaped) // 2]
    (OUT / "broken.json").write_text(truncated, encoding="utf-8")

    # ---- the oracle: Hugging Face encodes the corpus ----------------------
    reference = Tokenizer.from_file(str(OUT / "mini.json"))
    rows = []
    for text in CASES:
        ids = reference.encode(text, add_special_tokens=False).ids
        rows.append("%s\t%s" % (text.encode("utf-8").hex(), ",".join(str(i) for i in ids)))
    (OUT / "cases.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # One case per byte: the text is the byte-level character the byte maps to,
    # so the alphabet is exercised over its whole range — including the bytes
    # that are not printable and the code points NFC rewrites (U+00A8 and three
    # neighbours decompose to space + a combining mark). What the reference does
    # with those is the answer, not what we expect it to do.
    alphabet = byte_level_alphabet()
    rows = []
    for byte in range(256):
        text = chr(alphabet[byte])
        ids = reference.encode(text, add_special_tokens=False).ids
        rows.append("%s\t%s" % (text.encode("utf-8").hex(), ",".join(str(i) for i in ids)))
    (OUT / "bytes.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # The reference must agree with itself when the merges are written as pairs,
    # otherwise `mini_pairs.json` cannot be used as a second serialization of the
    # same tokenizer. Older `tokenizers` may reject the pair form outright.
    try:
        pairs = Tokenizer.from_file(str(OUT / "mini_pairs.json"))
        for text in CASES:
            if pairs.encode(text, add_special_tokens=False).ids != reference.encode(
                text, add_special_tokens=False
            ).ids:
                print("mini_pairs.json 与 mini.json 编码不一致，请检查", file=sys.stderr)
                return 1
    except Exception as exc:  # pragma: no cover - depends on the installed version
        print("tokenizers 无法读取 pair 形式的 merges：%s" % exc, file=sys.stderr)

    print("wrote", len(CASES), "cases and 256 byte cases to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
