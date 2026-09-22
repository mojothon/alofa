#!/usr/bin/env python3
"""Build `tests/fixtures/qwen2.5-0.5b-hf/` — a directory laid out the way Hugging
Face publishes a checkpoint.

Why a directory and not three loose files: what the e2e is supposed to answer is
"can this engine read the artefact the ecosystem ships, in the shape it ships
it". A test that points at three hand-picked files has already done the work of
deciding which files matter; pointing at a directory forces the loader to find
them the way a user's checkpoint directory presents them.

Why symlinks instead of copies: the real `model.safetensors` is 988 MB of
bfloat16 and `tokenizer.json` is 7 MB. Copying either into the repository would
double a fixture that is already checked in elsewhere, and for the weights it
would put a second gigabyte on disk for no test value — the bytes under the
symlink are the bytes Hugging Face wrote.

`tokenizer.json` links to the copy already committed under
`tests/fixtures/qwen2.5-0.5b/` (the one the differential corpus is checked
against), so the e2e and the corpus cannot drift apart. `config.json` is copied
because it is 681 bytes and a reader should see a plain file.

Usage (offline — nothing here touches the network):

    python3 scripts/build_real_model_dir.py
    python3 scripts/build_real_model_dir.py --src /path/to/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "tests" / "fixtures" / "qwen2.5-0.5b-hf"
CACHE_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*"
)

# The three files a Qwen2 checkpoint directory is read from here.
CONFIG = "config.json"
TOKENIZER = "tokenizer.json"
WEIGHTS = "model.safetensors"


def find_source(src: str | None) -> Path:
    if src is not None:
        path = Path(src).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit("source directory not found: %s" % path)
        return path
    candidates = sorted(glob.glob(CACHE_GLOB))
    for candidate in candidates:
        path = Path(candidate)
        if (path / WEIGHTS).is_file() and (path / CONFIG).is_file():
            return path.resolve()
    raise SystemExit(
        "no cached Qwen2.5-0.5B with %s and %s under %s; pass --src"
        % (CONFIG, WEIGHTS, CACHE_GLOB)
    )


def dtypes_and_count(path: Path) -> tuple[dict[str, int], int]:
    """(dtype -> tensor count, total tensors) from a safetensors header."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    counts: dict[str, int] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        counts[meta["dtype"]] = counts.get(meta["dtype"], 0) + 1
    return counts, len([k for k in header if k != "__metadata__"])


def link_or_copy(source: Path, target: Path, copy: bool) -> None:
    if target.is_symlink() or target.exists():
        target.unlink()
    if copy:
        target.write_bytes(source.read_bytes())
    else:
        target.symlink_to(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=None, help="a Qwen2.5-0.5B checkpoint directory")
    args = parser.parse_args()

    source = find_source(args.src)
    weights = source / WEIGHTS
    config = source / CONFIG
    tokenizer = source / TOKENIZER
    for path in (weights, config, tokenizer):
        if not path.is_file():
            raise SystemExit("source directory is missing %s" % path.name)

    DEST.mkdir(parents=True, exist_ok=True)

    # `config.json` is copied: at 681 bytes there is nothing to save, and a
    # reader that dereferences a link is one fewer thing to reason about.
    link_or_copy(config, DEST / CONFIG, copy=True)

    # `tokenizer.json` links to the copy already in the repository, so the e2e
    # and the 4560-case differential corpus are provably the same file.
    committed = REPO / "tests" / "fixtures" / "qwen2.5-0.5b" / TOKENIZER
    if committed.is_file():
        link_or_copy(committed, DEST / TOKENIZER, copy=False)
    else:
        link_or_copy(tokenizer, DEST / TOKENIZER, copy=False)

    link_or_copy(weights, DEST / WEIGHTS, copy=False)

    counts, total = dtypes_and_count(weights)
    provenance = "\n".join(
        [
            "source: %s" % source,
            "weights: %s (%d bytes, %s)" % (WEIGHTS, weights.stat().st_size, counts),
            "tensors: %d" % total,
            "config: %s (copied)" % CONFIG,
            "tokenizer: %s -> %s" % (TOKENIZER, (DEST / TOKENIZER).resolve()),
            "",
        ]
    )
    (DEST / "SOURCE.txt").write_text(provenance)

    print("wrote %s" % DEST)
    print(provenance, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
