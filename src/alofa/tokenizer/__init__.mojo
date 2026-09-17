"""Qwen2 tokenizer: NFC normalization, pre-tokenization and BPE in pure Mojo.

    from alofa.tokenizer.tokenizer import load_tokenizer

    var tok = load_tokenizer("tests/fixtures/qwen2.5-0.5b")
    var ids = tok.encode("Hello, world!")

The tokenizer reads the fixtures `scripts/dump_reference.py` writes, so it never
touches Python at runtime: what it agrees with is a checked-in reference, which
is what makes the 4560-case differential gate reproducible.
"""

from .bpe import Merges, Vocab, encode_piece
from .codes import tokenizer_error, tokenizer_error_name
from .nfc import nfc, nfc_range
from .pretokenize import pretokenize
from .tokenizer import Tokenizer, load_tokenizer
from .unicode import Unicode
