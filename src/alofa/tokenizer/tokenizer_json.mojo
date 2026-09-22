"""Loader for a real Hugging Face `tokenizer.json` — the file the ecosystem ships.

The TSV fixtures (`tokenizer.mojo`) exist so the differential gate can run on a
machine without torch or transformers. They are, however, *our* format: nobody
outside this repo produces them. This module reads the artifact the ecosystem
actually publishes, so a checkpoint can be dropped in without a conversion step.

What it reads, and nothing else:

- `model.vocab`: token (byte-level unicode) -> id;
- `model.merges`: rank-ordered pairs, either `"A B"` or `["A", "B"]` depending on
  the `tokenizers` version that wrote the file;
- `added_tokens`: `{id, content}` — **all** of them, not only `special: true`.

Byte-level tokens are stored as unicode, one code point per byte (GPT-2's
alphabet), so each token string is mapped back to raw bytes on the way in.

Everything else in the file is *checked, not trusted*. `pretokenize.mojo` is a
hand-written matcher for one specific regex, `nfc.mojo` implements NFC, and
`decode` is byte-level — so a file whose `pre_tokenizer` regex is a different
pattern, whose `normalizer` is not NFC, or whose `decoder` is not byte-level
would produce ids that look plausible and mean something else. Those raise
`unsupported_tokenizer` instead: refusing is the only honest answer when the
file describes a tokenizer this code does not implement.

Not implemented: WordPiece / Unigram models, `add_prefix_space`, and normalizers
other than NFC. They are refused by name rather than silently approximated.
"""

from .bpe import NO_TOKEN, Merges, Vocab
from .codes import (
    TOK_ERR_BAD_JSON,
    TOK_ERR_BAD_MERGES,
    TOK_ERR_BAD_VOCAB,
    TOK_ERR_UNSUPPORTED_TOKENIZER,
    tokenizer_error,
)
from .pretokenize import TOKENIZER_PATTERN
from .tokenizer import Tokenizer, read_text_file
from .unicode import Decoded, Unicode, utf8_encode_codepoint

comptime QUOTE = 0x22
comptime BACKSLASH = 0x5C
comptime LBRACE = 0x7B
comptime RBRACE = 0x7D
comptime LBRACKET = 0x5B
comptime RBRACKET = 0x5D
comptime COLON = 0x3A
comptime COMMA = 0x2C
comptime SPACE_BYTE = 0x20

comptime HEX_U = 0x75  # 'u'
comptime MINUS = 0x2D
comptime NULL_START = 0x6E  # 'n', as in `null`
comptime TRUE_START = 0x74  # 't', as in `true`


struct TokenizerJsonParts(Copyable, Movable):
    """What one `tokenizer.json` yielded, before it is assembled.

    Parsing is separated from building because `added_tokens` appears *before*
    `model` in the files Hugging Face writes, while the vocabulary has to exist
    before anything can be registered against it.
    """

    var keys: List[UInt8]  # scratch: member names and token strings
    var vocab_ids: List[Int]
    var vocab_blob: List[UInt8]
    var vocab_starts: List[Int]  # n + 1 offsets
    var vocab_text_starts: List[Int]  # n + 1 offsets into `keys`
    var merge_blob: List[UInt8]
    var merge_starts: List[Int]  # 2 * merges + 1 offsets
    var added_ids: List[Int]
    var added_blob: List[UInt8]
    var added_starts: List[Int]  # n + 1 offsets
    var model_type: String
    var normalizer_type: String  # "" means `null`
    var decoder_type: String
    var saw_pre_tokenizer: Bool
    var regex_found: Bool
    var regex_ok: Bool
    var prefix_space: Bool

    def __init__(out self):
        self.keys = List[UInt8]()
        self.vocab_ids = List[Int]()
        self.vocab_blob = List[UInt8]()
        self.vocab_starts = List[Int]()
        self.vocab_text_starts = List[Int]()
        self.merge_blob = List[UInt8]()
        self.merge_starts = List[Int]()
        self.added_ids = List[Int]()
        self.added_blob = List[UInt8]()
        self.added_starts = List[Int]()
        self.model_type = ""
        self.normalizer_type = ""
        self.decoder_type = ""
        self.saw_pre_tokenizer = False
        self.regex_found = False
        self.regex_ok = False
        self.prefix_space = False


def load_tokenizer_json(imm path: String) raises -> Tokenizer:
    """Load a Hugging Face `tokenizer.json` and return a ready tokenizer.

    Raises `bad_json` when the file is not well formed and
    `unsupported_tokenizer` when it describes a tokenizer this implementation
    does not match (see the module docstring).
    """
    var raw = bytes_of(read_text_file(path))
    var parts = TokenizerJsonParts()
    var at = 0

    _skip_ws(raw, at)
    _expect(raw, at, LBRACE)
    while True:
        _skip_ws(raw, at)
        if _peek(raw, at) == RBRACE:
            at += 1
            break
        var key_start = len(parts.keys)
        _read_string(raw, at, parts.keys)
        var key_end = len(parts.keys)
        _skip_ws(raw, at)
        _expect(raw, at, COLON)
        _skip_ws(raw, at)
        if _key_is(parts.keys, key_start, key_end, "model"):
            _read_model(raw, at, parts)
        elif _key_is(parts.keys, key_start, key_end, "added_tokens"):
            _read_added_tokens(raw, at, parts)
        elif _key_is(parts.keys, key_start, key_end, "normalizer"):
            parts.normalizer_type = _read_type_or_null(raw, at, parts.keys)
        elif _key_is(parts.keys, key_start, key_end, "decoder"):
            parts.decoder_type = _read_type_or_null(raw, at, parts.keys)
        elif _key_is(parts.keys, key_start, key_end, "pre_tokenizer"):
            parts.saw_pre_tokenizer = True
            _walk_for_regex(raw, at, parts)
        else:
            _skip_value(raw, at)
        _skip_ws(raw, at)
        var after = _peek(raw, at)
        if after == COMMA:
            at += 1
        elif after == RBRACE:
            at += 1
            break
        else:
            raise _bad("expected ',' or '}' at " + String(at))

    _check_supported(parts)
    return _build(parts)


def byte_from_unicode(cp: Int) -> Int:
    """The byte a code point stands for in GPT-2's alphabet, or -1.

    The alphabet is three printable runs mapped to themselves, then — for the 68
    bytes that are not printable — a run starting at 256. Written out rather than
    tabulated because it is four comparisons either way.
    """
    if 33 <= cp <= 126:
        return cp
    if 161 <= cp <= 172:
        return cp
    if 174 <= cp <= 255:
        return cp
    var index = cp - 256
    if 0 <= index <= 32:  # bytes 0..32
        return index
    if 33 <= index <= 66:  # bytes 127..160
        return index + 94
    if index == 67:  # byte 173
        return 173
    return -1


def bytes_of(imm source: String) -> List[UInt8]:
    """Materialise a file's bytes once.

    `String.as_bytes()` rebuilds its result on every call, so indexing it per
    byte costs a full copy per byte — quadratic over a 7 MB `tokenizer.json`.
    """
    var view = source.as_bytes()
    var out = List[UInt8]()
    var index = 0
    var total = source.byte_length()
    while index < total:
        out.append(view[index])
        index += 1
    return out^


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def _check_supported(imm parts: TokenizerJsonParts) raises:
    """Refuse files that describe a tokenizer this code does not implement."""
    if parts.model_type != "BPE":
        raise _unsupported(
            "model type is '" + parts.model_type + "'; only BPE is implemented"
        )
    if parts.normalizer_type != "" and parts.normalizer_type != "NFC":
        raise _unsupported(
            "normalizer '"
            + parts.normalizer_type
            + "'; only NFC (or none) is implemented"
        )
    if parts.decoder_type != "ByteLevel":
        raise _unsupported(
            "decoder '" + parts.decoder_type + "'; only ByteLevel is implemented"
        )
    if not parts.saw_pre_tokenizer or not parts.regex_found:
        raise _unsupported(
            "pre_tokenizer has no regex pattern to check the hard-coded"
            + " pre-tokenizer against"
        )
    if not parts.regex_ok:
        raise _unsupported(
            "pre_tokenizer regex differs from the one implemented in"
            + " pretokenize.mojo"
        )
    if parts.prefix_space:
        raise _unsupported("ByteLevel(add_prefix_space=true) is not implemented")


def _build(mut parts: TokenizerJsonParts) raises -> Tokenizer:
    var count = len(parts.vocab_ids)
    if count == 0:
        raise tokenizer_error(TOK_ERR_BAD_VOCAB, "tokenizer.json has no vocabulary")

    # Ids arrive in file order, which is not a guarantee; the vocab is built in
    # id order, so place each entry first. Sorting by id is a counting sort: ids
    # are unique and, as `Vocab` requires, dense from zero.
    var order = List[Int]()
    var i = 0
    while i < count:
        order.append(-1)
        i += 1
    i = 0
    while i < count:
        var id = parts.vocab_ids[i]
        if id < 0 or id >= count:
            raise tokenizer_error(
                TOK_ERR_BAD_VOCAB,
                "token id "
                + String(id)
                + " is outside 0.."
                + String(count - 1)
                + "; ids must be dense from zero",
            )
        if order[id] != -1:
            raise tokenizer_error(TOK_ERR_BAD_VOCAB, "duplicate token id " + String(id))
        order[id] = i
        i += 1

    var vocab = Vocab(count + len(parts.added_ids))
    i = 0
    while i < count:
        var entry = order[i]
        vocab.append_bytes(
            parts.vocab_blob, parts.vocab_starts[entry], parts.vocab_starts[entry + 1]
        )
        i += 1

    vocab.seal()

    # Added tokens extend the vocabulary: the reference's `get_vocab()` reports
    # them, and `decode` has to answer with their content. They are matched while
    # encoding by the added-token pass, not by BPE.
    #
    # A content that is already a token keeps the vocabulary's id — observed
    # against Hugging Face: an added token `é` whose content is the byte token
    # U+00E9 encodes as that token (233), not as the id the file declares (263).
    # The token is still split out as an added token, which is what distinguishes
    # it from the byte-level path (`xéy` -> [120, 233, 121], not [120, 195, 169, 121]).
    #
    # The comparison is between *strings*, as the reference's added-token trie
    # does it: the content is raw text, the vocabulary holds byte-level tokens,
    # and the two meet only as the strings the file spells out.
    var resolved = List[Int]()
    var appended = 0
    i = 0
    while i < len(parts.added_ids):
        var start = parts.added_starts[i]
        var end = parts.added_starts[i + 1]
        var existing = _find_token_text(parts, start, end)
        if existing != NO_TOKEN:
            resolved.append(existing)
        else:
            var declared = parts.added_ids[i]
            if declared != vocab.count():
                raise tokenizer_error(
                    TOK_ERR_BAD_VOCAB,
                    "added token id "
                    + String(declared)
                    + " does not follow the vocabulary at "
                    + String(vocab.count())
                    + "; ids must be dense from zero",
                )
            vocab.append_bytes(parts.added_blob, start, end)
            resolved.append(declared)
            appended += 1
        i += 1
    if appended > 0:
        vocab.index_from(count)

    var merges = Merges(len(parts.merge_starts) // 2)
    i = 0
    while i < len(parts.merge_starts) // 2:
        var left = vocab.find_bytes(
            parts.merge_blob, parts.merge_starts[2 * i], parts.merge_starts[2 * i + 1]
        )
        var right = vocab.find_bytes(
            parts.merge_blob,
            parts.merge_starts[2 * i + 1],
            parts.merge_starts[2 * i + 2],
        )
        if left == NO_TOKEN or right == NO_TOKEN:
            raise tokenizer_error(
                TOK_ERR_BAD_MERGES,
                "merge " + String(i) + " names a token that is not in the vocabulary",
            )
        merges.insert(left, right, i)
        i += 1

    var tokenizer = Tokenizer(Unicode(), vocab, merges)
    i = 0
    while i < len(parts.added_ids):
        tokenizer.register_special(
            resolved[i], parts.added_blob, parts.added_starts[i], parts.added_starts[i + 1]
        )
        i += 1
    return tokenizer^


# --------------------------------------------------------------------------
# members
# --------------------------------------------------------------------------


def _find_token_text(imm parts: TokenizerJsonParts, start: Int, end: Int) -> Int:
    """The id of the vocabulary token whose *string* is `added_blob[start:end]`.

    Added tokens are matched as text, so what is compared here is the string the
    file spells out, not the bytes the token stands for. Linear because it runs
    once per added token, and the length test rejects almost every candidate.
    """
    var length = end - start
    var i = 0
    while i < len(parts.vocab_ids):
        var text_start = parts.vocab_text_starts[i]
        if parts.vocab_text_starts[i + 1] - text_start == length:
            if _blob_equals(parts.keys, text_start, parts.added_blob, start, length):
                return parts.vocab_ids[i]
        i += 1
    return NO_TOKEN


def _blob_equals(
    imm a: List[UInt8], a_start: Int, imm b: List[UInt8], b_start: Int, length: Int
) -> Bool:
    var index = 0
    while index < length:
        if a[a_start + index] != b[b_start + index]:
            return False
        index += 1
    return True


def _read_model(imm raw: List[UInt8], mut at: Int, mut parts: TokenizerJsonParts) raises:
    _skip_ws(raw, at)
    _expect(raw, at, LBRACE)
    while True:
        _skip_ws(raw, at)
        if _peek(raw, at) == RBRACE:
            at += 1
            return
        var key_start = len(parts.keys)
        _read_string(raw, at, parts.keys)
        var key_end = len(parts.keys)
        _skip_ws(raw, at)
        _expect(raw, at, COLON)
        _skip_ws(raw, at)
        if _key_is(parts.keys, key_start, key_end, "type"):
            parts.model_type = _read_string_value(raw, at, parts.keys)
        elif _key_is(parts.keys, key_start, key_end, "vocab"):
            _read_vocab(raw, at, parts)
        elif _key_is(parts.keys, key_start, key_end, "merges"):
            _read_merges(raw, at, parts)
        else:
            _skip_value(raw, at)
        _skip_ws(raw, at)
        var after = _peek(raw, at)
        if after == COMMA:
            at += 1
        elif after == RBRACE:
            at += 1
            return
        else:
            raise _bad("expected ',' or '}' in model at " + String(at))


def _read_vocab(imm raw: List[UInt8], mut at: Int, mut parts: TokenizerJsonParts) raises:
    _skip_ws(raw, at)
    _expect(raw, at, LBRACE)
    while True:
        _skip_ws(raw, at)
        if _peek(raw, at) == RBRACE:
            at += 1
            parts.vocab_starts.append(len(parts.vocab_blob))
            parts.vocab_text_starts.append(len(parts.keys))
            return
        var text_start = len(parts.keys)
        _read_string(raw, at, parts.keys)
        var text_end = len(parts.keys)
        _skip_ws(raw, at)
        _expect(raw, at, COLON)
        _skip_ws(raw, at)
        parts.vocab_ids.append(_read_int(raw, at))
        parts.vocab_starts.append(len(parts.vocab_blob))
        parts.vocab_text_starts.append(text_start)
        _byte_level_bytes(parts.keys, text_start, text_end, parts.vocab_blob)
        _skip_ws(raw, at)
        var after = _peek(raw, at)
        if after == COMMA:
            at += 1
        elif after == RBRACE:
            at += 1
            parts.vocab_starts.append(len(parts.vocab_blob))
            parts.vocab_text_starts.append(len(parts.keys))
            return
        else:
            raise _bad("expected ',' or '}' in vocab at " + String(at))


def _read_merges(imm raw: List[UInt8], mut at: Int, mut parts: TokenizerJsonParts) raises:
    _skip_ws(raw, at)
    _expect(raw, at, LBRACKET)
    while True:
        _skip_ws(raw, at)
        if _peek(raw, at) == RBRACKET:
            at += 1
            parts.merge_starts.append(len(parts.merge_blob))
            return
        var lead = _peek(raw, at)
        if lead == QUOTE:
            # The `tokenizers` format up to 0.19: one string, "left right".
            # The space is unambiguous as a separator because the alphabet maps
            # byte 0x20 to U+0120, so no token content ever contains one.
            var start = len(parts.keys)
            _read_string(raw, at, parts.keys)
            var end = len(parts.keys)
            var split = _scan_byte(parts.keys, start, end, SPACE_BYTE)
            if split < 0:
                raise _bad("merge entry has no space separator")
            parts.merge_starts.append(len(parts.merge_blob))
            _byte_level_bytes(parts.keys, start, split, parts.merge_blob)
            parts.merge_starts.append(len(parts.merge_blob))
            _byte_level_bytes(parts.keys, split + 1, end, parts.merge_blob)
        elif lead == LBRACKET:
            # 0.20 and later write a pair instead.
            at += 1
            _skip_ws(raw, at)
            var left_start = len(parts.keys)
            _read_string(raw, at, parts.keys)
            var left_end = len(parts.keys)
            parts.merge_starts.append(len(parts.merge_blob))
            _byte_level_bytes(parts.keys, left_start, left_end, parts.merge_blob)
            _skip_ws(raw, at)
            _expect(raw, at, COMMA)
            _skip_ws(raw, at)
            var right_start = len(parts.keys)
            _read_string(raw, at, parts.keys)
            var right_end = len(parts.keys)
            parts.merge_starts.append(len(parts.merge_blob))
            _byte_level_bytes(parts.keys, right_start, right_end, parts.merge_blob)
            _skip_ws(raw, at)
            _expect(raw, at, RBRACKET)
        else:
            raise _bad("merge entry is neither a string nor a pair")
        _skip_ws(raw, at)
        var after = _peek(raw, at)
        if after == COMMA:
            at += 1
        elif after == RBRACKET:
            at += 1
            parts.merge_starts.append(len(parts.merge_blob))
            return
        else:
            raise _bad("expected ',' or ']' in merges at " + String(at))


def _read_added_tokens(
    imm raw: List[UInt8], mut at: Int, mut parts: TokenizerJsonParts) raises:
    """Read every added token, `special` or not.

    The reference splits text on all of them; `special: true` only decides
    membership of `all_special_tokens`, which is a decode-side concern.
    """
    _skip_ws(raw, at)
    _expect(raw, at, LBRACKET)
    while True:
        _skip_ws(raw, at)
        if _peek(raw, at) == RBRACKET:
            at += 1
            parts.added_starts.append(len(parts.added_blob))
            return
        _expect(raw, at, LBRACE)
        var id = -1
        var content_start = -1
        while True:
            _skip_ws(raw, at)
            if _peek(raw, at) == RBRACE:
                at += 1
                break
            var key_start = len(parts.keys)
            _read_string(raw, at, parts.keys)
            var key_end = len(parts.keys)
            _skip_ws(raw, at)
            _expect(raw, at, COLON)
            _skip_ws(raw, at)
            if _key_is(parts.keys, key_start, key_end, "id"):
                id = _read_int(raw, at)
            elif _key_is(parts.keys, key_start, key_end, "content"):
                content_start = len(parts.added_blob)
                _read_string(raw, at, parts.added_blob)
            else:
                _skip_value(raw, at)
            _skip_ws(raw, at)
            var after = _peek(raw, at)
            if after == COMMA:
                at += 1
            elif after == RBRACE:
                at += 1
                break
            else:
                raise _bad("expected ',' or '}' in added token at " + String(at))
        if id < 0 or content_start < 0:
            raise _bad("added token is missing 'id' or 'content'")
        parts.added_ids.append(id)
        parts.added_starts.append(content_start)
        _skip_ws(raw, at)
        var after = _peek(raw, at)
        if after == COMMA:
            at += 1
        elif after == RBRACKET:
            at += 1
            parts.added_starts.append(len(parts.added_blob))
            return
        else:
            raise _bad("expected ',' or ']' in added_tokens at " + String(at))


def _read_type_or_null(imm raw: List[UInt8], mut at: Int, mut scratch: List[UInt8]) raises -> String:
    """The `type` of a component object, or "" when the value is `null`."""
    _skip_ws(raw, at)
    if _peek(raw, at) == NULL_START:
        _skip_value(raw, at)
        return ""
    var found = ""
    _expect(raw, at, LBRACE)
    while True:
        _skip_ws(raw, at)
        if _peek(raw, at) == RBRACE:
            at += 1
            return found
        var key_start = len(scratch)
        _read_string(raw, at, scratch)
        var key_end = len(scratch)
        _skip_ws(raw, at)
        _expect(raw, at, COLON)
        _skip_ws(raw, at)
        if _key_is(scratch, key_start, key_end, "type"):
            found = _read_string_value(raw, at, scratch)
        else:
            _skip_value(raw, at)
        _skip_ws(raw, at)
        var after = _peek(raw, at)
        if after == COMMA:
            at += 1
        elif after == RBRACE:
            at += 1
            return found
        else:
            raise _bad("expected ',' or '}' at " + String(at))


def _walk_for_regex(
    imm raw: List[UInt8], mut at: Int, mut parts: TokenizerJsonParts) raises:
    """Walk a `pre_tokenizer` subtree looking for the regex and for `add_prefix_space`.

    The shape is not known up front — `Split` may sit inside a `Sequence`, on its
    own, or inside something else again — so the walk is structural rather than
    a search for a fixed path.
    """
    _skip_ws(raw, at)
    var lead = _peek(raw, at)
    if lead == LBRACE:
        at += 1
        while True:
            _skip_ws(raw, at)
            if _peek(raw, at) == RBRACE:
                at += 1
                return
            var key_start = len(parts.keys)
            _read_string(raw, at, parts.keys)
            var key_end = len(parts.keys)
            _skip_ws(raw, at)
            _expect(raw, at, COLON)
            _skip_ws(raw, at)
            if _key_is(parts.keys, key_start, key_end, "Regex"):
                var value_start = len(parts.keys)
                _read_string(raw, at, parts.keys)
                parts.regex_found = True
                parts.regex_ok = _range_equals(
                    parts.keys, value_start, len(parts.keys), TOKENIZER_PATTERN
                )
            elif _key_is(parts.keys, key_start, key_end, "add_prefix_space"):
                if _peek(raw, at) == TRUE_START:
                    parts.prefix_space = True
                _skip_value(raw, at)
            else:
                _walk_for_regex(raw, at, parts)
            _skip_ws(raw, at)
            var after = _peek(raw, at)
            if after == COMMA:
                at += 1
            elif after == RBRACE:
                at += 1
                return
            else:
                raise _bad("expected ',' or '}' in pre_tokenizer at " + String(at))
    elif lead == LBRACKET:
        at += 1
        while True:
            _skip_ws(raw, at)
            if _peek(raw, at) == RBRACKET:
                at += 1
                return
            _walk_for_regex(raw, at, parts)
            _skip_ws(raw, at)
            var after = _peek(raw, at)
            if after == COMMA:
                at += 1
            elif after == RBRACKET:
                at += 1
                return
            else:
                raise _bad("expected ',' or ']' in pre_tokenizer at " + String(at))
    else:
        _skip_value(raw, at)


# --------------------------------------------------------------------------
# byte-level tokens
# --------------------------------------------------------------------------


def _byte_level_bytes(
    imm source: List[UInt8], start: Int, end: Int, mut sink: List[UInt8]) raises:
    """Decode a byte-level token string into the bytes it stands for."""
    var index = start
    while index < end:
        var decoded = _decode_utf8(source, index, end)
        var byte = byte_from_unicode(decoded.cp)
        if byte < 0:
            raise _bad(
                "code point "
                + String(decoded.cp)
                + " is not in the byte-level alphabet"
            )
        sink.append(UInt8(byte))
        index += decoded.width


def _decode_utf8(imm raw: List[UInt8], index: Int, end: Int) raises -> Decoded:
    """One strict UTF-8 code point. `tokenizer.json` is UTF-8 by definition, so
    malformed input is an error here rather than a replacement character."""
    if index >= end:
        raise _bad("truncated UTF-8 sequence at " + String(index))
    var lead = Int(raw[index])
    if lead < 0x80:
        return Decoded(lead, 1)
    var extra = 3 if lead >= 0xF0 else (2 if lead >= 0xE0 else (1 if lead >= 0xC0 else -1))
    if extra < 0:
        raise _bad("invalid UTF-8 lead byte at " + String(index))
    if index + extra >= end:
        raise _bad("truncated UTF-8 sequence at " + String(index))
    var mask = 0x07 if extra == 3 else (0x0F if extra == 2 else 0x1F)
    var cp = lead & mask
    var i = 1
    while i <= extra:
        var continuation = Int(raw[index + i])
        if continuation & 0xC0 != 0x80:
            raise _bad("invalid UTF-8 continuation byte at " + String(index + i))
        cp = (cp << 6) | (continuation & 0x3F)
        i += 1
    return Decoded(cp, extra + 1)


# --------------------------------------------------------------------------
# JSON primitives
# --------------------------------------------------------------------------


def _read_string(imm raw: List[UInt8], mut at: Int, mut sink: List[UInt8]) raises:
    """Decode one JSON string into `sink` as UTF-8 bytes, escapes resolved."""
    if _peek(raw, at) != QUOTE:
        raise _bad("expected a string at " + String(at))
    at += 1
    while True:
        if at >= len(raw):
            raise _bad("unterminated string at " + String(at))
        var byte = Int(raw[at])
        if byte == QUOTE:
            at += 1
            return
        if byte != BACKSLASH:
            sink.append(UInt8(byte))
            at += 1
            continue
        at += 1
        if at >= len(raw):
            raise _bad("unterminated escape at " + String(at))
        var escape = Int(raw[at])
        at += 1
        if escape == HEX_U:
            var cp = _read_hex4(raw, at)
            # A surrogate is only meaningful as a pair; a lone one is not a
            # scalar value and cannot be encoded.
            if cp >= 0xD800 and cp <= 0xDBFF:
                if _peek(raw, at) == BACKSLASH and _peek(raw, at + 1) == HEX_U:
                    at += 2
                    var low = _read_hex4(raw, at)
                    if low >= 0xDC00 and low <= 0xDFFF:
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00)
            utf8_encode_codepoint(cp, sink)
            continue
        var simple = _escape_byte(escape)
        if simple < 0:
            raise _bad("unsupported escape at " + String(at))
        sink.append(UInt8(simple))


def _read_string_value(imm raw: List[UInt8], mut at: Int, mut scratch: List[UInt8]) raises -> String:
    var start = len(scratch)
    _read_string(raw, at, scratch)
    var out = List[UInt8]()
    var index = start
    while index < len(scratch):
        out.append(scratch[index])
        index += 1
    return String(unsafe_from_utf8=out)


def _read_hex4(imm raw: List[UInt8], mut at: Int) raises -> Int:
    if at + 4 > len(raw):
        raise _bad("truncated \\u escape at " + String(at))
    var value = 0
    var i = 0
    while i < 4:
        var digit = _hex_value(Int(raw[at + i]))
        if digit < 0:
            raise _bad("not a hex digit at " + String(at + i))
        value = value * 16 + digit
        i += 1
    at += 4
    return value


def _hex_value(byte: Int) -> Int:
    if 48 <= byte <= 57:
        return byte - 48
    if 97 <= byte <= 102:
        return byte - 87
    if 65 <= byte <= 70:
        return byte - 55
    return -1


def _escape_byte(escape: Int) -> Int:
    if escape == 0x22:
        return 0x22
    if escape == 0x5C:
        return 0x5C
    if escape == 0x2F:
        return 0x2F
    if escape == 0x62:
        return 0x08
    if escape == 0x66:
        return 0x0C
    if escape == 0x6E:
        return 0x0A
    if escape == 0x72:
        return 0x0D
    if escape == 0x74:
        return 0x09
    return -1


def _read_int(imm raw: List[UInt8], mut at: Int) raises -> Int:
    var negative = False
    if _peek(raw, at) == MINUS:
        negative = True
        at += 1
    var start = at
    var value = 0
    while at < len(raw):
        var digit = Int(raw[at]) - 48
        if digit < 0 or digit > 9:
            break
        value = value * 10 + digit
        at += 1
    if at == start:
        raise _bad("expected a number at " + String(at))
    if negative:
        return -value
    return value


def _skip_value(imm raw: List[UInt8], mut at: Int) raises:
    _skip_ws(raw, at)
    var lead = _peek(raw, at)
    if lead == LBRACE or lead == LBRACKET:
        _skip_container(raw, at)
        return
    if lead == QUOTE:
        _skip_string(raw, at)
        return
    while at < len(raw):
        var byte = Int(raw[at])
        if byte == COMMA or byte == RBRACE or byte == RBRACKET:
            return
        if byte == 0x20 or byte == 0x09 or byte == 0x0A or byte == 0x0D:
            return
        at += 1


def _skip_container(imm raw: List[UInt8], mut at: Int) raises:
    var depth = 0
    while at < len(raw):
        var byte = Int(raw[at])
        if byte == QUOTE:
            _skip_string(raw, at)
            continue
        if byte == LBRACE or byte == LBRACKET:
            depth += 1
            at += 1
            continue
        if byte == RBRACE or byte == RBRACKET:
            depth -= 1
            at += 1
            if depth == 0:
                return
            continue
        at += 1
    raise _bad("unterminated container")


def _skip_string(imm raw: List[UInt8], mut at: Int) raises:
    if _peek(raw, at) != QUOTE:
        raise _bad("expected a string at " + String(at))
    at += 1
    while at < len(raw):
        var byte = Int(raw[at])
        if byte == BACKSLASH:
            at += 2
            continue
        if byte == QUOTE:
            at += 1
            return
        at += 1
    raise _bad("unterminated string at " + String(at))


def _skip_ws(imm raw: List[UInt8], mut at: Int):
    while at < len(raw):
        var byte = Int(raw[at])
        if byte != 0x20 and byte != 0x09 and byte != 0x0A and byte != 0x0D:
            return
        at += 1


def _peek(imm raw: List[UInt8], at: Int) -> Int:
    if at < 0 or at >= len(raw):
        return -1
    return Int(raw[at])


def _expect(imm raw: List[UInt8], mut at: Int, expected: Int) raises:
    if _peek(raw, at) != expected:
        raise _bad("unexpected byte at " + String(at))
    at += 1


def _scan_byte(imm source: List[UInt8], start: Int, end: Int, target: Int) -> Int:
    var index = start
    while index < end:
        if Int(source[index]) == target:
            return index
        index += 1
    return -1


def _key_is(imm buffer: List[UInt8], start: Int, end: Int, imm name: String) -> Bool:
    return _range_equals(buffer, start, end, name)


def _range_equals(imm buffer: List[UInt8], start: Int, end: Int, imm text: String) -> Bool:
    var length = end - start
    if length != text.byte_length():
        return False
    var view = text.as_bytes()
    var index = 0
    while index < length:
        if buffer[start + index] != view[index]:
            return False
        index += 1
    return True


def _bad(detail: String) -> Error:
    return tokenizer_error(TOK_ERR_BAD_JSON, detail)


def _unsupported(detail: String) -> Error:
    return tokenizer_error(TOK_ERR_UNSUPPORTED_TOKENIZER, detail)
