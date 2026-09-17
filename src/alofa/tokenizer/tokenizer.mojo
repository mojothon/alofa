"""The Qwen2 tokenizer: loading, encoding and decoding.

Fixtures under one directory describe everything this needs — `vocab.tsv`,
`merges.tsv` and `special.tsv`, all produced by `scripts/dump_reference.py`.
That choice is deliberate: the object under test compares itself against data,
not against a live Python process, so the differential gate runs in CI on
machines that have neither torch nor transformers installed, and the reference
cannot drift underneath it between runs.

Pipeline order matches the reference implementation:

1. extract added (special) tokens, longest match wins;
2. NFC-normalize each remaining chunk;
3. pre-tokenize per Qwen2's regex (`pretokenize.mojo`);
4. merge each pre-token by rank (`bpe.mojo`).

Skipping step 1 is not cosmetic: the reference splits added tokens even with
`add_special_tokens=False`, so text containing `<|im_end|>` tokenizes
differently once added tokens are ignored.
"""

from std.io import FileHandle

from .bpe import NO_TOKEN, Merges, Vocab, encode_piece
from .codes import (
    TOK_ERR_BAD_FIXTURE_LINE,
    TOK_ERR_BAD_VOCAB,
    TOK_ERR_IO,
    TOK_ERR_UNKNOWN_ID,
    tokenizer_error,
)
from .nfc import nfc_range
from .pretokenize import pretokenize
from .unicode import Unicode, utf8_decode_at, utf8_encode_codepoint

comptime LF = 0x0A
comptime TAB = 0x09


struct TextLine(Copyable, Movable):
    """One line of a fixture file as a byte range, plus where the next one starts.

    A struct rather than several `out` parameters because Mojo 1.0 forbids more
    than one `out` argument per function.
    """

    var start: Int
    var end: Int

    def __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end

    def is_end(imm self) -> Bool:
        """True for the sentinel returned once the file is exhausted."""
        return self.start < 0


struct SpecialMatch(Copyable, Movable):
    """Which added token matched, and how many bytes it covers."""

    var id: Int
    var length: Int

    def __init__(out self, id: Int, length: Int):
        self.id = id
        self.length = length


struct Tokenizer(Copyable, Movable):
    """A loaded Qwen2 tokenizer, ready to encode or decode."""

    var uni: Unicode
    var vocab: Vocab
    var merges: Merges
    var special_ids: List[Int]
    var special_blob: List[UInt8]
    var special_starts: List[Int]

    def __init__(out self, imm uni: Unicode, imm vocab: Vocab, imm merges: Merges):
        self.uni = uni.copy()
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_ids = List[Int]()
        self.special_blob = List[UInt8]()
        self.special_starts = List[Int]()
        self.special_starts.append(0)

    def register_special(mut self, id: Int, imm source: List[UInt8], start: Int, end: Int):
        """Register an added token; its content is what splits input text."""
        var index = start
        while index < end:
            self.special_blob.append(source[index])
            index += 1
        self.special_ids.append(id)
        self.special_starts.append(len(self.special_blob))

    def special_at(imm self, imm text: String, index: Int) -> SpecialMatch:
        """The added token starting at `index`, or `NO_TOKEN` if there is none.

        Longest match wins, which is what the reference's Aho-Corasick pass does
        when one added token happens to be a prefix of another.
        """
        var best_id = NO_TOKEN
        var best_length = 0
        var slot = 0
        while slot < len(self.special_ids):
            var content_start = self.special_starts[slot]
            var content_end = self.special_starts[slot + 1]
            var size = content_end - content_start
            if size > best_length and _equal_range(
                text, index, self.special_blob, content_start, content_end
            ):
                best_length = size
                best_id = self.special_ids[slot]
            slot += 1
        return SpecialMatch(best_id, best_length)

    def encode(imm self, imm text: String, mut sink: List[Int]) raises:
        """Tokenize `text` without prepending anything.

        Equivalent to the reference's `add_special_tokens=False`: nothing is
        added, yet added tokens appearing in the text still tokenize as
        themselves.
        """
        var index = 0
        var chunk_start = 0
        var n = text.byte_length()
        while index < n:
            var hit = self.special_at(text, index)
            if hit.id != NO_TOKEN:
                self._encode_plain(text, chunk_start, index, sink)
                sink.append(hit.id)
                index += hit.length
                chunk_start = index
            else:
                # One byte is enough: added tokens are ASCII, so no match can
                # start inside a multi-byte sequence and be missed by this scan.
                index += 1
        self._encode_plain(text, chunk_start, n, sink)

    def encode(imm self, imm text: String) raises -> List[Int]:
        """Tokenize `text` and return the ids."""
        var ids = List[Int]()
        self.encode(text, ids)
        return ids^

    def _encode_plain(imm self, imm text: String, start: Int, end: Int, mut sink: List[Int]) raises:
        var normalized = nfc_range(text, start, end, self.uni)
        var pieces = pretokenize(normalized, self.uni)
        var i = 0
        while i < len(pieces):
            encode_piece(normalized, pieces[i], pieces[i + 1], self.vocab, self.merges, sink)
            i += 2

    def decode(imm self, imm ids: List[Int]) raises -> String:
        """Map ids back to text, replacing malformed UTF-8 as the reference does."""
        var bytes = List[UInt8]()
        var i = 0
        while i < len(ids):
            var id = ids[i]
            if id < 0 or id >= self.vocab.count():
                raise tokenizer_error(
                    TOK_ERR_UNKNOWN_ID, "id " + String(id) + " is outside the vocabulary"
                )
            var token_start = self.vocab.token_start(id)
            var token_end = self.vocab.token_end(id)
            var k = token_start
            while k < token_end:
                bytes.append(self.vocab.blob[k])
                k += 1
            i += 1

        var raw = String(unsafe_from_utf8=bytes)
        var text = List[UInt8]()
        var index = 0
        while index < raw.byte_length():
            var decoded = utf8_decode_at(raw, index)
            if decoded.width <= 0:
                break
            utf8_encode_codepoint(decoded.cp, text)
            index += decoded.width
        return String(unsafe_from_utf8=text)


def load_tokenizer(imm directory: String) raises -> Tokenizer:
    """Load a tokenizer from a fixture directory.

    No Python is involved: this reads the three TSV files that
    `scripts/dump_reference.py` wrote, which is what lets the differential gate
    run anywhere.
    """
    var vocab_text = read_text_file(directory + "/vocab.tsv")
    var merge_text = read_text_file(directory + "/merges.tsv")
    var special_text = read_text_file(directory + "/special.tsv")

    var vocab_count = count_lines(vocab_text)
    var merge_count = count_lines(merge_text)
    if vocab_count == 0 or merge_count == 0:
        raise tokenizer_error(TOK_ERR_BAD_VOCAB, "fixture files are empty")

    var uni = Unicode()
    var vocab = Vocab(vocab_count)
    var merges = Merges(merge_count)
    var scratch = List[UInt8]()

    # vocab.tsv: "<id>\t<token bytes as hex>", ids ascending from zero.
    var cursor = 0
    var line = next_line(vocab_text, cursor)
    while not line.is_end():
        _read_vocab_line(vocab_text, line.start, line.end, vocab, scratch)
        line = next_line(vocab_text, cursor)
    vocab.seal()

    cursor = 0
    line = next_line(merge_text, cursor)
    while not line.is_end():
        _read_merge_line(merge_text, line.start, line.end, merges)
        line = next_line(merge_text, cursor)

    var tokenizer = Tokenizer(uni, vocab, merges)
    cursor = 0
    line = next_line(special_text, cursor)
    while not line.is_end():
        _read_special_line(special_text, line.start, line.end, tokenizer, scratch)
        line = next_line(special_text, cursor)
    return tokenizer^


def read_text_file(imm path: String) raises -> String:
    """Read a whole file, naming the path when it cannot be read."""
    var handle: FileHandle
    try:
        handle = FileHandle(path, "r")
    except e:
        raise tokenizer_error(TOK_ERR_IO, "cannot open " + path + ": " + String(e))
    var content = handle.read()
    handle.close()
    return content^


def count_lines(imm source: String) -> Int:
    """Number of `\n`-terminated lines; the count sizes the hash tables up front."""
    # Byte access goes through one materialised copy: `as_bytes()` rebuilds the
    # sequence, so calling it per byte here is quadratic in file size.
    var bytes = source.as_bytes()
    var n = source.byte_length()
    var total = 0
    var index = 0
    while index < n:
        if Int(bytes[index]) == LF:
            total += 1
        index += 1
    # A final line without a trailing newline still counts.
    if n > 0 and Int(bytes[n - 1]) != LF:
        total += 1
    return total


def next_line(imm source: String, mut cursor: Int) -> TextLine:
    """Return the byte range of the next line, advancing the cursor.

    Ends with a sentinel whose `start` is negative, so the caller can loop on
    `is_end` without a separate index variable.
    """
    var bytes = source.as_bytes()
    var n = source.byte_length()
    if cursor >= n:
        return TextLine(-1, -1)
    var end = cursor
    while end < n and Int(bytes[end]) != LF:
        end += 1
    var start = cursor
    cursor = end + 1
    return TextLine(start, end)


def _read_vocab_line(
    imm source: String, start: Int, end: Int, mut vocab: Vocab, mut scratch: List[UInt8]
) raises:
    var tab = _scan_for(source, start, end, TAB)
    if tab < 0:
        raise tokenizer_error(TOK_ERR_BAD_FIXTURE_LINE, "vocab line has no tab separator")
    var id = _parse_int(source, start, tab)
    if id != vocab.count():
        raise tokenizer_error(
            TOK_ERR_BAD_VOCAB,
            "token id "
            + String(id)
            + " is out of order: expected "
            + String(vocab.count())
            + "; ids must ascend from zero by one",
        )
    scratch.clear()
    _decode_hex(source, tab + 1, end, scratch)
    vocab.append_bytes(scratch, 0, len(scratch))


def _read_merge_line(imm source: String, start: Int, end: Int, mut merges: Merges) raises:
    var first_tab = _scan_for(source, start, end, TAB)
    if first_tab < 0:
        raise tokenizer_error(TOK_ERR_BAD_FIXTURE_LINE, "merge line has no tab separators")
    var second_tab = _scan_for(source, first_tab + 1, end, TAB)
    if second_tab < 0:
        raise tokenizer_error(TOK_ERR_BAD_FIXTURE_LINE, "merge line needs two tab separators")
    var rank = _parse_int(source, start, first_tab)
    var left = _parse_int(source, first_tab + 1, second_tab)
    var right = _parse_int(source, second_tab + 1, end)
    merges.insert(left, right, rank)


def _read_special_line(
    imm source: String,
    start: Int,
    end: Int,
    mut tokenizer: Tokenizer,
    mut scratch: List[UInt8]) raises:
    var tab = _scan_for(source, start, end, TAB)
    if tab < 0:
        raise tokenizer_error(TOK_ERR_BAD_FIXTURE_LINE, "special line has no tab separator")
    var id = _parse_int(source, start, tab)
    scratch.clear()
    _decode_hex(source, tab + 1, end, scratch)
    tokenizer.register_special(id, scratch, 0, len(scratch))


def _equal_range(
    imm source: String, start: Int, imm blob: List[UInt8], token_start: Int, token_end: Int
) -> Bool:
    var length = token_end - token_start
    if start + length > source.byte_length():
        return False
    var bytes = source.as_bytes()
    var offset = 0
    while offset < length:
        if bytes[start + offset] != blob[token_start + offset]:
            return False
        offset += 1
    return True


def _scan_for(imm source: String, start: Int, end: Int, target: Int) -> Int:
    var bytes = source.as_bytes()
    var index = start
    while index < end:
        if Int(bytes[index]) == target:
            return index
        index += 1
    return -1


def _parse_int(imm source: String, start: Int, end: Int) raises -> Int:
    if start >= end:
        raise tokenizer_error(TOK_ERR_BAD_FIXTURE_LINE, "empty numeric field")
    var bytes = source.as_bytes()
    var value = 0
    var index = start
    while index < end:
        var digit = Int(bytes[index]) - 48
        if digit < 0 or digit > 9:
            raise tokenizer_error(TOK_ERR_BAD_FIXTURE_LINE, "non-digit in numeric field")
        value = value * 10 + digit
        index += 1
    return value


def _hex_value(byte: UInt8) raises -> Int:
    var value = Int(byte)
    if value >= 48 and value <= 57:
        return value - 48
    if value >= 97 and value <= 102:
        return value - 87
    if value >= 65 and value <= 70:
        return value - 55
    raise tokenizer_error(TOK_ERR_BAD_FIXTURE_LINE, "not a hex digit")


def _decode_hex(imm source: String, start: Int, end: Int, mut sink: List[UInt8]) raises:
    """Decode a hex encoded byte string, so fixtures stay free of binary blobs.

    Text fixtures mean a human can read `vocab.tsv` in a diff; binary would be
    more compact and considerably less auditable.
    """
    var bytes = source.as_bytes()
    var index = start
    while index + 1 < end:
        var high = _hex_value(bytes[index])
        var low = _hex_value(bytes[index + 1])
        sink.append(UInt8(high * 16 + low))
        index += 2
    if index != end:
        raise tokenizer_error(TOK_ERR_BAD_FIXTURE_LINE, "hex string has an odd length")
