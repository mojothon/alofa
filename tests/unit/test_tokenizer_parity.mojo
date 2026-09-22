"""Differential gate: our tokenizer against the reference, one case at a time.

This is the gate that decides whether the tokenizer is real. Unit tests written
against my own expectations would only prove the code agrees with itself; here
every case comes from `tests/fixtures/qwen2.5-0.5b/tokenizer_cases.tsv`, which
`scripts/dump_reference.py` produced by running HuggingFace's fast tokenizer
over a deliberately nasty corpus — docs prose, contractions, CRLF, emoji, NFC
traps, random sampled vocabulary tokens, added tokens mid-sentence.

The gate fails on **any** mismatch, not on a failure rate: this test asserts
byte-level equivalence of implementation details, and a 0.1% mismatch rate
still means the tokenizer is wrong for some inputs.

Run:
    pixi run mojo run -I src tests/unit/test_tokenizer_parity.mojo
"""

from std.io import FileHandle
from std.testing import TestSuite, assert_equal, assert_true

from alofa.tokenizer.nfc import nfc
from alofa.tokenizer.tokenizer import Tokenizer, load_tokenizer
from alofa.tokenizer.tokenizer_json import load_tokenizer_json
from alofa.tokenizer.unicode import Unicode

comptime FIXTURE_DIR = "tests/fixtures/qwen2.5-0.5b"
comptime CASES_PATH = FIXTURE_DIR + "/tokenizer_cases.tsv"
# The file Hugging Face publishes, verbatim — the same tokenizer the corpus was
# produced with, so it is the same oracle read through a different loader.
comptime TOKENIZER_JSON = FIXTURE_DIR + "/tokenizer.json"

comptime LF = 0x0A
comptime TAB = 0x09

# The corpus size the ledger promises. Generating fewer cases must fail rather
# than quietly lower the bar.
comptime MIN_CASES = 4560


struct ParityResult(Copyable, Movable):
    """Outcome over the whole corpus: counts plus a few failures to print."""

    var total: Int
    var failed: Int
    var report: String

    def __init__(out self):
        self.total = 0
        self.failed = 0
        self.report = ""


def run_corpus(imm tokenizer: Tokenizer) raises -> ParityResult:
    """Encode every reference case and compare ids one by one."""
    var cases = read_text_file(CASES_PATH)
    var raw = bytes_of(cases)
    var result = ParityResult()
    var scratch = List[UInt8]()
    var expected = List[Int]()
    var actual = List[Int]()

    var cursor = 0
    while cursor < len(raw):
        var start = cursor
        var end = next_newline(raw, cursor)
        cursor = end + 1
        if end <= start:
            break

        var tab = scan_for(raw, start, end, TAB)
        if tab < 0:
            raise Error("case line has no tab separator")

        scratch.clear()
        decode_hex(raw, start, tab, scratch)
        expected.clear()
        parse_id_list(raw, tab + 1, end, expected)

        var text = String(unsafe_from_utf8=scratch)
        actual.clear()
        tokenizer.encode(text, actual)
        result.total += 1
        if not same_ids(actual, expected):
            result.failed += 1
            if result.failed <= 5:
                result.report += (
                    String("  case ")
                    + String(result.total)
                    + " ["
                    + String(unsafe_from_utf8=scratch)
                    + "]\n    expected "
                    + describe_ids(expected)
                    + "\n    actual   "
                    + describe_ids(actual)
                    + "\n"
                )
    return result^


def test_corpus_matches_reference_exactly() raises:
    """Every one of the reference cases must produce the same ids."""
    var tokenizer = load_tokenizer(FIXTURE_DIR)
    var result = run_corpus(tokenizer)
    if result.failed > 0:
        print(result.report)
    assert_true(result.total >= MIN_CASES, "corpus is too small: " + String(result.total))
    assert_equal(result.failed, 0)


def test_json_loader_matches_the_reference_corpus() raises:
    """The same oracle, read from a real `tokenizer.json` instead of our TSV.

    This is the one that says the ecosystem's file is loadable: the corpus was
    produced by Hugging Face's fast tokenizer, and here it is re-encoded from
    the artifact that tokenizer itself serializes.
    """
    var tokenizer = load_tokenizer_json(TOKENIZER_JSON)
    var result = run_corpus(tokenizer)
    if result.failed > 0:
        print(result.report)
    assert_true(result.total >= MIN_CASES, "corpus is too small: " + String(result.total))
    assert_equal(result.failed, 0)


def test_json_and_tsv_loaders_agree_on_every_token() raises:
    """Two readers of one tokenizer must agree on all 151665 token contents.

    The corpus only exercises the tokens it happens to sample; this compares the
    whole vocabulary byte for byte, which is what catches an off-by-one in id
    placement that no sampled case would reach.
    """
    var from_json = load_tokenizer_json(TOKENIZER_JSON)
    var from_tsv = load_tokenizer(FIXTURE_DIR)
    assert_equal(from_json.vocab.count(), from_tsv.vocab.count())

    var mismatched = 0
    var id = 0
    while id < from_json.vocab.count():
        var json_start = from_json.vocab.token_start(id)
        var json_end = from_json.vocab.token_end(id)
        var tsv_start = from_tsv.vocab.token_start(id)
        if json_end - json_start != from_tsv.vocab.token_end(id) - tsv_start:
            mismatched += 1
        else:
            var offset = 0
            while offset < json_end - json_start:
                if from_json.vocab.blob[json_start + offset] != from_tsv.vocab.blob[
                    tsv_start + offset
                ]:
                    mismatched += 1
                    break
                offset += 1
        id += 1
    assert_equal(mismatched, 0)


def test_round_trip_preserves_the_input() raises:
    """Decoding a reference encoding gives back the *normalized* input.

    Catches what the encode direction hides: ids that are correct themselves
    while their token contents do not concatenate back to the input. The
    comparison is against NFC of the input rather than the raw bytes because
    the tokenizer normalizes before it splits — a decomposed input comes back
    composed, and that is the reference's behaviour too.
    """
    var tokenizer = load_tokenizer(FIXTURE_DIR)
    var cases = read_text_file(CASES_PATH)
    var raw = bytes_of(cases)
    var scratch = List[UInt8]()
    var expected = List[Int]()

    var mismatches = 0
    var checked = 0
    var cursor = 0
    while cursor < len(raw):
        var start = cursor
        var end = next_newline(raw, cursor)
        cursor = end + 1
        if end <= start:
            break
        var tab = scan_for(raw, start, end, TAB)
        if tab < 0:
            raise Error("case line has no tab separator")

        scratch.clear()
        decode_hex(raw, start, tab, scratch)
        expected.clear()
        parse_id_list(raw, tab + 1, end, expected)
        if len(expected) == 0:
            continue

        var normalized = List[UInt8]()
        _append_bytes(normalized, nfc(String(unsafe_from_utf8=scratch), tokenizer.uni))
        var decoded = tokenizer.decode(expected)
        checked += 1
        if not same_bytes(decoded, normalized):
            mismatches += 1
            if mismatches <= 5:
                print(String("  round-trip differs for [") + String(unsafe_from_utf8=scratch) + "]")
    assert_true(checked > 0, "no round-trip cases were checked")
    assert_equal(mismatches, 0)


def test_nfc_handles_the_reference_shapes() raises:
    """NFC on inputs chosen specifically to break naive normalization."""
    var uni = Unicode()
    assert_equal(nfc("e\u0301", uni), "\u00e9")  # combining mark -> precomposed
    assert_equal(nfc("\u212b", uni), "\u00c5")  # singleton -> precomposed
    assert_equal(nfc("\u00c5", uni), "\u00c5")  # already composed stays put
    assert_equal(nfc("\u1100\u1161", uni), "\uac00")  # Hangul L + V -> syllable
    assert_equal(nfc("\uac00\u11a8", uni), "\uac01")  # LV + T -> LVT
    # Dot above (class 230) sorts after cedilla (202); neither composes with q.
    assert_equal(nfc("q\u0307\u0327", uni), "q\u0327\u0307")


def read_text_file(path: String) raises -> String:
    var handle = FileHandle(path, "r")
    var content = handle.read()
    handle.close()
    return content^


def bytes_of(source: String) -> List[UInt8]:
    """Materialise the corpus bytes once.

    `String.as_bytes()` rebuilds its result on every call, so a helper that
    indexes it     per byte costs a full copy per byte. Taking one copy up front
    turns the whole corpus walk from quadratic into linear.
    """
    var view = source.as_bytes()
    var out = List[UInt8]()
    var index = 0
    var total = source.byte_length()
    while index < total:
        out.append(view[index])
        index += 1
    return out^


def next_newline(imm raw: List[UInt8], start: Int) -> Int:
    """Offset of the next LF, or the end of the data."""
    var index = start
    while index < len(raw):
        if Int(raw[index]) == LF:
            return index
        index += 1
    return len(raw)


def scan_for(imm raw: List[UInt8], start: Int, end: Int, target: Int) -> Int:
    var index = start
    while index < end:
        if Int(raw[index]) == target:
            return index
        index += 1
    return -1


def hex_value(byte: UInt8) -> Int:
    var value = Int(byte)
    if value >= 48 and value <= 57:
        return value - 48
    if value >= 97 and value <= 102:
        return value - 87
    if value >= 65 and value <= 70:
        return value - 55
    return -1


def decode_hex(imm raw: List[UInt8], start: Int, end: Int, mut sink: List[UInt8]) raises:
    var index = start
    while index + 1 < end:
        var high = hex_value(raw[index])
        var low = hex_value(raw[index + 1])
        if high < 0 or low < 0:
            raise Error("not a hex digit at " + String(index))
        sink.append(UInt8(high * 16 + low))
        index += 2
    if index != end:
        raise Error("hex string has an odd length")


def parse_id_list(imm raw: List[UInt8], start: Int, end: Int, mut sink: List[Int]) raises:
    """Parse `"12,34,5"`. An empty field means the input produced no tokens."""
    var index = start
    while index <= end:
        var comma = scan_for(raw, index, end, 44)
        var stop = end
        if comma >= 0:
            stop = comma
        if stop > index:
            var value = 0
            var digit = index
            while digit < stop:
                var number = Int(raw[digit]) - 48
                if number < 0 or number > 9:
                    raise Error("non-digit in expected ids")
                value = value * 10 + number
                digit += 1
            sink.append(value)
        if comma < 0:
            return
        index = comma + 1


def _append_bytes(mut sink: List[UInt8], source: String):
    var view = source.as_bytes()
    var index = 0
    while index < source.byte_length():
        sink.append(view[index])
        index += 1


def same_ids(actual: List[Int], expected: List[Int]) -> Bool:
    if len(actual) != len(expected):
        return False
    var index = 0
    while index < len(expected):
        if actual[index] != expected[index]:
            return False
        index += 1
    return True


def same_bytes(text: String, expected: List[UInt8]) -> Bool:
    if text.byte_length() != len(expected):
        return False
    var view = text.as_bytes()
    var index = 0
    while index < len(expected):
        if view[index] != expected[index]:
            return False
        index += 1
    return True


def describe_ids(ids: List[Int]) -> String:
    """First two dozen ids, so a failure prints what actually differed."""
    var text = "["
    var index = 0
    while index < len(ids) and index < 24:
        if index > 0:
            text += ","
        text += String(ids[index])
        index += 1
    if len(ids) > 24:
        text += ",...]"
    else:
        text += "]"
    return text


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
