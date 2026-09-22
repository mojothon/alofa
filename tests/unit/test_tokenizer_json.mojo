"""The `tokenizer.json` loader: real files, and the files it must refuse.

Two things are checked here that the Qwen gate cannot check:

- the shapes Hugging Face is allowed to write but this particular file does not
  contain — `\\uXXXX` escapes and merges written as a pair instead of a string.
  They are covered by `tests/fixtures/tokenizer-json/mini.json`, a real
  `tokenizer.json` small enough to commit, whose expected ids come from Hugging
  Face's own `tokenizers` (`scripts/gen_tokenizer_json_fixtures.py`);
- the refusals. A file describing a non-BPE model, a different regex,
  `add_prefix_space`, sparse ids, or broken JSON must raise: approximating any
  of them silently would produce ids that look plausible and mean something
  else.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_tokenizer_json.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.tokenizer.tokenizer import Tokenizer, read_text_file
from alofa.tokenizer.tokenizer_json import bytes_of, load_tokenizer_json

comptime FIXTURE_DIR = "tests/fixtures/tokenizer-json"
comptime CASES_PATH = FIXTURE_DIR + "/cases.tsv"
# One case per byte: the text is the byte-level character that byte maps to.
comptime BYTES_PATH = FIXTURE_DIR + "/bytes.tsv"
comptime BYTE_CASES = 256
comptime ESCAPED_JSON = FIXTURE_DIR + "/mini.json"
comptime PAIRS_JSON = FIXTURE_DIR + "/mini_pairs.json"

comptime LF = 0x0A
comptime TAB = 0x09

# What `scripts/gen_tokenizer_json_fixtures.py` writes. Fewer cases must fail
# rather than quietly lower the bar.
comptime MIN_CASES = 14


struct CaseResult(Copyable, Movable):
    """Outcome over the case file: counts plus a few failures to print."""

    var total: Int
    var failed: Int
    var report: String

    def __init__(out self):
        self.total = 0
        self.failed = 0
        self.report = ""


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_mini_json_matches_the_reference_cases() raises:
    """`mini.json` escapes every non-ASCII token, unlike the Qwen file.

    Same schema, same pipeline, ids produced by the reference: this is what
    catches an escape decoder that has only ever seen UTF-8.
    """
    var tokenizer = load_tokenizer_json(ESCAPED_JSON)
    var result = run_cases(tokenizer, CASES_PATH)
    if result.failed > 0:
        print(result.report)
    assert_true(result.total >= MIN_CASES, "corpus is too small: " + String(result.total))
    assert_equal(result.failed, 0)


def test_pair_form_merges_tokenize_the_same() raises:
    """`tokenizers` >= 0.20 writes `["A", "B"]` where 0.19 wrote `"A B"`.

    The reference version installed here cannot read the pair form, so the
    oracle is the string form: reading the same tokenizer twice, once per
    serialization, must give identical ids — and those ids are the reference's,
    which `test_mini_json_matches_the_reference_cases` has just established.
    """
    var from_escaped = load_tokenizer_json(ESCAPED_JSON)
    var from_pairs = load_tokenizer_json(PAIRS_JSON)
    assert_equal(from_escaped.vocab.count(), from_pairs.vocab.count())

    var raw = bytes_of(read_text_file(CASES_PATH))
    var scratch = List[UInt8]()
    var escaped_ids = List[Int]()
    var pair_ids = List[Int]()

    var cursor = 0
    var compared = 0
    var mismatched = 0
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
        escaped_ids.clear()
        pair_ids.clear()
        from_escaped.encode(String(unsafe_from_utf8=scratch), escaped_ids)
        from_pairs.encode(String(unsafe_from_utf8=scratch), pair_ids)
        compared += 1
        if not same_ids(escaped_ids, pair_ids):
            mismatched += 1
    assert_true(compared >= MIN_CASES, "corpus is too small: " + String(compared))
    assert_equal(mismatched, 0)


def test_every_byte_matches_the_reference() raises:
    """The byte-level alphabet over its whole range, judged by the reference.

    `bytes.tsv` holds one case per byte: the text is the code point that byte
    stands for in the alphabet. It covers the 68 bytes that are not printable —
    and therefore not mapped to themselves — which no prose corpus reaches, and
    it takes the reference's word on each, rather than our expectation of what a
    byte should do.
    """
    var tokenizer = load_tokenizer_json(ESCAPED_JSON)
    var result = run_cases(tokenizer, BYTES_PATH)
    if result.failed > 0:
        print(result.report)
    assert_equal(result.total, BYTE_CASES)
    assert_equal(result.failed, 0)


def test_foreign_model_is_refused() raises:
    var error = refusal(FIXTURE_DIR + "/bad_model_type.json")
    assert_true(
        "unsupported_tokenizer" in error,
        "a Unigram model must be refused, got: " + error,
    )


def test_foreign_regex_is_refused() raises:
    """The pre-tokenizer is a hand-written matcher for one regex. A file asking
    for another one has to fail loudly, not tokenize differently in silence."""
    var error = refusal(FIXTURE_DIR + "/bad_regex.json")
    assert_true(
        "unsupported_tokenizer" in error,
        "a different pre-tokenizer regex must be refused, got: " + error,
    )


def test_prefix_space_is_refused() raises:
    var error = refusal(FIXTURE_DIR + "/bad_prefix_space.json")
    assert_true(
        "unsupported_tokenizer" in error,
        "add_prefix_space must be refused, got: " + error,
    )


def test_sparse_ids_are_refused() raises:
    """`Vocab` addresses tokens by id, so a hole in the ids would silently
    shift every token after it."""
    var error = refusal(FIXTURE_DIR + "/bad_vocab.json")
    assert_true("bad_vocab" in error, "sparse ids must be refused, got: " + error)


def test_broken_json_is_refused() raises:
    var error = refusal(FIXTURE_DIR + "/broken.json")
    assert_true("bad_json" in error, "truncated JSON must be refused, got: " + error)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def refusal(imm path: String) -> String:
    """The error text a file is refused with, or "" when it loads cleanly."""
    var message = ""
    try:
        _ = load_tokenizer_json(path)
    except err:
        message = String(err)
    return message


def run_cases(imm tokenizer: Tokenizer, imm path: String) raises -> CaseResult:
    """Encode every case and compare ids with the reference.

    `expected` comes from Hugging Face, not from us: the case file is written by
    `scripts/gen_tokenizer_json_fixtures.py`, which loads the fixture with the
    reference `tokenizers` and encodes the same corpus.
    """
    var raw = bytes_of(read_text_file(path))
    var result = CaseResult()
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

        actual.clear()
        tokenizer.encode(String(unsafe_from_utf8=scratch), actual)
        result.total += 1
        if not same_ids(actual, expected):
            result.failed += 1
            if result.failed <= 5:
                result.report += (
                    "  case "
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


def next_newline(imm raw: List[UInt8], start: Int) -> Int:
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


def hex_value(byte: UInt8) -> Int:
    var value = Int(byte)
    if value >= 48 and value <= 57:
        return value - 48
    if value >= 97 and value <= 102:
        return value - 87
    if value >= 65 and value <= 70:
        return value - 55
    return -1


def same_ids(actual: List[Int], expected: List[Int]) -> Bool:
    if len(actual) != len(expected):
        return False
    var index = 0
    while index < len(expected):
        if actual[index] != expected[index]:
            return False
        index += 1
    return True


def describe_ids(ids: List[Int]) -> String:
    var text = "["
    var index = 0
    while index < len(ids):
        if index > 0:
            text += ","
        text += String(ids[index])
        index += 1
    text += "]"
    return text
