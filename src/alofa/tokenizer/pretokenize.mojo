"""Qwen2's pre-tokenizer: the regex that decides where words begin and end.

The pattern itself comes straight out of `tokenizer.json`:

    (?i:'s|'t|'re|'ve|'m|'ll|'d)
    |[^\\r\\n p{L} p{N}]? p{L}+
    | p{N}
    | ?[^\\s p{L} p{N}]+[\\r\\n]*
    |\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

(spaces inserted above for readability; see `TOKENIZER_PATTERN`.)

Two properties of the Rust regex engine shape this implementation:

- **alternatives are leftmost-first**, so at a given offset the first branch in
  written order that matches wins. That is why `a  b` cuts as `a`, `' '`,
  `' b'`: the letter branch fires at the second space before the whitespace
  branches get a chance, and the space is absorbed as that branch's optional
  leading character. Trying branches in any other order silently changes the
  token stream.
- **quantifiers are greedy and the engine backtracks**, which matters for the
  whitespace branches. `\\s+` followed by `(?!\\S)` matches the whole run at end
  of input, and the run *minus its last character* otherwise.

Rather than port a regex engine, each branch is matched directly against the
UTF-8 bytes. That keeps the semantics above explicit and reviewable: a future
reader can check a branch against the pattern line it implements.
"""

from .unicode import Unicode, utf8_decode_at

# `COMPILE_TIME_PATTERN` is documentation only: the branches below are the
# implementation, and this constant is what they must keep agreeing with.
comptime TOKENIZER_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

comptime APOSTROPHE = 0x27
comptime SPACE = 0x20
comptime CR = 0x0D
comptime LF = 0x0A


def pretokenize(imm text: String, imm uni: Unicode) -> List[Int]:
    """Split normalized text into pre-tokens.

    Returns a flat list of half-open byte offsets: `[start, end, start, end,
    ...]`. Offsets rather than substrings because the pieces are immediately
    converted to token ids, so copying the bytes first would buy nothing.

    Text no branch matches is kept as its own piece instead of being dropped:
    `Split` keeps whatever the pattern did not cover, and dropping it would
    silently delete characters from the decoded round-trip.
    """
    var pieces = List[Int]()
    var n = text.byte_length()
    var index = 0
    var gap_start = -1

    while index < n:
        var length = _match_branch(text, index, uni)
        if length <= 0:
            if gap_start < 0:
                gap_start = index
            # Unmatched: advance one whole character so code points are never
            # split down the middle.
            index += _char_width(text, index)
            continue
        if gap_start >= 0:
            pieces.append(gap_start)
            pieces.append(index)
            gap_start = -1
        pieces.append(index)
        pieces.append(index + length)
        index += length

    if gap_start >= 0:
        pieces.append(gap_start)
        pieces.append(n)
    return pieces^


def _match_branch(imm text: String, start: Int, imm uni: Unicode) -> Int:
    """Length of the branch that matches at `start`, or 0 when none does.

    Branch order is the order in `TOKENIZER_PATTERN`; see the module docstring
    for why that order is part of the contract.
    """
    var contraction = _match_contraction(text, start)
    if contraction > 0:
        return contraction

    var letters = _match_letters(text, start, uni)
    if letters > 0:
        return letters

    var number = _match_number(text, start, uni)
    if number > 0:
        return number

    var punctuation = _match_punctuation(text, start, uni)
    if punctuation > 0:
        return punctuation

    return _match_whitespace(text, start, uni)


def _char_width(imm text: String, index: Int) -> Int:
    var decoded = utf8_decode_at(text, index)
    # `utf8_decode_at` reports width 0 past the end; treat a malformed lead byte
    # the same way here so a bad byte can never stall the scan.
    if decoded.width <= 0:
        return 1
    return decoded.width


def _simple_fold(cp: Int) -> Int:
    """Lower-case `cp` the way Unicode simple case folding does, for our needs.

    Rust applies full Unicode simple folding inside `(?i:...)`, so straying here
    means disagreeing with the reference on inputs that contain long s (U+017F)
    or Kelvin sign (U+212A). ASCII covers everything else the contraction branch
    can reach, since that branch only ever looks at Latin letters.
    """
    if cp >= 0x41 and cp <= 0x5A:
        return cp + 32
    if cp == 0x017F:  # long s folds to 's'
        return 0x73
    if cp == 0x212A:  # Kelvin sign folds to 'k'
        return 0x6B
    return cp


def _match_contraction(imm text: String, start: Int) -> Int:
    """Branch 1: `(?i:'s|'t|'re|'ve|'m|'ll|'d)`.

    The alternatives are distinguished by their first letter, so trying them in
    written order and taking the first that fits reproduces leftmost-first exactly.
    """
    var head = utf8_decode_at(text, start)
    if head.cp != APOSTROPHE:
        return 0

    var first = utf8_decode_at(text, start + head.width)
    if first.width <= 0:
        return 0
    var base = start + head.width + first.width
    var folded = _simple_fold(first.cp)

    if folded == 0x73 or folded == 0x74 or folded == 0x6D or folded == 0x64:
        return base - start

    # 're, 've, 'll need a second letter; without it this branch fails and the
    # apostrophe falls through to the punctuation branch.
    if folded != 0x72 and folded != 0x76 and folded != 0x6C:
        return 0
    var second = utf8_decode_at(text, base)
    if second.width <= 0:
        return 0
    var expected = 0x65
    if folded == 0x6C:
        expected = 0x6C
    if _simple_fold(second.cp) != expected:
        return 0
    return base + second.width - start


def _match_letters(imm text: String, start: Int, imm uni: Unicode) -> Int:
    """Branch 2: `[^\\r\\n p{L} p{N}]? p{L}+` — one optional prefix plus letters.

    Numbers are excluded from the prefix and CR/LF is excluded outright, which
    is what keeps `'\\n'` with the newline branches instead of being swallowed
    by a following word.
    """
    var head = utf8_decode_at(text, start)
    if head.width <= 0 or uni.is_number(head.cp) or head.cp == CR or head.cp == LF:
        return 0

    var index = start
    if not uni.is_letter(head.cp):
        # Consume the optional prefix only when letters actually follow; the
        # `[^\\r\\n p{L} p{N}]?` asks for one character, not one-or-none greedy.
        index = start + head.width
        var next = utf8_decode_at(text, index)
        if next.width <= 0 or not uni.is_letter(next.cp):
            return 0

    while index < text.byte_length():
        var decoded = utf8_decode_at(text, index)
        if not uni.is_letter(decoded.cp):
            break
        index += decoded.width
    return index - start


def _match_number(imm text: String, start: Int, imm uni: Unicode) -> Int:
    """Branch 3: ` p{N}` — exactly one digit.

    Qwen2 quantifies this differently from GPT-4's `{1,3}`, and it shows: the
    reference tokenizes `12345` as five separate ids.
    """
    var head = utf8_decode_at(text, start)
    if head.width <= 0 or not uni.is_number(head.cp):
        return 0
    return head.width


def _match_punctuation(imm text: String, start: Int, imm uni: Unicode) -> Int:
    """Branch 4: ` ?[^\\s p{L} p{N}]+[\\r\\n]*`, including its trailing newlines.

    The optional leading character here is a literal ASCII space, not `\\s`: a
    tab before punctuation does not join the piece.
    """
    # `as_bytes()` materialises a whole copy, so it is taken once per call
    # rather than once per iteration below.
    var bytes = text.as_bytes()
    var index = start
    if start < text.byte_length() and Int(bytes[start]) == SPACE:
        index = start + 1

    var scan = index
    while scan < text.byte_length():
        var decoded = utf8_decode_at(text, scan)
        if (
            uni.is_space(decoded.cp)
            or uni.is_letter(decoded.cp)
            or uni.is_number(decoded.cp)
        ):
            break
        scan += decoded.width
    if scan == index:
        return 0

    # `[\r\n]*`: newlines belong to this piece, which is why `!!!\n` stays one
    # pre-token rather than leaving a separate newline behind.
    index = scan
    while index < text.byte_length():
        var decoded = utf8_decode_at(text, index)
        if decoded.cp != CR and decoded.cp != LF:
            break
        index += decoded.width
    return index - start


def _match_whitespace(imm text: String, start: Int, imm uni: Unicode) -> Int:
    """Branches 5–7: `\\s*[\\r\\n]+`, `\\s+(?!\\S)`, `\\s+`, in that order.

    All three act on the maximal whitespace run starting here, so it is computed
    once and each branch is expressed as what it does to that run.
    """
    var index = start
    var last_start = start
    var last_codepoint = -1
    var newline_start = -1
    var newline_width = 0

    while index < text.byte_length():
        var decoded = utf8_decode_at(text, index)
        if not uni.is_space(decoded.cp):
            break
        if decoded.cp == CR or decoded.cp == LF:
            newline_start = index
            newline_width = decoded.width
        last_start = index
        last_codepoint = decoded.cp
        index += decoded.width

    if index == start:
        return 0

    # Branch 5: everything through the last newline in the run. `\\s*` is greedy
    # and backtracks from the end of the run, and `[\r\n]+` then extends to the
    # end of that newline sequence — which is the last newline itself.
    if newline_start >= 0:
        return newline_start + newline_width - start

    # Branch 6: `\\s+(?!\\S)`. The lookahead can only succeed either at the very
    # end of the text, or after giving one character back.
    if index == text.byte_length():
        return index - start
    if last_start > start and last_codepoint >= 0:
        return last_start - start

    # Branch 7: `\\s+` takes the run as it stands.
    return index - start
