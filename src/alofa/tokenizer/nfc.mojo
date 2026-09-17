"""NFC normalization, which is what Qwen2's tokenizer applies before splitting.

Three steps in the UAX #15 order — decompose, reorder combining marks, recompose:

1. every code point is fully canonically decomposed (Hangul algorithmically);
2. runs of combining marks are stably sorted by their canonical combining class;
3. marks are composited back into the starter that precedes them, unless a
   mark in between blocks the match.

Skipping this is not an option even though most inputs survive it: in the
tokenizer's differential gate, a single mis-normalized byte maps the whole
string onto different merges, so an NFC bug surfaces as token mismatches with
no obvious cause. The combination that makes NFC cheap here is that the tables
are generated (see `unicode.mojo`) and used directly, rather than re-derived
per code point.
"""

from .unicode import Unicode, utf8_decode_at, utf8_encode_codepoint


def nfc(imm text: String, imm uni: Unicode) raises -> String:
    """Return `text` in Normalization Form C."""
    return nfc_range(text, 0, text.byte_length(), uni)


def nfc_range(imm text: String, start: Int, end: Int, imm uni: Unicode) raises -> String:
    """NFC over `text[start:end]`.

    The range form exists because encoding splits input into chunks around added
    tokens, and copying every chunk just to normalize it would put a fresh
    allocation on the encode path for a boundary that rarely occurs.
    """
    var decomposed = List[Int]()
    var index = start
    while index < end:
        var decoded = utf8_decode_at(text, index)
        if decoded.width == 0:
            break
        uni.decompose_into(decoded.cp, decomposed)
        index += decoded.width

    _canonical_order(decomposed, uni)

    var composed = _compose(decomposed, uni)

    var bytes = List[UInt8]()
    var k = 0
    while k < len(composed):
        utf8_encode_codepoint(composed[k], bytes)
        k += 1
    return String(unsafe_from_utf8=bytes)


def _canonical_order(mut sequence: List[Int], imm uni: Unicode) raises:
    """Sort each run of combining marks by combining class, stably.

    A stable bubble pass rather than a general sort: combining sequences are
    short, and stability is required — two marks of equal class must keep their
    original relative order or the result differs from NFC.
    """
    var i = 1
    while i < len(sequence):
        var current = uni.combining_class(sequence[i])
        if current != 0:
            var j = i
            while j > 0 and uni.combining_class(sequence[j - 1]) > current:
                var tmp = sequence[j]
                sequence[j] = sequence[j - 1]
                sequence[j - 1] = tmp
                j -= 1
        i += 1


def _compose(imm sequence: List[Int], imm uni: Unicode) raises -> List[Int]:
    """Recompose marks into their starter, honouring the blocking rule.

    A mark can only combine with the starter immediately preceding it if no
    mark in between has an equal or greater combining class; `last_mark`
    tracks that class and is only updated for marks that actually survived,
    since a mark consumed by a composition no longer sits between the starter
    and anything that follows.
    """
    var out = List[Int]()
    var starter = -1
    var last_mark = -1

    var i = 0
    while i < len(sequence):
        var value = sequence[i]
        var cc = uni.combining_class(value)
        i += 1
        if starter >= 0 and (cc == 0 or cc > last_mark):
            var merged = uni.compose(out[starter], value)
            if merged >= 0:
                out[starter] = merged
                continue
        out.append(value)
        if cc == 0:
            starter = len(out) - 1
            last_mark = -1
        else:
            last_mark = cc
    return out^
