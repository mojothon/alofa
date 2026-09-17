"""Unicode property lookups shared by pre-tokenization and NFC normalization.

The tables come from `scripts/gen_unicode_tables.py`, which derives them from
Python's `unicodedata` — **do not hand-edit them**. A hand-written category
table is the classic quiet failure here: it agrees with the reference on
almost every script and disagrees on one, and then it is a randomly located
differential failure rather than an obvious bug.

This layer answers exactly four questions: is this code point a letter, is it
a number, is it whitespace, and what does it canonically decompose to. How to
cut words or rank merges is somebody else's job.
"""

from .unicode_data import (
    combining_class_table,
    composition_table,
    decomposition_table,
    letter_ranges,
    number_ranges,
    whitespace_points,
)

# Hangul syllables follow the algorithm in UAX #15: keeping 11172 of them out
# of the tables is what keeps those tables two orders of magnitude smaller.
comptime SBASE = 0xAC00
comptime LBASE = 0x1100
comptime VBASE = 0x1161
comptime TBASE = 0x11A7
comptime LCOUNT = 19
comptime VCOUNT = 21
comptime TCOUNT = 28
comptime NCOUNT = 588  # VCOUNT * TCOUNT
comptime SCOUNT = 11172  # LCOUNT * NCOUNT

comptime REPLACEMENT = 0xFFFD


struct Decoded(Copyable, Movable):
    """One UTF-8 character: the code point and how many bytes encode it."""

    var cp: Int
    var width: Int

    def __init__(out self, cp: Int, width: Int):
        self.cp = cp
        self.width = width


def utf8_decode_at(imm text: String, index: Int) -> Decoded:
    """Decode one code point at a byte offset.

    Malformed sequences yield `REPLACEMENT` and a width of one byte, which is
    the same trade-off UTF-8 libraries make: scanning has to make progress, and
    a tokenizer that threw on bad input would be unable to process the input it
    is most likely to meet in the wild.
    """
    if index < 0 or index >= text.byte_length():
        return Decoded(REPLACEMENT, 0)

    # `as_bytes()` is materialised once: indexing it inside a loop would rebuild
    # the whole byte sequence on every iteration.
    var bytes = text.as_bytes()
    var lead = Int(bytes[index])
    if lead < 0x80:
        return Decoded(lead, 1)

    var extra = 3 if lead >= 0xF0 else (2 if lead >= 0xE0 else (1 if lead >= 0xC0 else -1))
    if extra < 0:
        # A continuation byte where a lead byte belongs is not a valid start.
        return Decoded(REPLACEMENT, 1)

    var mask = 0x07 if extra == 3 else (0x0F if extra == 2 else 0x1F)
    var cp = lead & mask

    if index + extra >= text.byte_length():
        return Decoded(REPLACEMENT, 1)

    var i = 1
    while i <= extra:
        var cont = Int(bytes[index + i])
        if (cont & 0xC0) != 0x80:
            return Decoded(REPLACEMENT, 1)
        cp = (cp << 6) | (cont & 0x3F)
        i += 1
    return Decoded(cp, extra + 1)


def utf8_encode_codepoint(cp: Int, mut sink: List[UInt8]) raises:
    """Append one code point as UTF-8. Surrogates are rejected."""
    if cp < 0 or cp > 0x10FFFF or (cp >= 0xD800 and cp <= 0xDFFF):
        raise Error("not a scalar value: " + String(cp))
    if cp < 0x80:
        sink.append(UInt8(cp))
    elif cp < 0x800:
        sink.append(UInt8(0xC0 | (cp >> 6)))
        sink.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        sink.append(UInt8(0xE0 | (cp >> 12)))
        sink.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        sink.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        sink.append(UInt8(0xF0 | (cp >> 18)))
        sink.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        sink.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        sink.append(UInt8(0x80 | (cp & 0x3F)))


struct Unicode(Copyable, Movable):
    """Unicode property tables, loaded once and shared.

    Loaded once because building them costs some twelve thousand appends; a
    tokenizer that rebuilt them per call would spend all its time on setup.
    """

    var letters: List[Int]
    var numbers: List[Int]
    var spaces: List[Int]
    var decompositions: List[Int]  # [cp, count, cp...] sorted by cp
    var decomp_keys: List[Int]  # code points only, for binary search
    var decomp_offsets: List[Int]  # where each entry starts in `decompositions`
    var compositions: List[Int]  # [a, b, cp] sorted by (a, b)
    var combining_classes: List[Int]  # [cp, ccc] sorted by cp

    def __init__(out self):
        self.letters = letter_ranges()
        self.numbers = number_ranges()
        self.spaces = whitespace_points()
        self.decompositions = decomposition_table()
        self.compositions = composition_table()
        self.combining_classes = combining_class_table()

        # Entries are variable width, so the flat table cannot be binary
        # searched directly: index it by code point here, then search and
        # follow the offset. Scanning linearly per lookup would turn the
        # 5736-case gate into hundreds of millions of comparisons.
        self.decomp_keys = List[Int]()
        self.decomp_offsets = List[Int]()
        var i = 0
        while i < len(self.decompositions):
            self.decomp_keys.append(self.decompositions[i])
            self.decomp_offsets.append(i)
            i += 2 + self.decompositions[i + 1]

    def is_letter(imm self, cp: Int) -> Bool:
        """`p{L}`: letters, as the pre-tokenizer's pattern means it."""
        return _in_ranges(self.letters, cp)

    def is_number(imm self, cp: Int) -> Bool:
        """`p{N}`: decimal digits, letter numbers and other numbers."""
        return _in_ranges(self.numbers, cp)

    def is_space(imm self, cp: Int) -> Bool:
        """`s` in the regex sense: the Unicode White_Space property.

        Deliberately not `isspace()`: the two disagree on 0x1C–0x1F, and the
        pattern being matched is a regex, so the regex definition wins.
        """
        return _in_points(self.spaces, cp)

    def combining_class(imm self, cp: Int) -> Int:
        """Canonical combining class; 0 for anything that is not a mark."""
        var value = _lookup_pairs(self.combining_classes, cp)
        if value < 0:
            return 0
        return value

    def decompose_into(imm self, cp: Int, mut sink: List[Int]):
        """Append `cp`'s full canonical decomposition onto `sink`.

        The `_into` form exists because the obtain-one-list variant would
        allocate once per input character, and normalization runs over every
        character of every prompt.
        """
        if cp >= SBASE and cp < SBASE + SCOUNT:
            var offset = cp - SBASE
            var final = offset % TCOUNT
            sink.append(LBASE + offset // NCOUNT)
            sink.append(VBASE + (offset % NCOUNT) // TCOUNT)
            if final != 0:
                sink.append(TBASE + final)
            return

        var slot = _index_of(self.decomp_keys, cp)
        if slot >= 0:
            var base = self.decomp_offsets[slot]
            var n = self.decompositions[base + 1]
            var k = 0
            while k < n:
                sink.append(self.decompositions[base + 2 + k])
                k += 1
            return
        sink.append(cp)

    def decompose(imm self, cp: Int) -> List[Int]:
        """Full canonical decomposition, Hangul syllables included.

        Returns a single-element list when the code point stands alone, so
        callers can treat the result uniformly.
        """
        var out = List[Int]()
        self.decompose_into(cp, out)
        return out^

    def compose(imm self, first: Int, second: Int) -> Int:
        """Canonical composition of two code points, or -1 if they do not.

        Hangul is composited arithmetically; everything else comes from the
        generated table, which only contains pairs the reference
        implementation actually composites back.
        """
        if first >= LBASE and first < LBASE + LCOUNT:
            if second >= VBASE and second < VBASE + VCOUNT:
                return SBASE + ((first - LBASE) * VCOUNT + (second - VBASE)) * TCOUNT
            return -1
        if (
            first >= SBASE
            and first < SBASE + SCOUNT
            and (first - SBASE) % TCOUNT == 0
            and second > TBASE
            and second < TBASE + TCOUNT
        ):
            return first + (second - TBASE)

        var target = first * 0x110000 + second
        var lo = 0
        var hi = len(self.compositions) // 3
        while lo < hi:
            var mid = (lo + hi) // 2
            var a = self.compositions[3 * mid]
            var b = self.compositions[3 * mid + 1]
            var key = a * 0x110000 + b
            if key == target:
                return self.compositions[3 * mid + 2]
            if key < target:
                lo = mid + 1
            else:
                hi = mid
        return -1


def _in_ranges(imm ranges: List[Int], cp: Int) -> Bool:
    var lo = 0
    var hi = len(ranges) // 2
    while lo < hi:
        var mid = (lo + hi) // 2
        if cp < ranges[2 * mid]:
            hi = mid
        elif cp > ranges[2 * mid + 1]:
            lo = mid + 1
        else:
            return True
    return False


def _in_points(imm points: List[Int], cp: Int) -> Bool:
    var lo = 0
    var hi = len(points)
    while lo < hi:
        var mid = (lo + hi) // 2
        if points[mid] == cp:
            return True
        if points[mid] < cp:
            lo = mid + 1
        else:
            hi = mid
    return False


def _lookup_pairs(imm table: List[Int], key: Int) -> Int:
    """`table` is [key, value, ...] sorted by key; -1 when absent."""
    var lo = 0
    var hi = len(table) // 2
    while lo < hi:
        var mid = (lo + hi) // 2
        var candidate = table[2 * mid]
        if candidate == key:
            return table[2 * mid + 1]
        if candidate < key:
            lo = mid + 1
        else:
            hi = mid
    return -1


def _index_of(imm sorted_keys: List[Int], key: Int) -> Int:
    """Binary index of `key` in a sorted list, or -1 when absent."""
    var lo = 0
    var hi = len(sorted_keys)
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_keys[mid] == key:
            return mid
        if sorted_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return -1
