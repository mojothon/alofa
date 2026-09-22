"""Vocabulary and merge tables, plus the merge loop itself.

Everything here works on **raw bytes** rather than on GPT-2's byte-level
alphabet. Those two views are isomorphic — each byte maps to exactly one
alphabet character — and bytes are what the reference expresses its tokens in,
so comparing against them directly removes a whole class of "did I map the
alphabet the other way round?" bugs.

Two details decide whether this agrees with tiktoken:

- merges are applied **by rank**, not left to right: every step takes the pair
  with the lowest rank anywhere in the piece. A greedy left-to-right scan picks
  different pairs and therefore different ids.
- ties are broken by position (earliest first), because that is what tiktoken's
  priority queue does.
"""

from .codes import TOK_ERR_MISSING_BYTE_TOKEN, TOK_ERR_MISSING_MERGE_TOKEN, tokenizer_error

comptime NO_TOKEN = -1
comptime NO_RANK = -1

# Token ids fit in 18 bits, so a merge pair packs into one Int key.
comptime ID_SHIFT = 18

# FNV-1a masked to keep intermediate products inside Int64.
comptime _FNV_PRIME = 1099511628211
comptime _MASK = 0x7FFFFFFFFFFFFFFF


def _hash_range(imm source: String, start: Int, end: Int) -> Int:
    var bytes = source.as_bytes()
    var hash_value: Int = 1469598103934665603
    var index = start
    while index < end:
        hash_value = ((hash_value ^ Int(bytes[index])) * _FNV_PRIME) & _MASK
        index += 1
    return hash_value


def _hash_blob(imm source: List[UInt8], start: Int, end: Int) -> Int:
    var hash_value: Int = 1469598103934665603
    var index = start
    while index < end:
        hash_value = ((hash_value ^ Int(source[index])) * _FNV_PRIME) & _MASK
        index += 1
    return hash_value


def _table_capacity(expected: Int) -> Int:
    """Power of two at least twice `expected`, so probes stay short."""
    var capacity = 16
    while capacity < 2 * expected:
        capacity *= 2
    return capacity


struct Vocab(Copyable, Movable):
    """Token bytes in one blob, plus a hash index over them.

    Tokens are appended in id order and then indexed by `seal`. The index stores
    only ids and every probe compares the actual bytes, so the table stays
    narrow and a hash collision cannot answer with the wrong token.
    """

    var blob: List[UInt8]
    var starts: List[Int]  # offsets into `blob`; length = token count + 1
    var buckets: List[Int]
    var mask: Int

    def __init__(out self, expected: Int):
        self.blob = List[UInt8]()
        self.starts = List[Int]()
        self.starts.append(0)
        var capacity = _table_capacity(expected)
        self.mask = capacity - 1
        self.buckets = List[Int]()
        var i = 0
        while i < capacity:
            self.buckets.append(NO_TOKEN)
            i += 1

    def count(imm self) -> Int:
        return len(self.starts) - 1

    def append_bytes(mut self, imm source: List[UInt8], start: Int, end: Int):
        """Append the token with bytes `source[start:end]` as the next id.

        Tokens must be appended in id order: `starts` doubles as the id -> offset
        map, so any other order silently scrambles every lookup.
        """
        var index = start
        while index < end:
            self.blob.append(source[index])
            index += 1
        self.starts.append(len(self.blob))

    def seal(mut self):
        """Index every token. Tokens are unique by construction, so probing only
        needs an empty slot rather than an equality check."""
        self.index_from(0)

    def index_from(mut self, first: Int):
        """Index tokens from `first` onwards; earlier ones are already indexed.

        Lets a vocabulary be extended after it was sealed without re-inserting
        every existing token.
        """
        var slot = first
        while slot < self.count():
            var probe = _hash_blob(self.blob, self.starts[slot], self.starts[slot + 1]) & self.mask
            while self.buckets[probe] != NO_TOKEN:
                probe = (probe + 1) & self.mask
            self.buckets[probe] = slot
            slot += 1

    def find(imm self, imm source: String, start: Int, end: Int) -> Int:
        """Id of the token whose bytes equal `source[start:end]`, or `NO_TOKEN`."""
        var bytes = source.as_bytes()
        var length = end - start
        var probe = _hash_range(source, start, end) & self.mask
        while True:
            var candidate = self.buckets[probe]
            if candidate == NO_TOKEN:
                return NO_TOKEN
            var token_start = self.starts[candidate]
            if self.starts[candidate + 1] - token_start == length:
                var offset = 0
                var matched = True
                while offset < length:
                    if bytes[start + offset] != self.blob[token_start + offset]:
                        matched = False
                        break
                    offset += 1
                if matched:
                    return candidate
            probe = (probe + 1) & self.mask

    def find_bytes(imm self, imm source: List[UInt8], start: Int, end: Int) -> Int:
        """Id of the token whose bytes equal `source[start:end]`, or `NO_TOKEN`.

        The byte-list form of `find`: token contents that come out of a
        `tokenizer.json` are held as bytes, not as a `String`, and building a
        `String` per token just to compare bytes would copy every token twice.
        """
        var length = end - start
        var probe = _hash_blob(source, start, end) & self.mask
        while True:
            var candidate = self.buckets[probe]
            if candidate == NO_TOKEN:
                return NO_TOKEN
            var token_start = self.starts[candidate]
            if self.starts[candidate + 1] - token_start == length:
                var offset = 0
                var matched = True
                while offset < length:
                    if source[start + offset] != self.blob[token_start + offset]:
                        matched = False
                        break
                    offset += 1
                if matched:
                    return candidate
            probe = (probe + 1) & self.mask

    def token_start(imm self, slot: Int) -> Int:
        return self.starts[slot]

    def token_end(imm self, slot: Int) -> Int:
        return self.starts[slot + 1]


struct Merges(Copyable, Movable):
    """`(left id, right id) -> rank`, where a lower rank merges sooner."""

    var keys: List[Int]
    var ranks: List[Int]
    var mask: Int

    def __init__(out self, expected: Int):
        var capacity = _table_capacity(expected)
        self.mask = capacity - 1
        self.keys = List[Int]()
        self.ranks = List[Int]()
        var i = 0
        while i < capacity:
            self.keys.append(NO_RANK)
            self.ranks.append(NO_RANK)
            i += 1

    def insert(mut self, left: Int, right: Int, rank: Int):
        """Record that `left` followed by `right` merges at `rank`."""
        var key = (left << ID_SHIFT) | right
        var probe = _hash_key(key) & self.mask
        while self.keys[probe] != NO_RANK:
            probe = (probe + 1) & self.mask
        self.keys[probe] = key
        self.ranks[probe] = rank

    def rank(imm self, left: Int, right: Int) -> Int:
        """Rank of this pair, or `NO_RANK` when it never merges."""
        var key = (left << ID_SHIFT) | right
        var probe = _hash_key(key) & self.mask
        while True:
            if self.keys[probe] == NO_RANK:
                return NO_RANK
            if self.keys[probe] == key:
                return self.ranks[probe]
            probe = (probe + 1) & self.mask


def _hash_key(key: Int) -> Int:
    # A small integer-mix; the table is powers of two, so using the raw packed
    # key would let structured input cluster.
    var hash_value = ((key ^ (key >> 33)) * 0xFF51AFD7ED558CCD) & _MASK
    hash_value = ((hash_value ^ (hash_value >> 33)) * 0xC4CEB9FE1A85EC53) & _MASK
    return hash_value ^ (hash_value >> 33)


def encode_piece(
    imm text: String,
    start: Int,
    end: Int,
    imm vocab: Vocab,
    imm merges: Merges,
    mut sink: List[Int]) raises:
    """Run BPE on one pre-token and append its ids to `sink`.

    `sink` rather than a returned list: the caller already owns a growing output
    list, and returning a list per pre-token would allocate once per piece.
    """
    var length = end - start
    if length <= 0:
        return

    if length == 1:
        var single = vocab.find(text, start, end)
        if single == NO_TOKEN:
            raise tokenizer_error(
                TOK_ERR_MISSING_BYTE_TOKEN,
                "byte "
                + String(Int(text.as_bytes()[start]))
                + " has no single-byte token; the vocabulary is not byte-complete",
            )
        sink.append(single)
        return

    # `bounds` are the boundaries between symbols (`length + 1` of them) and
    # `ids` holds the current token of each symbol. A merge drops a boundary.
    var bounds = List[Int]()
    var ids = List[Int]()
    var i = 0
    while i <= length:
        bounds.append(start + i)
        i += 1
    i = 0
    while i < length:
        var byte_id = vocab.find(text, start + i, start + i + 1)
        if byte_id == NO_TOKEN:
            raise tokenizer_error(
                TOK_ERR_MISSING_BYTE_TOKEN,
                "byte "
                + String(Int(text.as_bytes()[start + i]))
                + " has no single-byte token; the vocabulary is not byte-complete",
            )
        ids.append(byte_id)
        i += 1

    while len(ids) > 1:
        var best_rank = 0
        var best_at = NO_RANK
        i = 0
        while i + 1 < len(ids):
            var candidate = merges.rank(ids[i], ids[i + 1])
            if candidate != NO_RANK and (best_at == NO_RANK or candidate < best_rank):
                best_rank = candidate
                best_at = i
            i += 1
        if best_at == NO_RANK:
            break

        var merged = vocab.find(text, bounds[best_at], bounds[best_at + 2])
        if merged == NO_TOKEN:
            raise tokenizer_error(
                TOK_ERR_MISSING_MERGE_TOKEN,
                "rank "
                + String(best_rank)
                + " merges ids "
                + String(ids[best_at])
                + ","
                + String(ids[best_at + 1])
                + " but the result is not in the vocabulary",
            )
        ids[best_at] = merged
        _remove_at(ids, best_at + 1)
        _remove_at(bounds, best_at + 1)

    i = 0
    while i < len(ids):
        sink.append(ids[i])
        i += 1


def _remove_at(mut items: List[Int], index: Int):
    var i = index
    while i + 1 < len(items):
        items[i] = items[i + 1]
        i += 1
    _ = items.pop()
