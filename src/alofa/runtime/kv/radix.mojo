"""View B: a radix tree whose nodes point into the block pool.

Why a radix tree and not a hash chain
-------------------------------------

A hash chain (vLLM's APC) can only match prefixes that are block-aligned,
because the chain is keyed on whole blocks. A radix tree matches at token
granularity and forks anywhere, which is what makes multi-turn and agentic
workloads reuse anything at all: their shared prefixes are conversation-shaped,
not block-shaped.

The reason the tree can sit directly on a block pool — the thing SGLang's tree
and pager cannot do without a translation step — is that a node stores
`(offset, ntok)` next to its block list instead of a flat token range. Splitting
a node is then three integer updates:

    parent: (offset, k,             blocks[.. ceil((offset+k)/bs)])
    child:  ((offset+k) % bs, ntok-k, blocks[(offset+k)/bs ..])

No KV byte moves. When the split point falls inside a block, that one block ends
up referenced by *both* nodes, which is exactly what the shared refcount is for.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_kv_pool.mojo
"""

from alofa.core.error import ERR_CAPACITY, ERR_INVALID_ARGUMENT, AlofaError
from alofa.runtime.kv import (
    INVALID,
    MAX_NODE_BLOCKS,
    MAX_NODES,
    MAX_NODE_TOKENS,
    MAX_PATH,
    MAX_SEQ_TOKENS,
    ROOT,
    ceil_div,
    imin,
)


struct Match:
    """Result of a prefix lookup.

    `path` / `starts` are what the caller needs to *take* the matched blocks:
    the match may span several chained nodes, and each of them contributes its
    own blocks.
    """

    var node: Int
    var matched: Int
    var path: InlineArray[Int, MAX_PATH]
    var starts: InlineArray[Int, MAX_PATH]
    var n_path: Int

    def __init__(out self):
        self.node = INVALID
        self.matched = 0
        self.path = InlineArray[Int, MAX_PATH](fill=0)
        self.starts = InlineArray[Int, MAX_PATH](fill=0)
        self.n_path = 0


struct RadixTree:
    """Token-granularity prefix tree over block-granularity storage."""

    var alive: InlineArray[Int, MAX_NODES]
    var parent: InlineArray[Int, MAX_NODES]
    var child: InlineArray[Int, MAX_NODES]
    var sibling: InlineArray[Int, MAX_NODES]
    var ntok: InlineArray[Int, MAX_NODES]
    var offset: InlineArray[Int, MAX_NODES]
    var nblk: InlineArray[Int, MAX_NODES]
    var freq: InlineArray[Int, MAX_NODES]
    var tokens: InlineArray[Int, MAX_NODES * MAX_NODE_TOKENS]
    var blocks: InlineArray[Int, MAX_NODES * MAX_NODE_BLOCKS]
    var n_live: Int

    def __init__(out self):
        self.alive = InlineArray[Int, MAX_NODES](fill=0)
        self.parent = InlineArray[Int, MAX_NODES](fill=0)
        self.child = InlineArray[Int, MAX_NODES](fill=0)
        self.sibling = InlineArray[Int, MAX_NODES](fill=0)
        self.ntok = InlineArray[Int, MAX_NODES](fill=0)
        self.offset = InlineArray[Int, MAX_NODES](fill=0)
        self.nblk = InlineArray[Int, MAX_NODES](fill=0)
        self.freq = InlineArray[Int, MAX_NODES](fill=0)
        self.tokens = InlineArray[Int, MAX_NODES * MAX_NODE_TOKENS](fill=0)
        self.blocks = InlineArray[Int, MAX_NODES * MAX_NODE_BLOCKS](fill=0)
        for i in range(MAX_NODES):
            self.parent[i] = INVALID
            self.child[i] = INVALID
            self.sibling[i] = INVALID
        # The root holds no tokens; it exists so every real node has a parent.
        self.alive[ROOT] = 1
        self.n_live = 1

    # --- node plumbing ---

    def alloc_node(mut self) raises AlofaError -> Int:
        var node = INVALID
        var i = 1
        while node < 0 and i < MAX_NODES:
            if self.alive[i] == 0:
                node = i
            i += 1
        if node < 0:
            raise AlofaError(ERR_CAPACITY, "radix tree is full")
        self.alive[node] = 1
        self.parent[node] = INVALID
        self.child[node] = INVALID
        self.sibling[node] = INVALID
        self.ntok[node] = 0
        self.offset[node] = 0
        self.nblk[node] = 0
        self.freq[node] = 0
        for j in range(MAX_NODE_TOKENS):
            self.tokens[node * MAX_NODE_TOKENS + j] = 0
        for j in range(MAX_NODE_BLOCKS):
            self.blocks[node * MAX_NODE_BLOCKS + j] = 0
        self.n_live += 1
        return node

    def add_child(mut self, parent: Int, node: Int) raises AlofaError:
        if parent < 0 or parent >= MAX_NODES or self.alive[parent] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "add_child: bad parent")
        if node < 0 or node >= MAX_NODES or self.alive[node] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "add_child: bad node")
        self.sibling[node] = self.child[parent]
        self.child[parent] = node
        self.parent[node] = parent

    def detach(mut self, node: Int) raises AlofaError:
        """Unlink `node` from its parent. Its own children stay attached to it."""
        if node <= ROOT or node >= MAX_NODES or self.alive[node] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "detach: bad node")
        var parent = self.parent[node]
        if parent >= 0:
            if self.child[parent] == node:
                self.child[parent] = self.sibling[node]
            else:
                var c = self.child[parent]
                while c != INVALID:
                    if self.sibling[c] == node:
                        self.sibling[c] = self.sibling[node]
                        c = INVALID
                    else:
                        c = self.sibling[c]
        self.parent[node] = INVALID
        self.sibling[node] = INVALID

    def kill(mut self, node: Int) raises AlofaError:
        if node <= ROOT or node >= MAX_NODES or self.alive[node] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "kill: bad node")
        self.alive[node] = 0
        self.ntok[node] = 0
        self.nblk[node] = 0
        self.offset[node] = 0
        self.child[node] = INVALID
        self.sibling[node] = INVALID
        self.parent[node] = INVALID
        self.n_live -= 1

    # --- accessors ---

    def token_at(self, node: Int, i: Int) -> Int:
        return self.tokens[node * MAX_NODE_TOKENS + i]

    def block_at(self, node: Int, i: Int) -> Int:
        return self.blocks[node * MAX_NODE_BLOCKS + i]

    def set_token(mut self, node: Int, i: Int, value: Int):
        self.tokens[node * MAX_NODE_TOKENS + i] = value

    def set_block(mut self, node: Int, i: Int, value: Int):
        self.blocks[node * MAX_NODE_BLOCKS + i] = value

    # --- lookup and reshape ---

    def match(
        mut self, tokens: InlineArray[Int, MAX_SEQ_TOKENS], n: Int
    ) raises AlofaError -> Match:
        """Longest cached prefix of `tokens[0:n]`.

        Walks down comparing tokens; stops at the first divergence, whether that
        divergence is inside a node or between a node's children. Matching does
        not allocate and does not touch the pool — it is a pure lookup, which is
        what makes `MAT` in the fixture a no-op apart from the frequency bump.
        """
        var m = Match()
        var cur = self.child[ROOT]
        while cur != INVALID:
            var nt = self.ntok[cur]
            var i = 0
            while i < nt and m.matched + i < n:
                if self.token_at(cur, i) != tokens[m.matched + i]:
                    break
                i += 1
            if i == 0:
                break
            if m.n_path >= MAX_PATH:
                raise AlofaError(ERR_CAPACITY, "match path too deep")
            self.path_push(m, cur, m.matched)
            m.matched += i
            m.node = cur
            if i < nt:
                break  # diverged inside this node
            if m.matched >= n:
                break
            var nxt = INVALID
            var c = self.child[cur]
            while c != INVALID:
                if self.ntok[c] > 0:
                    if self.token_at(c, 0) == tokens[m.matched]:
                        nxt = c
                c = self.sibling[c]
            cur = nxt
        if m.node != INVALID:
            self.freq[m.node] += 1
        return m^

    def path_push(mut self, mut m: Match, node: Int, start: Int):
        m.path[m.n_path] = node
        m.starts[m.n_path] = start
        m.n_path += 1

    def split(mut self, node: Int, k: Int, block_size: Int) raises AlofaError -> Int:
        """Split `node` after `k` tokens; return the new child.

        Metadata only: the child's block ids are a suffix of the parent's, and
        the block the split point falls in is now referenced by both. The caller
        owns the refcount arithmetic (retain the child's blocks, drop the
        parent's tail) because only it knows the pool.
        """
        if node <= ROOT or node >= MAX_NODES or self.alive[node] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "split: bad node")
        if k <= 0 or k >= self.ntok[node]:
            raise AlofaError(ERR_INVALID_ARGUMENT, "split point must be interior")
        var child = self.alloc_node()
        var off = self.offset[node]
        var slot = off + k
        var first = slot // block_size
        var coff = slot % block_size
        var cn = self.ntok[node] - k

        self.offset[child] = coff
        self.ntok[child] = cn
        for i in range(cn):
            self.set_token(child, i, self.token_at(node, k + i))
        var cnblk = ceil_div(coff + cn, block_size)
        if cnblk > MAX_NODE_BLOCKS:
            raise AlofaError(ERR_CAPACITY, "split would overflow node blocks")
        self.nblk[child] = cnblk
        for i in range(cnblk):
            self.set_block(child, i, self.block_at(node, first + i))

        # The parent keeps its own prefix; note this leaves stale ids in the
        # parent's block array past nblk — invariants and digests only ever read
        # the first nblk entries, and the caller needs the tail to drop refs.
        self.ntok[node] = k
        self.nblk[node] = ceil_div(off + k, block_size)

        # The child takes over the parent's children: it *is* the continuation.
        self.child[child] = self.child[node]
        var c = self.child[child]
        while c != INVALID:
            self.parent[c] = child
            c = self.sibling[c]
        self.child[node] = child
        self.parent[child] = node
        self.sibling[child] = INVALID
        return child

    def digest(self) -> Int:
        var h = 0
        for i in range(MAX_NODES):
            h = (h * 1000003 + self.alive[i]) % 2147483647
            h = (h * 1000003 + self.ntok[i]) % 2147483647
            h = (h * 1000003 + self.offset[i]) % 2147483647
            h = (h * 1000003 + self.nblk[i]) % 2147483647
            h = (h * 1000003 + self.freq[i]) % 2147483647
            h = (h * 1000003 + self.parent[i] + 1) % 2147483647
            h = (h * 1000003 + self.child[i] + 1) % 2147483647
            h = (h * 1000003 + self.sibling[i] + 1) % 2147483647
        for i in range(MAX_NODES * MAX_NODE_TOKENS):
            h = (h * 1000003 + self.tokens[i]) % 2147483647
        for i in range(MAX_NODES * MAX_NODE_BLOCKS):
            h = (h * 1000003 + self.blocks[i]) % 2147483647
        return h
