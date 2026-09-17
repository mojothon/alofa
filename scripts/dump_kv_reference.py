#!/usr/bin/env python3
"""Export reference traces for the unified KV address space.

This is an *independent* implementation of the same policy as
`src/alofa/runtime/kv/`. That independence is the whole value of the file: if
the fixture came out of the Mojo code, a bug in the Mojo code would export a
fixture that agrees with itself and the gate would stay green forever.

It is a line-by-line port, so it reads like the Mojo and not like idiomatic
Python. That is deliberate — every difference in style is a place where the two
could disagree without either being wrong, and the gate exists to make that
disagreement loud.

Two variants are exported:

- `split` (default): a partial match splits the cached node, so a request that
  shares 3 tokens of a 16-token node shares those 3 tokens.
- `nosplit`: a partial match is truncated to the last fully matched node — the
  "block/node aligned sharing, no reshaping" behaviour a hash-chain design
  falls back to. It is self-consistent and it is worse. `alt.trace` is built
  from it, so the fixture is proved to be sensitive to the policy and not just
  to the file format.

Run:
    /home/rontom/anaconda3/bin/python scripts/dump_kv_reference.py
"""

import argparse
import os

MAX_BLOCKS = 112
MAX_BLOCKS_PER_SEQ = 16
MAX_REQUESTS = 8
MAX_NODES = 32
MAX_NODE_TOKENS = 16
MAX_NODE_BLOCKS = 4
MAX_SEQ_TOKENS = 64
MAX_PATH = 8
ROOT = 0
INVALID = -1
DIGEST_MOD = 2147483647
DIGEST_MUL = 1000003

OUT_DIR = "tests/fixtures/kv"


def ceil_div(a, b):
    return (a + b - 1) // b


class AlofaError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


ERR_CAPACITY = 6
ERR_DOUBLE_FREE = 11
ERR_INVALID_ARGUMENT = 2
ERR_OUT_OF_RANGE = 3


class BlockPool:
    def __init__(self, block_size):
        if block_size <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "block_size must be positive")
        self.block_size = block_size
        self.refcnt = [0] * MAX_BLOCKS
        self.free_stack = [MAX_BLOCKS - 1 - i for i in range(MAX_BLOCKS)]
        self.n_free = MAX_BLOCKS
        self.used = 0

    def alloc_one(self):
        if self.n_free <= 0:
            raise AlofaError(ERR_CAPACITY, "kv block pool is exhausted")
        self.n_free -= 1
        block = self.free_stack[self.n_free]
        if self.refcnt[block] != 0:
            raise AlofaError(ERR_DOUBLE_FREE, "free list holds a referenced block")
        self.refcnt[block] = 1
        self.used += 1
        return block

    def retain(self, block):
        if block < 0 or block >= MAX_BLOCKS:
            raise AlofaError(ERR_OUT_OF_RANGE, "block id out of range")
        if self.refcnt[block] <= 0:
            raise AlofaError(ERR_DOUBLE_FREE, "retain of a block nobody holds")
        self.refcnt[block] += 1

    def release(self, block):
        if block < 0 or block >= MAX_BLOCKS:
            raise AlofaError(ERR_OUT_OF_RANGE, "block id out of range")
        if self.refcnt[block] <= 0:
            raise AlofaError(ERR_DOUBLE_FREE, "release of a block nobody holds")
        self.refcnt[block] -= 1
        if self.refcnt[block] > 0:
            return False
        self.free_stack[self.n_free] = block
        self.n_free += 1
        self.used -= 1
        return True

    def digest(self):
        h = 0
        for i in range(MAX_BLOCKS):
            h = (h * DIGEST_MUL + self.refcnt[i]) % DIGEST_MOD
        for i in range(self.n_free):
            h = (h * DIGEST_MUL + self.free_stack[i]) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.n_free) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.used) % DIGEST_MOD
        return h


class Match:
    def __init__(self):
        self.node = INVALID
        self.matched = 0
        self.path = [0] * MAX_PATH
        self.starts = [0] * MAX_PATH
        self.n_path = 0


class RadixTree:
    def __init__(self):
        self.alive = [0] * MAX_NODES
        self.parent = [INVALID] * MAX_NODES
        self.child = [INVALID] * MAX_NODES
        self.sibling = [INVALID] * MAX_NODES
        self.ntok = [0] * MAX_NODES
        self.offset = [0] * MAX_NODES
        self.nblk = [0] * MAX_NODES
        self.freq = [0] * MAX_NODES
        self.tokens = [0] * (MAX_NODES * MAX_NODE_TOKENS)
        self.blocks = [0] * (MAX_NODES * MAX_NODE_BLOCKS)
        self.n_live = 1
        self.alive[ROOT] = 1

    def alloc_node(self):
        node = INVALID
        i = 1
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

    def add_child(self, parent, node):
        if parent < 0 or parent >= MAX_NODES or self.alive[parent] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "add_child: bad parent")
        self.sibling[node] = self.child[parent]
        self.child[parent] = node
        self.parent[node] = parent

    def detach(self, node):
        if node <= ROOT or node >= MAX_NODES or self.alive[node] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "detach: bad node")
        parent = self.parent[node]
        if parent >= 0:
            if self.child[parent] == node:
                self.child[parent] = self.sibling[node]
            else:
                c = self.child[parent]
                while c != INVALID:
                    if self.sibling[c] == node:
                        self.sibling[c] = self.sibling[node]
                        c = INVALID
                    else:
                        c = self.sibling[c]
        self.parent[node] = INVALID
        self.sibling[node] = INVALID

    def kill(self, node):
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

    def token_at(self, node, i):
        return self.tokens[node * MAX_NODE_TOKENS + i]

    def block_at(self, node, i):
        return self.blocks[node * MAX_NODE_BLOCKS + i]

    def match(self, tokens, n):
        m = Match()
        cur = self.child[ROOT]
        while cur != INVALID:
            nt = self.ntok[cur]
            i = 0
            while i < nt and m.matched + i < n:
                if self.token_at(cur, i) != tokens[m.matched + i]:
                    break
                i += 1
            if i == 0:
                break
            if m.n_path >= MAX_PATH:
                raise AlofaError(ERR_CAPACITY, "match path too deep")
            m.path[m.n_path] = cur
            m.starts[m.n_path] = m.matched
            m.n_path += 1
            m.matched += i
            m.node = cur
            if i < nt:
                break
            if m.matched >= n:
                break
            nxt = INVALID
            c = self.child[cur]
            while c != INVALID:
                if self.ntok[c] > 0 and self.token_at(c, 0) == tokens[m.matched]:
                    nxt = c
                c = self.sibling[c]
            cur = nxt
        if m.node != INVALID:
            self.freq[m.node] += 1
        return m

    def split(self, node, k, block_size):
        if node <= ROOT or node >= MAX_NODES or self.alive[node] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "split: bad node")
        if k <= 0 or k >= self.ntok[node]:
            raise AlofaError(ERR_INVALID_ARGUMENT, "split point must be interior")
        child = self.alloc_node()
        off = self.offset[node]
        slot = off + k
        first = slot // block_size
        coff = slot % block_size
        cn = self.ntok[node] - k
        self.offset[child] = coff
        self.ntok[child] = cn
        for i in range(cn):
            self.tokens[child * MAX_NODE_TOKENS + i] = self.token_at(node, k + i)
        cnblk = ceil_div(coff + cn, block_size)
        if cnblk > MAX_NODE_BLOCKS:
            raise AlofaError(ERR_CAPACITY, "split would overflow node blocks")
        self.nblk[child] = cnblk
        for i in range(cnblk):
            self.blocks[child * MAX_NODE_BLOCKS + i] = self.block_at(node, first + i)
        self.ntok[node] = k
        self.nblk[node] = ceil_div(off + k, block_size)
        self.child[child] = self.child[node]
        c = self.child[child]
        while c != INVALID:
            self.parent[c] = child
            c = self.sibling[c]
        self.child[node] = child
        self.parent[child] = node
        self.sibling[child] = INVALID
        return child

    def digest(self):
        h = 0
        for i in range(MAX_NODES):
            h = (h * 1000003 + self.alive[i]) % DIGEST_MOD
            h = (h * 1000003 + self.ntok[i]) % DIGEST_MOD
            h = (h * 1000003 + self.offset[i]) % DIGEST_MOD
            h = (h * 1000003 + self.nblk[i]) % DIGEST_MOD
            h = (h * 1000003 + self.freq[i]) % DIGEST_MOD
            h = (h * 1000003 + self.parent[i] + 1) % DIGEST_MOD
            h = (h * 1000003 + self.child[i] + 1) % DIGEST_MOD
            h = (h * 1000003 + self.sibling[i] + 1) % DIGEST_MOD
        for i in range(MAX_NODES * MAX_NODE_TOKENS):
            h = (h * 1000003 + self.tokens[i]) % DIGEST_MOD
        for i in range(MAX_NODES * MAX_NODE_BLOCKS):
            h = (h * 1000003 + self.blocks[i]) % DIGEST_MOD
        return h


class OpResult:
    def __init__(self):
        self.shared = []
        self.fresh = []
        self.nodes = []
        self.matched = INVALID
        self.offset = INVALID
        self.node = INVALID
        self.used = 0
        self.n_free = 0

    def push_unique(self, lst, value):
        if lst and lst[-1] == value:
            return
        lst.append(value)


class KvSpace:
    def __init__(self, block_size, variant="split"):
        self.pool = BlockPool(block_size)
        self.tree = RadixTree()
        self.variant = variant
        self.rq_id = [0] * MAX_REQUESTS
        self.rq_live = [0] * MAX_REQUESTS
        self.rq_ntok = [0] * MAX_REQUESTS
        self.rq_off = [0] * MAX_REQUESTS
        self.rq_nblk = [0] * MAX_REQUESTS
        self.rq_priv = [0] * MAX_REQUESTS
        self.rq_tok = [0] * (MAX_REQUESTS * MAX_SEQ_TOKENS)
        self.rq_blk = [0] * (MAX_REQUESTS * MAX_BLOCKS_PER_SEQ)
        self.rq_blen = [0] * (MAX_REQUESTS * MAX_BLOCKS_PER_SEQ)
        self.rq_bstart = [0] * (MAX_REQUESTS * MAX_BLOCKS_PER_SEQ)

    def block_at(self, slot, i):
        return self.rq_blk[slot * MAX_BLOCKS_PER_SEQ + i]

    def blen_at(self, slot, i):
        return self.rq_blen[slot * MAX_BLOCKS_PER_SEQ + i]

    def bstart_at(self, slot, i):
        return self.rq_bstart[slot * MAX_BLOCKS_PER_SEQ + i]

    def token_at(self, slot, i):
        return self.rq_tok[slot * MAX_SEQ_TOKENS + i]

    def find_request(self, req):
        for i in range(MAX_REQUESTS):
            if self.rq_live[i] == 1 and self.rq_id[i] == req:
                return i
        return INVALID

    def free_slot(self):
        for i in range(MAX_REQUESTS):
            if self.rq_live[i] == 0:
                return i
        raise AlofaError(ERR_CAPACITY, "no free request slot")

    def slot_of(self, req):
        slot = self.find_request(req)
        if slot < 0:
            raise AlofaError(ERR_OUT_OF_RANGE, "unknown request id")
        return slot

    def add_block(self, slot, block, start, take):
        """Returns True when the block became a new page-table entry.

        A refcount is one per view entry: two consecutive tree nodes may
        contribute the same (boundary) block to one request, and retaining it
        twice would put the pool out of step with the page table.

        Merging requires the run to be slot-contiguous, not merely the same
        block: sharing a block does not make two runs adjacent inside it.
        """
        idx = self.rq_nblk[slot]
        if idx > 0 and self.block_at(slot, idx - 1) == block:
            if self.bstart_at(slot, idx - 1) + self.blen_at(slot, idx - 1) == start:
                self.rq_blen[slot * MAX_BLOCKS_PER_SEQ + idx - 1] += take
                return False
        if idx >= MAX_BLOCKS_PER_SEQ:
            raise AlofaError(ERR_CAPACITY, "request block table is full")
        self.rq_blk[slot * MAX_BLOCKS_PER_SEQ + idx] = block
        self.rq_blen[slot * MAX_BLOCKS_PER_SEQ + idx] = take
        self.rq_bstart[slot * MAX_BLOCKS_PER_SEQ + idx] = start
        self.rq_nblk[slot] = idx + 1
        return True

    def adjust(self, m):
        """Variant hook: `nosplit` drops a partially matched node."""
        if self.variant != "nosplit":
            return m
        if m.n_path > 0:
            last = m.path[m.n_path - 1]
            used_here = m.matched - m.starts[m.n_path - 1]
            if used_here < self.tree.ntok[last]:
                m.n_path -= 1
                m.matched = m.starts[m.n_path] if m.n_path > 0 else 0
                m.node = m.path[m.n_path - 1] if m.n_path > 0 else INVALID
        return m

    def new_request(self, req, tokens, n):
        if n <= 0 or n > MAX_SEQ_TOKENS:
            raise AlofaError(ERR_INVALID_ARGUMENT, "bad token count")
        if self.find_request(req) >= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "duplicate request id")
        slot = self.free_slot()
        bs = self.pool.block_size
        res = OpResult()
        for i in range(n):
            self.rq_tok[slot * MAX_SEQ_TOKENS + i] = tokens[i]
        m = self.adjust(self.tree.match(tokens, n))
        res.matched = m.matched
        self.rq_off[slot] = 0
        if m.matched > 0:
            res.node = m.node
            res.offset = self.tree.offset[m.path[0]]
            self.rq_off[slot] = res.offset
            for p in range(m.n_path):
                node = m.path[p]
                off = self.tree.offset[node]
                stop_at = min(m.matched, m.starts[p] + self.tree.ntok[node])
                used_here = stop_at - m.starts[p]
                if used_here <= 0:
                    continue
                need = ceil_div(off + used_here, bs)
                for j in range(need):
                    block = self.tree.block_at(node, j)
                    start = max(j * bs, off)
                    stop = min((j + 1) * bs, off + used_here)
                    take = stop - start
                    if take <= 0:
                        continue
                    if self.add_block(slot, block, start - j * bs, take):
                        self.pool.retain(block)
                        res.push_unique(res.shared, block)
        remaining = n - m.matched
        while remaining > 0:
            block = self.pool.alloc_one()
            take = min(bs, remaining)
            self.add_block(slot, block, 0, take)
            res.fresh.append(block)
            remaining -= take
        self.rq_ntok[slot] = n
        self.rq_id[slot] = req
        self.rq_live[slot] = 1
        self.rq_priv[slot] = 1 if res.fresh else 0
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res

    def append_tokens(self, slot, n):
        if n <= 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "bad token count")
        bs = self.pool.block_size
        res = OpResult()
        remaining = n
        if self.rq_priv[slot] == 1 and self.rq_nblk[slot] > 0:
            last = self.rq_nblk[slot] - 1
            room = bs - self.blen_at(slot, last)
            if room > 0:
                take = min(room, remaining)
                self.rq_blen[slot * MAX_BLOCKS_PER_SEQ + last] += take
                remaining -= take
        while remaining > 0:
            block = self.pool.alloc_one()
            take = min(bs, remaining)
            self.add_block(slot, block, 0, take)
            res.fresh.append(block)
            remaining -= take
        self.rq_ntok[slot] += n
        self.rq_priv[slot] = 1
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res

    def commit(self, slot):
        bs = self.pool.block_size
        res = OpResult()
        n = self.rq_ntok[slot]
        tokens = [self.rq_tok[slot * MAX_SEQ_TOKENS + i] for i in range(n)]
        m = self.adjust(self.tree.match(tokens, n))
        res.matched = m.matched
        parent = ROOT
        if m.matched > 0:
            last = m.node
            used_here = m.matched - m.starts[m.n_path - 1]
            if used_here < self.tree.ntok[last]:
                old_nblk = self.tree.nblk[last]
                child = self.tree.split(last, used_here, bs)
                for i in range(self.tree.nblk[child]):
                    self.pool.retain(self.tree.block_at(child, i))
                i = self.tree.nblk[last]
                while i < old_nblk:
                    self.pool.release(self.tree.block_at(last, i))
                    i += 1
                res.nodes.append(child)
            parent = last
            res.node = last
        k = m.matched
        remaining = n - k
        cursor = 0
        acc = 0
        while cursor < self.rq_nblk[slot]:
            if acc + self.blen_at(slot, cursor) > k:
                break
            acc += self.blen_at(slot, cursor)
            cursor += 1
        inblk = k - acc
        if cursor < self.rq_nblk[slot]:
            inblk += self.bstart_at(slot, cursor)
        while remaining > 0:
            cap = min(MAX_NODE_TOKENS, MAX_NODE_BLOCKS * bs - inblk)
            take = min(cap, remaining)
            nb = ceil_div(inblk + take, bs)
            if cursor + nb > self.rq_nblk[slot]:
                raise AlofaError(ERR_INVALID_ARGUMENT, "request blocks exhausted")
            node = self.tree.alloc_node()
            self.tree.offset[node] = inblk
            self.tree.ntok[node] = take
            for i in range(take):
                self.tree.tokens[node * MAX_NODE_TOKENS + i] = tokens[k + i]
            self.tree.nblk[node] = nb
            for i in range(nb):
                block = self.block_at(slot, cursor + i)
                self.tree.blocks[node * MAX_NODE_BLOCKS + i] = block
                self.pool.retain(block)
                res.push_unique(res.shared, block)
            self.tree.add_child(parent, node)
            res.nodes.append(node)
            parent = node
            k += take
            remaining -= take
            slots = inblk + take
            cursor += slots // bs
            inblk = slots % bs
        self.rq_priv[slot] = 0
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res

    def match_only(self, tokens, n):
        res = OpResult()
        m = self.adjust(self.tree.match(tokens, n))
        res.matched = m.matched
        if m.matched > 0:
            res.node = m.node
            res.offset = self.tree.offset[m.path[0]]
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res

    def release(self, slot):
        res = OpResult()
        for i in range(self.rq_nblk[slot]):
            block = self.block_at(slot, i)
            if self.pool.release(block):
                res.fresh.append(block)
        self.rq_live[slot] = 0
        self.rq_nblk[slot] = 0
        self.rq_ntok[slot] = 0
        self.rq_off[slot] = 0
        self.rq_priv[slot] = 0
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res

    def evict(self, node):
        if node <= ROOT or node >= MAX_NODES or self.tree.alive[node] == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "evict: bad node")
        res = OpResult()
        self.tree.detach(node)
        stack = [node]
        while stack:
            cur = stack.pop()
            c = self.tree.child[cur]
            while c != INVALID:
                stack.append(c)
                c = self.tree.sibling[c]
            for i in range(self.tree.nblk[cur]):
                block = self.tree.block_at(cur, i)
                if self.pool.release(block):
                    res.fresh.append(block)
            self.tree.kill(cur)
        res.node = node
        res.used = self.pool.used
        res.n_free = self.pool.n_free
        return res

    def digest(self):
        h = self.pool.digest()
        h = (h * DIGEST_MUL + self.tree.digest()) % DIGEST_MOD
        for slot in range(MAX_REQUESTS):
            h = (h * DIGEST_MUL + self.rq_live[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_id[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_ntok[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_off[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_nblk[slot]) % DIGEST_MOD
            h = (h * DIGEST_MUL + self.rq_priv[slot]) % DIGEST_MOD
            for i in range(self.rq_nblk[slot]):
                h = (h * DIGEST_MUL + self.block_at(slot, i)) % DIGEST_MOD
                h = (h * DIGEST_MUL + self.blen_at(slot, i)) % DIGEST_MOD
                h = (h * DIGEST_MUL + self.bstart_at(slot, i)) % DIGEST_MOD
        return h

    def check_invariants(self):
        """Mirrors the Mojo check so the reference itself is not silently wrong."""
        bad = 0
        counted = [0] * MAX_BLOCKS
        for slot in range(MAX_REQUESTS):
            if self.rq_live[slot] == 0:
                continue
            total = 0
            for i in range(self.rq_nblk[slot]):
                block = self.block_at(slot, i)
                if block < 0 or block >= MAX_BLOCKS:
                    bad += 1
                    continue
                counted[block] += 1
                total += self.blen_at(slot, i)
                if self.blen_at(slot, i) <= 0 or self.blen_at(slot, i) > self.pool.block_size:
                    bad += 1
                if self.bstart_at(slot, i) < 0:
                    bad += 1
                if self.bstart_at(slot, i) + self.blen_at(slot, i) > self.pool.block_size:
                    bad += 1
            if total != self.rq_ntok[slot]:
                bad += 1
        for node in range(MAX_NODES):
            if self.tree.alive[node] == 0 or node == ROOT:
                continue
            if self.tree.ntok[node] <= 0 or self.tree.ntok[node] > MAX_NODE_TOKENS:
                bad += 1
            if self.tree.nblk[node] != ceil_div(
                self.tree.offset[node] + self.tree.ntok[node], self.pool.block_size
            ):
                bad += 1
            if self.tree.nblk[node] > MAX_NODE_BLOCKS:
                bad += 1
            for i in range(self.tree.nblk[node]):
                counted[self.tree.block_at(node, i)] += 1
        for block in range(MAX_BLOCKS):
            if counted[block] != self.pool.refcnt[block]:
                bad += 1
        if self.pool.n_free + self.pool.used != MAX_BLOCKS:
            bad += 1
        seen = [0] * MAX_BLOCKS
        for i in range(self.pool.n_free):
            seen[self.pool.free_stack[i]] += 1
        for block in range(MAX_BLOCKS):
            want = 0 if self.pool.refcnt[block] > 0 else 1
            if seen[block] != want:
                bad += 1
        return bad


# --- trace format (must match src/alofa/runtime/kv/trace.mojo) ---

def dash(value):
    return "-" if value == INVALID else str(value)


def ids_line(ids):
    return ",".join(str(x) for x in ids) if ids else "-"


def config_line(block_size):
    return (
        "CFG bs=%d nb=%d nr=%d nn=%d nt=%d"
        % (block_size, MAX_BLOCKS, MAX_REQUESTS, MAX_NODES, MAX_NODE_TOKENS)
    )


def op_text(op):
    kind = op[0]
    if kind == "NEW":
        return "OP=NEW r=%d T=%s" % (op[1], ",".join(str(t) for t in op[2]))
    if kind == "MAT":
        return "OP=MAT T=%s" % ",".join(str(t) for t in op[1])
    if kind == "APP":
        return "OP=APP r=%d n=%d" % (op[1], op[2])
    if kind == "EVI":
        return "OP=EVI v=%d" % op[1]
    return "OP=%s r=%d" % (kind, op[1])


def result_line(kind, res, digest):
    if kind == "NEW":
        s = "m=%s o=%s S=%s A=%s" % (
            dash(res.matched), dash(res.offset), ids_line(res.shared), ids_line(res.fresh)
        )
    elif kind == "APP":
        s = "A=%s" % ids_line(res.fresh)
    elif kind == "COM":
        s = "m=%s N=%s S=%s" % (dash(res.matched), ids_line(res.nodes), ids_line(res.shared))
    elif kind == "MAT":
        s = "m=%s H=%s o=%s" % (dash(res.matched), dash(res.node), dash(res.offset))
    elif kind == "REL":
        s = "X=%s" % ids_line(res.fresh)
    elif kind == "EVI":
        s = "X=%s" % ids_line(res.fresh)
    else:
        raise ValueError("unknown op " + kind)
    return s + " u=%d f=%d d=%d" % (res.used, res.n_free, digest)


def apply_op(space, op):
    """Run one scenario op against a space and return its result.

    Split out of `run` so that other exporters (see `dump_paged_reference.py`)
    can drive a `KvSpace` with the same op lists and keep the address space
    they describe in step with the traces this file writes.
    """
    kind = op[0]
    if kind == "NEW":
        return space.new_request(op[1], list(op[2]), len(op[2]))
    if kind == "MAT":
        return space.match_only(list(op[1]), len(op[1]))
    if kind == "APP":
        return space.append_tokens(space.slot_of(op[1]), op[2])
    if kind == "COM":
        return space.commit(space.slot_of(op[1]))
    if kind == "REL":
        return space.release(space.slot_of(op[1]))
    if kind == "EVI":
        return space.evict(op[1])
    raise ValueError("unknown op " + kind)


def run(name, description, block_size, ops, variant="split"):
    space = KvSpace(block_size, variant)
    lines = [
        "# alofa kv trace v1",
        "# scenario: %s (block_size=%d, variant=%s)" % (name, block_size, variant),
        "# %s" % description,
        config_line(block_size),
    ]
    for op in ops:
        kind = op[0]
        res = apply_op(space, op)
        bad = space.check_invariants()
        if bad != 0:
            raise SystemExit(
                "reference itself broke %d invariants on %s after %s" % (bad, name, op_text(op))
            )
        lines.append("%s R=%s" % (op_text(op), result_line(kind, res, space.digest())))
    return "\n".join(lines) + "\n"


P = list(range(1, 21))  # a 20-token prompt
Q = list(range(100, 140))  # a 40-token prompt


def scenarios():
    return [
        (
            "s01_prompt_sharing",
            "three requests over one cached prompt: share, diverge, re-match",
            16,
            [
                ("NEW", 1, P, None),
                ("COM", 1, None, None),
                ("MAT", P, None),
                ("NEW", 2, P[:5] + [50], None),
                ("COM", 2, None, None),
                ("NEW", 3, P + [99], None),
                ("COM", 3, None, None),
                ("REL", 1, None, None),
                ("REL", 2, None, None),
                ("REL", 3, None, None),
                ("MAT", P, None),
                ("MAT", P[:5] + [50], None),
            ],
        ),
        (
            "s02_split_divergence",
            "two divergences inside one cached node: the split must be metadata only",
            16,
            [
                ("NEW", 1, [7] * 8, None),
                ("COM", 1, None, None),
                ("NEW", 2, [7, 7, 7, 8], None),
                ("COM", 2, None, None),
                ("NEW", 3, [7] * 7 + [9], None),
                ("COM", 3, None, None),
                ("MAT", [7] * 7 + [9], None),
                ("MAT", [7, 7, 7, 8], None),
                ("MAT", [7, 7], None),
                ("REL", 1, None, None),
                ("REL", 2, None, None),
                ("REL", 3, None, None),
                ("EVI", 1, None, None),
                ("MAT", [7] * 8, None),
            ],
        ),
        (
            "s03_long_prompt_chain",
            "a 40-token prompt becomes a chain of nodes; a later split lands mid-chain",
            16,
            [
                ("NEW", 1, Q, None),
                ("COM", 1, None, None),
                ("MAT", Q[:20], None),
                ("NEW", 2, Q[:20] + [200], None),
                ("COM", 2, None, None),
                ("NEW", 3, Q[:5] + [201], None),
                ("COM", 3, None, None),
                ("REL", 1, None, None),
                ("REL", 2, None, None),
                ("REL", 3, None, None),
                ("MAT", Q, None),
                ("MAT", Q[:5] + [201], None),
            ],
        ),
        (
            "s04_append_grow",
            "decode grows a request: in place while the tail block is private",
            16,
            [
                ("NEW", 1, [11, 12, 13], None),
                ("APP", 1, 5, None),
                ("APP", 1, 10, None),
                ("COM", 1, None, None),
                ("MAT", [11, 12, 13], None),
                (
                    "NEW",
                    2,
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                    None,
                ),
                ("APP", 2, 4, None),
                ("COM", 2, None, None),
                ("REL", 2, None, None),
                ("REL", 1, None, None),
                ("MAT", [11, 12, 13], None),
            ],
        ),
        (
            "s05_evict_and_reuse",
            "evict a subtree, then watch the free list hand the blocks back in order",
            16,
            [
                ("NEW", 1, list(range(1, 11)), None),
                ("COM", 1, None, None),
                ("NEW", 2, list(range(1, 11)) + [21], None),
                ("COM", 2, None, None),
                ("NEW", 3, list(range(1, 11)) + [22], None),
                ("COM", 3, None, None),
                ("REL", 1, None, None),
                ("REL", 2, None, None),
                ("REL", 3, None, None),
                ("MAT", list(range(1, 11)) + [21], None),
                ("EVI", 1, None, None),
                ("MAT", list(range(1, 11)), None),
                ("NEW", 4, list(range(1, 11)), None),
                ("COM", 4, None, None),
                ("REL", 4, None, None),
            ],
        ),
        (
            "s06_small_blocks",
            "block_size=4: many blocks per request, sharing across a small block",
            4,
            [
                ("NEW", 1, list(range(1, 21)), None),
                ("COM", 1, None, None),
                ("NEW", 2, list(range(1, 11)) + [99], None),
                ("COM", 2, None, None),
                ("APP", 2, 3, None),
                ("REL", 1, None, None),
                ("NEW", 3, list(range(1, 11)), None),
                ("COM", 3, None, None),
                ("REL", 2, None, None),
                ("REL", 3, None, None),
                ("MAT", list(range(1, 21)), None),
                ("MAT", list(range(1, 11)) + [99], None),
            ],
        ),
        (
            "s07_refcount_pressure",
            "one block shared by a cached node and its continuation: both must let go",
            16,
            [
                ("NEW", 1, list(range(1, 13)) + [31], None),
                ("COM", 1, None, None),
                ("NEW", 2, list(range(1, 13)) + [32], None),
                ("COM", 2, None, None),
                ("REL", 1, None, None),
                ("NEW", 3, list(range(1, 13)) + [33], None),
                ("COM", 3, None, None),
                ("REL", 2, None, None),
                ("EVI", 2, None, None),
                ("MAT", list(range(1, 13)) + [31], None),
                ("EVI", 1, None, None),
                ("MAT", list(range(1, 13)), None),
                ("REL", 3, None, None),
            ],
        ),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for name, desc, bs, ops in scenarios():
        if args.only and args.only != name:
            continue
        text = run(name, desc, bs, ops)
        path = os.path.join(args.out_dir, name + ".trace")
        with open(path, "w") as fh:
            fh.write(text)
        print("wrote %s (%d ops)" % (path, len(ops)))

    # A fixture that is wrong on purpose: the gate must reject it, otherwise the
    # byte-exact comparison is not comparing anything.
    src = os.path.join(args.out_dir, "s02_split_divergence.trace")
    with open(src) as fh:
        good = fh.read().rstrip("\n").split("\n")
    lines = []
    for line in good:
        if line.startswith("OP=COM r=2"):
            head, res = line.split(" R=")
            fields = res.split(" ")
            fixed = []
            for f in fields:
                if f.startswith("u="):
                    fixed.append("u=%d" % (int(f[2:]) + 1))
                else:
                    fixed.append(f)
            line = head + " R=" + " ".join(fixed)
        lines.append(line)
    bad_path = os.path.join(args.out_dir, "bad.trace")
    with open(bad_path, "w") as fh:
        fh.write(
            "# alofa kv trace v1\n"
            "# scenario: bad — s02 with one op's used count bumped by one\n"
            "# this file must be REJECTED by the gate\n"
            + "\n".join(l for l in lines if not l.startswith("#"))
            + "\n"
        )
    print("wrote %s" % bad_path)

    # The equally-consistent-but-different policy: no node splitting.
    alt = run(
        "alt",
        "same plan as s02 but with node splitting disabled: prefixes only match "
        "up to the last fully matched node",
        16,
        [op for op in scenarios()[1][3]],
        variant="nosplit",
    )
    alt_path = os.path.join(args.out_dir, "alt.trace")
    with open(alt_path, "w") as fh:
        fh.write(alt)
    print("wrote %s" % alt_path)


if __name__ == "__main__":
    main()
