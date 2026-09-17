"""Turning a KV operation into bytes, and bytes back into an operation.

Same contract as the scheduler's trace, for the same reason: the fixtures under
`tests/fixtures/kv/` come from `scripts/dump_kv_reference.py`, an *independent*
implementation of this policy. One character of disagreement fails the gate.

Rules inherited from that format: integers only, fields in a fixed order read by
position, and `-` for "nothing" rather than an empty string.

Two notes specific to this one:

- The per-operation `d=` digest covers the *whole* space (pool refcounts, free
  list order, every request, every node). Two runs that allocate the same number
  of blocks but disagree about which ones therefore differ on the first op where
  it matters, instead of silently agreeing on a count.
- The helpers here duplicate `parts_of` / `value_after` from
  `alofa.engine.trace`. That duplication is deliberate: L3 must not import L4,
  and a layering violation is a worse cost than eight lines of string splitting.

This file allocates. Recording a trace is opt-in and is not part of the
per-token path.

Run:
    pixi run mojo run -I src tests/unit/test_kv_pool.mojo
"""

from alofa.core.error import ERR_PARSE, AlofaError
from alofa.core.text import parse_int
from alofa.runtime.kv import (
    INVALID,
    MAX_BLOCKS,
    MAX_NODES,
    MAX_NODE_TOKENS,
    MAX_REQUESTS,
    MAX_SEQ_TOKENS,
    IdList,
    OpResult,
)
from alofa.runtime.kv.space import KvSpace

comptime EMPTY = "-"

comptime CFG_PREFIX = "CFG"
comptime OP_PREFIX = "OP="
comptime R_PREFIX = "R="

comptime OP_NEW = 0
comptime OP_APP = 1
comptime OP_COM = 2
comptime OP_MAT = 3
comptime OP_REL = 4
comptime OP_EVI = 5


def parts_of(text: String, sep: String) -> List[String]:
    """Split keeping empty fields as empty strings (they mean something here)."""
    var out = List[String]()
    for span in text.split(sep):
        out.append(String(span))
    return out^


def value_after(text: String, sep: String) raises AlofaError -> String:
    """Everything after the *first* `sep`."""
    if text.find(sep) < 0:
        raise AlofaError(ERR_PARSE, "missing '" + sep + "' in '" + text + "'")
    var pieces = List[String]()
    for span in text.split(sep, 1):
        pieces.append(String(span))
    if len(pieces) < 2:
        raise AlofaError(ERR_PARSE, "empty value in '" + text + "'")
    return pieces[1]


def num_or_dash(value: Int) -> String:
    if value == INVALID:
        return EMPTY
    return String(value)


def ids_line(ids: IdList) -> String:
    if ids.n == 0:
        return EMPTY
    var s = String(ids.at(0))
    var i = 1
    while i < ids.n:
        s += ","
        s += String(ids.at(i))
        i += 1
    return s


def config_line(block_size: Int) -> String:
    """`CFG bs=16 nb=112 nr=8 nn=32 nt=16` — the capacities the gate ran with."""
    return (
        CFG_PREFIX
        + " bs="
        + String(block_size)
        + " nb="
        + String(MAX_BLOCKS)
        + " nr="
        + String(MAX_REQUESTS)
        + " nn="
        + String(MAX_NODES)
        + " nt="
        + String(MAX_NODE_TOKENS)
    )


def parse_config(line: String) raises AlofaError -> Int:
    """Read back the block size, refusing a fixture built for other capacities.

    A capacity mismatch used to be the friendliest-looking failure in the repo:
    the ops would run, most would agree, and only the tail would differ. Better
    to refuse the file.
    """
    var fields = parts_of(line, " ")
    if len(fields) != 6:
        raise AlofaError(ERR_PARSE, "config line must have CFG plus five fields")
    if fields[0] != CFG_PREFIX:
        raise AlofaError(ERR_PARSE, "config line must start with CFG")
    var block_size = parse_int(value_after(fields[1], "="))
    var nb = parse_int(value_after(fields[2], "="))
    var nr = parse_int(value_after(fields[3], "="))
    var nn = parse_int(value_after(fields[4], "="))
    var nt = parse_int(value_after(fields[5], "="))
    if nb != MAX_BLOCKS:
        raise AlofaError(ERR_PARSE, "fixture was built for a different pool size")
    if nr != MAX_REQUESTS:
        raise AlofaError(ERR_PARSE, "fixture was built for a different request table")
    if nn != MAX_NODES:
        raise AlofaError(ERR_PARSE, "fixture was built for a different tree size")
    if nt != MAX_NODE_TOKENS:
        raise AlofaError(ERR_PARSE, "fixture was built for a different node width")
    return block_size


struct KvOp:
    """One operation read from a fixture."""

    var kind: Int
    var req: Int
    var node: Int
    var n: Int
    var tokens: InlineArray[Int, MAX_SEQ_TOKENS]

    def __init__(out self):
        self.kind = INVALID
        self.req = INVALID
        self.node = INVALID
        self.n = 0
        self.tokens = InlineArray[Int, MAX_SEQ_TOKENS](fill=0)


def parse_op(line: String) raises AlofaError -> KvOp:
    """Parse the `OP=...` half of a fixture line (the `R=` half is compared, not parsed)."""
    var op = KvOp()
    var fields = parts_of(line, " ")
    if len(fields) < 2:
        raise AlofaError(ERR_PARSE, "operation line needs at least two fields")
    var name = value_after(fields[0], "=")
    if name == "NEW":
        op.kind = OP_NEW
    elif name == "APP":
        op.kind = OP_APP
    elif name == "COM":
        op.kind = OP_COM
    elif name == "MAT":
        op.kind = OP_MAT
    elif name == "REL":
        op.kind = OP_REL
    elif name == "EVI":
        op.kind = OP_EVI
    else:
        raise AlofaError(ERR_PARSE, "unknown operation '" + name + "'")

    var i = 1
    while i < len(fields):
        var field = fields[i]
        if field.find("T=") == 0:
            var body = value_after(field, "=")
            var items = parts_of(body, ",")
            if len(items) > MAX_SEQ_TOKENS:
                raise AlofaError(ERR_PARSE, "too many tokens in one operation")
            for j in range(len(items)):
                op.tokens[j] = parse_int(items[j])
            op.n = len(items)
        elif field.find("r=") == 0:
            op.req = parse_int(value_after(field, "="))
        elif field.find("n=") == 0:
            op.n = parse_int(value_after(field, "="))
        elif field.find("v=") == 0:
            op.node = parse_int(value_after(field, "="))
        else:
            raise AlofaError(ERR_PARSE, "unknown field '" + field + "'")
        i += 1

    if op.kind == OP_NEW or op.kind == OP_MAT:
        if op.n <= 0:
            raise AlofaError(ERR_PARSE, "operation needs a token list")
    return op^


def apply_op(mut space: KvSpace, op: KvOp) raises AlofaError -> OpResult:
    """Run one parsed operation against a space.

    Lives here rather than in each test because it is a dispatch table: two
    copies of it would drift, and the copy that drifted would be the one
    deciding what a `REL` means.
    """
    if op.kind == OP_NEW:
        return space.new_request(op.req, op.tokens, op.n)
    if op.kind == OP_APP:
        return space.append_tokens(space.slot_of(op.req), op.n)
    if op.kind == OP_COM:
        return space.commit(space.slot_of(op.req))
    if op.kind == OP_MAT:
        return space.match_only(op.tokens, op.n)
    if op.kind == OP_REL:
        return space.release(space.slot_of(op.req))
    return space.evict(op.node)


def result_line(
    kind: Int, res: OpResult, used: Int, n_free: Int, digest: Int
) -> String:
    """Serialise what an operation did. Field order is part of the contract."""
    var s: String
    if kind == OP_NEW:
        s = (
            "m="
            + num_or_dash(res.matched)
            + " o="
            + num_or_dash(res.offset)
            + " S="
            + ids_line(res.shared)
            + " A="
            + ids_line(res.fresh)
        )
    elif kind == OP_APP:
        s = "A=" + ids_line(res.fresh)
    elif kind == OP_COM:
        s = (
            "m="
            + num_or_dash(res.matched)
            + " N="
            + ids_line(res.nodes)
            + " S="
            + ids_line(res.shared)
        )
    elif kind == OP_MAT:
        s = (
            "m="
            + num_or_dash(res.matched)
            + " H="
            + num_or_dash(res.node)
            + " o="
            + num_or_dash(res.offset)
        )
    elif kind == OP_REL:
        s = "X=" + ids_line(res.fresh)
    elif kind == OP_EVI:
        s = "X=" + ids_line(res.fresh)
    else:
        s = "?"
    s += " u=" + String(used) + " f=" + String(n_free) + " d=" + String(digest)
    return s


def op_line(kind: Int, req: Int, node: Int, n: Int, tokens: List[Int]) -> String:
    """Write one operation. Used by tests that build a fixture on the fly."""
    var name = "NEW"
    if kind == OP_APP:
        name = "APP"
    elif kind == OP_COM:
        name = "COM"
    elif kind == OP_MAT:
        name = "MAT"
    elif kind == OP_REL:
        name = "REL"
    elif kind == OP_EVI:
        name = "EVI"
    var s = OP_PREFIX + name
    if kind == OP_NEW or kind == OP_MAT:
        s += " T="
        for i in range(len(tokens)):
            if i > 0:
                s += ","
            s += String(tokens[i])
    elif kind == OP_APP:
        s += " r=" + String(req) + " n=" + String(n)
    elif kind == OP_EVI:
        s += " v=" + String(node)
    else:
        s += " r=" + String(req)
    return s


def expected_of(line: String) raises AlofaError -> String:
    """The `R=` half of a fixture line: what the reference says should happen."""
    return value_after(line, " R=")
