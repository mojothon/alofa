"""引擎的 KV 房间：调度器数的块，是真块。

判定纪律
--------

**参照物是独立实现。** `tests/fixtures/kvroom/*.trace` 由
`scripts/dump_kv_room_reference.py` 导出 —— 同一份策略的 Python 实现，自己维护
一本账（每请求 prompt 长度、是否已发布）。参照物若由被测实现导出，门恒真。

**逐字节，不容差。** 这一层输出的不是数值而是**归属**：多少块在用、多少块空闲、
多少块属于缓存、整棵树的摘要。归属差一块就是另一套归属。

**三条常驻对照。** 一个不会失败的门等于没有门：
- `bad_room.trace`（把第一拍的 used 加一）→ 必须判红，否则逐字节比对根本没在比；
- `alt_room.trace`（换个 prompt 再跑同一串操作）→ 必须判红，否则 fixture 只分辨
  得了格式、分辨不了内容；
- `bad_room_alloc.mojo`（给房间加会增长的堆容器）→ 零分配扫描必须判违规，否则
  忙碌循环里每拍都可能 malloc。

**还有一类不依赖参照物的性质。** 参照物两边同时漏掉一次 retain 是可能的，所以
每拍都从视图重算一遍不变量（房间自己的 `invariants()`），并且断言
`used + n_free == MAX_BLOCKS` —— 池子里的块要么在用要么空闲，没有第三种状态。

边界（写在这里以免被误读）：`runtime/kv` 的并发上限是 `MAX_REQUESTS`（8）、单条
序列上限 `MAX_SEQ_TOKENS`（64）。房间对第 9 条报 `capacity`、对更长的序列报
`out_of_range`，这些限制被**断言**而不是被绕过 —— 悄悄少给一个请求几块，会在三拍
之后变成"少了一个答案"，那是最坏的位置。

Run:
    pixi run mojo run -I src tests/unit/test_kv_room.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.text import parse_int, read_text
from alofa.engine.kv_room import KvRoom
from alofa.runtime.kv import MAX_BLOCKS, MAX_REQUESTS, MAX_SEQ_TOKENS

comptime FIXTURE = "tests/fixtures/kvroom/"
comptime BAD_SRC = "tests/fixtures/bad_room_alloc.mojo"
comptime ROOM_SRC = "src/alofa/engine/kv_room.mojo"
comptime SCENARIOS = 5


def lines_of(path: String) raises -> List[String]:
    var out = List[String]()
    for span in read_text(path).split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        out.append(line)
    return out^


def parts_of(text: String) -> List[String]:
    var out = List[String]()
    for span in text.split(" "):
        var item = String(span)
        if item.byte_length() == 0:
            continue
        out.append(item)
    return out^


def value_of(field: String) raises -> Int:
    """`r=7` → 7。"""

    var at = field.find("=")
    if at < 0:
        raise AlofaError(1, "field without '=': " + field)
    return parse_int(String(field[byte=at + 1:]))


struct ReplayStats:
    """一次重放的体检结果。"""

    var ops: Int
    var mismatches: Int
    var inv_bad: Int
    var pool_bad: Int
    var first_want: String
    var first_got: String

    def __init__(out self):
        self.ops = 0
        self.mismatches = 0
        self.inv_bad = 0
        self.pool_bad = 0
        self.first_want = String("")
        self.first_got = String("")


def tokens_of(field: String) raises -> InlineArray[Int, MAX_SEQ_TOKENS]:
    """`tk=1001,1002,…` → 定长数组。"""

    var toks = InlineArray[Int, MAX_SEQ_TOKENS](fill=0)
    var at = field.find("=")
    if at < 0:
        raise AlofaError(1, "token field without '='")
    var body = field[byte=at + 1:]
    var n = 0
    for span in body.split(","):
        var item = String(span)
        if item.byte_length() == 0:
            continue
        if n >= MAX_SEQ_TOKENS:
            raise AlofaError(1, "fixture has more tokens than a sequence holds")
        toks[n] = parse_int(item)
        n += 1
    return toks^


def tail_of(fields: List[String]) -> String:
    """最后四个字段原样拼回 —— 比对的是字节，不是解析之后的数。"""

    var last = len(fields)
    var out = String("")
    for i in range(last - 4, last):
        if i > last - 4:
            out += " "
        out += fields[i]
    return out^


def observed(mut room: KvRoom) raises -> String:
    return (
        "u="
        + String(room.used())
        + " f="
        + String(room.n_free())
        + " c="
        + String(room.cached_blocks())
        + " d="
        + String(room.digest())
    )


def replay(path: String, mut stats: ReplayStats) raises -> Int:
    var lines = lines_of(path)
    if len(lines) < 2:
        raise AlofaError(1, "fixture must start with CFG and have at least one op")
    var head = parts_of(lines[0])
    if len(head) != 2 or head[0] != "CFG":
        raise AlofaError(1, "fixture must start with 'CFG bs=…'")
    var room = KvRoom(value_of(head[1]))
    var i = 1
    while i < len(lines):
        var fields = parts_of(lines[i])
        var op = fields[0]
        if op == "ADM":
            _ = room.admit(
                value_of(fields[1]),
                tokens_of(fields[4]),
                value_of(fields[2]),
                value_of(fields[3]),
            )
        elif op == "EXT":
            _ = room.extend(value_of(fields[1]), value_of(fields[2]))
        elif op == "PUB":
            _ = room.publish(value_of(fields[1]))
        elif op == "DRP":
            _ = room.drop(value_of(fields[1]))
        elif op == "REC":
            _ = room.reclaim(value_of(fields[1]))
        else:
            raise AlofaError(1, "unknown op: " + op)
        stats.ops += 1
        var want = tail_of(fields)
        var got = observed(room)
        if want != got:
            stats.mismatches += 1
            if stats.first_want.byte_length() == 0:
                stats.first_want = want
                stats.first_got = got
        # 两本账都从视图重算，不依赖参照物：参照物两边同时漏一次 retain 也能抓到。
        if room.invariants() != 0:
            stats.inv_bad += 1
        if room.used() + room.n_free() != MAX_BLOCKS:
            stats.pool_bad += 1
        i += 1
    return stats.ops


def scenario_names() -> InlineArray[String, SCENARIOS]:
    var out = InlineArray[String, SCENARIOS](fill="")
    out[0] = "r01_second_prompt_is_free"
    out[1] = "r02_sliced_prompt"
    out[2] = "r03_decode_growth"
    out[3] = "r04_evict_returns_cache"
    out[4] = "r05_hit_then_reclaim"
    return out^


def scan_growable(text: String) -> String:
    """第一个出现的"会增长的容器"，没有则空串。"""

    if text.find("List[") >= 0:
        return "List["
    if text.find("Dict[") >= 0:
        return "Dict["
    if text.find("alloc(") >= 0:
        return "alloc("
    if text.find("String(") >= 0:
        return "String("
    return ""


def prompt_tokens(base: Int, n: Int) -> InlineArray[Int, MAX_SEQ_TOKENS]:
    var toks = InlineArray[Int, MAX_SEQ_TOKENS](fill=0)
    for i in range(n):
        toks[i] = base + i
    return toks^


# --- 1. fixture 逐字节 ---


def test_every_room_fixture_replays_byte_for_byte() raises:
    var names = scenario_names()
    var total = 0
    for name in names:
        var stats = ReplayStats()
        _ = replay(FIXTURE + name + ".trace", stats)
        assert_true(stats.ops > 0, name + " 是空的")
        assert_equal(
            stats.mismatches,
            0,
            name
            + " 第 "
            + String(stats.mismatches)
            + " 处不一致：想要 "
            + stats.first_want
            + " 得到 "
            + stats.first_got,
        )
        assert_equal(stats.inv_bad, 0, name + " 有拍子不满足不变量")
        assert_equal(stats.pool_bad, 0, name + " 有拍子 used + n_free != MAX_BLOCKS")
        total += stats.ops
    assert_true(total > 40, "fixture 太少，门没有真的在比")


def test_a_corrupted_fixture_is_caught() raises:
    """逐字节比对必须**会失败**：坏夹具若不判红，上面那条门是恒真的。"""
    var stats = ReplayStats()
    _ = replay(FIXTURE + "bad_room.trace", stats)
    assert_true(stats.mismatches > 0, "被改坏的 fixture 竟然对上了")


def test_a_different_content_is_caught() raises:
    """换一个 prompt 重放同一串操作，摘要必须不同。

    只验格式的话，任何一串合法操作都会通过；这条保证 fixture 分辨得了**内容**。
    """
    var stats = ReplayStats()
    _ = replay(FIXTURE + "alt_room.trace", stats)
    assert_true(stats.mismatches > 0, "换了 prompt 的 fixture 竟然对上了")


# --- 2. 性质：不依赖参照物 ---


def test_identical_prompts_share_blocks() raises:
    """第二条相同 prompt 的命中长度必须等于 prompt 长度，且不再新拿块。

    反面是"每条请求都自己算一遍"：输出逐字节也一样，前缀缓存就只是个名字。
    """
    var room = KvRoom(16)
    var toks = prompt_tokens(1001, 32)
    _ = room.admit(1, toks, 32, 32)
    _ = room.publish(1)
    _ = room.drop(1)
    var after_first = room.used()
    _ = room.admit(2, toks, 32, 32)
    assert_equal(room.last_matched, 32, "相同 prompt 没有命中已缓存的前缀")
    assert_equal(room.last_fresh, 0, "命中之后竟然还新拿了块")
    assert_equal(room.used(), after_first, "共享之后池子用量变了")


def test_a_finished_request_keeps_its_blocks_until_reclaim() raises:
    """完成即发布：块换了主人（进缓存），used 不降；只有回收能把它们还回来。

    反面是"完成即归零"：调度器会以为池子空着，而缓存其实还占着。
    """
    var room = KvRoom(16)
    _ = room.admit(1, prompt_tokens(2001, 32), 32, 32)
    _ = room.grow_to(1, 40)
    var held = room.used()
    _ = room.publish(1)
    _ = room.drop(1)
    assert_equal(room.used(), held, "发布之后占用竟然降了 —— 块凭空消失了")
    assert_true(room.cached_blocks() > 0, "发布之后缓存是空的")
    var free_before = room.n_free()
    var freed = room.reclaim(room.cached_blocks())
    assert_true(freed > 0, "回收一个满是缓存的房间竟然还回 0 块")
    assert_equal(room.n_free(), free_before + freed, "还回来的块没有真的回到池子")
    assert_equal(room.take_freed(), freed, "上报给调度器的数和实测的不一样")
    assert_equal(room.take_freed(), 0, "同一次归还被上报了两次")


def test_an_unpublished_drop_gives_the_blocks_back() raises:
    """抢占/取消没有发布过，块必须真的回池 —— 它们和缓存无关。"""
    var room = KvRoom(16)
    _ = room.admit(1, prompt_tokens(3001, 32), 32, 32)
    var held = room.used()
    var free_before = room.n_free()
    _ = room.drop(1)
    assert_true(room.used() < held, "未发布就丢弃，块却没回池")
    assert_equal(room.n_free() - free_before, held - room.used())
    assert_equal(room.cached_blocks(), 0, "没发布过的请求竟然留下了缓存")


def test_reclaim_cannot_touch_a_live_request() raises:
    """活着的请求的块不能被回收 —— 回收的是缓存，不是正在读的 KV。

    驱逐一个正在被引用的节点只能剪掉树上的引用；块本身还在。所以"要 64 块"在这
    里必须还回 0，而不是把活请求的历史抽走。
    """
    var room = KvRoom(16)
    _ = room.admit(1, prompt_tokens(4001, 32), 32, 32)
    var held = room.used()
    var freed = room.reclaim(64)
    assert_equal(freed, 0, "回收竟然动了活着的请求的块")
    assert_equal(room.used(), held, "活着的请求的块数变了")
    assert_equal(room.n_tok(1), 32, "活着的请求的序列被截短了")


def test_a_half_delivered_prompt_is_not_published() raises:
    """prompt 没交完就不许发布：半个前缀进缓存，后面的人会命中半个前缀。"""
    var room = KvRoom(16)
    _ = room.admit(1, prompt_tokens(5001, 64), 64, 16)
    var kind = String("")
    try:
        _ = room.publish(1)
    except err:
        kind = err.name()
    assert_true(kind == "invalid_argument", "半个 prompt 竟然能发布，得到 " + kind)


# --- 3. 边界与具名拒绝 ---


def test_the_room_refuses_what_it_cannot_hold() raises:
    """第 9 条并发请求、超长序列、不存在的请求 —— 都得具名拒绝。

    悄悄少给一块会在几拍之后变成"少一个 token"，那是最坏的位置；所以这里是
    `capacity` / `out_of_range` / `invalid_argument`，不是静默降级。
    """
    var room = KvRoom(16)
    for i in range(MAX_REQUESTS):
        _ = room.admit(i + 1, prompt_tokens(6001 + 100 * i, 16), 16, 16)
    var kind = String("")
    try:
        _ = room.admit(99, prompt_tokens(9001, 16), 16, 16)
    except err:
        kind = err.name()
    assert_equal(kind, "capacity", "第 9 条请求竟然被收下了")

    var long_room = KvRoom(16)
    kind = ""
    try:
        _ = long_room.admit(1, prompt_tokens(7001, MAX_SEQ_TOKENS), MAX_SEQ_TOKENS + 1, MAX_SEQ_TOKENS + 1)
    except err:
        kind = err.name()
    assert_equal(kind, "out_of_range", "超长序列竟然被收下了")

    var empty_room = KvRoom(16)
    kind = ""
    try:
        _ = empty_room.drop(77)
    except err:
        kind = err.name()
    assert_equal(kind, "invalid_argument", "丢弃一个不存在的请求竟然成功了")
    kind = ""
    try:
        _ = empty_room.extend(77, 1)
    except err:
        kind = err.name()
    assert_equal(kind, "invalid_argument", "给一个不存在的请求加 token 竟然成功了")


def test_the_zero_alloc_gate_can_fail() raises:
    """零分配这条**源码门**自己也得会红，否则它只是个摆设。

    真正的保证在类型层面（全是编译期定长的 `InlineArray`）；这条扫的是没人把它
    改回会增长的样子。它拦不住 libc 里的小块分配，也不等于进程 RSS 不动。
    """
    var bad_in_good = scan_growable(read_text(ROOM_SRC))
    assert_equal(bad_in_good, "", "房间源码里出现了会增长的容器：" + bad_in_good)
    var bad_in_bad = scan_growable(read_text(BAD_SRC))
    assert_true(
        bad_in_bad.byte_length() > 0,
        "负向对照源码里没有任何会增长的容器 —— 门不会红",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
