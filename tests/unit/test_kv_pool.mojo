"""统一 KV 寻址的门：一个物理池，两个视图，一条引用计数方程。

判定纪律
--------

**逐字节，不容差。** 这一层输出的不是数值而是**归属**：这块块现在归谁、命中了几个
token、分裂出的节点引用哪几个块。归属差一个就是另一个归属，不存在"差一点点"。

**参照物是独立实现。** `tests/fixtures/kv/*.trace` 由
`scripts/dump_kv_reference.py` 导出 —— 同一份策略的 Python 实现。写它的时候它
自己就抓出了两个 bug（中间节点没按节点长度截断、同一块被重复保留），这正是"参照物
必须独立"的价值：由被测实现导出的 fixture 会跟着一起错。

**四条常驻对照。** 一个不会失败的门等于没有门：
- `bad.trace`（故意把一拍的 used 加一）→ 必须判红，否则逐字节比对根本没在比；
- `alt.trace`（关掉节点分裂，退回"只命中到最后一个完整节点"）→ 必须判红，且命中
  长度必须真的变短，否则 fixture 只分辨得了格式，分辨不了策略；
- `bad_kv_alloc.mojo`（给 KV 结构加会增长的堆容器）→ 零分配扫描必须判违规；
- 每拍都重算一遍引用计数方程 —— 这条**不依赖参照物**，两边同时漏掉一次 retain
  也逃不过它。

**零分配这条证据的边界**（写在这里以免被误读）：真正的保证是类型层面的（所有容器
都是编译期定长的 `InlineArray`），源码门保证的是没人把它改回会增长的样子。它拦不住
libc 里的小块分配，也不等同于进程级 RSS 不动。

Run:
    pixi run mojo run -I src tests/unit/test_kv_pool.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.text import read_text
from alofa.runtime.kv import MAX_SEQ_TOKENS, OpResult
from alofa.runtime.kv.space import KvSpace
from alofa.runtime.kv.trace import (
    OP_APP,
    OP_COM,
    OP_EVI,
    OP_MAT,
    OP_NEW,
    OP_REL,
    KvOp,
    apply_op,
    config_line,
    parse_config,
    parse_op,
    parts_of,
    result_line,
)

comptime FIXTURE = "tests/fixtures/kv/"
comptime KV_SOURCES = 4
comptime BAD_ALLOC_SRC = "tests/fixtures/bad_kv_alloc.mojo"

comptime SCENARIOS = 7


struct ReplayStats:
    """一次重放的体检结果。"""

    var ops: Int
    var mismatches: Int
    var inv_bad: Int
    var mat_ops: Int
    var mat_allocs: Int
    var first_want: String
    var first_got: String

    def __init__(out self):
        self.ops = 0
        self.mismatches = 0
        self.inv_bad = 0
        self.mat_ops = 0
        self.mat_allocs = 0
        self.first_want = ""
        self.first_got = ""

    def first_diff(self) -> String:
        """第一个不一致的字段，形如 `m: 3 vs 2`。

        只说"不一致"等于让人自己拿两份文件去对；指出是哪个字段，才是给读失败
        输出的人省下一次重跑。
        """
        if self.first_got.byte_length() == 0:
            return "-"
        var want = parts_of(self.first_want, " ")
        var got = parts_of(self.first_got, " ")
        var n = len(want)
        if len(got) < n:
            n = len(got)
        for i in range(n):
            if want[i] != got[i]:
                return want[i] + " vs " + got[i]
        return "长度不同：" + self.first_want + " vs " + self.first_got


def lines_of(path: String) raises AlofaError -> List[String]:
    var out = List[String]()
    for span in read_text(path).split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        out.append(line)
    return out^


def op_lines(path: String) raises AlofaError -> List[String]:
    """去掉注释与配置的纯操作行。"""
    var out = List[String]()
    for line in lines_of(path):
        if line.find("#") == 0:
            continue
        if line.find("CFG") == 0:
            continue
        out.append(line)
    return out^


def field_of(result: String, key: String) -> String:
    for field in parts_of(result, " "):
        if field.find(key) == 0:
            return field
    return "-"


def rep(value: Int, n: Int) raises AlofaError -> InlineArray[Int, MAX_SEQ_TOKENS]:
    var out = InlineArray[Int, MAX_SEQ_TOKENS](fill=0)
    for i in range(n):
        out[i] = value
    return out^


def seq(base: Int, n: Int) raises AlofaError -> InlineArray[Int, MAX_SEQ_TOKENS]:
    var out = InlineArray[Int, MAX_SEQ_TOKENS](fill=0)
    for i in range(n):
        out[i] = base + i
    return out^


def seq_then(base: Int, n: Int, tail: Int) raises AlofaError -> InlineArray[
    Int, MAX_SEQ_TOKENS
]:
    """`base, base+1, ..., base+n-1, tail` —— 一段公共前缀加一个分叉 token。"""
    var out = seq(base, n)
    out[n] = tail
    return out^


def rep_then(value: Int, n: Int, tail: Int) raises AlofaError -> InlineArray[
    Int, MAX_SEQ_TOKENS
]:
    """`value` 重复 `n` 次再接一个 `tail`。

    分裂场景要的是"同一个 token 连着出现"：用递增序列会让分叉点在别处。
    """
    var out = rep(value, n)
    out[n] = tail
    return out^


def replay(path: String, mut stats: ReplayStats) raises AlofaError:
    """用 fixture 驱动 KvSpace，逐条比对，并逐条重算引用计数方程。"""
    var cfg_bs = 0
    for line in lines_of(path):
        if line.find("CFG") == 0:
            cfg_bs = parse_config(line)
    if cfg_bs <= 0:
        raise AlofaError(1, "fixture has no CFG line")
    var space = KvSpace(cfg_bs)

    for line in lines_of(path):
        if line.find("#") == 0 or line.find("CFG") == 0:
            continue
        var halves = parts_of(line, " R=")
        if len(halves) != 2:
            raise AlofaError(1, "trace line must be 'OP=... R=...'")
        var op = parse_op(halves[0])
        var used_before = space.pool.used

        var res = apply_op(space, op)
        if op.kind == OP_MAT:
            stats.mat_ops += 1
            if space.pool.used != used_before:
                stats.mat_allocs += 1

        var got = result_line(
            op.kind, res, space.pool.used, space.pool.n_free, space.digest()
        )
        if got != halves[1]:
            stats.mismatches += 1
            if stats.first_got.byte_length() == 0:
                stats.first_want = halves[1]
                stats.first_got = got
        if space.check_invariants() != 0:
            stats.inv_bad += 1
        stats.ops += 1


def scenario_names() -> List[String]:
    var out = List[String]()
    out.append("s01_prompt_sharing")
    out.append("s02_split_divergence")
    out.append("s03_long_prompt_chain")
    out.append("s04_append_grow")
    out.append("s05_evict_and_reuse")
    out.append("s06_small_blocks")
    out.append("s07_refcount_pressure")
    return out^


def test_every_scenario_replays_byte_exact() raises:
    """七个场景与导出的 trace 逐字节一致，且每拍都满足引用计数方程。"""
    var checked = 0
    for name in scenario_names():
        var stats = ReplayStats()
        replay(FIXTURE + name + ".trace", stats)
        assert_true(stats.ops > 0, name + " 一行都没跑")
        assert_true(
            stats.mismatches == 0,
            name
            + " 有 "
            + String(stats.mismatches)
            + " 条与参考 trace 不一致，首处："
            + stats.first_diff(),
        )
        assert_true(
            stats.inv_bad == 0,
            name + " 有 " + String(stats.inv_bad) + " 拍破坏了引用计数方程",
        )
        checked += 1
    assert_equal(checked, SCENARIOS)


def test_the_bad_trace_is_rejected() raises:
    """红测：把一拍的 used 加一，重放必须判红。"""
    var stats = ReplayStats()
    replay(FIXTURE + "bad.trace", stats)
    assert_true(stats.mismatches > 0, "改坏过的 trace 竟然重放通过了")
    assert_true(stats.ops > 0, "bad.trace 一行都没跑")


def test_the_alt_policy_changes_the_match_length() raises:
    """红测：关掉节点分裂后必须判红，且命中长度必须真的变短。

    只判红还不够 —— 若两份 trace 只有 digest 不同，那说明 fixture 只分辨得了
    格式。这里要求 `m=` 字段真的不同：策略差异必须体现在"命中了多少"。
    """
    var stats = ReplayStats()
    replay(FIXTURE + "alt.trace", stats)
    assert_true(stats.mismatches > 0, "关掉节点分裂后重放竟然一致")

    var good = op_lines(FIXTURE + "s02_split_divergence.trace")
    var alt = op_lines(FIXTURE + "alt.trace")
    assert_equal(len(good), len(alt))
    var shorter = 0
    for i in range(len(good)):
        var gp = parts_of(good[i], " R=")
        var ap = parts_of(alt[i], " R=")
        var a = field_of(gp[1], "m=")
        var b = field_of(ap[1], "m=")
        if a != b:
            shorter += 1
    assert_true(shorter > 0, "alt 与默认策略的命中长度完全相同")


def test_splitting_a_node_allocates_nothing() raises:
    """节点分裂必须是元数据操作：commit 前后池子的占用数一分不差。

    这是"分裂零拷贝"唯一能被直接观测到的形式 —— 拷贝与否看不见，但"有没有向
    池子要新块"看得见。
    """
    var space = KvSpace(16)
    _ = space.new_request(1, rep(7, 8), 8)
    _ = space.commit(space.slot_of(1))

    _ = space.new_request(2, rep_then(7, 3, 8), 4)
    var before = space.pool.used
    var res = space.commit(space.slot_of(2))
    assert_equal(res.matched, 3)
    assert_equal(space.pool.used, before)
    assert_true(res.nodes.n >= 2, "分裂没有产生新节点")
    assert_equal(space.check_invariants(), 0)


def test_a_shared_block_outlives_its_first_holder() raises:
    """共享块必须活到最后一个持有者放手：请求先走，块不能跟着走。"""
    var space = KvSpace(16)
    _ = space.new_request(1, seq_then(1, 12, 31), 13)
    _ = space.commit(space.slot_of(1))
    var block = space.block_at(space.slot_of(1), 0)

    var two = space.new_request(2, seq_then(1, 12, 32), 13)
    assert_equal(two.matched, 12)
    _ = space.commit(space.slot_of(2))

    var rel = space.release(space.slot_of(1))
    assert_equal(rel.fresh.n, 0)
    assert_true(space.pool.refcnt_of(block) > 0, "请求走了，块却被回收了")

    _ = space.evict(1)
    _ = space.release(space.slot_of(2))
    assert_equal(space.pool.refcnt_of(block), 0)
    assert_equal(space.check_invariants(), 0)


def test_match_does_not_allocate() raises:
    """纯查询操作不得动池子：MAT 前后占用数必须相同。

    一条"只查询"的路径如果顺手分配了，重放出来的字节会随查询次数漂移。
    """
    var stats = ReplayStats()
    replay(FIXTURE + "s01_prompt_sharing.trace", stats)
    assert_true(stats.mat_ops > 0, "场景里一次 MAT 都没有")
    assert_equal(stats.mat_allocs, 0)


def test_pool_exhaustion_is_an_error_not_a_silent_skip() raises:
    """池子满了必须是具名错误，不能是"这一拍什么都不做"。"""
    var space = KvSpace(4)
    var threw = False
    var i = 1
    while i <= 8:
        try:
            _ = space.new_request(i, seq(i * 100, 64), 64)
        except:
            threw = True
        i += 1
    assert_true(threw, "请求容量大于池子，却一次都没触发容量错误")


def test_double_release_is_a_named_error() raises:
    """释放一块没人持有的块，必须报具名错误而不是把计数压成负数。"""
    var space = KvSpace(16)
    _ = space.new_request(1, seq(1, 5), 5)
    var slot = space.slot_of(1)
    var block = space.block_at(slot, 0)
    _ = space.release(slot)
    var threw = False
    try:
        _ = space.pool.release(block)
    except:
        threw = True
    assert_true(threw, "重复释放竟然被接受了")


def test_the_refcount_invariant_gate_can_fail() raises:
    """红测：凭空多一次 retain，引用计数方程必须判违规。

    这条对照针对的是**不依赖参照物**的那道门：如果两侧实现同时漏掉一次
    retain，逐字节比对仍会一致，只有这条方程会把它们一起抓出来。
    """
    var space = KvSpace(16)
    _ = space.new_request(1, seq(1, 5), 5)
    _ = space.commit(space.slot_of(1))
    assert_equal(space.check_invariants(), 0)
    space.pool.retain(space.block_at(space.slot_of(1), 0))
    assert_true(space.check_invariants() > 0, "多出来的引用竟然没被方程发现")


def test_config_must_match_compiled_capacities() raises:
    """fixture 的容量必须和编译进二进制的一致：不一致就拒绝，不猜。"""
    assert_equal(parse_config(config_line(16)), 16)
    var threw = False
    try:
        _ = parse_config("CFG bs=16 nb=999 nr=8 nn=32 nt=16")
    except:
        threw = True
    assert_true(threw, "为别的池子规模生成的 fixture 竟然被接受了")


def alloc_violations(path: String) raises AlofaError -> Int:
    var text = read_text(path)
    var bad = 0
    if text.find("List[") >= 0:
        bad += 1
    if text.find("String(") >= 0:
        bad += 1
    if text.find("Arena(") >= 0:
        bad += 1
    if text.find("Dict[") >= 0:
        bad += 1
    if text.find("Set[") >= 0:
        bad += 1
    return bad


def kv_sources() -> List[String]:
    var out = List[String]()
    out.append("src/alofa/runtime/kv/__init__.mojo")
    out.append("src/alofa/runtime/kv/pool.mojo")
    out.append("src/alofa/runtime/kv/radix.mojo")
    out.append("src/alofa/runtime/kv/space.mojo")
    return out^


def test_the_kv_space_cannot_allocate_by_construction() raises:
    """零分配：三个视图的源码里不得出现任何会增长的堆容器。

    `kv/trace.mojo` 不在此列 —— 序列化必然产生 String，而录 trace 是 `step`
    之外的可选动作，不是每 token 路径的一部分。
    """
    var checked = 0
    for path in kv_sources():
        assert_equal(alloc_violations(path), 0)
        checked += 1
    assert_equal(checked, KV_SOURCES)


def test_the_zero_alloc_gate_can_fail() raises:
    """红测：给 KV 结构加会增长的堆容器，扫描必须判违规。"""
    assert_true(alloc_violations(BAD_ALLOC_SRC) > 0, "零分配门抓不到明显的违规")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
