"""2.2 的门：分页寻址只改"行在哪里"，不改"算什么".

这个门要同时证明两件不同的事，所以用两把不同的尺子。

**第一件是寻址，用逐位相等。** 分页和连续布局的区别只在行的地址上，没有任何一项
运算被重排，所以两者必须**逐位相同**。这里若用容差，就是在替一个本不该存在的差
异留余地：差一个 ulp 就说明地址算错了，而地址算错的后果是读到了别人的 token ——
一个看起来完全合理的数。逐位相等不给这种错误留下藏身之处。

**第二件是公式，用容差。** 自己和自己逐位相等证明不了公式对不对。所以期望值由
`scripts/dump_paged_reference.py` 里另写的一份参照算出来，两者不共享任何代码，按
1e-5 相对误差比对。唯一的跨语言差异来源是 `exp` 的最后一位，这也正是容差该干的
事。

**表不是编的。** 夹具里的页表绝大多数是 `KvSpace`（2.1）对真实操作序列重放出来
的，和 `tests/fixtures/kv/` 是同一批操作 —— 这就是 2.2 要证明的"两层能拼起来"。
有三例是**块内偏移**用例，它们标记为 `syn_*`：当前树策略下每个请求都从槽位 0 开
始，所以这种表走 `KvSpace` 造不出来，但走 `PagedTable` 造得出来，内核就必须对它
负责。

**五条常驻对照。** 一个不会失败的门等于没有门：
- `expected_bad.tsv`（把期望值第一个元素改 0.05）→ 容差比对必须判红；
- `expected_hmap.tsv`（GQA 用 `h % n_kv_heads` 而不是 `h // group`）→ 必须判红，
  否则"多头映射"这一维根本没被比到；
- `bad_paged_alloc.mojo`（给内核加会增长的堆容器）→ 零分配扫描必须判违规；
- 把每段 run 的**尾部槽位**灌成垃圾值，输出必须逐位不变 —— 这条不依赖任何参照
  物，是"内核读没读过自己的长度"唯一能被直接观测的形式；
- 共享同一个块的两个请求，读到的字节必须相同。

Run:
    pixi run mojo run -O0 -I src tests/unit/test_paged_attention.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.dtype import DT_FP32
from alofa.core.error import AlofaError
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, Shape, TensorView, f32_data
from alofa.core.text import parse_float64, parse_int, read_text
from alofa.kernels.cpu.paged import PagedTable, paged_attention, paged_gather
from alofa.kernels.cpu.scalar import attention
from alofa.runtime.kv import MAX_BLOCKS_PER_SEQ
from alofa.runtime.kv.paging import fill_page_table
from alofa.runtime.kv.space import KvSpace
from alofa.runtime.kv.trace import apply_op, parse_config, parse_op, parts_of

comptime FIXTURE = "tests/fixtures/paged/"
comptime KV_FIXTURE = "tests/fixtures/kv/"
comptime BAD_ALLOC_SRC = "tests/fixtures/bad_paged_alloc.mojo"
comptime KERNEL_SRC = "src/alofa/kernels/cpu/paged.mojo"

comptime MAX_ENTRIES = 8
comptime CASES = 11
comptime KV_CASES = 8
comptime SYNTHETIC_CASES = 3
comptime MID_BLOCK_RUNS = 7
comptime BLOCKS_PER_TEST = 8

# One part in 1e-5. The only cross-language difference available here is the
# last bit of `exp`; anything larger than this is a formula disagreement, not
# rounding.
comptime TOLERANCE = 1e-5

comptime KvPageTable = PagedTable[MAX_BLOCKS_PER_SEQ]


struct Config:
    """`config.tsv`, plus the shapes every case has to agree with."""

    var n_blocks: Int
    var block_size: Int
    var kv_cols: Int
    var q_cols: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var n_cases: Int

    def __init__(out self):
        self.n_blocks = 0
        self.block_size = 0
        self.kv_cols = 0
        self.q_cols = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.head_dim = 0
        self.n_cases = 0

    def cache_cells(self) -> Int:
        return self.n_blocks * self.block_size * self.kv_cols


struct Case(Copyable, Movable):
    """One fixture case: a page table, and the query that reads through it."""

    var name: String
    var scenario: String
    var req: Int
    var ops: Int
    var q_len: Int
    var n_entries: Int
    var kv_len: Int
    var blocks: InlineArray[Int, MAX_ENTRIES]
    var starts: InlineArray[Int, MAX_ENTRIES]
    var lens: InlineArray[Int, MAX_ENTRIES]

    def __init__(out self):
        self.name = ""
        self.scenario = ""
        self.req = 0
        self.ops = 0
        self.q_len = 0
        self.n_entries = 0
        self.kv_len = 0
        self.blocks = InlineArray[Int, MAX_ENTRIES](fill=0)
        self.starts = InlineArray[Int, MAX_ENTRIES](fill=0)
        self.lens = InlineArray[Int, MAX_ENTRIES](fill=0)

    def from_kv_space(self) -> Bool:
        """False for the synthetic mid-block cases: no op sequence behind them."""
        return self.scenario != "-"

    def table(self, block_size: Int) -> PagedTable[MAX_ENTRIES]:
        var out = PagedTable[MAX_ENTRIES](block_size)
        for i in range(self.n_entries):
            _ = out.push(self.blocks[i], self.starts[i], self.lens[i])
        return out^

    def mid_block_runs(self) -> Int:
        var count = 0
        for i in range(self.n_entries):
            if self.starts[i] != 0:
                count += 1
        return count


def lines_of(path: String) raises AlofaError -> List[String]:
    var text = read_text(path)
    var out = List[String]()
    for span in text.split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        out.append(line)
    return out^


def load_config() raises -> Config:
    var cfg = Config()
    for line in lines_of(FIXTURE + "config.tsv"):
        var fields = parts_of(line, "\t")
        if len(fields) != 2:
            raise AlofaError(1, "config line must have a key and a value")
        var value = parse_int(fields[1])
        if fields[0] == "n_blocks":
            cfg.n_blocks = value
        elif fields[0] == "block_size":
            cfg.block_size = value
        elif fields[0] == "kv_cols":
            cfg.kv_cols = value
        elif fields[0] == "q_cols":
            cfg.q_cols = value
        elif fields[0] == "n_heads":
            cfg.n_heads = value
        elif fields[0] == "n_kv_heads":
            cfg.n_kv_heads = value
        elif fields[0] == "head_dim":
            cfg.head_dim = value
        elif fields[0] == "n_cases":
            cfg.n_cases = value
    return cfg^


def load_cases() raises -> List[Case]:
    """Read every case: name, provenance, op cut-off, query length, runs."""
    var out = List[Case]()
    for line in lines_of(FIXTURE + "cases.tsv"):
        if line.find("#") == 0:
            continue
        var f = parts_of(line, "\t")
        var c = Case()
        c.name = f[0]
        c.scenario = f[1]
        c.req = parse_int(f[2])
        c.ops = parse_int(f[3])
        c.q_len = parse_int(f[4])
        c.n_entries = parse_int(f[5])
        if c.n_entries > MAX_ENTRIES:
            raise AlofaError(1, "case has more runs than the test's table")
        var total = 0
        for i in range(c.n_entries):
            c.blocks[i] = parse_int(f[6 + 3 * i])
            c.starts[i] = parse_int(f[7 + 3 * i])
            c.lens[i] = parse_int(f[8 + 3 * i])
            total += c.lens[i]
        c.kv_len = total
        out.append(c^)
    return out^


def index_of(cases: List[Case], name: String) raises -> Int:
    """Index of the named case; a lookup, not a copy, so callers stay borrowed."""
    for i in range(len(cases)):
        if cases[i].name == name:
            return i
    raise AlofaError(1, "no such fixture case")


def tensor2(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    return TensorView(arena.alloc(rows * cols * 4), Shape(dims), DT_FP32)


def tensor3(mut arena: Arena, a: Int, b: Int, c: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(a)
    dims.append(b)
    dims.append(c)
    return TensorView(arena.alloc(a * b * c * 4), Shape(dims), DT_FP32)


def load_floats(path: String, dst: F32Ptr, count: Int) raises:
    """Read exactly `count` fixed-point values, one per line.

    Exactly, not at most: a fixture that lost a line would otherwise shift
    every later value by one and be blamed on the kernel.
    """
    var text = read_text(path)
    var i = 0
    for span in text.split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        if i >= count:
            raise AlofaError(1, "fixture has more values than the shape holds")
        dst[unsafe_offset=i] = Float32(parse_float64(line))
        i += 1
    if i != count:
        raise AlofaError(1, "fixture has fewer values than the shape holds")


def load_cache(mut arena: Arena, cfg: Config, path: String) raises -> TensorView:
    var cache = tensor3(arena, cfg.n_blocks, cfg.block_size, cfg.kv_cols)
    load_floats(path, f32_data(cache), cfg.cache_cells())
    return cache^


def load_q(mut arena: Arena, c: Case, cfg: Config) raises -> TensorView:
    var q = tensor2(arena, c.q_len, cfg.q_cols)
    load_floats(FIXTURE + c.name + ".q.tsv", f32_data(q), c.q_len * cfg.q_cols)
    return q^


def load_expected(mut arena: Arena, name: String, c: Case, cfg: Config) raises -> TensorView:
    var want = tensor2(arena, c.q_len, cfg.q_cols)
    load_floats(FIXTURE + name, f32_data(want), c.q_len * cfg.q_cols)
    return want^


def gather_into(
    dst_k: TensorView,
    dst_v: TensorView,
    c: Case,
    cfg: Config,
    k_cache: TensorView,
    v_cache: TensorView,
) raises:
    var table = c.table(cfg.block_size)
    paged_gather(dst_k, k_cache, table)
    paged_gather(dst_v, v_cache, table)


def run_paged(
    dst: TensorView,
    scores: TensorView,
    q: TensorView,
    c: Case,
    cfg: Config,
    k_cache: TensorView,
    v_cache: TensorView,
) raises:
    paged_attention(
        dst,
        q,
        k_cache,
        v_cache,
        c.table(cfg.block_size),
        scores,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
    )


def bit_diffs(a: TensorView, b: TensorView) raises -> Int:
    var pa = f32_data(a)
    var pb = f32_data(b)
    var bad = 0
    for i in range(a.numel()):
        if pa[unsafe_offset=i] != pb[unsafe_offset=i]:
            bad += 1
    return bad


def worst_relative(got: TensorView, want: TensorView) raises -> Float64:
    """Largest `|got - want| / max(1, |want|)` over the tensor."""
    var pg = f32_data(got)
    var pw = f32_data(want)
    var worst = Float64(0)
    for i in range(got.numel()):
        var g = Float64(pg[unsafe_offset=i])
        var w = Float64(pw[unsafe_offset=i])
        var scale = w
        if scale < 0:
            scale = -scale
        if scale < 1.0:
            scale = 1.0
        var diff = g - w
        if diff < 0:
            diff = -diff
        var rel = diff / scale
        if rel > worst:
            worst = rel
    return worst


def block_size_of(scenario: String) raises -> Int:
    """The block size the scenario's trace was built for, from its `CFG` line."""
    for line in lines_of(KV_FIXTURE + scenario + ".trace"):
        if line.find("CFG") == 0:
            return parse_config(line)
    raise AlofaError(1, "trace has no config line")


def replay_until(mut space: KvSpace, scenario: String, ops: Int) raises -> Int:
    """Apply the first `ops` operations of a 2.1 scenario trace."""
    var done = 0
    for line in lines_of(KV_FIXTURE + scenario + ".trace"):
        if line.find("#") == 0 or line.find("CFG") == 0:
            continue
        if done >= ops:
            break
        # Only the `OP=` half is an argument; `R=` is the reference's answer.
        _ = apply_op(space, parse_op(parts_of(line, " R=")[0]))
        done += 1
    return done


def test_gather_reproduces_the_reference_rows() raises:
    """`paged_gather` 与参照导出的连续行必须逐位一致.

    这是打破循环论证的那一半：如果分页内核和 gather 犯了同一个寻址错误，
    "内核 == gather + 连续内核"依然成立。所以 gather 必须由参照物单独钉住 ——
    它是 (block, start, length) 三段式在 Mojo 这一侧的第一次落地。
    """
    var cfg = load_config()
    var cases = load_cases()
    assert_equal(len(cases), CASES)
    var arena = Arena(1 << 24)
    var k_cache = load_cache(arena, cfg, FIXTURE + "k_cache.tsv")
    var v_cache = load_cache(arena, cfg, FIXTURE + "v_cache.tsv")

    var bad = 0
    var first = ""
    for c in cases:
        var kg = tensor2(arena, c.kv_len, cfg.kv_cols)
        var vg = tensor2(arena, c.kv_len, cfg.kv_cols)
        gather_into(kg, vg, c, cfg, k_cache, v_cache)
        var want = tensor2(arena, 2 * c.kv_len, cfg.kv_cols)
        load_floats(FIXTURE + c.name + ".kv.tsv", f32_data(want), 2 * c.kv_len * cfg.kv_cols)
        var pw = f32_data(want)
        var pk = f32_data(kg)
        var pv = f32_data(vg)
        var cells = c.kv_len * cfg.kv_cols
        for i in range(cells):
            if pk[unsafe_offset=i] != pw[unsafe_offset=i]:
                bad += 1
                if first.byte_length() == 0:
                    first = c.name + " k 行内第 " + String(i)
            if pv[unsafe_offset=i] != pw[unsafe_offset=cells + i]:
                bad += 1
                if first.byte_length() == 0:
                    first = c.name + " v 行内第 " + String(i)
    assert_true(
        bad == 0, "gather 与参照的连续行有 " + String(bad) + " 个元素不一致，首处：" + first
    )
    arena.keep_alive()


def test_paged_matches_the_contiguous_oracle_bit_for_bit() raises:
    """分页寻址只改行的位置，因此必须与连续 oracle 逐位相同."""
    var cfg = load_config()
    var cases = load_cases()
    var arena = Arena(1 << 24)
    var k_cache = load_cache(arena, cfg, FIXTURE + "k_cache.tsv")
    var v_cache = load_cache(arena, cfg, FIXTURE + "v_cache.tsv")

    var bad = 0
    var first = ""
    for c in cases:
        var kg = tensor2(arena, c.kv_len, cfg.kv_cols)
        var vg = tensor2(arena, c.kv_len, cfg.kv_cols)
        gather_into(kg, vg, c, cfg, k_cache, v_cache)
        var q = load_q(arena, c, cfg)
        var oracle = tensor2(arena, c.q_len, cfg.q_cols)
        var paged = tensor2(arena, c.q_len, cfg.q_cols)
        var scores = tensor2(arena, c.q_len, c.kv_len)
        attention(
            oracle,
            q,
            kg,
            vg,
            scores,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
        )
        run_paged(paged, scores, q, c, cfg, k_cache, v_cache)
        var diffs = bit_diffs(paged, oracle)
        if diffs > 0:
            bad += diffs
            if first.byte_length() == 0:
                first = c.name
    assert_true(
        bad == 0,
        "分页与连续 oracle 有 "
        + String(bad)
        + " 个元素逐位不同，首个用例："
        + first,
    )
    arena.keep_alive()


def test_paged_matches_the_reference_within_tolerance() raises:
    """公式本身由独立参照钉住，这里只允许 `exp` 的最后一位差异."""
    var cfg = load_config()
    var cases = load_cases()
    var arena = Arena(1 << 24)
    var k_cache = load_cache(arena, cfg, FIXTURE + "k_cache.tsv")
    var v_cache = load_cache(arena, cfg, FIXTURE + "v_cache.tsv")

    var worst = Float64(0)
    var worst_case = ""
    var checked = 0
    for c in cases:
        var q = load_q(arena, c, cfg)
        var got = tensor2(arena, c.q_len, cfg.q_cols)
        var scores = tensor2(arena, c.q_len, c.kv_len)
        run_paged(got, scores, q, c, cfg, k_cache, v_cache)
        var want = load_expected(arena, c.name + ".y.tsv", c, cfg)
        var rel = worst_relative(got, want)
        if rel > worst:
            worst = rel
            worst_case = c.name
        checked += 1
    assert_equal(checked, CASES)
    assert_true(
        worst <= TOLERANCE,
        "与参照的最大相对误差 "
        + String(worst)
        + " 超出容差，最差用例 "
        + worst_case,
    )
    arena.keep_alive()


def test_both_reference_controls_are_rejected() raises:
    """红测：改坏的期望值、用错 GQA 映射的期望值，都必须被拒绝."""
    var cfg = load_config()
    var cases = load_cases()
    var first = index_of(cases, "s01_c3_r1")
    var arena = Arena(1 << 24)
    var k_cache = load_cache(arena, cfg, FIXTURE + "k_cache.tsv")
    var v_cache = load_cache(arena, cfg, FIXTURE + "v_cache.tsv")
    var q = load_q(arena, cases[first], cfg)
    var got = tensor2(arena, cases[first].q_len, cfg.q_cols)
    var scores = tensor2(arena, cases[first].q_len, cases[first].kv_len)
    run_paged(got, scores, q, cases[first], cfg, k_cache, v_cache)

    var bad = load_expected(arena, "expected_bad.tsv", cases[first], cfg)
    assert_true(
        worst_relative(got, bad) > TOLERANCE,
        "把一个元素改了 0.05，容差比对竟然通过了",
    )

    var hmap = load_expected(arena, "expected_hmap.tsv", cases[first], cfg)
    assert_true(
        worst_relative(got, hmap) > TOLERANCE,
        "GQA 映射写错了，容差比对竟然通过了",
    )
    arena.keep_alive()


def test_a_run_is_never_read_past_its_length() raises:
    """把每段 run 的尾部槽位灌成垃圾值，输出必须逐位不变.

    不依赖参照物：这是"内核只读自己那段"唯一能被直接观测的形式。块是共享的，
    尾部那些槽位属于别的请求（或者干脆是空的），读到它们不会报错，只会算出一个
    看似合理的错答案。
    """
    var cfg = load_config()
    var cases = load_cases()
    var first = index_of(cases, "s01_c3_r1")
    var arena = Arena(1 << 24)
    var k_cache = load_cache(arena, cfg, FIXTURE + "k_cache.tsv")
    var v_cache = load_cache(arena, cfg, FIXTURE + "v_cache.tsv")

    var q = load_q(arena, cases[first], cfg)
    var scores = tensor2(arena, cases[first].q_len, cases[first].kv_len)
    var before = tensor2(arena, cases[first].q_len, cfg.q_cols)
    run_paged(before, scores, q, cases[first], cfg, k_cache, v_cache)

    # Which (block, slot) pairs this case legitimately owns.
    var covered = InlineArray[Int, BLOCKS_PER_TEST * 64](fill=0)
    for i in range(cases[first].n_entries):
        for s in range(cases[first].lens[i]):
            covered[cases[first].blocks[i] * cfg.block_size + cases[first].starts[i] + s] = 1
    var pk = f32_data(k_cache)
    var pv = f32_data(v_cache)
    var poisoned = 0
    for block in range(cfg.n_blocks):
        for slot in range(cfg.block_size):
            if covered[block * cfg.block_size + slot] != 0:
                continue
            for c in range(cfg.kv_cols):
                var at = (block * cfg.block_size + slot) * cfg.kv_cols + c
                pk[unsafe_offset=at] = Float32(1234.5)
                pv[unsafe_offset=at] = Float32(-999.25)
            poisoned += 1

    var after = tensor2(arena, cases[first].q_len, cfg.q_cols)
    run_paged(after, scores, q, cases[first], cfg, k_cache, v_cache)
    assert_true(poisoned > 0, "没有一个槽位可灌 —— 用例把整块都占满了")
    assert_equal(bit_diffs(before, after), 0)
    arena.keep_alive()


def test_the_address_space_and_the_kernel_agree_on_the_table() raises:
    """2.1 的页表喂给内核，必须与夹具里的 (block, start, length) 完全一致.

    这是 2.2 的"接"：夹具里的表本身就是 `KvSpace` 重放出来的，所以这条比对同时
    说明两件事 —— 绑定没丢字段，且夹具确实是地址空间的输出而不是手写的数字。
    """
    var cfg = load_config()
    var cases = load_cases()
    var checked = 0
    for c in cases:
        if not c.from_kv_space():
            continue
        var space = KvSpace(block_size_of(c.scenario))
        assert_true(
            replay_until(space, c.scenario, c.ops) == c.ops,
            c.name + " 的轨迹没重放完",
        )
        assert_equal(space.check_invariants(), 0)

        var table = KvPageTable(cfg.block_size)
        var ntok = fill_page_table(space, space.slot_of(c.req), table)
        assert_true(ntok == c.kv_len, c.name + " 的 token 数不一致")
        assert_true(table.n == c.n_entries, c.name + " 的 run 数不一致")
        for i in range(table.n):
            var at = c.name + " 第 " + String(i) + " 段"
            assert_true(table.block_at(i) == c.blocks[i], at + " 的块号不一致")
            assert_true(table.start_at(i) == c.starts[i], at + " 的起始槽位不一致")
            assert_true(table.length_at(i) == c.lens[i], at + " 的长度不一致")
        checked += 1
    assert_equal(checked, KV_CASES)


def test_a_shared_block_reads_the_same_bytes_for_both_requests() raises:
    """共享一个块的两个请求，读到的必须是同一段字节.

    前缀共享在数值上的意义就是这个：两条请求的前若干行不是"相等"，而是**同一份
    数据**。它证明的是寻址，不是"省了内存" —— 哪天内核改成按块拷一份私有副本，
    这条仍然会过。
    """
    var cfg = load_config()
    var cases = load_cases()
    var a = index_of(cases, "s01_c3_r1")
    var b = index_of(cases, "s01_c3_r2")
    assert_true(cases[a].kv_len > cases[b].kv_len, "两个用例的前缀关系反了")

    var arena = Arena(1 << 24)
    var k_cache = load_cache(arena, cfg, FIXTURE + "k_cache.tsv")
    var v_cache = load_cache(arena, cfg, FIXTURE + "v_cache.tsv")
    var ka = tensor2(arena, cases[a].kv_len, cfg.kv_cols)
    var va = tensor2(arena, cases[a].kv_len, cfg.kv_cols)
    var kb = tensor2(arena, cases[b].kv_len, cfg.kv_cols)
    var vb = tensor2(arena, cases[b].kv_len, cfg.kv_cols)
    gather_into(ka, va, cases[a], cfg, k_cache, v_cache)
    gather_into(kb, vb, cases[b], cfg, k_cache, v_cache)

    var shared = 0
    for i in range(cases[a].n_entries):
        var same_block = cases[a].blocks[i] == cases[b].blocks[0]
        var same_start = cases[a].starts[i] == cases[b].starts[0]
        if same_block and same_start:
            shared = cases[a].lens[i]
            if cases[b].lens[0] < shared:
                shared = cases[b].lens[0]
    assert_true(
        shared == cases[b].lens[0],
        "两个用例没有共享同一个块的同一段：前缀共享这一维根本没被比到",
    )
    var pka = f32_data(ka)
    var pkb = f32_data(kb)
    var bad = 0
    for i in range(shared * cfg.kv_cols):
        if pka[unsafe_offset=i] != pkb[unsafe_offset=i]:
            bad += 1
    assert_equal(bad, 0)
    arena.keep_alive()


def test_mid_block_runs_are_exercised() raises:
    """块内偏移这一维必须真的被比到，而且来源要说清楚.

    前一半：真实的 `KvSpace` 轨迹里，每个请求的每一段都从槽位 0 开始 —— 因为树
    只从根开始共享。这是当前策略的性质，写成断言是为了它哪天变了会有人知道，而
    不是因为它天然成立。后一半：块内偏移用例确实存在，否则 `start` 就是个没人
    验过的字段。
    """
    var cases = load_cases()
    var from_space = 0
    var from_space_nonzero = 0
    var synthetic = 0
    var mid_block = 0
    for c in cases:
        if c.from_kv_space():
            from_space += 1
            from_space_nonzero += c.mid_block_runs()
        else:
            synthetic += 1
        mid_block += c.mid_block_runs()
    assert_equal(from_space, KV_CASES)
    assert_equal(synthetic, SYNTHETIC_CASES)
    assert_true(
        from_space_nonzero == 0,
        "树策略变了：真实轨迹里出现了块内起始的 run，夹具该重导出了",
    )
    assert_equal(mid_block, MID_BLOCK_RUNS)


def test_an_invalid_table_is_a_named_error() raises:
    """读越界的页表必须报错，不能截断后算出一个看似合理的数."""
    var cfg = load_config()
    var arena = Arena(1 << 20)
    var k = tensor3(arena, cfg.n_blocks, cfg.block_size, cfg.kv_cols)
    var v = tensor3(arena, cfg.n_blocks, cfg.block_size, cfg.kv_cols)
    for i in range(cfg.cache_cells()):
        f32_data(k)[unsafe_offset=i] = Float32(0.5)
        f32_data(v)[unsafe_offset=i] = Float32(0.5)
    var q = tensor2(arena, 1, cfg.q_cols)
    var dst = tensor2(arena, 1, cfg.q_cols)
    var scores = tensor2(arena, 1, cfg.block_size)

    # A run that leaves its block: start + length > block_size.
    var over = PagedTable[MAX_ENTRIES](cfg.block_size)
    _ = over.push(0, cfg.block_size - 2, 4)
    var threw = False
    try:
        paged_attention(dst, q, k, v, over, scores, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
    except:
        threw = True
    assert_true(threw, "越出块的 run 竟然被接受了")

    # A block the cache does not have: every address would be somebody else's.
    var absent = PagedTable[MAX_ENTRIES](cfg.block_size)
    _ = absent.push(cfg.n_blocks, 0, 1)
    threw = False
    try:
        paged_attention(
            dst, q, k, v, absent, scores, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        )
    except:
        threw = True
    assert_true(threw, "指向不存在的块竟然被接受了")

    # An empty run: zero tokens, so the context silently gets shorter.
    var empty = PagedTable[MAX_ENTRIES](cfg.block_size)
    _ = empty.push(0, 0, 0)
    threw = False
    try:
        paged_attention(dst, q, k, v, empty, scores, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
    except:
        threw = True
    assert_true(threw, "空 run 竟然被接受了")

    # A block size that disagrees with the cache: every address would be wrong.
    var mismatched = PagedTable[MAX_ENTRIES](cfg.block_size * 2)
    _ = mismatched.push(0, 0, 1)
    threw = False
    try:
        paged_attention(
            dst, q, k, v, mismatched, scores, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        )
    except:
        threw = True
    assert_true(threw, "块大小与缓存不符竟然被接受了")
    arena.keep_alive()


def alloc_violations(path: String) raises -> Int:
    """扫会增长的堆容器：`PagedTable` 的定长是编译期的事，这条保证没人改回去."""
    var text = read_text(path)
    var bad = 0
    if text.find("List[") >= 0:
        bad += 1
    if text.find("Dict[") >= 0:
        bad += 1
    if text.find("Set[") >= 0:
        bad += 1
    if text.find("Arena(") >= 0:
        bad += 1
    if text.find("String(") >= 0:
        bad += 1
    return bad


def test_the_paged_kernel_cannot_allocate_by_construction() raises:
    """零分配：内核源码里不得出现任何会增长的容器，连错误信息也不许.

    `paged.mojo` 的错误全部是"具名类型 + 固定句子"，没有一处拼接字符串 ——
    这不是洁癖，是因为断言是文本的：只要留一处 `String(...)`，整条门就只能被
    削成一个更弱的版本。
    """
    assert_equal(alloc_violations(KERNEL_SRC), 0)


def test_the_zero_alloc_gate_can_fail() raises:
    """红测：给内核加会增长的堆容器，扫描必须判违规."""
    assert_true(alloc_violations(BAD_ALLOC_SRC) > 0, "零分配门抓不到明显的违规")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
