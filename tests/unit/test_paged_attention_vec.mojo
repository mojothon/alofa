"""分页注意力向量版的门：**与 `avx2.attention` 逐位一致**，全部夹具用例。

这一版核的全部主张就是一条恒等式：同样的值、同样的累加顺序，只有取数路径不同
（连续 `j * kv_cols` 换成页表的 `(block, slot)` 地址）—— 所以判据是**逐位相等**，
对全部 11 个夹具用例（含块内偏移、共享前缀、q_len>1 的因果截断）。参照物是
`avx2.attention` 读 gather 出来的连续矩阵：它自己被 oracle 逐位门与独立 python
参照钉着，这里再把分页寻址钉到它身上，"批 == 单"的核级一致性就闭环了。

标量 oracle（`paged.attention`）的地位不变：它继续当独立参照（它的逐位门照旧），
向量化带来的点积重排只允许出现在这一版，且被上面那条恒等式罩住。

**四条常驻对照：**
- 把 run 尾部之外的槽位灌成垃圾值，输出必须逐位不变（"只读自己那段"的直接观测）；
- 越出块的页表必须报具名错误，不能截断后算出看似合理的数；
- 改坏一个输出元素，逐位比对必须红 —— 不会失败的门等于没门；
- 公式本身仍与独立 python 参照按 1e-5 比对（防止"两版一起错"）。

Run:
    pixi run mojo run -O0 -I src tests/unit/test_paged_attention_vec.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.dtype import DT_FP32
from alofa.core.error import AlofaError
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, Shape, TensorView, f32_data
from alofa.core.text import parse_float64, parse_int, read_text
from alofa.kernels.cpu.avx2 import attention, paged_attention_vec
from alofa.kernels.cpu.paged import PagedTable, paged_gather
from alofa.runtime.kv.trace import parts_of

comptime FIXTURE = "tests/fixtures/paged/"
comptime CASES = 11
comptime MAX_ENTRIES = 8
comptime POISON_BLOCKS = 8
comptime TOLERANCE = 1e-5


struct Cfg:
    var n_blocks: Int
    var block_size: Int
    var kv_cols: Int
    var q_cols: Int
    var n_heads: Int
    var head_dim: Int

    def __init__(out self):
        self.n_blocks = 0
        self.block_size = 0
        self.kv_cols = 0
        self.q_cols = 0
        self.n_heads = 0
        self.head_dim = 0

    def n_kv(self) -> Int:
        return self.kv_cols // self.head_dim

    def cache_cells(self) -> Int:
        return self.n_blocks * self.block_size * self.kv_cols


struct Case(Copyable, Movable):
    """一个夹具用例：页表三段式 + 查询长度。名字留作报错时的第一现场。"""

    var name: String
    var q_len: Int
    var n_entries: Int
    var kv_len: Int
    var blocks: InlineArray[Int, MAX_ENTRIES]
    var starts: InlineArray[Int, MAX_ENTRIES]
    var lens: InlineArray[Int, MAX_ENTRIES]

    def __init__(out self):
        self.name = ""
        self.q_len = 0
        self.n_entries = 0
        self.kv_len = 0
        self.blocks = InlineArray[Int, MAX_ENTRIES](fill=0)
        self.starts = InlineArray[Int, MAX_ENTRIES](fill=0)
        self.lens = InlineArray[Int, MAX_ENTRIES](fill=0)

    def table(self, block_size: Int) -> PagedTable[MAX_ENTRIES]:
        var out = PagedTable[MAX_ENTRIES](block_size)
        for i in range(self.n_entries):
            _ = out.push(self.blocks[i], self.starts[i], self.lens[i])
        return out^


def lines_of(path: String) raises -> List[String]:
    var text = read_text(path)
    var out = List[String]()
    for span in text.split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        out.append(line)
    return out^


def load_config() raises -> Cfg:
    var cfg = Cfg()
    for line in lines_of(FIXTURE + "config.tsv"):
        var f = parts_of(line, "\t")
        if len(f) != 2:
            continue
        if f[0] == "n_blocks":
            cfg.n_blocks = parse_int(f[1])
        elif f[0] == "block_size":
            cfg.block_size = parse_int(f[1])
        elif f[0] == "kv_cols":
            cfg.kv_cols = parse_int(f[1])
        elif f[0] == "q_cols":
            cfg.q_cols = parse_int(f[1])
        elif f[0] == "n_heads":
            cfg.n_heads = parse_int(f[1])
        elif f[0] == "head_dim":
            cfg.head_dim = parse_int(f[1])
    return cfg^


def load_cases() raises -> List[Case]:
    var out = List[Case]()
    for line in lines_of(FIXTURE + "cases.tsv"):
        if line.find("#") == 0:
            continue
        var f = parts_of(line, "\t")
        var c = Case()
        c.name = f[0]
        c.q_len = parse_int(f[4])
        c.n_entries = parse_int(f[5])
        var total = 0
        for i in range(c.n_entries):
            c.blocks[i] = parse_int(f[6 + 3 * i])
            c.starts[i] = parse_int(f[7 + 3 * i])
            c.lens[i] = parse_int(f[8 + 3 * i])
            total += c.lens[i]
        c.kv_len = total
        out.append(c^)
    return out^


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
    """恰好多不少：丢一行会让后面全部错位，却被账算到内核头上。"""
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


struct Caches(Movable):
    """一对缓存。这版编译器既不收 Tuple 返回类型注解，也不收多个 out 参数。"""

    var k: TensorView
    var v: TensorView

    def __init__(out self, mut arena: Arena, cfg: Cfg) raises:
        self.k = tensor3(arena, cfg.n_blocks, cfg.block_size, cfg.kv_cols)
        self.v = tensor3(arena, cfg.n_blocks, cfg.block_size, cfg.kv_cols)
        load_floats(FIXTURE + "k_cache.tsv", f32_data(self.k), cfg.cache_cells())
        load_floats(FIXTURE + "v_cache.tsv", f32_data(self.v), cfg.cache_cells())


def bit_diffs(a: TensorView, b: TensorView) raises -> Int:
    var pa = f32_data(a)
    var pb = f32_data(b)
    var bad = 0
    for i in range(a.numel()):
        if pa[unsafe_offset=i] != pb[unsafe_offset=i]:
            bad += 1
    return bad


def worst_relative(got: TensorView, want: TensorView) raises -> Float64:
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


def test_vec_matches_the_vector_contiguous_kernel_bit_for_bit() raises:
    """全部夹具用例：分页向量版与连续向量版必须逐位相同."""
    var cfg = load_config()
    var cases = load_cases()
    assert_equal(len(cases), CASES)
    var arena = Arena(1 << 24)
    var caches = Caches(arena, cfg)
    var n_kv = cfg.n_kv()

    var bad = 0
    var first = ""
    for c in cases:
        var table = c.table(cfg.block_size)
        # gather 出连续矩阵，给参照核用。
        var kg = tensor2(arena, c.kv_len, cfg.kv_cols)
        var vg = tensor2(arena, c.kv_len, cfg.kv_cols)
        paged_gather(kg, caches.k, table)
        paged_gather(vg, caches.v, table)
        var q = tensor2(arena, c.q_len, cfg.q_cols)
        load_floats(FIXTURE + c.name + ".q.tsv", f32_data(q), c.q_len * cfg.q_cols)

        var ref_out = tensor2(arena, c.q_len, cfg.q_cols)
        var ref_scores = tensor2(arena, c.q_len, c.kv_len)
        attention(ref_out, q, kg, vg, ref_scores, cfg.n_heads, n_kv, cfg.head_dim)

        var got = tensor2(arena, c.q_len, cfg.q_cols)
        var scores = tensor2(arena, c.q_len, c.kv_len)
        paged_attention_vec(
            got, q, caches.k, caches.v, table, scores, cfg.n_heads, n_kv, cfg.head_dim
        )
        var diffs = bit_diffs(got, ref_out)
        if diffs > 0:
            bad += diffs
            if first.byte_length() == 0:
                first = c.name
    assert_true(
        bad == 0,
        "分页向量版与连续向量版有 "
        + String(bad)
        + " 个元素逐位不同，首个用例："
        + first,
    )
    arena.keep_alive()


def test_the_formula_stays_within_the_reference() raises:
    """公式仍由独立 python 参照钉住（防"两版一起错"）."""
    var cfg = load_config()
    var cases = load_cases()
    var arena = Arena(1 << 24)
    var caches = Caches(arena, cfg)
    var n_kv = cfg.n_kv()
    var worst = Float64(0)
    var worst_case = ""
    for c in cases:
        var table = c.table(cfg.block_size)
        var q = tensor2(arena, c.q_len, cfg.q_cols)
        load_floats(FIXTURE + c.name + ".q.tsv", f32_data(q), c.q_len * cfg.q_cols)
        var got = tensor2(arena, c.q_len, cfg.q_cols)
        var scores = tensor2(arena, c.q_len, c.kv_len)
        paged_attention_vec(
            got, q, caches.k, caches.v, table, scores, cfg.n_heads, n_kv, cfg.head_dim
        )
        var want = tensor2(arena, c.q_len, cfg.q_cols)
        load_floats(FIXTURE + c.name + ".y.tsv", f32_data(want), c.q_len * cfg.q_cols)
        var rel = worst_relative(got, want)
        if rel > worst:
            worst = rel
            worst_case = c.name
    assert_true(
        worst <= TOLERANCE,
        "与参照的最大相对误差 " + String(worst) + " 超出容差，最差用例 " + worst_case,
    )
    arena.keep_alive()


def test_a_run_is_never_read_past_its_length() raises:
    """把不属于自己的槽位灌成垃圾值，输出必须逐位不变."""
    var cfg = load_config()
    var cases = load_cases()
    var arena = Arena(1 << 24)
    var caches = Caches(arena, cfg)
    var n_kv = cfg.n_kv()
    var table = cases[0].table(cfg.block_size)
    var q = tensor2(arena, cases[0].q_len, cfg.q_cols)
    load_floats(FIXTURE + cases[0].name + ".q.tsv", f32_data(q), cases[0].q_len * cfg.q_cols)

    var before = tensor2(arena, cases[0].q_len, cfg.q_cols)
    var scores = tensor2(arena, cases[0].q_len, cases[0].kv_len)
    paged_attention_vec(
        before, q, caches.k, caches.v, table, scores, cfg.n_heads, n_kv, cfg.head_dim
    )

    var covered = InlineArray[Int, POISON_BLOCKS * 64](fill=0)
    for i in range(cases[0].n_entries):
        for s in range(cases[0].lens[i]):
            covered[cases[0].blocks[i] * cfg.block_size + cases[0].starts[i] + s] = 1
    var pk = f32_data(caches.k)
    var pv = f32_data(caches.v)
    var poisoned = 0
    for block in range(cfg.n_blocks):
        for slot in range(cfg.block_size):
            if covered[block * cfg.block_size + slot] != 0:
                continue
            for col in range(cfg.kv_cols):
                var at = (block * cfg.block_size + slot) * cfg.kv_cols + col
                pk[unsafe_offset=at] = Float32(1234.5)
                pv[unsafe_offset=at] = Float32(-999.25)
            poisoned += 1

    var after = tensor2(arena, cases[0].q_len, cfg.q_cols)
    paged_attention_vec(
        after, q, caches.k, caches.v, table, scores, cfg.n_heads, n_kv, cfg.head_dim
    )
    assert_true(poisoned > 0, "没有一个槽位可灌 —— 用例把整块都占满了")
    assert_equal(bit_diffs(before, after), 0)
    arena.keep_alive()


def test_an_invalid_table_is_a_named_error() raises:
    """越出块的 run 必须报错，不能截断后算出一个看似合理的数."""
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

    var over = PagedTable[MAX_ENTRIES](cfg.block_size)
    _ = over.push(0, cfg.block_size - 2, 4)
    var threw = False
    try:
        paged_attention_vec(dst, q, k, v, over, scores, cfg.n_heads, cfg.n_kv(), cfg.head_dim)
    except:
        threw = True
    assert_true(threw, "越出块的 run 竟然被接受了")
    arena.keep_alive()


def test_a_corrupted_output_is_caught() raises:
    """负向对照：改坏一个输出元素，逐位比对必须红."""
    var cfg = load_config()
    var cases = load_cases()
    var arena = Arena(1 << 24)
    var caches = Caches(arena, cfg)
    var n_kv = cfg.n_kv()
    var table = cases[0].table(cfg.block_size)
    var q = tensor2(arena, cases[0].q_len, cfg.q_cols)
    load_floats(FIXTURE + cases[0].name + ".q.tsv", f32_data(q), cases[0].q_len * cfg.q_cols)

    var kg = tensor2(arena, cases[0].kv_len, cfg.kv_cols)
    var vg = tensor2(arena, cases[0].kv_len, cfg.kv_cols)
    paged_gather(kg, caches.k, table)
    paged_gather(vg, caches.v, table)
    var ref_out = tensor2(arena, cases[0].q_len, cfg.q_cols)
    var ref_scores = tensor2(arena, cases[0].q_len, cases[0].kv_len)
    attention(ref_out, q, kg, vg, ref_scores, cfg.n_heads, n_kv, cfg.head_dim)

    var got = tensor2(arena, cases[0].q_len, cfg.q_cols)
    var scores = tensor2(arena, cases[0].q_len, cases[0].kv_len)
    paged_attention_vec(
        got, q, caches.k, caches.v, table, scores, cfg.n_heads, n_kv, cfg.head_dim
    )
    assert_equal(bit_diffs(got, ref_out), 0, "基线必须逐位一致")
    f32_data(got)[unsafe_offset=3] = Float32(12345.0)
    assert_true(bit_diffs(got, ref_out) == 1, "篡改必须被逐位比较抓到")
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
