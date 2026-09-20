"""融合 matmul 的**结构性**差分门（`q4vec` 夹具）。

这个文件和 `test_q4_parity.mojo` 分工不同，别合起来
----------------------------------------------------

`test_q4_parity` 回答的是"**值对不对**"，用的四个用例全是真实投影、每行 28 个块。
它管不到一件事：向量化实现里最容易写错的那一段 —— **尾巴**。28 能被 1/2/4/7/14/28
整除，所以任何"按 N 块展开、余数走标量尾巴"的写法都能在四个用例里全套绕过尾巴，
而尾巴一旦多读或少读一个块，28 的倍数也会照样通过别处的检查。

所以这里用另一份夹具 `tests/fixtures/.../q4vec/`（由
`scripts/dump_q4_matmul_cases.py` 导出），它的形状是**专门为把尾巴露出来**挑的：
块/行取 1、3、5、7、11、17、28、152，行数取 1、2、3、5、7、13。再加上三种数值
情形（全零块即 `d == 0`、全踩在 ±amax 使 nibble 只有 0/15、一堆 1e-8 夹一个 1.0）。

期望值来自同一个 numpy 参考脚本、同一套算法，所以**两条实现路径共用参照物**：
今天的标量 `matmul_q4_f32`，以及将来落地的向量化版本。此刻文件里只有前者 ——
那是有意的：**在还没有新东西可测的时候先把配额架设好**，等 `matmul_q4_f32_vec`
进来时它是往同一个门里加一条断言，而不是"顺便"顺手造一个会绿的检查。

三条负向对照（一个不会失败的门等于没门）
----------------------------------------

每条都跑同一个夹具、同一套判据，必须**被拒**：

- 省掉 `-8` 偏移 → 在 nibble 只有 0/15 的用例上偏约 8/7 倍；
- 高低半字节装反 → 布局读错的最小形态；
- 丢掉最后一个块 → 证明判据真的在看**全部** `cols`，少一段就会被抓。

Run:
    pixi run mojo run -O0 -I src tests/unit/test_q4_matmul_vec.mojo
"""

from std.testing import TestSuite, assert_true

from alofa.core.error import ERR_IO, AlofaError
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.mmap import MappedFile
from alofa.core.tensor import F32Ptr
from alofa.core.text import parse_int, read_text
from alofa.kernels.cpu.avx2 import _matmul_q4_f32acc
from alofa.kernels.cpu.avx2 import _matmul_q4_halves
from alofa.kernels.cpu.avx2 import matmul_q4_f32 as matmul_q4_f32_v
from alofa.kernels.cpu.quant import block_scale, matmul_q4_f32

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b/q4vec/"

# 沿用全项目的统一判据：`1e-5 × max(1, |ref|)`，Float64 累加。
comptime REL_TOL = Float64(1e-5)

comptime Q4_BLOCK = 32
comptime Q4_BYTES = 18


def lines_of(path: String) raises AlofaError -> List[String]:
    var text = read_text(path)
    var out = List[String]()
    for span in text.split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        out.append(line)
    return out^


def fields_of(line: String) raises AlofaError -> List[String]:
    var out = List[String]()
    for span in line.split("\t"):
        out.append(String(span))
    return out^


struct MatCase(Copyable, Movable):
    """`cases.tsv` 的一行。`cases.tsv` 还有第 6 列（权重怎么造出来的），那是给
    人看的存档，测试不读它 —— 免得改构造方式就要改断言。"""

    var name: String
    var rows: Int
    var cols: Int
    var block_off: Int
    var n_blocks: Int

    def __init__(out self):
        self.name = ""
        self.rows = 0
        self.cols = 0
        self.block_off = 0
        self.n_blocks = 0

    def blocks_per_row(imm self) -> Int:
        return self.cols // Q4_BLOCK


def load_cases() raises AlofaError -> List[MatCase]:
    var out = List[MatCase]()
    for line in lines_of(FIXTURE + "cases.tsv"):
        var f = fields_of(line)
        var row = MatCase()
        row.name = f[0]
        row.rows = parse_int(f[1])
        row.cols = parse_int(f[2])
        row.block_off = parse_int(f[3])
        row.n_blocks = parse_int(f[4])
        out.append(row.copy())
    return out^


def index_field(path: String, name: String) raises AlofaError -> List[Int]:
    """某个 `name <tab> offset <tab> numel` 索引文件里的一行。"""
    var out = List[Int]()
    for line in lines_of(path):
        var f = fields_of(line)
        if f[0] != name:
            continue
        out.append(parse_int(f[1]))
        out.append(parse_int(f[2]))
        return out^
    raise AlofaError(ERR_IO, "name is not in the fixture index", "name=" + name)


def worst_deviation(got: F32Ptr, want: RawPtr, rows: Int) -> Float64:
    """与参考值的最大相对偏差；参考值小于 1 时按 1 兜底（全项目统一判据）。"""
    var worst = Float64(0)
    for r in range(rows):
        var reference = want.unsafe_offset(r * 8).unsafe_bitcast[Float64]()[0]
        var d = Float64(got[unsafe_offset=r]) - reference
        if d < Float64(0):
            d = -d
        var scale = reference
        if scale < Float64(0):
            scale = -scale
        if scale < Float64(1):
            scale = Float64(1)
        d = d / scale
        if d > worst:
            worst = d
    return worst


def case_peak(want: RawPtr, rows: Int) -> Float64:
    """这一例参考值的最大绝对值 —— 用来区分"真的因为这条例外被拒"和
    "整例输出就是 0，怎么改都拒不掉"（`one_block` 是全零权重）。"""
    var peak = Float64(0)
    for r in range(rows):
        var v = want.unsafe_offset(r * 8).unsafe_bitcast[Float64]()[0]
        if v < Float64(0):
            v = -v
        if v > peak:
            peak = v
    return peak


def blocks_at(blocks_file: RawPtr, block_off: Int) -> RawPtr:
    return blocks_file.unsafe_offset(block_off * Q4_BYTES)


# ---------------------------------------------------------------------------
# 负向对照用的三种变异。它们各自只改**一件事**，其余部分照抄真实实现 ——
# 否则"红"就不知道该算到谁头上。
# ---------------------------------------------------------------------------


def variant_without_offset(dst: F32Ptr, x: F32Ptr, blocks: RawPtr, rows: Int, cols: Int) raises:
    """忘了 `-8`：nibble 直接当值用。"""
    var blocks_per_row = cols // Q4_BLOCK
    for row in range(rows):
        var acc = Float64(0)
        for b in range(blocks_per_row):
            var base = (row * blocks_per_row + b) * Q4_BYTES
            var d = Float64(block_scale(blocks, base))
            var x_base = b * Q4_BLOCK
            for j in range(16):
                var byte = Int(blocks.unsafe_offset(base + 2 + j)[0])
                var w0 = d * Float64(byte & 0x0F)
                var w1 = d * Float64((byte >> 4) & 0x0F)
                acc += w0 * Float64(x[unsafe_offset=x_base + j])
                acc += w1 * Float64(x[unsafe_offset=x_base + j + 16])
        dst[unsafe_offset=row] = Float32(acc)


def variant_swapped_nibbles(dst: F32Ptr, x: F32Ptr, blocks: RawPtr, rows: Int, cols: Int) raises:
    """高低半字节装反：第 j 个值去读高半字节。"""
    var blocks_per_row = cols // Q4_BLOCK
    for row in range(rows):
        var acc = Float64(0)
        for b in range(blocks_per_row):
            var base = (row * blocks_per_row + b) * Q4_BYTES
            var d = Float64(block_scale(blocks, base))
            var x_base = b * Q4_BLOCK
            for j in range(16):
                var byte = Int(blocks.unsafe_offset(base + 2 + j)[0])
                var w0 = Float64(((byte >> 4) & 0x0F) - 8) * d
                var w1 = Float64((byte & 0x0F) - 8) * d
                acc += w0 * Float64(x[unsafe_offset=x_base + j])
                acc += w1 * Float64(x[unsafe_offset=x_base + j + 16])
        dst[unsafe_offset=row] = Float32(acc)


def variant_dropped_last_block(dst: F32Ptr, x: F32Ptr, blocks: RawPtr, rows: Int, cols: Int) raises:
    """少读最后一块：证明判据真的看完了整个 `cols`。"""
    var blocks_per_row = cols // Q4_BLOCK
    for row in range(rows):
        var acc = Float64(0)
        for b in range(blocks_per_row - 1):
            var base = (row * blocks_per_row + b) * Q4_BYTES
            var d = Float64(block_scale(blocks, base))
            var x_base = b * Q4_BLOCK
            for j in range(16):
                var byte = Int(blocks.unsafe_offset(base + 2 + j)[0])
                var w0 = Float64((byte & 0x0F) - 8) * d
                var w1 = Float64(((byte >> 4) & 0x0F) - 8) * d
                acc += w0 * Float64(x[unsafe_offset=x_base + j])
                acc += w1 * Float64(x[unsafe_offset=x_base + j + 16])
        dst[unsafe_offset=row] = Float32(acc)


# ---------------------------------------------------------------------------


def check_kernel_against_reference(
    cases: List[MatCase],
    blocks_file: RawPtr,
    act_file: RawPtr,
    expect_file: RawPtr,
    kernel: String,
    report: Bool,
) raises -> Int:
    var checked = 0
    for row in cases:
        var act_meta = index_field(FIXTURE + "acts.tsv", row.name)
        var out_meta = index_field(FIXTURE + "out_expected.tsv", row.name)
        assert_true(
            act_meta[1] == row.cols, row.name + " 激活长度与 cols 不一致"
        )
        assert_true(out_meta[1] == row.rows, row.name + " 期望值行数与用例不一致")

        var arena = Arena(row.rows * 4 + 64)
        var got = arena.alloc(row.rows * 4).unsafe_bitcast[Float32]()
        var act = act_file.unsafe_offset(act_meta[0] * 4).unsafe_bitcast[Float32]()
        var want = expect_file.unsafe_offset(out_meta[0] * 8)
        var blocks = blocks_at(blocks_file, row.block_off)

        if kernel == "halves":
            _matmul_q4_halves(
                got,
                act,
                blocks.unsafe_bitcast[UInt8](),
                row.rows,
                row.cols,
                got,
                False,
            )
        elif kernel == "f32acc":
            _matmul_q4_f32acc(
                got,
                act,
                blocks.unsafe_bitcast[UInt8](),
                row.rows,
                row.cols,
            )
        elif kernel == "vector":
            matmul_q4_f32_v(
                got,
                act,
                blocks.unsafe_bitcast[UInt8](),
                row.rows,
                row.cols,
            )
        else:
            matmul_q4_f32(
                got,
                act,
                blocks.unsafe_bitcast[UInt8](),
                row.rows,
                row.cols,
            )
        var worst = worst_deviation(got, want, row.rows)
        if report:
            print("    " + row.name + " 最大相对偏差 = " + String(worst))
        assert_true(
            worst <= REL_TOL,
            row.name + " 与参考的最大相对偏差 " + String(worst) + " 超过 " + String(REL_TOL),
        )
        checked += 1
        arena.keep_alive()
    return checked


def assert_mutation_is_rejected(
    label: String,
    kernel: String,
    cases: List[MatCase],
    blocks_file: RawPtr,
    act_file: RawPtr,
    expect_file: RawPtr,
) raises -> None:
    """跑一种变异实现：凡是输出**不为零**的用例，都必须被判据拒掉。

    跳过"参考值本来就是 0"的用例（`one_block` 是全零权重）不是放水：那种用例上
    任何变异都得到 0，通不过/过得了都说明不了判据的分辨力 —— 真靠它，等于给门
    留一个恒真的假名将。
    """
    var informative = 0
    var rejected = 0
    var blind = ""
    var blind_names = List[String]()
    var prod_rejected = 0
    for row in cases:
        var act_meta = index_field(FIXTURE + "acts.tsv", row.name)
        var out_meta = index_field(FIXTURE + "out_expected.tsv", row.name)
        var arena = Arena(row.rows * 4 + 64)
        var got = arena.alloc(row.rows * 4).unsafe_bitcast[Float32]()
        var act = act_file.unsafe_offset(act_meta[0] * 4).unsafe_bitcast[Float32]()
        var want = expect_file.unsafe_offset(out_meta[0] * 8)
        var blocks = blocks_at(blocks_file, row.block_off)

        if kernel == "no_offset":
            variant_without_offset(got, act, blocks, row.rows, row.cols)
        elif kernel == "swapped":
            variant_swapped_nibbles(got, act, blocks, row.rows, row.cols)
        else:
            variant_dropped_last_block(got, act, blocks, row.rows, row.cols)

        if case_peak(want, row.rows) <= Float64(0):
            arena.keep_alive()
            continue
        informative += 1
        var worst = worst_deviation(got, want, row.rows)
        if worst > REL_TOL:
            rejected += 1
            if row.name == "real13" or row.name == "down_like":
                prod_rejected += 1
        else:
            blind += " " + row.name
            blind_names.append(row.name)
        arena.keep_alive()

    # 诊断行：绿的时候也要看得见"哪几条根本没参加"，否则失败信息里那句
    # "看不见的用例" 会被当成一句套话。
    print("  " + label + " 拒 " + String(rejected) + "/" + String(informative) + "，看不见：" + blind)

    # 每种变异**在数学上**看不见的用例，连原因一起列出来。用这张表而不是降阈值：
    # 只要这张表之外又多出一个"看不见"的用例，就说明夹具变了、或者实现里多了一处
    # 解释不掉的巧合 —— 两种情况都该让人停下来看一眼。
    var expected_blind = List[String]()
    if kernel == "no_offset":
        # w 与 x 都 ±交替：省掉的常量 8d 乘上 Σx 正好为 0，误差被代数地消掉。
        expected_blind.append("tail7")
    elif kernel == "swapped":
        # 整行同一值 → 所有 nibble 相同，装反等于没动。
        expected_blind.append("odd_blocks")
        # ±交替的周期是 2，j ↔ j+16 落在同一相位上。
        expected_blind.append("tail7")
        # 几乎整块都落在代表 0 的那个台阶上，装反前后都是 0。
        expected_blind.append("tail11")
        # 见下一条：该例 |ref| ≈ 1.8e-5，判据在这里退化成绝对容差。
        expected_blind.append("const_row")
    else:
        # ⚠️ 这条不是"变异无害"，是**判据在这里没牙**：该例参考值约 1.8e-5，而全项目
        # 统一的 `1e-5 × max(1, |ref|)` 在 |ref| ≪ 1 时退化为绝对 1e-5 —— 少读一整块的
        # 贡献约 5e-6，正好躲在下面。这是已知弱点，写在账本里，不假装没看见。
        expected_blind.append("const_row")

    var unexpected_blind = ""
    for i in range(blind_names.__len__()):
        var name = blind_names[i].copy()
        var known = False
        for j in range(expected_blind.__len__()):
            if expected_blind[j] == name:
                known = True
        if not known:
            unexpected_blind += " " + name

    assert_true(
        informative >= 8,
        label + " 有信息的用例只有 " + String(informative) + " 个，夹具被改薄了",
    )
    assert_true(
        prod_rejected == 2,
        label
        + " 没有在两个真实用例（real13 / down_like）上被拒 —— "
        + "那是合成数据之外唯一两条真实分布的指望",
    )
    assert_true(
        unexpected_blind == "",
        label
        + " 出现了没解释过的不可见用例："
        + unexpected_blind
        + " —— 要么夹具被改了，要么实现里多了一处巧合",
    )


def test_fixture_shapes_expose_the_simd_tail() raises:
    """夹具必须真的露出尾巴，否则上面所有断言都只是看着像覆盖了。

    这条守卫的是**夹具本身**：将来谁为了省几 KB 把奇形怪状的用例删了，这里会红，
    而不是让某个向量化实现带着一条从没被走过的尾巴混进去。
    """
    var cases = load_cases()
    assert_true(cases.__len__() >= 8, "夹具用例被删减了")

    var visible = List[Int]()
    for row in cases:
        var bpr = row.blocks_per_row()
        var worse_than_for = 0
        for unroll in range(4):
            var u = 1 << (unroll + 1)
            if bpr % u != 0:
                worse_than_for += 1
        if worse_than_for > 0:
            visible.append(bpr)
    assert_true(
        visible.__len__() >= 5,
        "能对 2/4/8/16 任一展开产生余数的用例只剩 "
        + String(visible.__len__())
        + " 个 —— 尾巴从夹具里消失了",
    )

    var row_tails = 0
    for row in cases:
        if row.rows % 4 != 0:
            row_tails += 1
    assert_true(row_tails >= 3, "行数不被 4 整除的用例不足，行方向尾巴没被覆盖")

    var single_block = 0
    for row in cases:
        if row.blocks_per_row() == 1:
            single_block += 1
    assert_true(
        single_block >= 1,
        "缺少「每行只有一个块」的用例 —— 那种形状下任何展开倍数都走尾巴",
    )


def test_scalar_path_matches_the_reference() raises:
    """今天的融合 matmul（`matmul_q4_f32`）在这份夹具上必须对得上。

    现在它是唯一的实现，所以这一条看上去像"又在测标量"。它真正的作用是**先把
    参考栈踩通**：夹具读得对、判据量得对、 九个形状全都走得到。等向量化版本
    进来，它跑的是同一个 `check_*`，差别只在调用哪个核。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var cases = load_cases()

    var checked = check_kernel_against_reference(
        cases, blocks_file.ptr(), act_file.ptr(), expect_file.ptr(), "scalar", False
    )
    assert_true(checked == 9, "expected 9 cases, got " + String(checked))

    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()


def test_vector_path_matches_the_reference() raises:
    """向量版必须落在同一份参考值上 —— 这是本轮的兑现点。

    它跑的是与标量那条完全相同的 `check_kernel_against_reference`，差别只有传进去的
    核。九个用例里的形状（块/行 1、3、5、7、11、17、28、152）就是为了这一条挑的：
    按 `j % 8` 切通道的写法如果写成了 `j % 4`，或者把"两个半块"写成"一个半块"，
    28 的倍数那几条照样会绿，塌的是奇数块那几条。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var cases = load_cases()

    var checked = check_kernel_against_reference(
        cases, blocks_file.ptr(), act_file.ptr(), expect_file.ptr(), "vector", False
    )
    assert_true(checked == 9, "expected 9 cases, got " + String(checked))

    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()


def test_f32_accumulation_deviation_is_measured() raises:
    """把 f32 累加的代价**量出来**，而不是争论它。

    每一条用例的最大相对偏差会被打印出来（看 logger 输出），因此这不是"能不能过"的
    判断题，而是给"要不要换累积类型"这件事留证据。特别注意 `real13` 与 `down_like`
    这两条**真实数据**的用例：合成数据上的 1e-7 说明不了什么，相消相长只会在真实
    激活上发生。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var cases = load_cases()

    var checked = check_kernel_against_reference(
        cases, blocks_file.ptr(), act_file.ptr(), expect_file.ptr(), "f32acc", True
    )
    assert_true(checked == 9, "expected 9 cases, got " + String(checked))

    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()


def test_halves_kernel_deviation_is_measured() raises:
    """半块切分（一次 16 值）的偏差也**量出来**，不靠"应该差不多"。

    它和 `test_f32_accumulation_deviation_is_measured` 一样是 f32 累加，所以预期
    同一量级；差别只在于块内怎么切向量。把两条并排留着，是为了让"换的是切法还是
    换的是精度"这件事在证据上分得开 —— 两件事混在一起时，变快了都不知道归谁。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var cases = load_cases()

    var checked = check_kernel_against_reference(
        cases, blocks_file.ptr(), act_file.ptr(), expect_file.ptr(), "halves", True
    )
    assert_true(checked == 9, "expected 9 cases, got " + String(checked))

    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()


def test_missing_offset_is_rejected() raises:
    """忘了 `-8` 必须被拒 —— 否则 nibble 到权重的映射没被验证。"""
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var cases = load_cases()
    assert_mutation_is_rejected(
        "去掉 -8 偏移",
        "no_offset",
        cases,
        blocks_file.ptr(),
        act_file.ptr(),
        expect_file.ptr(),
    )
    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()


def test_swapped_nibbles_are_rejected() raises:
    """高低半字节装反必须被拒 —— 这是块布局层面最小的那个错误。"""
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var cases = load_cases()
    assert_mutation_is_rejected(
        "高低半字节装反",
        "swapped",
        cases,
        blocks_file.ptr(),
        act_file.ptr(),
        expect_file.ptr(),
    )
    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()


def test_a_short_inner_loop_is_rejected() raises:
    """少读最后一块必须被拒 —— 证明判据看的是**全部** `cols`。

    这条对照是给"将来有人把内层切成两半并行"预备的：那种改法最典型的失败是某
    一半的边界差一块，而顺序遍历的版本在这里就已经红过了。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var cases = load_cases()
    assert_mutation_is_rejected(
        "丢掉最后一块",
        "dropped",
        cases,
        blocks_file.ptr(),
        act_file.ptr(),
        expect_file.ptr(),
    )
    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
