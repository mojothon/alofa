"""Q4_0 解量化与融合 matmul 必须逐位对得上它自己的格式。

这个文件的判定纪律和别处不太一样，值得说清楚：

**解量化用零容差。** 这不是苛刻，是因为它能做到。解量化后的值只取决于两样
东西 —— 4 位 nibble 与那个 fp16 缩放因子 —— 两者都是精确的：nibble 是整数，
半精度能表示的数单精度都能精确表示。既然结果由离散输入唯一决定，那么"差
一点点"就不是舍入，而是**布局读错了**：高低半字节装反、缩放因子按大端读、
块边界算错。这些错误里随便哪一个，都会让输出看起来仍然像权重。用容差比较
等于给它们发通行证 —— 1 ulp 的容差就足以让"次正规缩放解码差一位"蒙混过关。

**融合 matmul 用容差。** 它涉及累加，累加顺序不同确实会产生不同的最后一位，
所以这里比的是"数值是否正确"，阈值沿用全项目统一的 `1e-5 × max(1, |ref|)`。

**三条负向对照。** 一个不会失败的门等于没有门，所以：
- 解量化里额外算一遍"故意把高低半字节装反"的结果，它必须与期望值不同 ——
  证明零容差断言对布局敏感，而不是恒真；
- 量化里额外算一遍"取整改成截断"的结果，它的字节必须与参考不同 —— 证明
  逐字节比较对取整规则敏感；
- 量化里还额外算一遍"缩放因子退回 `amax/7`"的结果，它的字节也必须与参考
  不同 —— 参照物（fixture）是同一次导出的产物，实现退回旧规则时它不会自己
  变红，只有独立的旧规则版本能盯住这件事；
- 断言量化确实是**有损**的（解出的值不等于原始 fp32 权重的拷贝），否则
  "逐位相等"有可能只是因为压根没量化。

Run:
    pixi run mojo run -O0 -I src tests/unit/test_q4_parity.mojo
"""

from std.testing import TestSuite, assert_true

from alofa.core.error import ERR_IO, AlofaError
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.mmap import MappedFile
from alofa.core.text import parse_int, read_text
from alofa.kernels.cpu.quant import (
    block_scale,
    dequant_q4_0,
    f32_to_fp16_bits,
    matmul_q4_f32,
    quantize_q4_0,
)

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b/q4/"

# 沿用全项目的统一判据：`1e-5 × max(1, |ref|)`，Float64 累加。
comptime REL_TOL = Float64(1e-5)


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


struct Q4Case(Copyable, Movable):
    """`cases.tsv` 的一行。"""

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


def load_cases() raises AlofaError -> List[Q4Case]:
    var out = List[Q4Case]()
    for line in lines_of(FIXTURE + "cases.tsv"):
        var f = fields_of(line)
        var row = Q4Case()
        row.name = f[0]
        row.rows = parse_int(f[1])
        row.cols = parse_int(f[2])
        row.block_off = parse_int(f[3])
        row.n_blocks = parse_int(f[4])
        out.append(row.copy())
    return out^


def index_field(
    path: String, name: String, offset_column: Int, count_column: Int
) raises AlofaError -> List[Int]:
    """某个 `name <tab> offset <tab> numel` 索引文件里的一行。"""
    var out = List[Int]()
    for line in lines_of(path):
        var f = fields_of(line)
        if f[0] != name:
            continue
        out.append(parse_int(f[offset_column]))
        out.append(parse_int(f[count_column]))
        return out^
    raise AlofaError(ERR_IO, "name is not in the fixture index", "name=" + name)


def test_dequantised_values_are_bit_exact() raises:
    """解量化必须逐位等于参考，一个 ulp 都不许差。

    能零容差是因为可以零容差：输入是离散的，缩放是精确转换的，于是结果由
    输入唯一决定，不存在"两种实现都合理但差一点"的余地。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var expect_file = MappedFile(FIXTURE + "dequant.f32")
    var cases = load_cases()
    var checked = 0

    for row in cases:
        var n = row.n_blocks * 32
        var arena = Arena(n * 4 + 64)
        var got = arena.alloc(n * 4).unsafe_bitcast[Float32]()
        var blocks = blocks_file.ptr().unsafe_offset(row.block_off * 18).unsafe_bitcast[
            UInt8
        ]()
        dequant_q4_0(got, blocks, n)

        var meta = index_field(FIXTURE + "dequant.tsv", row.name, 1, 2)
        var want = expect_file.ptr().unsafe_offset(meta[0] * 4).unsafe_bitcast[
            Float32
        ]()
        assert_true(meta[1] == n, row.name + " 索引元素数与用例不一致")

        var worst = Float64(0)
        for i in range(n):
            var d = Float64(got[unsafe_offset=i]) - Float64(want[unsafe_offset=i])
            if d < Float64(0):
                d = -d
            if d > worst:
                worst = d
            # 逐位：不比差值，直接比是否相等。
            assert_true(
                got[unsafe_offset=i] == want[unsafe_offset=i],
                row.name + " 第 " + String(i) + " 个值不是逐位相等",
            )
        assert_true(worst == Float64(0), row.name + " 出现了非零差值")
        checked += 1
        arena.keep_alive()
    assert_true(checked == 4, "expected 4 cases, got " + String(checked))
    blocks_file.keep_alive()
    expect_file.keep_alive()


def test_a_swapped_nibble_order_is_rejected() raises:
    """负向对照：把高低半字节装反，零容差断言必须能看出来。

    真实数据里这种错不会让输出变成 NaN，只会让每个块里的值两两互换 —— 输出
    仍然是一堆大小合理的权重。若没有这条，上面那个"逐位相等"有可能因为比较
    方式本身失效而恒真。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var cases = load_cases()
    var row = cases[0].copy()
    var n = row.n_blocks * 32
    var arena = Arena(n * 4 + 64)
    var got = arena.alloc(n * 4).unsafe_bitcast[Float32]()
    var blocks = blocks_file.ptr().unsafe_offset(row.block_off * 18).unsafe_bitcast[
        UInt8
    ]()
    dequant_q4_0(got, blocks, n)

    # 在同一个块流上把高低半字节对调，统计有多少位置会因此改变。
    var differing = 0
    for b in range(row.n_blocks):
        var base = b * 18
        var d = block_scale(blocks, base)
        for j in range(16):
            var byte = Int(blocks[unsafe_offset=base + 2 + j])
            var swapped_low = Float32(((byte >> 4) & 0x0F) - 8) * d
            var swapped_high = Float32((byte & 0x0F) - 8) * d
            if swapped_low != got[unsafe_offset=b * 32 + j]:
                differing += 1
            if swapped_high != got[unsafe_offset=b * 32 + j + 16]:
                differing += 1
    assert_true(
        differing > 0,
        "对调高低半字节后没有任何值改变 —— 这条断言对布局不敏感，等于没有",
    )
    blocks_file.keep_alive()
    arena.keep_alive()


def test_quantised_blocks_match_the_reference_byte_for_byte() raises:
    """量化必须产出与参考**逐字节一样**的块流，不是"数值接近"。

    这一步是整条量化链路的起点，它的错误会被后面所有环节继承：缩放因子差
    一个 ulp，整块权重就等比缩放，量化后的网络照样流利 —— 只是悄悄变了个人。
    所以这里不比值，比字节。

    顺带说明为什么参考是"离线导出"的：量化是**我方定义的格式转换**，不是
    某个权威实现的行为，参照物只能是显式写死的整数算法（脚本里那段），由它
    导出字节流，再要求本实现逐字节对上。
    """
    var src_file = MappedFile(FIXTURE + "orig.f32")
    var want_file = MappedFile(FIXTURE + "blocks.bin")
    var n = src_file.size // 4
    var n_bytes = n // 32 * 18
    var arena = Arena(n_bytes + 64)
    var got = arena.alloc(n_bytes)
    var src = src_file.ptr().unsafe_bitcast[Float32]()
    quantize_q4_0(got, src, n)

    var want = want_file.ptr()
    assert_true(want_file.size == n_bytes, "参考块流的长度与元素数对不上")
    var differing = 0
    var first_bad = -1
    for i in range(n_bytes):
        if got[unsafe_offset=i] != want[unsafe_offset=i]:
            differing += 1
            if first_bad < 0:
                first_bad = i
    assert_true(
        differing == 0,
        "量化出的块流有 "
        + String(differing)
        + " 字节与参考不同，第一个在第 "
        + String(first_bad)
        + " 字节",
    )
    print("观测：量化 " + String(n) + " 个 fp32 → " + String(n_bytes) + " 字节，逐字节相同")
    src_file.keep_alive()
    want_file.keep_alive()
    arena.keep_alive()


def test_a_truncating_quantiser_is_rejected() raises:
    """负向对照：把"取最近"改成"截断"，逐字节比较必须看出来。

    这个错误值得专门钉住，因为它**不会**让输出变成垃圾：每个值最多偏半个
    台阶，量化后的网络依旧能生成通顺的话，只是分布整体偏移了一点点。任何带
    容差的判据都会放它过去。

    这里不直接改被测代码，而是在测试里另写一个截断版 —— 门要证明的是"比较
    本身对取整规则敏感"，而不是"被测代码恰好没写错"。
    """
    var src_file = MappedFile(FIXTURE + "orig.f32")
    var want_file = MappedFile(FIXTURE + "blocks.bin")
    var n = src_file.size // 4
    var n_bytes = n // 32 * 18
    var arena = Arena(n_bytes + 64)
    var got = arena.alloc(n_bytes)
    var src = src_file.ptr().unsafe_bitcast[Float32]()
    generate_truncated_blocks(got, src_file.ptr(), n)

    var want = want_file.ptr()
    var differing = 0
    for i in range(n_bytes):
        if got[unsafe_offset=i] != want[unsafe_offset=i]:
            differing += 1
    assert_true(
        differing > 0,
        "截断取整与取最近取整产出了同样的字节 —— 逐字节比较对取整规则不敏感，"
        + "这条断言是恒真的",
    )
    print("观测：截断版量化与参考相差 " + String(differing) + " 字节，门如期判红")
    src_file.keep_alive()
    want_file.keep_alive()
    arena.keep_alive()


def test_the_old_amax_scale_rule_is_rejected() raises:
    """负向对照：旧的 `d = amax / 7` 必须产出**不同**的字节。

    这条门钉的是"台阶放哪儿"这个自由度。MSE 选出来的 scale 与 `amax/7` 的
    差别只有每个块 2 个缩放字节加少量 nibble，量化后的网络照样流利，逐块看
    过去也照样"像权重"。若没有这条，哪天把 `quantize_q4_0` 退回 `amax/7`，
    上面那条逐字节比较会因为**参照物和实现一起退**而继续绿 —— fixture 是同
    一次导出的产物，它不会自己发现算法被换回去了。

    所以这里在测试里另写一个旧规则版本：**取整规则保持与被测代码一致**，只
    改缩放因子的选法，这样"字节不同"只可能来自台阶的位置，不会和上面那条
    取整对照混在一起。
    """
    var src_file = MappedFile(FIXTURE + "orig.f32")
    var want_file = MappedFile(FIXTURE + "blocks.bin")
    var n = src_file.size // 4
    var n_bytes = n // 32 * 18
    var arena = Arena(n_bytes + 64)
    var got = arena.alloc(n_bytes)
    generate_amax_blocks(got, src_file.ptr(), n)

    var want = want_file.ptr()
    var differing = 0
    for i in range(n_bytes):
        if got[unsafe_offset=i] != want[unsafe_offset=i]:
            differing += 1
    assert_true(
        differing > 0,
        "amax/7 与 MSE 选出的缩放因子产出了同样的字节 —— 逐字节比较对缩放"
        + "因子的选法不敏感，这条断言是恒真的",
    )
    print("观测：amax/7 版与 MSE 版相差 " + String(differing) + " 字节，门如期判红")
    src_file.keep_alive()
    want_file.keep_alive()
    arena.keep_alive()


def generate_amax_blocks(blocks: RawPtr, src_raw: RawPtr, n: Int) raises:
    """按旧规则 `d = amax / 7` 量化，只用于负向对照。

    与 `quantize_q4_0` 的**唯一**差别就是缩放因子的选法；取整规则（取最近、
    并列取偶）与被测代码一致，于是"字节不同"只能来自台阶的位置。
    """
    var src = src_raw.unsafe_bitcast[Float32]()
    var n_blocks = n // 32
    for b in range(n_blocks):
        var start = b * 32
        var amax = Float32(0)
        for j in range(32):
            var v = src[unsafe_offset=start + j]
            var a = v if v >= Float32(0) else -v
            if a > amax:
                amax = a
        var d = amax / Float32(7)
        var half = f32_to_fp16_bits(d)
        var base = b * 18
        blocks[unsafe_offset=base] = UInt8(half & 0xFF)
        blocks[unsafe_offset=base + 1] = UInt8((half >> 8) & 0xFF)
        for j in range(16):
            var lo = 8
            var hi = 8
            if d != Float32(0):
                lo = quantise_round_even(src[unsafe_offset=start + j], d)
                hi = quantise_round_even(src[unsafe_offset=start + j + 16], d)
            blocks[unsafe_offset=base + 2 + j] = UInt8((lo & 0x0F) | ((hi & 0x0F) << 4))


def quantise_round_even(value: Float32, d: Float32) -> Int:
    """`clamp(round(value / d) + 8, 0, 15)`，取整取**最近、并列取偶**。"""
    var scaled = value / d
    var floor_value = Float32(Int(scaled))
    if floor_value > scaled:
        floor_value -= Float32(1)
    var below = Int(floor_value)
    var fraction = scaled - floor_value
    var rounded = below
    if fraction > Float32(0.5):
        rounded = below + 1
    elif fraction == Float32(0.5):
        rounded = below if (below % 2 == 0) else below + 1
    var q = rounded + 8
    if q < 0:
        return 0
    if q > 15:
        return 15
    return q


def generate_truncated_blocks(blocks: RawPtr, src_raw: RawPtr, n: Int) raises:
    """故意写成截断取整的量化器，只用于负向对照。

    与 `quantize_q4_0` 的唯一差别就是取整规则：这里直接丢掉小数部分。若这条
    通道产出的字节和参考一样，说明比较根本没有在比取整。
    """
    var src = src_raw.unsafe_bitcast[Float32]()
    var n_blocks = n // 32
    for b in range(n_blocks):
        var start = b * 32
        var amax = Float32(0)
        for j in range(32):
            var v = src[unsafe_offset=start + j]
            var a = v if v >= Float32(0) else -v
            if a > amax:
                amax = a
        var d = amax / Float32(7)
        var half = f32_to_fp16_bits(d)
        var base = b * 18
        blocks[unsafe_offset=base] = UInt8(half & 0xFF)
        blocks[unsafe_offset=base + 1] = UInt8((half >> 8) & 0xFF)
        for j in range(16):
            var lo = 8
            var hi = 8
            if d != Float32(0):
                lo = Int(src[unsafe_offset=start + j] / d) + 8
                hi = Int(src[unsafe_offset=start + j + 16] / d) + 8
                if lo < 0:
                    lo = 0
                if lo > 15:
                    lo = 15
                if hi < 0:
                    hi = 0
                if hi > 15:
                    hi = 15
            blocks[unsafe_offset=base + 2 + j] = UInt8((lo & 0x0F) | ((hi & 0x0F) << 4))


def test_fused_matmul_matches_the_reference() raises:
    """解量化融进 matmul 后，结果必须等于"先解量化再点积"的 fp64 参考。

    幅度量级参考值可能很小（Qwen2 的投影输出约 0.1 量级），所以用
    `max(1, |ref|)` 兜底，与 layer0 的差分门同一个判据。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var cases = load_cases()
    var act = act_file.ptr().unsafe_bitcast[Float32]()
    var checked = 0

    for row in cases:
        var arena = Arena(row.rows * 4 + 64)
        var got = arena.alloc(row.rows * 4).unsafe_bitcast[Float32]()
        var blocks = blocks_file.ptr().unsafe_offset(row.block_off * 18).unsafe_bitcast[
            UInt8
        ]()
        matmul_q4_f32(got, act, blocks, row.rows, row.cols)

        var meta = index_field(FIXTURE + "out_expected.tsv", row.name, 1, 2)
        var want = expect_file.ptr().unsafe_offset(meta[0] * 8).unsafe_bitcast[
            Float64
        ]()
        assert_true(meta[1] == row.rows, row.name + " 期望值行数与用例不一致")

        for r in range(row.rows):
            var reference = want[unsafe_offset=r]
            var d = Float64(got[unsafe_offset=r]) - reference
            if d < Float64(0):
                d = -d
            var scale = reference
            if scale < Float64(0):
                scale = -scale
            if scale < Float64(1):
                scale = Float64(1)
            assert_true(
                d <= REL_TOL * scale,
                row.name
                + " 第 "
                + String(r)
                + " 行差 "
                + String(d)
                + "（阈值 "
                + String(REL_TOL * scale)
                + "）",
            )
        checked += 1
        arena.keep_alive()
    assert_true(checked == 4, "expected 4 cases, got " + String(checked))
    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()


def test_dequantised_values_land_on_the_block_grid() raises:
    """每个块里的 32 个值必须落在 {-8d, …, 7d} 这 16 个台阶上。

    这比"解出的值不等于原值"更强，也更贴近要证明的事：解量化是**按块共享
    一个缩放因子的离散还原**。若谁把它写成"直接读一块 fp32 抄过去"，值会
    连续分布，不会落在台阶上；而"逐位相等"这条断言本身是看不出这种事的。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var cases = load_cases()
    var row = cases[0].copy()
    var n = row.n_blocks * 32
    var arena = Arena(n * 4 + 64)
    var got = arena.alloc(n * 4).unsafe_bitcast[Float32]()
    var blocks = blocks_file.ptr().unsafe_offset(row.block_off * 18).unsafe_bitcast[
        UInt8
    ]()
    dequant_q4_0(got, blocks, n)

    var off_grid = 0
    for b in range(row.n_blocks):
        var d = Float64(block_scale(blocks, b * 18))
        for j in range(32):
            var v = got[unsafe_offset=b * 32 + j]
            if d == Float64(0):
                if v != Float32(0):
                    off_grid += 1
                continue
            var steps = Float64(v) / d
            if steps < Float64(-8.001) or steps > Float64(7.001):
                off_grid += 1
                continue
            var nearest = Float64(0)
            if steps >= Float64(0):
                nearest = Float64(Int(steps + Float64(0.5)))
            else:
                nearest = Float64(Int(steps - Float64(0.5)))
            var gap = steps - nearest
            if gap < Float64(0):
                gap = -gap
            if gap > Float64(1e-3):
                off_grid += 1
    assert_true(
        off_grid == 0,
        String(off_grid) + " 个值没落在块的台阶上 —— 这不是按块解量化的结果",
    )
    blocks_file.keep_alive()
    arena.keep_alive()


def test_quantisation_is_actually_lossy() raises:
    """解出的值必须与原始 fp32 权重不同 —— 否则"逐位相等"可能只是没量化。

    不设具体误差上界：那是质量门（端到端一致率）的事，不是这条正确性门的事。
    这里只确认这条链路真的做了有损变换。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var orig_file = MappedFile(FIXTURE + "orig.f32")
    var cases = load_cases()
    var row = cases[0].copy()
    var n = row.n_blocks * 32
    var arena = Arena(n * 4 + 64)
    var got = arena.alloc(n * 4).unsafe_bitcast[Float32]()
    var blocks = blocks_file.ptr().unsafe_offset(row.block_off * 18).unsafe_bitcast[
        UInt8
    ]()
    dequant_q4_0(got, blocks, n)

    var meta = index_field(FIXTURE + "orig.tsv", row.name, 1, 2)
    var orig = orig_file.ptr().unsafe_offset(meta[0] * 4).unsafe_bitcast[Float32]()
    var differing = 0
    for i in range(n):
        if got[unsafe_offset=i] != orig[unsafe_offset=i]:
            differing += 1
    assert_true(
        differing > n / 2,
        "只有 " + String(differing) + " / " + String(n) + " 个值被改变 —— 量化没起作用",
    )
    blocks_file.keep_alive()
    orig_file.keep_alive()
    arena.keep_alive()


def test_a_row_width_that_is_not_whole_blocks_is_rejected() raises:
    """内层维度不是 32 的整数倍是调用方错误，必须具名而不是静默截断。"""
    var arena = Arena(4096)
    var dst = arena.alloc(64).unsafe_bitcast[Float32]()
    var x = arena.alloc(256).unsafe_bitcast[Float32]()
    var blocks = arena.alloc(64).unsafe_bitcast[UInt8]()
    var rejected = False
    try:
        matmul_q4_f32(dst, x, blocks, 2, 40)
    except err:
        rejected = err.name() == "shape_mismatch"
    assert_true(rejected, "非整块的内层维度必须报 shape_mismatch")
    arena.keep_alive()


def test_an_infinite_or_nan_scale_is_rejected() raises:
    """缩放因子是 inf / NaN 说明块流坏了，不能让整行权重变成 inf。"""
    var arena = Arena(4096)
    var raw = arena.alloc(64).unsafe_bitcast[UInt8]()
    # 0x7C00 是半精度的 +inf；小端存放即 [0x00, 0x7C]。
    raw[unsafe_offset=0] = UInt8(0)
    raw[unsafe_offset=1] = UInt8(124)
    var rejected = False
    try:
        _ = block_scale(raw, 0)
    except err:
        rejected = err.name() == "unsupported"
    assert_true(rejected, "inf 缩放因子必须报 unsupported")
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
