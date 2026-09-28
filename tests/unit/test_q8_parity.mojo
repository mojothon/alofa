"""q8_0 的量化 / 解量化 / 融合 matmul 必须对得上它自己的格式。

判定纪律与 `test_q4_parity.mojo` 同一套，值得重复一遍：

**解量化用零容差。** 能这么做是因为可以这么做：解出的值只取决于 8 位码与那个
fp16 缩放因子，两者都是**精确**的（码是整数，fp16 能表示的数 fp32 都能精确表示
⇒ 转换无损）。既然结果由离散输入唯一决定，"差一点点"就不是舍入，而是**读错了
字节**：忘了符号扩展、块步长算错、缩放按大端读。随便哪一个都会让输出看起来仍
然像权重 —— 用容差等于给它们发通行证。

**融合 matmul 用容差** `1e-5 × max(1, |ref|)`（全项目统一判据）。期望值是用**解
量化后**的权重按 fp64 算出来的，所以这条只盯**累加顺序**这一件事 —— 若拿原始
fp32 权重当期望，量化误差（q8 相对 L2 就有 0.54%）会一口吃掉 1e-5 的判据。

**三条负向对照。** 一个不会失败的门等于没有门：
- 忘了**符号扩展**（把有符号码当无符号读）必须与期望不同 —— 证明零容差断言对
  这一个字节的解读敏感；
- 取整改成**截断**后，量化出的字节必须与参考不同 —— 证明逐字节比较对取整规则
  敏感；
- 块**步长**按 32 而不是 34 读，结果必须与期望不同 —— 证明门盯得住块边界。

**为什么有 q8_0**（2026-09-28 量出来的理由，不是格式偏好）：同机同 prompt 与
llama.cpp 对比时 alofa 的 q4_0 档只有对方 0.54…0.60×，而同流量对照（把 llama.cpp
的嵌入/输出张量也拖回 F32，它从 25.60 掉到 13.44，与 alofa 的 13.83…15.37 重合）
证明差距全在**输出投影的存储宽度**。输出投影压到 **q4_0 是量过、否掉的**（教师
强制贪心一致率 0.8691 → 0.77），8 bit 是它在质量上还站得住的最窄一档 —— 本文件
断言它的相对 L2 **小于 1%**（实测 0.54%，q4_0 是 10.3%）。

Run:
    pixi run mojo run -O0 -I src tests/unit/test_q8_parity.mojo
"""

from std.testing import TestSuite, assert_true

from alofa.core.memory import Arena
from alofa.core.mmap import MappedFile
from alofa.core.tensor import F32Ptr
from alofa.kernels.cpu.avx2 import matmul_q8_f32_vec
from alofa.kernels.cpu.quant import (
    U8Ptr,
    block_scale,
    dequant_q8_0,
    f32_to_fp16_bits,
    matmul_q8_f32,
    quantize_q8_0,
)

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b/q8/"

# `cases.tsv` 里那一行（`lm_head.weight` 的前 64 行，真实权重）。
comptime ROWS = 64
comptime COLS = 896
comptime N = ROWS * COLS
comptime N_BLOCKS = N // 32
comptime BLK = 34
comptime SCALE_BYTES = 2

comptime REL_TOL = Float64(1e-5)
# 相对 L2 的上界：q8_0 实测 0.54%，q4_0 是 10.3% —— 这条门的存在理由就是**把
# q4 挡在外面**：哪天有人把输出投影退回 q4_0，它会立刻红。
comptime REL_L2_MAX_SQ = Float64(1e-4)


def test_dequantised_values_are_bit_exact() raises:
    """解量化必须逐位等于参考，一个 ulp 都不许差。"""
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var expect_file = MappedFile(FIXTURE + "dequant.f32")
    var blocks = blocks_file.ptr().unsafe_bitcast[UInt8]()
    var want = expect_file.ptr().unsafe_bitcast[Float32]()

    var arena = Arena(N * 4 + 64)
    var got = arena.alloc(N * 4).unsafe_bitcast[Float32]()
    dequant_q8_0(got, blocks, N)

    for i in range(N):
        assert_true(
            got[unsafe_offset=i] == want[unsafe_offset=i],
            "第 " + String(i) + " 个值不是逐位相等",
        )
    blocks_file.keep_alive()
    expect_file.keep_alive()
    arena.keep_alive()


def test_quantised_blocks_match_the_reference_byte_for_byte() raises:
    """量化必须产出与参考**逐字节一样**的块流，不是"数值接近"。

    它是整条链路的起点，错在这里会被后面所有环节继承：缩放差一个 ulp，整块权重
    等比缩放，量化后的网络照样流利 —— 只是悄悄变了个人。所以不比值，比字节。
    """
    var orig_file = MappedFile(FIXTURE + "orig.f32")
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var orig = orig_file.ptr().unsafe_bitcast[Float32]()
    var want = blocks_file.ptr().unsafe_bitcast[UInt8]()

    var arena = Arena(N_BLOCKS * BLK + 64)
    var got = arena.alloc(N_BLOCKS * BLK).unsafe_bitcast[UInt8]()
    quantize_q8_0(got, orig, N)

    for i in range(N_BLOCKS * BLK):
        assert_true(
            got[unsafe_offset=i] == want[unsafe_offset=i],
            "第 " + String(i) + " 个字节与参考不同",
        )
    orig_file.keep_alive()
    blocks_file.keep_alive()
    arena.keep_alive()


def test_negative_missing_sign_extension_is_caught() raises:
    """负向对照：把有符号码当无符号读，零容差断言必须能看出来。

    q8_0 与 q4_0 在这里**不一样**：q4_0 存的是无符号 nibble（还原时减 8），q8_0
    存的是补码。照 q4 的习惯直接读字节，所有负权重会整体跳到 +128 那侧 —— 输出
    仍然是一堆大小合理的数，只是错的。这条钉住它。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var blocks = blocks_file.ptr().unsafe_bitcast[UInt8]()
    var arena = Arena(N * 4 + 64)
    var got = arena.alloc(N * 4).unsafe_bitcast[Float32]()
    dequant_q8_0(got, blocks, N)

    var differing = 0
    for b in range(N_BLOCKS):
        var base = b * BLK
        var d = block_scale(blocks, base)
        for j in range(32):
            var raw = Int(blocks[unsafe_offset=base + SCALE_BYTES + j])
            var unsigned = Float32(raw) * d  # 忘了补码 ⇒ 负权重变成大正数
            if unsigned != got[unsafe_offset=b * 32 + j]:
                differing += 1
    assert_true(
        differing > 0,
        "按无符号读与正确结果完全相同 —— 这条断言对符号扩展不敏感，等于没有",
    )
    blocks_file.keep_alive()
    arena.keep_alive()


def _quantise_truncated(value: Float32, d: Float32) -> Int:
    """`clamp(trunc(value / d), -128, 127)` —— 向零截断，不是取最近。

    截断会让每个值平均偏小半个台阶，于是量化出的字节必须整体不同。
    """
    var q = Int(value / d)
    if q < -128:
        return -128
    if q > 127:
        return 127
    return q


def _quantize_truncating(blocks: U8Ptr, src: F32Ptr, n: Int) raises:
    """与 `quantize_q8_0` 同布局，只把取整换成截断。"""
    var n_blocks = n // 32
    for b in range(n_blocks):
        var start = b * 32
        var amax = Float32(0)
        for j in range(32):
            var v = src[unsafe_offset=start + j]
            var a = v if v >= Float32(0) else -v
            if a > amax:
                amax = a
        var d = Float32(0)
        if amax != Float32(0):
            d = amax / Float32(127)
        var half = f32_to_fp16_bits(d)
        var base = b * BLK
        blocks[unsafe_offset=base] = UInt8(half & 0xFF)
        blocks[unsafe_offset=base + 1] = UInt8((half >> 8) & 0xFF)
        for j in range(32):
            var q = 0
            if d != Float32(0):
                q = _quantise_truncated(src[unsafe_offset=start + j], d)
            blocks[unsafe_offset=base + SCALE_BYTES + j] = UInt8(q & 0xFF)


def test_negative_truncating_round_is_rejected() raises:
    """负向对照：取整改成截断后，字节必须与参考不同。

    参照物（fixture）是同一次导出的产物：把实现退回截断，fixture **不会**自己
    变红 —— 只有独立的旧规则版本能盯住这件事。
    """
    var orig_file = MappedFile(FIXTURE + "orig.f32")
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var orig = orig_file.ptr().unsafe_bitcast[Float32]()
    var want = blocks_file.ptr().unsafe_bitcast[UInt8]()

    var arena = Arena(N_BLOCKS * BLK + 64)
    var got = arena.alloc(N_BLOCKS * BLK).unsafe_bitcast[UInt8]()
    _quantize_truncating(got, orig, N)

    var differing = 0
    for i in range(N_BLOCKS * BLK):
        if got[unsafe_offset=i] != want[unsafe_offset=i]:
            differing += 1
    assert_true(
        differing > 0,
        "截断取整与取最近产出了完全相同的字节 —— 逐字节比较对取整不敏感",
    )
    orig_file.keep_alive()
    blocks_file.keep_alive()
    arena.keep_alive()


def test_negative_wrong_block_stride_is_rejected() raises:
    """负向对照：块步长按 32 而不是 34 读，结果必须与期望不同。

    34 这个数里 2 个字节是缩放、32 个是码；想当然地按"32 个值 32 字节"去读，是
    最容易犯的错 —— 而且**第一块永远是对的**（偏移都是 0），只有这条能抓住它。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var blocks = blocks_file.ptr().unsafe_bitcast[UInt8]()

    # ⚠️ 这里**不**走 `block_scale` 去解出浮点值：错步长读到的两个字节可能根本不
    # 是合法 fp16（实测读到 half_bits=32740，逼近 inf），`fp16_bits_to_f32` 会直接
    # 抛"scale is infinity or NaN" —— 那是**另一种**证据，但它是崩溃不是判红。
    # 要的是"断言能看出来"，所以比的是**字节本身**：正确起点 `b*34` 与错起点
    # `b*32` 读到的那 34 个字节是否不同。
    var differing = 0
    # 只比前 64 块：错步长会把读指针一路往前推，64 块足够证明敏感性又不越界。
    for b in range(64):
        for k in range(BLK):
            var good = Int(blocks[unsafe_offset=b * BLK + k])
            var bad = Int(blocks[unsafe_offset=b * 32 + k])  # 错：应为 b * 34
            if good != bad:
                differing += 1
    assert_true(
        differing > 0,
        "按 32 字节步长读到的块与正确的完全相同 —— 这条断言对块边界不敏感",
    )
    blocks_file.keep_alive()


def test_quantisation_is_actually_lossy() raises:
    """解出的值**不等于**原始 fp32 —— 否则"逐位相等"可能只是压根没量化。"""
    var orig_file = MappedFile(FIXTURE + "orig.f32")
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var orig = orig_file.ptr().unsafe_bitcast[Float32]()
    var blocks = blocks_file.ptr().unsafe_bitcast[UInt8]()

    var arena = Arena(N * 4 + 64)
    var got = arena.alloc(N * 4).unsafe_bitcast[Float32]()
    dequant_q8_0(got, blocks, N)

    var differing = 0
    for i in range(N):
        if got[unsafe_offset=i] != orig[unsafe_offset=i]:
            differing += 1
    assert_true(differing > 0, "解量化结果与原始 fp32 完全相同 —— 没有真的量化")
    orig_file.keep_alive()
    blocks_file.keep_alive()
    arena.keep_alive()


def test_relative_l2_is_under_one_percent() raises:
    """相对 L2 必须 < 1%。

    这条门的作用是把 **q4_0 挡在输出投影外面**：q8_0 实测 0.54%，q4_0 是 10.3%
    （账本里"每块一个 scale 的自由度已经榨干"那一节）。哪天有人把 `lm_head` 退回
    q4_0 去省那 76 MB，这里立刻红。

    比平方是为了不开方（本机没有可信的 `sqrt` 依赖，平方比较等价）。
    """
    var orig_file = MappedFile(FIXTURE + "orig.f32")
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var orig = orig_file.ptr().unsafe_bitcast[Float32]()
    var blocks = blocks_file.ptr().unsafe_bitcast[UInt8]()

    var arena = Arena(N * 4 + 64)
    var got = arena.alloc(N * 4).unsafe_bitcast[Float32]()
    dequant_q8_0(got, blocks, N)

    var num = Float64(0)
    var den = Float64(0)
    for i in range(N):
        var diff = Float64(got[unsafe_offset=i]) - Float64(orig[unsafe_offset=i])
        num += diff * diff
        var w = Float64(orig[unsafe_offset=i])
        den += w * w
    assert_true(den > Float64(0), "原始权重全为零，这条用例没有意义")
    assert_true(
        num / den < REL_L2_MAX_SQ,
        "相对 L2 超过 1%（平方比 " + String(num / den) + "）",
    )
    orig_file.keep_alive()
    blocks_file.keep_alive()
    arena.keep_alive()


def _check_matmul_against_reference[vector: Bool]() raises:
    """融合 matmul 与 fp64 参考的比对，`vector` 选走哪条实现。

    期望值用**解量化后**的权重算 ⇒ 判据只盯**累加结构**：量化误差（0.54%）不在这
    条门的射程内，它在相对 L2 那条门里。
    """
    var blocks_file = MappedFile(FIXTURE + "blocks.bin")
    var act_file = MappedFile(FIXTURE + "act.f32")
    var expect_file = MappedFile(FIXTURE + "out_expected.f64")
    var blocks = blocks_file.ptr().unsafe_bitcast[UInt8]()
    var act = act_file.ptr().unsafe_bitcast[Float32]()
    var want = expect_file.ptr().unsafe_bitcast[Float64]()

    var arena = Arena(ROWS * 4 + 64)
    var got = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    comptime if vector:
        matmul_q8_f32_vec(got, act, blocks, ROWS, COLS)
    else:
        matmul_q8_f32(got, act, blocks, ROWS, COLS)

    var worst = Float64(0)
    for r in range(ROWS):
        var refv = want[unsafe_offset=r]
        var mag = refv if refv >= Float64(0) else -refv
        if mag < Float64(1):
            mag = Float64(1)
        var diff = Float64(got[unsafe_offset=r]) - refv
        if diff < Float64(0):
            diff = -diff
        if diff > worst:
            worst = diff
        assert_true(
            diff <= REL_TOL * mag,
            "第 " + String(r) + " 行偏离参考（差 " + String(diff) + "）",
        )
    print("观测：融合 matmul 与 fp64 参考的最大绝对差 = " + String(worst))
    blocks_file.keep_alive()
    act_file.keep_alive()
    expect_file.keep_alive()
    arena.keep_alive()


def test_matmul_matches_the_reference() raises:
    """标量版融合 matmul 必须等于"解量化后的权重 × 真实激活"（f64 累加）。"""
    _check_matmul_against_reference[False]()


def test_matmul_vector_backend_matches_the_reference() raises:
    """向量版融合 matmul 对**同一个** fp64 参考，判据同样是 1e-5。

    为什么单开一条而不是让上面那条也跑向量版：两条实现的累加结构**不同**（f32 通道
    + 行末归约 vs f64 一路累加），它们互相之间**不逐位相等** —— 那不是 bug，是换通
    路要付的精度，q4 那条向量通路也是同一个规矩。所以两边各自对着**同一个** fp64
    参考比，谁都不许越 1e-5。

    ⚠️ 这条存在的直接理由：输出投影接进模型之后，标量版让端到端**变慢**（q4_0 档
    `shards=8` 从 9.62…12.08 掉到 7.12…9.36 tok/s）—— 136M 次乘加压在标量上，省下的
    379 MB/token 流量全被吃了回去。向量版就是为了让那份流量真正兑现，而它必须先证
    明自己算得对。
    """
    _check_matmul_against_reference[True]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
