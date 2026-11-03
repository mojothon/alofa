"""q8 行带核（`matmul_q8_f32_rows_band`）的核级差分门 —— 批路输出投影专用。

被测的这一条：**一个 q8 块只 load/解量化一次、`RB` 个 token 行共用**（账本二十七：
批路此前把头交给 fp32 通路，一步读 544 MB；行带让 165 MB 的 q8 头在批宽内只读
一遍）。它换访存量，**不许换一个比特** —— 每 token 行的累加结构（四条 f32 向量
按块序、行末一次归约进 f64）与单行版 `matmul_q8_f32_vec` 一字不差。

判据：

① **逐位等于单行通路**（`avx2.matmul_q8_f32_vec` 逐 token 调）：零容差。
② **哨兵 + 守卫区**：批路 logits 行是隔开的（`dst_stride > rows`），最容易出的错
   就是"某行没写"或"写穿到下一行"。门里两种都布防。
③ **负向对照**：改掉结果里一个元素 → ①必须红；漏写余数 token 行 → 哨兵必须红。

⚠️ 形状故意除不尽：`n_tok` 取 `RB` 整数倍 ±1（含 1、`RB-1`），逼余数后备路径出场；
`cols` 取 3 个块（96，不是 32 的整十倍）。
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.dtype import DT_FP32
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import Q8_BLOCK, matmul_q8_f32_rows_band
from alofa.kernels.cpu.avx2 import matmul_q8_f32_vec
from alofa.kernels.cpu.quant import quantize_q8_0

comptime GUARD = 64
comptime SENTINEL = Float32(-999)
comptime ROWS = 96
comptime COLS = Q8_BLOCK * 3
comptime TOK = 19  # = 2×8 + 3：整带、余带、单行余数三条路都走到


def ptr_of(raw: RawPtr, n: Int) raises -> F32Ptr:
    return f32_data(TensorView(raw, shape2(n, 1), DT_FP32))


def paint(p: F32Ptr, n: Int) raises -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = SENTINEL
        i += 1


def seed_w(p: F32Ptr, n: Int) raises -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = Float32(((i * 37) % 101) - 50) * Float32(0.25)
        i += 1


def seed_x(p: F32Ptr, n: Int) raises -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = Float32(((i * 17) % 53) - 26) * Float32(0.125)
        i += 1


def bitwise_diffs(a: F32Ptr, b: F32Ptr, n: Int) raises -> Int:
    var bad = 0
    var i = 0
    while i < n:
        if a[unsafe_offset=i] != b[unsafe_offset=i]:
            bad += 1
        i += 1
    return bad


def test_band_matches_per_token_bit_for_bit() raises:
    """行带输出必须逐位等于逐 token 的单行核 —— 一个 ulp 都不许差。"""
    var arena = Arena(1 << 24)
    var w_bytes = ROWS * COLS // 32 * 34
    var w_f32 = ptr_of(arena.alloc(ROWS * COLS * 4), ROWS * COLS)
    seed_w(w_f32, ROWS * COLS)
    var blocks = arena.alloc(w_bytes).unsafe_bitcast[UInt8]()
    quantize_q8_0(blocks, w_f32, ROWS * COLS)

    var x = ptr_of(arena.alloc(TOK * COLS * 4), TOK * COLS)
    seed_x(x, TOK * COLS)

    # 单行参照：逐 token 调 vec 核，写进紧凑的 n_tok×rows。
    var ref_buf = arena.alloc(TOK * ROWS * 4)
    var ref_p = ptr_of(ref_buf, TOK * ROWS)
    paint(ref_p, TOK * ROWS)
    var t = 0
    while t < TOK:
        matmul_q8_f32_vec(
            ref_p.unsafe_offset(t * ROWS),
            x.unsafe_offset(t * COLS),
            blocks,
            ROWS,
            COLS,
        )
        t += 1

    # 行带：隔开的行（dst_stride = ROWS + 7），前后哨兵 + 守卫区。
    comptime STRIDE = ROWS + 7
    var span = TOK * STRIDE + GUARD
    var got_raw = arena.alloc(span * 4)
    var got = ptr_of(got_raw, span)
    paint(got, span)
    matmul_q8_f32_rows_band[8](got, x, blocks, TOK, ROWS, COLS, STRIDE)

    t = 0
    while t < TOK:
        var bad = bitwise_diffs(
            got.unsafe_offset(t * STRIDE),
            ref_p.unsafe_offset(t * ROWS),
            ROWS,
        )
        assert_equal(bad, 0, "第 " + String(t) + " 个 token 行与单行核不同")
        t += 1
    # 行间的空隙与守卫区必须原封不动 —— 写穿就是错了别人的份。
    var i = 0
    while i < TOK:
        var g = 0
        while g < 7:
            assert_equal(
                got[unsafe_offset=i * STRIDE + ROWS + g],
                SENTINEL,
                "第 " + String(i) + " 行被写穿了",
            )
            g += 1
        i += 1
    i = TOK * STRIDE
    while i < span:
        assert_equal(got[unsafe_offset=i], SENTINEL, "守卫区被写穿")
        i += 1
    # Arena 是 last-use 语义：不调 keep_alive，映射在最后一次 alloc 后就没了。
    arena.keep_alive()


def test_a_corrupted_result_is_caught() raises:
    """负向对照：改掉一个输出元素，逐位比较必须红 —— 不会失败的门等于没门。"""
    var arena = Arena(1 << 24)
    var w_bytes = ROWS * COLS // 32 * 34
    var w_f32 = ptr_of(arena.alloc(ROWS * COLS * 4), ROWS * COLS)
    seed_w(w_f32, ROWS * COLS)
    var blocks = arena.alloc(w_bytes).unsafe_bitcast[UInt8]()
    quantize_q8_0(blocks, w_f32, ROWS * COLS)
    var x = ptr_of(arena.alloc(8 * COLS * 4), 8 * COLS)
    seed_x(x, 8 * COLS)

    var ref_p = ptr_of(arena.alloc(8 * ROWS * 4), 8 * ROWS)
    paint(ref_p, 8 * ROWS)
    var t = 0
    while t < 8:
        matmul_q8_f32_vec(
            ref_p.unsafe_offset(t * ROWS),
            x.unsafe_offset(t * COLS),
            blocks,
            ROWS,
            COLS,
        )
        t += 1
    var got = ptr_of(arena.alloc(8 * ROWS * 4), 8 * ROWS)
    paint(got, 8 * ROWS)
    matmul_q8_f32_rows_band[8](got, x, blocks, 8, ROWS, COLS, ROWS)
    assert_equal(bitwise_diffs(got, ref_p, 8 * ROWS), 0, "基线必须逐位一致")
    got[unsafe_offset=3] = Float32(12345.0)
    assert_true(bitwise_diffs(got, ref_p, 8 * ROWS) == 1, "篡改必须被逐位比较抓到")
    arena.keep_alive()


def test_missing_remainder_rows_are_caught() raises:
    """负向对照：只写整带那几行、漏掉余数行 —— 哨兵检查必须红。

    防的是"余数后备路径没接"这种错：n_tok 不是 RB 整数倍时，余下的行若没人写，
    它们就该还是哨兵 —— 这里把哨兵当探测器用。
    """
    var arena = Arena(1 << 24)
    var w_bytes = ROWS * COLS // 32 * 34
    var w_f32 = ptr_of(arena.alloc(ROWS * COLS * 4), ROWS * COLS)
    seed_w(w_f32, ROWS * COLS)
    var blocks = arena.alloc(w_bytes).unsafe_bitcast[UInt8]()
    quantize_q8_0(blocks, w_f32, ROWS * COLS)
    comptime NTOK = 8 + 3
    var x = ptr_of(arena.alloc(NTOK * COLS * 4), NTOK * COLS)
    seed_x(x, NTOK * COLS)
    var got = ptr_of(arena.alloc(NTOK * ROWS * 4), NTOK * ROWS)
    paint(got, NTOK * ROWS)
    matmul_q8_f32_rows_band[8](got, x, blocks, NTOK, ROWS, COLS, ROWS)
    # 全部行都必须被写（包括余数那 3 行）—— 有哨兵活着就是漏写。
    var i = 0
    var live = 0
    while i < NTOK * ROWS:
        if got[unsafe_offset=i] != SENTINEL:
            live += 1
        i += 1
    assert_equal(live, NTOK * ROWS, "有行没被写 —— 余数后备路径没接上")
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
