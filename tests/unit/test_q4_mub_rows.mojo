"""`vpmaddubsw` 行带通路的核级差分门。

为什么要有这个门，而不是只留整网那条一致率
------------------------------------------
整网一致率（`tests/unit/test_q4_mub_greedy.mojo`）是个**整数计数**，它回答"质量掉没掉"，
但一旦红了，不知道是核写错了、还是开关没接到、还是量化误差真的变大了。这个门把前
两种死法单独钉住，而且**不要 2 GB 权重**，所以能进 `pixi run test` 天天跑。

它守两件事，缺一件这个门就等于没有：

① **够近**：`vpmaddubsw` 行带与默认 f32 行带的相对 L2 ≤ 1%。实测 0.33–0.38%，留
   二十倍余量是为了让"核坏了"和"数值抖动"分开 —— 相对 L2 涨到 1% 意味着激活量化那一层
   出了结构性问题（比如退回整批共用一个步长，那是 0.75%，就在这条线附近）。

② **确实在算**：两条通路的**逐位**结果必须**不同**。这一条是 2026-09-29 那次误判的
   常驻红测：开关挂在 `q4_matmul_k` 上而模型走的是行带入口，mub 一次都没执行，于是
   "无增益"和"一致率没变"两个假象同时出现 —— 而"逐位完全相同"正是"开关没生效"长什么样。
   把它写成断言，下次就不用靠 objdump 和 env 追踪去猜。

③ **守卫区**：`dst` 按元素偏。2026-09-22 真发生过把它写成 RawPtr 的字节习惯（多乘
   一个 4）的 bug，长相是"前几个 token 对、后面的写到别人地盘上"。门里把那个写法
   **原样复原**一遍并断言守卫区会被改动（负向对照：一个抓不住这个 bug 的守卫等于没守卫）。

形状故意**除不尽**：`cols` 取 3 / 5 个块（96 / 160），`n_tok` 取 1 / 2 / 5 / 8，
逼 `RB=2` 的整带与余下的单行后备路径都走到。
"""

from std.math import sqrt
from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr
from alofa.kernels.cpu.avx2 import (
    matmul_q4_f32_bias_rows_band,
    matmul_q4_f32_rows_band,
    matmul_q4_mub_bias_rows_band,
    matmul_q4_mub_rows_band,
)
from alofa.kernels.cpu.quant import Q4_BLOCK, Q4_BYTES, quantize_q4_0

comptime ROWS = 12
comptime GUARD = 16
comptime MUB_REL_L2_GATE = Float64(0.01)


struct CaseResult:
    """一次对照的结果：`rel` 是相对 L2，`diffs` 是逐位不同的元素数。"""

    var rel: Float64
    var diffs: Int

    def __init__(out self):
        self.rel = Float64(0)
        self.diffs = 0


def u32_seed_next(seed: UInt64) -> UInt64:
    return seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)


def fill_activation(x: F32Ptr, n_tok: Int, cols: Int, skew: Bool):
    """同一批里各行幅度差 20 倍（`skew`）：专治"整批共用一个激活步长"那类退化。"""
    var seed = UInt64(20260929)
    for t in range(n_tok):
        var scale = Float32(1)
        if skew:
            scale = Float32(1) / Float32(1 + t * 9)
        for j in range(cols):
            seed = u32_seed_next(seed)
            x[unsafe_offset=t * cols + j] = (
                Float32(Int((seed >> 40) & UInt64(0xFFFF))) / Float32(32768.0)
                - Float32(1.0)
            ) * scale


def rel_l2(a: F32Ptr, b: F32Ptr, n: Int) -> Float64:
    var num = Float64(0)
    var den = Float64(0)
    for i in range(n):
        var d = Float64(a[unsafe_offset=i]) - Float64(b[unsafe_offset=i])
        num += d * d
        den += Float64(b[unsafe_offset=i]) * Float64(b[unsafe_offset=i])
    if den == Float64(0):
        return Float64(0)
    return sqrt(num / den)


def bitwise_diffs(a: F32Ptr, b: F32Ptr, n: Int) -> Int:
    var bad = 0
    for i in range(n):
        if a[unsafe_offset=i] != b[unsafe_offset=i]:
            bad += 1
    return bad


def compare_case(
    n_tok: Int, cols: Int, use_bias: Bool, skew: Bool, mut r: CaseResult
) raises:
    """同一份权重与激活，两条通路各跑一遍，比相对 L2 与逐位差。

    顺带查守卫区：`dst` 后面多留 `GUARD` 个元素填哨兵，正常走法**不许**碰它们。
    """
    var nb = cols // Q4_BLOCK
    var nbytes = (
        ROWS * cols * 4
        + ROWS * nb * Q4_BYTES
        + n_tok * cols * 4
        + 2 * (n_tok * ROWS + GUARD) * 4
        + ROWS * 4
        + 65536
    )
    var arena = Arena(nbytes)
    var w = arena.alloc(ROWS * cols * 4).unsafe_bitcast[Float32]()
    var blocks = arena.alloc(ROWS * nb * Q4_BYTES).unsafe_bitcast[UInt8]()
    var x = arena.alloc(n_tok * cols * 4).unsafe_bitcast[Float32]()
    var d0 = arena.alloc((n_tok * ROWS + GUARD) * 4).unsafe_bitcast[Float32]()
    var d1 = arena.alloc((n_tok * ROWS + GUARD) * 4).unsafe_bitcast[Float32]()
    var bias = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()

    var seed = UInt64(7654321)
    for i in range(ROWS * cols):
        seed = u32_seed_next(seed)
        w[unsafe_offset=i] = Float32(Int((seed >> 40) & UInt64(0xFFFF))) / Float32(
            32768.0
        ) - Float32(1.0)
    for i in range(ROWS):
        seed = u32_seed_next(seed)
        bias[unsafe_offset=i] = Float32(Int((seed >> 40) & UInt64(0xFFFF))) / Float32(
            32768.0
        )
    fill_activation(x, n_tok, cols, skew)
    quantize_q4_0(blocks, w, ROWS * cols)

    for i in range(n_tok * ROWS, n_tok * ROWS + GUARD):
        d0[unsafe_offset=i] = Float32(-7.5)
        d1[unsafe_offset=i] = Float32(-7.5)

    if use_bias:
        matmul_q4_f32_bias_rows_band(d0, x, blocks, n_tok, ROWS, cols, bias, ROWS)
        matmul_q4_mub_bias_rows_band(d1, x, blocks, n_tok, ROWS, cols, bias, ROWS)
    else:
        matmul_q4_f32_rows_band(d0, x, blocks, n_tok, ROWS, cols, ROWS)
        matmul_q4_mub_rows_band(d1, x, blocks, n_tok, ROWS, cols, ROWS)

    r.rel = rel_l2(d1, d0, n_tok * ROWS)
    r.diffs = bitwise_diffs(d1, d0, n_tok * ROWS)
    # 守卫区：正常走法不许写到 `n_tok * ROWS` 之后
    var touched = bitwise_diffs(d1, d0, n_tok * ROWS + GUARD) - r.diffs
    assert_equal(touched, 0, "行带写到了 dst 的守卫区（dst 按元素偏，不是按字节）")
    arena.keep_alive()


def test_mub_band_tracks_the_f32_band() raises:
    """① 够近：相对 L2 ≤ 1%，且各形状都不许退化。"""
    var toks = InlineArray[Int, 4](fill=0)
    toks[0] = 1
    toks[1] = 2
    toks[2] = 5
    toks[3] = 8
    var colss = InlineArray[Int, 2](fill=0)
    colss[0] = 96
    colss[1] = 160
    for ti in range(4):
        for ci in range(2):
            var r = CaseResult()
            compare_case(toks[ti], colss[ci], False, False, r)
            assert_true(
                r.rel <= MUB_REL_L2_GATE,
                "mub 行带与 f32 行带的相对 L2 超了：n_tok="
                + String(toks[ti])
                + " cols="
                + String(colss[ci])
                + " rel="
                + String(r.rel),
            )


def test_mub_band_is_really_a_different_path() raises:
    """② 确实在算：两条通路**不许**逐位相同。

    `diffs == 0` 是"开关没接到这条路上"长什么样（2026-09-29 那次：开关挂在
    `q4_matmul_k`，而模型走行带入口，于是两次测量的"无增益"和"一致率没变"都是假象）。
    """
    var toks = InlineArray[Int, 4](fill=0)
    toks[0] = 1
    toks[1] = 2
    toks[2] = 5
    toks[3] = 8
    for ti in range(4):
        var r = CaseResult()
        compare_case(toks[ti], 96, False, False, r)
        assert_true(
            r.diffs > 0,
            "mub 行带与 f32 行带逐位相同：这条通路根本没在算（n_tok="
            + String(toks[ti])
            + "）",
        )


def test_skewed_rows_stay_within_the_gate() raises:
    """各行幅度差 20 倍时不许退化 —— 整批共用一个步长那版在这里是 1.00%。"""
    var r = CaseResult()
    compare_case(5, 160, False, True, r)
    assert_true(
        r.rel <= MUB_REL_L2_GATE,
        "各行幅度差 20 倍时相对 L2 超了（像是退回整批共用一个步长）：rel="
        + String(r.rel),
    )


def test_bias_band_tracks_the_f32_bias_band() raises:
    """带偏置那条（q/k/v）：同样是"够近 + 确实在算"。"""
    var r = CaseResult()
    compare_case(5, 96, True, False, r)
    assert_true(
        r.rel <= MUB_REL_L2_GATE,
        "带偏置的 mub 行带相对 L2 超了：rel=" + String(r.rel),
    )
    assert_true(r.diffs > 0, "带偏置的 mub 行带与 f32 行带逐位相同：没在算")


def test_a_bad_destination_stride_trips_the_guard() raises:
    """③ 的负向对照：多乘一个 4 的旧写法**必须**被守卫区抓住。

    一个抓不住它的守卫等于没守卫 —— 所以这条红测是守卫本身的存在证明。
    """
    comptime N_TOK = 3
    comptime COLS = 96
    var nb = COLS // Q4_BLOCK
    # 旧写法会写到 `N_TOK * ROWS` 之外，所以缓冲得按**错误步长**开，别按正确步长开
    var bad_stride = ROWS * 4
    var nfloats = N_TOK * bad_stride + GUARD * 4
    var arena = Arena(nfloats * 4 + 65536)
    var w = arena.alloc(ROWS * COLS * 4).unsafe_bitcast[Float32]()
    var blocks = arena.alloc(ROWS * nb * Q4_BYTES).unsafe_bitcast[UInt8]()
    var x = arena.alloc(N_TOK * COLS * 4).unsafe_bitcast[Float32]()
    var d = arena.alloc(nfloats * 4).unsafe_bitcast[Float32]()
    var seed = UInt64(13572468)
    for i in range(ROWS * COLS):
        seed = u32_seed_next(seed)
        w[unsafe_offset=i] = Float32(Int((seed >> 40) & UInt64(0xFFFF))) / Float32(
            32768.0
        ) - Float32(1.0)
    fill_activation(x, N_TOK, COLS, False)
    quantize_q4_0(blocks, w, ROWS * COLS)
    for i in range(N_TOK * ROWS, N_TOK * ROWS + GUARD):
        d[unsafe_offset=i] = Float32(-7.5)

    # 旧 bug 原样复原：`dst` 按元素走，被当成 RawPtr 按字节（多乘一个 4）
    for t in range(N_TOK):
        matmul_q4_mub_rows_band(
            d.unsafe_offset(t * bad_stride),
            x.unsafe_offset(t * COLS),
            blocks,
            1,
            ROWS,
            COLS,
            ROWS,
        )
    var touched = 0
    for i in range(N_TOK * ROWS, N_TOK * ROWS + GUARD):
        if d[unsafe_offset=i] != Float32(-7.5):
            touched += 1
    assert_true(
        touched > 0,
        "守卫区没抓住多乘 4 的步长：这个守卫抓不住它本该抓的 bug",
    )
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
