"""分片并发的投影必须与不切片的那次**算出同一个值**。

为什么这道门值得单独存在
------------------------
把一次投影切成 N 片，看起来只是"把同一段循环分给几个人"。但切分引入了一类
**只在大分片数下才出现**的错误：片的起点算错、`rem`（除不尽的余数）没分给任何
一片、偏置跟着片走却忘了偏移、`q4_0` 的块偏移按元素算而不是按字节算。这些错误
在 `out` 能被 N 整除时**全部看不见** —— 而真实形状里 `out` 是 896、151936，能被
2/4/7/8 整除，于是"跑通了"和"对了"是两件事。

所以这里的形状表专门挑**除不尽**的：`out` 取 1 / 3 / 5 / 7 / 11 / 17 / 28，
片数取 2 / 3 / 5 / 8 —— 余数必须真的被某一片算掉。

判据为什么是"同一个值"而不是"差一点点"
--------------------------------------
切分**不改变任何一次浮点运算**：每个输出元素仍然由它自己那一行与 `x` 完整归约
而来，累加顺序一模一样（`fp32` 通路是每条输出一个 f64 累加器，`q4_0` 通路是每条
输出一个 f32 累加器，都在片内闭合）。所以正确的实现给出的是**同一个浮点值**，
不是"差 1e-7"。拿容差去比，等于把"片起点算错"这种错误也放过去 —— 它就藏在
"差一点点"里。

⚠️ 负向对照（第 7、8 项）是这道门的另一半：一条**故意写错**的切分必须被同一套
比较抓出来。门会绿，不代表门在守东西。
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.tensor import TensorView, f32_data, shape2
from alofa.kernels.cpu.quant import Q4_BLOCK, Q4_BYTES, quantize_q4_0
from alofa.model.arch.qwen import (
    BACKEND_AVX2,
    MAX_SHARDS,
    PREFILL_SHARDS_MEASURED,
    linear_bias_k_shards,
    linear_k_shards,
    prefill_shards,
    q4_matmul_bias_k_shards,
    q4_matmul_k_shards,
    shard_count,
)

comptime B = BACKEND_AVX2

# 除不尽的那一批：`out` 取这些值时，2/3/5/8 片没有一片能整除它。
comptime OUTS = 17


def fill(a: TensorView, seed: Int) raises:
    var p = f32_data(a)
    for i in range(a.shape.numel()):
        p[unsafe_offset=i] = Float32((i * 31 + seed * 7) % 17) - Float32(8)


def all_equal(a: TensorView, b: TensorView) raises -> Bool:
    """两个视图逐元素**完全**相等（不是"差一点点"）。"""
    var pa = f32_data(a)
    var pb = f32_data(b)
    for i in range(a.shape.numel()):
        if pa[unsafe_offset=i] != pb[unsafe_offset=i]:
            return False
    return True


# --------------------------------------------------------------------------
# 片数本身
# --------------------------------------------------------------------------


def test_shard_count_clamps() raises:
    """片数不许超过输出个数，也不许为 0；`<= 1` 就是不切。"""
    assert_equal(shard_count(1, 8), 1)
    assert_equal(shard_count(896, 8), 8)
    # 输出只有 5 个却要 9 片：多出来的 4 片是空的，而空片也要付一次调度。
    assert_equal(shard_count(5, 9), 5)
    assert_equal(shard_count(100, 0), 1)
    assert_equal(shard_count(100, 1), 1)
    assert_equal(shard_count(100000, 64), MAX_SHARDS)


# --------------------------------------------------------------------------
# fp32 通路：decode（批 = 1，切输出列）
# --------------------------------------------------------------------------


def test_fp32_shards_match_single_pass() raises:
    var inner = 96
    var outs = List[Int]()
    for v in [1, 3, 5, 7, 11, 17, 28]:
        outs.append(v)
    var shardss = List[Int]()
    for v in [2, 3, 5, 8]:
        shardss.append(v)
    for oi in range(len(outs)):
        var out = outs[oi]
        var arena = Arena(out * inner * 4 + inner * 4 + out * 4 * 2 + 4096)
        var w = TensorView(arena.alloc(out * inner * 4), shape2(out, inner), DT_FP32)
        var x = TensorView(arena.alloc(inner * 4), shape2(1, inner), DT_FP32)
        var d1 = TensorView(arena.alloc(out * 4), shape2(1, out), DT_FP32)
        var d2 = TensorView(arena.alloc(out * 4), shape2(1, out), DT_FP32)
        fill(w, 1)
        fill(x, 2)
        linear_k_shards[B](d1, x, w, 1)
        for si in range(len(shardss)):
            linear_k_shards[B](d2, x, w, shardss[si])
            assert_true(
                all_equal(d1, d2),
                "out=" + String(out) + " shards=" + String(shardss[si]),
            )
        arena.keep_alive()


def test_fp32_bias_shards_match_single_pass() raises:
    """偏置必须跟着输出列走：切列时每片只取自己那几列的偏置。"""
    var inner = 96
    var out = OUTS
    var arena = Arena(out * inner * 4 + inner * 4 + out * 4 * 3 + 4096)
    var w = TensorView(arena.alloc(out * inner * 4), shape2(out, inner), DT_FP32)
    var x = TensorView(arena.alloc(inner * 4), shape2(1, inner), DT_FP32)
    var b = TensorView(arena.alloc(out * 4), shape2(1, out), DT_FP32)
    var d1 = TensorView(arena.alloc(out * 4), shape2(1, out), DT_FP32)
    var d2 = TensorView(arena.alloc(out * 4), shape2(1, out), DT_FP32)
    fill(w, 3)
    fill(x, 4)
    fill(b, 5)
    linear_bias_k_shards[B](d1, x, w, b, 1)
    for shards in [2, 3, 5, 8]:
        linear_bias_k_shards[B](d2, x, w, b, shards)
        assert_true(all_equal(d1, d2), "shards=" + String(shards))
    arena.keep_alive()


# --------------------------------------------------------------------------
# fp32 通路：prefill（批 > 1，改切输出行）
# --------------------------------------------------------------------------


def test_fp32_prefill_shards_match_single_pass() raises:
    """批大于 1 时 `dst` 是行主序，列不连续 —— 这时必须改切**行**。

    `rows` 取 1/2/3/5/7/8/9/13/32/33：① 除不尽的那几档（余数那行必须被算到）；
    ② 8/9/13/32/33 是 `avx2._gemm_tile[RB]` 的分块边界（`RB` 是 8/4/2/1，`rows`
    跨过 8 时块怎么切都会变）；③ **33 专门跨 `PREFILL_ROWS_MEASURED`** —— 那一边
    片数不再被压到实测档，是另一条支路。
    """
    var out = 11
    var inner = 96
    var rows_list = List[Int]()
    for v in [1, 2, 3, 5, 7, 8, 9, 13, 32, 33]:
        rows_list.append(v)
    for ri in range(len(rows_list)):
        var rows = rows_list[ri]
        var arena = Arena(
            out * inner * 4 + rows * inner * 4 + rows * out * 4 * 2 + 4096
        )
        var w = TensorView(
            arena.alloc(out * inner * 4), shape2(out, inner), DT_FP32
        )
        var x = TensorView(
            arena.alloc(rows * inner * 4), shape2(rows, inner), DT_FP32
        )
        var d1 = TensorView(
            arena.alloc(rows * out * 4), shape2(rows, out), DT_FP32
        )
        var d2 = TensorView(
            arena.alloc(rows * out * 4), shape2(rows, out), DT_FP32
        )
        fill(w, 6)
        fill(x, 7)
        linear_k_shards[B](d1, x, w, 1)
        for shards in [2, 3, 5, 8]:
            linear_k_shards[B](d2, x, w, shards)
            assert_true(
                all_equal(d1, d2),
                "rows=" + String(rows) + " shards=" + String(shards),
            )
        arena.keep_alive()


def test_prefill_shards_prefers_the_measured_tier() raises:
    """prefill 的片数上限：**只启用量过的那一档**。

    `prefill_shards()` 现在会决定一次 prefill 用几片，而它只改**并行度**、不改
    任何一次浮点运算 —— 所以这条守的是"它把片数压到哪儿"，不是"它对不对"
    （对不对由上一条与端到端门守）。

    规矩与 `SHARDS_MEASURED` 是同一条：**量过的范围内用实测最好的 4 片**，
    **`rows` 超出量过的上界就原样返回** —— 那里没量过，改动前的样子最不坏。
    """
    # 量过的范围内：压到 4。
    assert_equal(prefill_shards(8, 8), PREFILL_SHARDS_MEASURED)
    assert_equal(prefill_shards(16, 8), PREFILL_SHARDS_MEASURED)
    assert_equal(prefill_shards(32, 8), PREFILL_SHARDS_MEASURED)
    # 调用方本来就要得更少 → 尊重调用方（它可能是显式关掉并发的）。
    assert_equal(prefill_shards(8, 2), 2)
    assert_equal(prefill_shards(8, 1), 1)
    # 超出量过的上界 → 不假装量过，原样返回。
    assert_equal(prefill_shards(33, 8), 8)
    assert_equal(prefill_shards(128, 8), 8)


# --------------------------------------------------------------------------
# q4_0 通路：块偏移是**字节**，不是元素
# --------------------------------------------------------------------------


def test_q4_shards_match_single_pass() raises:
    var rows_list = List[Int]()
    for v in [1, 3, 5, 7, 11, 17, 28]:
        rows_list.append(v)
    for ri in range(len(rows_list)):
        var rows = rows_list[ri]
        var cols = 96
        var n = rows * cols
        var qb = n // Q4_BLOCK * Q4_BYTES
        var arena = Arena(n * 4 + cols * 4 + rows * 4 * 2 + qb + 4096)
        var w = TensorView(arena.alloc(n * 4), shape2(rows, cols), DT_FP32)
        var x = TensorView(arena.alloc(cols * 4), shape2(1, cols), DT_FP32)
        var blocks = arena.alloc(qb)
        var d1 = TensorView(arena.alloc(rows * 4), shape2(1, rows), DT_FP32)
        var d2 = TensorView(arena.alloc(rows * 4), shape2(1, rows), DT_FP32)
        fill(w, 8)
        fill(x, 9)
        quantize_q4_0(blocks, f32_data(w), n)
        q4_matmul_k_shards[B](f32_data(d1), f32_data(x), blocks, rows, cols, 1)
        for shards in [2, 3, 5, 8]:
            q4_matmul_k_shards[B](f32_data(d2), f32_data(x), blocks, rows, cols, shards)
            assert_true(
                all_equal(d1, d2),
                "rows=" + String(rows) + " shards=" + String(shards),
            )
        arena.keep_alive()


def test_q4_bias_shards_match_single_pass() raises:
    var rows = OUTS
    var cols = 96
    var n = rows * cols
    var qb = n // Q4_BLOCK * Q4_BYTES
    var arena = Arena(n * 4 + cols * 4 + rows * 4 * 3 + qb + 4096)
    var w = TensorView(arena.alloc(n * 4), shape2(rows, cols), DT_FP32)
    var x = TensorView(arena.alloc(cols * 4), shape2(1, cols), DT_FP32)
    var b = TensorView(arena.alloc(rows * 4), shape2(1, rows), DT_FP32)
    var blocks = arena.alloc(qb)
    var d1 = TensorView(arena.alloc(rows * 4), shape2(1, rows), DT_FP32)
    var d2 = TensorView(arena.alloc(rows * 4), shape2(1, rows), DT_FP32)
    fill(w, 10)
    fill(x, 11)
    fill(b, 12)
    quantize_q4_0(blocks, f32_data(w), n)
    q4_matmul_bias_k_shards[B](
        f32_data(d1), f32_data(x), blocks, rows, cols, f32_data(b), 1
    )
    for shards in [2, 3, 5, 8]:
        q4_matmul_bias_k_shards[B](
            f32_data(d2), f32_data(x), blocks, rows, cols, f32_data(b), shards
        )
        assert_true(all_equal(d1, d2), "shards=" + String(shards))
    arena.keep_alive()


# --------------------------------------------------------------------------
# 负向对照：门会绿，不代表门在守东西
# --------------------------------------------------------------------------


def test_broken_split_is_caught() raises:
    """一条**故意少算一片**的切分必须被同一套比较抓出来。

    它守的是"门真的在比东西"：如果 `all_equal` 写错成永远返回 `True`，或者形状
    表退化成"除得尽"，这条会先红。
    """
    var inner = 96
    var out = OUTS
    var shards = 5
    var arena = Arena(out * inner * 4 + inner * 4 + out * 4 * 2 + 4096)
    var w = TensorView(arena.alloc(out * inner * 4), shape2(out, inner), DT_FP32)
    var x = TensorView(arena.alloc(inner * 4), shape2(1, inner), DT_FP32)
    var d1 = TensorView(arena.alloc(out * 4), shape2(1, out), DT_FP32)
    var d2 = TensorView(arena.alloc(out * 4), shape2(1, out), DT_FP32)
    fill(w, 13)
    fill(x, 14)
    linear_k_shards[B](d1, x, w, 1)
    linear_k_shards[B](d2, x, w, shards)
    assert_true(all_equal(d1, d2))

    # ⚠️ 先把 d2 涂成哨兵值。不做这一步，漏掉的那片会**保留上一次全量算出的正确
    # 值**，于是"少算一片"和"全算"逐位相同 —— 负向对照自己就是绿的，等于没守。
    # （2026-09-20 实测：漏了这步，这条一直红不起来。）
    var pd = f32_data(d2)
    for i in range(out):
        pd[unsafe_offset=i] = Float32(-999)

    # 故意只算前 shards-1 片：最后一片（它才含除不尽的余数）永远没人算。
    var per = out // shards
    for k in range(shards - 1):
        var c0 = k * per
        var wv = TensorView(
            w.data.unsafe_offset(w.byte_offset + c0 * inner * 4),
            shape2(per, inner),
            DT_FP32,
        )
        var dv = TensorView(
            d2.data.unsafe_offset(d2.byte_offset + c0 * 4), shape2(1, per), DT_FP32
        )
        linear_k_shards[B](dv, x, wv, 1)
    assert_true(
        not all_equal(d1, d2), "a shard that is never computed must be noticed"
    )
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
