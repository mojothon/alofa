"""RU/RB 扫描：mub 带状核在 n_tok=8（批解码）与 n_tok=32（prefill）的 ilp 空间。

四十五的链条：服务批解码 109 ms/步 → in-process t=8 前向 ~90% 是核时间 →
批路 o/gate/up 走 RB=8 ⇒ `RU = MUB_ROW_ACC_TARGET(8) // RB(8) = 1`，每输出行
只有**一条**独立累加链。本基准不改产品码，用 `RU_OVERRIDE` 直接扫 RB/RU 组合。

（首轮已见：n_tok=8 时 RB=8/RU=1 比 RB=2/RU=4 慢 ~2.4×，权重流量相同 —— 纯 ILP。）
"""
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr
from alofa.kernels.cpu.avx2 import Q4_U8, _matmul_q4_mub_rows

alias BLOCK = 32
alias BB = 18
alias REPS = 30


def fill_scales(blocks: Q4_U8, nbytes: Int):
    var p = blocks.bitcast[UInt64]()
    var n8 = nbytes // 8
    var z = UInt64(0)
    for i in range(n8):
        p[i] = z
    var nb = nbytes // BB
    for b in range(nb):
        var off = b * BB
        blocks[unsafe_offset=off] = UInt8(0)
        blocks[unsafe_offset=off + 1] = UInt8(0x3C)


def bench_ru[
    RB: Int, RU: Int
](dst: F32Ptr, x: F32Ptr, w: Q4_U8, n_out: Int, inner: Int, n_tok: Int) raises -> Float64:
    _matmul_q4_mub_rows[RB, RU](dst, x, w, n_out, inner, dst, False, inner)
    var best = 1.0e18
    for _ in range(3):
        var t0 = monotonic_ns()
        for _ in range(REPS):
            _matmul_q4_mub_rows[RB, RU](dst, x, w, n_out, inner, dst, False, inner)
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(REPS)
        if us < best:
            best = us
    return best


def sweep[OUT: Int, INNER: Int](
    label: String, mut arena: Arena, n_tok: Int
) raises:
    var wbytes = OUT * INNER // BLOCK * BB
    var w = arena.alloc(wbytes)
    fill_scales(w, wbytes)
    var x = F32Ptr(arena.alloc(n_tok * INNER * 4).bitcast[Float32]())
    var dst = F32Ptr(arena.alloc(n_tok * OUT * 4).bitcast[Float32]())
    for i in range(n_tok * INNER):
        x[unsafe_offset=i] = Float32(i % 13) * 0.01

    var w_mb = Float64(wbytes) / 1048576.0
    print("==", label, " 权重", String(Int(w_mb * 10)), "(0.1MB)")
    # RB=2 与 RB=8；RU 组合受 RBV*RUV<=16 寄存器预算约束（与核内注释一致）
    var us_2_1 = bench_ru[2, 1](dst, x, w, OUT, INNER, n_tok)
    var us_2_2 = bench_ru[2, 2](dst, x, w, OUT, INNER, n_tok)
    var us_2_4 = bench_ru[2, 4](dst, x, w, OUT, INNER, n_tok)
    var us_2_8 = bench_ru[2, 8](dst, x, w, OUT, INNER, n_tok)
    var us_8_1 = bench_ru[8, 1](dst, x, w, OUT, INNER, n_tok)
    var us_8_2 = bench_ru[8, 2](dst, x, w, OUT, INNER, n_tok)
    print("  RB=2 RU=1:", String(Int(us_2_1 * 10)), "  RU=2:", String(Int(us_2_2 * 10)),
          "  RU=4:", String(Int(us_2_4 * 10)), "  RU=8:", String(Int(us_2_8 * 10)), " (0.1us)")
    print("  RB=8 RU=1:", String(Int(us_8_1 * 10)), "  RU=2:", String(Int(us_8_2 * 10)), " (0.1us)")
    print("  慢/快比 RB8RU1 / RB2RU4:", String(Int(Float64(us_8_1) / us_2_4 * 100)), "%")


def main() raises:
    var arena = Arena(16 << 20)
    sweep[4864, 896]("gate/up 形状 n_tok=8", arena, 8)
    sweep[896, 896]("o 形状 n_tok=8", arena, 8)
    sweep[4864, 896]("gate/up 形状 n_tok=32", arena, 32)
    sweep[896, 896]("o 形状 n_tok=32", arena, 32)
