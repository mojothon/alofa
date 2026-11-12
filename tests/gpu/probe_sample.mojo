"""账本一百二十三诊断：sample_rb 分布歪。①GPU 前 8 seed 的
token；②host 精确模拟同 seed 的 argmax(logit+T·G)——若一致则
LCG 质量问题，若不一致则核内算术 bug；③LCG u 序列均匀性统计。"""

from std.math import exp, log
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def lcg_next(s: Int) -> Int:
    return (s * 1664525 + 1013904223) & 0xFFFFFFFF


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)
    var toks = List[Int]()
    toks.append(100)
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    a.ctx.synchronize()

    # GPU 采样：seeds 1000..1007
    print("GPU tokens:")
    for n in range(8):
        _ = a.prefill_step[
            INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](toks, False, False, False, 1.0, Int32(1000 + n))
        a.ctx.synchronize()
        print("  seed", 1000 + n, "->", Int(a.h_tokout[0]))

    # host 精确模拟 seed=1000：元素 j 由 tid=j%256 处理，第 k=j//256
    # 次迭代，state_k = LCG^k(seed ^ (j%256)*40503)
    var seed = 1000
    var best = Float32(-1e30)
    var bi = -1
    for j in range(151936):
        var tid = j % 256
        var k = j // 256
        var s = (seed ^ (tid * 40503)) & 0xFFFFFFFF
        for _r in range(k):
            s = lcg_next(s)
        var ux = (s >> 8) & 16777215
        var u = (Float32(ux) + 0.5) / 16777216.0
        var g = -log(-log(u))
        var v = a.h_logits[j] + g
        if v > best:
            best = v
            bi = j
    print("host sim seed=1000 ->", bi)

    # LCG 均匀性：tid=0 序列前 100000 个 u 的统计
    var s2 = (1000 ^ 0) & 0xFFFFFFFF
    var cnt_hi = 0
    var cnt_lo = 0
    var su = 0.0
    for n in range(100000):
        s2 = lcg_next(s2)
        var ux2 = (s2 >> 8) & 16777215
        var u2 = (Float32(ux2) + 0.5) / 16777216.0
        su += Float64(u2)
        if u2 > 0.99:
            cnt_hi += 1
        if u2 < 0.01:
            cnt_lo += 1
    print("LCG u: mean", su / 100000.0,
          " >0.99:", cnt_hi, " <0.01:", cnt_lo, "(期望 ~1000/~1000)")
