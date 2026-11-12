"""账本一百二十三诊断二：host 模拟 200 seeds 的分布——判别
公式/PRNG 正确性 vs GPU 核内 bug。"""

from std.math import log
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
    var greedy = Int(a.h_tokout[0])
    print("greedy:", greedy)

    var hits = 0
    for n in range(200):
        var seed = 2000 + n
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
        if bi == greedy:
            hits += 1
        if n < 8:
            print("  sim seed", seed, "->", bi)
    print("host top-1 hits:", hits, "/200 (期望 ~136)")
