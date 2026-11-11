"""账本一百二十-b：批规模扩展 bench（B=8/16/32）+ sample 模式
D2H 成本量化。同 run 交错（钟态判据），copy_logits=False 只回
4B×B tokout；对照组 B=8 带 logits D2H。KV 0→40 增长。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 40
comptime REPS = 3


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)

    @parameter
    for b in range(8, 65):
        comptime if b == 8 or b == 16 or b == 32 or b == 64:
            var best = 1e30
            for rep in range(REPS):
                for i in range(64):
                    a.h_posarr[i] = Int32(0)
                var t0 = monotonic_ns()
                for s in range(ITERS):
                    var toks = List[Int]()
                    for i in range(b):
                        toks.append(100 + i * 10 + s)
                    _ = a.prefill_step[
                        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
                        DType.float32, True, True, True, True, True, True,
                        True, True, False, True
                    ](toks, False, True, False)
                a.ctx.synchronize()
                var d = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
                if d < best:
                    best = d
            print("B=", b, " min ms/step:", best,
                  " per-token ms:", best / Float64(b))

    # 对照：B=8 带 logits D2H（量化 sample 模式收益）
    var bestc = 1e30
    for rep in range(REPS):
        for i in range(32):
            a.h_posarr[i] = Int32(0)
        var t0 = monotonic_ns()
        for s in range(ITERS):
            var toks = List[Int]()
            for i in range(8):
                toks.append(100 + i * 10 + s)
            _ = a.prefill_step[
                INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
                DType.float32, True, True, True, True, True, True,
                True, True, False, True
            ](toks)
        a.ctx.synchronize()
        var d = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
        if d < bestc:
            bestc = d
    print("B=8+logitsD2H min ms/step:", bestc)
