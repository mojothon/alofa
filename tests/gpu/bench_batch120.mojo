"""账本一百二十：批解码吞吐 bench。B=8 vs B=1（同 run 交错，
钟态公平），KV 从 0 增长到 24。每 token 成本 = 步时/B。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 24
comptime REPS = 3


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)

    @parameter
    for b in range(1, 9):
        comptime if b == 1 or b == 8:
            var best = 1e30
            for rep in range(REPS):
                for i in range(8):
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
                    ](toks)
                a.ctx.synchronize()
                var d = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
                if d < best:
                    best = d
            print("B=", b, " min ms/step:", best,
                  " per-token ms:", best / Float64(b))
