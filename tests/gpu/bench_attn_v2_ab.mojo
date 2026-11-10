"""attention V2 端到端交错 A/B（账本八十）。

分相实测：V 相 16.3µs → V2（T512，j 分 4 组并行）6.6µs，kernel
46.1 → 36.5µs。端到端裁决：default（T128+V1）vs（T512+V2）逐 token
交错。⚠️ V2 的 j 求和序 + T512 的 sum 树序都变（f64 重结合）——
门 1e-5 容差 + argmax 判据（bit-exact 判据不再适用）。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime PAIRS = 36
comptime WARMUP = 12
comptime ROUNDS = 3
comptime MAXT = WARMUP + 2 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True](TOKEN)
    session.ctx.synchronize()

    var names = "默认(T128+V1),T512+V2".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 2](fill=0.0)
        var bests = InlineArray[Float64, 2](fill=1e30)
        for i in range(PAIRS):
            for m in range(2):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(2):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
