"""attention T1024 端到端交错 A/B（账本八十一）。

分相实测：T1024（V2 变 8 组）35.18µs vs T512（4 组）37.35µs。
端到端裁决：默认 T512 vs T1024，逐 token 交错。
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
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 1024, True, True](TOKEN)
    session.ctx.synchronize()

    var names = "T512,T1024".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 2](fill=0.0)
        var bests = InlineArray[Float64, 2](fill=1e30)
        for i in range(PAIRS):
            for m in range(2):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 1024, True, True](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(2):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
