"""dual OUTS 摊销交错重扫（账本八十八）：每块固定成本（x 载入 +
norm 树 + act 尾）摊到更多权重字节上。

八十七锚定 HBM 上限 1050GB/s：head 76%（每块 14.3KB 权重，无
norm）、dual 67%（每块 14.4KB 权重但有 x+norm 固定成本）。O4→O16
块数 1216→304，固定成本 /4。O8 在七十五曾与 O4 打平（2.95 时代、
K2' 之前）——管线结构已变，交错口径重扫。
modes：o4（默认）/ o8 / o16。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime PAIRS = 24
comptime WARMUP = 12
comptime ROUNDS = 4
comptime MAXT = WARMUP + 3 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 8](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 16](TOKEN)
    session.ctx.synchronize()

    var names = "o4,o8,o16".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 3](fill=0.0)
        var bests = InlineArray[Float64, 3](fill=1e30)
        for i in range(PAIRS):
            for m in range(3):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 8](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 16](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(3):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
