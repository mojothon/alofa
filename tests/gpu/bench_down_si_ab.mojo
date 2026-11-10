"""down split-inner A/B（账本八十七）：归约维切 SPLIT 份破延迟界。

down 现状 448 块 × 128 线程、x smem 19KB ⇒ 驻留 ~7 块/SM，
670GB/s（管线真值）。si 切 4/8 份 → 1792/3584 块，每块 x/w 只装
1/SPLIT，combine 微核收口。
modes：base / si4 / si8。
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
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 8, False, 4](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 8, False, 8](TOKEN)
    session.ctx.synchronize()

    var names = "base,si4,si8".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 3](fill=0.0)
        var bests = InlineArray[Float64, 3](fill=1e30)
        for i in range(PAIRS):
            for m in range(3):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 8, False, 4](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 8, False, 8](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(3):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
