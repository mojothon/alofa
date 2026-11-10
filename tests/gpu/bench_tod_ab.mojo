"""tri/o/down 线程数交错重扫（账本八十七）：896 块延迟界族。

tri 245GB/s、o ~250、down 588 —— 全是 896 块 × 128 线程的延迟界
候选；t512 每块内层迭代 7→2、树 7→9 层，驻留块 8.3→4。t 参是
comptime 现成参，八十一/八十七只扫过 64/128。
modes：base(128/128/128) / tri512 / down512+o512 / all512。
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
comptime MAXT = WARMUP + 4 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 512](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 512, 512](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 512, 0, 512, 512](TOKEN)
    session.ctx.synchronize()

    var names = "base,tri512,dn-o512,all512".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 4](fill=0.0)
        var bests = InlineArray[Float64, 4](fill=1e30)
        for i in range(PAIRS):
            for m in range(4):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 512](TOKEN)
                elif m == 2:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 512, 512](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 512, 0, 512, 512](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(4):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
