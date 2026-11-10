"""tri 多列 A/B（账本八十四）。

旧 tri：896 块每块 1 输出，245GB/s latency 主导。多列：每块 COLS
输出，x smem 载入/rmsnorm 摊薄，块数 ÷COLS，每线程 COLS×3 条独立
load。k 序与树规约同旧 ⇒ 逐位同（门可零漂验证）。
modes：旧（tri_cols=0）/ COLS=8 / COLS=16。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime PAIRS = 30
comptime WARMUP = 12
comptime ROUNDS = 4
comptime MAXT = WARMUP + 3 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 2](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 256, 4](TOKEN)
    session.ctx.synchronize()

    var names = "旧,c2t128,c4t256".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 3](fill=0.0)
        var bests = InlineArray[Float64, 3](fill=1e30)
        for i in range(PAIRS):
            for m in range(3):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 2](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 256, 4](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(3):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
