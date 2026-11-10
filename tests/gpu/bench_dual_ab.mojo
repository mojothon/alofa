"""dual（FF1 gate+up）发射形状扫描（账本八十五）。

0.70ms/24 层 = 29µs/层 @ 16.7MB = ~600GB/s —— 距 DRAM/LSU 顶都远。
账本 74 O2 交错后每线程每 chunk 8 条 16B load 在飞；扫描 dual_t/
dual_o 找 ILP/occupancy 平衡点（t64o4 = 1216 块 × 64 线程为默认）。
modes：t64o4（默认）/ t64o8（16 load 在飞）/ t128o4 / t128o8。
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
        _ = session.decode_step[INNER, INNER_FF, 64, 8](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 128, 4](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 128, 8](TOKEN)
    session.ctx.synchronize()

    var names = "t64o4,t64o8,t128o4,t128o8".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 4](fill=0.0)
        var bests = InlineArray[Float64, 4](fill=1e30)
        for i in range(PAIRS):
            for m in range(4):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 8](TOKEN)
                elif m == 2:
                    _ = session.decode_step[INNER, INNER_FF, 128, 4](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 128, 8](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(4):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
