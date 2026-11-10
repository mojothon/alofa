"""dual norm 出核 + 无 smem 变体 A/B（账本八十五）。

形状扫描证伪后（t64o4 565GB/s 最优），破 x 摊薄 vs occupancy 死锁：
norm 出核（rmsnorm 单发 +1 发射）+ dual 去 sx/NORM 直读 L1 —— o2 时
2432 块 × 2 warp = 44 warp/SM。dot 序同旧 ⇒ 对同一归一化输入逐位同。
modes：fused t64o4（默认）/ nx t64o4 / nx t64o2 / nx t128o4。
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
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 8, True](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 2, 128, 0, 128, 128, 512, True, True, 0, 8, True](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 128, 4, 128, 0, 128, 128, 512, True, True, 0, 8, True](TOKEN)
    session.ctx.synchronize()

    var names = "fused,nx4,nx2,nx4t128".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 4](fill=0.0)
        var bests = InlineArray[Float64, 4](fill=1e30)
        for i in range(PAIRS):
            for m in range(4):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 8, True](TOKEN)
                elif m == 2:
                    _ = session.decode_step[INNER, INNER_FF, 64, 2, 128, 0, 128, 128, 512, True, True, 0, 8, True](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 128, 4, 128, 0, 128, 128, 512, True, True, 0, 8, True](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(4):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
