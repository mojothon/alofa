"""attention 跨块 j 分块三趟 A/B（账本八十二）。

八十一判决：scores 15.4µs@kv300 锁 per-SM 天花板（14 块）。三趟：
K1 scores ×TILES 块 → K2 V 分块（冗余 softmax）→ K3 合并。scores 与
total 逐位同旧；V 部分和重结合（门容差判据）。端到端裁决：
默认单 kernel vs tiles=4，逐 token 交错。
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
comptime ROUNDS = 5
comptime MAXT = WARMUP + 2 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True, 0, 8](TOKEN)
    session.ctx.synchronize()

    var names = "tiles=4,tiles=8".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 2](fill=0.0)
        var bests = InlineArray[Float64, 2](fill=1e30)
        for i in range(PAIRS):
            for m in range(2):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True, 0, 4](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True, 0, 8](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(2):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
