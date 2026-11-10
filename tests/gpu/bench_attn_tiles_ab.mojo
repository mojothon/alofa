"""attn_tiles 交错重扫（账本八十七）：K1/K2' grid = (14, tiles) 块。

八十二的 tiles 扫描疑为 standalone 口径（热漂污染时代）。K1/K2'
112 块（tiles=8）每 SM 摊 ~1.1 块 —— block 饥饿延迟界候选；tiles
加倍 → 块数加倍、每块 chunk 减半（gmax 每块全行冗余扫描不变贵：
nsc<512 时每线程恰 1 读 + 9 层树）。
modes：tiles 8（默认）/ 16 / 32 / 4。
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
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 16](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 32](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 4](TOKEN)
    session.ctx.synchronize()

    var names = "t8,t16,t32,t4".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 4](fill=0.0)
        var bests = InlineArray[Float64, 4](fill=1e30)
        for i in range(PAIRS):
            for m in range(4):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 16](TOKEN)
                elif m == 2:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 32](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 4](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(4):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
