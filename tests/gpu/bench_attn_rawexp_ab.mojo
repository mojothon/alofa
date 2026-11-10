"""K2' RAWEXP（跳过 gmax 减稳）A/B（账本八十六）。

管线归因（bench_skip_attr）：attention 三趟 = 0.472ms（#2 成本）。
K2' 的 gmax 相 = 全行扫描 + 10 层 barrier 树；score 值域 ~±15 << f32
exp 上限 88 → 跳过 gmax 直取 exp(s)（min 80 钳防离群）。
modes：tiles=8（gmax）/ tiles=40（编码 = tiles 8 + RAWEXP）。
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
comptime MAXT = WARMUP + 2 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 40](TOKEN)
    session.ctx.synchronize()

    var names = "gmax,rawexp".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 2](fill=0.0)
        var bests = InlineArray[Float64, 2](fill=1e30)
        for i in range(PAIRS):
            for m in range(2):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True, 0, 40](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(2):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
