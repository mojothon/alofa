"""K2' 冗余 softmax 消除 A/B（账本八十三）。

旧 K2：8 tile × 14 头每块全量重扫 scores（8× exp + 8× 写回 +
gsum 树 10 屏障）。K2'：gmax 冗余只读、exp 本 tile smem 缓存
（全行 1×）、gsum 推迟 K3' 合并 —— kernel 数不加（仍 3 发）。
modes：tiles=8（旧）/ tiles=-8（新）。
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
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True, 0, 8](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True, 0, -8](TOKEN)
    session.ctx.synchronize()

    var names = "K2旧,K2新".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 2](fill=0.0)
        var bests = InlineArray[Float64, 2](fill=1e30)
        for i in range(PAIRS):
            for m in range(2):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True, 0, 8](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, True, True, 0, -8](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(2):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
