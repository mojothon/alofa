"""skip 掩码管线归因（账本八十六）。

sections bench 是 q4 旧口径，不能核 f16 管线的账。改用 skip 掩码：
跳过某族 kernel（noop 替换）的差值 = 该族管线内真实成本（含发射税、
不含被隐藏的部分）。全家族差值之和 vs 端到端 → 剩余 = 真正的地板。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import SKIP_ATTN, SKIP_DUAL, SKIP_DOWN, SKIP_GATHER, SKIP_HEAD, SKIP_O, SKIP_TRI
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 14
comptime WARMUP = 12
comptime ROUNDS = 4
comptime MAXT = WARMUP + 8 * ITERS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
    session.ctx.synchronize()

    var names = "full,no-tri,no-attn,no-o,no-dual,no-down,no-head,no-gather".split(",")
    var masks = InlineArray[Int, 8](fill=0)
    masks[1] = 1
    masks[2] = 2
    masks[3] = 4
    masks[4] = 8
    masks[5] = 16
    masks[6] = 32
    masks[7] = 64
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 8](fill=0.0)
        var bests = InlineArray[Float64, 8](fill=1e30)
        for i in range(ITERS):
            for m in range(8):
                var t0 = monotonic_ns()
                _ = session.decode_step[INNER, INNER_FF](TOKEN, masks[m])
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(8):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(ITERS), " best=", bests[m])
    model.params.keep_alive()
