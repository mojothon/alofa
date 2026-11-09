"""attention THREADS 交错 A/B（账本七十九）。

七十九开局发现：attn_rope grid=n_heads=**14 块**，128 线程 —— 全卡
线程槽 <1%，比"相位 5→3"更重的病。THREADS 是 kernel 泛型参（smem
行宽 + 树归约宽度 + j 步距），提 T 直接加倍并行 j 数：
T128→256→512。max 树归约序无关（bit-exact）；total 求和序随 T 变
（f64 末位差，门 1e-5 容差内裁决）。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime PAIRS = 36
comptime WARMUP = 12
comptime ROUNDS = 3
comptime MAXT = WARMUP + 3 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 256](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512](TOKEN)
    session.ctx.synchronize()

    var names = "T128(默认),T256,T512".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 3](fill=0.0)
        var bests = InlineArray[Float64, 3](fill=1e30)
        for i in range(PAIRS):
            for m in range(3):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 256](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(3):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
