"""dual/down 线程数交错 A/B（账本七十八）。

tri 轮（七十七）证明占据率病换 T 一刀就中。本轮同法两个单变量：
- dual：T64/O4（默认，35% 线程槽）→ T128/O4（70%）
- down：T128/O2（26%）→ T256/O2（52%）
- o：T128 → T256（顺带，同 fat 核）

四模式同会话逐 token 交替；步数预算 = WARMUP + 4×PAIRS×ROUNDS
（七十七坑：模式翻倍步数翻倍，压在 MAXT<368 崩线下）。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime PAIRS = 26
comptime WARMUP = 12
comptime ROUNDS = 3
comptime MAXT = WARMUP + 4 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    # 热身：每模式 3 步（JIT 就位）
    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 128](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 256](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 256](TOKEN)
    session.ctx.synchronize()

    var names = "base,dualT128,downT256,oT256".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 4](fill=0.0)
        var bests = InlineArray[Float64, 4](fill=1e30)
        for i in range(PAIRS):
            for m in range(4):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 128](TOKEN)
                elif m == 2:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 256](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 256](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(4):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
