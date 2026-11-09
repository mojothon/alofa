"""dual_o 交错 A/B（账本七十八下半）。

T128 拒绝后（bench_fat_ab：三轮一致 +0.15），剩 dual 的输出列宽 O：
O4（默认，608 块…实为 ceildiv(4864,4)=1216 块 ×64 线程=35% 槽，4+4 链）
→ O2（2432 块 ×64=70% 槽，2+2 链）/ O8（608 块=17%，8+8 链）。
prof 六十九的 O4 冠军出自微基准（L2 暖假象），真链交错重裁。
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
        _ = session.decode_step[INNER, INNER_FF, 64, 2](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 8](TOKEN)
    session.ctx.synchronize()

    var names = "O4(默认),O2,O8".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 3](fill=0.0)
        var bests = InlineArray[Float64, 3](fill=1e30)
        for i in range(PAIRS):
            for m in range(3):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 2](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 8](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(3):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
