"""attention 向量化 scores 交错 A/B（账本七十九下半）。

T 扫无效（bench_attn_ab 三轮无稳定序）——瓶颈不是块数是 scores 相位
的访存结构：每 j 单线程 128 条标量 2B 读串行链。VEC：k 行 load[8]
16B 向量读（16 条替代 128 条，相互独立可全在飞）。求和序 d 升序
不变 ⇒ **scores 逐位同旧** —— 门应零漂、argmax 逐位不变（强验证）。
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
comptime MAXT = WARMUP + 2 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    for rep in range(3):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 128, True](TOKEN)
    session.ctx.synchronize()

    var names = "标量,VEC".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 2](fill=0.0)
        var bests = InlineArray[Float64, 2](fill=1e30)
        for i in range(PAIRS):
            for m in range(2):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 128, True](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(2):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
