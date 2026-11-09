"""tri 形状交错 A/B（账本七十七）。

七十六实测 tri 138GB/s（真链 0.44ms/24 层 ≈ 18.3µs/层）——896 块 ×
32 线程只装 13% 线程槽，正是 dual 重排前的病。T 提升（块内线程增多、
每块同 grid）直接提高占据率。三形状同会话逐 token 交替，负载/时钟/
kv 历史完全共享（七十五方法论）。

⚠️ tri 是 NORM 融合核：跨线程归约序随 T 变 —— 与默认 T32 非逐位同。
判据 = 交错计时差 + 门容差（test_cuda_diff）与 argmax 不变。
⚠️ MAXT ≳ 400 会话 AsyncRT 崩（七十五在案），总步数压在 368 内。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime PAIRS = 38
comptime WARMUP = 12
comptime ROUNDS = 3
comptime MAXT = WARMUP + 3 * PAIRS * ROUNDS + 12


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    # 热身：每形状 4 步（JIT 就位）
    for rep in range(4):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 128](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4, 256](TOKEN)
    session.ctx.synchronize()

    var names = "T32(默认),T128,T256".split(",")
    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 3](fill=0.0)
        var bests = InlineArray[Float64, 3](fill=1e30)
        for i in range(PAIRS):
            for m in range(3):
                var t0 = monotonic_ns()
                if m == 0:
                    _ = session.decode_step[INNER, INNER_FF](TOKEN)
                elif m == 1:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 128](TOKEN)
                else:
                    _ = session.decode_step[INNER, INNER_FF, 64, 4, 256](TOKEN)
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        for m in range(3):
            print("round ", r, " ", names[m], ": mean=", totals[m] / Float64(PAIRS), " best=", bests[m])
    model.params.keep_alive()
