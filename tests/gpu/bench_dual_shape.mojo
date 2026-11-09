"""dual 真链形状扫（账本七十五）。

⚠️ ⚠️ 本基准的方法论已被否决（账本七十五）：共享 GPU 上顺序跑形状，
每个形状的测量窗口落在不同负载时段，排名是负载漂移假象（一度示
T128/O8 快 0.39ms，交错 A/B 证伪 —— 两形状逐位打平）。保留本文件
只为存证；形状裁决一律用 bench_dual_ab（逐 token 交错）。


六十九的 T/O 冠军（dual T64/O4、down T128/O2）全部出自微基准 —— 同权重
连发 ITERS 次，8.7MB 权重自第二次起驻留 40MB L2，BW 数字是暖缓存假象
（七十四：dual 微基准 27.2µs vs 真链 ~36µs 的 gap 即此）。真链每层权重
冷读（f16 全权重 ~500MB >> L2），延迟主导下的最优形状可以不同。

口径同 bench_decode：每形状**独立会话**（kv_pos 归零、缓存冷启动一致），
WARMUP 后 ROUNDS 轮取 best；七十四的重排（k 外层行内层）在位。 Attention
随 kv 微涨（kv ≤ 133 时 ~1µs/层内），对形状间相对比较无碍。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime WARMUP = 300
comptime ITERS = 30
comptime ROUNDS = 2
comptime MAXT = WARMUP + ITERS * ROUNDS + 8


def _sweep[T: Int, O: Int](mut model: QwenForward, label: String) raises:
    var session = GpuSession(model, MAXT)
    for i in range(WARMUP):
        _ = session.decode_step[INNER, INNER_FF, T, O](TOKEN)
    session.ctx.synchronize()

    var best = Float64(1e30)
    for r in range(ROUNDS):
        var s0 = monotonic_ns()
        for i in range(ITERS):
            _ = session.decode_step[INNER, INNER_FF, T, O](TOKEN)
        var ms = Float64(monotonic_ns() - s0) / 1e6 / Float64(ITERS)
        if ms < best:
            best = ms
    print("dual T" + String(T) + " O" + String(O) + ": ", best, " ms/token ", label)
    _ = label


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    comptime if True:
        _sweep[128, 8](model, "(七十五真链冠军)")
        _sweep[64, 4](model, "(六十九微基准冠军)")
    model.params.keep_alive()
