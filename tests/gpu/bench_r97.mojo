"""账本九十七：慢/快区切换条件——阶段化最小 bench。
A 冷（构造后首发）/ B 热连发 / C idle 5s 后 / D 再连发。
配 nsys --gpu-metrics-device=all --gpu-metrics-frequency=10000 看 SM 频率。
"""

from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from std.ffi import external_call
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)

    var s = GpuSession(model, 512)
    # A：无 gap 连发 ×2（=生产现状）
    for k in range(2):
        var t0 = monotonic_ns()
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32](ids, False, True)
        s.ctx.synchronize()
        print("A 无gap", k, Float64(monotonic_ns()) / 1e9, Float64(monotonic_ns() - t0) / 1e6, "ms")
    # B：gap 30us ×2（同 session，kv 继续增）
    for k in range(2):
        var t0 = monotonic_ns()
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32](ids, False, True, 30)
        s.ctx.synchronize()
        print("B gap30", k, Float64(monotonic_ns()) / 1e9, Float64(monotonic_ns() - t0) / 1e6, "ms")
    # C：换序——先 gap 后无 gap（排除顺序效应）
    var s2 = GpuSession(model, 512)
    for k in range(2):
        var t0 = monotonic_ns()
        _ = s2.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32](ids, False, True, 30)
        s2.ctx.synchronize()
        print("C gap30", k, Float64(monotonic_ns()) / 1e9, Float64(monotonic_ns() - t0) / 1e6, "ms")
    for k in range(2):
        var t0 = monotonic_ns()
        _ = s2.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32](ids, False, True)
        s2.ctx.synchronize()
        print("D 无gap", k, Float64(monotonic_ns()) / 1e9, Float64(monotonic_ns() - t0) / 1e6, "ms")
    model.params.keep_alive()
