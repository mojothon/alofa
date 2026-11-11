"""账本一百一十七-a：prefill 全景复测（128 token，当前默认
rb8/f32/MMA 全开/wpack）。无 profile 取 5 rep min 为真值；
profile=True 一发看十桶占比（段间 sync 破坏流水，只看占比）。"""

from alofa.core.ffi.posix import monotonic_ns
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
    var best = 1e30
    for rep in range(5):
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32](ids)
        s.ctx.synchronize()
        s.kv_pos = 0
    for rep in range(5):
        var t0 = monotonic_ns()
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32](ids)
        s.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best:
            best = ms
        s.kv_pos = 0
    print("prefill total min ms:", best)

    var t1 = monotonic_ns()
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32](ids, True)
    s.ctx.synchronize()
    print("profile total(钟内含 sync):", Float64(monotonic_ns() - t1) / 1e6, "ms")
