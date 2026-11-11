"""账本一百一十九-b：①4 路交错吞吐；②prefill 核 t 缩放
（真批处理上限：同核 n_rows 翻倍是否近恒定成本）。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 24
comptime REPS = 4
comptime KREWIND = 96


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    print("creating A")
    var a = GpuSession(model, 512)
    print("creating B")
    var b = GpuSession(model, 512)
    print("sessions ok")
    for i in range(6):
        _ = a.decode_step[INNER, INNER_FF](42, 0, False, True)
        _ = b.decode_step[INNER, INNER_FF](42, 0, False, True)
    a.ctx.synchronize()
    b.ctx.synchronize()
    print("warm ok")

    var best2 = 1e30
    for rep in range(REPS):
        a.kv_pos = KREWIND
        b.kv_pos = KREWIND
        var t0 = monotonic_ns()
        for i in range(ITERS):
            _ = a.decode_step[INNER, INNER_FF](42, 0, False, True)
            _ = b.decode_step[INNER, INNER_FF](42, 0, False, True)
        a.ctx.synchronize()
        b.ctx.synchronize()
        var d1 = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS) * 2.0
        if d1 < best2:
            best2 = d1

    print("2-way per-step ms:", best2)
    print("interleave ok")

    # prefill t 缩放（同核 n_rows 变化；kv 回卷 0）
    var ids = List[Int]()
    for i in range(64):
        ids.append(1000 + (i * 37) % 900)
    @parameter
    for tt in range(1, 9):
        comptime if tt == 1 or tt == 2 or tt == 4 or tt == 8:
            var sub = List[Int]()
            for i in range(tt):
                sub.append(1000 + i)
            var best = 1e30
            for rep in range(4):
                a.kv_pos = 0
                _ = a.prefill_step[
                    INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
                    DType.float32
                ](sub)
                a.ctx.synchronize()
                var t0 = monotonic_ns()
                _ = a.prefill_step[
                    INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
                    DType.float32
                ](sub)
                a.ctx.synchronize()
                var ms = Float64(monotonic_ns() - t0) / 1e6
                if ms < best:
                    best = ms
                a.kv_pos = 0
            print("prefill t=", tt, " min ms:", best)
        else:
            pass
