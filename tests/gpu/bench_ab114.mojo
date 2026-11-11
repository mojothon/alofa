"""一百一十四-c：decode attention 三趟（attn_tiles=8）vs 单发
（attn_tiles=0, attn_rope_gpu 一核全包）。占比探针显示 attention 段
~0.5ms/步且不随 kv_len 增长——疑似发射开销主导（72 次/步）。
两 session 交替配对，各 40 步/轮。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime PRIME = 150
comptime STEPS = 40
comptime REPS = 3


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var s_new = GpuSession(model, 512)
    var s_old = GpuSession(model, 512)
    for i in range(8):
        _ = s_new.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128](
            42, 0
        )
        _ = s_old.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128](
            42, 0
        )
    s_new.ctx.synchronize()
    s_old.ctx.synchronize()
    while s_new.kv_pos < PRIME:
        _ = s_new.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128](
            42, 0
        )
        _ = s_old.decode_step[INNER, INNER_FF, 64, 4, 128, 0, 128, 128](
            42, 0
        )
    s_new.ctx.synchronize()
    s_old.ctx.synchronize()

    var bnew = 1e30
    var bold = 1e30
    for k in range(REPS):
        var t0 = monotonic_ns()
        if k % 2 == 0:
            for i in range(STEPS):
                _ = s_new.decode_step[
                    INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True,
                    0, 0,
                ](42, 0)
            s_new.ctx.synchronize()
            var d0 = Float64(monotonic_ns() - t0) / 1e6 / Float64(STEPS)
            s_old.kv_pos = s_new.kv_pos
            var t1 = monotonic_ns()
            for i in range(STEPS):
                _ = s_old.decode_step[
                    INNER, INNER_FF, 64, 4, 128, 0, 128, 128,
                ](42, 0)
            s_old.ctx.synchronize()
            var d1 = Float64(monotonic_ns() - t1) / 1e6 / Float64(STEPS)
            print("rep", k, " single:", d0, " tri:", d1)
            if d0 < bnew:
                bnew = d0
            if d1 < bold:
                bold = d1
        else:
            for i in range(STEPS):
                _ = s_old.decode_step[
                    INNER, INNER_FF, 64, 4, 128, 0, 128, 128,
                ](42, 0)
            s_old.ctx.synchronize()
            var d1 = Float64(monotonic_ns() - t0) / 1e6 / Float64(STEPS)
            s_new.kv_pos = s_old.kv_pos
            var t1 = monotonic_ns()
            for i in range(STEPS):
                _ = s_new.decode_step[
                    INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512, True, True,
                    0, 0,
                ](42, 0)
            s_new.ctx.synchronize()
            var d0 = Float64(monotonic_ns() - t1) / 1e6 / Float64(STEPS)
            print("rep", k, " single:", d0, " tri:", d1)
            if d0 < bnew:
                bnew = d0
            if d1 < bold:
                bold = d1
    print("SINGLE min ms:", bnew, " TRI min ms:", bold,
          " delta%:", (bnew - bold) / bold * 100)
