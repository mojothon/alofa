"""账本一百一十五-b：decode GEMM e2e 形状/开关扫型（零新核）。

变体（decode_step comptime 参，其余默认）：
  0 baseline (dual 64/4, down_t 128, o_t 128, tri_t 128, down_si 0)
  1 down_si=8   2 down_si=16   3 down_si=4
  4 dual_o=2    5 tri_t=256    6 o_t=256
每变体前把 kv_pos 回卷到 96（计时用途，store 数据过期无妨），
32 步计时 → 各变体同 kv 相位配对。取 4 轮 min。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 32
comptime REPS = 4
comptime NVAR = 13
comptime KREWIND = 96


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var s = GpuSession(model, 512)
    for i in range(8):
        _ = s.decode_step[INNER, INNER_FF](42, 0)
    s.ctx.synchronize()
    while s.kv_pos < KREWIND + 8:
        _ = s.decode_step[INNER, INNER_FF](42, 0)
    s.ctx.synchronize()

    var best = InlineArray[Float64, NVAR](fill=1e30)
    for rep in range(REPS):
        @parameter
        for v in range(NVAR):
            s.kv_pos = KREWIND
            var t0 = monotonic_ns()
            comptime if v == 0:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 0)
            elif v == 1:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512,
                        True, True, 0, 8, False, 8,
                    ](42, 0)
            elif v == 2:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512,
                        True, True, 0, 8, False, 16,
                    ](42, 0)
            elif v == 3:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 128, 128, 512,
                        True, True, 0, 8, False, 4,
                    ](42, 0)
            elif v == 4:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 2, 128, 0, 128, 128,
                    ](42, 0)
            elif v == 5:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 256, 128,
                    ](42, 0)
            elif v == 6:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 256, 128,
                        512, True, True, 0, 8, False, 0, 1,
                    ](42, 0)
            elif v == 7:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 128, 128,
                        512, True, True, 0, 8, False, 0, 1,
                    ](42, 0)
            elif v == 8:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 256, 128,
                        512, True, True, 0, 8, False, 0, 4,
                    ](42, 0)
            elif v == 9:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 128, 128,
                        512, True, True, 0, 8, False, 0, 4,
                    ](42, 0)
            elif v == 10:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 256, 256,
                    ](42, 0)
            elif v == 11:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 128, 128,
                        512, True, True, 0, 8, False, 0, 2, True,
                    ](42, 0)
            else:
                for i in range(ITERS):
                    _ = s.decode_step[
                        INNER, INNER_FF, 64, 4, 128, 0, 256, 128,
                        512, True, True, 0, 8, False, 0, 1, True,
                    ](42, 0)
            s.ctx.synchronize()
            var d = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
            if d < best[v]:
                best[v] = d
    @parameter
    for v in range(NVAR):
        comptime if v == 0:
            print("base    min ms:", best[v])
        elif v == 1:
            print("dsi8    min ms:", best[v])
        elif v == 2:
            print("dsi16   min ms:", best[v])
        elif v == 3:
            print("dsi4    min ms:", best[v])
        elif v == 4:
            print("dual_o2 min ms:", best[v])
        elif v == 5:
            print("down_t256 min ms:", best[v])
        elif v == 6:
            print("dt256+oo1 min ms:", best[v])
        elif v == 7:
            print("o_outs1 min ms:", best[v])
        elif v == 8:
            print("dt256+oo4 min ms:", best[v])
        elif v == 9:
            print("o_outs4 min ms:", best[v])
        elif v == 10:
            print("dt256+ot256 min ms:", best[v])
        elif v == 11:
            print("qkvp    min ms:", best[v])
        else:
            print("FULL    min ms:", best[v])
