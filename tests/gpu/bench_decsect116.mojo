"""账本一百一十六-a：decode 全景复测（115-b 后新默认：
qkv_pack + down_t256 + o_outs1）。skip 掩码八点扫描，含全跳
（63）看固定底。段字节：TRI 49.5 / O 38.6 / DUAL 418 /
DOWN 209 / HEAD 272.3MB。每变体前 kv_pos 回卷 96 保同相位。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 28
comptime REPS = 3
comptime KREWIND = 96


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var s = GpuSession(model, 512)
    for i in range(8):
        _ = s.decode_step[INNER, INNER_FF](42, 0)
    s.ctx.synchronize()

    var best = InlineArray[Float64, 8](fill=1e30)
    for rep in range(REPS):
        @parameter
        for v in range(8):
            s.kv_pos = KREWIND
            var t0 = monotonic_ns()
            comptime if v == 0:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 0)
            elif v == 1:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 1)
            elif v == 2:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 2)
            elif v == 3:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 4)
            elif v == 4:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 8)
            elif v == 5:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 16)
            elif v == 6:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 32)
            else:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 63)
            s.ctx.synchronize()
            var d = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
            if d < best[v]:
                best[v] = d
    @parameter
    for v in range(8):
        comptime if v == 0:
            print("full      min ms:", best[v])
        elif v == 1:
            print("skipTRI   min ms:", best[v])
        elif v == 2:
            print("skipATTN  min ms:", best[v])
        elif v == 3:
            print("skipO     min ms:", best[v])
        elif v == 4:
            print("skipDUAL  min ms:", best[v])
        elif v == 5:
            print("skipDOWN  min ms:", best[v])
        elif v == 6:
            print("skipHEAD  min ms:", best[v])
        else:
            print("skipALL   min ms:", best[v])
    var full = best[0]
    print("sections ms: TRI", full - best[1], " ATTN", full - best[2],
          " O", full - best[3], " DUAL", full - best[4],
          " DOWN", full - best[5], " HEAD", full - best[6])
    print("unaccounted ms:", full - (
        (full - best[1]) + (full - best[2]) + (full - best[3])
        + (full - best[4]) + (full - best[5]) + (full - best[6])
        + best[7]
    ), " floor(skipALL):", best[7])
