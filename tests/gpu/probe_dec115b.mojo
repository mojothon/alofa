"""账本一百一十五-b 前置：decode 分段带宽效率测量。

skip 掩码切七段（TRI=1 qkv / ATTN=2 / O=4 / DUAL=8 gate+up /
DOWN=16 / HEAD=32 lm_head）。段时 = full − skip_variant（同流串行
发射，近似可加）。已知权重字节（24 层 f16）：
  TRI 896x1152x2x24=49.5MB  O 896x896x2x24=38.6MB
  DUAL 2x896x4864x2x24=418MB  DOWN 4864x896x2x24=209MB
  HEAD 896x151936x2=272MB   合计 ~987MB
理论下限 @1.4TB/s ≈ 0.7ms/步；实测 2.35ms ⇒ 3.4×。找最弱段。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 28
comptime PRIME = 256


def _timed(mut s: GpuSession, skip: Int) raises -> Float64:
    var t0 = monotonic_ns()
    for i in range(ITERS):
        _ = s.decode_step[INNER, INNER_FF](42, skip)
    s.ctx.synchronize()
    return Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var s = GpuSession(model, 512)
    for i in range(8):
        _ = s.decode_step[INNER, INNER_FF](42, 0)
    s.ctx.synchronize()
    while s.kv_pos < PRIME:
        _ = s.decode_step[INNER, INNER_FF](42, 0)
    s.ctx.synchronize()

    var full = _timed(s, 0)
    print("full: ", full)
    @parameter
    for v in range(6):
        comptime SK = 1 << v
        var d = _timed(s, SK)
        var sec = full - d
        var name = "TRI "
        var mb = 49.5
        comptime if SK == 2:
            name = "ATTN"
            mb = 1.0
        comptime if SK == 4:
            name = "O   "
            mb = 38.6
        comptime if SK == 8:
            name = "DUAL"
            mb = 418.0
        comptime if SK == 16:
            name = "DOWN"
            mb = 209.0
        comptime if SK == 32:
            name = "HEAD"
            mb = 272.3
        print(name, " skip-t:", d, " section:", sec,
              " GB/s:", mb / sec * 1e3)
