"""账本一百一十四-c 前置：decode attention 占比测量。

用 skip 掩码（SKIP_ATTN=2）在三个 kv_len 阶段（64/256/508）各测
skip=0 与 skip=2 的每步稳态时间，差值即 attention+rope+store 段。
占比大才值得做 decode GQA 分组。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 60


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

    for target in [64, 256, 384]:
        while s.kv_pos < target:
            _ = s.decode_step[INNER, INNER_FF](42, 0)
        s.ctx.synchronize()
        var full = _timed(s, 0)
        var noattn = _timed(s, 2)
        while s.kv_pos < target + 8:
            _ = s.decode_step[INNER, INNER_FF](42, 0)
        s.ctx.synchronize()
        print("kv_len", target, " full:", full, " no-attn:", noattn,
              " attn-share:", (full - noattn) / full)
