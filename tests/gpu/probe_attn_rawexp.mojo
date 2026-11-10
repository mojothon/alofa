"""RAWEXP 数值探针（账本八十六）：单 session 单步直读中间量。

双 session 同设备崩（CUDA_ERROR_INVALID_VALUE）。改单 session 走
tiles=40（RAWEXP 默认）一步，拷回 d_scores[head0 行 0..40]、
d_attot（tsum 112 个 f64）、d_attn（head0 128 个）—— 对照值域定位
失败模式（gsum=0 / exp 溢出 / scores 乱）。
"""

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime STEPS = 3
comptime MAXT = STEPS + 8


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)
    var hscores = session.ctx.enqueue_create_host_buffer[DType.float32](14 * MAXT)
    var htsum = session.ctx.enqueue_create_host_buffer[DType.float64](112)

    for step in range(STEPS):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
        session.ctx.synchronize()
        session.ctx.enqueue_copy(dst_buf=hscores, src_buf=session.d_scores)
        session.ctx.enqueue_copy(dst_buf=htsum, src_buf=session.d_attot)
        session.ctx.synchronize()
        var mx = Float64(-1e30)
        var have_nan = False
        for i in range(14 * MAXT):
            var v = Float64(hscores[i])
            if v != v:
                have_nan = True
            if v > mx:
                mx = v
        print("step ", step, " kv=", step + 1, " scores max=", mx, " nan=", have_nan, " tsum0..2=", Float64(htsum[0]), " ", Float64(htsum[1]), " ", Float64(htsum[2]))

    model.params.keep_alive()
