"""账本九十七：GEMM 时钟探针——每 500ms 发射 10 次 o-GEMM（SM-bound），
打印 (绝对时刻, 单发 µs)。SM 时钟 765→1410 时单发时长约 1.84×。
后台 nohup 运行，与 bench_r97 同 CLOCK_MONOTONIC 对齐。
"""

from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from std.ffi import external_call
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.matmul import f16_bmm_fat_gpu
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var s = GpuSession(model, 512)
    var anp = s.d_attnb.unsafe_ptr()
    var wo = s.d_w16_o[0].unsafe_ptr()
    var dxp = s.d_xb.unsafe_ptr()
    # 预热 30 发
    for w in range(30):
        s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
            anp, wo, dxp, Int32(896), Int32(128),
            grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
        )
    s.ctx.synchronize()
    print("probe ready:", Float64(monotonic_ns()) / 1e9)
    var n = 0
    while n < 1500:
        var t0 = monotonic_ns()
        for k in range(10):
            s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
                anp, wo, dxp, Int32(896), Int32(128),
                grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
            )
        s.ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / 10
        n += 1
        print("probe", n, Float64(monotonic_ns()) / 1e9, us, "us")
        _ = external_call["usleep", Int32](Int32(470000))
    model.params.keep_alive()
