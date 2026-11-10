"""账本九十七收口：间隙阶梯扫描——burst 间隙 [0,1,2,5,10,20,50,100,200,500]ms
每档 6 burst × 10 发 o-GEMM，打印 档位/时长。定位 快->慢 的切换间隙阈值。
"""

from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from std.ffi import external_call
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.matmul import f16_bmm_fat_gpu
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var s = GpuSession(model, 512)
    var anp = s.d_attnb.unsafe_ptr()
    var wo = s.d_w16_o[0].unsafe_ptr()
    var dxp = s.d_xb.unsafe_ptr()
    # 预热 + 连续高速档（模拟重放流水）
    for w in range(100):
        s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
            anp, wo, dxp, Int32(896), Int32(128),
            grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
        )
    s.ctx.synchronize()
    print("ready", Float64(monotonic_ns()) / 1e9)
    # 连续流基线：无间隙 60 发
    var t0 = monotonic_ns()
    for k in range(60):
        s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
            anp, wo, dxp, Int32(896), Int32(128),
            grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
        )
    s.ctx.synchronize()
    print("gap0 连续60发:", Float64(monotonic_ns() - t0) / 1e3 / 60, "us/发")
    # 间隙阶梯
    var gaps = InlineArray[Int, 10](fill=0)
    gaps[1] = 1
    gaps[2] = 2
    gaps[3] = 5
    gaps[4] = 10
    gaps[5] = 20
    gaps[6] = 50
    gaps[7] = 100
    gaps[8] = 200
    gaps[9] = 500
    for gi in range(10):
        var g = gaps[gi]
        # 每档先重新唤醒：连发 30 发
        for k in range(30):
            s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
                anp, wo, dxp, Int32(896), Int32(128),
                grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
            )
        s.ctx.synchronize()
        # 测 6 burst，每 burst 10 发、间隙 g
        var acc = 0.0
        for b in range(6):
            var tb = monotonic_ns()
            for k in range(10):
                s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
                    anp, wo, dxp, Int32(896), Int32(128),
                    grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
                )
            s.ctx.synchronize()
            acc += Float64(monotonic_ns() - tb) / 1e3 / 10
            _ = external_call["usleep", Int32](Int32(g * 1000))
        print("gap", g, "ms:", acc / 6, "us/发")
    model.params.keep_alive()
