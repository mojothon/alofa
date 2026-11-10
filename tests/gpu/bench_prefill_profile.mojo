"""prefill_step 分段计时（账本九十三）：profile=True 段间 sync，
只看占比（流水被破坏，总时长偏大）。128 token 单 rep。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.matmul import f16_bmm_fat_gpu
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128
comptime REPS = 10


def _bench[rb: Int, AF: DType](mut model: QwenForward, ids: List[Int]) raises:
    var session = GpuSession(model, 512)
    _ = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, rb, AF](ids, profile=True)
    session.ctx.synchronize()
    print("=== profile run rb=", rb, " af=", AF)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)

    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)

    # t=1 对照：同 43 launch，GPU 工作量 ≈ 0 —— CPU 发射开销直接现形
    var ids1 = List[Int]()
    ids1.append(1000)
    _bench[8, DType.float32](model, ids1)
    # t=1 无 profile 总时长（43 launch 流水化、GPU 工作量 ≈ 0）
    var s1 = GpuSession(model, 512)
    _ = s1.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids1)
    s1.ctx.synchronize()
    var b1 = 1e30
    for rep in range(5):
        var s2 = GpuSession(model, 512)
        var t0 = monotonic_ns()
        _ = s2.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids1)
        s2.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < b1:
            b1 = ms
    print("t=1 no-profile:", b1, "ms")

    # session 环境连发 down 核（同 buffer 同参）——孤立 vs session 差异分辨
    var s3 = GpuSession(model, 512)
    _ = s3.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids)
    s3.ctx.synchronize()
    var b3 = 1e30
    for rep in range(3):
        var t0 = monotonic_ns()
        var ap = s3.d_actb.unsafe_ptr()
        var wp = s3.d_w16_down[0].unsafe_ptr()
        var dp = s3.d_xb.unsafe_ptr()
        for i in range(30):
            s3.ctx.enqueue_function[
                f16_bmm_fat_gpu[128, 4864, 2, 8, True, DType.float32]
            ](
                ap, wp, dp,
                Int32(896), Int32(128),
                grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
            )
        s3.ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / 30
        if us < b3:
            b3 = us
    print("session-env down x30:", b3, "us/launch")

    _bench[8, DType.float32](model, ids)

    var session = GpuSession(model, 512)
    var best = 1e30
    for i in range(64):
        var t0 = monotonic_ns()
        _ = session.decode_step[INNER, INNER_FF](2000 + i)
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best:
            best = ms
    print("decode 64 x1: best=", best, " ms")
    model.params.keep_alive()
