"""账本九十九：fat2/dual2 A/B——①kernel 级 device 侧逐位对照（diff
kernel 归约 + 小数组 D2H）；②重放 harness 总时长（同轮交替 30 rep）。
"""

from max.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.matmul import (
    f16_bmm_fat_gpu, f16_bmm_dual_gpu, f16_bmm_fat3_gpu, f16_bmm_dual3_gpu,
)
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128
comptime T = 128


def diff99_gpu(
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        dst[Int(block_idx.x)] = max(dst[Int(block_idx.x)], d)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)

    var s = GpuSession(model, 512)
    var ctx = s.ctx
    var t = T

    # 跑一次 prefill 让中间 buffer 有真实数据
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids, False, True)

    var anp = s.d_attnb.unsafe_ptr()
    var wo = s.d_w16_o[0].unsafe_ptr()
    var dxp = s.d_xb.unsafe_ptr()
    var abp = s.d_actb.unsafe_ptr()
    var wd = s.d_w16_down[0].unsafe_ptr()
    var xnp = s.d_xnormb.unsafe_ptr()
    var wg = s.d_w16_gate[0].unsafe_ptr()
    var wu = s.d_w16_up[0].unsafe_ptr()

    # ===== kernel 级逐位对照 =====
    var nblocks = (t * 4864 + 255) // 256
    var dbuf = ctx.enqueue_create_buffer[DType.float32](nblocks)
    var hdiff = ctx.enqueue_create_host_buffer[DType.float32](nblocks)
    var dxa = ctx.enqueue_create_buffer[DType.float32](t * 896)
    var dxd = ctx.enqueue_create_buffer[DType.float32](t * 896)
    var g1 = ctx.enqueue_create_buffer[DType.float32](t * 4864)
    var u1 = ctx.enqueue_create_buffer[DType.float32](t * 4864)
    var a1 = ctx.enqueue_create_buffer[DType.float32](t * 4864)
    var g2 = ctx.enqueue_create_buffer[DType.float32](t * 4864)
    var u2 = ctx.enqueue_create_buffer[DType.float32](t * 4864)
    var a2 = ctx.enqueue_create_buffer[DType.float32](t * 4864)
    ctx.synchronize()

    # fat o
    ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, False, DType.float32]](
        anp, wo, dxa.unsafe_ptr(), Int32(896), Int32(t),
        grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[f16_bmm_fat3_gpu[128, 896, 4, 8, False, DType.float32]](
        anp, wo, dxd.unsafe_ptr(), Int32(896), Int32(t),
        grid_dim=(224, 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[diff99_gpu](
        dxa.unsafe_ptr(), dxd.unsafe_ptr(), dbuf.unsafe_ptr(), Int32(t * 896),
        grid_dim=((t * 896 + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hdiff, src_buf=dbuf)
    ctx.synchronize()
    var mx = 0.0
    for i in range((t * 896 + 255) // 256):
        if Float64(hdiff[i]) > mx:
            mx = Float64(hdiff[i])
    print("fat o(896) maxdiff:", mx)

    # fat down（ACC=True 写 dxa/dxd 同为累加目标——先清零语义：ACC 对两版一致即可，输入 abp 同源）
    ctx.enqueue_function[f16_bmm_fat_gpu[128, 4864, 2, 8, False, DType.float32]](
        abp, wd, dxa.unsafe_ptr(), Int32(896), Int32(t),
        grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[f16_bmm_fat3_gpu[128, 4864, 4, 8, False, DType.float32]](
        abp, wd, dxd.unsafe_ptr(), Int32(896), Int32(t),
        grid_dim=(224, 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[diff99_gpu](
        dxa.unsafe_ptr(), dxd.unsafe_ptr(), dbuf.unsafe_ptr(), Int32(t * 896),
        grid_dim=((t * 896 + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hdiff, src_buf=dbuf)
    ctx.synchronize()
    mx = 0.0
    for i in range((t * 896 + 255) // 256):
        if Float64(hdiff[i]) > mx:
            mx = Float64(hdiff[i])
    print("fat down(4864) maxdiff:", mx)

    # dual
    ctx.enqueue_function[f16_bmm_dual_gpu[64, 896, 2, 8, DType.float32]](
        xnp, wg, wu, g1.unsafe_ptr(), u1.unsafe_ptr(), a1.unsafe_ptr(),
        Int32(4864), Int32(t),
        grid_dim=(2432, 16, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_function[f16_bmm_dual3_gpu[64, 896, 2, 8, DType.float32]](
        xnp, wg, wu, g2.unsafe_ptr(), u2.unsafe_ptr(), a2.unsafe_ptr(),
        Int32(4864), Int32(t),
        grid_dim=(2432, 16, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_function[diff99_gpu](
        g1.unsafe_ptr(), g2.unsafe_ptr(), dbuf.unsafe_ptr(), Int32(t * 4864),
        grid_dim=(nblocks, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hdiff, src_buf=dbuf)
    ctx.synchronize()
    mx = 0.0
    for i in range(nblocks):
        if Float64(hdiff[i]) > mx:
            mx = Float64(hdiff[i])
    print("dual gate maxdiff:", mx)
    ctx.enqueue_function[diff99_gpu](
        a1.unsafe_ptr(), a2.unsafe_ptr(), dbuf.unsafe_ptr(), Int32(t * 4864),
        grid_dim=(nblocks, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hdiff, src_buf=dbuf)
    ctx.synchronize()
    mx = 0.0
    for i in range(nblocks):
        if Float64(hdiff[i]) > mx:
            mx = Float64(hdiff[i])
    print("dual act maxdiff:", mx)

    # ===== 时长 A/B（交替 30 rep）=====
    var tn = 0.0
    var t2 = 0.0
    for rep in range(30):
        var t0 = monotonic_ns()
        ctx.enqueue_function[f16_bmm_fat_gpu[128, 4864, 2, 8, True, DType.float32]](
            abp, wd, dxp, Int32(896), Int32(t),
            grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        tn += Float64(monotonic_ns() - t0) / 1e3
        t0 = monotonic_ns()
        ctx.enqueue_function[f16_bmm_fat3_gpu[128, 4864, 4, 8, True, DType.float32]](
            abp, wd, dxp, Int32(896), Int32(t),
            grid_dim=(224, 16, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        t2 += Float64(monotonic_ns() - t0) / 1e3
    print("down 旧:", tn / 30, "us 新(fat3):", t2 / 30, "us")

    tn = 0.0
    t2 = 0.0
    for rep in range(30):
        var t0 = monotonic_ns()
        ctx.enqueue_function[f16_bmm_dual_gpu[64, 896, 2, 8, DType.float32]](
            xnp, wg, wu, g1.unsafe_ptr(), u1.unsafe_ptr(), a1.unsafe_ptr(),
            Int32(4864), Int32(t),
            grid_dim=(2432, 16, 1), block_dim=(64, 1, 1),
        )
        ctx.synchronize()
        tn += Float64(monotonic_ns() - t0) / 1e3
        t0 = monotonic_ns()
        ctx.enqueue_function[f16_bmm_dual3_gpu[64, 896, 2, 8, DType.float32]](
            xnp, wg, wu, g2.unsafe_ptr(), u2.unsafe_ptr(), a2.unsafe_ptr(),
            Int32(4864), Int32(t),
            grid_dim=(1216, 16, 1), block_dim=(64, 1, 1),
        )
        ctx.synchronize()
        t2 += Float64(monotonic_ns() - t0) / 1e3
    print("dual 旧:", tn / 30, "us 新(fat3):", t2 / 30, "us")

    tn = 0.0
    t2 = 0.0
    for rep in range(30):
        var t0 = monotonic_ns()
        ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
            anp, wo, dxp, Int32(896), Int32(t),
            grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        tn += Float64(monotonic_ns() - t0) / 1e3
        t0 = monotonic_ns()
        ctx.enqueue_function[f16_bmm_fat3_gpu[128, 896, 4, 8, True, DType.float32]](
            anp, wo, dxp, Int32(896), Int32(t),
            grid_dim=(224, 16, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        t2 += Float64(monotonic_ns() - t0) / 1e3
    print("o 旧:", tn / 30, "us 新(fat3):", t2 / 30, "us")
    model.params.keep_alive()
