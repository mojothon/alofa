"""账本一百零五-a：q MMA（bias 尾）与 down MMA@896（1 块+余数，
o 用配置）孤立验证——生产形状 t=128 与 t=8。"""

from std.math import exp
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.matmul import (
    f16_mma_down_gpu,
    f16_mma_q_gpu,
    x_to_f16_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime INNER = 896

    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](INNER * INNER)
    var hbi = ctx.enqueue_create_host_buffer[DType.float32](INNER)
    ctx.synchronize()
    for i in range(T * INNER):
        hx[i] = Float32(Float64((i * 11) % 17 - 8) / 9.0)
    for i in range(INNER * INNER):
        hw[i] = Float16(Float64((i * 3) % 19 - 9) / 7.0)
    for i in range(INNER):
        hbi[i] = Float32(Float64((i * 5) % 7 - 3) / 11.0)

    var dx = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dxh = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    var dw = ctx.enqueue_create_buffer[DType.float16](INNER * INNER)
    var dbi = ctx.enqueue_create_buffer[DType.float32](INNER)
    var dq = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dxb = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.enqueue_copy(dst_buf=dbi, src_buf=hbi)
    ctx.synchronize()
    ctx.enqueue_function[x_to_f16_gpu](
        dx.unsafe_ptr(), dxh.unsafe_ptr(), Int32(T * INNER),
        grid_dim=((T * INNER + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    # down@896 的 dst 预填 3.5（ACC）
    var hb0 = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    ctx.synchronize()
    for i in range(T * INNER):
        hb0[i] = Float32(3.5)
    ctx.enqueue_copy(dst_buf=dxb, src_buf=hb0)
    ctx.synchronize()

    ctx.enqueue_function[f16_mma_q_gpu[INNER, 608]](
        dxh.unsafe_ptr(), dw.unsafe_ptr(), dbi.unsafe_ptr(),
        dq.unsafe_ptr(), Int32(INNER), Int32(T),
        grid_dim=(INNER // 128, T // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[f16_mma_down_gpu[INNER, 608]](
        dxh.unsafe_ptr(), dw.unsafe_ptr(), dxb.unsafe_ptr(),
        Int32(INNER), Int32(T),
        grid_dim=(INNER // 128, T // 16, 1), block_dim=(128, 1, 1),
    )

    var hq = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    var hxb = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    ctx.enqueue_copy(dst_buf=hq, src_buf=dq)
    ctx.enqueue_copy(dst_buf=hxb, src_buf=dxb)
    ctx.synchronize()

    var mxq = 0.0
    var mxo = 0.0
    for r in range(0, T, 13):
        for c in range(0, INNER, 29):
            var rr = 0.0
            for k in range(INNER):
                rr += Float64(Float32(Float16(hx[r * INNER + k]))) * Float64(
                    Float32(hw[c * INNER + k])
                )
            var dqv = Float64(hq[r * INNER + c]) - (
                rr + Float64(hbi[c])
            )
            if dqv < 0:
                dqv = -dqv
            if dqv > mxq:
                mxq = dqv
            var dov = Float64(hxb[r * INNER + c]) - (3.5 + rr)
            if dov < 0:
                dov = -dov
            if dov > mxo:
                mxo = dov
    print("q MMA (bias) maxdiff:", mxq)
    print("down MMA @896 ACC maxdiff:", mxo)
