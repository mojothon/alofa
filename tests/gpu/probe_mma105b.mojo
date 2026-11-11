"""账本一百零五-b：dual MMA 在 t=8（<16 行，块填 16 行）下的孤立
验证——down/q 同构对照。"""

from std.math import exp
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.matmul import (
    f16_mma_down_gpu,
    f16_mma_dual_gpu,
    x_to_f16_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 8
    comptime INNER = 896
    comptime INTER = 4864

    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    var hwa = ctx.enqueue_create_host_buffer[DType.float16](INTER * INNER)
    var hwb = ctx.enqueue_create_host_buffer[DType.float16](INTER * INNER)
    var hwd = ctx.enqueue_create_host_buffer[DType.float16](INTER * INNER)
    ctx.synchronize()
    for i in range(T * INNER):
        hx[i] = Float32(Float64((i * 11) % 17 - 8) / 9.0)
    for i in range(INTER * INNER):
        hwa[i] = Float16(Float64((i * 3) % 19 - 9) / 7.0)
        hwb[i] = Float16(Float64((i * 5) % 13 - 6) / 5.0)
        hwd[i] = Float16(Float64((i * 7) % 11 - 5) / 4.0)

    var dx = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dxh = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    var dwa = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwb = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwd = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dga = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dub = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dac = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dxb = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dwa, src_buf=hwa)
    ctx.enqueue_copy(dst_buf=dwb, src_buf=hwb)
    ctx.enqueue_copy(dst_buf=dwd, src_buf=hwd)
    ctx.synchronize()
    ctx.enqueue_function[x_to_f16_gpu](
        dx.unsafe_ptr(), dxh.unsafe_ptr(), Int32(T * INNER),
        grid_dim=((T * INNER + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    ctx.enqueue_function[f16_mma_dual_gpu[INNER, 608]](
        dxh.unsafe_ptr(), dwa.unsafe_ptr(), dwb.unsafe_ptr(),
        dga.unsafe_ptr(), dub.unsafe_ptr(), dac.unsafe_ptr(),
        Int32(INTER), Int32(T),
        grid_dim=(INTER // 64, (T + 15) // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[f16_mma_down_gpu[INTER, 608]](
        dxh.unsafe_ptr(), dwd.unsafe_ptr(), dxb.unsafe_ptr(),
        Int32(INTER), Int32(T),
        grid_dim=(INTER // 128, (T + 15) // 16, 1), block_dim=(128, 1, 1),
    )

    var hga = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    var hub = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    var hac = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    var hxb = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    ctx.enqueue_copy(dst_buf=hga, src_buf=dga)
    ctx.enqueue_copy(dst_buf=hub, src_buf=dub)
    ctx.enqueue_copy(dst_buf=hac, src_buf=dac)
    ctx.enqueue_copy(dst_buf=hxb, src_buf=dxb)
    ctx.synchronize()

    var mxa = 0.0
    var mxu = 0.0
    var mxc = 0.0
    var mxd = 0.0
    var badr = -1
    var badc = -1
    for r in range(T):
        for c in range(0, INTER, 61):
            var ra = 0.0
            var rb = 0.0
            var rd = 0.0
            for k in range(INNER):
                var xv = Float64(Float32(Float16(hx[r * INNER + k])))
                ra += xv * Float64(Float32(hwa[c * INNER + k]))
                rb += xv * Float64(Float32(hwb[c * INNER + k]))
                rd += xv * Float64(Float32(hwd[c * INNER + k]))
            var da = Float64(hga[r * INTER + c]) - ra
            if da < 0:
                da = -da
            if da > mxa:
                mxa = da
                badr = r
                badc = c
            var du = Float64(hub[r * INTER + c]) - rb
            if du < 0:
                du = -du
            if du > mxu:
                mxu = du
            var want_c = ra / (1.0 + exp(-ra)) * rb
            var dc = Float64(hac[r * INTER + c]) - want_c
            if dc < 0:
                dc = -dc
            if dc > mxc:
                mxc = dc
            var dd = Float64(hxb[r * INTER + c]) - rd
            if dd < 0:
                dd = -dd
            if dd > mxd:
                mxd = dd
    print("t=8 dual gate maxdiff:", mxa, " at r=", badr, " c=", badc)
    print("t=8 dual up   maxdiff:", mxu)
    print("t=8 dual act  maxdiff:", mxc)
    print("t=8 down      maxdiff:", mxd)
