"""账本一百零七-c：批量吞吐计时（20 连发单次 sync，除以 20）——
排除每次 enqueue+sync 的固定开销，得到真实的 kernel 吞吐，用来校准
packed 布局对 down/dual 的真实收益。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.matmul import (
    f16_mma_down_gpu,
    f16_mma_down_pk_gpu,
    f16_mma_dual_gpu,
    f16_mma_dual_pk_gpu,
    pack_w_mma_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime INTER = 4864
    comptime HIDDEN = 896
    comptime WARP = 8
    comptime REP = 20

    var hx = ctx.enqueue_create_host_buffer[DType.float16](T * INTER)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](HIDDEN * INTER)
    var hxf = ctx.enqueue_create_host_buffer[DType.float16](T * HIDDEN)
    var hg = ctx.enqueue_create_host_buffer[DType.float16](INTER * HIDDEN)
    ctx.synchronize()
    for i in range(T * INTER):
        hx[i] = Float16(Float64((i * 13) % 23 - 11) / 6.0)
    for i in range(HIDDEN * INTER):
        hw[i] = Float16(Float64((i * 7) % 11 - 5) / 4.0)
    for i in range(T * HIDDEN):
        hxf[i] = Float16(Float64((i * 5) % 17 - 8) / 5.0)
    for i in range(INTER * HIDDEN):
        hg[i] = Float16(Float64((i * 3) % 13 - 6) / 3.0)

    var dx = ctx.enqueue_create_buffer[DType.float16](T * INTER)
    var dw = ctx.enqueue_create_buffer[DType.float16](HIDDEN * INTER)
    var dwp = ctx.enqueue_create_buffer[DType.float16](HIDDEN * INTER)
    var dxf = ctx.enqueue_create_buffer[DType.float16](T * HIDDEN)
    var dg = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var dgp = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var dg2 = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var dgp2 = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var dout = ctx.enqueue_create_buffer[DType.float32](T * HIDDEN)
    var dact = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dga = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dgb = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.enqueue_copy(dst_buf=dxf, src_buf=hxf)
    ctx.enqueue_copy(dst_buf=dg, src_buf=hg)
    ctx.enqueue_copy(dst_buf=dg2, src_buf=hg)
    ctx.synchronize()
    ctx.enqueue_function[pack_w_mma_gpu[INTER, 2, WARP]](
        dw.unsafe_ptr(), dwp.unsafe_ptr(),
        grid_dim=(HIDDEN // 128, INTER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[HIDDEN, 2, WARP]](
        dg.unsafe_ptr(), dgp.unsafe_ptr(),
        grid_dim=(INTER // 128, HIDDEN // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[HIDDEN, 2, WARP]](
        dg2.unsafe_ptr(), dgp2.unsafe_ptr(),
        grid_dim=(INTER // 128, HIDDEN // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()

    comptime NV = 4
    var best = InlineArray[Float64, NV](fill=1e30)
    for rep in range(5):
        @parameter
        for v in range(NV):
            var t0 = monotonic_ns()
            for r in range(REP):
                comptime if v == 0:
                    ctx.enqueue_function[
                        f16_mma_down_gpu[INTER, 608, 2, WARP]
                    ](
                        dx.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                        Int32(HIDDEN), Int32(T),
                        grid_dim=(HIDDEN // 128, T // 16, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 1:
                    ctx.enqueue_function[
                        f16_mma_down_pk_gpu[INTER, 608, 2, WARP]
                    ](
                        dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
                        Int32(HIDDEN), Int32(T),
                        grid_dim=(HIDDEN // 128, T // 16, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 2:
                    ctx.enqueue_function[f16_mma_dual_gpu[HIDDEN, 608, WARP]](
                        dxf.unsafe_ptr(), dg.unsafe_ptr(), dg2.unsafe_ptr(),
                        dga.unsafe_ptr(), dgb.unsafe_ptr(), dact.unsafe_ptr(),
                        Int32(INTER), Int32(T),
                        grid_dim=(INTER // 128, T // 16, 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    ctx.enqueue_function[
                        f16_mma_dual_pk_gpu[HIDDEN, 608, WARP]
                    ](
                        dxf.unsafe_ptr(), dgp.unsafe_ptr(), dgp2.unsafe_ptr(),
                        dga.unsafe_ptr(), dgb.unsafe_ptr(), dact.unsafe_ptr(),
                        Int32(INTER), Int32(T),
                        grid_dim=(INTER // 128, T // 16, 1),
                        block_dim=(256, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP)
            if per < best[v]:
                best[v] = per
    @parameter
    for v in range(NV):
        print("cfg", v, " us/call:", best[v])
