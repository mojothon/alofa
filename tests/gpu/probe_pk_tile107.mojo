"""账本一百零七-b：packed down 的分块重扫——w 载入整型化后，warp 数
（并发）与 tile 数（A 复用）的权衡变了。生产形状 t=128、INNER=4864、
HIDDEN=896。配置：(WARP,OUTS,CH) = (8,2,608) / (4,4,608) / (8,2,304)
/ (8,2,1216)。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.matmul import (
    f16_mma_down_pk_gpu,
    pack_w_mma_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime INTER = 4864
    comptime HIDDEN = 896

    var hx = ctx.enqueue_create_host_buffer[DType.float16](T * INTER)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](HIDDEN * INTER)
    ctx.synchronize()
    for i in range(T * INTER):
        hx[i] = Float16(Float64((i * 13) % 23 - 11) / 6.0)
    for i in range(HIDDEN * INTER):
        hw[i] = Float16(Float64((i * 7) % 11 - 5) / 4.0)

    var dx = ctx.enqueue_create_buffer[DType.float16](T * INTER)
    var dw = ctx.enqueue_create_buffer[DType.float16](HIDDEN * INTER)
    var dwp = ctx.enqueue_create_buffer[DType.float16](HIDDEN * INTER)
    var dout = ctx.enqueue_create_buffer[DType.float32](T * HIDDEN)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()
    ctx.enqueue_function[pack_w_mma_gpu[INTER, 2, 8]](
        dw.unsafe_ptr(), dwp.unsafe_ptr(),
        grid_dim=(HIDDEN // 128, INTER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()

    comptime NV = 4
    var times = InlineArray[InlineArray[Float64, NV], 10](
        fill=InlineArray[Float64, NV](fill=0.0)
    )
    for rep in range(10):
        @parameter
        for v in range(NV):
            var t0 = monotonic_ns()
            comptime if v == 0:
                ctx.enqueue_function[f16_mma_down_pk_gpu[INTER, 608, 2, 8]](
                    dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(HIDDEN), Int32(T),
                    grid_dim=(HIDDEN // 128, T // 16, 1),
                    block_dim=(256, 1, 1),
                )
            elif v == 1:
                ctx.enqueue_function[f16_mma_down_pk_gpu[INTER, 608, 4, 4]](
                    dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(HIDDEN), Int32(T),
                    grid_dim=(HIDDEN // 128, T // 16, 1),
                    block_dim=(128, 1, 1),
                )
            elif v == 2:
                ctx.enqueue_function[f16_mma_down_pk_gpu[INTER, 304, 2, 8]](
                    dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(HIDDEN), Int32(T),
                    grid_dim=(HIDDEN // 128, T // 16, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                ctx.enqueue_function[f16_mma_down_pk_gpu[INTER, 1216, 2, 8]](
                    dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(HIDDEN), Int32(T),
                    grid_dim=(HIDDEN // 128, T // 16, 1),
                    block_dim=(256, 1, 1),
                )
            ctx.synchronize()
            times[rep][v] = Float64(monotonic_ns() - t0) / 1e6
    for v in range(NV):
        var arr = InlineArray[Float64, 8](fill=0.0)
        for rep in range(2, 10):
            arr[rep - 2] = times[rep][v]
        for a in range(8):
            for b in range(a + 1, 8):
                if arr[b] < arr[a]:
                    var tmp = arr[a]
                    arr[a] = arr[b]
                    arr[b] = tmp
        print("cfg", v, "median ms:", arr[4])
