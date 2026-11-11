"""账本一百零七-a：w 打包布局（fragment 连续 256B）对 down/dual 的
影响——计时 + 数值逐位恒等验证（同 mma 序，diff 必须 = 0）。
生产形状：down t=128/INNER=4864/HIDDEN=896；dual t=128/INNER=896/INTER=4864。
WARP=8、CH=608、down OUTS=2（COLS=128）。"""

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

    # --- down 数据 ---
    var hx = ctx.enqueue_create_host_buffer[DType.float16](T * INTER)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](HIDDEN * INTER)
    # --- dual 数据 ---
    var hxf = ctx.enqueue_create_host_buffer[DType.float16](T * HIDDEN)
    var hg = ctx.enqueue_create_host_buffer[DType.float16](INTER * HIDDEN)
    var hu = ctx.enqueue_create_host_buffer[DType.float16](INTER * HIDDEN)
    ctx.synchronize()
    for i in range(T * INTER):
        hx[i] = Float16(Float64((i * 13) % 23 - 11) / 6.0)
    for i in range(HIDDEN * INTER):
        hw[i] = Float16(Float64((i * 7) % 11 - 5) / 4.0)
    for i in range(T * HIDDEN):
        hxf[i] = Float16(Float64((i * 5) % 17 - 8) / 5.0)
    for i in range(INTER * HIDDEN):
        hg[i] = Float16(Float64((i * 3) % 13 - 6) / 3.0)
        hu[i] = Float16(Float64((i * 11) % 19 - 9) / 7.0)

    var dx = ctx.enqueue_create_buffer[DType.float16](T * INTER)
    var dw = ctx.enqueue_create_buffer[DType.float16](HIDDEN * INTER)
    var dwp = ctx.enqueue_create_buffer[DType.float16](HIDDEN * INTER)
    var dxf = ctx.enqueue_create_buffer[DType.float16](T * HIDDEN)
    var dg = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var dgp = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var du = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var dup = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var dout = ctx.enqueue_create_buffer[DType.float32](T * HIDDEN)
    var dact = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dga = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dgb = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.enqueue_copy(dst_buf=dxf, src_buf=hxf)
    ctx.enqueue_copy(dst_buf=dg, src_buf=hg)
    ctx.enqueue_copy(dst_buf=du, src_buf=hu)
    ctx.synchronize()

    # --- 打包（一次性）---
    var tp0 = monotonic_ns()
    ctx.enqueue_function[pack_w_mma_gpu[INTER, 2, WARP]](
        dw.unsafe_ptr(), dwp.unsafe_ptr(),
        grid_dim=(HIDDEN // 128, INTER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[HIDDEN, 2, WARP]](
        dg.unsafe_ptr(), dgp.unsafe_ptr(),
        grid_dim=(INTER // 128, HIDDEN // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[HIDDEN, 2, WARP]](
        du.unsafe_ptr(), dup.unsafe_ptr(),
        grid_dim=(INTER // 128, HIDDEN // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()
    print("pack 3 矩阵 ms:", Float64(monotonic_ns() - tp0) / 1e6)

    comptime NV = 4
    var times = InlineArray[InlineArray[Float64, NV], 12](
        fill=InlineArray[Float64, NV](fill=0.0)
    )
    for rep in range(12):
        @parameter
        for v in range(NV):
            var t0 = monotonic_ns()
            comptime if v == 0:
                ctx.enqueue_function[f16_mma_down_gpu[INTER, 608, 2, WARP]](
                    dx.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(HIDDEN), Int32(T),
                    grid_dim=(HIDDEN // 128, T // 16, 1),
                    block_dim=(256, 1, 1),
                )
            elif v == 1:
                ctx.enqueue_function[f16_mma_down_pk_gpu[INTER, 608, 2, WARP]](
                    dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(HIDDEN), Int32(T),
                    grid_dim=(HIDDEN // 128, T // 16, 1),
                    block_dim=(256, 1, 1),
                )
            elif v == 2:
                ctx.enqueue_function[f16_mma_dual_gpu[HIDDEN, 608, WARP]](
                    dxf.unsafe_ptr(), dg.unsafe_ptr(), du.unsafe_ptr(),
                    dga.unsafe_ptr(), dgb.unsafe_ptr(), dact.unsafe_ptr(),
                    Int32(INTER), Int32(T),
                    grid_dim=(INTER // 128, T // 16, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                ctx.enqueue_function[f16_mma_dual_pk_gpu[HIDDEN, 608, WARP]](
                    dxf.unsafe_ptr(), dgp.unsafe_ptr(), dup.unsafe_ptr(),
                    dga.unsafe_ptr(), dgb.unsafe_ptr(), dact.unsafe_ptr(),
                    Int32(INTER), Int32(T),
                    grid_dim=(INTER // 128, T // 16, 1),
                    block_dim=(256, 1, 1),
                )
            ctx.synchronize()
            times[rep][v] = Float64(monotonic_ns() - t0) / 1e6
    for v in range(NV):
        var arr = InlineArray[Float64, 10](fill=0.0)
        for rep in range(2, 12):
            arr[rep - 2] = times[rep][v]
        for a in range(10):
            for b in range(a + 1, 10):
                if arr[b] < arr[a]:
                    var tmp = arr[a]
                    arr[a] = arr[b]
                    arr[b] = tmp
        print("cfg", v, "median ms:", arr[5])

    # --- 数值逐位恒等：down ---
    var hA = ctx.enqueue_create_host_buffer[DType.float32](T * HIDDEN)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](T * HIDDEN)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](T * HIDDEN)
    for i in range(T * HIDDEN):
        hz[i] = Float32(0)
    ctx.enqueue_copy(dst_buf=dout, src_buf=hz)
    ctx.synchronize()
    ctx.enqueue_function[f16_mma_down_gpu[INTER, 608, 2, WARP]](
        dx.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
        Int32(HIDDEN), Int32(T),
        grid_dim=(HIDDEN // 128, T // 16, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hA, src_buf=dout)
    ctx.enqueue_copy(dst_buf=dout, src_buf=hz)
    ctx.synchronize()
    ctx.enqueue_function[f16_mma_down_pk_gpu[INTER, 608, 2, WARP]](
        dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
        Int32(HIDDEN), Int32(T),
        grid_dim=(HIDDEN // 128, T // 16, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hB, src_buf=dout)
    ctx.synchronize()
    var mxd = 0.0
    for i in range(T * HIDDEN):
        var dv = Float64(hB[i]) - Float64(hA[i])
        if dv < 0:
            dv = -dv
        if dv > mxd:
            mxd = dv
    print("down packed-vs-orig maxdiff:", mxd)

    # --- 数值逐位恒等：dual（act）---
    var hactA = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    var hactB = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    var hz2 = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    for i in range(T * INTER):
        hz2[i] = Float32(0)
    ctx.enqueue_copy(dst_buf=dact, src_buf=hz2)
    ctx.synchronize()
    ctx.enqueue_function[f16_mma_dual_gpu[HIDDEN, 608, WARP]](
        dxf.unsafe_ptr(), dg.unsafe_ptr(), du.unsafe_ptr(),
        dga.unsafe_ptr(), dgb.unsafe_ptr(), dact.unsafe_ptr(),
        Int32(INTER), Int32(T),
        grid_dim=(INTER // 128, T // 16, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hactA, src_buf=dact)
    ctx.enqueue_copy(dst_buf=dact, src_buf=hz2)
    ctx.synchronize()
    ctx.enqueue_function[f16_mma_dual_pk_gpu[HIDDEN, 608, WARP]](
        dxf.unsafe_ptr(), dgp.unsafe_ptr(), dup.unsafe_ptr(),
        dga.unsafe_ptr(), dgb.unsafe_ptr(), dact.unsafe_ptr(),
        Int32(INTER), Int32(T),
        grid_dim=(INTER // 128, T // 16, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hactB, src_buf=dact)
    ctx.synchronize()
    var mxa = 0.0
    for i in range(T * INTER):
        var dv = Float64(hactB[i]) - Float64(hactA[i])
        if dv < 0:
            dv = -dv
        if dv > mxa:
            mxa = dv
    print("dual packed-vs-orig act maxdiff:", mxa)
