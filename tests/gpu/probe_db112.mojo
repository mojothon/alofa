"""账本一百一十二-b：down_pk 的 x smem 双缓冲（pk2）vs 原版 pk ——
真实 down 形状 INNER=4864→OUT=896、CH=608、OUTS=1、WARP=8，
8 套独立打包权重轮转（冷权重），20 发 ÷REP；另验逐位恒等。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.kernels.cuda.matmul import (
    f16_mma_down_pk_gpu,
    f16_mma_down_pk2_gpu,
    pack_w_mma_gpu,
)

comptime T = 128
comptime INNER = 4864
comptime OUT = 896
comptime NSET = 8
comptime REP = 10


def main() raises:
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float16](T * INNER)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](OUT * INNER)
    ctx.synchronize()
    for i in range(T * INNER):
        hx[i] = Float16(Float64((i * 13) % 23 - 11) / 16.0)
    for i in range(OUT * INNER):
        hw[i] = Float16(Float64((i * 7) % 11 - 5) / 32.0)

    var dx = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    var dw = ctx.enqueue_create_buffer[DType.float16](OUT * INNER)
    var dref = ctx.enqueue_create_buffer[DType.float32](T * OUT)
    var dnew = ctx.enqueue_create_buffer[DType.float32](T * OUT)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()

    var wpk = List[DeviceBuffer[DType.float16]]()
    for i in range(NSET):
        wpk.append(ctx.enqueue_create_buffer[DType.float16](OUT * INNER))
    for s in range(NSET):
        ctx.enqueue_function[pack_w_mma_gpu[INNER, 1, 8]](
            dw.unsafe_ptr(), wpk[s].unsafe_ptr(),
            grid_dim=(OUT // 64, INNER // 16, 1), block_dim=(128, 1, 1),
        )
    ctx.synchronize()

    var best = InlineArray[Float64, 2](fill=1e30)
    for rep in range(5):
        @parameter
        for v in range(2):
            var t0 = monotonic_ns()
            for r in range(REP):
                for s in range(NSET):
                    comptime if v == 0:
                        ctx.enqueue_function[
                            f16_mma_down_pk_gpu[INNER, 608, 1, 8]
                        ](
                            dx.unsafe_ptr(), wpk[s].unsafe_ptr(),
                            dref.unsafe_ptr(), Int32(OUT), Int32(T),
                            grid_dim=(OUT // 64, (T + 15) // 16, 1),
                            block_dim=(256, 1, 1),
                        )
                    else:
                        ctx.enqueue_function[
                            f16_mma_down_pk2_gpu[INNER, 608, 1, 8]
                        ](
                            dx.unsafe_ptr(), wpk[s].unsafe_ptr(),
                            dnew.unsafe_ptr(), Int32(OUT), Int32(T),
                            grid_dim=(OUT // 64, (T + 15) // 16, 1),
                            block_dim=(256, 1, 1),
                        )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP * NSET)
            if per < best[v]:
                best[v] = per
    print("pk us/call:", best[0], " pk2 us/call:", best[1],
          " speedup:", best[0] / best[1])

    for s in range(NSET):
        ctx.enqueue_function[f16_mma_down_pk_gpu[INNER, 608, 1, 8]](
            dx.unsafe_ptr(), wpk[s].unsafe_ptr(), dref.unsafe_ptr(),
            Int32(OUT), Int32(T),
            grid_dim=(OUT // 64, (T + 15) // 16, 1), block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[f16_mma_down_pk2_gpu[INNER, 608, 1, 8]](
            dx.unsafe_ptr(), wpk[s].unsafe_ptr(), dnew.unsafe_ptr(),
            Int32(OUT), Int32(T),
            grid_dim=(OUT // 64, (T + 15) // 16, 1), block_dim=(256, 1, 1),
        )
    var hA = ctx.enqueue_create_host_buffer[DType.float32](T * OUT)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](T * OUT)
    ctx.enqueue_copy(dst_buf=hA, src_buf=dref)
    ctx.enqueue_copy(dst_buf=hB, src_buf=dnew)
    ctx.synchronize()
    var mx = 0.0
    var nz = 0
    for i in range(T * OUT):
        var dv = Float64(hB[i]) - Float64(hA[i])
        if dv < 0:
            dv = -dv
        if dv > 0:
            nz += 1
        if dv > mx:
            mx = dv
    print("pk vs pk2 maxdiff:", mx, " nonzero:", nz)
