"""账本一百零八-a：packed GEMM（q2/o 的 896×896 小 GEMM）微基准 +
与标量 q2/o 核的对照。形状 t=128、INNER=896、out=896。
cfg0 = f16_bmm_q2_gpu（标量 q2，带 bias）
cfg1 = f16_mma_gemm_pk_gpu[896,608,2,8,True]（packed MMA，带 bias）
cfg2 = f16_bmm_fat3_gpu（标量 o）
cfg3 = f16_mma_gemm_pk_gpu[896,608,2,8,False]（packed MMA，无 bias）
批量计时：20 连发单次 sync ÷ 20，取 5 轮最小。
数值：MMA 输出 vs 标量输出 maxdiff（f16 x 舍入预期 ~1e-3）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.matmul import (
    x_to_f16_gpu,
    f16_bmm_q2_gpu,
    f16_bmm_fat3_gpu,
    f16_mma_gemm_pk_gpu,
    pack_w_mma_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime M = 896
    comptime REP = 20

    var hx = ctx.enqueue_create_host_buffer[DType.float16](T * M)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](M * M)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](M)
    ctx.synchronize()
    for i in range(T * M):
        hx[i] = Float16(Float64((i * 13) % 23 - 11) / 8.0)
    for i in range(M * M):
        hw[i] = Float16(Float64((i * 7) % 11 - 5) / 8.0)
    for i in range(M):
        hb[i] = Float32(Float64((i % 5) - 2) / 16.0)

    var dx = ctx.enqueue_create_buffer[DType.float16](T * M)
    var dx32 = ctx.enqueue_create_buffer[DType.float32](T * M)
    var dw = ctx.enqueue_create_buffer[DType.float16](M * M)
    var dwp = ctx.enqueue_create_buffer[DType.float16](M * M)
    var db = ctx.enqueue_create_buffer[DType.float32](M)
    var dout = ctx.enqueue_create_buffer[DType.float32](T * M)
    var dref = ctx.enqueue_create_buffer[DType.float32](T * M)
    var dzero = ctx.enqueue_create_buffer[DType.float32](T * M)
    var hx32 = ctx.enqueue_create_host_buffer[DType.float32](T * M)
    for i in range(T * M):
        hx32[i] = Float32(hx[i])
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dx32, src_buf=hx32)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.synchronize()
    ctx.enqueue_function[pack_w_mma_gpu[M, 2, 8]](
        dw.unsafe_ptr(), dwp.unsafe_ptr(),
        grid_dim=(M // 128, M // 16, 1), block_dim=(128, 1, 1),
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
                    ctx.enqueue_function[f16_bmm_q2_gpu[128, M, 4, 8,
                        DType.float32]](
                        dx32.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(),
                        dout.unsafe_ptr(), Int32(M), Int32(T),
                        grid_dim=((M + 3) // 4, (T + 7) // 8, 1),
                        block_dim=(128, 1, 1),
                    )
                elif v == 1:
                    ctx.enqueue_function[
                        f16_mma_gemm_pk_gpu[M, 608, 2, 8, True]
                    ](
                        dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
                        db.unsafe_ptr(), Int32(M), Int32(T),
                        grid_dim=(M // 128, (T + 15) // 16, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 2:
                    ctx.enqueue_function[f16_bmm_fat3_gpu[128, M, 4, 8,
                        True, DType.float32]](
                        dx32.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                        Int32(M), Int32(T),
                        grid_dim=((M + 3) // 4, (T + 7) // 8, 1),
                        block_dim=(128, 1, 1),
                    )
                else:
                    ctx.enqueue_function[
                        f16_mma_gemm_pk_gpu[M, 608, 2, 8, False]
                    ](
                        dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
                        db.unsafe_ptr(), Int32(M), Int32(T),
                        grid_dim=(M // 128, (T + 15) // 16, 1),
                        block_dim=(256, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP)
            if per < best[v]:
                best[v] = per
    @parameter
    for v in range(NV):
        print("cfg", v, " us/call:", best[v])

    # 数值：cfg1（MMA+bias）对 cfg0（标量 q2）
    var hz = ctx.enqueue_create_host_buffer[DType.float32](T * M)
    for i in range(T * M):
        hz[i] = Float32(0)
    var hA = ctx.enqueue_create_host_buffer[DType.float32](T * M)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](T * M)
    ctx.enqueue_copy(dst_buf=dref, src_buf=hz)
    ctx.enqueue_copy(dst_buf=dout, src_buf=hz)
    ctx.synchronize()
    ctx.enqueue_function[f16_bmm_q2_gpu[128, M, 4, 8, DType.float32]](
        dx32.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(), dref.unsafe_ptr(),
        Int32(M), Int32(T),
        grid_dim=((M + 3) // 4, (T + 7) // 8, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[f16_mma_gemm_pk_gpu[M, 608, 2, 8, True]](
        dx.unsafe_ptr(), dwp.unsafe_ptr(), dout.unsafe_ptr(),
        db.unsafe_ptr(), Int32(M), Int32(T),
        grid_dim=(M // 128, (T + 15) // 16, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hA, src_buf=dref)
    ctx.enqueue_copy(dst_buf=hB, src_buf=dout)
    ctx.synchronize()
    var mx = 0.0
    var rel = 0.0
    for i in range(T * M):
        var dv = Float64(hB[i]) - Float64(hA[i])
        if dv < 0:
            dv = -dv
        if dv > mx:
            mx = dv
        var a = Float64(hA[i])
        if a < 0:
            a = -a
        if a > rel:
            rel = a
    print("q2 MMA-vs-scalar maxdiff:", mx, " max|ref|:", rel)
