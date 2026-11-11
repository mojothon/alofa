"""账本一百零九-e：t=128 下 down 的 MMA 分块在**满时钟 + 冷权重**下的
真收益。t=128 时 (WARP=8,OUTS=2) 只有 7×8=56 块 < 108 SM ⇒ 一半 SM
空转；更细的列块（COLS=64/32 → 112/224 块）可能翻正。

用 8 份**独立打包缓冲**轮转（每份 8.7MB，轮转即冷）模拟流水里的冷权重，
避免「20 发同一份权重」把 L2 加热造成的乐观偏差。

形状：t=128、INNER=896、out=4864（down）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.kernels.cuda.matmul import f16_mma_down_pk_gpu, pack_w_mma_gpu

comptime T = 128
comptime INNER = 4864
comptime OUT = 896
comptime NSET = 8
comptime REP = 6
comptime NCFG = 5


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
    var dxb = ctx.enqueue_create_buffer[DType.float32](T * OUT)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()

    var wpk = List[DeviceBuffer[DType.float16]]()
    for i in range(NCFG * NSET):
        wpk.append(ctx.enqueue_create_buffer[DType.float16](OUT * INNER))

    @parameter
    for v in range(NCFG):
        comptime WARPV = 4 if v == 1 or v == 3 else 8
        comptime OUTSV = 1 if v == 3 or v == 2 else 2
        comptime COLSV = WARPV * OUTSV * 8
        for s in range(NSET):
            ctx.enqueue_function[pack_w_mma_gpu[INNER, OUTSV, WARPV]](
                dw.unsafe_ptr(), wpk[v * NSET + s].unsafe_ptr(),
                grid_dim=(OUT // COLSV, INNER // 16, 1),
                block_dim=(128, 1, 1),
            )
    ctx.synchronize()

    var best = InlineArray[Float64, NCFG](fill=1e30)
    for rep in range(4):
        @parameter
        for v in range(NCFG):
            comptime WARPV = 4 if v == 1 or v == 3 else 8
            comptime OUTSV = 1 if v == 3 or v == 2 else 2
            comptime COLSV = WARPV * OUTSV * 8
            comptime CHV = 1216 if v == 4 else 608
            var t0 = monotonic_ns()
            for r in range(REP):
                for s in range(NSET):
                    ctx.enqueue_function[
                        f16_mma_down_pk_gpu[INNER, CHV, OUTSV, WARPV]
                    ](
                        dx.unsafe_ptr(), wpk[v * NSET + s].unsafe_ptr(),
                        dxb.unsafe_ptr(), Int32(OUT), Int32(T),
                        grid_dim=(OUT // COLSV, (T + 15) // 16, 1),
                        block_dim=(256, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP * NSET)
            if per < best[v]:
                best[v] = per
    @parameter
    for v in range(NCFG):
        comptime WARPV = 4 if v == 1 or v == 3 else 8
        comptime OUTSV = 1 if v == 3 or v == 2 else 2
        comptime CHV = 1216 if v == 4 else 608
        print("cfg", v, " WARP:", WARPV, " OUTS:", OUTSV, " COLS:",
              WARPV * OUTSV * 8, " blocks:",
              OUT // (WARPV * OUTSV * 8) * 8, " CH:", CHV,
              " us/call:", best[v])
