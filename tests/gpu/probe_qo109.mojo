"""账本一百一十-a：q2/o 的 896×896 GEMM 分块（冷权重轮转，GPU 1 满
时钟）。现配置 WARP=8/OUTS=2/CH=608 只有 7×8=56 块（半数 SM 空转）；
试 OUTS=1（COLS 64 → 112 块）与 CH=896（单段无尾余）。

用 f16_mma_down_pk_gpu 计时（与 q2 的 gemm_pk 同几何，仅差 bias，
bias 是 O(1) 尾部开销）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.kernels.cuda.matmul import f16_mma_down_pk_gpu, pack_w_mma_gpu

comptime T = 128
comptime INNER = 896
comptime OUT = 896
comptime NSET = 8
comptime REP = 10
comptime NCFG = 4


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
    var dout = ctx.enqueue_create_buffer[DType.float32](T * OUT)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()

    var wpk = List[DeviceBuffer[DType.float16]]()
    for i in range(NCFG * NSET):
        wpk.append(ctx.enqueue_create_buffer[DType.float16](OUT * INNER))

    @parameter
    for v in range(NCFG):
        comptime OS = 1 if v >= 2 else 2
        comptime COLS = 8 * OS * 8
        for s in range(NSET):
            ctx.enqueue_function[pack_w_mma_gpu[INNER, OS, 8]](
                dw.unsafe_ptr(), wpk[v * NSET + s].unsafe_ptr(),
                grid_dim=(OUT // COLS, INNER // 16, 1), block_dim=(128, 1, 1),
            )
    ctx.synchronize()

    var best = InlineArray[Float64, NCFG](fill=1e30)
    for rep in range(4):
        @parameter
        for v in range(NCFG):
            comptime OS = 1 if v >= 2 else 2
            comptime COLS = 8 * OS * 8
            comptime CHV = 896 if (v == 1 or v == 3) else 608
            var t0 = monotonic_ns()
            for r in range(REP):
                for s in range(NSET):
                    ctx.enqueue_function[
                        f16_mma_down_pk_gpu[INNER, CHV, OS, 8]
                    ](
                        dx.unsafe_ptr(), wpk[v * NSET + s].unsafe_ptr(),
                        dout.unsafe_ptr(), Int32(OUT), Int32(T),
                        grid_dim=(OUT // COLS, (T + 15) // 16, 1),
                        block_dim=(256, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP * NSET)
            if per < best[v]:
                best[v] = per
    @parameter
    for v in range(NCFG):
        comptime OS = 1 if v >= 2 else 2
        comptime CHV = 896 if (v == 1 or v == 3) else 608
        print("cfg", v, " OUTS:", OS, " COLS:", 8 * OS * 8, " CH:", CHV,
              " blocks:", OUT // (8 * OS * 8) * 8, " us/call:", best[v])
