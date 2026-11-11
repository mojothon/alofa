"""账本一百一十三-a：o 的 split-K 扫描 —— o 形状 INNER=896→OUT=896、
OUTS=2（现 pk：CH=896、grid (7,8)=56 块，一半 SM 空转）。
扫 S=2(CH=448)/S=4(CH=224) vs pk；8 套冷权重轮转；另验数值
（pk 为 ACC；spl+adds[S] 同语义）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.kernels.cuda.matmul import (
    f16_mma_down_pk_gpu,
    f16_mma_down_spl_gpu,
    f32_adds_gpu,
    pack_w_mma_gpu,
)

comptime T = 128
comptime INNER = 896
comptime OUT = 896
comptime NSET = 8
comptime REP = 12
comptime NV = 3


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
    var dpart = ctx.enqueue_create_buffer[DType.float32](8 * T * OUT)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()

    var wpk = List[DeviceBuffer[DType.float16]]()
    for i in range(NSET):
        wpk.append(ctx.enqueue_create_buffer[DType.float16](OUT * INNER))
    for s in range(NSET):
        ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
            dw.unsafe_ptr(), wpk[s].unsafe_ptr(),
            grid_dim=(OUT // 128, INNER // 16, 1), block_dim=(128, 1, 1),
        )
    ctx.synchronize()

    var best = InlineArray[Float64, NV](fill=1e30)
    for rep in range(5):
        @parameter
        for v in range(NV):
            comptime SPL = 1 << v  # 1 / 2 / 4
            var t0 = monotonic_ns()
            for r in range(REP):
                for s in range(NSET):
                    comptime if SPL == 1:
                        ctx.enqueue_function[
                            f16_mma_down_pk_gpu[INNER, 896, 2, 8]
                        ](
                            dx.unsafe_ptr(), wpk[s].unsafe_ptr(),
                            dref.unsafe_ptr(), Int32(OUT), Int32(T),
                            grid_dim=(OUT // 128, T // 16, 1),
                            block_dim=(256, 1, 1),
                        )
                    else:
                        comptime if SPL == 2:
                            ctx.enqueue_function[
                                f16_mma_down_spl_gpu[INNER, 448, 2, 8, 2]
                            ](
                                dx.unsafe_ptr(), wpk[s].unsafe_ptr(),
                                dpart.unsafe_ptr(), Int32(OUT), Int32(T),
                                grid_dim=(OUT // 128, T // 16, 2),
                                block_dim=(256, 1, 1),
                            )
                        else:
                            ctx.enqueue_function[
                                f16_mma_down_spl_gpu[INNER, 224, 2, 8, 4]
                            ](
                                dx.unsafe_ptr(), wpk[s].unsafe_ptr(),
                                dpart.unsafe_ptr(), Int32(OUT), Int32(T),
                                grid_dim=(OUT // 128, T // 16, 4),
                                block_dim=(256, 1, 1),
                            )
                        ctx.enqueue_function[f32_adds_gpu[SPL]](
                            dnew.unsafe_ptr(), dpart.unsafe_ptr(),
                            Int32(T * OUT),
                            grid_dim=((T * OUT + 255) // 256, 1, 1),
                            block_dim=(256, 1, 1),
                        )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP * NSET)
            if per < best[v]:
                best[v] = per
    @parameter
    for v in range(NV):
        print("S =", 1 << v, " us/call:", best[v])
    print("speedup S4 vs pk:", best[0] / best[2])

    # 数值对照（干净缓冲 + 显式清零）
    var dref2 = ctx.enqueue_create_buffer[DType.float32](T * OUT)
    var dnew2 = ctx.enqueue_create_buffer[DType.float32](T * OUT)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](T * OUT)
    for i in range(T * OUT):
        hz[i] = 0.0
    ctx.enqueue_copy(dst_buf=dref2, src_buf=hz)
    ctx.enqueue_copy(dst_buf=dnew2, src_buf=hz)
    ctx.enqueue_function[f16_mma_down_pk_gpu[INNER, 896, 2, 8]](
        dx.unsafe_ptr(), wpk[0].unsafe_ptr(), dref2.unsafe_ptr(),
        Int32(OUT), Int32(T),
        grid_dim=(OUT // 128, T // 16, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[f16_mma_down_spl_gpu[INNER, 224, 2, 8, 4]](
        dx.unsafe_ptr(), wpk[0].unsafe_ptr(), dpart.unsafe_ptr(),
        Int32(OUT), Int32(T),
        grid_dim=(OUT // 128, T // 16, 4), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[f32_adds_gpu[4]](
        dnew2.unsafe_ptr(), dpart.unsafe_ptr(), Int32(T * OUT),
        grid_dim=((T * OUT + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    var hA = ctx.enqueue_create_host_buffer[DType.float32](T * OUT)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](T * OUT)
    ctx.enqueue_copy(dst_buf=hA, src_buf=dref2)
    ctx.enqueue_copy(dst_buf=hB, src_buf=dnew2)
    ctx.synchronize()
    var mx = 0.0
    var rel = 0.0
    for i in range(T * OUT):
        var dv = Float64(hB[i]) - Float64(hA[i])
        if dv < 0:
            dv = -dv
        if dv > mx:
            mx = dv
        var a = Float64(hA[i])
        if a < 0:
            a = -a
        if a > 1.0 and dv / a > rel:
            rel = dv / a
    print("pk vs spl4 maxdiff:", mx, " rel:", rel)
