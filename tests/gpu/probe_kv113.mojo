"""账本一百一十三-b：f16_mma_gemm2_pk_gpu vs 标量 f16_bmm_kv2_gpu
—— 真实 kv2 形状 inner=896、kv_dim=128、t=128。验 k/v 两块输出的
数值一致性 + 计时。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.kernels.cuda.matmul import (
    cat2_f16_gpu,
    cat2_f32_gpu,
    f16_bmm_kv2_gpu,
    f16_mma_gemm2_pk_gpu,
    pack_w_mma_gpu,
    x_to_f16_gpu,
)

comptime T = 128
comptime INNER = 896
comptime KVD = 128
comptime NSET = 4
comptime REP = 20


def main() raises:
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    var hwk = ctx.enqueue_create_host_buffer[DType.float16](KVD * INNER)
    var hwv = ctx.enqueue_create_host_buffer[DType.float16](KVD * INNER)
    var hbk = ctx.enqueue_create_host_buffer[DType.float32](KVD)
    var hbv = ctx.enqueue_create_host_buffer[DType.float32](KVD)
    ctx.synchronize()
    for i in range(T * INNER):
        hx[i] = Float32(Float64((i * 13) % 23 - 11) / 16.0)
    for i in range(KVD * INNER):
        hwk[i] = Float16(Float64((i * 7) % 11 - 5) / 32.0)
        hwv[i] = Float16(Float64((i * 5) % 17 - 8) / 6.0)
    for i in range(KVD):
        hbk[i] = Float32(Float64((i * 3) % 9 - 4) / 8.0)
        hbv[i] = Float32(Float64((i * 11) % 13 - 6) / 8.0)

    var dx = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dx16 = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    var dwk = ctx.enqueue_create_buffer[DType.float16](KVD * INNER)
    var dwv = ctx.enqueue_create_buffer[DType.float16](KVD * INNER)
    var dbk = ctx.enqueue_create_buffer[DType.float32](KVD)
    var dbv = ctx.enqueue_create_buffer[DType.float32](KVD)
    var kvraw = ctx.enqueue_create_buffer[DType.float16](2 * KVD * INNER)
    var wpkv = ctx.enqueue_create_buffer[DType.float16](2 * KVD * INNER)
    var bkv = ctx.enqueue_create_buffer[DType.float32](2 * KVD)
    var kb_ref = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    var vb_ref = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    var kb_new = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    var vb_new = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dwk, src_buf=hwk)
    ctx.enqueue_copy(dst_buf=dwv, src_buf=hwv)
    ctx.enqueue_copy(dst_buf=dbk, src_buf=hbk)
    ctx.enqueue_copy(dst_buf=dbv, src_buf=hbv)
    ctx.synchronize()

    # 拼接打包（同 session 逻辑）
    ctx.enqueue_function[cat2_f16_gpu](
        dwk.unsafe_ptr(), dwv.unsafe_ptr(), kvraw.unsafe_ptr(),
        Int32(KVD * INNER),
        grid_dim=(2 * KVD * INNER // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
        kvraw.unsafe_ptr(), wpkv.unsafe_ptr(),
        grid_dim=(2 * KVD // 128, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[cat2_f32_gpu](
        dbk.unsafe_ptr(), dbv.unsafe_ptr(), bkv.unsafe_ptr(), Int32(KVD),
        grid_dim=(2 * KVD // 256 + 1, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # 参照：标量 kv2（prefill_rb=4，tri_t 同 session？先取 256 一档）
    ctx.enqueue_function[f16_bmm_kv2_gpu[64, INNER, 2, 4, DType.float32]](
        dx.unsafe_ptr(), dwk.unsafe_ptr(), dwv.unsafe_ptr(),
        dbk.unsafe_ptr(), dbv.unsafe_ptr(),
        kb_ref.unsafe_ptr(), vb_ref.unsafe_ptr(),
        Int32(KVD), Int32(T),
        grid_dim=(KVD // 2, T // 4, 1), block_dim=(64, 1, 1),
    )
    # 新：f16 x + 双输出 MMA
    ctx.enqueue_function[x_to_f16_gpu](
        dx.unsafe_ptr(), dx16.unsafe_ptr(), Int32(T * INNER),
        grid_dim=(T * INNER // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[f16_mma_gemm2_pk_gpu[INNER, 896, 8]](
        dx16.unsafe_ptr(), wpkv.unsafe_ptr(),
        kb_new.unsafe_ptr(), vb_new.unsafe_ptr(), bkv.unsafe_ptr(),
        Int32(KVD), Int32(T),
        grid_dim=(2 * KVD // 128, T // 16, 1), block_dim=(256, 1, 1),
    )
    var hA = ctx.enqueue_create_host_buffer[DType.float32](T * KVD)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](T * KVD)
    ctx.enqueue_copy(dst_buf=hA, src_buf=kb_ref)
    ctx.enqueue_copy(dst_buf=hB, src_buf=kb_new)
    ctx.synchronize()
    var mxk = 0.0
    for i in range(T * KVD):
        var d = Float64(hB[i]) - Float64(hA[i])
        if d < 0:
            d = -d
        if d > mxk:
            mxk = d
    ctx.enqueue_copy(dst_buf=hA, src_buf=vb_ref)
    ctx.enqueue_copy(dst_buf=hB, src_buf=vb_new)
    ctx.synchronize()
    var mxv = 0.0
    for i in range(T * KVD):
        var d = Float64(hB[i]) - Float64(hA[i])
        if d < 0:
            d = -d
        if d > mxv:
            mxv = d
    print("k maxdiff:", mxk, " v maxdiff:", mxv)

    # 计时（冷权重：4 套打包轮转）
    var wpks = List[DeviceBuffer[DType.float16]]()
    for s in range(NSET):
        wpks.append(ctx.enqueue_create_buffer[DType.float16](2 * KVD * INNER))
        ctx.enqueue_function[cat2_f16_gpu](
            dwk.unsafe_ptr(), dwv.unsafe_ptr(), kvraw.unsafe_ptr(),
            Int32(KVD * INNER),
            grid_dim=(2 * KVD * INNER // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
            kvraw.unsafe_ptr(), wpks[s].unsafe_ptr(),
            grid_dim=(2 * KVD // 128, INNER // 16, 1), block_dim=(128, 1, 1),
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
                            f16_bmm_kv2_gpu[64, INNER, 2, 4, DType.float32]
                        ](
                            dx.unsafe_ptr(), dwk.unsafe_ptr(),
                            dwv.unsafe_ptr(), dbk.unsafe_ptr(),
                            dbv.unsafe_ptr(), kb_ref.unsafe_ptr(),
                            vb_ref.unsafe_ptr(), Int32(KVD), Int32(T),
                            grid_dim=(KVD // 2, T // 4, 1),
                            block_dim=(64, 1, 1),
                        )
                    else:
                        ctx.enqueue_function[
                            f16_mma_gemm2_pk_gpu[INNER, 896, 8]
                        ](
                            dx16.unsafe_ptr(), wpks[s].unsafe_ptr(),
                            kb_new.unsafe_ptr(), vb_new.unsafe_ptr(),
                            bkv.unsafe_ptr(), Int32(KVD), Int32(T),
                            grid_dim=(2 * KVD // 128, T // 16, 1),
                            block_dim=(64, 1, 1),
                        )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP * NSET)
            if per < best[v]:
                best[v] = per
    print("scalar us/call:", best[0], " mma2 us/call:", best[1],
          " speedup:", best[0] / best[1])
