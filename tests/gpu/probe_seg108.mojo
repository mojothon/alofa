"""账本一百零八-d：剩余段落的逐核批量微基准（t=128、kv_len=128、
pos0=0）——rmsnorm / x_to_f16 / kv2 / RS / VT2 / COMB，attention 三核
再扫 attn_t=512/256/128（t=128 时每块只有 t 个 j 活跃，512 线程空转
3/4）。20 连发单次 sync ÷ 20，取 5 轮最小。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.matmul import f16_bmm_kv2_gpu, x_to_f16_gpu
from alofa.kernels.cuda.basic import rmsnorm_gpu
from alofa.kernels.cuda.attention import (
    attn_rs_b_gpu,
    attn_vt2_b_gpu,
    attn_comb2_b_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime M = 896
    comptime HEADS = 14
    comptime KVH = 2
    comptime HD = 64
    comptime KL = 128
    comptime REP = 20

    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * M)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](M)
    var hw16 = ctx.enqueue_create_host_buffer[DType.float16](M * M)
    var hk = ctx.enqueue_create_host_buffer[DType.float16](KVH * HD * M)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](KVH * HD)
    ctx.synchronize()
    for i in range(T * M):
        hx[i] = Float32(Float64((i * 13) % 23 - 11) / 8.0)
    for i in range(M):
        hw[i] = Float32(1)
    for i in range(M * M):
        hw16[i] = Float16(Float64((i * 7) % 11 - 5) / 8.0)
    for i in range(KVH * HD * M):
        hk[i] = Float16(Float64((i * 5) % 17 - 8) / 9.0)
    for i in range(KVH * HD):
        hb[i] = Float32(0)

    var dx = ctx.enqueue_create_buffer[DType.float32](T * M)
    var dw = ctx.enqueue_create_buffer[DType.float32](M)
    var dq16 = ctx.enqueue_create_buffer[DType.float16](M * M)
    var dk16 = ctx.enqueue_create_buffer[DType.float16](KVH * HD * M)
    var dv16 = ctx.enqueue_create_buffer[DType.float16](KVH * HD * M)
    var db = ctx.enqueue_create_buffer[DType.float32](KVH * HD)
    var db2 = ctx.enqueue_create_buffer[DType.float32](KVH * HD)
    var dx16 = ctx.enqueue_create_buffer[DType.float16](T * M)
    var dnorm = ctx.enqueue_create_buffer[DType.float32](T * M)
    var dkb = ctx.enqueue_create_buffer[DType.float32](T * KVH * HD)
    var dvb = ctx.enqueue_create_buffer[DType.float32](T * KVH * HD)
    var dq = ctx.enqueue_create_buffer[DType.float32](T * HEADS * HD)
    var dks = ctx.enqueue_create_buffer[DType.float32](T * KVH * HD)
    var dvs = ctx.enqueue_create_buffer[DType.float32](T * KVH * HD)
    var dsc = ctx.enqueue_create_buffer[DType.float32](T * HEADS * KL)
    var dpart = ctx.enqueue_create_buffer[DType.float32](T * HEADS * HD)
    var dts = ctx.enqueue_create_buffer[DType.float64](T * HEADS)
    var dattn = ctx.enqueue_create_buffer[DType.float32](T * HEADS * HD)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.enqueue_copy(dst_buf=dq16, src_buf=hw16)
    ctx.enqueue_copy(dst_buf=dk16, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dv16, src_buf=hk)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.enqueue_copy(dst_buf=db2, src_buf=hb)
    ctx.synchronize()

    comptime NV = 8
    var best = InlineArray[Float64, NV](fill=1e30)
    for rep in range(5):
        @parameter
        for v in range(NV):
            var t0 = monotonic_ns()
            for r in range(REP):
                comptime if v == 0:
                    ctx.enqueue_function[rmsnorm_gpu](
                        dx.unsafe_ptr(), dw.unsafe_ptr(), dnorm.unsafe_ptr(),
                        Int32(M), Int32(T), Float32(1e-6),
                        grid_dim=(T, 1, 1), block_dim=(256, 1, 1),
                    )
                elif v == 1:
                    ctx.enqueue_function[x_to_f16_gpu](
                        dx.unsafe_ptr(), dx16.unsafe_ptr(), Int32(T * M),
                        grid_dim=((T * M + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 2:
                    ctx.enqueue_function[f16_bmm_kv2_gpu[128, M, 2, 8,
                        DType.float32]](
                        dx.unsafe_ptr(), dk16.unsafe_ptr(), dv16.unsafe_ptr(),
                        db.unsafe_ptr(), db2.unsafe_ptr(), dkb.unsafe_ptr(),
                        dvb.unsafe_ptr(), Int32(KVH * HD), Int32(T),
                        grid_dim=((KVH * HD + 1) // 2, (T + 7) // 8, 1),
                        block_dim=(128, 1, 1),
                    )
                elif v == 3:
                    ctx.enqueue_function[attn_rs_b_gpu[512, True, 1]](
                        dq.unsafe_ptr(), dks.unsafe_ptr(), dsc.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0),
                        grid_dim=(HEADS, 1, T), block_dim=(512, 1, 1),
                    )
                elif v == 4:
                    ctx.enqueue_function[attn_vt2_b_gpu[512, 1]](
                        dsc.unsafe_ptr(), dvs.unsafe_ptr(), dpart.unsafe_ptr(),
                        dts.unsafe_ptr(), Int32(HEADS), Int32(KVH),
                        Int32(HD), Int32(KL), Int32(0),
                        grid_dim=(HEADS, 1, T), block_dim=(512, 1, 1),
                    )
                elif v == 5:
                    ctx.enqueue_function[attn_comb2_b_gpu[1]](
                        dattn.unsafe_ptr(), dpart.unsafe_ptr(),
                        dts.unsafe_ptr(), Int32(HEADS), Int32(HD),
                        grid_dim=(HEADS, 1, T), block_dim=(512, 1, 1),
                    )
                elif v == 6:
                    ctx.enqueue_function[attn_rs_b_gpu[128, True, 1]](
                        dq.unsafe_ptr(), dks.unsafe_ptr(), dsc.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0),
                        grid_dim=(HEADS, 1, T), block_dim=(128, 1, 1),
                    )
                else:
                    ctx.enqueue_function[attn_vt2_b_gpu[128, 1]](
                        dsc.unsafe_ptr(), dvs.unsafe_ptr(), dpart.unsafe_ptr(),
                        dts.unsafe_ptr(), Int32(HEADS), Int32(KVH),
                        Int32(HD), Int32(KL), Int32(0),
                        grid_dim=(HEADS, 1, T), block_dim=(128, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP)
            if per < best[v]:
                best[v] = per
    @parameter
    for v in range(NV):
        print("cfg", v, " us/call:", best[v])
