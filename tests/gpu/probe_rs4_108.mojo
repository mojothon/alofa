"""账本一百零八-e：attn_rs_b4_gpu（ROWS 行/块）对 attn_rs_b_gpu ——
t=128、kv_len=128、pos0=0、n_heads=14。计时（20 连发单次 sync ÷20，
5 轮最小，两种顺序各测） + 逐位恒等（scores 必须 maxdiff 0.0）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.attention import attn_rs_b_gpu, attn_rs_b4_gpu


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime HEADS = 14
    comptime KVH = 2
    comptime HD = 64
    comptime KL = 128
    comptime REP = 20

    var hq = ctx.enqueue_create_host_buffer[DType.float32](T * HEADS * HD)
    var hk = ctx.enqueue_create_host_buffer[DType.float32](KL * KVH * HD)
    ctx.synchronize()
    for i in range(T * HEADS * HD):
        hq[i] = Float32(Float64((i * 7) % 13 - 6) / 4.0)
    for i in range(KL * KVH * HD):
        hk[i] = Float32(Float64((i * 5) % 17 - 8) / 6.0)

    var dq = ctx.enqueue_create_buffer[DType.float32](T * HEADS * HD)
    var dk = ctx.enqueue_create_buffer[DType.float32](KL * KVH * HD)
    var dsc = ctx.enqueue_create_buffer[DType.float32](T * HEADS * KL)
    var dsc2 = ctx.enqueue_create_buffer[DType.float32](T * HEADS * KL)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](T * HEADS * KL)
    for i in range(T * HEADS * KL):
        hz[i] = Float32(0)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dsc, src_buf=hz)
    ctx.enqueue_copy(dst_buf=dsc2, src_buf=hz)
    ctx.synchronize()

    comptime NV = 3
    var best = InlineArray[Float64, NV](fill=1e30)
    for rep in range(5):
        @parameter
        for v in range(NV):
            var t0 = monotonic_ns()
            for r in range(REP):
                comptime if v == 0:
                    ctx.enqueue_function[attn_rs_b_gpu[512, True, 1]](
                        dq.unsafe_ptr(), dk.unsafe_ptr(), dsc.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0),
                        grid_dim=(HEADS, 1, T), block_dim=(512, 1, 1),
                    )
                elif v == 1:
                    ctx.enqueue_function[attn_rs_b4_gpu[512, 4, True]](
                        dq.unsafe_ptr(), dk.unsafe_ptr(), dsc2.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0), Int32(T),
                        grid_dim=(HEADS, 1, T // 4), block_dim=(512, 1, 1),
                    )
                else:
                    ctx.enqueue_function[attn_rs_b4_gpu[256, 2, True]](
                        dq.unsafe_ptr(), dk.unsafe_ptr(), dsc2.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0), Int32(T),
                        grid_dim=(HEADS, 1, T // 2), block_dim=(256, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP)
            if per < best[v]:
                best[v] = per
    @parameter
    for v in range(NV):
        print("cfg", v, " us/call:", best[v])

    # 逐位恒等：cfg0 已写 dsc；重跑 cfg1 写 dsc2，比较
    var hA = ctx.enqueue_create_host_buffer[DType.float32](T * HEADS * KL)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](T * HEADS * KL)
    ctx.enqueue_function[attn_rs_b_gpu[512, True, 1]](
        dq.unsafe_ptr(), dk.unsafe_ptr(), dsc.unsafe_ptr(),
        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL), Int32(0),
        grid_dim=(HEADS, 1, T), block_dim=(512, 1, 1),
    )
    ctx.enqueue_function[attn_rs_b4_gpu[512, 4, True]](
        dq.unsafe_ptr(), dk.unsafe_ptr(), dsc2.unsafe_ptr(),
        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL), Int32(0), Int32(T),
        grid_dim=(HEADS, 1, T // 4), block_dim=(512, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hA, src_buf=dsc)
    ctx.enqueue_copy(dst_buf=hB, src_buf=dsc2)
    ctx.synchronize()
    var mx = 0.0
    var nz = 0
    for i in range(T * HEADS * KL):
        var dv = Float64(hB[i]) - Float64(hA[i])
        if dv < 0:
            dv = -dv
        if dv > 0:
            nz += 1
        if dv > mx:
            mx = dv
    print("rs vs rs4 maxdiff:", mx, " nonzero:", nz)
