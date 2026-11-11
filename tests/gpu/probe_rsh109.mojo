"""账本一百零九-h：attn_rs_h_gpu（kv_head 分组）vs attn_rs_b_gpu ——
t=128、kv=128、14 头 / 2 kv 头（group=7）。批量 20 连发单次 sync ÷20。
另验 scores 逐位同（分组不改变任一点积的算序）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.attention import attn_rs_b_gpu, attn_rs_h_gpu


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
    var dref = ctx.enqueue_create_buffer[DType.float32](T * HEADS * KL)
    var dnew = ctx.enqueue_create_buffer[DType.float32](T * HEADS * KL)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.synchronize()

    comptime NV = 2
    var best = InlineArray[Float64, NV](fill=1e30)
    for rep in range(5):
        @parameter
        for v in range(NV):
            var t0 = monotonic_ns()
            for r in range(REP):
                comptime if v == 0:
                    ctx.enqueue_function[attn_rs_b_gpu[512, True, 1]](
                        dq.unsafe_ptr(), dk.unsafe_ptr(), dref.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0),
                        grid_dim=(HEADS, 1, T), block_dim=(512, 1, 1),
                    )
                else:
                    ctx.enqueue_function[attn_rs_h_gpu[512, 8, True]](
                        dq.unsafe_ptr(), dk.unsafe_ptr(), dnew.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0),
                        grid_dim=(KVH, 1, T), block_dim=(512, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP)
            if per < best[v]:
                best[v] = per
    print("rs  us/call:", best[0], " rs_h  us/call:", best[1],
          " speedup:", best[0] / best[1])

    var hA = ctx.enqueue_create_host_buffer[DType.float32](T * HEADS * KL)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](T * HEADS * KL)
    ctx.enqueue_function[attn_rs_b_gpu[512, True, 1]](
        dq.unsafe_ptr(), dk.unsafe_ptr(), dref.unsafe_ptr(),
        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL), Int32(0),
        grid_dim=(HEADS, 1, T), block_dim=(512, 1, 1),
    )
    ctx.enqueue_function[attn_rs_h_gpu[512, 8, True]](
        dq.unsafe_ptr(), dk.unsafe_ptr(), dnew.unsafe_ptr(),
        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL), Int32(0),
        grid_dim=(KVH, 1, T), block_dim=(512, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hA, src_buf=dref)
    ctx.enqueue_copy(dst_buf=hB, src_buf=dnew)
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
    print("rs vs rs_h maxdiff:", mx, " nonzero:", nz)
