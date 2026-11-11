"""账本一百一十一-a：attn_vt2_h_gpu（kv_head 分组）vs attn_vt2_b_gpu ——
t=128、kv=128、14 头/2 kv 头（group=7）、TILES=8。批量 20 连发单次
sync ÷20，另验 vpart/tsum 一致性；再补 nsc=1024 长序列用例验证跨步
exp 相位（chunk > TPP 时不得漏 key）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.attention import attn_vt2_b_gpu, attn_vt2_h_gpu


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime HEADS = 14
    comptime KVH = 2
    comptime HD = 64
    comptime KL = 128
    comptime TILES = 8
    comptime REP = 20

    var hs = ctx.enqueue_create_host_buffer[DType.float32](T * HEADS * KL)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](KL * KVH * HD)
    ctx.synchronize()
    for i in range(T * HEADS * KL):
        hs[i] = Float32(Float64((i * 7) % 13 - 6) / 4.0)
    for i in range(KL * KVH * HD):
        hv[i] = Float32(Float64((i * 5) % 17 - 8) / 6.0)

    var ds = ctx.enqueue_create_buffer[DType.float32](T * HEADS * KL)
    var dv = ctx.enqueue_create_buffer[DType.float32](KL * KVH * HD)
    var vref = ctx.enqueue_create_buffer[DType.float32](
        T * HEADS * TILES * HD
    )
    var vnew = ctx.enqueue_create_buffer[DType.float32](
        T * HEADS * TILES * HD
    )
    var tsref = ctx.enqueue_create_buffer[DType.float64](T * HEADS * TILES)
    var tsnew = ctx.enqueue_create_buffer[DType.float64](T * HEADS * TILES)
    ctx.enqueue_copy(dst_buf=ds, src_buf=hs)
    ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
    ctx.synchronize()

    comptime NV = 2
    var best = InlineArray[Float64, NV](fill=1e30)
    for rep in range(5):
        @parameter
        for v in range(NV):
            var t0 = monotonic_ns()
            for r in range(REP):
                comptime if v == 0:
                    ctx.enqueue_function[attn_vt2_b_gpu[512, TILES]](
                        ds.unsafe_ptr(), dv.unsafe_ptr(), vref.unsafe_ptr(),
                        tsref.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0),
                        grid_dim=(HEADS, TILES, T), block_dim=(512, 1, 1),
                    )
                else:
                    ctx.enqueue_function[attn_vt2_h_gpu[512, 8, TILES]](
                        ds.unsafe_ptr(), dv.unsafe_ptr(), vnew.unsafe_ptr(),
                        tsnew.unsafe_ptr(),
                        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL),
                        Int32(0),
                        grid_dim=(KVH, TILES, T), block_dim=(512, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP)
            if per < best[v]:
                best[v] = per
    print("vt2 us/call(ms):", best[0], " vt2_h(ms):", best[1],
          " speedup:", best[0] / best[1])

    ctx.enqueue_function[attn_vt2_b_gpu[512, TILES]](
        ds.unsafe_ptr(), dv.unsafe_ptr(), vref.unsafe_ptr(),
        tsref.unsafe_ptr(),
        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL), Int32(0),
        grid_dim=(HEADS, TILES, T), block_dim=(512, 1, 1),
    )
    ctx.enqueue_function[attn_vt2_h_gpu[512, 8, TILES]](
        ds.unsafe_ptr(), dv.unsafe_ptr(), vnew.unsafe_ptr(),
        tsnew.unsafe_ptr(),
        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL), Int32(0),
        grid_dim=(KVH, TILES, T), block_dim=(512, 1, 1),
    )
    var hA = ctx.enqueue_create_host_buffer[DType.float32](
        T * HEADS * TILES * HD
    )
    var hB = ctx.enqueue_create_host_buffer[DType.float32](
        T * HEADS * TILES * HD
    )
    var tA = ctx.enqueue_create_host_buffer[DType.float64](T * HEADS * TILES)
    var tB = ctx.enqueue_create_host_buffer[DType.float64](T * HEADS * TILES)
    ctx.enqueue_copy(dst_buf=hA, src_buf=vref)
    ctx.enqueue_copy(dst_buf=hB, src_buf=vnew)
    ctx.enqueue_copy(dst_buf=tA, src_buf=tsref)
    ctx.enqueue_copy(dst_buf=tB, src_buf=tsnew)
    ctx.synchronize()
    var mx = 0.0
    var mt = 0.0
    for i in range(T * HEADS * TILES * HD):
        var dvv = Float64(hB[i]) - Float64(hA[i])
        if dvv < 0:
            dvv = -dvv
        if dvv > mx:
            mx = dvv
    for i in range(T * HEADS * TILES):
        var dt = tB[i] - tA[i]
        if dt < 0:
            dt = -dt
        if dt > mt:
            mt = dt
    print("vpart maxdiff:", mx, " tsum maxdiff:", mt)
    long_case()


def long_case() raises:
    var ctx = DeviceContext()
    comptime T = 1
    comptime HEADS = 14
    comptime KVH = 2
    comptime HD = 64
    comptime KL = 1024
    comptime TILES = 8
    var hs = ctx.enqueue_create_host_buffer[DType.float32](T * HEADS * KL)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](KL * KVH * HD)
    ctx.synchronize()
    for i in range(T * HEADS * KL):
        hs[i] = Float32(Float64((i * 7) % 13 - 6) / 4.0)
    for i in range(KL * KVH * HD):
        hv[i] = Float32(Float64((i * 5) % 17 - 8) / 6.0)
    var ds = ctx.enqueue_create_buffer[DType.float32](T * HEADS * KL)
    var dv = ctx.enqueue_create_buffer[DType.float32](KL * KVH * HD)
    var vref = ctx.enqueue_create_buffer[DType.float32](
        T * HEADS * TILES * HD
    )
    var vnew = ctx.enqueue_create_buffer[DType.float32](
        T * HEADS * TILES * HD
    )
    var tsref = ctx.enqueue_create_buffer[DType.float64](T * HEADS * TILES)
    var tsnew = ctx.enqueue_create_buffer[DType.float64](T * HEADS * TILES)
    ctx.enqueue_copy(dst_buf=ds, src_buf=hs)
    ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
    ctx.synchronize()
    ctx.enqueue_function[attn_vt2_b_gpu[512, TILES]](
        ds.unsafe_ptr(), dv.unsafe_ptr(), vref.unsafe_ptr(),
        tsref.unsafe_ptr(),
        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL), Int32(1023),
        grid_dim=(HEADS, TILES, T), block_dim=(512, 1, 1),
    )
    ctx.enqueue_function[attn_vt2_h_gpu[512, 8, TILES]](
        ds.unsafe_ptr(), dv.unsafe_ptr(), vnew.unsafe_ptr(),
        tsnew.unsafe_ptr(),
        Int32(HEADS), Int32(KVH), Int32(HD), Int32(KL), Int32(1023),
        grid_dim=(KVH, TILES, T), block_dim=(512, 1, 1),
    )
    var hA = ctx.enqueue_create_host_buffer[DType.float32](
        T * HEADS * TILES * HD
    )
    var hB = ctx.enqueue_create_host_buffer[DType.float32](
        T * HEADS * TILES * HD
    )
    var tA = ctx.enqueue_create_host_buffer[DType.float64](T * HEADS * TILES)
    var tB = ctx.enqueue_create_host_buffer[DType.float64](T * HEADS * TILES)
    ctx.enqueue_copy(dst_buf=hA, src_buf=vref)
    ctx.enqueue_copy(dst_buf=hB, src_buf=vnew)
    ctx.enqueue_copy(dst_buf=tA, src_buf=tsref)
    ctx.enqueue_copy(dst_buf=tB, src_buf=tsnew)
    ctx.synchronize()
    var mx = 0.0
    var mt = 0.0
    for i in range(T * HEADS * TILES * HD):
        var dvv = Float64(hB[i]) - Float64(hA[i])
        if dvv < 0:
            dvv = -dvv
        if dvv > mx:
            mx = dvv
    for i in range(T * HEADS * TILES):
        var dt = tB[i] - tA[i]
        if dt < 0:
            dt = -dt
        if dt > mt:
            mt = dt
    print("long nsc=1024 vpart maxdiff:", mx, " tsum maxdiff:", mt)
