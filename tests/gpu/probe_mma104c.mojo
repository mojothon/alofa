"""账本一百零四-c：f16_mma_dual_gpu（gate/up/act 三输出）与
f16_mma_down_gpu（ACC 残加）正确性 + dual 计时。"""

from max.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from std.math import exp
from std.ffi import external_call
from alofa.core.ffi.posix import monotonic_ns
from alofa.kernels.cuda.matmul import (
    f16_mma_down_gpu,
    f16_mma_dual_gpu,
    x_to_f16_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime INTER = 4864
    comptime INNER = 896
    comptime HIDDEN = 896

    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    var hwa = ctx.enqueue_create_host_buffer[DType.float16](INTER * INNER)
    var hwb = ctx.enqueue_create_host_buffer[DType.float16](INTER * INNER)
    var hwd = ctx.enqueue_create_host_buffer[DType.float16](INTER * HIDDEN)
    var hxa = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    ctx.synchronize()
    for i in range(T * INNER):
        hx[i] = Float32(Float64((i * 11) % 17 - 8) / 9.0)
    for i in range(INTER * INNER):
        hwa[i] = Float16(Float64((i * 3) % 19 - 9) / 7.0)
        hwb[i] = Float16(Float64((i * 5) % 13 - 6) / 5.0)
    for i in range(INTER * HIDDEN):
        hwd[i] = Float16(Float64((i * 7) % 11 - 5) / 4.0)
    for i in range(T * INTER):
        hxa[i] = Float32(Float64((i * 13) % 23 - 11) / 6.0)

    var dx = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dxh = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    var dwa = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwb = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwd = ctx.enqueue_create_buffer[DType.float16](INTER * HIDDEN)
    var dxa = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dxah = ctx.enqueue_create_buffer[DType.float16](T * INTER)
    var dga = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dub = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dac = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dxb = ctx.enqueue_create_buffer[DType.float32](T * HIDDEN)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dwa, src_buf=hwa)
    ctx.enqueue_copy(dst_buf=dwb, src_buf=hwb)
    ctx.enqueue_copy(dst_buf=dwd, src_buf=hwd)
    ctx.enqueue_copy(dst_buf=dxa, src_buf=hxa)
    ctx.synchronize()
    var nc = (T * INNER + 255) // 256
    ctx.enqueue_function[x_to_f16_gpu](
        dx.unsafe_ptr(), dxh.unsafe_ptr(), Int32(T * INNER),
        grid_dim=(nc, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # ---- dual MMA ----
    ctx.enqueue_function[f16_mma_dual_gpu[INNER, 608]](
        dxh.unsafe_ptr(), dwa.unsafe_ptr(), dwb.unsafe_ptr(),
        dga.unsafe_ptr(), dub.unsafe_ptr(), dac.unsafe_ptr(),
        Int32(INTER), Int32(T),
        grid_dim=(INTER // 64, T // 16, 1), block_dim=(128, 1, 1),
    )
    # ---- down MMA（ACC：dxb 先填 3.5，结果应 = 3.5 + A·W）----
    var hxb0 = ctx.enqueue_create_host_buffer[DType.float32](T * HIDDEN)
    ctx.synchronize()
    for i in range(T * HIDDEN):
        hxb0[i] = Float32(3.5)
    ctx.enqueue_copy(dst_buf=dxb, src_buf=hxb0)
    ctx.synchronize()
    var nc2 = (T * INTER + 255) // 256
    ctx.enqueue_function[x_to_f16_gpu](
        dxa.unsafe_ptr(), dxah.unsafe_ptr(), Int32(T * INTER),
        grid_dim=(nc2, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[f16_mma_down_gpu[INTER, 608]](
        dxah.unsafe_ptr(), dwd.unsafe_ptr(), dxb.unsafe_ptr(),
        Int32(HIDDEN), Int32(T),
        grid_dim=(HIDDEN // 128, T // 16, 1), block_dim=(128, 1, 1),
    )

    var hga = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    var hub = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    var hac = ctx.enqueue_create_host_buffer[DType.float32](T * INTER)
    var hxb = ctx.enqueue_create_host_buffer[DType.float32](T * HIDDEN)
    ctx.enqueue_copy(dst_buf=hga, src_buf=dga)
    ctx.enqueue_copy(dst_buf=hub, src_buf=dub)
    ctx.enqueue_copy(dst_buf=hac, src_buf=dac)
    ctx.enqueue_copy(dst_buf=hxb, src_buf=dxb)
    ctx.synchronize()

    # 稀疏 f64 参照
    var mxa = 0.0
    var mxu = 0.0
    var mxc = 0.0
    var mxd = 0.0
    for r in range(0, T, 17):
        for c in range(0, INTER, 37):
            var ra = 0.0
            var rb = 0.0
            for k in range(INNER):
                ra += Float64(Float32(Float16(hx[r * INNER + k]))) * Float64(
                    Float32(hwa[c * INNER + k])
                )
                rb += Float64(Float32(Float16(hx[r * INNER + k]))) * Float64(
                    Float32(hwb[c * INNER + k])
                )
            var da = Float64(hga[r * INTER + c]) - ra
            if da < 0:
                da = -da
            if da > mxa:
                mxa = da
            var du = Float64(hub[r * INTER + c]) - rb
            if du < 0:
                du = -du
            if du > mxu:
                mxu = du
            var want_c = ra / (1.0 + exp(-ra)) * rb
            var dc = Float64(hac[r * INTER + c]) - want_c
            if dc < 0:
                dc = -dc
            if dc > mxc:
                mxc = dc
    for r in range(0, T, 17):
        for c in range(0, HIDDEN, 41):
            var rd = 0.0
            for k in range(INTER):
                rd += Float64(Float32(Float16(hxa[r * INTER + k]))) * Float64(
                    Float32(hwd[c * INTER + k])
                )
            var dd = Float64(hxb[r * HIDDEN + c]) - (3.5 + rd)
            if dd < 0:
                dd = -dd
            if dd > mxd:
                mxd = dd
    print("dual gate maxdiff:", mxa)
    print("dual up   maxdiff:", mxu)
    print("dual act  maxdiff:", mxc)
    print("down ACC  maxdiff:", mxd)

    # dual 计时（10 rep）
    for rep in range(10):
        var tt = monotonic_ns()
        ctx.enqueue_function[f16_mma_dual_gpu[INNER, 608]](
            dxh.unsafe_ptr(), dwa.unsafe_ptr(), dwb.unsafe_ptr(),
            dga.unsafe_ptr(), dub.unsafe_ptr(), dac.unsafe_ptr(),
            Int32(INTER), Int32(T),
            grid_dim=(INTER // 64, T // 16, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        if rep == 9:
            print("dual MMA 末次 us:", Float64(monotonic_ns() - tt) / 1e3)
    _ = external_call["usleep", Int32](Int32(1000))
