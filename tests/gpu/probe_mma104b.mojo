"""账本一百零四-b：V3 基线复测 + V4（smem x + NT4）+ V5（NT4 U4）+
V6（NT8）。V4 的 x tile 16×896 f16 = 28KB smem，块内一次转换。"""

from max.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu import AddressSpace, barrier
from std.ffi import external_call
from alofa.core.ffi.posix import monotonic_ns
from max.gpu.compute.mma import mma


def x_to_f16_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    xh: Pointer[Float16, MutUntrackedOrigin],
    n: Int32,
) -> None:
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        xh[i] = Float16(x[i])


def down_g[
    NT: Int, U: Int, INNER: Int, SMEMX: Bool,
](
    xh: Pointer[Float16, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    var cb = Int(block_idx.x) * (32 * NT)
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * INNER, Float16, address_space = AddressSpace.SHARED
    ]()
    comptime if SMEMX:
        var i = tid
        while i < 16 * INNER:
            sxf[unsafe_offset=i] = xh[unsafe_offset=r0 * INNER + i]
            i += 128
        barrier()

    var accA = InlineArray[SIMD[DType.float32, 4], NT](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], NT](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ks in range(INNER // (16 * U)):
        @parameter
        for u in range(U):
            var k = (ks * U + u) * 16
            var av: SIMD[DType.float16, 8]
            comptime if SMEMX:
                av = SIMD[DType.float16, 8](
                    sxf[unsafe_offset=gid * INNER + k + tig * 2],
                    sxf[unsafe_offset=gid * INNER + k + tig * 2 + 1],
                    sxf[unsafe_offset=(gid + 8) * INNER + k + tig * 2],
                    sxf[unsafe_offset=(gid + 8) * INNER + k + tig * 2 + 1],
                    sxf[unsafe_offset=gid * INNER + k + tig * 2 + 8],
                    sxf[unsafe_offset=gid * INNER + k + tig * 2 + 9],
                    sxf[unsafe_offset=(gid + 8) * INNER + k + tig * 2 + 8],
                    sxf[unsafe_offset=(gid + 8) * INNER + k + tig * 2 + 9],
                )
            else:
                av = SIMD[DType.float16, 8](
                    xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
                    xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
                    xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
                    xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
                    xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
                    xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
                    xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
                    xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
                )
            @parameter
            for j in range(NT):
                var col = cb + warp * 8 + j * 32
                var bv = SIMD[DType.float16, 4](
                    w[unsafe_offset=(col + gid) * INNER + k + tig * 2],
                    w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 1],
                    w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 8],
                    w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 9],
                )
                var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                comptime if u % 2 == 0:
                    mma(dv, av, bv, accA[j])
                    accA[j] = dv
                else:
                    mma(dv, av, bv, accB[j])
                    accB[j] = dv

    @parameter
    for j in range(NT):
        var col = cb + warp * 8 + j * 32
        var acc = accA[j] + accB[j]
        if r0 + gid < Int(n_rows):
            dst[unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2] = acc[0]
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = acc[1]
        if r0 + gid + 8 < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = acc[2]
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = acc[3]


def main() raises:
    var ctx = DeviceContext()
    comptime T = 128
    comptime OUTC = 4864
    comptime IN = 896

    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * IN)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](OUTC * IN)
    var hout = ctx.enqueue_create_host_buffer[DType.float32](T * OUTC)
    ctx.synchronize()
    for i in range(T * IN):
        hx[i] = Float32(Float64((i * 11) % 17 - 8) / 9.0)
    for i in range(OUTC * IN):
        hw[i] = Float16(Float64((i * 3) % 19 - 9) / 7.0)

    var dx = ctx.enqueue_create_buffer[DType.float32](T * IN)
    var dxh = ctx.enqueue_create_buffer[DType.float16](T * IN)
    var dw = ctx.enqueue_create_buffer[DType.float16](OUTC * IN)
    var dout = ctx.enqueue_create_buffer[DType.float32](T * OUTC)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()
    var nconv = (T * IN + 255) // 256
    ctx.enqueue_function[x_to_f16_gpu](
        dx.unsafe_ptr(), dxh.unsafe_ptr(), Int32(T * IN),
        grid_dim=(nconv, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # V3 作为第一发吃 JIT（排除编译干扰），然后交替计时
    ctx.enqueue_function[down_g[4, 2, IN, False]](
        dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
        Int32(OUTC), Int32(T),
        grid_dim=(OUTC // 128, T // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()

    comptime NV = 4
    var times = InlineArray[InlineArray[Float64, NV], 12](
        fill=InlineArray[Float64, NV](fill=0.0)
    )
    for rep in range(12):
        @parameter
        for v in range(NV):
            var tt = monotonic_ns()
            comptime if v == 0:
                # V3 复测：NT4 U2 全局 x
                ctx.enqueue_function[down_g[4, 2, IN, False]](
                    dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(OUTC), Int32(T),
                    grid_dim=(OUTC // 128, T // 16, 1), block_dim=(128, 1, 1),
                )
            elif v == 1:
                # V4：NT4 U2 smem x
                ctx.enqueue_function[down_g[4, 2, IN, True]](
                    dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(OUTC), Int32(T),
                    grid_dim=(OUTC // 128, T // 16, 1), block_dim=(128, 1, 1),
                )
            elif v == 2:
                # V5：NT4 U4 smem x
                ctx.enqueue_function[down_g[4, 4, IN, True]](
                    dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(OUTC), Int32(T),
                    grid_dim=(OUTC // 128, T // 16, 1), block_dim=(128, 1, 1),
                )
            else:
                # V6：NT8 U2 全局 x
                ctx.enqueue_function[down_g[8, 2, IN, False]](
                    dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(OUTC), Int32(T),
                    grid_dim=(OUTC // 256, T // 16, 1), block_dim=(128, 1, 1),
                )
            ctx.synchronize()
            times[rep][v] = Float64(monotonic_ns() - tt) / 1e3
    comptime names = ("V3 g NT4U2", "V4 s NT4U2", "V5 s NT4U4", "V6 g NT8U2")
    @parameter
    for v in range(NV):
        var mn = 1e30
        var arr = InlineArray[Float64, 10](fill=0.0)
        for rep in range(10):
            arr[rep] = times[rep + 2][v]
            if arr[rep] < mn:
                mn = arr[rep]
        for a in range(10):
            for b in range(a + 1, 10):
                if arr[b] < arr[a]:
                    var tmp = arr[a]
                    arr[a] = arr[b]
                    arr[b] = tmp
        print(names[v], " min:", mn, " median:", arr[5])

    # 正确性：V4（代表）
    ctx.enqueue_function[down_g[4, 2, IN, True]](
        dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
        Int32(OUTC), Int32(T),
        grid_dim=(OUTC // 128, T // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hout, src_buf=dout)
    ctx.synchronize()
    var mx = 0.0
    for r in range(0, T, 17):
        for c in range(0, OUTC, 37):
            var rr = 0.0
            for k in range(IN):
                rr += Float64(Float32(Float16(hx[r * IN + k]))) * Float64(
                    Float32(hw[c * IN + k])
                )
            var d = Float64(hout[r * OUTC + c]) - rr
            if d < 0:
                d = -d
            if d > mx:
                mx = d
    print("V4 sparse maxdiff:", mx)
    _ = external_call["usleep", Int32](Int32(1000))
