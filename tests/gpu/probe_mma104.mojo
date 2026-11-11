"""账本一百零四：MMA down kernel 深挖——x 预转 f16 + 多输出 tile + k 双链
展开。V0=NT1,U1（基线：消 smem 转换）；V1=NT1,U2；V2=NT2,U2；V3=NT4,U2。
统一 NT×32 列/块、4 warp、grid (OUTC/(32·NT), T/16)。"""

from max.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
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


def down_v[
    NT: Int, U: Int, INNER: Int,
](
    xh: Pointer[Float16, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """块 = 16 行 × (32·NT) 列，4 warp，每 warp NT 个 16×8 tile（间隔
    32 列）。k 按 U 个 16-tile 分组，偶链 accA/奇链 accB 交替（ILP），
    终值相加（f32 重组，非逐位）。A fragment 全局 f16 x（L2 驻留）。"""
    var cb = Int(block_idx.x) * (32 * NT)
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

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
            var av = SIMD[DType.float16, 8](
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
    # x 预转 f16（计时）
    var t0c = monotonic_ns()
    var nconv = (T * IN + 255) // 256
    ctx.enqueue_function[x_to_f16_gpu](
        dx.unsafe_ptr(), dxh.unsafe_ptr(), Int32(T * IN),
        grid_dim=(nconv, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    print("x 转 f16 us:", Float64(monotonic_ns() - t0c) / 1e3)

    # 预热 + 交替计时（12 轮 × 4 变体，抗漂移；取 min/median）
    comptime NV = 4
    var times = InlineArray[InlineArray[Float64, NV], 12](
        fill=InlineArray[Float64, NV](fill=0.0)
    )
    for rep in range(12):
        @parameter
        for v in range(NV):
            var tt = monotonic_ns()
            comptime if v == 0:
                ctx.enqueue_function[down_v[1, 1, IN]](
                    dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(OUTC), Int32(T),
                    grid_dim=(OUTC // 32, T // 16, 1), block_dim=(128, 1, 1),
                )
            elif v == 1:
                ctx.enqueue_function[down_v[1, 2, IN]](
                    dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(OUTC), Int32(T),
                    grid_dim=(OUTC // 32, T // 16, 1), block_dim=(128, 1, 1),
                )
            elif v == 2:
                ctx.enqueue_function[down_v[2, 2, IN]](
                    dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(OUTC), Int32(T),
                    grid_dim=(OUTC // 64, T // 16, 1), block_dim=(128, 1, 1),
                )
            else:
                ctx.enqueue_function[down_v[4, 2, IN]](
                    dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
                    Int32(OUTC), Int32(T),
                    grid_dim=(OUTC // 128, T // 16, 1), block_dim=(128, 1, 1),
                )
            ctx.synchronize()
            times[rep][v] = Float64(monotonic_ns() - tt) / 1e3
    comptime names = ("V0 NT1U1", "V1 NT1U2", "V2 NT2U2", "V3 NT4U2")
    @parameter
    for v in range(NV):
        var mn = 1e30
        var md = 0.0
        for rep in range(2, 12):
            if times[rep][v] < mn:
                mn = times[rep][v]
        # 中位数：简单冒泡取 rep2..11 第 5 小
        var arr = InlineArray[Float64, 10](fill=0.0)
        for rep in range(10):
            arr[rep] = times[rep + 2][v]
        for a in range(10):
            for b in range(a + 1, 10):
                if arr[b] < arr[a]:
                    var tmp = arr[a]
                    arr[a] = arr[b]
                    arr[b] = tmp
        md = arr[5]
        print(names[v], " min:", mn, " median:", md)

    # 正确性：V2（代表）稀疏 f64 参照（f16 舍入 x）
    ctx.enqueue_function[down_v[2, 2, IN]](
        dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
        Int32(OUTC), Int32(T),
        grid_dim=(OUTC // 64, T // 16, 1), block_dim=(128, 1, 1),
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
    print("V2 sparse maxdiff:", mx)
    _ = external_call["usleep", Int32](Int32(1000))
