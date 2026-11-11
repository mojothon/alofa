"""账本一百零三：MMA 最小验证——m16n8k16 f16×f16→f32 fragment 映射
正确性（对 CPU 参照）+ down 形状（4864×896, t=128）简单 WMMA kernel
吞吐 ballpark。"""

from max.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu import AddressSpace, barrier
from std.ffi import external_call
from alofa.core.ffi.posix import monotonic_ns

from max.gpu.compute.mma import mma

comptime M = 16
comptime N = 8
comptime K = 16


def mma_check_gpu(
    a: Pointer[Float16, MutUntrackedOrigin],
    b: Pointer[Float16, MutUntrackedOrigin],
    c: Pointer[Float32, MutUntrackedOrigin],
    d: Pointer[Float32, MutUntrackedOrigin],
) -> None:
    """一个 warp 一块 16×8 输出：d = a(16×16)·b(16×8) + c。4 warp 独立。"""
    var lane = Int(thread_idx.x) % 32
    var warp = Int(thread_idx.x) // 32
    var gid = lane >> 2
    var tig = lane % 4
    var abase = warp * 256
    var av = SIMD[DType.float16, 8](
        a[abase + gid * 16 + tig * 2],
        a[abase + gid * 16 + tig * 2 + 1],
        a[abase + (gid + 8) * 16 + tig * 2],
        a[abase + (gid + 8) * 16 + tig * 2 + 1],
        a[abase + gid * 16 + tig * 2 + 8],
        a[abase + gid * 16 + tig * 2 + 9],
        a[abase + (gid + 8) * 16 + tig * 2 + 8],
        a[abase + (gid + 8) * 16 + tig * 2 + 9],
    )
    var bbase = warp * 128
    var bv = SIMD[DType.float16, 4](
        b[bbase + (tig * 2) * 8 + gid],
        b[bbase + (tig * 2 + 1) * 8 + gid],
        b[bbase + (tig * 2 + 8) * 8 + gid],
        b[bbase + (tig * 2 + 9) * 8 + gid],
    )
    var cv = SIMD[DType.float32, 4](
        c[warp * 128 + gid * 8 + tig * 2],
        c[warp * 128 + gid * 8 + tig * 2 + 1],
        c[warp * 128 + (gid + 8) * 8 + tig * 2],
        c[warp * 128 + (gid + 8) * 8 + tig * 2 + 1],
    )
    var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
    mma(dv, av, bv, cv)
    d[warp * 128 + gid * 8 + tig * 2] = dv[0]
    d[warp * 128 + gid * 8 + tig * 2 + 1] = dv[1]
    d[warp * 128 + (gid + 8) * 8 + tig * 2] = dv[2]
    d[warp * 128 + (gid + 8) * 8 + tig * 2 + 1] = dv[3]


def down_mma_gpu[
    WARPS: Int, INNER: Int, SMEM: Bool,
](
    x: Pointer[Float32, MutUntrackedOrigin],
    xh: Pointer[Float16, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """down 形状简单 WMMA：块 = 4 warp（16 行 × 32 列），x tile 转
    f16 进 smem，w fragment 直读 global。块覆盖 16×32 输出。
    grid = (ceildiv(out_cols, 32), ceildiv(n_rows, 16))。"""
    var cb = Int(block_idx.x) * 32
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    # x tile f16: 16 行 × inner 列（每行 inner*2 字节）
    var sxf = stack_allocation[
        16 * INNER, Float16, address_space = AddressSpace.SHARED
    ]()
    comptime if SMEM:
        var i = tid
        while i < 16 * INNER:
            sxf[unsafe_offset=i] = Float16(
                x[unsafe_offset=(r0 + i // INNER) * INNER + i % INNER]
            )
            i += 128
        barrier()

    # 每 warp 输出 16 行 × 8 列（块 4 warp → 32 列），K=inner 分 16 步进
    var col = cb + warp * 8
    var acc0 = SIMD[DType.float32, 4](0, 0, 0, 0)
    @parameter
    for ks in range(INNER // 16):
        var k = ks * 16
        var av: SIMD[DType.float16, 8]
        comptime if SMEM:
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
        var bv = SIMD[DType.float16, 4](
            w[unsafe_offset=(col + gid) * INNER + k + tig * 2],
            w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 1],
            w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 8],
            w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 9],
        )
        mma(acc0, av, bv, acc0)
    if r0 + gid < Int(n_rows):
        dst[unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2] = acc0[0]
        dst[unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1] = acc0[1]
    if r0 + gid + 8 < Int(n_rows):
        dst[unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2] = acc0[2]
        dst[unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1] = acc0[3]


def main() raises:
    var ctx = DeviceContext()

    # ---- 正确性：4 warp × (16×16)·(16×8)+c ----
    var nw = 4
    var ha = ctx.enqueue_create_host_buffer[DType.float16](nw * 256)
    var hb = ctx.enqueue_create_host_buffer[DType.float16](nw * 128)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](nw * 128)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](nw * 128)
    ctx.synchronize()
    for i in range(nw * 256):
        ha[i] = Float16(Float64((i * 7) % 13 - 6) / 4.0)
    for i in range(nw * 128):
        hb[i] = Float16(Float64((i * 5) % 11 - 5) / 3.0)
        hc[i] = Float32(Float64(i % 7) / 8.0)
    var da = ctx.enqueue_create_buffer[DType.float16](nw * 256)
    var db = ctx.enqueue_create_buffer[DType.float16](nw * 128)
    var dc = ctx.enqueue_create_buffer[DType.float32](nw * 128)
    var dd = ctx.enqueue_create_buffer[DType.float32](nw * 128)
    ctx.enqueue_copy(dst_buf=da, src_buf=ha)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.enqueue_copy(dst_buf=dc, src_buf=hc)
    ctx.synchronize()
    ctx.enqueue_function[mma_check_gpu](
        da.unsafe_ptr(), db.unsafe_ptr(), dc.unsafe_ptr(), dd.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()
    var mx = 0.0
    for w in range(nw):
        for r in range(16):
            for c in range(8):
                var rref = Float64(hc[w * 128 + r * 8 + c])
                for k in range(16):
                    rref += Float64(Float32(ha[w * 256 + r * 16 + k])) * Float64(
                        Float32(hb[w * 128 + k * 8 + c])
                    )
                var got = Float64(hd[w * 128 + r * 8 + c])
                var d = got - rref
                if d < 0:
                    d = -d
                if d > mx:
                    mx = d
    print("mma check maxdiff:", mx)

    # ---- down 形状吞吐 ----
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
    var hxh = ctx.enqueue_create_host_buffer[DType.float16](T * IN)
    for i in range(T * IN):
        hxh[i] = Float16(hx[i])
    var dxh = ctx.enqueue_create_buffer[DType.float16](T * IN)
    var dx = ctx.enqueue_create_buffer[DType.float32](T * IN)
    var dw = ctx.enqueue_create_buffer[DType.float16](OUTC * IN)
    var dout = ctx.enqueue_create_buffer[DType.float32](T * OUTC)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dxh, src_buf=hxh)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()

    # 预热 + 计时 20 rep
    for rep in range(22):
        var t0 = monotonic_ns()
        ctx.enqueue_function[down_mma_gpu[4, IN, True]](
            dx.unsafe_ptr(), dxh.unsafe_ptr(), dw.unsafe_ptr(), dout.unsafe_ptr(),
            Int32(OUTC), Int32(T),
            grid_dim=(OUTC // 32, T // 16, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        if rep >= 2:
            if rep == 2:
                print("down_mma 单次 us:", Float64(monotonic_ns() - t0) / 1e3)
            if rep == 21:
                var tavg = Float64(monotonic_ns() - t0) / 1e3
                print("down_mma 末次 us:", tavg)
    ctx.enqueue_copy(dst_buf=hout, src_buf=dout)
    ctx.synchronize()
    # 参照（f64 逐行）
    var mx2 = 0.0
    for r in range(0, T, 17):
        for c in range(0, OUTC, 37):
            var rref = 0.0
            for k in range(IN):
                rref += Float64(Float32(Float16(hx[r * IN + k]))) * Float64(
                    Float32(hw[c * IN + k])
                )
            var d = Float64(hout[r * OUTC + c]) - rref
            if d < 0:
                d = -d
            if d > mx2:
                mx2 = d
    print("down_mma maxdiff(稀疏):", mx2)
    var tflops = 2.0 * T * OUTC * IN / (328.0 * 1e-6) / 1e12
    print("对照: 现 down scalar 328us =", tflops, "TFLOPS")

    # ---- 小尺寸密集对照（T=16, OUTC=32, IN=32）定位错位模式 ----
    comptime T2 = 16
    comptime C2 = 32
    comptime K2 = 32
    var hx2 = ctx.enqueue_create_host_buffer[DType.float32](T2 * K2)
    var hw2 = ctx.enqueue_create_host_buffer[DType.float16](C2 * K2)
    var ho2 = ctx.enqueue_create_host_buffer[DType.float32](T2 * C2)
    ctx.synchronize()
    for i in range(T2 * K2):
        hx2[i] = Float32(Float64((i * 11) % 17 - 8) / 9.0)
    for i in range(C2 * K2):
        hw2[i] = Float16(Float64((i * 3) % 19 - 9) / 7.0)
    var hxh2 = ctx.enqueue_create_host_buffer[DType.float16](T2 * K2)
    for i in range(T2 * K2):
        hxh2[i] = Float16(hx2[i])
    var dxh2 = ctx.enqueue_create_buffer[DType.float16](T2 * K2)
    var dx2 = ctx.enqueue_create_buffer[DType.float32](T2 * K2)
    var dw2 = ctx.enqueue_create_buffer[DType.float16](C2 * K2)
    var do2 = ctx.enqueue_create_buffer[DType.float32](T2 * C2)
    ctx.enqueue_copy(dst_buf=dx2, src_buf=hx2)
    ctx.enqueue_copy(dst_buf=dxh2, src_buf=hxh2)
    ctx.enqueue_copy(dst_buf=dw2, src_buf=hw2)
    ctx.synchronize()
    ctx.enqueue_function[down_mma_gpu[4, K2, False]](
        dx2.unsafe_ptr(), dxh2.unsafe_ptr(), dw2.unsafe_ptr(), do2.unsafe_ptr(),
        Int32(C2), Int32(T2),
        grid_dim=(C2 // 32, T2 // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=ho2, src_buf=do2)
    ctx.synchronize()
    # want 矩阵前 2 行 + got 矩阵前 2 行
    for r in range(2):
        var wrow = ""
        var grow = ""
        for c in range(C2):
            var rr = 0.0
            for k in range(K2):
                rr += Float64(Float32(Float16(hx2[r * K2 + k]))) * Float64(
                    Float32(hw2[c * K2 + k])
                )
            wrow += String(Float64(rr), 4) + " "
            grow += String(Float64(ho2[r * C2 + c]), 4) + " "
        print("want r", r, ":", wrow)
        print("got  r", r, ":", grow)
    _ = external_call["usleep", Int32](Int32(1000))
