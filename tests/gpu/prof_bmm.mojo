"""down-bmm 单核扫型（账本九十）：110ms prefill 拆解第一步。
down 形状（inner=4864, out=896, t=128 行）。理论对照：f64 FMA
1.1 GFLOP/launch @9.7T = 115µs；w 流量 ceil(128/RB)×8.7MB @1050
GB/s（RB8=139MB → 132µs）；x 流量 128×896×4864×4B=2.2GB L2。
若实测 ≫ 两者 max → 访存/占用率病；若 ≈ max → matmul 无罪，
110ms 的凶手在 attention/gather/norm。
（注：泛型 _run 包装触发编译器实例化失败，四臂内联展开。）
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu import AddressSpace, barrier
from alofa.kernels.cuda.matmul import f16_bmm_fat_gpu, f16_matmul_fat_gpu

comptime INNER = 4864
comptime OUT = 896
comptime T = 128
comptime WARMUP = 10
comptime ITERS = 50




def f16_bmm_tiled_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, CH: Int, ACC: Bool](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """smem x-tile GEMM 探针（账本九十）：x RB×CH 块进 smem，块内
    OUTS×RB 全复用；w 每 RB 行流式一次。x 冷读 = out_blocks×RB×INNER
    ×4B（O8 = 279MB vs 现在 2.2GB）。f32 acc（8 FMA/cycle/SM 级吞吐
    足够）。CH=256 → x tile 8KB + sacc 32KB = 40KB 静态 smem。"""
    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[
        RB * CH, Float32, address_space = AddressSpace.SHARED
    ]()
    var acc = SIMD[DType.float32, OUTS * RB](0)
    @parameter
    for kc in range((INNER + CH - 1) // CH):
        var xbase = kc * CH
        var idx = tid
        while idx < RB * CH:
            var rr = idx // CH
            var cc = idx - rr * CH
            var r = r0 + rr
            if r < Int(n_rows):
                if xbase + cc < Int(INNER):
                    sx[unsafe_offset=idx] = x[
                        unsafe_offset=r * Int(INNER) + xbase + cc
                    ]
            idx += THREADS
        barrier()
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                var w_base = o * Int(INNER)
                var k = xbase + tid * 8
                var kend = xbase + CH
                if kend > Int(INNER):
                    kend = Int(INNER)
                while k < kend:
                    var wv = w.load[8](w_base + k)
                    @parameter
                    for rr in range(RB):
                        var xv = sx.load[8](rr * CH + k - xbase)
                        var prod = xv * wv.cast[DType.float32]()
                        acc[oo * RB + rr] += prod.reduce_add()
                    k += THREADS * 8
        barrier()
    var sacc = stack_allocation[
        OUTS * RB * THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx2 in range(OUTS * RB):
        sacc[unsafe_offset=idx2 * THREADS + tid] = acc[idx2]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx2 in range(OUTS * RB):
                sacc[unsafe_offset=idx2 * THREADS + tid] += sacc[
                    unsafe_offset=idx2 * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = sacc[unsafe_offset=(oo * RB + rr) * THREADS]
                        comptime if ACC:
                            dst[unsafe_offset=r * Int(out_cols) + o] = (
                                dst[unsafe_offset=r * Int(out_cols) + o] + v
                            )
                        else:
                            dst[unsafe_offset=r * Int(out_cols) + o] = v



def f16_bmm_sacc_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """acc-smem 探针（账本九十）：验证 SIMD lane 累加 lowering 假说
    —— acc 放 smem（[lane][tid] 布局，warp 无 bank 冲突），每 FMA
    smem RMW。若 ≫ 快于寄存器 lane 版 ⇒ 假说成立。"""
    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var sacc_w = stack_allocation[
        OUTS * RB * THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(OUTS * RB):
        sacc_w[unsafe_offset=idx * THREADS + tid] = Float32(0)
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        if o < Int(out_cols):
            var w_base = o * Int(INNER)
            var k = tid * 8
            while k < Int(INNER):
                var wv = w.load[8](w_base + k)
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var xv = x.load[8](r * Int(INNER) + k)
                        @parameter
                        for j in range(8):
                            sacc_w[unsafe_offset=(oo * RB + rr) * THREADS + tid] += (
                                xv[j] * Float32(wv[j])
                            )
                k += THREADS * 8
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(OUTS * RB):
                sacc_w[unsafe_offset=idx * THREADS + tid] += sacc_w[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        dst[unsafe_offset=r * Int(out_cols) + o] = sacc_w[
                            unsafe_offset=(oo * RB + rr) * THREADS
                        ]



def f16_bmm_vec_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, ACC: Bool](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """vec 载入探针（账本九十一）：bitcast 到 SIMD 指针再 load，
    强制 32B/16B 对齐向量载入（对照 load[8] 的 8×b32 标量灾难）。"""
    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[DType.float32, OUTS * RB](0)
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        if o < Int(out_cols):
            var w_base = o * Int(INNER)
            var k = tid * 8
            while k < Int(INNER):
                var wv = w.load[8, alignment=16](w_base + k)
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                        var prod = xv * wv.cast[DType.float32]()
                        acc[oo * RB + rr] += prod.reduce_add()
                k += THREADS * 8
    var sacc = stack_allocation[
        OUTS * RB * THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = sacc[unsafe_offset=(oo * RB + rr) * THREADS]
                        comptime if ACC:
                            dst[unsafe_offset=r * Int(out_cols) + o] = (
                                dst[unsafe_offset=r * Int(out_cols) + o] + v
                            )
                        else:
                            dst[unsafe_offset=r * Int(out_cols) + o] = v



def f16_bmm_vec2_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, ACC: Bool](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """vec + 标量 j 累加探针（账本九十一）：向量载入 + 无 reduce_add
    树（每元素 1 FMA 直入 acc lane）。"""
    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[DType.float32, OUTS * RB](0)
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        if o < Int(out_cols):
            var w_base = o * Int(INNER)
            var k = tid * 8
            while k < Int(INNER):
                var wv = w.load[8, alignment=16](w_base + k)
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                        @parameter
                        for j in range(8):
                            acc[oo * RB + rr] += Scalar[DType.float32](
                                xv[j] * Float32(wv[j])
                            )
                k += THREADS * 8
    var sacc = stack_allocation[
        OUTS * RB * THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = sacc[unsafe_offset=(oo * RB + rr) * THREADS]
                        comptime if ACC:
                            dst[unsafe_offset=r * Int(out_cols) + o] = (
                                dst[unsafe_offset=r * Int(out_cols) + o] + v
                            )
                        else:
                            dst[unsafe_offset=r * Int(out_cols) + o] = v

def main() raises:
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var w = ctx.enqueue_create_buffer[DType.float16](OUT * INNER)
    var dst = ctx.enqueue_create_buffer[DType.float32](T * OUT)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](OUT * INNER)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](T * OUT)
    ctx.synchronize()
    for i in range(T * INNER):
        hx[i] = Float32((i % 17) - 8) * 0.01
    for i in range(OUT * INNER):
        hw[i] = Float16((i % 13) - 6) * 0.01
    for i in range(T * OUT):
        hd[i] = Float32(0)
    ctx.enqueue_copy(dst_buf=x, src_buf=hx)
    ctx.enqueue_copy(dst_buf=w, src_buf=hw)
    ctx.enqueue_copy(dst_buf=dst, src_buf=hd)
    ctx.synchronize()

    var dst2 = ctx.enqueue_create_buffer[DType.float32](OUT)
    ctx.synchronize()
    var xp = x.unsafe_ptr()
    var wp = w.unsafe_ptr()
    var dp = dst.unsafe_ptr()

    # 臂 1: T128 RB8 f64
    var best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, INNER, 2, 8, True, DType.float64]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("T128 RB8 f64:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 2: T128 RB8 f32
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, INNER, 2, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("T128 RB8 f32:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 3: T256 RB8 f32
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[256, INNER, 2, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 16, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("T256 RB8 f32:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 4: T128 RB4 f64（w-passes 32，对照带宽墙）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, INNER, 2, 4, True, DType.float64]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 32, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("T128 RB4 f64:", best, "us/launch  有效bw=", Float64(32 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 5: T128 RB8 OUTS8 f32（x L2 流量 ÷4 假说验证）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, INNER, 8, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 8, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("T128 RB8 O8 f32:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 7: tiled O8 RB8 CH256 f32
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_tiled_gpu[128, INNER, 8, 8, 256, True]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 8, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("tiled O8 RB8 CH256:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + (OUT // 8) * 16 * 8 * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 8: tiled O2 RB8 CH256 f32
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_tiled_gpu[128, INNER, 2, 8, 256, True]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("tiled O2 RB8 CH256:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + (OUT // 2) * 16 * 8 * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 9: tiled T64 O8 RB8 CH512 f32（CH=64*8 全线程勤；smem 16+16=32KB）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_tiled_gpu[64, INNER, 8, 8, 512, True]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 8, 16, 1), block_dim=(64, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("tiled T64 O8 RB8 CH512:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + (OUT // 8) * 16 * 8 * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 11: decode 原版 fat 核对照（smem x，1 行，grid 448×1）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_matmul_fat_gpu[128, INNER, 2, False, True]](
                xp, wp, dp, dst2.unsafe_ptr(), Int32(INNER), Int32(OUT),
                grid_dim=(OUT // 2, 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("decode-fat 1row:", best, "us/launch  有效bw=", Float64(OUT * INNER * 2 + INNER * 4 + OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 12: bmm RB=1 特例（全局 x，1 行，ACC=False）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, INNER, 2, 1, False, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(1),
                grid_dim=(OUT // 2, 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("bmm RB1 1row:", best, "us/launch  有效bw=", Float64(OUT * INNER * 2 + INNER * 4 + OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 13: bmm RB=8 但 n_rows=1（同 128 块 8 守卫空转——隔离 grid.y 因子）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, INNER, 2, 8, False, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(1),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("bmm RB8 n1:", best, "us/launch")

    # 臂 14: RB8 acc(16 lanes) 但 grid.y=1（448 块）——分 acc 与块数
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, INNER, 2, 8, False, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(1),
                grid_dim=(OUT // 2, 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("bmm RB8 n1 gridY1:", best, "us/launch")

    # 臂 15: RB1 acc(2 lanes) 但 grid.y=16（7168 块）——纯块数因子
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, INNER, 2, 1, False, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(1),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("bmm RB1 gridY16:", best, "us/launch")

    # 臂 16: acc-smem O2 RB8 f32（对照寄存器 lane 版 1372µs）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_sacc_gpu[128, INNER, 2, 8]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("sacc O2 RB8 f32:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 17: tiled T128 RB8 CH1024 O2 f32（CH=128*8 全勤；smem 32+16=48KB 压线）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_tiled_gpu[128, INNER, 2, 8, 1024, True]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("tiled T128 RB8 CH1024:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 18: vec 载入 O2 RB8 f32（对照标量版 1372µs）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_vec_gpu[128, INNER, 2, 8, True]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("vec O2 RB8 f32:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # 臂 19: vec + 标量 j 累加（对照 vec 树版 876µs）
    best = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_vec2_gpu[128, INNER, 2, 8, True]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(OUT // 2, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("vec2 scalar-j O2 RB8 f32:", best, "us/launch  有效bw=", Float64(16 * OUT * INNER * 2 + T * INNER * 4 + T * OUT * 4) / (best * 1e-6) / 1e9, "GB/s")

    # ===== 九十四：INNER=896（o 形状）THREADS 扫描——单 k 迭代延迟暴露定位 =====
    # w/x/dst 缓冲复用（o 只用前 896×896 与 128×896）。迭代数 = ceil(896/(T*8)):
    # T128=1, T112=1(无闲线程), T64=2, T32=4, T16=7。ACC=True 同生产 o 核。
    var b8 = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < b8:
            b8 = us
    print("o-shape T128 (1it):", b8, "us/launch")

    var b9 = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[112, 896, 2, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(448, 16, 1), block_dim=(112, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < b9:
            b9 = us
    print("o-shape T112 (1it 全勤):", b9, "us/launch")

    var b10 = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[64, 896, 2, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(448, 16, 1), block_dim=(64, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < b10:
            b10 = us
    print("o-shape T64 (2it):", b10, "us/launch")

    var b11 = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[32, 896, 2, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(448, 16, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < b11:
            b11 = us
    print("o-shape T32 (4it):", b11, "us/launch")

    var b12 = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[16, 896, 2, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(448, 16, 1), block_dim=(16, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < b12:
            b12 = us
    print("o-shape T16 (7it):", b12, "us/launch")

    var b13 = Float64(1e30)
    for r in range(3):
        ctx.synchronize()
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 8, 8, True, DType.float32]](
                xp, wp, dp, Int32(OUT), Int32(T),
                grid_dim=(112, 16, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < b13:
            b13 = us
    print("o-shape T128 O8 (1it 16LD ILP):", b13, "us/launch")
