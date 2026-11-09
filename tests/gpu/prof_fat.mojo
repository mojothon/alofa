"""fat-CTA 选型（账本六十六）：OUTS 输出/CTA 摊薄 x 载入相位与 L2 流量。

冠军基线对照：q T32 12.0us、gate T32 31.9us、down T128 22.8us、
lm_head block[256] 1.02ms（六十五实测）。
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.core.dtype import DT_FP32
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.core.memory import Arena
from alofa.kernels.cpu.quant import quantize_q4_0
from alofa.kernels.cuda.matmul import q4_matmul_fat_gpu, q4_matmul_smem_gpu
from alofa.kernels.cuda.basic import linear_block_gpu, linear_fat_gpu

comptime WARMUP = 20
comptime ITERS = 200
comptime ROUNDS = 3


def _q4_upload(
    mut ctx: DeviceContext, mut arena: Arena, n_out: Int, n_in: Int
) raises -> DeviceBuffer[DType.uint8]:
    var n_blocks = n_out * n_in // 32
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n_blocks * 18)
    ctx.synchronize()
    var dims = List[Int]()
    dims.append(n_out)
    dims.append(n_in)
    var v = TensorView(arena.alloc(n_out * n_in * 4), Shape(dims), DT_FP32)
    for i in range(n_out * n_in):
        f32_data(v)[unsafe_offset=i] = Float32(i % 13) * 0.01 - 0.06
    quantize_q4_0(host.unsafe_ptr(), f32_data(v), n_blocks)
    var d = ctx.enqueue_create_buffer[DType.uint8](n_blocks * 18)
    ctx.enqueue_copy(dst_buf=d, src_buf=host)
    ctx.synchronize()
    return d


def _run_fat[THREADS: Int, INNER: Int, NOUT: Int, OUTS: Int](
    mut ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.uint8],
    mut dd: DeviceBuffer[DType.float32],
    mut dummy: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x_ptr = dx.unsafe_ptr()
    var w_ptr = dq.unsafe_ptr()
    var d_ptr = dd.unsafe_ptr()
    # HAS_BIAS=False 的 bias 占位：aliasing 规要求四个指针两两不同源
    var b_ptr = dummy.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_fat_gpu[THREADS, INNER, OUTS, False]](
            x_ptr, w_ptr, d_ptr, b_ptr,
            Int32(INNER), Int32(NOUT),
            grid_dim=(NOUT // OUTS, 1, 1), block_dim=(THREADS, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_fat_gpu[THREADS, INNER, OUTS, False]](
                x_ptr, w_ptr, d_ptr, b_ptr,
                Int32(INNER), Int32(NOUT),
                grid_dim=(NOUT // OUTS, 1, 1), block_dim=(THREADS, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    return best


def _run_champ[THREADS: Int, INNER: Int, NOUT: Int](
    mut ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.uint8],
    mut dd: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x_ptr = dx.unsafe_ptr()
    var w_ptr = dq.unsafe_ptr()
    var d_ptr = dd.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_smem_gpu[THREADS, INNER]](
            x_ptr, w_ptr, d_ptr,
            Int32(INNER), Int32(NOUT),
            grid_dim=(NOUT, 1, 1), block_dim=(THREADS, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_smem_gpu[THREADS, INNER]](
                x_ptr, w_ptr, d_ptr,
                Int32(INNER), Int32(NOUT),
                grid_dim=(NOUT, 1, 1), block_dim=(THREADS, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    return best


def _run_head[THREADS: Int, INNER: Int, OUTS: Int](
    mut ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32],
    mut dd: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x_ptr = dx.unsafe_ptr()
    var w_ptr = dw.unsafe_ptr()
    var d_ptr = dd.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[linear_fat_gpu[THREADS, INNER, OUTS]](
            x_ptr, w_ptr, d_ptr, Int32(151936),
            grid_dim=(151936 // OUTS, 1, 1), block_dim=(THREADS, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[linear_fat_gpu[THREADS, INNER, OUTS]](
                x_ptr, w_ptr, d_ptr, Int32(151936),
                grid_dim=(151936 // OUTS, 1, 1), block_dim=(THREADS, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
        if ms < best:
            best = ms
    return best


def _run_head_block[THREADS: Int, INNER: Int](
    mut ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32],
    mut dd: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x_ptr = dx.unsafe_ptr()
    var w_ptr = dw.unsafe_ptr()
    var d_ptr = dd.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[linear_block_gpu[THREADS]](
            x_ptr, w_ptr, d_ptr, Int32(INNER), Int32(151936),
            grid_dim=(151936, 1, 1), block_dim=(THREADS, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[linear_block_gpu[THREADS]](
                x_ptr, w_ptr, d_ptr, Int32(INNER), Int32(151936),
                grid_dim=(151936, 1, 1), block_dim=(THREADS, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
        if ms < best:
            best = ms
    return best


def main() raises:
    var ctx = DeviceContext()
    var arena = Arena(1 << 26)
    var dq_q = _q4_upload(ctx, arena, 896, 896)
    var dq_g = _q4_upload(ctx, arena, 4864, 896)
    var dq_d = _q4_upload(ctx, arena, 896, 4864)
    var dx = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd = ctx.enqueue_create_buffer[DType.float32](4864)
    # lm_head 权重 544MB f32
    var dw_head = ctx.enqueue_create_buffer[DType.float32](151936 * 896)
    var dummy = ctx.enqueue_create_buffer[DType.float32](4)
    ctx.synchronize()
    print("q 896x896:")
    print("  champ T32:      ", _run_champ[32, 896, 896](ctx, dx, dq_q, dd), " us")
    print("  fat T32 O2:     ", _run_fat[32, 896, 896, 2](ctx, dx, dq_q, dd, dummy), " us")
    print("  fat T32 O4:     ", _run_fat[32, 896, 896, 4](ctx, dx, dq_q, dd, dummy), " us")
    print("  fat T64 O4:     ", _run_fat[64, 896, 896, 4](ctx, dx, dq_q, dd, dummy), " us")
    print("gate 4864x896:")
    print("  champ T32:      ", _run_champ[32, 896, 4864](ctx, dx, dq_g, dd), " us")
    print("  fat T32 O2:     ", _run_fat[32, 896, 4864, 2](ctx, dx, dq_g, dd, dummy), " us")
    print("  fat T32 O4:     ", _run_fat[32, 896, 4864, 4](ctx, dx, dq_g, dd, dummy), " us")
    print("  fat T64 O4:     ", _run_fat[64, 896, 4864, 4](ctx, dx, dq_g, dd, dummy), " us")
    print("  fat T64 O8:     ", _run_fat[64, 896, 4864, 8](ctx, dx, dq_g, dd, dummy), " us")
    print("down 896x4864:")
    print("  champ T128:     ", _run_champ[128, 4864, 896](ctx, dx, dq_d, dd), " us")
    print("  fat T128 O2:    ", _run_fat[128, 4864, 896, 2](ctx, dx, dq_d, dd, dummy), " us")
    print("  fat T128 O4:    ", _run_fat[128, 4864, 896, 4](ctx, dx, dq_d, dd, dummy), " us")
    print("  fat T64 O4:     ", _run_fat[64, 4864, 896, 4](ctx, dx, dq_d, dd, dummy), " us")
    print("lm_head 151936x896:")
    print("  block T256:     ", _run_head_block[256, 896](ctx, dx, dw_head, dd), " ms")
    print("  fat T128 O16:   ", _run_head[128, 896, 16](ctx, dx, dw_head, dd), " ms")
    print("  fat T128 O32:   ", _run_head[128, 896, 32](ctx, dx, dw_head, dd), " ms")
    print("  fat T256 O8:    ", _run_head[256, 896, 8](ctx, dx, dw_head, dd), " ms")
    print("  fat T256 O16:   ", _run_head[256, 896, 16](ctx, dx, dw_head, dd), " ms")
    ctx.synchronize()
    arena.keep_alive()
