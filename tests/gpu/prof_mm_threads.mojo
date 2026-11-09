"""THREADS 重调（账本六十五）：五十六的"inner≤1024→32"是分解实验语境下的
结论；decode 计时显示大 matmul 74µs/层 vs 内存下限 ~4µs。本基准在三个
真实形状（q 896x896、gate 4864x896、down 896x4864）下扫 THREADS
{32,64,128,256}，找每形状最优。
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.core.dtype import DT_FP32
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.core.memory import Arena
from alofa.kernels.cpu.quant import quantize_q4_0
from alofa.kernels.cuda.matmul import q4_matmul_smem_gpu

comptime WARMUP = 30
comptime ITERS = 300
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


def _run[THREADS: Int, INNER: Int, NOUT: Int](
    mut ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.uint8],
    mut dd: DeviceBuffer[DType.float32],
) raises -> Float64:
    # buffer 参数必须标 mut（borrowed 的 unsafe_ptr() 给 mut=False 指针，
    # 撞 enqueue_function 的约束重载）；指针在 callee 内现取（账本六十二/六十五）。
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


def main() raises:
    var ctx = DeviceContext()
    var arena = Arena(1 << 26)
    var dq_q = _q4_upload(ctx, arena, 896, 896)
    var dq_g = _q4_upload(ctx, arena, 4864, 896)
    var dq_d = _q4_upload(ctx, arena, 896, 4864)
    var dx = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd = ctx.enqueue_create_buffer[DType.float32](4864)
    ctx.synchronize()

    print("q 896x896 (floor ~0.3us):")
    print("  T32: ", _run[32, 896, 896](ctx, dx, dq_q, dd), " us")
    print("  T64: ", _run[64, 896, 896](ctx, dx, dq_q, dd), " us")
    print("  T128:", _run[128, 896, 896](ctx, dx, dq_q, dd), " us")
    print("  T256:", _run[256, 896, 896](ctx, dx, dq_q, dd), " us")
    print("gate 4864x896 (floor ~1.6us):")
    print("  T32: ", _run[32, 896, 4864](ctx, dx, dq_g, dd), " us")
    print("  T64: ", _run[64, 896, 4864](ctx, dx, dq_g, dd), " us")
    print("  T128:", _run[128, 896, 4864](ctx, dx, dq_g, dd), " us")
    print("  T256:", _run[256, 896, 4864](ctx, dx, dq_g, dd), " us")
    print("down 896x4864 (floor ~1.6us):")
    print("  T128:", _run[128, 4864, 896](ctx, dx, dq_d, dd), " us")
    ctx.synchronize()
    arena.keep_alive()
