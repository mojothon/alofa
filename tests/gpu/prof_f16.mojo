"""f16 预反量化 vs q4-load[4] 探针（账本六十八）：方向决策数据。

基线 = 六十七 session 在用：gate dual-fat T64/O4 23.0us、down fat T128/O2
26.9us（gate+up 合跑）。探针：A=f16 GEMV（8/16 宽）、B=q4 32B 块 load[4]。
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.core.dtype import DT_FP32
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.core.memory import Arena
from alofa.kernels.cpu.quant import dequant_q4_0, quantize_q4_0
from alofa.kernels.cuda.matmul import (
    f16_matmul_fat_gpu,
    q4_matmul_dual_fat_gpu,
    q4_matmul_fat_gpu,
)

comptime WARMUP = 20
comptime ITERS = 200
comptime ROUNDS = 3


def _upload_q4(mut ctx: DeviceContext, mut arena: Arena, n_out: Int, n_in: Int) raises -> DeviceBuffer[DType.uint8]:
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


def _upload_f16(mut ctx: DeviceContext, mut arena: Arena, n_out: Int, n_in: Int) raises -> DeviceBuffer[DType.float16]:
    """q4 -> f16 预反量化（装载期路径模拟）。"""
    var n = n_out * n_in
    var n_blocks = n // 32
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n_blocks * 18)
    ctx.synchronize()
    var dims = List[Int]()
    dims.append(n_out)
    dims.append(n_in)
    var v = TensorView(arena.alloc(n * 4), Shape(dims), DT_FP32)
    for i in range(n):
        f32_data(v)[unsafe_offset=i] = Float32(i % 13) * 0.01 - 0.06
    quantize_q4_0(host.unsafe_ptr(), f32_data(v), n)
    var dq = arena.alloc(n * 4)
    dequant_q4_0(dq.unsafe_bitcast[Float32](), host.unsafe_ptr(), n)
    var dq32 = dq.unsafe_bitcast[Float32]()
    var h16 = ctx.enqueue_create_host_buffer[DType.float16](n)
    ctx.synchronize()
    for i in range(n):
        h16[i] = Float16(dq32[unsafe_offset=i])
    var d = ctx.enqueue_create_buffer[DType.float16](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=h16)
    ctx.synchronize()
    return d


def _run_f16[THREADS: Int, INNER: Int, NOUT: Int, OUTS: Int](
    mut ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float16],
    mut dd: DeviceBuffer[DType.float32],
    mut dd2: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x_ptr = dx.unsafe_ptr()
    var w_ptr = dw.unsafe_ptr()
    var d_ptr = dd.unsafe_ptr()
    # HAS_BIAS=False 的 bias 占位：第二个真实缓冲（mut 参数互异可过；与
    # 函数内新建的局部缓冲反而冲突 —— aliasing 检查的怪癖，账本六十八）
    var b_ptr = dd2.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[f16_matmul_fat_gpu[THREADS, INNER, OUTS, False]](
            x_ptr, w_ptr, d_ptr, b_ptr,
            Int32(INNER), Int32(NOUT),
            grid_dim=(NOUT // OUTS, 1, 1), block_dim=(THREADS, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_matmul_fat_gpu[THREADS, INNER, OUTS, False]](
                x_ptr, w_ptr, d_ptr, b_ptr,
                Int32(INNER), Int32(NOUT),
                grid_dim=(NOUT // OUTS, 1, 1), block_dim=(THREADS, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    return best


def main() raises:
    var ctx = DeviceContext()
    var arena = Arena(1 << 28)
    var dq_ga = _upload_q4(ctx, arena, 4864, 896)
    var dq_gb = _upload_q4(ctx, arena, 4864, 896)
    var dq_da = _upload_q4(ctx, arena, 896, 4864)
    var dq_db = _upload_q4(ctx, arena, 896, 4864)
    var dw_g = _upload_f16(ctx, arena, 4864, 896)
    var dw_d = _upload_f16(ctx, arena, 896, 4864)
    var dx = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd = ctx.enqueue_create_buffer[DType.float32](4864)
    ctx.synchronize()
    var dummy = ctx.enqueue_create_buffer[DType.float32](4)
    print("== gate+up 4864x896 x2 (dual 基线) ==")
    # dual/fat_down 探针内联：mut 参数派生 origin 被视为一类，helper 形态过不了
    # aliasing 检查；main 局部变量形态与门文件同款（六十五起反复验证）。
    var dd2 = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd3 = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd4 = ctx.enqueue_create_buffer[DType.float32](4864)
    var xg = dx.unsafe_ptr()
    var wag = dq_ga.unsafe_ptr()
    var wbg = dq_gb.unsafe_ptr()
    var dg = dd.unsafe_ptr()
    var dg2 = dd2.unsafe_ptr()
    var dg3 = dd3.unsafe_ptr()
    var dg4 = dd4.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_dual_fat_gpu[64, 896, 4, False]](
            xg, wag, wbg, dg, dg2, dg3, dg4,
            Int32(896), Int32(4864),
            grid_dim=(1216, 1, 1), block_dim=(64, 1, 1),
        )
    ctx.synchronize()
    var best_g = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_dual_fat_gpu[64, 896, 4, False]](
                xg, wag, wbg, dg, dg2, dg3, dg4,
                Int32(896), Int32(4864),
                grid_dim=(1216, 1, 1), block_dim=(64, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best_g:
            best_g = us
    print("  dual T64 O4 (champ):   ", best_g, " us")
    print("  f16 T64 O4:            ", _run_f16[64, 896, 4864, 4](ctx, dx, dw_g, dd, dd2), " us")
    print("  f16 T128 O4:           ", _run_f16[128, 896, 4864, 4](ctx, dx, dw_g, dd, dd2), " us")
    print("  f16 T256 O4:           ", _run_f16[256, 896, 4864, 4](ctx, dx, dw_g, dd, dd2), " us")
    print("  f16 T128 O2:           ", _run_f16[128, 896, 4864, 2](ctx, dx, dw_g, dd, dd2), " us")
    print("== down 896x4864 (fat 基线) ==")
    var xd = dx.unsafe_ptr()
    var wd = dq_da.unsafe_ptr()
    var dgd = dd.unsafe_ptr()
    var dgd2 = dd2.unsafe_ptr()
    var dgd3 = dd3.unsafe_ptr()
    var dgd4 = dd4.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_fat_gpu[128, 4864, 2, False]](
            xd, wd, dgd, dgd2,
            Int32(4864), Int32(896),
            grid_dim=(448, 1, 1), block_dim=(128, 1, 1),
        )
    ctx.synchronize()
    var best_d = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_fat_gpu[128, 4864, 2, False]](
                xd, wd, dgd, dgd2,
                Int32(4864), Int32(896),
                grid_dim=(448, 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best_d:
            best_d = us
    print("  fat T128 O2 (champ):   ", best_d, " us")
    print("  f16 T128 O2:           ", _run_f16[128, 4864, 896, 2](ctx, dx, dw_d, dd, dd2), " us")
    print("  f16 T256 O2:           ", _run_f16[256, 4864, 896, 2](ctx, dx, dw_d, dd, dd2), " us")
    ctx.synchronize()
    arena.keep_alive()


def _run_fat_down(
    mut ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.uint8],
    mut dd: DeviceBuffer[DType.float32],
) raises -> Float64:
    # down 基线（fat，18B 标准布局）；占位在函数内创建（mut 参数同源问题）
    var dummy = ctx.enqueue_create_buffer[DType.float32](4)
    ctx.synchronize()
    var dd_local = dd
    var x_ptr = dx.unsafe_ptr()
    var w_ptr = dw.unsafe_ptr()
    var d_ptr = dd_local.unsafe_ptr()
    var b_ptr = dummy.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_fat_gpu[128, 4864, 2, False]](
            x_ptr, w_ptr, d_ptr, b_ptr,
            Int32(4864), Int32(896),
            grid_dim=(448, 1, 1), block_dim=(128, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_fat_gpu[128, 4864, 2, False]](
                x_ptr, w_ptr, d_ptr, b_ptr,
                Int32(4864), Int32(896),
                grid_dim=(448, 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    return best
