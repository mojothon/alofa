"""异构 kernel 交替发射的开销基准（账本六十四）。

分段计时发现：单 kernel 稳态合计 ~82µs/层，但 20 发射的层链要 343µs ——
差值疑似"链式上下文里每次发射的 GPU 侧间隔"。本基准交替发射两个不同
kernel（q matmul 与 attention），对照同 kernel 连发，把"异构切换开销"从
"kernel 执行"里剥出来。
"""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.core.dtype import DT_FP32
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.core.memory import Arena
from alofa.kernels.cpu.quant import quantize_q4_0
from alofa.kernels.cuda.attention import attention_par_gpu
from alofa.kernels.cuda.elementwise import add_gpu, copy_gpu, swiglu_gpu
from alofa.kernels.cuda.matmul import q4_matmul_smem_gpu

comptime WARMUP = 50
comptime PAIRS = 500
comptime ROUNDS = 3
comptime INNER = 896
comptime OUT = 896
comptime QOUT = 896
comptime KVDIM = 128
comptime HEADS = 14
comptime KVHEADS = 2
comptime HD = 64


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
        f32_data(v)[unsafe_offset=i] = Float32(i % 7) * 0.01 - 0.03
    quantize_q4_0(host.unsafe_ptr(), f32_data(v), n_blocks)
    var d = ctx.enqueue_create_buffer[DType.uint8](n_blocks * 18)
    ctx.enqueue_copy(dst_buf=d, src_buf=host)
    ctx.synchronize()
    return d


def main() raises:
    var ctx = DeviceContext()
    var arena = Arena(1 << 24)
    var dq = _q4_upload(ctx, arena, OUT, INNER)
    var dx = ctx.enqueue_create_buffer[DType.float32](INNER)
    var dd = ctx.enqueue_create_buffer[DType.float32](OUT)
    var d_q = ctx.enqueue_create_buffer[DType.float32](QOUT)
    var d_attn = ctx.enqueue_create_buffer[DType.float32](QOUT)
    var d_k = ctx.enqueue_create_buffer[DType.float32](KVDIM)
    var d_v = ctx.enqueue_create_buffer[DType.float32](KVDIM)
    var d_scores = ctx.enqueue_create_buffer[DType.float32](HEADS * 4)
    var d_src = ctx.enqueue_create_buffer[DType.float32](KVDIM)
    var d_dst = ctx.enqueue_create_buffer[DType.float32](KVDIM)
    ctx.synchronize()

    # 同 kernel 连发
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_smem_gpu[32, INNER]](
            dx.unsafe_ptr(), dq.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(INNER), Int32(OUT),
            grid_dim=(OUT, 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    var best_q = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(PAIRS):
            ctx.enqueue_function[q4_matmul_smem_gpu[32, INNER]](
                dx.unsafe_ptr(), dq.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(INNER), Int32(OUT),
                grid_dim=(OUT, 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(PAIRS)
        if us < best_q:
            best_q = us
    print("same-kernel q matmul: ", best_q, " us/launch")

    # 异构交替：q matmul + attention 成对
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_smem_gpu[32, INNER]](
            dx.unsafe_ptr(), dq.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(INNER), Int32(OUT),
            grid_dim=(OUT, 1, 1), block_dim=(32, 1, 1),
        )
        ctx.enqueue_function[attention_par_gpu[128]](
            d_attn.unsafe_ptr(), d_q.unsafe_ptr(), d_k.unsafe_ptr(),
            d_v.unsafe_ptr(), d_scores.unsafe_ptr(),
            Int32(HEADS), Int32(KVHEADS), Int32(HD), Int32(1), Int32(1),
            grid_dim=(HEADS, 1, 1), block_dim=(128, 1, 1),
        )
    ctx.synchronize()
    var best_alt = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(PAIRS):
            ctx.enqueue_function[q4_matmul_smem_gpu[32, INNER]](
                dx.unsafe_ptr(), dq.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(INNER), Int32(OUT),
                grid_dim=(OUT, 1, 1), block_dim=(32, 1, 1),
            )
            ctx.enqueue_function[attention_par_gpu[128]](
                d_attn.unsafe_ptr(), d_q.unsafe_ptr(), d_k.unsafe_ptr(),
                d_v.unsafe_ptr(), d_scores.unsafe_ptr(),
                Int32(HEADS), Int32(KVHEADS), Int32(HD), Int32(1), Int32(1),
                grid_dim=(HEADS, 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(PAIRS)
        if us < best_alt:
            best_alt = us
    print("alt q+attention: ", best_alt, " us/pair (2 launches)")
    ctx.synchronize()

    # 小 kernel 轮转：copy/add/swiglu 三种循环 —— 复现"链里小 kernel 19µs/发"?
    var d_a896 = ctx.enqueue_create_buffer[DType.float32](896)
    var d_b896 = ctx.enqueue_create_buffer[DType.float32](896)
    var d_c896 = ctx.enqueue_create_buffer[DType.float32](896)
    var d_g = ctx.enqueue_create_buffer[DType.float32](4864)
    var d_u = ctx.enqueue_create_buffer[DType.float32](4864)
    var d_o = ctx.enqueue_create_buffer[DType.float32](4864)
    ctx.synchronize()
    for i in range(WARMUP):
        ctx.enqueue_function[copy_gpu](
            d_dst.unsafe_ptr(), d_src.unsafe_ptr(), Int32(KVDIM),
            grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
        )
        ctx.enqueue_function[add_gpu](
            d_c896.unsafe_ptr(), d_a896.unsafe_ptr(), d_b896.unsafe_ptr(),
            Int32(896), grid_dim=(14, 1, 1), block_dim=(64, 1, 1),
        )
        ctx.enqueue_function[swiglu_gpu](
            d_o.unsafe_ptr(), d_g.unsafe_ptr(), d_u.unsafe_ptr(), Int32(4864),
            grid_dim=(76, 1, 1), block_dim=(64, 1, 1),
        )
    ctx.synchronize()
    var best_cycle = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(PAIRS):
            ctx.enqueue_function[copy_gpu](
                d_dst.unsafe_ptr(), d_src.unsafe_ptr(), Int32(KVDIM),
                grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
            )
            ctx.enqueue_function[add_gpu](
                d_c896.unsafe_ptr(), d_a896.unsafe_ptr(), d_b896.unsafe_ptr(),
                Int32(896), grid_dim=(14, 1, 1), block_dim=(64, 1, 1),
            )
            ctx.enqueue_function[swiglu_gpu](
                d_o.unsafe_ptr(), d_g.unsafe_ptr(), d_u.unsafe_ptr(), Int32(4864),
                grid_dim=(76, 1, 1), block_dim=(64, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(PAIRS)
        if us < best_cycle:
            best_cycle = us
    print("cycle copy+add+swiglu: ", best_cycle, " us/triple (3 launches)")
    ctx.synchronize()
