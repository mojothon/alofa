"""ncu 变通 + 分解实验：把 blk32 的 ~40µs 拆成 裸内存 / f64 / 规约 三份。

ncu 2023.1 的注入库不转发 `cuFuncGetParamInfo`（MAX 运行时在 ncu 下起不来，
直跑正常），驱动 560 够新 —— 问题在老 ncu。等不了升级，直接用**变体 kernel
分解**：同一分工、同一形状，逐个去掉一个成分，看墙钟掉多少。

四个臂（全部 rows=1，gate/up 896→4864，best-of-3 × 200）：
    blk32      完整版（基线，五十三/五十四的冠军，~40µs）
    loads_only 加载 + nibble 解包照做，但 f32 累加、无共享内存无 barrier
    f32_acc    完整结构（barrier + 树规约），只把累加换成 f32
    barrier_only 什么加载都不做，acc=tid 直接进树规约 —— 规约骨架的裸成本

正确性不在此文件判断（差分门的事），这里只看时间。
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation

from max.gpu import AddressSpace, barrier
from max.gpu.host import DeviceContext
from alofa.core.dtype import DT_FP32
from alofa.core.ffi.posix import monotonic_ns
from alofa.core.memory import Arena
from alofa.kernels.cpu.quant import Q4_BYTES, Q4_SCALE_BYTES, quantize_q4_0
from alofa.kernels.cuda.matmul import q4_matmul_block_gpu
from alofa.kernels.cuda.basic import _fp16_bits_to_f32_device


comptime DT = DType.float32
comptime INNER = 896
comptime OUT_COLS = 4864
comptime WARMUP = 10
comptime ITERS = 200
comptime ROUNDS = 3


def blk32_loads_only_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """内存 + 解包路径的裸成本：无共享内存、无 barrier、f32 累加。"""
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var x_base = row * Int(inner)
    var nblk = Int(inner) // 32
    var w_row = o * nblk
    var acc = Float32(0)
    var b = tid
    while b < nblk:
        var base = (w_row + b) * Q4_BYTES
        var low = Int(w_blocks[unsafe_offset=base])
        var high = Int(w_blocks[unsafe_offset=base + 1])
        var d = _fp16_bits_to_f32_device(low | (high << 8))
        var kb = b * 32
        for j in range(16):
            var byte = Int(w_blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
            acc += Float32((byte & 0x0F) - 8) * d * x[unsafe_offset=x_base + kb + j]
            acc += Float32(((byte >> 4) & 0x0F) - 8) * d * x[unsafe_offset=x_base + kb + j + 16]
        b += 32
    if tid == 0:
        dst[unsafe_offset=row * Int(out_cols) + o] = acc


def blk32_f32acc_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """完整结构（barrier + 树规约），只把累加换成 f32 —— 隔离 f64 的代价。"""
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var x_base = row * Int(inner)
    var nblk = Int(inner) // 32
    var w_row = o * nblk
    var acc = Float32(0)
    var b = tid
    while b < nblk:
        var base = (w_row + b) * Q4_BYTES
        var low = Int(w_blocks[unsafe_offset=base])
        var high = Int(w_blocks[unsafe_offset=base + 1])
        var d = _fp16_bits_to_f32_device(low | (high << 8))
        var kb = b * 32
        for j in range(16):
            var byte = Int(w_blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
            acc += Float32((byte & 0x0F) - 8) * d * x[unsafe_offset=x_base + kb + j]
            acc += Float32(((byte >> 4) & 0x0F) - 8) * d * x[unsafe_offset=x_base + kb + j + 16]
        b += 32
    var smem = stack_allocation[32, Float32, address_space = AddressSpace.SHARED]()
    smem[tid] = acc
    barrier()
    var stride = 16
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        dst[unsafe_offset=row * Int(out_cols) + o] = smem[0]


def blk32_smemx_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """x 行协作载入共享内存：真凶假设的直接验证。

    分解实验把 40µs 钉在加载路径上，而权重 byte load 已被 L1 兜底 —— 剩下的
    嫌疑是 **x**：每个 CTA 把整行 x 重读一遍，且线程读 `x[kb+j]`（warp 内
    间隔 128B，一条指令 32 个散射扇区，放大 ~8×）。修法是教科书式的：CTA
    开头协作把 x 行载入 smem（相邻线程读相邻地址，真合并），之后全部从 smem
    取。累加结构与 blk32 完全一致，预期 diff 仍 0.0。
    """
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[896, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    # 协作加载：k = tid, tid+32, ... —— 相邻线程读相邻 x，合并。
    var k = tid
    while k < Int(inner):
        sx[k] = x[unsafe_offset=x_base + k]
        k += 32
    barrier()
    var nblk = Int(inner) // 32
    var w_row = o * nblk
    var acc = Float64(0)
    var b = tid
    while b < nblk:
        var base = (w_row + b) * Q4_BYTES
        var low = Int(w_blocks[unsafe_offset=base])
        var high = Int(w_blocks[unsafe_offset=base + 1])
        var d = _fp16_bits_to_f32_device(low | (high << 8))
        var kb = b * 32
        for j in range(16):
            var byte = Int(w_blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
            acc += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(sx[kb + j])
            acc += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(sx[kb + j + 16])
        b += 32
    var smem = stack_allocation[32, Float64, address_space = AddressSpace.SHARED]()
    smem[tid] = acc
    barrier()
    var stride = 16
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        dst[unsafe_offset=row * Int(out_cols) + o] = Float32(smem[0])


def blk32_barrier_only_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """规约骨架的裸成本：不碰权重不碰 x，acc=tid 直接进树规约。"""
    _ = x
    _ = w_blocks
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var smem = stack_allocation[32, Float64, address_space = AddressSpace.SHARED]()
    smem[tid] = Float64(tid)
    barrier()
    var stride = 16
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        dst[unsafe_offset=row * Int(out_cols) + o] = Float32(smem[0])


def main() raises:
    var arena = Arena(1 << 26)
    var w_bytes = OUT_COLS * (INNER // 32) * Q4_BYTES

    var blocks = arena.alloc(w_bytes)
    var x_raw = arena.alloc(INNER * 4)
    var px = x_raw.unsafe_bitcast[Float32]()
    for i in range(INNER):
        px[unsafe_offset=i] = Float32((i % 19) - 9) * Float32(0.1)
    var wt = arena.alloc(OUT_COLS * INNER * 4).unsafe_bitcast[Float32]()
    for i in range(OUT_COLS * INNER):
        wt[unsafe_offset=i] = Float32((i % 17) - 8) * Float32(0.05)
    quantize_q4_0(blocks, wt, OUT_COLS * INNER)

    var ctx = DeviceContext()
    var n_x = INNER
    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hw = ctx.enqueue_create_host_buffer[DType.uint8](w_bytes)
    ctx.synchronize()
    for i in range(n_x):
        hx[i] = px[unsafe_offset=i]
    for i in range(w_bytes):
        hw[i] = blocks[unsafe_offset=i]

    var dx = ctx.enqueue_create_buffer[DT](n_x)
    var dw = ctx.enqueue_create_buffer[DType.uint8](w_bytes)
    var dd = ctx.enqueue_create_buffer[DT](OUT_COLS)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()

    # ---- 完整版（产品 kernel）----
    var best_full = Float64(1e30)
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_block_gpu[32]](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_block_gpu[32]](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_full:
            best_full = ms

    # ---- loads_only ----
    var best_loads = Float64(1e30)
    for i in range(WARMUP):
        ctx.enqueue_function[blk32_loads_only_gpu](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[blk32_loads_only_gpu](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_loads:
            best_loads = ms

    # ---- f32_acc ----
    var best_f32 = Float64(1e30)
    for i in range(WARMUP):
        ctx.enqueue_function[blk32_f32acc_gpu](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[blk32_f32acc_gpu](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_f32:
            best_f32 = ms

    # ---- barrier_only ----
    var best_barrier = Float64(1e30)
    for i in range(WARMUP):
        ctx.enqueue_function[blk32_barrier_only_gpu](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[blk32_barrier_only_gpu](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_barrier:
            best_barrier = ms

    # ---- smem_x ----
    var best_smemx = Float64(1e30)
    for i in range(WARMUP):
        ctx.enqueue_function[blk32_smemx_gpu](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[blk32_smemx_gpu](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(INNER), Int32(OUT_COLS), grid_dim=(Int(OUT_COLS), 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_smemx:
            best_smemx = ms

    var scale = Float64(ITERS)
    print(
        "full(blk32)  "
        + String(best_full / scale * 1e3)
        + " µs/iter"
    )
    print(
        "loads_only   "
        + String(best_loads / scale * 1e3)
        + " µs/iter"
    )
    print(
        "f32_acc      "
        + String(best_f32 / scale * 1e3)
        + " µs/iter"
    )
    print(
        "barrier_only "
        + String(best_barrier / scale * 1e3)
        + " µs/iter"
    )
    print(
        "smem_x       "
        + String(best_smemx / scale * 1e3)
        + " µs/iter"
    )
    arena.keep_alive()
