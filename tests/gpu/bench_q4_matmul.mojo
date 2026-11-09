"""q4 融合 matmul 的 A100 性能基线（账本五十二）。

**这独立于正确性门**：差分只验一次（小输入），性能跑真实投影形状。架构纪律是
两者分离 —— 门不追规模，基准不设正确性闸。

两个形状（Qwen 0.5B 的实际投影，`inner` 都是 32 的倍数，块永不跨行）：
    gate/up:  inner=896,  out=4864  （≈2.45 MB q4 权重）
    down:     inner=4864, out=896   （≈2.45 MB q4 权重）
两臂：`q4_matmul_gpu`（权重 0.5625 B/值）与 `linear_gpu`（fp32 权重 4 B/值），
理论字节比约 7:1。

**口径**：`inner×out` 权重 + `rows×inner` 的 x + 写 `rows×out`；解码形状
rows=1。同步粒度是「先排满 ITERS 次发射再同步一次」—— 测的是稳态吞吐；如果
结果被发射端（CPU enqueue 速率）而不是 GPU 带宽限制，数字会说话。

注意：远程跑必须带 `-I src`；本地会卡 ptxas sm_52（预期边界）。
"""

from std.math import ceildiv

from max.gpu.host import DeviceContext
from alofa.core.dtype import DT_FP32
from alofa.core.ffi.mem import RawPtr
from alofa.core.ffi.posix import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.kernels.cpu.quant import Q4_BYTES, dequant_q4_0, quantize_q4_0
from alofa.kernels.cpu.scalar import linear
from alofa.kernels.cuda.basic import linear_gpu
from alofa.kernels.cuda.matmul import q4_matmul_block_gpu, q4_matmul_gpu, q4_matmul_smem_gpu, q4_matmul_vec_gpu
from alofa.verify.compare import max_abs, max_abs_diff


comptime DT = DType.float32
comptime REL_TOLERANCE = Float64(1e-5)
comptime Q4_BLOCK = 32
comptime WARMUP = 20
comptime ITERS = 500
comptime ROUNDS = 3
comptime BLOCK = 128


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, Shape(dims), DT_FP32)


def fill(t: TensorView, seed: Int) raises -> None:
    """确定性、不整齐的输入（与差分门同一惯例）。"""
    var p = f32_data(t)
    for i in range(t.numel()):
        var k = i + seed * 7919
        p[unsafe_offset=i] = Float32((k % 251) - 125) * Float32(0.023)


def _view(raw: RawPtr, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    return TensorView(raw, Shape(dims), DT_FP32)


def _bench_shape[inner: Int, out_cols: Int](
    mut arena: Arena,
    mut ctx: DeviceContext,
) raises:
    """一个形状的完整流程：正确性检查 → q4 臂 → fp32 臂。"""
    var nblk = inner // Q4_BLOCK
    var w_bytes = out_cols * nblk * Q4_BYTES
    var n_w_f32 = out_cols * inner

    # ---- 数据（CPU 侧，产品代码造）----
    var x = scratch(arena, 1, inner)
    fill(x, 1)
    var wt = scratch(arena, out_cols, inner)
    fill(wt, 2)
    var blocks = arena.alloc(w_bytes)
    # 平铺量化：行主 [out, inner] 且 inner 是 32 的倍数 ⇒ 块流天然就是行主块流。
    quantize_q4_0(blocks, f32_data(wt), out_cols * inner)
    var wf_raw = arena.alloc(n_w_f32 * 4)
    # 参照路径：产品 dequant + 产品 linear（不是另写一份）。
    dequant_q4_0(wf_raw.unsafe_bitcast[Float32](), blocks, n_w_f32)
    var ref_raw = arena.alloc(out_cols * 4)
    linear(_view(ref_raw, 1, out_cols), x, _view(wf_raw, out_cols, inner))

    # ---- 上卡 ----
    var hb_x = ctx.enqueue_create_host_buffer[DT](inner)
    var hb_w = ctx.enqueue_create_host_buffer[DType.uint8](w_bytes)
    var hb_wf = ctx.enqueue_create_host_buffer[DT](n_w_f32)
    ctx.synchronize()
    var px = f32_data(x)
    var pwf = wf_raw.unsafe_bitcast[Float32]()
    for i in range(inner):
        hb_x[i] = px[unsafe_offset=i]
    for i in range(w_bytes):
        hb_w[i] = blocks[unsafe_offset=i]
    for i in range(n_w_f32):
        hb_wf[i] = pwf[unsafe_offset=i]

    var dx = ctx.enqueue_create_buffer[DT](inner)
    var dw = ctx.enqueue_create_buffer[DType.uint8](w_bytes)
    var dwf = ctx.enqueue_create_buffer[DT](n_w_f32)
    var dd = ctx.enqueue_create_buffer[DT](out_cols)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hb_x)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hb_w)
    ctx.enqueue_copy(dst_buf=dwf, src_buf=hb_wf)
    ctx.synchronize()

    var grid = (Int(ceildiv(out_cols, BLOCK)), 1, 1)
    var blk = (BLOCK, 1, 1)

    # ---- 正确性（rows=1，与解码形状一致）----
    ctx.enqueue_function[q4_matmul_gpu](
        dx.unsafe_ptr(),
        dw.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(inner),
        Int32(out_cols),
        grid_dim=grid,
        block_dim=blk,
    )
    var hd = ctx.enqueue_create_host_buffer[DT](out_cols)
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()
    var n = out_cols
    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(_view(ref_raw, 1, out_cols)), n)
    var scale = max_abs(f32_data(_view(ref_raw, 1, out_cols)), n)
    if scale <= 0.0:
        raise Error("reference all zeros")
    var tol = REL_TOLERANCE * scale
    if diff > tol:
        raise Error(
            "q4_matmul: diff="
            + String(diff)
            + " tol="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    print(
        "  [check] inner="
        + String(inner)
        + " out="
        + String(out_cols)
        + " diff="
        + String(diff)
        + " (tol "
        + String(tol)
        + ")"
    )

    # ---- 性能：q4 臂 ----
    var bytes_q4 = w_bytes + inner * 4 + out_cols * 4
    var best_q4 = Float64(1e30)
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_gpu](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(inner), Int32(out_cols), grid_dim=grid, block_dim=blk,
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_gpu](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(inner), Int32(out_cols), grid_dim=grid, block_dim=blk,
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_q4:
            best_q4 = ms

    # ---- 性能：fp32 臂（同一形状，权重 4 B/值）----
    var bytes_f32 = n_w_f32 * 4 + inner * 4 + out_cols * 4
    var best_f32 = Float64(1e30)
    for i in range(WARMUP):
        ctx.enqueue_function[linear_gpu](
            dx.unsafe_ptr(), dwf.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(inner), Int32(out_cols), grid_dim=grid, block_dim=blk,
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[linear_gpu](
                dx.unsafe_ptr(), dwf.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(inner), Int32(out_cols), grid_dim=grid, block_dim=blk,
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_f32:
            best_f32 = ms

    # best 是 ms；换算成 µs/iter（此前按 µs 直打，差了 1000 倍 —— 账本五十二记坑）。
    # ---- 性能 + 正确性：block 规约臂（THREADS=32 与 128）----
    # 规约顺序与串行版不同 ⇒ 判据是相对容差（对 CPU 参照），不是逐位。
    var best_b32 = Float64(1e30)
    var diff_b32 = Float64(-1)
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_block_gpu[32]](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_block_gpu[32]](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_b32:
            best_b32 = ms
    ctx.enqueue_function[q4_matmul_block_gpu[32]](
        dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
        Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()
    diff_b32 = max_abs_diff(hd.unsafe_ptr(), f32_data(_view(ref_raw, 1, out_cols)), n)

    # ---- vec 臂：每线程一块 + 16B 向量加载（THREADS=32）----
    var best_v32 = Float64(1e30)
    var diff_v32 = Float64(-1)
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_vec_gpu[32]](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_vec_gpu[32]](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_v32:
            best_v32 = ms
    ctx.enqueue_function[q4_matmul_vec_gpu[32]](
        dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
        Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()
    diff_v32 = max_abs_diff(hd.unsafe_ptr(), f32_data(_view(ref_raw, 1, out_cols)), n)

    # ---- smem 臂：x 行协作载入共享内存（冠军候选）----
    var best_s32 = Float64(1e30)
    var diff_s32 = Float64(-1)
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_smem_gpu[32, inner]](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_smem_gpu[32, inner]](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_s32:
            best_s32 = ms
    ctx.enqueue_function[q4_matmul_smem_gpu[32, inner]](
        dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
        Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()
    diff_s32 = max_abs_diff(hd.unsafe_ptr(), f32_data(_view(ref_raw, 1, out_cols)), n)

    @parameter
    for threads in range(64, 129, 64):
        var best_t = Float64(1e30)
        var diff_t = Float64(-1)
        for i in range(WARMUP):
            ctx.enqueue_function[q4_matmul_smem_gpu[threads, inner]](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(threads, 1, 1),
            )
        ctx.synchronize()
        for r in range(ROUNDS):
            var t0 = monotonic_ns()
            for i in range(ITERS):
                ctx.enqueue_function[q4_matmul_smem_gpu[threads, inner]](
                    dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                    Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(threads, 1, 1),
                )
            ctx.synchronize()
            var ms = Float64(monotonic_ns() - t0) / 1e6
            if ms < best_t:
                best_t = ms
        ctx.enqueue_function[q4_matmul_smem_gpu[threads, inner]](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(threads, 1, 1),
        )
        ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
        ctx.synchronize()
        diff_t = max_abs_diff(hd.unsafe_ptr(), f32_data(_view(ref_raw, 1, out_cols)), n)
        var us_t = best_t * Float64(1000) / Float64(ITERS)
        var gb_t = Float64(bytes_q4) * Float64(ITERS) / (best_t / 1e3) / 1e9
        print(
            "  q4_smem"
            + String(threads)
            + "   "
            + String(us_t)
            + " µs/iter  "
            + String(gb_t)
            + " GB/s  diff="
            + String(diff_t)
        )

    var best_b128 = Float64(1e30)
    var diff_b128 = Float64(-1)
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_block_gpu[128]](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
            Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(128, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_block_gpu[128]](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
                Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_b128:
            best_b128 = ms
    ctx.enqueue_function[q4_matmul_block_gpu[128]](
        dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(),
        Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), 1, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()
    diff_b128 = max_abs_diff(hd.unsafe_ptr(), f32_data(_view(ref_raw, 1, out_cols)), n)

    var us_q4 = best_q4 * Float64(1000) / Float64(ITERS)
    var us_f32 = best_f32 * Float64(1000) / Float64(ITERS)
    var gb_q4 = Float64(bytes_q4) * Float64(ITERS) / (best_q4 / 1e3) / 1e9
    var gb_f32 = Float64(bytes_f32) * Float64(ITERS) / (best_f32 / 1e3) / 1e9
    # ---- 发射地板：空 kernel，同一发射节奏 ----
    var hdn = ctx.enqueue_create_host_buffer[DT](1)
    var dn = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()
    var best_null = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[null_gpu](
                dn.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_null:
            best_null = ms
    var us_null = best_null * Float64(1000) / Float64(ITERS)

    var us_b32 = best_b32 * Float64(1000) / Float64(ITERS)
    var us_b128 = best_b128 * Float64(1000) / Float64(ITERS)
    var gb_b32 = Float64(bytes_q4) * Float64(ITERS) / (best_b32 / 1e3) / 1e9
    var gb_b128 = Float64(bytes_q4) * Float64(ITERS) / (best_b128 / 1e3) / 1e9
    print(
        "shape inner="
        + String(inner)
        + " out="
        + String(out_cols)
        + " rows=1"
    )
    print(
        "  q4_row     "
        + String(us_q4)
        + " µs/iter  "
        + String(gb_q4)
        + " GB/s"
    )
    print(
        "  fp32_row   "
        + String(us_f32)
        + " µs/iter  "
        + String(gb_f32)
        + " GB/s"
    )
    print(
        "  q4_blk32   "
        + String(us_b32)
        + " µs/iter  "
        + String(gb_b32)
        + " GB/s  diff="
        + String(diff_b32)
        + " (tol "
        + String(REL_TOLERANCE * scale)
        + ")"
    )
    print(
        "  q4_blk128  "
        + String(us_b128)
        + " µs/iter  "
        + String(gb_b128)
        + " GB/s  diff="
        + String(diff_b128)
    )
    var us_s32 = best_s32 * Float64(1000) / Float64(ITERS)
    var gb_s32 = Float64(bytes_q4) * Float64(ITERS) / (best_s32 / 1e3) / 1e9
    var us_v32 = best_v32 * Float64(1000) / Float64(ITERS)
    var gb_v32 = Float64(bytes_q4) * Float64(ITERS) / (best_v32 / 1e3) / 1e9
    print(
        "  q4_vec32   "
        + String(us_v32)
        + " µs/iter  "
        + String(gb_v32)
        + " GB/s  diff="
        + String(diff_v32)
    )
    print(
        "  q4_smem32  "
        + String(us_s32)
        + " µs/iter  "
        + String(gb_s32)
        + " GB/s  diff="
        + String(diff_s32)
    )
    print("  launch_floor " + String(us_null) + " µs/iter (空 kernel)")

    # ---- rows=32 批形状：首次真正走 block_idx.y 维度 ----
    var nrows = 32
    var n_x32 = nrows * inner
    var x32 = arena.alloc(n_x32 * 4)
    var px32 = x32.unsafe_bitcast[Float32]()
    for i in range(n_x32):
        px32[unsafe_offset=i] = Float32((i * 31 + 7) % 211 - 105) * Float32(0.021)
    # CPU 参照：整批 linear（一次调用，不做逐行）
    var ref32_raw = arena.alloc(nrows * out_cols * 4)
    var x32_view = _view(x32, nrows, inner)
    linear(_view(ref32_raw, nrows, out_cols), x32_view, _view(wf_raw, out_cols, inner))

    var hx32 = ctx.enqueue_create_host_buffer[DT](n_x32)
    var hd32 = ctx.enqueue_create_host_buffer[DT](nrows * out_cols)
    ctx.synchronize()
    for i in range(n_x32):
        hx32[i] = px32[unsafe_offset=i]
    var dx32 = ctx.enqueue_create_buffer[DT](n_x32)
    var dd32 = ctx.enqueue_create_buffer[DT](nrows * out_cols)
    ctx.enqueue_copy(dst_buf=dx32, src_buf=hx32)
    ctx.synchronize()

    # 正确性
    ctx.enqueue_function[q4_matmul_smem_gpu[32, inner]](
        dx32.unsafe_ptr(), dw.unsafe_ptr(), dd32.unsafe_ptr(),
        Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), Int(nrows), 1), block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd32, src_buf=dd32)
    ctx.synchronize()
    var n32 = nrows * out_cols
    var diff32 = max_abs_diff(hd32.unsafe_ptr(), f32_data(_view(ref32_raw, nrows, out_cols)), n32)
    var scale32 = max_abs(f32_data(_view(ref32_raw, nrows, out_cols)), n32)
    if scale32 <= 0.0:
        raise Error("batch reference all zeros")
    if diff32 > REL_TOLERANCE * scale32:
        raise Error("rows=32: diff=" + String(diff32))

    # 吞吐
    var bytes_b32 = w_bytes + n_x32 * 4 + n32 * 4
    var best_batch = Float64(1e30)
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_smem_gpu[32, inner]](
            dx32.unsafe_ptr(), dw.unsafe_ptr(), dd32.unsafe_ptr(),
            Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), Int(nrows), 1), block_dim=(32, 1, 1),
        )
    ctx.synchronize()
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_smem_gpu[32, inner]](
                dx32.unsafe_ptr(), dw.unsafe_ptr(), dd32.unsafe_ptr(),
                Int32(inner), Int32(out_cols), grid_dim=(Int(out_cols), Int(nrows), 1), block_dim=(32, 1, 1),
            )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best_batch:
            best_batch = ms
    var us_batch = best_batch * Float64(1000) / Float64(ITERS)
    var gb_batch = Float64(bytes_b32) * Float64(ITERS) / (best_batch / 1e3) / 1e9
    print(
        "  rows=32 smem32  "
        + String(us_batch)
        + " µs/iter  "
        + String(gb_batch)
        + " GB/s  diff="
        + String(diff32)
    )


def null_gpu(dst: Pointer[Float32, MutUntrackedOrigin]) -> None:
    """空 kernel：只碰一个字节，量发射+调度地板。"""
    dst[unsafe_offset=0] = Float32(0)


def main() raises:
    var arena = Arena(1 << 28)
    var ctx = DeviceContext()
    _bench_shape[896, 4864](arena, ctx)
    _bench_shape[4864, 896](arena, ctx)
    arena.keep_alive()
