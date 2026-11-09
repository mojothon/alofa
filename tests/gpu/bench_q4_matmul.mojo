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
from alofa.kernels.cuda.matmul import q4_matmul_gpu
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
    var us_q4 = best_q4 * Float64(1000) / Float64(ITERS)
    var us_f32 = best_f32 * Float64(1000) / Float64(ITERS)
    var gb_q4 = Float64(bytes_q4) * Float64(ITERS) / (best_q4 / 1e3) / 1e9
    var gb_f32 = Float64(bytes_f32) * Float64(ITERS) / (best_f32 / 1e3) / 1e9
    print(
        "shape inner="
        + String(inner)
        + " out="
        + String(out_cols)
        + " rows=1  |  q4: "
        + String(us_q4)
        + " µs/iter  "
        + String(gb_q4)
        + " GB/s  |  fp32: "
        + String(us_f32)
        + " µs/iter  "
        + String(gb_f32)
        + " GB/s  |  字节比 "
        + String(us_f32 / us_q4)
    )


def main() raises:
    var arena = Arena(1 << 28)
    var ctx = DeviceContext()
    _bench_shape[896, 4864](arena, ctx)
    _bench_shape[4864, 896](arena, ctx)
    arena.keep_alive()
