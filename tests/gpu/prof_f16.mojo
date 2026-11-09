"""f16 matmul 扫型（账本六十九）：session 实际核口径的每形状冠军。

六十八的探针不对称（f16 臂用单矩阵 fat、q4 基线用 dual）；本轮全部换成
session 在用的核：gate/up 用 `f16_matmul_dual_gpu`（两矩阵一次发射），
down/o 用 `f16_matmul_fat_gpu`，head 用 `linear_fat_f16_gpu`。每臂打印
有效带宽（权重 f16 字节 + x + dst）/ 时间，冠军回填 session。
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.core.dtype import DT_FP32
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.core.memory import Arena
from alofa.kernels.cpu.quant import dequant_q4_0, quantize_q4_0
from alofa.kernels.cuda.basic import linear_fat_f16_gpu
from alofa.kernels.cuda.matmul import (
    f16_matmul_dual_gpu,
    f16_matmul_fat_gpu,
    q4_matmul_dual_fat_gpu,
    q4_matmul_fat_gpu,
)

comptime WARMUP = 20
comptime ITERS = 200
comptime ROUNDS = 3


def _bw(bytes: Int, us: Float64) -> Float64:
    return Float64(bytes) / (us * 1e-6) / 1e9


def _upload_q4(
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


def _upload_f16_from_q4(
    mut ctx: DeviceContext, mut arena: Arena, dq4: DeviceBuffer[DType.uint8], n: Int
) raises -> DeviceBuffer[DType.float16]:
    """q4 设备流的 f16 精确反量化（host 绕一圈，仅装载期）。"""
    var host_q4 = ctx.enqueue_create_host_buffer[DType.uint8](n // 32 * 18)
    ctx.enqueue_copy(dst_buf=host_q4, src_buf=dq4)
    ctx.synchronize()
    var dims = List[Int]()
    dims.append(1)
    dims.append(n)
    var v = TensorView(arena.alloc(n * 4), Shape(dims), DT_FP32)
    var dq32 = f32_data(v)
    dequant_q4_0(dq32, host_q4.unsafe_ptr(), n)
    var host16 = ctx.enqueue_create_host_buffer[DType.float16](n)
    ctx.synchronize()
    for i in range(n):
        host16[i] = Float16(dq32[unsafe_offset=i])
    var d = ctx.enqueue_create_buffer[DType.float16](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=host16)
    ctx.synchronize()
    return d


def _bench_q4_dual(
    mut ctx: DeviceContext,
    label: String,
    mut dx: DeviceBuffer[DType.float32],
    mut dwa: DeviceBuffer[DType.uint8],
    mut dwb: DeviceBuffer[DType.uint8],
    mut dd1: DeviceBuffer[DType.float32],
    mut dd2: DeviceBuffer[DType.float32],
    mut db1: DeviceBuffer[DType.float32],
    mut db2: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x = dx.unsafe_ptr()
    var wa = dwa.unsafe_ptr()
    var wb = dwb.unsafe_ptr()
    var d1 = dd1.unsafe_ptr()
    var d2 = dd2.unsafe_ptr()
    var b1 = db1.unsafe_ptr()
    var b2 = db2.unsafe_ptr()
    """q4 dual 基线（T64/O4 六十七冠军）。wa/wb 收 uint8 指针由调用处转。"""
    var best = Float64(1e30)
    for r in range(ROUNDS + WARMUP // 10):
        var t0 = monotonic_ns()
        for i in range(ITERS if r > 0 else 20):
            ctx.enqueue_function[q4_matmul_dual_fat_gpu[64, 896, 4, False]](
                x, wa, wb, d1, d2, b1, b2,
                Int32(896), Int32(4864),
                grid_dim=(1216, 1, 1), block_dim=(64, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS if r > 0 else 20)
        if us < best:
            best = us
    print("  ", label, best, "us  bw=", _bw(4864 * 896 // 32 * 18, best), "GB/s(q4字节)")
    return best


def _bench_dual[T: Int, O: Int](
    mut ctx: DeviceContext,
    label: String,
    mut dx: DeviceBuffer[DType.float32],
    mut dwa: DeviceBuffer[DType.float16],
    mut dwb: DeviceBuffer[DType.float16],
    mut dd1: DeviceBuffer[DType.float32],
    mut dd2: DeviceBuffer[DType.float32],
    mut db1: DeviceBuffer[DType.float32],
    mut db2: DeviceBuffer[DType.float32],
    mut dact: DeviceBuffer[DType.float32],
    mut dnm: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x = dx.unsafe_ptr()
    var wa = dwa.unsafe_ptr()
    var wb = dwb.unsafe_ptr()
    var d1 = dd1.unsafe_ptr()
    var d2 = dd2.unsafe_ptr()
    var b1 = db1.unsafe_ptr()
    var b2 = db2.unsafe_ptr()
    var act = dact.unsafe_ptr()
    var nm = dnm.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[f16_matmul_dual_gpu[T, 896, O, False, False, False]](
            x, wa, wb, d1, d2, b1, b2, act, nm, Float32(0.0),
            Int32(896), Int32(4864),
            grid_dim=(4864 // O, 1, 1), block_dim=(T, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_matmul_dual_gpu[T, 896, O, False, False, False]](
                x, wa, wb, d1, d2, b1, b2, act, nm, Float32(0.0),
                Int32(896), Int32(4864),
                grid_dim=(4864 // O, 1, 1), block_dim=(T, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("  ", label, best, "us  bw=", _bw(4864 * 896 * 2 * 2 + 896 * 4 + 4864 * 8, best), "GB/s")
    return best


def _bench_fat[T: Int, INNER: Int, O: Int](
    mut ctx: DeviceContext,
    label: String,
    nout: Int,
    mut dx: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float16],
    mut dd: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x = dx.unsafe_ptr()
    var w = dw.unsafe_ptr()
    var d = dd.unsafe_ptr()
    var b = db.unsafe_ptr()
    for i in range(WARMUP):
        ctx.enqueue_function[f16_matmul_fat_gpu[T, INNER, O, False]](
            x, w, d, b,
            Int32(INNER), Int32(nout),
            grid_dim=(nout // O, 1, 1), block_dim=(T, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[f16_matmul_fat_gpu[T, INNER, O, False]](
                x, w, d, b,
                Int32(INNER), Int32(nout),
                grid_dim=(nout // O, 1, 1), block_dim=(T, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best:
            best = us
    print("  ", label, best, "us  bw=", _bw(nout * INNER * 2 + INNER * 4 + nout * 4, best), "GB/s")
    return best


def _bench_head[T: Int, INNER: Int, O: Int](
    mut ctx: DeviceContext,
    label: String,
    n_vocab: Int,
    mut dx: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float16],
    mut dd: DeviceBuffer[DType.float32],
    mut dnm: DeviceBuffer[DType.float32],
) raises -> Float64:
    var x = dx.unsafe_ptr()
    var w = dw.unsafe_ptr()
    var d = dd.unsafe_ptr()
    for i in range(WARMUP // 2):
        ctx.enqueue_function[linear_fat_f16_gpu[T, INNER, O]](
            x, w, d, dnm.unsafe_ptr(), Float32(0.0), Int32(n_vocab),
            grid_dim=(n_vocab // O, 1, 1), block_dim=(T, 1, 1),
        )
    ctx.synchronize()
    var best = Float64(1e30)
    for r in range(3):
        var t0 = monotonic_ns()
        for i in range(30):
            ctx.enqueue_function[linear_fat_f16_gpu[T, INNER, O]](
                x, w, d, dnm.unsafe_ptr(), Float32(0.0), Int32(n_vocab),
                grid_dim=(n_vocab // O, 1, 1), block_dim=(T, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / 30.0
        if us < best:
            best = us
    print("  ", label, best, "us  bw=", _bw(n_vocab * INNER * 2 + INNER * 4 + n_vocab * 4, best), "GB/s")
    return best


def main() raises:
    var ctx = DeviceContext()
    var arena = Arena(1 << 28)
    var dummy1 = ctx.enqueue_create_buffer[DType.float32](4864)
    var dummy2 = ctx.enqueue_create_buffer[DType.float32](4864)
    ctx.synchronize()

    print("== gate/up 4864x896 x2：q4 dual vs f16 dual ==")
    var dq_ga = _upload_q4(ctx, arena, 4864, 896)
    var dq_gb = _upload_q4(ctx, arena, 4864, 896)
    var dw_ga = _upload_f16_from_q4(ctx, arena, dq_ga, 4864 * 896)
    var dw_gb = _upload_f16_from_q4(ctx, arena, dq_gb, 4864 * 896)
    var dx = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd2 = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd3 = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd4 = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd5 = ctx.enqueue_create_buffer[DType.float32](4864)
    var dd6 = ctx.enqueue_create_buffer[DType.float32](4864)
    ctx.synchronize()
    _bench_q4_dual(ctx, "q4 dual T64 O4:", dx, dq_ga, dq_gb, dd, dd2, dd3, dd4)
    _bench_dual[64, 4](ctx, "f16 dual T64 O4:", dx, dw_ga, dw_gb, dd, dd2, dd3, dd4, dd5, dd6)
    _bench_dual[128, 4](ctx, "f16 dual T128 O4:", dx, dw_ga, dw_gb, dd, dd2, dd3, dd4, dd5, dd6)
    _bench_dual[64, 8](ctx, "f16 dual T64 O8:", dx, dw_ga, dw_gb, dd, dd2, dd3, dd4, dd5, dd6)
    _bench_dual[32, 4](ctx, "f16 dual T32 O4:", dx, dw_ga, dw_gb, dd, dd2, dd3, dd4, dd5, dd6)
    # ⚠️ 七十二：以下 O2/O1 臂的数字被同核连发的跨迭代流水污染
    #（O1"冠军"21.7µs 在真链上反而慢 0.3ms —— 4864 块 = 1.4 波）。
    # 留作微基准/链上背离的存证，勿据其选型；选型判据见账本七十二。
    _bench_dual[64, 2](ctx, "f16 dual T64 O2:", dx, dw_ga, dw_gb, dd, dd2, dd3, dd4, dd5, dd6)
    _bench_dual[64, 1](ctx, "f16 dual T64 O1:", dx, dw_ga, dw_gb, dd, dd2, dd3, dd4, dd5, dd6)
    _bench_dual[32, 1](ctx, "f16 dual T32 O1:", dx, dw_ga, dw_gb, dd, dd2, dd3, dd4, dd5, dd6)

    print("== down 896x4864：q4 fat 基线 vs f16 fat ==")
    var dq_da = _upload_q4(ctx, arena, 896, 4864)
    var dw_da = _upload_f16_from_q4(ctx, arena, dq_da, 896 * 4864)
    var dxd = ctx.enqueue_create_buffer[DType.float32](4864)
    var ddd = ctx.enqueue_create_buffer[DType.float32](4864)
    var ddd2 = ctx.enqueue_create_buffer[DType.float32](4864)
    ctx.synchronize()
    for i in range(WARMUP):
        ctx.enqueue_function[q4_matmul_fat_gpu[128, 4864, 2, False]](
            dxd.unsafe_ptr(), dq_da.unsafe_ptr(), ddd.unsafe_ptr(), ddd2.unsafe_ptr(),
            Int32(4864), Int32(896),
            grid_dim=(448, 1, 1), block_dim=(128, 1, 1),
        )
    ctx.synchronize()
    var best4d = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[q4_matmul_fat_gpu[128, 4864, 2, False]](
                dxd.unsafe_ptr(), dq_da.unsafe_ptr(), ddd.unsafe_ptr(), ddd2.unsafe_ptr(),
                Int32(4864), Int32(896),
                grid_dim=(448, 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3 / Float64(ITERS)
        if us < best4d:
            best4d = us
    print("   q4 fat T128 O2:", best4d, "us")
    _bench_fat[128, 4864, 2](ctx, "f16 fat T128 O2:", 896, dxd, dw_da, ddd, ddd2)
    _bench_fat[256, 4864, 2](ctx, "f16 fat T256 O2:", 896, dxd, dw_da, ddd, ddd2)
    _bench_fat[128, 4864, 4](ctx, "f16 fat T128 O4:", 896, dxd, dw_da, ddd, ddd2)
    _bench_fat[256, 4864, 4](ctx, "f16 fat T256 O4:", 896, dxd, dw_da, ddd, ddd2)

    print("== o 896x896（fat） ==")
    var dq_oa = _upload_q4(ctx, arena, 896, 896)
    var dw_oa = _upload_f16_from_q4(ctx, arena, dq_oa, 896 * 896)
    var dxo = ctx.enqueue_create_buffer[DType.float32](896)
    var ddo = ctx.enqueue_create_buffer[DType.float32](896)
    var ddo2 = ctx.enqueue_create_buffer[DType.float32](896)
    ctx.synchronize()
    _bench_fat[128, 896, 2](ctx, "f16 fat T128 O2:", 896, dxo, dw_oa, ddo, ddo2)
    _bench_fat[64, 896, 2](ctx, "f16 fat T64 O2:", 896, dxo, dw_oa, ddo, ddo2)
    _bench_fat[128, 896, 4](ctx, "f16 fat T128 O4:", 896, dxo, dw_oa, ddo, ddo2)
    _bench_fat[32, 896, 2](ctx, "f16 fat T32 O2:", 896, dxo, dw_oa, ddo, ddo2)

    print("== head 151936x896（linear_fat_f16） ==")
    comptime N_VOCAB = 151936
    var dxh = ctx.enqueue_create_buffer[DType.float32](896)
    var ddh = ctx.enqueue_create_buffer[DType.float32](N_VOCAB)
    var dnmh = ctx.enqueue_create_buffer[DType.float32](896)
    ctx.synchronize()
    # 大权重直接 host 填 pattern 上传（f16 字节流本身即可，无需真实反量化）
    var host16 = ctx.enqueue_create_host_buffer[DType.float16](N_VOCAB * 896)
    ctx.synchronize()
    for i in range(N_VOCAB * 896):
        host16[i] = Float16((Int(i) % 13) - 6) * Float16(0.01)
    var dwh = ctx.enqueue_create_buffer[DType.float16](N_VOCAB * 896)
    ctx.enqueue_copy(dst_buf=dwh, src_buf=host16)
    ctx.synchronize()
    _bench_head[256, 896, 8](ctx, "head T256 O8:", N_VOCAB, dxh, dwh, ddh, dnmh)
    _bench_head[512, 896, 8](ctx, "head T512 O8:", N_VOCAB, dxh, dwh, ddh, dnmh)
    _bench_head[256, 896, 16](ctx, "head T256 O16:", N_VOCAB, dxh, dwh, ddh, dnmh)
    _bench_head[256, 896, 4](ctx, "head T256 O4:", N_VOCAB, dxh, dwh, ddh, dnmh)
    _bench_head[128, 896, 8](ctx, "head T128 O8:", N_VOCAB, dxh, dwh, ddh, dnmh)

    arena.keep_alive()
