"""attention decode 形状的 A100 性能基准 —— 串行版 vs 块内并行版。

decode（q_len=1）是 attention 的关键路：串行版一个线程扛 kv_len 全部工作
（kv_len=2048 时每线程串行几百次 exp + 点积），并行版一个 block 64/128 线程。
形状用 Qwen 0.5B 的头配置（14 查询头 : 2 kv 头，head_dim=64）。

正确性差分（对产品标量 attention，1e-5 相对容差）与计时同脚本；差分越界会
raise（与差分门同一底线，性能脚本不许洗掉错误）。
"""

from max.gpu.host import DeviceContext
from alofa.core.dtype import DT_FP32
from alofa.core.ffi.mem import RawPtr
from alofa.core.ffi.posix import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.kernels.cpu.scalar import attention
from alofa.kernels.cuda.attention import attention_gpu, attention_par_gpu
from alofa.verify.compare import max_abs, max_abs_diff


comptime DT = DType.float32
comptime N_HEADS = 14
comptime N_KV_HEADS = 2
comptime HEAD_DIM = 64
comptime WARMUP = 10
comptime ITERS = 100
comptime ROUNDS = 3
comptime REL_TOLERANCE = Float64(1e-5)


def _view(raw: RawPtr, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    return TensorView(raw, Shape(dims), DT_FP32)


def main() raises:
    var arena = Arena(1 << 24)
    var q_cols = Int(N_HEADS) * Int(HEAD_DIM)
    var kv_cols = Int(N_KV_HEADS) * Int(HEAD_DIM)

    var ctx = DeviceContext()

    @parameter
    for kv_i in range(2):
        comptime kv_len = 512 * (1 + kv_i) * (1 + kv_i)
        var n_q = q_cols
        var n_kv = kv_len * kv_cols
        var n_s = Int(N_HEADS) * kv_len

        var x = arena.alloc((n_q + 2 * n_kv) * 4)
        var px = x.unsafe_bitcast[Float32]()
        for i in range(n_q):
            px[unsafe_offset=i] = Float32((i * 37 + 11) % 97 - 48) * Float32(0.031)
        for i in range(2 * n_kv):
            px[unsafe_offset=n_q + i] = Float32((i * 41 + 12) % 89 - 44) * Float32(0.027)

        # CPU 参照（产品 scalar attention，一次调用）
        var dst_raw = arena.alloc(n_q * 4)
        var scores_raw = arena.alloc(n_s * 4)
        attention(
            _view(dst_raw, 1, q_cols),
            _view(x, 1, q_cols),
            _view(x.unsafe_offset(n_q * 4), kv_len, kv_cols),
            _view(x.unsafe_offset((n_q + n_kv) * 4), kv_len, kv_cols),
            _view(scores_raw, 1, kv_len),
            Int(N_HEADS),
            Int(N_KV_HEADS),
            Int(HEAD_DIM),
        )
        var ref_max = max_abs(f32_data(_view(dst_raw, 1, q_cols)), n_q)

        # 设备缓冲
        var hq = ctx.enqueue_create_host_buffer[DT](n_q)
        var hk = ctx.enqueue_create_host_buffer[DT](n_kv)
        var hv = ctx.enqueue_create_host_buffer[DT](n_kv)
        var hd = ctx.enqueue_create_host_buffer[DT](n_q)
        ctx.synchronize()
        for i in range(n_q):
            hq[i] = px[unsafe_offset=i]
        for i in range(n_kv):
            hk[i] = px[unsafe_offset=n_q + i]
            hv[i] = px[unsafe_offset=n_q + n_kv + i]
        var dq = ctx.enqueue_create_buffer[DT](n_q)
        var dk = ctx.enqueue_create_buffer[DT](n_kv)
        var dv = ctx.enqueue_create_buffer[DT](n_kv)
        var ds = ctx.enqueue_create_buffer[DT](n_s)
        var dd = ctx.enqueue_create_buffer[DT](n_q)
        ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
        ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
        ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
        ctx.synchronize()

        var bytes_att = Float64(2 * n_kv * 4 + n_q * 4 + n_s * 4)

        # ---- 串行臂 ----
        var best_ser = Float64(1e30)
        for i in range(WARMUP):
            ctx.enqueue_function[attention_gpu](
                dd.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
                ds.unsafe_ptr(), Int32(N_HEADS), Int32(N_KV_HEADS), Int32(HEAD_DIM),
                Int32(1), Int32(kv_len), grid_dim=(Int(N_HEADS), 1, 1), block_dim=(1, 1, 1),
            )
        ctx.synchronize()
        for r in range(ROUNDS):
            var t0 = monotonic_ns()
            for i in range(ITERS):
                ctx.enqueue_function[attention_gpu](
                    dd.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
                    ds.unsafe_ptr(), Int32(N_HEADS), Int32(N_KV_HEADS), Int32(HEAD_DIM),
                    Int32(1), Int32(kv_len), grid_dim=(Int(N_HEADS), 1, 1), block_dim=(1, 1, 1),
                )
            ctx.synchronize()
            var ms = Float64(monotonic_ns() - t0) / 1e6
            if ms < best_ser:
                best_ser = ms
        ctx.enqueue_function[attention_gpu](
            dd.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
            ds.unsafe_ptr(), Int32(N_HEADS), Int32(N_KV_HEADS), Int32(HEAD_DIM),
            Int32(1), Int32(kv_len), grid_dim=(Int(N_HEADS), 1, 1), block_dim=(1, 1, 1),
        )
        ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
        ctx.synchronize()
        var diff_ser = max_abs_diff(
            hd.unsafe_ptr(), f32_data(_view(dst_raw, 1, q_cols)), n_q
        )

        # ---- 并行臂（THREADS=64 与 128）----
        var best_p64 = Float64(1e30)
        var diff_p64 = Float64(-1)
        var best_p128 = Float64(1e30)
        var diff_p128 = Float64(-1)
        @parameter
        for threads in range(64, 129, 64):
            var best_t = Float64(1e30)
            for i in range(WARMUP):
                ctx.enqueue_function[attention_par_gpu[threads]](
                    dd.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
                    ds.unsafe_ptr(), Int32(N_HEADS), Int32(N_KV_HEADS), Int32(HEAD_DIM),
                    Int32(1), Int32(kv_len), grid_dim=(Int(N_HEADS), 1, 1),
                    block_dim=(threads, 1, 1),
                )
            ctx.synchronize()
            for r in range(ROUNDS):
                var t0 = monotonic_ns()
                for i in range(ITERS):
                    ctx.enqueue_function[attention_par_gpu[threads]](
                        dd.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
                        ds.unsafe_ptr(), Int32(N_HEADS), Int32(N_KV_HEADS), Int32(HEAD_DIM),
                        Int32(1), Int32(kv_len), grid_dim=(Int(N_HEADS), 1, 1),
                        block_dim=(threads, 1, 1),
                    )
                ctx.synchronize()
                var ms = Float64(monotonic_ns() - t0) / 1e6
                if ms < best_t:
                    best_t = ms
            ctx.enqueue_function[attention_par_gpu[threads]](
                dd.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
                ds.unsafe_ptr(), Int32(N_HEADS), Int32(N_KV_HEADS), Int32(HEAD_DIM),
                Int32(1), Int32(kv_len), grid_dim=(Int(N_HEADS), 1, 1),
                block_dim=(threads, 1, 1),
            )
            ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
            ctx.synchronize()
            var diff_t = max_abs_diff(
                hd.unsafe_ptr(), f32_data(_view(dst_raw, 1, q_cols)), n_q
            )
            if threads == 64:
                best_p64 = best_t
                diff_p64 = diff_t
            else:
                best_p128 = best_t
                diff_p128 = diff_t

        var us_ser = best_ser * Float64(1000) / Float64(ITERS)
        var us_p64 = best_p64 * Float64(1000) / Float64(ITERS)
        var us_p128 = best_p128 * Float64(1000) / Float64(ITERS)
        var gb_ser = bytes_att * Float64(ITERS) / (best_ser / 1e3) / 1e9
        var gb_p64 = bytes_att * Float64(ITERS) / (best_p64 / 1e3) / 1e9
        var gb_p128 = bytes_att * Float64(ITERS) / (best_p128 / 1e3) / 1e9
        print(
            "kv_len="
            + String(kv_len)
            + " q_len=1 heads=14:2 hd=64  ref_max="
            + String(ref_max)
        )
        if diff_ser > REL_TOLERANCE * ref_max:
            raise Error("serial diff too large: " + String(diff_ser))
        if diff_p64 > REL_TOLERANCE * ref_max:
            raise Error("par64 diff too large: " + String(diff_p64))
        if diff_p128 > REL_TOLERANCE * ref_max:
            raise Error("par128 diff too large: " + String(diff_p128))
        print(
            "  serial      "
            + String(us_ser)
            + " µs/iter  "
            + String(gb_ser)
            + " GB/s  diff="
            + String(diff_ser)
        )
        print(
            "  par[64]     "
            + String(us_p64)
            + " µs/iter  "
            + String(gb_p64)
            + " GB/s  diff="
            + String(diff_p64)
        )
        print(
            "  par[128]    "
            + String(us_p128)
            + " µs/iter  "
            + String(gb_p128)
            + " GB/s  diff="
            + String(diff_p128)
        )
    arena.keep_alive()
