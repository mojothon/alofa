"""账本一百零二：attn 融合 A/B——vpart/tsum 逐位对照 + 交替 30 rep。"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_idx, thread_idx
from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.attention import (
    attn_rs_b_gpu, attn_vt2_b_gpu, attn_fused_b_gpu,
)
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128
comptime T = 128


def diff102_gpu(
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        dst[Int(block_idx.x)] = max(dst[Int(block_idx.x)], d)


def diff102d_gpu(
    a: Pointer[Float64, MutUntrackedOrigin],
    b: Pointer[Float64, MutUntrackedOrigin],
    dst: Pointer[Float64, MutUntrackedOrigin],
    n: Int32,
) -> None:
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        dst[Int(block_idx.x)] = max(dst[Int(block_idx.x)], d)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)
    var s = GpuSession(model, 512)
    var ctx = s.ctx
    for k in range(4):
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32](ids, False, True)

    var qrot = s.d_qrot_b.unsafe_ptr()
    var pk0 = s.d_k_store[0].unsafe_ptr()
    var pv0 = s.d_v_store[0].unsafe_ptr()

    # 两组 scratch：old 路径（scores/vpart/tsum）与 fused（vpart2/tsum2）
    var vp1 = ctx.enqueue_create_buffer[DType.float32](T * 16 * 128)
    var ts1 = ctx.enqueue_create_buffer[DType.float64](T * 16)
    var vp2 = ctx.enqueue_create_buffer[DType.float32](T * 16 * 128)
    var ts2 = ctx.enqueue_create_buffer[DType.float64](T * 16)
    var nblocks = (T * 16 * 128 + 255) // 256
    var dbuf = ctx.enqueue_create_buffer[DType.float32](nblocks)
    var hdiff = ctx.enqueue_create_host_buffer[DType.float32](nblocks)
    var dbuf64 = ctx.enqueue_create_buffer[DType.float64](T * 16)
    var hdiff64 = ctx.enqueue_create_host_buffer[DType.float64](T * 16)
    ctx.synchronize()

    var kvlens = InlineArray[Int, 2](fill=128)
    kvlens[1] = 512
    for ki in range(2):
        var L = kvlens[ki]
        var pos0 = Int32(L - T)
        ctx.enqueue_function[attn_rs_b_gpu[512, True, 1]](
            qrot, pk0, s.d_scoresb.unsafe_ptr(),
            Int32(s.n_heads), Int32(s.n_kv_heads), Int32(s.head_dim),
            Int32(L), pos0,
            grid_dim=(s.n_heads, 1, T), block_dim=(512, 1, 1),
        )
        ctx.enqueue_function[attn_vt2_b_gpu[512, 1]](
            s.d_scoresb.unsafe_ptr(), pv0,
            vp1.unsafe_ptr(), ts1.unsafe_ptr(),
            Int32(s.n_heads), Int32(s.n_kv_heads), Int32(s.head_dim),
            Int32(L), pos0,
            grid_dim=(s.n_heads, 1, T), block_dim=(512, 1, 1),
        )
        ctx.enqueue_function[attn_fused_b_gpu[512, True, 1]](
            qrot, pk0, pv0,
            vp2.unsafe_ptr(), ts2.unsafe_ptr(),
            Int32(s.n_heads), Int32(s.n_kv_heads), Int32(s.head_dim),
            Int32(L), pos0,
            grid_dim=(s.n_heads, 1, T), block_dim=(512, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_function[diff102_gpu](
            vp1.unsafe_ptr(), vp2.unsafe_ptr(), dbuf.unsafe_ptr(), Int32(T * 16 * 128),
            grid_dim=(nblocks, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.enqueue_copy(dst_buf=hdiff, src_buf=dbuf)
        ctx.synchronize()
        var mxv = 0.0
        for i in range(nblocks):
            if Float64(hdiff[i]) > mxv:
                mxv = Float64(hdiff[i])
        print("kv", L, "vpart maxdiff:", mxv)
        # tsum f64：按位 diff 不适用 f32 kernel —— 转 f32 粗比
        var nbd = (T * 16 + 255) // 256
        ctx.enqueue_function[diff102d_gpu](
            ts1.unsafe_ptr(), ts2.unsafe_ptr(), dbuf64.unsafe_ptr(), Int32(T * 16),
            grid_dim=(nbd, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.enqueue_copy(dst_buf=hdiff64, src_buf=dbuf64)
        ctx.synchronize()
        var mxd = 0.0
        for i in range(nbd):
            if Float64(hdiff64[i]) > mxd:
                mxd = Float64(hdiff64[i])
        print("kv", L, "tsum maxdiff:", mxd)

    # 时长：交替 30 rep（kv=512）
    var L = 512
    var pos0 = Int32(L - T)
    var told = 0.0
    var tnew = 0.0
    for rep in range(30):
        var t0 = monotonic_ns()
        ctx.enqueue_function[attn_rs_b_gpu[512, True, 1]](
            qrot, pk0, s.d_scoresb.unsafe_ptr(),
            Int32(s.n_heads), Int32(s.n_kv_heads), Int32(s.head_dim),
            Int32(L), pos0,
            grid_dim=(s.n_heads, 1, T), block_dim=(512, 1, 1),
        )
        ctx.enqueue_function[attn_vt2_b_gpu[512, 1]](
            s.d_scoresb.unsafe_ptr(), pv0,
            vp1.unsafe_ptr(), ts1.unsafe_ptr(),
            Int32(s.n_heads), Int32(s.n_kv_heads), Int32(s.head_dim),
            Int32(L), pos0,
            grid_dim=(s.n_heads, 1, T), block_dim=(512, 1, 1),
        )
        ctx.synchronize()
        told += Float64(monotonic_ns() - t0) / 1e3
        t0 = monotonic_ns()
        ctx.enqueue_function[attn_fused_b_gpu[512, True, 1]](
            qrot, pk0, pv0,
            vp2.unsafe_ptr(), ts2.unsafe_ptr(),
            Int32(s.n_heads), Int32(s.n_kv_heads), Int32(s.head_dim),
            Int32(L), pos0,
            grid_dim=(s.n_heads, 1, T), block_dim=(512, 1, 1),
        )
        ctx.synchronize()
        tnew += Float64(monotonic_ns() - t0) / 1e3
    print("rs+vt2 旧:", told / 30, "us  fused:", tnew / 30, "us")
    model.params.keep_alive()
