"""账本一百：q2/kv2 A/B——device 侧逐位对照 + 交替 30 rep 时长。"""

from max.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.matmul import (
    f16_bmm_qkv_gpu, f16_bmm_q2_gpu, f16_bmm_kv2_gpu,
)
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128
comptime T = 128


def diff99_gpu(
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


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)
    var s = GpuSession(model, 512)
    var ctx = s.ctx
    var t = T

    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids, False, True)

    var xnp = s.d_xnormb.unsafe_ptr()
    var wq = s.d_w16_q[0].unsafe_ptr()
    var wk = s.d_w16_k[0].unsafe_ptr()
    var wv = s.d_w16_v[0].unsafe_ptr()
    var bq = s.d_bias_q[0].unsafe_ptr()
    var bk = s.d_bias_k[0].unsafe_ptr()
    var bv = s.d_bias_v[0].unsafe_ptr()
    var qo = Int32(s.q_out)
    var kvd = Int32(s.kv_dim)

    # 旧输出 & 新输出缓冲
    var q1 = ctx.enqueue_create_buffer[DType.float32](t * 896)
    var k1 = ctx.enqueue_create_buffer[DType.float32](t * 128)
    var v1 = ctx.enqueue_create_buffer[DType.float32](t * 128)
    var q2 = ctx.enqueue_create_buffer[DType.float32](t * 896)
    var k2 = ctx.enqueue_create_buffer[DType.float32](t * 128)
    var v2 = ctx.enqueue_create_buffer[DType.float32](t * 128)
    var nblocks = (t * 896 + 255) // 256
    var dbuf = ctx.enqueue_create_buffer[DType.float32](nblocks)
    var hdiff = ctx.enqueue_create_host_buffer[DType.float32](nblocks)
    ctx.synchronize()

    ctx.enqueue_function[f16_bmm_qkv_gpu[128, 896, 8, DType.float32]](
        xnp, wq, wk, wv, bq, bk, bv,
        q1.unsafe_ptr(), k1.unsafe_ptr(), v1.unsafe_ptr(),
        qo, kvd, Int32(t),
        grid_dim=(896, 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[f16_bmm_q2_gpu[128, 896, 4, 8, DType.float32]](
        xnp, wq, bq, q2.unsafe_ptr(), qo, Int32(t),
        grid_dim=(224, 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[f16_bmm_kv2_gpu[128, 896, 2, 8, DType.float32]](
        xnp, wk, wv, bk, bv, k2.unsafe_ptr(), v2.unsafe_ptr(), kvd, Int32(t),
        grid_dim=(64, 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[diff99_gpu](
        q1.unsafe_ptr(), q2.unsafe_ptr(), dbuf.unsafe_ptr(), Int32(t * 896),
        grid_dim=(nblocks, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hdiff, src_buf=dbuf)
    ctx.synchronize()
    var mx = 0.0
    for i in range(nblocks):
        if Float64(hdiff[i]) > mx:
            mx = Float64(hdiff[i])
    print("q maxdiff:", mx)

    ctx.enqueue_function[diff99_gpu](
        k1.unsafe_ptr(), k2.unsafe_ptr(), dbuf.unsafe_ptr(), Int32(t * 128),
        grid_dim=((t * 128 + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hdiff, src_buf=dbuf)
    ctx.synchronize()
    mx = 0.0
    for i in range((t * 128 + 255) // 256):
        if Float64(hdiff[i]) > mx:
            mx = Float64(hdiff[i])
    print("k maxdiff:", mx)

    ctx.enqueue_function[diff99_gpu](
        v1.unsafe_ptr(), v2.unsafe_ptr(), dbuf.unsafe_ptr(), Int32(t * 128),
        grid_dim=((t * 128 + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hdiff, src_buf=dbuf)
    ctx.synchronize()
    mx = 0.0
    for i in range((t * 128 + 255) // 256):
        if Float64(hdiff[i]) > mx:
            mx = Float64(hdiff[i])
    print("v maxdiff:", mx)

    # 时长：旧 qkv vs 新 q2+kv2（交替 30 rep）
    var tn = 0.0
    var t2s = 0.0
    for rep in range(30):
        var t0 = monotonic_ns()
        ctx.enqueue_function[f16_bmm_qkv_gpu[128, 896, 8, DType.float32]](
            xnp, wq, wk, wv, bq, bk, bv,
            q1.unsafe_ptr(), k1.unsafe_ptr(), v1.unsafe_ptr(),
            qo, kvd, Int32(t),
            grid_dim=(896, 16, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        tn += Float64(monotonic_ns() - t0) / 1e3
        t0 = monotonic_ns()
        ctx.enqueue_function[f16_bmm_q2_gpu[128, 896, 4, 8, DType.float32]](
            xnp, wq, bq, q2.unsafe_ptr(), qo, Int32(t),
            grid_dim=(224, 16, 1), block_dim=(128, 1, 1),
        )
        ctx.enqueue_function[f16_bmm_kv2_gpu[128, 896, 2, 8, DType.float32]](
            xnp, wk, wv, bk, bv, k2.unsafe_ptr(), v2.unsafe_ptr(), kvd, Int32(t),
            grid_dim=(64, 16, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        t2s += Float64(monotonic_ns() - t0) / 1e3
    print("qkv 旧:", tn / 30, "us  新(q2+kv2):", t2s / 30, "us")
    model.params.keep_alive()
