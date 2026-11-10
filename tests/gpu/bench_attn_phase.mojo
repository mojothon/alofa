"""attention 分相计时（账本八十）。

七十九留题：attention 20µs/层主导项未定位。`attn_rope_gpu` 加 `cut`
编译期相位裁剪参（1=scores 2=max归约 4=exp 8=sum归约 16=V），独立
kernel 循环发射计时，累计裁剪差值 = 各相成本：
V = L0-L1, sum = L1-L2, exp = L2-L3, max = L3-L4, scores = L4-L5。
kv 预热到 ~300（真实 kv 长度口径）；⚠️ 裁剪态输出是垃圾/inf，仅计时。

⚠️ 直发 import kernel 必须先 `from max.gpu.host import ...`（Mojo 怪象：
无此行 kernel 约定推断成 thin，绑不上 enqueue_function 的 capturing
thin 候选 —— round 76 六个 mini 探针全灭的根因，账本八十在案）。
"""

from max.gpu.host import HostBuffer

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.attention import attn_rope_gpu
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime MAXT = 320
comptime ITERS = 120


def bench_level[
    CUT: Int
](
    mut session: GpuSession,
    name: String,
    kv_len: Int,
    hd: Int,
) raises:
    var hd2 = hd
    var pos = (kv_len - 1) * hd2
    var cos_row = session.d_cos.unsafe_ptr() + pos
    var sin_row = session.d_sin.unsafe_ptr() + pos
    var pk = session.d_k_store[0].unsafe_ptr()
    var pv = session.d_v_store[0].unsafe_ptr()
    # 热身 3 发
    for r in range(3):
        session.ctx.enqueue_function[attn_rope_gpu[128, True, CUT]](
            session.d_attn.unsafe_ptr(),
            session.d_q_rot.unsafe_ptr(),
            session.d_q.unsafe_ptr(),
            session.d_k.unsafe_ptr(),
            session.d_v.unsafe_ptr(),
            pk,
            pv,
            cos_row,
            sin_row,
            session.d_scores.unsafe_ptr(),
            Int32(session.n_heads),
            Int32(session.n_kv_heads),
            Int32(hd),
            Int32(kv_len),
            grid_dim=(session.n_heads, 1, 1),
            block_dim=(128, 1, 1),
        )
    session.ctx.synchronize()
    var total = 0.0
    var bestv = 1e30
    for r in range(ITERS):
        var t0 = monotonic_ns()
        session.ctx.enqueue_function[attn_rope_gpu[128, True, CUT]](
            session.d_attn.unsafe_ptr(),
            session.d_q_rot.unsafe_ptr(),
            session.d_q.unsafe_ptr(),
            session.d_k.unsafe_ptr(),
            session.d_v.unsafe_ptr(),
            pk,
            pv,
            cos_row,
            sin_row,
            session.d_scores.unsafe_ptr(),
            Int32(session.n_heads),
            Int32(session.n_kv_heads),
            Int32(hd),
            Int32(kv_len),
            grid_dim=(session.n_heads, 1, 1),
            block_dim=(128, 1, 1),
        )
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3
        total += us
        if us < bestv:
            bestv = us
    print(name, ": mean=", total / Float64(ITERS), "us  best=", bestv, "us")


def bench_level_v2(
    mut session: GpuSession,
    name: String,
    kv_len: Int,
    hd: Int,
) raises:
    # V2 需 THREADS > head_dim：T512 → 4 组
    for r in range(3):
        session.ctx.enqueue_function[attn_rope_gpu[512, True, 0, True]](
            session.d_attn.unsafe_ptr(),
            session.d_q_rot.unsafe_ptr(),
            session.d_q.unsafe_ptr(),
            session.d_k.unsafe_ptr(),
            session.d_v.unsafe_ptr(),
            session.d_k_store[0].unsafe_ptr(),
            session.d_v_store[0].unsafe_ptr(),
            session.d_cos.unsafe_ptr() + (kv_len - 1) * hd,
            session.d_sin.unsafe_ptr() + (kv_len - 1) * hd,
            session.d_scores.unsafe_ptr(),
            Int32(session.n_heads),
            Int32(session.n_kv_heads),
            Int32(hd),
            Int32(kv_len),
            grid_dim=(session.n_heads, 1, 1),
            block_dim=(512, 1, 1),
        )
    session.ctx.synchronize()
    var total = 0.0
    var bestv = 1e30
    for r in range(ITERS):
        var t0 = monotonic_ns()
        session.ctx.enqueue_function[attn_rope_gpu[512, True, 0, True]](
            session.d_attn.unsafe_ptr(),
            session.d_q_rot.unsafe_ptr(),
            session.d_q.unsafe_ptr(),
            session.d_k.unsafe_ptr(),
            session.d_v.unsafe_ptr(),
            session.d_k_store[0].unsafe_ptr(),
            session.d_v_store[0].unsafe_ptr(),
            session.d_cos.unsafe_ptr() + (kv_len - 1) * hd,
            session.d_sin.unsafe_ptr() + (kv_len - 1) * hd,
            session.d_scores.unsafe_ptr(),
            Int32(session.n_heads),
            Int32(session.n_kv_heads),
            Int32(hd),
            Int32(kv_len),
            grid_dim=(session.n_heads, 1, 1),
            block_dim=(512, 1, 1),
        )
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3
        total += us
        if us < bestv:
            bestv = us
    print(name, ": mean=", total / Float64(ITERS), "us  best=", bestv, "us")


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    # kv 预热到 ~300
    for i in range(300):
        _ = session.decode_step[896, 4864](TOKEN)
    session.ctx.synchronize()
    var kv_len = session.kv_pos
    var hd = session.head_dim
    var cos_row = session.d_cos.unsafe_ptr() + (kv_len - 1) * hd
    var sin_row = session.d_sin.unsafe_ptr() + (kv_len - 1) * hd
    var pk = session.d_k_store[0].unsafe_ptr()
    var pv = session.d_v_store[0].unsafe_ptr()
    print("kv_len=", kv_len)

    bench_level[0](session, "full", kv_len, hd)
    bench_level_v2(session, "full-V2", kv_len, hd)
    bench_level[16](session, "  -V", kv_len, hd)
    bench_level[24](session, "  -V-sum", kv_len, hd)
    bench_level[28](session, "  -V-sum-exp", kv_len, hd)
    bench_level[30](session, "  -V-sum-exp-max", kv_len, hd)
    bench_level[31](session, "  only-scores", kv_len, hd)
    model.params.keep_alive()
