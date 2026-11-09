"""decode step 分段计时（账本六十四：10.2ms/token 未过 2ms 判据的定位）。

发射成本已排除（bench_launch：2.3µs/次 → 592 次 = 1.4ms host 侧）。本基准
把一步拆成三段各测稳态：整层链（21 次发射，×28 外推）、终 norm+lm_head、
每步 token 路径（pinned alloc + 2×sync + H2D）。谁吃了 8ms 一目了然。
"""

from max.gpu.host import HostBuffer
from std.math import ceildiv
from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import (
    Q4_SLOT_DOWN,
    Q4_SLOT_GATE,
    Q4_SLOT_K,
    Q4_SLOT_O,
    Q4_SLOT_Q,
    Q4_SLOT_UP,
    Q4_SLOT_V,
    QwenForward,
)
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.attention import attention_par_gpu
from alofa.kernels.cuda.basic import linear_fat_f16_gpu, linear_gpu
from alofa.kernels.cuda.elementwise import copy_gpu, rope_gpu, swiglu_gpu

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime MAXT = 64
comptime WARMUP = 20
comptime ITERS = 200
comptime ROUNDS = 3


def _time(mut session: GpuSession, name: String, iters: Int) raises:
    # 调用方已把要测的发射热身过；这里只做 ITERS 次纯计时的占位 —— 实际
    # 测量在 main 里逐段手写（发射序列各不相同，闭包传不进 comptime）。
    _ = session
    _ = name
    _ = iters


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)
    var h = model.cfg.hidden
    var q_out = session.q_out
    var kv_dim = session.kv_dim
    var hd = model.cfg.head_dim
    var inter = model.cfg.intermediate
    _ = h

    # 热身：走一遍层 0 全链 + lm_head（编译/缓存就位）。
    _ = session.debug_gather(TOKEN)
    session._norm(0)
    session._mm_bias[INNER](Q4_SLOT_Q, q_out, 0, 0, 0)
    session._mm_bias[INNER](Q4_SLOT_K, kv_dim, 1, 0, 1)
    session._mm_bias[INNER](Q4_SLOT_V, kv_dim, 2, 0, 2)
    session.ctx.enqueue_function[rope_gpu](
        session.d_q_rot.unsafe_ptr(), session.d_q.unsafe_ptr(),
        session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
        Int32(q_out), Int32(hd), Int32(1),
        grid_dim=(ceildiv(q_out // 2, 64), 1, 1), block_dim=(64, 1, 1),
    )
    session.ctx.enqueue_function[rope_gpu](
        session.d_k_rot.unsafe_ptr(), session.d_k.unsafe_ptr(),
        session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
        Int32(kv_dim), Int32(hd), Int32(1),
        grid_dim=(ceildiv(kv_dim // 2, 64), 1, 1), block_dim=(64, 1, 1),
    )
    session.ctx.enqueue_function[copy_gpu](
        session.d_k_store[0].unsafe_ptr(), session.d_k_rot.unsafe_ptr(),
        Int32(kv_dim), grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
    )
    session.ctx.enqueue_function[copy_gpu](
        session.d_v_store[0].unsafe_ptr(), session.d_v.unsafe_ptr(),
        Int32(kv_dim), grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
    )
    var pk = session.d_k_store[0].unsafe_ptr()
    var pv = session.d_v_store[0].unsafe_ptr()
    session.ctx.enqueue_function[attention_par_gpu[128]](
        session.d_attn.unsafe_ptr(), session.d_q_rot.unsafe_ptr(), pk, pv,
        session.d_scores.unsafe_ptr(),
        Int32(session.n_heads), Int32(session.n_kv_heads), Int32(hd),
        Int32(1), Int32(1),
        grid_dim=(session.n_heads, 1, 1), block_dim=(128, 1, 1),
    )
    session._mm[INNER](Q4_SLOT_O, h, 1, 3)
    session._residual()
    session._norm(1)
    session._mm[INNER](Q4_SLOT_GATE, inter, 0, 4)
    session._mm[INNER](Q4_SLOT_UP, inter, 0, 5)
    session.ctx.enqueue_function[swiglu_gpu](
        session.d_act.unsafe_ptr(), session.d_gate.unsafe_ptr(),
        session.d_up.unsafe_ptr(), Int32(inter),
        grid_dim=(ceildiv(inter, 64), 1, 1), block_dim=(64, 1, 1),
    )
    session._mm[INNER_FF](Q4_SLOT_DOWN, h, 2, 3)
    session._residual()
    session._norm(2 * session.n_layers)
    session.ctx.enqueue_function[linear_fat_f16_gpu[256, 896, 8]](
        session.d_normed.unsafe_ptr(), session.d_embed16.unsafe_ptr(),
        session.d_logits.unsafe_ptr(),
        Int32(session.vocab),
        grid_dim=(session.vocab // 8, 1, 1), block_dim=(256, 1, 1),
    )
    session.ctx.synchronize()

    var best_layer = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            session._norm(0)
            session._mm_bias[INNER](Q4_SLOT_Q, q_out, 0, 0, 0)
            session._mm_bias[INNER](Q4_SLOT_K, kv_dim, 1, 0, 1)
            session._mm_bias[INNER](Q4_SLOT_V, kv_dim, 2, 0, 2)
            session.ctx.enqueue_function[rope_gpu](
                session.d_q_rot.unsafe_ptr(), session.d_q.unsafe_ptr(),
                session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
                Int32(q_out), Int32(hd), Int32(1),
                grid_dim=(ceildiv(q_out // 2, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[rope_gpu](
                session.d_k_rot.unsafe_ptr(), session.d_k.unsafe_ptr(),
                session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
                Int32(kv_dim), Int32(hd), Int32(1),
                grid_dim=(ceildiv(kv_dim // 2, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[copy_gpu](
                session.d_k_store[0].unsafe_ptr(), session.d_k_rot.unsafe_ptr(),
                Int32(kv_dim), grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[copy_gpu](
                session.d_v_store[0].unsafe_ptr(), session.d_v.unsafe_ptr(),
                Int32(kv_dim), grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
            )
            # pk/pv 紧贴发射前提取：中间任何 session 可变借用都会使内部
            # 引用失效（账本六十四，比账本六十三的规则更细）。
            var pk = session.d_k_store[0].unsafe_ptr()
            var pv = session.d_v_store[0].unsafe_ptr()
            session.ctx.enqueue_function[attention_par_gpu[128]](
                session.d_attn.unsafe_ptr(), session.d_q_rot.unsafe_ptr(), pk, pv,
                session.d_scores.unsafe_ptr(),
                Int32(session.n_heads), Int32(session.n_kv_heads), Int32(hd),
                Int32(1), Int32(1),
                grid_dim=(session.n_heads, 1, 1), block_dim=(128, 1, 1),
            )
            session._mm[INNER](Q4_SLOT_O, h, 1, 3)
            session._residual()
            session._norm(1)
            session._mm[INNER](Q4_SLOT_GATE, inter, 0, 4)
            session._mm[INNER](Q4_SLOT_UP, inter, 0, 5)
            session.ctx.enqueue_function[swiglu_gpu](
                session.d_act.unsafe_ptr(), session.d_gate.unsafe_ptr(),
                session.d_up.unsafe_ptr(), Int32(inter),
                grid_dim=(ceildiv(inter, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session._mm[INNER_FF](Q4_SLOT_DOWN, h, 2, 3)
            session._residual()
        session.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
        if ms < best_layer:
            best_layer = ms
    print("A full layer chain: ", best_layer, " ms  -> x28 = ", best_layer * 28.0)

    var best_head = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            session._norm(2 * session.n_layers)
            session.ctx.enqueue_function[linear_fat_f16_gpu[256, 896, 8]](
                session.d_normed.unsafe_ptr(), session.d_embed16.unsafe_ptr(),
                session.d_logits.unsafe_ptr(),
                Int32(session.vocab),
                grid_dim=(session.vocab // 8, 1, 1), block_dim=(256, 1, 1),
            )
        session.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
        if ms < best_head:
            best_head = ms
    print("B final norm + lm_head: ", best_head, " ms")

    var best_tok = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            var ht = session.ctx.enqueue_create_host_buffer[DType.int32](1)
            session.ctx.synchronize()
            ht[0] = Int32(TOKEN)
            session.ctx.enqueue_copy(dst_buf=session.d_token, src_buf=ht)
            session.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
        if ms < best_tok:
            best_tok = ms
    print("C token path (pinned alloc + 2 sync + H2D): ", best_tok, " ms")

    var t0 = monotonic_ns()
    for i in range(ITERS):
        var hl = session.ctx.enqueue_create_host_buffer[DType.float32](
            session.vocab
        )
        session.ctx.synchronize()
        session.ctx.enqueue_copy(dst_buf=hl, src_buf=session.d_logits)
        session.ctx.synchronize()
    var d2h_ms = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
    print("D logits d2h (608 KB + pinned alloc + 2 sync): ", d2h_ms, " ms")

    # E：单 kernel 稳态 —— q(896x896)/gate(4864x896)/down(896x4864)/attention。
    var shapes = List[String]()
    var slots = List[Int]()
    var outs = List[Int]()
    shapes.append("E q    896x896 (q4)")
    slots.append(Q4_SLOT_Q)
    outs.append(q_out)
    shapes.append("E gate 4864x896 (q4)")
    slots.append(Q4_SLOT_GATE)
    outs.append(inter)
    shapes.append("E down 896x4864 (q4)")
    slots.append(Q4_SLOT_DOWN)
    outs.append(h)
    for si in range(3):
        var slot = slots[si]
        var n_out = outs[si]
        var name = shapes[si]
        var best = Float64(1e30)
        for r in range(ROUNDS):
            var t1 = monotonic_ns()
            for i in range(ITERS):
                session._mm[INNER](slot, n_out, 0, 3)
            session.ctx.synchronize()
            var us = Float64(monotonic_ns() - t1) / 1e3 / Float64(ITERS)
            if us < best:
                best = us
        print(name, " matmul: ", best, " us")
    var best_attn = Float64(1e30)
    for r in range(ROUNDS):
        var t1 = monotonic_ns()
        for i in range(ITERS):
            var pk = session.d_k_store[0].unsafe_ptr()
            var pv = session.d_v_store[0].unsafe_ptr()
            session.ctx.enqueue_function[attention_par_gpu[128]](
                session.d_attn.unsafe_ptr(), session.d_q_rot.unsafe_ptr(), pk, pv,
                session.d_scores.unsafe_ptr(),
                Int32(session.n_heads), Int32(session.n_kv_heads), Int32(hd),
                Int32(1), Int32(1),
                grid_dim=(session.n_heads, 1, 1), block_dim=(128, 1, 1),
            )
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t1) / 1e3 / Float64(ITERS)
        if us < best_attn:
            best_attn = us
    print("E attention: ", best_attn, " us")

    # F：小 kernel 组（12 发射：2 norm + 3 bias add + 残差x2 + rope x2 +
    # copy x2 + swiglu）与 G：大 kernel 组（7 matmul + attention）。
    var best_small = Float64(1e30)
    for r in range(ROUNDS):
        var t1 = monotonic_ns()
        for i in range(ITERS):
            session._norm(0)
            session._mm_bias[INNER](Q4_SLOT_K, kv_dim, 1, 0, 1)
            session._mm_bias[INNER](Q4_SLOT_V, kv_dim, 2, 0, 2)
            session.ctx.enqueue_function[rope_gpu](
                session.d_q_rot.unsafe_ptr(), session.d_q.unsafe_ptr(),
                session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
                Int32(q_out), Int32(hd), Int32(1),
                grid_dim=(ceildiv(q_out // 2, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[rope_gpu](
                session.d_k_rot.unsafe_ptr(), session.d_k.unsafe_ptr(),
                session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
                Int32(kv_dim), Int32(hd), Int32(1),
                grid_dim=(ceildiv(kv_dim // 2, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[copy_gpu](
                session.d_k_store[0].unsafe_ptr(), session.d_k_rot.unsafe_ptr(),
                Int32(kv_dim), grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[copy_gpu](
                session.d_v_store[0].unsafe_ptr(), session.d_v.unsafe_ptr(),
                Int32(kv_dim), grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
            )
            session._residual()
            session._norm(1)
            session.ctx.enqueue_function[swiglu_gpu](
                session.d_act.unsafe_ptr(), session.d_gate.unsafe_ptr(),
                session.d_up.unsafe_ptr(), Int32(inter),
                grid_dim=(ceildiv(inter, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session._residual()
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t1) / 1e3 / Float64(ITERS)
        if us < best_small:
            best_small = us
    print("F small-kernel set (12 launches): ", best_small, " us")

    # F 拆段：F1 = 2 norm；F2 = k/v mm_bias（4 发射）；F3 = 其余 7 发射。
    var best_f1 = Float64(1e30)
    for r in range(ROUNDS):
        var t1 = monotonic_ns()
        for i in range(ITERS):
            session._norm(0)
            session._norm(1)
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t1) / 1e3 / Float64(ITERS)
        if us < best_f1:
            best_f1 = us
    print("F1 norms x2: ", best_f1, " us")
    var best_f2 = Float64(1e30)
    for r in range(ROUNDS):
        var t1 = monotonic_ns()
        for i in range(ITERS):
            session._mm_bias[INNER](Q4_SLOT_K, kv_dim, 1, 0, 1)
            session._mm_bias[INNER](Q4_SLOT_V, kv_dim, 2, 0, 2)
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t1) / 1e3 / Float64(ITERS)
        if us < best_f2:
            best_f2 = us
    print("F2 k/v mm_bias: ", best_f2, " us")
    var best_f3 = Float64(1e30)
    for r in range(ROUNDS):
        var t1 = monotonic_ns()
        for i in range(ITERS):
            session.ctx.enqueue_function[rope_gpu](
                session.d_q_rot.unsafe_ptr(), session.d_q.unsafe_ptr(),
                session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
                Int32(q_out), Int32(hd), Int32(1),
                grid_dim=(ceildiv(q_out // 2, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[rope_gpu](
                session.d_k_rot.unsafe_ptr(), session.d_k.unsafe_ptr(),
                session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
                Int32(kv_dim), Int32(hd), Int32(1),
                grid_dim=(ceildiv(kv_dim // 2, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[copy_gpu](
                session.d_k_store[0].unsafe_ptr(), session.d_k_rot.unsafe_ptr(),
                Int32(kv_dim), grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
            )
            session.ctx.enqueue_function[copy_gpu](
                session.d_v_store[0].unsafe_ptr(), session.d_v.unsafe_ptr(),
                Int32(kv_dim), grid_dim=(1, 1, 1), block_dim=(64, 1, 1),
            )
            session._residual()
            session.ctx.enqueue_function[swiglu_gpu](
                session.d_act.unsafe_ptr(), session.d_gate.unsafe_ptr(),
                session.d_up.unsafe_ptr(), Int32(inter),
                grid_dim=(ceildiv(inter, 64), 1, 1), block_dim=(64, 1, 1),
            )
            session._residual()
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t1) / 1e3 / Float64(ITERS)
        if us < best_f3:
            best_f3 = us
    print("F3 rope/copy/swiglu/residual: ", best_f3, " us")
    var best_big = Float64(1e30)
    for r in range(ROUNDS):
        var t1 = monotonic_ns()
        for i in range(ITERS):
            session._mm[INNER](Q4_SLOT_Q, q_out, 0, 0)
            session._mm[INNER](Q4_SLOT_O, h, 1, 3)
            session._mm[INNER](Q4_SLOT_GATE, inter, 0, 4)
            session._mm[INNER](Q4_SLOT_UP, inter, 0, 5)
            session._mm[INNER_FF](Q4_SLOT_DOWN, h, 2, 3)
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t1) / 1e3 / Float64(ITERS)
        if us < best_big:
            best_big = us
    print("G big-kernel set (5 matmuls): ", best_big, " us")
    session.ctx.synchronize()
    model.params.keep_alive()
