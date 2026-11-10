"""prefill_step 分段计时（账本九十三）：profile=True 段间 sync，
只看占比（流水被破坏，总时长偏大）。128 token 单 rep。
"""

from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from std.ffi import external_call
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.matmul import (
    f16_bmm_fat_gpu, f16_bmm_dual_gpu, f16_bmm_qkv_gpu,
)
from alofa.kernels.cuda.attention import (
    kv_store_b_gpu, attn_rs_b_gpu, attn_vt2_b_gpu, attn_comb2_b_gpu,
)
from alofa.kernels.cuda.basic import rmsnorm_gpu, linear_fat_f16_gpu
from alofa.kernels.cuda.elementwise import gather_gpu
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128
comptime REPS = 10


def _bench[rb: Int, AF: DType](mut model: QwenForward, ids: List[Int]) raises:
    var session = GpuSession(model, 512)
    # 热身一发（钟内计——看首发射成本），第二次带 profile 同 session
    var t00 = monotonic_ns()
    _ = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, rb, AF](ids)
    session.ctx.synchronize()
    print("hot-1 total:", Float64(monotonic_ns() - t00) / 1e6, "ms")
    var t01 = monotonic_ns()
    _ = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, rb, AF](ids, profile=True)
    session.ctx.synchronize()
    print("hot-2 total(钟内含 ht):", Float64(monotonic_ns() - t01) / 1e6, "ms")
    print("=== profile run rb=", rb, " af=", AF)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)

    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)

    # t=1 对照：同 43 launch，GPU 工作量 ≈ 0 —— CPU 发射开销直接现形
    var ids1 = List[Int]()
    ids1.append(1000)
    _bench[8, DType.float32](model, ids1)
    # t=1 无 profile 总时长（43 launch 流水化、GPU 工作量 ≈ 0）
    var s1 = GpuSession(model, 512)
    _ = s1.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids1)
    s1.ctx.synchronize()
    var b1 = 1e30
    for rep in range(5):
        var s2 = GpuSession(model, 512)
        var t0 = monotonic_ns()
        _ = s2.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids1)
        s2.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < b1:
            b1 = ms
    print("t=1 no-profile:", b1, "ms")

    # 非泛型上下文：session 热两发后手写 o 对照
    var s = GpuSession(model, 512)
    var t00 = monotonic_ns()
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids)
    s.ctx.synchronize()
    print("main hot-1:", Float64(monotonic_ns() - t00) / 1e6, "ms")
    var t01 = monotonic_ns()
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids)
    s.ctx.synchronize()
    print("main hot-2:", Float64(monotonic_ns() - t01) / 1e6, "ms")
    var t03 = monotonic_ns()
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids)
    s.ctx.synchronize()
    print("main hot-3:", Float64(monotonic_ns() - t03) / 1e6, "ms")
    var t04 = monotonic_ns()
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids)
    s.ctx.synchronize()
    print("main hot-4:", Float64(monotonic_ns() - t04) / 1e6, "ms")
    # 九十五：拆分 发射段 / sync 段 / sleep 后 sync 段
    var s10 = GpuSession(model, 512)
    # 权重预热：同一 session 先跑 2 次（冷权重首读），再进入计时区
    _ = s10.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids, False, False)
    s10.ctx.synchronize()
    print("warm0 (冷权重首读): 完")
    _ = s10.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids, False, False)
    s10.ctx.synchronize()
    print("warm1: 完")
    for k in range(4):
        var tka = monotonic_ns()
        _ = s10.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, 8, DType.float32](ids, False, False)
        var tkb = monotonic_ns()
        s10.ctx.synchronize()
        var tkc = monotonic_ns()
        print("call", k, " 发射:", Float64(tkb - tka) / 1e6, " sync:", Float64(tkc - tkb) / 1e6)
        _ = external_call["usleep", Int32](Int32(120000))
        _ = external_call["usleep", Int32](Int32(120000))
    var anp = s.d_attnb.unsafe_ptr()
    var woL = s.d_w16_o[0].unsafe_ptr()
    var dxp = s.d_xb.unsafe_ptr()
    var b_o = 1e30
    for rep in range(3):
        s.ctx.synchronize()
        var t0 = monotonic_ns()
        s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
            anp, woL, dxp, Int32(896), Int32(128),
            grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
        )
        s.ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3
        if us < b_o:
            b_o = us
    print("bench-proc 手写 o:", b_o, "us/launch")
    print("手写指针: x=", anp, " w=", woL, " dst=", dxp)

    # ===== bench 进程内 4 层重放 + gather + lm_head（终极对账）=====
    var xnp = s.d_xnormb.unsafe_ptr()
    var qbp = s.d_qb.unsafe_ptr()
    var kbp = s.d_kb.unsafe_ptr()
    var vbp = s.d_vb.unsafe_ptr()
    var qrp = s.d_qrot_b.unsafe_ptr()
    var scp = s.d_scoresb.unsafe_ptr()
    var vpp = s.d_vpartb.unsafe_ptr()
    var atp = s.d_attotb.unsafe_ptr()
    var anp2 = s.d_attnb.unsafe_ptr()
    var gbp = s.d_gateb.unsafe_ptr()
    var ubp = s.d_upb.unsafe_ptr()
    var abp = s.d_actb.unsafe_ptr()
    var tbp = s.d_tokenb.unsafe_ptr()
    var emp = s.d_embed.unsafe_ptr()
    var cp = s.d_cos.unsafe_ptr()
    var sp = s.d_sin.unsafe_ptr()
    var np2 = s.d_normed.unsafe_ptr()
    var lp = s.d_logits.unsafe_ptr()
    var wsp = s.d_ws.unsafe_ptr()
    var eps2 = s.eps
    var nh = s.n_heads
    var nkv = s.n_kv_heads
    var hd = s.head_dim
    var qo = s.q_out
    var kvd = s.kv_dim
    var b7 = 1e30
    for rep in range(3):
        s.ctx.synchronize()
        var t0 = monotonic_ns()
        s.ctx.enqueue_function[gather_gpu](
            dxp, emp, tbp, Int32(896), Int32(128),
            grid_dim=(1792, 1, 1), block_dim=(64, 1, 1),
        )
        for layer in range(12):
            var pkL = s.d_k_store[layer].unsafe_ptr()
            var pvL = s.d_v_store[layer].unsafe_ptr()
            var nA = s.d_norms[2 * layer].unsafe_ptr()
            var nB = s.d_norms[2 * layer + 1].unsafe_ptr()
            var wqL = s.d_w16_q[layer].unsafe_ptr()
            var wkL = s.d_w16_k[layer].unsafe_ptr()
            var wvL = s.d_w16_v[layer].unsafe_ptr()
            var bqL = s.d_bias_q[layer].unsafe_ptr()
            var bkL = s.d_bias_k[layer].unsafe_ptr()
            var bvL = s.d_bias_v[layer].unsafe_ptr()
            var woL2 = s.d_w16_o[layer].unsafe_ptr()
            var wgL = s.d_w16_gate[layer].unsafe_ptr()
            var wuL = s.d_w16_up[layer].unsafe_ptr()
            var wdL = s.d_w16_down[layer].unsafe_ptr()
            var tseg = monotonic_ns()
            s.ctx.enqueue_function[rmsnorm_gpu](
                dxp, nA, xnp, Int32(896), Int32(128), eps2,
                grid_dim=(128, 1, 1), block_dim=(256, 1, 1),
            )
            s.ctx.enqueue_function[f16_bmm_qkv_gpu[128, 896, 8, DType.float32]](
                xnp, wqL, wkL, wvL, bqL, bkL, bvL, qbp, kbp, vbp,
                Int32(qo), Int32(kvd), Int32(128),
                grid_dim=(896, 16, 1), block_dim=(128, 1, 1),
            )
            s.ctx.synchronize()
            var tri_ms = Float64(monotonic_ns() - tseg) / 1e6
            tseg = monotonic_ns()
            s.ctx.enqueue_function[kv_store_b_gpu[512]](
                qrp, qbp, kbp, vbp, pkL, pvL, cp, sp,
                Int32(nh), Int32(nkv), Int32(hd), Int32(0),
                grid_dim=(nh, 1, 128), block_dim=(512, 1, 1),
            )
            s.ctx.enqueue_function[attn_rs_b_gpu[512, True, 8]](
                qrp, pkL, scp, Int32(nh), Int32(nkv), Int32(hd), Int32(128), Int32(0),
                grid_dim=(nh, 8, 128), block_dim=(512, 1, 1),
            )
            s.ctx.enqueue_function[attn_vt2_b_gpu[512, 8]](
                scp, pvL, vpp, atp, Int32(nh), Int32(nkv), Int32(hd), Int32(128), Int32(0),
                grid_dim=(nh, 8, 128), block_dim=(512, 1, 1),
            )
            s.ctx.enqueue_function[attn_comb2_b_gpu[8]](
                anp2, vpp, atp, Int32(nh), Int32(hd),
                grid_dim=(nh, 1, 128), block_dim=(512, 1, 1),
            )
            s.ctx.synchronize()
            var attn_ms = Float64(monotonic_ns() - tseg) / 1e6
            tseg = monotonic_ns()
            s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
                anp2, woL2, dxp, Int32(896), Int32(128),
                grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
            )
            s.ctx.synchronize()
            var o_ms = Float64(monotonic_ns() - tseg) / 1e6
            tseg = monotonic_ns()
            s.ctx.enqueue_function[rmsnorm_gpu](
                dxp, nB, xnp, Int32(896), Int32(128), eps2,
                grid_dim=(128, 1, 1), block_dim=(256, 1, 1),
            )
            s.ctx.enqueue_function[f16_bmm_dual_gpu[64, 896, 2, 8, DType.float32]](
                xnp, wgL, wuL, gbp, ubp, abp, Int32(4864), Int32(128),
                grid_dim=(2432, 16, 1), block_dim=(64, 1, 1),
            )
            s.ctx.synchronize()
            var ff_ms = Float64(monotonic_ns() - tseg) / 1e6
            tseg = monotonic_ns()
            s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 4864, 2, 8, True, DType.float32]](
                abp, wdL, dxp, Int32(896), Int32(128),
                grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
            )
            s.ctx.synchronize()
            var down_ms = Float64(monotonic_ns() - tseg) / 1e6
            print("重放层", layer, " tri:", tri_ms, " attn:", attn_ms, " o:", o_ms, " ff:", ff_ms, " down:", down_ms)
        s.ctx.enqueue_function[rmsnorm_gpu](
            dxp, s.d_norms[8].unsafe_ptr(), np2, Int32(896), Int32(1), eps2,
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
        )
        s.ctx.enqueue_function[linear_fat_f16_gpu[256, 896, 8]](
            np2, s.d_embed16.unsafe_ptr(), lp, wsp, eps2, Int32(s.vocab),
            grid_dim=(s.vocab // 8, 1, 1), block_dim=(256, 1, 1),
        )
        s.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < b7:
            b7 = ms
    print("bench-proc 4层完整重放:", b7, "ms")

    _bench[8, DType.float32](model, ids)

    var session = GpuSession(model, 512)
    var best = 1e30
    for i in range(64):
        var t0 = monotonic_ns()
        _ = session.decode_step[INNER, INNER_FF](2000 + i)
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best:
            best = ms
    print("decode 64 x1: best=", best, " ms")
    model.params.keep_alive()
