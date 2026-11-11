"""账本九十九决定性 AB：同 session 内 old,new,new,old（kv 平衡 320/320、
顺序抵消），12 层重放，打印每次总时长。另反向 session new,old,old,new。
"""

from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.matmul import (
    f16_bmm_fat_gpu, f16_bmm_dual_gpu, f16_bmm_fat3_gpu, f16_bmm_dual3_gpu,
    f16_bmm_qkv_gpu,
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
comptime T = 128


def run_layers[NEW: Bool](mut s: GpuSession, ids: List[Int]) raises:
    var t = T
    var t0 = monotonic_ns()
    var dxp = s.d_xb.unsafe_ptr()
    var emp = s.d_embed.unsafe_ptr()
    var tbp = s.d_tokenb.unsafe_ptr()
    var xnp = s.d_xnormb.unsafe_ptr()
    var qbp = s.d_qb.unsafe_ptr()
    var kbp = s.d_kb.unsafe_ptr()
    var vbp = s.d_vb.unsafe_ptr()
    var qrp = s.d_qrot_b.unsafe_ptr()
    var scp = s.d_scoresb.unsafe_ptr()
    var vpp = s.d_vpartb.unsafe_ptr()
    var atp = s.d_attotb.unsafe_ptr()
    var anp = s.d_attnb.unsafe_ptr()
    var gbp = s.d_gateb.unsafe_ptr()
    var ubp = s.d_upb.unsafe_ptr()
    var abp = s.d_actb.unsafe_ptr()
    var np2 = s.d_normed.unsafe_ptr()
    var lp = s.d_logits.unsafe_ptr()
    var wsp = s.d_ws.unsafe_ptr()
    var cp = s.d_cos.unsafe_ptr()
    var sp = s.d_sin.unsafe_ptr()
    var eps = s.eps
    var nh = s.n_heads
    var nkv = s.n_kv_heads
    var hd = s.head_dim
    var qo = s.q_out
    var kvd = s.kv_dim
    s.ctx.enqueue_function[gather_gpu](
        dxp, emp, tbp, Int32(896), Int32(t),
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
        var woL = s.d_w16_o[layer].unsafe_ptr()
        var wgL = s.d_w16_gate[layer].unsafe_ptr()
        var wuL = s.d_w16_up[layer].unsafe_ptr()
        var wdL = s.d_w16_down[layer].unsafe_ptr()
        s.ctx.enqueue_function[rmsnorm_gpu](
            dxp, nA, xnp, Int32(896), Int32(t), eps,
            grid_dim=(t, 1, 1), block_dim=(256, 1, 1),
        )
        s.ctx.enqueue_function[f16_bmm_qkv_gpu[128, 896, 8, DType.float32]](
            xnp, wqL, wkL, wvL, bqL, bkL, bvL, qbp, kbp, vbp,
            Int32(qo), Int32(kvd), Int32(t),
            grid_dim=(896, 16, 1), block_dim=(128, 1, 1),
        )
        s.ctx.enqueue_function[kv_store_b_gpu[512]](
            qrp, qbp, kbp, vbp, pkL, pvL, cp, sp,
            Int32(nh), Int32(nkv), Int32(hd), Int32(0),
            grid_dim=(nh, 1, t), block_dim=(512, 1, 1),
        )
        s.ctx.enqueue_function[attn_rs_b_gpu[512, True, 8]](
            qrp, pkL, scp, Int32(nh), Int32(nkv), Int32(hd), Int32(t), Int32(0),
            grid_dim=(nh, 8, t), block_dim=(512, 1, 1),
        )
        s.ctx.enqueue_function[attn_vt2_b_gpu[512, 8]](
            scp, pvL, vpp, atp, Int32(nh), Int32(nkv), Int32(hd), Int32(t), Int32(0),
            grid_dim=(nh, 8, t), block_dim=(512, 1, 1),
        )
        s.ctx.enqueue_function[attn_comb2_b_gpu[8]](
            anp, vpp, atp, Int32(nh), Int32(hd),
            grid_dim=(nh, 1, t), block_dim=(512, 1, 1),
        )
        comptime if NEW:
            s.ctx.enqueue_function[f16_bmm_fat3_gpu[128, 896, 4, 8, True, DType.float32]](
                anp, woL, dxp, Int32(896), Int32(t),
                grid_dim=(224, 16, 1), block_dim=(128, 1, 1),
            )
        else:
            s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
                anp, woL, dxp, Int32(896), Int32(t),
                grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
            )
        s.ctx.enqueue_function[rmsnorm_gpu](
            dxp, nB, xnp, Int32(896), Int32(t), eps,
            grid_dim=(t, 1, 1), block_dim=(256, 1, 1),
        )
        comptime if NEW:
            s.ctx.enqueue_function[f16_bmm_dual3_gpu[64, 896, 2, 8, DType.float32]](
                xnp, wgL, wuL, gbp, ubp, abp, Int32(4864), Int32(t),
                grid_dim=(2432, 16, 1), block_dim=(64, 1, 1),
            )
        else:
            s.ctx.enqueue_function[f16_bmm_dual_gpu[64, 896, 2, 8, DType.float32]](
                xnp, wgL, wuL, gbp, ubp, abp, Int32(4864), Int32(t),
                grid_dim=(2432, 16, 1), block_dim=(64, 1, 1),
            )
        comptime if NEW:
            s.ctx.enqueue_function[f16_bmm_fat3_gpu[128, 4864, 4, 8, True, DType.float32]](
                abp, wdL, dxp, Int32(896), Int32(t),
                grid_dim=(224, 16, 1), block_dim=(128, 1, 1),
            )
        else:
            s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 4864, 2, 8, True, DType.float32]](
                abp, wdL, dxp, Int32(896), Int32(t),
                grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
            )
    s.ctx.enqueue_function[rmsnorm_gpu](
        dxp, s.d_norms[8].unsafe_ptr(), np2, Int32(896), Int32(1), eps,
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )
    s.ctx.enqueue_function[linear_fat_f16_gpu[256, 896, 8]](
        np2, s.d_embed16.unsafe_ptr(), lp, wsp, eps, Int32(s.vocab),
        grid_dim=(s.vocab // 8, 1, 1), block_dim=(256, 1, 1),
    )
    s.ctx.synchronize()
    var ms = Float64(monotonic_ns() - t0) / 1e6
    comptime if NEW:
        print("NEW:", ms, "ms")
    else:
        print("OLD:", ms, "ms")


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)
    var sA = GpuSession(model, 512)
    var sB = GpuSession(model, 512)
    # 双 session 交错：old,new,new,old | new,old,old,new（kv 与顺序双平衡）
    # 每 session 4 发 × 128 = 512 kv 满，仅一轮
    if True:
        _ = run_layers[False](sA, ids)
        _ = run_layers[True](sB, ids)
        _ = run_layers[True](sA, ids)
        _ = run_layers[False](sB, ids)
        _ = run_layers[False](sA, ids)
        _ = run_layers[True](sB, ids)
        _ = run_layers[True](sA, ids)
        _ = run_layers[False](sB, ids)
    model.params.keep_alive()
