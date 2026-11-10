"""账本九十七终极验证：层间功耗余量假说。
臂1 连续 12 层（=生产模式，应撞功耗墙慢 ~3.5ms/层）
臂2 层间 sync+usleep 300us（应全速 ~2ms/层）
臂3 层间 sync+usleep 100us
臂4 层间 sync 不 sleep
各臂独立新 session（kv 独立，pos0=0）。
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
comptime T = 128


def run_layers[NEST: Int](mut s: GpuSession, ids: List[Int]) raises:
    """NEST=0 连续；NEST>0 层间 sync+usleep(NEST us)。12 层 + gather + tail。"""
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
        s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 896, 2, 8, True, DType.float32]](
            anp, woL, dxp, Int32(896), Int32(t),
            grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
        )
        s.ctx.enqueue_function[rmsnorm_gpu](
            dxp, nB, xnp, Int32(896), Int32(t), eps,
            grid_dim=(t, 1, 1), block_dim=(256, 1, 1),
        )
        s.ctx.enqueue_function[f16_bmm_dual_gpu[64, 896, 2, 8, DType.float32]](
            xnp, wgL, wuL, gbp, ubp, abp, Int32(4864), Int32(t),
            grid_dim=(2432, 16, 1), block_dim=(64, 1, 1),
        )
        s.ctx.enqueue_function[f16_bmm_fat_gpu[128, 4864, 2, 8, True, DType.float32]](
            abp, wdL, dxp, Int32(896), Int32(t),
            grid_dim=(448, 16, 1), block_dim=(128, 1, 1),
        )
        if NEST > 0:
            s.ctx.synchronize()
            _ = external_call["usleep", Int32](Int32(NEST))
        elif NEST == -1:
            s.ctx.synchronize()
    s.ctx.enqueue_function[rmsnorm_gpu](
        dxp, s.d_norms[8].unsafe_ptr(), np2, Int32(896), Int32(1), eps,
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )
    s.ctx.enqueue_function[linear_fat_f16_gpu[256, 896, 8]](
        np2, s.d_embed16.unsafe_ptr(), lp, wsp, eps, Int32(s.vocab),
        grid_dim=(s.vocab // 8, 1, 1), block_dim=(256, 1, 1),
    )
    s.ctx.synchronize()
    print("nest", NEST, ":", Float64(monotonic_ns() - t0) / 1e6, "ms")


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)
    var s1 = GpuSession(model, 512)
    _ = run_layers[-1](s1, ids)
    _ = run_layers[-1](s1, ids)
    print("--- 臂5 层间纯 sync 无 sleep")
    var s2 = GpuSession(model, 512)
    _ = run_layers[20](s2, ids)
    _ = run_layers[20](s2, ids)
    print("--- 臂6 层间 sync+20us")
    var s3 = GpuSession(model, 512)
    _ = run_layers[30](s3, ids)
    _ = run_layers[30](s3, ids)
    print("--- 臂7 层间 sync+30us")
    var s4 = GpuSession(model, 512)
    _ = run_layers[50](s4, ids)
    _ = run_layers[50](s4, ids)
    print("--- 臂8 层间 sync+50us（复验）")
    model.params.keep_alive()
