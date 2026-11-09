"""GpuSession 层 0 重放（常驻回归，账本六十三）：每步 vs 产品 CPU 通路。

层 0 的每一步（gather → norm → q/k/v → rope → append → attention → o → 残差
→ norm → gate/up → swiglu → down → 残差）逐原语重放并与产品标量通路对照。
判据：gather 零容差；其余相对容差 1e-5 —— **链式重放的"零容差"必须算继承
差**：q 的重组差会被 rope/store 原样继承（q_rot diff == q diff 恰是 rope
干净的证据），把继承差当成零容差红出来是探针自己冤枉自己。

它抓住过两个生产 bug（账本六十三）：q4 上传的 pinned-staging 竞态（S2 q 差
0.77/79）与 down 块流截断（NaN 一路传进 logits）；后者还暴露了
`max_abs_diff` 的 NaN 假绿（已修：NaN → +inf）。
"""

from max.gpu.host import HostBuffer
from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.tensor import Shape, TensorView, f32_data
from std.math import ceildiv
from alofa.model.arch.qwen import (
    BACKEND_SCALAR,
    EMBED,
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
from alofa.kernels.cpu.scalar import add, attention, rmsnorm, rope, swiglu
from alofa.kernels.cuda.elementwise import copy_gpu, rope_gpu, swiglu_gpu
from alofa.kernels.cuda.attention import attention_par_gpu
from alofa.verify.compare import max_abs, max_abs_diff

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime MAXT = 64
comptime REL_TOLERANCE = Float64(1e-5)


def _scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    return TensorView(arena.alloc(rows * cols * 4), Shape(dims), DT_FP32)


def _check(
    tag: String,
    mut h: HostBuffer[DType.float32],
    want: TensorView,
    mut arena: Arena,
    zero: Bool,
) raises:
    var n = want.numel()
    var got = _scratch(arena, 1, n)
    for i in range(n):
        f32_data(got)[unsafe_offset=i] = h[i]
    var d = max_abs_diff(f32_data(got), f32_data(want), n)
    var s = max_abs(f32_data(want), n)
    print(tag, " diff=", d, " ref_max=", s)
    if d != d:
        raise Error(tag + ": GPU SIDE HAS NaN")
    if zero and d != 0.0:
        raise Error(tag + ": nonzero at zero tolerance")
    if not zero and s > 0 and d > REL_TOLERANCE * s:
        raise Error(tag + ": exceeds 1e-5 relative")


def _fill_from_host(mut v: TensorView, mut h: HostBuffer[DType.float32], n: Int) raises:
    for i in range(n):
        f32_data(v)[unsafe_offset=i] = h[i]


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)
    var h = model.cfg.hidden
    var q_out = session.q_out
    var kv_dim = session.kv_dim
    var hd = model.cfg.head_dim
    var inter = model.cfg.intermediate
    var arena = Arena(1 << 22)

    # ---- 输入：真实 embedding 行（host）----
    var dims = List[Int]()
    dims.append(1)
    dims.append(h)
    var x_cpu = _scratch(arena, 1, h)
    var pe = f32_data(model.params.view(EMBED))
    for j in range(h):
        f32_data(x_cpu)[unsafe_offset=j] = pe[unsafe_offset=TOKEN * h + j]

    # ---- CPU 参照链（产品标量通路）----
    var normed_cpu = _scratch(arena, 1, h)
    rmsnorm(
        normed_cpu,
        x_cpu,
        model.params.view("model.layers.0.input_layernorm.weight"),
        model.cfg.eps,
    )
    var q_cpu = _scratch(arena, 1, q_out)
    var k_cpu = _scratch(arena, 1, kv_dim)
    var v_cpu = _scratch(arena, 1, kv_dim)
    model.project[BACKEND_SCALAR](
        q_cpu, normed_cpu, "model.layers.0.self_attn.q_proj.weight", Q4_SLOT_Q,
        model.params.view("model.layers.0.self_attn.q_proj.bias"), True,
    )
    model.project[BACKEND_SCALAR](
        k_cpu, normed_cpu, "model.layers.0.self_attn.k_proj.weight", Q4_SLOT_K,
        model.params.view("model.layers.0.self_attn.k_proj.bias"), True,
    )
    model.project[BACKEND_SCALAR](
        v_cpu, normed_cpu, "model.layers.0.self_attn.v_proj.weight", Q4_SLOT_V,
        model.params.view("model.layers.0.self_attn.v_proj.bias"), True,
    )
    var qrot_cpu = _scratch(arena, 1, q_out)
    var krot_cpu = _scratch(arena, 1, kv_dim)
    rope(qrot_cpu, krot_cpu, q_cpu, k_cpu, model.cos, model.sin, hd)
    var attn_cpu = _scratch(arena, 1, q_out)
    var scores_cpu = _scratch(arena, 1, 1)
    attention(
        attn_cpu, qrot_cpu, krot_cpu, v_cpu, scores_cpu,
        session.n_heads, session.n_kv_heads, hd,
    )
    var o_cpu = _scratch(arena, 1, h)
    model.project[BACKEND_SCALAR](
        o_cpu, attn_cpu, "model.layers.0.self_attn.o_proj.weight", Q4_SLOT_O,
        o_cpu, False,
    )
    var hid1_cpu = _scratch(arena, 1, h)
    add(hid1_cpu, x_cpu, o_cpu)
    var normed2_cpu = _scratch(arena, 1, h)
    rmsnorm(
        normed2_cpu,
        hid1_cpu,
        model.params.view("model.layers.0.post_attention_layernorm.weight"),
        model.cfg.eps,
    )
    var gate_cpu = _scratch(arena, 1, inter)
    model.project[BACKEND_SCALAR](
        gate_cpu, normed2_cpu, "model.layers.0.mlp.gate_proj.weight", Q4_SLOT_GATE,
        gate_cpu, False,
    )
    var up_cpu = _scratch(arena, 1, inter)
    model.project[BACKEND_SCALAR](
        up_cpu, normed2_cpu, "model.layers.0.mlp.up_proj.weight", Q4_SLOT_UP,
        up_cpu, False,
    )
    var act_cpu = _scratch(arena, 1, inter)
    swiglu(act_cpu, gate_cpu, up_cpu)
    var down_cpu = _scratch(arena, 1, h)
    model.project[BACKEND_SCALAR](
        down_cpu, act_cpu, "model.layers.0.mlp.down_proj.weight", Q4_SLOT_DOWN,
        down_cpu, False,
    )
    var hid2_cpu = _scratch(arena, 1, h)
    add(hid2_cpu, hid1_cpu, down_cpu)

    # ---- GPU 重放（session 私有原语；gather 已由探针 1 验证）----
    var hx = session.debug_gather(TOKEN)
    _check("S1 gather", hx, x_cpu, arena, True)

    session._norm(0)
    var hb1 = session.debug_read(1, h)
    _check("S2 normed", hb1, normed_cpu, arena, True)

    session._mm_bias[INNER](Q4_SLOT_Q, q_out, 0, 0, 0)
    session._mm_bias[INNER](Q4_SLOT_K, kv_dim, 1, 0, 1)
    session._mm_bias[INNER](Q4_SLOT_V, kv_dim, 2, 0, 2)
    var hb2 = session.debug_read(2, q_out)
    _check("S2 q", hb2, q_cpu, arena, False)
    var hb3 = session.debug_read(3, kv_dim)
    _check("S2 k", hb3, k_cpu, arena, False)
    var hb4 = session.debug_read(4, kv_dim)
    _check("S2 v", hb4, v_cpu, arena, False)

    session.ctx.enqueue_function[rope_gpu](
        session.d_q_rot.unsafe_ptr(),
        session.d_q.unsafe_ptr(),
        session.d_cos.unsafe_ptr(),
        session.d_sin.unsafe_ptr(),
        Int32(q_out),
        Int32(hd),
        Int32(1),
        grid_dim=(ceildiv(q_out // 2, 64), 1, 1),
        block_dim=(64, 1, 1),
    )
    session.ctx.enqueue_function[rope_gpu](
        session.d_k_rot.unsafe_ptr(),
        session.d_k.unsafe_ptr(),
        session.d_cos.unsafe_ptr(),
        session.d_sin.unsafe_ptr(),
        Int32(kv_dim),
        Int32(hd),
        Int32(1),
        grid_dim=(ceildiv(kv_dim // 2, 64), 1, 1),
        block_dim=(64, 1, 1),
    )
    session.ctx.synchronize()
    var hb5 = session.debug_read(5, q_out)
    _check("S2 q_rot", hb5, qrot_cpu, arena, False)  # rope 继承 q 的重组差
    var hb6 = session.debug_read(6, kv_dim)
    _check("S2 k_rot", hb6, krot_cpu, arena, False)

    session.ctx.enqueue_function[copy_gpu](
        session.d_k_store[0].unsafe_ptr(),
        session.d_k_rot.unsafe_ptr(),
        Int32(kv_dim),
        grid_dim=(ceildiv(kv_dim, 64), 1, 1),
        block_dim=(64, 1, 1),
    )
    session.ctx.enqueue_function[copy_gpu](
        session.d_v_store[0].unsafe_ptr(),
        session.d_v.unsafe_ptr(),
        Int32(kv_dim),
        grid_dim=(ceildiv(kv_dim, 64), 1, 1),
        block_dim=(64, 1, 1),
    )
    var pk = session.d_k_store[0].unsafe_ptr()
    var pv = session.d_v_store[0].unsafe_ptr()
    session.ctx.enqueue_function[attention_par_gpu[128]](
        session.d_attn.unsafe_ptr(),
        session.d_q_rot.unsafe_ptr(),
        pk,
        pv,
        session.d_scores.unsafe_ptr(),
        Int32(session.n_heads),
        Int32(session.n_kv_heads),
        Int32(hd),
        Int32(1),
        Int32(1),
        grid_dim=(session.n_heads, 1, 1),
        block_dim=(128, 1, 1),
    )
    session.ctx.synchronize()
    var hbks = session.debug_read(11, kv_dim)
    var hbvs = session.debug_read(12, kv_dim)
    _check("S2 store_k", hbks, krot_cpu, arena, False)
    _check("S2 store_v", hbvs, v_cpu, arena, False)
    var hb7 = session.debug_read(7, q_out)
    var ag2 = _scratch(arena, 1, q_out)
    for i in range(q_out):
        f32_data(ag2)[unsafe_offset=i] = hb7[i]
    print("S2 attn gpu[:4]=", f32_data(ag2)[unsafe_offset=0], f32_data(ag2)[unsafe_offset=1], f32_data(ag2)[unsafe_offset=2], f32_data(ag2)[unsafe_offset=3])
    print("S2 attn cpu[:4]=", f32_data(attn_cpu)[unsafe_offset=0], f32_data(attn_cpu)[unsafe_offset=1], f32_data(attn_cpu)[unsafe_offset=2], f32_data(attn_cpu)[unsafe_offset=3])
    _check("S2 attn", hb7, attn_cpu, arena, False)

    session._mm[INNER](Q4_SLOT_O, h, 1, 3)
    var hb8 = session.debug_read(8, h)
    _check("S3 o", hb8, o_cpu, arena, False)
    session._residual()
    var hb9 = session.debug_read(0, h)
    _check("S3 hidden+o", hb9, hid1_cpu, arena, False)
    session._norm(1)
    var hb10 = session.debug_read(1, h)
    _check("S3 normed2", hb10, normed2_cpu, arena, False)  # 继承链上前级重组差
    session._mm[INNER](Q4_SLOT_GATE, inter, 0, 4)
    session._mm[INNER](Q4_SLOT_UP, inter, 0, 5)
    var hb11 = session.debug_read(13, inter)
    _check("S3 gate", hb11, gate_cpu, arena, False)
    var hb12 = session.debug_read(14, inter)
    _check("S3 up", hb12, up_cpu, arena, False)
    session.ctx.enqueue_function[swiglu_gpu](
        session.d_act.unsafe_ptr(),
        session.d_gate.unsafe_ptr(),
        session.d_up.unsafe_ptr(),
        Int32(inter),
        grid_dim=(ceildiv(inter, 64), 1, 1),
        block_dim=(64, 1, 1),
    )
    session.ctx.synchronize()
    var hb13 = session.debug_read(9, inter)
    _check("S3 act", hb13, act_cpu, arena, False)
    session._mm[INNER_FF](Q4_SLOT_DOWN, h, 2, 3)
    var hb14 = session.debug_read(8, h)
    _check("S3 down", hb14, down_cpu, arena, False)
    session._residual()
    var hb15 = session.debug_read(0, h)
    _check("S3 hidden_final", hb15, hid2_cpu, arena, False)
    print("*** PROBE COMPLETE: layer 0 chain clean ***")
    model.params.keep_alive()
    arena.keep_alive()
