"""向量 kernel 与标量 kernel 对着**同一份 fixture、同一条容差**判。

这里是 `test_layer0_parity.mojo` 的镜像：参照物同样是 Hugging Face 导出的输入/输
出对，容差同样是 `1e-5 × max(1, |参考|ₘₐₓ)`。两边共用 fixture 与判据，是为了让
"向量化有没有算错"这个问题**只**取决于向量化本身 —— 换一份 fixture 或放宽一点
容差，这个门就再也说不清自己在判什么了。

三条容易被忽略、所以特意写进来的东西：

**主机必须有 AVX2，否则具名失败，不静默跳过。** 一个"环境不支持就跳过"的测试
在 CI 上永远是绿的，而它绿的原因是没有跑 —— 那样的门比没有更危险，它会让人以
为这条通路是被验证过的。

**除了对参考，还与标量后端互比。** 对参考能发现"两个后端一起错"（比如都把布局
读错了，概率低），与标量互比能发现"向量版单独错"（概率高，因为它是新写的）。
两个方向都看，失败时才知道该修谁。

**负向对照钉的是"尾巴"。** 向量主循环只处理整向量，不足一个向量的部分走标量尾
巴。fixture 里的形状（896、4864）恰好都能被 4 和 8 整除，于是尾巴**一次都没被
真正测到** —— 这种"恰好整除所以没事"的依赖，换个模型就塌。所以这里专门写一个
只跑主循环、丢掉尾巴的版本，喂一个长度不是 8 的倍数的输入，它必须被判红。

跑法（不需要 2 GB 权重，只用到 layer0 那一份）：

    pixi run mojo run -O0 -I src tests/unit/test_avx2_parity.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.core.text import parse_float64, parse_int, read_text
from alofa.kernels.cpu.avx2 import add as add_v
from alofa.kernels.cpu.avx2 import attention as attention_v
from alofa.kernels.cpu.avx2 import linear as linear_v
from alofa.kernels.cpu.avx2 import linear_bias as linear_bias_v
from alofa.kernels.cpu.avx2 import rmsnorm as rmsnorm_v
from alofa.kernels.cpu.avx2 import rope as rope_v
from alofa.kernels.cpu.avx2 import swiglu as swiglu_v
from alofa.kernels.cpu.scalar import (
    add,
    attention,
    linear,
    linear_bias,
    rmsnorm,
    rope,
    swiglu,
)
from alofa.model.loader import TensorFile, config_value
from alofa.verify.compare import max_abs, max_abs_diff

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b/layer0"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"

# 与标量门**同一个数**，不是"向量化可以宽一点"。
comptime REL_TOLERANCE = Float64(1e-5)


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, Shape(dims), DT_FP32)


def assert_close(op: String, got: TensorView, expected: TensorView) raises:
    assert_equal(
        got.numel(), expected.numel(), op + ": element count differs from the reference"
    )
    var n = expected.numel()
    var diff = max_abs_diff(f32_data(got), f32_data(expected), n)
    var scale = max_abs(f32_data(expected), n)
    if scale < 1.0:
        scale = 1.0
    var tol = REL_TOLERANCE * scale
    assert_true(
        diff <= tol,
        op
        + ": max_abs_diff="
        + String(diff)
        + " tolerance="
        + String(tol)
        + " ref_max="
        + String(scale),
    )
    assert_true(scale > 0, op + ": the reference tensor is all zeros")


def test_host_actually_has_avx2() raises:
    """主机没有 AVX2 就**具名失败**，不静默跳过。

    静默跳过的测试在 CI 上是绿的，而它绿的原因恰恰是它没跑 —— 那样的门会让人
    以为这条通路被验证过。
    """
    var info = read_text("/proc/cpuinfo")
    assert_true(
        info.find(" avx2 ") >= 0,
        "这台机器的 /proc/cpuinfo 里没有 avx2 —— 向量门必须失败，不能悄悄跳过",
    )


def test_rmsnorm_matches_the_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var x = store.view("embed_out")
    var out = scratch(arena, x.shape.dims[0], x.shape.dims[1])
    var eps = Float32(parse_float64(config_value(CONFIG, "eps")))
    rmsnorm_v(out, x, store.view("norm_w"), eps)
    assert_close("rmsnorm", out, store.view("norm_out"))
    arena.keep_alive()
    store.keep_alive()


def test_qkv_projections_match_the_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var x = store.view("norm_out")
    var tokens = x.shape.dims[0]
    var hidden = x.shape.dims[1]

    var q = scratch(arena, tokens, hidden)
    linear_bias_v(q, x, store.view("q_w"), store.view("q_b"))
    assert_close("q_proj", q, store.view("q_out"))

    var kv_dim = store.view("k_out").shape.dims[1]
    var k = scratch(arena, tokens, kv_dim)
    linear_bias_v(k, x, store.view("k_w"), store.view("k_b"))
    assert_close("k_proj", k, store.view("k_out"))

    var v = scratch(arena, tokens, kv_dim)
    linear_bias_v(v, x, store.view("v_w"), store.view("v_b"))
    assert_close("v_proj", v, store.view("v_out"))
    arena.keep_alive()
    store.keep_alive()


def test_output_projection_and_residual_match_the_reference() raises:
    """o_proj → 残差加 → 第二个 RMSNorm，一条串起来的残差通路。

    只覆盖到 fixture 里确实导出了权重的那几步：gate/up/down 三个 MLP 投影的
    **权重**没有导出（只导出了它们的输出），所以本轮不拿它们对参考 —— 拿输出当
    输入自乘一遍，测的就不是被测代码了。
    """
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var tokens = store.view("attn_out").shape.dims[0]
    var hidden = store.view("attn_out").shape.dims[1]

    var proj = scratch(arena, tokens, hidden)
    linear_v(proj, store.view("attn_out"), store.view("o_w"))
    assert_close("o_proj", proj, store.view("o_out"))

    var hidden1 = scratch(arena, tokens, hidden)
    add_v(hidden1, store.view("embed_out"), store.view("o_out"))
    assert_close("add", hidden1, store.view("hidden1"))

    var norm2 = scratch(arena, tokens, hidden)
    var eps = Float32(parse_float64(config_value(CONFIG, "eps")))
    rmsnorm_v(norm2, hidden1, store.view("norm2_w"), eps)
    assert_close("rmsnorm2", norm2, store.view("norm2_out"))
    arena.keep_alive()
    store.keep_alive()


def test_swiglu_matches_the_reference() raises:
    """激活函数：`silu(gate) * up`，参照物是 HF 的真实 `swiglu_out`。"""
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var gate = store.view("gate_out")
    var out = scratch(arena, gate.shape.dims[0], gate.shape.dims[1])
    swiglu_v(out, gate, store.view("up_out"))
    assert_close("swiglu", out, store.view("swiglu_out"))
    arena.keep_alive()
    store.keep_alive()


def test_rope_and_attention_match_the_reference() raises:
    """`rope` 与 `attention` 的向量版对着 HF 导出的同一份输入输出判。

    镜像 `test_layer0_parity.mojo` 里两个同名测试：同一 fixture、同一容差
    （1e-5 × 量级），换一份 fixture 或放宽一点，这个门就再也说不清自己在判什么。
    """
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var head_dim = parse_int(config_value(CONFIG, "head_dim"))
    var q = store.view("q_out")
    var k = store.view("k_out")
    var tokens = q.shape.dims[0]
    var out_q = scratch(arena, tokens, q.shape.dims[1])
    var out_k = scratch(arena, tokens, k.shape.dims[1])
    rope_v(out_q, out_k, q, k, store.view("cos"), store.view("sin"), head_dim)
    assert_close("rope.q", out_q, store.view("q_rot"))
    assert_close("rope.k", out_k, store.view("k_rot"))

    var n_heads = parse_int(config_value(CONFIG, "n_heads"))
    var n_kv_heads = parse_int(config_value(CONFIG, "n_kv_heads"))
    var attn = scratch(arena, tokens, n_heads * head_dim)
    var scores = scratch(arena, tokens, tokens)
    attention_v(
        attn,
        store.view("q_rot"),
        store.view("k_rot"),
        store.view("v_out"),
        scores,
        n_heads,
        n_kv_heads,
        head_dim,
    )
    assert_close("attention", attn, store.view("attn_out"))
    arena.keep_alive()
    store.keep_alive()


def assert_agree(op: String, got: TensorView, want: TensorView) raises:
    """两个后端对同一输入的差，按**该算子自己的量级**判。

    判据与对参考时同一条（1e-5 × max(1, |量级|)），不因为"这是自己人比自己人"
    就放松 —— 放松之后它证明的就只是"两边都出了个数"。
    """
    var n = want.numel()
    var diff = max_abs_diff(f32_data(got), f32_data(want), n)
    var scale = max_abs(f32_data(want), n)
    if scale < 1.0:
        scale = 1.0
    var tol = REL_TOLERANCE * scale
    print(
        "观测："
        + op
        + " 向量 vs 标量 最大绝对差 "
        + String(diff)
        + "（判据 "
        + String(tol)
        + "，参考量级 "
        + String(scale)
        + "）"
    )
    assert_true(
        diff <= tol,
        op + ": 向量与标量相差 " + String(diff) + "，超过 " + String(tol),
    )


def test_vector_kernels_agree_with_the_scalar_backend() raises:
    """两个后端对同一输入互比。

    对**参考**能发现"两边一起错"，对**标量**能发现"向量版单独错"。后者的概率高
    得多（向量版是新写的），所以这条虽然用的是同一份 fixture，失败时指向的却是
    另一处。

    每个算子各自开缓冲：`swiglu` 的宽度是 `intermediate`（4864）而不是
    `hidden`（896），共用一块 [token, hidden] 的缓冲会在形状检查上炸 —— 那不是
    数值不一致，是测试自己写错了。
    """
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 22)
    var tokens = store.view("norm_out").shape.dims[0]
    var hidden = store.view("norm_out").shape.dims[1]

    var va = scratch(arena, tokens, hidden)
    var sa = scratch(arena, tokens, hidden)
    rmsnorm_v(va, store.view("embed_out"), store.view("norm_w"), Float32(1e-6))
    rmsnorm(sa, store.view("embed_out"), store.view("norm_w"), Float32(1e-6))
    assert_agree("rmsnorm", va, sa)

    linear_bias_v(va, store.view("norm_out"), store.view("q_w"), store.view("q_b"))
    linear_bias(sa, store.view("norm_out"), store.view("q_w"), store.view("q_b"))
    assert_agree("q_proj", va, sa)

    linear_v(va, store.view("attn_out"), store.view("o_w"))
    linear(sa, store.view("attn_out"), store.view("o_w"))
    assert_agree("o_proj", va, sa)

    add_v(va, store.view("embed_out"), store.view("o_out"))
    add(sa, store.view("embed_out"), store.view("o_out"))
    assert_agree("add", va, sa)

    var gate = store.view("gate_out")
    var vg = scratch(arena, gate.shape.dims[0], gate.shape.dims[1])
    var sg = scratch(arena, gate.shape.dims[0], gate.shape.dims[1])
    swiglu_v(vg, gate, store.view("up_out"))
    swiglu(sg, gate, store.view("up_out"))
    assert_agree("swiglu", vg, sg)

    var head_dim = parse_int(config_value(CONFIG, "head_dim"))
    var n_heads = parse_int(config_value(CONFIG, "n_heads"))
    var n_kv_heads = parse_int(config_value(CONFIG, "n_kv_heads"))
    var q_raw = store.view("q_out")
    var k_raw = store.view("k_out")
    var qv = scratch(arena, tokens, q_raw.shape.dims[1])
    var qs = scratch(arena, tokens, q_raw.shape.dims[1])
    var kv_vec = scratch(arena, tokens, k_raw.shape.dims[1])
    var kv_sca = scratch(arena, tokens, k_raw.shape.dims[1])
    rope_v(qv, kv_vec, q_raw, k_raw, store.view("cos"), store.view("sin"), head_dim)
    rope(qs, kv_sca, q_raw, k_raw, store.view("cos"), store.view("sin"), head_dim)
    assert_agree("rope.q", qv, qs)
    assert_agree("rope.k", kv_vec, kv_sca)

    var av = scratch(arena, tokens, n_heads * head_dim)
    var sa_attn = scratch(arena, tokens, n_heads * head_dim)
    var sv = scratch(arena, tokens, tokens)
    var ss = scratch(arena, tokens, tokens)
    attention_v(
        av, qv, kv_vec, store.view("v_out"), sv, n_heads, n_kv_heads, head_dim
    )
    attention(
        sa_attn, qs, kv_sca, store.view("v_out"), ss, n_heads, n_kv_heads, head_dim
    )
    assert_agree("attention", av, sa_attn)
    arena.keep_alive()
    store.keep_alive()


def test_tail_is_written_when_the_shape_is_not_divisible() raises:
    """负向对照：形状不整除时，`rope`/`attention` 的标量尾巴**必须被写到**。

    `head_dim=6` 时半个头只有 3 个通道，凑不满一个 8 通道向量；注意力那 6 个通道
    也凑不满两个 4 通道向量 —— 这两个形状让主循环**一次都不执行**，全程走尾巴。
    fixture 里的 head_dim=64 恰好整除，于是尾巴从未被跑到；这里的 6 就是为了让它
    被跑到。

    输出先填哨兵：尾巴若被丢掉，最后一个元素仍是哨兵，这条会**具名失败**，而不
    是"恰好整除所以看起来没事"。
    """
    var arena = Arena(1 << 16)
    var head_dim = 6
    var n_heads = 2
    var n_kv_heads = 1
    var tokens = 2
    var kv_len = 3
    var q_cols = n_heads * head_dim
    var kv_cols = n_kv_heads * head_dim

    var q = scratch(arena, tokens, q_cols)
    var cos = scratch(arena, tokens, head_dim)
    var sin = scratch(arena, tokens, head_dim)
    for i in range(tokens * q_cols):
        f32_data(q)[unsafe_offset=i] = Float32(i % 7) - Float32(3)
    for i in range(tokens * head_dim):
        f32_data(cos)[unsafe_offset=i] = Float32(i % 5) * Float32(0.25) - Float32(0.5)
        f32_data(sin)[unsafe_offset=i] = Float32(i % 3) * Float32(0.125) - Float32(0.25)

    var vq = scratch(arena, tokens, q_cols)
    var vk = scratch(arena, tokens, q_cols)
    var sq = scratch(arena, tokens, q_cols)
    var sk = scratch(arena, tokens, q_cols)
    for i in range(tokens * q_cols):
        f32_data(vq)[unsafe_offset=i] = Float32(-7)
        f32_data(vk)[unsafe_offset=i] = Float32(-7)
        f32_data(sq)[unsafe_offset=i] = Float32(-7)
        f32_data(sk)[unsafe_offset=i] = Float32(-7)

    rope_v(vq, vk, q, q, cos, sin, head_dim)
    rope(sq, sk, q, q, cos, sin, head_dim)
    assert_agree("rope.tail", vq, sq)
    assert_agree("rope.tail.k", vk, sk)
    assert_true(
        f32_data(vq)[unsafe_offset=tokens * q_cols - 1] != Float32(-7),
        "rope 的最后一个元素还是哨兵 —— 向量 kernel 把标量尾巴丢掉了",
    )

    var ka = scratch(arena, kv_len, kv_cols)
    var va = scratch(arena, kv_len, kv_cols)
    for i in range(kv_len * kv_cols):
        f32_data(ka)[unsafe_offset=i] = Float32(i % 5) * Float32(0.5) - Float32(1)
        f32_data(va)[unsafe_offset=i] = Float32(i % 4) - Float32(1.5)
    var out_v = scratch(arena, tokens, q_cols)
    var out_s = scratch(arena, tokens, q_cols)
    var sc_v = scratch(arena, tokens, kv_len)
    var sc_s = scratch(arena, tokens, kv_len)
    for i in range(tokens * q_cols):
        f32_data(out_v)[unsafe_offset=i] = Float32(-7)
        f32_data(out_s)[unsafe_offset=i] = Float32(-7)
    attention_v(out_v, vq, ka, va, sc_v, n_heads, n_kv_heads, head_dim)
    attention(out_s, sq, ka, va, sc_s, n_heads, n_kv_heads, head_dim)
    assert_agree("attention.tail", out_v, out_s)
    assert_true(
        f32_data(out_v)[unsafe_offset=tokens * q_cols - 1] != Float32(-7),
        "attention 的最后一个元素还是哨兵 —— 向量 kernel 把标量尾巴丢掉了",
    )
    arena.keep_alive()


def add_head_only(dst: TensorView, a: TensorView, b: TensorView) raises:
    """只跑向量主循环、丢掉尾巴的版本，**仅用于负向对照**。

    它看起来和 `avx2.add` 一模一样，唯一的差别是没有处理末尾不足 8 个元素的那
    一段。喂一个长度不是 8 的倍数的输入，它必须被判红 —— 否则说明判据对"尾巴
    被丢掉"这件事不敏感。
    """
    var n = a.numel()
    var pa = f32_data(a)
    var pb = f32_data(b)
    var po = f32_data(dst)
    var i = 0
    while i + 8 <= n:
        po.unsafe_store[width=8](
            i, pa.unsafe_load[width=8](i) + pb.unsafe_load[width=8](i)
        )
        i += 8


def test_a_kernel_that_drops_the_tail_is_rejected() raises:
    """负向对照：丢掉标量尾巴的向量 kernel 必须被判出来。

    fixture 里的形状恰好都能被通道数整除，于是尾巴**从未被真正执行过** —— 这种
    "恰好没事"的依赖最该有一条专门的测试顶住。这里用长度 10（不是 8 的倍数），
    输出先填一个不可能等于正确答案的哨兵值，于是"尾巴没写"必然暴露。
    """
    var arena = Arena(1 << 16)
    var n = 10
    var a = scratch(arena, 1, n)
    var b = scratch(arena, 1, n)
    var got = scratch(arena, 1, n)
    var want = scratch(arena, 1, n)
    for i in range(n):
        f32_data(a)[unsafe_offset=i] = Float32(i) * Float32(0.5) - Float32(2)
        f32_data(b)[unsafe_offset=i] = Float32(i % 3) - Float32(1)
        # 哨兵：正确答案不可能等于它，于是"没写"和"写对了"可以区分开。
        f32_data(got)[unsafe_offset=i] = Float32(-7)
        f32_data(want)[unsafe_offset=i] = Float32(-7)

    add_head_only(got, a, b)
    add(want, a, b)

    var diff = max_abs_diff(f32_data(got), f32_data(want), n)
    assert_true(
        diff > 0,
        "丢掉尾巴的向量 kernel 与正确答案完全一致 —— 说明判据对尾巴不敏感，"
        + "这条对照是恒真的",
    )
    # 顺带钉住"尾巴确实被丢了"这件事本身：末尾两个元素必须还是哨兵。
    assert_true(
        f32_data(got)[unsafe_offset=n - 1] == Float32(-7),
        "丢尾巴的版本竟然写到了最后一个元素，对照没有构造成功",
    )
    print("观测：丢掉尾巴的版本与正确答案相差 " + String(diff) + "，门如期判红")
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
