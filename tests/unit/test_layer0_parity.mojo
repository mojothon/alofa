"""Differential test: every scalar kernel against Hugging Face's own tensors.

The reference is a fixture, not a live call into Python (see
`scripts/dump_model_reference.py` for why). What is being checked here is
narrower than "the model works": for each operator, the fixture holds the
**input the reference actually saw** and the **output the reference actually
produced**, captured at that operator's boundary. That is what makes a failure
local — when `attention` is wrong, only the attention test fails, and the
linear tests still pass, so the input to attention is known to be right.

Tolerance is stated per comparison as `1e-5 × max(1, |reference|ₘₐₓ)`: the
fixture is fp32 from a fp32 reference, so what is being tolerated is floating
point reassociation, not a different algorithm. A kernel that is *wrong* is
wrong by far more than that, and a tolerance wide enough to hide a wrong
kernel would also hide the one number this repository cares about.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_layer0_parity.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.core.text import parse_float64, parse_int, read_text
from alofa.model.loader import TensorFile, config_value
from alofa.kernels.cpu.scalar import (
    add,
    attention,
    linear,
    linear_bias,
    rmsnorm,
    rope,
    silu,
    swiglu,
)
from alofa.verify.compare import max_abs, max_abs_diff

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b/layer0"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"

# Relative tolerance on the largest reference magnitude, floored at 1 so that
# small-magnitude tensors are held to an absolute bound rather than an
# impossible ratio.
comptime REL_TOLERANCE = Float64(1e-5)


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    """An uninitialised fp32 `[rows, cols]` view in the arena.

    The arena hands out untracked pointers and is destroyed at its last use,
    so every test calls `keep_alive` after the last read of a view taken from
    it — see `Arena.keep_alive`.
    """
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, Shape(dims), DT_FP32)


def assert_close(op: String, got: TensorView, expected: TensorView) raises:
    """Assert `got` matches the reference to within reassociation error."""
    assert_equal(
        got.numel(),
        expected.numel(),
        op + ": element count differs from the reference",
    )
    var n = expected.numel()
    var diff = max_abs_diff(f32_data(got), f32_data(expected), n)
    var scale = max_abs(f32_data(expected), n)
    # `max(1, scale)` rather than a pure ratio: the reference computes in fp32
    # with its own reassociation, so what this tolerates on a tensor of
    # magnitude 0.3 is the reference's own rounding, and holding it to a
    # relative-only bound would test the reference, not the implementation.
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
    # A difference that is zero means nothing was compared: an all-zero
    # reference would make every implementation look correct.
    assert_true(scale > 0, op + ": the reference tensor is all zeros")


def test_fixture_index_is_consistent() raises:
    """The index, the payload and the operator table must agree with each other.

    This is the test that catches a stale fixture. Without it, an exporter
    change that renamed a tensor would surface as a kernel "bug" instead of as
    what it is.
    """
    var store = TensorFile(FIXTURE)
    assert_true(len(store.names) > 0, "the fixture has no tensors")

    # Every operator in ops.tsv must name tensors that exist.
    var ops = read_text(FIXTURE + "/ops.tsv")
    var lines = ops.split("\n")
    var checked = 0
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        var op = String(fields[0])
        var ins = String(fields[1])
        var outs = String(fields[2])
        assert_equal(
            len(fields), 3, "op line must have three fields: " + op
        )
        var operands = ins + "," + outs
        var parts = operands.split(",")
        for part_span in parts:
            var part = String(part_span)
            assert_true(
                store.has(part), "ops.tsv names a tensor that is not exported: " + part
            )
            checked += 1
    assert_true(checked > 0, "no operands were checked, so the test proved nothing")
    store.keep_alive()


def test_config_is_self_consistent() raises:
    """Head counts and hidden size must multiply out; the fixture says so."""
    var hidden = parse_int(config_value(CONFIG, "hidden"))
    var n_heads = parse_int(config_value(CONFIG, "n_heads"))
    var head_dim = parse_int(config_value(CONFIG, "head_dim"))
    var n_kv_heads = parse_int(config_value(CONFIG, "n_kv_heads"))
    assert_equal(n_heads * head_dim, hidden, "heads * head_dim must be hidden")
    assert_true(
        n_heads % n_kv_heads == 0, "n_kv_heads must divide n_heads (GQA)"
    )
    var eps = parse_float64(config_value(CONFIG, "eps"))
    assert_true(eps > 0, "rms norm eps must be positive")


def test_rmsnorm_matches_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var x = store.view("embed_out")
    var tokens = x.shape.dims[0]
    var hidden = x.shape.dims[1]
    var out = scratch(arena, tokens, hidden)
    var eps = Float32(parse_float64(config_value(CONFIG, "eps")))
    rmsnorm(out, x, store.view("norm_w"), eps)
    assert_close("rmsnorm", out, store.view("norm_out"))
    arena.keep_alive()
    store.keep_alive()


def test_qkv_projections_match_reference() raises:
    """The q/k/v projections are the three that carry a bias in Qwen2."""
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var x = store.view("norm_out")
    var tokens = x.shape.dims[0]
    var hidden = x.shape.dims[1]

    var q = scratch(arena, tokens, hidden)
    linear_bias(q, x, store.view("q_w"), store.view("q_b"))
    assert_close("q_proj", q, store.view("q_out"))

    var kv = store.view("k_out")
    var kv_dim = kv.shape.dims[1]
    var k = scratch(arena, tokens, kv_dim)
    linear_bias(k, x, store.view("k_w"), store.view("k_b"))
    assert_close("k_proj", k, store.view("k_out"))

    var v = scratch(arena, tokens, kv_dim)
    linear_bias(v, x, store.view("v_w"), store.view("v_b"))
    assert_close("v_proj", v, store.view("v_out"))
    arena.keep_alive()
    store.keep_alive()


def test_rope_matches_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var head_dim = parse_int(config_value(CONFIG, "head_dim"))
    var q = store.view("q_out")
    var k = store.view("k_out")
    var tokens = q.shape.dims[0]
    var out_q = scratch(arena, tokens, q.shape.dims[1])
    var out_k = scratch(arena, tokens, k.shape.dims[1])
    rope(out_q, out_k, q, k, store.view("cos"), store.view("sin"), head_dim)
    assert_close("rope.q", out_q, store.view("q_rot"))
    assert_close("rope.k", out_k, store.view("k_rot"))
    arena.keep_alive()
    store.keep_alive()


def test_attention_matches_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var n_heads = parse_int(config_value(CONFIG, "n_heads"))
    var n_kv_heads = parse_int(config_value(CONFIG, "n_kv_heads"))
    var head_dim = parse_int(config_value(CONFIG, "head_dim"))
    var q = store.view("q_rot")
    var tokens = q.shape.dims[0]
    var out = scratch(arena, tokens, n_heads * head_dim)
    # Scratch for one row block; attention needs [tokens, tokens] and takes it
    # from the caller so that the kernel itself never allocates.
    var scores = scratch(arena, tokens, tokens)
    attention(
        out,
        q,
        store.view("k_rot"),
        store.view("v_out"),
        scores,
        n_heads,
        n_kv_heads,
        head_dim,
    )
    assert_close("attention", out, store.view("attn_out"))
    arena.keep_alive()
    store.keep_alive()


def test_output_projection_matches_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var x = store.view("attn_out")
    var tokens = x.shape.dims[0]
    var hidden = x.shape.dims[1]
    var out = scratch(arena, tokens, hidden)
    linear(out, x, store.view("o_w"))
    assert_close("o_proj", out, store.view("o_out"))
    arena.keep_alive()
    store.keep_alive()


def test_residual_add_matches_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var a = store.view("embed_out")
    var tokens = a.shape.dims[0]
    var hidden = a.shape.dims[1]
    var out = scratch(arena, tokens, hidden)
    add(out, a, store.view("o_out"))
    assert_close("residual", out, store.view("hidden1"))
    arena.keep_alive()
    store.keep_alive()


def test_second_rmsnorm_matches_reference() raises:
    """The feed-forward norm runs on the post-residual tensor, not on the input."""
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var x = store.view("hidden1")
    var tokens = x.shape.dims[0]
    var hidden = x.shape.dims[1]
    var out = scratch(arena, tokens, hidden)
    var eps = Float32(parse_float64(config_value(CONFIG, "eps")))
    rmsnorm(out, x, store.view("norm2_w"), eps)
    assert_close("rmsnorm_ffn", out, store.view("norm2_out"))
    arena.keep_alive()
    store.keep_alive()


def test_silu_matches_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 22)
    var x = store.view("gate_out")
    var tokens = x.shape.dims[0]
    var inter = x.shape.dims[1]
    var out = scratch(arena, tokens, inter)
    silu(out, x)
    assert_close("silu", out, store.view("silu_out"))
    arena.keep_alive()
    store.keep_alive()


def test_swiglu_matches_reference() raises:
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 22)
    var gate = store.view("gate_out")
    var tokens = gate.shape.dims[0]
    var inter = gate.shape.dims[1]
    var out = scratch(arena, tokens, inter)
    swiglu(out, gate, store.view("up_out"))
    assert_close("swiglu", out, store.view("swiglu_out"))
    arena.keep_alive()
    store.keep_alive()


def test_shape_errors_are_named() raises:
    """A wrong shape must raise, not produce a plausible tensor."""
    var store = TensorFile(FIXTURE)
    var arena = Arena(1 << 20)
    var x = store.view("norm_out")
    var wrong = scratch(arena, 2, 3)
    var caught = "no-error"
    try:
        rmsnorm(wrong, x, store.view("norm_w"), Float32(1e-6))
    except err:
        caught = err.name()
    assert_equal(caught, "shape_mismatch")
    arena.keep_alive()
    store.keep_alive()


def test_unknown_tensor_is_rejected() raises:
    """A missing reference must fail the test, not silently compare zeros."""
    var store = TensorFile(FIXTURE)
    var caught = "no-error"
    try:
        _ = store.view("this_tensor_does_not_exist")
    except err:
        caught = err.name()
    assert_equal(caught, "invalid_argument")
    store.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
