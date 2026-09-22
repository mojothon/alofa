"""Tests for Hugging Face config.json and single-file safetensors loading."""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.tensor import f32_data
from alofa.model.arch.qwen import QwenConfig
from alofa.model.config import json_bool, json_float, json_int
from alofa.model.loader import TensorFile
from alofa.model.safetensors import SafeTensorFile

comptime FIXTURE = "tests/fixtures/model-formats"
comptime CONFIG = FIXTURE + "/config.json"
comptime SAFE = FIXTURE + "/tiny.safetensors"
# Widening is checked against a file that holds one payload in two dtypes; both
# are written by `scripts/gen_model_formats_fixtures.py`.
comptime MIXED = FIXTURE + "/tiny_mixed.safetensors"
comptime HALF = FIXTURE + "/tiny_f16.safetensors"


def test_huggingface_config_scalars() raises:
    assert_equal(json_int(CONFIG, "num_hidden_layers"), 2)
    assert_equal(json_int(CONFIG, "hidden_size"), 4)
    assert_equal(json_int(CONFIG, "num_attention_heads"), 2)
    assert_equal(json_int(CONFIG, "num_key_value_heads"), 1)
    assert_equal(json_int(CONFIG, "head_dim"), 2)
    assert_equal(json_int(CONFIG, "intermediate_size"), 8)
    assert_equal(json_int(CONFIG, "vocab_size"), 16)
    assert_true(json_float(CONFIG, "rms_norm_eps") > 0.0)
    assert_equal(json_bool(CONFIG, "tie_word_embeddings"), True)


def test_qwen_config_accepts_huggingface_json() raises:
    var cfg = QwenConfig(CONFIG)
    assert_equal(cfg.n_layers, 2)
    assert_equal(cfg.hidden, 4)
    assert_equal(cfg.n_kv_heads, 1)
    assert_equal(cfg.kv_dim(), 2)


def test_safetensors_metadata_and_f32_view() raises:
    var safe = SafeTensorFile(SAFE)
    assert_true(safe.has("x"))
    assert_true(safe.has("matrix"))
    assert_equal(safe.numel("x"), 2)
    assert_equal(safe.numel("matrix"), 6)
    var x = safe.ptr("x")
    assert_equal(x[0], 1.5)
    assert_equal(x[1], -2.0)
    safe.keep_alive()


def test_tensor_file_accepts_safetensors_path() raises:
    var file = TensorFile(SAFE)
    assert_equal(file.numel("matrix"), 6)
    var dims = file.dims("matrix")
    assert_equal(len(dims), 2)
    assert_equal(dims[0], 2)
    assert_equal(dims[1], 3)
    var view = file.view("matrix")
    assert_equal(view.shape.dims[0], 2)
    assert_equal(view.shape.dims[1], 3)
    file.keep_alive()


def test_safetensors_missing_tensor_is_named() raises:
    var safe = SafeTensorFile(SAFE)
    var raised = False
    try:
        _ = safe.numel("missing")
    except err:
        raised = err.name() == "invalid_argument"
    assert_true(raised, "missing safetensors tensor must be a named error")


def test_bf16_payload_widens_to_the_same_fp32() raises:
    """The six numbers, read back from bf16, bit-identical to the fp32 copy.

    The fixture stores them in both dtypes and every one of them is exactly
    representable in bf16, so "widened correctly" is not a tolerance here — it
    is equality. A shift in the wrong direction, a byte order read backwards,
    or a payload offset counted in elements rather than bytes all land orders
    of magnitude away.
    """
    var file = TensorFile(MIXED)
    var plain = f32_data(file.view("f32_row"))
    var widened = f32_data(file.view("bf16_row"))
    assert_equal(file.numel("bf16_row"), 6)
    for i in range(6):
        assert_equal(
            widened[unsafe_offset=i],
            plain[unsafe_offset=i],
            "widened bf16 differs from fp32 at " + String(i),
        )
    file.keep_alive()


def test_bf16_widening_lands_near_the_value_it_was_given() raises:
    """One third: not representable, so this is the neighbourhood check.

    bf16 carries eight mantissa bits, which bounds the gap at 2^-9 relative —
    about 7e-4 here. The point is that a widening which produces the right bit
    pattern for the exact values but nonsense for everything else fails here.
    """
    var file = TensorFile(MIXED)
    var third = Float64(f32_data(file.view("third"))[unsafe_offset=0])
    var gap = third - Float64(1.0) / Float64(3.0)
    if gap < 0.0:
        gap = -gap
    assert_true(
        gap <= Float64(0.002),
        "widened third is not near one third: " + String(third),
    )
    file.keep_alive()


def test_f16_payload_is_still_refused_by_name() raises:
    """FP16 is not bfloat16, and the loader has to say so.

    Widening bf16 is a shift; fp16 has a different exponent bias and a
    different mantissa width, so reading it the same way would be silent
    corruption. The negative control for the two tests above: those pass on
    bf16, this one keeps fp16 out.
    """
    var raised = False
    var message = ""
    try:
        _ = TensorFile(HALF)
    except err:
        raised = err.name() == "unsupported"
        message = err.message
    assert_true(raised, "F16 must be refused by name, got: " + message)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
