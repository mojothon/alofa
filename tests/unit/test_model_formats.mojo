"""Tests for Hugging Face config.json and single-file safetensors loading."""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.model.arch.qwen import QwenConfig
from alofa.model.config import json_bool, json_float, json_int
from alofa.model.loader import TensorFile
from alofa.model.safetensors import SafeTensorFile

comptime FIXTURE = "tests/fixtures/model-formats"
comptime CONFIG = FIXTURE + "/config.json"
comptime SAFE = FIXTURE + "/tiny.safetensors"


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
