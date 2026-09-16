"""Tests for element types and block-quantization layout metadata.

The byte counts are asserted against the GGUF / llama.cpp layouts. If they are
wrong, every weight offset derived from them is wrong too, and the mistake
surfaces much later as a model that loads cleanly and then produces garbage —
so these numbers are pinned exactly, not approximately.

Run:
    pixi run mojo run -I src tests/unit/test_core_dtype.mojo
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from alofa.core.dtype import (
    DT_BF16,
    DT_FP16,
    DT_FP32,
    DT_INT32,
    DT_INT64,
    DT_INT8,
    DT_UINT8,
    QUANT_FP8,
    QUANT_INT8,
    QUANT_NONE,
    QUANT_Q4_0,
    QUANT_Q4_K,
    QUANT_Q8_0,
    elem_name,
    elem_size_bytes,
    quant_block_size,
    quant_bytes_per_block,
    quant_has_sub_scale,
    quant_name,
    quant_scale_bytes,
    quant_storage_bytes,
)


def test_element_sizes() raises:
    assert_equal(elem_size_bytes(DT_FP32), 4)
    assert_equal(elem_size_bytes(DT_FP16), 2)
    assert_equal(elem_size_bytes(DT_BF16), 2)
    assert_equal(elem_size_bytes(DT_INT8), 1)
    assert_equal(elem_size_bytes(DT_UINT8), 1)
    assert_equal(elem_size_bytes(DT_INT32), 4)
    assert_equal(elem_size_bytes(DT_INT64), 8)


def test_unknown_element_type_reports_zero_size() raises:
    assert_equal(elem_size_bytes(999), 0)
    assert_equal(elem_name(999), "unknown")


def test_q4_0_layout_matches_gguf() raises:
    """32 elements per block in 18 bytes: 2 B scale + 32 * 4 bit."""
    assert_equal(quant_block_size[QUANT_Q4_0](), 32)
    assert_equal(quant_bytes_per_block[QUANT_Q4_0](), 18)
    assert_equal(quant_scale_bytes[QUANT_Q4_0](), 2)


def test_q8_0_layout_matches_gguf() raises:
    assert_equal(quant_block_size[QUANT_Q8_0](), 32)
    assert_equal(quant_bytes_per_block[QUANT_Q8_0](), 34)
    assert_equal(quant_scale_bytes[QUANT_Q8_0](), 2)


def test_q4_k_layout_matches_gguf() raises:
    """256 elements in 144 bytes: 2 B d + 2 B dmin + 12 B scales + 128 B data."""
    assert_equal(quant_block_size[QUANT_Q4_K](), 256)
    assert_equal(quant_bytes_per_block[QUANT_Q4_K](), 144)
    assert_equal(quant_scale_bytes[QUANT_Q4_K](), 4)


def test_only_q4_k_has_a_sub_scale() raises:
    assert_true(quant_has_sub_scale[QUANT_Q4_K]())
    assert_false(quant_has_sub_scale[QUANT_Q4_0]())
    assert_false(quant_has_sub_scale[QUANT_Q8_0]())


def test_non_block_formats_report_no_block() raises:
    assert_equal(quant_block_size[QUANT_NONE](), 0)
    assert_equal(quant_bytes_per_block[QUANT_INT8](), 0)


def test_storage_bytes_round_up_to_whole_blocks() raises:
    """Weight files store element counts, so the tail block is always padded."""
    assert_equal(quant_storage_bytes[QUANT_Q4_0](32), 18)
    assert_equal(quant_storage_bytes[QUANT_Q4_0](33), 36)
    assert_equal(quant_storage_bytes[QUANT_Q4_0](64), 36)


def test_storage_bytes_for_non_block_formats() raises:
    assert_equal(quant_storage_bytes[QUANT_INT8](10), 10)


def test_quant_names_cover_every_format() raises:
    assert_equal(quant_name(QUANT_NONE), "none")
    assert_equal(quant_name(QUANT_Q4_0), "q4_0")
    assert_equal(quant_name(QUANT_Q4_K), "q4_k")
    assert_equal(quant_name(QUANT_Q8_0), "q8_0")
    assert_equal(quant_name(QUANT_INT8), "int8")
    assert_equal(quant_name(QUANT_FP8), "fp8")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
