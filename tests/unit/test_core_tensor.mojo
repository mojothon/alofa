"""Tests for tensor views.

The load-bearing property is that a view does not own its data: a slice must
address the same bytes as the view it came from. The tests prove that by
writing through one view and reading through another, rather than by checking
that a pointer field happens to compare equal.

Run:
    pixi run mojo run -I src tests/unit/test_core_tensor.mojo
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from alofa.core.dtype import DT_FP32, DT_INT8
from alofa.core.ffi.mem import mmap_anonymous, munmap
from alofa.core.tensor import Shape, TensorView


def test_shape_reports_rank_and_numel() raises:
    var dims: List[Int] = [2, 3, 4]
    var shape = Shape(dims)
    assert_equal(shape.rank(), 3)
    assert_equal(shape.numel(), 24)


def test_scalar_shape_has_one_element() raises:
    var dims: List[Int] = []
    var shape = Shape(dims)
    assert_equal(shape.rank(), 0)
    assert_equal(shape.numel(), 1)


def test_contiguous_strides_are_row_major() raises:
    var dims: List[Int] = [2, 3]
    var strides = Shape(dims).contiguous_strides()
    assert_equal(strides[0], 3)
    assert_equal(strides[1], 1)


def test_byte_offset_of_a_contiguous_index() raises:
    """Element [1, 2] of a 2x3 fp32 tensor sits at (1 * 3 + 2) * 4 = 20."""
    var p = mmap_anonymous(64)
    var dims: List[Int] = [2, 3]
    var view = TensorView(p, Shape(dims), DT_FP32)
    var index: List[Int] = [1, 2]
    assert_equal(view.byte_offset_at(index), 20)
    _ = munmap(p, 64)


def test_byte_size_scales_with_element_type() raises:
    var p = mmap_anonymous(64)
    var dims: List[Int] = [4, 4]
    assert_equal(TensorView(p, Shape(dims), DT_FP32).byte_size(), 64)
    assert_equal(TensorView(p, Shape(dims), DT_INT8).byte_size(), 16)
    _ = munmap(p, 64)


def test_out_of_range_index_is_rejected() raises:
    var p = mmap_anonymous(64)
    var dims: List[Int] = [2, 3]
    var view = TensorView(p, Shape(dims), DT_FP32)
    var caught = "no-error"
    var index: List[Int] = [2, 0]
    try:
        _ = view.byte_offset_at(index)
    except err:
        caught = err.name()
    assert_equal(caught, "out_of_range")
    _ = munmap(p, 64)


def test_index_of_the_wrong_rank_is_rejected() raises:
    var p = mmap_anonymous(64)
    var dims: List[Int] = [2, 3]
    var view = TensorView(p, Shape(dims), DT_FP32)
    var caught = "no-error"
    var index: List[Int] = [1]
    try:
        _ = view.byte_offset_at(index)
    except err:
        caught = err.name()
    assert_equal(caught, "shape_mismatch")
    _ = munmap(p, 64)


def test_default_view_is_contiguous() raises:
    var p = mmap_anonymous(64)
    var dims: List[Int] = [2, 3]
    var view = TensorView(p, Shape(dims), DT_FP32)
    assert_true(view.is_contiguous(), "a default view should be contiguous")
    _ = munmap(p, 64)


def test_explicit_strides_can_make_a_view_non_contiguous() raises:
    var p = mmap_anonymous(64)
    var dims: List[Int] = [2, 3]
    var strides: List[Int] = [3, 2]
    var view = TensorView(p, Shape(dims), strides, DT_FP32)
    assert_false(view.is_contiguous(), "strided view reported as contiguous")
    _ = munmap(p, 64)


def test_slice_addresses_the_same_bytes_as_its_parent() raises:
    """A slice is a view, not a copy: a write through one is seen by the other."""
    var p = mmap_anonymous(64)
    var dims: List[Int] = [4, 4]
    var view = TensorView(p, Shape(dims), DT_FP32)
    var row = view.slice_dim(0, 2, 1)
    var in_slice: List[Int] = [0, 1]
    p[unsafe_offset=row.byte_offset_at(in_slice)] = 99
    var in_parent: List[Int] = [2, 1]
    assert_equal(p[unsafe_offset=view.byte_offset_at(in_parent)], 99)
    _ = munmap(p, 64)


def test_slice_out_of_bounds_is_rejected() raises:
    var p = mmap_anonymous(64)
    var dims: List[Int] = [4, 4]
    var view = TensorView(p, Shape(dims), DT_FP32)
    var caught = "no-error"
    try:
        _ = view.slice_dim(0, 3, 5)
    except err:
        caught = err.name()
    assert_equal(caught, "out_of_range")
    _ = munmap(p, 64)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
