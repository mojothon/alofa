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
from alofa.core.tensor import MAX_RANK, Shape, TensorView, shape2
from alofa.core.text import read_text


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


def test_a_shape_built_in_a_step_is_not_a_container() raises:
    """`shape2` is inline metadata: the hot path's shapes do not allocate.

    What is pinned here is the *representation*, because the representation is
    what makes the allocation gate mean anything. A `List` inside `Shape` would
    be one allocation per view per step, and the gate that greps the engine for
    `List[` would still be green — the list would live in the core, one layer
    below where anyone was looking.
    """
    var shape = shape2(4, 5)
    assert_equal(shape.rank(), 2)
    assert_equal(shape.numel(), 20)
    assert_equal(shape.dims[0], 4)
    assert_equal(shape.dims[1], 5)


def test_a_rank_a_view_cannot_describe_is_refused() raises:
    """Rank over `MAX_RANK` is an error, never a truncation.

    A shape that quietly dropped a dimension would still report a rank and a
    sensible element count for the wrong tensor, which is the one failure this
    layer cannot afford: nothing downstream re-checks it.
    """
    var dims = List[Int]()
    for i in range(MAX_RANK + 1):
        dims.append(1)
    var caught = "no-error"
    try:
        _ = Shape(dims)
    except err:
        caught = err.name()
    assert_equal(caught, "out_of_range")


def step_path_sources() -> List[String]:
    """The files one step actually runs through.

    Not "the engine": a step also builds views, and the views are built in the
    core and in the kernels. A gate that only looked at `engine/` would have
    been green all along while every row of scratch allocated a shape.
    """
    var out = List[String]()
    out.append("src/alofa/core/tensor.mojo")
    out.append("src/alofa/engine/core.mojo")
    out.append("src/alofa/engine/executor.mojo")
    out.append("src/alofa/kernels/cpu/paged.mojo")
    out.append("src/alofa/kernels/cpu/scalar.mojo")
    out.append("src/alofa/kernels/cpu/segments.mojo")
    return out^


def shape_alloc_violations(path: String) raises -> Int:
    """How many ways this file builds a shape out of a heap container."""
    var text = read_text(path)
    var bad = 0
    if text.find("List[Int]()") >= 0:
        bad += 1
    if text.find("Shape(") >= 0 and path != "src/alofa/core/tensor.mojo":
        bad += 1
    return bad


def test_no_view_on_the_step_path_builds_a_heap_shape() raises:
    """Every file a step runs through builds shapes inline, or not at all.

    One row of scratch is one view; a step has rows. So a shape that allocated
    would be an allocation *per row per step* — the one number the engine's
    allocation promise is about — and it would sit in the core, below the layer
    the engine's own gate reads. `rows_view` is in `core/tensor.mojo` for
    exactly this reason: it is on the path, so it has to be under this gate.
    """
    var checked = 0
    for path in step_path_sources():
        assert_equal(
            shape_alloc_violations(path),
            0,
            "a shape is still being built out of a heap container in " + path,
        )
        checked += 1
    assert_equal(checked, 6, "the gate stopped checking some of the step path")


def test_the_shape_alloc_gate_can_fail() raises:
    """Red test: the old `rows_view` must be judged a violation."""
    assert_true(
        shape_alloc_violations("tests/fixtures/bad_shape_alloc.mojo") > 0,
        "the shape gate cannot see a shape built from a List",
    )


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
    var strides = InlineArray[Int, MAX_RANK](fill=1)
    strides[0] = 3
    strides[1] = 2
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
