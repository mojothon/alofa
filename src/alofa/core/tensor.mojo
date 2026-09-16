"""Tensor views for the L0 core layer.

A view never owns its data. Ownership stays with whoever allocated it — an
arena, a mapping, a caller's buffer — and the view records only shape, strides,
element type and a byte offset into memory it does not control. That is what
makes slicing free: a slice is a new view over the same bytes, never a copy.

Strides are in **elements**, not bytes; byte offsets multiply by the element
size at the point of use. Keeping them in elements lets the same view describe
a tensor whose dtype is not known until the file is read.

Run:
    pixi run mojo run -I src tests/unit/test_core_tensor.mojo
"""

from alofa.core.dtype import elem_size_bytes
from alofa.core.error import ERR_OUT_OF_RANGE, ERR_SHAPE_MISMATCH, AlofaError
from alofa.core.ffi.mem import RawPtr


struct Shape(Copyable, Movable):
    """A tensor's extents — the one place rank and element count are derived."""

    var dims: List[Int]

    def __init__(out self, dims: List[Int]):
        self.dims = dims.copy()

    def rank(self) -> Int:
        return len(self.dims)

    def numel(self) -> Int:
        """Total elements. An empty dims list is a scalar: one element."""
        var total = 1
        for d in self.dims:
            total *= d
        return total

    def contiguous_strides(self) -> List[Int]:
        """Row-major (C-order) element strides, e.g. [2, 3] -> [3, 1]."""
        var reversed_strides = List[Int]()
        var acc = 1
        var i = len(self.dims) - 1
        while i >= 0:
            reversed_strides.append(acc)
            acc *= self.dims[i]
            i -= 1

        var strides = List[Int]()
        var j = len(reversed_strides) - 1
        while j >= 0:
            strides.append(reversed_strides[j])
            j -= 1
        return strides^


struct TensorView(Copyable, Movable):
    """A non-owning view: shape, strides, element type, byte offset."""

    var data: RawPtr
    var shape: Shape
    var strides: List[Int]
    var dtype: Int
    var byte_offset: Int

    def __init__(
        out self,
        data: RawPtr,
        shape: Shape,
        dtype: Int,
        byte_offset: Int = 0,
    ):
        self.data = data
        self.shape = shape.copy()
        self.strides = shape.contiguous_strides()
        self.dtype = dtype
        self.byte_offset = byte_offset

    def __init__(
        out self,
        data: RawPtr,
        shape: Shape,
        strides: List[Int],
        dtype: Int,
        byte_offset: Int = 0,
    ):
        self.data = data
        self.shape = shape.copy()
        self.strides = strides.copy()
        self.dtype = dtype
        self.byte_offset = byte_offset

    def rank(self) -> Int:
        return self.shape.rank()

    def numel(self) -> Int:
        return self.shape.numel()

    def byte_size(self) -> Int:
        """Bytes spanned by this view's elements."""
        return self.numel() * elem_size_bytes(self.dtype)

    def is_contiguous(self) -> Bool:
        """Whether the strides are exactly the row-major ones."""
        var expected = 1
        var i = self.rank() - 1
        while i >= 0:
            if self.strides[i] != expected:
                return False
            expected *= self.shape.dims[i]
            i -= 1
        return True

    def byte_offset_at(self, indices: List[Int]) raises AlofaError -> Int:
        """Byte offset of `indices` from the start of the underlying memory.

        Raises rather than returning -1: an out-of-range index reaching a
        consumer is a bug, and a silent -1 would read whatever sits just before
        the tensor instead of failing.
        """
        if len(indices) != self.rank():
            raise AlofaError(
                ERR_SHAPE_MISMATCH,
                "index rank does not match tensor rank",
                "given=" + String(len(indices)) + " want=" + String(self.rank()),
            )
        var offset = self.byte_offset
        var i = 0
        while i < self.rank():
            var index = indices[i]
            if index < 0 or index >= self.shape.dims[i]:
                raise AlofaError(
                    ERR_OUT_OF_RANGE,
                    "index out of range",
                    "dim=" + String(i) + " index=" + String(index),
                )
            offset += index * self.strides[i] * elem_size_bytes(self.dtype)
            i += 1
        return offset

    def slice_dim(
        self, dim: Int, start: Int, length: Int
    ) raises AlofaError -> TensorView:
        """A narrower view along one dimension — a new view, never a copy."""
        if dim < 0 or dim >= self.rank():
            raise AlofaError(
                ERR_OUT_OF_RANGE, "slice dim out of range", "dim=" + String(dim)
            )
        if start < 0 or length < 0 or start + length > self.shape.dims[dim]:
            raise AlofaError(
                ERR_OUT_OF_RANGE,
                "slice range out of bounds",
                "dim="
                + String(dim)
                + " start="
                + String(start)
                + " len="
                + String(length),
            )

        var dims = List[Int]()
        var i = 0
        while i < self.rank():
            dims.append(self.shape.dims[i])
            i += 1
        dims[dim] = length

        return TensorView(
            self.data,
            Shape(dims),
            self.strides,
            self.dtype,
            self.byte_offset
            + start * self.strides[dim] * elem_size_bytes(self.dtype),
        )
