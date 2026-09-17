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

from alofa.core.dtype import DT_FP32, elem_name, elem_size_bytes
from alofa.core.error import (
    ERR_OUT_OF_RANGE,
    ERR_SHAPE_MISMATCH,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.ffi.mem import RawPtr

# The element type every numerical layer starts from. It is named here, next to
# the view, rather than in a compute module: a view is what carries the dtype,
# so the accessor that has to check it belongs beside it.
comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]

# The largest rank a view can describe. Eight is not a guess at what anybody
# will need — it is the number past which a shape stops being a shape and
# becomes a container, and this layer does not own one of those.
comptime MAX_RANK = 8


def rows_view(base: RawPtr, rows: Int, cols: Int) raises AlofaError -> TensorView:
    """A `[rows, cols]` fp32 view over memory that already exists.

    A shorter view over a longer allocation — how both phases share one buffer
    without reallocating when the token count changes.

    It lives in the core rather than in a caller because *a view is what this
    layer is for*, and because the shape it builds has to be built the way every
    other shape in a step is built: inline, with no container. A caller that
    spelled the shape out would put the allocation one layer below the gate that
    forbids it, which is the same allocation with a better alibi.
    """
    if rows <= 0 or cols <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "view needs a positive number of rows and columns",
            "rows=" + String(rows) + " cols=" + String(cols),
        )
    return TensorView(base, shape2(rows, cols), DT_FP32)


def copy_dims(
    src: InlineArray[Int, MAX_RANK],
) -> InlineArray[Int, MAX_RANK]:
    """A rank-bounded array is movable, not copyable — this is the copy.

    Element by element, because the type refuses to be duplicated implicitly and
    every caller here has a borrowed value it must not consume.
    """
    var out = InlineArray[Int, MAX_RANK](fill=1)
    for i in range(MAX_RANK):
        out[i] = src[i]
    return out^


def shape2(a: Int, b: Int) raises AlofaError -> Shape:
    """A `[a, b]` shape, built without touching the allocator.

    The hot path calls this once per row of scratch per step, and a shape that
    allocated would be an allocation *per row per step* — unbounded in the one
    place P2 says must be bounded. Ranks past `MAX_RANK` are refused rather than
    truncated: a shape that quietly drops a dimension produces a view that is
    exactly the right size for the wrong tensor.
    """
    return _shape_from(a, b, 1, 2)


def shape3(a: Int, b: Int, c: Int) raises AlofaError -> Shape:
    """A `[a, b, c]` shape, built without touching the allocator."""
    return _shape_from(a, b, c, 3)


def _shape_from(a: Int, b: Int, c: Int, rank: Int) raises AlofaError -> Shape:
    var out = InlineArray[Int, MAX_RANK](fill=1)
    out[0] = a
    out[1] = b
    out[2] = c
    for i in range(rank):
        if out[i] <= 0:
            raise AlofaError(
                ERR_OUT_OF_RANGE,
                "a dimension must be positive",
                "dim=" + String(i) + " value=" + String(out[i]),
            )
    return Shape(out^, rank)


struct Shape(Copyable, Movable):
    """A tensor's extents — the one place rank and element count are derived.

    Rank-bounded and inline: `dims` is a fixed-length register array, not a
    heap container, so a shape is a value that can be built and copied inside a
    loop that is not allowed to allocate. `Shape(list)` still exists for the
    places that read a rank off a file — those run once, at load time — but
    everything inside a step calls `shape2` / `shape3` instead.
    """

    var dims: InlineArray[Int, MAX_RANK]
    var n: Int

    def __init__(out self, dims: InlineArray[Int, MAX_RANK], n: Int):
        self.dims = copy_dims(dims)
        self.n = n

    def __copyinit__(out self, other: Self):
        self.dims = copy_dims(other.dims)
        self.n = other.n

    def copy(self) -> Shape:
        return Shape(copy_dims(self.dims), self.n)

    def __init__(out self, dims: List[Int]) raises AlofaError:
        if len(dims) > MAX_RANK:
            raise AlofaError(
                ERR_OUT_OF_RANGE,
                "rank is larger than a view can describe",
                "rank=" + String(len(dims)) + " max=" + String(MAX_RANK),
            )
        self.dims = InlineArray[Int, MAX_RANK](fill=1)
        self.n = len(dims)
        for i in range(len(dims)):
            self.dims[i] = dims[i]

    def rank(self) -> Int:
        return self.n

    def numel(self) -> Int:
        """Total elements. A rank of zero is a scalar: one element."""
        var total = 1
        for i in range(self.n):
            total *= self.dims[i]
        return total

    def contiguous_strides(self) -> InlineArray[Int, MAX_RANK]:
        """Row-major (C-order) element strides, e.g. [2, 3] -> [3, 1]."""
        var strides = InlineArray[Int, MAX_RANK](fill=1)
        var acc = 1
        var i = self.n - 1
        while i >= 0:
            strides[i] = acc
            acc *= self.dims[i]
            i -= 1
        return strides^


struct TensorView(Copyable, Movable):
    """A non-owning view: shape, strides, element type, byte offset."""

    var data: RawPtr
    var shape: Shape
    var strides: InlineArray[Int, MAX_RANK]
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
        strides: InlineArray[Int, MAX_RANK],
        dtype: Int,
        byte_offset: Int = 0,
    ):
        self.data = data
        self.shape = shape.copy()
        self.strides = copy_dims(strides)
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

        # The dims are copied out of an inline array into an inline array:
        # `slice_dim` is how a view is narrowed, and narrowing must not be the
        # one operation in a step that reaches for the heap.
        var dims = copy_dims(self.shape.dims)
        dims[dim] = length

        return TensorView(
            self.data,
            Shape(dims^, self.rank()),
            self.strides,
            self.dtype,
            self.byte_offset
            + start * self.strides[dim] * elem_size_bytes(self.dtype),
        )


def view3(base: RawPtr, a: Int, b: Int, c: Int) raises AlofaError -> TensorView:
    """A `[a, b, c]` fp32 view over memory that already exists.

    Here rather than in a caller because a shape is rank-bounded metadata, and
    the layers that run inside a busy loop are the ones that may not grow
    anything: they call this instead of spelling the shape out.
    """
    return TensorView(base, shape3(a, b, c), DT_FP32)


def f32_data(view: TensorView) raises AlofaError -> F32Ptr:
    """Typed access to a contiguous fp32 view's elements.

    Two things are checked rather than assumed. The dtype, because every
    consumer here indexes in `Float32` units and a view that turned out to be
    fp16 would read twice as far as it should. Contiguity, because a strided
    view's elements are not `numel()` floats apart from each other, and a
    slice of a tensor is exactly such a view.

    The pointer is untracked, so its owner's lifetime is the caller's to
    state — see `Arena.keep_alive` and `MappedFile.keep_alive`.
    """
    if view.dtype != DT_FP32:
        raise AlofaError(
            ERR_UNSUPPORTED,
            "view is not fp32",
            "dtype=" + elem_name(view.dtype),
        )
    if not view.is_contiguous():
        raise AlofaError(
            ERR_UNSUPPORTED, "view is not contiguous", "strides-mismatch"
        )
    return view.data.unsafe_offset(view.byte_offset).unsafe_bitcast[Float32]()
