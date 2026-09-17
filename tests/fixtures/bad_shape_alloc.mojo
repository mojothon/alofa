"""A red test: a view that builds its shape out of a heap container.

This is what `rows_view` used to look like, and what it must never look like
again. Every row of scratch gets a view, so a shape that allocates is an
allocation per row per step — and the allocation is in `core/tensor.mojo`, one
layer below where the engine's zero-allocation gate was looking.
"""

from alofa.core.error import AlofaError
from alofa.core.ffi.mem import RawPtr
from alofa.core.tensor import DT_FP32, Shape, TensorView


def rows_view(base: RawPtr, rows: Int, cols: Int) raises AlofaError -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    return TensorView(base, Shape(dims), DT_FP32)
