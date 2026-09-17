"""Comparison mathematics for differential testing.

Three functions, each with one job: the largest gap between two arrays, the
largest magnitude in one, and the cosine similarity of two. They are separated
from the fixtures and the loaders on purpose — a tolerance decision is easier
to review when the arithmetic it rests on is a page of code with no I/O in it.

Everything accumulates in `Float64`. The values under comparison are fp32, and
the interesting comparisons run over 151936-wide vectors: a fp32 accumulator's
error there is comparable to the gaps being tested for, and a similarity whose
own error is the size of its verdict proves nothing.

Run:
    pixi run mojo run -I src tests/unit/test_layer0_parity.mojo
"""

from std.math import sqrt

from alofa.core.tensor import F32Ptr


def max_abs(values: F32Ptr, n: Int) -> Float64:
    """Largest magnitude in `values`; 0 for an empty range."""
    var best = Float64(0)
    for i in range(n):
        var v = Float64(values[unsafe_offset=i])
        if v < 0:
            v = -v
        if v > best:
            best = v
    return best


def max_abs_diff(a: F32Ptr, b: F32Ptr, n: Int) -> Float64:
    """Largest element-wise gap between two arrays of the same length."""
    var worst = Float64(0)
    for i in range(n):
        var d = Float64(a[unsafe_offset=i]) - Float64(b[unsafe_offset=i])
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    return worst


def cosine(a: F32Ptr, b: F32Ptr, n: Int) -> Float64:
    """Cosine similarity of two arrays.

    Returns 0 when either side is all zeros rather than dividing by zero: an
    all-zero reference is a broken fixture, and the shape comparison around
    this call is what reports that.
    """
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(n):
        var x = Float64(a[unsafe_offset=i])
        var y = Float64(b[unsafe_offset=i])
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0
    return dot / (sqrt(na) * sqrt(nb))
