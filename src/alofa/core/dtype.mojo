"""Data types and block-quantization layout metadata.

Two things live here:

1. Element types (`DT_FP32`, ...) — what a tensor's values are.
2. Quantization formats (`QUANT_Q4_0`, ...) and their block layouts.

Block layouts are exposed as functions taking the format as a **comptime
parameter**, not as a runtime table lookup. They are the foundation of the
compile-time backend specialization design: a compute loop gets instantiated
per (quantization format, ISA, block shape), and that triple has to be fixed
before it is compiled. A runtime lookup here would push every such loop back
into a runtime switch, which is exactly what the design avoids.

Element types are plain `Int` codes rather than Mojo's `DType` because dtype
information arrives from a file at runtime; `DType` is a compile-time type and
cannot carry a value read from disk.

Run:
    pixi run mojo run -I src tests/unit/test_core_dtype.mojo
"""

comptime DT_FP32 = 0
comptime DT_FP16 = 1
comptime DT_BF16 = 2
comptime DT_INT8 = 3
comptime DT_UINT8 = 4
comptime DT_INT32 = 5
comptime DT_INT64 = 6
comptime DT_LAST = 6

comptime QUANT_NONE = 0
comptime QUANT_Q4_0 = 1
comptime QUANT_Q4_K = 2
comptime QUANT_Q8_0 = 3
comptime QUANT_INT8 = 4
comptime QUANT_FP8 = 5
comptime QUANT_LAST = 5


def elem_size_bytes(dt: Int) -> Int:
    """Bytes per element, or 0 for an unrecognized type.

    0 rather than a raise: this is called while describing what is on disk, and
    an unknown type is worth reporting with the surrounding context (which
    tensor, which file) rather than as a bare number.
    """
    if dt == DT_FP32:
        return 4
    elif dt == DT_FP16:
        return 2
    elif dt == DT_BF16:
        return 2
    elif dt == DT_INT8:
        return 1
    elif dt == DT_UINT8:
        return 1
    elif dt == DT_INT32:
        return 4
    elif dt == DT_INT64:
        return 8
    return 0


def elem_name(dt: Int) -> String:
    """The identifier for an element type code, or "unknown"."""
    if dt == DT_FP32:
        return "fp32"
    elif dt == DT_FP16:
        return "fp16"
    elif dt == DT_BF16:
        return "bf16"
    elif dt == DT_INT8:
        return "int8"
    elif dt == DT_UINT8:
        return "uint8"
    elif dt == DT_INT32:
        return "int32"
    elif dt == DT_INT64:
        return "int64"
    return "unknown"


@always_inline
def quant_block_size[fmt: Int]() -> Int:
    """Elements per quantization block; 0 for formats that are not block-wise."""
    if fmt == QUANT_Q4_0:
        return 32
    elif fmt == QUANT_Q4_K:
        return 256
    elif fmt == QUANT_Q8_0:
        return 32
    return 0


@always_inline
def quant_bytes_per_block[fmt: Int]() -> Int:
    """Storage bytes per block, including its scale; 0 if not block-wise."""
    if fmt == QUANT_Q4_0:
        return 18  # 2 B fp16 scale + 32 * 4 bit
    elif fmt == QUANT_Q4_K:
        return 144  # 2 B d + 2 B dmin + 12 B scales + 256 * 4 bit
    elif fmt == QUANT_Q8_0:
        return 34  # 2 B fp16 scale + 32 * 8 bit
    return 0


@always_inline
def quant_scale_bytes[fmt: Int]() -> Int:
    """Bytes of scale data at the head of a block; 0 if not block-wise."""
    if fmt == QUANT_Q4_0:
        return 2
    elif fmt == QUANT_Q4_K:
        return 4  # d + dmin, both fp16
    elif fmt == QUANT_Q8_0:
        return 2
    return 0


@always_inline
def quant_has_sub_scale[fmt: Int]() -> Bool:
    """Whether the format carries a second-level scale (q4_k's per-64 group)."""
    return fmt == QUANT_Q4_K


def quant_storage_bytes[fmt: Int](n_elements: Int) -> Int:
    """Storage bytes for `n_elements`, rounded up to whole blocks.

    This is how a stored array's on-disk size is derived: quantized formats do
    not store a byte count, they store an element count, and the tail block is
    always padded out to a full block.
    """
    var block = quant_block_size[fmt]()
    if block == 0:
        return n_elements
    var blocks = (n_elements + block - 1) // block
    return blocks * quant_bytes_per_block[fmt]()


def quant_name(fmt: Int) -> String:
    """The identifier for a quantization format code, or "unknown"."""
    if fmt == QUANT_NONE:
        return "none"
    elif fmt == QUANT_Q4_0:
        return "q4_0"
    elif fmt == QUANT_Q4_K:
        return "q4_k"
    elif fmt == QUANT_Q8_0:
        return "q8_0"
    elif fmt == QUANT_INT8:
        return "int8"
    elif fmt == QUANT_FP8:
        return "fp8"
    return "unknown"
