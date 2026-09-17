"""L1 compute layer — the arithmetic every model is built from.

This layer knows shape and element types (L0) and nothing above it: it has no
notion of a model file, a schedule or a request. `cpu/` holds the host backends,
starting with a scalar one whose only job is to be obviously correct — it is the
oracle that a specialized backend is measured against, so clarity beats speed
here and every loop is written in the order the formula is written.

Submodules are imported directly by callers
(`from alofa.kernels.cpu.scalar import ...`).
"""
