"""CPU backends for the L1 compute layer.

Backends are selected at compile time, by which module the caller imports,
rather than at run time by a flag: the specialization this repository is built
around is that a compute loop is compiled once per (format, ISA, block shape)
triple, and a run-time switch would collapse them all back into one loop.
"""
