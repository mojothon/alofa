"""L0 core layer — the base every other layer builds on.

Nothing here may reference a concept from a higher layer (see
`tests/capability/test_layering.mojo`, which enforces that on every CI run).

Submodules are imported directly by callers (`from alofa.core.error import
...`); this module re-exports the common symbols for convenience.
"""

from .error import (
    ERR_ALIGNMENT,
    ERR_CAPACITY,
    ERR_DOUBLE_FREE,
    ERR_INVALID_ARGUMENT,
    ERR_IO,
    ERR_LAST,
    ERR_MMAP,
    ERR_NONE,
    ERR_NOT_INITIALIZED,
    ERR_OUT_OF_MEMORY,
    ERR_OUT_OF_RANGE,
    ERR_PARSE,
    ERR_SHAPE_MISMATCH,
    ERR_UNSUPPORTED,
    AlofaError,
    error_name,
)
from .ffi import Timespec, monotonic_ms, monotonic_ns
from .log import (
    LEVEL_DEBUG,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_WARN,
    Logger,
    escape_json,
    level_name,
    log_line,
)
