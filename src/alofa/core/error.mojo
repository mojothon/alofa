"""Named errors for alofa's L0 core layer.

Every failure is reported as an `AlofaError` carrying a **compile-time named
code**. Bare strings are never used as error codes.

Why named codes instead of strings:

- a string code cannot be enumerated, so `verify/` has no way to assert that a
  failure path is covered;
- a typo in a string silently creates a *new* error kind at runtime rather
  than failing to compile;
- `verify/` consumes these errors directly and needs identifiers that stay
  stable when a human-facing message is reworded.

Codes are `comptime` constants, so `AlofaError(ERR_OUT_OF_RANGE, ...)` cannot
misspell its category: a wrong name is a compile error, not a new error kind.

Code values are a contract once anything persists them: **never renumber an
existing code, only append.**

Run:
    pixi run mojo run -I src tests/unit/test_core_error.mojo
"""

comptime ERR_NONE = 0
comptime ERR_INVALID_ARGUMENT = 1
comptime ERR_OUT_OF_RANGE = 2
comptime ERR_OUT_OF_MEMORY = 3
comptime ERR_ALIGNMENT = 4
comptime ERR_CAPACITY = 5
comptime ERR_IO = 6
comptime ERR_MMAP = 7
comptime ERR_UNSUPPORTED = 8
comptime ERR_SHAPE_MISMATCH = 9
comptime ERR_NOT_INITIALIZED = 10
comptime ERR_DOUBLE_FREE = 11
comptime ERR_PARSE = 12

comptime ERR_LAST = 12


def error_name(code: Int) -> String:
    """Return the identifier of `code`, or "unknown" if none is defined.

    The test suite enumerates every code from `ERR_NONE` through `ERR_LAST`
    and asserts that none maps to "unknown". That is what catches a newly
    added code that was never named here — without it, a code would silently
    report itself as "unknown" everywhere it surfaced.
    """
    if code == ERR_NONE:
        return "none"
    elif code == ERR_INVALID_ARGUMENT:
        return "invalid_argument"
    elif code == ERR_OUT_OF_RANGE:
        return "out_of_range"
    elif code == ERR_OUT_OF_MEMORY:
        return "out_of_memory"
    elif code == ERR_ALIGNMENT:
        return "alignment"
    elif code == ERR_CAPACITY:
        return "capacity"
    elif code == ERR_IO:
        return "io"
    elif code == ERR_MMAP:
        return "mmap"
    elif code == ERR_UNSUPPORTED:
        return "unsupported"
    elif code == ERR_SHAPE_MISMATCH:
        return "shape_mismatch"
    elif code == ERR_NOT_INITIALIZED:
        return "not_initialized"
    elif code == ERR_DOUBLE_FREE:
        return "double_free"
    elif code == ERR_PARSE:
        return "parse"
    return "unknown"


struct AlofaError(Copyable, Movable, Writable):
    """A failure with a named code, a human message, and machine-readable detail.

    `detail` carries the context that a consumer needs to act on — the
    offending value, the path, the shape — so `verify/` can assert on it
    without parsing the human-facing `message`.
    """

    var code: Int
    var message: String
    var detail: String

    def __init__(out self, code: Int, message: String):
        self.code = code
        self.message = message
        self.detail = ""

    def __init__(out self, code: Int, message: String, detail: String):
        self.code = code
        self.message = message
        self.detail = detail

    def name(self) -> String:
        """The identifier of this error's code."""
        return error_name(self.code)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.name(), "(", self.code, "): ", self.message)
        if self.detail.byte_length() > 0:
            writer.write(" [", self.detail, "]")
