"""Reading numbers and whole files from text, with nothing left implicit.

Mojo 1.0 ships no number parsing for `String`, and the alternative — calling
`strtod` through FFI — would make reading a configuration file depend on
libc's locale (a decimal comma would then be somebody else's bug). The parsers
here accept exactly what this repository's exporters write: decimal integers,
and fixed-point decimals without an exponent.

`read_text` converts the untyped I/O errors of `FileHandle` into named
`AlofaError`s, so that a caller can catch by name without dragging an
untyped `raises` through every signature above it.

Run:
    pixi run mojo run -I src tests/unit/test_core_text.mojo
"""

from std.io import FileHandle

from alofa.core.error import ERR_IO, ERR_PARSE, AlofaError


def read_text(path: String) raises AlofaError -> String:
    """The whole file as a string, or a named I/O error."""
    var handle: FileHandle
    try:
        handle = FileHandle(path, "r")
    except:
        raise AlofaError(ERR_IO, "cannot open file", "path=" + path)
    var text: String
    try:
        text = handle.read()
    except:
        raise AlofaError(ERR_IO, "cannot read file", "path=" + path)
    try:
        handle.close()
    except:
        # The read succeeded; a close failure must not turn it into an error.
        pass
    return text


def parse_int(text: String) raises AlofaError -> Int:
    """Parse an optionally negative decimal integer; nothing else is accepted.

    Rejecting anything that is not a digit is what stops a mis-typed column
    from being read as a number: `parse_int("")` and `parse_int("1e3")` are
    both errors rather than 0 and 1000.
    """
    var n = text.byte_length()
    if n == 0:
        raise AlofaError(ERR_PARSE, "empty integer", "")
    var raw = text.as_bytes()
    var i = 0
    var negative = False
    if raw[0] == 45:  # '-'
        negative = True
        i = 1
    if i >= n:
        raise AlofaError(ERR_PARSE, "sign with no digits", "text=" + text)
    var value = 0
    while i < n:
        var c = Int(raw[i])
        if c < 48 or c > 57:
            raise AlofaError(ERR_PARSE, "not a decimal digit", "text=" + text)
        value = value * 10 + (c - 48)
        i += 1
    if negative:
        value = -value
    return value


def parse_float64(text: String) raises AlofaError -> Float64:
    """Parse a fixed-point decimal. Exponents are not accepted.

    Exporters in this repository write floats with a fixed-point format, so a
    number carrying an exponent would mean the file was written by something
    else — better to fail here than to silently accept a value nobody agreed
    on.
    """
    var n = text.byte_length()
    if n == 0:
        raise AlofaError(ERR_PARSE, "empty number", "")
    var raw = text.as_bytes()
    var i = 0
    var negative = False
    if raw[0] == 45:  # '-'
        negative = True
        i = 1
    elif raw[0] == 43:  # '+'
        i = 1

    var whole = Float64(0)
    var digits = 0
    while i < n and raw[i] != 46:  # '.'
        var c = Int(raw[i])
        if c < 48 or c > 57:
            raise AlofaError(ERR_PARSE, "not a decimal digit", "text=" + text)
        whole = whole * 10 + Float64(c - 48)
        digits += 1
        i += 1
    if digits == 0:
        raise AlofaError(ERR_PARSE, "no digits before the point", "text=" + text)

    var fraction = Float64(0)
    var scale = Float64(1)
    if i < n and raw[i] == 46:
        i += 1
        while i < n:
            var c = Int(raw[i])
            if c < 48 or c > 57:
                raise AlofaError(ERR_PARSE, "not a decimal digit", "text=" + text)
            scale = scale / Float64(10)
            fraction = fraction + Float64(c - 48) * scale
            i += 1

    var value = whole + fraction
    if negative:
        value = -value
    return value
