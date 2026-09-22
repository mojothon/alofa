"""Qwen configuration readers for TSV and Hugging Face JSON files."""

from alofa.core.error import ERR_INVALID_ARGUMENT, ERR_PARSE, AlofaError
from alofa.core.text import parse_float64, parse_int, read_text


def _text_slice(raw: String, start: Int, end: Int) -> String:
    var bytes = List[UInt8]()
    var source = raw.as_bytes()
    for i in range(start, end):
        bytes.append(source[i])
    return String(unsafe_from_utf8=bytes)


def json_has(path: String, key: String) raises AlofaError -> Bool:
    """Whether `key` appears as a JSON field in `path`.

    Not every field is in every file: Qwen2.5's `config.json` has no
    `head_dim`, because the reference derives it from the hidden size and the
    head count. A reader that demanded every field would refuse the artefact it
    exists to read, so the optional ones are asked about before they are read.
    """
    var raw = read_text(path).as_bytes()
    var needle = key.as_bytes()
    var i = 0
    while i + len(needle) + 2 <= len(raw):
        if raw[i] != 34:
            i += 1
            continue
        var matches = True
        for j in range(len(needle)):
            if raw[i + 1 + j] != needle[j]:
                matches = False
        if matches and raw[i + 1 + len(needle)] == 34:
            var p = i + len(needle) + 2
            while p < len(raw) and (
                raw[p] == 32 or raw[p] == 9 or raw[p] == 10 or raw[p] == 13
            ):
                p += 1
            if p < len(raw) and raw[p] == 58:
                return True
        i += 1
    return False


def json_value(path: String, key: String) raises AlofaError -> String:
    """Return one primitive JSON value for `key`.

    Supports quoted strings, booleans, and number tokens. Nested objects and
    arrays are skipped as values; this reader is deliberately strict about
    duplicate or missing scalar configuration fields.
    """
    var text = read_text(path)
    var raw = text.as_bytes()
    var needle = key.as_bytes()
    var found = False
    var answer = ""
    var i = 0
    while i + len(needle) + 2 <= len(raw):
        if raw[i] != 34:
            i += 1
            continue
        var matches = True
        for j in range(len(needle)):
            if raw[i + 1 + j] != needle[j]:
                matches = False
        if not matches or raw[i + 1 + len(needle)] != 34:
            i += 1
            continue
        var p = i + len(needle) + 2
        while p < len(raw) and (raw[p] == 32 or raw[p] == 9 or raw[p] == 10 or raw[p] == 13):
            p += 1
        if p >= len(raw) or raw[p] != 58:
            raise AlofaError(ERR_PARSE, "JSON key is not followed by colon", "key=" + key)
        p += 1
        while p < len(raw) and (raw[p] == 32 or raw[p] == 9 or raw[p] == 10 or raw[p] == 13):
            p += 1
        if p >= len(raw):
            raise AlofaError(ERR_PARSE, "JSON value is missing", "key=" + key)
        var start = p
        if raw[p] == 34:
            p += 1
            start = p
            while p < len(raw) and raw[p] != 34:
                if raw[p] == 92:
                    raise AlofaError(ERR_PARSE, "escaped JSON strings are unsupported", "key=" + key)
                p += 1
            if p >= len(raw):
                raise AlofaError(ERR_PARSE, "unterminated JSON string", "key=" + key)
        else:
            while p < len(raw) and raw[p] != 44 and raw[p] != 125 and raw[p] != 93 and raw[p] != 32 and raw[p] != 9 and raw[p] != 10 and raw[p] != 13:
                p += 1
        answer = _text_slice(text, start, p)
        if found:
            raise AlofaError(ERR_PARSE, "duplicate JSON configuration key", "key=" + key)
        found = True
        i = p + 1
    if not found:
        raise AlofaError(ERR_INVALID_ARGUMENT, "configuration key not found", "key=" + key)
    return answer


def _ascii_int(text: String) raises AlofaError -> Int:
    """Parse an unsigned JSON integer without materializing a byte-list String."""
    var raw = text.as_bytes()
    if len(raw) == 0:
        raise AlofaError(ERR_PARSE, "empty integer", "text=" + text)
    var out = 0
    var start = 0
    var sign = 1
    if raw[0] == 45:
        sign = -1
        start = 1
    for i in range(start, len(raw)):
        if raw[i] < 48 or raw[i] > 57:
            raise AlofaError(ERR_PARSE, "invalid integer", "text=" + text)
        out = out * 10 + Int(raw[i] - 48)
    return sign * out


def json_int(path: String, key: String) raises AlofaError -> Int:
    """Read one integer-valued JSON configuration field."""
    return _ascii_int(json_value(path, key))


def json_float(path: String, key: String) raises AlofaError -> Float64:
    """Read a JSON number, including exponent notation."""
    var token = json_value(path, key)
    var raw = token.as_bytes()
    var split = -1
    for i in range(len(raw)):
        if raw[i] == 101 or raw[i] == 69:
            split = i
            break
    if split < 0:
        return parse_float64(token)
    var exponent = _ascii_int(_text_slice(token, split + 1, len(raw)))
    var mantissa = _text_slice(token, 0, split)
    var value = parse_float64(mantissa)
    if exponent > 0:
        for _ in range(exponent):
            value *= 10.0
    elif exponent < 0:
        for _ in range(-exponent):
            value /= 10.0
    return value


def json_bool(path: String, key: String) raises AlofaError -> Bool:
    """Read a JSON boolean configuration field."""
    var value = json_value(path, key)
    if value == "true":
        return True
    if value == "false":
        return False
    raise AlofaError(ERR_PARSE, "JSON value is not boolean", "key=" + key + " value=" + value)
