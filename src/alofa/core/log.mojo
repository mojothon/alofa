"""Structured JSONL logging for the L0 core layer.

One object per line with a fixed field set, so `verify/` can consume logs
without an ad-hoc parser per call site:

    {"ts_ms":0,"level":"info","component":"core","event":"arena.reset","msg":"","detail":""}

Logging **never raises**. A logger that can fail on the error path would mask
the very error it is being asked to report, and would turn "record this" into
a second failure mode for every caller. Everything here is best-effort: the
clock falls back to 0 (see `monotonic_ns`) and output goes straight to stdout.

Timestamps come from `CLOCK_MONOTONIC`, not wall time: what `verify/` compares
is the interval between events, and those intervals must not jump when the
system clock is adjusted.

`log_line` is a pure function and is what the tests assert on — checking log
output by capturing stdout would only prove that `print` works.

Run:
    pixi run mojo run -I src tests/unit/test_core_log.mojo
"""

from alofa.core.error import AlofaError
from alofa.core.ffi import monotonic_ms

comptime LEVEL_DEBUG = 10
comptime LEVEL_INFO = 20
comptime LEVEL_WARN = 30
comptime LEVEL_ERROR = 40


def level_name(level: Int) -> String:
    """The identifier used in the `level` field for a numeric level."""
    if level <= LEVEL_DEBUG:
        return "debug"
    elif level <= LEVEL_INFO:
        return "info"
    elif level <= LEVEL_WARN:
        return "warn"
    return "error"


def escape_json(s: String) -> String:
    """Escape the two characters that would otherwise break the JSON line.

    Minimal on purpose: log fields are identifiers and paths, not arbitrary
    documents. Escaping only `"` and `\\` keeps the line valid without
    pretending to be a general JSON writer.
    """
    var out = String("")
    var i = 0
    while i < s.byte_length():
        var ch = s[byte=i]
        if ch == "\"":
            out += "\\\""
        elif ch == "\\":
            out += "\\\\"
        else:
            var piece = String(ch)
            out += piece
        i += 1
    return out


def log_line(
    ts_ms: Int,
    level: Int,
    component: String,
    event: String,
    message: String,
    detail: String,
) -> String:
    """Render one JSONL record. Pure — same inputs always give same bytes."""
    return "{{\"ts_ms\":{},\"level\":\"{}\",\"component\":\"{}\",\"event\":\"{}\",\"msg\":\"{}\",\"detail\":\"{}\"}}".format(
        ts_ms,
        level_name(level),
        escape_json(component),
        escape_json(event),
        escape_json(message),
        escape_json(detail),
    )


struct Logger(Copyable, Movable):
    """A component-tagged emitter with a minimum level.

    The level filter is a field so that a caller can hold a quiet logger for
    a hot path and a verbose one elsewhere without a global switch.
    """

    var component: String
    var level: Int

    def __init__(out self, component: String, level: Int = LEVEL_INFO):
        self.component = component
        self.level = level

    def is_enabled(self, level: Int) -> Bool:
        """Whether a record at `level` would be emitted."""
        return level >= self.level

    def emit(
        self,
        level: Int,
        event: String,
        message: String = "",
        detail: String = "",
    ):
        """Write one record if `level` passes the filter. Never raises."""
        if not self.is_enabled(level):
            return
        print(
            log_line(
                monotonic_ms(),
                level,
                self.component,
                event,
                message,
                detail,
            )
        )

    def debug(self, event: String, message: String = "", detail: String = ""):
        self.emit(LEVEL_DEBUG, event, message, detail)

    def info(self, event: String, message: String = "", detail: String = ""):
        self.emit(LEVEL_INFO, event, message, detail)

    def warn(self, event: String, message: String = "", detail: String = ""):
        self.emit(LEVEL_WARN, event, message, detail)

    def error(self, event: String, message: String = "", detail: String = ""):
        self.emit(LEVEL_ERROR, event, message, detail)

    def emit_error(self, event: String, err: AlofaError):
        """Record a failure, carrying its named code in `detail`.

        The code rides along as `code=<name>` rather than in the message so
        that a consumer can filter on it after the message is reworded.
        """
        self.emit(LEVEL_ERROR, event, err.message, "code=" + err.name())
