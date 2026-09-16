"""Tests for the structured-log layer.

The assertions cover `log_line` (a pure function) rather than captured stdout:
checking that `print` was called only proves that `print` works.

Run:
    pixi run mojo run -I src tests/unit/test_core_log.mojo
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from alofa.core.log import (
    LEVEL_DEBUG,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_WARN,
    Logger,
    escape_json,
    level_name,
    log_line,
)


def test_level_name_covers_every_level() raises:
    assert_equal(level_name(LEVEL_DEBUG), "debug")
    assert_equal(level_name(LEVEL_INFO), "info")
    assert_equal(level_name(LEVEL_WARN), "warn")
    assert_equal(level_name(LEVEL_ERROR), "error")


def test_log_line_renders_exact_json() raises:
    var line = log_line(0, LEVEL_INFO, "core", "arena.reset", "", "")
    assert_equal(
        line,
        "{\"ts_ms\":0,\"level\":\"info\",\"component\":\"core\","
        + "\"event\":\"arena.reset\",\"msg\":\"\",\"detail\":\"\"}",
    )


def test_log_line_carries_timestamp_and_detail() raises:
    var line = log_line(42, LEVEL_ERROR, "core", "weights.load", "nope", "path=x")
    assert_true(line.find("\"ts_ms\":42") >= 0, "timestamp field is wrong")
    assert_true(line.find("\"detail\":\"path=x\"") >= 0, "detail field is wrong")


def test_escape_json_escapes_quotes_and_backslashes() raises:
    assert_equal(escape_json("a\"b"), "a\\\"b")
    assert_equal(escape_json("a\\b"), "a\\\\b")
    assert_equal(escape_json("plain"), "plain")


def test_escaped_message_keeps_the_line_parseable() raises:
    """A quote in a message must not break the record."""
    var line = log_line(0, LEVEL_WARN, "core", "e", "say \"hi\"", "")
    assert_true(line.find("say \\\"hi\\\"") >= 0, "message was not escaped")


def test_logger_filters_below_its_minimum_level() raises:
    var log = Logger("core", LEVEL_WARN)
    assert_false(log.is_enabled(LEVEL_DEBUG))
    assert_false(log.is_enabled(LEVEL_INFO))
    assert_true(log.is_enabled(LEVEL_WARN))
    assert_true(log.is_enabled(LEVEL_ERROR))


def test_logger_debug_level_enables_everything() raises:
    var log = Logger("core", LEVEL_DEBUG)
    assert_true(log.is_enabled(LEVEL_DEBUG))
    assert_true(log.is_enabled(LEVEL_ERROR))


def test_logger_keeps_its_component() raises:
    var log = Logger("arena", LEVEL_INFO)
    assert_equal(log.component, "arena")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
