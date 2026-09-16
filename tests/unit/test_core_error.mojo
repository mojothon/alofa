"""Tests for the named-error layer.

Errors are asserted exactly: a code is an integer and a name is a string, so
`assert_equal` everywhere — there is nothing here to approximate.

Run:
    pixi run mojo run -I src tests/unit/test_core_error.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_IO,
    ERR_LAST,
    ERR_NONE,
    ERR_OUT_OF_RANGE,
    ERR_PARSE,
    AlofaError,
    error_name,
)


def test_every_error_code_has_a_name() raises:
    """No code from ERR_NONE to ERR_LAST may map to "unknown".

    This is the tripwire for adding a code and forgetting to name it. Without
    it, the new code silently reports itself as "unknown" everywhere it
    surfaces, and the omission is only noticed when someone reads a log.
    """
    var code = 0
    while code <= ERR_LAST:
        assert_true(
            error_name(code) != "unknown",
            "error code has no name: " + String(code),
        )
        code += 1


def test_known_codes_map_to_expected_names() raises:
    assert_equal(error_name(ERR_NONE), "none")
    assert_equal(error_name(ERR_INVALID_ARGUMENT), "invalid_argument")
    assert_equal(error_name(ERR_OUT_OF_RANGE), "out_of_range")


def test_code_beyond_the_last_is_unknown() raises:
    assert_equal(error_name(ERR_LAST + 1), "unknown")
    assert_equal(error_name(9999), "unknown")


def test_error_carries_code_message_and_detail() raises:
    var err = AlofaError(ERR_OUT_OF_RANGE, "index past end", "index=7 size=3")
    assert_equal(err.code, ERR_OUT_OF_RANGE)
    assert_equal(err.message, "index past end")
    assert_equal(err.detail, "index=7 size=3")
    assert_equal(err.name(), "out_of_range")


def test_error_without_detail_has_empty_detail() raises:
    var err = AlofaError(ERR_IO, "read failed")
    assert_equal(err.detail, "")


def test_named_code_survives_being_raised() raises:
    """The whole point of a named error: the code reaches the handler."""
    var caught = -1  # sentinel: stays -1 if nothing is raised
    try:
        raise AlofaError(ERR_OUT_OF_RANGE, "out of bounds", "index=9")
    except err:
        caught = err.code
    assert_equal(caught, ERR_OUT_OF_RANGE)


def test_error_renders_name_message_and_detail() raises:
    var text = String(AlofaError(ERR_PARSE, "bad header", "offset=0"))
    assert_true(text.find("parse") >= 0, "render is missing the code name")
    assert_true(text.find("bad header") >= 0, "render is missing the message")
    assert_true(text.find("offset=0") >= 0, "render is missing the detail")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
