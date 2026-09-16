"""Tests for read-only file mappings.

The oracle is Mojo's own `FileHandle`: the mapping must expose exactly the
bytes an ordinary read would. A mapping that returned the wrong region would
still look plausible on its own, so the comparison is byte-for-byte against a
second, independent reader.

Run:
    pixi run mojo run -I src tests/unit/test_core_mmap.mojo
"""

from std.io import FileHandle
from std.os import stat
from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.mmap import MappedFile

comptime FIXTURE = "tests/fixtures/ledger_bad.md"


def test_mapping_size_matches_stat() raises:
    var mapped = MappedFile(FIXTURE)
    assert_equal(mapped.size, Int(stat(FIXTURE).st_size))


def test_mapping_exposes_the_same_bytes_as_filehandle() raises:
    var mapped = MappedFile(FIXTURE)
    var handle = FileHandle(FIXTURE, "r")
    var expected = handle.read(16)
    handle.close()

    # Byte-for-byte, not codepoint-for-codepoint: a mapping holds bytes, and
    # the fixture is UTF-8, so comparing against codepoints would only ever
    # agree on the ASCII prefix.
    var compared = 0
    var i = 0
    for expected_byte in expected.as_bytes():
        assert_equal(Int(mapped.byte_at(i)), Int(expected_byte))
        i += 1
        compared += 1
    assert_true(compared > 0, "compared no bytes, so the test proved nothing")


def test_byte_past_the_end_is_rejected() raises:
    var mapped = MappedFile(FIXTURE)
    var caught = "no-error"
    try:
        _ = mapped.byte_at(mapped.size)
    except err:
        caught = err.name()
    assert_equal(caught, "out_of_range")


def test_missing_file_is_rejected() raises:
    var caught = "no-error"
    try:
        var mapped = MappedFile("tests/fixtures/this_file_does_not_exist.md")
        _ = mapped.size
    except err:
        caught = err.name()
    assert_equal(caught, "io")


def test_empty_file_is_rejected() raises:
    """/dev/null is zero-length; mmap would fail with EINVAL, so name it."""
    var caught = "no-error"
    try:
        var mapped = MappedFile("/dev/null")
        _ = mapped.size
    except err:
        caught = err.name()
    assert_equal(caught, "invalid_argument")


def test_pointer_into_the_mapping_stays_valid() raises:
    """`ptr` hands out an untracked pointer; `keep_alive` is what keeps it so."""
    var mapped = MappedFile(FIXTURE)
    var start = mapped.ptr()
    assert_equal(Int(start[unsafe_offset=0]), Int(mapped.byte_at(0)))
    mapped.keep_alive()


def test_access_hints_are_accepted() raises:
    var mapped = MappedFile(FIXTURE)
    assert_equal(mapped.advise_sequential(), 0)
    assert_equal(mapped.advise_willneed(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
