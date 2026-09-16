"""Tests for the arena allocator.

Alignment is checked through `bytes_used` rather than by inspecting pointer
values: Mojo offers no way to turn a pointer into an integer here, and the
padding that alignment introduces is exactly what the byte count shows.

That allocations do not overlap is checked by writing through one pointer and
reading through the next — a bump allocator that forgot to advance would pass
a pointer-equality test and fail this one.

Run:
    pixi run mojo run -I src tests/unit/test_core_memory.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.memory import Arena


def test_first_allocation_starts_at_zero() raises:
    var arena = Arena(1024)
    assert_equal(arena.bytes_used(), 0)
    _ = arena.alloc(10)
    assert_equal(arena.bytes_used(), 10)


def test_alignment_padding_shows_up_in_bytes_used() raises:
    """Two 1-byte allocations at alignment 64 occupy 65 bytes, not 2."""
    var arena = Arena(1024)
    _ = arena.alloc(1, 64)
    _ = arena.alloc(1, 64)
    assert_equal(arena.bytes_used(), 65)


def test_allocations_do_not_overlap() raises:
    var arena = Arena(1024)
    var first = arena.alloc(8)
    var second = arena.alloc(8)
    first[unsafe_offset=0] = 1
    second[unsafe_offset=0] = 2
    assert_equal(first[unsafe_offset=0], 1)
    assert_equal(second[unsafe_offset=0], 2)
    # The arena's last use has to come after the last use of its allocations,
    # or Mojo unmaps the mapping while these two pointers are still in play.
    arena.keep_alive()


def test_bytes_free_tracks_capacity() raises:
    var arena = Arena(1024)
    assert_equal(arena.bytes_free(), 1024)
    _ = arena.alloc(24)
    assert_equal(arena.bytes_free(), 1000)


def test_reset_reclaims_everything() raises:
    var arena = Arena(1024)
    _ = arena.alloc(100)
    assert_equal(arena.bytes_used(), 100)
    arena.reset()
    assert_equal(arena.bytes_used(), 0)
    assert_equal(arena.bytes_free(), 1024)


def test_region_is_reusable_after_reset() raises:
    """Reset has to do more than zero a counter — the bytes must be usable."""
    var arena = Arena(1024)
    var first = arena.alloc(8)
    first[unsafe_offset=0] = 7
    arena.reset()
    var second = arena.alloc(8)
    second[unsafe_offset=0] = 9
    assert_equal(second[unsafe_offset=0], 9)
    arena.keep_alive()


def test_exceeding_capacity_is_rejected() raises:
    var arena = Arena(64)
    var caught = "no-error"
    try:
        _ = arena.alloc(65)
    except err:
        caught = err.name()
    assert_equal(caught, "capacity")


def test_alignment_that_is_not_a_power_of_two_is_rejected() raises:
    var arena = Arena(64)
    var caught = "no-error"
    try:
        _ = arena.alloc(4, 3)
    except err:
        caught = err.name()
    assert_equal(caught, "alignment")


def test_zero_size_allocation_is_rejected() raises:
    var arena = Arena(64)
    var caught = "no-error"
    try:
        _ = arena.alloc(0)
    except err:
        caught = err.name()
    assert_equal(caught, "invalid_argument")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
