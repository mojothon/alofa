"""Tests for the platform-primitive bindings.

These assert that alofa's own bindings work — that a returned fd is usable and
that a mapping round-trips bytes. They are not the platform probe:
`tests/capability/test_libc_ffi.mojo` answers "does this machine have these
primitives?" and stands on its own so it can still answer that if these
bindings regress.

Every fd opened here is closed in the same test, so a failing assertion cannot
leak descriptors into the rest of the suite.

Run:
    pixi run mojo run -I src tests/unit/test_core_ffi.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.ffi import (
    close_fd,
    enable_reuseport,
    epoll_create,
    eventfd_counter,
    monotonic_ms,
    socket_stream,
    timerfd_monotonic,
)
from alofa.core.ffi.mem import (
    MADV_WILLNEED,
    is_mapped,
    madvise,
    mmap_anonymous,
    munmap,
)


def test_monotonic_ms_is_non_negative() raises:
    assert_true(monotonic_ms() >= 0, "monotonic_ms returned a negative value")


def test_monotonic_ms_does_not_go_backwards() raises:
    var first = monotonic_ms()
    var second = monotonic_ms()
    assert_true(second >= first, "monotonic clock moved backwards")


def test_socket_stream_returns_a_closable_fd() raises:
    var fd = socket_stream()
    assert_true(fd >= 0, "socket_stream() did not return a valid fd")
    assert_equal(close_fd(fd), 0)


def test_epoll_create_returns_a_closable_fd() raises:
    var fd = epoll_create()
    assert_true(fd >= 0, "epoll_create() did not return a valid fd")
    assert_equal(close_fd(fd), 0)


def test_timerfd_returns_a_closable_fd() raises:
    var fd = timerfd_monotonic()
    assert_true(fd >= 0, "timerfd_monotonic() did not return a valid fd")
    assert_equal(close_fd(fd), 0)


def test_eventfd_returns_a_closable_fd() raises:
    var fd = eventfd_counter()
    assert_true(fd >= 0, "eventfd_counter() did not return a valid fd")
    assert_equal(close_fd(fd), 0)


def test_reuseport_can_be_enabled_on_a_socket() raises:
    var fd = socket_stream()
    assert_true(fd >= 0, "socket_stream() failed")
    assert_equal(enable_reuseport(fd), 0)
    assert_equal(close_fd(fd), 0)


def test_anonymous_mapping_round_trips_a_byte() raises:
    var p = mmap_anonymous(4096)
    p[unsafe_offset=0] = 42
    assert_equal(p[unsafe_offset=0], 42)
    assert_equal(munmap(p, 4096), 0)


def test_live_mapping_reports_as_mapped() raises:
    var p = mmap_anonymous(4096)
    assert_true(is_mapped(p, 4096), "a fresh mapping reported as unmapped")
    assert_equal(munmap(p, 4096), 0)


def test_unmapped_region_no_longer_reports_as_mapped() raises:
    """`is_mapped` must answer False after munmap, without touching memory."""
    var p = mmap_anonymous(4096)
    assert_equal(munmap(p, 4096), 0)
    assert_true(not is_mapped(p, 4096), "unmapped region still reports as mapped")


def test_madvise_is_accepted_for_a_live_mapping() raises:
    var p = mmap_anonymous(4096)
    assert_equal(madvise(p, 4096, MADV_WILLNEED), 0)
    assert_equal(munmap(p, 4096), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
