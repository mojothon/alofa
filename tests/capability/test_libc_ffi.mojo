"""Capability gate: libc FFI availability.

This is the load-bearing assumption behind alofa's service layer (L5). Mojo
1.0.0 has no language-level `async` (roadmap Phase 2, not started) and no
stdlib networking, so the event loop must be built directly on libc
primitives through `external_call`.

If this test fails on a target platform, the pure-Mojo service layer is NOT
viable there and the thin-sidecar deployment form described in
`docs/plan/02-architecture.md` must be used instead.

Run:
    pixi run mojo run tests/capability/test_libc_ffi.mojo

NOTE: never call `fork()` from a file run under `mojo run` - the JIT shares the
compiler process and it will crash. `fork()` is only exercised from compiled
binaries, and is covered by the e2e suite instead.
"""

from std.ffi import external_call
from std.testing import TestSuite, assert_true


def test_socket() raises:
    """Verify socket(AF_INET, SOCK_STREAM, 0) returns a valid fd."""
    # AF_INET=2, SOCK_STREAM=1 on linux-64.
    var fd = external_call["socket", Int32](Int32(2), Int32(1), Int32(0))
    assert_true(fd >= 0, "socket() failed: libc FFI unusable")
    _ = external_call["close", Int32](fd)


def test_epoll() raises:
    """Verify epoll_create1 works - the event loop needs no language async."""
    var fd = external_call["epoll_create1", Int32](Int32(0))
    assert_true(fd >= 0, "epoll_create1() failed")
    _ = external_call["close", Int32](fd)


def test_timerfd() raises:
    """Verify timerfd_create works - the scheduler tick is a loop event."""
    comptime CLOCK_MONOTONIC = 1
    var fd = external_call["timerfd_create", Int32](
        Int32(CLOCK_MONOTONIC), Int32(0)
    )
    assert_true(fd >= 0, "timerfd_create() failed")
    _ = external_call["close", Int32](fd)


def test_eventfd() raises:
    """Verify eventfd works - the wakeup channel between master and workers."""
    var fd = external_call["eventfd", Int32](UInt32(0), Int32(0))
    assert_true(fd >= 0, "eventfd() failed")
    _ = external_call["close", Int32](fd)


def test_reuseport_setsockopt() raises:
    """Verify SO_REUSEPORT can be set - the horizontal scaling mechanism.

    AF_INET=2, SOCK_STREAM=1, SOL_SOCKET=1, SO_REUSEPORT=15 on linux-64.
    """
    var fd = external_call["socket", Int32](Int32(2), Int32(1), Int32(0))
    assert_true(fd >= 0, "socket() failed")
    var one = Int32(1)
    var rc = external_call["setsockopt", Int32](
        fd, Int32(1), Int32(15), Pointer(to=one), UInt32(4)
    )
    assert_true(rc == 0, "setsockopt(SO_REUSEPORT) failed")
    _ = external_call["close", Int32](fd)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
