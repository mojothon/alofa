"""POSIX primitives used by the L0 core layer.

Thin `external_call` wrappers — no retry logic, no buffering, no state. This
layer describes what the platform offers; policy lives with the caller.

Resource-returning calls hand back the raw libc value: an fd is `>= 0` on
success and `-1` on failure. Nothing here inspects `errno`, because the only
thing a caller can do with most of these failures is report them, and the
caller has the context to say which operation failed.

`tests/capability/test_libc_ffi.mojo` proves these primitives exist on a given
machine. That test is a *platform probe*: it calls libc directly and
deliberately does not import this module, so it keeps answering "does this
platform have epoll?" even if these bindings regress.

Run:
    pixi run mojo run -I src tests/unit/test_core_ffi.mojo
"""

from std.ffi import external_call

# --- clocks ----------------------------------------------------------------
comptime CLOCK_REALTIME = 0
comptime CLOCK_MONOTONIC = 1

# --- sockets ---------------------------------------------------------------
comptime AF_INET = 2
comptime SOCK_STREAM = 1
comptime SOL_SOCKET = 1
comptime SO_REUSEPORT = 15

# --- open flags (Linux x86-64) ---------------------------------------------
comptime O_RDONLY = 0
comptime O_WRONLY = 1
comptime O_RDWR = 2
comptime O_CLOEXEC = 0x80000

# `AT_FDCWD` — "resolve relative to the current directory", which makes
# `openat` a drop-in for `open`. `open` is deliberately avoided here: it is
# declared with a different signature elsewhere in the stdlib, and two
# `external_call`s that disagree about one libc symbol fail to lower.
comptime AT_FDCWD = -100

# --- fd / flag bits (Linux x86-64) -----------------------------------------
comptime EPOLL_CLOEXEC = 0x80000
comptime TFD_CLOEXEC = 0x80000
comptime TFD_NONBLOCK = 0x800
comptime EFD_CLOEXEC = 0x80000
comptime EFD_NONBLOCK = 0x800


@fieldwise_init
struct Timespec(Copyable, Movable):
    """`struct timespec`: two `long`s, which are `Int` on linux-64."""

    var tv_sec: Int
    var tv_nsec: Int


def monotonic_ns() -> Int:
    """Nanoseconds on `CLOCK_MONOTONIC`.

    Never raises and never returns a partial value: a logger that can fail
    would mask the error it is trying to report. On failure it returns 0,
    which a consumer reads as "no timestamp" — `clock_gettime(CLOCK_MONOTONIC)`
    does not fail on Linux, so this branch is defensive only.
    """
    var ts = Timespec(0, 0)
    var rc = external_call["clock_gettime", Int32](
        Int32(CLOCK_MONOTONIC), Pointer(to=ts)
    )
    if rc != 0:
        return 0
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def monotonic_ms() -> Int:
    """Milliseconds on `CLOCK_MONOTONIC`. See `monotonic_ns` on failure."""
    return monotonic_ns() // 1_000_000


def socket_stream() -> Int32:
    """`socket(AF_INET, SOCK_STREAM, 0)` → fd, or -1 on failure."""
    return external_call["socket", Int32](
        Int32(AF_INET), Int32(SOCK_STREAM), Int32(0)
    )


def epoll_create() -> Int32:
    """`epoll_create1(EPOLL_CLOEXEC)` → fd, or -1 on failure."""
    return external_call["epoll_create1", Int32](Int32(EPOLL_CLOEXEC))


def timerfd_monotonic() -> Int32:
    """`timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC)` → fd, or -1 on failure."""
    return external_call["timerfd_create", Int32](
        Int32(CLOCK_MONOTONIC), Int32(TFD_CLOEXEC)
    )


def eventfd_counter() -> Int32:
    """`eventfd(0, EFD_CLOEXEC)` → fd, or -1 on failure."""
    return external_call["eventfd", Int32](UInt32(0), Int32(EFD_CLOEXEC))


def enable_reuseport(fd: Int32) -> Int32:
    """`setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, 1)` → 0, or -1 on failure."""
    var one = Int32(1)
    return external_call["setsockopt", Int32](
        fd, Int32(SOL_SOCKET), Int32(SO_REUSEPORT), Pointer(to=one), UInt32(4)
    )


def close_fd(fd: Int32) -> Int32:
    """`close(fd)` → 0, or -1 on failure."""
    return external_call["close", Int32](fd)


def open_read(path: String) -> Int32:
    """`openat(AT_FDCWD, path, O_RDONLY)` → fd, or -1 on failure.

    A `String` has to cross into C as a `CStringSlice`. `as_c_string_slice` is
    mutating, so the path needs to be a variable, and the slice must be reduced
    with `unsafe_ptr` before the call. Naming an untracked origin here does not
    work: the slice's origin is `origin_of(mutable)`, and Mojo will not
    implicitly widen it, so the pointer is handed over exactly as it is.

    `openat` rather than `open`: `open` is already declared by the stdlib's own
    file I/O, and two `external_call`s that disagree about one libc symbol make
    the whole module fail to lower — with an error that points here rather than
    at the conflict. `openat(AT_FDCWD, ...)` is the same open, named apart.
    """
    var mutable = path
    var slice = mutable.as_c_string_slice()
    var raw = slice.unsafe_ptr()
    return external_call["openat", Int32](
        Int32(AT_FDCWD), raw, Int32(O_RDONLY)
    )
