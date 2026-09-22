"""Platform primitive bindings for the L0 core layer.

`posix` covers clocks and file descriptors, `mem` covers mappings. Both are
thin `external_call` wrappers that carry no policy of their own.
"""

from .mem import (
    MADV_DONTNEED,
    MADV_NORMAL,
    MADV_RANDOM,
    MADV_SEQUENTIAL,
    MADV_WILLNEED,
    MAP_ANONYMOUS,
    MAP_PRIVATE,
    MAP_SHARED,
    PROT_READ,
    PROT_WRITE,
    RawPtr,
    is_mapped,
    madvise,
    mlock,
    mmap_anonymous,
    mmap_file,
    munlock,
    munmap,
)
from .posix import (
    AF_INET,
    CLOCK_MONOTONIC,
    CLOCK_REALTIME,
    EFD_CLOEXEC,
    EPOLL_CLOEXEC,
    SOCK_STREAM,
    SOL_SOCKET,
    SO_REUSEPORT,
    TFD_CLOEXEC,
    Timespec,
    close_fd,
    dup_to,
    enable_reuseport,
    epoll_create,
    eventfd_counter,
    fd_is_open,
    monotonic_ms,
    monotonic_ns,
    open_read,
    socket_stream,
    timerfd_monotonic,
)
