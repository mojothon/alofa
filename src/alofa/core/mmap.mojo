"""Read-only file mappings.

Large files are mapped, not read, so that a multi-gigabyte file costs address
space and page cache rather than a second copy in the heap. The mapping is
read-only and private: nothing is written back, and the OS stays free to
evict pages under memory pressure instead of having to flush them.

`MappedFile` is move-only, so a mapping is unmapped exactly once. A copyable
mapping would leave two objects pointing at the same region, and the second
destruction would pull the pages out from under the first.

Run:
    pixi run mojo run -I src tests/unit/test_core_mmap.mojo
"""

from std.os import stat

from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_IO,
    ERR_OUT_OF_RANGE,
    AlofaError,
)
from alofa.core.ffi import close_fd, open_read
from alofa.core.ffi.mem import (
    MADV_SEQUENTIAL,
    MADV_WILLNEED,
    RawPtr,
    madvise,
    mmap_file,
    munmap,
)
from alofa.core.log import LEVEL_INFO, Logger


struct MappedFile(Movable):
    """A whole file mapped read-only, unmapped when it goes out of scope."""

    var data: RawPtr
    var size: Int
    var log: Logger

    def __init__(out self, path: String) raises AlofaError:
        var fd = open_read(path)
        if fd < 0:
            raise AlofaError(ERR_IO, "cannot open file", "path=" + path)

        var size: Int
        try:
            size = Int(stat(path).st_size)
        except:
            _ = close_fd(fd)
            raise AlofaError(ERR_IO, "cannot stat file", "path=" + path)

        # Mapping a zero-length region is EINVAL, and an empty file is a real
        # failure worth naming rather than a mapping of nothing.
        if size <= 0:
            _ = close_fd(fd)
            raise AlofaError(ERR_INVALID_ARGUMENT, "file is empty", "path=" + path)

        var data: RawPtr
        try:
            data = mmap_file(size, fd, 0)
        except err:
            _ = close_fd(fd)
            raise err.copy()

        # The mapping holds its own reference to the file; keeping the fd would
        # leak one descriptor per mapped file.
        _ = close_fd(fd)

        self.data = data
        self.size = size
        self.log = Logger("mmap", LEVEL_INFO)
        self.log.info(
            "mmap.open", "file mapped", "path=" + path + " size=" + String(size)
        )

    def __deinit__(deinit self):
        _ = munmap(self.data, self.size)

    def byte_at(self, index: Int) raises AlofaError -> UInt8:
        """The byte at `index`. Raises rather than reading past the file."""
        if index < 0 or index >= self.size:
            raise AlofaError(
                ERR_OUT_OF_RANGE,
                "byte index out of range",
                "index=" + String(index) + " size=" + String(self.size),
            )
        return self.data[unsafe_offset=index]

    def ptr(self) -> RawPtr:
        """Start of the mapping, for handing to a tensor view.

        The pointer is untracked, so Mojo can unmap the file at the mapping's
        last use — which may be before the last use of this pointer. As with
        `Arena.keep_alive`, callers that hand the pointer on must call
        `keep_alive` afterwards to say where the last use is.
        """
        return self.data

    def keep_alive(self):
        """State that the mapping is still in use here.

        See `ptr`: this does nothing at run time, it exists to be the last use
        of the mapping so that Mojo does not unmap the file underneath a
        pointer that outlived it.
        """
        pass

    def advise_sequential(self) -> Int32:
        """Hint that the file will be read front-to-back in one pass."""
        return madvise(self.data, self.size, MADV_SEQUENTIAL)

    def advise_willneed(self) -> Int32:
        """Hint that the whole file is about to be read."""
        return madvise(self.data, self.size, MADV_WILLNEED)
