"""Arena allocation for the L0 core layer.

An arena hands out bumps of a single anonymous mapping: allocating is a
pointer bump, and the only way to reclaim is `reset`, which reclaims
everything at once. There is deliberately no per-block free. The workload this
is for allocates the same shapes over and over, which is exactly the pattern a
bump allocator handles well and a general allocator only fragments on. If a
future phase needs individual frees, that is a different allocator and should
say so in the ledger rather than growing a free path here.

The arena is **move-only**: copying it would produce two objects that both
believe they own the mapping, and whichever was destroyed second would unmap
memory the other is still using.

**Lifetime.** Allocations come back as raw pointers, and Mojo destroys a value
at its *last use* — not at the end of its scope. For an arena that last use is
usually the last `alloc`, which is earlier than the last use of the pointers
handed out, so a naive caller sees the mapping disappear from under it. The
caller must therefore say where the arena's last use is, by calling
`keep_alive` after the last use of any pointer taken from it. This is not a
quirk to be designed around later: it is the price of handing out untracked
pointers, and the alternative — a borrow-tracked pointer — would forbid
holding two allocations at once, which a bump allocator cannot live with.

Run:
    pixi run mojo run -I src tests/unit/test_core_memory.mojo
"""

from alofa.core.error import (
    ERR_ALIGNMENT,
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    AlofaError,
)
from alofa.core.ffi.mem import RawPtr, mmap_anonymous, munmap
from alofa.core.log import LEVEL_DEBUG, Logger

comptime DEFAULT_ALIGNMENT = 64


struct Arena(Movable):
    """A bump allocator over one anonymous mapping."""

    var data: RawPtr
    var capacity: Int
    var used: Int
    var log: Logger

    def __init__(out self, capacity: Int) raises AlofaError:
        if capacity <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "arena capacity must be positive",
                "capacity=" + String(capacity),
            )
        self.data = mmap_anonymous(capacity)
        self.capacity = capacity
        self.used = 0
        self.log = Logger("arena", LEVEL_DEBUG)
        self.log.debug(
            "arena.create", "arena created", "capacity=" + String(capacity)
        )

    def __deinit__(deinit self):
        _ = munmap(self.data, self.capacity)

    def keep_alive(self):
        """State that the arena is still in use here.

        See the module docstring: Mojo destroys a value at its last use, and
        for an arena that has to be *after* the last use of its allocations.
        Calling this is how a caller says so. It does nothing at run time; it
        exists to be the arena's last use.
        """
        pass

    def alloc(
        mut self, size: Int, alignment: Int = DEFAULT_ALIGNMENT
    ) raises AlofaError -> RawPtr:
        """Reserve `size` bytes aligned to `alignment`.

        The bytes are not zeroed — the caller owns what it asks for and an
        arena that silently cleared memory would charge every allocation for a
        cost most do not need.

        The returned pointer is untracked, so the arena's lifetime is the
        caller's to state; see `keep_alive`.
        """
        if size <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "allocation size must be positive",
                "size=" + String(size),
            )
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise AlofaError(
                ERR_ALIGNMENT,
                "alignment must be a power of two",
                "alignment=" + String(alignment),
            )

        var aligned = (self.used + alignment - 1) // alignment * alignment
        if aligned + size > self.capacity:
            raise AlofaError(
                ERR_CAPACITY,
                "arena is out of capacity",
                "need="
                + String(aligned + size)
                + " capacity="
                + String(self.capacity),
            )

        self.used = aligned + size
        # `unsafe_offset` counts bytes for a `Pointer[UInt8]`, which is what
        # `aligned` is: an offset from the start of the mapping, not an index.
        return self.data.unsafe_offset(aligned)

    def reset(mut self):
        """Reclaim every allocation at once."""
        self.log.debug("arena.reset", "arena reset", "reclaimed=" + String(self.used))
        self.used = 0

    def bytes_used(self) -> Int:
        return self.used

    def bytes_free(self) -> Int:
        return self.capacity - self.used
