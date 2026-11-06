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
# 共享内存的前 64 字节是头（借用标记 + 对齐余量）。
comptime REGION_HEADER = 64


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


struct SharedRegion(Movable):
    """一块**只读共享内存**，按进程寿命持有（永不 munmap）—— 除非没人借过。

    为什么不带所有权/引用计数（试过，退回来了）：Mojo 按**最后一次使用**析构
    字段，而调用方拿到的是未跟踪的裸指针 —— 它不参与任何所有权。于是"谁最后死"
    在这里无法可靠表达：借用计数会在临时量的析构里被吃掉，字段又可能在构造完就
    被回收，实测结果是借用方在第一次使用之前就读到了已解映射的地址（SIGSEGV）。

    所以这里用一个**只进不出的标记**：映射头部留 64 字节，只要发生过一次
    `borrowed`，标记置位，此后**所有**句柄的析构都只做减法意义上的空操作 ——
    谁先死都不会把对手的字节抽走。**没人借过的单实例照常 munmap**（这是常态：
    一个进程一份模型，重复加载的测试不该累积）。

    分配语义与 `Arena` 一致（bump、不回收），`alloc` 返回未跟踪指针，调用方要
    用 `keep_alive` 声明最后使用点。
    """

    var base: RawPtr
    var data: RawPtr
    var capacity: Int
    var used: Int
    var log: Logger

    def __init__(out self, capacity: Int) raises AlofaError:
        if capacity <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "region capacity must be positive",
                "capacity=" + String(capacity),
            )
        self.base = mmap_anonymous(capacity + REGION_HEADER)
        self.data = self.base.unsafe_offset(REGION_HEADER)
        self.capacity = capacity
        self.used = 0
        self.log = Logger("region", LEVEL_DEBUG)
        self.pin_ptr()[] = 0
        self.log.debug(
            "region.create", "region created", "capacity=" + String(capacity)
        )

    def __init__(out self, base: RawPtr, capacity: Int, used: Int):
        """搭在别处已经分配好的字节上（借用方走这条；不接管所有权）。"""
        self.base = base
        self.data = base.unsafe_offset(REGION_HEADER)
        self.capacity = capacity
        self.used = used
        self.log = Logger("region", LEVEL_DEBUG)

    def pin_ptr(self) -> Pointer[Int32, MutUntrackedOrigin]:
        return self.base.unsafe_bitcast[Int32]()

    def borrowed(self) -> SharedRegion:
        """同一批字节的另一个句柄，并**永久**标记这块内存不回收。"""
        self.pin_ptr()[] = 1
        return SharedRegion(self.base, self.capacity, self.used)

    def __deinit__(deinit self):
        # 被借过就不还：见结构体文档（借用方手里的裸指针不参与所有权）。
        if self.pin_ptr()[] == 0:
            _ = munmap(self.base, self.capacity + REGION_HEADER)

    def keep_alive(self):
        """声明这块内存在这里还在用（与 `Arena.keep_alive` 同一个道理）。"""
        pass

    def alloc(
        mut self, size: Int, alignment: Int = DEFAULT_ALIGNMENT
    ) raises AlofaError -> RawPtr:
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
                "region is out of capacity",
                "need="
                + String(aligned + size)
                + " capacity="
                + String(self.capacity),
            )
        self.used = aligned + size
        return self.data.unsafe_offset(aligned)

