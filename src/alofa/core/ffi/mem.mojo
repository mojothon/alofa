"""Memory-mapping primitives used by the L0 core layer.

Large files are read through a mapping rather than `read()`, so that page
cache behaviour stays under our control and a multi-gigabyte file does not
become a multi-gigabyte copy in the heap.

Detecting `mmap` failure needs an explanation. libc signals it by returning
`(void*)-1`, but Mojo 1.0 offers no way to compare a `Pointer` against that
sentinel — `bitcast` only converts between SIMD types, and `rebind` rejects a
pointer-vs-scalar reinterpretation. `madvise` is therefore used as the probe:
it is the one call that reports whether an address range is mapped *without
touching the memory*, which turns a would-be segfault into a return code.

Run:
    pixi run mojo run -I src tests/unit/test_core_ffi.mojo
"""

from std.ffi import external_call

from alofa.core.error import ERR_MMAP, AlofaError

comptime PROT_READ = 0x1
comptime PROT_WRITE = 0x2

comptime MAP_SHARED = 0x01
comptime MAP_PRIVATE = 0x02
comptime MAP_ANONYMOUS = 0x20

comptime MADV_NORMAL = 0
comptime MADV_RANDOM = 1
comptime MADV_SEQUENTIAL = 2
comptime MADV_WILLNEED = 3
comptime MADV_DONTNEED = 4

comptime RawPtr = Pointer[UInt8, MutUntrackedOrigin]


def is_mapped(p: RawPtr, size: Int) -> Bool:
    """Whether `p` refers to a live mapping of at least `size` bytes.

    See the module docstring for why this is an `madvise` probe rather than a
    comparison against `MAP_FAILED`.
    """
    return external_call["madvise", Int32](p, Int(size), Int32(MADV_NORMAL)) == 0


def mmap_anonymous(size: Int) raises AlofaError -> RawPtr:
    """Map `size` zero-filled anonymous bytes, read-write.

    Raises `ERR_MMAP` if the mapping failed instead of returning a pointer
    that would segfault on first touch.
    """
    var p = external_call["mmap", RawPtr](
        Int(0),
        Int(size),
        Int32(PROT_READ | PROT_WRITE),
        Int32(MAP_PRIVATE | MAP_ANONYMOUS),
        Int32(-1),
        Int(0),
    )
    if not is_mapped(p, size):
        raise AlofaError(
            ERR_MMAP, "mmap(anonymous) failed", "size=" + String(size)
        )
    return p


def mmap_file(size: Int, fd: Int32, offset: Int) raises AlofaError -> RawPtr:
    """Map `size` bytes of `fd` from `offset`, read-only and private.

    Read-only private is the whole-file-scan case: nothing is written back, so
    access hints never have to account for dirty pages.
    """
    var p = external_call["mmap", RawPtr](
        Int(0), Int(size), Int32(PROT_READ), Int32(MAP_PRIVATE), fd, Int(offset)
    )
    if not is_mapped(p, size):
        raise AlofaError(
            ERR_MMAP,
            "mmap(file) failed",
            "size=" + String(size) + " fd=" + String(Int(fd)),
        )
    return p


comptime RLIMIT_STACK = 3
comptime RLIM_INFINITY = UInt64(0xFFFFFFFFFFFFFFFF)


def raise_stack_limit(want_bytes: Int) raises -> Int:
    """把 `RLIMIT_STACK` 的 soft 上限抬到至少 `want_bytes`，返回生效值（-1 = 失败）。

    为什么需要它：`pthread_create`（NULL attr）起的线程，默认栈 =
    `RLIMIT_STACK` 的 soft 值（本机 8 MiB）。本项目的批前向调用链（执行器与
    模型按值传递的大结构体 + 宽带核的深调用链）在 8 MiB 上**栈溢出**：溢出点
    是谁先碰到 guard page 谁死，所以表现为「崩溃点随机落点、插桩即愈、gdb
    掩盖」，2026-11-04 用 64 MiB 栈 A/B 钉死（账本三十五）。必须在**任何线程
    spawn 之前**调用 —— 它只影响之后创建的线程。抬 soft 不需要特权
    （hard = unlimited）。
    """
    var p = mmap_anonymous(4096)
    var rc = external_call["getrlimit", Int32](Int32(RLIMIT_STACK), p)
    if rc != 0:
        munmap(p, 4096)
        return -1
    var words = p.unsafe_bitcast[UInt64]()
    var soft = words[unsafe_offset=0]
    var hard = words[unsafe_offset=1]
    var new_soft = UInt64(want_bytes)
    if soft == RLIM_INFINITY or soft >= new_soft:
        munmap(p, 4096)
        return Int(soft)
    if hard != RLIM_INFINITY and new_soft > hard:
        new_soft = hard
    words[unsafe_offset=0] = new_soft
    rc = external_call["setrlimit", Int32](Int32(RLIMIT_STACK), p)
    munmap(p, 4096)
    if rc != 0:
        return -1
    return Int(new_soft)


def munmap(p: RawPtr, size: Int) -> Int32:
    """Unmap a region. Returns 0, or -1 on failure."""
    return external_call["munmap", Int32](p, Int(size))


def madvise(p: RawPtr, size: Int, advice: Int32) -> Int32:
    """Hint the OS about expected access. Returns 0, or -1 on failure."""
    return external_call["madvise", Int32](p, Int(size), advice)


def mlock(p: RawPtr, size: Int) -> Int32:
    """Lock a region into RAM. Returns 0, or -1 on failure (often: no permission)."""
    return external_call["mlock", Int32](p, Int(size))


def munlock(p: RawPtr, size: Int) -> Int32:
    """Undo `mlock`. Returns 0, or -1 on failure."""
    return external_call["munlock", Int32](p, Int(size))
