# Upstream issue 草稿 —— flare `Mutex` 双线程并发 lock/unlock 腐坏自身堆存储

> 状态：材料齐备，待提交（目标仓库 `modular/modular`，Bugs 分类）。
> 复现器：`scripts/diag_repro_mailbox_burst.mojo`（构建命令见文件头）。
> 全部数字为 2026-11-06 本机新鲜实测（账本四十一/四十二）。

---

## Title

`flare.runtime.mutex.Mutex` corrupts its own heap storage under concurrent lock/unlock from two threads (EINVAL on unlock, SIGILL/SIGSEGV in allocator)

## Body

### Summary

`flare.runtime.mutex.Mutex` becomes corrupted when two threads concurrently `lock()`/`unlock()` the **same** instance — even with **zero other work**: no `List`, no user allocation on the contended path, nothing but lock/unlock in a tight loop. Within seconds, `pthread_mutex_unlock` starts returning `rc=22` (`EINVAL`) via flare's own assert (`src/flare/runtime/mutex.mojo:114:41`), the process dies with SIGILL, or hangs. Our byte-accounting instrument shows the mutex's backing heap storage (an `unsafe_alloc` block) being **overwritten while the mutex is not held**, and later live heap blocks get allocated on top of it (a `String` buffer landed on the mutex storage) — i.e. the mutex's own storage is being corrupted, which then poisons the general heap.

### Environment

- Mojo 1.0.0 (ed45d567), installed via pixi (bundled libAsyncRTRuntimeGlobals.so / libKGENCompilerRTShared.so)
- Ubuntu 22.04.5 LTS, glibc from 22.04
- Intel Core i7-9700K (8C, AVX2 only), 62 GB RAM

### Minimal reproducer

Two threads, one shared `Mutex`, nothing else. Nondeterministic within an 8–10 s run: observed outcomes over 4 runs were 1× SIGILL (assert `pthread_mutex_unlock failed with rc=22` from flare's `mutex.mojo:114`), 1× hang, 2× survived.

```mojo
from std.ffi import external_call
from std.memory import Pointer
from std.time import sleep
from std.time import time_function

from flare.runtime._thread import ThreadHandle
from flare.runtime.mutex import Mutex

comptime OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]


struct Shared(Movable):
    var m: Mutex

    def __init__(out self) raises:
        self.m = Mutex()


def worker(arg: OpaquePtr) -> OpaquePtr:
    var shared = arg.unsafe_bitcast[Shared]()
    var n = 0
    while n < 200_000_000:
        shared[].m.lock()
        shared[].m.unlock()
        n += 1
    print("worker done: locks=" + String(n))
    var zero = Int(0)
    return OpaquePtr(unsafe_from_address=zero)


def main() raises:
    var shared = Shared()
    var t = ThreadHandle.spawn[worker](
        OpaquePtr(unsafe_from_address=Int(Pointer(to=shared)))
    )
    var n = 0
    while n < 200_000_000:
        shared.m.lock()
        shared.m.unlock()
        n += 1
    t.join()
    print("main done: locks=" + String(n))
```

Observed failure modes on the spawned thread (this is flare's own assert, so the runtime already knows something is wrong):

```
At: src/flare/runtime/mutex.mojo:114:41: Assert Error: pthread_mutex_unlock failed with rc=22
```

with `rc=22` = `EINVAL` — i.e. the `pthread_mutex_t` inside the mutex's storage no longer describes a valid, unlocked normal mutex, even though nobody wrote to it through any user path.

### Deterministic amplifier (crashes within seconds, every run)

The full reproducer (`scripts/diag_repro_mailbox_burst.mojo`) adds the workload shape where we first hit this in production — a two-thread mailbox shuttling **large inline elements** (8 KB jobs / 16 KB results in `List[Job]`/`List[Result]`, guarded by the flare `Mutex`) while the other thread churns small string allocations:

- flare `Mutex` mailbox: **SIGSEGV within seconds** (fresh run: crash after 764,370 jobs), stack inside the allocator, e.g. `String._add → alloc → libAsyncRTRuntimeGlobals.so+0x4f2c2`; across sessions also tcmalloc sized-delete CHECK paths (`TCMallocInternalDeleteArraySizedAligned`).
- Same program, same sequence, same lists — mutex swapped for a plain inline 48-byte `pthread_mutex_t` (zero-init, raw `pthread_mutex_lock/unlock` via `external_call`): **survives**, 754,275 job pairs in 15 s, exit 0. (4/4 across sessions.)
- Same program, **single-threaded** self-loop through the identical push/pop sequence: **survives**, 9,089,456 ops in 10 s — so the large-element `List` push/pop path is not the bug.
- Same program with `Mutex` lock/unlock only, no lists at all: SIGILL / `rc=22` / hang as above.

### Byte-accounting evidence (storage is being overwritten while unlocked)

The reproducer snapshots bytes 16–39 of `Mutex._storage` (the `__kind` field and list pointers of the embedded `pthread_mutex_t`; bytes 0–3 are the futex word, which lock/unlock legitimately touches) immediately after construction, and re-checks them on every loop iteration **from the main thread while the mutex is not held**. In corrupted runs these bytes change, and in one caught instance a subsequent `String` allocation received a buffer overlapping the mutex's storage — the two live blocks overlapped, which is heap-corruption territory, not a locking-logic bug.

### Impact

Any program that uses a flare `Mutex` from more than one thread can corrupt the heap within seconds under contention, with crash sites far away from the mutex (allocator internals, unrelated allocations). We root-caused a long-running server crash (generations on engine threads) to exactly this: swapping the flare `Mutex` in the request mailbox for a raw `pthread_mutex_t` made an isolated repro that crashed 4–6/6 rounds survive 6/6, and the server has been soak-clean since (FAKE workload 0/6 × 60 s, real model 0/3 × 120 s).

### Workaround

Do not use `flare.runtime.mutex.Mutex` across threads; use a raw zero-initialized 48-byte `pthread_mutex_t` through `external_call["pthread_mutex_lock"/"pthread_mutex_unlock"]` (identical API surface, inline storage, no heap block to corrupt).

### Questions

1. Is `Mutex._storage` (the `unsafe_alloc` heap block) subject to being freed/moved/reused by something in the runtime when a second thread is spawned or when allocations happen on other threads?
2. Is there an intended lifetime rule (e.g. mutex must be pinned / constructed on a specific thread / must not outlive something) that two-thread lock/unlock violates? The reproducer keeps the owning struct alive for the whole program, so nothing is dropped mid-run.
