"""The seam between the KV address space (2.1) and the paged kernel (2.2).

`KvSpace` owns the accounting — who holds which block, and how many tokens of it
— and the kernel owns the arithmetic. Neither should know about the other, so
this module is the one place that knows both, and it is deliberately tiny: it
copies a request's page table into the shape the kernel reads.

Doing more here would be easy and wrong. A `fill_page_table` that also chose
blocks, filled them with KV, or decided what to evict would make the kernel
untestable without a pool and a model, which is exactly the situation 2.1 was
built to get out of.

The copy is a copy on purpose: the kernel takes a table it can read without
reaching back into the space, so the paged kernel's tests can hand it a table
made of numbers from a file.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_paged_attention.mojo
"""

from alofa.core.error import ERR_CAPACITY, AlofaError
from alofa.kernels.cpu.paged import PagedTable
from alofa.runtime.kv import MAX_BLOCKS_PER_SEQ
from alofa.runtime.kv.space import KvSpace

comptime KvPageTable = PagedTable[MAX_BLOCKS_PER_SEQ]


def fill_page_table(space: KvSpace, slot: Int, mut table: KvPageTable) raises AlofaError -> Int:
    """Copy request `slot`'s runs into `table`; return its token count.

    The three numbers per run are the whole of what "unified addressing" buys a
    paged kernel: which block, from which slot inside it, for how many tokens.
    A table that knew only (block, length) would read a shared prefix from the
    start of its block — somebody else's tokens.
    """
    table.clear()
    for i in range(space.rq_nblk[slot]):
        var ok = table.push(
            space.block_at(slot, i),
            space.bstart_at(slot, i),
            space.blen_at(slot, i),
        )
        if not ok:
            # Unreachable while both sides are sized by MAX_BLOCKS_PER_SEQ; a
            # silent truncation here would drop the tail of a context.
            raise AlofaError(
                ERR_CAPACITY, "page table is smaller than the request's block table"
            )
    return space.rq_ntok[slot]
