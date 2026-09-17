#!/usr/bin/env python3
"""导出 `paged_scatter` 的参照夹具 —— 与 Mojo 不共享任何代码的第二份实现。

`paged_scatter` 是批量前向**唯一**往 KV 缓存里写东西的地方：它决定第 j 个 token
落在哪一块的哪一个槽内。写错位置的后果不是数值差一点，而是"一整段历史是别人的"
—— 而且每一步看起来都对。所以这一层要的是**逐位**，不是容差：期望值全是 fp32 能
精确表示的小数，写错一个位置就对不上。

参照实现在这里**另写一遍**布局展开（runs → 位置列表），而不是复用 Mojo 的
`row_offset`：由被测实现导出的夹具会跟着一起错。

输出的每个用例包含四样东西：初始 cache（全部元素）、页表（若干 run）、src、
以及 scatter 之后的**整个** cache。比对整个 cache 而不是只比对写过的部分，是为了
顺带管住"顺手改了别人的块" —— 共享前缀让两个请求指向同一块，那条纪律在这一层
正好可以被夹具验到（见 `s05_shared_block`）。

Run:
    python3 scripts/dump_paged_scatter_reference.py
"""

import os

OUT = "tests/fixtures/pagedscatter"

# (name, n_blocks, block_size, cols, runs, n_src, first)
#   runs: [(block, start, length)]，长度之和即上下文长度
CASES = [
    # 一整张表从头写满：最朴素的情形，用来定位格式错误。
    ("s01_fill_all", 3, 4, 2, [(0, 0, 4), (1, 0, 3)], 7, 0),
    # 从中间开始写：只有 3 行被改，前后都要保持原样。
    ("s02_from_middle", 3, 4, 2, [(0, 0, 4), (1, 0, 3)], 3, 2),
    # 一次写跨过 run 边界：位置 3 在块 0 的最后一槽，位置 4 在块 1 的第 0 槽。
    ("s03_across_runs", 3, 4, 2, [(0, 0, 4), (1, 0, 3)], 3, 3),
    # run 从块的中间开始（共享前缀的典型形状）：start 不是 0。
    ("s04_mid_block_start", 3, 4, 2, [(0, 2, 2), (0, 0, 2)], 4, 0),
    # 两个 run 落在同一块里：共享前缀的另一种形状，写必须只落在自己的槽上。
    ("s05_shared_block", 3, 4, 2, [(2, 0, 2), (2, 2, 2)], 4, 0),
    # 倒着排的块：块号大在前，验证"按 run 走"而不是"按块号排序"。
    ("s06_out_of_order", 3, 4, 2, [(2, 0, 3), (0, 0, 2)], 5, 0),
]


def fmt(v):
    """fp32 能精确表示的值，写成不会丢精度的形式。"""
    return repr(float(v))


def make_cache(n_blocks, block_size, cols):
    """初始 cache：块号、槽号、列号都编进数值里，任何两个位置都不同。"""
    out = []
    for b in range(n_blocks):
        for s in range(block_size):
            for c in range(cols):
                out.append(b * 1000 + s * 10 + c + 0.5)
    return out


def make_src(n_src, cols):
    out = []
    for j in range(n_src):
        for c in range(cols):
            out.append(5000 + j * 10 + c + 0.25)
    return out


def scatter_ref(cache, block_size, cols, runs, first, src):
    """参照实现：把 runs 展开成位置列表，再按位置写。

    故意不用任何"算术求地址"的写法（`(first + j) // block_size` 之类）：
    被测实现正是靠这种算术出错的，参照物得从另一条路走到同一个答案。
    """
    layout = []
    for block, start, length in runs:
        for i in range(length):
            layout.append((block, start + i))
    out = list(cache)
    n_src = len(src) // cols
    for j in range(n_src):
        block, slot = layout[first + j]
        base = (block * block_size + slot) * cols
        for c in range(cols):
            out[base + c] = src[j * cols + c]
    return out


def write_values(path, values):
    with open(path, "w") as handle:
        for v in values:
            handle.write(fmt(v) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cases.tsv"), "w") as handle:
        handle.write("# name n_blocks block_size cols n_src first n_runs\n")
        for name, n_blocks, block_size, cols, runs, n_src, first in CASES:
            cache = make_cache(n_blocks, block_size, cols)
            src = make_src(n_src, cols)
            want = scatter_ref(cache, block_size, cols, runs, first, src)
            handle.write(
                "{} {} {} {} {} {} {}\n".format(
                    name, n_blocks, block_size, cols, n_src, first, len(runs)
                )
            )
            with open(os.path.join(OUT, name + ".tab.tsv"), "w") as t:
                for block, start, length in runs:
                    t.write("{} {} {}\n".format(block, start, length))
            write_values(os.path.join(OUT, name + ".cache.tsv"), cache)
            write_values(os.path.join(OUT, name + ".src.tsv"), src)
            write_values(os.path.join(OUT, name + ".want.tsv"), want)
    print("wrote {} cases to {}".format(len(CASES), OUT))


if __name__ == "__main__":
    main()
