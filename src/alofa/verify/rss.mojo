"""常驻内存观测：从 `/proc/self/status` 读 VmRSS 与 VmHWM。

为什么是 `/proc` 而不是别的
----------------------------

内存门要量的是**这个进程自己**占了多少物理内存，包括被 mmap 进来并且已经
触碰过的那些页。内核已经把这个数算好了，读 `/proc/self/status` 就是读内核
的答案 —— 不引入任何依赖，也不需要链接器帮忙。

两个量，别混用
--------------

- `rss_bytes()`（`VmRSS`）：**此刻**的常驻量。会涨也会落。
- `peak_rss_bytes()`（`VmHWM`）：**峰值**常驻量，只涨不落。

内存门必须用峰值。用此刻的量去判"加载权重之后占多少"是错的：内核可以在任
意时刻回收干净的文件页，于是同一个程序量两次，第二次可能比第一次小，门就
变成了"看内核心情"。峰值没有这个问题 —— 它记的是"最紧张的那个瞬间"，而那
正是内存门关心的东西。

`VmHWM` 从 Linux 2.6.28 起就在 `/proc/self/status` 里。读不到就**具名报错**
而不是返回 0：返回 0 会让 `rss <= 1.15 × 权重` 这种断言恒真，而一个恒真的
门比没有门更糟，它会给出一个谁都不会再去看的绿灯。

Run:
    pixi run mojo run -I src tests/unit/test_memory_gate.mojo
"""

from alofa.core.error import ERR_IO, AlofaError
from alofa.core.text import parse_int, read_text

comptime STATUS_PATH = "/proc/self/status"

comptime RSS_KEY = "VmRSS"
comptime PEAK_KEY = "VmHWM"


def first_number(text: String) raises AlofaError -> Int:
    """一行里第一个十进制数。

    `/proc/self/status` 的分隔符是制表符与空格混着来的（`VmHWM:\t  12345 kB`），
    所以先按制表符、再按空格各切一次，取第一个非空片段。没找到就报错，不返回
    0 —— 返回 0 会让"常驻量 ≤ 预算"这类断言恒真。
    """
    var candidate = text
    for part in text.split("\t"):
        var piece = String(part)
        if piece.byte_length() > 0:
            candidate = piece
            break
    for token in candidate.split(" "):
        var word = String(token)
        if word.byte_length() == 0:
            continue
        return parse_int(word)
    raise AlofaError(ERR_IO, "no number in this field", "text=" + text)


def status_value(key: String) raises AlofaError -> Int:
    """`/proc/self/status` 里某一行的数值（kB）。"""
    var text = read_text(STATUS_PATH)
    for span in text.split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        var halves = line.split(":")
        var head = ""
        var tail = ""
        var first = True
        for part in halves:
            var piece = String(part)
            if first:
                head = piece
                first = False
            elif tail.byte_length() == 0:
                tail = piece
        if head != key:
            continue
        return first_number(tail)
    raise AlofaError(
        ERR_IO,
        "the kernel did not report this value",
        "key=" + key + " path=" + STATUS_PATH,
    )


def rss_bytes() raises AlofaError -> Int:
    """此刻的常驻物理内存（字节）。"""
    return status_value(RSS_KEY) * 1024


def peak_rss_bytes() raises AlofaError -> Int:
    """进程至今的**峰值**常驻物理内存（字节）。内存门要用这个。"""
    return status_value(PEAK_KEY) * 1024
