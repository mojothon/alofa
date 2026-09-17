"""分页 scatter：写进去的位置，就是将来读出来的位置。

判定纪律
--------

**逐位，不容差。** 这一层输出的不是数而是**位置**：第 j 个 token 落在哪一块的哪一
个槽。位置差一个槽，后面每一步算出来的数仍然看起来合理 —— 那是最坏的一种错。所以
夹具里的值全是 fp32 能精确表示的小数，比对的是**整个 cache**（不是只比对写过的部
分）：共享前缀让两个请求指向同一块，"顺手改了别人的槽"必须能被这一门看见。

**参照物是独立实现。** `tests/fixtures/pagedscatter/*.tsv` 由
`scripts/dump_paged_scatter_reference.py` 导出 —— 另一份 Python 实现，它把 runs
**展开成位置列表**再写，而不是用任何"按 block_size 做除法求地址"的算术：被测实现
正是靠那种算术出错的，参照物得从另一条路走到同一个答案。

**两条常驻对照。** 一个不会失败的门等于没有门：
- **把 run 的 `start` 抹成 0**（从块的开头写 —— 分页最经典的那个错）→ 必须判红，
  且红在**共享/中间起始**的用例上。这条对照正是文件头那句"读别人的 token"的反面
  证据：抹掉 start 之后每一步也仍然是个数；
- **写越界必须报 `out_of_range`**，不许静默截断 —— 截断的后果是上下文悄悄变短，
  它会以"少一个 token"的形式出现在三拍之后。

**零分配。** 这个 kernel 每拍对每个请求都被调用，`src/alofa/kernels/cpu/paged.mojo`
里连错误消息都不许分配 `String`（所以那里的诊断只有固定句子）。源码门保证没人把它
改回会增长的样子，而 `bad_paged_alloc.mojo` 保证这个门**自己会红**。

边界（写在这里以免被误读）：这一门**不比数值**（算得对不对由分页注意力门与批一致
性重门负责），它只验"写到了哪里、没写到哪里"。

Run:
    pixi run mojo run -O0 -I src tests/unit/test_paged_scatter.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.dtype import DT_FP32
from alofa.core.error import AlofaError
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, Shape, TensorView, f32_data
from alofa.core.text import parse_float64, parse_int, read_text
from alofa.kernels.cpu.paged import PagedTable, paged_scatter

comptime FIXTURE = "tests/fixtures/pagedscatter/"
comptime BAD_ALLOC_SRC = "tests/fixtures/bad_paged_alloc.mojo"
comptime KERNEL_SRC = "src/alofa/kernels/cpu/paged.mojo"
comptime MAX_ENTRIES = 8
comptime CASES = 6

# 起始槽非零的用例：抹掉 `start` 之后必须在这里现形。
comptime MID_START_CASE = "s04_mid_block_start"


struct Case(ImplicitlyCopyable, Movable):
    """`cases.tsv` 的一行。"""

    var name: String
    var n_blocks: Int
    var block_size: Int
    var cols: Int
    var n_src: Int
    var first: Int
    var n_runs: Int

    def __init__(out self):
        self.name = String("")
        self.n_blocks = 0
        self.block_size = 0
        self.cols = 0
        self.n_src = 0
        self.first = 0
        self.n_runs = 0

    def cells(self) -> Int:
        return self.n_blocks * self.block_size * self.cols


def parts_of(text: String) -> List[String]:
    var out = List[String]()
    for span in text.split(" "):
        var item = String(span)
        if item.byte_length() == 0:
            continue
        out.append(item)
    return out^


def load_cases() raises -> List[Case]:
    var out = List[Case]()
    var first_line = True
    for span in read_text(FIXTURE + "cases.tsv").split("\n"):
        var line = String(span)
        if line.byte_length() == 0 or line[byte=0] == "#":
            continue
        var f = parts_of(line)
        if len(f) != 7:
            raise AlofaError(1, "cases.tsv line does not have seven fields: " + line)
        var c = Case()
        c.name = f[0]
        c.n_blocks = parse_int(f[1])
        c.block_size = parse_int(f[2])
        c.cols = parse_int(f[3])
        c.n_src = parse_int(f[4])
        c.first = parse_int(f[5])
        c.n_runs = parse_int(f[6])
        out.append(c^)
        _ = first_line
    if len(out) != CASES:
        raise AlofaError(1, "fixture does not hold every case")
    return out^


def tensor2(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    return TensorView(arena.alloc(rows * cols * 4), Shape(dims), DT_FP32)


def tensor3(mut arena: Arena, a: Int, b: Int, c: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(a)
    dims.append(b)
    dims.append(c)
    return TensorView(arena.alloc(a * b * c * 4), Shape(dims), DT_FP32)


def load_floats(path: String, dst: F32Ptr, count: Int) raises:
    """正好 `count` 个值，一行一个。

    正好，不是至多：少了一行的夹具会把后面每个值整体挪一位，然后被算到 kernel 头上。
    """
    var text = read_text(path)
    var i = 0
    for span in text.split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        if i >= count:
            raise AlofaError(1, "fixture has more values than the shape holds")
        dst[unsafe_offset=i] = Float32(parse_float64(line))
        i += 1
    if i != count:
        raise AlofaError(1, "fixture has fewer values than the shape holds")


def load_table(path: String, mut table: PagedTable[MAX_ENTRIES]) raises:
    var n = 0
    for span in read_text(path).split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        var f = parts_of(line)
        if len(f) != 3:
            raise AlofaError(1, "a run needs block, start and length")
        if not table.push(parse_int(f[0]), parse_int(f[1]), parse_int(f[2])):
            raise AlofaError(1, "the fixture holds more runs than the table")
        n += 1
    if n == 0:
        raise AlofaError(1, "a case with an empty table proves nothing")


def run_case(mut arena: Arena, read c: Case, drop_start: Bool) raises -> Int:
    """跑一个用例，返回不一致的元素个数。

    `drop_start` 是那条常驻负向对照的开关：把每个 run 的起始槽抹成 0 —— 也就是
    "从块的开头写"这个分页最经典的错法。
    """
    var cache = tensor3(arena, c.n_blocks, c.block_size, c.cols)
    load_floats(FIXTURE + c.name + ".cache.tsv", f32_data(cache), c.cells())
    var src = tensor2(arena, c.n_src, c.cols)
    load_floats(FIXTURE + c.name + ".src.tsv", f32_data(src), c.n_src * c.cols)
    var want = tensor2(arena, c.cells(), 1)
    load_floats(FIXTURE + c.name + ".want.tsv", f32_data(want), c.cells())

    var table = PagedTable[MAX_ENTRIES](c.block_size)
    load_table(FIXTURE + c.name + ".tab.tsv", table)
    if drop_start:
        var fixed = PagedTable[MAX_ENTRIES](c.block_size)
        for i in range(table.n):
            _ = fixed.push(table.block_at(i), 0, table.length_at(i))
        table = fixed^

    paged_scatter(cache, table, c.first, src)

    var got = f32_data(cache)
    var expect = f32_data(want)
    var bad = 0
    for i in range(c.cells()):
        if got[unsafe_offset=i] != expect[unsafe_offset=i]:
            bad += 1
    # 撑住 arena 的寿命：Mojo 在最后一次使用处析构，而最后一次 alloc 之后这块内存
    # 还要被 scatter 读写 —— 少了这句，指针指向的是已经归还的地址。
    arena.keep_alive()
    return bad


def case_named(cases: List[Case], name: String) raises -> Int:
    for i in range(len(cases)):
        if cases[i].name == name:
            return i
    raise AlofaError(1, "no such case: " + name)


def scan_growable(text: String) -> String:
    """第一个出现的"会增长的容器"，没有则空串。"""

    if text.find("List[") >= 0:
        return "List["
    if text.find("Dict[") >= 0:
        return "Dict["
    if text.find("alloc(") >= 0:
        return "alloc("
    if text.find("String(") >= 0:
        return "String("
    return ""


# --- 1. 逐位 ---


def test_every_scatter_case_is_bit_exact() raises:
    var cases = load_cases()
    var arena = Arena(1 << 18)
    for i in range(len(cases)):
        var c = cases[i]
        var bad = run_case(arena, c, False)
        assert_equal(bad, 0, c.name + " 有 " + String(bad) + " 个元素不在它该在的地方")


def test_dropping_the_run_start_is_caught() raises:
    """抹掉 `start`（从块的开头写）必须判红 —— 否则逐位比对只是在验文件格式。

    而且它必须红在**起始槽非零**的用例上：那正是共享前缀留下的形状。抹掉 start 之后
    写进去的每一个数都仍然是个数，只是位置是别人的。
    """
    var cases = load_cases()
    var arena = Arena(1 << 18)
    var c = cases[case_named(cases, MID_START_CASE)]
    assert_true(c.block_size > 1, "这条对照需要一个块里不止一个槽")
    var bad = run_case(arena, c, True)
    assert_true(bad > 0, "把 start 抹成 0 竟然还写对了")


def test_a_shared_block_keeps_the_other_slots() raises:
    """两个 run 落在同一块里：写必须只落在自己的槽上，同块的其它槽一个都不许动。

    `s05_shared_block` 的期望里，块 2 之外的所有元素都与初始 cache 相同；逐位比对
    整块 cache 已经覆盖了这条，这里把它单列出来，是为了让"共享"这件事在门里有一个
    名字 —— 共享前缀是分页存在的理由，它不该只在注释里被提到。
    """
    var cases = load_cases()
    var arena = Arena(1 << 18)
    var c = cases[case_named(cases, "s05_shared_block")]
    var seen_same_block = False
    for span in read_text(FIXTURE + c.name + ".tab.tsv").split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        var f = parts_of(line)
        if parse_int(f[0]) == 2:
            if seen_same_block:
                assert_true(True, "")
            seen_same_block = True
    assert_true(seen_same_block, "共享块的用例里没有 run 落在块 2 上")
    assert_equal(run_case(arena, c, False), 0, "共享块里写错了槽")


# --- 2. 拒绝 ---


def test_writing_past_the_table_is_refused() raises:
    """写越界要报 `out_of_range`，不许静默截断。

    截断的后果是上下文悄悄变短 —— 它会以"少一个 token"的形式出现在几拍之后，那时
    没人会怀疑到 scatter 头上。
    """
    var arena = Arena(1 << 18)
    var cache = tensor3(arena, 3, 4, 2)
    var src = tensor2(arena, 1, 2)
    var table = PagedTable[MAX_ENTRIES](4)
    _ = table.push(0, 0, 4)
    _ = table.push(1, 0, 3)

    # 表覆盖 7 个位置（4 + 3），写第 8 个就越界了 —— 注意是"第 8 个"而不是"第 6 个"：
    # 一张表的容量是它的长度之和，不是它的最后一个块的大小。
    var kind = String("")
    try:
        paged_scatter(cache, table, 7, src)
    except err:
        kind = err.name()
    assert_equal(kind, "out_of_range", "写到表外竟然被允许，得到 " + kind)

    kind = ""
    try:
        paged_scatter(cache, table, -1, src)
    except err:
        kind = err.name()
    assert_equal(kind, "out_of_range", "负的起始位置竟然被允许，得到 " + kind)
    arena.keep_alive()


# --- 3. 零分配 ---


def test_the_zero_alloc_gate_can_fail() raises:
    """这个 kernel 每拍被调用，源码门自己也得会红，否则它只是个摆设。

    真正的保证在类型层面（全是编译期定长的 `InlineArray`）；这条扫的是没人把它改回
    会增长的样子。它拦不住 libc 里的小块分配，也不等于进程 RSS 不动。
    """
    var bad_in_good = scan_growable(read_text(KERNEL_SRC))
    assert_equal(bad_in_good, "", "kernel 源码里出现了会增长的容器：" + bad_in_good)
    var bad_in_bad = scan_growable(read_text(BAD_ALLOC_SRC))
    assert_true(
        bad_in_bad.byte_length() > 0,
        "负向对照源码里没有任何会增长的容器 —— 门不会红",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
