"""屏障探针：一次 TaskGroup 的 `dispatch → 分片 start/end → wait` 五段账。

**要回答的问题**（四十六改判后的下一问）：批解码每步 ~90% 是核时间，而 8 条
worker 在核时间里只忙 ~57% —— 那 43% 是**哪一段**？三种可能，本探针分得开：

    a. 派发→开跑（唤醒）：`min(start) - dispatch` 大 ⇒ 派发/唤醒延迟；
    b. 分片之间不均（掉队）：`max 分片时长 - 平均分片时长` 大 ⇒ 行数分配或缓存运气；
    c. 跑完→`wait()` 返回（收尾）：`wait - max(end)` 大 ⇒ 完成通知/join 延迟。

**挂法**：只挂批路的三条派发器（q4 无偏置、q4 带偏置、q8 头，含单流头）。fp32
通路与注意力不挂 —— 先把占步时最大的几项量出来。

**为什么一条记录能复用**：所有 TG 都是"派发完就 `wait()`、返回才派下一个"，即
**全局串行**；同一条记录同一时刻只有一个 TG 在写。游标在缓冲第 0 个字，只由
派发侧（engine 线程）推进，任务侧只读 `rec`。

**空指针**：不挂探针的调用点（脚本/基准）传 `probe_null()`；`probe_dispatch`
见到 0 返回 -1，任务侧见到 `rec < 0` 全跳过。

**成本**：`ALOFA_BARRIER_PROBE=False` 时全部在编译期折掉（零指令）；打开时每
任务 2 次 `monotonic_ns()` + 2 次写 —— 每步约 2700 次，相对 ~100 ms 可忽略。

⚠️ 探针**不进产品判据**：只写一块进程内缓冲，不改任何计算路径，也不参与
`pixi run test`。量完就关（`False`）。
"""

from alofa.core.error import AlofaError
from alofa.core.ffi.mem import RawPtr, mmap_anonymous
from alofa.core.ffi.posix import monotonic_ns

comptime ALOFA_BARRIER_PROBE = False

# 一次 TG 最多看几个分片（本项目 `shards` 上限 8，留一倍余量）
comptime PROBE_MAX_SHARDS = 16

# 一条记录占的字数：dispatch / wait / n / kind / starts[MAX] / ends[MAX]
comptime PROBE_STRIDE = 4 + 2 * PROBE_MAX_SHARDS

# 一块缓冲装多少条记录：一步 = 24 层 × 7 投影 + 1 头 = 169，留一步半
comptime PROBE_CAP = 256

# kind 标签（dump 时按类聚合）
comptime PROBE_KIND_ROWS = 0
comptime PROBE_KIND_BIAS = 1
comptime PROBE_KIND_HEAD = 2


def probe_null() -> RawPtr:
    """不挂探针的调用点传这个：`probe_dispatch` 记 -1，任务侧全跳过。"""
    return RawPtr(unsafe_from_address=Int(0))


def probe_alloc() raises AlofaError -> RawPtr:
    """一块 `(1 + CAP * STRIDE)` 个 Int64 的匿名映射，第 0 字是游标。"""
    return mmap_anonymous((1 + PROBE_CAP * PROBE_STRIDE) * 8)


def probe_reset(probe: RawPtr):
    """把游标清零（热身步的样本不想要时用）。"""
    comptime if not ALOFA_BARRIER_PROBE:
        return
    if Int(probe) == 0:
        return
    var p = probe.bitcast[Int64]()
    p[unsafe_offset=0] = Int64(0)


def probe_dispatch(probe: RawPtr, kind: Int, n_shards: Int) -> Int:
    """派发侧：占一条记录，写 dispatch 时刻/分片数/类别，返回记录号。

    ⚠️ 只在 `wait()` 之前调用一次（同一条记录不能让两个 TG 同时写）。
    """
    comptime if not ALOFA_BARRIER_PROBE:
        _ = kind
        _ = n_shards
        return -1
    if Int(probe) == 0:
        return -1
    var p = probe.bitcast[Int64]()
    var cur = Int(p[unsafe_offset=0])
    if cur >= PROBE_CAP:
        # 缓冲满：丢样本而不是覆盖正在写的记录。
        return -1
    p[unsafe_offset=0] = Int64(cur + 1)
    var base = 1 + cur * PROBE_STRIDE
    p[unsafe_offset=base] = Int64(monotonic_ns())
    p[unsafe_offset=base + 2] = Int64(n_shards)
    p[unsafe_offset=base + 3] = Int64(kind)
    return cur


def probe_mark_start(probe: RawPtr, rec: Int, shard: Int):
    """任务侧：入口戳。"""
    comptime if not ALOFA_BARRIER_PROBE:
        return
    if rec < 0:
        return
    var p = probe.bitcast[Int64]()
    p[unsafe_offset=1 + rec * PROBE_STRIDE + 4 + shard] = Int64(monotonic_ns())


def probe_mark_end(probe: RawPtr, rec: Int, shard: Int):
    """任务侧：出口戳。"""
    comptime if not ALOFA_BARRIER_PROBE:
        return
    if rec < 0:
        return
    var p = probe.bitcast[Int64]()
    p[unsafe_offset=1 + rec * PROBE_STRIDE + 4 + PROBE_MAX_SHARDS + shard] = (
        Int64(monotonic_ns())
    )


def probe_wait_done(probe: RawPtr, rec: Int):
    """派发侧：`wait()` 返回时刻。"""
    comptime if not ALOFA_BARRIER_PROBE:
        return
    if rec < 0:
        return
    var p = probe.bitcast[Int64]()
    p[unsafe_offset=1 + rec * PROBE_STRIDE + 1] = Int64(monotonic_ns())


def _us(imm ns: Int) -> String:
    """纳秒 → 微秒，一位小数。"""
    var whole = ns // 1000
    var frac = (ns % 1000) // 100
    return String(whole) + "." + String(frac)


def probe_trace(probe: RawPtr, first: Int = 0, count: Int = 6):
    """把前几条记录逐个分片打印出来（相对 dispatch 的 µs 偏移）。"""
    comptime if not ALOFA_BARRIER_PROBE:
        _ = first
        _ = count
        return
    if Int(probe) == 0:
        return
    var p = probe.bitcast[Int64]()
    var n_rec = Int(p[unsafe_offset=0])
    var last = first + count
    if last > n_rec:
        last = n_rec
    for r in range(first, last):
        var base = 1 + r * PROBE_STRIDE
        var n = Int(p[unsafe_offset=base + 2])
        var t_disp = Int(p[unsafe_offset=base])
        var t_wait = Int(p[unsafe_offset=base + 1])
        var line = (
            "   [trace] rec "
            + String(r)
            + " kind "
            + String(Int(p[unsafe_offset=base + 3]))
            + " n "
            + String(n)
            + " wall "
            + _us(t_wait - t_disp)
            + " |"
        )
        for k in range(n):
            if k >= PROBE_MAX_SHARDS:
                break
            var s = Int(p[unsafe_offset=base + 4 + k])
            var e = Int(p[unsafe_offset=base + 4 + PROBE_MAX_SHARDS + k])
            if s <= 0 or e <= 0:
                line += " k" + String(k) + "=(缺)"
                continue
            line += (
                " k"
                + String(k)
                + "=["
                + _us(s - t_disp)
                + ","
                + _us(e - t_disp)
                + "]"
            )
        print(line)


def _kind_name(imm kind: Int) -> String:
    if kind == PROBE_KIND_ROWS:
        return "q4 无偏置"
    if kind == PROBE_KIND_BIAS:
        return "q4 带偏置"
    return "q8 头"


def probe_dump(probe: RawPtr, up_to: Int = PROBE_CAP):
    """按类聚合：唤醒（dispatch→min start）、掉队（max − 平均分片时长）、
    收尾（wait−max end），外加每类 TG 的并行利用率。"""
    comptime if not ALOFA_BARRIER_PROBE:
        _ = up_to
        print("   [probe] 未启用（ALOFA_BARRIER_PROBE=False）")
        return
    if Int(probe) == 0:
        print("   [probe] 这块模型没挂探针（probe 为空）")
        return
    var p = probe.bitcast[Int64]()
    var n_rec = Int(p[unsafe_offset=0])
    if n_rec > up_to:
        n_rec = up_to
    if n_rec <= 0:
        print("   [probe] 没有样本")
        return

    var kinds = InlineArray[Int, 3](fill=0)
    var wake_sum = InlineArray[Int, 3](fill=0)
    var skew_sum = InlineArray[Int, 3](fill=0)
    var tail_sum = InlineArray[Int, 3](fill=0)
    var wall_sum = InlineArray[Int, 3](fill=0)
    var busy_sum = InlineArray[Int, 3](fill=0)
    var shards_sum = InlineArray[Int, 3](fill=0)

    for r in range(n_rec):
        var base = 1 + r * PROBE_STRIDE
        var kind = Int(p[unsafe_offset=base + 3])
        if kind < 0 or kind > 2:
            continue
        var n = Int(p[unsafe_offset=base + 2])
        var t_disp = Int(p[unsafe_offset=base])
        var t_wait = Int(p[unsafe_offset=base + 1])
        if t_wait <= 0:
            continue
        var min_start = 0
        var max_end = 0
        var max_dur = 0
        var sum_dur = 0
        var seen = 0
        for k in range(n):
            if k >= PROBE_MAX_SHARDS:
                break
            var s = Int(p[unsafe_offset=base + 4 + k])
            var e = Int(p[unsafe_offset=base + 4 + PROBE_MAX_SHARDS + k])
            if s <= 0 or e <= 0:
                continue
            if seen == 0 or s < min_start:
                min_start = s
            if e > max_end:
                max_end = e
            if e - s > max_dur:
                max_dur = e - s
            sum_dur += e - s
            seen += 1
        if seen == 0:
            continue
        kinds[kind] += 1
        wall_sum[kind] += t_wait - t_disp
        wake_sum[kind] += min_start - t_disp
        tail_sum[kind] += t_wait - max_end
        skew_sum[kind] += max_dur - sum_dur // seen
        busy_sum[kind] += sum_dur
        shards_sum[kind] += seen

    print("   [probe] 屏障账（每类 TG 的平均值，µs）· 样本 " + String(n_rec) + " 条")
    print(
        "     类别           TG数 分片  墙钟   唤醒  掉队  收尾  分片合计 并行利用率"
    )
    for kind in range(3):
        var c = kinds[kind]
        if c == 0:
            continue
        var wall = wall_sum[kind] // c
        var sh = shards_sum[kind] // c
        var busy = busy_sum[kind] // c
        var util = 0
        if wall > 0 and sh > 0:
            util = busy * 100 // (wall * sh)
        print(
            "     "
            + _kind_name(kind)
            + "  "
            + String(c)
            + "  "
            + String(sh)
            + "  "
            + _us(wall)
            + "  "
            + _us(wake_sum[kind] // c)
            + "  "
            + _us(skew_sum[kind] // c)
            + "  "
            + _us(tail_sum[kind] // c)
            + "  "
            + _us(busy)
            + "  "
            + String(util)
            + "%"
        )
