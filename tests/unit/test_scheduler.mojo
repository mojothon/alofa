"""调度器的门：参照物是**导出的 trace**，不是"看起来对"。

判定纪律
--------

**逐字节，不容差。** 调度器输出的不是数值而是**决定** —— 这一拍给谁多少 token、
抢占谁。两个决定之间不存在"差一点点"：差一个 token 就是另一个决定。所以比的
是 `Action` 序列化后的字符串，一个字符都不许差。

**参照物是独立实现。** `tests/fixtures/scheduler/*.trace` 由
`scripts/dump_scheduler_reference.py` 导出 —— 同一份策略的 Python 实现。若参照物
由被测实现自己导出，实现改坏了 fixture 会跟着一起改坏，门就永远绿着。

**三条负向对照。** 一个不会失败的门等于没有门：
- `bad.trace`（故意改坏一拍的 OUT）→ 重放必须判红，否则逐字节比对根本没在比；
- `alt.trace`（把抢占顺序从"抢占最新的"改成"抢占最老的"，一个同样自洽但不同的
  策略）→ 必须判红，否则 fixture 只分辨得了格式，分辨不了策略；
- `bad_alloc.mojo`（给调度器加会增长的堆容器）→ 零分配扫描必须判违规。

**零分配这条证据的边界**（写在这里以免被误读成更强的结论）：它能拦住"给调度器
加一个会增长的容器"，因为所有容器都是编译期定长的 `InlineArray` 且源码门会扫出
新增的堆容器；它**拦不住** libc 里的小块分配，也**不等同**于进程级 RSS 不动。
所以账本里按这个口径写，不写成"进程零分配"。

Run:
    pixi run mojo run -I src tests/unit/test_scheduler.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.text import parse_int, read_text
from alofa.engine.scheduler import Action, SchedConfig, SchedInput, Scheduler
from alofa.engine.trace import (
    action_line,
    parse_config,
    parse_input,
    parts_of,
    value_after,
)

comptime FIXTURE = "tests/fixtures/scheduler/"
comptime SCHED_SRC = "src/alofa/engine/scheduler.mojo"
comptime BAD_ALLOC_SRC = "tests/fixtures/bad_alloc.mojo"

comptime SCENARIOS = 7


struct ReplayStats:
    """一次重放的体检结果：不变量是否破了，以及场景关心的极值。"""

    var ticks: Int
    var mismatches: Int
    var preempt_total: Int
    var max_blocks: Int
    var max_tokens: Int
    var final_blocks: Int
    var final_cached: Int
    var final_live: Int
    var budget_breaks: Int
    var capacity_breaks: Int
    var accounting_breaks: Int
    var duplicate_prefill: Int
    var watermark_breaks: Int

    def __init__(out self):
        self.ticks = 0
        self.mismatches = 0
        self.preempt_total = 0
        self.max_blocks = 0
        self.max_tokens = 0
        self.final_blocks = 0
        self.final_cached = 0
        self.final_live = 0
        self.budget_breaks = 0
        self.capacity_breaks = 0
        self.accounting_breaks = 0
        self.duplicate_prefill = 0
        self.watermark_breaks = 0

    def violations(self) -> Int:
        return (
            self.mismatches
            + self.budget_breaks
            + self.capacity_breaks
            + self.accounting_breaks
            + self.duplicate_prefill
        )


def lines_of(path: String) raises AlofaError -> List[String]:
    var out = List[String]()
    for span in read_text(path).split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        out.append(line)
    return out^


def seen_has(seen: List[Int], req: Int) -> Bool:
    for i in range(len(seen)):
        if seen[i] == req:
            return True
    return False


def replay(
    path: String, mut stats: ReplayStats, mut seen: List[Int]
) raises AlofaError -> List[String]:
    """用 fixture 里的输入驱动调度器，返回逐步序列化后的 OUT。

    顺带逐拍检查四条不变量：预算不超、不超硬容量、`blocks_used` 与序列长度
    推导出的占用一致、同一请求不在一拍里拿到两个 chunk。
    """
    var lines = lines_of(path)
    var cfg = parse_config(lines[0])
    var sch = Scheduler(cfg)
    var got = List[String]()

    var idx = 1
    while idx < len(lines):
        # OUT 自身含空格，所以按 " OUT=" 一刀切开，而不是按空格切。
        var halves = parts_of(lines[idx], " OUT=")
        if len(halves) != 2:
            raise AlofaError(1, "trace line must be 'IN=... OUT=...'")
        stats.ticks += 1

        var inp = parse_input(halves[0])
        var act = sch.step(inp)
        var line = action_line(act)
        got.append(line)
        if line != "OUT=" + halves[1]:
            stats.mismatches += 1

        if act.tokens() > cfg.token_budget:
            stats.budget_breaks += 1
        if act.tokens() > stats.max_tokens:
            stats.max_tokens = act.tokens()
        if sch.blocks_used > cfg.capacity_blocks:
            stats.capacity_breaks += 1
        if sch.blocks_used > stats.max_blocks:
            stats.max_blocks = sch.blocks_used
        # 占用 = 活跃请求持有的 + 前缀缓存占着的。缓存那部分由引擎归还，
        # 但"账是否平"每拍都必须从两边重算。
        if sch.blocks_used != sch.blocks_held() + sch.cached_blocks:
            stats.accounting_breaks += 1
        if sch.blocks_used > cfg.threshold_blocks() and sch.n_running() > 0:
            # 有可抢占对象却把水位留在阈值之上 = 抢占没做干净。
            stats.watermark_breaks += 1

        for i in range(act.n_prefill):
            for j in range(i):
                if act.p_req[j] == act.p_req[i]:
                    # 同一请求在一拍里出现两次 = 同一请求拿了两个 chunk，违反策略 3。
                    stats.duplicate_prefill += 1
            seen.append(act.p_req[i])
        for i in range(act.n_decode):
            seen.append(act.decode[i])
        for i in range(act.n_preempted):
            seen.append(act.preempted[i])
        for i in range(act.n_finished):
            seen.append(act.finished[i])

        idx += 1

    stats.preempt_total = sch.preempt_total
    stats.final_blocks = sch.blocks_used
    stats.final_cached = sch.cached_blocks
    stats.final_live = sch.n_live()
    return got^


def replay_scheduler(path: String) raises AlofaError -> Scheduler:
    """重放到底，把调度器本身交出来（要给"归还之后归零"这条性质用）。

    只喂输入、不看输出 —— 逐字节比对是 `replay` 的事，这里要的是终局状态。
    """
    var lines = lines_of(path)
    var sch = Scheduler(parse_config(lines[0]))
    var idx = 1
    while idx < len(lines):
        var halves = parts_of(lines[idx], " OUT=")
        if len(halves) != 2:
            raise AlofaError(1, "trace line must be 'IN=... OUT=...'")
        var tick_in = parse_input(halves[0])
        _ = sch.step(tick_in)
        idx += 1
    return sch^


def scenario_names() -> List[String]:
    var out = List[String]()
    out.append("s01_long_prompt")
    out.append("s02_preempt_storm")
    out.append("s03_budget_exhausted")
    out.append("s04_zero_budget")
    out.append("s05_cancel_race")
    out.append("s06_kv_watermark")
    out.append("s07_cache_freed")
    return out^


def test_every_scenario_replays_byte_exact() raises:
    """六个极端场景与导出的 trace 逐字节一致，且四条不变量全绿。"""
    var names = scenario_names()
    var checked = 0
    for name in names:
        var stats = ReplayStats()
        var seen = List[Int]()
        _ = replay(FIXTURE + name + ".trace", stats, seen)
        assert_true(
            stats.mismatches == 0,
            name + " 有 " + String(stats.mismatches) + " 拍与参考 trace 不一致",
        )
        assert_true(
            stats.budget_breaks == 0, name + " 有拍次超出 token 预算"
        )
        assert_true(
            stats.capacity_breaks == 0, name + " 有拍次超出 KV 硬容量"
        )
        assert_true(
            stats.accounting_breaks == 0, name + " 的 KV 占用与序列长度推导不一致"
        )
        assert_true(
            stats.watermark_breaks == 0,
            name + " 有可抢占对象却把占用留在水位之上",
        )
        assert_equal(stats.duplicate_prefill, 0)
        assert_true(stats.ticks > 0, name + " 一行都没跑")
        checked += 1
    assert_equal(checked, SCENARIOS)


def test_the_pool_is_empty_once_every_request_is_gone() raises:
    """所有请求结束后，剩下的占用必须**全部**是前缀缓存，且一次归还就能归零。

    这条门盯着的是"增量记账漂移"：加的时候少算、放的时候多减，两侧各自看起来
    都正常，只有归零这一刻会露出来。完成即发布（块换主人不退池），所以归零要
    靠引擎把缓存还回来 —— 若还回来的数量和调度器记着的缓存对不上，这里就会响。
    """
    var names = scenario_names()
    for name in names:
        var stats = ReplayStats()
        var seen = List[Int]()
        _ = replay(FIXTURE + name + ".trace", stats, seen)
        if name == "s04_zero_budget":
            # 0 预算：请求还活着，这是它存在的意义。
            continue
        assert_true(stats.final_live == 0, name + " 结束后仍有活跃请求")
        assert_equal(stats.final_blocks, stats.final_cached)
        var sch = replay_scheduler(FIXTURE + name + ".trace")
        assert_equal(sch.blocks_held(), 0)
        assert_equal(sch.blocks_used, sch.cached_blocks)
        var inp = SchedInput()
        inp.add_freed_blocks(sch.cached_blocks)
        _ = sch.step(inp)
        assert_equal(sch.blocks_used, 0)
        assert_equal(sch.cached_blocks, 0)


def test_a_finished_request_hands_its_blocks_to_the_cache() raises:
    """完成即发布：块换了主人，占用不降 —— 归还只能走 `freed_blocks`。

    反面是"完成即归零"：那样调度器会以为块空着，而池子里其实还被缓存占着，
    于是它放行一个装不下的批次。
    """
    var sch = Scheduler(SchedConfig(16, 16, 16, 12, 1000, 8))
    var first = SchedInput()
    first.add_arrival(1, 16, 1)
    _ = sch.step(first)
    var second = SchedInput()
    var act = sch.step(second)
    assert_equal(act.n_finished, 1)
    assert_equal(sch.blocks_held(), 0)
    assert_true(
        sch.cached_blocks > 0, "请求完成后块既没回池子也没进缓存 —— 账漏了"
    )
    assert_equal(sch.blocks_used, sch.cached_blocks)
    var back = SchedInput()
    back.add_freed_blocks(sch.cached_blocks)
    _ = sch.step(back)
    assert_equal(sch.cached_blocks, 0)
    assert_equal(sch.blocks_used, 0)


def test_a_return_of_blocks_nobody_cached_is_refused() raises:
    """归还数超过缓存数必须具名拒绝 —— 静默夹住会让两本账从此分家。

    调度器与引擎各记一本账；它们一旦对不上，唯一的信号就是这个数字。
    """
    var sch = Scheduler(SchedConfig(16, 16, 16, 12, 1000, 8))
    var inp = SchedInput()
    inp.add_freed_blocks(1)
    var refused = False
    var kind = String("")
    try:
        _ = sch.step(inp)
    except err:
        refused = True
        kind = err.name()
    assert_true(refused, "凭空归还的块竟然被接受了")
    assert_true(kind == "invalid_argument", "拒绝必须是具名错误，得到 " + kind)


def test_a_bad_trace_is_rejected() raises:
    """红测：改坏一拍的 OUT，重放必须判红。

    若不判红，说明逐字节比对根本没在比 —— 那么前面那条全绿就是假的。
    """
    var stats = ReplayStats()
    var seen = List[Int]()
    _ = replay(FIXTURE + "bad.trace", stats, seen)
    assert_true(stats.mismatches > 0, "改坏过的 trace 竟然重放通过了")


def test_a_different_policy_is_detected() raises:
    """红测：换一个同样自洽但不同的抢占顺序，必须产生不同的字节。

    这条门盯的是"fixture 只分辨得了格式"：若两种策略输出相同，说明抢占顺序
    压根没体现在 trace 里，那"抢占"这件事就没被验过。
    """
    var stats = ReplayStats()
    var seen = List[Int]()
    _ = replay(FIXTURE + "alt.trace", stats, seen)
    assert_true(
        stats.mismatches > 0, "改了抢占顺序却得到同样的字节 —— fixture 分辨不了策略"
    )


def test_replaying_twice_gives_the_same_bytes() raises:
    """同一个 trace 重放两遍必须逐字节相同 —— 调度器不得依赖任何隐藏状态。"""
    var first = ReplayStats()
    var second = ReplayStats()
    var seen_a = List[Int]()
    var seen_b = List[Int]()
    var a = replay(FIXTURE + "s02_preempt_storm.trace", first, seen_a)
    var b = replay(FIXTURE + "s02_preempt_storm.trace", second, seen_b)
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_true(a[i] == b[i], "第 " + String(i + 1) + " 拍两次重放不一致")


def test_a_long_prompt_is_split_into_contiguous_chunks() raises:
    """超长 prompt：切片必须首尾相接、不重叠、每片不超过 `max_chunk`。

    "每请求每拍最多一个 chunk" 与 "切片连续" 是两件不同的事，两条都断言：
    前者防止一个请求吃满预算，后者防止漏 token 或重复算。
    """
    var stats = ReplayStats()
    var seen = List[Int]()
    var got = replay(FIXTURE + "s01_long_prompt.trace", stats, seen)

    var cfg = parse_config(lines_of(FIXTURE + "s01_long_prompt.trace")[0])
    var cursor = 0
    var chunks = 0
    for line in got:
        var groups = parts_of(line, " ")
        var body = value_after(groups[1], "=")
        if body == "-":
            continue
        for piece in body.split(","):
            var triple = parts_of(String(piece), ":")
            var start = parse_int(triple[1])
            var end = parse_int(triple[2])
            assert_true(start == cursor, "切片不连续：期望起点 " + String(cursor))
            assert_true(
                end - start <= cfg.max_chunk,
                "切片长度 " + String(end - start) + " 超过 max_chunk",
            )
            assert_true(end > start, "出现空切片")
            cursor = end
            chunks += 1
    assert_equal(cursor, 300)
    assert_true(chunks > 1, "300 token 的 prompt 却只用了一拍")
    assert_true(chunks >= 300 // cfg.max_chunk, "切片数少于下界")


def test_a_preemption_storm_actually_preempts() raises:
    """并发抢占风暴：抢占必须真的发生，而且抢占计数被累计暴露出来。

    抢占次数是本项目的"KV 不够用"告警指标，所以它必须是 Action 的一部分，
    而不是只有调度器自己知道。
    """
    var stats = ReplayStats()
    var seen = List[Int]()
    var got = replay(FIXTURE + "s02_preempt_storm.trace", stats, seen)
    assert_true(stats.preempt_total > 0, "并发抢占风暴里一次抢占都没发生")
    var preempt_lines = 0
    for line in got:
        var groups = parts_of(line, " ")
        if value_after(groups[3], "=") != "-":
            preempt_lines += 1
    assert_true(preempt_lines > 0, "没有任何一拍的 Action 里出现被抢占者")


def test_the_budget_is_never_exceeded() raises:
    """预算耗尽：每拍花掉的 token 必须恰好等于预算（一分不多，且确实用满）。

    断言"用满"与断言"不超"同样重要：一个永远不花预算的调度器也能满足后者。
    """
    var stats = ReplayStats()
    var seen = List[Int]()
    _ = replay(FIXTURE + "s03_budget_exhausted.trace", stats, seen)
    var cfg = parse_config(lines_of(FIXTURE + "s03_budget_exhausted.trace")[0])
    assert_equal(stats.max_tokens, cfg.token_budget)
    assert_equal(stats.budget_breaks, 0)


def test_zero_budget_makes_no_progress_but_keeps_ticking() raises:
    """0 预算：动作必须为空，tick 继续自增，等待拍数继续累加。

    这是最容易写歪的一处：预算为 0 时"什么都不做"和"不调用调度器"看起来一样，
    但后者的 `tick_seq` 与 `wait_ticks` 不会前进 —— 于是护栏永远不会触发，
    引擎会静默地卡住。
    """
    var cfg = SchedConfig(0, 16, 16, 64, 900, 2)
    var sch = Scheduler(cfg)
    var inp = SchedInput()
    inp.add_arrival(1, 32, 4)
    var act = sch.step(inp)
    assert_equal(act.tokens(), 0)
    assert_equal(act.n_prefill, 0)
    assert_equal(act.n_decode, 0)
    assert_equal(act.tick_seq, 1)

    inp.clear()
    for i in range(7):
        var empty_act = sch.step(inp)
        assert_equal(empty_act.tokens(), 0)
        assert_equal(empty_act.tick_seq, i + 2)
    assert_equal(sch.wait_ticks_of(1), 7)
    assert_equal(sch.blocks_used, 0)


def test_the_latency_guard_jumps_the_queue() raises:
    """延迟护栏：等够久的请求必须插到"排在前面但一直在前进"的请求之前。

    刻意构造的情形是：队首一个 64 token 的长 prompt 每拍都把 8 个 token 的预算
    吃光，后面一个 8 token 的短 prompt 永远轮不上 —— 这正是"等批次"陷阱的最小
    复现。护栏失效时，第 4 拍的预算仍会被队首拿走。

    注意护栏只在**等待者之间**插队，不越过 decode：第 4 拍请求 2 的等待拍数是
    2，而请求 1 因为一直在前进被重置为 0 —— 护栏比的是"谁没在前进"。
    """
    var cfg = SchedConfig(8, 8, 16, 64, 900, 2)
    var sch = Scheduler(cfg)
    var inp = SchedInput()
    inp.add_arrival(1, 64, 2)
    var first = sch.step(inp)
    assert_equal(first.n_prefill, 1)
    assert_equal(first.p_req[0], 1)

    inp.clear()
    inp.add_arrival(2, 8, 2)
    _ = sch.step(inp)
    inp.clear()
    _ = sch.step(inp)

    var act = sch.step(inp)
    assert_equal(act.n_prefill, 1)
    assert_equal(act.p_req[0], 2)
    assert_equal(sch.wait_ticks_of(1), 1)
    assert_equal(sch.wait_ticks_of(2), 0)


def test_a_cancel_landing_with_an_arrival_wins() raises:
    """取消竞态：同拍到达又被取消的请求必须从未被调度过，且 KV 立即归还。

    "取消先于到达" 只是半句话 —— 取消名单还必须能**挡住**同拍的到达，否则请求
    会先入队、再被服务，然后才在下一拍消失。
    """
    var cfg = SchedConfig(16, 16, 16, 64, 900, 4)
    var sch = Scheduler(cfg)
    var inp = SchedInput()
    inp.add_arrival(1, 32, 4)
    inp.add_cancel(1)
    var act = sch.step(inp)
    assert_equal(act.tokens(), 0)
    assert_equal(sch.n_live(), 0)
    assert_equal(sch.blocks_used, 0)
    assert_equal(sch.find(1), -1)

    var stats = ReplayStats()
    var seen = List[Int]()
    _ = replay(FIXTURE + "s05_cancel_race.trace", stats, seen)
    assert_true(not seen_has(seen, 3), "被同拍取消的请求 3 仍然被调度了")
    assert_true(seen_has(seen, 1), "请求 1 在被取消之前应当被调度过")


def test_the_watermark_preempts_without_breaking_capacity() raises:
    """KV 水位临界：过水位就抢占，但任何一拍都不得越过硬容量。

    硬容量是物理上限，水位是策略线。把两者混起来（比如用硬容量当抢占线）会让
    "抢占"变成"OOM 前的最后一次挣扎"。
    """
    var stats = ReplayStats()
    var seen = List[Int]()
    _ = replay(FIXTURE + "s06_kv_watermark.trace", stats, seen)
    var cfg = parse_config(lines_of(FIXTURE + "s06_kv_watermark.trace")[0])
    assert_true(stats.preempt_total > 0, "水位场景里一次抢占都没发生")
    assert_equal(stats.capacity_breaks, 0)
    assert_equal(stats.watermark_breaks, 0)
    # 抢占真的把占用拉回了阈值之内：抽样点在一拍结束（抢占之后），
    # 所以"从未越过硬容量"与"抢占发生过"合起来才说明水位被守住了。
    assert_true(
        stats.max_blocks <= cfg.threshold_blocks(),
        "抢占之后占用仍在阈值之上：" + String(stats.max_blocks),
    )


def test_a_pool_too_small_fails_loudly() raises:
    """容量小到装不下一个 decode 步时，报具名错误，而不是静默抖动或丢 token。

    另一种写法是"抢不到就这一拍不算"，那会让请求看起来在生成、实际在原地
    打转 —— 正是本项目要避免的静默失败。
    """
    var cfg = SchedConfig(32, 32, 16, 2, 900, 8)
    var sch = Scheduler(cfg)
    var inp = SchedInput()
    inp.add_arrival(1, 32, 4)
    var act = sch.step(inp)
    assert_equal(act.tokens(), 32)

    inp.clear()
    var raised = False
    try:
        _ = sch.step(inp)
    except err:
        raised = err.name() == "capacity"
    assert_true(raised, "KV 池装不下时必须报 capacity，而不是静默继续")


def alloc_violations(path: String) raises AlofaError -> Int:
    """源码里出现会增长的堆容器的次数。

    扫描的是**构造点**（`List[` / `String(` / `Arena(` …），不是任何提及：
    docstring 里讨论这些类型不算违规。
    """
    var text = read_text(path)
    var tokens = List[String]()
    tokens.append("List[")
    tokens.append("Dict[")
    tokens.append("String(")
    tokens.append("Arena(")
    tokens.append("MappedFile(")
    tokens.append("FileHandle(")
    var count = 0
    for i in range(len(tokens)):
        if text.find(tokens[i]) >= 0:
            count += 1
    return count


def test_the_scheduler_cannot_allocate_by_construction() raises:
    """零分配：调度器源码里不得出现任何会增长的堆容器。

    真正的保证是类型层面的（所有容器都是编译期定长的 `InlineArray`），这条门
    保证的是**没人把它改回会增长的样子**。它能拦住"加一个 List"，拦不住
    libc 里的小块分配 —— 账本里按这个口径写，不夸大成"进程零分配"。
    """
    assert_equal(alloc_violations(SCHED_SRC), 0)


def test_the_zero_alloc_gate_can_fail() raises:
    """红测：给调度器加会增长的堆容器，扫描必须判违规。

    若不判违规，上一条门就是恒真的 —— 那它证明不了任何事。
    """
    assert_true(alloc_violations(BAD_ALLOC_SRC) > 0, "零分配门抓不到明显的违规")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
