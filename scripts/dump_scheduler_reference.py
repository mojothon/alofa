"""导出调度器的参考 trace —— 一份**独立实现**的同一份策略。

为什么算法要在这个脚本里再写一遍
---------------------------------

`src/alofa/engine/scheduler.mojo` 与这个文件是同一份策略的两个实现。这不是重复
劳动，而是差分门的前提：若参照物由被测实现自己导出，那么实现改坏了，fixture 会
跟着一起改坏，门就永远绿着。q4_0 那条门也是这么做的 —— 量化算法显式写在导出脚本
里，Mojo 与它逐字节相同。

所以：**改策略要同时改两处**。只改一处会让门变红，那是门在正常工作。

策略（与 `scheduler.mojo` 的 docstring 逐条对应）
------------------------------------------------

1. 单一 token 预算：prefill 与 decode 花同一种额度。
2. 一拍之内的顺序：取消（同时挡住同拍到达）→ 引擎报告完成 → 等待拍数 +1 →
   到达 → 提升 → decode → 饥饿 prefill → 常规 prefill → 水位抢占。
3. 每请求每拍最多一个 chunk（`max_chunk`）。
4. 已在 decode 的请求先拿 token。
5. 等待拍数达到 `max_wait_ticks` 的请求插到 decode 之前。
6. 抢占只针对 RUNNING，取**最新**到达者（slot 最大），被抢占者 KV 全部作废
   （`done` 与 `generated` 都归零），重算语义；抢占计数累计暴露。
7. 水位是千分数整数（`watermark_permille`），不写浮点 —— 浮点阈值会把逐字节门
   变成容差门。

产物
----

- `tests/fixtures/scheduler/<场景>.trace`：首行 `CFG ...`，其后每拍一行
  `IN=... OUT=...`。
- `bad.trace`：把 `s01` 某一拍的 OUT 改一个字段（故意改坏）。重放门**必须判红**，
  否则这条门恒真。
- `alt.trace`：把抢占顺序改成"抢占最老的"（一个同样自洽但不同的策略）跑同一场景。
  Mojo 的输出**必须与之不同** —— 证明 fixture 真的能分辨策略，而不是只分辨格式。

Run:
    python scripts/dump_scheduler_reference.py
"""

from __future__ import annotations

import os

MAX_BATCH = 32
ST_FREE, ST_WAITING, ST_RUNNING = 0, 1, 2
DIGEST_MOD = 2147483647
DIGEST_MUL = 1000003

OUT_DIR = os.path.join("tests", "fixtures", "scheduler")


def blocks_for(seq: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return 0 if seq <= 0 else (seq + block_size - 1) // block_size


class Config:
    def __init__(self, budget, chunk, block_size, capacity, watermark, max_wait):
        self.budget = budget
        self.chunk = chunk
        self.block_size = block_size
        self.capacity = capacity
        self.watermark = watermark
        self.max_wait = max_wait

    def threshold(self) -> int:
        return self.capacity * self.watermark // 1000

    def line(self) -> str:
        return (
            f"CFG b={self.budget} c={self.chunk} s={self.block_size} "
            f"k={self.capacity} w={self.watermark} t={self.max_wait}"
        )


class Scheduler:
    """与 Mojo 侧逐字段对应的参考实现。

    `variant` 只在生成 `alt.trace` 时用：它把抢占顺序从"最新"改成"最老"，
    用来证明逐字节门对策略敏感。
    """

    def __init__(self, cfg: Config, variant: str = "newest"):
        self.cfg = cfg
        self.variant = variant
        self.ids = [0] * MAX_BATCH
        self.prompt_len = [0] * MAX_BATCH
        self.done = [0] * MAX_BATCH
        self.generated = [0] * MAX_BATCH
        self.max_new = [0] * MAX_BATCH
        self.state = [ST_FREE] * MAX_BATCH
        self.wait_ticks = [0] * MAX_BATCH
        self.preempt_count = [0] * MAX_BATCH
        self.progressed = [0] * MAX_BATCH
        self.blocks_used = 0
        # 完成即发布到前缀缓存：块换了主人，没有回池子。只有引擎能把它们还
        # 回来（freed_blocks）。不变量：blocks_used == blocks_held + cached。
        self.cached_blocks = 0
        self.preempt_total = 0
        self.tick_seq = 0

    # --- 查询 ---

    def find(self, req: int) -> int:
        for i in range(MAX_BATCH):
            if self.state[i] != ST_FREE and self.ids[i] == req:
                return i
        return -1

    def digest(self) -> int:
        h = 0
        for i in range(MAX_BATCH):
            for value in (
                self.state[i],
                self.ids[i],
                self.prompt_len[i],
                self.done[i],
                self.generated[i],
                self.max_new[i],
                self.wait_ticks[i],
                self.preempt_count[i],
            ):
                h = (h * DIGEST_MUL + value) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.blocks_used) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.cached_blocks) % DIGEST_MOD
        h = (h * DIGEST_MUL + self.preempt_total) % DIGEST_MOD
        return h

    # --- 状态迁移 ---

    def admit(self, req: int, prompt_len: int, max_new: int) -> None:
        if prompt_len <= 0 or max_new <= 0:
            raise ValueError("prompt_len and max_new must be positive")
        if self.find(req) >= 0:
            raise ValueError(f"duplicate request id {req}")
        if blocks_for(prompt_len, self.cfg.block_size) > self.cfg.capacity:
            raise ValueError(f"prompt {prompt_len} does not fit the kv pool")
        # 准入按水位判据，而不是原始容量：一旦并发序列的稳态足迹超过
        # threshold()，第 9 步就会每拍抢占，而被抢占者从零重算 —— 谁也跑不完。
        # 在这里拒绝，是为了让那个状态不可达。首个请求一律准入，否则过紧的
        # 水位会一条都进不来。缓存不计入判据：它不可抢占，不是并发足迹。
        committed = 0
        for i in range(MAX_BATCH):
            if self.state[i] != ST_FREE:
                committed += blocks_for(
                    self.prompt_len[i] + self.max_new[i], self.cfg.block_size
                )
        whole = blocks_for(prompt_len + max_new, self.cfg.block_size)
        if committed > 0 and committed + whole > self.cfg.threshold():
            raise ValueError("concurrent sequences exceed the kv watermark")
        slot = next((i for i in range(MAX_BATCH) if self.state[i] == ST_FREE), -1)
        if slot < 0:
            raise ValueError("scheduler is full")
        self.ids[slot] = req
        self.prompt_len[slot] = prompt_len
        self.max_new[slot] = max_new
        self.done[slot] = 0
        self.generated[slot] = 0
        self.wait_ticks[slot] = 0
        self.preempt_count[slot] = 0
        self.state[slot] = ST_WAITING

    def release(self, slot: int, to_cache: bool) -> None:
        held = blocks_for(self.done[slot] + self.generated[slot], self.cfg.block_size)
        if to_cache:
            # 发布的是整条序列：prompt 全长 + 要求生成的每一个 token。generated 数的是
            # decode 拍，而续写的第一个 token 是「把 prompt 喂完的那一步」选出来的，
            # 不算一拍 —— 它每次都少一个 token，跨块时就少一整块。
            whole = blocks_for(
                self.prompt_len[slot] + self.max_new[slot], self.cfg.block_size
            )
            self.blocks_used += whole - held
            self.cached_blocks += whole
        else:
            self.blocks_used -= held
        self.ids[slot] = 0
        self.prompt_len[slot] = 0
        self.done[slot] = 0
        self.generated[slot] = 0
        self.max_new[slot] = 0
        self.wait_ticks[slot] = 0
        self.preempt_count[slot] = 0
        self.state[slot] = ST_FREE

    def preempt_one(self, act: dict, protect: int) -> bool:
        order = range(MAX_BATCH)
        if self.variant == "oldest":
            order = range(MAX_BATCH)
        else:
            order = range(MAX_BATCH - 1, -1, -1)
        for i in order:
            if i == protect:
                continue
            if self.state[i] != ST_RUNNING:
                continue
            if self.done[i] + self.generated[i] <= 0:
                continue
            act["X"].append(self.ids[i])
            self.blocks_used -= blocks_for(
                self.done[i] + self.generated[i], self.cfg.block_size
            )
            self.done[i] = 0
            self.generated[i] = 0
            self.wait_ticks[i] = 0
            self.preempt_count[i] += 1
            self.preempt_total += 1
            self.state[i] = ST_WAITING
            return True
        return False

    def ensure_room(self, need: int, protect: int, act: dict) -> None:
        while self.blocks_used + need > self.cfg.capacity:
            if not self.preempt_one(act, protect):
                raise ValueError("kv pool too small for this batch")

    def try_prefill(self, slot: int, budget: int, act: dict) -> int:
        if budget <= 0:
            return budget
        if self.state[slot] != ST_WAITING:
            return budget
        if self.progressed[slot] != 0:
            return budget
        remaining = self.prompt_len[slot] - self.done[slot]
        if remaining <= 0:
            return budget
        chunk = min(self.cfg.chunk, remaining, budget)
        if chunk <= 0:
            return budget
        new_done = self.done[slot] + chunk
        need = blocks_for(new_done, self.cfg.block_size) - blocks_for(
            self.done[slot], self.cfg.block_size
        )
        if self.blocks_used + need > self.cfg.capacity:
            return budget
        act["P"].append((self.ids[slot], self.done[slot], new_done))
        self.blocks_used += need
        self.done[slot] = new_done
        self.progressed[slot] = 1
        self.wait_ticks[slot] = 0
        return budget - chunk

    def step(self, arrivals, cancels, finished, freed_blocks: int = 0) -> dict:
        act = {"t": 0, "P": [], "D": [], "X": [], "F": [], "C": 0, "K": 0}
        self.tick_seq += 1
        act["t"] = self.tick_seq
        self.progressed = [0] * MAX_BATCH

        # 0. 引擎从缓存里还回来的块：上一拍的消息，必须最早入账。
        if freed_blocks:
            if freed_blocks > self.cached_blocks:
                raise ValueError("engine freed more blocks than are cached")
            self.cached_blocks -= freed_blocks
            self.blocks_used -= freed_blocks

        for req in cancels:
            slot = self.find(req)
            if slot >= 0:
                self.release(slot, False)

        for req in finished:
            slot = self.find(req)
            if slot >= 0:
                act["F"].append(self.ids[slot])
                self.release(slot, True)

        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING:
                self.wait_ticks[i] += 1

        # 同拍取消也挡住到达：只"先处理取消"是不够的，取消名单必须能把同拍的
        # 到达一并拦下，否则请求会先入队再被服务。
        for req, prompt_len, max_new in arrivals:
            if req in cancels:
                continue
            self.admit(req, prompt_len, max_new)

        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING and self.prompt_len[i] > 0:
                if self.done[i] >= self.prompt_len[i]:
                    self.state[i] = ST_RUNNING
                    self.wait_ticks[i] = 0

        budget = self.cfg.budget

        # decode 先（每请求 1 token，总量有界），再是延迟护栏，再是常规 prefill。
        for i in range(MAX_BATCH):
            if budget <= 0:
                break
            if self.state[i] != ST_RUNNING:
                continue
            seq = self.done[i] + self.generated[i]
            need = blocks_for(seq + 1, self.cfg.block_size) - blocks_for(
                seq, self.cfg.block_size
            )
            if need > 0:
                self.ensure_room(need, i, act)
            self.blocks_used += need
            self.generated[i] += 1
            act["D"].append(self.ids[i])
            budget -= 1
            if self.generated[i] >= self.max_new[i]:
                act["F"].append(self.ids[i])
                self.release(i, True)

        # 延迟护栏：等待够久的请求排在其它**等待者**之前（不越过 decode）。
        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING and self.wait_ticks[i] >= self.cfg.max_wait:
                budget = self.try_prefill(i, budget, act)

        for i in range(MAX_BATCH):
            if self.state[i] == ST_WAITING:
                budget = self.try_prefill(i, budget, act)

        limit = self.cfg.threshold()
        while self.blocks_used > limit:
            if not self.preempt_one(act, -1):
                break

        act["C"] = self.preempt_total
        act["K"] = self.digest()
        return act


# --- 序列化（与 src/alofa/engine/trace.mojo 逐字节相同）---


def join_or_dash(items) -> str:
    return "-" if not items else ",".join(str(x) for x in items)


def input_line(arrivals, cancels, finished, freed: int = 0) -> str:
    a = "-" if not arrivals else ",".join(f"{r}:{p}:{m}" for r, p, m in arrivals)
    return (
        f"IN=A={a}|C={join_or_dash(cancels)}|F={join_or_dash(finished)}|R={freed}"
    )


def action_line(act: dict) -> str:
    p = (
        "-"
        if not act["P"]
        else ",".join(f"{r}:{s}:{e}" for r, s, e in act["P"])
    )
    return (
        f"OUT=t={act['t']} P={p} D={join_or_dash(act['D'])} "
        f"X={join_or_dash(act['X'])} F={join_or_dash(act['F'])} "
        f"C={act['C']} K={act['K']}"
    )


def reclaim_tail(cfg: Config, ticks, variant: str = "newest", cap: int = 256):
    """补上"引擎回收缓存"的拍：主干里队列被缓存卡住时插一拍，末尾再排空。

    完成即发布以后，缓存会占着块；纯调度器重放里没人回收，队列就再也排不空 ——
    那不是调度器的错，是场景少了一个角色。补拍是确定性的（每拍把当时缓存的
    量原样还回去），Mojo 侧照着重放即可。

    ⚠️ 这个"角色"必须**贯穿**主干，不能只在结尾补：否则缓存会把池子吃满、
    并发掉到 1，抢占顺序就再也影响不到输出 —— 抗原 s02 会悄悄失效。
    """
    sch = Scheduler(cfg, variant=variant)
    out = []
    for tick in ticks:
        freed = tick[3] if len(tick) > 3 else 0
        out.append((tick[0], tick[1], tick[2], freed))
        act = sch.step(tick[0], tick[1], tick[2], freed)
        stalled = not act["P"] and any(
            sch.state[i] == ST_WAITING for i in range(MAX_BATCH)
        )
        if sch.cached_blocks and stalled:
            out.append(([], [], [], sch.cached_blocks))
            sch.step([], [], [], sch.cached_blocks)
    while sum(1 for s in sch.state if s) > 0 and len(out) < cap:
        freed = sch.cached_blocks
        out.append(([], [], [], freed))
        sch.step([], [], [], freed)
    return out


def run(cfg: Config, ticks, variant: str = "newest"):
    sch = Scheduler(cfg, variant=variant)
    lines = [cfg.line()]
    for tick in ticks:
        arrivals, cancels, finished = tick[0], tick[1], tick[2]
        freed = tick[3] if len(tick) > 3 else 0
        act = sch.step(arrivals, cancels, finished, freed)
        lines.append(
            f"{input_line(arrivals, cancels, finished, freed)} {action_line(act)}"
        )
    return lines, sch


def empty(n: int):
    return [([], [], []) for _ in range(n)]


# --- 六个极端场景 ---


def scenarios():
    out = []

    # 1. 超长 prompt：300 token、chunk 16 → 必须跨多拍切片，且切片首尾相接。
    cfg = Config(budget=64, chunk=16, block_size=16, capacity=64, watermark=900, max_wait=8)
    ticks = [([(1, 300, 8)], [], [])] + empty(31)
    out.append(("s01_long_prompt", cfg, ticks))

    # 2. 并发抢占风暴：超限不再由「并发足迹顶破水位」制造 —— 那样的 state 现在被
    #    准入拦在门外，而它本也不该到达（抢占者跑不完）。改由**前缀缓存**制造：
    #    前两条跑完，各自把整条序列发布进缓存（缓存不可抢占），把 blocks_used
    #    顶到阈值；随后三条并发到达 —— 它们自身的稳态足迹合规、准入放行，但叠加
    #    缓存后就过线了，第 9 步开始抢占，受害者从零重算，形成风暴。
    #    收尾靠 reclaim_tail 补的「引擎归还缓存」：缓存一退，请求才跑得完。
    cfg = Config(budget=32, chunk=32, block_size=16, capacity=12, watermark=500, max_wait=8)
    ticks = (
        [([(1, 16, 1)], [], [])]
        + empty(3)
        + [([(2, 16, 1)], [], [])]
        + empty(3)
        + [([(3, 16, 3), (4, 16, 3), (5, 16, 3)], [], [])]
        + empty(40)
    )
    out.append(("s02_preempt_storm", cfg, ticks))

    # 3. 预算耗尽：预算 8、4 个请求各 32 token → 每拍最多 8 个 token。
    cfg = Config(budget=8, chunk=16, block_size=16, capacity=64, watermark=900, max_wait=4)
    ticks = [([(i, 32, 2) for i in range(1, 5)], [], [])] + empty(23)
    out.append(("s03_budget_exhausted", cfg, ticks))

    # 4. 0 预算：一个有预算也不能前进的引擎，动作必须为空但 tick 继续自增。
    cfg = Config(budget=0, chunk=16, block_size=16, capacity=64, watermark=900, max_wait=2)
    ticks = [([(1, 32, 4)], [], [])] + empty(7)
    out.append(("s04_zero_budget", cfg, ticks))

    # 5. 取消竞态：同拍到达又被取消的请求必须从未被调度过；取消优先于完成。
    cfg = Config(budget=16, chunk=16, block_size=16, capacity=64, watermark=900, max_wait=4)
    ticks = [
        ([(1, 32, 2), (2, 16, 2)], [], []),
        ([(3, 16, 2)], [1, 3], []),          # 3 到达即被取消；1 在跑着被取消
        ([(4, 32, 2), (5, 32, 2)], [], []),
        ([], [5], [5]),                       # 取消与完成同拍：取消先手，完成落空
        ([], [], [4]),                        # 引擎报告 4 完成
    ] + empty(8)
    out.append(("s05_cancel_race", cfg, ticks))

    # 6. KV 水位临界：同 s02 的机制（缓存顶高 → 水位抢占），但池子更宽裕，用来验
    #    「水位是策略线、硬容量是物理线」这两件事没有混起来：抢占全程把 blocks_used
    #    拉回阈值之内，任何一拍都没有越过硬容量。
    cfg = Config(budget=32, chunk=32, block_size=16, capacity=16, watermark=500, max_wait=8)
    ticks = (
        [([(1, 16, 1)], [], [])]
        + empty(3)
        + [([(2, 16, 1)], [], [])]
        + empty(3)
        + [([(3, 16, 1)], [], [])]
        + empty(3)
        + [([(4, 16, 3), (5, 16, 3), (6, 16, 3)], [], [])]
        + empty(40)
    )
    out.append(("s06_kv_watermark", cfg, ticks))

    # 7. 前缀缓存归还：完成即发布到缓存（块换了主人，没回池子），只有引擎的
    #    freed_blocks 能把它们还回来。这条通道不接上，缓存会把 blocks_used
    #    一路顶高，后面的请求永远排不上 —— 看起来像"池子太小"，其实是账没接。
    #    水位取 1000‰：这个场景要验的是归还通道，不是抢占。
    cfg = Config(budget=16, chunk=16, block_size=16, capacity=12, watermark=1000, max_wait=8)
    ticks = [
        ([(1, 16, 1), (2, 16, 1)], [], [], 0),
        ([], [], [], 0),
        ([(3, 16, 1), (4, 16, 1)], [], [], 0),
        ([], [], [], 0),
        ([], [], [], 0),
        ([(5, 16, 1), (6, 16, 1)], [], [], 4),   # 引擎清掉 4 个缓存块
        ([], [], [], 0),
        ([(7, 16, 1), (8, 16, 1)], [], [], 2),   # 再还 2 个
        ([], [], [], 0),
        ([], [], [], 1),                          # 部分归还
    ] + empty(6)
    out.append(("s07_cache_freed", cfg, ticks))

    return out


def write(name: str, lines) -> None:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"  {path}  ({len(lines) - 1} ticks)")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("scheduler trace fixtures:")

    produced = {}
    for name, cfg, ticks in scenarios():
        base = ticks
        # s04 故意永远排不空（0 预算），其余场景补上"引擎回收缓存"的收尾拍。
        if name != "s04_zero_budget":
            ticks = reclaim_tail(cfg, ticks)
        lines, sch = run(cfg, ticks)
        write(f"{name}.trace", lines)
        print(
            f"    preempt_total={sch.preempt_total} "
            f"blocks_used={sch.blocks_used} live={sum(1 for s in sch.state if s)}"
        )
        produced[name] = (cfg, ticks, lines, base)

    # 抗原一：故意改坏一拍的 OUT（把 s01 第 1 拍的切片终点从 16 改成 15）。
    # 重放门必须判红；若它判绿，说明逐字节比对根本没在比。
    s01_lines = produced["s01_long_prompt"][2]
    bad = list(s01_lines)
    bad[1] = bad[1].replace("P=1:0:16", "P=1:0:15")
    assert bad[1] != s01_lines[1], "改坏失败：找不到要改的字段"
    write("bad.trace", bad)

    # 抗原二：换一个同样自洽但不同的策略（抢占最老的而不是最新的）跑**有抢占**
    # 的场景。Mojo 的输出必须与之不同 —— 否则 fixture 只分辨得了格式，分辨不了策略。
    s02_cfg, _, s02_lines, s02_base = produced["s02_preempt_storm"]
    # 只跑主干、不补回收拍：回收拍是策略相关的（缓存量不同），喂给另一种策略会
    # 触发"归还数超过缓存数"。抗原要验的是"输出字节不同"，不是"输入序列不同"。
    alt_lines, _ = run(s02_cfg, s02_base, variant="oldest")
    write("alt.trace", alt_lines)
    if alt_lines == s02_lines:
        raise SystemExit("alt 策略与默认策略产生了同样的字节 —— 这条抗原无效")


if __name__ == "__main__":
    main()
