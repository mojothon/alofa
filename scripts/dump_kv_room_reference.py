#!/usr/bin/env python
"""导出「引擎 KV 房间」的参照 trace（独立于 Mojo 的 Python 实现）。

房间是**翻译层**：把引擎的请求（切片 prefill、每拍一个 token 的 decode、完成即
发布）翻译成 `KvSpace` 的操作，并且是唯一知道"哪些块不属于任何请求"的地方。
它自己的算法（`KvSpace`）已由 `dump_kv_reference.py` 逐字节验过 —— 那份 Python
实现被直接 import 复用，所以这里只多写一层"房间"，不重写块池。

被验的三件事：
1. 切片不等于 prompt（prompt 没看完不许发布进缓存）；
2. 完成即发布：块换了主人，占用不降；
3. 归还只有一扇门：驱逐回来的块数必须等于"缓存块数的减少量"。

产物：`tests/fixtures/kvroom/*.trace`，Mojo 侧 `tests/unit/test_engine_kv.mojo`
逐字节重放。
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "tests", "fixtures", "kvroom")

spec = importlib.util.spec_from_file_location(
    "dump_kv_reference", os.path.join(HERE, "dump_kv_reference.py")
)
kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv)

MAX_BLOCKS = kv.MAX_BLOCKS
MAX_REQUESTS = kv.MAX_REQUESTS
MAX_SEQ_TOKENS = kv.MAX_SEQ_TOKENS
MAX_NODES = kv.MAX_NODES


class KvRoom:
    """引擎侧的 KV 房间（Python 版，与 `engine/kv_room.mojo` 同策略）。"""

    def __init__(self, block_size, policy="coldest"):
        self.space = kv.KvSpace(block_size)
        self.policy = policy
        self.n_prompt = [0] * MAX_REQUESTS
        self.published = [0] * MAX_REQUESTS
        self.freed_pending = 0
        self.last_matched = 0
        self.last_fresh = 0

    def cached_blocks(self):
        owned = [0] * MAX_BLOCKS
        for s in range(MAX_REQUESTS):
            if self.space.rq_live[s] == 1:
                for i in range(self.space.rq_nblk[s]):
                    owned[self.space.block_at(s, i)] = 1
        return sum(
            1 for b in range(MAX_BLOCKS) if self.space.pool.refcnt[b] > 0 and not owned[b]
        )

    def admit(self, req, tokens, prompt_len, n_now):
        if prompt_len <= 0 or prompt_len > MAX_SEQ_TOKENS:
            raise kv.AlofaError(kv.ERR_OUT_OF_RANGE, "bad prompt length")
        if n_now <= 0 or n_now > prompt_len:
            raise kv.AlofaError(kv.ERR_OUT_OF_RANGE, "first slice exceeds the prompt")
        res = self.space.new_request(req, tokens, n_now)
        slot = self.space.slot_of(req)
        self.n_prompt[slot] = prompt_len
        self.published[slot] = 0
        self.last_matched = max(res.matched, 0)
        self.last_fresh = len(res.fresh)
        return res.used

    def extend(self, req, n):
        slot = self.space.slot_of(req)
        if self.published[slot] == 1:
            raise kv.AlofaError(kv.ERR_INVALID_ARGUMENT, "cannot grow a published request")
        res = self.space.append_tokens(slot, n)
        return res.used

    def prompt_done(self, req):
        slot = self.space.slot_of(req)
        return self.space.rq_ntok[slot] >= self.n_prompt[slot]

    def publish(self, req):
        slot = self.space.slot_of(req)
        if self.space.rq_ntok[slot] < self.n_prompt[slot]:
            raise kv.AlofaError(
                kv.ERR_INVALID_ARGUMENT, "refusing to cache a prompt that was never fully seen"
            )
        self.space.commit(slot)
        self.published[slot] = 1
        return self.cached_blocks()

    def drop(self, req):
        slot = self.space.slot_of(req)
        res = self.space.release(slot)
        self.n_prompt[slot] = 0
        self.published[slot] = 0
        return res.used

    def coldest_node(self):
        best = -1
        best_freq = 0
        for n in range(1, MAX_NODES):
            if self.space.tree.alive[n] == 1:
                if self.policy == "hottest":
                    if best < 0 or self.space.tree.freq[n] >= best_freq:
                        best = n
                        best_freq = self.space.tree.freq[n]
                else:
                    if best < 0 or self.space.tree.freq[n] < best_freq:
                        best = n
                        best_freq = self.space.tree.freq[n]
        return best

    def reclaim(self, need):
        """归还至少 need 个缓存块（驱逐的单位是节点，所以可能多还）。"""
        if need <= 0:
            return 0
        before = self.cached_blocks()
        guard = 0
        while before - self.cached_blocks() < need and guard < MAX_NODES:
            node = self.coldest_node()
            if node < 0:
                break
            self.space.evict(node)
            guard += 1
        freed = before - self.cached_blocks()
        self.freed_pending += freed
        return freed


def prompt_tokens(base, n):
    """一条可复现的 prompt：base 决定起点，n 决定长度。"""
    return [(base + i) % 30000 + 1 for i in range(n)]


def op_text(op):
    kind = op[0]
    if kind == "ADM":
        return f"ADM r={op[1]} pl={op[2]} n={op[3]} tk={','.join(str(t) for t in op[4])}"
    if kind == "EXT":
        return f"EXT r={op[1]} n={op[2]}"
    if kind == "PUB":
        return f"PUB r={op[1]}"
    if kind == "DRP":
        return f"DRP r={op[1]}"
    if kind == "REC":
        return f"REC need={op[1]}"
    raise SystemExit(f"unknown op {kind}")


def apply_op(room, op):
    kind = op[0]
    if kind == "ADM":
        room.admit(op[1], op[4], op[2], op[3])
    elif kind == "EXT":
        room.extend(op[1], op[2])
    elif kind == "PUB":
        room.publish(op[1])
    elif kind == "DRP":
        room.drop(op[1])
    elif kind == "REC":
        room.reclaim(op[1])
    else:
        raise SystemExit(f"unknown op {kind}")


def result_text(room):
    return (
        f"u={room.space.pool.used} f={room.space.pool.n_free} "
        f"c={room.cached_blocks()} d={room.space.digest()}"
    )


def run(block_size, ops, policy="coldest"):
    room = KvRoom(block_size, policy=policy)
    lines = [f"CFG bs={block_size}"]
    for op in ops:
        apply_op(room, op)
        lines.append(f"{op_text(op)} {result_text(room)}")
        bad = room.space.check_invariants()
        if bad:
            raise SystemExit(f"参照实现自己就违反不变量（{bad} 条）：{op_text(op)}")
    return lines, room


def scenarios():
    out = []

    # 1. 第二条一模一样的 prompt：命中缓存，一个新块都不用。
    tok = prompt_tokens(1000, 32)
    out.append(
        (
            "r01_second_prompt_is_free",
            16,
            [
                ("ADM", 1, 32, 32, tok),
                ("PUB", 1),
                ("DRP", 1),
                ("ADM", 2, 32, 32, tok),   # 命中 32 个 token，0 个新块
                ("EXT", 2, 1),             # decode 一步：越过共享前缀，才要新块
                ("PUB", 2),
                ("DRP", 2),
            ],
        )
    )

    # 2. 切片 prefill：prompt 分两拍才给完，发布前必须给完。
    tok2 = prompt_tokens(2000, 32)
    out.append(
        (
            "r02_sliced_prompt",
            16,
            [
                ("ADM", 1, 32, 16, tok2),
                ("EXT", 1, 16),
                ("PUB", 1),
                ("DRP", 1),
                ("ADM", 2, 32, 16, tok2),
                ("EXT", 2, 16),
                ("PUB", 2),
                ("DRP", 2),
            ],
        )
    )

    # 3. decode：每拍一个 token，块按需增长。
    tok3 = prompt_tokens(3000, 8)
    out.append(
        (
            "r03_decode_growth",
            16,
            [
                ("ADM", 1, 8, 8, tok3),
                ("EXT", 1, 1),
                ("EXT", 1, 1),
                ("EXT", 1, 1),
                ("PUB", 1),
                ("DRP", 1),
            ],
        )
    )

    # 4. 压力与驱逐：先把池子塞满 —— 靠缓存，而不是靠活着的请求（活着的请求最多
    #    8×64 个 token，根本塞不满 112 个块）。塞满之后回收才是真在驱逐，归还数
    #    才真的等于"缓存块数的减少量"。
    ops = []
    for req in range(1, 7):
        ops.append(("ADM", req, 64, 64, prompt_tokens(4000 + 100 * req, 64)))
        ops.append(("PUB", req))
        ops.append(("DRP", req))
    ops.append(("REC", 4))     # 驱逐一个节点：还回 4 个（节点是驱逐的单位）
    ops.append(("REC", 8))     # 再驱逐两个节点
    ops.append(("REC", 64))    # 剩下的全清掉
    ops.append(("REC", 4))     # 已经没有缓存了：归还 0
    out.append(("r04_evict_returns_cache", 16, ops))

    # 5. 混合：一边发布一边命中，中间插一次回收。
    tok5 = prompt_tokens(5000, 48)
    out.append(
        (
            "r05_hit_then_reclaim",
            16,
            [
                ("ADM", 1, 48, 16, tok5),
                ("EXT", 1, 32),
                ("PUB", 1),
                ("DRP", 1),
                ("ADM", 2, 48, 48, tok5),   # 整段命中
                ("PUB", 2),
                ("DRP", 2),
                ("ADM", 3, 48, 24, tok5),   # 只给一半也命中前缀
                ("EXT", 3, 24),
                ("REC", 2),
                ("PUB", 3),
                ("DRP", 3),
            ],
        )
    )

    return out


def write(name, lines):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"  {path}  ({len(lines) - 1} ops)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("kvroom trace fixtures:")

    produced = {}
    for name, block_size, ops in scenarios():
        lines, room = run(block_size, ops)
        write(f"{name}.trace", lines)
        produced[name] = (block_size, ops, lines)
        print(
            f"    used={room.space.pool.used} cached={room.cached_blocks()} "
            f"freed_pending={room.freed_pending}"
        )

    # 抗原一：故意改坏一拍的结果（把第一拍的占用加一）。重放门必须判红。
    base = produced["r01_second_prompt_is_free"][2]
    bad = list(base)
    bad[1] = bad[1].replace("u=2", "u=3", 1)
    assert bad[1] != base[1], "改坏失败：找不到要改的字段"
    write("bad_room.trace", bad)

    # 抗原二：换一个同样自洽但不同的驱逐策略（驱逐**最热**的节点）。
    bs, ops, _ = produced["r04_evict_returns_cache"]
    alt_lines, _ = run(bs, ops, policy="hottest")
    write("alt_room.trace", alt_lines)
    if alt_lines == produced["r04_evict_returns_cache"][2]:
        raise SystemExit("alt 策略与默认策略产生了同样的字节 —— 这条抗原无效")


if __name__ == "__main__":
    sys.exit(main())
