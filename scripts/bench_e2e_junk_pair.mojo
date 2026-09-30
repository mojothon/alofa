"""端到端剂量-响应：**同进程两臂配对**（`JUNK=0` 对 `JUNK=3`）。

为什么要这份
------------
跨进程 A/B 在本机判不了（runq 常年 6–31，同档两次能差 ±25%），而 2026-09-30 那次
跨进程测出的"+65%"刚好落在噪声里。同进程两臂配对把 runq/调度/缺页这些**共同的**东西
抵消掉：两臂在同一个进程里各实例化一份核（`objdump` 实证 `JUNK=0` ×205、`JUNK=3` ×207），
`QwenForward.set_junk_arm()` 每次调用切一次（168 次投影里的一个 `if`，不是每块）。

⚠️ 加料是**数值中性**的：`JUNK` 只把剂量累加进 `sink`，收尾判给 `+inf`（永不命中）
⇒ 与 `JUNK=0` 那臂**逐位相同**。这条由脚本自己验：两臂生成的 token 序列必须完全
一致，不一致就是加料改了数值、整份测量作废（2026-09-30 用旧的"改累加器"加料就踩了
这个：生成文本变成 `ṡṡṡṡ…`）。

读法
----
比值 `JUNK=3 / JUNK=0`：
  * 若 ≈1.0 ⇒ 核内指令不是瓶颈，端到端那 2× 得去别处找（带宽/调度）；
  * 若 ≈1.3…1.5（接近单核探针的 +49% 按核内占比折算）⇒ **核内是瓶颈**，
    砍指令能换到端到端时间 ⇒ 值得做。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_e2e_junk_pair.mojo \\
        -o target/bench_e2e_junk_pair
    ./target/bench_e2e_junk_pair
"""

from alofa.core.ffi import monotonic_ns
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward, q4_mub_enabled
from alofa.tokenizer import Tokenizer, load_tokenizer


comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime PROMPT = "The capital of France is"
comptime MAX_TOK = 64
comptime NEW = 32
comptime WARMUP_NEW = 2
comptime ROUNDS = 9
comptime BEST = 5
# 主战档：shards=8（多核端到端），权重档 q4_0 + q8_head。
comptime SHARDS = 8


def f2(v: Float64) -> String:
    """两位小数。本机没有 sprintf，整数拼是唯一不引入新依赖的写法。"""
    var i = Int(v * 100.0)
    var frac = i % 100
    var t = String(frac)
    if t.byte_length() < 2:
        t = "0" + t
    return String(Int(v)) + "." + t


def ms(ns: Int) -> String:
    return f2(Float64(ns) / 1000000.0)


def run_once[backend: Int](
    mut model: QwenForward, imm tok: Tokenizer, n_new: Int, shards: Int
) raises -> List[Int]:
    """跑一趟，返回 `[prefill_ns, step0_ns, …]`（与 `bench_vs_llamacpp` 同一个约定）。"""
    model.set_shards(shards)
    model.reset()
    var out = List[Int]()
    var ids = tok.encode(PROMPT)

    var t0 = monotonic_ns()
    var logits = model.prefill[backend](ids)
    out.append(monotonic_ns() - t0)

    var next = model.argmax(logits)
    for _ in range(n_new):
        var t1 = monotonic_ns()
        var lg = model.step[backend](next)
        out.append(monotonic_ns() - t1)
        next = model.argmax(lg)
    return out^


def run_ids[backend: Int](
    mut model: QwenForward, imm tok: Tokenizer, n_new: Int, shards: Int
) raises -> List[Int]:
    """跑一趟，返回**生成的 token id 序列**（给数值中性自检用）。"""
    model.set_shards(shards)
    model.reset()
    var ids = tok.encode(PROMPT)
    var got = List[Int]()
    var logits = model.prefill[backend](ids)
    var next = model.argmax(logits)
    got.append(next)
    for _ in range(n_new - 1):
        var lg = model.step[backend](next)
        next = model.argmax(lg)
        got.append(next)
    return got^


def best_step(mut model: QwenForward, imm tok: Tokenizer) raises -> Int:
    """`BEST` 趟里最快的那一步（best-of：本机噪声下 min 比均值稳得多）。"""
    var best = Int(0)
    for b in range(BEST):
        var r = run_once[BACKEND_AVX2](model, tok, NEW, SHARDS)
        var mn = r[1]
        var i = 2
        while i < len(r):
            if r[i] < mn:
                mn = r[i]
            i += 1
        if b == 0 or mn < best:
            best = mn
    return best


def main() raises:
    print("")
    print("== 端到端剂量-响应：同进程两臂配对（JUNK=0 vs JUNK=3）==")
    print(
        "   q4 通路："
        + ("vpmaddubsw（默认）" if q4_mub_enabled() else "f32 展开（ALOFA_Q4_MUB=0）")
    )
    print("   shards=" + String(SHARDS) + " · 每轮每臂 best-of-" + String(BEST)
        + " · " + String(ROUNDS) + " 轮 · 每轮轮换出场顺序")

    var tok = load_tokenizer(FIXTURE)
    var model = QwenForward(
        FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK, True
    )
    model.enable_q8_output()

    # ① 数值中性自检：两臂必须生成**完全相同**的 token 序列。
    #    不一致 ⇒ 加料改了数值 ⇒ 后面所有时间都不算数。
    model.set_junk_arm(0)
    var ids0 = run_ids[BACKEND_AVX2](model, tok, NEW, SHARDS)
    model.set_junk_arm(3)
    var ids3 = run_ids[BACKEND_AVX2](model, tok, NEW, SHARDS)
    var same = len(ids0) == len(ids3)
    var i = 0
    while same and i < len(ids0):
        if ids0[i] != ids3[i]:
            same = False
        i += 1
    if same:
        print("   [自检] 两臂生成 " + String(len(ids0)) + " 个 token 逐位相同 ⇒ 加料数值中性 ✓")
    else:
        print("   [自检] ❌ 两臂 token 序列不同 ⇒ 加料改了数值，以下时间全部作废")
        return

    # ② 预热（丢弃）：量的是 mmap 缺页，不是计算。
    for k in range(2):
        model.set_junk_arm(0 if k == 0 else 3)
        _ = run_once[BACKEND_AVX2](model, tok, WARMUP_NEW, SHARDS)

    # ③ 配对：每轮两臂各 best-of-BEST，轮换出场顺序。
    var lo = Float64(9.99)
    var hi = Float64(0.0)
    var acc = Float64(0.0)
    for rep in range(ROUNDS):
        var t0 = Int(0)
        var t3 = Int(0)
        for k in range(2):
            # 轮换：偶数轮 0→3，奇数轮 3→0
            var first = (rep + k) % 2
            var arm = 0 if first == 0 else 3
            model.set_junk_arm(arm)
            var t = best_step(model, tok)
            if arm == 0:
                t0 = t
            else:
                t3 = t
        var ratio = Float64(t3) / Float64(t0)
        if ratio < lo:
            lo = ratio
        if ratio > hi:
            hi = ratio
        acc += ratio
        print(
            "   round=" + String(rep) + "  JUNK=0 " + ms(t0) + " ms · JUNK=3 "
            + ms(t3) + " ms · 比值 " + f2(ratio)
        )

    print("")
    print("   比值 JUNK=3/JUNK=0 ：" + f2(lo) + " … " + f2(hi)
        + "（中位 " + f2(acc / Float64(ROUNDS)) + "）")
    print("   读法：≈1.0 ⇒ 核内不是瓶颈；明显 >1 ⇒ 端到端时间跟着核内指令走")
