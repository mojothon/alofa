"""与 llama.cpp 同机同 prompt 对齐的端到端计时（alofa 侧）。

为什么单独写这份
----------------
账本「与 llama.cpp 的同机同 prompt 对比」那一栏至今是 `missing`，而它是**唯一被
接受的性能参考基准**。上一次尝试（2026-09-18）在 A100 共享机上做，三条口径没
对齐，所以数字没被接受：

  ① alofa 报的是**含 prefill** 的墙钟，llama.cpp 的 eval time 是**纯 decode**；
  ② 两边 prompt token 数不同（5 vs 13 —— 当时没对账 tokenizer）；
  ③ 共享机（6 卡全忙）有噪声。

这份脚本只解决①和②：

  * **prefill 与 decode 分开报** —— llama-bench 的 `pp` / `tg` 就是分开的，alofa
    混在一起就没法比；
  * **把 tokenizer 出来的 id 打印出来** —— 拿去和 `llama-tokenize` 对账。id 不同
    就不是"同 prompt"，后面的数字一律不成立；
  * **greedy（argmax）** —— 排除采样随机性对时间的干扰；
  * **预热轮丢弃** —— 它量的是 mmap 缺页，不是计算；
  * **中位数**为主统计量（本机 runq 常年 6–27，均值会被一次抢占拖走）；
  * **两档 shards（1 / 8）交错跑** —— 对应 llama.cpp 的 `-t 1` / `-t 8`；
  * **两档权重（fp32 / q4_0）** —— q4_0 是 llama.cpp 的强项档，只报 fp32 会把
    差距说反。⚠️ alofa 的 q4 是**加载期量化**（同一份 fp32 权重 + `enable_q4`），
    推理期不量化；llama.cpp 那边的 q4_0 由 `llama-quantize` 从**同一个** f32 GGUF
    转出，两边块布局相同，但**缩放因子的选法不同**（alofa 按 MSE，llama.cpp 按
    amax/7 一类）⇒ 质量不同、流量相同，所以**速度可比、质量不可比**。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_vs_llamacpp.mojo \\
        -o target/bench_vs_llamacpp
    ./target/bench_vs_llamacpp

输出每行带 `[alofa]` 前缀，便于与 llama.cpp 的输出并排 grep。

⚠️ 边界（不可省略）：本机长期过载，绝对值只作**同轮同机相对比较**；批=1 恒
memory 受限，所以这里量的是"alofa 今天的水平"，不是它的天花板。
"""

from alofa.core.ffi import monotonic_ns
from alofa.core.text import parse_int, read_text
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward
from alofa.tokenizer import Tokenizer, load_tokenizer

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime PROMPT = "The capital of France is"
# 历史长度上限：prompt 5 个 + 32 个新 token。
comptime MAX_TOK = 64
comptime NEW = 32
comptime WARMUP_NEW = 2
comptime ROUNDS = 3
# 两档：1 = 不切（单线程语义），8 = 本机核数。
comptime ARMS = 2
# 两档权重：fp32 与 q4_0（加载期量化）。
# 三档：fp32 / q4_0 / **q4_0 + q8_0 输出投影**。第三档是这轮要看的那档 —— 它只改
# 输出投影的宽度（544 MB → 165 MB），24 层与 q4_0 档逐位相同，于是两档之差就是
# "这一个张量的流量"值多少钱。
comptime KINDS = 3


def shard_of(i: Int) -> Int:
    return 1 if i == 0 else 8


def f2(v: Float64) -> String:
    """两位小数。本机没有 sprintf，整数拼是唯一不引入新依赖的写法。"""
    var i = Int(v * 100.0)
    var frac = i % 100
    var t = String(frac)
    if frac < 10:
        t = "0" + t
    return String(i // 100) + "." + t


def median_ns(imm steps: List[Int], start: Int) -> Int:
    """`steps[start:]` 的中位数（start 用来跳过第 0 格的 prefill）。"""
    var v = List[Int]()
    for i in range(start, len(steps)):
        v.append(steps[i])
    var i = 1
    while i < len(v):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
        i += 1
    return v[len(v) // 2]


def runq_len() raises -> Int:
    """/proc/loadavg 第 4 个字段里 `/` 左边的数：此刻**可运行**的进程数。"""
    var text = read_text("/proc/loadavg")
    var toks = List[String]()
    for t in text.split(" "):
        var word = String(t)
        if word.byte_length() > 0:
            toks.append(word)
    if len(toks) < 4:
        return -1
    var fourth = toks[3]
    for part in fourth.split("/"):
        return parse_int(String(part))
    return -1


def run_once[backend: Int](
    mut model: QwenForward,
    imm tok: Tokenizer,
    n_new: Int,
    shards: Int,
) raises -> List[Int]:
    """跑一趟，返回 `[prefill_ns, step0_ns, step1_ns, …]`。

    **prefill 单独占第 0 格**：llama-bench 把 `pp` 与 `tg` 分开报，alofa 混进分母
    就会让 tok/s 随生成长度漂移，也就没法对齐。

    ⚠️ 为什么把两个输出塞进一个 List：Mojo 1.0 **一个函数只允许一个 `out` 参数**
    （编译期就拒），而 `mut` 借用的调用语法在本机这个版本上又不对。返回一条 List、
    按约定取第 0 格，是这里唯一不跟编译器较劲的写法。
    """
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


def main() raises:
    print("")
    print("== alofa：与 llama.cpp 同机同 prompt 对齐的端到端计时 ==")
    print("   backend avx2 / greedy / prefill 与 decode 分开报 / fp32 与 q4 两档")

    var tok = load_tokenizer(FIXTURE)

    # ① 与 llama.cpp 对账用：这段 prompt 的 id 必须和 `llama-tokenize` 一致。
    var ids = tok.encode(PROMPT)
    var ids_s = ""
    for i in range(len(ids)):
        if i > 0:
            ids_s = ids_s + ","
        ids_s = ids_s + String(ids[i])
    print("[alofa] prompt=" + PROMPT)
    print("[alofa] prompt_tokens=" + String(len(ids)) + " ids=[" + ids_s + "]")
    print("[alofa] new_tokens=" + String(NEW) + " rounds=" + String(ROUNDS))

    for kind in range(KINDS):
        var kind_s = String("fp32")
        var use_q4 = False
        var use_q8_head = False
        if kind == 1:
            kind_s = "q4_0"
            use_q4 = True
        elif kind == 2:
            kind_s = "q4_0+q8_head"
            use_q4 = True
            use_q8_head = True

        print("")
        print("   ---- 权重档：" + kind_s + " ----")
        var t0 = monotonic_ns()
        var model = QwenForward(
            FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK, use_q4
        )
        var load_ms = (monotonic_ns() - t0) // 1000000
        if model.q4_enabled:
            print("   q4_enabled=true（推理确实走量化通路）")
        else:
            print("   q4_enabled=false")
        if use_q8_head:
            var tq = monotonic_ns()
            model.enable_q8_output()
            print(
                "   q8_head_enabled=true（输出投影走 q8_0 通路），量化耗时 "
                + String((monotonic_ns() - tq) // 1000000)
                + " ms"
            )
        print("   权重加载（含量化）: " + String(load_ms) + " ms")

        # 预热：两档各跑一遍丢弃（量的是 mmap 缺页，不是计算）。
        for k in range(ARMS):
            _ = run_once[BACKEND_AVX2](model, tok, WARMUP_NEW, shard_of(k))

        var med = List[Int]()
        var pfs = List[Int]()
        for _ in range(ROUNDS * ARMS):
            med.append(0)
            pfs.append(0)

        for rep in range(ROUNDS):
            var q0 = runq_len()
            for k in range(ARMS):
                var arm = (rep + k) % ARMS
                var out = run_once[BACKEND_AVX2](model, tok, NEW, shard_of(arm))
                var pf = out[0]
                var m = median_ns(out, 1)
                med[rep * ARMS + arm] = m
                pfs[rep * ARMS + arm] = pf
                print(
                    "[alofa] kind="
                    + kind_s
                    + " round="
                    + String(rep + 1)
                    + " shards="
                    + String(shard_of(arm))
                    + " prefill_ms="
                    + f2(Float64(pf) / 1000000.0)
                    + " decode_ms_per_tok="
                    + f2(Float64(m) / 1000000.0)
                    + " decode_tok_s="
                    + f2(1000000000.0 / Float64(m))
                    + " runq="
                    + String(q0)
                )

        # 单独一趟只为拿生成文本：与 llama.cpp 的输出对账，确认两边是**同一个
        # 模型**在说同一句话（数字可比的前提）。这趟不计入上面的统计。
        model.set_shards(shard_of(0))
        model.reset()
        var lgt = model.prefill[BACKEND_AVX2](tok.encode(PROMPT))
        var nx = model.argmax(lgt)
        var gids = List[Int]()
        for _ in range(NEW):
            gids.append(nx)
            nx = model.argmax(model.step[BACKEND_AVX2](nx))
        var text = tok.decode(gids)

        print("   汇总（" + kind_s + "，每档 " + String(ROUNDS) + " 趟，min…max）")
        for arm in range(ARMS):
            var lo = med[arm]
            var hi = med[arm]
            var plo = pfs[arm]
            var phi = pfs[arm]
            for rep in range(1, ROUNDS):
                var x = med[rep * ARMS + arm]
                if x < lo:
                    lo = x
                if x > hi:
                    hi = x
                var y = pfs[rep * ARMS + arm]
                if y < plo:
                    plo = y
                if y > phi:
                    phi = y
            print(
                "      shards="
                + String(shard_of(arm))
                + " : decode "
                + f2(Float64(lo) / 1000000.0)
                + "…"
                + f2(Float64(hi) / 1000000.0)
                + " ms/token  ("
                + f2(1000000000.0 / Float64(hi))
                + "…"
                + f2(1000000000.0 / Float64(lo))
                + " tok/s)   prefill "
                + f2(Float64(plo) / 1000000.0)
                + "…"
                + f2(Float64(phi) / 1000000.0)
                + " ms"
            )
        print("[alofa] text(" + kind_s + ")=" + text)

    print("")
