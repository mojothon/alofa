"""端到端 decode A/B：把一次投影切成 N 片并发跑，一个 token 能快多少。

它接着 `bench_thread_matmul.mojo` 往下走一站
--------------------------------------------
那份量的是**核级**：`down_proj`（896 × 4864）按输出切成 T 份，`fp32/avx2` 在
T=4 上给 2.00–2.38×、`q4_0/avx2` 在 T=8 上给 3.69–5.76×。核级快不等于端到端快：

  * 一个 token 要走 **169 次**投影，其中最大的 `down_proj` 只占权重流量的一部分，
    而 `lm_head`（151936 × 896，544 MB）一家就占每 token 权重流量的 **27.6%**；
  * 注意力 / RMSNorm / RoPE / 残差那几段**一次都没被切**，它们按 Amdahl 原样留在
    串行段里 —— 串行段不小，天花板就不是核级那个数；
  * 169 次投影意味着 **169 × N 次协程调度**，核级只调度 N 次。

所以这里量的是**真的一个 token**：加载真实的 0.5B fp32 权重，prefill 一个短
prompt，然后逐 token 贪心解码，取每 token 时间的**中位数**。

为什么走 `QwenForward.prefill / step` 而不是引擎
------------------------------------------------
引擎（`EngineCore.tick`）把一次前向拆成调度、分页 KV、批处理三段，它要回答的是
"批起来对不对"；这里要回答的是"一个 token 的算术与访存能不能并行"，多走一层
只会多一个解释变量。`model.step` 就是那 169 次投影本身。

口径（与 `src/alofa/cli.mojo` 那张重复交错表同源）
-------------------------------------------------
  * **必须 `-O2`**。
  * **交错**：外层轮次、内层片数，轮换起跑顺序（`arm = (rep + k) % ARMS`）——
    谁排在前面不该成为解释变量。
  * 主口径 = **同轮配对比值**（同一轮里 `shards=1` 的中位数 ÷ `shards=N` 的中位
    数）：同一进程、同一份权重、挨着跑，机器慢下来的那一段被除掉。
  * 保守对照 = **非配对区间**（各档自己的 min…max，按相反方向相除）。
  * 两个口径（每 token 中位数 / decode 均值）**同向且都排除 1.0** 才给结论；
    默认值永远是"量不出结论"，不许取好看的那个。
  * 预热轮（2 个 token）丢弃：它量的是 mmap 的缺页，不是 decode。
  * `shards=1` 那档走的是**直呼**（`shard_count` 在 `<= 1` 时返回 1），也就是今天
    线上那条路 —— 否则量出来的是"换调度机制的收益"，不是"并行的收益"。
  * 顺带读 `/proc/loadavg` 的 runq：本机 8 核 runq 常年 6–27，掉速那一轮如果
    runq 也高，至少说明"那段时间机器上还有别人"。它证明不了谁抢走了什么。
  * **不拿绝对值当结论**：本机同日绝对 tok/s 抖 2.8×，绝对值只用来看量级。

跑法：

    pixi run mojo run -O2 -I src scripts/bench_model_shards.mojo
"""

from alofa.core.ffi import monotonic_ns
from alofa.core.text import parse_int, read_text
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward
from alofa.tokenizer import Tokenizer, load_tokenizer

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime PROMPT = "The capital of France is"
# 历史长度上限：prompt 5 个 + 32 个新 token。与 `cli.mojo` 的 `ROWS` 同量。
comptime MAX_TOK = 64
comptime NEW = 32
comptime WARMUP_NEW = 2
comptime ROUNDS = 3

# 四档片数：1 是不切（基线），8 是本机的核数。
comptime ARMS = 4


def shard_of(i: Int) -> Int:
    if i == 0:
        return 1
    if i == 1:
        return 2
    if i == 2:
        return 4
    return 8


def shard_label(i: Int) -> String:
    var s = String(shard_of(i))
    while s.byte_length() < 2:
        s = " " + s
    return s


def f2(v: Float64) -> String:
    """两位小数。本机没有 sprintf，整数拼是唯一不引入新依赖的写法。"""
    var i = Int(v * 100.0)
    var frac = i % 100
    var t = String(frac)
    if frac < 10:
        t = "0" + t
    return String(i // 100) + "." + t


def median_ns(imm steps: List[Int]) -> Int:
    var v = List[Int]()
    for i in range(len(steps)):
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
    if len(v) == 0:
        return 0
    return v[len(v) // 2]


def sum_ns(imm steps: List[Int]) -> Int:
    var s = 0
    for i in range(len(steps)):
        s += steps[i]
    return s


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
    """跑一趟（prefill + `n_new` 个 token），返回**逐 token** 的 decode 时间。

    prefill 的时间**不在返回值里**：它是"一次"而不是"每 token"，混进分母会让
    tok/s 随生成长度漂移，也就没法跟 llama.cpp 的 eval time 对齐。
    """
    model.set_shards(shards)
    model.reset()
    var steps = List[Int]()
    var logits = model.prefill[backend](tok.encode(PROMPT))
    var next = model.argmax(logits)
    for _ in range(n_new):
        var t0 = monotonic_ns()
        var lg = model.step[backend](next)
        steps.append(monotonic_ns() - t0)
        next = model.argmax(lg)
    return steps^


def ns_of(imm v: List[Int], arm: Int, pick_min: Bool) -> Int:
    var best = v[arm]
    for rep in range(1, ROUNDS):
        var x = v[rep * ARMS + arm]
        if pick_min:
            if x < best:
                best = x
        elif x > best:
            best = x
    return best


def sample_min(r: List[Float64]) -> Float64:
    var v = r[0]
    for i in range(1, len(r)):
        if r[i] < v:
            v = r[i]
    return v


def sample_max(r: List[Float64]) -> Float64:
    var v = r[0]
    for i in range(1, len(r)):
        if r[i] > v:
            v = r[i]
    return v


def span_text(r: List[Float64]) -> String:
    return f2(sample_min(r)) + "× … " + f2(sample_max(r)) + "×"


def direction(r: List[Float64]) -> Int:
    """+1 = 整串在 1.0 之上（分片更快）；-1 = 整串在 1.0 之下；0 = 含 1.0。"""
    var lo = sample_min(r)
    var hi = sample_max(r)
    if lo <= Float64(0):
        return 0
    if lo > Float64(1):
        return 1
    if hi < Float64(1):
        return -1
    return 0


def combined(p50: List[Float64], mean: List[Float64]) -> String:
    """两个口径**同向且都排除 1.0** 才给结论。默认值是"量不出结论"。"""
    var a = direction(p50)
    var b = direction(mean)
    if a != 0 and a == b:
        if a > 0:
            return "两个口径同向 → 分片更快"
        return "两个口径同向 → 分片更慢"
    if a != 0 or b != 0:
        return "两个口径不同向 → 量不出结论（不许取好看的那个）"
    return "两个口径都含 1.0 → 量不出结论"


def unpaired(imm med: List[Int], arm: Int) -> String:
    """保守对照：各档自己的 min…max，按相反方向相除。"""
    var base_lo = Float64(ns_of(med, 0, True))
    var base_hi = Float64(ns_of(med, 0, False))
    var arm_lo = Float64(ns_of(med, arm, True))
    var arm_hi = Float64(ns_of(med, arm, False))
    var text = f2(base_lo / arm_hi) + "× … " + f2(base_hi / arm_lo) + "×   区间"
    if arm_hi < base_lo:
        return text + "不重合 → 分片更快"
    if arm_lo > base_hi:
        return text + "不重合 → 分片更慢"
    return text + "重合 → 这份数据区分不出两者"


def main() raises:
    print("")
    print(
        "== 端到端 decode：投影切成 N 片并发（Qwen2.5-0.5B fp32，"
        + "avx2，同一进程同一份权重） =="
    )
    var t0 = monotonic_ns()
    var model = QwenForward(FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK)
    print(
        "   权重加载 : "
        + String((monotonic_ns() - t0) // 1000000)
        + " ms（mmap，惰性；真正的读页在预热轮里）"
    )
    var tok = load_tokenizer(FIXTURE)

    print("   预热轮（" + String(WARMUP_NEW) + " 个 token，结果丢弃）")
    for k in range(ARMS):
        _ = run_once[BACKEND_AVX2](model, tok, WARMUP_NEW, shard_of(k))

    var med = List[Int]()
    var tot = List[Int]()
    for _ in range(ROUNDS * ARMS):
        med.append(0)
        tot.append(0)
    var runq = List[Int]()
    for _ in range(ROUNDS):
        runq.append(0)

    for rep in range(ROUNDS):
        print("   -- 第 " + String(rep + 1) + " 轮")
        for k in range(ARMS):
            var arm = (rep + k) % ARMS
            var q0 = runq_len()
            var steps = run_once[BACKEND_AVX2](model, tok, NEW, shard_of(arm))
            var q1 = runq_len()
            if q0 > runq[rep]:
                runq[rep] = q0
            if q1 > runq[rep]:
                runq[rep] = q1
            med[rep * ARMS + arm] = median_ns(steps)
            tot[rep * ARMS + arm] = sum_ns(steps)
        print("      runq " + String(runq[rep]) + "（本机 8 核，常年 6–27）")
        for arm in range(ARMS):
            var m = med[rep * ARMS + arm]
            print(
                "      shards="
                + shard_label(arm)
                + " : "
                + f2(Float64(m) / 1000000.0)
                + " ms/token   "
                + f2(1000000000.0 / Float64(m))
                + " tok/s"
            )

    print("")
    print("   同轮配对比值（主口径：不切 ÷ 切片，同轮、同进程）")
    for arm in range(1, ARMS):
        var p50 = List[Float64]()
        var mean = List[Float64]()
        for rep in range(ROUNDS):
            p50.append(
                Float64(med[rep * ARMS]) / Float64(med[rep * ARMS + arm])
            )
            mean.append(
                Float64(tot[rep * ARMS]) / Float64(tot[rep * ARMS + arm])
            )
        print("      shards=" + shard_label(arm))
        print("          中位数口径 : " + span_text(p50))
        print("          均值口径   : " + span_text(mean))
        print("          → " + combined(p50, mean))

    print("")
    print("   非配对区间（各档自己的 min…max）—— 保守对照")
    for arm in range(1, ARMS):
        print("      shards=" + shard_label(arm) + " : " + unpaired(med, arm))
    print("")
