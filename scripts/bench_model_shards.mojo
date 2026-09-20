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

最后一段：离天花板还有多远（**决定下一个杠杆**）
------------------------------------------------
"分片快 1.7×，而 llama.cpp 同机是 4.24×" 差的那一截在哪？两个候选互斥：

  * **串行段**（注意力 / RMSNorm / RoPE / 残差一次都没被切）+ 每 token 169 × N
    次协程调度 → 那该去切它们；
  * **已经贴到天花板**（聚合读带宽就这么多）→ 切串行段**一无所获**，该去改访存
    模式或转向批。

区分它们只需要一个比：**每 token 实测带宽 ÷ 天花板**。天花板**必须在同一进程
现测**（本机同脚本不同时刻的读带宽 12.0–15.0 GB/s 会漂；`bench_thread_bandwidth`
那个 22.69 GB/s 还是"累加器链"的顶），所以这里现测两个：

  * **B_T（纯读天花板）**：T 个任务扫 256 MB，每任务 **8 个独立 f64 累加器**
    （一个累加器时顶是 25.6 GB/s，那是链的顶不是 DRAM 的顶）。
  * **K_T（kernel 天花板）**：用**真实** `linear`（avx2）在 8 份互不相交的权重
    （139 MB ≫ L3 12 MB）上各算一趟 —— 它含 kernel 自己所有的低效，是"如果整
    个 token 都是这种投影"能到的最好成绩。

判定阈值**在看数之前定死**：对 K_T 的利用率 **≥ 900‰ = 贴顶**（串行段没油水）、
**≤ 700‰ = 明显没贴顶**（时间里有 ≥30% 不在投影上）。中间地带就写"量不出结论"。

跑法：

    pixi run mojo run -O2 -I src scripts/bench_model_shards.mojo
"""

from std.runtime.asyncrt import TaskGroup

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import RawPtr, TensorView, f32_data, shape2
from alofa.core.text import parse_int, read_text
from alofa.kernels.cpu.avx2 import linear
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

# 每 token 要流的权重字节数。`cli.mojo` 的 `BYTES_PER_TOKEN_FP32`：权重文件的
# 字节数 1,976,393,216 B，不是估算（`lm_head` 与 `embed_tokens` 绑定共用一份）。
comptime BYTES_PER_TOKEN = 1.976e9

# 纯读天花板的数组：256 MB ≫ L3（12 MB）。
comptime CEIL_BYTES = 256 * 1024 * 1024
comptime CEIL_ELEMS = CEIL_BYTES // 4
comptime CEIL_ROUNDS = 3

# kernel 天花板：`down_proj` 896 × 4864，8 份互不相交的权重 = 139 MB ≫ L3。
comptime K_COLS = 896
comptime K_INNER = 4864
comptime K_REGIONS = 8
comptime K_W_BYTES = K_COLS * K_INNER * 4


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


def read_seq(p: RawPtr, n: Int, sink: RawPtr, slot: Int) raises -> None:
    """扫一遍 `[p, p+n)`，8 个**独立**的 f64 累加器。

    ⚠️ 必须是 8 个独立的而不是一个：`add` 延迟 4 周期 → 单累加器时每 4 周期才
    8 个元素 = **25.6 GB/s** 的顶，那是"累加器链"的顶，不是 DRAM 的顶
    （`bench_thread_bandwidth.mojo` 报的 22.69 GB/s 就是这么来的）。
    """
    var pd = f32_data(TensorView(p, shape2(1, n), DT_FP32))
    var sd = f32_data(TensorView(sink, shape2(1, 16), DT_FP32))
    var a0: Float64 = 0
    var a1: Float64 = 0
    var a2: Float64 = 0
    var a3: Float64 = 0
    var a4: Float64 = 0
    var a5: Float64 = 0
    var a6: Float64 = 0
    var a7: Float64 = 0
    var i = 0
    while i + 8 <= n:
        a0 += Float64(pd[unsafe_offset=i])
        a1 += Float64(pd[unsafe_offset=i + 1])
        a2 += Float64(pd[unsafe_offset=i + 2])
        a3 += Float64(pd[unsafe_offset=i + 3])
        a4 += Float64(pd[unsafe_offset=i + 4])
        a5 += Float64(pd[unsafe_offset=i + 5])
        a6 += Float64(pd[unsafe_offset=i + 6])
        a7 += Float64(pd[unsafe_offset=i + 7])
        i += 8
    while i < n:
        a0 += Float64(pd[unsafe_offset=i])
        i += 1
    # 写一个槽位：这段读不能被当成可消除的死代码。
    sd[unsafe_offset=slot] = Float32(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)


async def read_slice(p: RawPtr, n: Int, sink: RawPtr, slot: Int) -> None:
    """`read_seq` 的协程包装。

    ⚠️ 协程参数只能是平凡值（指针 / 整数），视图在协程**内**现造 —— `TensorView`
    直接当协程参数实测会**静默写错地方**。
    """
    try:
        read_seq(p, n, sink, slot)
    except err:
        _ = err


def slice_len(t: Int, k: Int, per: Int, total: Int) -> Int:
    if k == t - 1:
        return total - per * (t - 1)
    return per


def sweep_read(t: Int, base: RawPtr, sink: RawPtr) raises -> Int:
    """把 256 MB 切成 `t` 段并发扫一遍，返回纳秒；**自检失败返回 -1**。

    ⚠️ `t == 1` 走**直呼**而不是"一个任务的 TaskGroup"：基线档必须与模型里
    `shards <= 1` 那条路**同一形态**，否则量到的是"换调度机制的收益"。

    ⚠️ **必须自检**：数组填的是全 1，所以第 k 片累加出来必须**精确等于**该片的
    元素数。少算了任何一片（或者协程参数被写坏、那片压根没跑），这里就会红 ——
    没有这一步，"天花板"可以报出 56 GB/s 这种**超过 DDR4 双通道上限**的数而不
    自知（2026-09-20 实测：先报了 56.56 GB/s，本机理论上限只有 ~42 GB/s）。
    """
    var sd = f32_data(TensorView(sink, shape2(1, 16), DT_FP32))
    for k in range(16):
        sd[unsafe_offset=k] = Float32(0)
    var t0 = monotonic_ns()
    if t == 1:
        read_seq(base, CEIL_ELEMS, sink, 0)
        var dt = monotonic_ns() - t0
        if sd[unsafe_offset=0] != Float32(CEIL_ELEMS):
            return -1
        return dt
    var per = (CEIL_ELEMS // t) // 8 * 8
    var tg = TaskGroup()
    for k in range(t):
        tg.create_task(
            read_slice(base.unsafe_offset(k * per * 4), slice_len(t, k, per, CEIL_ELEMS), sink, k)
        )
    tg.wait()
    var dt = monotonic_ns() - t0
    for k in range(t):
        if sd[unsafe_offset=k] != Float32(slice_len(t, k, per, CEIL_ELEMS)):
            return -1
    return dt


def ceiling_read(t: Int) raises -> Float64:
    """T 个并发任务纯读能到多少字节/秒（3 趟取**中位数**，不取最优）。

    自检失败返回 **-1**：那个数不许进判定，也**不许**被引用。
    """
    var arena = Arena(CEIL_BYTES + 64)
    var base = arena.alloc(CEIL_BYTES)
    var sink = arena.alloc(64)
    var pv = f32_data(TensorView(base, shape2(1, CEIL_ELEMS), DT_FP32))
    for i in range(CEIL_ELEMS):
        pv[unsafe_offset=i] = Float32(1)
    # 预热：第一趟要付匿名页的缺页，那不是带宽。
    _ = sweep_read(t, base, sink)
    var times = List[Int]()
    var ok = True
    for _ in range(CEIL_ROUNDS):
        var dt = sweep_read(t, base, sink)
        if dt < 0:
            ok = False
        else:
            times.append(dt)
    _ = f32_data(TensorView(sink, shape2(1, 16), DT_FP32))[unsafe_offset=0]
    arena.keep_alive()
    if not ok or len(times) == 0:
        return -1.0
    return Float64(CEIL_BYTES) * 1000000000.0 / Float64(median_ns(times))


def gemv_seq(d: RawPtr, x: RawPtr, w: RawPtr, cols: Int, inner: Int) raises -> None:
    """一片输出列上的真实 `linear`（avx2）。"""
    linear(
        TensorView(d, shape2(1, cols), DT_FP32),
        TensorView(x, shape2(1, inner), DT_FP32),
        TensorView(w, shape2(cols, inner), DT_FP32),
    )


async def gemv_slice(d: RawPtr, x: RawPtr, w: RawPtr, cols: Int, inner: Int) -> None:
    """`gemv_seq` 的协程包装；参数同样只收平凡值。"""
    try:
        gemv_seq(d, x, w, cols, inner)
    except err:
        _ = err


def sweep_gemv(t: Int, w_raw: RawPtr, x_raw: RawPtr, d_raw: RawPtr) raises -> Int:
    """8 份互不相交的权重各算一趟，一趟内把 896 个输出切成 `t` 片。

    返回纳秒；**自检失败返回 -1**。`t == 1` 走**直呼**（同上：基线档必须与
    `shards <= 1` 那条路同形态）。

    ⚠️ 自检：先把 `d` 涂成哨兵 -999，权重与 `x` 都是全 1 → 每个输出**精确等于**
    `inner`。既查"算错了"也查"压根没算"（哨兵还在）。
    """
    var dv = f32_data(TensorView(d_raw, shape2(1, K_COLS), DT_FP32))
    for c in range(K_COLS):
        dv[unsafe_offset=c] = Float32(-999)
    var per = K_COLS // t
    var rem = K_COLS - per * t
    var t0 = monotonic_ns()
    for r in range(K_REGIONS):
        var wbase = w_raw.unsafe_offset(r * K_W_BYTES)
        if t == 1:
            gemv_seq(d_raw, x_raw, wbase, K_COLS, K_INNER)
            continue
        var tg = TaskGroup()
        var c0 = 0
        for k in range(t):
            var n = per + (rem if k == t - 1 else 0)
            tg.create_task(
                gemv_slice(
                    d_raw.unsafe_offset(c0 * 4),
                    x_raw,
                    wbase.unsafe_offset(c0 * K_INNER * 4),
                    n,
                    K_INNER,
                )
            )
            c0 += n
        tg.wait()
    var dt = monotonic_ns() - t0
    for c in range(K_COLS):
        if dv[unsafe_offset=c] != Float32(K_INNER):
            return -1
    return dt


def ceiling_kernel(t: Int) raises -> Float64:
    """T 个并发任务跑真实 GEMV 能到多少字节/秒；自检失败返回 **-1**。

    它含 kernel 自己所有的低效（访存模式、累加器、形状），所以是"如果整个
    token 都是这种投影"能到的**最好成绩** —— 用它当分母，剩下的才是串行段。
    """
    var arena = Arena(K_REGIONS * K_W_BYTES + K_INNER * 4 + K_COLS * 4 + 4096)
    var w_raw = arena.alloc(K_REGIONS * K_W_BYTES)
    var x_raw = arena.alloc(K_INNER * 4)
    var d_raw = arena.alloc(K_COLS * 4)
    var xv = f32_data(TensorView(x_raw, shape2(1, K_INNER), DT_FP32))
    for i in range(K_INNER):
        xv[unsafe_offset=i] = Float32(1)
    var wv = f32_data(
        TensorView(w_raw, shape2(1, K_REGIONS * K_COLS * K_INNER), DT_FP32)
    )
    for i in range(K_REGIONS * K_COLS * K_INNER):
        wv[unsafe_offset=i] = Float32(1)
    # ⚠️ 139 MB ≫ L3（12 MB）是必须的 —— 小于 L3 时重复的几趟会有一部分不走
    # DRAM，报出物理不可能的数（见变更日志 2026-09-20）。
    _ = sweep_gemv(t, w_raw, x_raw, d_raw)
    var times = List[Int]()
    var ok = True
    for _ in range(CEIL_ROUNDS):
        var dt = sweep_gemv(t, w_raw, x_raw, d_raw)
        if dt < 0:
            ok = False
        else:
            times.append(dt)
    _ = f32_data(TensorView(d_raw, shape2(1, K_COLS), DT_FP32))[unsafe_offset=0]
    arena.keep_alive()
    if not ok or len(times) == 0:
        return -1.0
    return Float64(K_REGIONS * K_W_BYTES) * 1000000000.0 / Float64(median_ns(times))


def gbps_text(lo: Float64, hi: Float64) -> String:
    return f2(lo / 1000000000.0) + "–" + f2(hi / 1000000000.0) + " GB/s"


def permille(r: Float64) -> String:
    return String(Int(r * 1000.0)) + "‰"


def median_f64(v: List[Float64]) -> Float64:
    var s = List[Float64]()
    for i in range(len(v)):
        s.append(v[i])
    var i = 1
    while i < len(s):
        var x = s[i]
        var j = i - 1
        while j >= 0 and s[j] > x:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = x
        i += 1
    return s[len(s) // 2]


def gbps_or_fail(v: Float64) -> String:
    if v <= Float64(0):
        return "自检失败（这一趟不许进判定）"
    return f2(v / 1000000000.0) + " GB/s"


def ceiling_verdict(lo: Float64, hi: Float64) -> String:
    """阈值在 docstring 里先定死：≥900‰ 贴顶、≤700‰ 明显没贴顶。"""
    if hi < 0.7:
        return (
            "离 kernel 天花板还有 ≥30% → 时间里有这么多不在投影上（串行段 / 调度），"
            "切串行段**有油水**"
        )
    if lo > 0.9:
        return "已贴到 kernel 天花板（≥90%）→ 串行段没油水，别去切注意力 / norms"
    return "量不出结论（区间跨过 700‰ 与 900‰）"


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

    # ⚠️ 天花板必须和 token 时间**同一时间窗**：本机读带宽跨运行能差 1.6×
    # （实测 B_1 一次 12.97 一次 8.09 GB/s），先跑完 3 轮再测天花板的话，机器
    # 变慢会让天花板偏低、利用率虚高（会算出 >1000‰ 这种不可能的数）。
    var b1s = List[Float64]()
    var b8s = List[Float64]()
    var k1s = List[Float64]()
    var k8s = List[Float64]()
    var ceil_ok = True

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
        # 同一轮里现测四个天花板（各 <1 s，与上面的 token 时间同窗口）。
        var v1 = ceiling_read(1)
        var v8 = ceiling_read(8)
        var g1 = ceiling_kernel(1)
        var g8 = ceiling_kernel(8)
        if v1 <= Float64(0) or v8 <= Float64(0):
            ceil_ok = False
        else:
            b1s.append(v1)
            b8s.append(v8)
        if g1 <= Float64(0) or g8 <= Float64(0):
            ceil_ok = False
        else:
            k1s.append(g1)
            k8s.append(g8)

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
    print(
        "   同进程现测的两个天花板（⚠️ 分母必须同进程现测：本机读带宽同脚本不同"
        "时刻 12.0–15.0 GB/s 会漂）"
    )
    if not ceil_ok or len(b1s) == 0:
        print(
            "      天花板自检失败（有一趟没算全）→ **不报任何利用率**：一个假的"
            "分母比没有分母更害人"
        )
        print("")
        return
    var b1 = median_f64(b1s)
    var b8 = median_f64(b8s)
    var k1 = median_f64(k1s)
    var k8 = median_f64(k8s)
    print(
        "      纯读 B_1 = "
        + gbps_or_fail(b1)
        + "    B_8 = "
        + gbps_or_fail(b8)
        + "   （8 个独立 f64 累加器，不是累加器链）"
    )
    print(
        "      kernel K_1 = "
        + gbps_or_fail(k1)
        + "    K_8 = "
        + gbps_or_fail(k8)
        + "   （真实 `linear`，8 份互不相交权重 139 MB ≫ L3）"
    )
    print(
        "      四个天花板都是**每轮测一次取中位数**，与 token 时间同一时间窗；"
        "跨运行它们能差 1.6×，所以只在同一进程内用"
    )

    print("")
    print("   每 token 实测带宽与利用率（每 token 流 1.976 GB = 权重文件字节数）")
    var focus = List[Int]()
    focus.append(0)
    focus.append(3)
    for j in range(2):
        var arm = focus[j]
        var sh = shard_of(arm)
        var b = b1 if sh == 1 else b8
        var k = k1 if sh == 1 else k8
        var t_lo = Float64(ns_of(med, arm, True))
        var t_hi = Float64(ns_of(med, arm, False))
        var g_lo = BYTES_PER_TOKEN * 1000000000.0 / t_hi
        var g_hi = BYTES_PER_TOKEN * 1000000000.0 / t_lo
        print("      shards=" + shard_label(arm) + " : " + gbps_text(g_lo, g_hi))
        if b <= Float64(0) or k <= Float64(0):
            print(
                "          天花板自检失败 → **不报利用率**（一个假的分母比没有"
                "分母更害人）"
            )
            continue
        print(
            "          对纯读天花板 : "
            + permille(g_lo / b)
            + " – "
            + permille(g_hi / b)
            + "   （kernel 自己离 DRAM 顶有多远）"
        )
        print(
            "          对 kernel 天花板 : "
            + permille(g_lo / k)
            + " – "
            + permille(g_hi / k)
            + "   （剩下的就是串行段 + 调度）"
        )
        print("          → " + ceiling_verdict(g_lo / k, g_hi / k))
    print("")
