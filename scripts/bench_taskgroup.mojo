"""一次 `TaskGroup` 的空转开销有多大 —— 乘上 169 次投影，看它配不配当瓶颈。

为什么突然要量这个
----------------
`bench_head_split.mojo` 把一次 decode 切成两半（真实 0.5B、q4_0 层 + q8_0 输出投影、
backend avx2、`shards=8`）：

        step 88.18 ms  =  输出投影 17.39 ms (19.7%)  +  其余 70.79 ms (80.3%)

也就是说**输出投影已经不是瓶颈**，把它清零也只能从 11.34 提到 14.13 tok/s。剩下的
80% 里藏着什么？一个候选是分片本身：一次 decode 要走 **169 次投影**，每次投影都建
一个 `TaskGroup`、往里塞 `shards` 个协程、然后 `wait` —— 一个 token 就是
`169 × shards` 次协程调度。`bench_model_shards.mojo` 的文档早就写了这件事，但**从
没量过它值多少毫秒**。

这里量的是**空转**：N 个什么都不做的协程 + 一次 `wait`。它是调度开销的**上界估计**
—— 真实的协程里有计算，能盖掉一部分，但盖不掉 `wait` 那次同步。把它乘 169 之后如果
落在几十 ms 量级，那"其余"里的大头就是它，而不是 q4 核的算术。

⚠️ 边界（不可省略）
* 空协程 ≠ 真实协程：真实的那 169 次里，每片还要读自己的那一段权重，调度与访存会
  部分重叠 ⇒ **×169 是上界**，不是预测值。
* 它不能回答"该不该分片"——那要看 `bench_model_shards.mojo` 的端到端 A/B。这里只回
  答"分片的固定成本是多少"。
* 本机长期过载 ⇒ 取**中位数**，且绝对值只作同轮内部比较。

跑法（`-O2`：`-O0` 下的协程调度没有参考价值）
--------------------------------------------
    pixi run mojo build -O2 -I src scripts/bench_taskgroup.mojo -o target/bench_taskgroup
    ./target/bench_taskgroup
"""

from std.runtime.asyncrt import TaskGroup, parallelism_level

from alofa.core.ffi import monotonic_ns

comptime REPS = 200
comptime WARMUP = 50
# 一个 token 里的投影次数：24 层 × 7（q/k/v/o/gate/up/down）+ 输出投影。
comptime PROJECTIONS = 169


def median_ns(mut steps: List[Int]) -> Int:
    """`steps` 的中位数（就地插入排序；样本小，够用）。"""
    var i = 1
    while i < len(steps):
        var v = steps[i]
        var j = i - 1
        while j >= 0 and steps[j] > v:
            steps[j + 1] = steps[j]
            j -= 1
        steps[j + 1] = v
        i += 1
    return steps[len(steps) // 2]


def f2(x: Float64) -> String:
    """两位小数。本机没有 sprintf，整数拼是唯一不引入新依赖的写法。"""
    var v = x
    var whole = Int(v)
    var frac = Int((v - Float64(whole)) * Float64(100) + Float64(0.5))
    if frac >= 100:
        whole += 1
        frac -= 100
    var s = String(whole) + "."
    if frac < 10:
        s = s + "0"
    return s + String(frac)


async def _noop():
    """什么都不做的协程：留给调度自己的那一段。"""
    return


def bench_one(shards: Int) -> Int:
    """N 个空协程 + 一次 `wait` 的中位耗时（ns）。"""
    for _ in range(WARMUP):
        var tg0 = TaskGroup()
        for k in range(shards):
            tg0.create_task(_noop())
        tg0.wait()
    var times = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        var tg = TaskGroup()
        for k in range(shards):
            tg.create_task(_noop())
        tg.wait()
        times.append(monotonic_ns() - t0)
    return median_ns(times)


def main() raises:
    print("")
    print("== 一次 TaskGroup 空转的开销 ==")
    print("   parallelism_level=" + String(parallelism_level()))
    print("   一个 token 走 " + String(PROJECTIONS) + " 次投影 ⇒ 调度次数 × " + String(PROJECTIONS))

    var arms = List[Int]()
    arms.append(1)
    arms.append(2)
    arms.append(4)
    arms.append(8)

    for a in range(len(arms)):
        var shards = arms[a]
        var ns = bench_one(shards)
        var per_us = Float64(ns) / 1000.0
        var per_token_ms = Float64(ns) * Float64(PROJECTIONS) / 1000000.0
        print(
            "[tg] shards="
            + String(shards)
            + " : 单次 "
            + f2(per_us)
            + " μs  ⇒  ×"
            + String(PROJECTIONS)
            + " = "
            + f2(per_token_ms)
            + " ms/token（上界；真实协程有计算可盖掉一部分）"
        )

    print("")
    print("   对照 `bench_head_split.mojo`（同机、q8_0 输出投影档，shards=8）：")
    print("     step 88.18 ms = 输出投影 17.39 ms + 其余 70.79 ms")
    print("   若上面的 shards=8 那行落在几十 ms，则「其余」里的大头就是调度，")
    print("   而不是 q4 核的算术 —— 下一步该动的是**分片的粒度**，不是核。")
    print("")
