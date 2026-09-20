"""端到端 prefill：一个 token 的权重字节，批起来能不能摊薄。

问的是什么
----------
`avx2._gemm` 原来是「行外层、列内层」——`rows` 行的前向把整个权重矩阵读 `rows`
遍，于是 `cli.mojo` 里那条「每 token 流 1.976 GB」在批 = 8 和批 = 1 上**是同一个
数**。改成「`RB` 行一块、列外层」之后，权重每块只读一遍，每 token 字节数应该
除以约 `RB`。

核级证据在 `scripts/bench_gemm_rows.mojo`（`rows=8` 一趟 23.31 → 3.51 ms）；
这份量的是**真的整段 prefill**：加载真实 0.5B fp32 权重，一次前向里还有注意力、
RMSNorm、RoPE、残差，它们不吃这份红利 —— 所以这里的倍数**必然小于**核级那个，
它回答的是"端到端还剩多少"。

口径
----
  * **必须 `-O2`**；预热一趟丢弃（它量的是 mmap 缺页，不是前向）。
  * 3 趟取 **min…max 区间**；主统计量是**每 token 时间**（总时间 ÷ `n` token）。
  * 判据是**每 token 等效带宽** = `1.976 GB ÷ 每 token 时间`：
    它超过同进程现测的 `T=1` 读带宽 → 权重确实被复用了（因为一份字节不可能
    以高于 DRAM 的速度进来）。它是一个**等效**量，不是真带宽 —— 真带宽得靠
    计数器，本机 `perf_event_paranoid=4`，拿不到。
  * 自检：三趟的 argmax 与前 16 个 logit 必须**逐位相等**（同一份权重、同一段
    id，前向是确定的）。不等说明有片没算或写坏了 —— 那会让时间偏快。

为什么不用分词器
----------------
`prefill` 吃的是 token id 列表，不是字符串；直接给 id 就少一个解释变量
（分词本身不进计时，但加载词表要 100 MB 级的内存与几百毫秒）。

跑法：

    pixi run mojo run -O2 -I src scripts/bench_prefill.mojo
"""

from alofa.core.ffi import monotonic_ns
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.dtype import DT_FP32
from alofa.core.tensor import TensorView, f32_data, shape2
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"

# prompt 长度：8 和 32。`MAX_PROMPT` 是 128，模型的历史上限更大。
comptime N_CASES = 2
comptime MAX_TOK = 128
comptime ROUNDS = 3

# 每 token 要流的权重字节（权重文件 1,976,393,216 B，不是估算）。
comptime BYTES_PER_TOKEN = 1.976e9


def token_ids(n: Int) -> List[Int]:
    """一段确定的 id；都远小于词表 151936，形状之外不引入别的变量。"""
    var v = List[Int]()
    for i in range(n):
        v.append(1000 + i * 7)
    return v^


def prefill_once(mut model: QwenForward, n: Int, sink: RawPtr) raises -> Int:
    """跑一趟 `n` 个 token 的 prefill，返回纳秒；把**指纹**写进 `sink`。

    指纹 = argmax 与前 16 个 logit 的和。同一份权重 + 同一段 id，前向是确定的：
    三趟指纹不等就说明有片没算（那会让时间偏快，把结论引向相反方向）。
    """
    model.reset()
    var t0 = monotonic_ns()
    var logits = model.prefill[BACKEND_AVX2](token_ids(n))
    var dt = monotonic_ns() - t0
    var pv = logits
    var total = Float32(0)
    for i in range(16):
        total += pv[unsafe_offset=i]
    var pd = f32_data(TensorView(sink, shape2(1, 4), DT_FP32))
    pd[unsafe_offset=0] = Float32(model.argmax(logits))
    pd[unsafe_offset=1] = total
    return dt


def main() raises:
    print("")
    print("== 端到端 prefill：每 token 的权重字节能不能被批摊薄 ==")
    var model = QwenForward(FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK)
    var arena = Arena(64)
    var sink = arena.alloc(64)

    var ns = List[Int]()
    var fp = List[Float32]()
    for _ in range(N_CASES * ROUNDS):
        ns.append(0)
        fp.append(Float32(0))

    var sizes = List[Int]()
    sizes.append(8)
    sizes.append(32)

    for c in range(N_CASES):
        var n = sizes[c]
        # 预热：第一趟要付 mmap 的缺页，那不是前向。
        _ = prefill_once(model, n, sink)
        for r in range(ROUNDS):
            ns[c * ROUNDS + r] = prefill_once(model, n, sink)
            fp[c * ROUNDS + r] = f32_data(
                TensorView(sink, shape2(1, 4), DT_FP32)
            )[unsafe_offset=1] + f32_data(TensorView(sink, shape2(1, 4), DT_FP32))[
                unsafe_offset=0
            ]
        var lo = ns[c * ROUNDS]
        var hi = ns[c * ROUNDS]
        for r in range(1, ROUNDS):
            if ns[c * ROUNDS + r] < lo:
                lo = ns[c * ROUNDS + r]
            if ns[c * ROUNDS + r] > hi:
                hi = ns[c * ROUNDS + r]
        var same = True
        for r in range(1, ROUNDS):
            if fp[c * ROUNDS + r] != fp[c * ROUNDS]:
                same = False
        var per_lo = Float64(lo) / Float64(n)
        var per_hi = Float64(hi) / Float64(n)
        print(
            "  n =",
            n,
            "  一趟",
            Float64(lo) / 1e6,
            "…",
            Float64(hi) / 1e6,
            "ms   每 token",
            per_lo / 1e6,
            "…",
            per_hi / 1e6,
            "ms",
        )
        # 1 GB/s = 1e9 B / 1e9 ns = 1 B/ns → 字节数 ÷ 纳秒 直接就是 GB/s。
        print(
            "         每 token 等效带宽",
            BYTES_PER_TOKEN / per_hi,
            "…",
            BYTES_PER_TOKEN / per_lo,
            "GB/s（1.976 GB ÷ 每 token 时间）   自检",
            "通过" if same else "失败 → 这格的数作废",
        )

    arena.keep_alive()
