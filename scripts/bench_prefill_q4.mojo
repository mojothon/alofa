"""端到端量 prefill：接进模型层的 q4 行复用（`q4_matmul_rows_k_shards`）到底有没有换到时间。

口径（跑之前定死）：
  - 后端统一 `BACKEND_AVX2`（标量后端走的是逐行兜底，拿不到复用）；
  - 每个长度预热 1 趟丢弃，之后 5 趟取 min…max；
  - 本机长期过载 → 数字只在**同一次会话内的 A/B** 里比，不与别处的数字比。

用法：`mojo build -O2 -I src scripts/bench_prefill_q4.mojo -o /tmp/x && /tmp/x`
"""

from alofa.core.ffi import monotonic_ns
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward

comptime WEIGHTS_DIR = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime WINDOW = 512
comptime ROUNDS = 5
comptime LENGTHS = 3


def nth_length(i: Int) -> Int:
    var table = List[Int]()
    table.append(16)
    table.append(32)
    table.append(64)
    return table[i]


def make_ids(n: Int) -> List[Int]:
    var out = List[Int]()
    var i = 0
    while i < n:
        out.append((i * 7 + 11) % 40000)
        i += 1
    return out^


def interval(ns: List[Int]) -> String:
    var lo = ns[0]
    var hi = ns[0]
    var i = 0
    while i < len(ns):
        var v = ns[i]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        i += 1
    return (
        String(Float64(lo) / Float64(1e6))
        + "–"
        + String(Float64(hi) / Float64(1e6))
        + " ms"
    )


def time_prefill(mut model: QwenForward, n: Int) raises -> String:
    var ts = List[Int]()
    var k = 0
    while k < ROUNDS + 1:
        model.reset()
        var start = monotonic_ns()
        _ = model.prefill[BACKEND_AVX2](make_ids(n))
        var end = monotonic_ns()
        if k > 0:
            ts.append(end - start)
        k += 1
    return interval(ts^)


def main() raises:
    var model = QwenForward(WEIGHTS_DIR, CONFIG, WINDOW, True)
    print("q4 通路 =", model.q4_enabled, " 分片 =", model.shards, " 窗口 =", WINDOW)
    print("每个长度先预热 1 趟丢弃，再取", ROUNDS, "趟的 min…max：")
    var i = 0
    while i < LENGTHS:
        var n = nth_length(i)
        print("  prefill 长度", n, " → ", time_prefill(model, n))
        i += 1
    model.keep_alive()
