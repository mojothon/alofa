"""decode step 端到端计时（设计 06-device-context.md 步骤 4，账本六十四）。

判据：单流 decode ≤ 2ms/token（账本五十八的 1.4ms kernel 级估计 + 传输与
launch 余量）。计时口径：**decode_step 整步**（含 token id 上传、每步 2 次
sync、logits 回传 —— 这些都是设计内的步进成本），WARMUP 后分 ROUNDS 轮、
每轮 ITERS 步取一轮均值、轮间取最小 —— 与 bench_q4_matmul 同方法论。

MAXT 必须 ≥ WARMUP + ITERS×ROUNDS + 1：kv_pos 每步前进，store 满了会抛。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime WARMUP = 5
comptime ITERS = 100
comptime ROUNDS = 3
comptime MAXT = WARMUP + ITERS * ROUNDS + 8


def main() raises:
    var t0 = monotonic_ns()
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var t1 = monotonic_ns()
    print(
        "model load+quantize: ",
        Float64(t1 - t0) / 1e9,
        " s (one-time, CPU side)",
    )
    var session = GpuSession(model, MAXT)
    var t2 = monotonic_ns()
    print(
        "gpu upload+alloc: ",
        Float64(t2 - t1) / 1e9,
        " s (one-time; ~550 MB weights + embedding)",
    )

    for i in range(WARMUP):
        _ = session.decode_step[INNER, INNER_FF](TOKEN)
    session.ctx.synchronize()

    var best = Float64(1e30)
    var worst_round = Float64(0)
    for r in range(ROUNDS):
        var s0 = monotonic_ns()
        for i in range(ITERS):
            _ = session.decode_step[INNER, INNER_FF](TOKEN)
        var ms = Float64(monotonic_ns() - s0) / 1e6 / Float64(ITERS)
        if ms < best:
            best = ms
        if ms > worst_round:
            worst_round = ms

    print("decode_step end-to-end: ", best, " ms/token (best of", ROUNDS, " rounds)")
    print("worst round: ", worst_round, " ms/token")
    print("throughput: ", 1000.0 / best, " tokens/s")
    print(
        "vs kernel-level estimate 1.4 ms (ledger 58) and criterion 2.0 ms/token"
    )
    if best <= 2.0:
        print("*** DECODE BENCH PASS *** ", best, " ms/token <= 2.0")
    else:
        print("*** DECODE BENCH MISS *** ", best, " ms/token > 2.0")
    model.params.keep_alive()
