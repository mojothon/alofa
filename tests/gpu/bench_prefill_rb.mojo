"""RB 扫（账本八十九，同核 A/B）：prefill_step 的 prefill_rb ∈ {1, 8, 16}。

rb=1 即八十八的 batched-GEMV 行为（每 block 一行，权重每行重读）——
同核同卡口径下的 GEMV 基线；rb=8/16 为真 GEMM 分块。位序：rb1 →
rb8 → rb16 各 1 rep（kv 累计 384）→ 128 步 decode 基线（max_tokens
=512，恰在 cos 表 fixture 上限内）。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128


def _bench[rb: Int, AF: DType](mut session: GpuSession, ids: List[Int]) raises:
    for rep in range(1):
        var t0 = monotonic_ns()
        _ = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, rb, AF](ids)
        session.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        print("rb=", rb, " af=", AF, " rep=", rep, ": ", ms, " ms")


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var session = GpuSession(model, 512)

    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)

    _bench[1, DType.float64](session, ids)
    _bench[8, DType.float64](session, ids)
    _bench[8, DType.float32](session, ids)

    var total = Float64(0)
    var best = 1e30
    for i in range(P):
        var t0 = monotonic_ns()
        _ = session.decode_step[INNER, INNER_FF](2000 + i)
        var ms = Float64(monotonic_ns() - t0) / 1e6
        total += ms
        if ms < best:
            best = ms
    print("decode ", P, " x1: mean=", total / Float64(P), " ms  best=", best, " ms")
    model.params.keep_alive()
