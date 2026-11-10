"""prefill TTFT 基线（账本八十八）：128-token prompt 批量 prefill
vs 128 × 逐 token decode。

会话位序：prefill(128) @0 → prefill(128) @128（两 rep）→ 128 步
decode（@256..384）。max_tokens=384 < cos 表 fixture 上限 512。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 384, True)
    var session = GpuSession(model, 384)

    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)

    # warmup + rep1
    for rep in range(2):
        var t0 = monotonic_ns()
        _ = session.prefill_step[INNER, INNER_FF](ids)
        session.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        print("prefill ", P, " tokens: ", ms, " ms  (", ms / Float64(P) * 1000.0, " us/token )")

    # 128 步逐 token decode 基线
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
