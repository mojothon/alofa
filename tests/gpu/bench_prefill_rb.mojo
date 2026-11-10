"""RB/AF 长窗口扫 v2（账本九十）：每 rep **新 GpuSession**（kv 归零
= 真同功 rep；同会话复用会把 128 token/rep 追加进 KV，rep 间注意力
工作量递增 —— 八十八/八十九的 rep 对比全部失真，此坑记账）。权重
复传在钟外。10 rep 取 min（共享卡唯一稳定口径）。kv: 每 rep 128 ≤
512 ✓。变体：rb1-f64（GEMV 同核基线）、rb8-f64（bit-exact 档）、
rb8-f32（向量化快路）。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128
comptime REPS = 10


def _bench[rb: Int, AF: DType](mut model: QwenForward, ids: List[Int]) raises:
    # warmup rep 不计钟：per-session 首发射成本（内存池首分配等）
    var warm = GpuSession(model, 512)
    _ = warm.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, rb, AF](ids)
    warm.ctx.synchronize()
    var best = 1e30
    for rep in range(REPS):
        var session = GpuSession(model, 512)
        var t0 = monotonic_ns()
        _ = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 8, rb, AF](
            ids
        )
        session.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best:
            best = ms
    print("rb=", rb, " af=", AF, ": min=", best, " ms  (", REPS, " fresh-session reps )")


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)

    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)

    _bench[1, DType.float64](model, ids)
    _bench[8, DType.float64](model, ids)
    _bench[8, DType.float32](model, ids)

    var session = GpuSession(model, 512)
    var best = 1e30
    for i in range(64):
        var t0 = monotonic_ns()
        _ = session.decode_step[INNER, INNER_FF](2000 + i)
        var ms = Float64(monotonic_ns() - t0) / 1e6
        if ms < best:
            best = ms
    print("decode 64 x1: best=", best, " ms")
    model.params.keep_alive()
