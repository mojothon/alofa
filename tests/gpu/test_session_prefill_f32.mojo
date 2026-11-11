"""prefill f32 快路径容差门（账本一百零五）：prefill_step(AF=float32)
vs AF=float64（同 f16 权重）——隔离 f32 快路径（f16 x 舍入 + MMA
累加序）的端到端数值偏差。f64 路径本身已由 test_session_prefill
（对 CPU run）钉死。

Run（A100）:
    mojo run -I src tests/gpu/test_session_prefill_f32.mojo
"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 64, True)
    var session = GpuSession(model, 64)

    var ids8 = List[Int]()
    for tid in [42, 17, 900, 7, 123, 5, 88, 2]:
        ids8.append(tid)

    # f64 精确路径先跑，快照（h_logits 会被 f32 跑覆写）
    var gpu64 = session.prefill_step[INNER, INNER_FF](ids8)
    var n = model.cfg.vocab
    var snap64 = List[Float64]()
    var scale = Float64(0)
    for i in range(n):
        snap64.append(Float64(gpu64[unsafe_offset=i]))
        var cv = snap64[i]
        if cv < 0:
            cv = -cv
        if cv > scale:
            scale = cv

    # ⚠️ kv_pos 是 chunked-prefill 流水游标（跨 run 接续）——每个
    # 变体 run 前必须重置 0，否则 attention 读到上次 run 的 KV 残留
    #（一百零五教训：之前所有连跑变体的 diff 都被此污染）。
    session.kv_pos = 0
    var gpu32 = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128,
        512, 1, 8, DType.float32](ids8)
    var diff = Float64(0)
    var best32 = 0
    for i in range(n):
        var d = Float64(gpu32[unsafe_offset=i]) - snap64[i]
        if d < 0:
            d = -d
        if d > diff:
            diff = d
        if gpu32[unsafe_offset=i] > gpu32[unsafe_offset=best32]:
            best32 = i
    var tol = Float64(0.02) * scale
    print("prefill f32-vs-f64: diff=", diff, " tol=", tol,
          " argmax32=", best32, " (want 5)")
    if diff > tol:
        raise Error("f32 fast path diff exceeded tolerance")
    if best32 != 5:
        raise Error("f32 fast path argmax mismatch")
    # MMA 诊断变体（kv_pos 重置后重测——一百零五重审）
    comptime VARIANTS = (
        ("dual-only", False, True),
        ("down-only", True, False),
    )
    @parameter
    for v in range(2):
        session.kv_pos = 0
        var gm = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128,
            128, 512, 1, 8, DType.float32, VARIANTS[v][1],
            VARIANTS[v][2]](ids8)
        var dm = Float64(0)
        var bm = 0
        for i in range(n):
            var d = Float64(gm[unsafe_offset=i]) - snap64[i]
            if d < 0:
                d = -d
            if d > dm:
                dm = d
            if gm[unsafe_offset=i] > gm[unsafe_offset=bm]:
                bm = i
        print("MMA", VARIANTS[v][0], ": diff=", dm, " argmax=", bm)
    print("*** PREFILL F32 GATE PASS ***")
