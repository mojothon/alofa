"""账本一百零一：attn_tiles=1 的 prefill 对数门（与主门同参照同判据）。

tsum/vpart 分块数变化 ⇒ 数值重组（非逐位），须落在 f16 专用容差
0.003×ref_max 内且 argmax 一致。

Run（A100）:
    mojo run -I src tests/gpu/test_session_prefill_t1.mojo
"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime F16_REL = Float64(0.003)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 64, True)
    var session = GpuSession(model, 64)

    var ids8 = List[Int]()
    for tid in [42, 17, 900, 7, 123, 5, 88, 2]:
        ids8.append(tid)

    var cpu8 = model.run(ids8)
    var n = model.cfg.vocab
    var snap8 = List[Float64]()
    for i in range(n):
        snap8.append(Float64(cpu8[unsafe_offset=i]))

    var g8 = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8](
        ids8
    )
    var diff8 = Float64(0)
    var scale8 = Float64(0)
    var best8 = 0
    var gbest8 = 0
    for i in range(n):
        var cv = snap8[i]
        if cv < 0:
            cv = -cv
        if cv > scale8:
            scale8 = cv
        var d = Float64(g8[unsafe_offset=i]) - snap8[i]
        if d < 0:
            d = -d
        if d > diff8:
            diff8 = d
        if snap8[i] > snap8[best8]:
            best8 = i
        if g8[unsafe_offset=i] > g8[unsafe_offset=gbest8]:
            gbest8 = i
    if diff8 > F16_REL * scale8:
        raise Error(
            "row8 t1: diff=" + String(diff8) + " tol=" + String(F16_REL * scale8)
        )
    if gbest8 != best8:
        raise Error("row8 t1 argmax mismatch: gpu=" + String(gbest8))
    print(
        "prefill row8 t1: diff=", diff8, " tol=", F16_REL * scale8,
        " argmax=", best8, " *** PASS ***",
    )
    model.params.keep_alive()
