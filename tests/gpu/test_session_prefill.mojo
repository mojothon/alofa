"""GpuSession prefill_step 端到端对数（账本八十八）：8 token 批量
prefill + 1 token decode vs 产品 `run`（BACKEND_SCALAR）。

判据：prefill 末行 logits（第 8 行）与 decode(第 9 token) logits
分别对 CPU run 同序列的对应行——相对容差同 decode 门（f16 专用
0.003 × ref_max）+ argmax 一致。prefill 的 attention 逐行算术与
decode 同构（同核同序同树），差应落在 decode 同带内；超出即 bug。

Run（A100）:
    mojo run -I src tests/gpu/test_session_prefill.mojo
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

    # ⚠️ run 返回的指针指向模型 arena：下一次 run 会复用覆写 ——
    # 每次比对前必须先快照到自有序。
    var cpu8 = model.run(ids8)
    var n = model.cfg.vocab
    var snap8 = List[Float64]()
    for i in range(n):
        snap8.append(Float64(cpu8[unsafe_offset=i]))
    # row9 参照：run 累积历史（此模型已到 8），另起一个干净模型
    # 单次跑 9 tokens —— 返回的才是 row9 的 logits。
    var model2 = QwenForward(WEIGHTS, CONFIG, 64, True)
    var ids9 = List[Int]()
    for tid in [42, 17, 900, 7, 123, 5, 88, 2, 31]:
        ids9.append(tid)
    var cpu9 = model2.run(ids9)
    var snap9 = List[Float64]()
    for i in range(n):
        snap9.append(Float64(cpu9[unsafe_offset=i]))
    model2.params.keep_alive()

    # ⚠️ h_logits 是共享 host 缓冲：decode_step 会覆写 prefill 的
    # 返回指针内容 —— row8 必须先比再 decode。
    var gpu8 = session.prefill_step[INNER, INNER_FF](ids8)

    # 手写比对（verify 助手只吃 F32Ptr；快照是 List[Float64]）
    var diff8 = Float64(0)
    var scale8 = Float64(0)
    var best8 = 0
    var gbest8 = 0
    for i in range(n):
        var cv = snap8[i]
        var gv = Float64(gpu8[unsafe_offset=i])
        if cv < 0:
            cv = -cv
        if cv > scale8:
            scale8 = cv
        var d = Float64(gpu8[unsafe_offset=i]) - snap8[i]
        if d < 0:
            d = -d
        if d > diff8:
            diff8 = d
        if snap8[i] > snap8[best8]:
            best8 = i
        if gpu8[unsafe_offset=i] > gpu8[unsafe_offset=gbest8]:
            gbest8 = i
    if diff8 > F16_REL * scale8:
        raise Error(
            "row8: diff=" + String(diff8) + " tol=" + String(F16_REL * scale8)
        )
    if gbest8 != best8:
        raise Error("row8 argmax mismatch: gpu=" + String(gbest8))
    print("prefill row8: diff=", diff8, " tol=", F16_REL * scale8,
          " argmax=", best8, " *** PASS ***")

    var gpu9 = session.decode_step[INNER, INNER_FF](31)
    var diff9 = Float64(0)
    var scale9 = Float64(0)
    var best9 = 0
    var gbest9 = 0
    for i in range(n):
        var cv = snap9[i]
        if cv < 0:
            cv = -cv
        if cv > scale9:
            scale9 = cv
        var d = Float64(gpu9[unsafe_offset=i]) - snap9[i]
        if d < 0:
            d = -d
        if d > diff9:
            diff9 = d
        if snap9[i] > snap9[best9]:
            best9 = i
        if gpu9[unsafe_offset=i] > gpu9[unsafe_offset=gbest9]:
            gbest9 = i
    if diff9 > F16_REL * scale9:
        raise Error(
            "row9: diff=" + String(diff9) + " tol=" + String(F16_REL * scale9)
        )
    if gbest9 != best9:
        raise Error("row9 argmax mismatch: gpu=" + String(gbest9))
    print("decode row9: diff=", diff9, " tol=", F16_REL * scale9,
          " argmax=", best9, " *** PASS ***")
    model.params.keep_alive()
