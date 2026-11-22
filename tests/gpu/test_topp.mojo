"""账本一百二十八：top-p（nucleus）采样门——sample_topp_gpu。

判据：①top_p=0.9 抽样 400 次全部落在 host 镜像核内（边界留
ε 容差）；②经验分布对齐 host 核内重归一 softmax（top-6 ±0.10）；
③同种子确定 / 异种子变异；④top_p=1.0 退化为全词表采样（分布
同 123 门参照，top-4 ±0.10）。"""

from std.math import exp

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime N = 400
comptime VOCAB = 151936
comptime TOP_P = 0.9


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)

    var toks = List[Int]()
    toks.append(100)

    # 贪心参照 + 取 batch tail logits（copy_logits=True 默认开）
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks, False, True, True, False, 0.0, 7)
    a.ctx.synchronize()
    var greedy = Int(a.h_tokout[0])
    print("greedy:", greedy)

    # ---- host 镜像二分（T=1，top_p=0.9）----
    var m = Float32(-1e30)
    for v in range(VOCAB):
        if a.h_logits_b[v] > m:
            m = a.h_logits_b[v]
    var zraw = Float32(0)
    for v in range(VOCAB):
        zraw += exp(a.h_logits_b[v] - m)
    var lo = m - 80.0
    var hi = m
    for _it in range(24):
        var tau = (lo + hi) / 2.0
        var acc = Float32(0)
        for v in range(VOCAB):
            if a.h_logits_b[v] >= tau:
                acc += exp(a.h_logits_b[v] - m)
        if acc / zraw >= Float32(TOP_P):
            lo = tau
        else:
            hi = tau
    # 核 = {logit ≥ lo}；核内重归一概率（zn 为核内 raw 和）
    var zn = Float32(0)
    for v in range(VOCAB):
        if a.h_logits_b[v] >= lo:
            zn += exp(a.h_logits_b[v] - m)
    print("host nucleus mass:", zn / zraw, "（≥0.9） raw zn:", zn)
    if zn / zraw < Float32(TOP_P):
        print("*** TOPP GATE FAIL: host nucleus under mass ***")
        return

    # top-6（核内按原始 logit 排）
    var hv = List[Float32]()
    var hidx = List[Int]()
    for k in range(6):
        hv.append(Float32(-1e30))
        hidx.append(-1)
    for v in range(VOCAB):
        var lv = a.h_logits_b[v]
        if lv < lo:
            continue
        for k in range(6):
            if lv > hv[k]:
                for mm in range(5, k, -1):
                    hv[mm] = hv[mm - 1]
                    hidx[mm] = hidx[mm - 1]
                hv[k] = lv
                hidx[k] = v
                break
    var hp = List[Float32]()
    for k in range(6):
        hp.append(exp(hv[k] - m) / zn)
    print("top6 idx:", hidx[0], hidx[1], hidx[2], hidx[3], hidx[4], hidx[5])
    for k in range(6):
        print("  exp p:", hp[k])

    # ---- ①② top_p=0.9：核成员 + 分布（N=400）----
    var hist = List[Int]()
    for k in range(6):
        hist.append(0)
    var outside = 0
    for n in range(N):
        _ = a.prefill_step[
            INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](toks, False, False, False, False, 1.0, Int32(2000 + n), Float32(TOP_P))
        a.ctx.synchronize()
        var tok = Int(a.h_tokout[0])
        # 成员断言（边界 ε=0.01 logit 容差）
        var intop6 = False
        for k in range(6):
            if tok == hidx[k]:
                hist[k] += 1
                intop6 = True
        if not intop6:
            var lv2 = a.h_logits_b[tok]
            if lv2 < lo - 0.01:
                outside += 1
                if outside < 4:
                    print("OUTSIDE-NUCLEUS token:", tok, "logit:", lv2,
                          "cut:", lo)
    var dist_ok = True
    for k in range(6):
        var emp = Float32(hist[k]) / Float32(N)
        var d = emp - hp[k]
        if d < Float32(-0.10) or d > Float32(0.10):
            dist_ok = False
        print("tok", hidx[k], " exp:", hp[k], " emp:", emp)
    print("outside-nucleus samples:", outside, "/400")

    # ---- ③ 种子确定性（top_p=0.9）----
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks, False, False, False, False, 1.0, 77, Float32(TOP_P))
    a.ctx.synchronize()
    var s77a = Int(a.h_tokout[0])
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks, False, False, False, False, 1.0, 77, Float32(TOP_P))
    a.ctx.synchronize()
    var s77b = Int(a.h_tokout[0])
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks, False, False, False, False, 1.0, 78, Float32(TOP_P))
    a.ctx.synchronize()
    var s78 = Int(a.h_tokout[0])
    print("seed77:", s77a, s77b, " seed78:", s78)

    # KV 回绕：①③共 ~400+ 步已近 512 上限（一百二十八坑：连续
    # prefill_step 采样调用推进 kv_pos，长门须重置）
    a.kv_pos = 0
    for i in range(64):
        a.h_posarr[i] = Int32(0)
    a.ctx.synchronize()

    # ---- ④ top_p=1.0：全词表（分布同 123 参照）----
    # 全量 softmax Z
    var zfull = zraw
    var hist1 = List[Int]()
    for k in range(4):
        hist1.append(0)
    for n in range(N):
        _ = a.prefill_step[
            INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](toks, False, False, False, False, 1.0, Int32(3000 + n), 1.0)
        a.ctx.synchronize()
        var tok1 = Int(a.h_tokout[0])
        for k in range(6):
            if tok1 == hidx[k] and k < 4:
                hist1[k] += 1
    var full_ok = True
    for k in range(4):
        var emp1 = Float32(hist1[k]) / Float32(N)
        var exp1 = Float32(0)
        if hidx[k] >= 0:
            exp1 = exp(a.h_logits_b[hidx[k]] - m) / zfull
        var d1 = emp1 - exp1
        if d1 < Float32(-0.10) or d1 > Float32(0.10):
            full_ok = False
        print("topp1 tok", hidx[k], " exp:", exp1, " emp:", emp1)

    var member_ok = outside == 0
    var det_ok = (s77a == s77b) and (s78 != s77a)
    print("member_ok:", member_ok, " dist_ok:", dist_ok,
          " det_ok:", det_ok, " full_ok:", full_ok)
    if member_ok and dist_ok and det_ok and full_ok:
        print("*** TOPP GATE PASS ***")
    else:
        print("*** TOPP GATE FAIL ***")
