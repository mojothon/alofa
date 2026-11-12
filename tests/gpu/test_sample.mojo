"""账本一百二十三：采样门——sample_rb（Gumbel-max 温度采样）。

判据：①temp=0 与贪心逐位一致；②同种子确定性、异种子变异；
③T=1 经验分布对齐 host softmax 参照（N=400，top-4 各 ±0.10）。"""

from std.math import exp

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime N = 400


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)

    # 单序列 prompt=100，batch t=1
    var toks = List[Int]()
    toks.append(100)

    # 贪心参照（argmax 路径）
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    a.ctx.synchronize()
    var greedy = Int(a.h_tokout[0])
    # 顺带取 logits 算 host softmax 参照（copy_logits=True 默认开）
    var hv = List[Float32]()
    var hidx = List[Int]()
    for v in range(8):
        hv.append(Float32(-1e30))
        hidx.append(0)
    for v in range(151936):
        var lv = a.h_logits_b[v]
        for k in range(8):
            if lv > hv[k]:
                for m in range(7, k, -1):
                    hv[m] = hv[m - 1]
                    hidx[m] = hidx[m - 1]
                hv[k] = lv
                hidx[k] = v
                break
    var zsum = Float32(0)
    for k in range(8):
        zsum += exp(hv[k])
    # 尾部 logits 求和（近似全词表 Z：非 top-8 贡献小，但为公平
    # 用全量再算一次精确 Z）
    var zfull = Float32(0)
    for v in range(151936):
        zfull += exp(a.h_logits_b[v])
    print("greedy:", greedy, " top8:",
          hidx[0], hidx[1], hidx[2], hidx[3],
          hidx[4], hidx[5], hidx[6], hidx[7])

    # ①temp=0：与贪心逐位一致
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks, False, False, False, 0.0, 7)
    a.ctx.synchronize()
    var t0ok = Int(a.h_tokout[0]) == greedy

    # ②种子确定性 + 异种子变异（T=1）
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks, False, False, False, 1.0, 42)
    a.ctx.synchronize()
    var s42a = Int(a.h_tokout[0])
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks, False, False, False, 1.0, 42)
    a.ctx.synchronize()
    var s42b = Int(a.h_tokout[0])
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks, False, False, False, 1.0, 43)
    a.ctx.synchronize()
    var s43 = Int(a.h_tokout[0])
    print("seed42:", s42a, s42b, " seed43:", s43)

    # ③分布对齐：N 次独立采样（seed=i）直方图 top-4
    var hist = List[Int]()
    for k in range(4):
        hist.append(0)
    for n in range(N):
        _ = a.prefill_step[
            INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](toks, False, False, False, 1.0, Int32(1000 + n))
        a.ctx.synchronize()
        var tok = Int(a.h_tokout[0])
        for k in range(4):
            if tok == hidx[k]:
                hist[k] += 1
                break
    var dist_ok = True
    for k in range(4):
        var emp = Float32(hist[k]) / Float32(N)
        var exp_p = exp(hv[k]) / zfull
        var d = emp - exp_p
        if d < Float32(-0.10) or d > Float32(0.10):
            dist_ok = False
        print("tok", hidx[k], " exp:", exp_p, " emp:", emp)
    print("t0 parity:", t0ok,
          " seed-det:", (s42a == s42b),
          " seed-var:", (s42a != s43),
          " dist-ok:", dist_ok)
    if t0ok and (s42a == s42b) and (s42a != s43) and dist_ok:
        print("*** SAMPLE GATE PASS ***")
    else:
        print("*** SAMPLE GATE FAIL ***")
