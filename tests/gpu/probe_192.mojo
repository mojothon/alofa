"""一百七十四-c 第三刀复现探针 K：验证 tick 漂移逐层归因。

probe_183 复现：验证 tick（trust_all=False）中 row 0 恒一致、rows
1-3 在两次相同运行间漂移（worst ~0.14，ndiff ~60 万）。本探针双
会话各跑 建底 tick + 验证 tick（dbg_layers=True，全 t 行校验和），
定位首个漂移层与首个漂移行。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime VOCAB = 151936


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var toks = List[Int]()
    for i in range(11):
        toks.append(100 + i)
    var vtoks = List[Int]()
    vtoks.append(1052)
    vtoks.append(12163)
    vtoks.append(264)
    vtoks.append(23335)
    var ln1 = List[Int]()
    ln1.append(11)
    var sq1 = List[Int]()
    sq1.append(0)
    var ln2 = List[Int]()
    ln2.append(4)
    var sq2 = List[Int]()
    sq2.append(0)

    var a1 = GpuSession(model, 512)
    _ = a1.spec_step[INNER, INNER_FF](toks, ln1, sq1, True)
    print("==== A1 verify ====")
    _ = a1.spec_step[INNER, INNER_FF](vtoks, ln2, sq2, False, True)
    var a2 = GpuSession(model, 512)
    _ = a2.spec_step[INNER, INNER_FF](toks, ln1, sq1, True)
    print("==== A2 verify ====")
    _ = a2.spec_step[INNER, INNER_FF](vtoks, ln2, sq2, False, True)

    # 终局 logits 逐行比对（行内 sum 指纹）
    a1.ctx.synchronize()
    a2.ctx.synchronize()
    for r in range(4):
        var s1 = Float64(0)
        var s2 = Float64(0)
        var mx1 = Float64(-1e30)
        var m1 = Int(-1)
        for i in range(r * VOCAB, (r + 1) * VOCAB):
            s1 += Float64(a1.h_logits_b[i])
            var v = Float64(a2.h_logits_b[i])
            s2 += v
            if v > mx1:
                mx1 = v
                m1 = i - r * VOCAB
        var d = s1 - s2
        if d < 0:
            d = -d
        print("row", r, " sumA1 =", s1, " sumA2 =", s2, " diff =",
            d, " A2argmax =", m1)
    print("done")
