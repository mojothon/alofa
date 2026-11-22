"""一百七十四-c 第三刀复现探针 B：spec（row_mapped）路径确定性。

账本 0089913 记录的"MMA 尾组非确定"当时与 embedp 别名 bug 并发出现；
修复后先在普通批 tick 上验证了确定性（probe_182：A1-vs-A2 逐位 0.0，
MMA-vs-noMMA 差 ~0.03 为累加序正常差）。本探针补 row_mapped spec 路径：

  ① prompt 建底 tick（trust_all，t=11）双会话逐位比。
  ② spec 建底 vs 普通 prefill_step（同 token 同位，恒等行映射）
     逐位比——174-b 的"identity-mapped bit-exact"回归判据。
  ③ draft 验证 tick（trust_all=False，x0+3 draft）双会话逐位比。
"""

from std.math import abs
from max.gpu.host import HostBuffer
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime VOCAB = 151936


def compare(
    ha: HostBuffer[DType.float32],
    hb: HostBuffer[DType.float32],
    rows: Int,
    label: String,
) raises -> Float64:
    var worst = Float64(0)
    var worst_row = Int(-1)
    var worst_col = Int(-1)
    var ndiff = Int(0)
    for r in range(rows):
        var rworst = Float64(0)
        var rcol = Int(-1)
        for i in range(r * VOCAB, (r + 1) * VOCAB):
            var d = abs(Float64(ha[i]) - Float64(hb[i]))
            if d > 0:
                ndiff += 1
            if d > rworst:
                rworst = d
                rcol = i - r * VOCAB
            if d > worst:
                worst = d
                worst_row = r
                worst_col = i - r * VOCAB
        if rworst > 0:
            print(
                label, " row", r, " worst =", rworst, " col", rcol,
                " a=", ha[r * VOCAB + rcol],
                " b=", hb[r * VOCAB + rcol]
            )
    print(
        label, " worst =", worst, " at row", worst_row, " col",
        worst_col, " ndiff =", ndiff
    )
    return worst


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var toks = List[Int]()
    for i in range(11):
        toks.append(100 + i)

    # ① prompt 建底 tick：确定性
    var a1 = GpuSession(model, 512)
    var a2 = GpuSession(model, 512)
    var ln1 = List[Int]()
    ln1.append(11)
    var sq1 = List[Int]()
    sq1.append(0)
    _ = a1.spec_step[INNER, INNER_FF](toks, ln1, sq1, True)
    _ = a2.spec_step[INNER, INNER_FF](toks, ln1, sq1, True)
    var d1 = compare(a1.h_logits_b, a2.h_logits_b, 11, "① spec 建底确定性")

    # ② spec 建底 vs 普通 prefill：恒等
    var a3 = GpuSession(model, 512)
    _ = a3.prefill_step[INNER, INNER_FF](toks)
    a3.ctx.synchronize()
    var d2 = compare(a1.h_logits_b, a3.h_logits_b, 11, "② spec-vs-prefill 恒等")

    # ③ draft 验证 tick：确定性（x0=1052 + 健康续写 3 draft）
    var vtoks = List[Int]()
    vtoks.append(1052)
    vtoks.append(12163)
    vtoks.append(264)
    vtoks.append(23335)
    var ln2 = List[Int]()
    ln2.append(4)
    var sq2 = List[Int]()
    sq2.append(0)
    var out1 = a1.spec_step[INNER, INNER_FF](vtoks, ln2, sq2, False)
    var out2 = a2.spec_step[INNER, INNER_FF](vtoks, ln2, sq2, False)
    var d3 = compare(a1.h_logits_b, a2.h_logits_b, 4, "③ 验证 tick 确定性")
    print("验证返回 out1 =", out1, " out2 =", out2)

    var ok = (d1 == 0.0) and (d2 == 0.0) and (d3 == 0.0)
    print("SPEC-DETERMINISM", "PASS" if ok else "FAIL")
    model.params.keep_alive()
    print("done")
