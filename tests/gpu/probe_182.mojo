"""一百七十四-c 第三刀复现探针：MMA 核族尾组部分填充非确定。

账本 0089913 记录：t=11（16 行 MMA t-块只填 11 行）时 dual_pk/down_spl
非确定（logits 全 inf、同输入重跑漂移）；no-mma 同输入逐位正确——
本探针把它固化成可重跑的判定：

  t=11：A1/A2 = 生产 MMA 配置同输入各跑一遍，h_logits_b 全 11 行
        逐位比（确定性）；B = no-mma oracle 同输入，逐行 max diff
        （正确性）。
  t=16：整组对照，三项应全为 0（生产 t=64/128 从未暴露的原因）。

no-mma oracle 关 dual/down/wpack/mma_q/mma_o/tile/spl，gqa/kv 保持
同生产（attention 路径一致，只隔离 GEMM 实现差）。
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


def tick_mma(mut sess: GpuSession, toks: List[Int]) raises:
    _ = sess.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    sess.ctx.synchronize()


def tick_nomma(mut sess: GpuSession, toks: List[Int]) raises:
    _ = sess.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, False, False, False, False, False, False,
        True, False, True, True
    ](toks)
    sess.ctx.synchronize()


def compare(
    ha: HostBuffer[DType.float32],
    hb: HostBuffer[DType.float32],
    rows: Int,
    label: String,
) raises -> Float64:
    var worst = Float64(0)
    for i in range(rows * VOCAB):
        var d = abs(Float64(ha[i]) - Float64(hb[i]))
        if d > worst:
            worst = d
    print(label, " worst =", worst)
    return worst


def run_case(
    mut model: QwenForward, t: Int, base: Int, label: String
) raises:
    print("====", label, "(t =", t, ") ====")
    var a1 = GpuSession(model, 512)
    var a2 = GpuSession(model, 512)
    var b = GpuSession(model, 512)
    var toks = List[Int]()
    for i in range(t):
        toks.append(base + i)

    tick_mma(a1, toks)
    var saved = a1.ctx.enqueue_create_host_buffer[DType.float32](
        t * VOCAB
    )
    a1.ctx.synchronize()
    for i in range(t * VOCAB):
        saved[i] = a1.h_logits_b[i]

    tick_mma(a2, toks)
    var d_det = compare(saved, a2.h_logits_b, t, "A1-vs-A2 确定性")
    tick_nomma(b, toks)
    var d_orc = compare(saved, b.h_logits_b, t, "A1-vs-oracle 正确性")
    if d_det != 0.0 or d_orc > 1e-3:
        print(label, " -> FAIL (det", d_det, "oracle", d_orc, ")")
    else:
        print(label, " -> PASS")
    model.params.keep_alive()


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    run_case(model, 11, 100, "尾组 t=11")
    run_case(model, 16, 200, "整组 t=16")
    print("done")
