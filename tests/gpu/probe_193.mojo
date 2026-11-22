"""一百七十四-c 第三刀复现探针 L：spec 路径正确性终裁（设备侧读回）。

probe_192 定案：spec tick 的 copy_logits=False，h_logits_b 从未
写入——此前 probe_183/185/186 的"漂移"全是读未初始化 host 缓冲的
假信号；DBGT 设备侧 d_logits_b 读回证明 A1==A2 逐位一致。
本探针绕开 h_logits_b，直接 enqueue_copy 读 d_logits_b：
  ① 建底+验证 tick 设备读回，双会话逐位比（真确定性）
  ② 每行 vs CPU 累积真值（干净模型）——spec 前向正确性
"""

from max.gpu.host import HostBuffer
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime VOCAB = 151936


def snap_rows(
    mut sess: GpuSession, rows: Int
) raises -> HostBuffer[DType.float32]:
    sess.ctx.synchronize()
    var h = sess.ctx.enqueue_create_host_buffer[DType.float32](
        rows * VOCAB
    )
    sess.ctx.enqueue_copy(dst_buf=h, src_buf=sess.d_logits_b)
    sess.ctx.synchronize()
    return h


def row_diff(
    label: String,
    h: HostBuffer[DType.float32],
    row: Int,
    refp: Pointer[Float32, MutUntrackedOrigin],
) raises -> Float64:
    var worst = Float64(0)
    var wcol = Int(-1)
    var am = Int(-1)
    var amv = Float64(-1e30)
    for i in range(VOCAB):
        var a = Float64(h[row * VOCAB + i])
        if a > amv:
            amv = a
            am = i
        var b = Float64(refp[unsafe_offset=i])
        var d = a - b
        if d < 0:
            d = -d
        if d > worst:
            worst = d
            wcol = i
    print(
        label, " row", row, " worst =", worst, " col", wcol,
        " gpu =", h[row * VOCAB + wcol], " cpu =",
        refp[unsafe_offset=wcol], " argmax =", am
    )
    return worst


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

    # ① 双会话确定性（设备读回）
    var a1 = GpuSession(model, 512)
    _ = a1.spec_step[INNER, INNER_FF](toks, ln1, sq1, True)
    _ = a1.spec_step[INNER, INNER_FF](vtoks, ln2, sq2, False)
    var g1 = snap_rows(a1, 4)
    var a2 = GpuSession(model, 512)
    _ = a2.spec_step[INNER, INNER_FF](toks, ln1, sq1, True)
    _ = a2.spec_step[INNER, INNER_FF](vtoks, ln2, sq2, False)
    var g2 = snap_rows(a2, 4)
    var worst = Float64(0)
    for i in range(4 * VOCAB):
        var d = Float64(g1[i]) - Float64(g2[i])
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    print("① 设备读回确定性 worst =", worst)

    # ② 每行 vs CPU 真值：row k = vtoks[k] @ pos 11+k —— 参照 =
    # [100..110] + vtoks[0..k]（共 12+k 个 token）的末位 logits
    for k in range(4):
        var sub = List[Int]()
        for i in range(11):
            sub.append(100 + i)
        for j in range(k + 1):
            sub.append(vtoks[j])
        var m = QwenForward(WEIGHTS, CONFIG, 64, True)
        var refp = m.run(sub)
        _ = row_diff("spec", g1, k, refp)
        m.params.keep_alive()
    model.params.keep_alive()
    print("done")
