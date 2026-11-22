"""一百七十四-c 第三刀复现探针 G：批路 tail 的 t 依赖裁决。

probe_187：批 t=11 行 0（token 100 pos 0）vs CPU 差 18.7（f32/f64
一致地错）。但 test_session_batch（批 t=8 vs 单 tick 重放）PASS——
GPU-GPU 互比不能发现共同错误。本探针用 CPU 真值直接裁决三种形态：
  ① 单 tick t=1 token 100（非批 tail）
  ② 批 t=8  tokens 100-107 行 0
  ③ 批 t=11 tokens 100-110 行 0
若 ① 对而 ②③ 错 → 批 tail（LMHEAD_K）t 依赖 bug；若 ②③ 错而
t=8 曾 PASS 是 GPU-GPU 假绿 → 单 tick 也需复核。
"""

from max.gpu.host import HostBuffer
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime VOCAB = 151936


def row_diff(
    label: String,
    h: HostBuffer[DType.float32],
    row: Int,
    refp: Pointer[Float32, MutUntrackedOrigin],
    rows_snap: Int,
) raises -> Float64:
    var worst = Float64(0)
    var wcol = Int(-1)
    for i in range(VOCAB):
        var a = Float64(h[row * rows_snap + i])
        var b = Float64(refp[unsafe_offset=i])
        var d = a - b
        if d < 0:
            d = -d
        if d > worst:
            worst = d
            wcol = i
    print(
        label, " worst =", worst, " col", wcol,
        " gpu =", h[row * rows_snap + wcol], " cpu =",
        refp[unsafe_offset=wcol]
    )
    return worst


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var mcpu = QwenForward(WEIGHTS, CONFIG, 64, True)
    var one = List[Int]()
    one.append(100)
    var ref0 = mcpu.run(one)
    mcpu.params.keep_alive()

    # ① 单 tick t=1（f32 显式全 MMA 列表，与 test_session_batch 同款）
    var s1 = GpuSession(model, 512)
    var t1 = List[Int]()
    t1.append(100)
    _ = s1.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32,
        True, True, True, True, True, True, True, True, False, True
    ](t1)
    s1.ctx.synchronize()
    var h1 = s1.ctx.enqueue_create_host_buffer[DType.float32](VOCAB)
    s1.ctx.synchronize()
    for i in range(VOCAB):
        h1[i] = s1.h_logits_b[i]
    _ = row_diff("① 单tick t=1 f32", h1, 0, ref0, VOCAB)

    # ② 批 t=8
    var s2 = GpuSession(model, 512)
    var t8 = List[Int]()
    for i in range(8):
        t8.append(100 + i)
    _ = s2.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32,
        True, True, True, True, True, True, True, True, False, True
    ](t8)
    s2.ctx.synchronize()
    var h2 = s2.ctx.enqueue_create_host_buffer[DType.float32](8 * VOCAB)
    s2.ctx.synchronize()
    for i in range(8 * VOCAB):
        h2[i] = s2.h_logits_b[i]
    _ = row_diff("② 批 t=8 f32 行0", h2, 0, ref0, VOCAB)

    # ③ 批 t=11
    var s3 = GpuSession(model, 512)
    var t11 = List[Int]()
    for i in range(11):
        t11.append(100 + i)
    _ = s3.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32,
        True, True, True, True, True, True, True, True, False, True
    ](t11)
    s3.ctx.synchronize()
    var h3 = s3.ctx.enqueue_create_host_buffer[DType.float32](11 * VOCAB)
    s3.ctx.synchronize()
    for i in range(11 * VOCAB):
        h3[i] = s3.h_logits_b[i]
    _ = row_diff("③ 批 t=11 f32 行0", h3, 0, ref0, VOCAB)
    model.params.keep_alive()
    print("done")
