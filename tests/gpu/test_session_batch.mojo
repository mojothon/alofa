"""账本一百二十：批解码正确性门 v2。

8 序列 × 3 步：batch=True prefill vs 单序列参照。参照改为
**逐序列独立重放**（单参照 session 顺序跑 8 序列会互相覆写 KV
历史——v1 的 step≥1 MISMATCH 是参照侧伪影，批侧 step 0 已
diff=0.0 逐位全对）。流程：批跑 3 步存 logits → 逐序列重放
对比。判据：diff<1e-4 + argmax 一致。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime B = 8
comptime VOCAB = 151936
comptime STEPS = 3


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)
    var b = GpuSession(model, 512)

    # 阶段一：批 3 步，logits 存 [STEPS*B][VOCAB]
    var saved = a.ctx.enqueue_create_host_buffer[DType.float32](
        STEPS * B * VOCAB
    )
    a.ctx.synchronize()
    for s in range(STEPS):
        var btoks = List[Int]()
        for i in range(B):
            btoks.append(100 + i * 10 + s)
        _ = a.prefill_step[
            INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](btoks)
        a.ctx.synchronize()
        for i in range(B * VOCAB):
            saved[s * B * VOCAB + i] = a.h_logits_b[i]

    # 阶段二：逐序列重放（各序列在自己的 0..STEPS-1 连续位上）
    var worst = 0.0
    var mismatch = 0
    for i in range(B):
        b.kv_pos = 0
        for s in range(STEPS):
            var toks = List[Int]()
            toks.append(100 + i * 10 + s)
            _ = b.prefill_step[
                INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
                DType.float32
            ](toks)
            b.ctx.synchronize()
            var diff = 0.0
            var hlog = b.h_logits
            for v in range(VOCAB):
                var d = Float64(
                    abs(hlog[v] - saved[(s * B + i) * VOCAB + v])
                )
                if d > diff:
                    diff = d
            if diff > worst:
                worst = diff
            var hv = Float32(-1e30)
            var hi = 0
            for v in range(VOCAB):
                if hlog[v] > hv:
                    hv = hlog[v]
                    hi = v
            # 批侧 argmax：重放时的保存值不可得——用批侧 tokout
            # 只在最后一步可对照（h_tokout 是最后一步的）。改为
            # 比对 logits argmax vs 保存的批 logits argmax。
            var bv = Float32(-1e30)
            var bi = 0
            for v in range(VOCAB):
                if saved[(s * B + i) * VOCAB + v] > bv:
                    bv = saved[(s * B + i) * VOCAB + v]
                    bi = v
            if hi != bi:
                mismatch += 1
                print("MISMATCH seq", i, "step", s,
                      "batch:", bi, "ref:", hi)

    print("worst logits diff:", worst, " argmax mismatch:", mismatch)
    if worst < 1e-4 and mismatch == 0:
        print("*** BATCH GATE PASS ***")
    else:
        print("*** BATCH GATE FAIL ***")
