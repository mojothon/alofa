"""账本一百二十：B=1 批解码 bisect——batch=True t=1 vs 单序列 t=1。
若 B=1 仍坏：bug 在批管道自身（_bb 核/store/head）；若好：跨行串扰。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime VOCAB = 151936


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)
    var b = GpuSession(model, 512)

    for s in range(2):
        var toks = List[Int]()
        toks.append(100 + s)
        _ = a.prefill_step[
            INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](toks)
        a.ctx.synchronize()

        b.kv_pos = s
        _ = b.prefill_step[
            INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32
        ](toks)
        b.ctx.synchronize()

        var diff = 0.0
        var hlog = b.h_logits
        for v in range(VOCAB):
            var d = Float64(abs(hlog[v] - a.h_logits_b[v]))
            if d > diff:
                diff = d
        print("step", s, "maxdiff:", diff,
              " batch argmax:", Int(a.h_tokout[0]))
