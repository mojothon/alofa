"""账本一百二十四：堆审计二分二——第三个 session 用小 max_tokens，
判别 per-context 资源上限 vs 尺寸依赖。GPU3 空卡跑。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    print("creating s1 (512)")
    var a = GpuSession(model, 512)
    print("s1 ok")
    print("creating s2 (512)")
    var b = GpuSession(model, 512)
    print("s2 ok")
    print("creating s3 (32)")
    var c = GpuSession(model, 32)
    print("s3 ok")

    var toks = List[Int]()
    for i in range(8):
        toks.append(100 + i * 10)
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    a.ctx.synchronize()
    print("batch step a ok, row0:", Int(a.h_tokout[0]))
    _ = b.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    b.ctx.synchronize()
    print("batch step b ok, row0:", Int(b.h_tokout[0]))
    _ = c.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    c.ctx.synchronize()
    print("batch step c ok, row0:", Int(c.h_tokout[0]))
    print("*** HEAP PROBE2 ALL PASS ***")
