"""一百七十六-c：f64 vs f32 prefill 逐层对账——首个发散层定位。

同 token（42）单 token 前向，两个 session 分别 af=f64（默认）与
af=f32（生产），dbg_layers 打印逐层 d_xb 校验和，找首个发散层。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 64, True)
    var s64 = GpuSession(model, 64)
    var s32 = GpuSession(model, 64)
    var one = List[Int]()
    one.append(42)
    print("==== af=f64 ====")
    _ = s64.prefill_step[INNER, INNER_FF](
        one, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, False,
        True
    )
    print("==== af=f32 ====")
    _ = s32.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32
    ](
        one, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, False,
        True
    )
    print("done")
