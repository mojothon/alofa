"""账本一百零九-b：prefill 的伸缩性——t=128 vs t=256（单次调用、
kv_pos 重置）。若时间近似翻倍 ⇒ 计算受限；若明显小于翻倍 ⇒ 存在
与 batch 无关的每核间隙/固定开销（则可推出「融合核数」的收益上限）。

同时打印 prefill 与单层核数的比例：层数 24、每层核数量级 20。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 1024, True)
    var s = GpuSession(model, 1024)
    var ids = List[Int]()
    for i in range(128):
        ids.append(1000 + (i * 37) % 900)
    var ids256 = List[Int]()
    for i in range(256):
        ids256.append(1000 + (i * 53) % 900)

    # 预热
    s.kv_pos = 0
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids256, False, True)
    s.ctx.synchronize()

    var b128 = 1e30
    var b256 = 1e30
    for k in range(3):
        s.kv_pos = 0
        var t0 = monotonic_ns()
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32](ids256, False, True)
        var d0 = Float64(monotonic_ns() - t0) / 1e6
        s.kv_pos = 0
        var t1 = monotonic_ns()
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32](ids256, False, True)
        var d1 = Float64(monotonic_ns() - t1) / 1e6
        if d0 < b128:
            b128 = d0
        if d1 < b256:
            b256 = d1
        print("rep", k, " t=128:", d0, " t=256:", d1)
    print("min t=128:", b128, " t=256:", b256,
          " ratio:", b256 / b128)
    model.params.keep_alive()
