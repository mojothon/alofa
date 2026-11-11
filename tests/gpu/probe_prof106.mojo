"""账本一百零六-b：prefill 分段 profile（生产配置 t=128）。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 256, True)
    var session = GpuSession(model, 256)
    var ids = List[Int]()
    for i in range(128):
        ids.append((i * 37 + 11) % 1000)
    session.kv_pos = 0
    _ = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, False, True)
    session.kv_pos = 0
    _ = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, True, True)
    session.kv_pos = 0
    _ = session.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, True, True)
