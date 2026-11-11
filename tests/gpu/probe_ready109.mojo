"""账本一百零九-d：wp_ready 是否跨 prefill 调用保留？（nsys 显示打包
核每次调用都跑 24 层份 = 1.25ms/前向白费）"""

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var s = GpuSession(model, 512)
    var ids = List[Int]()
    for i in range(128):
        ids.append(1000 + (i * 37) % 900)
    print("init wp_ready:", s.wp_ready, " n_layers:", s.n_layers)
    s.kv_pos = 0
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, False, True)
    print("after call1 wp_ready:", s.wp_ready)
    s.kv_pos = 0
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, False, True)
    print("after call2 wp_ready:", s.wp_ready)
    model.params.keep_alive()
