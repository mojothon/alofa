"""账本一百零九-c：nsys 时间线用最小 probe —— 1 次预热 + 1 次被测
prefill（t=128、生产配置、无 gap）。配合：

    nsys profile -o /tmp/t109 --force-overwrite true --trace=cuda \
        mojo run -I src tests/gpu/probe_nsys109.mojo
    nsys stats --report cuda_gpu_kern_sum /tmp/t109.nsys-rep

给出每个核的总耗时（区分 ramp/drain 与真实占用）。"""

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
    s.kv_pos = 0
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, False, True)
    s.ctx.synchronize()
    print("=== MEASURED RUN START ===")
    s.kv_pos = 0
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, False, True)
    print("=== MEASURED RUN END ===")
    model.params.keep_alive()
