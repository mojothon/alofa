"""账本一百零八-c：prefill 是 host 发射受限还是 GPU 受限？
sync_tail=False 时返回时刻 = 纯 host enqueue 时间；随后 synchronize
的增量 = GPU 实际执行。若 enqueue ≈ 总时间 ⇒ 主机侧发射是瓶颈
（核级优化白费）。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)
    var s = GpuSession(model, 512)
    # 预热
    s.kv_pos = 0
    _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, False, True)
    s.ctx.synchronize()

    for k in range(3):
        s.kv_pos = 0
        var t0 = monotonic_ns()
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32](ids, False, False)
        var t1 = monotonic_ns()
        s.ctx.synchronize()
        var t2 = monotonic_ns()
        print("rep", k,
              " host-enqueue ms:", Float64(t1 - t0) / 1e6,
              " gpu-tail ms:", Float64(t2 - t1) / 1e6,
              " total ms:", Float64(t2 - t0) / 1e6)
    model.params.keep_alive()
