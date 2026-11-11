"""账本一百零一：attn TILES 扫描（8/4/2/1×两轮正反序）。每臂新 session、
4 发 t=128（kv 128→512），报 call1-3（call0 为暖机）。"""
from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128


def arm[TILES: Int](mut model: QwenForward, ids: List[Int], tag: String) raises:
    var s = GpuSession(model, 512)
    for k in range(4):
        var t0 = monotonic_ns()
        _ = s.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, TILES, 8, DType.float32](ids, False, True)
        s.ctx.synchronize()
        print(tag, k, Float64(monotonic_ns() - t0) / 1e6, "ms")


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)
    # 正序
    arm[8](model, ids, "T8")
    arm[4](model, ids, "T4")
    arm[2](model, ids, "T2")
    arm[1](model, ids, "T1")
    # 反序（抗环境漂移）
    arm[1](model, ids, "U1")
    arm[2](model, ids, "U2")
    arm[4](model, ids, "U4")
    arm[8](model, ids, "U8")
    model.params.keep_alive()
