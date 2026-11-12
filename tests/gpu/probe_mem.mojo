"""账本一百二十七：per-session 显存实量——1 session 落地后停顿，
由外部 nvidia-smi 采样；再建 3 session 停顿，采样差额。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"


def gen_load(mut sess: GpuSession) raises:
    """B=8 × 24 token 真实负载——触碰权重/KV/激活，物理页提交。"""
    var f8 = List[Int]()
    var l8 = List[Int]()
    var t8 = 3000
    for i in range(8):
        var li = 6 + i
        l8.append(li)
        for _r in range(li):
            f8.append(t8)
            t8 += 1
    var out = sess.generate_batch[
        896, 4864, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32
    ](f8, l8, 24, 200000, List[Int](), List[Int]())
    var n = 0
    for i in range(8 * 24):
        if out[i] != -1:
            n += 1
    print("  load useful:", n, "/192")


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    print("PHASE1: 1 session + load")
    var a = GpuSession(model, 512)
    _ = gen_load(a)
    print("phase1 holding 150s")
    var _t1 = monotonic_ns() + 150_000_000_000
    while monotonic_ns() < _t1:
        pass
    print("PHASE2: 3 sessions + loads")
    var b = GpuSession(model, 512)
    var c = GpuSession(model, 512)
    _ = gen_load(b)
    _ = gen_load(c)
    print("phase2 holding 150s")
    var _t2 = monotonic_ns() + 150_000_000_000
    while monotonic_ns() < _t2:
        pass
    print("DONE")
