"""账本一百一十九-a：跨序列重叠探针（2×GpuSession 交错 span 步）。

依据：①116-c 证伪了单序列 span（sync 被发射流水吸收），但那是
单流视角——两个 session 各持独立 DeviceContext（独立流），交错
发射可让 A 流的尾部与 B 流的头部重叠；②decode t=1 GEMM 网格
饥渴（56 块级），双流并发正好填 SM。
臂 A 串行：A×N（span）→sync→B×N（span）→sync。
臂 B 交错：{A.step(span); B.step(span)}×N → 尾部一次 sync。
固定 token 42 保确定性；每变体 kv 回卷 96 同相位。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime ITERS = 24
comptime REPS = 4
comptime KREWIND = 96


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)
    var b = GpuSession(model, 512)
    for i in range(8):
        _ = a.decode_step[INNER, INNER_FF](42, 0, False, True)
        _ = b.decode_step[INNER, INNER_FF](42, 0, False, True)
    a.ctx.synchronize()
    b.ctx.synchronize()

    var best_serial = 1e30
    var best_inter = 1e30
    for rep in range(REPS):
        a.kv_pos = KREWIND
        b.kv_pos = KREWIND
        var t0 = monotonic_ns()
        for i in range(ITERS):
            _ = a.decode_step[INNER, INNER_FF](42, 0, False, True)
        a.ctx.synchronize()
        for i in range(ITERS):
            _ = b.decode_step[INNER, INNER_FF](42, 0, False, True)
        b.ctx.synchronize()
        var d = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS) * 2.0
        if d < best_serial:
            best_serial = d

        a.kv_pos = KREWIND
        b.kv_pos = KREWIND
        t0 = monotonic_ns()
        for i in range(ITERS):
            _ = a.decode_step[INNER, INNER_FF](42, 0, False, True)
            _ = b.decode_step[INNER, INNER_FF](42, 0, False, True)
        a.ctx.synchronize()
        b.ctx.synchronize()
        d = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS) * 2.0
        if d < best_inter:
            best_inter = d

    print("serial  per-pair ms:", best_serial)
    print("interleaved per-pair ms:", best_inter,
          "  speedup:", best_serial / best_inter)

    # 正确性抽查：交错后两 session 各补一发 sample 步，token 应一致
    a.kv_pos = KREWIND
    b.kv_pos = KREWIND
    _ = a.decode_step[INNER, INNER_FF](42, 0, False, True)
    _ = b.decode_step[INNER, INNER_FF](42, 0, False, True)
    a.ctx.synchronize()
    b.ctx.synchronize()
    _ = a.decode_step[INNER, INNER_FF](42, 0, True)
    _ = b.decode_step[INNER, INNER_FF](42, 0, True)
    print("A token:", a.last_token, " B token:", b.last_token,
          " MATCH" if a.last_token == b.last_token else " MISMATCH")
