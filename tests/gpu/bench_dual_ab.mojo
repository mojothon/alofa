"""dual 形状交错 A/B（账本七十五）。

背景：同参重扫（独立会话顺序跑）里 T128/O8 = 2.572 vs T64/O4 = 2.964，
但 standalone bench_decode 里 T128/O8 = 2.974 纹丝不动 —— 顺序扫型的
排名不可复现到独立进程。本机是共享 GPU（他租户负载随时段波动，
窗口间绝对值漂移 ±0.3ms+），顺序扫型的形状间可比性被负载漂移污染。

裁决法：**逐 token 交错** —— 两形状在同一会话、同一分钟窗口内交替
调用，负载/时钟/kv 历史（kv 差 1 可忽略）完全一致；decode_step 内部
每步 2 次 sync，逐调用 monotonic 计时精确。判 dual 形状差是真是假。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime WARMUP = 50
comptime PAIRS = 50
comptime ROUNDS = 3
comptime MAXT = WARMUP + 2 * PAIRS * ROUNDS + 8


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)
    for i in range(WARMUP):
        _ = session.decode_step[INNER, INNER_FF, 128, 8](TOKEN)
        _ = session.decode_step[INNER, INNER_FF, 64, 4](TOKEN)
    session.ctx.synchronize()

    for r in range(ROUNDS):
        var tot_a = Float64(0)
        var tot_b = Float64(0)
        var best_a = Float64(1e30)
        var best_b = Float64(1e30)
        for i in range(PAIRS):
            var t0 = monotonic_ns()
            _ = session.decode_step[INNER, INNER_FF, 128, 8](TOKEN)
            var ms_a = Float64(monotonic_ns() - t0) / 1e6
            tot_a += ms_a
            if ms_a < best_a:
                best_a = ms_a
            t0 = monotonic_ns()
            _ = session.decode_step[INNER, INNER_FF, 64, 4](TOKEN)
            var ms_b = Float64(monotonic_ns() - t0) / 1e6
            tot_b += ms_b
            if ms_b < best_b:
                best_b = ms_b
        print(
            "round ",
            r,
            ":  T128/O8 mean=",
            tot_a / Float64(PAIRS),
            " best=",
            best_a,
            " | T64/O4 mean=",
            tot_b / Float64(PAIRS),
            " best=",
            best_b,
        )
    model.params.keep_alive()
