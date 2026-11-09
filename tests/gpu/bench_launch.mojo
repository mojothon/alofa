"""launch 开销隔离基准（账本六十四）。

decode_step 未过 2ms 判据（10.2ms/token），每步 ~592 次发射 —— 本基准把
host 端 enqueue 成本与 GPU 执行分开：小 kernel 连发 1000 次不 sync（测
enqueue 吞吐），再 sync（测真实执行）。若 enqueue ≥ 10µs/次，则 decode
瓶颈在 host 而非 GPU，优化方向是减发射数而不是调 kernel。
"""

from std.math import ceildiv
from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.kernels.cuda.elementwise import copy_gpu

comptime WARMUP = 100
comptime ITERS = 1000
comptime ROUNDS = 3
comptime N = 128


def main() raises:
    var ctx = DeviceContext()
    var src = ctx.enqueue_create_buffer[DType.float32](N)
    var dst = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.synchronize()

    for i in range(WARMUP):
        ctx.enqueue_function[copy_gpu](
            dst.unsafe_ptr(),
            src.unsafe_ptr(),
            Int32(N),
            grid_dim=(1, 1, 1),
            block_dim=(64, 1, 1),
        )
    ctx.synchronize()

    var best_enqueue = Float64(1e30)
    var best_total = Float64(1e30)
    for r in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(ITERS):
            ctx.enqueue_function[copy_gpu](
                dst.unsafe_ptr(),
                src.unsafe_ptr(),
                Int32(N),
                grid_dim=(1, 1, 1),
                block_dim=(64, 1, 1),
            )
        var t1 = monotonic_ns()
        ctx.synchronize()
        var t2 = monotonic_ns()
        var us = Float64(t1 - t0) / 1e3 / Float64(ITERS)
        var tot = Float64(t2 - t0) / 1e3 / Float64(ITERS)
        if us < best_enqueue:
            best_enqueue = us
        if tot < best_total:
            best_total = tot

    print(
        "enqueue-only: ",
        best_enqueue,
        " us/launch (best of ",
        ROUNDS,
        " rounds, ",
        ITERS,
        " iters)",
    )
    print("enqueue+exec: ", best_total, " us/launch")
    print("592 launches/step would cost: ", best_enqueue * 592.0 / 1000.0, " ms host-side")
    ctx.synchronize()
