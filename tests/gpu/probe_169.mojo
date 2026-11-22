"""账本一百六十九：host 发射路径 A/B——「编译并启动」路径（每发查
编译缓存）vs 预编译 DeviceFunction 直发（跳过查表）。测纯 host 墙钟
（发射排队、不 sync），500 发 × 5 轮取最小。若 B 路径省 ≥0.5µs/发，
生产 390 发/tick 即省 ≥0.2ms/tick——值得全面改造发射点。

来源：https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceContext.md
—— compile_function（一次，declared_arg_types 编译期类型检查）+
enqueue_function(DeviceFunction, ...)（多次，无查表）为官方支持的
「编译一次、多次 launch」路径。
"""

from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.kernels.cuda.basic import rmsnorm_gpu


def main() raises:
    var ctx = DeviceContext()
    comptime T = 28
    comptime INNER = 896
    comptime N = 500
    comptime ROUNDS = 5

    var dx = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dw = ctx.enqueue_create_buffer[DType.float32](INNER)
    var dnorm = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dnorm16 = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    ctx.enqueue_memset[DType.float32](dx, 1.0)
    ctx.enqueue_memset[DType.float32](dw, 1.0)
    ctx.synchronize()

    # B 路径：预编译一次（declared_arg_types 自动推断）
    var df = ctx.compile_function[rmsnorm_gpu]()

    # 预热两路径
    ctx.enqueue_function[rmsnorm_gpu](
        dx.unsafe_ptr(), dw.unsafe_ptr(), dnorm.unsafe_ptr(),
        dnorm16.unsafe_ptr(), Int32(INNER), Int32(T), Float32(1e-5),
        grid_dim=(T, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function(df,
        dx.unsafe_ptr(), dw.unsafe_ptr(), dnorm.unsafe_ptr(),
        dnorm16.unsafe_ptr(), Int32(INNER), Int32(T), Float32(1e-5),
        grid_dim=(T, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    var best_a = Float64(1e30)
    var best_b = Float64(1e30)
    for _r in range(ROUNDS):
        # A：编译并启动路径
        var t0 = monotonic_ns()
        for _k in range(N):
            ctx.enqueue_function[rmsnorm_gpu](
                dx.unsafe_ptr(), dw.unsafe_ptr(), dnorm.unsafe_ptr(),
                dnorm16.unsafe_ptr(), Int32(INNER), Int32(T), Float32(1e-5),
                grid_dim=(T, 1, 1), block_dim=(256, 1, 1),
            )
        var t1 = monotonic_ns()
        ctx.synchronize()
        var us_a = Float64(t1 - t0) / 1e3 / N
        if us_a < best_a:
            best_a = us_a

        # B：预编译直发
        t0 = monotonic_ns()
        for _k in range(N):
            ctx.enqueue_function(df,
                dx.unsafe_ptr(), dw.unsafe_ptr(), dnorm.unsafe_ptr(),
                dnorm16.unsafe_ptr(), Int32(INNER), Int32(T), Float32(1e-5),
                grid_dim=(T, 1, 1), block_dim=(256, 1, 1),
            )
        t1 = monotonic_ns()
        ctx.synchronize()
        var us_b = Float64(t1 - t0) / 1e3 / N
        if us_b < best_b:
            best_b = us_b

    # C：每 tick compile 一次（缓存命中成本）+ 直发
    var best_c = Float64(1e30)
    var best_cc = Float64(1e30)
    for _r in range(ROUNDS):
        var t0 = monotonic_ns()
        var df2 = ctx.compile_function[rmsnorm_gpu]()
        var t1 = monotonic_ns()
        ctx.synchronize()
        var us_cc = Float64(t1 - t0) / 1e3
        if us_cc < best_cc:
            best_cc = us_cc
        t0 = monotonic_ns()
        for _k in range(N):
            ctx.enqueue_function(df2,
                dx.unsafe_ptr(), dw.unsafe_ptr(), dnorm.unsafe_ptr(),
                dnorm16.unsafe_ptr(), Int32(INNER), Int32(T), Float32(1e-5),
                grid_dim=(T, 1, 1), block_dim=(256, 1, 1),
            )
        t1 = monotonic_ns()
        ctx.synchronize()
        var us_c = Float64(t1 - t0) / 1e3 / N
        if us_c < best_c:
            best_c = us_c
    print("C compile-then-launch =", best_c, "us/launch (host)")
    print("  (compile_function one-shot =", best_cc, "us incl sync)")

    print("A compile-per-launch =", best_a, "us/launch (host)")
    print("B precompiled        =", best_b, "us/launch (host)")
    print("delta =", best_a - best_b, "us/launch -> x390/tick =",
          (best_a - best_b) * 390, "us/tick")
    print("probe169 done")
