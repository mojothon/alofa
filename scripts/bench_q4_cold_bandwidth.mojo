"""冷 cache 下，q4 **真入口**与 fp32 各自吃到的带宽（核级）。

为什么不能直接复用 `bench_decode_roofline.mojo` 的 q4 那档
----------------------------------------------------------
它量的是 `_matmul_q4` / `matmul_q4_f32`（**普通核**），而 decode 的真入口是**行带**
`q4_matmul_rows_k`（`n_tok=1` ⇒ 单行分支）。开关只挂在普通核上时**一次都没执行过**
（2026-09-29 记的那条坑）⇒ 那档数不能拿来判 q4 通路。这里量的是真入口。

第二个必须处理的事：L3 12 MB
-----------------------------
`down_proj` 的 q4 权重只有 2.45 MB，反复调用会一直**驻留在 cache** 里 ⇒ 量到的是算力
不是 DRAM；而真模型每 token 读 346 MB，权重每次进来都是冷的。⇒ 每次计时前先扫一遍
256 MB 的冲刷缓冲，把权重从 cache 里挤出去；顺带把这趟纯读也计时，当作**同轮的冷读
带宽标定**（省得再去翻账本里的老数）。

跑法（`-O2`；`fork()` 在 JIT 下会崩，所以先 build）：

    pixi run mojo build -O2 -I src scripts/bench_q4_cold_bandwidth.mojo -o target/q4cold
    ./target/q4cold                  # mub（默认）
    ALOFA_Q4_MUB=0 ./target/q4cold   # 退回 f32 展开
"""

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import linear as linear_v
from alofa.kernels.cpu.quant import Q4_BLOCK, Q4_BYTES, quantize_q4_0
from alofa.model.arch.qwen import BACKEND_AVX2, q4_matmul_rows_k, q4_mub_enabled

# Qwen2.5-0.5B 的三个真投影形状（`tests/fixtures/qwen2.5-0.5b/config.tsv`）。
comptime HIDDEN = 896
comptime INTER = 4864
comptime KV = 128
comptime FLUSH_FLOATS = 64 * 1024 * 1024  # 256 MB ≫ L3 12 MB
comptime ATTEMPTS = 5


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, shape2(rows, cols), DT_FP32)


def fill(p: F32Ptr, n: Int) -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1


def read_sum(p: F32Ptr, n: Int) -> Float64:
    """把 `n` 个 f32 扫一遍求和 —— 只用来**读**（冷 cache 时这就是纯 DRAM 读）。

    结果必须被用掉，否则会被整段优化掉；调用方把它累加进 sink 并在最后打印。
    """
    var acc = Float64(0)
    var i = 0
    while i < n:
        acc += Float64(p[unsafe_offset=i])
        i += 1
    return acc


def rate_gb_s(nbytes: Float64, ns: Float64) -> Float64:
    return nbytes * 1_000_000_000.0 / ns / 1e9


def bench_shape(name: String, rows: Int, inner: Int, flush_p: F32Ptr) raises -> Float64:
    """一个真形状：`dst[1, rows] = x[1, inner] @ w[rows, inner]ᵀ`（decode 形状）。

    返回 sink（防止读循环被优化掉）。
    """
    var n = rows * inner
    var q4_bytes = n // Q4_BLOCK * Q4_BYTES
    var arena = Arena((n + inner + rows) * 4 + q4_bytes + 64)
    var w = scratch(arena, rows, inner)
    var x = scratch(arena, 1, inner)
    var dst = scratch(arena, 1, rows)
    fill(f32_data(w), n)
    fill(f32_data(x), inner)
    var blocks = arena.alloc(q4_bytes)
    quantize_q4_0(blocks, f32_data(w), n)

    var pd = f32_data(dst)
    var px = f32_data(x)
    var sink = Float64(0)

    # 预热三趟（不计入）：把页表、分支预测带起来。
    linear_v(dst, x, w)
    q4_matmul_rows_k[BACKEND_AVX2](pd, px, blocks, 1, rows, inner, rows)

    var best_fp32 = Int(0)
    var best_q4 = Int(0)
    for attempt in range(ATTEMPTS):
        sink += read_sum(flush_p, FLUSH_FLOATS)
        var t0 = monotonic_ns()
        linear_v(dst, x, w)
        var t1 = monotonic_ns()
        if best_fp32 == 0 or t1 - t0 < best_fp32:
            best_fp32 = t1 - t0

        sink += read_sum(flush_p, FLUSH_FLOATS)
        var t2 = monotonic_ns()
        q4_matmul_rows_k[BACKEND_AVX2](pd, px, blocks, 1, rows, inner, rows)
        var t3 = monotonic_ns()
        if best_q4 == 0 or t3 - t2 < best_q4:
            best_q4 = t3 - t2

    # 同一条通路、同一块权重，**不冲刷**：权重留在 cache 里。
    # 冷/热两个时间差不多 ⇒ 病在核内（发射 / 依赖 / 预取），不在 DRAM。
    var best_hot_fp32 = Int(0)
    var best_hot_q4 = Int(0)
    for attempt in range(ATTEMPTS):
        var h0 = monotonic_ns()
        linear_v(dst, x, w)
        var h1 = monotonic_ns()
        if best_hot_fp32 == 0 or h1 - h0 < best_hot_fp32:
            best_hot_fp32 = h1 - h0

        var h2 = monotonic_ns()
        q4_matmul_rows_k[BACKEND_AVX2](pd, px, blocks, 1, rows, inner, rows)
        var h3 = monotonic_ns()
        if best_hot_q4 == 0 or h3 - h2 < best_hot_q4:
            best_hot_q4 = h3 - h2

    var fp32_bytes = Float64(n * 4)
    var bw_fp32 = rate_gb_s(fp32_bytes, Float64(best_fp32))
    var bw_q4 = rate_gb_s(Float64(q4_bytes), Float64(best_q4))
    print(
        name
        + " rows=" + String(rows) + " inner=" + String(inner)
        + " | fp32 冷 " + String(best_fp32) + " ns " + String(bw_fp32) + " GB/s"
        + " / 热 " + String(best_hot_fp32) + " ns"
        + " (" + String(fp32_bytes / 1e6) + " MB)"
        + "\n      q4   冷 " + String(best_q4) + " ns " + String(bw_q4) + " GB/s"
        + " / 热 " + String(best_hot_q4) + " ns"
        + " (" + String(Float64(q4_bytes) / 1e6) + " MB)"
        + "\n      冷/热 fp32=" + String(Float64(best_fp32) / Float64(best_hot_fp32))
        + " q4=" + String(Float64(best_q4) / Float64(best_hot_q4))
        + " | 带宽比 q4/fp32=" + String(bw_q4 / bw_fp32)
        + " 时间比 fp32/q4=" + String(Float64(best_fp32) / Float64(best_q4))
        + " 字节比=" + String(fp32_bytes / Float64(q4_bytes))
    )
    arena.keep_alive()
    return sink


def main() raises:
    if q4_mub_enabled():
        print("q4 通路：mub（vpmaddubsw 整数点积）")
    else:
        print("q4 通路：f32 展开（ALOFA_Q4_MUB=0）")

    var farena = Arena(FLUSH_FLOATS * 4 + 64)
    var flushv = scratch(farena, FLUSH_FLOATS, 1)
    var flush_p = f32_data(flushv)
    fill(flush_p, FLUSH_FLOATS)

    # 同轮冷读标定：把 256 MB 扫一遍要多久（这就是"纯 DRAM 读"在本轮机器上的速度）。
    var best_read = Int(0)
    var sink = Float64(0)
    for attempt in range(ATTEMPTS):
        var t0 = monotonic_ns()
        sink += read_sum(flush_p, FLUSH_FLOATS)
        var t1 = monotonic_ns()
        if best_read == 0 or t1 - t0 < best_read:
            best_read = t1 - t0
    # ⚠️ 这一趟只是**冲刷用的标量读循环**，不是带宽标定（标量求和本身就慢，实测
    # 3.7 GB/s 远低于探针测到的 12…16 GB/s）。它的作用是保证计时的那一段是冷的。
    print(
        "冲刷缓冲 256 MB 扫一遍：" + String(best_read) + " ns（标量读循环，"
        + "**不是**带宽标定；只用来把权重挤出 cache）"
    )
    print("")

    sink += bench_shape("q_proj/o_proj ", HIDDEN, HIDDEN, flush_p)
    sink += bench_shape("gate/up/down  ", INTER, HIDDEN, flush_p)
    sink += bench_shape("k_proj/v_proj ", KV, HIDDEN, flush_p)

    print("")
    print("unreachable " + String(sink))
    farena.keep_alive()
