"""batch-1 decode 卡在带宽还是算力：三个数现场测出来，交给 roofline 去判。

结论**不在这里写死**，这里只产出被同一台机、同一分钟实测喂进去的两个峰值与两个真
形状的 `Counter`，判定交给 `Roofline.bottleneck` —— 它是**交叉相乘比大小**，不
依赖任何容差，所以不会出现"放宽一点就两边都成立"的情况。

为什么峰值必须现测，不能写死
----------------------------
roofline 的两条天花板是**调用方传入**的（见 `verify/roofline.mojo`）。写死一个
"这台机大概 X GB/s"就把结论变成了写进数字里的偏见：填得低就人人带宽受限，填得高
就人人算力受限。所以两个峰值都由本文件实测：

  * **带宽峰值 B** —— 顺序扫一个远大于 L3 的 fp32 数组求和。选这个形态是因为它
    就是 GEMV 读权重的形式（一段连续内存读一遍、每个元素只用一次），比 STREAM 的
    copy/triad 更贴近这里要判的东西。
  * **算力峰值 P** —— 权重**常驻 cache** 时同一份 `linear` 能跑到多少 GFLOP/s。
    ⚠️ 这是一个**下界**：连 L2/L3 带宽都还可能参与限制，真实上限只会更高。用下界
    当天花板会让"算力受限"**更容易**被判出来 —— 它不袒护"带宽受限"的结论。

还要看的一个比
--------------
除了 bottleneck，每条都额外印 `read_only_ns / me_asured_ns`：**把这块权重原样纯读
一遍需要的时间 ÷ 实际跑完的时间**。接近 1 说明算术被访存完全盖住 —— 那才是"带宽
受限"最直白的说法，不依赖任何模型。

跑法（`-O2`：比 `-O0` 才谈得上毫秒级 timing）：

    pixi run mojo run -O2 -I src scripts/bench_decode_roofline.mojo
"""

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import linear as linear_v
from alofa.kernels.cpu.avx2 import matmul_q4_f32 as matmul_q4_f32_v
from alofa.kernels.cpu.avx2 import _matmul_q4
from alofa.kernels.cpu.quant import (
    Q4_BLOCK,
    Q4_BYTES,
    matmul_q4_f32,
    quantize_q4_0,
)
from alofa.kernels.cpu.scalar import linear as linear_s
from alofa.verify.roofline import Counter, Roofline, bottleneck_name

# Qwen2.5-0.5B（tests/fixtures/qwen2.5-0.5b/config.tsv）—— 只用来挑被测量的形状。
comptime HIDDEN = 896
comptime VOCAB = 151936
comptime INTER = 4864


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, shape2(rows, cols), DT_FP32)


def fill(p: F32Ptr, n: Int) -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = Float32(i % 17) - Float32(8)
        i += 1


def rate_per_s(count: Float64, ns: Float64) -> Float64:
    """`count` 个东西在 `ns` 纳秒里完成 → 每秒多少个。"""
    return count * 1_000_000_000.0 / ns


def bench_read_bandwidth() raises -> Float64:
    """顺序读带宽峰值 B（bytes/s），取三次最快的一次。

    远大于 L3（本地 12 MB），所以每次访问都必须真的走到 DRAM。
    """
    var elems = 64 * 1024 * 1024  # 256 MB of fp32
    var arena = Arena(elems * 4)
    var p = f32_data(scratch(arena, 1, elems))
    fill(p, elems)
    var best = Float64(0)
    var sink = Float64(0)
    for attempt in range(3):
        var acc = SIMD[DType.float32, 8](Float32(0))
        var t0 = monotonic_ns()
        var j = 0
        while j < elems:
            acc += p.unsafe_load[width=8](j)
            j += 8
        var t1 = monotonic_ns()
        sink += Float64(acc.reduce_add())
        var bw = rate_per_s(Float64(elems * 4), Float64(t1 - t0))
        if bw > best:
            best = bw
    if sink > Float64(1e30):
        print("unreachable " + String(sink))
    print("peak.bandwidth_bytes_per_s " + String(Int(best)))
    arena.keep_alive()
    return best


def bench_compute_peak() raises -> Float64:
    """权重常驻 cache 时的 GFLOP/s —— 算力天花板的**下界**，见模块 docstring。

    256×256 的 fp32 权重是 256 KB：装得进 L2 / L3，反复调用不再有 DRAM 流量。留意
    打印出来的数是否超过 AVX2 的理论上限（约 96 GFLOP/s f32、48 GFLOP/s f64；本
    kernel 走 f64 累加） —— 超了说明编译器把这个循环优化掉了，那个数就不能用。
    """
    var inner = 256
    var cols = 256
    var reps = 20000
    var arena = Arena((cols * inner + inner + cols) * 4)
    var w = scratch(arena, cols, inner)
    var x = scratch(arena, 1, inner)
    var dst = scratch(arena, 1, cols)
    fill(f32_data(w), cols * inner)
    fill(f32_data(x), inner)
    var pd = f32_data(dst)
    linear_v(dst, x, w)  # 预热：先把页表和 cache 带起来，这一趟不计进我能看到的数
    var best = Float64(0)
    var sink = Float64(0)
    for attempt in range(3):
        var t0 = monotonic_ns()
        for r in range(reps):
            linear_v(dst, x, w)
            # 读一点结果，别让重复的计算被当成可消除的死代码。一次 load 相对
            # 2*256*256 次乘加可以忽略，但它让「跑完了」这件事不可省略。
            sink += Float64(pd[unsafe_offset=0])
        var t1 = monotonic_ns()
        var per_call = Float64(t1 - t0) / Float64(reps)
        var gf = rate_per_s(Float64(2 * cols * inner), per_call)
        if gf > best:
            best = gf
    if sink > Float64(1e30):
        print("unreachable " + String(sink))
    print("peak.flops_per_s " + String(Int(best)) + "  (下限；权重 256KB 常驻 cache)")
    arena.keep_alive()
    return best


def bench_fma_peak() raises -> Float64:
    """不受依赖链限制的 f64 FMA 吞吐（GFLOP/s）—— **机器**天花板的近似。

    `_gemm` 的累加器只有一条链：`acc += xv * wv` 必须等上一拍的 `acc` 算完，于是
    吞吐被 FMA 的**延迟**按住，而不是被 FMA 的**数量**按住 —— 那是 kernel 的性质，
    不是机器的性质。这里开 8 条互不依赖的链，让 FMA 单元自己成为唯一限制。

    这个数与 `bench_compute_peak` 的差，就是"现在的 kernel 在 ALU 上留了多少"。
    """
    var iters = 50_000_000
    var one = SIMD[DType.float64, 4](Float64(1.0000001))
    var a0 = SIMD[DType.float64, 4](Float64(0))
    var a1 = SIMD[DType.float64, 4](Float64(0))
    var a2 = SIMD[DType.float64, 4](Float64(0))
    var a3 = SIMD[DType.float64, 4](Float64(0))
    var a4 = SIMD[DType.float64, 4](Float64(0))
    var a5 = SIMD[DType.float64, 4](Float64(0))
    var a6 = SIMD[DType.float64, 4](Float64(0))
    var a7 = SIMD[DType.float64, 4](Float64(0))
    var best = Float64(0)
    for attempt in range(3):
        var t0 = monotonic_ns()
        var i = 0
        while i < iters:
            a0 += one * one
            a1 += one * one
            a2 += one * one
            a3 += one * one
            a4 += one * one
            a5 += one * one
            a6 += one * one
            a7 += one * one
            i += 1
        var t1 = monotonic_ns()
        # 每条链 8 次 flop（4 通道 × 乘加），8 条链 → 每轮 64 次 flop。
        var gf = rate_per_s(Float64(iters * 64), Float64(t1 - t0))
        if gf > best:
            best = gf
    var sink = Float64(
        (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7).reduce_add()
    )
    if sink > Float64(1e30):
        print("unreachable " + String(sink))
    print("machine.flops_per_s " + String(Int(best)) + "  (8 条独立累加链)")
    return best


def bench_shape(
    machine_roof: Roofline,
    kernel_roof: Roofline,
    name: String,
    cols: Int,
    inner: Int,
    read_peak: Float64,
) raises -> None:
    """一个真形状：`dst[1, cols] = x[1, inner] @ w[cols, inner]ᵀ`。

    算术强度**恒为 0.5 flops/byte**：批大小为 1 时每个 fp32 权重必须被读一次、也
    只被用一次 —— 没有任何分块技巧能绕开这件事，这里把它印出来是为了让"是不是
    刚好只有这个 kernel 不行"这个疑问当场消掉。
    """
    var arena = Arena((cols * inner + inner + cols) * 4)
    var w = scratch(arena, cols, inner)
    var x = scratch(arena, 1, inner)
    var dst = scratch(arena, 1, cols)
    fill(f32_data(w), cols * inner)
    fill(f32_data(x), inner)
    linear_v(dst, x, w)  # 预热

    var best_ns = Int(0)
    for attempt in range(3):
        var t0 = monotonic_ns()
        linear_v(dst, x, w)
        var t1 = monotonic_ns()
        if best_ns == 0 or t1 - t0 < best_ns:
            best_ns = t1 - t0

    var nbytes = cols * inner * 4
    var nflops = 2 * cols * inner
    var counter = Counter(name, nbytes, nflops, best_ns)
    # 把这块权重原样纯读一遍要多久：访存时间的地板，不含任何算术。
    var read_only_ns = Float64(nbytes) / read_peak * 1_000_000_000.0

    print(
        name
        + ": bytes="
        + String(nbytes)
        + " flops="
        + String(nflops)
        + " ns="
        + String(best_ns)
        + " | AI="
        + String(Float64(nflops) / Float64(nbytes))
        + " flops/byte"
        + " | bw="
        + String(counter.bandwidth_bytes_per_s())
        + " B/s = "
        + String(machine_roof.bandwidth_utilization_permille(counter))
        + "‰ of read peak"
        + " | compute="
        + String(counter.flops_per_s())
        + " FLOP/s = "
        + String(kernel_roof.compute_utilization_permille(counter))
        + "‰ of what this kernel reaches"
        + " ("
        + String(machine_roof.compute_utilization_permille(counter))
        + "‰ of the machine)"
        + " | 纯读同样字节/实测="
        + String(read_only_ns / Float64(best_ns))
        + " | verdict: vs 机器 = "
        + bottleneck_name(machine_roof.bottleneck(counter))
        + " ; vs 当下 kernel = "
        + bottleneck_name(kernel_roof.bottleneck(counter))
    )
    arena.keep_alive()


def q4_line(
    label: String, ns: Int, nbytes: Int, nflops: Int, read_peak: Float64
) -> String:
    """一条通路的结果行：它读的字节、达到的算力，以及它**自己**的访存地板。

    `floor_ns / ns` 那一列是这里唯一要盯的东西：把它当成"算术被访存盖住的程度"。
    接近 1 说明这条路已经在 bw 上限上；远小于 1 说明卡在算术 —— 而且是**这条通路
    自己**的算术，跟别的通路没关系。
    """
    var floor_ns = Float64(nbytes) / read_peak * 1_000_000_000.0
    return (
        "\n    "
        + label
        + " ns="
        + String(ns)
        + " | "
        + String(rate_per_s(Float64(nbytes), Float64(ns)) / 1e9)
        + " GB/s"
        + " | "
        + String(rate_per_s(Float64(nflops), Float64(ns)) / 1e9)
        + " GFLOP/s"
        + " | 自己的访存地板/实测 = "
        + String(floor_ns / Float64(ns))
        + " (1.0 = 算术完全被访存盖住)"
    )


def bench_q4_shape(
    name: String, rows: Int, inner: Int, read_peak: Float64
) raises -> None:
    """同一块真权重，三条通路各自的时间，以及各自**自己的**访存地板。

    加这一档是因为端到端先给了个反直觉的结果：Q4 对 fp32 只快 1.06–1.14×（标量档），
    而按「字节少了 7.1×」本该接近 2.76×。要判断"少的那部分时间去哪了"，就得把同一
    块矩阵的三条通路放在同一台机上量：

      * `fp32/avx2`   —— `linear_v`
      * `fp32/scalar` —— `linear`
      * `q4_0/scalar` —— `matmul_q4_f32`（`quant.mojo`，标量）
      * `q4_0/avx2`   —— 默认通路（`_matmul_q4_halves`：按半块切 16 值、f32 累加）
      * `q4_0/f64-8w` —— `_matmul_q4`（按块内 `j%8` 切 f64 通道，与标量版逐位一致）

    每条都按**它自己读的字节**算地板（fp32 是 4 字节/权重，q4 是 4.5 bit/权重），
    所以三条地板的高度差就是量化真正省下的访存时间；实测高出地板多少，就是该通路
    自己的算术开销 —— 高得越多说明"已经不在bw瓶颈上"这件事越确定。
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

    # 预热三趟，不计入。
    linear_v(dst, x, w)
    linear_s(dst, x, w)
    matmul_q4_f32(pd, px, blocks, rows, inner)
    matmul_q4_f32_v(pd, px, blocks, rows, inner)
    _matmul_q4(pd, px, blocks, rows, inner, pd, False)

    var best_vec = Int(0)
    var best_scl = Int(0)
    var best_q4 = Int(0)
    var best_q4v = Int(0)
    var best_q4h = Int(0)
    var sink = Float64(0)
    for attempt in range(3):
        var t0 = monotonic_ns()
        linear_v(dst, x, w)
        var t1 = monotonic_ns()
        linear_s(dst, x, w)
        var t2 = monotonic_ns()
        matmul_q4_f32(pd, px, blocks, rows, inner)
        var t3 = monotonic_ns()
        matmul_q4_f32_v(pd, px, blocks, rows, inner)
        var t4 = monotonic_ns()
        _matmul_q4(pd, px, blocks, rows, inner, pd, False)
        var t5 = monotonic_ns()
        sink += Float64(pd[unsafe_offset=0])
        if best_vec == 0 or t1 - t0 < best_vec:
            best_vec = t1 - t0
        if best_scl == 0 or t2 - t1 < best_scl:
            best_scl = t2 - t1
        if best_q4 == 0 or t3 - t2 < best_q4:
            best_q4 = t3 - t2
        if best_q4v == 0 or t4 - t3 < best_q4v:
            best_q4v = t4 - t3
        if best_q4h == 0 or t5 - t4 < best_q4h:
            best_q4h = t5 - t4
    if sink > Float64(1e30):
        print("unreachable " + String(sink))

    print(
        name
        + " rows="
        + String(rows)
        + " inner="
        + String(inner)
        + " fp32="
        + String(n * 4)
        + "B q4="
        + String(q4_bytes)
        + "B"
        + q4_line("fp32/avx2  ", best_vec, n * 4, 2 * n, read_peak)
        + q4_line("fp32/scalar", best_scl, n * 4, 2 * n, read_peak)
        + q4_line("q4_0/scalar", best_q4, q4_bytes, 2 * n, read_peak)
        + q4_line("q4_0/avx2  ", best_q4v, q4_bytes, 2 * n, read_peak)
        + q4_line("q4_0/f64-8w ", best_q4h, q4_bytes, 2 * n, read_peak)
    )
    arena.keep_alive()


def main() raises:
    var read_peak = bench_read_bandwidth()
    var kernel_flops = bench_compute_peak()
    var machine_flops = bench_fma_peak()
    var kernel_roof = Roofline(Int(read_peak), Int(kernel_flops))
    var machine_roof = Roofline(Int(read_peak), Int(machine_flops))
    print(
        "balance_point vs 机器 "
        + String(machine_flops / read_peak)
        + " flops/byte"
        + " | vs 当下 kernel "
        + String(kernel_flops / read_peak)
        + " flops/byte —— 强度低于平衡点才是带宽受限"
    )
    bench_shape(machine_roof, kernel_roof, "hidden_proj", HIDDEN, HIDDEN, read_peak)
    bench_shape(machine_roof, kernel_roof, "lm_head", VOCAB, HIDDEN, read_peak)
    # 真实的前馈 `down_proj`：4864×896 —— 每层三块大矩阵里最大的一块。
    bench_q4_shape("down_proj", INTER, HIDDEN, read_peak)
