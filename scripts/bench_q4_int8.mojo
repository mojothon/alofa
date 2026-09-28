"""q4_0 权重 × int8 激活的**整数点积**：这条路走得通吗，当场量。

为什么突然有这条路
----------------
`bench_decode_roofline.mojo` 把 q4 层核钉在了**算术受限**上：`down_proj` 4864×896，
fp32/avx2 读 17.4 MB 跑 11.47 GB/s，而 q4_0/avx2 读 2.45 MB（八倍小于 L3，已在 cache
里）只跑 **2.11 GB/s**。块流在 cache 里跑输给 fp32 从内存跑 —— 它不是带宽受限，是**每
个值的算术太贵**：现在是 `vpmovsxbd`(int8→int32) + `vcvtdq2ps`(→f32) + `vmulps`(×d)
+ `vfmadd` 四条一组。

整数点积换掉的是这套算术：`vpmaddwd` 一条指令做 **16 个 int16 乘加**（相邻对相乘后
相加），每块的 32 个乘积被它压成 8 个 int32 —— 这 8 个通道再转 f32 乘一次该块的 `d`。
对比之下，f32 fma 一条只推进 8 个点积各 1 项。

⚠️ 为什么是 `pmadd.wd` 而不是 `pmaddubsw`
----------------------------------------
llama.cpp 的 AVX2 路径（`ggml_vec_dot_q4_0_q8_0`）在有 AVX-VNNI 时用 `vpdpbssd`，没有时
用 `vpmaddubsw` 加一套符号技巧。本机（i7-9700K，仅 AVX2）走后者那条。但 Mojo 1.0 能叫到
的 LLVM intrinsic 名实测只有 **`llvm.x86.avx2.pmadd.wd`**（`pmaddwd`）—— `pmaddubsw` 的
几种命名变体本机都报"找不到"（见账本 2026-09-28 那笔）。所以这里走 **int16 路径**：
int8 先 widen 成 int16（`vpmovsxbw`，4 条/块），再交给 `pmadd.wd`。它比 llama.cpp 多付
那 4 条 widen，这是本方案**已知的劣势**，量出来才知道值不值。

判据不是"快了就上线"
--------------------
* **精度**：整数点积本身**精确**（整数加法没有舍入），误差只来自两处量化 —— 权重
  q4_0（4 bit）与激活 int8（8 bit）。参照 `pixi run test-q8-head` 那条门：q8_0 输出投影的
  logits 相对 L2 是 **0.77%**，这里同量级才算站得住。
* **参照是独立另写的**：参考值用**未量化的 fp32 权重**与 fp32 激活做点积，`Float64`
  累加 —— 它不复用被测核的任何一行，也不复用量化器。
* 本机长期过载 ⇒ 取**中位数**，且只作**同轮相对比较**。

跑法（`-O2`：`-O0` 下 `pmadd.wd` 的调度没有参考价值）
----------------------------------------------------
    pixi run mojo build -O2 -I src scripts/bench_q4_int8.mojo -o target/bench_q4_int8
    ./target/bench_q4_int8
"""

from std.math import round
from std.sys import llvm_intrinsic

from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr
from alofa.kernels.cpu.avx2 import Q4_U8, matmul_q4_f32
from alofa.kernels.cpu.quant import (
    Q4_BLOCK,
    Q4_BYTES,
    Q4_SCALE_BYTES,
    U8Ptr,
    block_scale,
    quantize_q4_0,
)

# 真实的 `down_proj`：每层三块大矩阵里最大的一块，也是 roofline 量过的那个形状。
comptime ROWS = 4864
comptime COLS = 896
comptime NB = COLS // Q4_BLOCK
# 本机过载 ⇒ 少跑几趟、看中位数。
comptime REPS = 5
comptime WARMUP = 2

# 实测能叫到的名字（pmaddubsw 的各种拼法本机都找不到，见文件头）。
comptime MADDWD = "llvm.x86.avx2.pmadd.wd"

alias I8x16 = SIMD[DType.int8, 16]

# 激活的 int8 缓冲：直接按 `Int8` 取指针，`unsafe_load` 即得符号扩展 —— 若按 `UInt8`
# 取再 `cast[DType.int8]()`，补码字节（如 200 = −56）会被**饱和成 +127**（Mojo 的
# 整数 `cast` 是数值转换，不是位重解释），负值全废。
comptime I8Ptr = Pointer[Int8, MutUntrackedOrigin]
comptime I16Ptr = Pointer[Int16, MutUntrackedOrigin]
alias U8x16 = SIMD[DType.uint8, 16]
alias I16x16 = SIMD[DType.int16, 16]
alias I32x8 = SIMD[DType.int32, 8]
alias F32x8 = SIMD[DType.float32, 8]


def median_ns(mut steps: List[Int]) -> Int:
    """中位数（就地插入排序；样本小，够用）。"""
    var i = 1
    while i < len(steps):
        var v = steps[i]
        var j = i - 1
        while j >= 0 and steps[j] > v:
            steps[j + 1] = steps[j]
            j -= 1
        steps[j + 1] = v
        i += 1
    return steps[len(steps) // 2]


def f2(x: Float64) -> String:
    """两位小数。本机没有 sprintf，整数拼是唯一不引入新依赖的写法。"""
    var v = x
    var whole = Int(v)
    var frac = Int((v - Float64(whole)) * Float64(100) + Float64(0.5))
    if frac >= 100:
        whole += 1
        frac -= 100
    var s = String(whole) + "."
    if frac < 10:
        s = s + "0"
    return s + String(frac)


def q4_int8_dot(
    dst: F32Ptr,
    xq: I8Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    sx: Float32,
) raises:
    """`dst = W · (sx · xq)`，`W` 是 q4_0 块流，`xq` 是 int8 激活（补码字节）。

    每块 32 个值：nibble → int8(−8..7) → widen 成 int16 → `pmadd.wd` 压成 8 个 int32
    → 转 f32 乘该块的 `d` 与激活的 `sx` 累加。整数部分**精确**，误差只来自量化。
    """
    var nb = cols // Q4_BLOCK
    var m4 = U8x16(0x0F)
    var eight = I8x16(8)
    for r in range(rows):
        var acc = F32x8(0)
        for b in range(nb):
            var base = (r * nb + b) * Q4_BYTES
            var qs = blocks.unsafe_load[width=16](base + Q4_SCALE_BYTES)
            # 低 nibble = w[j]，高 nibble = w[j+16]；值 = nibble − 8。
            var lo = ((qs & m4).cast[DType.int8]()) - eight
            var hi = (((qs >> U8x16(4)) & m4).cast[DType.int8]()) - eight
            var y0 = xq.unsafe_load[width=16](b * Q4_BLOCK + 0)
            var y1 = xq.unsafe_load[width=16](b * Q4_BLOCK + 16)
            var p0 = llvm_intrinsic[MADDWD, I32x8](lo.cast[DType.int16](), y0.cast[DType.int16]())
            var p1 = llvm_intrinsic[MADDWD, I32x8](hi.cast[DType.int16](), y1.cast[DType.int16]())
            var d = block_scale(blocks, base)
            acc = (p0 + p1).cast[DType.float32]() * F32x8(d * sx) + acc
        dst[unsafe_offset=r] = acc.reduce_add()


def rel_l2(a: F32Ptr, b: F32Ptr, n: Int) -> Float64:
    """`‖a−b‖ / ‖b‖`。分母用参照物，不是被测值 —— 否则误差大反而显得小。"""
    var num = Float64(0)
    var den = Float64(0)
    for i in range(n):
        var av = Float64(a[unsafe_offset=i])
        var bv = Float64(b[unsafe_offset=i])
        num += (av - bv) * (av - bv)
        den += bv * bv
    if den == Float64(0):
        return Float64(0)
    var q = num / den
    # 手开平方：本机 std.math 有 sqrt，但为的是不引入第二条依赖路径，牛顿法两步足够。
    var g = Float64(1)
    for _ in range(60):
        g = (g + q / g) * Float64(0.5)
    return g


def main() raises:
    print("")
    print("== q4_0 × int8 整数点积 vs 现在的 f32 解量化核 ==")
    print("   down_proj " + String(ROWS) + "×" + String(COLS) + "；backend avx2")

    var arena = Arena(ROWS * COLS * 4 + ROWS * NB * Q4_BYTES + ROWS * 4 * 4 + COLS * 8)
    var w = arena.alloc(ROWS * COLS * 4).unsafe_bitcast[Float32]()
    var blocks = arena.alloc(ROWS * NB * Q4_BYTES).unsafe_bitcast[UInt8]()
    var x = arena.alloc(COLS * 4).unsafe_bitcast[Float32]()
        # ⚠️ 存 **int8**（补码）。试过直接存 widen 好的 int16（想让循环内每块省掉 2 条
    # `vpmovsxbw`），实测 **更慢**（0.96× vs 1.07×）：省下的 2 条 widen 抵不过激活
    # 体积翻倍带来的 L1 压力。所以这里保持紧凑的 int8，widen 留在循环里做。
    var xq = arena.alloc(COLS).unsafe_bitcast[Int8]()
    var dst_f32 = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    var dst_int = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    var refv = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()

    # 权重：确定性 LCG，值域 ±0.3（与真实权重一个量级）。
    var seed = UInt64(12345)
    for i in range(ROWS * COLS):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(Int((seed >> 40) & UInt64(0xFFFF))) / Float32(32768.0)
        w[unsafe_offset=i] = (u - Float32(1.0)) * Float32(0.3)

    # 激活：同样确定性。
    seed = UInt64(777)
    for i in range(COLS):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(Int((seed >> 40) & UInt64(0xFFFF))) / Float32(32768.0)
        x[unsafe_offset=i] = (u - Float32(1.0)) * Float32(0.5)

    # 压成 q4_0 块流（与生产路径同一个量化器）。
    # ⚠️ 一次调用整块矩阵而不是逐行：COLS=896 是 Q4_BLOCK 的整数倍，块边界**不会跨行**，
    # 所以这与逐行量化**逐字节相同**。省这一步只是绕开 `offset`，不是改语义。
    quantize_q4_0(blocks, w, ROWS * COLS)

    # 激活压成 int8：sx = amax/127，与 q8_0 同法。
    var amax = Float32(0)
    for i in range(COLS):
        var av = x[unsafe_offset=i]
        if av < Float32(0):
            av = -av
        if av > amax:
            amax = av
    var sx = amax / Float32(127.0)
    for i in range(COLS):
        var q = Int(round(Float64(x[unsafe_offset=i]) / Float64(sx)))
        if q > 127:
            q = 127
        if q < -128:
            q = -128
        xq[unsafe_offset=i] = Int8(q)

    # 参照物：未量化的 fp32 权重 × fp32 激活，fp64 累加。**不复用**上面任何一个核。
    for r in range(ROWS):
        var s = Float64(0)
        for c in range(COLS):
            s += Float64(w[unsafe_offset=r * COLS + c]) * Float64(x[unsafe_offset=c])
        refv[unsafe_offset=r] = Float32(s)

    # ① 现在的核：q4 → f32 解量化 + fma
    for _ in range(WARMUP):
        matmul_q4_f32(dst_f32, x, blocks, ROWS, COLS)
    var t_f32 = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        matmul_q4_f32(dst_f32, x, blocks, ROWS, COLS)
        t_f32.append(monotonic_ns() - t0)

    # ② 整数点积
    for _ in range(WARMUP):
        q4_int8_dot(dst_int, xq, blocks, ROWS, COLS, sx)
    var t_int = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        q4_int8_dot(dst_int, xq, blocks, ROWS, COLS, sx)
        t_int.append(monotonic_ns() - t0)

    var ns_f32 = median_ns(t_f32)
    var ns_int = median_ns(t_int)
    var bytes = ROWS * NB * Q4_BYTES
    var bw_f32 = Float64(bytes) * Float64(1e9) / Float64(ns_f32)
    var bw_int = Float64(bytes) * Float64(1e9) / Float64(ns_int)

    print(
        "[q4int] 现在的 f32 解量化核 : " + String(ns_f32 // 1000000) + " ms  " +
        f2(bw_f32 / 1e9) + " GB/s（按 q4 块流 " + String(bytes // 1024) + " KB）"
    )
    print(
        "[q4int] 整数点积（pmadd.wd）: " + String(ns_int // 1000000) + " ms  " +
        f2(bw_int / 1e9) + " GB/s"
    )
    print(
        "[q4int] 对比: " + f2(Float64(ns_f32) / Float64(ns_int)) +
        "×（>1 是整数点积更快）"
    )
    print(
        "[q4int] 相对 L2：f32 核 " + f2(rel_l2(dst_f32, refv, ROWS) * Float64(100)) +
        "%  整数点积 " + f2(rel_l2(dst_int, refv, ROWS) * Float64(100)) + "%"
    )
    print("[q4int] 参照：q8_0 输出投影那条门（test-q8-head）的 logits 相对 L2 是 0.77%")
    print("")

    arena.keep_alive()
