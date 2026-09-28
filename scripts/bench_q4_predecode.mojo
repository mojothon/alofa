"""`block_scale` 那 90.6% 能不能拿回来 —— 核级验证。

起因
----
`bench_block_scale.mojo` 量出来：一次 `block_scale` 要 **5.4 ns**，折合 **38.6 条 f32 fma**；
而一个 q4_0 块只有 32 个值，把它们乘完只需 **4 条 fma**（8 宽）⇒ **每块成本的 90.6%
花在解码那个 fp16 缩放因子 `d` 上**。它是 `quant.mojo` 里手写的标量位运算（指数/尾数/
符号三段 + 次正规分支 + 一个 `raise`），除法最贵。

为什么这是条好路
----------------
它**不增流量**：q4 的 0.5 B/值一点没变，省下的纯是算术。批=1 decode 是 memory 受限，
但 q4 层核现在只跑到 2.85 GB/s（而几乎无算术的 f16 核能到 11.03）⇒ 中间这段差距就是
能白拿的。这跟 f16 那条路正好相反：f16 是"省算术、付 4× 流量"，这条是"只省算术"。

这个文件做什么
--------------
`_matmul_q4_halves` 抄**两份**，每份只改循环里那一行 `var d = ...`：

    ① 生产核           d = block_scale(blocks, base)        每块现算，5.4 ns
    ② 预解码表版       d = dscale[block_id]                 加载时算好，一次 load
    ③ F16C 版          d = vcvtph2ps(块内那两个字节)        硬件解，不加表

其余**逐字不动**。于是判据不是"误差多大"，而是**逐位相同**：三个核读同一份块流、用
同一个 `d`（②③ 算出来的就是 `block_scale` 的返回值），结果必须一字节不差。这条成立
⇒ 性能差就是纯收益，不涉及任何精度取舍，不需要重跑 `test_q4_matmul_vec` 那套偏差判据。

②③ 的差别只在**接线成本**：② 要改加载路径、多占 4 B/块（+22% 的权重字节）；③ 只动
`block_scale` 内部，内存布局与调用面都不变。若 ③ 跟 ② 一样快，那就该走 ③。

⚠️ 这是**探路用的验证**，不是门，不进 `pixi run test`。它不改生产核，也不改任何
现有通路 —— 生产核在接线前保持原样。
⚠️ 本机长期过载 ⇒ 绝对时间跨轮不可比，只作**同轮相对**比较（三档挨着跑）。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_q4_predecode.mojo -o target/bench_q4_predecode
    ./target/bench_q4_predecode
"""

from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.kernels.cpu.avx2 import (
    Q4_BLOCK,
    Q4_BYTES,
    Q4_SCALE_BYTES,
    Q4_U8,
    W_Q4_HALF,
    matmul_q4_f32,
)
from alofa.kernels.cpu.quant import block_scale, quantize_q4_0

# 与 bench_f16_w / bench_decode_roofline 同一个形状：down_proj 4864×896。
comptime ROWS = 4864
comptime COLS = 896
comptime NBLK = ROWS * (COLS // Q4_BLOCK)
comptime REPS = 5
comptime WARMUP = 2

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]


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


def _halves_predecoded(
    dst: F32Ptr,
    x: F32Ptr,
    blocks: Q4_U8,
    dscale: F32Ptr,
    rows: Int,
    cols: Int,
    bias: F32Ptr,
    has_bias: Bool,
) raises:
    """`_matmul_q4_halves` 的改动版之一：`d` 从预解码表来，其余逐字不动。

    表里存的就是 `block_scale` 的返回值，所以 `d` 的值与生产核**完全相同** ⇒ 结果
    必须逐位相同（下面有硬断言）。
    """
    var blocks_per_row = cols // Q4_BLOCK
    var low_mask = SIMD[DType.uint8, W_Q4_HALF](0x0F)
    var eight = SIMD[DType.float32, W_Q4_HALF](8.0)
    for row in range(rows):
        var acc = Float64(0)
        if has_bias:
            acc = Float64(bias[unsafe_offset=row])
        var s0 = SIMD[DType.float32, W_Q4_HALF](0)
        var s1 = SIMD[DType.float32, W_Q4_HALF](0)
        for b in range(blocks_per_row):
            var block_id = row * blocks_per_row + b
            var base = block_id * Q4_BYTES
            # ↓↓↓ 与生产核唯一的差别在这一行 ↓↓↓
            var d = dscale[unsafe_offset=block_id]
            var xb = b * Q4_BLOCK
            var raw = blocks.unsafe_load[width=W_Q4_HALF](base + Q4_SCALE_BYTES)
            var lo = (raw & low_mask).cast[DType.float32]() - eight
            var hi = ((raw >> 4) & low_mask).cast[DType.float32]() - eight
            var x0 = x.unsafe_load[width=W_Q4_HALF](xb) * d
            var x1 = x.unsafe_load[width=W_Q4_HALF](xb + W_Q4_HALF) * d
            s0 += lo * x0
            s1 += hi * x1
        acc += Float64((s0 + s1).reduce_add())
        dst[unsafe_offset=row] = Float32(acc)


def _halves_f16c(
    dst: F32Ptr,
    x: F32Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    bias: F32Ptr,
    has_bias: Bool,
) raises:
    """**不加表**的那一档：`d` 走 F16C 硬件指令（`vcvtph2ps`），其余仍逐字不动。

    块里那个 fp16 在字节 0..1，块长 18 是 2 的倍数 ⇒ 可以按 `Float16` 直接索引
    （第 b 块是第 9b 个 fp16）。fp16 能表示的每个数 fp32 都能精确表示，所以硬件
    转换与手写位运算**应当**给出同一个值 —— 逐位相同由下面统一断言。

    这一档的意义是**接线成本**：它不改内存布局、不改加载路径、不加那 22% 内存，
    只动 `block_scale` 内部。若它和预解码表一样快，就没必要去改加载路径。
    """
    var bf16 = blocks.unsafe_bitcast[Float16]()
    var blocks_per_row = cols // Q4_BLOCK
    var low_mask = SIMD[DType.uint8, W_Q4_HALF](0x0F)
    var eight = SIMD[DType.float32, W_Q4_HALF](8.0)
    for row in range(rows):
        var acc = Float64(0)
        if has_bias:
            acc = Float64(bias[unsafe_offset=row])
        var s0 = SIMD[DType.float32, W_Q4_HALF](0)
        var s1 = SIMD[DType.float32, W_Q4_HALF](0)
        for b in range(blocks_per_row):
            var block_id = row * blocks_per_row + b
            var base = block_id * Q4_BYTES
            # ↓↓↓ 与生产核唯一的差别在这一行 ↓↓↓
            var d = bf16.unsafe_load[width=1](block_id * 9).cast[DType.float32]()[0]
            var xb = b * Q4_BLOCK
            var raw = blocks.unsafe_load[width=W_Q4_HALF](base + Q4_SCALE_BYTES)
            var lo = (raw & low_mask).cast[DType.float32]() - eight
            var hi = ((raw >> 4) & low_mask).cast[DType.float32]() - eight
            var x0 = x.unsafe_load[width=W_Q4_HALF](xb) * d
            var x1 = x.unsafe_load[width=W_Q4_HALF](xb + W_Q4_HALF) * d
            s0 += lo * x0
            s1 += hi * x1
        acc += Float64((s0 + s1).reduce_add())
        dst[unsafe_offset=row] = Float32(acc)


def main() raises:
    print("")
    print("== `d` 的三种解法：核级验证 ==")
    print(
        "   down_proj " + String(ROWS) + "×" + String(COLS) + "；" + String(NBLK) +
        " 块；backend avx2"
    )

    var arena = Arena(
        ROWS * COLS * 4 + NBLK * Q4_BYTES + NBLK * 4 + COLS * 4 + ROWS * 4 * 4
    )
    var w = arena.alloc(ROWS * COLS * 4).unsafe_bitcast[Float32]()
    var blocks = arena.alloc(NBLK * Q4_BYTES).unsafe_bitcast[UInt8]()
    var dscale = arena.alloc(NBLK * 4).unsafe_bitcast[Float32]()
    var x = arena.alloc(COLS * 4).unsafe_bitcast[Float32]()
    var dst_prod = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    var dst_pre = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    var dst_f16c = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()

    # 权重：确定性 LCG，值域 ±0.3（与真实权重一个量级）。
    var seed = UInt64(12345)
    for i in range(ROWS * COLS):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(Int((seed >> 40) & UInt64(0xFFFF))) / Float32(32768.0)
        w[unsafe_offset=i] = (u - Float32(1.0)) * Float32(0.3)

    seed = UInt64(777)
    for i in range(COLS):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(Int((seed >> 40) & UInt64(0xFFFF))) / Float32(32768.0)
        x[unsafe_offset=i] = (u - Float32(1.0)) * Float32(0.5)

    quantize_q4_0(blocks, w, ROWS * COLS)

    # 预解码表：加载时做一次。表里存的就是 `block_scale` 的返回值 —— 不重新实现
    # fp16 解码，也就没有"两处解码不一致"的可能。
    var t_pre0 = monotonic_ns()
    for i in range(NBLK):
        dscale[unsafe_offset=i] = block_scale(blocks, i * Q4_BYTES)
    var build_ns = monotonic_ns() - t_pre0

    # ① 生产核（d 每块现算）
    for _ in range(WARMUP):
        matmul_q4_f32(dst_prod, x, blocks, ROWS, COLS)
    var t_prod = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        matmul_q4_f32(dst_prod, x, blocks, ROWS, COLS)
        t_prod.append(monotonic_ns() - t0)

    # ② 预解码表版（只改 d 的来源）
    for _ in range(WARMUP):
        _halves_predecoded(dst_pre, x, blocks, dscale, ROWS, COLS, dst_pre, False)
    var t_pre = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        _halves_predecoded(dst_pre, x, blocks, dscale, ROWS, COLS, dst_pre, False)
        t_pre.append(monotonic_ns() - t0)

    # ③ F16C 版（不加表，只换 d 的解码方式）
    for _ in range(WARMUP):
        _halves_f16c(dst_f16c, x, blocks, ROWS, COLS, dst_f16c, False)
    var t_f16c = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        _halves_f16c(dst_f16c, x, blocks, ROWS, COLS, dst_f16c, False)
        t_f16c.append(monotonic_ns() - t0)

    var ns_prod = median_ns(t_prod)
    var ns_pre = median_ns(t_pre)
    var ns_f16c = median_ns(t_f16c)
    var nbytes = NBLK * Q4_BYTES

    # 判据：逐位相同。不比误差，比**值**。
    var diff_pre = 0
    var diff_f16c = 0
    var worst = Float64(0)
    for r in range(ROWS):
        var a = dst_prod[unsafe_offset=r]
        var b = dst_pre[unsafe_offset=r]
        var c = dst_f16c[unsafe_offset=r]
        if a != b:
            diff_pre += 1
        if a != c:
            diff_f16c += 1
        var da = Float64(a) - Float64(b)
        if da < Float64(0):
            da = -da
        var dc = Float64(a) - Float64(c)
        if dc < Float64(0):
            dc = -dc
        if da > worst:
            worst = da
        if dc > worst:
            worst = dc

    print("")
    print(
        "[predec] ① 生产核（d 现算）   : " + String(ns_prod // 1000) + " µs  " +
        f2(Float64(nbytes) / Float64(ns_prod) * Float64(1000)) + " MB/s"
    )
    print(
        "[predec] ② 预解码表版（+表）  : " + String(ns_pre // 1000) + " µs  " +
        f2(Float64(nbytes) / Float64(ns_pre) * Float64(1000)) + " MB/s  加速 " +
        f2(Float64(ns_prod) / Float64(ns_pre)) + "×"
    )
    print(
        "[predec] ③ F16C 版（不加表）  : " + String(ns_f16c // 1000) + " µs  " +
        f2(Float64(nbytes) / Float64(ns_f16c) * Float64(1000)) + " MB/s  加速 " +
        f2(Float64(ns_prod) / Float64(ns_f16c)) + "×"
    )
    print(
        "[predec] 逐位相同：② " + ("是" if diff_pre == 0 else "否") + "（不同 " +
        String(diff_pre) + " 行）  ③ " + ("是" if diff_f16c == 0 else "否") +
        "（不同 " + String(diff_f16c) + " 行）  最大绝对差 " + f2(worst)
    )
    print(
        "[predec] 预解码表构造 " + String(build_ns // 1000000) +
        " ms（加载时一次）；+4 B/块 = +" +
        f2(Float64(100) * Float64(4) / Float64(Q4_BYTES)) + "% 权重字节"
    )
    print("[predec] ⚠️ 绝对时间跨轮不可比（本机过载），只看同轮三档的相对关系")
    print("")

    arena.keep_alive()
