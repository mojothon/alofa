"""`block_scale` 到底有多贵 —— f16 那条路量出来的**附带发现**。

起因
----
`bench_f16_w.mojo` 同场量了 f16 与 q4：`down_proj` 4864×896 上，我那个几乎没有算术的
f16 核跑 **11.03 GB/s**，而 q4 核只有 **2.85 GB/s** —— 差 **3.9×**。可是两者的指令数
差不了那么多（f16 每 8 个值 load+cvt+fma 三条；q4 每 8 个值五条上下）。**指令数差 1.25×，
时间差 3.9×** ⇒ 慢的原因不是"SIMD 算术太重"，得往别处找。

最可疑的是每块都要调一次的 `block_scale`：它读块里那个 fp16 缩放因子，走的是
`quant.mojo` 里**手写的标量位运算**（指数/尾数/符号三段 + 次正规分支 + 一个 `raise`）。
那个 `raise` 尤其要紧 —— 带异常出口的函数，编译器没法像对待纯算术那样随意重排。

这个文件只回答一件事
--------------------
**一次 `block_scale` 值多少纳秒**，以及**换成 F16C 硬件指令能省多少**。

对照不是"感觉"，是量出来的等效价：把一次调用折成"相当于几条 f32 fma"（AVX2 上 fma
吞吐 2 条/周期，3.6 GHz ⇒ 约 0.14 ns/条）。每块只有 32 个值，也就是 **4 条 fma**
（8 宽）就把这一块算完了 —— 所以只要 `block_scale` 超过几纳秒，它就是主角。

⚠️ 这是**探路用的 micro-bench**，不是门，不进 `pixi run test`。
⚠️ 本机长期过载 ⇒ 取**中位数**，只看**同轮相对**比较。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_block_scale.mojo -o target/bench_block_scale
    ./target/bench_block_scale
"""

from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.kernels.cpu.quant import Q4_BYTES, block_scale

# 与 bench_f16_w 同一个形状：down_proj 4864×896。
comptime ROWS = 4864
comptime COLS = 896
comptime NBLK = ROWS * (COLS // 32)
comptime CALLS = 2000000
# 本机过载 ⇒ 少跑几趟、看中位数。
comptime REPS = 5

comptime F16Ptr = Pointer[Float16, MutUntrackedOrigin]


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


def main() raises:
    print("")
    print("== block_scale 一次值多少纳秒 ==")
    print("   形状 down_proj " + String(ROWS) + "×" + String(COLS) + "；backend avx2")

    var arena = Arena(NBLK * Q4_BYTES + 64)
    var blocks = arena.alloc(NBLK * Q4_BYTES)
    # qs 字节随便填；⚠️ 但**缩放因子必须是合法的 fp16** —— 撞上 exponent==31 会被
    # `block_scale` 的 NaN/Inf 保护直接抛出来（我第一版就被它拦下了，这正好说明那个
    # `raise` 是活代码，编译器必须为它生成分支）。指数固定取 12、尾数变化，让每次
    # 调用的取值都不同，免得整个循环被常量折叠。
    var seed = UInt64(99)
    for i in range(NBLK * Q4_BYTES):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        blocks[unsafe_offset=i] = UInt8((seed >> 40) & UInt64(0xFF))
    for b in range(NBLK):
        var half = 0x3000 + (b % 2048)
        blocks[unsafe_offset=b * Q4_BYTES + 0] = UInt8(half & 0xFF)
        blocks[unsafe_offset=b * Q4_BYTES + 1] = UInt8((half >> 8) & 0xFF)

    # ① 现在的写法：手写标量位运算（含次正规分支 + 一个 raise 出口）
    var t_manual = List[Int]()
    var sink = Float32(0)
    for _ in range(REPS):
        var t0 = monotonic_ns()
        for i in range(CALLS):
            sink += block_scale(blocks, (i % NBLK) * Q4_BYTES)
        t_manual.append(monotonic_ns() - t0)
    var ns_manual = median_ns(t_manual)

    # ② 换成 F16C 硬件指令：块里那个 fp16 在字节 0..1，块长 18 是 2 的倍数
    #    ⇒ 可以按 Float16 直接索引（第 b 块是第 9b 个 fp16）。
    var bf16 = blocks.unsafe_bitcast[Float16]()
    var t_f16c = List[Int]()
    var sink2 = Float32(0)
    for _ in range(REPS):
        var t0 = monotonic_ns()
        for i in range(CALLS):
            var b = i % NBLK
            sink2 += bf16.unsafe_load[width=1](b * 9).cast[DType.float32]()[0]
        t_f16c.append(monotonic_ns() - t0)
    var ns_f16c = median_ns(t_f16c)

    var per_manual = Float64(ns_manual) / Float64(CALLS)
    var per_f16c = Float64(ns_f16c) / Float64(CALLS)
    # 折成"相当于几条 f32 fma"：AVX2 fma 吞吐 2/周期 @3.6GHz ≈ 0.14 ns/条。
    var fma_ns = Float64(0.14)

    print("")
    print(
        "[bscale] 手写位运算（现在）: " + f2(per_manual * Float64(1000)) +
        " ps/call ≈ " + f2(per_manual / fma_ns) + " 条 fma"
    )
    print(
        "[bscale] F16C 硬件转换     : " + f2(per_f16c * Float64(1000)) +
        " ps/call ≈ " + f2(per_f16c / fma_ns) + " 条 fma"
    )
    print(
        "[bscale] 差 " + f2(per_manual / per_f16c) + "×（>1 是手写版更贵）"
    )
    print(
        "[bscale] ⚠️ 每块只有 32 个值 = 4 条 fma（8 宽）就把这块算完了 ⇒ " +
        "手写版占每块成本的 " +
        f2(Float64(100) * per_manual / (per_manual + Float64(4) * fma_ns)) + "%"
    )
    print("[bscale] sink=" + f2(Float64(sink) * Float64(0)) + "（只为别让循环被当死代码删掉）")
    print("")

    arena.keep_alive()
