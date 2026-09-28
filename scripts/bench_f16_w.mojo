"""f16 权重那条路：核级量一次，并把「流量 ×4」那笔账算上。

为什么突然想试它
----------------
`bench_decode_roofline.mojo` 把 q4 层核钉在了**算术受限**上：`down_proj` 4864×896，
q4_0/avx2 只读 2.45 MB（常驻 L3）却只跑 **2.11–2.28 GB/s**，地板比 0.163；同一形状
fp32/avx2 读 **17.4 MB**（八倍于 q4，从内存来）反而跑 **11.47 GB/s**。也就是说：q4 把
字节省下来了，但**每个值摊到的算术更贵** —— `vpmovsxbd` + `vcvtdq2ps` + `vmulps`(×d)
+ `vfmadd` 四条一组，零头都没剩下。

f16 换掉的是这套算术：每 8 个值一条 `vcvtph2ps`（F16C，本机有）+ 一条 fma，没有
nibble 拆分、没有 per-block 缩放。代价写在脸上 —— **每值 2 字节，是 q4 的 4 倍**。

⚠️ 所以这里量的不只是「谁快」
------------------------------
批=1 decode 是 **memory 受限**（算术强度 0.5 flops/byte），一个 token 要把整个层权重
拖过内存一遍。于是「核级快 X 倍」**不等于**「端到端快 X 倍」：

    端到端的账 ≈ 4 / X      （f16 流量 4×，核级快 X×）

只有当 **X > 4** 时 f16 才在端到端上赢 —— 而这个脚本存在的意义，就是去看 X 有多大。
核级赢 ≠ 该接线；这是它和 `bench_q4_int8.mojo` 共用的判据（那条路 X=1.07，已收手）。

⚠️ f16 不是"另一档量化"，它不替代 q4
--------------------------------------
q4 的 0.5 B/值是为了让**大模型**装得下；f16 是给"算术比带宽贵"的场合用的。本脚本只
回答"层核那一下谁快"，不回答"0.5B 该存哪种格式"。

判据
----
* 参照物**独立另写**：未量化的 fp32 权重 × fp32 激活，`Float64` 累加 —— 不复用被测核
  的任何一行，也不复用量化器。
* 精度：f16 有 11 位尾数，权重相对误差 ~5e-4 ⇒ 点积的相对 L2 应远好于 q4 的 **6.39%**
  （同一形状、同一批合成权重）。若 f16 的误差接近 q4，说明是核写错了，不是格式问题。
* 本机长期过载 ⇒ 取**中位数**，且只作**同轮相对比较**。

跑法（`-O2`：`-O0` 下 `vcvtph2ps` 的调度没有参考价值）
------------------------------------------------------
    pixi run mojo build -O2 -I src scripts/bench_f16_w.mojo -o target/bench_f16_w
    ./target/bench_f16_w
"""

from std.math import round

from alofa.core.dtype import DT_FP32
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, shape2
from alofa.kernels.cpu.avx2 import Q4_U8, linear as linear_v, matmul_q4_f32
from alofa.kernels.cpu.quant import Q4_BYTES, quantize_q4_0

# 真实的 `down_proj`：每层三块大矩阵里最大的一块，也是 roofline 量过的那个形状。
comptime ROWS = 4864
comptime COLS = 896
# 本机过载 ⇒ 少跑几趟、看中位数。
comptime REPS = 5
comptime WARMUP = 2

comptime F16Ptr = Pointer[Float16, MutUntrackedOrigin]
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


def matmul_f16_w(
    dst: F32Ptr, x: F32Ptr, w: F16Ptr, rows: Int, cols: Int
) raises:
    """`dst = W · x`，`W` 为 f16（每值 2 字节），单累加器。

    每 8 个值：一条 load(f16) + 一条 `vcvtph2ps` + 一条 load(x) + 一条 fma。
    对比 q4 那套四联指令，且**没有** per-block 缩放 —— `d` 是 q4 独有的负担。
    """
    for r in range(rows):
        var acc = F32x8(0)
        var base = r * cols
        for c in range(0, cols, 8):
            var wv = w.unsafe_load[width=8](base + c).cast[DType.float32]()
            var xv = x.unsafe_load[width=8](c)
            acc = wv * xv + acc
        dst[unsafe_offset=r] = acc.reduce_add()


def matmul_f16_w4(
    dst: F32Ptr, x: F32Ptr, w: F16Ptr, rows: Int, cols: Int
) raises:
    """同上，**4 个累加器展开**：把四条依赖链分开，让 fma 流水线填满。

    f16 核的每步只依赖上一步的累加器，单累加器时吞吐被 fma 延迟（4 拍）钉住。
    """
    for r in range(rows):
        var a0 = F32x8(0)
        var a1 = F32x8(0)
        var a2 = F32x8(0)
        var a3 = F32x8(0)
        var base = r * cols
        for c in range(0, cols, 32):
            a0 = w.unsafe_load[width=8](base + c + 0).cast[DType.float32]() * x.unsafe_load[
                width=8
            ](c + 0) + a0
            a1 = w.unsafe_load[width=8](base + c + 8).cast[DType.float32]() * x.unsafe_load[
                width=8
            ](c + 8) + a1
            a2 = w.unsafe_load[width=8](base + c + 16).cast[DType.float32]() * x.unsafe_load[
                width=8
            ](c + 16) + a2
            a3 = w.unsafe_load[width=8](base + c + 24).cast[DType.float32]() * x.unsafe_load[
                width=8
            ](c + 24) + a3
        dst[unsafe_offset=r] = (a0 + a1 + a2 + a3).reduce_add()


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
    var g = Float64(1)
    for _ in range(60):
        g = (g + q / g) * Float64(0.5)
    return g


def main() raises:
    print("")
    print("== f16 权重那条路：核级 ===")
    print("   down_proj " + String(ROWS) + "×" + String(COLS) + "；backend avx2；F16C 有")

    # ~46 MB：fp32 权重 17.4 + f16 权重 8.7 + q4 块流 2.45，再加上 fp32 对照要的
    # 另一份 17.4（`linear` 走 TensorView 接口，自己要一块），以及 dst 与激活。
    var arena = Arena(
        ROWS * COLS * 4 * 2 + ROWS * COLS * 2 + ROWS * COLS // 32 * Q4_BYTES
        + ROWS * 4 * 7 + COLS * 16 + 1048576
    )
    var w = arena.alloc(ROWS * COLS * 4).unsafe_bitcast[Float32]()
    var wf16 = arena.alloc(ROWS * COLS * 2).unsafe_bitcast[Float16]()
    var nblk = ROWS * (COLS // 32) * Q4_BYTES
    var blocks = arena.alloc(nblk).unsafe_bitcast[UInt8]()
    var x = arena.alloc(COLS * 4).unsafe_bitcast[Float32]()
    var dst_q4 = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    var dst_f16 = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    var dst_f16x4 = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
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

    # f16 权重：fp32 → f16（IEEE binary16，就近舍入）。
    for i in range(ROWS * COLS):
        wf16[unsafe_offset=i] = Float16(w[unsafe_offset=i])

    # q4 块流（对照）：COLS 是 32 的整数倍，块边界不跨行 ⇒ 一次调用与逐行**逐字节相同**。
    quantize_q4_0(blocks, w, ROWS * COLS)

    # 参照物：未量化的 fp32 权重 × fp32 激活，fp64 累加。**不复用**上面任何一个核。
    for r in range(ROWS):
        var s = Float64(0)
        for c in range(COLS):
            s += Float64(w[unsafe_offset=r * COLS + c]) * Float64(x[unsafe_offset=c])
        refv[unsafe_offset=r] = Float32(s)

    # fp32 对照走项目的 `linear`（TensorView 接口），不是我另写的一版。
    # ⚠️ 这一路**只量时间**：下面三块是刚分配的内存，没填权重（填了也不影响速度，
    # 而 `linear` 的结果在这里不参与精度判定 —— 精度只由 q4 / f16 与参照物比）。
    var tv_w = TensorView(arena.alloc(ROWS * COLS * 4), shape2(ROWS, COLS), DT_FP32)
    var tv_x = TensorView(arena.alloc(COLS * 4), shape2(1, COLS), DT_FP32)
    var tv_d = TensorView(arena.alloc(ROWS * 4), shape2(1, ROWS), DT_FP32)

    # ① 现在的 q4 核
    for _ in range(WARMUP):
        matmul_q4_f32(dst_q4, x, blocks, ROWS, COLS)
    var t_q4 = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        matmul_q4_f32(dst_q4, x, blocks, ROWS, COLS)
        t_q4.append(monotonic_ns() - t0)

    # ② f16 单累加器
    for _ in range(WARMUP):
        matmul_f16_w(dst_f16, x, wf16, ROWS, COLS)
    var t_f16 = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        matmul_f16_w(dst_f16, x, wf16, ROWS, COLS)
        t_f16.append(monotonic_ns() - t0)

    # ③ f16 四路展开
    for _ in range(WARMUP):
        matmul_f16_w4(dst_f16x4, x, wf16, ROWS, COLS)
    var t_f16x4 = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        matmul_f16_w4(dst_f16x4, x, wf16, ROWS, COLS)
        t_f16x4.append(monotonic_ns() - t0)

    # ④ fp32 对照（读 17.4 MB，从内存来）
    for _ in range(WARMUP):
        linear_v(tv_d, tv_x, tv_w)
    var t_fp32 = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        linear_v(tv_d, tv_x, tv_w)
        t_fp32.append(monotonic_ns() - t0)

    var ns_q4 = median_ns(t_q4)
    var ns_f16 = median_ns(t_f16)
    var ns_f16x4 = median_ns(t_f16x4)
    var ns_fp32 = median_ns(t_fp32)
    var b_q4 = nblk
    var b_f16 = ROWS * COLS * 2
    var b_fp32 = ROWS * COLS * 4

    print(
        "[f16w] q4_0   读 " + String(b_q4 // 1024) + " KB : " + String(ns_q4 // 1000000) +
        " ms  " + f2(Float64(b_q4) * 1e9 / Float64(ns_q4) / 1e9) + " GB/s"
    )
    print(
        "[f16w] f16    读 " + String(b_f16 // 1024) + " KB : " + String(ns_f16 // 1000000) +
        " ms  " + f2(Float64(b_f16) * 1e9 / Float64(ns_f16) / 1e9) + " GB/s"
    )
    print(
        "[f16w] f16×4  读 " + String(b_f16 // 1024) + " KB : " + String(ns_f16x4 // 1000000) +
        " ms  " + f2(Float64(b_f16) * 1e9 / Float64(ns_f16x4) / 1e9) + " GB/s"
    )
    print(
        "[f16w] fp32   读 " + String(b_fp32 // 1024) + " KB : " + String(ns_fp32 // 1000000) +
        " ms  " + f2(Float64(b_fp32) * 1e9 / Float64(ns_fp32) / 1e9) + " GB/s"
    )
    print(
        "[f16w] 核级对比：f16 最快版是 q4 的 " +
        f2(Float64(ns_q4) / Float64(min(ns_f16, ns_f16x4))) + "×（>1 是 f16 更快）"
    )
    print(
        "[f16w] 相对 L2：q4 " + f2(rel_l2(dst_q4, refv, ROWS) * 100) + "%  f16 " +
        f2(rel_l2(dst_f16, refv, ROWS) * 100) + "%  f16×4 " +
        f2(rel_l2(dst_f16x4, refv, ROWS) * 100) + "%"
    )
    print(
        "[f16w] ⚠️ 端到端的账：f16 流量是 q4 的 4× ⇒ 端到端 ≈ 4 / " +
        f2(Float64(ns_q4) / Float64(min(ns_f16, ns_f16x4))) + " = " +
        f2(Float64(4) * Float64(min(ns_f16, ns_f16x4)) / Float64(ns_q4)) +
        "×（>1 是 f16 端到端更慢）"
    )
    print("[f16w] 判据：只有当核级 >4× 时，f16 才值得接线（整数点积那条路是 1.07×，已收手）")
    print("")

    arena.keep_alive()
