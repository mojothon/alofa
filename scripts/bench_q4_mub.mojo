"""`vpmaddubsw` 通路：核级值不值。

起因
----
单线程端到端 alofa 12.40…12.59 tok/s，llama.cpp `-t 1` 27.05 ⇒ 差 **2.15×**。
两边都是 q4_0：同样 0.5 B/值、同样 32 值/块、同样 fp16 缩放因子。所以差不在格式，
在**核**。

ggml 的 `ggml_vec_dot_q4_0_q8_0` 靠的是 `vpmaddubsw`（u8 × i8 → i16，一条指令吃掉
16 对乘法）。这条路 alofa 在 2026-09-28 探过，**只值 1.07× ⇒ 收手**，但收手的原因
不是路不对，是**指令没叫到**：当时六种拼法都拿不到 `vpmaddubsw`，只能退到 int16 路
（`pmadd.wd`），每块因此多付 4 条 `vpmovsxbw`。

**本轮拿到了**：`llvm.x86.avx2.pmadd.ub.sw` 能编译、且 objdump 确认生成
`vpmaddubsw`（此前试的是 `pmaddub.sw` / `pmadd.ubsw`，差的就是这个点号位置）。

为什么值得一试
--------------
反汇编 `_matmul_q4_halves` 的热循环（`objdump`，24 条指令/块）数出来：

    vcvtph2ps ×1          解 d（已修）
    vpand/vpsrlw/vpand    拆 nibble
    vpmovzxbd ×4          16 个 nibble → i32
    vcvtdq2ps ×4          i32 → f32
    vaddps ×4             −8
    vmulps ×2 + vfmadd ×2 真正有用的乘加（只有 4 条）

**16 条在展开，4 条在算**。而 `vpmovzxbd` / `vcvtdq2ps` 都走 **port 5**（每 cycle
1 条）⇒ 每块 8 条 p5 = 8 cycle；实测每块 ~13 cycle ⇒ **层核剩下的瓶颈是 port 5**。
`vpmaddubsw` 走的是乘加端口，正好绕开它。

这个文件做什么
--------------
`down_proj` 4864×896 上同场比两档：

    ① 生产核 `matmul_q4_f32`（f32 展开路）
    ② `vpmaddubsw` 路：拆 nibble → join 成 32 个 u8 → `vpmaddubsw` → `pmadd.wd`(配 ones) → 整数求和

⚠️ ② 用一个**恒等式**换掉 −8：`Σ(n−8)·x = Σn·x − 8·Σx`。因为 `vpmaddubsw` 把第一个
操作数当 **u8**，而 nibble−8 的负值（如 −8）按 u8 读是 248 ⇒ 只能送**未减 8** 的
nibble 进去，再在整数域减掉 `8·Σx`（每段 x 的和**预计算**，一个 token 算一次，所有行共用）。
整数域做减法 ⇒ 不引入任何新的浮点误差。

⚠️ 本机长期过载 ⇒ 绝对时间跨轮不可比，只看**同轮两档的相对关系**。
⚠️ 这是**探路用的验证**，不是门，不进 `pixi run test`；它不改生产核。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_q4_mub.mojo -o target/bench_q4_mub
    ./target/bench_q4_mub
"""

from std.sys import llvm_intrinsic

from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.kernels.cpu.avx2 import (
    Q4_BLOCK,
    Q4_BYTES,
    Q4_SCALE_BYTES,
    Q4_U8,
    matmul_q4_f32,
)
from alofa.kernels.cpu.quant import block_scale, quantize_q4_0

comptime ROWS = 4864
comptime COLS = 896
comptime NB = COLS // Q4_BLOCK
comptime REPS = 5
comptime WARMUP = 2

# 实测能叫到的名字。`pmaddub.sw` / `pmadd.ubsw` 都叫不到 —— 差的就是点号位置。
comptime MUBSW = "llvm.x86.avx2.pmadd.ub.sw"
comptime MADDWD = "llvm.x86.avx2.pmadd.wd"

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]
comptime I8Ptr = Pointer[Int8, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]

alias U8x16 = SIMD[DType.uint8, 16]
alias I16x16 = SIMD[DType.int16, 16]
alias I32x8 = SIMD[DType.int32, 8]


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


def q4_mub_dot(
    dst: F32Ptr,
    xq: I8Ptr,
    sumx: I32Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    sx: Float32,
) raises:
    """`dst = W · (sx · xq)`；每块走 `vpmaddubsw`。

    与 f32 展开路的差别只在中间那一步：展开路把 nibble 变成 f32 再乘（8 条 port-5
    指令），这里让 `vpmaddubsw` 直接在整数域吃掉 16 对乘法（1 条，走乘加端口）。
    """
    var nb = cols // Q4_BLOCK
    var m4 = U8x16(0x0F)
    var four = U8x16(4)
    var ones = I16x16(1)
    for r in range(rows):
        var acc = Float32(0)
        for b in range(nb):
            var base = (r * nb + b) * Q4_BYTES
            var qs = blocks.unsafe_load[width=16](base + Q4_SCALE_BYTES)
            # 未减 8 的 nibble：`vpmaddubsw` 把第一个操作数当 u8，减过 8 的负值
            # （如 −8 → 0xF8）按 u8 读是 248，那就错了。
            var lo = qs & m4
            var hi = (qs >> four) & m4
            var w32 = lo.join(hi)
            var x32 = xq.unsafe_load[width=32](b * Q4_BLOCK)
            var p = llvm_intrinsic[MUBSW, I16x16](w32, x32)
            var q = llvm_intrinsic[MADDWD, I32x8](p, ones)
            var d = block_scale(blocks, base)
            # Σ(n−8)·x = Σn·x − 8·Σx —— 减法是**整数**的，不添浮点误差。
            var v = Float32(q.reduce_add() - 8 * sumx[unsafe_offset=b])
            acc += d * sx * v
        dst[unsafe_offset=r] = acc


def q4_mub_dot_vec(
    dst: F32Ptr,
    xq: I8Ptr,
    sumxv: I32Ptr,
    blocks: Q4_U8,
    rows: Int,
    cols: Int,
    sx: Float32,
) raises:
    """同 ②，但**不在块内求和** —— `d` 每块不同，逼着 ② 每块做一次 `reduce_add`
    （8 个 i32 → 1，好几条 shuffle）。

    这里改成：把 `−8·Σx` 预先做成**一个 8 宽的 i32 向量**（只 lane 0 非零），
    块内只做 `q + shift`（1 条整数加）就算完了，累加进 **f32 向量累加器**，
    整行结束才 `reduce_add` 一次 ⇒ 每块省掉那次求和。
    """
    var nb = cols // Q4_BLOCK
    var m4 = U8x16(0x0F)
    var four = U8x16(4)
    var ones = I16x16(1)
    for r in range(rows):
        var acc = SIMD[DType.float32, 8](0)
        for b in range(nb):
            var base = (r * nb + b) * Q4_BYTES
            var qs = blocks.unsafe_load[width=16](base + Q4_SCALE_BYTES)
            var lo = qs & m4
            var hi = (qs >> four) & m4
            var w32 = lo.join(hi)
            var x32 = xq.unsafe_load[width=32](b * Q4_BLOCK)
            var p = llvm_intrinsic[MUBSW, I16x16](w32, x32)
            var q = llvm_intrinsic[MADDWD, I32x8](p, ones)
            var sh = sumxv.unsafe_load[width=8](b * 8)
            var d = block_scale(blocks, base)
            acc = (q + sh).cast[DType.float32]() * SIMD[DType.float32, 8](d * sx) + acc
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
    var g = Float64(1)
    for _ in range(60):
        g = (g + q / g) * Float64(0.5)
    return g


def main() raises:
    print("")
    print("== vpmaddubsw 通路 vs 现在的 f32 展开核 ==")
    print(
        "   down_proj " + String(ROWS) + "×" + String(COLS) + "；" + String(NB) +
        " 块/行；backend avx2"
    )

    var arena = Arena(
        ROWS * COLS * 4 + ROWS * NB * Q4_BYTES + ROWS * 4 * 2 + COLS * 4 + COLS + NB * 4
        + 1048576
    )
    var w = arena.alloc(ROWS * COLS * 4).unsafe_bitcast[Float32]()
    var blocks = arena.alloc(ROWS * NB * Q4_BYTES).unsafe_bitcast[UInt8]()
    var x = arena.alloc(COLS * 4).unsafe_bitcast[Float32]()
    var xq = arena.alloc(COLS).unsafe_bitcast[Int8]()
    var dst_f32 = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    var dst_mub = arena.alloc(ROWS * 4).unsafe_bitcast[Float32]()
    var sumx = arena.alloc(NB * 4).unsafe_bitcast[Int32]()
    var sumxv = arena.alloc(NB * 8 * 4).unsafe_bitcast[Int32]()

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

    # 激活量化成 int8：`vpmaddubsw` 的第二个操作数按 i8 解释，所以必须按 `Int8`
    # 指针存补码字节（按 UInt8 存再 `cast[int8]` 会被**饱和**成 +127，负值全废）。
    var xmax = Float32(0)
    for i in range(COLS):
        var v = x[unsafe_offset=i]
        if v < Float32(0):
            v = -v
        if v > xmax:
            xmax = v
    var sx = xmax / Float32(127.0)
    if sx == Float32(0):
        sx = Float32(1)
    for i in range(COLS):
        var qv = Int(x[unsafe_offset=i] / sx)
        if qv > 127:
            qv = 127
        if qv < -128:
            qv = -128
        xq[unsafe_offset=i] = Int8(qv)
    # 每段的 Σx：一个 token 算一次，所有行共用 —— 这是 −8 能在整数域减掉的前提。
    for b in range(NB):
        var s = Int(0)
        for j in range(Q4_BLOCK):
            s += Int(xq[unsafe_offset=b * Q4_BLOCK + j])
        sumx[unsafe_offset=b] = Int32(s)
        # ③ 要的向量形式：只 lane 0 放 −8·Σx，其余为 0 ⇒ 块内一条整数加就算完。
        for k in range(8):
            if k == 0:
                sumxv[unsafe_offset=b * 8 + k] = Int32(-8 * s)
            else:
                sumxv[unsafe_offset=b * 8 + k] = Int32(0)

    # ① 生产核
    for _ in range(WARMUP):
        matmul_q4_f32(dst_f32, x, blocks, ROWS, COLS)
    var t1 = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        matmul_q4_f32(dst_f32, x, blocks, ROWS, COLS)
        t1.append(monotonic_ns() - t0)

    # ② vpmaddubsw 路
    for _ in range(WARMUP):
        q4_mub_dot(dst_mub, xq, sumx, blocks, ROWS, COLS, sx)
    var t2 = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        q4_mub_dot(dst_mub, xq, sumx, blocks, ROWS, COLS, sx)
        t2.append(monotonic_ns() - t0)

    # ⚠️ 先量 ② 的精度 —— ③ 马上要往同一个 dst 里写，会把它盖掉。
    var l2_2 = rel_l2(dst_mub, dst_f32, ROWS)

    # ③ vpmaddubsw 路（块内不求和，整行末才 reduce 一次）
    for _ in range(WARMUP):
        q4_mub_dot_vec(dst_mub, xq, sumxv, blocks, ROWS, COLS, sx)
    var t3 = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        q4_mub_dot_vec(dst_mub, xq, sumxv, blocks, ROWS, COLS, sx)
        t3.append(monotonic_ns() - t0)

    var ns1 = median_ns(t1)
    var ns2 = median_ns(t2)
    var ns3 = median_ns(t3)
    var nbytes = ROWS * NB * Q4_BYTES

    print("")
    print(
        "[mub] ① 生产核（f32 展开） : " + String(ns1 // 1000) + " µs  " +
        f2(Float64(nbytes) / Float64(ns1) * Float64(1000)) + " MB/s"
    )
    print(
        "[mub] ② vpmaddubsw 通路    : " + String(ns2 // 1000) + " µs  " +
        f2(Float64(nbytes) / Float64(ns2) * Float64(1000)) + " MB/s  加速 " +
        f2(Float64(ns1) / Float64(ns2)) + "×"
    )
    print(
        "[mub] ③ 同 ② 但块内不求和     : " + String(ns3 // 1000) + " µs  " +
        f2(Float64(nbytes) / Float64(ns3) * Float64(1000)) + " MB/s  加速 " +
        f2(Float64(ns1) / Float64(ns3)) + "×"
    )
    print(
        "[mub] 相对 L2（分母取①）：② " + f2(l2_2 * Float64(100)) + "%  ③ " +
        f2(rel_l2(dst_mub, dst_f32, ROWS) * Float64(100)) + "%"
    )
    print("[mub] ⚠️ 绝对时间跨轮不可比（本机过载），只看同轮两档的相对关系")
    print("")

    arena.keep_alive()
