"""一次 `attention` 调用值多少纳秒 —— decode 形状下，单线程。

为什么要有这个文件
------------------
账本上"注意力占 26.4%"和"注意力里 30/44 是 data-dependent 分支"这两条，**没有
一把能复现的尺子**：`grep` 全部 `scripts/bench_*.mojo`，没有任何一个直接调
`attention` 核（`bench_head_split` 只在正文里提"注意力"，它量的是输出投影）。

而算术核对说那个 26.4% 不可能是算术：

    24 层 × 14 头 × 1 query × 17 kv × 64 dim × 2（打分 + 值归约） ≈ 0.73 MFLOP

若它真是 26.4% × 24.6 ms = 6.5 ms，则有效算力只有 **0.11 GFLOP/s** —— 比同一颗
核的向量能力低三个数量级。所以那 6.5 ms 要么混进了 `copy_into` / `rope` /
`prof.now()` 的开销，要么是**每次调用的固定成本**（形状校验、`rows_of`、`exp`
库调用）。这两条病因对应完全不同的修法，**在量出 ns/调用之前改代码就是猜**。

它回答三件事
------------
1. 真实 decode 形状（`kv_len` = 17 / 33 / 65）下，一次 `attention` 多少 ns；
2. 折算成"占一个 token 的百分之几"（÷ 24 层 ÷ `TOKEN_MS`），与账本的 26.4% 对质；
3. 按头切分有没有引入回归：`0..14` 一次 vs `0..7` + `7..14` 两次（**同轮对照**）。

口径
----
* **单线程**，不走 `TaskGroup` —— 本机长期过载，多线程在这里量不出东西。
* 每档跑 `REPS` 轮、每轮 `CALLS` 次，取 **min**：过载机器上 min 比中位数更抗抢核
  （被抢的那几轮直接偏大，最小值最接近"这核自己有多快"）。
* 权重与激活都填**非零随机值**：全零会让 `exp`/比较分支走最省的路，量出来的
  是"最好情况"，不是 decode。
* `dst` 每次都被核覆写 ⇒ 不存在被优化掉的循环。
* ⚠️ micro-bench，不是门，不进 `pixi run test`。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_attn_decode.mojo -o target/bench_attn_decode
    ./target/bench_attn_decode
"""

from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.ffi import monotonic_ns
from alofa.core.tensor import TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import attention

# Qwen2.5-0.5B 的真实 decode 形状。
comptime N_HEADS = 14
comptime N_KV = 2
comptime HEAD_DIM = 64
comptime Q_COLS = N_HEADS * HEAD_DIM
comptime KV_COLS = N_KV * HEAD_DIM

# decode 到第 17 / 33 / 65 个位置。
comptime KVS = 3
comptime KV_LEN0 = 17
comptime KV_STEP = 16

comptime REPS = 7
comptime CALLS = 400

# 账本里"一个 token 24.6 ms、注意力 6.5 ms (26.4%)"的那个分母。
comptime TOKEN_MS = 24.6


def min_ns(mut v: List[Int]) -> Int:
    """样本最小值。过载机器上被抢核的那几轮只会偏大。"""
    var m = v[0]
    for i in range(1, len(v)):
        if v[i] < m:
            m = v[i]
    return m


def fill_noise(p: Pointer[Float32, MutUntrackedOrigin], n: Int, seed_in: UInt64):
    """确定性伪随机，取值落在 ±1 内。"""
    var seed = seed_in
    for i in range(n):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        # 取高 24 位 → [0,1) → [-1,1)
        var u = Float32((seed >> 40) & UInt64(0xFFFFFF)) / Float32(16777216.0)
        p[i] = (u * Float32(2.0)) - Float32(1.0)


def f2(x: Float64) -> String:
    """两位小数（本机没有 sprintf）。负数要单独带符号：切分代价可能 < 0。"""
    var neg = x < Float64(0)
    var v = -x if neg else x
    var whole = Int(v)
    var frac = Int((v - Float64(whole)) * Float64(100) + Float64(0.5))
    if frac >= 100:
        whole += 1
        frac -= 100
    var s = String(whole) + "."
    if frac < 10:
        s = s + "0"
    s = s + String(frac)
    return "-" + s if neg else s


def time_full(
    dst: TensorView, q: TensorView, k: TensorView, v: TensorView, sc: TensorView
) raises -> Int:
    """全量 14 头，返回每次调用的最小 ns。"""
    var t = List[Int]()
    print("     [full] calling")
    attention(dst, q, k, v, sc, N_HEADS, N_KV, HEAD_DIM)
    print("     [full] first call ok")
    for _ in range(REPS):
        var t0 = monotonic_ns()
        for _ in range(CALLS):
            attention(dst, q, k, v, sc, N_HEADS, N_KV, HEAD_DIM)
        t.append((monotonic_ns() - t0) // CALLS)
    return min_ns(t)


def time_halves(
    dst: TensorView, q: TensorView, k: TensorView, v: TensorView, sc: TensorView
) raises -> Int:
    """同样 14 个头，但拆成 0..7 + 7..14 两次 —— 并行版走的就是这条路的形态。

    两次加起来的工作量与全量**完全一样**（头之间无共享状态），所以两者之差就是
    "按头切分"本身的代价：多一次调用的固定成本（形状校验等）。这条对照与机器
    负载无关（同一轮、同一进程、紧挨着跑）。
    """
    var t = List[Int]()
    for _ in range(REPS):
        var t0 = monotonic_ns()
        for _ in range(CALLS):
            attention(dst, q, k, v, sc, N_HEADS, N_KV, HEAD_DIM, 0, 7)
            attention(dst, q, k, v, sc, N_HEADS, N_KV, HEAD_DIM, 7, N_HEADS)
        t.append((monotonic_ns() - t0) // CALLS)
    return min_ns(t)


def main() raises:
    print("")
    print("== 一次 attention 调用值多少纳秒（decode 形状，单线程） ==")
    print(
        "   n_heads "
        + String(N_HEADS)
        + "  n_kv_heads "
        + String(N_KV)
        + "  head_dim "
        + String(HEAD_DIM)
        + "  q_len 1"
    )
    print("   每档 " + String(REPS) + " 轮 × " + String(CALLS) + " 次，取 min")

    # k/v 最大 65 行。
    comptime MAX_KV = KV_LEN0 + KV_STEP * (KVS - 1)
    var need = 1024 * 1024
    var arena = Arena(need)
    print("   [a] arena ok")
    # 分配出来的都是 RawPtr（UInt8）；填数用 `unsafe_bitcast[Float32]`，
    # 传进 `TensorView` 直接用 RawPtr —— 与 `bench_q4_rows` 同一条路。
    var q_raw = arena.alloc(Q_COLS * 4)
    var k_raw = arena.alloc(KV_COLS * MAX_KV * 4)
    var v_raw = arena.alloc(KV_COLS * MAX_KV * 4)
    var o_raw = arena.alloc(Q_COLS * 4)
    var s_raw = arena.alloc(MAX_KV * 4)
    print("   [b] allocs ok")
    # ⚠️ `keep_alive()` 放在 main 的**最后**（见文件末尾）：它只是借用 `self`，
    # 而 Mojo 的 ASAP 销毁发生在**最后一次使用之后** —— 放在 alloc 后面等于把
    # 寿命截断在那里，紧接着的填数就会写进已 munmap 的地址（本探针踩过两次）。


    print("")
    print("   （占一个 token = ns/次 × 24 层 ÷ " + String(TOKEN_MS) + " ms）")
    print("   ------------------------------------------------------------------")

    for idx in range(KVS):
        var kv_len = KV_LEN0 + KV_STEP * idx
        var dst = TensorView(o_raw, shape2(1, Q_COLS), DT_FP32)
        var q = TensorView(q_raw, shape2(1, Q_COLS), DT_FP32)
        var k = TensorView(k_raw, shape2(kv_len, KV_COLS), DT_FP32)
        var v = TensorView(v_raw, shape2(kv_len, KV_COLS), DT_FP32)
        var sc = TensorView(s_raw, shape2(1, kv_len), DT_FP32)
        # 真正写数据的指针走 `f32_data`（`bench_q4_rows` 同一条路）：`unsafe_bitcast`
        # 那条路在本探针里量出的指针是坏的，写第一个元素就 SIGSEGV。
        fill_noise(f32_data(q), Q_COLS, UInt64(7))
        fill_noise(f32_data(k), KV_COLS * kv_len, UInt64(11))
        fill_noise(f32_data(v), KV_COLS * kv_len, UInt64(13))
        print("   [kv=" + String(kv_len) + "] filled")

        var full = time_full(dst, q, k, v, sc)
        var halves = time_halves(dst, q, k, v, sc)
        var cost = Float64(halves - full) / Float64(full) * Float64(100)
        var share = Float64(full) * Float64(24) / (Float64(TOKEN_MS) * Float64(1e6)) * Float64(100)

        print(
            "   kv_len=" + String(kv_len)
            + "   全量 " + String(full) + " ns"
            + "   两半 " + String(halves) + " ns"
            + "   切分代价 " + f2(cost) + "%"
            + "   占一个 token " + f2(share) + "%"
        )

    print("")
    print("   读法（三条，缺一条就不能下结论）：")
    print("   ① 「占一个 token」若远小于账本的 26.4% ⇒ 那个桶里混了别的东西")
    print("      （copy_into / rope / prof.now），不是 attention 本身贵。")
    print("   ② 「切分代价」为 0 附近 ⇒ 按头分片的固定成本可以忽略，瓶颈在别处。")
    print("   ③ 如果 ns/次 与 kv_len 几乎无关 ⇒ 时间是**每次调用的固定成本**，")
    print("      不是 O(kv_len) 的算术 —— 那种情况该治的是入口，不是内层循环。")

    arena.keep_alive()
