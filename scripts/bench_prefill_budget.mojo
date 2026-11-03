"""prefill 一趟到底花在哪 —— 真实模型、真实权重、单进程。

为什么要有这个文件
------------------
decode 已经有分段预算了（`bench_decode_budget.mojo`，账本 11-03 第七条），
它把"串行算子占 55%"这个整体叙事推翻成 **1.3%**。prefill 一直没有同样的表，
而 `bench_prefill.mojo` 只给端到端"每 token 时间 × 片数"，看不到**时间落在哪一段**。

prefill 的成本模型与 decode 不同，这里一并打出来对账：

    权重被读的遍数 = 片数 × ceil(每片行数 / RB)        （RB=8，块内复用 8 行）
    n=128, shards=8 → 每片 16 行 → ceil(16/8)=2 → 16 遍 → 16 × 201 MB = 3.2 GB

如果实测"墙钟 × 隐含带宽"贴着整机 DRAM 天花板（账本 11-03 第八条量到 **16.6 GB/s**），
那 prefill 就是**权重遍数**受限，杠杆是"把权重留在寄存器里、让 token 在内层"；
如果隐含带宽远低于 16.6，那它受限在别处（算术/调度），遍数就不是杠杆。

⚠️ 要开 `comptime ALOFA_PROFILE = True`（`src/alofa/model/profiler.mojo`）。
插桩拖慢绝对值 ⇒ **只读段占比**；每 token 时间与隐含带宽也受插桩影响，当**量级**看。
⚠️ 探路用 micro-bench，不是门。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_prefill_budget.mojo -o target/prefill_budget
    ./target/prefill_budget
"""

from alofa.core.ffi import monotonic_ns
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime MAX_TOK = 1024

# Qwen2.5-0.5B 的形状（与 `bench_proj_mlp.mojo` 同一组常量）
comptime HID = 896
comptime INTER = 4864
comptime KV = 128
comptime LAYERS = 24
# 24 层 7 个投影的 q4_0 字节数，每 token 一遍
comptime PROJ_BYTES = LAYERS * ((2 * HID * HID + 2 * KV * HID + 3 * INTER * HID) // 32 * 18)
# 输出头 151936 × 896
comptime HEAD_BYTES = (151936 * HID // 32) * 18


def f2(v: Float64) -> String:
    var w = Int(v)
    var frac = Int((v - Float64(w)) * 100.0 + 0.5)
    if frac >= 100:
        w += 1
        frac -= 100
    var s = String(frac)
    if frac < 10:
        s = "0" + s
    return String(w) + "." + s


def token_ids(n: Int) -> List[Int]:
    """`n` 个确定性 id（不用分词器，少一个解释变量；与 `bench_prefill.mojo` 同口径）。"""
    var v = List[Int]()
    for i in range(n):
        v.append(1000 + i * 7)
    return v^


def passes(n: Int, shards: Int) -> Int:
    """权重被读的遍数：`shards × ceil(每片行数 / RB)`。

    `RB` 跟着 `qwen.mojo` 的 `Q4_MUB_ROWS_MIN_TOK` 分流：每片 ≥4 行走 `RB=8`，
    否则 `RB=2`（decode 恒为 1 行 ⇒ 窄带）。⚠️ 这个模型假定 `RB` 能被整片吃完；
    `n=32, shards=8`（每片 4 行）时 `RB=8` 实际只用到 4，遍数仍是 1 —— 同式。
    """
    var per = (n + shards - 1) // shards
    var rb = 8 if per >= 4 else 2
    return shards * ((per + rb - 1) // rb)


def run_one[profile: Bool](
    mut model: QwenForward, ids: List[Int], shards: Int
) raises -> Int:
    """跑一趟 prefill，返回墙钟 ns；`profile=True` 时先清样本再计时。"""
    model.set_shards(shards)
    model.reset()
    comptime if profile:
        model.set_profile(True)
    var t0 = monotonic_ns()
    var lg = model.prefill[BACKEND_AVX2](ids)
    var el = monotonic_ns() - t0
    # argmax 必须留在计时外仍然被执行：否则优化器可以把整个前向当死代码。
    var nx = model.argmax(lg)
    if nx == -12345:
        print("      (unreachable)")
    return el


def main() raises:
    print("")
    print("== prefill 分段预算（真实权重 q4_0 + q8_head，单进程，avx2） ==")
    print(
        "   层投影每遍 "
        + String(PROJ_BYTES // (1024 * 1024))
        + " MB，输出头 "
        + String(HEAD_BYTES // (1024 * 1024))
        + " MB；整机 DRAM 天花板约 16.6 GB/s（账本 11-03 第八条）"
    )
    var model = QwenForward(FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK, True)
    model.enable_q8_output()

    var lengths = List[Int]()
    lengths.append(32)
    lengths.append(128)
    var shard_set = List[Int]()
    shard_set.append(1)
    shard_set.append(4)
    shard_set.append(8)

    # 预热：把 mmap 的页读进来，否则第一趟会把缺页算进 K_GEMM。
    var warm = token_ids(128)
    _ = run_one[False](model, warm, 1)
    _ = run_one[False](model, warm, 8)

    for li in range(len(lengths)):
        var n = lengths[li]
        var ids = token_ids(n)
        # 同长度先热身一遍（K/V 缓存按新长度重建）
        _ = run_one[False](model, ids, 1)
        for si in range(len(shard_set)):
            var sh = shard_set[si]
            var p = passes(n, sh)
            var wbytes = p * PROJ_BYTES + HEAD_BYTES
            # 取两趟的较小值（机器过载，min 比 mean 稳）
            var t1 = run_one[True](model, ids, sh)
            var t2 = run_one[True](model, ids, sh)
            var el = t1 if t1 < t2 else t2
            var gbs = Float64(wbytes) / Float64(el)
            print("")
            print(
                "   n="
                + String(n)
                + "  shards="
                + String(sh)
                + "   墙钟 "
                + f2(Float64(el) / 1e6)
                + " ms ⇒ 每 token "
                + f2(Float64(el) / Float64(n) / 1e3)
                + " µs"
            )
            print(
                "     遍数模型 P="
                + String(p)
                + " ⇒ 权重流量 "
                + f2(Float64(wbytes) / 1048576.0)
                + " MB ⇒ **隐含带宽 "
                + f2(gbs)
                + " GB/s**（贴 16.6 = 带宽受限；远低 = 受限在别处）"
            )
            # 先 dump 再关：`set_profile(False)` 会 reset()，把样本一起清掉。
            model.dump_profile()
            model.set_profile(False)

    print("")
    print("   读法：")
    print("   ① 隐含带宽贴着 16.6 GB/s ⇒ prefill 受**权重遍数**限，杠杆是"
          "「权重留寄存器、token 走内层」，不是再抠核。")
    print("   ② 隐含带宽远低于 16.6 ⇒ 遍数不是主要矛盾，先看 K_GEMM 段里"
          "算术/调度占比。")
    print("   ③ 插桩拖慢绝对值 ⇒ 段占比可信，每 token 时间只看量级。")
    model.keep_alive()
