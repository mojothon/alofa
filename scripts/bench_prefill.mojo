"""端到端 prefill：片数 × prompt 长度，谁快？

问的是什么
----------
上一轮把 `avx2._gemm` 换成「`RB` 行一块、列外层」之后，`rows > 1` 的前向不再把
权重读 `rows` 遍 —— **但复用只在块内发生**。而 `prefill` 是按**输出行**切片的：
`rows` 行 ÷ `shards` 片 = 每片 `rows/shards` 行。于是

    rows=32, shards=8 → 每片 4 行 → `RB=4`，块内复用 4 份   （实测 2.68–3.66×）
    rows=8,  shards=8 → 每片 1 行 → `RB=1`，**块内没有第二行** （实测：量不出差别）

所以「片数」现在是 prefill 的一个真旋钮：**片数越少，每片行数越多，复用越充分；
片数越多，线程并行度越高**。这两个方向相反，谁是主导只能量。

口径
----
  * **必须 `-O2`**；每个 `(n, shards)` 预热一趟丢弃（它量的是 mmap 缺页）。
  * 3 趟，**轮外层、片数内层**交错（`arm = (rep + k) % 4` 那种共模消除的同一套
    道理：机器变慢时所有档一起慢），取 **min…max 区间**。
  * 主统计量是**每 token 时间**（总时间 ÷ `n`）。
  * **不变量自检**：同一段 id、同一份权重，前向与怎么切无关 —— 每个输出各自一个
    累加器、按同样的 `k` 次序累加，切分只决定"谁算哪几行"。故四种片数的
    **argmax + 前 16 个 logit 必须逐位相等**。不等说明有片没算或写坏了 —— 那会让
    时间偏快，把结论引向相反方向。
  * 判据沿用测量六条②：**区间不重合才算有差别**。

为什么不用分词器
----------------
`prefill` 吃的是 token id 列表；直接给 id 就少一个解释变量。

跑法：

    pixi run mojo build -O2 -I src scripts/bench_prefill.mojo -o target/bench_prefill
    ./target/bench_prefill
"""

from alofa.core.ffi import monotonic_ns
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.dtype import DT_FP32
from alofa.core.tensor import TensorView, f32_data, shape2
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"

# prompt 长度 8 / 16 / 32；片数 1 / 2 / 4 / 8。
comptime N_SIZES = 3
comptime SH_CASES = 4
comptime MAX_TOK = 128
comptime ROUNDS = 3


def n_of(c: Int) -> Int:
    """第 `c` 档 prompt 长度。"""
    if c == 0:
        return 8
    if c == 1:
        return 16
    return 32


def sh_of(s: Int) -> Int:
    """第 `s` 档片数。"""
    if s == 0:
        return 1
    if s == 1:
        return 2
    if s == 2:
        return 4
    return 8


def token_ids(n: Int) -> List[Int]:
    """一段确定的 id；都远小于词表 151936，形状之外不引入别的变量。"""
    var v = List[Int]()
    for i in range(n):
        v.append(1000 + i * 7)
    return v^


def prefill_once(mut model: QwenForward, n: Int, sink: RawPtr) raises -> Int:
    """跑一趟 `n` 个 token 的 prefill，返回纳秒；把**指纹**写进 `sink`。

    指纹 = argmax 与前 16 个 logit 之和。前向与怎么切无关，所以同一 `n` 下四种
    片数的指纹必须**逐位相等** —— 这是整份脚本的自检。
    """
    model.reset()
    var t0 = monotonic_ns()
    var logits = model.prefill[BACKEND_AVX2](token_ids(n))
    var dt = monotonic_ns() - t0
    var pv = logits
    var total = Float32(0)
    for i in range(16):
        total += pv[unsafe_offset=i]
    var pd = f32_data(TensorView(sink, shape2(1, 4), DT_FP32))
    pd[unsafe_offset=0] = Float32(model.argmax(logits))
    pd[unsafe_offset=1] = total
    return dt


def main() raises:
    print("")
    print("== 端到端 prefill：片数 × prompt 长度（复用只在块内发生）==")
    var model = QwenForward(FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK)
    var arena = Arena(64)
    var sink = arena.alloc(64)

    var ns = List[Int]()
    var fp = List[Float32]()
    for _ in range(N_SIZES * SH_CASES * ROUNDS):
        ns.append(0)
        fp.append(Float32(0))

    # 先量**默认片数**（一次 `set_shards` 都不调）：改动到底有没有端到端生效，
    # 只有这一行说了算 —— 下面那张扫描表是给"该定几片"用的，它显式设了片数，
    # 会把策略绕过去。
    for c in range(N_SIZES):
        var n = n_of(c)
        _ = prefill_once(model, n, sink)
        var lo = Int(1) << 62
        var hi = Int(0)
        for r in range(ROUNDS):
            var dt = prefill_once(model, n, sink)
            if dt < lo:
                lo = dt
            if dt > hi:
                hi = dt
        print(
            "  默认片数  n =",
            n,
            "  每 token",
            Float64(lo) / Float64(n) / 1e6,
            "…",
            Float64(hi) / Float64(n) / 1e6,
            "ms",
        )

    for c in range(N_SIZES):
        var n = n_of(c)
        # 预热：每个 (n, shards) 的第一趟要付 mmap 缺页，那不是前向。
        for s in range(SH_CASES):
            model.set_shards(sh_of(s))
            _ = prefill_once(model, n, sink)
        # 轮外层、片数内层 → 交错，机器变慢时所有档一起慢。
        for r in range(ROUNDS):
            for s in range(SH_CASES):
                model.set_shards(sh_of(s))
                var i = (c * SH_CASES + s) * ROUNDS + r
                ns[i] = prefill_once(model, n, sink)
                var pd = f32_data(TensorView(sink, shape2(1, 4), DT_FP32))
                fp[i] = pd[unsafe_offset=0] + pd[unsafe_offset=1]

        # 自检：四种片数必须逐位相等。
        var same = True
        for s in range(1, SH_CASES):
            for r in range(ROUNDS):
                if fp[(c * SH_CASES + s) * ROUNDS + r] != fp[c * SH_CASES * ROUNDS + r]:
                    same = False

        print("")
        print(
            "  n =",
            n,
            "  自检",
            "通过（四种片数逐位相等）" if same else "失败 → 这格的数作废",
        )
        for s in range(SH_CASES):
            var lo = ns[(c * SH_CASES + s) * ROUNDS]
            var hi = ns[(c * SH_CASES + s) * ROUNDS]
            for r in range(1, ROUNDS):
                if ns[(c * SH_CASES + s) * ROUNDS + r] < lo:
                    lo = ns[(c * SH_CASES + s) * ROUNDS + r]
                if ns[(c * SH_CASES + s) * ROUNDS + r] > hi:
                    hi = ns[(c * SH_CASES + s) * ROUNDS + r]
            print(
                "      shards =",
                sh_of(s),
                "  每 token",
                Float64(lo) / Float64(n) / 1e6,
                "…",
                Float64(hi) / Float64(n) / 1e6,
                "ms",
            )

    arena.keep_alive()
