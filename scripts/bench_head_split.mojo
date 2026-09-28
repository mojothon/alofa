"""一次 decode 里**输出投影占多少**、其余占多少 —— 当场切开量，不反推。

为什么要专门写它
----------------
q8_0 输出投影接上之后，q4 档从 `13.72…15.38` 提到 `17.70…20.23 tok/s`（+30%），但离
llama.cpp 的 25.60 还差 27%。那 27% 去哪儿了，**不能靠反推**：拿两个流量档去拟合
"带宽 + 固定开销"，等于把两档的流量估算误差一起除进答案里 —— 拟合出来的"每 token
24 ms 固定开销"是个诱人的数字，但它可能是我自己对"层流量有多少 MB"猜错了。

所以这里直接切：

    step_ms      一次完整的 `model.step`（层 + 输出投影 + argmax + …）
    head_ms      只调输出投影，**真实形状** 151936×896、**真实** `shards` 分片
    rest_ms      `step_ms − head_ms` = 层 + norm + attention + argmax + 其余

`rest_ms` 才是这轮要盯的数：如果它占了大部分，那么继续优化 q8 核就是**在已经不是
瓶颈的地方使劲**（objdump 已经确认那个核是 4×`vpmovsxbd`+`vcvtdq2ps`+`vfmadd231ps`
的真向量代码，每块 20 条向量指令 / 32 个值）。

⚠️ 边界（不可省略）
* 单独计时用的激活是**常数**向量，不是真的 hidden：这条脚本量的是**时间**不是数值，
  访存形态（一遍连续读块流 + 一次写 608 KB 的 `dst`）与真跑一致，数值正确性由
  `pixi run test-q8-head` 负责，不由这里负责。
* `head_ms` 里**不含** argmax（`step` 里那次是在 `step` 之后跑的，落在 `rest` 里）。
  所以 `rest` 不是纯层，是"除输出投影外的一切"。
* 本机长期过载 ⇒ 绝对值只作**同轮内部**比较；这里看的是**占比**，比绝对值稳。

跑法（`-O2`）
------------
    pixi run mojo build -O2 -I src scripts/bench_head_split.mojo -o target/bench_head_split
    ./target/bench_head_split
"""

from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.tensor import f32_data
from alofa.model.arch.qwen import (
    BACKEND_AVX2,
    QwenForward,
    q8_matmul_k_shards,
)
from alofa.tokenizer import load_tokenizer

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime PROMPT = "The capital of France is"
comptime MAX_TOK = 64
# 每档跑几趟取中位数。本机过载 ⇒ 少跑几趟、看中位数比看最好值诚实。
comptime ROUNDS = 7
comptime WARMUP = 3


def median_ns(mut steps: List[Int]) -> Int:
    """`steps` 的中位数（就地排序，趟数很小，插入排序够用）。

    ⚠️ 参数是 `mut` 而不是借用：`List` 不 `ImplicitlyCopyable`，而调用方后面还要
    重新建一个 List，就地排序正好省掉一次拷贝。
    """
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
    var neg = x < Float64(0)
    var v = x if not neg else -x
    var whole = Int(v)
    var frac = Int((v - Float64(whole)) * Float64(100) + Float64(0.5))
    if frac >= 100:
        whole += 1
        frac -= 100
    var s = String(whole) + "."
    if frac < 10:
        s = s + "0"
    s = s + String(frac)
    return ("-" + s) if neg else s


def main() raises:
    print("")
    print("== 一次 decode 里，输出投影 vs 其余 ==")
    print("   q4_0 层 + q8_0 输出投影；backend avx2；两档 shards 1 / 8")

    var tok = load_tokenizer(FIXTURE)
    var ids = tok.encode(PROMPT)
    print("[split] prompt=" + PROMPT + " tokens=" + String(len(ids)))

    var t0 = monotonic_ns()
    var model = QwenForward(FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK, True)
    var load_ms = (monotonic_ns() - t0) // 1000000
    t0 = monotonic_ns()
    model.enable_q8_output()
    var q8_ms = (monotonic_ns() - t0) // 1000000
    print("[split] 加载+量化（24 层 q4_0）: " + String(load_ms) + " ms")
    print("[split] 输出投影压 q8_0: " + String(q8_ms) + " ms")

    var h = model.cfg.hidden
    var vocab = model.cfg.vocab
    print("[split] hidden=" + String(h) + " vocab=" + String(vocab))

    # 输出投影的输入/输出缓冲。x 填常数：这条脚本量时间，不量数值。
    var arena = Arena(vocab * 4 + h * 4 + 128)
    var x = arena.alloc(h * 4).unsafe_bitcast[Float32]()
    var dst = arena.alloc(vocab * 4).unsafe_bitcast[Float32]()
    for i in range(h):
        x[unsafe_offset=i] = Float32(i % 13) * Float32(0.01) - Float32(0.06)

    var arms = List[Int]()
    arms.append(1)
    arms.append(8)

    for a in range(2):
        var shards = arms[a]
        model.set_shards(shards)
        model.reset()
        var lgt = model.prefill[BACKEND_AVX2](ids)
        var nx = model.argmax(lgt)

        # ① 完整 step
        for _ in range(WARMUP):
            nx = model.argmax(model.step[BACKEND_AVX2](nx))
        var steps = List[Int]()
        for _ in range(ROUNDS):
            var s0 = monotonic_ns()
            nx = model.argmax(model.step[BACKEND_AVX2](nx))
            steps.append(monotonic_ns() - s0)

        # ② 只跑输出投影
        for _ in range(WARMUP):
            q8_matmul_k_shards[BACKEND_AVX2](
                dst, x, model.q8_blocks[0], vocab, h, shards
            )
        var heads = List[Int]()
        for _ in range(ROUNDS):
            var s0 = monotonic_ns()
            q8_matmul_k_shards[BACKEND_AVX2](
                dst, x, model.q8_blocks[0], vocab, h, shards
            )
            heads.append(monotonic_ns() - s0)

        var step_ms = Float64(median_ns(steps)) / 1000000.0
        var head_ms = Float64(median_ns(heads)) / 1000000.0
        var rest_ms = step_ms - head_ms
        var share = head_ms / step_ms * Float64(100)
        print(
            "[split] shards="
            + String(shards)
            + " : step "
            + f2(step_ms)
            + " ms  =  输出投影 "
            + f2(head_ms)
            + " ms ("
            + f2(share)
            + "%)  +  其余 "
            + f2(rest_ms)
            + " ms"
        )
        # 后两个是**目标值**不是预测：它们分别回答"把另一半压到 0 能到多少"，
        # 用来判断该往哪边使劲 —— llama.cpp 的 25.60 是两边的共同参照。
        print(
            "[split]          换算：step "
            + f2(Float64(1000) / step_ms)
            + " tok/s；[其余] 清零可到 "
            + f2(Float64(1000) / head_ms)
            + "；输出投影清零可到 "
            + f2(Float64(1000) / rest_ms)
        )

    print("")
    arena.keep_alive()
