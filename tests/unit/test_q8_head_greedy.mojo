"""q8_0 输出投影接进整网之后，模型还认不认得自己。

`test_q8_parity.mojo` 证明的是**块格式**对：解量化逐位相同、量化逐字节相同、融
合 matmul 与参考一致。它证明不了"把 151936 × 896 的输出投影换成 q8_0 之后，端到
端还能不能生成" —— 逐块 0.54% 的误差会在 151936 维的 argmax 上翻盘，也会在自回
归里滚起来。这个门看的是**端到端**：教师强制下每一步贪心选出的 token，与 fp32
参考的一致率。

**为什么是 8 bit 而不是 4 bit**（账本里量过，这里是它的断言形态）：输出投影压到
q4_0 会让教师强制贪心一致率从 **0.8691 掉到 0.77**（省 76 MB 不值）；q8_0 的相对
L2 是 **0.5359%**，q4_0 是 10.3%。所以本门不单看"q8 档一致率够高"，还要看**它与
"输出投影留在 fp32"那一档差多少** —— 差得越少，"这条通路值得开"这句话才站得住。

**负向对照。** 一致率只是一个数，只看它会绿得毫无意义：
- 若通路**静默退回 fp32**，一致率会正好等于 fp32 那一档，看起来"很好"。所以另有
  一条断言：**q8 档的 logits 必须与 fp32 档不同**（数值层面证明块流真的在用）；
- 若**符号扩展丢了**或**行步长算错**，logits 会直接变成垃圾：一致率崩到 0.75 以
  下，同时相对 L2 冲过 1%。两头一起看，任一侧单独失效都会红。

⚠️ **tied embedding 是最容易漏的一侧**：`lm_head` 与 `embed_tokens` 是**同一份字
节**，量化它之后 **prefill 的查表也必须改读块流**，否则 prefill 会把 q8 的字节当
fp32 解释 —— 而且**只在 prefill 上错**，decode 看起来完全正常。这一侧单独有一条
用例盯着（按行解出的 embedding 行 vs fp32 那一行）。

跑法（依赖 2 GB 本地导出，**故意不进 `pixi run test`**）：

    pixi run test-q8-head
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import ERR_SHAPE_MISMATCH, AlofaError
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, f32_data
from alofa.core.text import parse_int, read_text
from alofa.kernels.cpu.quant import Q8_BLOCK, Q8_BYTES, dequant_q8_0
from alofa.model.arch.qwen import BACKEND_AVX2, BACKEND_SCALAR, EMBED, QwenForward
from alofa.model.loader import config_value

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime WEIGHTS_DIR = FIXTURE + "/weights"
comptime CONFIG = FIXTURE + "/config.tsv"
comptime PROMPTS = FIXTURE + "/prompts.tsv"
comptime GREEDY = FIXTURE + "/greedy.tsv"

comptime MAX_TOKENS = 256

# "通路在算东西"的下限：与 `test_q4_greedy.mojo` 的回归下限同值（0.85）。含义也
# 一样 —— 低于它不像量化误差，更像解码反了或者喂错了行。
comptime Q8_PATH_ALIVE_GATE = Float64(0.85)

# **q8 输出投影相对"输出投影留在 fp32"的容忍差。** 0.03 不是随手取的：q8_0 的权重
# 相对 L2 是 0.5359%，而 151936 维 argmax 上翻盘的是**最接近的那几个** logit ——
# 4 条 prompt 合计上百步里，它该改动的只是个位数。若这里掉到 0.03 以上，说明误差
# 比"权重宽 8 bit"该有的样子大得多，那条通路就不该开。
comptime Q8_HEAD_MAX_DROP = Float64(0.03)

# logits 的相对 L2 上界 1%（平方比 1e-4）：权重那侧是 0.5359%，过 1% 就是别的错。
comptime LOGITS_REL_L2_MAX_SQ = Float64(1e-4)


def ids_from_csv(text: String) raises AlofaError -> List[Int]:
    var ids = List[Int]()
    if text.byte_length() == 0:
        return ids^
    var parts = text.split(",")
    for part_span in parts:
        var part = String(part_span)
        if part.byte_length() == 0:
            continue
        ids.append(parse_int(part))
    return ids^


def load_prompts() raises AlofaError -> List[List[Int]]:
    var text = read_text(PROMPTS)
    var out = List[List[Int]]()
    for line_span in text.split("\n"):
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        out.append(ids_from_csv(String(fields[1])))
    return out^


def load_greedy() raises AlofaError -> List[List[Int]]:
    var text = read_text(GREEDY)
    var out = List[List[Int]]()
    for line_span in text.split("\n"):
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        out.append(ids_from_csv(line))
    return out^


def config_int(name: String) raises AlofaError -> Int:
    return parse_int(config_value(CONFIG, name))


def argmax_of(values: F32Ptr, n: Int) -> Int:
    var best = 0
    var best_value = Float32(-3.4028234663852886e38)
    for i in range(n):
        if values[unsafe_offset=i] > best_value:
            best_value = values[unsafe_offset=i]
            best = i
    return best


def _agreement_rates[backend: Int]() raises -> List[Float64]:
    """两档同跑，返回 `[输出投影 fp32 的一致率, q8_0 的一致率]`。

    两档共用同一份**教师强制**序列，于是"这一步选得对不对"只取决于当前这步的量化
    误差，不会被前面步骤的误差放大污染 —— 一致率的差才是**输出投影量化本身**的度量。

    `backend` 是参数而不是写死默认：端到端那个 "+30%" 是**向量**后端跑出来的（标量
    版反而更慢），所以两条后端都得量 —— 只测默认那条，等于没测提速走的那条路。
    """
    var n = config_int("vocab")
    var prompts = load_prompts()
    var greedy = load_greedy()
    assert_equal(len(greedy), len(prompts), "one greedy row per prompt")

    var plain = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, True)
    var quant = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, True)
    quant.enable_q8_output()
    assert_true(
        quant.q8_enabled, "输出投影没有走 q8_0 通路，这个门测的是 fp32 对 fp32"
    )

    var total = 0
    var plain_agreed = 0
    var quant_agreed = 0
    var index = 0
    for ids in prompts:
        plain.reset()
        quant.reset()
        var expected = greedy[index].copy()
        var lp = plain.prefill[backend](ids)
        var lq = quant.prefill[backend](ids)
        total += 1
        if argmax_of(lp, n) == expected[0]:
            plain_agreed += 1
        if argmax_of(lq, n) == expected[0]:
            quant_agreed += 1
        var seen = 1
        while seen < len(expected):
            var fed = expected[seen - 1]
            var sp = plain.step[backend](fed)
            var sq = quant.step[backend](fed)
            total += 1
            if argmax_of(sp, n) == expected[seen]:
                plain_agreed += 1
            if argmax_of(sq, n) == expected[seen]:
                quant_agreed += 1
            seen += 1
        index += 1

    var rate_plain = Float64(plain_agreed) / Float64(total)
    var rate_quant = Float64(quant_agreed) / Float64(total)
    print(
        "观测：教师强制贪心一致率（backend="
        + String(backend)
        + "）—— 输出投影 fp32 "
        + String(rate_plain)
        + " ，q8_0 "
        + String(rate_quant)
        + " （共 "
        + String(total)
        + " 步）"
    )
    var out = List[Float64]()
    out.append(rate_plain)
    out.append(rate_quant)
    plain.keep_alive()
    quant.keep_alive()
    return out^


def _assert_rates(rates: List[Float64]) raises:
    var rate_plain = rates[0]
    var rate_quant = rates[1]
    assert_true(
        rate_quant >= Q8_PATH_ALIVE_GATE,
        "q8_0 输出投影的一致率只有 "
        + String(rate_quant)
        + "（下限 "
        + String(Q8_PATH_ALIVE_GATE)
        + "）—— 低到不像量化误差，更像块流读错了",
    )
    var drop = rate_plain - rate_quant
    if drop < Float64(0):
        drop = -drop
    assert_true(
        drop <= Q8_HEAD_MAX_DROP,
        "q8_0 输出投影让一致率从 "
        + String(rate_plain)
        + " 掉到 "
        + String(rate_quant)
        + "（差 "
        + String(drop)
        + "，容忍 "
        + String(Q8_HEAD_MAX_DROP)
        + "）—— 输出投影压 q4_0 时掉的是 0.8691 → 0.77 那种量级，"
        + "q8_0 不该是这个样子",
    )


def test_q8_head_holds_the_greedy_rate() raises:
    """标量后端：输出投影压成 q8_0 之后，一致率必须几乎不动。"""
    _assert_rates(_agreement_rates[BACKEND_SCALAR]())


def test_q8_head_vector_backend_holds_the_greedy_rate() raises:
    """向量后端：同上 —— **这条才是端到端提速走的那条路**。

    为什么必须单独有一条：向量版与标量版的累加结构不同（f32 通道 + 行末归约 vs
    f64 一路累加），它们**不逐位相等**，所以标量版绿推不出向量版绿。而基准里那个
    "+30%" 是向量版跑出来的 —— 只测默认后端，等于没测提速走的那条路。
    """
    _assert_rates(_agreement_rates[BACKEND_AVX2]())


def test_q8_head_logits_differ_but_stay_close() raises:
    """负向对照 + 数值上界：logits 必须**与 fp32 不同**，但差得不超过 1%。

    两侧一起看才有意义：
    - 完全相同 ⇒ 前向**根本没走** q8 通路（一致率那条会绿得毫无意义）；
    - 差过 1% ⇒ 权重那侧是 0.5359%，logits 不该比它大一个量级。
    """
    var n = config_int("vocab")
    var prompts = load_prompts()
    var greedy = load_greedy()

    var plain = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, True)
    var quant = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, True)
    quant.enable_q8_output()

    var differing = 0
    var diff_sq = Float64(0)
    var ref_sq = Float64(0)
    var index = 0
    for ids in prompts:
        plain.reset()
        quant.reset()
        var expected = greedy[index].copy()
        _accumulate(plain.prefill(ids), quant.prefill(ids), n, differing, diff_sq, ref_sq)
        var seen = 1
        while seen < len(expected):
            var fed = expected[seen - 1]
            _accumulate(plain.step(fed), quant.step(fed), n, differing, diff_sq, ref_sq)
            seen += 1
        index += 1

    print(
        "观测：q8_0 与 fp32 输出投影的 logits —— 不同的位置 "
        + String(differing)
        + " 个，相对 L2 平方比 "
        + String(diff_sq / ref_sq)
    )
    assert_true(
        differing > 0,
        "q8_0 与 fp32 的 logits 完全相同 —— 输出投影压根没走量化通路",
    )
    assert_true(ref_sq > Float64(0), "logits 全为零，这条用例没有意义")
    assert_true(
        diff_sq / ref_sq < LOGITS_REL_L2_MAX_SQ,
        "q8_0 与 fp32 的 logits 相对 L2 平方比是 "
        + String(diff_sq / ref_sq)
        + "（上界 "
        + String(LOGITS_REL_L2_MAX_SQ)
        + "）—— 权重那侧只有 0.5359%，这里不该大一个量级",
    )
    plain.keep_alive()
    quant.keep_alive()


def _accumulate(
    plain: F32Ptr,
    quant: F32Ptr,
    n: Int,
    mut differing: Int,
    mut diff_sq: Float64,
    mut ref_sq: Float64,
):
    for i in range(n):
        var a = Float64(plain[unsafe_offset=i])
        var b = Float64(quant[unsafe_offset=i])
        if plain[unsafe_offset=i] != quant[unsafe_offset=i]:
            differing += 1
        var d = a - b
        diff_sq += d * d
        ref_sq += a * a


def test_q8_embedding_rows_match_the_fp32_rows() raises:
    """tied embedding 那一侧：查表按行解出来的行，必须与 fp32 的那一行吻合。

    为什么单独开一条：`lm_head` 与 `embed_tokens` 是**同一份字节**，输出投影量化
    之后 prefill 的查表也得改读块流。这条路径只在 prefill 上走，一致率那条门对它
    的覆盖是间接的 —— 行步长 `h / 32 * 34` 算错一个字，prefill 就整段喂错，而
    decode 看起来完全正常。
    """
    var h = config_int("hidden")
    var vocab = config_int("vocab")
    var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, False)
    model.enable_q8_output()
    assert_true(model.q8_enabled, "q8_0 通路没开，这条用例测的是 fp32 对 fp32")
    if h % Q8_BLOCK != 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "a q8 embedding row must be a whole number of blocks",
            "h=" + String(h),
        )

    var row_bytes = h // Q8_BLOCK * Q8_BYTES
    var pe = f32_data(model.params.view(EMBED))
    var blocks = model.q8_blocks[0]
    var arena = Arena(h * 4 + 64)
    var got = arena.alloc(h * 4).unsafe_bitcast[Float32]()

    # 头一行、一个中间行、接近词表末尾的一行：行偏移在大数上才容易露出乘法错误。
    var rows = List[Int]()
    rows.append(0)
    rows.append(1)
    rows.append(1234)
    rows.append(vocab - 1)

    var diff_sq = Float64(0)
    var ref_sq = Float64(0)
    var differing = 0
    for row in rows:
        dequant_q8_0(
            got, blocks.unsafe_offset(row * row_bytes).unsafe_bitcast[UInt8](), h
        )
        for j in range(h):
            var want = pe[unsafe_offset=row * h + j]
            if got[unsafe_offset=j] != want:
                differing += 1
            var d = Float64(got[unsafe_offset=j]) - Float64(want)
            diff_sq += d * d
            var w = Float64(want)
            ref_sq += w * w

    print(
        "观测：按行解出的 embedding 与 fp32 —— 不同的位置 "
        + String(differing)
        + " 个，相对 L2 平方比 "
        + String(diff_sq / ref_sq)
    )
    assert_true(differing > 0, "解出的 embedding 行与 fp32 完全相同 —— 没真的量化")
    assert_true(ref_sq > Float64(0), "embedding 行全为零，这条用例没有意义")
    assert_true(
        diff_sq / ref_sq < LOGITS_REL_L2_MAX_SQ,
        "按行解出的 embedding 相对 L2 平方比是 "
        + String(diff_sq / ref_sq)
        + " —— 行步长算错会让这里直接爆掉",
    )
    model.keep_alive()
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
