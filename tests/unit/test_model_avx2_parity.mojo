"""整网跑在**向量后端**上时，必须落在与 fp32 参考同一个地方。

`tests/unit/test_avx2_parity.mojo` 验的是**算子**：五个核各自对着同一份
fixture，与标量后端逐位相同。它验不到"整网" —— 算子装进 24 层、跑 128 步
自回归之后，模型还认不认得自己。

为什么要单独一个文件，而不是给 `test_model_parity` 加个参数：两条门要回答的
问题不同。标量整网门回答"前向对不对"，这条回答"**换掉实现之后**前向还对不对"。
前者红了后者一定红，于是后者的失败是有歧义的 —— 分不清是模型错了还是后端错了。
分开之后，两个红各自有唯一解释。

**这条门能证明什么、不能证明什么**（写下来是因为它决定了怎么读这个绿）：

- 能证明：整网装在 `BACKEND_AVX2` 上时，logits 的 cos 与 argmax、128 步贪心
  、以及逐 token 解码，都落在 HF 参考上。**向量核写错 → 这条门红**，因为参照
  物（fp32 参考）与两个后端都无关。
- **不能**用数值证明"跑的确实是向量后端"：向量后端与标量后端在这些形状上是
  **逐位相同**的（上一轮量出来的，也是设计目标），所以任何数值比较都区分不了
  它们。能区分的那条对照在 `test_avx2_parity.mojo` 里（丢掉标量尾巴的版本必须
  判红）；而"后端参数有没有真的接上"由下面第一条断言加 `pixi run
  test-backend-guard` 那道编译期门盯着 —— 拼错的后端常量必须**编译失败**，
  不能悄悄退化成标量。

Run:
    pixi run test-model-avx2
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.mmap import MappedFile
from alofa.core.tensor import F32Ptr, f32_data
from alofa.core.text import parse_int, read_text
from alofa.model.arch.qwen import (
    BACKEND_AVX2,
    BACKEND_SCALAR,
    QwenForward,
    backend_label,
)
from alofa.model.loader import config_value
from alofa.verify.compare import cosine, max_abs_diff

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime WEIGHTS_DIR = FIXTURE + "/weights"
comptime CONFIG = FIXTURE + "/config.tsv"
comptime PROMPTS = FIXTURE + "/prompts.tsv"
comptime GREEDY = FIXTURE + "/greedy.tsv"
comptime LOGITS = FIXTURE + "/logits_last.f32"

# 与标量整网门**同一组**常量：同一批 prompt、同一段参考、同一个 cos 门。
# 判据不许在换后端的时候跟着换 —— 换的是实现，不是标准。
comptime MAX_TOKENS = 256
comptime COSINE_GATE = Float64(0.999)


def ids_from_csv(text: String) raises AlofaError -> List[Int]:
    """A comma-separated line of token ids."""
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
    """Prompt token ids; the text column is hex and only for humans."""
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
    """Reference greedy continuations, one line of ids per prompt."""
    var text = read_text(GREEDY)
    var out = List[List[Int]]()
    for line_span in text.split("\n"):
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        out.append(ids_from_csv(line))
    return out^


def vocab_size() raises AlofaError -> Int:
    return parse_int(config_value(CONFIG, "vocab"))


def reference_logits(mapped: MappedFile, prompt: Int) raises AlofaError -> F32Ptr:
    """The reference logits row for one prompt, from the mapped fixture."""
    var n = vocab_size()
    return mapped.ptr().unsafe_offset(prompt * n * 4).unsafe_bitcast[Float32]()


def argmax_of(values: F32Ptr, n: Int) -> Int:
    """Largest element's index, ties to the lowest index (as `torch.argmax`)."""
    var best = 0
    var best_value = Float32(-3.4028234663852886e38)
    for i in range(n):
        if values[unsafe_offset=i] > best_value:
            best_value = values[unsafe_offset=i]
            best = i
    return best


def test_the_backend_label_matches_the_constant_we_ask_for() raises:
    """`backend_label` 必须与传进去的常量一致，两个方向都断言。

    否则这条断言是恒真的 —— 若 `backend_label` 永远返回 `"avx2"`，只看一个
    方向看不出来。它守的是"下面几条的 `BACKEND_AVX2` 真的是向量那套"：名字
    与算子分发共用同一个判断，判断写错了这里先红。
    """
    assert_equal(backend_label[BACKEND_AVX2](), "avx2", "avx2 后端的名字不对")
    assert_equal(backend_label[BACKEND_SCALAR](), "scalar", "scalar 后端的名字不对")


def test_prefill_logits_match_reference() raises:
    """Cosine and argmax of the last position, for every prompt."""
    var n = vocab_size()
    var mapped = MappedFile(LOGITS)
    var prompts = load_prompts()
    assert_true(len(prompts) > 0, "the fixture has no prompts")

    var index = 0
    for ids in prompts:
        var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
        var logits = model.prefill[BACKEND_AVX2](ids)
        var expected_logits = reference_logits(mapped, index)
        var sim = cosine(logits, expected_logits, n)
        assert_true(
            sim >= COSINE_GATE,
            "logits cosine below the gate: prompt="
            + String(index)
            + " cosine="
            + String(sim),
        )
        assert_equal(
            argmax_of(logits, n),
            argmax_of(expected_logits, n),
            "argmax differs from the reference: prompt=" + String(index),
        )
        model.keep_alive()
        index += 1
    mapped.keep_alive()


def test_greedy_generation_matches_token_for_token() raises:
    """128 tokens, identical to the reference, for every prompt.

    这条是向量后端真正吃劲的地方：一个只在算子级看不出差别的实现误差，会在
    128 步自回归里滚成"第 30 个 token 开始跑偏"。
    """
    var n = vocab_size()
    var prompts = load_prompts()
    var greedy = load_greedy()
    assert_equal(len(greedy), len(prompts), "one greedy row per prompt")

    var index = 0
    for ids in prompts:
        var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
        var expected = greedy[index].copy()
        var first = model.prefill[BACKEND_AVX2](ids)
        var seen = 0
        var token = argmax_of(first, n)
        assert_equal(
            token, expected[0], "first greedy token differs: prompt=" + String(index)
        )
        seen += 1
        while seen < len(expected):
            var logits = model.step[BACKEND_AVX2](expected[seen - 1])
            token = argmax_of(logits, n)
            assert_equal(
                token,
                expected[seen],
                "greedy differs at token "
                + String(seen)
                + ": prompt="
                + String(index),
            )
            seen += 1
        model.keep_alive()
        index += 1


def test_incremental_decode_matches_full_prefill() raises:
    """One token at a time must land where consuming the whole prompt lands.

    注意力（`attention`）与旋转位置编码（`rope`）本轮只有标量实现，所以这条
    同时也守着"向量路径与标量路径混跑时两个阶段仍然一致" —— 混跑不是理想
    状态，但它就是当前的事实，得有人盯着。
    """
    var n = vocab_size()
    var prompts = load_prompts()
    var index = 0
    for ids in prompts:
        var whole = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
        var at_once = whole.prefill[BACKEND_AVX2](ids)

        var stepped = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
        var one_at_a_time = stepped.step[BACKEND_AVX2](ids[0])
        for i in range(1, len(ids)):
            one_at_a_time = stepped.step[BACKEND_AVX2](ids[i])

        var diff = max_abs_diff(at_once, one_at_a_time, n)
        assert_true(
            diff <= Float64(1e-4),
            "incremental decode differs from prefill: prompt="
            + String(index)
            + " max_abs_diff="
            + String(diff),
        )
        whole.keep_alive()
        stepped.keep_alive()
        index += 1


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
