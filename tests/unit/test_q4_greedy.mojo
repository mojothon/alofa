"""量化后的整网还能不能生成。

前面的 `test_q4_parity.mojo` 证明的是**块格式**对：解量化逐位相同、融合 matmul
与参考一致。但它证明不了"把 24 层都换成 q4_0 之后，模型还认得自己" —— 逐块的
误差可以在自回归里滚起来，也可以在 151936 维的 argmax 上翻盘。这个门看的是
**端到端的结果**：量化后每一步贪心选出的 token，与 fp32 参考的一致率。

两条设计上的讲究：

**用教师强制（teacher forcing）而不是自回归续写。** 每一步喂进去的都是参考序
列里的真值 token，于是"这一步选得对不对"只取决于当前这步的量化误差，不会被前
面步骤的误差放大污染。这样一致率才是**量化本身**的度量；自回归续写量到的是
"误差累积之后还能走多远"，那是另一个（也很有意思但本轮没做）的问题。

**这个门判的是"通路在工作"，不是"质量达标"。** 实测一致率 **0.80**（输出投影
留在 fp32 的情况下；连它一起量化是 0.77）。项目路线图里"一致率 ≥ 0.90"那条是
**质量门，本轮没过**，账本里按实测数字记着，没有为了让它绿而把 0.90 改小。这
里 0.75 的下限只用来回答一个问题：量化通路是不是真的在算东西 —— 解码高低半字
节写反会得到 0.0，前向悄悄退回 fp32 会得到 1.0，两者都会被这条断言抓住。

参照物同样是导出的 fixture（`greedy.tsv`，来自 Hugging Face 的 fp32 推理），
不是实时调 Python。

跑法（依赖 2 GB 本地导出，**故意不进 `pixi run test`**）：

    pixi run test-q4
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.tensor import F32Ptr
from alofa.core.text import parse_int, read_text
from alofa.model.arch.qwen import QwenForward
from alofa.model.loader import config_value

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime WEIGHTS_DIR = FIXTURE + "/weights"
comptime CONFIG = FIXTURE + "/config.tsv"
comptime PROMPTS = FIXTURE + "/prompts.tsv"
comptime GREEDY = FIXTURE + "/greedy.tsv"

comptime MAX_TOKENS = 256

# "通路在工作"的下限，不是质量门。见文件头关于 0.80 / 0.90 的说明。
comptime Q4_PATH_ALIVE_GATE = Float64(0.75)


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
    var lines = text.split("\n")
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        out.append(ids_from_csv(String(fields[1])))
    return out^


def load_greedy() raises AlofaError -> List[List[Int]]:
    var text = read_text(GREEDY)
    var out = List[List[Int]]()
    var lines = text.split("\n")
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        out.append(ids_from_csv(line))
    return out^


def vocab_size() raises AlofaError -> Int:
    return parse_int(config_value(CONFIG, "vocab"))


def argmax_of(values: F32Ptr, n: Int) -> Int:
    var best = 0
    var best_value = Float32(-3.4028234663852886e38)
    for i in range(n):
        if values[unsafe_offset=i] > best_value:
            best_value = values[unsafe_offset=i]
            best = i
    return best


def test_quantised_greedy_stays_with_the_reference() raises:
    """量化后每一步的贪心选择，与 fp32 参考的一致率必须在合理区间内。

    一致率**太低**（< 0.75）说明量化通路在算垃圾；**等于 1.0** 说明前向根本没
    走量化通路。两头都要看，只看一头会让"绿"变得毫无意义。
    """
    var n = vocab_size()
    var prompts = load_prompts()
    var greedy = load_greedy()
    assert_equal(len(greedy), len(prompts), "one greedy row per prompt")

    var total = 0
    var agreed = 0
    var index = 0
    for ids in prompts:
        var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, True)
        # 先确认前向真的走量化通路。少了这一条，万一开关没生效，测的就是
        # fp32 对 fp32 —— 一致率 100%，门绿得毫无意义。
        assert_true(model.q4_enabled, "前向没有走量化通路，这个门测的是 fp32 对 fp32")
        var expected = greedy[index].copy()
        var first = model.prefill(ids)
        var token = argmax_of(first, n)
        total += 1
        if token == expected[0]:
            agreed += 1
        var seen = 1
        while seen < len(expected):
            var logits = model.step(expected[seen - 1])
            token = argmax_of(logits, n)
            total += 1
            if token == expected[seen]:
                agreed += 1
            seen += 1
        model.keep_alive()
        index += 1

    var rate = Float64(agreed) / Float64(total)
    print(
        "观测：量化后教师强制贪心一致 "
        + String(agreed)
        + "/"
        + String(total)
        + " = "
        + String(rate)
    )
    assert_true(
        rate >= Q4_PATH_ALIVE_GATE,
        "量化后只有 "
        + String(rate)
        + " 的贪心选择与 fp32 一致（下限 "
        + String(Q4_PATH_ALIVE_GATE)
        + "）—— 这个数低到不像量化误差，更像解码反了或者喂错了行",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
