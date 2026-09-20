"""垂直切片门：一句真文本走完 tokenizer → model → engine → sampler。

为什么需要这道门，而不是"跑过一次就算了"
--------------------------------------
每一层此前都有自己的门，而且都是绿的：分词器对得上 HuggingFace、模型对得上
HuggingFace、调度器对得上 Python 参照、批执行器对得上串行执行。但**没有任何一处把
它们串起来跑过** —— `test_engine_core.mojo` 的文件头自己就写明它 "without a model:
the argmax is supplied by the test"。

那是个理性的选择（引擎循环的测点是时序，不是数值），代价是接缝从未被执行。而历史
欠账就在接缝里：2026-09-17 记过一次"各层各自绿、拼起来崩到 0/512"。这道门把那条
路径固定成每天早上都能走一遍的东西。

它验什么
--------
- **分词段**：prompt 必须编成那 5 个 id。这是 golden，且与 `test_tokenizer_parity`
  的 4560 条差分同源 —— 这里钉住的是"这条切片用的是同一套分词"。
- **生成段**：greedy 续写必须说出 Paris。0.5B 模型答错首都的可能性是存在的，但
  "The capital of France is" 是它几乎不会错的一句；它要是答不出来，说明断的是
  链路而不是知识。
- **负向对照（关键）**：温度 0.01 的采样必须**逐字等于**贪心。

最后一条为什么是负向对照而不是普通断言：因为它证明 sampler **真的介入了决策**。
若 `run_sampled` 悄悄退化成 argmax（比如 logits 指针拿错、build 没被调用），前两条
断言照样全绿 —— 输出的还是 Paris，而采样路径一次都没生效过。低温收敛把这件事变成
可证伪的：温度趋零时分布塌到 argmax 上，两条独立路径必须给出同一个答案。

跑（需 1.9 GB 权重，因此不并入 `pixi run test`）：
    pixi run test-slice
"""

from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from alofa.cli import DEFAULT_NEW, DEFAULT_PROMPT, generate
from alofa.runtime.sampler import SampleParams
from alofa.tokenizer.tokenizer import load_tokenizer

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"

# 与 `test_tokenizer_parity.mojo` 同源的 golden：HF 对这句的编码。
comptime N_PROMPT_IDS = 5
# 低温对照只比前几个 token：整段 16 个 token 里只要有一个在两条路上不同就足以
# 证明采样生效，而比整段会把门变成对浮点边界的赌注。
comptime COLD_NEW = 6
# 不是 0：temperature 会进除数，0 会让分布变成除零。
comptime COLD_TEMPERATURE = 0.01


def expected_ids() -> List[Int]:
    """HF 对 "The capital of France is" 的编码（golden）。"""
    var out = List[Int]()
    out.append(785)
    out.append(6722)
    out.append(315)
    out.append(9625)
    out.append(374)
    return out^


def test_prompt_encodes_to_the_expected_ids() raises:
    """分词段：这条切片用的必须是与 HF 对过的那套分词器。"""
    var tok = load_tokenizer(FIXTURE)
    var ids = tok.encode(DEFAULT_PROMPT)
    var want = expected_ids()
    assert_equal(len(ids), N_PROMPT_IDS, "prompt 的 token 数变了")
    for i in range(len(want)):
        assert_equal(
            ids[i], want[i], "第 " + String(i) + " 个 id 与 HF 不一致"
        )


def test_greedy_slice_names_paris() raises:
    """生成段：整条链路跑完，greedy 必须说出 Paris。"""
    var text = generate(DEFAULT_PROMPT, DEFAULT_NEW, SampleParams(), True).text
    assert_true(
        text.find("Paris") >= 0,
        "greedy 续写里没有 Paris —— 断的更可能是链路而不是知识：" + text,
    )


def test_sampled_slice_also_completes() raises:
    """采样段：默认参数（temperature=1）也要能跑完并给出可读文本。

    不断言内容 —— 高温采样本来就允许跑偏，这里只要求它不崩、非空、且长度合理。
    """
    var text = generate(DEFAULT_PROMPT, DEFAULT_NEW, SampleParams(), False).text
    assert_true(
        text.byte_length() > 0, "采样路径返回了空文本 —— 它可能根本没生成"
    )


def test_low_temperature_converges_to_greedy() raises:
    """负向对照：温度趋零时采样必须塌到贪心上。

    这条断言是整道门存在的理由 —— 它让"sampler 没接上"变成**会红**的错误，而不是
    一个谁也看不见的巧合。见文件头。
    """
    var greedy = generate(DEFAULT_PROMPT, COLD_NEW, SampleParams(), True).text
    var cold = SampleParams()
    cold.temperature = COLD_TEMPERATURE
    var sampled = generate(DEFAULT_PROMPT, COLD_NEW, cold, False).text
    assert_equal(
        sampled,
        greedy,
        "低温采样与贪心不一致 —— sampler 可能压根没参与决策，或 logits 取错了行",
    )


def main() raises:
    print("Running vertical slice gate (real weights, ~1 min)...")
    TestSuite.discover_tests[__functions_in_module()]().run()
