"""`vpmaddubsw` 通路的**整网**质量门（backend = `BACKEND_AVX2`）。

为什么单独一个门，而不是让 `test-q4` 顺带覆盖
--------------------------------------------
`test-q4`（`tests/unit/test_q4_greedy.mojo`）跑的是**标量后端**（`prefill` / `step`
的 `backend` 参数默认 `BACKEND_SCALAR`），而 `vpmaddubsw` 只存在于向量分支 ——
2026-09-29 之前这条通路就是"开关开着但没人测"，于是"无增益"和"一致率没变"两个假象
同时出现：它们都长着"开关没生效"的样子。这个门把后端换成 `BACKEND_AVX2`，用**同一份
fixture、同一套教师强制贪心、同一组下限**（0.75 / 0.85）再判一遍。

判什么、不判什么
----------------
- 判：**质量** —— 0.75 是"通路在工作"（解码高低半字节写反会得到 0.0，前向悄悄退回
  fp32 会得到 1.0，都会被它抓住），0.85 是"不该掉的掉了"。实测 **444/512 = 0.8672**
  （默认 f32 展开是 445/512 = 0.8691，只差一个 token）。
- 不判：这条**分不清**"核写错了"和"开关没接到" —— 那两种死法由核级门
  `tests/unit/test_q4_mub_rows.mojo`（进 `pixi run test`，不要 2 GB 权重）分别钉住。
- 也不判"跑的是不是 mub"：那是下面第一条断言的事（它只保证开关读到的值，真正证明
  在算的是核级门那条"两条通路不许逐位相同"）。

跑法（`-O2`，依赖 2 GB 本地导出，**故意不进 `pixi run test`**）：

    pixi run test-q4-mub
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.tensor import F32Ptr
from alofa.core.text import parse_int, read_text
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward, q4_mub_enabled
from alofa.model.loader import config_value

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime WEIGHTS_DIR = FIXTURE + "/weights"
comptime CONFIG = FIXTURE + "/config.tsv"
comptime PROMPTS = FIXTURE + "/prompts.tsv"
comptime GREEDY = FIXTURE + "/greedy.tsv"

comptime MAX_TOKENS = 256

# 与 `test_q4_greedy.mojo` **同一组**下限，一把尺子才比得动：0.75 看通路是否在算，
# 0.85 挡"不该掉的掉了"（退回整批共用一个激活步长那版是 0.8184，就红在 0.85 上）。
comptime Q4_PATH_ALIVE_GATE = Float64(0.75)
comptime Q4_REGRESSION_FLOOR = Float64(0.85)


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


def greedy_rate() raises -> Float64:
    """教师强制贪心的一致率（整数计数，不受本机噪声影响）。"""
    var n = vocab_size()
    var prompts = load_prompts()
    var greedy = load_greedy()
    var total = 0
    var agreed = 0
    var index = 0
    for ids in prompts:
        var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, True)
        if not model.q4_enabled:
            raise AlofaError(
                1, "前向没有走量化通路，这轮测的是 fp32 对 fp32", "q4_enabled=False"
            )
        var expected = greedy[index].copy()
        var first = model.prefill[BACKEND_AVX2](ids)
        var token = argmax_of(first, n)
        total += 1
        if token == expected[0]:
            agreed += 1
        var seen = 1
        while seen < len(expected):
            var logits = model.step[BACKEND_AVX2](expected[seen - 1])
            token = argmax_of(logits, n)
            total += 1
            if token == expected[seen]:
                agreed += 1
            seen += 1
        model.keep_alive()
        index += 1
    return Float64(agreed) / Float64(total)


def test_the_mub_path_is_the_one_being_measured() raises:
    """开关必须是开的 —— 否则下面那条量到的是 f32 通路，绿得毫无意义。

    ⚠️ 这条只能挡住"开关被关掉/默认被翻回去"，挡不住"开关开着但挂在没人走的入口上"
    （2026-09-29 那个坑），后者由核级门 `test_q4_mub_rows.mojo` 的"两条通路不许逐位
    相同"盯着。
    """
    assert_true(
        q4_mub_enabled(),
        "ALOFA_Q4_MUB=0：这一轮量到的是 f32 展开，不是 vpmaddubsw",
    )


def test_whole_network_greedy_holds_both_floors() raises:
    """整网一致率：0.75（通路在工作）与 0.85（不该掉的掉了）两道都过。

    一次前向两分钟，所以两条下限写在同一个 test 里 —— 拆成两个 test 就要把 2 GB
    权重压两遍，而失败信息里已经写清是哪条下限红的。
    """
    var rate = greedy_rate()
    print("[q4mub] 一致率 = " + String(rate) + "（backend = AVX2, vpmaddubsw）")
    assert_true(
        rate >= Q4_PATH_ALIVE_GATE,
        "一致率低于 0.75：量化通路在算垃圾（rate=" + String(rate) + "）",
    )
    assert_true(
        rate >= Q4_REGRESSION_FLOOR,
        "一致率低于 0.85 回归下限：不该掉的掉了（rate=" + String(rate) + "）",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
