"""`vpmaddubsw` 可选通路在**整网**上还认不认得自己。

为什么要有这个文件，而不是直接跑 `pixi run test-q4`
----------------------------------------------------
`test-q4`（`tests/unit/test_q4_greedy.mojo`）量到的 0.8691 是**标量后端**的：
`prefill` / `step` 的 `backend` 参数默认就是 `BACKEND_SCALAR`。而 `vpmaddubsw` 是
AVX2 指令，开关 `ALOFA_Q4_MUB` 只挂在向量分支上 —— 所以直接跑 `test-q4`，开关开
不开结果都一样（实测两遍都是 445/512，一模一样，就是这个原因）。

于是这里把同一份判据在 **`BACKEND_AVX2`** 上再跑一遍：同样的 fixture、同样的教师
强制贪心、同样的两条下限（0.75 / 0.85）。它**不是门**（在 `scripts/` 下，不进
`pixi run test`，不登记账本），只是一个回答"那 0.75% 顶不顶得住"的测量。

判什么
------
`vpmaddubsw` 这条通路核级快 1.87×，代价是**多量化一层激活**（相对 L2 约 0.75%，
见 `bench_q4_mub.mojo` 与 `avx2.mojo` 里那节）。核级的相对 L2 不等于端到端的
一致率：argmax 是离散的，0.75% 的扰动可能一步都不翻转，也可能在最接近的两条
logit 上翻掉一批。这里量的是后者。

跑法（`-O2`，依赖 2 GB 本地导出）
--------------------------------
    pixi run mojo build -O2 -I src scripts/verify_q4_mub_greedy.mojo -o target/verify_q4_mub
    ./target/verify_q4_mub                  # AVX2 + 默认通路（对照）
    ALOFA_Q4_MUB=1 ./target/verify_q4_mub   # AVX2 + vpmaddubsw 通路

⚠️ 两条线必须**同轮**跑：本机长期过载，跨轮的绝对数字不可比；但这里比的是一致率
（整数计数），不是时间，受噪声影响小得多。
"""

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

# 与 `test_q4_greedy.mojo` **同一组**下限：0.75 是"通路在工作"，0.85 是"不该掉的
# 掉了"。这里不是门，但用同一把尺子才比得动。
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


def main() raises:
    var n = vocab_size()
    var prompts = load_prompts()
    var greedy = load_greedy()
    print("")
    print("== q4_0 整网贪心一致率（backend = AVX2）==")
    print(
        "   通路："
        + ("vpmaddubsw（ALOFA_Q4_MUB=1）" if q4_mub_enabled() else "默认 f32 展开")
    )

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

    var rate = Float64(agreed) / Float64(total)
    print(
        "[q4mub] 一致 "
        + String(agreed)
        + "/"
        + String(total)
        + " = "
        + String(rate)
        + "   0.75 通路门 "
        + ("过" if rate >= Q4_PATH_ALIVE_GATE else "未过")
        + "；0.85 回归下限 "
        + ("过" if rate >= Q4_REGRESSION_FLOOR else "未过")
    )
    print("[q4mub] 这轮不是门，只是把两条下限用同一把尺子量一遍")
    print("")
