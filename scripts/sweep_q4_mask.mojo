# 一百七十九（实验）：q4_0 逐层敏感度扫描 —— 混合精度的数据底座
#
# 问题：整网 q4_0（除输出投影）的教师强制贪心一致率 0.8691，距自设 ≥0.90 质量门
# 差 3 个点。哪几层把一致率拉下来的？不量出来，"混合精度"就只能是猜。
#
# 方法：**leave-one-out**——除层 i 外全量化（`enable_q4(skip_layers=[i])`），
# 量教师强制贪心一致率。读法：
#   - 全量化基线 0.8691；
#   - 把层 i 放回 fp32 后一致率**升得最多**的层，就是对一致率伤害最大的层；
#   - 升幅≈0（0.8691 不动）的层，量化它不亏——混合精度不需要动它。
#
# 判据与 `test_q4_greedy.mojo` 完全同源（同 fixtures、同教师强制循环、同 argmax），
# 只是**不设下限**：扫描要看原始数字，包括低于回归下限的那些。
#
# 跑法（依赖 2 GB 本地导出 + 上面的 24 次模型构造，-O2）：
#
#     pixi run mojo run -O2 -I src scripts/sweep_q4_mask.mojo
#
# ⚠️ 只定义在 CPU 前向上（GPU 会话整体上传 q4_blocks，见 enable_q4 注释）。

from alofa.core.error import AlofaError, ERR_INVALID_ARGUMENT
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


def measure(skip_layer: Int, prompts: List[List[Int]], greedy: List[List[Int]], n: Int) raises -> Float64:
    """除 skip_layer 外全量化的教师强制贪心一致率。skip_layer = -1 → 全量化基线。"""
    var skip = List[Int]()
    if skip_layer >= 0:
        skip.append(skip_layer)
    var total = 0
    var agreed = 0
    for index in range(len(prompts)):
        var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, True, skip)
        if not model.q4_enabled:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "量化通路没开 —— 扫描测的是 fp32 对 fp32，数字无意义",
                "skip_layer=" + String(skip_layer),
            )
        var expected = greedy[index].copy()
        var first = model.prefill(prompts[index])
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
    return Float64(agreed) / Float64(total)


def main() raises:
    var prompts = load_prompts()
    var greedy = load_greedy()
    var n = vocab_size()

    var base = measure(-1, prompts, greedy, n)
    print("baseline(all-q4) = " + String(base))

    for layer in range(24):
        var rate = measure(layer, prompts, greedy, n)
        var gain = rate - base
        print(
            "leave-out "
            + String(layer)
            + " = "
            + String(rate)
            + "  (gain "
            + String(gain)
            + ")"
        )
