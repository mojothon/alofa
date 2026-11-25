# 一百七十九（实验二）：q4_0 混合精度的渐进曲线 —— "留几层 fp32 才到 0.90"
#
# leave-one-out 扫描（sweep_q4_mask.mojo，账本一百七十九）的结论：单层放回 fp32
# 最多 +1.37%（层 21/23），大部分层在 ±0.6% 内 —— 误差是**弥散**的，不集中在某
# 几层。但 LO 增益未必线性可加（量化误差沿深度复合）—— 渐进实验直接量"复合后"
# 的曲线：按单层增益降序，把 top-k 层留在 fp32（其余全量化），量教师强制贪心
# 一致率。
#
# 读法：曲线在哪越过 0.90，对应几层 fp32 = 多少内存/流量代价 —— 越不过去，
# 层级混合精度这个杠杆就被数据否决，账本如实记。
#
# 层序来源：sweep_q4_mask.mojo 的 LO gain 降序（2026-11-25 本机实测）。
#
# 跑法：
#
#     pixi run mojo run -O2 -I src scripts/sweep_q4_topk.mojo

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

# LO gain 降序（sweep_q4_mask.mojo 2026-11-25 实测，账本一百七十九）：
# 21/23 +1.37%、4 +0.78%、0/12/16 +0.59%、5 +0.39%、8/10/13/17/19 +0.20%
def top_order() -> List[Int]:
    # LO gain 降序（sweep_q4_mask.mojo 2026-11-25 实测，账本一百七十九）
    var out = List[Int]()
    out.append(21)
    out.append(23)
    out.append(4)
    out.append(0)
    out.append(12)
    out.append(16)
    out.append(5)
    out.append(8)
    out.append(10)
    out.append(13)
    out.append(17)
    out.append(19)
    return out^


def ids_from_csv(text: String) raises -> List[Int]:
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


def load_prompts() raises -> List[List[Int]]:
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


def load_greedy() raises -> List[List[Int]]:
    var text = read_text(GREEDY)
    var out = List[List[Int]]()
    var lines = text.split("\n")
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        out.append(ids_from_csv(line))
    return out^


def vocab_size() raises -> Int:
    return parse_int(config_value(CONFIG, "vocab"))


def argmax_of(values: F32Ptr, n: Int) -> Int:
    var best = 0
    var best_value = Float32(-3.4028234663852886e38)
    for i in range(n):
        if values[unsafe_offset=i] > best_value:
            best_value = values[unsafe_offset=i]
            best = i
    return best


def measure(skip: List[Int], prompts: List[List[Int]], greedy: List[List[Int]], n: Int) raises -> Float64:
    var total = 0
    var agreed = 0
    for index in range(len(prompts)):
        var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS, True, skip)
        assert model.q4_enabled
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

    var order = top_order()
    var ks = List[Int]()
    ks.append(1)
    ks.append(2)
    ks.append(3)
    ks.append(4)
    ks.append(6)
    ks.append(8)
    ks.append(12)
    for ki in range(len(ks)):
        var k = ks[ki]
        var mask = List[Int]()
        for i in range(k):
            mask.append(order[i])
        var rate = measure(mask^, prompts, greedy, n)
        print("top-" + String(k) + " fp32 layers = " + String(rate))
