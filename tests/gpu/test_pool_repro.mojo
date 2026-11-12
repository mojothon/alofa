"""账本一百三十二诊断刀：池化后「同一 session 第 K 条流换世界」最小复现器。

现象（从主门二分而来）：slot0 在 cjk 流之后，PROMPT 独跑的贪心链永久
换成另一个世界（572-world），KV reset（kv_pos=0 + h_posarr 清零）不救。
本复现器逐流打印 head，钉住「第几条流开始换世界」与「是否与 prompt
内容有关」。
"""

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime PROMPT = "Once upon a time, in a land far away,"
comptime CJK = "给我讲一个简短的睡前故事。"
comptime MAXNEW = 24


def _run(mut svc: GpuService, rid: Int, prompt: String, n: Int) raises -> List[Int]:
    svc.stream_begin(prompt, n, 0.0, 1.0, rid)
    var c = List[Int]()
    var done = False
    while not done:
        var fr = svc.stream_next(rid)
        c.append(svc.streams[0].last)
        if fr.done:
            done = True
    svc.stream_end(rid)
    return c^


def _head(c: List[Int]) -> String:
    var s = String("")
    for i in range(4):
        if i < len(c):
            s = s + String(c[i]) + " "
    return s


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var tokenizer = load_tokenizer_json(TOKJSON)
    var svc = GpuService(
        model, 2, 512, tokenizer^, 151936, 42,
        WEIGHTS, CONFIG, TOKJSON,
    )

    # F0：complete 先行（generate_batch 冷启动一遍）——主门 chain1 英文
    # 世界的唯一前置差异。若 F1 因此变英文 ⇒ 纯流路冷启动缺状态。
    var comp0 = svc.complete(PROMPT, 8, 0.0, 1.0)
    print("F0 (complete)      :", comp0.completion_tokens, "tokens |",
          comp0.text)
    var c1 = _run(svc, 1, PROMPT, MAXNEW)
    print("F1 (PROMPT)        :", _head(c1))
    var c2 = _run(svc, 2, PROMPT, MAXNEW)
    print("F2 (PROMPT) == F1  :", _same(c2, c1))
    var c3 = _run(svc, 3, PROMPT, MAXNEW)
    print("F3 (PROMPT) == F1  :", _same(c3, c1))
    var c4 = _run(svc, 4, CJK, 16)
    print("F4 (CJK)           :", _head(c4))
    # F4b：**英文新 prompt**——判别「换世界」是否 CJK 专属（若英文新
    # prompt 也错 ⇒ 病在「流路 prefill 对新 prompt 内容」，与 CJK 无关）。
    var c4b = _run(svc, 14, "The quick brown fox jumps over", 16)
    print("F4b (new EN prompt):", _head(c4b))
    var c4c = _run(svc, 15, "She said that he would", 16)
    print("F4c (2nd EN prompt):", _head(c4c))
    var c5 = _run(svc, 5, PROMPT, MAXNEW)
    print("F5 (PROMPT) == F1  :", _same(c5, c1), " head:", _head(c5))
    var c6 = _run(svc, 6, PROMPT, MAXNEW)
    print("F6 (PROMPT) == F1  :", _same(c6, c1), " == F5:", _same(c6, c5))
    var c7 = _run(svc, 7, CJK, 16)
    print("F7 (CJK) == F4     :", _same(c7, c4))
    var c8 = _run(svc, 8, PROMPT, MAXNEW)
    print("F8 (PROMPT) == F1  :", _same(c8, c1), " == F5:", _same(c8, c5))
