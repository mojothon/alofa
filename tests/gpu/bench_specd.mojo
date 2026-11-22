"""账本一百七十四-d A/B bench：solo spec（prompt-lookup 草稿验证）
vs solo t=1 合批——同一 GpuService 拨 spec_on 字段两相对比。

两个 prompt 域：
* story：叙事续写（对话域代表——草稿命中率低，spec 退化为 t=1 + 
  验证开销）；
* pattern：计数序列（重复模式域——prompt-lookup 主战场，高接受率）。

判据面：①链 parity（spec vs base 文本逐位同）；②tps（complete()
墙钟，每相两轮取后轮——首轮付惰性装载）；③接受率遥测。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime STORY = "Once upon a time, in a land far away, there was a"
comptime PATTERN = (
    "Count with me. one, two, three, four, one, two, three, four,"
    " one, two,"
)
comptime GEN = 128


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var tok = load_tokenizer_json(TOKJSON)
    var svc = GpuService(
        model, 1, 512, tok^, 151936, 7, WEIGHTS, CONFIG, TOKJSON
    )

    for mode in range(2):
        svc.spec_on = (mode == 1)
        for p in range(2):
            var tag = String("")
            var prompt = String("")
            if p == 0:
                prompt = STORY
                tag = "story"
            else:
                prompt = PATTERN
                tag = "pattern"
            if mode == 1:
                tag += " spec"
            else:
                tag += " base"
            var pre_calls = svc.spec_tel.n_calls
            var pre_acc = svc.spec_tel.n_accepted
            var txt0 = String("")
            var parity = True
            var tps = Float64(0)
            var n_out = 0
            for r in range(2):
                var t0 = monotonic_ns()
                var c = svc.complete(prompt, GEN, 0.0, 1.0)
                var t1 = monotonic_ns()
                var ms = Float64(t1 - t0) / 1e6
                tps = Float64(c.completion_tokens) / (ms / 1000.0)
                n_out = c.completion_tokens
                if r == 0:
                    txt0 = c.text
                else:
                    parity = c.text == txt0
            var calls = svc.spec_tel.n_calls - pre_calls
            var acc = svc.spec_tel.n_accepted - pre_acc
            print(
                tag, ": tps =", tps, " replay_parity =", parity,
                " n_out =", n_out, " calls =", calls,
                " accepted =", acc, " acc_rate =",
                Float64(acc) / Float64(max(calls * 4, 1)),
            )
    print("done")
