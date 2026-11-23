"""账本一百七十五：spec 草稿参数域扫描——(k, ngram) 网格在两个生产
代表域上的 e2e tps（174-d 的 K=4/NGRAM=4 是拍脑袋定值；接受率 95%+
的域可能吃得到更长链）。

配置面：svc.spec_k / svc.spec_ngram 运行时拨字段（ALOFA_SPEC_K/NGRAM
的同款通道）。每相预热一轮 + 计测一轮（首轮付惰性装载）；链 parity
照查（重放逐位同——参数只影响 tick 数，不改输出链）。

判据面：tps（complete() 墙钟）+ acc_rate + replay parity。基线
(4,4) 必须在场；parity 全 True 是任何参数变更的硬门。
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
    svc.spec_on = True
    # (k, ngram) 网格：k 扫链长（2/4/6/8），ngram 扫匹配窗（3/4/6）。
    var ks = List[Int]()
    var ngs = List[Int]()
    ks.append(2)
    ngs.append(4)
    ks.append(4)
    ngs.append(4)
    ks.append(6)
    ngs.append(4)
    ks.append(8)
    ngs.append(4)
    ks.append(8)
    ngs.append(3)
    ks.append(8)
    ngs.append(6)
    for cfg in range(len(ks)):
        svc.spec_k = ks[cfg]
        svc.spec_ngram = ngs[cfg]
        for p in range(2):
            var prompt = STORY if p == 0 else PATTERN
            var tag = "k" + String(ks[cfg]) + "n" + String(ngs[cfg])
            tag += " story" if p == 0 else " pattern"
            var txt0 = String("")
            var parity = True
            var tps = Float64(0)
            var pre_calls = svc.spec_tel.n_calls
            var pre_acc = svc.spec_tel.n_accepted
            for r in range(2):
                var t0 = monotonic_ns()
                var c = svc.complete(prompt, GEN, 0.0, 1.0)
                var t1 = monotonic_ns()
                var ms = Float64(t1 - t0) / 1e6
                tps = Float64(c.completion_tokens) / (ms / 1000.0)
                if r == 0:
                    txt0 = c.text
                else:
                    parity = c.text == txt0
            var calls = svc.spec_tel.n_calls - pre_calls
            var acc = svc.spec_tel.n_accepted - pre_acc
            print(
                tag, ": tps =", tps, " parity =", parity,
                " calls =", calls, " accepted =", acc,
                " acc_rate =",
                Float64(acc) / Float64(max(calls * ks[cfg], 1)),
            )
    print("done")
