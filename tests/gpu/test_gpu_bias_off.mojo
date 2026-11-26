"""账本一百八十四：GPU bias-optional 门——zero-bias checkpoint 与
bias-on 路全链逐 token 一致。

原理：把 0.5B 权重里的 72 个 `*.bias` 张量删掉另存（zerobias fixture）。
bias-on 路对它逐位等价的数学事实是「加 0.0 精确」——bias-off 家族
（GPU session 的 attn_bias=False：零占位 + List 填零引用 + 融合核
HAS_BIAS 分流）必须产出与 bias-on 路相同的贪心链。这条门同时钉住：
① session 的 attn_bias 判据（层 0 张量清单）；② 零占位填充与上传
分支；③ f16_mma_gemm2_pk / mergb 系 / fat 核的 HAS_BIAS 模板位。
"""
from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS_ON = "tests/fixtures/qwen2.5-0.5b-zerobias"
comptime CONFIG_ON = "tests/fixtures/qwen2.5-0.5b-zerobias/config.json"
comptime WEIGHTS_OFF = "tests/fixtures/qwen2.5-0.5b-nobias"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime PROMPT = "The capital of France is"
comptime MAXNEW = 16


def main() raises:
    var tok = load_tokenizer_json(TOKJSON)

    # bias-on（现役 0.5B TSV fixture）。
    var m_on = QwenForward(
        WEIGHTS_ON + "/model.safetensors", CONFIG_ON, 512, True
    )
    var svc_on = GpuService(
        m_on,
        4,
        512,
        tok^,
        151936,
        7,
        WEIGHTS_ON + "/model.safetensors",
        CONFIG_ON,
        TOKJSON,
    )
    if not svc_on.session.attn_bias:
        print("FAIL: bias-on fixture judged attn_bias=False")
        return

    # bias-off（zerobias safetensors：无 *.bias 张量）。
    # 单文件形态（180：QwenForward 的 weights 参数是权重文件自己）。
    var m_off = QwenForward(
        WEIGHTS_OFF + "/model.safetensors",
        WEIGHTS_OFF + "/config.json",
        512,
        True,
    )
    var tok2 = load_tokenizer_json(TOKJSON)
    var svc_off = GpuService(
        m_off,
        4,
        512,
        tok2^,
        151936,
        7,
        WEIGHTS_OFF + "/model.safetensors",
        WEIGHTS_OFF + "/config.json",
        TOKJSON,
    )
    if svc_off.session.attn_bias:
        print("FAIL: zerobias fixture judged attn_bias=True")
        return

    # bias-on 链。
    var c_on = List[Int]()
    svc_on.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 9001)
    var slot_on = svc_on._slot_of(9001)
    var done_on = False
    while not done_on:
        var fr = svc_on.stream_next(9001)
        if not fr.waiting:
            c_on.append(svc_on.streams[slot_on].last)
            if fr.done:
                done_on = True
    svc_on.stream_end(9001)

    # bias-off 链。
    var c_off = List[Int]()
    svc_off.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 9002)
    var slot_off = svc_off._slot_of(9002)
    var done_off = False
    while not done_off:
        var fr = svc_off.stream_next(9002)
        if not fr.waiting:
            c_off.append(svc_off.streams[slot_off].last)
            if fr.done:
                done_off = True
    svc_off.stream_end(9002)

    var ok = len(c_on) == MAXNEW and len(c_off) == MAXNEW
    if ok:
        for i in range(MAXNEW):
            if c_on[i] != c_off[i]:
                ok = False
                print("diverge at", i, ":", c_on[i], "vs", c_off[i])
                break
    print("on-chain :", len(c_on), "tokens")
    print("off-chain:", len(c_off), "tokens")
    if ok:
        print("*** GPU BIAS-OFF GATE PASS ***")
    else:
        print("*** GPU BIAS-OFF GATE FAIL ***")
