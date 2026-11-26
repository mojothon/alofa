"""账本一百八十五：GPU QK-norm 门——CPU 参照链（181 已对 HF fp32 逐
token 一致）与 GPU 链在注入 qk_norm 的 fixture 上逐 token 一致。

fixture：0.5B + 全 1 的 q_norm/k_norm 注入（48 张 bf16，[n_heads·hd] /
[n_kv·hd] 平铺）——attn_bias（0.5B 原生有）与 qk_norm 两个开关同时
开，正好覆盖开关共存的组合。GPU 侧实现 = CPU 181 的换视图论证照搬：
rmsnorm_gpu 的 rows/cols 参数化（155）直接吃 [t, q_dim] ≡
[t·n_heads, head_dim] 的同构，decode 与 prefill 两路的 rope 前各两发。
"""
from std.testing import assert_true

from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b-qknorm"
comptime WEIGHTS = FIXTURE + "/model.safetensors"
comptime CONFIG = FIXTURE + "/config.json"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime PROMPT = "The capital of France is"
comptime MAXNEW = 12


def main() raises:
    var tok = load_tokenizer_json(TOKJSON)
    var ids = tok.encode(PROMPT)

    # CPU 参照链（QK-norm 换视图路，181 对 HF 验证过）。
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    assert_true(model.qk_norm, "fixture must switch qk_norm on")
    assert_true(model.attn_bias, "fixture must keep attn_bias on")
    var cpu = List[Int]()
    model.reset()
    var token = model.argmax(model.prefill(ids))
    var n = 0
    while n < MAXNEW:
        cpu.append(token)
        n += 1
        if n < MAXNEW:
            token = model.argmax(model.step(token))

    # GPU 链。
    var svc = GpuService(
        model, 4, 512, load_tokenizer_json(TOKJSON)^, 151936, 7,
        WEIGHTS, CONFIG, TOKJSON,
    )
    assert_true(svc.session.qk_norm, "session must switch qk_norm on")
    var gpu = List[Int]()
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 9003)
    var slot = svc._slot_of(9003)
    var done = False
    while not done:
        var fr = svc.stream_next(9003)
        if not fr.waiting:
            gpu.append(svc.streams[slot].last)
            if fr.done:
                done = True
    svc.stream_end(9003)

    var ok = len(cpu) == MAXNEW and len(gpu) == MAXNEW
    if ok:
        for i in range(MAXNEW):
            if cpu[i] != gpu[i]:
                ok = False
                print("diverge at", i, ":", cpu[i], "vs", gpu[i])
                break
    print("cpu-chain:", len(cpu), "tokens")
    print("gpu-chain:", len(gpu), "tokens")
    if ok:
        print("*** GPU QK-NORM GATE PASS ***")
    else:
        print("*** GPU QK-NORM GATE FAIL ***")
