"""Qwen3-8B 教师强制 logits 对拍（alofa 侧）——沿 HF fp32 的贪心轨迹逐位比。

与 /tmp/qwen3_hf_dump.py 配对：同一串输入（ChatML prompt + HF 生成的 16 个
token），两边都**教师强制**（不吃自己上一步的 argmax，排除闭环混沌），
每步打印 top-3（id, logit）。数学对得上 ⇒ 逐位差应在 1e-2 以下；
对不上 ⇒ QK-norm 接线有真偏差。

远端运行：
    ALOFA_WEIGHTS=/path/to/Qwen3-8B a100.sh run 3 scripts/qwen3_logits_dump.mojo
"""

from std.os import getenv

from alofa.core.tensor import F32Ptr
from alofa.model.arch.qwen import QwenForward

comptime MAX_TOKENS = 64


def weights_dir() -> String:
    """`ALOFA_WEIGHTS`（分片目录；config.json 在其中），未设返回空串。"""
    var raw = getenv("ALOFA_WEIGHTS", "")
    if raw.byte_length() == 0:
        print("ALOFA_WEIGHTS not set")
        return String("")
    return raw^


def top3(values: F32Ptr, n: Int) raises:
    """单遍扫出 top-3（id, logit）并打印。"""
    var i0 = -1
    var v0 = Float32(-3.4028234663852886e38)
    var i1 = -1
    var v1 = Float32(-3.4028234663852886e38)
    var i2 = -1
    var v2 = Float32(-3.4028234663852886e38)
    for i in range(n):
        var v = values[unsafe_offset=i]
        if v > v0:
            v2 = v1
            i2 = i1
            v1 = v0
            i1 = i0
            v0 = v
            i0 = i
        elif v > v1:
            v2 = v1
            i2 = i1
            v1 = v
            i1 = i
        elif v > v2:
            v2 = v
            i2 = i
    print("  top:", i0, Float64(v0), "|", i1, Float64(v1), "|", i2, Float64(v2))


def main() raises:
    var prompt = List[Int]()
    prompt.append(151644)
    prompt.append(872)
    prompt.append(198)
    prompt.append(16)
    prompt.append(220)
    prompt.append(17)
    prompt.append(220)
    prompt.append(18)
    prompt.append(220)
    prompt.append(19)
    prompt.append(220)
    prompt.append(20)
    prompt.append(220)
    prompt.append(21)
    prompt.append(151645)
    prompt.append(198)
    prompt.append(151644)
    prompt.append(77091)
    prompt.append(198)
    # HF fp32 轨迹（教师强制喂入，不吃自己的 argmax）
    var gen = List[Int]()
    gen.append(151667)
    gen.append(198)
    gen.append(32313)
    gen.append(11)
    gen.append(279)
    gen.append(1196)
    gen.append(3897)
    gen.append(264)
    gen.append(8500)
    gen.append(315)
    gen.append(5109)
    gen.append(25)
    gen.append(220)
    gen.append(16)
    gen.append(220)
    gen.append(17)

    # 权重目录走 `ALOFA_WEIGHTS`（分片目录；config.json 在其中）。
    var weights = weights_dir()
    var model = QwenForward(weights, weights + "/config.json", MAX_TOKENS, False)
    print("prompt rows:", len(prompt), "gen:", len(gen))
    var logits = model.prefill(prompt)
    print("step 0:")
    top3(logits, 151936)
    for s in range(1, len(gen)):
        logits = model.step(gen[s - 1])
        print("step", s, "(fed", gen[s - 1], "):")
        top3(logits, 151936)
    model.keep_alive()
    print("ALOFA-DUMP-DONE")
