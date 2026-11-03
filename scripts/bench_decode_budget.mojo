"""decode 一个 token 到底花在哪 —— 真实模型、真实权重、单进程。

为什么要有这个文件
------------------
账本 11-03 第六条量出"decode 的注意力只占 0.69–2.01%"，推翻了账本里
"注意力 26.4%"那条 —— 那张表（GEMM 53.3% / 注意力 26.4% / 归一化 20.3%）
是**非 decode 形状**的，不能套到单 token 上。

于是这里用 profiler 的 8 个分段，在**真实 decode 形状**上重新出预算：

    K_NORM        RMSNorm
    K_ROPE_ATTN   RoPE + KV 追加 + 注意力（三者同一个桶，账本 26.4% 就是这个桶）
    K_ACT         SwiGLU
    K_RESID       残差相加
    K_HEAD        输出头（q8）+ argmax
    K_VIEW        权重视图解析（`params.view`）
    K_PREP        解码前的准备
    K_GEMM        7 个投影 + 输出头的矩阵乘

同时跑 `shards=1` 与 `shards=8` 两档：既给出"哪段最贵"，也给出"shards 到底
省下了哪一段、哪一段完全没动" —— 后者是判断并行扩展还剩多少空间的关键。

⚠️ 要开 `comptime ALOFA_PROFILE = True`（`src/alofa/model/profiler.mojo`）。
插桩会拖慢绝对值，所以**只读各段占比，不读 tok/s**；跑完记得把常量改回 False。
⚠️ 这是探路用的 micro-bench，不是门，不进 `pixi run test`。

跑法
----
    pixi run mojo build -O2 -I src scripts/bench_decode_budget.mojo -o target/decode_budget
    ./target/decode_budget
"""

from alofa.core.ffi import monotonic_ns
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward
from alofa.tokenizer import Tokenizer, load_tokenizer

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime PROMPT = "The capital of France is"
comptime MAX_TOK = 64
comptime NEW = 16


def measure(mut model: QwenForward, imm tok: Tokenizer, shards: Int) raises -> Int:
    """跑一趟 decode 并打印分段预算，返回**每 token 的墙钟纳秒**。"""
    model.set_shards(shards)
    model.reset()
    var logits = model.prefill[BACKEND_AVX2](tok.encode(PROMPT))
    var next = model.argmax(logits)

    # prefill 不计入：profile 在 prefill 之后才开，且 reset 清零。
    model.set_profile(True)
    var t0 = monotonic_ns()
    for _ in range(NEW):
        var lg = model.step[BACKEND_AVX2](next)
        next = model.argmax(lg)
    var wall = (monotonic_ns() - t0) // NEW
    print("   shards=" + String(shards) + "   墙钟 " + String(wall // 1000) + " µs/token")
    # ⚠️ 先 dump 再关：`set_profile(False)` 会 `reset()`，把样本一起清掉。
    model.dump_profile()
    model.set_profile(False)
    return wall


def main() raises:
    print("")
    print("== decode 分段预算（真实权重 q4_0 + q8_head，单进程，avx2） ==")
    var model = QwenForward(FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK, True)
    model.enable_q8_output()
    var tok = load_tokenizer(FIXTURE)

    # 预热一轮：把 mmap 的页读进来，否则第一趟会把缺页算进 K_GEMM。
    model.set_shards(1)
    model.reset()
    var lg = model.prefill[BACKEND_AVX2](tok.encode(PROMPT))
    var nx = model.argmax(lg)
    for _ in range(2):
        nx = model.argmax(model.step[BACKEND_AVX2](nx))

    _ = measure(model, tok, 1)
    _ = measure(model, tok, 8)

    print("")
    print("   读法：")
    print("   ① 占比最大的那一段就是下一个杠杆；注意力桶（K_ROPE_ATTN）若很小，")
    print("      说明并行扩展已经没有空间，别再往那投。")
    print("   ② 对比 shards=1 与 8 的同段耗时：没变的那段就是没被并行的段。")
    print("   ③ 插桩会拖慢绝对值 ⇒ 只看占比，不看 tok/s。")
