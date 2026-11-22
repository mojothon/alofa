"""账本一百七十四-c 第四刀门：`generate_spec`——prompt-lookup 草稿
源驱动 `spec_step` 的贪心生成闭环。判据：

① 链 parity：spec 链与「t=1 逐 token 贪心参照」逐位一致——spec 的
   接受语义是「草稿 d_i 只有在模型 argmax 恰好产出 d_i 时才被接受」，
   草稿只省 tick，不改输出。参照与 spec 同为 f64 累加面（默认
   comptime），t=1 非批 vs 行映射批的核变体在 f64 下逐位同
   （一百七十四-b 的恒等映射结论）。
   ⚠️ 参照不用 generate_batch：f64 默认实例化会触发 KGEN 编译器
   崩（门历史只用 f32 实例化过它，见同日 probe_190 注记）。
② 跨会话确定性：spec 两次运行（独立会话）链逐位同；
③ 遥测健全：n_calls ≥ 1、tokens_per_tick 打印——>1 即 spec 净
   收益（每 tick 多产 token）。

提示词为重复 token 模式（4-gram × 4）——prompt-lookup 的目标域
（复述/代码/格式化文本），草稿命中率应显著非零。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime PLEN = 16
comptime MAXNEW = 48
comptime EOS = -1
comptime SPEC_K = 4
comptime SPEC_NGRAM = 4


def _pattern_prompt() -> List[Int]:
    """4-gram 模式 ×4（16 token）的重复模式提示词。"""
    var prompt = List[Int]()
    for _r in range(4):
        prompt.append(264)
        prompt.append(23335)
        prompt.append(12163)
        prompt.append(1052)
    return prompt^


def _greedy_ref(mut sess: GpuSession, prompt: List[Int]) raises -> List[Int]:
    """t=1 逐 token 贪心参照链——经 spec_step（trust_all，ln=1）：
    acc=0、next = 行 0 的 argmax。与被测路径同实例化族（模块内只
    留一个 prefill_step 实例化——多实例化触发 KGEN 编译器崩，见
    probe_190 注记），f64 累加面下 t=1 与批行逐位同（174-b）。"""
    sess.h_posarr[0] = Int32(0)
    var out = List[Int]()
    var prev = 0
    var steps = PLEN - 1 + MAXNEW
    for j in range(steps):
        var toks = List[Int]()
        if j < PLEN:
            toks.append(prompt[j])
        else:
            toks.append(prev)
        var ln = List[Int]()
        ln.append(1)
        var sq = List[Int]()
        sq.append(0)
        var r = sess.spec_step[INNER, INNER_FF](toks^, ln, sq, True)
        prev = r[1]
        if j >= PLEN - 1:
            out.append(prev)
    return out^


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)

    # ① 参照：t=1 贪心链
    var b = GpuSession(model, 512)
    var t0 = monotonic_ns()
    var refc = _greedy_ref(b, _pattern_prompt())
    var t1 = monotonic_ns()
    var ref_ms = Float64(t1 - t0) / 1e6

    # ② spec 闭环
    var a = GpuSession(model, 512)
    t0 = monotonic_ns()
    var res = a.generate_spec[INNER, INNER_FF](
        _pattern_prompt(), MAXNEW, EOS, SPEC_K, SPEC_NGRAM
    )
    t1 = monotonic_ns()
    var spec_ms = Float64(t1 - t0) / 1e6

    # ③ 跨会话确定性
    var c = GpuSession(model, 512)
    var res2 = c.generate_spec[INNER, INNER_FF](
        _pattern_prompt(), MAXNEW, EOS, SPEC_K, SPEC_NGRAM
    )

    var n = len(res.tokens)
    var mismatch = 0
    var first_bad = -1
    for i in range(n):
        if res.tokens[i] != refc[i]:
            mismatch += 1
            if first_bad < 0:
                first_bad = i
    var det_bad = 0
    if n != len(res2.tokens):
        det_bad = -1
    else:
        for i in range(n):
            if res.tokens[i] != res2.tokens[i]:
                det_bad += 1

    print("chain: spec", n, "tokens vs ref budget", MAXNEW,
          " mismatch =", mismatch, " first_bad =", first_bad)
    print("determinism (spec rerun):", det_bad)
    print("telemetry: ticks =", res.n_ticks, " calls =",
          res.tel.n_calls, " hits =", res.tel.n_hits,
          " drafted =", res.tel.n_drafted,
          " accepted =", res.tel.n_accepted)
    print("  hit_rate =", res.tel.hit_rate(), " accept_rate =",
          res.tel.accept_rate(), " tokens/tick =",
          res.tel.tokens_per_tick())
    print("wall: spec =", spec_ms, "ms  t1-ref =", ref_ms, "ms")

    var ok = (
        mismatch == 0
        and det_bad == 0
        and n == MAXNEW
        and len(refc) == MAXNEW
        and res.tel.n_calls >= 1
    )
    if ok:
        print("SPEC-GEN PASS")
    else:
        print("SPEC-GEN FAIL")
