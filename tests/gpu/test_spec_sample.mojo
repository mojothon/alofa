"""账本一百七十五-补九门：采样验证——spec 路在 temp>0 下与 plain
批路**逐位同**（采样即判据：h_tokout 从 argmax 变样本，tokout[j]==
草稿才接受、首失配处的样本即 next——输出分布恒等于目标分布，无偏）。
判据：

① 采样 parity：generate_spec(temp=0.7, seed=42) 链与 generate_batch
   (f32 实例化，同温度同种子) 逐位一致——种子按生成步派生
   （步 s 用 seed+s），spec 验证行与 plain 行同位同种子同核
   （sample_topp_row_gpu），故逐位同。
② 确定性：spec 采样链两次运行（独立会话）逐位同。
③ 种子多样性：seed=42 vs 43 链不同（采样真实生效，未退化贪心）。

提示词沿用 4-gram 重复模式（草稿源充足，验证行带真实草稿链）。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime MAXNEW = 48
comptime EOS = -1
comptime SPEC_K = 4
comptime SPEC_NGRAM = 4
comptime TEMP = 0.7
comptime SEED = 42


def _pattern_prompt() -> List[Int]:
    """4-gram 模式 ×4（16 token）。"""
    var prompt = List[Int]()
    for _r in range(4):
        prompt.append(264)
        prompt.append(23335)
        prompt.append(12163)
        prompt.append(1052)
    return prompt^


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)

    # ① 参照：plain 批路采样链（f32 实例化——f64 默认触发 KGEN 崩，
    #    门历史只用 f32，见 test_generate_batch / probe_190 注记）。
    var b = GpuSession(model, 512)
    var flat = List[Int]()
    for t in _pattern_prompt():
        flat.append(t)
    var lens = List[Int]()
    lens.append(16)
    var refc = b.generate_batch[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32
    ](
        flat^, lens^, MAXNEW, EOS, List[Int](), List[Int](),
        TEMP, Int32(SEED), 1.0
    )

    # ② spec 采样闭环（同温度同种子）
    var a = GpuSession(model, 512)
    var res = a.generate_spec[INNER, INNER_FF](
        _pattern_prompt(), MAXNEW, EOS, SPEC_K, SPEC_NGRAM,
        TEMP, Int32(SEED)
    )

    # ③ 确定性：独立会话重跑
    var c = GpuSession(model, 512)
    var res2 = c.generate_spec[INNER, INNER_FF](
        _pattern_prompt(), MAXNEW, EOS, SPEC_K, SPEC_NGRAM,
        TEMP, Int32(SEED)
    )

    # ④ 种子多样性
    var d = GpuSession(model, 512)
    var res3 = d.generate_spec[INNER, INNER_FF](
        _pattern_prompt(), MAXNEW, EOS, SPEC_K, SPEC_NGRAM,
        TEMP, Int32(SEED + 1)
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
    var diverge = True
    if n == len(res3.tokens):
        diverge = False
        for i in range(n):
            if res.tokens[i] != res3.tokens[i]:
                diverge = True
                break

    print("chain: spec-sampled", n, "vs plain-batch budget", MAXNEW,
          " mismatch =", mismatch, " first_bad =", first_bad)
    print("determinism (spec rerun):", det_bad)
    print("seed diversity (42 vs 43):", "differs" if diverge else "IDENTICAL")
    print("telemetry: ticks =", res.n_ticks, " calls =",
          res.tel.n_calls, " hits =", res.tel.n_hits,
          " drafted =", res.tel.n_drafted,
          " accepted =", res.tel.n_accepted)
    print("  accept_rate =", res.tel.accept_rate(), " tokens/tick =",
          res.tel.tokens_per_tick())

    var ok = (
        mismatch == 0
        and det_bad == 0
        and n == MAXNEW
        and diverge
        and res.tel.n_calls >= 1
    )
    if ok:
        print("SPEC-SAMPLE PASS")
    else:
        print("SPEC-SAMPLE FAIL")
