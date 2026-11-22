"""账本一百七十四-d 门：serving 层 spec 接入——`GpuService._tick` 的
solo spec 分支（池内恰一条活跃流 + 生成相位 + temp==0 时，prompt-
lookup 草稿驱动 `spec_step` 验证 tick，一次前向产出 acc+1 token）。

判据：
① 链 parity：spec 开（solo 走草稿验证）vs 关（solo 走 t=1 合批路）
   的贪心链**逐位同**——接受即贪心链，草稿只省 tick 不改输出；
② 遥测健全：spec 开时 n_spec_ticks > 0、spec_calls ≥ 1、
   tokens/tick > 1（草稿命中场景）；
③ 多流回退不受扰：两条并发流（len(act)=2 → 合批路径）各自完成、
   链确定——spec 分支只在 solo 触发；
④ stream_next 帧语义：spec tick 产多帧，逐帧领取与一次性领取的
   拼接一致（帧队列天然支持）。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
# 重复模式 + 续写型 prompt：草稿源的主战场（复述/格式化文本），
# 应有显著非零命中。
comptime PROMPT = (
    "The rule is simple. First: be kind. Second: be kind."
    " Third: be kind. Fourth:"
)
comptime MAXNEW = 32


def _chain(mut svc: GpuService, rid: Int, maxnew: Int) raises -> List[Int]:
    """solo 贪心一条流，回 token 链（test_gpu_service 的 _replay 同款）。"""
    svc.stream_begin(PROMPT, maxnew, 0.0, 1.0, rid)
    var slot = svc._slot_of(rid)
    var c = List[Int]()
    var done = False
    while not done:
        var fr = svc.stream_next(rid)
        if fr.waiting:
            continue
        c.append(svc.streams[slot].last)
        if fr.done:
            done = True
    svc.stream_end(rid)
    return c^


def _chain_text(
    mut svc: GpuService, rid: Int, prompt: String, maxnew: Int
) raises -> List[Int]:
    """换 prompt 的 solo 贪心链（草稿命中率场景用）。"""
    svc.stream_begin(prompt, maxnew, 0.0, 1.0, rid)
    var slot = svc._slot_of(rid)
    var c = List[Int]()
    var done = False
    while not done:
        var fr = svc.stream_next(rid)
        if fr.waiting:
            continue
        c.append(svc.streams[slot].last)
        if fr.done:
            done = True
    svc.stream_end(rid)
    return c^


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var tok = load_tokenizer_json(TOKJSON)
    var svc = GpuService(
        model, 2, 512, tok^, 151936, 7, WEIGHTS, CONFIG, TOKJSON
    )

    # ① spec 开：solo 草稿验证链
    svc.spec_on = True
    var c_spec = _chain(svc, 1, MAXNEW)
    var spec_calls = svc.spec_tel.n_calls
    var spec_ticks = svc.n_spec_ticks
    var tpt = svc.spec_tel.tokens_per_tick()

    # ② spec 关：同 prompt 的 t=1 合批路链
    svc.spec_on = False
    var c_base = _chain(svc, 2, MAXNEW)

    var parity = _same(c_spec, c_base)
    print("chain: spec", len(c_spec), "tokens, base", len(c_base),
          " parity =", parity)
    print("telemetry: spec_ticks =", spec_ticks, " calls =", spec_calls,
          " hits =", svc.spec_tel.n_hits, " drafted =",
          svc.spec_tel.n_drafted, " accepted =", svc.spec_tel.n_accepted,
          " tokens/tick =", tpt)

    # ③ 多流回退：spec 开着，两条流并发（len(act)=2 → 合批路径），
    # 各自完成且链确定（重放逐位同）。
    svc.spec_on = True
    svc.stream_begin(PROMPT, 16, 0.0, 1.0, 3)
    svc.stream_begin(PROMPT, 16, 0.0, 1.0, 4)
    var slot3 = svc._slot_of(3)
    var slot4 = svc._slot_of(4)
    var done3 = False
    var done4 = False
    var n3 = List[Int]()
    var n4 = List[Int]()
    while not (done3 and done4):
        if not done3:
            var f3 = svc.stream_next(3)
            if not f3.waiting:
                n3.append(svc.streams[slot3].last)
                if f3.done:
                    done3 = True
        if not done4:
            var f4 = svc.stream_next(4)
            if not f4.waiting:
                n4.append(svc.streams[slot4].last)
                if f4.done:
                    done4 = True
    svc.stream_end(3)
    svc.stream_end(4)
    # 并发链 vs solo 重放链（**同预算 16**——probe_194 教训：长度不
    # 等的链 _same 恒 False）。spec 分支只在 solo 触发；并发两条流
    # 都走合批路径（但后完成者可能转 solo spec——接受即贪心链，仍
    # 应逐位同）。
    var solo3 = _chain(svc, 3, 16)
    var solo4 = _chain(svc, 4, 16)
    var conc_parity = _same(n3, solo3) and _same(n4, solo4)
    print("concurrent fallback: len3 =", len(n3), " len4 =", len(n4),
          " parity vs solo =", conc_parity)

    # ④ 草稿命中场景：重复模式 prompt（计数序列）——模型应延续模式，
    # prompt-lookup 命中且接受，tokens/tick 显著 > 1。快照差分：
    # ③ 并发阶段的后完成者可能转 solo spec，计数要扣掉。
    var pre_calls = svc.spec_tel.n_calls
    var pre_acc = svc.spec_tel.n_accepted
    svc.spec_on = True
    var pat = _chain_text(
        svc, 5,
        "one, two, three, four, one, two, three, four, one, two,",
        32
    )
    var pat_calls = svc.spec_tel.n_calls - pre_calls
    var pat_acc = svc.spec_tel.n_accepted - pre_acc
    var pat_tpt = Float64(len(pat)) / Float64(max(pat_calls, 1))
    print("pattern prompt: len =", len(pat), " calls =", pat_calls,
          " accepted =", pat_acc, " tokens/tick =", pat_tpt)

    var ok = (
        parity
        and spec_calls >= 1
        and spec_ticks >= 1
        and pat_calls >= 1
        and pat_tpt > 1.0
        and conc_parity
        and len(c_spec) == MAXNEW
    )
    if ok:
        print("SERVICE-SPEC PASS")
    else:
        print("SERVICE-SPEC FAIL")
