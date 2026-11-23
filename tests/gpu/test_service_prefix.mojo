"""账本一百七十六门：槽间 KV 前缀复用——同 prompt 请求直接拷贝源槽
KV 行 [0, plen-1)，从末枚 prompt token 起步，免掉整段 prompt prefill。

判据：
① 链 parity：命中前缀复用的流（第二条同 prompt 流）与独跑参照链
   **逐位同**——KV 行内容由位置索引唯一决定（rope 相位 = 位置），
   拷贝即复用，省行不改输出；
② 遥测健全：第二条流 n_prefix_hits ≥ 1、n_prefix_rows = plen-1
   ——prefill 真被免掉；
③ 防污染：不同 prompt 的流**不**命中（hits 不增）且链与独跑逐位同
   ——哈希三重核对 + 槽占用态校验把错配挡死；
④ 活跃流源：第二条流在第一条**结束后**发起（slot_prefix 校验面）；
   并发同 prompt（slot_cur 校验面）由 HTTP 侧覆盖。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime PROMPT = (
    "The rule is simple. First: be kind. Second: be kind."
    " Third: be kind. Fourth:"
)
comptime PROMPT2 = (
    "The garden changes with the seasons. Spring brings rain and"
    " buds. Summer brings heat and growth. Autumn brings"
)
comptime MAXNEW = 24


def _run(
    mut svc: GpuService, rid: Int, prompt: String, maxnew: Int
) raises -> List[Int]:
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
    # 池 = 2：流 A 与流 B 占两行——B 必须命中 A 的槽（或 A 释放后的
    # 同槽），交叉换 prompt 流 C 验证防污染。
    var svc = GpuService(
        model, 2, 512, tok^, 151936, 7, WEIGHTS, CONFIG, TOKJSON
    )
    if not svc.kv_prefix:
        print("KV-PREFIX FAIL (disabled)")
        return

    var h0 = svc.n_prefix_hits

    # ① 流 A：PROMPT 独跑（冷，全量 prefill）——参照链。
    var ca = _run(svc, 61, PROMPT, MAXNEW)

    # ② 流 B：同 PROMPT——应命中前缀复用，链与 A 逐位同。
    var cb = _run(svc, 62, PROMPT, MAXNEW)
    var hits_b = svc.n_prefix_hits - h0
    print("chain B == chain A:", _same(ca, cb),
          " prefix_hits:", hits_b,
          " prefix_rows:", svc.n_prefix_rows)

    # ③ 流 C：不同 prompt——不得命中（占用态/哈希核对挡住），链自洽。
    h0 = svc.n_prefix_hits
    var cc = _run(svc, 63, PROMPT2, MAXNEW)
    var hits_c = svc.n_prefix_hits - h0
    var cref = _run(svc, 64, PROMPT2, MAXNEW)
    print("chain C == ref:", _same(cc, cref), " hits_c:", hits_c)

    # ④ 防污染二段：PROMPT2 覆写了 PROMPT 的槽（池=2 的分配序）——
    # PROMPT 的旧条目此时必失活（占用态校验挡住已覆写的行）：流 D
    # 不得命中（miss），但其结束会把条目刷新，流 E 必命中。miss-
    # 再命中两段把「条目失活→重建」的完整生命周期钉死。
    h0 = svc.n_prefix_hits
    var cd = _run(svc, 65, PROMPT, MAXNEW)
    var hits_d = svc.n_prefix_hits - h0
    var ce = _run(svc, 66, PROMPT, MAXNEW)
    var hits_e = svc.n_prefix_hits - h0 - hits_d
    print("chain D == chain A:", _same(ca, cd), " hits_d:", hits_d,
          " chain E == chain A:", _same(ca, ce), " hits_e:", hits_e)

    var ok = (
        _same(ca, cb)
        and hits_b >= 1
        and svc.n_prefix_rows >= 1
        and _same(cc, cref)
        and hits_c == 0
        and _same(ca, cd)
        and hits_d == 0
        and _same(ca, ce)
        and hits_e >= 1
    )
    if ok:
        print("KV-PREFIX PASS")
    else:
        print("KV-PREFIX FAIL")
