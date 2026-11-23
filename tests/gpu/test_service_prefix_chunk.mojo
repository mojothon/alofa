"""账本一百七十六-b 门：部分前缀复用——多轮对话/共享 preamble 场景，
新 prompt 是旧 prompt 的**延长**（精确全等匹配永远 miss）。chunk 链
式哈希（16-token 边界）给出最长公共前缀，命中即拷贝行 [0, lcp)，
剩余 prompt 行正常 prefill。

判据（全部排在冷链重跑之前——冷链会覆写槽行并使条目失活，这是
占用态校验的设计性质，不是缺陷）：
① 延长链 parity：B 的 prompt = A 的 prompt + 延长句——B 必经 chunk
   路（n_chunk_hits ≥ 1、rows 为 16 的倍数），且命中链与 B 的**冷链**
   （kv_prefix 关、全量 prefill）**逐位同**；
② 另一延长同样命中 + parity（多轮生产形态）；
③ 不相关 preamble 的 prompt 不命中（hits 不增）且链与冷链逐位同。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"


def _mkprompt(rep: Int, tail: String) -> String:
    """合成 prompt：rep 段重复 preamble（确保 > 16 token，chunk 边界
    多枚）+ 延长尾句（可为空）。"""
    var s = String("")
    for _i in range(rep):
        s += "The committee reviewed the annual budget report carefully. "
    s += tail
    return s


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
    var svc = GpuService(
        model, 2, 512, tok^, 151936, 7, WEIGHTS, CONFIG, TOKJSON
    )
    if not svc.kv_prefix:
        print("KV-PREFIX-CHUNK FAIL (disabled)")
        return

    var p_a = _mkprompt(6, "")
    var p_b = _mkprompt(6, " Now summarize the report briefly.")
    var p_c = _mkprompt(6, " List every finding in detail with dates.")
    var p_d = (
        "Zebra quixotic journeys vault beyond murky glassy ponds."
        " Frozen meadows glisten under pale starlight tonight."
    )
    var maxnew = 20

    # ① A 冷跑登记 chunk 边界。
    _ = _run(svc, 71, p_a, maxnew)
    var h0 = svc.n_prefix_hits

    # ② B = A + 延长——chunk 路命中，链与冷链逐位同。
    var chain_hit_b = _run(svc, 72, p_b, maxnew)
    var hits_b = svc.n_prefix_hits - h0
    var rows_b = svc.n_prefix_rows
    var chunk_b = svc.n_chunk_hits
    print("B hits:", hits_b, " chunk:", chunk_b, " rows:", rows_b)

    # ③ C = A + 另一延长——同样命中（B 刚结束、其条目活跃）。
    h0 = svc.n_prefix_hits
    var chain_hit_c = _run(svc, 74, p_c, maxnew)
    var hits_c = svc.n_prefix_hits - h0
    print("C hits:", hits_c)

    # ④ 不相关 preamble——不命中。
    h0 = svc.n_prefix_hits
    var chain_hit_d = _run(svc, 76, p_d, maxnew)
    var hits_d = svc.n_prefix_hits - h0
    print("D hits:", hits_d)

    # 冷链参照（prefix 关、全量 prefill）——放最后：冷链会覆写槽行
    # 使条目失活（占用态校验的设计性质）。
    svc.kv_prefix = False
    var chain_cold_b = _run(svc, 73, p_b, maxnew)
    var chain_cold_c = _run(svc, 75, p_c, maxnew)
    var chain_cold_d = _run(svc, 77, p_d, maxnew)
    svc.kv_prefix = True

    var ok = (
        _same(chain_hit_b, chain_cold_b)
        and hits_b >= 1
        and chunk_b >= 1
        and rows_b >= 16
        and rows_b % 16 == 0
        and _same(chain_hit_c, chain_cold_c)
        and hits_c >= 1
        and _same(chain_hit_d, chain_cold_d)
        and hits_d == 0
    )
    if ok:
        print("KV-PREFIX-CHUNK PASS")
    else:
        print("KV-PREFIX-CHUNK FAIL")
