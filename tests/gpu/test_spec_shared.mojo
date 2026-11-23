"""账本一百七十五-补四：会话级共享输出 n-gram 表（跨流草稿源）门。

背景：175-补三画像暴露 prompt-lookup 在真问答域 hit≈0（草稿只能
来自 prompt 内部复现）。SpecTable 把每条流的产出以滑动 n-gram
窗口入共享表，查询链式跟随重建草稿链——同前缀的历史生成逐位重放。

门判据：
① 表关参照区：QA prompt 两次 solo 链逐位同、shared_hits==0、
   表 entries==0（关即零插桩零召回）；
② 表开暖跑：贪心链与参照逐位同（表召回不改贪心链）；
③ 表开重放：同 prompt 二跑链逐位同（召回不改链），且本段
   shared_hits ≥ 1（确实从表拿到了草稿）；
④ 异 prompt 链与表关参照逐位同（表的 QA 内容对其零扰动）。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
# 真问答 prompt：无内部复现结构（prompt-lookup 的盲区，175-补三
# 实测 hit 0.8%）——共享表的靶域。
comptime PROMPT_QA = (
    "User: What are the main causes of climate change? Give a detailed"
    " explanation with examples.\nAssistant:"
)
comptime PROMPT_OTHER = (
    "User: Explain step by step how to make pancakes at home.\nAssistant:"
)
comptime MAXNEW = 40
comptime MAXNEW_OTHER = 24


def _chain_text(
    mut svc: GpuService, rid: Int, prompt: String, maxnew: Int
) raises -> List[Int]:
    """solo 贪心一条流，回 token 链（test_service_spec 的 _chain_text 同款）。"""
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
        model, 8, 512, tok^, 151936, 7, WEIGHTS, CONFIG, TOKJSON
    )
    svc.spec_on = True

    # ① 表关参照区。
    svc.spec_shared = False
    var ref_qa1 = _chain_text(svc, 11, PROMPT_QA, MAXNEW)
    var ref_qa2 = _chain_text(svc, 12, PROMPT_QA, MAXNEW)
    var ref_other = _chain_text(svc, 13, PROMPT_OTHER, MAXNEW_OTHER)
    var ref_determinism = _same(ref_qa1, ref_qa2)
    var ref_no_shared = svc.spec_table.n_shared_hits == 0
    var ref_no_entries = len(svc.spec_table.map) == 0
    print(
        "① ref: qa1/qa2 determinism =", ref_determinism,
        " shared_hits =", svc.spec_table.n_shared_hits,
        " entries =", len(svc.spec_table.map)
    )

    # ② 表开：QA 暖跑（边生成边入表）。
    svc.spec_shared = True
    var warm = _chain_text(svc, 21, PROMPT_QA, MAXNEW)
    var warm_parity = _same(warm, ref_qa1)
    var hits_before = svc.spec_table.n_shared_hits
    print(
        "② warm: parity vs ref =", warm_parity,
        " entries =", len(svc.spec_table.map),
        " shared_hits so far =", svc.spec_table.n_shared_hits
    )

    # ③ 表开重放：同 prompt 二跑——共享表应供全链草稿。
    var replay = _chain_text(svc, 22, PROMPT_QA, MAXNEW)
    var replay_parity = _same(replay, ref_qa1)
    var shared_gained = svc.spec_table.n_shared_hits - hits_before
    print(
        "③ replay: parity vs ref =", replay_parity,
        " shared_hits gained =", shared_gained
    )

    # ④ 异 prompt 链不受扰。
    var other2 = _chain_text(svc, 23, PROMPT_OTHER, MAXNEW_OTHER)
    var other_parity = _same(other2, ref_other)
    print("④ other: parity vs ref =", other_parity)

    var ok = (
        ref_determinism
        and ref_no_shared
        and ref_no_entries
        and warm_parity
        and replay_parity
        and shared_gained >= 1
        and other_parity
    )
    print(
        "telemetry: inserts =", svc.spec_table.n_inserts,
        " shared_hits =", svc.spec_table.n_shared_hits,
        " drafted =", svc.spec_tel.n_drafted,
        " accepted =", svc.spec_tel.n_accepted
    )
    if ok:
        print("*** SPEC SHARED GATE PASS ***")
    else:
        print("*** SPEC SHARED GATE FAIL ***")
