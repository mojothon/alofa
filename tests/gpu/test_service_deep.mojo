"""账本一百七十五-补：深生成多流 spec 门——32 流 × 120 token 全贪心
并发，断言全部链跑满且与 solo 参照逐位同。

背景：多流 spec 上线时行预算守卫用错容量面（SPEC_MAX_ROWS=320 是
缓冲尺寸，prefill_step 批路硬容量是 PREFILL_MAX_BATCH=128），且
prefill 的 pos 检查按「行 b = 槽 b」假设扫 h_posarr[0..t]——对
row_mapped 的 spec 行越界读槽表外垃圾位，假报「kv full」。两层问题
在 HTTP 层被 SSE 清理路径掩盖（异常 → 补 [DONE] → 客户端零错误但
内容 ~6-10 token 截断，tok/s 虚高 80×）。conc8（t ≤ 72）从未触界，
本门以 t=288+ 的深生成场景封住这一类回归。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime PROMPT = (
    "Write a detailed story about a dragon who learns to cook. The dragon"
)
comptime MAXNEW = 120
comptime NSTREAMS = 32


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var tok = load_tokenizer_json(TOKJSON)
    var svc = GpuService(
        model, 64, 512, tok^, 151936, 7, WEIGHTS, CONFIG, TOKJSON
    )
    svc.spec_on = True
    # solo 参照链。
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 9999)
    var slot_ref = svc._slot_of(9999)
    var refc = List[Int]()
    var ref_done = False
    while not ref_done:
        var fr = svc.stream_next(9999)
        if not fr.waiting:
            refc.append(svc.streams[slot_ref].last)
            if fr.done:
                ref_done = True
    svc.stream_end(9999)
    var ref_ok = len(refc) == MAXNEW
    print("solo ref chain:", len(refc), "tokens ok =", ref_ok)

    # 32 流并发驱动（round-robin）。
    var lens = List[Int]()
    for r in range(NSTREAMS):
        svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 100 + r)
    var done = List[Bool]()
    for r in range(NSTREAMS):
        done.append(False)
    var ndone = 0
    var rounds = 0
    var all_full = True
    while ndone < NSTREAMS and rounds < 100000:
        rounds += 1
        for r in range(NSTREAMS):
            if done[r]:
                continue
            var fr = svc.stream_next(100 + r)
            if not fr.waiting:
                var slot = svc._slot_of(100 + r)
                if fr.done:
                    done[r] = True
                    ndone += 1
                    var n = svc.streams[slot].count
                    lens.append(n)
                    if n != MAXNEW:
                        all_full = False
                        print(
                            "stream", r, "finished SHORT: count =", n,
                            " steps =", svc.streams[slot].steps
                        )
    var mn = MAXNEW
    var mx = 0
    for n in lens:
        if n < mn:
            mn = n
        if n > mx:
            mx = n
    var slot0 = svc._slot_of(100)
    var parity = svc.streams[slot0].out == refc
    print(
        "concurrent chains: min =", mn, " max =", mx, " rounds =",
        rounds, " spec_ticks =", svc.n_spec_ticks,
        " spec_calls =", svc.spec_tel.n_calls
    )
    print("stream0 parity vs solo:", parity)
    var ok = (
        ref_ok
        and all_full
        and mn == MAXNEW
        and mx == MAXNEW
        and parity
        and svc.n_spec_ticks >= 1
    )
    if ok:
        print("*** SERVICE DEEP GATE PASS ***")
    else:
        print("*** SERVICE DEEP GATE FAIL ***")
