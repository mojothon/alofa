"""批行池最小诊断刀（账本一百三十四）：把「行号正确性 / 行 0 确定
性 / 跨 prompt 状态 / 合批数值」四个维度逐一钉死，每刀一条打印。"""

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime CJK = "给我讲一个简短的睡前故事。"
comptime EN = "Once upon a time, in a land far away,"


def _run(mut svc: GpuService, prompt: String, rid: Int) raises -> List[Int]:
    svc.stream_begin(prompt, 16, 0.0, 1.0, rid)
    var c = List[Int]()
    var done = False
    while not done:
        var fr = svc.stream_next(rid)
        c.append(svc.streams[0].last)
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
    var svc = GpuService(
        model, 2, 512, load_tokenizer_json(TOKJSON)^, 151936, 42,
        WEIGHTS, CONFIG, TOKJSON,
    )
    # c1：CJK 落 slot 0。
    var c1 = _run(svc, CJK, 1)
    print("c1 (CJK@slot0)     :", c1[0], c1[1], c1[2], c1[3], "len", len(c1))
    # c2：CJK 落 slot 1（row_base=1 的 t=1 前向）。
    var c2 = _run(svc, CJK, 2)
    print("c2 (CJK@slot1)==c1 :", _same(c2, c1))
    # c3：再回 slot 0（行 0 确定性）。
    var c3 = _run(svc, CJK, 3)
    print("c3 (CJK@slot0)==c1 :", _same(c3, c1), " ==c2:", _same(c3, c2))
    # c4：先一条别的 prompt，再回 CJK@slot0（跨 prompt 状态）。
    var c4 = _run(svc, EN, 4)
    var c5 = _run(svc, CJK, 5)
    print("c5 (after EN)==c1  :", _same(c5, c1), " ==c3:", _same(c5, c3))
    # c6：同一 slot 连跑两次 CJK（同 prompt 重放）。
    var c6 = _run(svc, CJK, 6)
    print("c6 (rerun)==c1     :", _same(c6, c1), " ==c5:", _same(c6, c5))
    # c7：再再跑一次（奇偶判别）。
    var c7 = _run(svc, CJK, 7)
    print("c7 (rerun2)==c1    :", _same(c7, c1), " ==c6:", _same(c7, c6))
    # c8：两流合批（t=2，同 prompt——两行都应 == c1）。
    svc.stream_begin(CJK, 16, 0.0, 1.0, 11)
    svc.stream_begin(CJK, 16, 0.0, 1.0, 12)
    var ca = List[Int]()
    var cb = List[Int]()
    var da = False
    var db = False
    while (not da) or (not db):
        if not da:
            var fa = svc.stream_next(11)
            ca.append(svc.streams[0].last)
            if fa.done:
                da = True
        if not db:
            var fb = svc.stream_next(12)
            cb.append(svc.streams[1].last)
            if fb.done:
                db = True
    svc.stream_end(11)
    svc.stream_end(12)
    print("c8-batch ca==c1    :", _same(ca, c1),
          " cb==c1:", _same(cb, c1),
          " len", len(ca), len(cb))
    # c9：合批之后再独跑（t=2 残留状态）。
    var c9 = _run(svc, CJK, 13)
    print("c9 (after batch)==c1:", _same(c9, c1))
    # c10：再独跑一次（奇偶判别）。
    var c10 = _run(svc, CJK, 14)
    print("c10 ==c1           :", _same(c10, c1), " ==c9:", _same(c10, c9))
    # c11：**异 prompt 异长**合批（EN 24 帧 / CJK 16 帧，主门 interleave
    # 的精确形态）——独跑各自参照。
    svc.stream_begin(EN, 24, 0.0, 1.0, 21)
    svc.stream_begin(CJK, 16, 0.0, 1.0, 22)
    var ma = List[Int]()
    var mb = List[Int]()
    var da2 = False
    var db2 = False
    var round = 0
    while (not da2) or (not db2):
        if not da2:
            var fa = svc.stream_next(21)
            ma.append(svc.streams[0].last)
            if fa.done:
                da2 = True
        if not db2:
            var fb = svc.stream_next(22)
            mb.append(svc.streams[1].last)
            if round < 6:
                print("  round", round, "A.last", svc.streams[0].last,
                      "B.last", svc.streams[1].last,
                      "A.count", svc.streams[0].count,
                      "B.count", svc.streams[1].count,
                      "A.plen", len(svc.streams[0].prompt),
                      "B.plen", len(svc.streams[1].prompt),
                      "A.pidx", svc.streams[0].pidx,
                      "B.pidx", svc.streams[1].pidx,
                      "ticks", svc.n_ticks, "batched", svc.n_batched)
            if fb.done:
                db2 = True
        round += 1
    svc.stream_end(21)
    svc.stream_end(22)
    # mb 应 == c1（同 CJK 独跑链）；ma 与 c1 无关但 length 应 24。
    print("c11-mixed mb==c1   :", _same(mb, c1), " len", len(ma), len(mb))
    # 分歧步：mb 与 c1 同 prompt，首个分歧 token 就是污染步。
    var n = len(mb) if len(mb) < len(c1) else len(c1)
    var div_at = -1
    for i in range(n):
        if mb[i] != c1[i]:
            div_at = i
            break
    print("  mb diverge at", div_at, " mb[]:", mb[0], mb[1], mb[2], mb[3],
          mb[4], mb[5], " c1[]:", c1[0], c1[1], c1[2], c1[3], c1[4], c1[5])
    # c12：异 prompt 合批之后再独跑。
    var c12 = _run(svc, CJK, 23)
    print("c12 (after mixed)==c1:", _same(c12, c1))
    var c13 = _run(svc, CJK, 24)
    print("c13 ==c1           :", _same(c13, c1), " ==c12:", _same(c13, c12))
