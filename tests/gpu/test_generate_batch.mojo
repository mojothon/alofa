"""账本一百二十一：generate_batch 门——padded lockstep vs 单序列
逐 token 参照。判据：①token 逐位一致（120-a 证明 batch t=1 与
单序列 t=1 逐位恒等，生成链应全程一致）；②二次调用确定性。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime B = 3
comptime MAXNEW = 5
comptime EOS = 200000


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)
    var b = GpuSession(model, 512)

    var flat = List[Int]()
    var lens = List[Int]()
    # seq0 len3: 100,101,102 ; seq1 len6: 200..205 ; seq2 len4: 300..303
    flat.append(100)
    flat.append(101)
    flat.append(102)
    flat.append(200)
    flat.append(201)
    flat.append(202)
    flat.append(203)
    flat.append(204)
    flat.append(205)
    flat.append(300)
    flat.append(301)
    flat.append(302)
    flat.append(303)
    lens.append(3)
    lens.append(6)
    lens.append(4)

    var out1 = a.generate_batch[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32
    ](flat, lens, MAXNEW, EOS, List[Int](), List[Int]())
    var out2 = a.generate_batch[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32
    ](flat, lens, MAXNEW, EOS, List[Int](), List[Int]())

    # 参照：单序列 t=1 prefill 逐 token 链
    var mismatch = 0
    var off = 0
    for i in range(B):
        var li = lens[i]
        b.kv_pos = 0
        var prev = 0
        # 对齐：gen[k] = 第 (li-1+k) 步的 argmax（gen[0] 来自末
        # prompt token；此前 k=0 误取首 token 的 argmax）
        var steps = li - 1 + MAXNEW
        for j in range(steps):
            var toks = List[Int]()
            if j < li:
                toks.append(flat[off + j])
            else:
                toks.append(prev)
            _ = b.prefill_step[
                INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
                DType.float32
            ](toks)
            b.ctx.synchronize()
            var hv = Float32(-1e30)
            var hi = 0
            for v in range(151936):
                if b.h_logits[v] > hv:
                    hv = b.h_logits[v]
                    hi = v
            prev = hi
            if j >= li - 1:
                var k = j - (li - 1)
                if hi != out1[i * MAXNEW + k]:
                    mismatch += 1
                    print("MISMATCH seq", i, "k", k,
                          "batch:", out1[i * MAXNEW + k], "ref:", hi)
        off += li

    var det = True
    for i in range(B * MAXNEW):
        if out1[i] != out2[i]:
            det = False
    print("parity mismatch:", mismatch, " deterministic:", det)
    for i in range(B):
        print("seq", i, "tokens:",
              out1[i * MAXNEW + 0], out1[i * MAXNEW + 1],
              out1[i * MAXNEW + 2], out1[i * MAXNEW + 3],
              out1[i * MAXNEW + 4])
    if mismatch == 0 and det:
        print("*** PARITY PASS ***")
    else:
        print("*** PARITY FAIL ***")

    # 一百二十二-b：join 接槽 parity——2 初始序列（len 3/4）
    # max_new=3，1 个 joiner（len 5）在首序列完成后接管槽位。
    # 若回收槽位有 KV 污染，joiner 的 token 链会偏离参照。
    var jflat = List[Int]()
    jflat.append(500)
    jflat.append(501)
    jflat.append(502)
    jflat.append(503)
    jflat.append(504)
    var jlens = List[Int]()
    jlens.append(5)
    # 2 初始序列（b=2）：joiner 落在 out 行 [2+jt]=[6..8]
    var flat2 = List[Int]()
    flat2.append(100)
    flat2.append(101)
    flat2.append(102)
    flat2.append(200)
    flat2.append(201)
    flat2.append(202)
    flat2.append(203)
    var lens2 = List[Int]()
    lens2.append(3)
    lens2.append(4)
    var out3 = a.generate_batch[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8, DType.float32
    ](flat2, lens2, 3, EOS, jflat, jlens)
    # 参照：joiner 单序列逐 token 链（3 个生成 token）
    b.kv_pos = 0
    var prevj = 0
    var jmismatch = 0
    for j in range(5 - 1 + 3):
        var toks = List[Int]()
        if j < 5:
            toks.append(jflat[j])
        else:
            toks.append(prevj)
        _ = b.prefill_step[
            INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
            DType.float32
        ](toks)
        b.ctx.synchronize()
        var hv2 = Float32(-1e30)
        var hi2 = 0
        for v in range(151936):
            if b.h_logits[v] > hv2:
                hv2 = b.h_logits[v]
                hi2 = v
        prevj = hi2
        if j >= 4:
            var k = j - 4
            if hi2 != out3[2 * 3 + k]:
                jmismatch += 1
                print("JOIN MISMATCH k", k,
                      "batch:", out3[2 * 3 + k], "ref:", hi2)
    print("join parity mismatch:", jmismatch,
          " joiner tokens:", out3[6], out3[7], out3[8])
    if mismatch == 0 and det and jmismatch == 0:
        print("*** GENERATE GATE PASS ***")
    else:
        print("*** GENERATE GATE FAIL ***")
