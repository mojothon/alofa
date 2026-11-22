"""账本一百七十四-c 第二刀：spec_step 机制验证（oracle 草稿）——
参考链（generate_batch 贪心）→ 槽 1 上 spec tick：
  tick A：prompt 全 token 行化喂入（建 KV + 断 next == refl[0]）
  tick B：x0=refl[0]、oracle drafts ref[1..4] → 断 acc=4、next=refl[5]
  tick C：x0=refl[5]、oracle drafts ref[6..9] → 断 acc=4、next=refl[10]
全过 = 行化组装、行映射消费、argmax 前缀验证、pos 推进全链正确。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession
from alofa.tokenizer import load_tokenizer_json


def main() raises:
    comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
    comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
    comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
    comptime PROMPT = "Once upon a time, in a land far away,"

    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var sess = GpuSession(model, 512)
    var tok = load_tokenizer_json(TOKJSON)
    var pids = tok.encode(PROMPT)
    var p = len(pids)
    print("prompt tokens:", p)

    # ---- 参考链：槽 0 贪心 12 token ----
    var join_flat = List[Int]()
    var join_lens = List[Int]()
    var ref_out = sess.generate_batch[896, 4864](
        pids, [p], 12, -1, join_flat, join_lens, 0.0, 0, 1.0
    )
    var refl = List[Int]()
    for i in range(12):
        refl.append(ref_out[i])
    print("ref:", refl[0], refl[1], refl[2], "...")

    # ---- 单行 tick ×3（换槽，pos=0 干净）——单行稳定性判定 ----
    for k in range(3):
        var one = List[Int]()
        one.append(pids[0])
        var sq1 = List[Int]()
        sq1.append(2 + k)
        var ln1 = List[Int]()
        ln1.append(1)
        var r1 = sess.spec_step[896, 4864](one, ln1, sq1, True)
        print("single tick", k, "slot", 2 + k, "pred:", r1[1])

    # ---- 干扰定位：两行 tick（A: 同序列两行；B: 异序列各一行）----
    var t2a = List[Int]()
    t2a.append(pids[0])
    t2a.append(pids[1])
    var sq2a = List[Int]()
    sq2a.append(5)
    var ln2a = List[Int]()
    ln2a.append(2)
    var r2a = sess.spec_step[896, 4864](t2a, ln2a, sq2a, True)
    print(
        "2row same-seq row0:", Int(sess.h_tokout[0]),
        "row1:", Int(sess.h_tokout[1]), "(row0 should be 140367)",
    )

    var t2b = List[Int]()
    t2b.append(pids[0])
    t2b.append(pids[0])
    var sq2b = List[Int]()
    sq2b.append(6)
    sq2b.append(7)
    var ln2b = List[Int]()
    ln2b.append(1)
    ln2b.append(1)
    var r2b = sess.spec_step[896, 4864](t2b, ln2b, sq2b, True)
    print("2row diff-seq preds:", r2b[1], r2b[3], "(both should be 140367)")

    # ---- tick A：prompt 全 token 行化喂入（槽 1，KV 从零建）----
    var seqs1 = List[Int]()
    seqs1.append(1)
    var lens1 = List[Int]()
    lens1.append(p)
    var tick_a = sess.spec_step[896, 4864](pids, lens1, seqs1, True)
    var acc_a = tick_a[0]
    var nxt_a = tick_a[1]
    print("tick A: acc =", acc_a, "next =", nxt_a)
    var rowpred = List[Int]()
    for i in range(p):
        rowpred.append(Int(sess.h_tokout[i]))
    print("rowpred:", rowpred[0], rowpred[1], rowpred[2],
        rowpred[3], rowpred[4], rowpred[5], rowpred[6],
        rowpred[7], rowpred[8], rowpred[9], rowpred[10])
    if nxt_a != refl[0]:
        print("tick A next != refl[0] -> FAIL")
        return
    var pos_ok_a = Int(sess.h_posarr[1]) == acc_a + 1
    if not pos_ok_a:
        print("tick A pos advance FAIL:", Int(sess.h_posarr[1]))
        return

    # ---- tick B：x0=refl[0]、oracle drafts ref[1..4]（k=4）----
    var tk_b = List[Int]()
    tk_b.append(refl[0])
    for i in range(4):
        tk_b.append(refl[1 + i])
    # ---- 诊断：关 MMA/spl 走旧核路径重跑 tick A ----
    sess.h_posarr[1] = Int32(11)
    for i in range(11):
        sess.h_rowseq[i] = Int32(1)
        sess.h_rowj[i] = Int32(i)
    _ = sess.prefill_step[
        896, 4864, prefill_mma_dual=False, prefill_mma_down=False,
        prefill_tile=False, prefill_spl=False
    ](pids, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, True)
    sess.ctx.synchronize()
    print(
        "tick A2 (no-mma): tokout[10] =", Int(sess.h_tokout[10]),
        "(want 84103)",
    )
    print("pre-tickB h_posarr[1] =", Int(sess.h_posarr[1]))
    for i in range(5):
        sess.h_rowseq[i] = Int32(1)
        sess.h_rowj[i] = Int32(i)
    _ = sess.prefill_step[896, 4864](
        tk_b, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    var acc_b = 0
    var nxt_b = Int(sess.h_tokout[0])
    print("tick B: acc =", acc_b, "next =", nxt_b)
    var best1 = Float64(-1e30)
    var best2 = Float64(-1e30)
    var i1 = 0
    var i2 = 0
    for i in range(151936):
        var v = Float64(sess.h_logits_b[i])
        if v > best1:
            best2 = best1
            i2 = i1
            best1 = v
            i1 = i
        elif v > best2:
            best2 = v
            i2 = i
    print("tickB row0 top1:", i1, best1, "top2:", i2, best2)

    # ---- t=1 手工版（同上下文窗 12，单行）----
    var tk_one = List[Int]()
    tk_one.append(refl[0])
    sess.h_rowseq[0] = Int32(1)
    sess.h_rowj[0] = Int32(0)
    _ = sess.prefill_step[896, 4864](
        tk_one, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    print("t1 manual tokout:", Int(sess.h_tokout[0]), "(want ref[1]=49014)")
    print("t1 manual logits top1: skipped (use FULL read)")
    # 全量读 d_logits_b（无尺寸歧义）
    var hl_full = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 151936
    )
    sess.ctx.enqueue_copy(dst_buf=hl_full, src_buf=sess.d_logits_b)
    sess.ctx.synchronize()
    var fb1 = Float64(-1e30)
    var fb2 = Float64(-1e30)
    var fi1 = 0
    var fi2 = 0
    for i in range(151936):
        var v = Float64(hl_full[i])
        if v > fb1:
            fb2 = fb1
            fi2 = fi1
            fb1 = v
            fi1 = i
        elif v > fb2:
            fb2 = v
            fi2 = i
    print(
        "FULL row0 top1:", fi1, fb1, "top2:", fi2, fb2,
        "lg[84231]:", Float64(hl_full[84231]),
        "lg[49014]:", Float64(hl_full[49014]),
    )
    print("tick B: acc = 0 next =", Int(sess.h_tokout[0]))
    if Int(sess.h_tokout[0]) != refl[5]:
        print("tick B next != refl[5] -> FAIL")
        print("tick B next != refl[5] -> FAIL")
        return
    if Int(sess.h_posarr[1]) != 5 + acc_a + 1:
        print("tick B pos advance FAIL:", Int(sess.h_posarr[1]))
        return

    # ---- tick C：x0=refl[5]、oracle drafts ref[6..9] ----
    var tk_c = List[Int]()
    tk_c.append(refl[5])
    for i in range(4):
        tk_c.append(refl[6 + i])
    var seqs_c = List[Int]()
    seqs_c.append(1)
    var lens_c = List[Int]()
    lens_c.append(5)
    var tick_c = sess.spec_step[896, 4864](tk_c, lens_c, seqs_c)
    var acc_c = tick_c[0]
    var nxt_c = tick_c[1]
    print("tick C: acc =", acc_c, "next =", nxt_c)
    if acc_c != 4:
        print("tick C oracle accept != 4 -> FAIL")
        return
    if nxt_c != refl[10]:
        print("tick C next != refl[10] -> FAIL")
        return

    print("probe176 SPEC-MECHANISM PASS")
