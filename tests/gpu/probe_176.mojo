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

    # ---- 会话首个 forward：槽 9 建底 A''（污染前基线）----
    var sq0 = List[Int]()
    sq0.append(9)
    var ln0 = List[Int]()
    ln0.append(11)
    _ = sess.spec_step[896, 4864](pids, ln0, sq0, True)
    print("A'' first-ever: tokout[10] =", Int(sess.h_tokout[10]),
        "(A on slot1 gives 84103)")

    # ---- 污染二分：A'' 后逐步插入操作，每步后测新槽 ----
    var sqx = List[Int]()
    var lnx = List[Int]()
    lnx.append(11)

    # 步 1：仅再跑一次 spec tick（槽 10）后测槽 11
    sqx.append(10)
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    sqx[0] = 11
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    print("step1 (spec+spec): slot11 tokout[10] =", Int(sess.h_tokout[10]))
    # 步 2：generate_batch（单序列 ref 链）后测新槽 12
    var gflat = List[Int]()
    for i in range(11):
        gflat.append(pids[i])
    var glens = List[Int]()
    glens.append(11)
    var gjoin = List[Int]()
    var gjoinl = List[Int]()
    _ = sess.generate_batch[896, 4864](
        gflat, glens, 2, -1, gjoin, gjoinl, 0.0, 0, 1.0
    )
    sqx[0] = 12
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    print("step2 (after gen_batch): slot12 tokout[10] =",
        Int(sess.h_tokout[10]))
    # 步 3：单行 tick ×3（tk=pids[0]）
    for k in range(3):
        var one = List[Int]()
        one.append(pids[0])
        var sq1x = List[Int]()
        sq1x.append(13 + k)
        var ln1x = List[Int]()
        ln1x.append(1)
        _ = sess.spec_step[896, 4864](one, ln1x, sq1x, True)
    sqx[0] = 20
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    print("step3 (after 3 single ticks): slot20 tokout[10] =",
        Int(sess.h_tokout[10]))

    # 步 4：2row same-seq（槽 21、lens=2）+ diff-seq（槽 22/23）
    var t2 = List[Int]()
    t2.append(pids[0])
    t2.append(pids[1])
    var sq21 = List[Int]()
    sq21.append(21)
    var ln2x = List[Int]()
    ln2x.append(2)
    _ = sess.spec_step[896, 4864](t2, ln2x, sq21, True)
    var sq22 = List[Int]()
    sq22.append(22)
    sq22.append(23)
    var ln11x = List[Int]()
    ln11x.append(1)
    ln11x.append(1)
    var t22 = List[Int]()
    t22.append(pids[0])
    t22.append(pids[0])
    _ = sess.spec_step[896, 4864](t22, ln11x, sq22, True)
    sqx[0] = 24
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    print("step4 (after 2row ticks): slot24 tokout[10] =",
        Int(sess.h_tokout[10]))

    # 步 5：FULL logits 读（大 copy）
    var hlA = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 151936
    )
    sess.ctx.enqueue_copy(dst_buf=hlA, src_buf=sess.d_logits_b)
    sess.ctx.synchronize()
    sqx[0] = 25
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    print("step5 (after full-logits read): slot25 tokout[10] =",
        Int(sess.h_tokout[10]))
    # 步 6：tick A（槽 1）+ tick B（draft 失败路径）
    var sq1y = List[Int]()
    sq1y.append(1)
    _ = sess.spec_step[896, 4864](pids, lnx, sq1y, True)
    print("step6a tickA: slot1 tokout[10] =", Int(sess.h_tokout[10]))
    var tkb = List[Int]()
    tkb.append(refl[0])
    for i in range(4):
        tkb.append(refl[1 + i])
    var ln5y = List[Int]()
    ln5y.append(5)
    _ = sess.spec_step[896, 4864](tkb, ln5y, sq1y)
    print("step6b tickB: tokout[0] =", Int(sess.h_tokout[0]))
    sqx[0] = 26
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    print("step6c (after draft-fail tick): slot26 tokout[10] =",
        Int(sess.h_tokout[10]))

    # 步 7：t1 manual（手工 prefill_step、copy_logits=True）
    var tk1 = List[Int]()
    tk1.append(refl[0])
    sess.h_rowseq[0] = Int32(1)
    sess.h_rowj[0] = Int32(0)
    _ = sess.prefill_step[896, 4864](
        tk1, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    print("step7a t1 manual: tokout =", Int(sess.h_tokout[0]))
    sqx[0] = 27
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    print("step7b (after t1 manual): slot27 tokout[10] =",
        Int(sess.h_tokout[10]))
    # 步 8：同槽 9 第二次建底（pos 重置 0）——上轮 84231 复现判定
    sess.h_posarr[9] = Int32(0)
    sqx[0] = 9
    _ = sess.spec_step[896, 4864](pids, lnx, sqx, True)
    print("step8 slot9 2nd build: tokout[10] =", Int(sess.h_tokout[10]),
        "(1st was 84103)")
    # 步 9：KV 数值对照（层 0）——槽 9（tick A 行化链）vs 槽 0（refl 链）
    # store 布局 [BMAX][max_tokens=512][kv_dim=128]，f32
    var kall = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 512 * 128
    )
    sess.ctx.enqueue_copy(dst_buf=kall, src_buf=sess.d_k_store_b[0])
    sess.ctx.synchronize()
    var dmax = Float64(0)
    var dsum = Float64(0)
    for r in range(11):
        for c in range(128):
            var a = Float64(kall[9 * 512 * 128 + r * 128 + c])
            var b = Float64(kall[0 * 512 * 128 + r * 128 + c])
            d = abs(a - b)
            if d > dmax:
                dmax = d
            dsum += d
    print("KV diff L0 rows0-10: max =", dmax, "sum =", dsum)
    # 分行 diff（随行号增长 = 尾组泄漏特征）
    var rr = List[Int]()
    rr.append(0)
    rr.append(1)
    rr.append(5)
    rr.append(9)
    rr.append(10)
    for r0 in rr:
        var rm = Float64(0)
        for c in range(128):
            var a = Float64(kall[9 * 512 * 128 + r0 * 128 + c])
            var b = Float64(kall[r0 * 128 + c])
            if abs(a - b) > rm:
                rm = abs(a - b)
        print("KV diff row", r0, "max =", rm)
    # 单行 tick 槽 13 vs 槽 0 的 row0（同 token 同 pos、t=1 路径）
    var s13 = Float64(0)
    for c in range(128):
        s13 += abs(
            Float64(kall[13 * 512 * 128 + c])
            - Float64(kall[c])
        )
    print("KV slot13-row0 vs slot0-row0 sum =", s13)

    var vall = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 512 * 128
    )
    sess.ctx.enqueue_copy(dst_buf=vall, src_buf=sess.d_v_store_b[0])
    sess.ctx.synchronize()
    var dvmax = Float64(0)
    for r in range(11):
        for c in range(128):
            var a = Float64(vall[9 * 512 * 128 + r * 128 + c])
            var b = Float64(vall[0 * 512 * 128 + r * 128 + c])
            d = abs(a - b)
            if d > dvmax:
                dvmax = d
    print("VV diff L0 rows0-10: max =", dvmax)
    # 步 10：d_kb 行 0 对照（spec 链 vs decode tick 链）
    var kb1 = sess.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess.ctx.enqueue_copy(dst_buf=kb1, src_buf=sess.d_kb)
    sess.ctx.synchronize()
    var s1 = Float64(0)
    for c in range(128):
        s1 += abs(Float64(kb1[c]) - Float64(kall[c]))
    print("kb-row0: spec vs slot0-KV sum =", s1, "(rope 前后应不同)")
    # 步 10b：decode tick（t=1 槽 0 pos=0 pids[0]）后对比 d_kb 行 0
    sess.h_posarr[0] = Int32(0)
    var dtok = List[Int]()
    dtok.append(pids[0])
    _ = sess.prefill_step[896, 4864](
        dtok, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    var kb2 = sess.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess.ctx.enqueue_copy(dst_buf=kb2, src_buf=sess.d_kb)
    sess.ctx.synchronize()
    var s2 = Float64(0)
    var smax = Float64(0)
    for c in range(128):
        var dd = abs(Float64(kb1[c]) - Float64(kb2[c]))
        s2 += dd
        if dd > smax:
            smax = dd
    print("d_kb row0 spec-t11 vs decode-t1: sum =", s2, "max =", smax)
    # vb 对照
    var vb1 = sess.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess.ctx.enqueue_copy(dst_buf=vb1, src_buf=sess.d_vb)
    sess.ctx.synchronize()
    var s3 = Float64(0)
    for c in range(128):
        s3 += abs(Float64(vb1[c]) - Float64(kb2[c]))
    print("d_vb row0 vs d_kb row0 (sanity): sum =", s3)

    # ---- 步 10c：d_xb/d_xnormb16/d_kb row0 三点对照（t=11 vs t=1）----
    var sq30 = List[Int]()
    sq30.append(30)
    var ln30 = List[Int]()
    ln30.append(11)
    _ = sess.spec_step[896, 4864](pids, ln30, sq30, True)
    sess.ctx.synchronize()
    var hx11 = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=hx11, src_buf=sess.d_xb)
    var hn11 = sess.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess.ctx.enqueue_copy(dst_buf=hn11, src_buf=sess.d_xnormb16)
    var hk11 = sess.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess.ctx.enqueue_copy(dst_buf=hk11, src_buf=sess.d_kb)
    sess.ctx.synchronize()
    sess.h_posarr[0] = Int32(0)
    sess.h_rowseq[0] = Int32(0)
    sess.h_rowj[0] = Int32(0)
    var dtok2 = List[Int]()
    dtok2.append(pids[0])
    _ = sess.prefill_step[896, 4864](
        dtok2, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    var hx1 = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=hx1, src_buf=sess.d_xb)
    var hn1 = sess.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess.ctx.enqueue_copy(dst_buf=hn1, src_buf=sess.d_xnormb16)
    var hk1 = sess.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess.ctx.enqueue_copy(dst_buf=hk1, src_buf=sess.d_kb)
    sess.ctx.synchronize()
    var xs = Float64(0)
    var xmax = Float64(0)
    var ns = Float64(0)
    var nmax = Float64(0)
    var ks2 = Float64(0)
    var kmax2 = Float64(0)
    for c in range(896):
        var dx = abs(Float64(hx11[c]) - Float64(hx1[c]))
        xs += dx
        if dx > xmax:
            xmax = dx
        var dn = abs(Float64(hn11[c]) - Float64(hn1[c]))
        ns += dn
        if dn > nmax:
            nmax = dn
    for c in range(128):
        var dk = abs(Float64(hk11[c]) - Float64(hk1[c]))
        ks2 += dk
        if dk > kmax2:
            kmax2 = dk
    print("10c d_xb row0: sum =", xs, "max =", xmax)
    print("10c d_xnormb16 row0: sum =", ns, "max =", nmax)
    print("10c d_kb row0: sum =", ks2, "max =", kmax2)

    # ---- 步 10d：同参数 t=11 spec tick ×2（槽 31/32）三点自比 ----
    var sq31 = List[Int]()
    sq31.append(31)
    _ = sess.spec_step[896, 4864](pids, ln30, sq31, True)
    sess.ctx.synchronize()
    var kbA = sess.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess.ctx.enqueue_copy(dst_buf=kbA, src_buf=sess.d_kb)
    var ksA = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 512 * 128
    )
    sess.ctx.enqueue_copy(dst_buf=ksA, src_buf=sess.d_k_store_b[0])
    var xbA2 = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=xbA2, src_buf=sess.d_xb)
    sess.ctx.synchronize()
    var sq32 = List[Int]()
    sq32.append(32)
    _ = sess.spec_step[896, 4864](pids, ln30, sq32, True)
    sess.ctx.synchronize()
    var kbB = sess.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess.ctx.enqueue_copy(dst_buf=kbB, src_buf=sess.d_kb)
    var ksB = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 512 * 128
    )
    sess.ctx.enqueue_copy(dst_buf=ksB, src_buf=sess.d_k_store_b[0])
    var xbB2 = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=xbB2, src_buf=sess.d_xb)
    sess.ctx.synchronize()
    var kb_d = Float64(0)
    var ks_d = Float64(0)
    var xb_d = Float64(0)
    for c in range(128):
        kb_d += abs(Float64(kbA[c]) - Float64(kbB[c]))
        ks_d += abs(
            Float64(ksA[31 * 512 * 128 + c]) - Float64(ksB[32 * 512 * 128 + c])
        )
    for c in range(896):
        xb_d += abs(Float64(xbA2[c]) - Float64(xbB2[c]))
    print("10d d_kb row0 A-vs-B: sum =", kb_d)
    print("10d kstore L0 row0 (s31 vs s32): sum =", ks_d)
    print("10d d_xb row0 A-vs-B: sum =", xb_d)

    # ---- 步 10e：L0 kstore row0 t11(s31=ksA) vs t1(slot0) ----
    print("10e kv_pos before decode =", sess.kv_pos)
    sess.kv_pos = 0
    var dtok3 = List[Int]()
    dtok3.append(pids[0])
    _ = sess.prefill_step[896, 4864](
        dtok3, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    var ksC = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 512 * 128
    )
    sess.ctx.enqueue_copy(dst_buf=ksC, src_buf=sess.d_k_store_b[0])
    sess.ctx.synchronize()
    var k0d = Float64(0)
    var k0m = Float64(0)
    for c in range(128):
        var dd = abs(
            Float64(ksA[31 * 512 * 128 + c]) - Float64(ksC[c])
        )
        k0d += dd
        if dd > k0m:
            k0m = dd
    print("10e L0 kstore row0 t11-vs-t1: sum =", k0d, "max =", k0m)

    # ---- 步 10f：ref2(generate_batch 重建 slot0) vs spec(slot33) 逐行 ----
    var jf2 = List[Int]()
    var jl2 = List[Int]()
    var ref2 = sess.generate_batch[896, 4864](
        pids, [p], 12, -1, jf2, jl2, 0.0, 0, 1.0
    )
    print("ref2:", ref2[0], ref2[1], ref2[2])
    var sq33 = List[Int]()
    sq33.append(33)
    _ = sess.spec_step[896, 4864](pids, ln30, sq33, True)
    sess.ctx.synchronize()
    var ksD = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 512 * 128
    )
    sess.ctx.enqueue_copy(dst_buf=ksD, src_buf=sess.d_k_store_b[0])
    sess.ctx.synchronize()
    for rr in range(11):
        var rd = Float64(0)
        var rm = Float64(0)
        for c in range(128):
            var dd = abs(
                Float64(ksD[0 * 512 * 128 + rr * 128 + c])
                - Float64(ksD[33 * 512 * 128 + rr * 128 + c])
            )
            rd += dd
            if dd > rm:
                rm = dd
        print("10f L0 K row", rr, ": sum =", rd, "max =", rm)

    # ---- 步 10g：全层 K 逐行比对 + ref3 判首调用一次性 ----
    var found_first = False
    for l2 in range(sess.n_layers):
        var kl2 = sess.ctx.enqueue_create_host_buffer[DType.float32](
            64 * 512 * 128
        )
        sess.ctx.enqueue_copy(dst_buf=kl2, src_buf=sess.d_k_store_b[l2])
        sess.ctx.synchronize()
        var ld = Float64(0)
        var lm = Float64(0)
        var lrow = Int(-1)
        for rr in range(11):
            for c in range(128):
                var dd = abs(
                    Float64(kl2[0 * 512 * 128 + rr * 128 + c])
                    - Float64(kl2[33 * 512 * 128 + rr * 128 + c])
                )
                ld += dd
                if dd > lm:
                    lm = dd
                if dd > 0 and lrow == -1:
                    lrow = rr
        if lm > 0:
            if not found_first:
                print("10g first diverging layer =", l2)
                found_first = True
            print("10g L", l2, "K diff: sum =", ld, "max =", lm,
                "firstrow =", lrow)
    if not found_first:
        print("10g all layers K identical rows0-10")
    var jf3 = List[Int]()
    var jl3 = List[Int]()
    var ref3 = sess.generate_batch[896, 4864](
        pids, [p], 12, -1, jf3, jl3, 0.0, 0, 1.0
    )
    print("ref3:", ref3[0], ref3[1], ref3[2])

    # ---- 步 10h：virgin session vs dirty session 的 t=1 MMA 输出对照 ----
    var sess2 = GpuSession(model, 512)
    sess.kv_pos = 0
    var dtok4 = List[Int]()
    dtok4.append(pids[0])
    _ = sess.prefill_step[896, 4864](
        dtok4, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    var hqA = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=hqA, src_buf=sess.d_qb)
    var hxA = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=hxA, src_buf=sess.d_xb)
    var hkA2 = sess.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess.ctx.enqueue_copy(dst_buf=hkA2, src_buf=sess.d_kb)
    sess.ctx.synchronize()
    _ = sess2.prefill_step[896, 4864](
        dtok4, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess2.ctx.synchronize()
    var hqB = sess2.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess2.ctx.enqueue_copy(dst_buf=hqB, src_buf=sess2.d_qb)
    var hxB = sess2.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess2.ctx.enqueue_copy(dst_buf=hxB, src_buf=sess2.d_xb)
    var hkB2 = sess2.ctx.enqueue_create_host_buffer[DType.float32](320 * 128)
    sess2.ctx.enqueue_copy(dst_buf=hkB2, src_buf=sess2.d_kb)
    sess2.ctx.synchronize()
    var qd = Float64(0)
    var qm = Float64(0)
    var xd = Float64(0)
    var xm = Float64(0)
    var kd2 = Float64(0)
    for c in range(896):
        var dq = abs(Float64(hqA[c]) - Float64(hqB[c]))
        qd += dq
        if dq > qm:
            qm = dq
        var dxx = abs(Float64(hxA[c]) - Float64(hxB[c]))
        xd += dxx
        if dxx > xm:
            xm = dxx
    for c in range(128):
        kd2 += abs(Float64(hkA2[c]) - Float64(hkB2[c]))
    print("10h d_qb row0 dirty-vs-virgin: sum =", qd, "max =", qm)
    print("10h d_xb row0 dirty-vs-virgin: sum =", xd, "max =", xm)
    print("10h d_kb row0 dirty-vs-virgin: sum =", kd2)

    # ---- 步 10i：sess2 上首跑 vs 重跑的 generate_batch 转变复现 ----
    var jf4 = List[Int]()
    var jl4 = List[Int]()
    var refA = sess2.generate_batch[896, 4864](
        pids, [p], 12, -1, jf4, jl4, 0.0, 0, 1.0
    )
    print("10i sess2 1st gen:", refA[0], refA[1], refA[2])
    var jf5 = List[Int]()
    var jl5 = List[Int]()
    var refB = sess2.generate_batch[896, 4864](
        pids, [p], 12, -1, jf5, jl5, 0.0, 0, 1.0
    )
    print("10i sess2 2nd gen:", refB[0], refB[1], refB[2])

    # ---- 步 10j：t=1 logits/normedb dirty(sess) vs virgin(sess2) ----
    sess.kv_pos = 0
    sess2.kv_pos = 0
    var dtok5 = List[Int]()
    dtok5.append(pids[0])
    _ = sess.prefill_step[896, 4864](
        dtok5, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    _ = sess2.prefill_step[896, 4864](
        dtok5, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess2.ctx.synchronize()
    var lgA = sess.ctx.enqueue_create_host_buffer[DType.float32](151936)
    sess.ctx.enqueue_copy(dst_buf=lgA, src_buf=sess.d_logits_b)
    var lgB = sess2.ctx.enqueue_create_host_buffer[DType.float32](151936)
    sess2.ctx.enqueue_copy(dst_buf=lgB, src_buf=sess2.d_logits_b)
    var neA = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=neA, src_buf=sess.d_normedb)
    var neB = sess2.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess2.ctx.enqueue_copy(dst_buf=neB, src_buf=sess2.d_normedb)
    sess.ctx.synchronize()
    sess2.ctx.synchronize()
    var ld2 = Float64(0)
    var lm2 = Float64(0)
    var nd = Float64(0)
    for c in range(151936):
        var dd = abs(Float64(lgA[c]) - Float64(lgB[c]))
        ld2 += dd
        if dd > lm2:
            lm2 = dd
    for c in range(896):
        nd += abs(Float64(neA[c]) - Float64(neB[c]))
    var amA = 0
    var avA = Float64(lgA[0])
    var amB = 0
    var avB = Float64(lgB[0])
    for c in range(1, 151936):
        if Float64(lgA[c]) > avA:
            avA = Float64(lgA[c])
            amA = c
        if Float64(lgB[c]) > avB:
            avB = Float64(lgB[c])
            amB = c
    print("10j logits diff: sum =", ld2, "max =", lm2)
    print("10j normedb diff: sum =", nd)
    print("10j argmax dirty =", amA, "virgin =", amB)
    var xbA3 = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=xbA3, src_buf=sess.d_xb)
    var xbB3 = sess2.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess2.ctx.enqueue_copy(dst_buf=xbB3, src_buf=sess2.d_xb)
    var atA3 = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=atA3, src_buf=sess.d_attnb)
    var atB3 = sess2.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess2.ctx.enqueue_copy(dst_buf=atB3, src_buf=sess2.d_attnb)
    var xnA3 = sess.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess.ctx.enqueue_copy(dst_buf=xnA3, src_buf=sess.d_xnormb16)
    var xnB3 = sess2.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess2.ctx.enqueue_copy(dst_buf=xnB3, src_buf=sess2.d_xnormb16)
    sess.ctx.synchronize()
    sess2.ctx.synchronize()
    var xbd = Float64(0)
    var xbm = Float64(0)
    var atd = Float64(0)
    var xnd = Float64(0)
    for c in range(896):
        var d1 = abs(Float64(xbA3[c]) - Float64(xbB3[c]))
        xbd += d1
        if d1 > xbm:
            xbm = d1
        atd += abs(Float64(atA3[c]) - Float64(atB3[c]))
        var d3 = abs(
            Float64(Float32(xnA3[c])) - Float64(Float32(xnB3[c]))
        )
        xnd += d3
    print("10j d_xb row0 diff: sum =", xbd, "max =", xbm)
    print("10j d_attb row0 diff: sum =", atd)
    print("10j d_xnormb16 row0 diff: sum =", xnd)

    # ---- 步 10k：d_normed row0 + 同 session 二连 tick 自比 ----
    var ndA = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=ndA, src_buf=sess.d_normed)
    var ndB = sess2.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess2.ctx.enqueue_copy(dst_buf=ndB, src_buf=sess2.d_normed)
    sess.ctx.synchronize()
    sess2.ctx.synchronize()
    var ndd = Float64(0)
    for c in range(896):
        ndd += abs(Float64(ndA[c]) - Float64(ndB[c]))
    print("10k d_normed row0 diff: sum =", ndd)
    _ = sess.prefill_step[896, 4864](
        dtok5, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    var xnA4 = sess.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess.ctx.enqueue_copy(dst_buf=xnA4, src_buf=sess.d_xnormb16)
    var xbA4 = sess.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess.ctx.enqueue_copy(dst_buf=xbA4, src_buf=sess.d_xb)
    sess.ctx.synchronize()
    var xnd2 = Float64(0)
    var xbd2 = Float64(0)
    for c in range(896):
        xnd2 += abs(
            Float64(Float32(xnA3[c])) - Float64(Float32(xnA4[c]))
        )
        xbd2 += abs(Float64(xbA3[c]) - Float64(xbA4[c]))
    print("10k sess self d_xnormb16 row0 (run1 vs run2): sum =", xnd2)
    print("10k sess self d_xb row0 (run1 vs run2): sum =", xbd2)
    # NaN/差异位置指纹
    var nbad = 0
    var first_bad = Int(-1)
    var last_bad = Int(-1)
    for c in range(896):
        var v = Float32(xnA4[c])
        if (v != v) or abs(Float64(v)) > 60000.0:
            if first_bad == -1:
                first_bad = c
            last_bad = c
            nbad += 1
    print("10l xnA4 nonfinite count =", nbad, "first =", first_bad,
        "last =", last_bad)
    var ncnt2 = 0
    var nfb2 = Int(-1)
    var nlb2 = Int(-1)
    for c in range(896):
        var d5 = abs(Float64(Float32(xnA3[c])) - Float64(Float32(xnB3[c])))
        if d5 > 0.001:
            if nfb2 == -1:
                nfb2 = c
            nlb2 = c
            ncnt2 += 1
    print("10l 10j-diff count =", ncnt2, "first =", nfb2, "last =", nlb2)
    # 复抓两次（无 tick 间隔）判异步写手
    var xnA5 = sess.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess.ctx.enqueue_copy(dst_buf=xnA5, src_buf=sess.d_xnormb16)
    var xnA6 = sess.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess.ctx.enqueue_copy(dst_buf=xnA6, src_buf=sess.d_xnormb16)
    sess.ctx.synchronize()
    var xnd3 = Float64(0)
    var xnd4 = Float64(0)
    var nbad3 = 0
    for c in range(896):
        xnd3 += abs(
            Float64(Float32(xnA4[c])) - Float64(Float32(xnA5[c]))
        )
        xnd4 += abs(
            Float64(Float32(xnA5[c])) - Float64(Float32(xnA6[c]))
        )
        var v5 = Float32(xnA5[c])
        if v5 != v5:
            nbad3 += 1
    print("10l regrab1 (run2 vs regrab): sum =", xnd3)
    print("10l regrab2 (regrab vs regrab2): sum =", xnd4, "nan count =", nbad3)

    # ---- 步 10m：镜像 vs f16(d_normed) 主机侧逐位对照 ----
    var badA = 0
    var badB = 0
    var fdA = Float64(0)
    var fdB = Float64(0)
    var fbadA = 0
    var fbadB = 0
    for c in range(896):
        var expA = Float16(Float32(ndA[c]))
        var expB = Float16(Float32(ndB[c]))
        var eA = Float64(Float32(xnA5[c])) - Float64(Float32(expA))
        var eB = Float64(Float32(xnB3[c])) - Float64(Float32(expB))
        fdA += abs(eA)
        fdB += abs(eB)
        if abs(eA) > 1e-4:
            badA += 1
        if abs(eB) > 1e-4:
            badB += 1
        var nA5 = Float32(xnA5[c])
        var nB3 = Float32(xnB3[c])
        if (nA5 != nA5) or abs(Float64(nA5)) > 60000.0:
            fbadA += 1
        if (nB3 != nB3) or abs(Float64(nB3)) > 60000.0:
            fbadB += 1
    print("10m sess mirror-vs-f16(normed): sum =", fdA, "bad =", badA,
        "nonfinite =", fbadA)
    print("10m sess2 mirror-vs-f16(normed): sum =", fdB, "bad =", badB,
        "nonfinite =", fbadB)
    var npos = List[Int]()
    for c in range(896):
        var v7 = Float32(xnB3[c])
        if (v7 != v7) or abs(Float64(v7)) > 60000.0:
            npos.append(c)
    print("10n sess2 nonfinite positions:", npos[0], npos[1], npos[2],
        npos[3], npos[4], npos[5], npos[6], npos[7], npos[8], npos[9],
        npos[10], npos[11], npos[12], npos[13], npos[14])
    # ---- 步 10o：virgin sess3 首个 t=1 tick 最小复现 ----
    var sess3 = GpuSession(model, 512)
    var dtok6 = List[Int]()
    dtok6.append(pids[0])
    _ = sess3.prefill_step[896, 4864](
        dtok6, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess3.ctx.synchronize()
    var ndC = sess3.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess3.ctx.enqueue_copy(dst_buf=ndC, src_buf=sess3.d_normed)
    var xnC = sess3.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess3.ctx.enqueue_copy(dst_buf=xnC, src_buf=sess3.d_xnormb16)
    sess3.ctx.synchronize()
    var badC = 0
    var fdC = Float64(0)
    var fnfC = 0
    var cpos = List[Int]()
    for c in range(896):
        var expC = Float16(Float32(ndC[c]))
        var eC = Float64(Float32(xnC[c])) - Float64(Float32(expC))
        fdC += abs(eC)
        if abs(eC) > 1e-4:
            badC += 1
        var v8 = Float32(xnC[c])
        if (v8 != v8) or abs(Float64(v8)) > 60000.0:
            fnfC += 1
            if len(cpos) < 16:
                cpos.append(c)
    print("10o sess3 first-tick mirror-vs-f16(normed): sum =", fdC,
        "bad =", badC, "nonfinite =", fnfC)
    if fnfC > 0:
        print("10o sess3 nonfinite pos:", cpos[0], cpos[1], cpos[2],
            cpos[3], cpos[4])

    # ---- 步 10p：sess4 历史二分（t1 → gen#1 → gen#2），当场抓镜像 ----
    var sess4 = GpuSession(model, 512)
    var dtok7 = List[Int]()
    dtok7.append(pids[0])

    def chk_mirror(sessx: GpuSession, tag: Int) raises:
        sessx.ctx.synchronize()
        var ndX = sessx.ctx.enqueue_create_host_buffer[DType.float32](896)
        sessx.ctx.enqueue_copy(dst_buf=ndX, src_buf=sessx.d_normedb)
        var xnX = sessx.ctx.enqueue_create_host_buffer[DType.float16](896)
        sessx.ctx.enqueue_copy(dst_buf=xnX, src_buf=sessx.d_xnormb16)
        sessx.ctx.synchronize()
        var bx = 0
        var fdx = Float64(0)
        var fnfx = 0
        for c in range(896):
            var eX = Float64(Float32(xnX[c])) - Float64(
                Float32(Float16(Float32(ndX[c])))
            )
            fdx += abs(eX)
            if abs(eX) > 1e-4:
                bx += 1
            var vX = Float32(xnX[c])
            if (vX != vX) or abs(Float64(vX)) > 60000.0:
                fnfx += 1
        print("10p chk", tag, ": mirror-vs-f16(normedb) sum =", fdx,
            "bad =", bx, "nonfinite =", fnfx)

    _ = sess4.prefill_step[896, 4864](
        dtok7, False, True, False, False, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    chk_mirror(sess4, 1)  # batch=False tick：normedb 是陈旧的，仅看 nonfinite
    var jf6 = List[Int]()
    var jl6 = List[Int]()
    var r4a = sess4.generate_batch[896, 4864](
        pids, [p], 12, -1, jf6, jl6, 0.0, 0, 1.0
    )
    print("10p gen1:", r4a[0], r4a[1], r4a[2])
    chk_mirror(sess4, 2)
    var jf7 = List[Int]()
    var jl7 = List[Int]()
    var r4b = sess4.generate_batch[896, 4864](
        pids, [p], 12, -1, jf7, jl7, 0.0, 0, 1.0
    )
    print("10p gen2:", r4b[0], r4b[1], r4b[2])
    chk_mirror(sess4, 3)
    _ = sess4.prefill_step[896, 4864](
        dtok7, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess4.kv_pos = 0
    _ = sess4.prefill_step[896, 4864](
        dtok7, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess4.ctx.synchronize()
    var ndY = sess4.ctx.enqueue_create_host_buffer[DType.float32](896)
    sess4.ctx.enqueue_copy(dst_buf=ndY, src_buf=sess4.d_normed)
    var xnY = sess4.ctx.enqueue_create_host_buffer[DType.float16](896)
    sess4.ctx.enqueue_copy(dst_buf=xnY, src_buf=sess4.d_xnormb16)
    sess4.ctx.synchronize()
    var by = 0
    var fdy = Float64(0)
    var fnfy = 0
    var ypos = List[Int]()
    for c in range(896):
        var eY = Float64(Float32(xnY[c])) - Float64(
            Float32(Float16(Float32(ndY[c])))
        )
        fdy += abs(eY)
        if abs(eY) > 1e-4:
            by += 1
        var vY = Float32(xnY[c])
        if (vY != vY) or abs(Float64(vY)) > 60000.0:
            fnfy += 1
            if len(ypos) < 16:
                ypos.append(c)
    print("10p chk4 (t1 copy_logits=True after gen2): sum =", fdy,
        "bad =", by, "nonfinite =", fnfy)
    if fnfy > 0:
        print("10p chk4 nonfinite pos:", ypos[0], ypos[1], ypos[2],
            ypos[3], ypos[4], ypos[5], ypos[6], ypos[7], ypos[8],
            ypos[9], ypos[10], ypos[11], ypos[12], ypos[13], ypos[14])

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
    # 二分：3 组参数 × 2 次重跑（判正确性 + 确定性）
    var combos = List[Int]()
    combos.append(1)  # 只关 dual
    combos.append(2)  # 只关 down+spl
    combos.append(3)  # 只关 tile
    for ci in combos:
        for rep in range(2):
            if ci == 1:
                _ = sess.prefill_step[
                    896, 4864, prefill_mma_dual=False
                ](pids, False, True, False, False, 0.0, 0, 1.0,
                  0, 0, 0, False, True)
            elif ci == 2:
                _ = sess.prefill_step[
                    896, 4864, prefill_mma_down=False, prefill_spl=False
                ](pids, False, True, False, False, 0.0, 0, 1.0,
                  0, 0, 0, False, True)
            else:
                _ = sess.prefill_step[
                    896, 4864, prefill_tile=False
                ](pids, False, True, False, False, 0.0, 0, 1.0,
                  0, 0, 0, False, True)
            sess.ctx.synchronize()
            print(
                "combo", ci, "rep", rep, "tokout[10] =",
                Int(sess.h_tokout[10]), "(want 84103)",
            )

    # ---- 干净槽 9 重演上轮复现序列：A'→B'→t1'→FULL ----
    var sq9 = List[Int]()
    sq9.append(9)
    sess.h_posarr[9] = Int32(0)
    _ = sess.spec_step[896, 4864](pids, lens1, sq9, True)
    print("A' tick: next =", Int(sess.h_tokout[10]), "(want 84103)")
    # taps：d_xb 行 0（终残差）+ d_tokenb 输入行
    var hxb = sess.ctx.enqueue_create_host_buffer[DType.float32](
        320 * 896
    )
    sess.ctx.enqueue_copy(dst_buf=hxb, src_buf=sess.d_xb)
    sess.ctx.synchronize()
    var s0 = Float64(0)
    var sn = Float64(0)
    for i in range(896):
        s0 += Float64(hxb[i])
        if Float64(hxb[i]) != 0.0:
            sn += 1.0
    print("A' d_xb row0 sum:", s0, "nonzero:", Int(sn))
    var htk = sess.ctx.enqueue_create_host_buffer[DType.int32](320)
    sess.ctx.enqueue_copy(dst_buf=htk, src_buf=sess.d_tokenb)
    sess.ctx.synchronize()
    print(
        "A' d_tokenb[0..3]:", Int(htk[0]), Int(htk[1]),
        Int(htk[2]), Int(htk[3]),
    )
    # taps：设备端 rowmap/posarr（A' 后应 rowseq=9、rowj=0..10、pos[9]=11）
    var hrs = sess.ctx.enqueue_create_host_buffer[DType.int32](320)
    sess.ctx.enqueue_copy(dst_buf=hrs, src_buf=sess.d_rowseq)
    var hrj = sess.ctx.enqueue_create_host_buffer[DType.int32](320)
    sess.ctx.enqueue_copy(dst_buf=hrj, src_buf=sess.d_rowj)
    var hpa = sess.ctx.enqueue_create_host_buffer[DType.int32](64)
    sess.ctx.enqueue_copy(dst_buf=hpa, src_buf=sess.d_posarr)
    sess.ctx.synchronize()
    print(
        "A' dev rowseq[0..2]:", Int(hrs[0]), Int(hrs[1]), Int(hrs[2]),
        "rowj[0..2]:", Int(hrj[0]), Int(hrj[1]), Int(hrj[2]),
        "posarr[1]:", Int(hpa[1]), "posarr[9]:", Int(hpa[9]),
    )

    var tkb2 = List[Int]()
    tkb2.append(refl[0])
    for i in range(4):
        tkb2.append(refl[1 + i])
    var ln5 = List[Int]()
    ln5.append(5)
    _ = sess.spec_step[896, 4864](tkb2, ln5, sq9)
    print("B' tick: tokout[0] =", Int(sess.h_tokout[0]),
        "(want refl[1]=49014)")

    var tk1b = List[Int]()
    tk1b.append(refl[0])
    sess.h_rowseq[0] = Int32(9)
    sess.h_rowj[0] = Int32(0)
    _ = sess.prefill_step[896, 4864](
        tk1, False, True, False, True, 0.0, 0, 1.0, 0, 0, 0, False, True
    )
    sess.ctx.synchronize()
    print("t1' manual: tokout =", Int(sess.h_tokout[0]),
        "(want 49014)")

    var hl9 = sess.ctx.enqueue_create_host_buffer[DType.float32](
        64 * 151936
    )
    sess.ctx.enqueue_copy(dst_buf=hl9, src_buf=sess.d_logits_b)
    sess.ctx.synchronize()
    print(
        "t1' logits[49014] =", Float64(hl9[49014]),
        "logits[0] =", Float64(hl9[0]),
        "logits[1000] =", Float64(hl9[1000]),
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
