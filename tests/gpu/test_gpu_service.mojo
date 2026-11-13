"""账本一百二十九：GPU 服务外壳门——`GpuService`（`Service` trait 的
GpuSession 后端实现，`ChatHandler`/HTTP/OpenAI/SSE 全复用）。

判据：
① 流式贪心 vs `complete`（generate_batch 单行）同链：**逐帧文本拼接**
   与整段 decode 逐位一致（整段前缀解码后，多字节字符跨 token 也不再
   产生乱码帧——CJK prompt 单独钉）；
② 流式贪心重放逐位确定（同 prompt 两次）；
③ 采样流式（T=1, top_p=0.9 **走参数面**）同种子逐位复现、异种子
   变异；
⑦ 合批并发（一百三十四/一百三十六）：单 session 两批行，两条流
   交替问——tick 一次推进全部活跃行（**任意采样参数组合**，
   per-row 核），各自与独跑**逐位同链**，且**真合批**；流活跃时
   complete 与并发流共存（一百三十七：complete 走 tick 池）；
④ 协议语义：错号 stream_next 指名红；拒第二条流；done 后继续问仍
   done=True；stream_end 幂等；
⑤ eos 终止：把链中一枚冒充 eos，流在第 3 帧收、不吐文本、done 后
   仍 done；complete 同一把刀。
"""

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import DEFAULT_EOS, GpuService
from alofa.tokenizer import Tokenizer, load_tokenizer_json

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime PROMPT = "Once upon a time, in a land far away,"
comptime CJK = "给我讲一个简短的睡前故事。"
comptime MAXNEW = 24


def _sample_run(mut svc: GpuService, rid: Int) raises -> List[Int]:
    """T=1, top_p=0.9（**参数面**）的一条流，收 token 序列。"""
    svc.stream_begin(PROMPT, 16, 1.0, 0.9, rid)
    var s = List[Int]()
    while True:
        var fr = svc.stream_next(rid)
        s.append(svc.streams[0].last)
        if fr.done:
            break
    svc.stream_end(rid)
    return s^


def _replay(mut svc: GpuService, rid: Int) raises -> List[Int]:
    """诊断探针：slot0 独跑 PROMPT 一条，回 token 链。"""
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, rid)
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
    var tokenizer = load_tokenizer_json(TOKJSON)
    # 门侧比对用的第二份（第一份被 ^ 移进 svc，Mojo 移动后不可再用）。
    var tok2 = load_tokenizer_json(TOKJSON)
    # 批行池 = 2（一百三十四：单 session 两批行，tick 合批）：
    # 两条流交替推进各与独跑逐位同链（⑦），第三条指名拒绝。
    var svc = GpuService(
        model, 2, 512, tokenizer^, 151936, 42,
        WEIGHTS, CONFIG, TOKJSON,
    )

    # ---- ① 流式贪心 vs complete（文本可比面）----
    var comp = svc.complete(PROMPT, MAXNEW, 0.0, 1.0)
    print("complete tokens:", comp.completion_tokens)
    print("complete text:", comp.text)

    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 0)
    var chain1 = List[Int]()
    var frames1 = List[String]()
    while True:
        var fr = svc.stream_next(0)
        chain1.append(svc.streams[0].last)
        frames1.append(fr.text)
        if fr.done:
            break
    svc.stream_end(0)
    svc.stream_end(0)  # 幂等
    var stream_text = tok2.decode(chain1)
    # 逐帧拼接必须逐字节等于整段 decode（前缀解码法的全部意义）。
    var joined = String("")
    for i in range(len(frames1)):
        joined = joined + frames1[i]
    var join_ok = joined == stream_text
    if not join_ok:
        # 诊断：首差异字节 + 各自长度 + 帧长序列头几枚。
        var jb = joined.as_bytes()
        var db = stream_text.as_bytes()
        var n = len(jb) if len(jb) < len(db) else len(db)
        var diff_at = -1
        for i in range(n):
            if jb[i] != db[i]:
                diff_at = i
                break
        print("  JOIN DIAG: joined_len", len(jb), "decode_len", len(db),
              "first_diff_at", diff_at)
        var flens = String("")
        for i in range(len(frames1)):
            if i > 6:
                break
            flens = flens + String(len(frames1[i].as_bytes())) + ","
        print("  frame byte lens:", flens)
        print("  svc out len:", len(svc.streams[0].out),
              "chain1 len:", len(chain1))
    var same1 = len(chain1) == comp.completion_tokens and stream_text == comp.text
    print("stream1 len:", len(chain1), "join==decode:", join_ok,
          "text:", stream_text)
    if not (same1 and join_ok):
        print("  PARITY MISMATCH: stream len", len(chain1),
              "vs complete", comp.completion_tokens)

    # ---- ② 重放逐位确定 ----
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 1)
    var chain2 = List[Int]()
    while True:
        var fr = svc.stream_next(1)
        chain2.append(svc.streams[0].last)
        if fr.done:
            break
    svc.stream_end(1)
    var det = len(chain1) == len(chain2)
    if det:
        for i in range(len(chain1)):
            if chain1[i] != chain2[i]:
                det = False
                print("replay diverge at", i)
                break
    print("replay det:", det)

    # ---- ④ 协议语义 ----
    var sem_ok = True
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 7)
    try:
        _ = svc.stream_next(8)
        sem_ok = False
        print("mismatch NOT caught!")
    except e:
        print("mismatch raised ok")
    # 池=2：第二条流落到另一个坑，**成功**；第三条才拒。
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 9)
    print("second stream admitted ok")
    try:
        svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 10)
        sem_ok = False
        print("third stream NOT rejected!")
    except e:
        print("third stream rejected ok")
    svc.stream_end(9)
    svc.stream_end(7)
    svc.stream_end(7)  # 幂等
    # done 后继续问仍 done（max_tokens=3 的短流）
    svc.stream_begin(PROMPT, 3, 0.0, 1.0, 11)
    var tail_done = True
    var n_done_at = -1
    for i in range(6):
        var fr3 = svc.stream_next(11)
        if fr3.done and n_done_at < 0:
            n_done_at = i
        if i >= 3 and not fr3.done:
            tail_done = False
    svc.stream_end(11)
    print("short stream: first done at frame", n_done_at,
          " tail all done:", tail_done)
    if n_done_at != 2 or not tail_done:
        sem_ok = False
    # ---- ⑤ eos 导出层 + 终止：链中一枚冒充 eos ----
    var eos_ok = True
    var eosl = List[Int]()
    eosl.append(svc.eos)
    var eos_text = tok2.decode(eosl)
    var eos_export_ok = svc.eos == 151645 and eos_text.find("<|im_end|>") == 0
    print("eos exported:", svc.eos, "text:", eos_text)
    if not eos_export_ok:
        eos_ok = False
    svc.eos = chain1[2]
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 31)
    var echain = List[Int]()
    var e_done_at = -1
    for i in range(MAXNEW + 2):
        var fre = svc.stream_next(31)
        if fre.done and e_done_at < 0:
            e_done_at = i
        if not fre.done:
            echain.append(svc.streams[0].last)
    var e_text = tok2.decode(echain)
    # 参照前缀（List 切片出 Span，decode 只收 List）。
    var pref = List[Int]()
    for i in range(len(echain)):
        pref.append(chain1[i])
    var pre2 = List[Int]()
    pre2.append(chain1[0])
    pre2.append(chain1[1])
    # 之后再问仍 done
    var fre2 = svc.stream_next(31)
    if not fre2.done:
        eos_ok = False
    svc.stream_end(31)
    var eos_complete = svc.complete(PROMPT, MAXNEW, 0.0, 1.0)
    print("eos stream: done at", e_done_at, "len", len(echain),
          "text prefix ok:", e_text == tok2.decode(pref),
          "| complete tokens:", eos_complete.completion_tokens,
          "(expect", len(echain), ")")
    if e_done_at != 2 or len(echain) != 2:
        eos_ok = False
    var ctok = eos_complete.completion_tokens
    if ctok != len(echain) and ctok != len(echain) + 1:
        eos_ok = False
    if e_text != tok2.decode(pre2):
        eos_ok = False
    svc.eos = DEFAULT_EOS

    # ---- ⑥ CJK 多字节跨 token：逐帧拼接仍逐字节等于整段 decode ----
    svc.stream_begin(CJK, 16, 0.0, 1.0, 41)
    var cjoin = String("")
    var cchain = List[Int]()
    while True:
        var frc = svc.stream_next(41)
        cjoin = cjoin + frc.text
        cchain.append(svc.streams[0].last)
        if frc.done:
            break
    svc.stream_end(41)
    var c_ok = cjoin == tok2.decode(cchain)
    print("cjk stream len:", len(cchain), "join==decode:", c_ok,
          "text:", cjoin)
    if not c_ok:
        sem_ok = False

    var g7b_ok = True
    # ---- ⑦b 帧积压守门（一百四十四）：构造"71 积压 16 帧 + 73 在
    # prompt 相位无帧"的状态，pending(73) 必须为 True；领完 71 清积压
    # 后必须解除；守门不改变任何流的输出内容。
    svc.stream_begin(CJK, 16, 0.0, 1.0, 71)
    var b7_slot71 = svc._slot_of(71)
    # 72 用自己的 next 驱动 tick，71 从不领帧（它的 gen 帧全程堆积）。
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 72)
    var guard_rounds = 0
    while svc.streams[b7_slot71].count < svc.streams[b7_slot71].steps:
        _ = svc.stream_next(72)
        guard_rounds += 1
        if guard_rounds > 400:
            break
    # 71 已走完：16+ 帧积压在队列里，71 的行不再活跃。
    svc.stream_end(72)  # 让出行的行位给 73（门池只有 2 行）
    # 73 begin：prompt 相位、无帧可领、池积压超限 → pending True。
    svc.stream_begin(CJK, 16, 0.0, 1.0, 73)
    var b7_slot73 = svc._slot_of(73)
    var saw_pending = svc.stream_pending(73)
    if not saw_pending:
        print("  gate: backlog guard never fired!")
        g7b_ok = False
    # 领完 71 的积压：守门必须解除。
    while True:
        var fr71 = svc.stream_next(71)
        if fr71.done:
            break
    svc.stream_end(71)
    var pending_after = svc.stream_pending(73)
    if pending_after:
        print("  gate: pending stuck after backlog drained!")
        g7b_ok = False
    # 73 走完，与 74 独跑逐位比（守门只挡"问"，不动内容）。
    var chain73 = List[Int]()
    while True:
        var fr73 = svc.stream_next(73)
        chain73.append(svc.streams[b7_slot73].last)
        if fr73.done:
            break
    svc.stream_end(73)
    svc.stream_end(72)
    svc.stream_begin(CJK, 16, 0.0, 1.0, 74)
    var b7_slot74 = svc._slot_of(74)
    var chain74 = List[Int]()
    while True:
        var fr74 = svc.stream_next(74)
        chain74.append(svc.streams[b7_slot74].last)
        if fr74.done:
            break
    svc.stream_end(74)
    var g_ok = len(chain73) == len(chain74)
    if g_ok:
        for i in range(len(chain73)):
            if chain73[i] != chain74[i]:
                g_ok = False
                print("  gate 73 diverges at", i)
                break
    print("gate 7b backlog-guard: pending-seen", saw_pending,
          "cleared", not pending_after, "bit-same", g_ok,
          "len", len(chain73), len(chain74))
    g7b_ok = g7b_ok and g_ok

    # ---- ⑦ 池并发：两条流交替推进，各自与独跑逐位同链 ----
    # B 独跑（temp0.8）参照：一百三十六起 B 带采样参数合批。
    svc.stream_begin(CJK, 16, 0.8, 1.0, 51)
    var slot51 = svc._slot_of(51)
    var cchain8 = List[Int]()
    while True:
        var frs = svc.stream_next(51)
        cchain8.append(svc.streams[slot51].last)
        if frs.done:
            break
    svc.stream_end(51)
    # 一百三十六：A=temp0（argmax 分支）、B=temp0.8（Gumbel 分支）
    # ——**混合参数同一批**，各与独跑逐位同链。
    svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 61)   # 坑 0
    svc.stream_begin(CJK, 16, 0.8, 1.0, 62)          # 坑 1
    # 一百三十八：真前向口径基线（n_ticks 混空转、n_batched 混
    # padding——n_forwards/n_forward_rows 是 launch 级干净数）。
    var fw0 = svc.session.n_forwards
    var fwr0 = svc.session.n_forward_rows
    var ia = List[Int]()
    var ib = List[Int]()
    var done_a = False
    var done_b = False
    while (not done_a) or (not done_b):
        if not done_a:
            var fra = svc.stream_next(61)
            ia.append(svc.streams[0].last)
            if fra.done:
                done_a = True
        if not done_b:
            var frb = svc.stream_next(62)
            ib.append(svc.streams[1].last)
            if frb.done:
                done_b = True
    svc.stream_end(61)
    svc.stream_end(62)
    var inter_ok = len(ia) == len(chain1) and len(ib) == len(cchain8)
    if inter_ok:
        for i in range(len(ia)):
            if ia[i] != chain1[i]:
                inter_ok = False
                print("interleave A diverge at", i)
                break
    if inter_ok:
        for i in range(len(ib)):
            if ib[i] != cchain8[i]:
                inter_ok = False
                print("interleave B diverge at", i)
                break
    print("interleave: a", len(ia), "b", len(ib), " same-as-solo:", inter_ok)
    var fw = svc.session.n_forwards - fw0
    var fwr = svc.session.n_forward_rows - fwr0
    print("  ticks:", svc.n_ticks, "batched:", svc.n_batched,
          "single:", svc.n_single, "| real forwards:", fw,
          "row-steps:", fwr)
    # 真前向必须显著小于串行总和（两流 prompt 相位 26 步 + 生成相位
    # max(24,16)=24 步 ≈ 34 理论；串行 sum = 40 帧 + 26 prompt 步）。
    if fw >= 40:
        print("  real forwards not batched below serial sum!")
        inter_ok = False
    if svc.n_single != 0:
        # 一百三十六：合批条件已消失——采样流不应再走逐行路径。
        print("  single-stepped forward still exists!")
        inter_ok = False
    # 诊断：交替之后 B 独占重放——== cchain 则行 1 无残留、分歧在
    # 合批前向本身（行间数值泄漏）；!= cchain 则行 1 带残渣。
    svc.stream_begin(CJK, 16, 0.0, 1.0, 81)
    var ib2 = List[Int]()
    var d_b2 = False
    while not d_b2:
        var fr8 = svc.stream_next(81)
        ib2.append(svc.streams[1].last)
        if fr8.done:
            d_b2 = True
    svc.stream_end(81)
    print("  B solo-rerun == cchain:", _same(ib2, cchain),
          " len:", len(ib2), "vs", len(cchain))
    # 判别：同状态再重放一次——两次互异 = 同状态不确定性（device 残留
    # /竞态）；两次相同但 != cchain = cchain 段的状态依赖。
    svc.stream_begin(CJK, 16, 0.0, 1.0, 82)
    var ib3 = List[Int]()
    var d_b3 = False
    while not d_b3:
        var fr9 = svc.stream_next(82)
        ib3.append(svc.streams[0].last)
        if fr9.done:
            d_b3 = True
    svc.stream_end(82)
    print("  B rerun2 == rerun1:", _same(ib3, ib2),
          " == cchain:", _same(ib3, cchain))
    if svc.n_batched == 0:
        # temp=0 的两条流交替必须真合批（一次前向推两行）。
        print("  NO batched tick despite two argmax streams!")
        inter_ok = False

    # complete 走合批路径（一百三十七）：占一行进 tick 池，与并发流
    # **共存**（行满才拒），不再是互斥语义。判据三层：
    #   a) 空闲 complete 正常；
    #   b) 流活跃时 complete **成功**，且文本与独跑逐位同
    #      （temp0.8 种子重放：st.seed 固定、count 从 0 起步）；
    #   c) complete 自驱动期间同一 tick 池里的流 71 与它的独跑参照
    #      逐位同链（complete 的 next 循环推进全池——合批语义）。
    var comp_idle = svc.complete(PROMPT, 8, 0.8, 1.0)
    # 71 流独跑参照（4 帧贪心）。
    svc.stream_begin(PROMPT, 4, 0.0, 1.0, 72)
    var cchain4 = List[Int]()
    while True:
        var fr7 = svc.stream_next(72)
        cchain4.append(svc.streams[svc._slot_of(72)].last)
        if fr7.done:
            break
    svc.stream_end(72)
    # busy complete：流 71 活跃期间 complete 挤入池中。
    svc.stream_begin(PROMPT, 4, 0.0, 1.0, 71)
    var slot71 = svc._slot_of(71)
    var busy_rejected = False
    var comp_busy_text = String("")
    try:
        var comp_busy = svc.complete(PROMPT, 8, 0.8, 1.0)
        comp_busy_text = comp_busy.text
    except e:
        busy_rejected = True
    # complete 自驱动已推完 71 流的帧——领完并对比参照。
    var s71 = List[Int]()
    var d71 = False
    while not d71:
        var fr71 = svc.stream_next(71)
        s71.append(svc.streams[slot71].last)
        if fr71.done:
            d71 = True
    svc.stream_end(71)
    var busy_ok = (not busy_rejected) and comp_busy_text == comp_idle.text
    var s71_ok = _same(s71, cchain4)
    print("complete: idle", comp_idle.completion_tokens, "tokens |",
          "while-stream ok:", busy_ok, "| stream71 chain ok:", s71_ok)
    if not busy_ok or not s71_ok:
        sem_ok = False

    # ---- ③ 采样流式（T=1, top_p=0.9 参数面）：确定 + 异种子变异 ----
    var s1 = _sample_run(svc, 21)
    var s2 = _sample_run(svc, 22)
    var sdet = len(s1) == len(s2)
    if sdet:
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                sdet = False
                print("sample replay diverge at", i)
                break
    svc.seed = 43
    var s3 = _sample_run(svc, 23)
    svc.seed = 42
    var svar = False
    if len(s1) != len(s3):
        svar = True
    else:
        for i in range(len(s1)):
            if s1[i] != s3[i]:
                svar = True
                break
    print("sample det:", sdet, " seed-varies:", svar,
          " lens:", len(s1), len(s2), len(s3))

    print("parity:", same1, " join:", join_ok, " det:", det,
          " sem_ok:", sem_ok, " eos_ok:", eos_ok, " cjk:", c_ok,
          " inter:", inter_ok, " sdet:", sdet, " svar:", svar)
    if (same1 and det and sem_ok and eos_ok and c_ok and inter_ok
            and sdet and svar and g7b_ok):
        print("*** GPU SERVICE GATE PASS ***")
    else:
        print("*** GPU SERVICE GATE FAIL ***")
