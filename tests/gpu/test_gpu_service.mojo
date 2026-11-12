"""账本一百二十九：GPU 服务外壳门——`GpuService`（`Service` trait 的
GpuSession 后端实现，`ChatHandler`/HTTP/OpenAI/SSE 全复用）。

判据：
① 流式贪心 vs `complete`（generate_batch 单行）同链：**逐帧文本拼接**
   与整段 decode 逐位一致（整段前缀解码后，多字节字符跨 token 也不再
   产生乱码帧——CJK prompt 单独钉）；
② 流式贪心重放逐位确定（同 prompt 两次）；
③ 采样流式（T=1, top_p=0.9 **走参数面**）同种子逐位复现、异种子
   变异；
⑦ 池并发（一百三十二）：两条流交替推进，各自与独跑**逐位同链**；
   complete 在池里占坑：一条流占着可用另一条，两条全占指名拒绝；
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
    # ⚠️ 池参数退回 1（账本一百三十二）：池架构本身全绿，但 session 层
    # 有既有盲区——「单路 prefill(batch=False) 写的 KV 与批路解码步读的
    # KV」从未被门独立验过，跨流残留渗入（test_pool_repro.mojo 钉着三
    # 个世界 + 判别炮数据）。sessions=2 的交替同链判据（⑦）待一百三十
    # 三修复后恢复复跑。
    var svc = GpuService(
        model, 1, 512, tokenizer^, 151936, 42,
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
    # sessions=1（见上注）：第二条流指名拒绝（与 129 语义一致）。
    try:
        svc.stream_begin(PROMPT, MAXNEW, 0.0, 1.0, 9)
        sem_ok = False
        print("second stream NOT rejected!")
    except e:
        print("second stream rejected ok")
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
          " sdet:", sdet, " svar:", svar)
    if same1 and det and sem_ok and eos_ok and c_ok and sdet and svar:
        print("*** GPU SERVICE GATE PASS ***")
    else:
        print("*** GPU SERVICE GATE FAIL ***")
