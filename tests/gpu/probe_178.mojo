"""指纹实验：K2 rows13-21 毒成 1e6*(j-12)，抓 attnb row0 差分。"""

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

    var snap0 = List[Float64]()
    var snap1 = List[Float64]()

    for rep in range(2):
        for i in range(64):
            sess.h_posarr[i] = Int32(0)
        if rep == 1:
            var kp = sess.ctx.enqueue_create_host_buffer[DType.float32](
                64 * 512 * 128
            )
            sess.ctx.enqueue_copy(dst_buf=kp, src_buf=sess.d_k_store_b[2])
            sess.ctx.synchronize()
            for rr in range(13, 22):
                for c in range(128):
                    kp[rr * 128 + c] = 1.0e30
            sess.ctx.enqueue_copy(dst_buf=sess.d_k_store_b[2], src_buf=kp)
            sess.ctx.synchronize()
            var er = sess.ctx.enqueue_create_host_buffer[DType.float32](
                151936 * 896
            )
            sess.ctx.enqueue_copy(dst_buf=er, src_buf=sess.d_embed)
            sess.ctx.synchronize()
            var hits = 0
            for rr in range(84095, 84115):
                for c in range(896):
                    if abs(er[rr * 896 + c]) > 1.0e20:
                        if hits < 12:
                            print("HOT embed row", rr, "dim", c,
                                "val =", Float64(er[rr * 896 + c]))
                        hits += 1
            print("embed hot total =", hits)
        var cur = Int(-1)
        var g = List[Int]()
        for step in range(p + 11):
            var tk = List[Int]()
            if step < p:
                tk.append(pids[step])
            else:
                tk.append(cur)
            _ = sess.prefill_step[
                896, 4864, 64, 4, 128, 128, 128, 512, 1, 8, DType.float64,
                True, True, True, True, True, True, True, True, False, True
            ](tk, False, True, False, False, 0.0, 0, 1.0)
            sess.ctx.synchronize()
            var to = sess.ctx.enqueue_create_host_buffer[DType.int32](8)
            sess.ctx.enqueue_copy(dst_buf=to, src_buf=sess.d_tokout)
            sess.ctx.synchronize()
            var tv = Int(to[0])
            if step >= p - 1:
                g.append(tv)
                cur = tv
                print("chain", rep, "step", step, ": tok =", tv)
                if step == p + 1:
                    var ab = sess.ctx.enqueue_create_host_buffer[
                        DType.float32
                    ](14 * 512)
                    sess.ctx.enqueue_copy(dst_buf=ab, src_buf=sess.d_scoresb)
                    sess.ctx.synchronize()
                    if rep == 0:
                        for c in range(14 * 512):
                            snap0.append(Float64(ab[c]))
                    else:
                        for c in range(14 * 512):
                            snap1.append(Float64(ab[c]))

    # 陈旧 scores（rep0 step p+1 时 rows13-21 = chain0 残留）
    # 注意：本版 snap 里存的是 attnb；scores 单独重抓太长，
    # 先看 attnb 差分形态
    for hh in range(2):
        var h = hh * 7
        for rr in range(12, 24):
            print("scores h", h, "row", rr, ": clean =", snap0[h * 512 + rr],
                "poison =", snap1[h * 512 + rr])
    print("done")


