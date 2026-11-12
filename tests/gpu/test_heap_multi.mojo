"""账本一百二十五：单 DeviceContext 多路复用门——突破 MAX 每
进程 2 个 DeviceContext 的上限（账本一百二十四判决）。

3 个 GpuSession 共享同一 ctx（GPU3 空卡，~28GB）。判据：
①创建不崩（124 中第 3 个 session 必崩）；②三 session 批步结果
逐位一致（同输入必同输出）；③generate_batch 跨 session 确定性
（a 与 b 同输入同输出）。"""

from max.gpu.host import DeviceContext

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def step_row(mut sess: GpuSession) raises -> List[Float32]:
    var toks = List[Int]()
    for i in range(8):
        toks.append(100 + i * 10)
    _ = sess.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    sess.ctx.synchronize()
    var out = List[Float32]()
    for v in range(64):
        out.append(sess.h_logits_b[v])
    return out^


def gen_two(mut sess: GpuSession) raises -> List[Int]:
    var flat = List[Int]()
    flat.append(100)
    flat.append(101)
    flat.append(102)
    flat.append(200)
    flat.append(201)
    flat.append(202)
    flat.append(203)
    var lens = List[Int]()
    lens.append(3)
    lens.append(4)
    return sess.generate_batch[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32
    ](flat, lens, 3, 200000, List[Int](), List[Int]())


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ctx = DeviceContext()
    print("creating s1 (shared ctx)")
    var a = GpuSession(model, 512, ctx)
    print("s1 ok")
    print("creating s2 (shared ctx)")
    var b = GpuSession(model, 512, ctx)
    print("s2 ok")
    print("creating s3 (shared ctx)")
    var c = GpuSession(model, 512, ctx)
    print("s3 ok  ← 124 中必崩点已过")

    var la = step_row(a)
    var lb = step_row(b)
    var lc = step_row(c)
    print("steps ok: a", Int(a.h_tokout[0]),
          " b", Int(b.h_tokout[0]), " c", Int(c.h_tokout[0]))

    var worst = Float32(0)
    for v in range(64):
        var d1 = la[v] - lb[v]
        if d1 < 0:
            d1 = -d1
        var d2 = la[v] - lc[v]
        if d2 < 0:
            d2 = -d2
        if d1 > worst:
            worst = d1
        if d2 > worst:
            worst = d2
    print("worst logits diff across sessions:", worst)

    var ga = gen_two(a)
    var gb = gen_two(b)
    var gmis = 0
    for i in range(2 * 3):
        if ga[i] != gb[i]:
            gmis += 1
    print("generate a:", ga[0], ga[1], ga[2], ga[3], ga[4], ga[5])
    print("generate b:", gb[0], gb[1], gb[2], gb[3], gb[4], gb[5])

    if worst == 0 and gmis == 0:
        print("*** SHARED CTX GATE PASS ***")
    else:
        print("*** SHARED CTX GATE FAIL ***")
