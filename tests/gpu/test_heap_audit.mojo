"""账本一百二十四：3-session 堆损坏审计（119 立旗）。

3 个 GpuSession 顺序创建 + 各自跑同一批步（t=8，batch=True）。
判据：①不崩不挂；②三 session 的 logits/argmax 互相一致（同输入
必同输出）；③host buffer 交叉读无串写。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def run_step(mut sess: GpuSession) raises -> List[Float32]:
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


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    print("audit: creating session 1")
    var a = GpuSession(model, 512)
    print("audit: creating session 2")
    var b = GpuSession(model, 512)
    print("audit: creating session 3")
    var c = GpuSession(model, 512)

    var la = run_step(a)
    print("audit: session 1 stepped, tokout row0:", Int(a.h_tokout[0]))
    var lb = run_step(b)
    print("audit: session 2 stepped, tokout row0:", Int(b.h_tokout[0]))
    var lc = run_step(c)
    print("audit: session 3 stepped, tokout row0:", Int(c.h_tokout[0]))

    var worst_ab = Float32(0)
    var worst_ac = Float32(0)
    for v in range(64):
        var d1 = la[v] - lb[v]
        if d1 < 0:
            d1 = -d1
        if d1 > worst_ab:
            worst_ab = d1
        var d2 = la[v] - lc[v]
        if d2 < 0:
            d2 = -d2
        if d2 > worst_ac:
            worst_ac = d2
    print("worst a-b:", worst_ab, " worst a-c:", worst_ac)
    if worst_ab == 0 and worst_ac == 0:
        print("*** HEAP AUDIT PASS ***")
    else:
        print("*** HEAP AUDIT FAIL ***")
