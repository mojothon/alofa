"""账本一百二十六：单进程多 session e2e——服务形态压验。

3 个 GpuSession 共享单 ctx（GPU3 空卡，~28.5GB）。判据：
①三 session 各自完整 generate_batch（125 前必崩形态）；
②跨 session 同输入同输出；③多轮确定性（中间穿插其他
session 的工作不扰动输出）；④三 session 轮转大负载跑通
（B=8 × 24 token × 3 轮）。"""

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime MAXNEW = 8


def gen2(mut sess: GpuSession) raises -> List[Int]:
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
    ](flat, lens, MAXNEW, 200000, List[Int](), List[Int]())


def gen_small(mut sess: GpuSession, base: Int) raises -> Int:
    """小负载（4 token prompt × 1 序列 × 6 生成），穿插用。"""
    var flat = List[Int]()
    flat.append(base)
    flat.append(base + 1)
    flat.append(base + 2)
    flat.append(base + 3)
    var lens = List[Int]()
    lens.append(4)
    var out = sess.generate_batch[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32
    ](flat, lens, 6, 200000, List[Int](), List[Int]())
    return out[0]


def gen8(mut sess: GpuSession, f8: List[Int], l8: List[Int], rep: Int) raises -> Int:
    """B=8 × 24 token 大负载；prompt 整体平移 rep 保证链不同。
    返回首序列首 token。"""
    var flat2 = List[Int]()
    for i in range(len(f8)):
        flat2.append(f8[i] + rep)
    var out = sess.generate_batch[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32
    ](flat2, l8, 24, 200000, List[Int](), List[Int]())
    var n = 0
    for i in range(8 * 24):
        if out[i] != -1:
            n += 1
    print("  gen8 useful:", n, "/192 first:", out[0])
    return out[0]


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    print("creating 3 sessions (shared ctx)")
    var a = GpuSession(model, 512)
    var b = GpuSession(model, 512)
    var c = GpuSession(model, 512)
    print("3 sessions ok")

    # ①② 跨 session 同输入同输出
    var ga = gen2(a)
    var gb = gen2(b)
    var gc = gen2(c)
    var mis = 0
    for i in range(2 * MAXNEW):
        if ga[i] != gb[i] or ga[i] != gc[i]:
            mis += 1
    print("cross-session parity mismatches:", mis)
    print("tokens a:", ga[0], ga[1], ga[2], " b:", gb[0], gb[1], gb[2])

    # ③ 多轮确定性：b/c 跑别的负载后 a 复跑
    _ = gen_small(b, 900)
    _ = gen_small(c, 950)
    var ga2 = gen2(a)
    var det = True
    for i in range(2 * MAXNEW):
        if ga[i] != ga2[i]:
            det = False
    print("multi-round deterministic:", det)

    # ④ 大负载轮转：B=8 × 24 token，rep 0/1/2 分别给 a/b/c
    var f8 = List[Int]()
    var l8 = List[Int]()
    var t8 = 1000
    for i in range(8):
        var li = 6 + i
        l8.append(li)
        for _r in range(li):
            f8.append(t8)
            t8 += 1
    print("big load round-robin:")
    _ = gen8(a, f8, l8, 0)
    _ = gen8(b, f8, l8, 100)
    _ = gen8(c, f8, l8, 200)

    if mis == 0 and det:
        print("*** MULTI-SESSION E2E PASS ***")
    else:
        print("*** MULTI-SESSION E2E FAIL ***")
