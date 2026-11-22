"""一百七十四-c 第三刀复现探针 H：CPU 参照稳定性仲裁。

probe_186-188 的 CPU 参照自相矛盾：GPU 会话之前 run([100]) 得
logits[145375]=0.0，之后 run 得 -18.7。本探针仲裁：
  r1 = GPU 会话建立**前**的干净 CPU run（独立实例）
  r2 = GPU 会话建立**后**的 CPU run（独立实例）
  逐点比 r1 vs r2 + 打印关键列与 argmax。
若 r1==r2 → CPU 稳定，此前的矛盾出在探针写法；若不同 → CPU 参照
被 GPU 会话初始化污染（多实例/arena 冲突），门里"同实例"模式才是
安全的（test_session_decode 的用法）。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime VOCAB = 151936


def summarize(label: String, r: Pointer[Float32, MutUntrackedOrigin]) raises:
    var mx = Float64(-1e30)
    var mxc = Int(-1)
    var s = Float64(0)
    for i in range(VOCAB):
        var v = Float64(r[unsafe_offset=i])
        s += v
        if v > mx:
            mx = v
            mxc = i
    print(
        label, ": logits[145375] =", r[unsafe_offset=145375],
        " argmax =", mxc, " max =", mx, " sum =", s
    )


def main() raises:
    var m1 = QwenForward(WEIGHTS, CONFIG, 64, True)
    var one = List[Int]()
    one.append(100)
    var r1 = m1.run(one)
    summarize("r1 (GPU 会话前)", r1)

    var s = GpuSession(m1, 512)
    m1.params.keep_alive()

    var m2 = QwenForward(WEIGHTS, CONFIG, 64, True)
    var r2 = m2.run(one)
    summarize("r2 (GPU 会话后)", r2)
    m2.params.keep_alive()

    var worst = Float64(0)
    var wcol = Int(-1)
    for i in range(VOCAB):
        var d = Float64(r1[unsafe_offset=i]) - Float64(r2[unsafe_offset=i])
        if d < 0:
            d = -d
        if d > worst:
            worst = d
            wcol = i
    print("r1-vs-r2 worst =", worst, " col", wcol)
    print("done")
