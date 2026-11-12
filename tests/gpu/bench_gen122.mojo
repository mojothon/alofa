"""账本一百二十二-a：generate_batch 服务级 e2e bench。

异质 prompt 长度（6-16 随机分布）× 生成 24 token，B=8/16 同 run
交错。有效 token = Σlen + B×GEN；padded lockstep 的浪费 = padding
步占比，从步数与有效 token 数直接算出——给 chunked prefill 定基
线。步数 = lmax + GEN - 1。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime GEN = 24
comptime REPS = 3


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)

    @parameter
    for b in range(1, 33):
        comptime if b == 8 or b == 16:
            # 异质长度：lcm 风格伪随机 6..16
            var lens = List[Int]()
            var flat = List[Int]()
            var total = 0
            for i in range(b):
                var li = 6 + ((i * 7 + 3) % 11)
                lens.append(li)
                for j in range(li):
                    flat.append(1000 + (i * 13 + j * 3) % 900)
                total += li
            var lmax = 0
            for i in range(b):
                if lens[i] > lmax:
                    lmax = lens[i]
            var steps = lmax + GEN - 1
            var useful = total + b * GEN
            var pad = Float64(steps * b - total - b * GEN) / Float64(steps * b)

            var best = 1e30
            var g0 = 0
            for rep in range(REPS):
                var t0 = monotonic_ns()
                var out = a.generate_batch[
                    INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
                    DType.float32
                ](flat, lens, GEN, 200000, List[Int](), List[Int]())
                var ms = Float64(monotonic_ns() - t0) / 1e6
                if ms < best:
                    best = ms
                    g0 = out[0]
            print("B=", b, " total ms:", best,
                  " ms/step:", best / Float64(steps),
                  " useful tok/s:", Float64(useful) / (best * 1e-3),
                  " idle_waste:", pad,
                  " gen0:", g0)
