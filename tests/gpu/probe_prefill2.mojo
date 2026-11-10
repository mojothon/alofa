"""prefill t=2 自检（账本八十八调试）：prefill([42]) vs CPU run([42])。

decode(42) 对同参照 diff≈0.0087（八十三基线）。t=1 prefill 与
decode 算术同构 —— 若此门 diff 落同带，b=0 偏移无辜，问题在
b>0 偏移或多行 grid；若爆，b 版核的基础偏移就错了。
"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession
from alofa.verify.compare import max_abs, max_abs_diff

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime F16_REL = Float64(0.003)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 64, True)
    var session = GpuSession(model, 64)
    var ids = List[Int]()
    ids.append(42)
    ids.append(17)
    var cpu = model.run(ids)
    var gpu = session.prefill_step[INNER, INNER_FF](ids)
    var n = model.cfg.vocab
    var diff = max_abs_diff(gpu, cpu, n)
    var scale = max_abs(cpu, n)
    var best = 0
    var gbest = 0
    for i in range(1, n):
        if cpu[unsafe_offset=i] > cpu[unsafe_offset=best]:
            best = i
        if gpu[unsafe_offset=i] > gpu[unsafe_offset=gbest]:
            gbest = i
    print(
        "t=1 prefill: diff=", diff,
        " tol=", F16_REL * scale,
        " argmax gpu=", gbest, " cpu=", best,
    )
    model.params.keep_alive()
