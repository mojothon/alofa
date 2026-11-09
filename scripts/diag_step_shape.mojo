"""量一次前向的墙钟：t=1（单流一步）vs t=8（批一步）vs shards。

判据问题（四十五）：批解码 109 ms/步、8 条线程各 ~52% —— 如果 t=8 的一步
"应该"只比 t=1 贵一点（权重读一次摊 8 行），那多出来的墙钟就是同步气泡；
如果它本来就是 ~2 倍，那空闲的账要另找主。这里不经过服务器，直接量前向。
输出单位 0.1 ms（整数打印，避开小数格式化）。
"""
from std.collections import List
from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward

alias FIXTURE = "tests/fixtures/qwen2.5-0.5b"
alias MAX_TOK = 4096


def bench_step(
    mut model: QwenForward, t: Int, rounds: Int
) raises -> Float64:
    var ids = List[Int]()
    for _ in range(t):
        ids.append(100)
    _ = model.run[BACKEND_AVX2](ids)
    var best = 1.0e18
    for _ in range(3):
        var t0 = monotonic_ns()
        for _ in range(rounds):
            model.reset()
            _ = model.run[BACKEND_AVX2](ids)
        var ms = Float64(monotonic_ns() - t0) / 1e6 / Float64(rounds)
        if ms < best:
            best = ms
    return best


def _row(mut model: QwenForward, shards: Int, t: Int) raises:
    model.reset()
    model.set_shards(shards)
    var ms = bench_step(model, t, 30)
    print(
        "shards=",
        String(shards),
        " t=",
        String(t),
        ": ",
        String(Int(ms * 10)),
        " (0.1ms)/步",
    )
    model.prof.dump()


def main() raises:
    var model1 = QwenForward(
        FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK, True
    )
    model1.enable_q8_output()
    _row(model1, 8, 1)
    var model8 = QwenForward(
        FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK, True
    )
    model8.enable_q8_output()
    _row(model8, 8, 8)
    var model32 = QwenForward(
        FIXTURE + "/weights", FIXTURE + "/config.tsv", MAX_TOK, True
    )
    model32.enable_q8_output()
    _row(model32, 8, 32)
