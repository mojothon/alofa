"""decode_step 链上时间分解 —— 骨架法 + 交错测量（账本七十六）。

七十五确立：本机共享 GPU，跨 run/顺序测量噪声带 ±0.4ms，唯一可信的
对比是同窗口交错。本基准用 `decode_step` 的骨架 skip 掩码（账本七十
六：置位类 kernel 换 1 块 32 线程 noop，发射结构/同步/拷贝原样），
八种模式在同一会话里逐 token 交替调用 —— 每类 kernel 的链上时间 ≈
wall(全) - wall(换掉它)（kernel 时间数据无关，垃圾输入不改变指令数；
noop 顶替只省该类 GPU 工作，launch 结构不变）。decode_step 每步内建
2 次 sync，逐调用 monotonic 计时精确（七十五已证）。

产出：skeleton 模式（全 no-op）= 纯发射/同步/拷贝地板；六类 kernel
的链上时间 —— 七十三收官时"间隙 ~0.6ms"是估算，这里给实测分解。
⚠️ MAXT ≳ 400 会话在 AsyncRT 崩（七十五在案），总步数压在 368 以内。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import (
    GpuSession,
    SKIP_ATTN,
    SKIP_DUAL,
    SKIP_DOWN,
    SKIP_GATHER,
    SKIP_HEAD,
    SKIP_O,
    SKIP_TRI,
)
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime WARMUP = 16
comptime ITERS = 20
comptime ROUNDS = 2
comptime MAXT = WARMUP + 8 * ITERS * ROUNDS + 8


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)

    # 热身：每模式 2 步（JIT 就位；skip=0 与完整路径逐指令相同）。
    for rep in range(2):
        for m in range(8):
            _ = session.decode_step[INNER, INNER_FF](TOKEN, _skip_of(m))
    session.ctx.synchronize()

    for r in range(ROUNDS):
        var totals = InlineArray[Float64, 8](fill=0.0)
        var bests = InlineArray[Float64, 8](fill=1e30)
        for i in range(ITERS):
            for m in range(8):
                var t0 = monotonic_ns()
                _ = session.decode_step[INNER, INNER_FF](TOKEN, _skip_of(m))
                var ms = Float64(monotonic_ns() - t0) / 1e6
                totals[m] += ms
                if ms < bests[m]:
                    bests[m] = ms
        var names = (
            "full,no-tri,no-attn,no-o,no-dual,no-down,no-head,skeleton"
        )
        var parts = names.split(",")
        for m in range(8):
            print(
                "round ",
                r,
                " ",
                parts[m],
                ": mean=",
                totals[m] / Float64(ITERS),
                " best=",
                bests[m],
            )
    model.params.keep_alive()


def _skip_of(m: Int) -> Int:
    """模式 → skip 掩码：0=全真；1-6 = 分别换掉 tri/attn/o/dual/down/
    head；7 = 全 no-op（骨架地板）。"""
    var skip = 0
    if m == 1:
        skip = SKIP_TRI
    elif m == 2:
        skip = SKIP_ATTN
    elif m == 3:
        skip = SKIP_O
    elif m == 4:
        skip = SKIP_DUAL
    elif m == 5:
        skip = SKIP_DOWN
    elif m == 6:
        skip = SKIP_HEAD
    elif m == 7:
        skip = SKIP_TRI + SKIP_ATTN + SKIP_O + SKIP_DUAL + SKIP_DOWN
        skip += SKIP_HEAD + SKIP_GATHER
    return skip
