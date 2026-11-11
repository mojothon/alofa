"""decode step 分段计时（账本六十四：10.2ms/token 未过 2ms 判据的定位）。

发射成本已排除（bench_launch：2.3µs/次 → 592 次 = 1.4ms host 侧）。本基准
把一步拆成三段各测稳态：整层链（21 次发射，×28 外推）、终 norm+lm_head、
每步 token 路径（pinned alloc + 2×sync + H2D）。谁吃了 8ms 一目了然。
"""

from max.gpu.host import HostBuffer
from std.math import ceildiv
from alofa.core.ffi.posix import monotonic_ns
from alofa.model.arch.qwen import (
    Q4_SLOT_DOWN,
    Q4_SLOT_GATE,
    Q4_SLOT_K,
    Q4_SLOT_O,
    Q4_SLOT_Q,
    Q4_SLOT_UP,
    Q4_SLOT_V,
    QwenForward,
)
from alofa.gpu.session import GpuSession
from alofa.kernels.cuda.attention import attention_par_gpu
from alofa.kernels.cuda.basic import linear_fat_f16_gpu, linear_gpu
from alofa.kernels.cuda.elementwise import copy_gpu, rope_gpu, swiglu_gpu

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
comptime INNER = 896
comptime INNER_FF = 4864
comptime MAXT = 64
comptime WARMUP = 20
comptime ITERS = 200
comptime ROUNDS = 3


def _time(mut session: GpuSession, name: String, iters: Int) raises:
    # 调用方已把要测的发射热身过；这里只做 ITERS 次纯计时的占位 —— 实际
    # 测量在 main 里逐段手写（发射序列各不相同，闭包传不进 comptime）。
    _ = session
    _ = name
    _ = iters


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, MAXT, True)
    var session = GpuSession(model, MAXT)
    var h = model.cfg.hidden
    var q_out = session.q_out
    var kv_dim = session.kv_dim
    var hd = model.cfg.head_dim
    var inter = model.cfg.intermediate
    _ = h

    # 热身：走一遍层 0 全链 + lm_head（编译/缓存就位）。
    _ = session.debug_gather(TOKEN)
    session._norm(0)
    session._mm_bias[INNER](Q4_SLOT_Q, q_out, 0, 0, 0)
    session._mm_bias[INNER](Q4_SLOT_K, kv_dim, 1, 0, 1)
    session._mm_bias[INNER](Q4_SLOT_V, kv_dim, 2, 0, 2)
    session.ctx.enqueue_function[rope_gpu](
        session.d_q_rot.unsafe_ptr(), session.d_q.unsafe_ptr(),
        session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
        Int32(q_out), Int32(hd), Int32(1),
        grid_dim=(ceildiv(q_out // 2, 64), 1, 1), block_dim=(64, 1, 1),
    )
    session.ctx.enqueue_function[rope_gpu](
        session.d_k_rot.unsafe_ptr(), session.d_k.unsafe_ptr(),
        session.d_cos.unsafe_ptr(), session.d_sin.unsafe_ptr(),
        Int32(kv_dim), Int32(hd), Int32(1),
        grid_dim=(ceildiv(kv_dim // 2, 64), 1, 1), block_dim=(64, 1, 1),
    )
    session.ctx.synchronize()
    print("secs-head ok")
