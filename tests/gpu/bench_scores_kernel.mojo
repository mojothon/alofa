"""scores 循环剥离微测（账本八十一）。

分相归因说 scores ≈ 17.6µs@kv300，但 sq/pf/pair 三个变体全不兑现
—— 假设（q 重读 / 合迹 / 带宽）逐一证伪。把点积循环剥成独立微核
直接测：现状（load[8] 循环）vs 整行预取（4×load[32]，load 全在飞）。
grid=(14) 块（n_heads）复刻真链并行度；W=9 空核出地板（launch+sync）。
"""

from max.gpu.host import HostBuffer

from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from std.gpu import block_idx, thread_idx


def scores_micro[W: Int](
    q: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    scores: Pointer[Float32, MutUntrackedOrigin],
    kv_cols: Int32,
    head_dim: Int32,
    kv_len: Int32,
    kv_head_base: Int32,
) -> None:
    """W=0 load[8] 循环；W=1 整行预取 4×load[32]；W=9 空核。"""
    comptime if W == 9:
        pass
    else:
        var tid = Int(thread_idx.x)
        var j = tid
        var upto = Int(kv_len) - 1
        var row = Int(block_idx.x) * Int(kv_len)
        var hd = Int(head_dim)
        comptime if W == 1:
            var v0 = k_store.load[32](0)
            var v1 = k_store.load[32](0)
            var v2 = k_store.load[32](0)
            var v3 = k_store.load[32](0)
            while j < upto:
                var kbase = j * Int(kv_cols) + Int(kv_head_base)
                v0 = k_store.load[32](kbase)
                v1 = k_store.load[32](kbase + 32)
                v2 = k_store.load[32](kbase + 64)
                v3 = k_store.load[32](kbase + 96)
                var acc = Float64(0)
                @parameter
                for dd in range(32):
                    acc += Float64(q[unsafe_offset=dd]) * Float64(v0[dd])
                @parameter
                for dd in range(32):
                    acc += Float64(q[unsafe_offset=32 + dd]) * Float64(v1[dd])
                @parameter
                for dd in range(32):
                    acc += Float64(q[unsafe_offset=64 + dd]) * Float64(v2[dd])
                @parameter
                for dd in range(32):
                    acc += Float64(q[unsafe_offset=96 + dd]) * Float64(v3[dd])
                scores[unsafe_offset=row + j] = Float32(acc)
                j += 512
        else:
            while j < upto:
                var kbase = j * Int(kv_cols) + Int(kv_head_base)
                var acc = Float64(0)
                for c in range(hd // 8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += Float64(q[unsafe_offset=c * 8 + dd]) * Float64(
                            kv8[dd]
                        )
                scores[unsafe_offset=row + j] = Float32(acc)
                j += 512


def run_w[
    W: Int
](
    mut session: GpuSession,
    name: String,
    kv_len: Int,
    hd: Int,
    iters: Int,
) raises:
    var qcol = session.d_q_rot.unsafe_ptr()
    for r in range(3):
        session.ctx.enqueue_function[scores_micro[W]](
            qcol,
            session.d_k_store[0].unsafe_ptr(),
            session.d_scores.unsafe_ptr(),
            Int32(session.kv_dim),
            Int32(hd),
            Int32(kv_len),
            Int32(0),
            grid_dim=(session.n_heads, 1, 1),
            block_dim=(512, 1, 1),
        )
    session.ctx.synchronize()
    var total = 0.0
    var bestv = 1e30
    for r in range(iters):
        var t0 = monotonic_ns()
        session.ctx.enqueue_function[scores_micro[W]](
            qcol,
            session.d_k_store[0].unsafe_ptr(),
            session.d_scores.unsafe_ptr(),
            Int32(session.kv_dim),
            Int32(hd),
            Int32(kv_len),
            Int32(0),
            grid_dim=(session.n_heads, 1, 1),
            block_dim=(512, 1, 1),
        )
        session.ctx.synchronize()
        var us = Float64(monotonic_ns() - t0) / 1e3
        total += us
        if us < bestv:
            bestv = us
    print(name, ": mean=", total / Float64(iters), "us  best=", bestv, "us")


def main() raises:
    var model = QwenForward(
        "tests/fixtures/qwen2.5-0.5b/weights",
        "tests/fixtures/qwen2.5-0.5b/config.tsv",
        320,
        True,
    )
    var session = GpuSession(model, 320)
    for i in range(300):
        _ = session.decode_step[896, 4864](42)
    session.ctx.synchronize()
    var kv_len = session.kv_pos
    var hd = session.head_dim
    print("kv_len=", kv_len)
    run_w[9](session, "floor      ", kv_len, hd, 150)
    run_w[0](session, "loop l[8]  ", kv_len, hd, 150)
    run_w[1](session, "prefetch   ", kv_len, hd, 150)
    model.params.keep_alive()
