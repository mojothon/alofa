"""账本一百零八-b：q2/o packed MMA 的同进程配对 A/B —— 两个 session
交替（全开 vs q/o 标量），每次 kv_pos/wp_ready 重置，奇偶轮换序，
取最小。其余段（dual/down packed）两配置相同，差值即 q/o 收益。"""

from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime P = 128


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ids = List[Int]()
    for i in range(P):
        ids.append(1000 + (i * 37) % 900)

    var s_on = GpuSession(model, 512)
    var s_off = GpuSession(model, 512)
    # 预热（含首次打包）
    s_on.kv_pos = 0
    s_on.wp_ready = False
    _ = s_on.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, False, True)
    s_off.kv_pos = 0
    s_off.wp_ready = False
    _ = s_off.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, False, False](ids, False, True)
    s_on.ctx.synchronize()

    var bon = 1e30
    var boff = 1e30
    for k in range(4):
        if k % 2 == 0:
            s_on.kv_pos = 0
            s_on.wp_ready = False
            var t0 = monotonic_ns()
            _ = s_on.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32](ids, False, True)
            s_on.ctx.synchronize()
            var d0 = Float64(monotonic_ns() - t0) / 1e6
            s_off.kv_pos = 0
            s_off.wp_ready = False
            var t1 = monotonic_ns()
            _ = s_off.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32, True, True, True, False, False](
                ids, False, True
            )
            s_off.ctx.synchronize()
            var d1 = Float64(monotonic_ns() - t1) / 1e6
            if d0 < bon:
                bon = d0
            if d1 < boff:
                boff = d1
            print("rep", k, " qo-on:", d0, " qo-off:", d1)
        else:
            s_off.kv_pos = 0
            s_off.wp_ready = False
            var t1 = monotonic_ns()
            _ = s_off.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32, True, True, True, False, False](
                ids, False, True
            )
            s_off.ctx.synchronize()
            var d1 = Float64(monotonic_ns() - t1) / 1e6
            s_on.kv_pos = 0
            s_on.wp_ready = False
            var t0 = monotonic_ns()
            _ = s_on.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32](ids, False, True)
            s_on.ctx.synchronize()
            var d0 = Float64(monotonic_ns() - t0) / 1e6
            if d0 < bon:
                bon = d0
            if d1 < boff:
                boff = d1
            print("rep", k, " qo-off:", d1, " qo-on:", d0)
    print("QO-ON min ms:", bon, " QO-OFF min ms:", boff,
          " delta%:", (boff - bon) / boff * 100.0)
    model.params.keep_alive()
