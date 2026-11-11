"""账本一百一十三-b：kv2 双输出 packed MMA（prefill_kv True/False）的
同进程配对 A/B——spl/gqa/tile 两边都开。"""

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

    var s_new = GpuSession(model, 512)
    var s_old = GpuSession(model, 512)
    s_new.kv_pos = 0
    s_new.wp_ready = False
    _ = s_new.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True, True](ids, False, True)
    s_old.kv_pos = 0
    s_old.wp_ready = False
    _ = s_old.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True, True, False](ids, False, True)
    s_new.ctx.synchronize()

    var bnew = 1e30
    var bold = 1e30
    for k in range(4):
        var t0 = monotonic_ns()
        if k % 2 == 0:
            s_new.kv_pos = 0
            _ = s_new.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32, True, True, True, True, True, True, True, True](ids, False, True)
            var d0 = Float64(monotonic_ns() - t0) / 1e6
            s_old.kv_pos = 0
            var t1 = monotonic_ns()
            _ = s_old.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32, True, True, True, True, True, True, True, True, False](
                ids, False, True
            )
            var d1 = Float64(monotonic_ns() - t1) / 1e6
            if d0 < bnew:
                bnew = d0
            if d1 < bold:
                bold = d1
            print("rep", k, " kv-on:", d0, " kv-off:", d1)
        else:
            s_old.kv_pos = 0
            var t1 = monotonic_ns()
            _ = s_old.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32, True, True, True, True, True, True, True, True, False](
                ids, False, True
            )
            var d1 = Float64(monotonic_ns() - t1) / 1e6
            s_new.kv_pos = 0
            var t2 = monotonic_ns()
            _ = s_new.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32, True, True, True, True, True, True, True, True](ids, False, True)
            var d0 = Float64(monotonic_ns() - t2) / 1e6
            if d0 < bnew:
                bnew = d0
            if d1 < bold:
                bold = d1
            print("rep", k, " kv-off:", d1, " kv-on:", d0)
    print("KV-ON min ms:", bnew, " KV-OFF min ms:", bold,
          " delta%:", (bold - bnew) / bold * 100.0)
    model.params.keep_alive()
