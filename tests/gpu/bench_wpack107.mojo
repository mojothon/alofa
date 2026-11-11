"""账本一百零七-d：w 打包布局的同进程配对 A/B —— 两个 session 交替
跑（packed / 原布局），每次 kv_pos=0，奇数轮交换顺序以抵消漂移。
取每配置最小值。"""

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

    var s_pk = GpuSession(model, 512)
    var s_raw = GpuSession(model, 512)
    # 预热（含首次 pack）
    s_pk.kv_pos = 0
    _ = s_pk.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32](ids, False, True)
    s_raw.kv_pos = 0
    _ = s_raw.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, False](ids, False, True)
    s_pk.ctx.synchronize()

    var bpk = 1e30
    var braw = 1e30
    for k in range(4):
        if k % 2 == 0:
            s_pk.kv_pos = 0
            var t0 = monotonic_ns()
            _ = s_pk.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32](ids, False, True)
            s_pk.ctx.synchronize()
            var d0 = Float64(monotonic_ns() - t0) / 1e6
            s_raw.kv_pos = 0
            var t1 = monotonic_ns()
            _ = s_raw.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32, True, True, False](ids, False, True)
            s_raw.ctx.synchronize()
            var d1 = Float64(monotonic_ns() - t1) / 1e6
            if d0 < bpk:
                bpk = d0
            if d1 < braw:
                braw = d1
            print("rep", k, " packed:", d0, " raw:", d1)
        else:
            s_raw.kv_pos = 0
            var t1 = monotonic_ns()
            _ = s_raw.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32, True, True, False](ids, False, True)
            s_raw.ctx.synchronize()
            var d1 = Float64(monotonic_ns() - t1) / 1e6
            s_pk.kv_pos = 0
            var t0 = monotonic_ns()
            _ = s_pk.prefill_step[INNER, INNER_FF, 64, 4, 128, 128, 128, 512,
                1, 8, DType.float32](ids, False, True)
            s_pk.ctx.synchronize()
            var d0 = Float64(monotonic_ns() - t0) / 1e6
            if d0 < bpk:
                bpk = d0
            if d1 < braw:
                braw = d1
            print("rep", k, " raw:", d1, " packed:", d0)
    print("PACKED min ms:", bpk, " RAW min ms:", braw,
          " delta%:", (braw - bpk) / braw * 100.0)
    model.params.keep_alive()
