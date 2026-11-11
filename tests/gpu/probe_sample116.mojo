"""账本一百一十六-b：sample 快尾探针。

1) argmax 双核正确性：151936 随机 logits（含并列最大值构造），
   device vs host 首现语义。
2) decode A/B：sample=False（608KB D2H 尾）vs sample=True
   （argmax + 4B 尾）——固定底 0.315ms 里 logits D2H + 双 sync
   的可回收份额。交错轮 + kv 回卷同相位。"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.kernels.cuda.basic import argmax_f32_gpu, argmax_comb_gpu
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession
from max.gpu.host import DeviceContext

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime VOCAB = 151936
comptime ITERS = 28
comptime REPS = 4
comptime KREWIND = 96


def main() raises:
    # ---- 1) argmax 核正确性 ----
    var ctx = DeviceContext()
    var hl = ctx.enqueue_create_host_buffer[DType.float32](VOCAB)
    ctx.synchronize()
    var ref_val = Float32(-1e30)
    var ref_idx = 0
    for i in range(VOCAB):
        # 伪随机：多数 -10..10，两处并列最大 42.5（i=5000, i=90000）
        var v = Float32(
            Float64((i * 7919) % 2000 - 1000) / 100.0
        )
        if i == 5000 or i == 90000:
            v = Float32(42.5)
        hl[i] = v
        if v > ref_val:
            ref_val = v
            ref_idx = i
    # 首现语义参考（严格 > 遍历会取 5000）
    ref_idx = 5000
    var dl = ctx.enqueue_create_buffer[DType.float32](VOCAB)
    var dpv = ctx.enqueue_create_buffer[DType.float32](128)
    var dpi = ctx.enqueue_create_buffer[DType.int32](128)
    var dout = ctx.enqueue_create_buffer[DType.int32](1)
    var hout = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=dl, src_buf=hl)
    ctx.synchronize()
    ctx.enqueue_function[argmax_f32_gpu](
        dl.unsafe_ptr(), dpv.unsafe_ptr(), dpi.unsafe_ptr(),
        Int32(VOCAB),
        grid_dim=(128, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[argmax_comb_gpu](
        dpv.unsafe_ptr(), dpi.unsafe_ptr(), dout.unsafe_ptr(),
        Int32(128),
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hout, src_buf=dout)
    ctx.synchronize()
    print("argmax got:", hout[0], " want:", ref_idx,
          " MATCH" if Int(hout[0]) == ref_idx else " MISMATCH")

    # ---- 2) decode A/B ----
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var s = GpuSession(model, 512)
    for i in range(8):
        _ = s.decode_step[INNER, INNER_FF](42, 0)
    s.ctx.synchronize()

    # 采样一致性：同 token 同相位，sample 两路 argmax 应一致
    s.kv_pos = KREWIND
    _ = s.decode_step[INNER, INNER_FF](42, 0)
    var hlog = s.h_logits
    var hv = Float32(-1e30)
    var hi = 0
    for i in range(VOCAB):
        if hlog[i] > hv:
            hv = hlog[i]
            hi = i
    s.kv_pos = KREWIND
    _ = s.decode_step[INNER, INNER_FF](42, 0, True)
    print("sample token:", s.last_token, " host argmax:", hi,
          " MATCH" if s.last_token == hi else " MISMATCH")

    var best = InlineArray[Float64, 3](fill=1e30)
    for rep in range(REPS):
        @parameter
        for v in range(3):
            s.kv_pos = KREWIND
            var t0 = monotonic_ns()
            comptime if v == 0:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 0)
            elif v == 1:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 0, True)
            else:
                for i in range(ITERS):
                    _ = s.decode_step[INNER, INNER_FF](42, 0, False, True)
            s.ctx.synchronize()
            var d = Float64(monotonic_ns() - t0) / 1e6 / Float64(ITERS)
            if d < best[v]:
                best[v] = d
    print("slow-tail min ms:", best[0])
    print("fast-tail min ms:", best[1], "  save:", best[0] - best[1])
    print("span      min ms:", best[2], "  save:", best[0] - best[2])
