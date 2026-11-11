"""账本一百二十：tail 探针 v2——精确尺寸拷贝 + 手动 rmsnorm 发射。"""

from alofa.kernels.cuda.basic import rmsnorm_gpu
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime HIDDEN = 896
comptime B = 8


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)
    var b = GpuSession(model, 512)

    var toks = List[Int]()
    toks.append(100)
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    a.ctx.synchronize()
    b.kv_pos = 0
    _ = b.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32
    ](toks)
    b.ctx.synchronize()

    var hxa = a.ctx.enqueue_create_host_buffer[DType.float32](B * HIDDEN)
    var hna = a.ctx.enqueue_create_host_buffer[DType.float32](B * HIDDEN)
    var hnb = b.ctx.enqueue_create_host_buffer[DType.float32](HIDDEN)
    var hman = a.ctx.enqueue_create_host_buffer[DType.float32](B * HIDDEN)
    a.ctx.enqueue_copy(dst_buf=hxa, src_buf=a.d_xb)
    a.ctx.enqueue_copy(dst_buf=hna, src_buf=a.d_normedb)
    b.ctx.enqueue_copy(dst_buf=hnb, src_buf=b.d_normed)
    a.ctx.synchronize()
    b.ctx.synchronize()

    print("a.d_xb[0..4]:", hxa[0], hxa[1], hxa[2], hxa[3], hxa[4])
    print("batch d_normedb[0..4]:", hna[0], hna[1], hna[2], hna[3], hna[4])
    print("ref d_normed[0..4]:", hnb[0], hnb[1], hnb[2], hnb[3], hnb[4])

    print("h_tokout:", Int(a.h_tokout[0]))
