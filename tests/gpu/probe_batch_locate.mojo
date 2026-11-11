"""账本一百二十：批管道定位探针——B=1 batch vs 单序列，下载
d_xb（backbone 残差流）与 d_normedb/d_logits_b（tail）分差。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime HIDDEN = 896
comptime VOCAB = 151936


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

    var hxa = a.ctx.enqueue_create_host_buffer[DType.float32](HIDDEN)
    var hxb = b.ctx.enqueue_create_host_buffer[DType.float32](HIDDEN)
    var hna = a.ctx.enqueue_create_host_buffer[DType.float32](HIDDEN)
    var hnb = b.ctx.enqueue_create_host_buffer[DType.float32](HIDDEN)
    var hla = a.ctx.enqueue_create_host_buffer[DType.float32](8)
    var hlb = b.ctx.enqueue_create_host_buffer[DType.float32](8)
    a.ctx.enqueue_copy(dst_buf=hxa, src_buf=a.d_xb)
    b.ctx.enqueue_copy(dst_buf=hxb, src_buf=b.d_xb)
    a.ctx.enqueue_copy(dst_buf=hna, src_buf=a.d_normedb)
    b.ctx.enqueue_copy(dst_buf=hnb, src_buf=b.d_normed)
    a.ctx.enqueue_copy(dst_buf=hla, src_buf=a.d_logits_b)
    b.ctx.enqueue_copy(dst_buf=hlb, src_buf=b.d_logits)
    a.ctx.synchronize()
    b.ctx.synchronize()

    var dx = 0.0
    for i in range(HIDDEN):
        var d = Float64(abs(hxa[i] - hxb[i]))
        if d > dx:
            dx = d
    var dn = 0.0
    for i in range(HIDDEN):
        var d = Float64(abs(hna[i] - hnb[i]))
        if d > dn:
            dn = d
    var dl = 0.0
    for i in range(8):
        var d = Float64(abs(hla[i] - hlb[i]))
        if d > dl:
            dl = d
    print("d_xb diff:", dx)
    print("d_normed diff:", dn)
    print("logits[0..8] diff:", dl)
    print("batch tokout:", Int(a.h_tokout[0]))
