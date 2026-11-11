"""账本一百二十：tail 数据探针——d_normedb 内容判读。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864
comptime HIDDEN = 896


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)
    var b = GpuSession(model, 512)

    var toks = List[Int]()
    toks.append(100)
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, False,
        True
    ](toks)
    a.ctx.synchronize()
    b.kv_pos = 0
    _ = b.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32
    ](toks)
    b.ctx.synchronize()

    var hna = a.ctx.enqueue_create_host_buffer[DType.float32](4 * HIDDEN)
    var hnb = b.ctx.enqueue_create_host_buffer[DType.float32](HIDDEN)
    a.ctx.enqueue_copy(dst_buf=hna, src_buf=a.d_normedb)
    b.ctx.enqueue_copy(dst_buf=hnb, src_buf=b.d_normed)
    a.ctx.synchronize()
    b.ctx.synchronize()

    print("batch normed row0 [0..8]:")
    for i in range(8):
        print("  ", hna[i])
    print("ref normed row0 [0..8]:")
    for i in range(8):
        print("  ", hnb[i])
    print("batch normed row1 [896..904] (应未写/无关):")
    for i in range(8):
        print("  ", hna[HIDDEN + i])
