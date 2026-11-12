"""账本一百二十四：堆审计二分。A=3 session 纯 H2D 拷贝；B=单序列
prefill 步；C=批步；D=跨 session 混跑。崩点即答案。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime INNER = 896
comptime INNER_FF = 4864


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var a = GpuSession(model, 512)
    var b = GpuSession(model, 512)
    var c = GpuSession(model, 512)
    print("sessions created")

    # A：纯 H2D 拷贝 × 3 session（无核发射）
    a.ctx.enqueue_copy(dst_buf=a.d_posarr, src_buf=a.h_posarr)
    a.ctx.synchronize()
    b.ctx.enqueue_copy(dst_buf=b.d_posarr, src_buf=b.h_posarr)
    b.ctx.synchronize()
    c.ctx.enqueue_copy(dst_buf=c.d_posarr, src_buf=c.h_posarr)
    c.ctx.synchronize()
    print("A pass: copy x3")

    # B：单序列 prefill t=8（batch=False）
    var toks = List[Int]()
    for i in range(8):
        toks.append(100 + i * 10)
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, False
    ](toks)
    a.ctx.synchronize()
    print("B pass: single prefill")

    # C：批步（batch=True）
    _ = a.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    a.ctx.synchronize()
    print("C pass: batch step")

    # D：跨 session 混跑
    _ = b.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    b.ctx.synchronize()
    _ = c.prefill_step[
        INNER, INNER_FF, 64, 4, 128, 128, 128, 512, 1, 8,
        DType.float32, True, True, True, True, True, True, True,
        True, False, True
    ](toks)
    c.ctx.synchronize()
    print("D pass: batch on b and c")
    print("*** HEAP PROBE ALL PASS ***")
