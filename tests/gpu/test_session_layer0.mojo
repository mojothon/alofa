"""GpuSession 骨架对数：单层 rmsnorm → q_proj，GPU vs 产品 CPU 通路（账本六十二）。

参照不是自写的算术，是**产品代码本身**：`scalar.rmsnorm` + `model.project`（run
逐层调用的同一个函数），权重取自真实 fixture（`quantize=True` 的 q4 块流），
输入取真实 embedding 表的第 TOKEN 行。判据：

- normed：**零容差**（rmsnorm 已在门里逐位对齐）；
- q：相对容差 1e-5 × ref_max（q4 matmul 冠军 kernel 在门里 diff=0.0，预期能到零）。

这也是 `06-device-context.md` §5 步骤 2 的验收：GpuSession 构造（1 session 1
ctx）、权重一次性上传、device 侧 rmsnorm→q4 matmul→bias 与 CPU 对数。

Run（A100）:
    mojo run -I src tests/gpu/test_session_layer0.mojo
"""

from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.model.arch.qwen import (
    BACKEND_SCALAR,
    EMBED,
    Q4_SLOT_Q,
    QwenForward,
)
from alofa.gpu.session import GpuSession
from alofa.kernels.cpu.scalar import rmsnorm
from alofa.verify.compare import max_abs, max_abs_diff

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
# smem kernel 的编译期列数 = Qwen 0.5B 的 hidden；session 运行期会核对。
comptime INNER = 896
comptime REL_TOLERANCE = Float64(1e-5)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 64, True)
    var session = GpuSession(model, 0)
    var h = model.cfg.hidden
    var q_out = session.q_out

    # 输入：真实 embedding 表的第 TOKEN 行（fp32 权重的 mmap）。
    var arena = Arena(1 << 18)
    var dims = List[Int]()
    dims.append(1)
    dims.append(h)
    var x = TensorView(arena.alloc(h * 4), Shape(dims), DT_FP32)
    var pe = f32_data(model.params.view(EMBED))
    var px = f32_data(x)
    for j in range(h):
        px[unsafe_offset=j] = pe[unsafe_offset=TOKEN * h + j]

    # CPU 参照：产品 rmsnorm + 产品 project（run 逐层调用的同一个函数）。
    var normed_cpu = TensorView(arena.alloc(h * 4), Shape(dims), DT_FP32)
    rmsnorm(
        normed_cpu,
        x,
        model.params.view("model.layers.0.input_layernorm.weight"),
        model.cfg.eps,
    )
    var q_dims = List[Int]()
    q_dims.append(1)
    q_dims.append(q_out)
    var q_cpu = TensorView(arena.alloc(q_out * 4), Shape(q_dims), DT_FP32)
    model.project[BACKEND_SCALAR](
        q_cpu,
        normed_cpu,
        "model.layers.0.self_attn.q_proj.weight",
        Q4_SLOT_Q,
        model.params.view("model.layers.0.self_attn.q_proj.bias"),
        True,
    )

    # GPU：session 原语。x 上传一次，之后零往返。
    var hx = session.ctx.enqueue_create_host_buffer[DType.float32](h)
    var hnormed = session.ctx.enqueue_create_host_buffer[DType.float32](h)
    var hq = session.ctx.enqueue_create_host_buffer[DType.float32](q_out)
    session.ctx.synchronize()
    for j in range(h):
        hx[j] = px[unsafe_offset=j]
    var dx = session.ctx.enqueue_create_buffer[DType.float32](h)
    var dnormed = session.ctx.enqueue_create_buffer[DType.float32](h)
    var dq = session.ctx.enqueue_create_buffer[DType.float32](q_out)
    session.ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    session.ctx.synchronize()
    session.enqueue_rmsnorm(dx, dnormed)
    session.enqueue_qproj[INNER](dnormed, dq)
    session.ctx.enqueue_copy(dst_buf=hnormed, src_buf=dnormed)
    session.ctx.enqueue_copy(dst_buf=hq, src_buf=dq)
    session.ctx.synchronize()

    var dn = max_abs_diff(hnormed.unsafe_ptr(), f32_data(normed_cpu), h)
    if dn != 0.0:
        raise Error(
            "normed: diff=" + String(dn) + " (zero tolerance vs product rmsnorm)"
        )
    var dqq = max_abs_diff(hq.unsafe_ptr(), f32_data(q_cpu), q_out)
    var scale = max_abs(f32_data(q_cpu), q_out)
    if scale <= 0.0:
        raise Error("q: the reference tensor is all zeros")
    if dqq > REL_TOLERANCE * scale:
        raise Error(
            "q: diff=" + String(dqq) + " tol=" + String(REL_TOLERANCE * scale)
        )
    print(
        "*** SESSION LAYER0 GPU/CPU PARITY PASS *** normed diff= 0.0  q diff=",
        dqq,
        " tol=",
        REL_TOLERANCE * scale,
    )
    model.params.keep_alive()
    arena.keep_alive()
