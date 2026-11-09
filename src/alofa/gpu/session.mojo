"""GpuSession —— GPU 路径的资源宿主（接线设计 `docs/plan/06-device-context.md`）。

设计定稿（账本六十）的落点：device 权重副本、激活缓冲与 `DeviceContext` 不进
`QwenForward`（原型被多线程共享、后端是方法参数），而是由一个平行于模型实例的
会话对象持有 —— 与 runtime 层孪生体同构：**一个 engine 线程一个 session，session
生命周期 ≤ 原型**。

骨架阶段（设计 §5 步骤 2）只做三件事，计算路径在差分测试里驱动：

1. **构造**：自持一个 `DeviceContext`（不共享、不传参 —— "复用"经验规则指的是
   同一进程内不要反复建/弃，跨 session 共享未经评估）。
2. **权重上传（一次性）**：构造时把第 `layer` 层 rmsnorm 权重、`q_proj` 偏置与
   q 槽位的 q4 块流拷上 device，结尾一次 `synchronize`。来源是模型的
   `params.view(...)`（mmap）与 `q4_blocks[slot]`（SharedRegion）—— session 只
   **读**模型，`QwenForward` 一行不改。
3. **计算原语**：`enqueue_rmsnorm` / `enqueue_qproj[inner]` 把门里过了差分的
   kernel 按本层形状拼起来。decode 单 token（t=1）起步；prefill 是后续步骤。

q4 块流的槽位语义与 `project` 一致：全局槽位 = `layer * Q4_SLOTS_PER_LAYER +
slot`；bias 是 fp32、不在 q4 里（`qwen.mojo:1999`）。

⚠️ `enqueue_qproj` 的 `inner` 是 comptime 参数（smem kernel 编译期定大小），
调用方必须传本层 `hidden` 的值；运行期核对，不一致就报错。
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.tensor import f32_data
from std.math import ceildiv
from alofa.kernels.cuda.basic import rmsnorm_gpu
from alofa.kernels.cuda.elementwise import add_gpu
from alofa.kernels.cuda.matmul import q4_matmul_smem_gpu
from alofa.model.arch.qwen import (
    Q4_SLOTS_PER_LAYER,
    Q4_SLOT_Q,
    QwenForward,
)


struct GpuSession:
    """一层的 GPU 资源宿主（骨架：q 投影路）。

    device 权重缓冲构造时一次性上传、此后只读；`d_mm` 是 q 投影的 matmul 工作
    区。激活（x/normed/q）由调用方创建传入 —— 骨架阶段先不接管 act 生命周期。
    """

    var ctx: DeviceContext
    var hidden: Int
    var q_out: Int
    var eps: Float32
    var d_norm_w: DeviceBuffer[DType.float32]
    var d_bias: DeviceBuffer[DType.float32]
    var d_q4: DeviceBuffer[DType.uint8]
    var d_mm: DeviceBuffer[DType.float32]

    def __init__(out self, model: QwenForward, layer: Int) raises:
        """上传第 `layer` 层的 rmsnorm 权重、q_proj 偏置与 q 槽位 q4 块流。

        要求 `model` 已按 `quantize=True` 构造（q4 通路开着），`layer` 在
        `0 .. n_layers-1` 内。上传结束同步一次 —— 此后权重在 device 上可读。
        """
        if layer < 0 or layer >= model.cfg.n_layers:
            raise AlofaError(
                ERR_OUT_OF_RANGE,
                "layer index out of range",
                "layer=" + String(layer) + " n_layers=" + String(model.cfg.n_layers),
            )
        if not model.q4_enabled:
            raise AlofaError(
                ERR_UNSUPPORTED,
                "GpuSession reads the q4 path; build the model with quantize=True",
                "",
            )
        self.ctx = DeviceContext()
        self.eps = model.cfg.eps
        var p = "model.layers." + String(layer) + "."
        var norm_w = model.params.view(p + "input_layernorm.weight")
        var bias = model.params.view(p + "self_attn.q_proj.bias")
        self.hidden = norm_w.numel()
        self.q_out = bias.numel()
        if self.hidden % 32 != 0 or self.hidden * self.q_out % 32 != 0:
            raise AlofaError(
                ERR_UNSUPPORTED,
                "q4 block stream needs dimensions that tile 32",
                "hidden=" + String(self.hidden) + " out=" + String(self.q_out),
            )
        var n_bytes = self.hidden * self.q_out // 32 * 18
        var blocks = model.q4_blocks[layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_Q]

        var hn = self.ctx.enqueue_create_host_buffer[DType.float32](self.hidden)
        var hb = self.ctx.enqueue_create_host_buffer[DType.float32](self.q_out)
        var hq = self.ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
        self.ctx.synchronize()
        var pn = f32_data(norm_w)
        var pb = f32_data(bias)
        for i in range(self.hidden):
            hn[i] = pn[unsafe_offset=i]
        for i in range(self.q_out):
            hb[i] = pb[unsafe_offset=i]
        for i in range(n_bytes):
            hq[i] = blocks[unsafe_offset=i]
        self.d_norm_w = self.ctx.enqueue_create_buffer[DType.float32](self.hidden)
        self.d_bias = self.ctx.enqueue_create_buffer[DType.float32](self.q_out)
        self.d_q4 = self.ctx.enqueue_create_buffer[DType.uint8](n_bytes)
        self.d_mm = self.ctx.enqueue_create_buffer[DType.float32](self.q_out)
        self.ctx.enqueue_copy(dst_buf=self.d_norm_w, src_buf=hn)
        self.ctx.enqueue_copy(dst_buf=self.d_bias, src_buf=hb)
        self.ctx.enqueue_copy(dst_buf=self.d_q4, src_buf=hq)
        self.ctx.synchronize()

    def enqueue_rmsnorm(
        mut self,
        mut dx: DeviceBuffer[DType.float32],
        mut dnormed: DeviceBuffer[DType.float32],
    ) raises:
        """`dnormed = rmsnorm(x, layer.input_layernorm.weight)`（t=1）。

        一行一线程的 kernel：t=1 就一个线程，内部按列循环。
        """
        self.ctx.enqueue_function[rmsnorm_gpu](
            dx.unsafe_ptr(),
            self.d_norm_w.unsafe_ptr(),
            dnormed.unsafe_ptr(),
            Int32(self.hidden),
            Int32(1),
            self.eps,
            grid_dim=(1, 1, 1),
            block_dim=(32, 1, 1),
        )

    def enqueue_qproj[inner: Int](
        mut self,
        mut dnormed: DeviceBuffer[DType.float32],
        mut dq: DeviceBuffer[DType.float32],
    ) raises:
        """`dq = q4_matmul(dnormed, q_slot) + bias`（t=1，两段 kernel）。

        先 matmul 进 `d_mm` 工作区，再 `add_gpu` 加偏置 —— 与 `project` 的
        bias 路同一语义（偏置在矩阵乘之后逐输出元素相加）。`inner` 必须等于
        本层 `hidden`（smem kernel 的编译期列数）。
        """
        if Int(inner) != self.hidden:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "comptime inner does not match the layer's hidden width",
                "inner=" + String(inner) + " hidden=" + String(self.hidden),
            )
        self.ctx.enqueue_function[q4_matmul_smem_gpu[32, inner]](
            dnormed.unsafe_ptr(),
            self.d_q4.unsafe_ptr(),
            self.d_mm.unsafe_ptr(),
            Int32(self.hidden),
            Int32(self.q_out),
            grid_dim=(self.q_out, 1, 1),
            block_dim=(32, 1, 1),
        )
        self.ctx.enqueue_function[add_gpu](
            dq.unsafe_ptr(),
            self.d_mm.unsafe_ptr(),
            self.d_bias.unsafe_ptr(),
            Int32(self.q_out),
            grid_dim=(ceildiv(self.q_out, 64), 1, 1),
            block_dim=(64, 1, 1),
        )
