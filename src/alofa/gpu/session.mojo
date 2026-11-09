"""GpuSession —— GPU 路径的资源宿主（接线设计 `docs/plan/06-device-context.md`）。

设计定稿（账本六十）的落点：device 权重副本、激活缓冲与 `DeviceContext` 不进
`QwenForward`（原型被多线程共享、后端是方法参数），而是由一个平行于模型实例的
会话对象持有 —— 与 runtime 层孪生体同构：**一个 engine 线程一个 session，session
生命周期 ≤ 原型**。`QwenForward` 一行不改，session 只读它的
`params` / `q4_blocks` / `cfg` / `cos` / `sin`。

构造即全量上传（结尾一次同步）：全部 `n_layers × 7` 个 q4 槽位、每层三个
q/k/v 偏置、每层两个 layernorm 权重加终 norm、rope 表（`[max_tokens,
head_dim]`，theta 推导表）、embedding 矩阵（tied 输出投影，fp32）。KV store
在 device 上按层分配（`2*n_layers` 个 `max_tokens × kv_dim` 缓冲），激活缓冲
构造期一次分配、跨步驻留 —— 每 token 只有 token id 进、logits 出，中间零
host 往返。

`decode_step[inner, inner_ff](token)` 是设计 §5 步骤 3 的入口：t=1 的完整
前向，逐层 rmsnorm → q/k/v（带 bias）→ rope → append → attention → o → 残差
→ rmsnorm → gate/up → swiglu → down → 残差 → 终 norm → lm_head，算子序与
`run` 逐步对应。`inner` / `inner_ff` 是 comptime 列数（smem kernel 编译期定
大小），运行期与 `hidden` / `inter` 核对。

⚠️ 经 kernel 指针的 `DeviceBuffer` 参数必须标 `mut`：方法参数默认借用
（mut=False），`unsafe_ptr()` 会给出只读指针，与 kernel 声明的
`Pointer[..., mut=True]` 约束失败（账本六十二）。
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.tensor import TensorView, f32_data
from std.math import ceildiv
from alofa.kernels.cuda.attention import attention_par_gpu
from alofa.kernels.cuda.basic import linear_block_gpu, rmsnorm_gpu
from alofa.kernels.cuda.elementwise import (
    add_into_gpu,
    copy2_gpu,
    rope2_gpu,
    gather_gpu,
    swiglu_gpu,
)
from alofa.kernels.cuda.matmul import q4_matmul_fused_gpu
from alofa.model.arch.qwen import (
    FINAL_NORM,
    Q4_SLOTS_PER_LAYER,
    Q4_SLOT_DOWN,
    Q4_SLOT_GATE,
    Q4_SLOT_K,
    Q4_SLOT_O,
    Q4_SLOT_Q,
    Q4_SLOT_UP,
    Q4_SLOT_V,
    QwenForward,
)



def _upload_f32(
    ctx: DeviceContext, view: TensorView
) raises -> DeviceBuffer[DType.float32]:
    """一个 fp32 TensorView 的 device 副本（小权重路径）。

    模块级函数而不是方法：`__init__` 半初始化状态下调不了 `self` 的方法
    （保守检查要求全字段就绪，账本六十三）；收 `ctx` 参数绕开。
    """
    var n = view.numel()
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    var p = f32_data(view)
    for i in range(n):
        h[i] = p[unsafe_offset=i]
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    return d


def _upload_u8(
    ctx: DeviceContext, mut h: HostBuffer[DType.uint8], n: Int
) raises -> DeviceBuffer[DType.uint8]:
    """host 暂存的前 `n` 字节 → device 新缓冲（同上，模块级）。

    结尾必须同步：调用方（`__init__` 的 q4 循环）**复用同一块 host 暂存**，
    在途拷贝还没读完 host 就回填下一槽 —— 异步拷贝读到的是被覆盖的字节
    （账本六十三：S2 q 差 0.77/79 的根因，经典 pinned-staging 竞态）。
    """
    var d = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()
    return d


struct GpuSession:
    """全模型 decode 的 GPU 资源宿主（t=1）。

    权重与 KV 都是 device 会话私有；激活跨步驻留。单层差分原语
    （`enqueue_rmsnorm` / `enqueue_qproj`）与整步入口（`decode_step`）共用
    同一套私有发射函数。
    """

    var ctx: DeviceContext
    var n_layers: Int
    var hidden: Int
    var q_out: Int
    var inter: Int
    var vocab: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var kv_dim: Int
    var max_tokens: Int
    var eps: Float32
    var kv_pos: Int

    var d_q4: List[DeviceBuffer[DType.uint8]]
    var d_norms: List[DeviceBuffer[DType.float32]]
    var d_biases: List[DeviceBuffer[DType.float32]]
    var d_embed: DeviceBuffer[DType.float32]
    var d_cos: DeviceBuffer[DType.float32]
    var d_sin: DeviceBuffer[DType.float32]
    var d_k_store: List[DeviceBuffer[DType.float32]]
    var d_v_store: List[DeviceBuffer[DType.float32]]

    var d_x: DeviceBuffer[DType.float32]
    var d_normed: DeviceBuffer[DType.float32]
    var d_q: DeviceBuffer[DType.float32]
    var d_q_rot: DeviceBuffer[DType.float32]
    var d_k: DeviceBuffer[DType.float32]
    var d_k_rot: DeviceBuffer[DType.float32]
    var d_v: DeviceBuffer[DType.float32]
    var d_attn: DeviceBuffer[DType.float32]
    var d_proj: DeviceBuffer[DType.float32]
    var d_gate: DeviceBuffer[DType.float32]
    var d_up: DeviceBuffer[DType.float32]
    var d_act: DeviceBuffer[DType.float32]
    var d_ws: DeviceBuffer[DType.float32]
    var d_scores: DeviceBuffer[DType.float32]
    var d_logits: DeviceBuffer[DType.float32]
    var d_token: DeviceBuffer[DType.int32]

    var h_logits: HostBuffer[DType.float32]

    def __init__(out self, model: QwenForward, max_tokens: Int) raises:
        """全量上传并分配激活/KV。要求 `model` 以 `quantize=True` 构造。"""
        if not model.q4_enabled:
            raise AlofaError(
                ERR_UNSUPPORTED,
                "GpuSession reads the q4 path; build the model with quantize=True",
                "",
            )
        self.ctx = DeviceContext()
        self.d_q4 = List[DeviceBuffer[DType.uint8]]()
        self.d_norms = List[DeviceBuffer[DType.float32]]()
        self.d_biases = List[DeviceBuffer[DType.float32]]()
        self.d_k_store = List[DeviceBuffer[DType.float32]]()
        self.d_v_store = List[DeviceBuffer[DType.float32]]()
        self.n_layers = model.cfg.n_layers
        self.hidden = model.cfg.hidden
        self.q_out = model.cfg.n_heads * model.cfg.head_dim
        self.inter = model.cfg.intermediate
        self.vocab = model.cfg.vocab
        self.n_heads = model.cfg.n_heads
        self.n_kv_heads = model.cfg.n_kv_heads
        self.head_dim = model.cfg.head_dim
        self.kv_dim = model.cfg.n_kv_heads * model.cfg.head_dim
        self.max_tokens = max_tokens
        self.eps = model.cfg.eps
        self.kv_pos = 0
        if self.hidden % 32 != 0 or self.inter % 32 != 0:
            raise AlofaError(
                ERR_UNSUPPORTED,
                "q4 block streams need dimensions that tile 32",
                "hidden=" + String(self.hidden) + " inter=" + String(self.inter),
            )

        # ---- 权重：q4 全槽位（复用一块 host 暂存，最大单槽是 down）----
        var max_bytes = self.inter * self.hidden // 32 * 18
        var hq4 = self.ctx.enqueue_create_host_buffer[DType.uint8](max_bytes)
        self.ctx.synchronize()
        for layer in range(self.n_layers):
            for slot in range(Q4_SLOTS_PER_LAYER):
                # 槽位宽度内联（__init__ 半初始化状态调不了 self 的方法：
                # 保守检查要求全字段就绪，账本六十三）。
                var n_out = self.hidden
                var n_in = self.hidden
                if slot == Q4_SLOT_Q:
                    n_out = self.q_out
                if slot == Q4_SLOT_K or slot == Q4_SLOT_V:
                    n_out = self.kv_dim
                if slot == Q4_SLOT_GATE or slot == Q4_SLOT_UP:
                    n_out = self.inter
                # 块数由**整个矩阵** out×in 决定；down 的输入是 inter 不是
                # hidden —— 2026-11-09 的第一版在这里只传了 1/5.4 的字节，
                # down matmul 越界读块流，NaN 一路传进 logits（账本六十三）。
                if slot == Q4_SLOT_DOWN:
                    n_in = self.inter
                var n_bytes = n_out * n_in // 32 * 18
                var blocks = model.q4_blocks[layer * Q4_SLOTS_PER_LAYER + slot]
                for i in range(n_bytes):
                    hq4[i] = blocks[unsafe_offset=i]
                self.d_q4.append(_upload_u8(self.ctx, hq4, n_bytes))

        # ---- fp32 小权重：norms（每层两个 + 终 norm）、q/k/v 偏置 ----
        for layer in range(self.n_layers):
            var p = "model.layers." + String(layer) + "."
            self.d_norms.append(
                _upload_f32(self.ctx, model.params.view(p + "input_layernorm.weight"))
            )
            self.d_norms.append(
                _upload_f32(
                    self.ctx, model.params.view(p + "post_attention_layernorm.weight")
                )
            )
            self.d_biases.append(
                _upload_f32(self.ctx, model.params.view(p + "self_attn.q_proj.bias"))
            )
            self.d_biases.append(
                _upload_f32(self.ctx, model.params.view(p + "self_attn.k_proj.bias"))
            )
            self.d_biases.append(
                _upload_f32(self.ctx, model.params.view(p + "self_attn.v_proj.bias"))
            )
        self.d_norms.append(_upload_f32(self.ctx, model.params.view(FINAL_NORM)))

        # ---- rope 表与 embedding（tied 输出投影）----
        var n_table = max_tokens * self.head_dim
        var hcos = self.ctx.enqueue_create_host_buffer[DType.float32](n_table)
        var hsin = self.ctx.enqueue_create_host_buffer[DType.float32](n_table)
        var pc = f32_data(model.cos)
        var ps = f32_data(model.sin)
        for i in range(n_table):
            hcos[i] = pc[unsafe_offset=i]
            hsin[i] = ps[unsafe_offset=i]
        self.d_cos = self.ctx.enqueue_create_buffer[DType.float32](n_table)
        self.d_sin = self.ctx.enqueue_create_buffer[DType.float32](n_table)
        self.ctx.enqueue_copy(dst_buf=self.d_cos, src_buf=hcos)
        self.ctx.enqueue_copy(dst_buf=self.d_sin, src_buf=hsin)
        var n_embed = self.vocab * self.hidden
        var he = self.ctx.enqueue_create_host_buffer[DType.float32](n_embed)
        var pem = f32_data(model.params.view(model.head))
        for i in range(n_embed):
            he[i] = pem[unsafe_offset=i]
        self.d_embed = self.ctx.enqueue_create_buffer[DType.float32](n_embed)
        self.ctx.enqueue_copy(dst_buf=self.d_embed, src_buf=he)

        # ---- KV store（device 侧，每层 k/v）与激活缓冲 ----
        var n_kv_row = max_tokens * self.kv_dim
        for i in range(self.n_layers):
            self.d_k_store.append(
                self.ctx.enqueue_create_buffer[DType.float32](n_kv_row)
            )
            self.d_v_store.append(
                self.ctx.enqueue_create_buffer[DType.float32](n_kv_row)
            )
        self.d_x = self.ctx.enqueue_create_buffer[DType.float32](self.hidden)
        self.d_normed = self.ctx.enqueue_create_buffer[DType.float32](self.hidden)
        self.d_q = self.ctx.enqueue_create_buffer[DType.float32](self.q_out)
        self.d_q_rot = self.ctx.enqueue_create_buffer[DType.float32](self.q_out)
        self.d_k = self.ctx.enqueue_create_buffer[DType.float32](self.kv_dim)
        self.d_k_rot = self.ctx.enqueue_create_buffer[DType.float32](self.kv_dim)
        self.d_v = self.ctx.enqueue_create_buffer[DType.float32](self.kv_dim)
        self.d_attn = self.ctx.enqueue_create_buffer[DType.float32](self.q_out)
        self.d_proj = self.ctx.enqueue_create_buffer[DType.float32](self.hidden)
        self.d_gate = self.ctx.enqueue_create_buffer[DType.float32](self.inter)
        self.d_up = self.ctx.enqueue_create_buffer[DType.float32](self.inter)
        self.d_act = self.ctx.enqueue_create_buffer[DType.float32](self.inter)
        self.d_ws = self.ctx.enqueue_create_buffer[DType.float32](self.hidden)
        self.d_scores = self.ctx.enqueue_create_buffer[DType.float32](
            self.n_heads * max_tokens
        )
        self.d_logits = self.ctx.enqueue_create_buffer[DType.float32](self.vocab)
        self.d_token = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.h_logits = self.ctx.enqueue_create_host_buffer[DType.float32](
            self.vocab
        )
        self.ctx.synchronize()

    # ---- 私有发射原语（t=1；只收索引，不收 self 派生缓冲参数）----
    #
    # aliasing 注记（账本六十三）：`mut self` 与从 `self` 取出的缓冲同时以
    # mutable 传入会被 aliasing 检查拒绝，所以这些函数的源/目标一律用**选择
    # 子索引**表达，缓冲在函数体内经 `_src`/`_dst` 取（按值返回句柄副本）。

    def _src(self, sel: Int) raises -> DeviceBuffer[DType.float32]:
        """matmul 源选择子：0=normed, 1=attn, 2=act。"""
        if sel == 0:
            return self.d_normed
        if sel == 1:
            return self.d_attn
        if sel == 2:
            return self.d_act
        raise AlofaError(ERR_OUT_OF_RANGE, "unknown src selector", "sel=" + String(sel))

    def _dst(self, sel: Int) raises -> DeviceBuffer[DType.float32]:
        """matmul 目标选择子：0=q, 1=k, 2=v, 3=proj, 4=gate, 5=up。"""
        if sel == 0:
            return self.d_q
        if sel == 1:
            return self.d_k
        if sel == 2:
            return self.d_v
        if sel == 3:
            return self.d_proj
        if sel == 4:
            return self.d_gate
        if sel == 5:
            return self.d_up
        raise AlofaError(ERR_OUT_OF_RANGE, "unknown dst selector", "sel=" + String(sel))

    def _norm(mut self, idx: Int) raises:
        """`d_normed = rmsnorm(d_x, d_norms[idx])`（一行一线程）。"""
        self.ctx.enqueue_function[rmsnorm_gpu](
            self.d_x.unsafe_ptr(),
            self.d_norms[idx].unsafe_ptr(),
            self.d_normed.unsafe_ptr(),
            Int32(self.hidden),
            Int32(1),
            self.eps,
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
        )

    def _residual(mut self) raises:
        """`d_x = d_x + d_proj`（残差）。

        用 `add_into_gpu` 而不是 `add_gpu(dst, a, b)`：dst 与 a 是同一缓冲，
        aliasing 检查拒绝同一对象以两个 mutable 参数进发射（账本六十三）。
        """
        self.ctx.enqueue_function[add_into_gpu](
            self.d_x.unsafe_ptr(),
            self.d_proj.unsafe_ptr(),
            Int32(self.hidden),
            grid_dim=(ceildiv(self.hidden, 64), 1, 1),
            block_dim=(64, 1, 1),
        )

    def _mm[inner: Int](
        mut self, slot: Int, n_out: Int, src_sel: Int, dst_sel: Int
    ) raises:
        """无偏置 q4 matmul（o/gate/up/down），直接写目标缓冲。

        THREADS 按账本五十六的每形状最优：inner ≤ 1024 用 32，否则 128。
        账本六十五起走 fused kernel（HAS_BIAS=False；bias 参数传 dst 占位，
        尾缀不读）。
        """
        var src = self._src(src_sel)
        var dst = self._dst(dst_sel)
        # HAS_BIAS=False 的 bias 参数不读，但 aliasing 检查不允许同一指针
        # 传两次 —— 占位传 d_ws（真缓冲、该实例化不读；同一指针出现两次必被拒，账本六十五）。
        comptime if inner <= 1024:
            self.ctx.enqueue_function[q4_matmul_fused_gpu[32, inner, False]](
                src.unsafe_ptr(),
                self.d_q4[slot].unsafe_ptr(),
                dst.unsafe_ptr(),
                self.d_ws.unsafe_ptr(),
                Int32(inner),
                Int32(n_out),
                grid_dim=(n_out, 1, 1),
                block_dim=(32, 1, 1),
            )
        else:
            self.ctx.enqueue_function[q4_matmul_fused_gpu[128, inner, False]](
                src.unsafe_ptr(),
                self.d_q4[slot].unsafe_ptr(),
                dst.unsafe_ptr(),
                self.d_ws.unsafe_ptr(),
                Int32(inner),
                Int32(n_out),
                grid_dim=(n_out, 1, 1),
                block_dim=(128, 1, 1),
            )

    def _mm_bias[inner: Int](
        mut self, slot: Int, n_out: Int, bias_idx: Int, src_sel: Int, dst_sel: Int
    ) raises:
        """带偏置 q4 matmul（q/k/v）：bias 融合在写回尾缀（账本六十五），
        一次发射直写目标缓冲 —— 原 matmul 进 `d_ws` 再 `add_gpu` 的两步撤销。
        """
        var src = self._src(src_sel)
        var dst = self._dst(dst_sel)
        comptime if inner <= 1024:
            self.ctx.enqueue_function[q4_matmul_fused_gpu[32, inner, True]](
                src.unsafe_ptr(),
                self.d_q4[slot].unsafe_ptr(),
                dst.unsafe_ptr(),
                self.d_biases[bias_idx].unsafe_ptr(),
                Int32(inner),
                Int32(n_out),
                grid_dim=(n_out, 1, 1),
                block_dim=(32, 1, 1),
            )
        else:
            self.ctx.enqueue_function[q4_matmul_fused_gpu[128, inner, True]](
                src.unsafe_ptr(),
                self.d_q4[slot].unsafe_ptr(),
                dst.unsafe_ptr(),
                self.d_biases[bias_idx].unsafe_ptr(),
                Int32(inner),
                Int32(n_out),
                grid_dim=(n_out, 1, 1),
                block_dim=(128, 1, 1),
            )

    # ---- 公开原语（单层差分用；layer 0 / q 槽；收调用方缓冲）----

    def enqueue_rmsnorm(
        mut self,
        mut dx: DeviceBuffer[DType.float32],
        mut dnormed: DeviceBuffer[DType.float32],
    ) raises:
        """`dnormed = rmsnorm(dx, layer0.input_layernorm)`（t=1）。"""
        self.ctx.enqueue_function[rmsnorm_gpu](
            dx.unsafe_ptr(),
            self.d_norms[0].unsafe_ptr(),
            dnormed.unsafe_ptr(),
            Int32(self.hidden),
            Int32(1),
            self.eps,
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
        )

    def enqueue_qproj[inner: Int](
        mut self,
        mut dnormed: DeviceBuffer[DType.float32],
        mut dq: DeviceBuffer[DType.float32],
    ) raises:
        """`dq = q4_matmul(dnormed, layer0.q_slot) + bias`（t=1）。"""
        if Int(inner) != self.hidden:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "comptime inner does not match the model's hidden width",
                "inner=" + String(inner) + " hidden=" + String(self.hidden),
            )
        # 与 decode 同一 fused 路径（账本六十五）：单发射直写 dq。
        self.ctx.enqueue_function[q4_matmul_fused_gpu[32, inner, True]](
            dnormed.unsafe_ptr(),
            self.d_q4[Q4_SLOT_Q].unsafe_ptr(),
            dq.unsafe_ptr(),
            self.d_biases[0].unsafe_ptr(),
            Int32(inner),
            Int32(self.q_out),
            grid_dim=(self.q_out, 1, 1),
            block_dim=(32, 1, 1),
        )

    # ---- 诊断原语（探针用；方法内部发射绕开外部 aliasing）----

    def debug_gather(mut self, token: Int) raises -> HostBuffer[DType.float32]:
        """gather 一步并下载 d_x（探针验证 d_embed 上传与 gather 正确性）。"""
        var ht = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.ctx.synchronize()
        ht[0] = Int32(token)
        self.ctx.enqueue_copy(dst_buf=self.d_token, src_buf=ht)
        self.ctx.enqueue_function[gather_gpu](
            self.d_x.unsafe_ptr(),
            self.d_embed.unsafe_ptr(),
            self.d_token.unsafe_ptr(),
            Int32(self.hidden),
            Int32(1),
            grid_dim=(ceildiv(self.hidden, 64), 1, 1),
            block_dim=(64, 1, 1),
        )
        var hx = self.ctx.enqueue_create_host_buffer[DType.float32](self.hidden)
        self.ctx.enqueue_copy(dst_buf=hx, src_buf=self.d_x)
        self.ctx.synchronize()
        return hx

    def debug_read(mut self, sel: Int, n: Int) raises -> HostBuffer[DType.float32]:
        """按选择子下载激活缓冲（探针用）：0=x 1=normed 2=q 3=k 4=v 5=q_rot
        6=k_rot 7=attn 8=proj 9=act 10=logits。"""
        var d = self.d_x
        if sel == 1:
            d = self.d_normed
        if sel == 2:
            d = self.d_q
        if sel == 3:
            d = self.d_k
        if sel == 4:
            d = self.d_v
        if sel == 5:
            d = self.d_q_rot
        if sel == 6:
            d = self.d_k_rot
        if sel == 7:
            d = self.d_attn
        if sel == 8:
            d = self.d_proj
        if sel == 9:
            d = self.d_act
        if sel == 10:
            d = self.d_logits
        if sel == 11:
            d = self.d_k_store[0]
        if sel == 12:
            d = self.d_v_store[0]
        if sel == 13:
            d = self.d_gate
        if sel == 14:
            d = self.d_up
        var h = self.ctx.enqueue_create_host_buffer[DType.float32](n)
        self.ctx.synchronize()
        self.ctx.enqueue_copy(dst_buf=h, src_buf=d)
        self.ctx.synchronize()
        return h

    # ---- 整步入口 ----

    def decode_step[inner: Int, inner_ff: Int](
        mut self, token: Int
    ) raises -> Pointer[Float32, MutUntrackedOrigin]:
        """t=1 的完整前向：返回 host logits 指针（下次调用前有效）。

        算子序与 `run` 逐步对应；KV 追加在 device 侧 store 上，`kv_pos` 是
        会话内的绝对位置。中间零 host 往返（token id 一次 4 字节上传）。
        """
        if Int(inner) != self.hidden or Int(inner_ff) != self.inter:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "comptime inner values do not match the model's widths",
                "inner="
                + String(inner)
                + " hidden="
                + String(self.hidden)
                + " inner_ff="
                + String(inner_ff)
                + " inter="
                + String(self.inter),
            )
        var pos = self.kv_pos
        var kv_len = pos + 1
        if kv_len > self.max_tokens:
            raise AlofaError(ERR_OUT_OF_RANGE, "kv store is full", "")

        var ht = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.ctx.synchronize()
        ht[0] = Int32(token)
        self.ctx.enqueue_copy(dst_buf=self.d_token, src_buf=ht)
        self.ctx.enqueue_function[gather_gpu](
            self.d_x.unsafe_ptr(),
            self.d_embed.unsafe_ptr(),
            self.d_token.unsafe_ptr(),
            Int32(self.hidden),
            Int32(1),
            grid_dim=(ceildiv(self.hidden, 64), 1, 1),
            block_dim=(64, 1, 1),
        )

        for layer in range(self.n_layers):
            var base = layer * Q4_SLOTS_PER_LAYER
            self._norm(2 * layer)
            self._mm_bias[inner](base + Q4_SLOT_Q, self.q_out, 3 * layer, 0, 0)
            self._mm_bias[inner](base + Q4_SLOT_K, self.kv_dim, 3 * layer + 1, 0, 1)
            self._mm_bias[inner](base + Q4_SLOT_V, self.kv_dim, 3 * layer + 2, 0, 2)
            # rope：表行 = 绝对位置 pos（指针偏移进第 pos 行）。q 转到
            # d_q_rot，k 转到 d_k_rot。
            # rope q+k 合成一次发射（账本六十五）；表行 = 绝对位置 pos。
            var cos_row = self.d_cos.unsafe_ptr() + pos * self.head_dim
            var sin_row = self.d_sin.unsafe_ptr() + pos * self.head_dim
            self.ctx.enqueue_function[rope2_gpu](
                self.d_q_rot.unsafe_ptr(),
                self.d_k_rot.unsafe_ptr(),
                self.d_q.unsafe_ptr(),
                self.d_k.unsafe_ptr(),
                cos_row,
                sin_row,
                Int32(self.q_out),
                Int32(self.kv_dim),
                Int32(self.head_dim),
                Int32(1),
                grid_dim=(ceildiv((self.q_out + self.kv_dim) // 2, 64), 1, 1),
                block_dim=(64, 1, 1),
            )
            # KV append（store 行 = pos）。
            # KV append k/v 合成一次发射（账本六十五）；store 行 = pos。
            var pk_row = self.d_k_store[layer].unsafe_ptr() + pos * self.kv_dim
            var pv_row = self.d_v_store[layer].unsafe_ptr() + pos * self.kv_dim
            self.ctx.enqueue_function[copy2_gpu](
                pk_row,
                pv_row,
                self.d_k_rot.unsafe_ptr(),
                self.d_v.unsafe_ptr(),
                Int32(self.kv_dim),
                grid_dim=(ceildiv(self.kv_dim * 2, 64), 1, 1),
                block_dim=(64, 1, 1),
            )
            # attention（decode 性能版：一 (head,t) 一个 block）。两个 store
            # 指针先落地局部变量 —— 同一 List 的两个下标表达式直接进发射会
            # 被 aliasing 检查当成潜在同源（账本六十三）。
            var pk = self.d_k_store[layer].unsafe_ptr()
            var pv = self.d_v_store[layer].unsafe_ptr()
            self.ctx.enqueue_function[attention_par_gpu[128]](
                self.d_attn.unsafe_ptr(),
                self.d_q_rot.unsafe_ptr(),
                pk,
                pv,
                self.d_scores.unsafe_ptr(),
                Int32(self.n_heads),
                Int32(self.n_kv_heads),
                Int32(self.head_dim),
                Int32(1),
                Int32(kv_len),
                grid_dim=(self.n_heads, 1, 1),
                block_dim=(128, 1, 1),
            )
            self._mm[inner](base + Q4_SLOT_O, self.hidden, 1, 3)
            self._residual()
            self._norm(2 * layer + 1)
            self._mm[inner](base + Q4_SLOT_GATE, self.inter, 0, 4)
            self._mm[inner](base + Q4_SLOT_UP, self.inter, 0, 5)
            self.ctx.enqueue_function[swiglu_gpu](
                self.d_act.unsafe_ptr(),
                self.d_gate.unsafe_ptr(),
                self.d_up.unsafe_ptr(),
                Int32(self.inter),
                grid_dim=(ceildiv(self.inter, 64), 1, 1),
                block_dim=(64, 1, 1),
            )
            self._mm[inner_ff](base + Q4_SLOT_DOWN, self.hidden, 2, 3)
            self._residual()

        self._norm(2 * self.n_layers)
        # lm_head 走块内合并读版（账本六十五：linear_gpu 每 warp 读 32 个
        # 不同 w 行零合并，1.14ms = 3x 内存下限；重组差由门的相对容差覆盖）。
        self.ctx.enqueue_function[linear_block_gpu[256]](
            self.d_normed.unsafe_ptr(),
            self.d_embed.unsafe_ptr(),
            self.d_logits.unsafe_ptr(),
            Int32(self.hidden),
            Int32(self.vocab),
            grid_dim=(self.vocab, 1, 1),
            block_dim=(256, 1, 1),
        )
        self.ctx.enqueue_copy(dst_buf=self.h_logits, src_buf=self.d_logits)
        self.ctx.synchronize()
        self.kv_pos = kv_len
        return self.h_logits.unsafe_ptr()
