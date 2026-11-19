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
from alofa.core.ffi.posix import monotonic_ns
from std.ffi import external_call

from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.tensor import TensorView, f32_data
from std.math import ceildiv
from alofa.kernels.cuda.attention import attn_rope_gpu, attn_rs_gpu, \
    attn_vt_gpu, attn_comb_gpu, attn_vt2_gpu, attn_comb2_gpu, \
    kv_store_b_gpu, attn_rs_b_gpu, attn_rs_h_gpu, attn_vt2_b_gpu, \
    attn_vt2_h_gpu, attn_comb2_b_gpu, kv_store_bb_gpu, \
    attn_rs_h_bb_gpu, attn_vt2_h_bb_gpu
from alofa.kernels.cuda.basic import linear_fat_f16_gpu, linear_fat_gpu, rmsnorm_gpu, argmax_f32_gpu, argmax_comb_gpu, argmax_rb_gpu, sample_rb_gpu, sample_topp_gpu, sample_topp_row_gpu
from alofa.kernels.cuda.elementwise import (
    add_into_gpu,
    f32_to_f16_gpu,
    q4_dequant_f16_gpu,
    gather_gpu,
    noop_gpu,
)
from alofa.kernels.cuda.matmul import (
    f16_matmul_dual_gpu,
    q4_matmul_dual_fat_gpu,
    f16_matmul_fat_gpu, f16_matmul_fat_si_gpu, fat_si_comb_gpu,
    f16_bmm_fat_gpu, f16_bmm_dual_gpu, f16_bmm_qkv_gpu,
    f16_bmm_q2_gpu, f16_bmm_kv2_gpu,
    f16_bmm_fat3_gpu, f16_bmm_dual3_gpu,
    x_to_f16_gpu, f16_mma_down_gpu, f16_mma_dual_gpu,
    f16_mma_down_pk_gpu, f16_mma_down_spl_gpu, f16_mma_dual_pk_gpu,
    f16_mma_gemm_pk_gpu, f16_mma_gemm2_pk_gpu, f32_adds_gpu,
    pack_w_mma_gpu, cat2_f16_gpu, cat2_f32_gpu,
        f16_matmul_dual_nx_gpu,
    f16_matmul_qkv_gpu,
    f16_matmul_qkv_mc_gpu,
    q4_matmul_fat_gpu,
    q4_matmul_fused_gpu,
    q4_matmul_qkv_gpu,
    f16_mma_gemm2_spl_gpu,
    f32_mergb2_gpu, f32_mergb_gpu,
    f16_matmul_qkv_fat_gpu,
    cat3_f16_gpu,
)

# f16 预反量化构建开关（账本六十八二分用）
# decode_step 的骨架测量位掩码（账本七十六）：置位 = 该类 kernel 换成
# noop（1 块 32 线程零工作）。运行时参数、主机分支零成本；skip=0 时
# GPU 序列与原路径逐指令相同。骨架分解 bench（bench_decode_span）用。
comptime SKIP_TRI = 1
comptime SKIP_ATTN = 2
comptime SKIP_O = 4
comptime SKIP_DUAL = 8
comptime SKIP_DOWN = 16
comptime SKIP_HEAD = 32  # 终 norm + lm_head
comptime SKIP_GATHER = 64

comptime F16_BUILD = True

# prefill 批上限（账本八十八）：缓冲按此定宽；超长 prompt 由调用方分块
# （分块间 KV 连续，天然 chunked prefill）。
alias PREFILL_MAX_BATCH = 128
# 一百二十-b：批解码序列上限（store/pos/logits 缓冲按此分配）
alias BMAX = 64
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
    # f16 权重（账本六十八：装载期 q4 -> f16 预反量化；decode 全部 matmul
    # 走 f16 向量读，qproj 留 q4 fused 保严格对数层）。每槽一个独立 List
    # —— 一次发射里每个 origin 只许出现一次（六十六/六十七定稿的规则）。
    var d_w16_q: List[DeviceBuffer[DType.float16]]
    var d_w16_k: List[DeviceBuffer[DType.float16]]
    var d_w16_v: List[DeviceBuffer[DType.float16]]
    var d_w16_o: List[DeviceBuffer[DType.float16]]
    var d_w16_gate: List[DeviceBuffer[DType.float16]]
    var d_w16_up: List[DeviceBuffer[DType.float16]]
    var d_w16_down: List[DeviceBuffer[DType.float16]]
    # 一百零七：gate/up/down 的 mma-fragment 连续打包副本（一次性，
    # 与 d_w16_* 同尺寸）；标量/f64 路径仍读原始布局。
    var d_wp_gate: List[DeviceBuffer[DType.float16]]
    var d_wp_up: List[DeviceBuffer[DType.float16]]
    var d_wp_down: List[DeviceBuffer[DType.float16]]
    var d_wp_q: List[DeviceBuffer[DType.float16]]
    var d_wp_o: List[DeviceBuffer[DType.float16]]
    var wp_ready: Bool
    var qkvp_ready: Bool
    # q/k/v 偏置的分流句柄（同 List 多下标同病同治）
    var d_bias_q: List[DeviceBuffer[DType.float32]]
    var d_bias_k: List[DeviceBuffer[DType.float32]]
    var d_bias_v: List[DeviceBuffer[DType.float32]]
    # 一百一十三-b：k/v 拼接打包（[2*kv_dim, inner] fragment 布局）
    var d_wp_kv: List[DeviceBuffer[DType.float16]]
    # 一百一十五-b：qkv 打包权重（q+2kv 单矩阵 f16，cat3 一次）
    var d_wp_qkv: List[DeviceBuffer[DType.float16]]
    var d_bias_kv: List[DeviceBuffer[DType.float32]]
    # 拼接打包的 scratch（同流顺序复用）
    var d_kvraw: DeviceBuffer[DType.float16]
    # lm_head 权重的 f16 副本（d_embed 本体留给 gather）
    var d_embed16: DeviceBuffer[DType.float16]
    # F16_BUILD=False 时置空句柄（conservative 检查要求全字段初始化）

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
    var d_xnorm: DeviceBuffer[DType.float32]
    var d_ws: DeviceBuffer[DType.float32]
    var d_scores: DeviceBuffer[DType.float32]
    var d_vpart: DeviceBuffer[DType.float32]
    var d_attot: DeviceBuffer[DType.float64]
    var d_dpart: DeviceBuffer[DType.float64]
    var d_xb: DeviceBuffer[DType.float32]
    var d_qb: DeviceBuffer[DType.float32]
    var d_kb: DeviceBuffer[DType.float32]
    var d_vb: DeviceBuffer[DType.float32]
    # 一百一十四-a：kv2 split-K partial（[8][mb*kv_dim]，f32_mergb2 归并）
    var d_kbpart: DeviceBuffer[DType.float32]
    var d_qpart: DeviceBuffer[DType.float32]
    var d_vbpart: DeviceBuffer[DType.float32]
    var d_qrot_b: DeviceBuffer[DType.float32]
    var d_attnb: DeviceBuffer[DType.float32]
    var d_gateb: DeviceBuffer[DType.float32]
    var d_upb: DeviceBuffer[DType.float32]
    var d_actb: DeviceBuffer[DType.float32]
    var d_xnormb: DeviceBuffer[DType.float32]
    var d_xnormb16: DeviceBuffer[DType.float16]
    var d_attnb16: DeviceBuffer[DType.float16]
    var d_actb16: DeviceBuffer[DType.float16]
    # 一百一十二-b：down split-K 的 partial（[8][mb][hidden] f32）
    var d_downpart: DeviceBuffer[DType.float32]
    var d_scoresb: DeviceBuffer[DType.float32]
    var d_vpartb: DeviceBuffer[DType.float32]
    var d_attotb: DeviceBuffer[DType.float64]
    var d_tokenb: DeviceBuffer[DType.int32]
    var d_logits: DeviceBuffer[DType.float32]
    var d_token: DeviceBuffer[DType.int32]

    var h_logits: HostBuffer[DType.float32]
    # 一百一十六-b：sample 快尾（device argmax + 4B 回传）的基建
    var h_token: HostBuffer[DType.int32]
    var d_amax: DeviceBuffer[DType.float32]
    var d_aidx: DeviceBuffer[DType.int32]
    var last_token: Int
    # 一百二十：批解码基建（batch=True 的 prefill；BMAX=8）
    var d_k_store_b: List[DeviceBuffer[DType.float32]]
    var d_v_store_b: List[DeviceBuffer[DType.float32]]
    var d_posarr: DeviceBuffer[DType.int32]
    var h_posarr: HostBuffer[DType.int32]
    var d_normedb: DeviceBuffer[DType.float32]
    var d_logits_b: DeviceBuffer[DType.float32]
    var h_logits_b: HostBuffer[DType.float32]
    var d_tokout: DeviceBuffer[DType.int32]
    var h_tokout: HostBuffer[DType.int32]
    # 一百三十六：per-row 采样参数（批行池合批用）——host 真源，host 写
    # 即时生效，forward 时随整块上传。种子 host 算好直传，与行号解耦。
    var d_stemp: DeviceBuffer[DType.float32]
    var h_stemp: HostBuffer[DType.float32]
    var d_stopps: DeviceBuffer[DType.float32]
    var h_stopps: HostBuffer[DType.float32]
    var d_sseeds: DeviceBuffer[DType.int32]
    var h_sseeds: HostBuffer[DType.int32]
    # 一百三十八：干净吞吐口径——n_ticks 混空转 tick（无活跃行
    # early-return），n_batched 混 padding 行；真前向次数与行·步总数
    # 在 prefill_step 入口计数（batched/单路/服务层全走这里）。
    var n_forwards: Int
    var n_forward_rows: Int
    # 一百五十三：profile 模式的最近一次分段（ms）——调用方开启
    # prefill_step(profile=True) 后可读此字段做池级均值统计。
    var last_prof: InlineArray[Float64, 10]

    def __init__(
        out self,
        model: QwenForward,
        max_tokens: Int,
        # 一百二十五：外传共享 DeviceContext——MAX 运行时每进程
        # DeviceContext 上限为 2（账本一百二十四），多 session 必须同
        # ctx 复用。省略则自建（单 session 场景不变）。
        shared_ctx: Optional[DeviceContext] = Optional[DeviceContext](),
    ) raises:
        """全量上传并分配激活/KV。要求 `model` 以 `quantize=True` 构造。"""
        if not model.q4_enabled:
            raise AlofaError(
                ERR_UNSUPPORTED,
                "GpuSession reads the q4 path; build the model with quantize=True",
                "",
            )
        if shared_ctx:
            self.ctx = shared_ctx.value()
        else:
            self.ctx = DeviceContext()
        self.d_q4 = List[DeviceBuffer[DType.uint8]]()
        # gate/up 单独再存一份（同一个 DeviceBuffer 句柄，零拷贝）：dual-fat
        # 发射要从两个不同 List 取权重指针 —— 同一 List 的两个下标即使提取
        # 局部变量也带着同一 origin，被 aliasing 检查拒（账本六十六）。
        self.d_w16_q = List[DeviceBuffer[DType.float16]]()
        self.d_w16_k = List[DeviceBuffer[DType.float16]]()
        self.d_w16_v = List[DeviceBuffer[DType.float16]]()
        self.d_w16_o = List[DeviceBuffer[DType.float16]]()
        self.d_w16_gate = List[DeviceBuffer[DType.float16]]()
        self.d_w16_up = List[DeviceBuffer[DType.float16]]()
        self.d_w16_down = List[DeviceBuffer[DType.float16]]()
        self.d_wp_gate = List[DeviceBuffer[DType.float16]]()
        self.d_wp_up = List[DeviceBuffer[DType.float16]]()
        self.d_wp_down = List[DeviceBuffer[DType.float16]]()
        self.d_wp_q = List[DeviceBuffer[DType.float16]]()
        self.d_wp_o = List[DeviceBuffer[DType.float16]]()
        # 一百一十三-b：k/v 拼接打包（[2*kv_dim, inner] fragment 布局）
        self.d_wp_kv = List[DeviceBuffer[DType.float16]]()
        self.d_wp_qkv = List[DeviceBuffer[DType.float16]]()
        self.d_bias_kv = List[DeviceBuffer[DType.float32]]()
        self.wp_ready = False
        self.qkvp_ready = False
        comptime if not F16_BUILD:
            self.d_embed16 = self.ctx.enqueue_create_buffer[DType.float16](1)
        self.d_bias_q = List[DeviceBuffer[DType.float32]]()
        self.d_bias_k = List[DeviceBuffer[DType.float32]]()
        self.d_bias_v = List[DeviceBuffer[DType.float32]]()
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

        # ---- 权重：q4 全槽位 + f16 预反量化副本（账本六十八）----
        # 复用 host 暂存（最大单槽是 down）：q4 字节流、反量化 f32、f16。
        var max_bytes = self.inter * self.hidden // 32 * 18
        var hq4 = self.ctx.enqueue_create_host_buffer[DType.uint8](max_bytes)
        self.ctx.synchronize()
        # 一百一十三-b：k/v 拼接打包缓冲 **加载循环前一次性连续分配**。
        # 账本六十三在案：AsyncRT 拷贝队列在「分配与上传交错」过多后
        # 抛 CUDA_ERROR_INVALID_VALUE（host 路径第 15 次 copy）——24 个
        # 中途分配正好推过悬崖，症状为非确定性（门时过时不过、bench
        # 必崩）。连续分配不与上传交错则无事。
        for _pl in range(self.n_layers):
            self.d_wp_kv.append(
                self.ctx.enqueue_create_buffer[DType.float16](
                    2 * self.kv_dim * self.hidden
                )
            )
            self.d_bias_kv.append(
                self.ctx.enqueue_create_buffer[DType.float32](2 * self.kv_dim)
            )
        # 一百二十五：装载三段化——账本六十三悬崖的根治。旧实现每槽
        # alloc+copy+kernel 交错 ~170 次/进程，AsyncRT 拷贝队列深度
        # 累积到第 3 个 session 越崖（CUDA_ERROR_INVALID_VALUE，账本
        # 一百二十四定性：2 个 session 恒好 / 3 个恒坏）。三段化后
        # alloc 与 copy 不再交错：A 全量预分配 → B 纯拷贝（每槽同步，
        # pinned 暂存回填护栏保留）→ C 纯反量化核（一次收尾同步）。
        var q4_nbytes = List[Int]()
        var w16_nelems = List[Int]()
        for _l in range(self.n_layers):
            for _s in range(Q4_SLOTS_PER_LAYER):
                var n_out0 = self.hidden
                var n_in0 = self.hidden
                if _s == Q4_SLOT_Q:
                    n_out0 = self.q_out
                if _s == Q4_SLOT_K or _s == Q4_SLOT_V:
                    n_out0 = self.kv_dim
                if _s == Q4_SLOT_GATE or _s == Q4_SLOT_UP:
                    n_out0 = self.inter
                if _s == Q4_SLOT_DOWN:
                    n_in0 = self.inter
                q4_nbytes.append(n_out0 * n_in0 // 32 * 18)
                w16_nelems.append(n_out0 * n_in0)

        # ---- A 段：全量预分配（零拷贝）----
        var w16_all = List[DeviceBuffer[DType.float16]]()
        for i in range(self.n_layers * Q4_SLOTS_PER_LAYER):
            self.d_q4.append(
                self.ctx.enqueue_create_buffer[DType.uint8](q4_nbytes[i])
            )
            var w16 = self.ctx.enqueue_create_buffer[DType.float16](
                w16_nelems[i]
            )
            w16_all.append(w16)
            var _sl = i % Q4_SLOTS_PER_LAYER
            if _sl == Q4_SLOT_Q:
                self.d_w16_q.append(w16)
                self.d_wp_q.append(
                    self.ctx.enqueue_create_buffer[DType.float16](
                        w16_nelems[i]
                    )
                )
            if _sl == Q4_SLOT_K:
                self.d_w16_k.append(w16)
            if _sl == Q4_SLOT_V:
                self.d_w16_v.append(w16)
            if _sl == Q4_SLOT_O:
                self.d_w16_o.append(w16)
                self.d_wp_o.append(
                    self.ctx.enqueue_create_buffer[DType.float16](
                        w16_nelems[i]
                    )
                )
            if _sl == Q4_SLOT_GATE:
                self.d_w16_gate.append(w16)
                self.d_wp_gate.append(
                    self.ctx.enqueue_create_buffer[DType.float16](
                        w16_nelems[i]
                    )
                )
            if _sl == Q4_SLOT_UP:
                self.d_w16_up.append(w16)
                self.d_wp_up.append(
                    self.ctx.enqueue_create_buffer[DType.float16](
                        w16_nelems[i]
                    )
                )
            if _sl == Q4_SLOT_DOWN:
                self.d_w16_down.append(w16)
                self.d_wp_down.append(
                    self.ctx.enqueue_create_buffer[DType.float16](
                        w16_nelems[i]
                    )
                )
        self.ctx.synchronize()

        # ---- B 段：q4 纯拷贝（无分配；每槽同步——同一块 pinned 暂存
        # 会被回填，在途拷贝读半截字节的竞态护栏保留，账本六十三）----
        for layer in range(self.n_layers):
            for slot in range(Q4_SLOTS_PER_LAYER):
                var i = layer * Q4_SLOTS_PER_LAYER + slot
                var n_bytes = q4_nbytes[i]
                var blocks = model.q4_blocks[i]
                for j in range(n_bytes):
                    hq4[j] = blocks[unsafe_offset=j]
                self.ctx.enqueue_copy(
                    dst_buf=self.d_q4[i], src_buf=hq4
                )
                self.ctx.synchronize()

        # ---- f16 预反量化（账本六十八）：C 段纯核——GPU 侧直读显存
        # d_q4 反量化成 f16，host 不再中转（数值判据见门 F16_REL）。
        comptime if F16_BUILD:
            for i in range(self.n_layers * Q4_SLOTS_PER_LAYER):
                self.ctx.enqueue_function[q4_dequant_f16_gpu](
                    self.d_q4[i].unsafe_ptr(),
                    w16_all[i].unsafe_ptr(),
                    Int32(w16_nelems[i]),
                    grid_dim=(ceildiv(w16_nelems[i], 256), 1, 1),
                    block_dim=(256, 1, 1),
                )
            self.ctx.synchronize()

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
            var bq = _upload_f32(self.ctx, model.params.view(p + "self_attn.q_proj.bias"))
            var bk = _upload_f32(self.ctx, model.params.view(p + "self_attn.k_proj.bias"))
            var bv = _upload_f32(self.ctx, model.params.view(p + "self_attn.v_proj.bias"))
            self.d_biases.append(bq)
            self.d_biases.append(bk)
            self.d_biases.append(bv)
            self.d_bias_q.append(bq)
            self.d_bias_k.append(bk)
            self.d_bias_v.append(bv)
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
        self.ctx.synchronize()
        # lm_head 的 f16 副本（账本六十八）：字节流量减半；d_embed 本体
        # 留给 gather（f32 精确嵌入）。
        comptime if F16_BUILD:
            # f32 -> f16 GPU 侧转换（账本六十八，同上绕开 host 路径）
            self.d_embed16 = self.ctx.enqueue_create_buffer[DType.float16](n_embed)
            self.ctx.enqueue_function[f32_to_f16_gpu](
                self.d_embed.unsafe_ptr(),
                self.d_embed16.unsafe_ptr(),
                Int32(n_embed),
                grid_dim=(ceildiv(n_embed, 256), 1, 1),
                block_dim=(256, 1, 1),
            )
            self.ctx.synchronize()

        # ---- KV store（device 侧，每层 k/v）与激活缓冲 ----
        var n_kv_row = max_tokens * self.kv_dim
        for i in range(self.n_layers):
            self.d_k_store.append(
                self.ctx.enqueue_create_buffer[DType.float32](n_kv_row)
            )
            self.d_v_store.append(
                self.ctx.enqueue_create_buffer[DType.float32](n_kv_row)
            )
        for i in range(self.n_layers):
            self.d_wp_qkv.append(
                self.ctx.enqueue_create_buffer[DType.float16](
                    (self.q_out + 2 * self.kv_dim) * self.hidden
                )
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
        self.d_xnorm = self.ctx.enqueue_create_buffer[DType.float32](self.hidden)
        self.d_ws = self.ctx.enqueue_create_buffer[DType.float32](self.hidden)
        self.d_scores = self.ctx.enqueue_create_buffer[DType.float32](
            self.n_heads * max_tokens
        )
        # 跨块 j 分块三趟（账本八十二）：V 部分和（容量按 TILES=8 预留）
        # 与每头 gsum。
        self.d_vpart = self.ctx.enqueue_create_buffer[DType.float32](
            self.n_heads * 8 * self.head_dim
        )
        # n_heads * 8（K2' tsum[head*TILES+tile]，TILES≤8；八十六修正：
        # 八十三起误配 n_heads，越界 784B 被分配器 slack 掩护未爆）
        self.d_attot = self.ctx.enqueue_create_buffer[DType.float64](
            self.n_heads * 8
        )
        # split-inner 部分和（hidden × 8 f64 上限；八十七）
        self.d_dpart = self.ctx.enqueue_create_buffer[DType.float64](
            self.hidden * 8
        )
        # prefill 批缓冲（账本八十八，PREFILL_MAX_BATCH 行）
        var mb = PREFILL_MAX_BATCH
        self.d_xb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.hidden
        )
        self.d_qb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.q_out
        )
        # 一百一十三-b：k/v 拼接打包 scratch（[2*kv_dim, inter] f16）
        self.d_kvraw = self.ctx.enqueue_create_buffer[DType.float16](
            2 * self.kv_dim * self.inter
        )
        self.d_kb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.kv_dim
        )
        self.d_vb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.kv_dim
        )
        # 一百一十四-a：kv2 split-K partial，S=8（与发射/归并同源）
        self.d_kbpart = self.ctx.enqueue_create_buffer[DType.float32](
            8 * mb * self.kv_dim
        )
        # 一百一十七-b：q2 split-K partial（[8][mb][q_out]）
        self.d_qpart = self.ctx.enqueue_create_buffer[DType.float32](
            8 * mb * self.q_out
        )
        self.d_vbpart = self.ctx.enqueue_create_buffer[DType.float32](
            8 * mb * self.kv_dim
        )
        self.d_qrot_b = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.q_out
        )
        self.d_attnb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.q_out
        )
        self.d_gateb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.inter
        )
        self.d_upb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.inter
        )
        self.d_actb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.inter
        )
        self.d_xnormb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.hidden
        )
        self.d_attnb16 = self.ctx.enqueue_create_buffer[DType.float16](
            mb * self.hidden
        )
        self.d_xnormb16 = self.ctx.enqueue_create_buffer[DType.float16](
            mb * self.hidden
        )
        self.d_actb16 = self.ctx.enqueue_create_buffer[DType.float16](
            mb * self.inter
        )
        # 一百一十二-b：down split-K 的 partial（[8][mb][hidden] f32）
        self.d_downpart = self.ctx.enqueue_create_buffer[DType.float32](
            8 * mb * self.hidden
        )
        self.d_scoresb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.n_heads * self.max_tokens
        )
        self.d_vpartb = self.ctx.enqueue_create_buffer[DType.float32](
            mb * self.n_heads * 8 * self.head_dim
        )
        self.d_attotb = self.ctx.enqueue_create_buffer[DType.float64](
            mb * self.n_heads * 8
        )
        self.d_tokenb = self.ctx.enqueue_create_buffer[DType.int32](mb)
        self.d_logits = self.ctx.enqueue_create_buffer[DType.float32](self.vocab)
        self.d_token = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.h_token = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.d_amax = self.ctx.enqueue_create_buffer[DType.float32](128)
        self.d_aidx = self.ctx.enqueue_create_buffer[DType.int32](128)
        self.last_token = 0
        self.h_logits = self.ctx.enqueue_create_host_buffer[DType.float32](
            self.vocab
        )
        # 一百二十：批解码分配（与上方同属纯分配区，不与拷贝交错——
        # 账本六十三的 AsyncRT 悬崖）。BMAX=32；store 布局
        # [B][max_tokens][kv_dim]，比单序列 store 多 32×（~460MB）。
        self.d_k_store_b = List[DeviceBuffer[DType.float32]]()
        self.d_v_store_b = List[DeviceBuffer[DType.float32]]()
        for _bl in range(self.n_layers):
            self.d_k_store_b.append(
                self.ctx.enqueue_create_buffer[DType.float32](
                    BMAX * self.max_tokens * self.kv_dim
                )
            )
            self.d_v_store_b.append(
                self.ctx.enqueue_create_buffer[DType.float32](
                    BMAX * self.max_tokens * self.kv_dim
                )
            )
        self.d_posarr = self.ctx.enqueue_create_buffer[DType.int32](BMAX)
        self.h_posarr = self.ctx.enqueue_create_host_buffer[DType.int32](BMAX)
        self.d_normedb = self.ctx.enqueue_create_buffer[DType.float32](
            BMAX * self.hidden
        )
        self.d_logits_b = self.ctx.enqueue_create_buffer[DType.float32](
            BMAX * self.vocab
        )
        self.h_logits_b = self.ctx.enqueue_create_host_buffer[DType.float32](
            BMAX * self.vocab
        )
        self.d_tokout = self.ctx.enqueue_create_buffer[DType.int32](BMAX)
        self.h_tokout = self.ctx.enqueue_create_host_buffer[DType.int32](BMAX)
        self.d_stemp = self.ctx.enqueue_create_buffer[DType.float32](BMAX)
        self.h_stemp = self.ctx.enqueue_create_host_buffer[DType.float32](BMAX)
        self.d_stopps = self.ctx.enqueue_create_buffer[DType.float32](BMAX)
        self.h_stopps = self.ctx.enqueue_create_host_buffer[DType.float32](BMAX)
        self.d_sseeds = self.ctx.enqueue_create_buffer[DType.int32](BMAX)
        self.h_sseeds = self.ctx.enqueue_create_host_buffer[DType.int32](BMAX)
        self.n_forwards = 0
        self.last_prof = InlineArray[Float64, 10](fill=Float64(0))
        self.n_forward_rows = 0
        self.ctx.synchronize()
        for i in range(BMAX):
            self.h_posarr[i] = Int32(0)
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

    def generate_batch[
        inner: Int, inner_ff: Int, dual_t: Int = 64, dual_o: Int = 4,
        tri_t: Int = 128, down_t: Int = 128, o_t: Int = 128,
        attn_t: Int = 512, attn_tiles: Int = 1,
        prefill_rb: Int = 8,
        prefill_af: DType = DType.float64,
        prefill_mma_dual: Bool = True,
        prefill_mma_down: Bool = True,
        prefill_wpack: Bool = True,
        prefill_mma_q: Bool = True,
        prefill_mma_o: Bool = True,
        prefill_tile: Bool = True,
        prefill_gqa: Bool = True,
        prefill_spl: Bool = True,
        prefill_kv: Bool = False,
    ](
        mut self,
        flat_prompts: List[Int],
        lens: List[Int],
        max_new: Int,
        eos: Int,
        join_flat: List[Int],
        join_lens: List[Int],
        temperature: Float32 = 0.0,
        seed: Int32 = 0,
        top_p: Float32 = 1.0,
    ) raises -> List[Int]:
        """连续批处理 v2（账本一百二十二-b）：逐行 token 流 +
        完成槽位回收。

        每行独立推进 prompt→生成流：喂 prompt 剩余 token（耗尽步
        的 tokout 即 gen[0]），此后链式生成；EOS 或预算用尽即完成
        ——join 队列非空时槽位立即让给新序列（h_posarr 归零；新
        序列按 pos 顺序覆写旧行，attention 只读 [0..pos] 故无污
        染——回收零 KV 搬运）。全程无 padding 行。步数 =
        max_i(plen_i + 生成数 + 接续链)。

        返回扁平 [B+jb][max_new]：初始行 i 在 [i]，第 jt 个 joiner
        在 [B+jt]；未接入的 joiner 行保持 -1。eos=-1 禁用早停。
        确定性：同输入同输出。
        """
        var b = len(lens)
        var jb = len(join_lens)
        if b > BMAX:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "generate_batch cap is 64",
                "b=" + String(b),
            )
        var poff = List[Int]()
        var plen = List[Int]()
        var pidx = List[Int]()
        var psrc = List[Int]()
        var genc = List[Int]()
        var cur = List[Int]()
        var fin = List[Bool]()
        var obase = List[Int]()
        var total = 0
        for i in range(b):
            poff.append(total)
            total += lens[i]
            plen.append(lens[i])
            pidx.append(0)
            psrc.append(0)
            genc.append(0)
            cur.append(0)
            fin.append(False)
            obase.append(i * max_new)
        var joff = List[Int]()
        var jtotal = 0
        for i in range(jb):
            joff.append(jtotal)
            jtotal += join_lens[i]
        var out = List[Int]()
        for i in range((b + jb) * max_new):
            out.append(-1)
        for i in range(BMAX):
            self.h_posarr[i] = Int32(0)

        var jt = 0
        var nfin = 0
        while nfin < b:
            # 1) 逐行取下一 token（prompt 优先，否则生成链上一环）
            var toks = List[Int]()
            var fp = List[Bool]()
            for i in range(b):
                if (not fin[i]) and pidx[i] < plen[i]:
                    if psrc[i] == 0:
                        toks.append(flat_prompts[poff[i] + pidx[i]])
                    else:
                        toks.append(join_flat[poff[i] + pidx[i]])
                    pidx[i] += 1
                    fp.append(True)
                else:
                    toks.append(cur[i])
                    fp.append(False)
            # 2) 批步
            _ = self.prefill_step[
                inner, inner_ff, dual_t, dual_o, tri_t, down_t, o_t,
                attn_t, attn_tiles, prefill_rb, prefill_af,
                prefill_mma_dual, prefill_mma_down, prefill_wpack,
                prefill_mma_q, prefill_mma_o, prefill_tile, prefill_gqa,
                prefill_spl, prefill_kv, True
            ](toks, False, False, False, temperature, seed, top_p)
            self.ctx.synchronize()
            # 3) 逐行消化 tokout
            for i in range(b):
                if fin[i]:
                    continue
                var tv = Int(self.h_tokout[i])
                var hit = tv == eos
                if fp[i]:
                    # 本步消费的是 prompt token
                    if pidx[i] == plen[i]:
                        # 恰为末 prompt token → gen[0]
                        out[obase[i] + 0] = tv
                        genc[i] = 1
                        if (not hit) and max_new > 1:
                            cur[i] = tv
                    else:
                        continue
                else:
                    # 生成链：cur = gen[genc-1] → 本步产 gen[genc]
                    out[obase[i] + genc[i]] = tv
                    genc[i] = genc[i] + 1
                    if (not hit) and genc[i] < max_new:
                        cur[i] = tv
                # 完成判定（EOS 或预算用尽）→ joiner 接槽或退场
                if hit or genc[i] >= max_new:
                    if jt < jb:
                        poff[i] = joff[jt]
                        plen[i] = join_lens[jt]
                        pidx[i] = 0
                        psrc[i] = 1
                        genc[i] = 0
                        cur[i] = 0
                        obase[i] = (b + jt) * max_new
                        jt += 1
                        self.h_posarr[i] = Int32(0)
                    else:
                        fin[i] = True
                        nfin += 1
        return out^


    def decode_step[
        inner: Int, inner_ff: Int, dual_t: Int = 64, dual_o: Int = 4,
        tri_t: Int = 128, tri_cols: Int = 0, down_t: Int = 256, o_t: Int = 128,
        attn_t: Int = 512, attn_vec: Bool = True, attn_v2: Bool = True,
        attn_smode: Int = 0, attn_tiles: Int = 8, dual_nx: Bool = False,
        down_si: Int = 0, o_outs: Int = 1,
        qkv_pack: Bool = True,
    ](
        mut self, token: Int, skip: Int = 0, sample: Bool = False,
        span: Bool = False
    ) raises -> Pointer[Float32, MutUntrackedOrigin]:
        """t=1 的完整前向：返回 host logits 指针（下次调用前有效）。

        算子序与 `run` 逐步对应；KV 追加在 device 侧 store 上，`kv_pos` 是
        会话内的绝对位置。中间零 host 往返（token id 一次 4 字节上传）。

        dual_t/dual_o（账本七十五）：dual 形状 comptime 参（默认 =
        六十九微基准冠军）。⚠️ 真链重扫一度示 T128/O8 快 0.39ms，被
        交错 A/B（bench_dual_ab）证伪 —— 本机是共享 GPU，顺序扫型的
        形状窗口落在不同负载时段，排名是漂移假象；交错下两形状逐位
        打平（2.9575 vs 2.9526）。参数保留作扫型基建，默认不动。

        tri_t（账本七十七）：tri 发射的线程数 comptime 参。交错 A/B
        （bench_tri_ab，三轮同序）冠军 T128：省 ~0.19ms（tri 18.3 →
        ~10.3µs/层 ≈ 245GB/s），默认随之切换。
        七十六实测 tri 138GB/s —— 896 块 × 32 线程只装 13% 线程槽；
        T 提升直接提高占据率。交错 A/B（bench_tri_ab）裁决，⚠️ NORM
        融合的跨线程归约序随 T 变，与默认非逐位同 —— 以门容差与
        argmax 不变为判据。
        down_t/o_t（账本七十八）：down/o 发射的线程数 comptime 参
        （默认 128）。fat 核 grid=ceildiv(hidden,2)=448 块 —— T128 只有
        26% 线程槽（tri 同款占据率病）；交错 A/B 裁决后定默认。
        attn_t/attn_vec/attn_v2（账本七十九/八十）：attention 发射
        形状/向量化/V 相并行开关。八十分相：V 相占 kernel 35%（16.3µs
        @kv300，只许 128 线程干活）——V2（T512，j 分 4 组并行 + smem
        部分和按组序合并）砍到 6.6µs，kernel 46.1 → 36.5µs；端到端
        交错三轮同序（默认 2.80/2.70/2.89 vs V2 2.79/2.55/2.60）。默认
        T512+V2。⚠️ j 求和序 + sum 树序变（f64 重结合，bit-exact 判据
        弃用，转门 1e-5 容差 + argmax）。
        T 扫无效（T256/512 噪声内）；VEC（k 行 load[8] 向量读，求和序
        不变 ⇒ scores 逐位同）三轮一致 -0.017ms，默认开。⚠️ attention
        20µs/层的主导项仍未定位 —— 14 块 SM 饥饿是结构性的（j 分块
        跨块归约是下一刀），VEC 只是零头。
        skip（账本七十六）：骨架测量掩码（SKIP_* 位，默认 0 = 完整
        路径）。置位的 kernel 类换成 noop 发射 —— 发射结构与同步/
        拷贝原样保留，只剥离该类 GPU 工作；skip=0 时与原路径逐指令
        相同。配 bench_decode_span 做链上时间分解（发射间隙 vs
        kernel 本体），不做正确性用途。
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

        # 一百一十六-b：持久 pinned 缓冲，去每步 alloc + 早 sync
        # （步骤末尾本就有 sync，上一循环的 H2D 必已完成）
        self.h_token[0] = Int32(token)
        self.ctx.enqueue_copy(dst_buf=self.d_token, src_buf=self.h_token)
        if skip & SKIP_GATHER == 0:
            self.ctx.enqueue_function[gather_gpu](
                self.d_x.unsafe_ptr(),
                self.d_embed.unsafe_ptr(),
                self.d_token.unsafe_ptr(),
                Int32(self.hidden),
                Int32(1),
                grid_dim=(ceildiv(self.hidden, 64), 1, 1),
                block_dim=(64, 1, 1),
            )
        else:
            self.ctx.enqueue_function[noop_gpu](
                grid_dim=(1, 1, 1), block_dim=(32, 1, 1)
            )

        for layer in range(self.n_layers):
            var base = layer * Q4_SLOTS_PER_LAYER
            # q/k/v 三融合走 f16 + NORM 就地归一化（账本六十八/七十一）：
            # 直接吃残差流 d_x，norm 单发整个省掉。
            var dst_qkv_q = self._dst(0)
            var dst_qkv_k = self._dst(1)
            var dst_qkv_v = self._dst(2)
            if skip & SKIP_TRI == 0:
                comptime if qkv_pack:
                    # 一百一十五-b：权重 cat3 打包一次（首步），此后
                    # 每层一发 qkv_fat（fat 结构 + 就地 norm + 三路
                    # bias，隔离对决 tri 1.51×）。
                    if not self.qkvp_ready:
                        for pl in range(self.n_layers):
                            self.ctx.enqueue_function[cat3_f16_gpu](
                                self.d_wp_qkv[pl].unsafe_ptr(),
                                self.d_w16_q[pl].unsafe_ptr(),
                                self.d_w16_k[pl].unsafe_ptr(),
                                self.d_w16_v[pl].unsafe_ptr(),
                                Int32(self.q_out * inner),
                                Int32(self.kv_dim * inner),
                                Int32(self.kv_dim * inner),
                                grid_dim=(
                                    ceildiv(
                                        (self.q_out + 2 * self.kv_dim)
                                        * inner,
                                        256,
                                    ),
                                    1,
                                    1,
                                ),
                                block_dim=(256, 1, 1),
                            )
                        self.qkvp_ready = True
                    self.ctx.enqueue_function[
                        f16_matmul_qkv_fat_gpu[tri_t, inner, 2]
                    ](
                        self.d_x.unsafe_ptr(),
                        self.d_wp_qkv[layer].unsafe_ptr(),
                        dst_qkv_q.unsafe_ptr(),
                        dst_qkv_k.unsafe_ptr(),
                        dst_qkv_v.unsafe_ptr(),
                        self.d_bias_q[layer].unsafe_ptr(),
                        self.d_bias_k[layer].unsafe_ptr(),
                        self.d_bias_v[layer].unsafe_ptr(),
                        self.d_norms[2 * layer].unsafe_ptr(),
                        self.eps,
                        Int32(inner),
                        Int32(self.q_out),
                        Int32(self.kv_dim),
                        grid_dim=(
                            (self.q_out + 2 * self.kv_dim) // 2,
                            1,
                            1,
                        ),
                        block_dim=(tri_t, 1, 1),
                    )
                elif tri_cols > 0:
                    self.ctx.enqueue_function[
                        f16_matmul_qkv_mc_gpu[tri_t, inner, tri_cols, True, True]
                    ](
                        self.d_x.unsafe_ptr(),
                        self.d_w16_q[layer].unsafe_ptr(),
                        self.d_w16_k[layer].unsafe_ptr(),
                        self.d_w16_v[layer].unsafe_ptr(),
                        dst_qkv_q.unsafe_ptr(),
                        dst_qkv_k.unsafe_ptr(),
                        dst_qkv_v.unsafe_ptr(),
                        self.d_bias_q[layer].unsafe_ptr(),
                        self.d_bias_k[layer].unsafe_ptr(),
                        self.d_bias_v[layer].unsafe_ptr(),
                        self.d_norms[2 * layer].unsafe_ptr(),
                        self.eps,
                        Int32(inner),
                        Int32(self.q_out),
                        Int32(self.kv_dim),
                        grid_dim=(ceildiv(self.q_out, tri_cols), 1, 1),
                        block_dim=(tri_t, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        f16_matmul_qkv_gpu[tri_t, inner, True, True]
                    ](
                    self.d_x.unsafe_ptr(),
                    self.d_w16_q[layer].unsafe_ptr(),
                    self.d_w16_k[layer].unsafe_ptr(),
                    self.d_w16_v[layer].unsafe_ptr(),
                    dst_qkv_q.unsafe_ptr(),
                    dst_qkv_k.unsafe_ptr(),
                    dst_qkv_v.unsafe_ptr(),
                    self.d_bias_q[layer].unsafe_ptr(),
                    self.d_bias_k[layer].unsafe_ptr(),
                    self.d_bias_v[layer].unsafe_ptr(),
                    self.d_norms[2 * layer].unsafe_ptr(),
                    self.eps,
                    Int32(inner),
                    Int32(self.q_out),
                    Int32(self.kv_dim),
                    grid_dim=(max(self.q_out, self.kv_dim), 1, 1),
                    block_dim=(tri_t, 1, 1),
                )
            else:
                self.ctx.enqueue_function[noop_gpu](
                    grid_dim=(1, 1, 1), block_dim=(32, 1, 1)
                )
            # attention + rope 四合一（账本七十三）：rope_kv 的活折进
            # attention 头部（q rope / k rope 直写本 token store 行 / v
            # 拷贝），当前 token 的 k/v 经块内 smem 中转 —— 同 launch 内
            # 读兄弟块刚写的 store 行是竞态。表行/store 行 = 绝对位置
            # pos（指针偏移）；store 基指针在 kernel 内按 upto 行定位。
            var cos_row = self.d_cos.unsafe_ptr() + pos * self.head_dim
            var sin_row = self.d_sin.unsafe_ptr() + pos * self.head_dim
            # 两个 store 指针先落地局部变量 —— 同一 List 的两个下标表达式
            # 直接进发射会被 aliasing 检查当成潜在同源（账本六十三）。
            # 一百三十三：KV 统一到**批 store 行 0**——批 store 布局
            # [b][max_kv][kv_cols]，行 0 段与单路布局 [pos][kv_cols] 逐
            # 字节同构（pos < max_kv 由 kv_len 检查保证）。旧行为写单路
            # store（d_k_store），而 prefill_step 批路解码读批 store
            # ——跨路径 KV 断层：纯流路冷启动读空 store 出数字噪声、
            # generate_batch 先行"治愈"是残留恰同 prompt 的假象（复现
            # 器 test_pool_repro.mojo 三个世界钉死）。
            var pk = self.d_k_store_b[layer].unsafe_ptr()
            var pv = self.d_v_store_b[layer].unsafe_ptr()
            if skip & SKIP_ATTN == 0:
                comptime if attn_tiles < 0:
                    # 负 tiles = 旧 K2/K3（八十二，8× 冗余 softmax），留 A/B 复现。
                    self.ctx.enqueue_function[
                        attn_vt_gpu[attn_t, -attn_tiles]
                    ](
                        self.d_scores.unsafe_ptr(),
                        pv,
                        self.d_vpart.unsafe_ptr(),
                        self.d_attot.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.n_kv_heads),
                        Int32(self.head_dim),
                        Int32(kv_len),
                        grid_dim=(self.n_heads, -attn_tiles, 1),
                        block_dim=(attn_t, 1, 1),
                    )
                    self.ctx.enqueue_function[
                        attn_comb_gpu[-attn_tiles]
                    ](
                        self.d_attn.unsafe_ptr(),
                        self.d_vpart.unsafe_ptr(),
                        self.d_attot.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.head_dim),
                        grid_dim=(self.n_heads, 1, 1),
                        block_dim=(attn_t, 1, 1),
                    )
                elif attn_tiles > 0:
                    # 三趟（八十二 K1 + 八十三 K2'/K3'）：gmax 冗余只读、
                    # exp 本 tile smem 缓存（全行 1×）、gsum 推迟 K3 合并。
                    self.ctx.enqueue_function[
                        attn_rs_gpu[attn_t, attn_vec, attn_tiles]
                    ](
                        self.d_q_rot.unsafe_ptr(),
                        self.d_q.unsafe_ptr(),
                        self.d_k.unsafe_ptr(),
                        self.d_v.unsafe_ptr(),
                        pk,
                        pv,
                        cos_row,
                        sin_row,
                        self.d_scores.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.n_kv_heads),
                        Int32(self.head_dim),
                        Int32(kv_len),
                        grid_dim=(self.n_heads, attn_tiles, 1),
                        block_dim=(attn_t, 1, 1),
                    )
                    self.ctx.enqueue_function[
                        attn_vt2_gpu[attn_t, attn_tiles]
                    ](
                        self.d_scores.unsafe_ptr(),
                        pv,
                        self.d_vpart.unsafe_ptr(),
                        self.d_attot.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.n_kv_heads),
                        Int32(self.head_dim),
                        Int32(kv_len),
                        grid_dim=(self.n_heads, attn_tiles, 1),
                        block_dim=(attn_t, 1, 1),
                    )
                    self.ctx.enqueue_function[
                        attn_comb2_gpu[attn_tiles]
                    ](
                        self.d_attn.unsafe_ptr(),
                        self.d_vpart.unsafe_ptr(),
                        self.d_attot.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.head_dim),
                        grid_dim=(self.n_heads, 1, 1),
                        block_dim=(attn_t, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        attn_rope_gpu[
                            attn_t, attn_vec, 0, attn_v2, attn_smode
                        ]
                    ](
                        self.d_attn.unsafe_ptr(),
                        self.d_q_rot.unsafe_ptr(),
                        self.d_q.unsafe_ptr(),
                        self.d_k.unsafe_ptr(),
                        self.d_v.unsafe_ptr(),
                        pk,
                        pv,
                        cos_row,
                        sin_row,
                        self.d_scores.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.n_kv_heads),
                        Int32(self.head_dim),
                        Int32(kv_len),
                        grid_dim=(self.n_heads, 1, 1),
                        block_dim=(attn_t, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[noop_gpu](
                    grid_dim=(1, 1, 1), block_dim=(32, 1, 1)
                )
            # o 走 f16 fat O2 + ACC（账本七十）：直接累加进 d_x，省残差
            # 发射；舍入顺序与"写 proj 再 add_into"逐位一致。
            var x_o = self._src(1)
            if skip & SKIP_O == 0:
                self.ctx.enqueue_function[
                    f16_matmul_fat_gpu[o_t, inner, o_outs, False, True]
                ](
                    x_o.unsafe_ptr(),
                    self.d_w16_o[layer].unsafe_ptr(),
                    self.d_x.unsafe_ptr(),
                    self.d_ws.unsafe_ptr(),
                    Int32(inner),
                    Int32(self.hidden),
                    grid_dim=(ceildiv(self.hidden, o_outs), 1, 1),
                    block_dim=(o_t, 1, 1),
                )
            else:
                self.ctx.enqueue_function[noop_gpu](
                    grid_dim=(1, 1, 1), block_dim=(32, 1, 1)
                )
            # gate/up dual f16 + swiglu 尾缀 + NORM 就地归一化（账本
            # 六十八/七十/七十一）：直接吃 d_x；T64/O4；act 在写回后
            # 顺带算。bias 槽是占位（HAS_BIAS=False 不解引用），但发射
            # aliasing 检查要求两两不同 —— 用 d_proj 占位。
            var w_gate = self.d_w16_gate[layer].unsafe_ptr()
            var w_up = self.d_w16_up[layer].unsafe_ptr()
            var dst_gate = self._dst(4)
            var dst_up = self._dst(5)
            if skip & SKIP_DUAL == 0:
                comptime if dual_nx:
                    # 账本八十五：norm 出核（rmsnorm 单发写 d_xnorm）+
                    # dual 去 smem 直读 L1 —— 破 x 摊薄 vs occupancy 死锁。
                    self.ctx.enqueue_function[rmsnorm_gpu](
                        self.d_x.unsafe_ptr(),
                        self.d_norms[2 * layer + 1].unsafe_ptr(),
                        self.d_xnorm.unsafe_ptr(),
                        Int32(inner),
                        Int32(1),
                        self.eps,
                        grid_dim=(1, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                    self.ctx.enqueue_function[
                        f16_matmul_dual_nx_gpu[dual_t, inner, dual_o, True]
                    ](
                        self.d_xnorm.unsafe_ptr(),
                        w_gate,
                        w_up,
                        dst_gate.unsafe_ptr(),
                        dst_up.unsafe_ptr(),
                        self.d_ws.unsafe_ptr(),
                        self.d_proj.unsafe_ptr(),
                        self.d_act.unsafe_ptr(),
                        Int32(self.inter),
                        grid_dim=(ceildiv(self.inter, dual_o), 1, 1),
                        block_dim=(dual_t, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        f16_matmul_dual_gpu[dual_t, inner, dual_o, False, True, True]
                    ](
                    self.d_x.unsafe_ptr(),
                    w_gate,
                    w_up,
                    dst_gate.unsafe_ptr(),
                    dst_up.unsafe_ptr(),
                    self.d_ws.unsafe_ptr(),
                    self.d_proj.unsafe_ptr(),
                    self.d_act.unsafe_ptr(),
                    self.d_norms[2 * layer + 1].unsafe_ptr(),
                    self.eps,
                    Int32(inner),
                    Int32(self.inter),
                    grid_dim=(ceildiv(self.inter, dual_o), 1, 1),
                    block_dim=(dual_t, 1, 1),
                )
            else:
                self.ctx.enqueue_function[noop_gpu](
                    grid_dim=(1, 1, 1), block_dim=(32, 1, 1)
                )
            # down f16 fat O2 + ACC（账本六十八/七十）。
            var w_down = self.d_w16_down[layer].unsafe_ptr()
            var x_down = self._src(2)
            # 六十九扫型：T128/O2 冠军（15.6 vs T256 15.8us）。
            if skip & SKIP_DOWN == 0:
                comptime if down_si > 0:
                    # split-inner（八十七）：块数 ×down_si、每块 x/w 只装
                    # 1/down_si chunk，combine 微核 s 升序收口 + 残差 ACC。
                    self.ctx.enqueue_function[
                        f16_matmul_fat_si_gpu[down_t, inner_ff, 2, down_si]
                    ](
                        x_down.unsafe_ptr(),
                        w_down,
                        self.d_dpart.unsafe_ptr(),
                        Int32(inner_ff),
                        Int32(self.hidden),
                        grid_dim=(ceildiv(self.hidden, 2), down_si, 1),
                        block_dim=(down_t, 1, 1),
                    )
                    self.ctx.enqueue_function[
                        fat_si_comb_gpu[8, down_si, True]
                    ](
                        self.d_dpart.unsafe_ptr(),
                        self.d_x.unsafe_ptr(),
                        Int32(self.hidden),
                        grid_dim=(ceildiv(self.hidden, 8), 1, 1),
                        block_dim=(128, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        f16_matmul_fat_gpu[down_t, inner_ff, 2, False, True]
                    ](
                        x_down.unsafe_ptr(),
                        w_down,
                        self.d_x.unsafe_ptr(),
                        self.d_ws.unsafe_ptr(),
                        Int32(inner_ff),
                        Int32(self.hidden),
                        grid_dim=(ceildiv(self.hidden, 2), 1, 1),
                        block_dim=(down_t, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[noop_gpu](
                    grid_dim=(1, 1, 1), block_dim=(32, 1, 1)
                )

        if skip & SKIP_HEAD == 0:
            self._norm(2 * self.n_layers)
            # lm_head 走 fat 版（账本六十六扫型 T256/O8 冠军）。NORM 融合
            # 试过（七十一）：18688 块 × 每块冗余归约 + 屏障链，实测比
            # norm 单发更慢，头部保持分离发射。
            self.ctx.enqueue_function[linear_fat_f16_gpu[256, inner, 8]](
                self.d_normed.unsafe_ptr(),
                self.d_embed16.unsafe_ptr(),
                self.d_logits.unsafe_ptr(),
                self.d_ws.unsafe_ptr(),
                self.eps,
                Int32(self.vocab),
                grid_dim=(ceildiv(self.vocab, 8), 1, 1),
                block_dim=(256, 1, 1),
            )
        else:
            self.ctx.enqueue_function[noop_gpu](
                grid_dim=(1, 1, 1), block_dim=(32, 1, 1)
            )
        if span:
            # 一百一十六-c 试验位：跳尾部回传与 sync（argmax 仍写
            # d_token，链条设备侧闭环）——量 N 步一 sync 的上限。
            self.ctx.enqueue_function[argmax_f32_gpu](
                self.d_logits.unsafe_ptr(),
                self.d_amax.unsafe_ptr(),
                self.d_aidx.unsafe_ptr(),
                Int32(self.vocab),
                grid_dim=(128, 1, 1),
                block_dim=(256, 1, 1),
            )
            self.ctx.enqueue_function[argmax_comb_gpu](
                self.d_amax.unsafe_ptr(),
                self.d_aidx.unsafe_ptr(),
                self.d_token.unsafe_ptr(),
                Int32(128),
                grid_dim=(1, 1, 1),
                block_dim=(256, 1, 1),
            )
            self.kv_pos = kv_len
            return self.h_logits.unsafe_ptr()
        if sample:
            # 一百一十六-b：device argmax 收口 + 4B 回传（vs 608KB
            # logits D2H）；完整 logits 走 sample=False 或 run()。
            self.ctx.enqueue_function[argmax_f32_gpu](
                self.d_logits.unsafe_ptr(),
                self.d_amax.unsafe_ptr(),
                self.d_aidx.unsafe_ptr(),
                Int32(self.vocab),
                grid_dim=(128, 1, 1),
                block_dim=(256, 1, 1),
            )
            self.ctx.enqueue_function[argmax_comb_gpu](
                self.d_amax.unsafe_ptr(),
                self.d_aidx.unsafe_ptr(),
                self.d_token.unsafe_ptr(),
                Int32(128),
                grid_dim=(1, 1, 1),
                block_dim=(256, 1, 1),
            )
            self.ctx.enqueue_copy(dst_buf=self.h_token, src_buf=self.d_token)
            self.ctx.synchronize()
            self.last_token = Int(self.h_token[0])
        else:
            self.ctx.enqueue_copy(
                dst_buf=self.h_logits, src_buf=self.d_logits
            )
            self.ctx.synchronize()
        self.kv_pos = kv_len
        return self.h_logits.unsafe_ptr()

    def prefill_step[
        inner: Int, inner_ff: Int, dual_t: Int = 64, dual_o: Int = 4,
        tri_t: Int = 128, down_t: Int = 128, o_t: Int = 128,
        attn_t: Int = 512, attn_tiles: Int = 1,
        prefill_rb: Int = 8,
        prefill_af: DType = DType.float64,
        # 一百零五：dual/down MMA 默认开（diff 1e-3 级、norm2 重转修正
        # 后）；q/o MMA 默认关——896×896 小 GEMM 的 MMA 网格仅 56 块，
        # 并行度不足，bench 实测负收益 ~2.4ms（26.97 vs 24.55）。
        # 一百零六：q/o MMA 已移除（896×896 小 GEMM 网格负收益 ~2.4ms，
        # 一百零五实测）；仅 dual/down 走 MMA。
        prefill_mma_dual: Bool = True,
        prefill_mma_down: Bool = True,
        # 一百零七：w 打包布局开关（False = 走原 [out_cols, INNER] 布局的
        # MMA 核，仅供同进程配对 A/B 校准；生产 True）
        prefill_wpack: Bool = True,
        # 一百零八：q2/o 也走 packed MMA（w 打包后 896×896 从 110/124µs
        # 降到 39/39µs；一百零五的负收益是散段载入时代的结论）
        prefill_mma_q: Bool = True,
        prefill_mma_o: Bool = True,
        # 一百零九：t=128 的满时钟/冷权重复扫（nsys 实测 down 188µs、
        # dual 96µs）——down 改 OUTS=1（COLS 64 → 112 块，-31%）、
        # dual 改 CH=896（单段无尾余，-17%）
        prefill_tile: Bool = True,
        # 一百零九：scores 按 kv_head 分组（同组 7 个 query head 共享一次
        # k 行读）——nsys 定位 RS 为访存受限，分组后 78→36µs（逐位同）
        prefill_gqa: Bool = True,
        # 一百一十二-b：down split-K（t=128 时 112 块只够每 SM 1 块 8 warp，
        # 藏不住冷权重延迟；沿 K 切 8 段 → 896 块，实测 1.96×）
        prefill_spl: Bool = True,
        # 一百一十三-b：kv2 换双输出 packed MMA（k/v 权重拼接一发算完）
        prefill_kv: Bool = False,
        # 一百二十：批解码模式——t 行 = t 序列各 1 token（pos_arr 独立
        # rope/KV 基址，_bb attention 核，B 行 head+argmax）
        batch: Bool = False,
    ](
        mut self, tokens: List[Int], profile: Bool = False, sync_tail: Bool = True,
        # 一百二十-b：False 时批 tail 跳过 B×vocab logits D2H（sample
        # 模式，只回 tokout 4B×B）
        copy_logits: Bool = True,
        # 一百二十三：>0 走 sample_rb（Gumbel-max 温度采样），=0 纯 argmax
        sample_temp: Float32 = 0.0,
        sample_seed: Int32 = 0,
        # 一百二十八：<1 启用 top-p nucleus（配合 sample_temp>0）；
        # =1 全词表（同 sample_rb）
        sample_topp: Float32 = 1.0,
        layer_gap_us: Int = 0, layer_gap_every: Int = 1,
        # 一百三十四：批路的**起始行号**——行 b（grid z）实际读写
        # store_b/posarr/scores 的行 row_base+b。合批服务把每条流钉在
        # 自己的批行上（slot≠0 的流单步前向不再错位）；padding 补齐或
        # 紧密压缩两种编排都直接可用。默认 0 = 原行为。
        row_base: Int = 0,
        # 一百三十六：per-row 采样——True 时忽略整批标量参数，行 b 用
        # h_stemp/h_stopps/h_sseeds[b]（host 直写，forward 时整块上传）。
        # 一个 launch 吃任意 temp/top_p 组合（核内每行自选分支）。
        per_row_sampling: Bool = False,
    ) raises -> Pointer[Float32, MutUntrackedOrigin]:
        """t 行批量前向（账本八十八，prefill/chunked prefill 之路）：
        gather → 每层 tri/attention 三趟 b 版/o/dual/down → 末行终
        norm → lm_head（只算末行 logits）。

        attn_tiles（账本一百零一）：默认 1——t=128 时 rs 的 chunk=nsc
        可与 512 线程对齐（T8 时每块仅 nsc/8≈16 线程活跃 + gmax 全行
        8× 冗余）。前置条件：nsc = pos0+b+1 ≤ THREADS(512)，由
        max_tokens ≤ attn_t(512) 的 kv 检查保证；更长上下文须用
        attn_tiles ≥ ceil(nsc/THREADS)。

        matmul 族（tri/o/dual/down）与 gather/rmsnorm 本就带 row 维
        （row = block_idx.y / 一行一块）—— grid y=t 直通；attention
        三趟用 b 版核（grid z=t，因果界 upto = pos0+b，scores 行 =
        (b*heads+head)*kv_len，stride 取终态 kv_len）。逐行算术与
        decode_step 同构（同核同序同树）；t=1 时与 decode_step 同门
        容差。KV 追加在 device 侧 store，行 b 落绝对位置 pos0+b ——
        多次调用即 chunked prefill，KV 连续。

        返回 host logits 指针（末行，下次调用前有效）。
        """
        var t = len(tokens)
        if t == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "no tokens given", "")
        # 一百三十八：真前向口径（launch 次数 × 行·步总数）——批次路
        # 与单路、服务层与会话门全走这里，n_ticks/n_batched 混的空转
        # 与 padding 在这两个数上不再失真。
        self.n_forwards += 1
        self.n_forward_rows += t
        if t > PREFILL_MAX_BATCH:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "prefill batch exceeds capacity (chunk the prompt)",
                "t=" + String(t) + " cap=" + String(PREFILL_MAX_BATCH),
            )
        var pos0 = self.kv_pos
        var kv_len = pos0 + t
        comptime if batch:
            if t > BMAX:
                raise AlofaError(
                    ERR_INVALID_ARGUMENT, "batch decode cap is 64",
                    "t=" + String(t),
                )
            for i in range(t):
                if Int(self.h_posarr[i]) + 1 > self.max_tokens:
                    raise AlofaError(
                        ERR_OUT_OF_RANGE, "kv store is full (batch)", ""
                    )
            self.ctx.enqueue_copy(
                dst_buf=self.d_posarr, src_buf=self.h_posarr
            )
        else:
            if kv_len > self.max_tokens:
                raise AlofaError(ERR_OUT_OF_RANGE, "kv store is full", "")

        # 分段计时（profile=True 时段间 synchronize——破坏流水只看占比）
        var prof = InlineArray[Float64, 10](fill=Float64(0))
        var prof_t0 = monotonic_ns()

        # 九十四：kernel 特化提为 alias（编译期构造，发射路径免重复特化查找）
        alias NORM_K = rmsnorm_gpu
        # 账本一百：q2/kv2——qkv 拆 q 链（OUTS=4）与 k/v 双链（OUTS=2）
        # 寄存器分块，qkv 607→188µs（3.2×，bit-exact 同旧核）。
        alias Q_K = f16_bmm_q2_gpu[tri_t, inner, 4, prefill_rb, prefill_af]
        alias KV_K = f16_bmm_kv2_gpu[tri_t, inner, 2, prefill_rb, prefill_af]
        alias KVS_K = kv_store_b_gpu[attn_t]
        alias RS_K = attn_rs_b_gpu[attn_t, True, attn_tiles]
        alias VT2_K = attn_vt2_b_gpu[attn_t, attn_tiles]
        alias COMB_K = attn_comb2_b_gpu[attn_tiles]
        # 账本九十九：fat3/dual3——x tile 寄存器复用（流量 ÷OUTS）+
        # w tile 寄存器组：o/down 1.6-1.8×、dual 1.8×（bit-exact 同旧核）。
        alias O_K = f16_bmm_fat3_gpu[o_t, inner, 4, prefill_rb, True, prefill_af]
        alias DUAL_K = f16_bmm_dual3_gpu[dual_t, inner, 2, prefill_rb, prefill_af]
        alias DOWN_K = f16_bmm_fat3_gpu[down_t, inner_ff, 4, prefill_rb, True, prefill_af]
        alias LMHEAD_K = linear_fat_f16_gpu[256, inner, 8]

        # host 侧与 d_tokenb 同宽（copy 要求同尺寸；尾部零填充无害）
        var ht = self.ctx.enqueue_create_host_buffer[DType.int32](
            PREFILL_MAX_BATCH
        )
        self.ctx.synchronize()
        for i in range(PREFILL_MAX_BATCH):
            ht[i] = Int32(tokens[i]) if i < t else Int32(0)
        self.ctx.enqueue_copy(dst_buf=self.d_tokenb, src_buf=ht)

        # gather：t 行 embedding（gather_gpu 本就按 t 行索引）
        self.ctx.enqueue_function[gather_gpu](
            self.d_xb.unsafe_ptr(),
            self.d_embed.unsafe_ptr(),
            self.d_tokenb.unsafe_ptr(),
            Int32(self.hidden),
            Int32(t),
            grid_dim=(ceildiv(t * self.hidden, 64), 1, 1),
            block_dim=(64, 1, 1),
        )
        if profile:
            self.ctx.synchronize()
            prof[0] += Float64(monotonic_ns() - prof_t0) / 1e6
            prof_t0 = monotonic_ns()

        # 一百零七：首次进 f32 快路径时把 gate/up/down 打成
        # fragment 连续布局（每层 3 次 launch，共 0.24ms 量级）
        comptime if prefill_af == DType.float32 and prefill_wpack and (
            prefill_mma_dual or prefill_mma_down or prefill_mma_q
            or prefill_mma_o
        ):
            if not self.wp_ready:
                for pl in range(self.n_layers):
                    self.ctx.enqueue_function[
                        pack_w_mma_gpu[inner, 2, 8]
                    ](
                        self.d_w16_gate[pl].unsafe_ptr(),
                        self.d_wp_gate[pl].unsafe_ptr(),
                        grid_dim=(ceildiv(self.inter, 128),
                        inner // 16, 1),
                        block_dim=(128, 1, 1),
                    )
                    self.ctx.enqueue_function[
                        pack_w_mma_gpu[inner, 2, 8]
                    ](
                        self.d_w16_up[pl].unsafe_ptr(),
                        self.d_wp_up[pl].unsafe_ptr(),
                        grid_dim=(ceildiv(self.inter, 128),
                        inner // 16, 1),
                        block_dim=(128, 1, 1),
                    )
                    comptime if prefill_tile:
                        self.ctx.enqueue_function[
                            pack_w_mma_gpu[inner_ff, 1, 8]
                        ](
                            self.d_w16_down[pl].unsafe_ptr(),
                            self.d_wp_down[pl].unsafe_ptr(),
                            grid_dim=(ceildiv(self.hidden, 64),
                            inner_ff // 16, 1),
                            block_dim=(128, 1, 1),
                        )
                    else:
                        self.ctx.enqueue_function[
                            pack_w_mma_gpu[inner_ff, 2, 8]
                        ](
                            self.d_w16_down[pl].unsafe_ptr(),
                            self.d_wp_down[pl].unsafe_ptr(),
                            grid_dim=(ceildiv(self.hidden, 128),
                            inner_ff // 16, 1),
                            block_dim=(128, 1, 1),
                        )
                    comptime if prefill_mma_q:
                        self.ctx.enqueue_function[
                            pack_w_mma_gpu[inner, 2, 8]
                        ](
                            self.d_w16_q[pl].unsafe_ptr(),
                            self.d_wp_q[pl].unsafe_ptr(),
                            grid_dim=(ceildiv(self.q_out, 128),
                            inner // 16, 1),
                            block_dim=(128, 1, 1),
                        )
                    comptime if prefill_kv and prefill_wpack and (
                        inner % 16 == 0 and inner % 896 == 0
                    ):
                        # 一百一十三-b：k/v 拼成 [2*kv_dim, inner] 再打包
                        # + bias 拼接（kvraw 同流顺序复用一块 scratch）
                        self.ctx.enqueue_function[cat2_f16_gpu](
                            self.d_w16_k[pl].unsafe_ptr(),
                            self.d_w16_v[pl].unsafe_ptr(),
                            self.d_kvraw.unsafe_ptr(),
                            Int32(self.kv_dim * inner),
                            grid_dim=(ceildiv(
                                self.kv_dim * inner, 256
                            ), 1, 1),
                            block_dim=(256, 1, 1),
                        )
                        self.ctx.enqueue_function[
                            pack_w_mma_gpu[inner, 2, 8]
                        ](
                            self.d_kvraw.unsafe_ptr(),
                            self.d_wp_kv[pl].unsafe_ptr(),
                            grid_dim=(ceildiv(2 * self.kv_dim, 128),
                            inner // 16, 1),
                            block_dim=(128, 1, 1),
                        )
                        self.ctx.enqueue_function[cat2_f32_gpu](
                            self.d_bias_k[pl].unsafe_ptr(),
                            self.d_bias_v[pl].unsafe_ptr(),
                            self.d_bias_kv[pl].unsafe_ptr(),
                            Int32(self.kv_dim),
                            grid_dim=(ceildiv(2 * self.kv_dim, 256), 1, 1),
                            block_dim=(256, 1, 1),
                        )
                    comptime if prefill_mma_o:
                        self.ctx.enqueue_function[
                            pack_w_mma_gpu[inner, 2, 8]
                        ](
                            self.d_w16_o[pl].unsafe_ptr(),
                            self.d_wp_o[pl].unsafe_ptr(),
                            grid_dim=(ceildiv(self.hidden, 128),
                            inner // 16, 1),
                            block_dim=(128, 1, 1),
                        )
                self.wp_ready = True

        for layer in range(self.n_layers):
            var ltri = 0.0
            var lattn = 0.0
            var lo = 0.0
            var lff = 0.0
            var ldown = 0.0
            # tri：norm 前置单发（t 行一次，树序重组差在门容差内）+
            # 真 GEMM 分块（八十九）：权重每 RB 行一次读，x 走 L2。
            self.ctx.enqueue_function[rmsnorm_gpu](
                self.d_xb.unsafe_ptr(),
                self.d_norms[2 * layer].unsafe_ptr(),
                self.d_xnormb.unsafe_ptr(),
                Int32(inner),
                Int32(t),
                self.eps,
                grid_dim=(t, 1, 1),
                block_dim=(256, 1, 1),
            )
            comptime if prefill_af == DType.float32 and (
                prefill_mma_q or prefill_mma_dual
            ):
                # 一百零五：xnormb f16 预转前置（q MMA 与 dual MMA 共用）
                self.ctx.enqueue_function[x_to_f16_gpu](
                    self.d_xnormb.unsafe_ptr(),
                    self.d_xnormb16.unsafe_ptr(),
                    Int32(t * inner),
                    grid_dim=((t * inner + 255) // 256, 1, 1),
                    block_dim=(256, 1, 1),
                )
            comptime if prefill_af == DType.float32 and prefill_mma_q and (
                prefill_wpack and inner % 16 == 0
            ):
                comptime if inner % 896 == 0:
                    # 一百一十七-b：q2 split-K（S=8/CH=112，56→448 块）。
                    # 冷转对决（probe_pf117b，48 套）：pk 45.2µs →
                    # spl8 28.2µs = 1.60×（每 SM 半块藏不住冷权重
                    # ~600 周期 DRAM 延迟，down_spl 同病同方）。
                    # 累加序变（~1e-4 级），门容差 + argmax 判据。
                    self.ctx.enqueue_function[
                        f16_mma_down_spl_gpu[inner, 112, 2, 8, 8]
                    ](
                        self.d_xnormb16.unsafe_ptr(),
                        self.d_wp_q[layer].unsafe_ptr(),
                        self.d_qpart.unsafe_ptr(),
                        Int32(self.q_out),
                        Int32(t),
                        grid_dim=(
                            ceildiv(self.q_out, 128), ceildiv(t, 16), 8,
                        ),
                        block_dim=(256, 1, 1),
                    )
                    self.ctx.enqueue_function[f32_mergb_gpu[8]](
                        self.d_qb.unsafe_ptr(),
                        self.d_qpart.unsafe_ptr(),
                        self.d_bias_q[layer].unsafe_ptr(),
                        Int32(self.q_out),
                        Int32(t * self.q_out),
                        grid_dim=((t * self.q_out + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        f16_mma_gemm_pk_gpu[
                            inner,
                            896 if prefill_tile and inner >= 896 else 608,
                            2,
                            8,
                            True,
                        ]
                    ](
                        self.d_xnormb16.unsafe_ptr(),
                        self.d_wp_q[layer].unsafe_ptr(),
                        self.d_qb.unsafe_ptr(),
                        self.d_bias_q[layer].unsafe_ptr(),
                        Int32(self.q_out),
                        Int32(t),
                        grid_dim=(
                            ceildiv(self.q_out, 128), ceildiv(t, 16), 1,
                        ),
                        block_dim=(256, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[
                    Q_K
                ](
                    self.d_xnormb.unsafe_ptr(),
                    self.d_w16_q[layer].unsafe_ptr(),
                    self.d_bias_q[layer].unsafe_ptr(),
                    self.d_qb.unsafe_ptr(),
                    Int32(self.q_out),
                    Int32(t),
                    grid_dim=(ceildiv(self.q_out, 4),
                    ceildiv(t, prefill_rb), 1),
                    block_dim=(tri_t, 1, 1),
                )
            if prefill_kv and prefill_af == DType.float32 and prefill_wpack and (
                inner % 16 == 0 and inner % 896 == 0
            ) and self.kv_dim % 64 == 0:
                # 一百一十三-b：双输出 packed MMA（k/v 拼接一发算完）。
                # 必须用运行时 if：comptime if 会把 else 绑到自己身上，
                # prefill_kv=False 时 kv2 被静默跳过（本轮踩过，f32 门
                # diff 18.7 / argmax 75223 的真凶）。
                # 一百一十四-a：inner 为 896 倍数时走 split-K
                # （S=8/CH=112，128 块；真冷 DRAM 下 1.84×，探针
                # 96 套轮转 >L2 逐位恒等）。两支都是完整发射，
                # else 落在 comptime if 上不会吞分支。
                comptime if inner % 896 == 0:
                    self.ctx.enqueue_function[
                        f16_mma_gemm2_spl_gpu[inner, 112, 8, 8]
                    ](
                        self.d_xnormb16.unsafe_ptr(),
                        self.d_wp_kv[layer].unsafe_ptr(),
                        self.d_kbpart.unsafe_ptr(),
                        self.d_vbpart.unsafe_ptr(),
                        Int32(self.kv_dim),
                        Int32(t),
                        grid_dim=(
                            ceildiv(2 * self.kv_dim, 128),
                            ceildiv(t, 16),
                            8,
                        ),
                        block_dim=(256, 1, 1),
                    )
                    self.ctx.enqueue_function[f32_mergb2_gpu[8]](
                        self.d_kb.unsafe_ptr(),
                        self.d_kbpart.unsafe_ptr(),
                        self.d_vb.unsafe_ptr(),
                        self.d_vbpart.unsafe_ptr(),
                        self.d_bias_kv[layer].unsafe_ptr(),
                        Int32(self.kv_dim),
                        Int32(t * self.kv_dim),
                        grid_dim=(ceildiv(t * self.kv_dim, 256), 1, 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        f16_mma_gemm2_pk_gpu[
                            inner,
                            896 if inner >= 896 and inner % 896 == 0 else inner,
                            8,
                        ]
                    ](
                        self.d_xnormb16.unsafe_ptr(),
                        self.d_wp_kv[layer].unsafe_ptr(),
                        self.d_kb.unsafe_ptr(),
                        self.d_vb.unsafe_ptr(),
                        self.d_bias_kv[layer].unsafe_ptr(),
                        Int32(self.kv_dim),
                        Int32(t),
                        grid_dim=(
                            ceildiv(2 * self.kv_dim, 128),
                            ceildiv(t, 16),
                            1,
                        ),
                        block_dim=(256, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[
                    KV_K
                ](
                    self.d_xnormb.unsafe_ptr(),
                    self.d_w16_k[layer].unsafe_ptr(),
                    self.d_w16_v[layer].unsafe_ptr(),
                    self.d_bias_k[layer].unsafe_ptr(),
                    self.d_bias_v[layer].unsafe_ptr(),
                    self.d_kb.unsafe_ptr(),
                    self.d_vb.unsafe_ptr(),
                    Int32(self.kv_dim),
                    Int32(t),
                    grid_dim=(ceildiv(self.kv_dim, 2), ceildiv(t, prefill_rb), 1),
                    block_dim=(tri_t, 1, 1),
                )
            if profile:
                self.ctx.synchronize()
                prof[1] += Float64(monotonic_ns() - prof_t0) / 1e6
                ltri = Float64(monotonic_ns() - prof_t0) / 1e6
                prof_t0 = monotonic_ns()
            # attention 三趟 b 版：grid z = t
            # 一百三十三：pk/pv 一律指批 store 行 0（与 decode_step 同步
            # 统一）——单路 prefill 的 KV 从此与批路解码步、generate_batch
            # 共读同一块 store，跨路径链（单路 prefill → 批解码步）成立。
            # 行 0 段与单路布局 [pos][kv_cols] 逐字节同构（pos < max_kv
            # 由 kv_len 检查保证）。单路 store（d_k_store）退役保留。
            # store 与 scores 分跨 launch 边界（同 launch 跨 block 写读
            # store 行是竞态：行 b 读 pos0..pos0+b-1，全由本批写入）。
            var pk = self.d_k_store_b[layer].unsafe_ptr()
            var pv = self.d_v_store_b[layer].unsafe_ptr()
            var row_off = row_base * self.max_tokens * self.kv_dim
            var row_pos = self.d_posarr.unsafe_ptr() + row_base
            var score_off = row_base * self.n_heads * self.max_tokens
            if batch:
                self.ctx.enqueue_function[kv_store_bb_gpu[attn_t]](
                    self.d_qrot_b.unsafe_ptr(),
                    self.d_qb.unsafe_ptr(),
                    self.d_kb.unsafe_ptr(),
                    self.d_vb.unsafe_ptr(),
                    self.d_k_store_b[layer].unsafe_ptr() + row_off,
                    self.d_v_store_b[layer].unsafe_ptr() + row_off,
                    self.d_cos.unsafe_ptr(),
                    self.d_sin.unsafe_ptr(),
                    row_pos,
                    Int32(self.n_heads),
                    Int32(self.n_kv_heads),
                    Int32(self.head_dim),
                    Int32(self.max_tokens),
                    grid_dim=(self.n_heads, 1, t),
                    block_dim=(attn_t, 1, 1),
                )
            else:
                self.ctx.enqueue_function[
                    KVS_K
                ](
                    self.d_qrot_b.unsafe_ptr(),
                    self.d_qb.unsafe_ptr(),
                    self.d_kb.unsafe_ptr(),
                    self.d_vb.unsafe_ptr(),
                    pk,
                    pv,
                    self.d_cos.unsafe_ptr(),
                    self.d_sin.unsafe_ptr(),
                    Int32(self.n_heads),
                    Int32(self.n_kv_heads),
                    Int32(self.head_dim),
                    Int32(pos0),
                    grid_dim=(self.n_heads, 1, t),
                    block_dim=(attn_t, 1, 1),
                )
            if profile:
                self.ctx.synchronize()
                prof[5] += Float64(monotonic_ns() - prof_t0) / 1e6
                lattn += Float64(monotonic_ns() - prof_t0) / 1e6
                prof_t0 = monotonic_ns()
            if profile:
                self.ctx.synchronize()
                prof[4] += Float64(monotonic_ns() - prof_t0) / 1e6
                lattn += Float64(monotonic_ns() - prof_t0) / 1e6
                prof_t0 = monotonic_ns()
            if profile:
                self.ctx.synchronize()
                prof[3] += Float64(monotonic_ns() - prof_t0) / 1e6
                lattn += Float64(monotonic_ns() - prof_t0) / 1e6
                prof_t0 = monotonic_ns()
            if profile:
                self.ctx.synchronize()
                prof[2] += Float64(monotonic_ns() - prof_t0) / 1e6
                lattn += Float64(monotonic_ns() - prof_t0) / 1e6
                prof_t0 = monotonic_ns()
            if batch:
                self.ctx.enqueue_function[
                    attn_rs_h_bb_gpu[attn_t, 8, True]
                ](
                    self.d_qrot_b.unsafe_ptr(),
                    self.d_k_store_b[layer].unsafe_ptr() + row_off,
                    self.d_scoresb.unsafe_ptr() + score_off,
                    row_pos,
                    Int32(self.n_heads),
                    Int32(self.n_kv_heads),
                    Int32(self.head_dim),
                    Int32(self.max_tokens),
                    grid_dim=(self.n_kv_heads, 1, t),
                    block_dim=(attn_t, 1, 1),
                )
            else:
                comptime if prefill_gqa and attn_tiles == 1 and attn_t % 8 == 0:
                    self.ctx.enqueue_function[
                        attn_rs_h_gpu[attn_t, 8, True]
                    ](
                        self.d_qrot_b.unsafe_ptr(),
                        pk,
                        self.d_scoresb.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.n_kv_heads),
                        Int32(self.head_dim),
                        Int32(kv_len),
                        Int32(pos0),
                        grid_dim=(self.n_kv_heads, 1, t),
                        block_dim=(attn_t, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        RS_K
                    ](
                        self.d_qrot_b.unsafe_ptr(),
                        pk,
                        self.d_scoresb.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.n_kv_heads),
                        Int32(self.head_dim),
                        Int32(kv_len),
                        Int32(pos0),
                        grid_dim=(self.n_heads, attn_tiles, t),
                        block_dim=(attn_t, 1, 1),
                    )
            if batch:
                self.ctx.enqueue_function[
                    attn_vt2_h_bb_gpu[attn_t, 8, attn_tiles, attn_tiles == 1]
                ](
                    self.d_scoresb.unsafe_ptr() + score_off,
                    self.d_v_store_b[layer].unsafe_ptr() + row_off,
                    self.d_vpartb.unsafe_ptr(),
                    self.d_attotb.unsafe_ptr(),
                    row_pos,
                    Int32(self.n_heads),
                    Int32(self.n_kv_heads),
                    Int32(self.head_dim),
                    Int32(self.max_tokens),
                    self.d_attnb.unsafe_ptr(),
                    grid_dim=(self.n_kv_heads, attn_tiles, t),
                    block_dim=(attn_t, 1, 1),
                )
            elif prefill_gqa and attn_t // 8 == self.head_dim and (
                self.n_heads // self.n_kv_heads <= 8
            ):
                comptime if attn_tiles == 1:
                    # 一百一十二：TILES=1 时 COMB 只是 vpart/tsum 缩放，
                    # 并进 VT2 尾写（省每层一发 13.6µs 的核）
                    self.ctx.enqueue_function[
                        attn_vt2_h_gpu[attn_t, 8, attn_tiles, True]
                    ](
                        self.d_scoresb.unsafe_ptr(),
                        pv,
                        self.d_vpartb.unsafe_ptr(),
                        self.d_attotb.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.n_kv_heads),
                        Int32(self.head_dim),
                        Int32(kv_len),
                        Int32(pos0),
                        self.d_attnb.unsafe_ptr(),
                        grid_dim=(self.n_kv_heads, attn_tiles, t),
                        block_dim=(attn_t, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        attn_vt2_h_gpu[attn_t, 8, attn_tiles]
                    ](
                        self.d_scoresb.unsafe_ptr(),
                        pv,
                        self.d_vpartb.unsafe_ptr(),
                        self.d_attotb.unsafe_ptr(),
                        Int32(self.n_heads),
                        Int32(self.n_kv_heads),
                        Int32(self.head_dim),
                        Int32(kv_len),
                        Int32(pos0),
                        self.d_attnb.unsafe_ptr(),
                        grid_dim=(self.n_kv_heads, attn_tiles, t),
                        block_dim=(attn_t, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[
                    VT2_K
                ](
                    self.d_scoresb.unsafe_ptr(),
                    pv,
                    self.d_vpartb.unsafe_ptr(),
                    self.d_attotb.unsafe_ptr(),
                    Int32(self.n_heads),
                    Int32(self.n_kv_heads),
                    Int32(self.head_dim),
                    Int32(kv_len),
                    Int32(pos0),
                    grid_dim=(self.n_heads, attn_tiles, t),
                    block_dim=(attn_t, 1, 1),
                )
            # 一百一十二：GQA 分组 + TILES=1 时 VT2 已尾写 d_attnb，跳过
            if not (
                prefill_gqa and attn_t // 8 == self.head_dim
                and self.n_heads // self.n_kv_heads <= 8 and attn_tiles == 1
            ):
                self.ctx.enqueue_function[
                    COMB_K
                ](
                    self.d_attnb.unsafe_ptr(),
                    self.d_vpartb.unsafe_ptr(),
                    self.d_attotb.unsafe_ptr(),
                    Int32(self.n_heads),
                    Int32(self.head_dim),
                    grid_dim=(self.n_heads, 1, t),
                    block_dim=(attn_t, 1, 1),
                )
            # o：真 GEMM 分块 + ACC 残差（八十九）
            comptime if prefill_af == DType.float32 and prefill_mma_o and (
                prefill_wpack and inner % 16 == 0
            ):
                # o 是残差累加（ACC=True）——几何与 down 同，直接复用
                self.ctx.enqueue_function[x_to_f16_gpu](
                    self.d_attnb.unsafe_ptr(),
                    self.d_attnb16.unsafe_ptr(),
                    Int32(t * self.hidden),
                    grid_dim=((t * self.hidden + 255) // 256, 1, 1),
                    block_dim=(256, 1, 1),
                )
                if prefill_spl and inner % 16 == 0 and (
                    inner // 4
                ) % 224 == 0:
                    # 一百一十三-a：o 同款 split-K（S=4，56→224 块）；
                    # partial 复用 d_downpart 前 4 面（层间顺序无冲突）
                    self.ctx.enqueue_function[
                        f16_mma_down_spl_gpu[inner, 224, 2, 8, 4]
                    ](
                        self.d_attnb16.unsafe_ptr(),
                        self.d_wp_o[layer].unsafe_ptr(),
                        self.d_downpart.unsafe_ptr(),
                        Int32(self.hidden),
                        Int32(t),
                        grid_dim=(ceildiv(self.hidden, 128), ceildiv(t, 16), 4),
                        block_dim=(256, 1, 1),
                    )
                    self.ctx.enqueue_function[f32_adds_gpu[4]](
                        self.d_xb.unsafe_ptr(),
                        self.d_downpart.unsafe_ptr(),
                        Int32(t * self.hidden),
                        grid_dim=((t * self.hidden + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        f16_mma_down_pk_gpu[
                            inner, 896 if prefill_tile and inner >= 896 else 608, 2, 8
                        ]
                    ](
                        self.d_attnb16.unsafe_ptr(),
                        self.d_wp_o[layer].unsafe_ptr(),
                        self.d_xb.unsafe_ptr(),
                        Int32(self.hidden),
                        Int32(t),
                        grid_dim=(ceildiv(self.hidden, 128), ceildiv(t, 16), 1),
                        block_dim=(256, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[
                    O_K
                ](
                    self.d_attnb.unsafe_ptr(),
                    self.d_w16_o[layer].unsafe_ptr(),
                    self.d_xb.unsafe_ptr(),
                    Int32(self.hidden),
                    Int32(t),
                    grid_dim=(ceildiv(self.hidden, 4),
                    ceildiv(t, prefill_rb), 1),
                    block_dim=(o_t, 1, 1),
                )
            if profile:
                self.ctx.synchronize()
                prof[6] += Float64(monotonic_ns() - prof_t0) / 1e6
                lo = Float64(monotonic_ns() - prof_t0) / 1e6
                prof_t0 = monotonic_ns()
            # dual：norm 前置单发 + 真 GEMM 分块（八十九），act 尾同款
            self.ctx.enqueue_function[rmsnorm_gpu](
                self.d_xb.unsafe_ptr(),
                self.d_norms[2 * layer + 1].unsafe_ptr(),
                self.d_xnormb.unsafe_ptr(),
                Int32(inner),
                Int32(t),
                self.eps,
                grid_dim=(t, 1, 1),
                block_dim=(256, 1, 1),
            )
            comptime if prefill_af == DType.float32 and (
                prefill_mma_dual and inner % 16 == 0
            ):
                # 一百零五修正：norm2 覆写 d_xnormb —— dual 的 f16 输入
                # 必须在 norm2 之后重转（norm1 后的转换只服务 q2）
                self.ctx.enqueue_function[x_to_f16_gpu](
                    self.d_xnormb.unsafe_ptr(),
                    self.d_xnormb16.unsafe_ptr(),
                    Int32(t * inner),
                    grid_dim=((t * inner + 255) // 256, 1, 1),
                    block_dim=(256, 1, 1),
                )
                comptime if prefill_wpack:
                    self.ctx.enqueue_function[
                        f16_mma_dual_pk_gpu[
                            inner, 896 if prefill_tile else 608, 8
                        ]
                    ](
                        self.d_xnormb16.unsafe_ptr(),
                        self.d_wp_gate[layer].unsafe_ptr(),
                        self.d_wp_up[layer].unsafe_ptr(),
                        self.d_gateb.unsafe_ptr(),
                        self.d_upb.unsafe_ptr(),
                        self.d_actb.unsafe_ptr(),
                        Int32(self.inter),
                        Int32(t),
                        grid_dim=(ceildiv(self.inter, 128), ceildiv(t, 16), 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        f16_mma_dual_gpu[inner, 608, 8]
                    ](
                        self.d_xnormb16.unsafe_ptr(),
                        self.d_w16_gate[layer].unsafe_ptr(),
                        self.d_w16_up[layer].unsafe_ptr(),
                        self.d_gateb.unsafe_ptr(),
                        self.d_upb.unsafe_ptr(),
                        self.d_actb.unsafe_ptr(),
                        Int32(self.inter),
                        Int32(t),
                        grid_dim=(ceildiv(self.inter, 128), ceildiv(t, 16), 1),
                        block_dim=(256, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[
                    DUAL_K
                ](
                    self.d_xnormb.unsafe_ptr(),
                    self.d_w16_gate[layer].unsafe_ptr(),
                    self.d_w16_up[layer].unsafe_ptr(),
                    self.d_gateb.unsafe_ptr(),
                    self.d_upb.unsafe_ptr(),
                    self.d_actb.unsafe_ptr(),
                    Int32(self.inter),
                    Int32(t),
                    grid_dim=(ceildiv(self.inter, 2), ceildiv(t, prefill_rb), 1),
                    block_dim=(dual_t, 1, 1),
                )
            if profile:
                self.ctx.synchronize()
                prof[7] += Float64(monotonic_ns() - prof_t0) / 1e6
                lff = Float64(monotonic_ns() - prof_t0) / 1e6
                prof_t0 = monotonic_ns()
            # down：真 GEMM 分块 + ACC 残差（八十九）
            comptime if prefill_af == DType.float32 and (
                prefill_mma_down and inner_ff % 16 == 0
            ):
                # 一百零四：down MMA（tensor core，f32 快路径）
                self.ctx.enqueue_function[x_to_f16_gpu](
                    self.d_actb.unsafe_ptr(),
                    self.d_actb16.unsafe_ptr(),
                    Int32(t * self.inter),
                    grid_dim=((t * self.inter + 255) // 256, 1, 1),
                    block_dim=(256, 1, 1),
                )
                comptime if prefill_wpack and prefill_spl and (
                    (inner_ff // 8) % 608 == 0 and inner_ff % 16 == 0
                ):
                    # 一百一十二-b：split-K——partial 裸写 + add4 归并残差
                    self.ctx.enqueue_function[
                        f16_mma_down_spl_gpu[inner_ff, 608, 1, 8, 8]
                    ](
                        self.d_actb16.unsafe_ptr(),
                        self.d_wp_down[layer].unsafe_ptr(),
                        self.d_downpart.unsafe_ptr(),
                        Int32(self.hidden),
                        Int32(t),
                        grid_dim=(
                            ceildiv(self.hidden, 64), ceildiv(t, 16), 8
                        ),
                        block_dim=(256, 1, 1),
                    )
                    self.ctx.enqueue_function[f32_adds_gpu[8]](
                        self.d_xb.unsafe_ptr(),
                        self.d_downpart.unsafe_ptr(),
                        Int32(t * self.hidden),
                        grid_dim=((t * self.hidden + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                elif prefill_wpack:
                    self.ctx.enqueue_function[
                        f16_mma_down_pk_gpu[
                            inner_ff, 608, 1 if prefill_tile else 2, 8
                        ]
                    ](
                        self.d_actb16.unsafe_ptr(),
                        self.d_wp_down[layer].unsafe_ptr(),
                        self.d_xb.unsafe_ptr(),
                        Int32(self.hidden),
                        Int32(t),
                        grid_dim=(ceildiv(
                            self.hidden, 64 if prefill_tile else 128
                        ), ceildiv(t, 16), 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[
                        f16_mma_down_gpu[inner_ff, 608, 2, 8]
                    ](
                        self.d_actb16.unsafe_ptr(),
                        self.d_w16_down[layer].unsafe_ptr(),
                        self.d_xb.unsafe_ptr(),
                        Int32(self.hidden),
                        Int32(t),
                        grid_dim=(ceildiv(self.hidden, 128), ceildiv(t, 16), 1),
                        block_dim=(256, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[
                    DOWN_K
                ](
                    self.d_actb.unsafe_ptr(),
                    self.d_w16_down[layer].unsafe_ptr(),
                    self.d_xb.unsafe_ptr(),
                    Int32(self.hidden),
                    Int32(t),
                    grid_dim=(ceildiv(self.hidden, 4), ceildiv(t, prefill_rb), 1),
                    block_dim=(down_t, 1, 1),
                )
            if profile:
                self.ctx.synchronize()
                prof[8] += Float64(monotonic_ns() - prof_t0) / 1e6
                ldown = Float64(monotonic_ns() - prof_t0) / 1e6
            if profile:
                print("层", layer, "tri:", ltri, "attn:", lattn, "o:", lo, "ff:", lff, "down:", ldown)
            # 账本九十七/九十八：层间功耗余量窗——连续满载在机箱热压下触发
            # A100 功耗墙降频（765/1410MHz = 0.54 ≈ 慢/快 kernel 倍率 1.84）。
            # 每 layer_gap_every 层末 sync+usleep(layer_gap_us)；默认 0=关闭
            # （安静环境连续执行即最优 22.3ms）。热环境推荐 (4,300)/(2,30)：
            # 实测 40.6→23.3ms。安静环境开启反而略亏（23.3 vs 22.3）。
            if layer_gap_us > 0 and (layer + 1) % layer_gap_every == 0:
                self.ctx.synchronize()
                _ = external_call["usleep", Int32](Int32(layer_gap_us))
                prof_t0 = monotonic_ns()

        if batch:
            # 一百二十：B 行终 norm + B 行 lm_head + 行 argmax
            self.ctx.enqueue_function[rmsnorm_gpu](
                self.d_xb.unsafe_ptr(),
                self.d_norms[2 * self.n_layers].unsafe_ptr(),
                self.d_normedb.unsafe_ptr(),
                Int32(self.hidden),
                Int32(t),
                self.eps,
                grid_dim=(t, 1, 1),
                block_dim=(256, 1, 1),
            )
            self.ctx.enqueue_function[LMHEAD_K](
                self.d_normedb.unsafe_ptr(),
                self.d_embed16.unsafe_ptr(),
                self.d_logits_b.unsafe_ptr(),
                self.d_ws.unsafe_ptr(),
                self.eps,
                Int32(self.vocab),
                grid_dim=(ceildiv(self.vocab, 8), t, 1),
                block_dim=(256, 1, 1),
            )
            if per_row_sampling:
                self.ctx.enqueue_copy(
                    dst_buf=self.d_stemp, src_buf=self.h_stemp
                )
                self.ctx.enqueue_copy(
                    dst_buf=self.d_stopps, src_buf=self.h_stopps
                )
                self.ctx.enqueue_copy(
                    dst_buf=self.d_sseeds, src_buf=self.h_sseeds
                )
                self.ctx.enqueue_function[sample_topp_row_gpu](
                    self.d_logits_b.unsafe_ptr(),
                    self.d_tokout.unsafe_ptr(),
                    Int32(self.vocab),
                    self.d_stemp.unsafe_ptr(),
                    self.d_stopps.unsafe_ptr(),
                    self.d_sseeds.unsafe_ptr(),
                    grid_dim=(t, 1, 1),
                    block_dim=(256, 1, 1),
                )
            elif sample_temp > Float32(0):
                if sample_topp < Float32(1):
                    self.ctx.enqueue_function[sample_topp_gpu](
                        self.d_logits_b.unsafe_ptr(),
                        self.d_tokout.unsafe_ptr(),
                        Int32(self.vocab),
                        sample_temp,
                        sample_topp,
                        sample_seed,
                        grid_dim=(t, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    self.ctx.enqueue_function[sample_rb_gpu](
                        self.d_logits_b.unsafe_ptr(),
                        self.d_tokout.unsafe_ptr(),
                        Int32(self.vocab),
                        sample_temp,
                        sample_seed,
                        grid_dim=(t, 1, 1),
                        block_dim=(256, 1, 1),
                    )
            else:
                self.ctx.enqueue_function[argmax_rb_gpu](
                    self.d_logits_b.unsafe_ptr(),
                    self.d_tokout.unsafe_ptr(),
                    Int32(self.vocab),
                    grid_dim=(t, 1, 1),
                    block_dim=(256, 1, 1),
                )
            if copy_logits:
                self.ctx.enqueue_copy(
                    dst_buf=self.h_logits_b, src_buf=self.d_logits_b
                )
            self.ctx.enqueue_copy(dst_buf=self.h_tokout, src_buf=self.d_tokout)
        else:
            # 末行终 norm + lm_head（只算末行 logits）
            self.ctx.enqueue_function[rmsnorm_gpu](
                self.d_xb.unsafe_ptr() + (t - 1) * self.hidden,
                self.d_norms[2 * self.n_layers].unsafe_ptr(),
                self.d_normed.unsafe_ptr(),
                Int32(self.hidden),
                Int32(1),
                self.eps,
                grid_dim=(1, 1, 1),
                block_dim=(256, 1, 1),
            )
            self.ctx.enqueue_function[LMHEAD_K](
                self.d_normed.unsafe_ptr(),
                self.d_embed16.unsafe_ptr(),
                self.d_logits.unsafe_ptr(),
                self.d_ws.unsafe_ptr(),
                self.eps,
                Int32(self.vocab),
                grid_dim=(ceildiv(self.vocab, 8), 1, 1),
                block_dim=(256, 1, 1),
            )
            self.ctx.enqueue_copy(dst_buf=self.h_logits, src_buf=self.d_logits)
        if profile:
            self.ctx.synchronize()
            prof[9] += Float64(monotonic_ns() - prof_t0) / 1e6
            prof_t0 = monotonic_ns()
        if sync_tail:
            self.ctx.synchronize()
        if profile:
            for pi in range(10):
                self.last_prof[pi] = prof[pi]
            print("prof gather:", prof[0], "ms")
            print("prof tri:", prof[1], "ms")
            print("prof kvs:", prof[2], "ms")
            print("prof scores:", prof[3], "ms")
            print("prof vt2:", prof[4], "ms")
            print("prof comb:", prof[5], "ms")
            print("prof o:", prof[6], "ms")
            print("prof ff:", prof[7], "ms")
            print("prof down:", prof[8], "ms")
            print("prof tail:", prof[9], "ms")
        if batch:
            for i in range(t):
                self.h_posarr[i] = self.h_posarr[i] + 1
            return self.h_logits_b.unsafe_ptr()
        self.kv_pos = kv_len
        return self.h_logits.unsafe_ptr()
