"""CUDA kernel 与标量后端的**逐值差分**门 —— **只能在 A100 远程验证机上运行**。

本地开发机是 Maxwell sm_52，现代 CUDA 栈与 MAX 均不支持，因此本文件**故意不加入
`pixi run test`**（本地跑必然失败，会污染能力账本的信号）。运行方式：

    ./scripts/a100.sh gpu                          # 先挑一张空闲卡（共享机）
    ./scripts/a100.sh run 4 tests/gpu/test_cuda_diff.mojo

覆盖十二项：`rmsnorm`、`linear`、`attention`、`attention_par`、`q4_matmul`（相对容差
1e-5）、`dequant_q4_0` 与 `dequant_q8_0`（**零容差**）。attention 的差分只面对一个新变量：exp 的 ulp 差
（libdevice vs libm），门把实际差值打出来记录量级。

本门**只验正确性、不报性能**（账本 §4 给这条路线定的解锁条件③）：GPU 与标量后端算
**同一个**输入，逐值比 fp32 **相对**容差 1e-5 —— 与 AVX2 门同一个数，不是「GPU 可以
宽一点」。

2026-11-09 实测结果（A100，卡 0，`mojo run -I src`）：
    *** RMSNORM   GPU/SCALAR DIFF PASS *** diff= 0.0  tol= 5.0350770950317384e-05
    *** DEQUANT Q4_0 GPU/SCALAR DIFF PASS *** diff= 0.0
    *** DEQUANT Q8_0 GPU/SCALAR DIFF PASS *** diff= 0.0
    *** LINEAR    GPU/SCALAR DIFF PASS *** diff= 0.0  tol= 0.024
    *** ATTENTION GPU/SCALAR DIFF PASS *** diff= 1.1920928955078125e-07  tol= 2.0e-05
    *** ATTENTION_PAR GPU/SCALAR DIFF PASS *** diff= 1.1920928955078125e-07  tol= 2.0e-05
    *** Q4_MATMUL GPU/REFERENCE DIFF PASS *** diff= 0.0  tol= 0.026
    *** ADD/COPY   GPU/SCALAR DIFF PASS *** diff= 0.0 (zero tolerance)
    *** SWIGLU     GPU/SCALAR DIFF PASS *** diff= 0.0  tol= 1.3e-05
    *** ROPE       GPU/SCALAR DIFF PASS *** diff= 0.0 (zero, no FMA contraction)
    *** GATHER     GPU/REFERENCE DIFF PASS *** diff= 0.0 (zero tolerance)
    （rope 的 `a*c - b*s` 没有被 FMA 折叠 —— 零容差站住了；swiglu 的 exp 路径
    在此数据上逐位一致。两个工程教训记在账本六十一：整门共享一个
    DeviceContext（每 check 各建会在 ~10 个上下文处崩）；`arena.keep_alive()`
    是生命周期标记，放在读取之前 = use-after-free，症状是 AsyncRT 栈里的
    段错误，极具误导性。）
    （前四个与标量**逐位一致** —— linear 的 diff=0.0 说明 GPU 编译器没把乘加
    折叠成改变舍入的 FMA。attention 的 diff = 1.19e-7 = 恰好 1 个 fp32 ulp：
    kernel 与标量逐句同构，唯一下游差异是 exp（libdevice vs libm）—— 这是门里
    第一个非零差分，量级 1 ulp，远低于 1e-5 容差。注意远程直跑必须带 `-I src`，
    否则 `alofa.verify` 会报 "does not refer to a nested package"。另外 `ref`
    是 Mojo 关键字，变量名避开。）

为什么 kernel 里一个线程要独立算完一整行（看起来很浪费）：本门只想让「GPU 算的数与
CPU 算的数是同一个」这句话成立。写成块内规约会引入**共享内存**和**浮点累加顺序**
两个额外的正确性变量，一旦数值对不上就分不清是布局读错还是规约写错。等这条路走通
了再谈性能内核 —— 那时换的是实现，不是这个门的判据。
"""

from std.gpu import global_idx
from std.math import ceildiv, sqrt

from max.gpu.host import DeviceContext, HostBuffer
from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.tensor import Shape, TensorView, f32_data, rows_view
from alofa.kernels.cpu.quant import (
    Q4_BLOCK,
    Q4_BYTES,
    Q8_BLOCK,
    Q8_BYTES,
    dequant_q4_0,
    dequant_q8_0,
    quantize_q4_0,
    quantize_q8_0,
)
from alofa.kernels.cpu.scalar import add, attention, linear, rmsnorm, rope, swiglu
from alofa.kernels.cuda.attention import attention_gpu, attention_par_gpu
from alofa.kernels.cuda.basic import (
    dequant_q4_0_gpu,
    dequant_q8_0_gpu,
    linear_gpu,
    linear_block_gpu,
    linear_fat_f16_gpu,
    rmsnorm_gpu,
)
from alofa.kernels.cuda.elementwise import (
    q4_dequant_f16_gpu,
    add_gpu,
    add_into_gpu,
    copy2_gpu,
    copy_gpu,
    gather_gpu,
    rope2_gpu,
    rope_gpu,
    rope_kv_gpu,
    swiglu_gpu,
)
from alofa.kernels.cuda.matmul import f16_matmul_dual_gpu, f16_matmul_fat_gpu, f16_matmul_qkv_gpu, q4_matmul_dual_fat_gpu, q4_matmul_fused_gpu, q4_matmul_qkv_gpu, q4_matmul_smem_gpu
from std.math import abs, exp
from alofa.verify.compare import max_abs, max_abs_diff

comptime DT = DType.float32
comptime ROWS = 8
comptime COLS = 896
# q4_0 反量化差分的规模：8 个块（256 个值），块结构与 `kernels/cpu/quant.mojo` 一致。
comptime Q4_BLOCKS = 8
comptime Q4_N = Q4_BLOCKS * Q4_BLOCK
# q8_0 差分：同样 8 个块（256 个值）。
comptime Q8_BLOCKS = 8
comptime Q8_N = Q8_BLOCKS * Q8_BLOCK
# linear 差分的规模：Qwen 0.5B 的 hidden=896，输出列取 16（差分门不追规模）。
comptime LM_ROWS = 4
comptime LM_INNER = 896
comptime LM_OUT = 16
# attention 差分：小而全 —— GQA 分组（2 查询头 : 1 kv 头）、q_len < kv_len
#（覆盖因果掩码与 decode 位置公式）、head_dim 取 16 让点积链够长。
comptime ATTN_HEADS = 2
comptime ATTN_KV_HEADS = 1
comptime ATTN_HEAD_DIM = 16
comptime ATTN_Q_LEN = 3
comptime ATTN_KV_LEN = 5
# q4 matmul 差分：产品冠军 kernel（smem 版）vs 产品 dequant + 产品 linear 的
# 组合参照（AVX2 的 SIMD 通道序会影响舍入，不作参照）。inner 取 896 对齐
# Qwen 形状且是 32 的倍数（块永不跨行，平铺量化即行主块流）。
comptime Q4MM_ROWS = 2
comptime Q4MM_INNER = 896
comptime Q4MM_OUT = 16
# 逐元素小算子差分的规模：小向量即可（判据是零容差/1e-5，不追规模）。
comptime EW_N = 64
comptime COPY_N = 100
comptime ROPE_TOKENS = 2
comptime ROPE_Q_COLS = 32
comptime ROPE_K_COLS = 16
comptime ROPE_HEAD_DIM = 16
comptime GATHER_TOKENS = 3
comptime GATHER_COLS = 32
comptime GATHER_VOCAB = 7
comptime EPS = Float32(1e-6)
comptime BLOCK = 64
# 与 AVX2 门**同一个**容差。
comptime REL_TOLERANCE = Float64(1e-5)
# f16 权重舍入判据（账本六十八）：每权重 rel err ≤ 2^-11，累积后对输出
# 的相对界实测定标 —— 先用此初值跑，按实测调整。
comptime F16_REL = Float64(0.05)


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, Shape(dims), DT_FP32)


def fill(t: TensorView, seed: Int) raises -> None:
    """确定性的、不整齐的输入。

    不用随机数，也不取 0/1 这类特殊值：这个门的判据是数值，输入本身必须可复现，
    且要能撑起 `ref_max` 的规模（全零输入会让相对容差退化成绝对容差，门形同虚设）。
    """
    var p = f32_data(t)
    for i in range(t.numel()):
        var k = i + seed * 7919
        p[unsafe_offset=i] = Float32((k % 251) - 125) * Float32(0.023)


def rmsnorm_diff_check(mut ctx: DeviceContext) raises:
    var arena = Arena(1 << 20)
    var x = scratch(arena, ROWS, COLS)
    var w = scratch(arena, 1, COLS)
    var dst_cpu = scratch(arena, ROWS, COLS)
    fill(x, 1)
    fill(w, 2)

    # 参照：CPU 标量后端（不是另写一份，是产品代码本身）
    rmsnorm(dst_cpu, x, w, EPS)

    var hx = ctx.enqueue_create_host_buffer[DT](ROWS * COLS)
    var hw = ctx.enqueue_create_host_buffer[DT](COLS)
    var hd = ctx.enqueue_create_host_buffer[DT](ROWS * COLS)
    ctx.synchronize()
    var px = f32_data(x)
    var pw = f32_data(w)
    for i in range(ROWS * COLS):
        hx[i] = px[unsafe_offset=i]
    for i in range(COLS):
        hw[i] = pw[unsafe_offset=i]

    var dx = ctx.enqueue_create_buffer[DT](ROWS * COLS)
    var dw = ctx.enqueue_create_buffer[DT](COLS)
    var dd = ctx.enqueue_create_buffer[DT](ROWS * COLS)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)

    ctx.enqueue_function[rmsnorm_gpu](
        dx.unsafe_ptr(),
        dw.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(COLS),
        Int32(ROWS),
        EPS,
        # 块内规约版（账本六十四）：一行一个 block、256 线程。树形归约与
        # 标量的顺序求和有合法重组差 —— 本检查本就是相对容差，覆盖。
        grid_dim=(ROWS, 1, 1),
        block_dim=(256,),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var n = ROWS * COLS
    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(dst_cpu), n)
    var scale = max_abs(f32_data(dst_cpu), n)
    if scale < 1.0:
        scale = 1.0
    var tol = REL_TOLERANCE * scale
    # arena 必须活到上面这些指针都不再用为止：Mojo 在值的**最后一次使用处**析构，而
    # `arena` 变量的最后一次使用是第三个 `scratch(...)` —— 比 `fill` 还早。不在这里
    # 声明，`Arena.__deinit__` 会在 `fill(x, 1)` 之前就把整块映射 `munmap` 掉，于是
    # 第一个写就落到已释放的地址上（实测：段错误，栈指向 `fill`，看起来像 GPU 的锅）。
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "rmsnorm: gpu vs scalar max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    # 参照张量若全为零，上面的相对容差会退化成绝对容差，门就形同虚设。
    if scale <= 0.0:
        raise Error("rmsnorm: the reference tensor is all zeros")
    print("*** RMSNORM GPU/SCALAR DIFF PASS *** diff=", diff, " tol=", tol)


def dequant_q4_diff_check(mut ctx: DeviceContext) raises:
    """q4_0 反量化的 GPU/标量差分，判据是**零容差**。

    为什么与 rmsnorm 不同：反量化结果只由 nibble 与缩放因子决定，两者都是精确
    的（`kernels/cpu/quant.mojo` 的 docstring 说得更狠 —— 用容差反而是放水，会把
    "半字节高低位装反"这类错伪装成"舍入差一点"）。差一个 ULP 就是错，没有第二句话。
    """
    var arena = Arena(1 << 20)
    var n_values = Int(Q4_N)
    var n_bytes = Int(Q4_BLOCKS) * Int(Q4_BYTES)
    var src_view = scratch(arena, 1, n_values)
    fill(src_view, 3)
    var blocks = arena.alloc(n_bytes)
    var ref_out = arena.alloc(n_values * 4).unsafe_bitcast[Float32]()
    # 块从**产品代码的量化器**来：顺带验了 GPU 读的块布局与量化器写的一致。
    quantize_q4_0(blocks, f32_data(src_view), n_values)
    # 参照：CPU 标量反量化（不是另写一份，是产品代码本身）。
    dequant_q4_0(ref_out, blocks, n_values)

    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var hd = ctx.enqueue_create_host_buffer[DT](n_values)
    ctx.synchronize()
    for i in range(n_bytes):
        hb[i] = blocks[unsafe_offset=i]

    var db = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var dd = ctx.enqueue_create_buffer[DT](n_values)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.enqueue_function[dequant_q4_0_gpu](
        db.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(Q4_BLOCKS),
        grid_dim=(Int(Q4_BLOCKS), 1, 1),
        block_dim=(Int(Q4_BLOCK),),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var diff = Float64(0)
    for i in range(n_values):
        var d = Float64(hd[i]) - Float64(ref_out[unsafe_offset=i])
        if d < 0.0:
            d = -d
        if d > diff:
            diff = d
    # 与 rmsnorm 同一个坑：arena 的最后一次使用在上面 alloc 处，不 keep_alive 会被
    # 提前 munmap（见 rmsnorm_diff_check 里的注释）。
    arena.keep_alive()
    if diff != 0.0:
        raise Error(
            "dequant_q4_0: gpu vs scalar diff=" + String(diff) + " (zero tolerance)"
        )
    print("*** DEQUANT Q4_0 GPU/SCALAR DIFF PASS *** diff=", diff)


def dequant_q8_diff_check(mut ctx: DeviceContext) raises:
    """q8_0 反量化的 GPU/标量差分，判据与 q4_0 一样是**零容差**。"""
    var arena = Arena(1 << 20)
    var n_values = Int(Q8_BLOCKS) * Int(Q8_BLOCK)
    var n_bytes = Int(Q8_BLOCKS) * Int(Q8_BYTES)
    var src_view = scratch(arena, 1, n_values)
    fill(src_view, 6)
    var blocks = arena.alloc(n_bytes)
    var ref_out = arena.alloc(n_values * 4).unsafe_bitcast[Float32]()
    quantize_q8_0(blocks, f32_data(src_view), n_values)
    dequant_q8_0(ref_out, blocks, n_values)

    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var hd = ctx.enqueue_create_host_buffer[DT](n_values)
    ctx.synchronize()
    for i in range(n_bytes):
        hb[i] = blocks[unsafe_offset=i]
    var db = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var dd = ctx.enqueue_create_buffer[DT](n_values)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.enqueue_function[dequant_q8_0_gpu](
        db.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(Q8_BLOCKS),
        grid_dim=(Int(Q8_BLOCKS), 1, 1),
        block_dim=(Int(Q8_BLOCK),),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var diff = Float64(0)
    for i in range(n_values):
        var d = Float64(hd[i]) - Float64(ref_out[unsafe_offset=i])
        if d < 0.0:
            d = -d
        if d > diff:
            diff = d
    arena.keep_alive()
    if diff != 0.0:
        raise Error(
            "dequant_q8_0: gpu vs scalar diff=" + String(diff) + " (zero tolerance)"
        )
    print("*** DEQUANT Q8_0 GPU/SCALAR DIFF PASS *** diff=", diff)


def linear_diff_check(mut ctx: DeviceContext) raises:
    """linear（无偏置 GEMM）的 GPU/标量差分，相对容差与 rmsnorm 同一个数。

    kernel 与 `_gemm` 同一个累加顺序（Float64、k 升序、末尾一次 Float32），预期
    也是逐位一致；判据仍用 1e-5 相对容差 —— 它是「同一算法」的门，不是「同一
    指令序列」的门，若某天 GPU 编译器把乘加折叠成 FMA 使差分变非零，门会把
    差值打出来让人看，而不是默默放行。
    """
    var arena = Arena(1 << 21)
    var x = scratch(arena, LM_ROWS, LM_INNER)
    var w = scratch(arena, LM_OUT, LM_INNER)
    var dst_cpu = scratch(arena, LM_ROWS, LM_OUT)
    fill(x, 4)
    fill(w, 5)
    # 参照：CPU 标量 `linear`（产品代码本身，`w` 按 [out, in] 存）。
    linear(dst_cpu, x, w)

    var n_x = Int(LM_ROWS) * Int(LM_INNER)
    var n_w = Int(LM_OUT) * Int(LM_INNER)
    var n_d = Int(LM_ROWS) * Int(LM_OUT)
    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hw = ctx.enqueue_create_host_buffer[DT](n_w)
    var hd = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    var px = f32_data(x)
    var pw = f32_data(w)
    for i in range(n_x):
        hx[i] = px[unsafe_offset=i]
    for i in range(n_w):
        hw[i] = pw[unsafe_offset=i]

    var dx = ctx.enqueue_create_buffer[DT](n_x)
    var dw = ctx.enqueue_create_buffer[DT](n_w)
    var dd = ctx.enqueue_create_buffer[DT](n_d)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.enqueue_function[linear_gpu](
        dx.unsafe_ptr(),
        dw.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(LM_INNER),
        Int32(LM_OUT),
        grid_dim=(Int(LM_OUT), Int(LM_ROWS), 1),
        block_dim=(1, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var n = n_d
    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(dst_cpu), n)
    var scale = max_abs(f32_data(dst_cpu), n)
    if scale < 1.0:
        scale = 1.0
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "linear: gpu vs scalar max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    if scale <= 0.0:
        raise Error("linear: the reference tensor is all zeros")
    print("*** LINEAR GPU/SCALAR DIFF PASS *** diff=", diff, " tol=", tol)


def attention_diff_check(mut ctx: DeviceContext) raises:
    """因果 GQA attention 全函数的 GPU/标量差分。

    参照是产品 `attention`（不是拆出来的 softmax）—— 差分对象与要接进模型的
    函数是同一个。kernel 与标量逐句同构（串行三遍、Float64 累加、同一掩码），
    所以差值只能来自 exp 的实现差：门把实际差值打出来，若是 0.0 则连 exp 都
    逐位一致，若非零则这个数字就是 libdevice vs libm 的量级记录。
    """
    var arena = Arena(1 << 20)
    var q_cols = Int(ATTN_HEADS) * Int(ATTN_HEAD_DIM)
    var kv_cols = Int(ATTN_KV_HEADS) * Int(ATTN_HEAD_DIM)
    var q = scratch(arena, Int(ATTN_Q_LEN), q_cols)
    var k = scratch(arena, Int(ATTN_KV_LEN), kv_cols)
    var v = scratch(arena, Int(ATTN_KV_LEN), kv_cols)
    var scores = scratch(arena, Int(ATTN_Q_LEN), Int(ATTN_KV_LEN))
    var dst_cpu = scratch(arena, Int(ATTN_Q_LEN), q_cols)
    fill(q, 11)
    fill(k, 12)
    fill(v, 13)
    # 参照：产品 scalar attention（无偏置、全头、因果掩码、GQA 分组）。
    attention(
        dst_cpu, q, k, v, scores, Int(ATTN_HEADS), Int(ATTN_KV_HEADS), Int(ATTN_HEAD_DIM)
    )

    var n_q = Int(ATTN_Q_LEN) * q_cols
    var n_kv = Int(ATTN_KV_LEN) * kv_cols
    var n_s = Int(ATTN_Q_LEN) * Int(ATTN_KV_LEN)
    var hq = ctx.enqueue_create_host_buffer[DT](n_q)
    var hk = ctx.enqueue_create_host_buffer[DT](n_kv)
    var hv = ctx.enqueue_create_host_buffer[DT](n_kv)
    var hs = ctx.enqueue_create_host_buffer[DT](n_s)
    var hd = ctx.enqueue_create_host_buffer[DT](n_q)
    ctx.synchronize()
    for i in range(n_q):
        hq[i] = f32_data(q)[unsafe_offset=i]
    for i in range(n_kv):
        hk[i] = f32_data(k)[unsafe_offset=i]
        hv[i] = f32_data(v)[unsafe_offset=i]

    var dq = ctx.enqueue_create_buffer[DT](n_q)
    var dk = ctx.enqueue_create_buffer[DT](n_kv)
    var dv = ctx.enqueue_create_buffer[DT](n_kv)
    var ds = ctx.enqueue_create_buffer[DT](n_s)
    var dd = ctx.enqueue_create_buffer[DT](n_q)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
    ctx.synchronize()
    ctx.enqueue_function[attention_gpu](
        dd.unsafe_ptr(),
        dq.unsafe_ptr(),
        dk.unsafe_ptr(),
        dv.unsafe_ptr(),
        ds.unsafe_ptr(),
        Int32(ATTN_HEADS),
        Int32(ATTN_KV_HEADS),
        Int32(ATTN_HEAD_DIM),
        Int32(ATTN_Q_LEN),
        Int32(ATTN_KV_LEN),
        grid_dim=(Int(ATTN_HEADS), Int(ATTN_Q_LEN), 1),
        block_dim=(1, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var n = n_q
    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(dst_cpu), n)
    var scale = max_abs(f32_data(dst_cpu), n)
    if scale <= 0.0:
        raise Error("attention: the reference tensor is all zeros")
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "attention: gpu vs scalar max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    print("*** ATTENTION GPU/SCALAR DIFF PASS *** diff=", diff, " tol=", tol)


def attention_par_diff_check(mut ctx: DeviceContext) raises:
    """attention 并行版（块内规约）对产品标量 `attention` 的差分。

    顺序变量清单（相对串行版/标量）：scores 逐位一致（单 j 内 d 升序同序）；
    max 规约无舍入、顺序自由；total 树规约是唯一新增舍入变量；V 累加每线程
    一个 d、j 升序同序。加上已知的 expf 1 ulp，预期差值仍在个位 ulp 量级。
    """
    var arena = Arena(1 << 20)
    var q_cols = Int(ATTN_HEADS) * Int(ATTN_HEAD_DIM)
    var kv_cols = Int(ATTN_KV_HEADS) * Int(ATTN_HEAD_DIM)
    var q = scratch(arena, Int(ATTN_Q_LEN), q_cols)
    var k = scratch(arena, Int(ATTN_KV_LEN), kv_cols)
    var v = scratch(arena, Int(ATTN_KV_LEN), kv_cols)
    var scores = scratch(arena, Int(ATTN_Q_LEN), Int(ATTN_KV_LEN))
    var dst_cpu = scratch(arena, Int(ATTN_Q_LEN), q_cols)
    fill(q, 11)
    fill(k, 12)
    fill(v, 13)
    attention(
        dst_cpu, q, k, v, scores, Int(ATTN_HEADS), Int(ATTN_KV_HEADS), Int(ATTN_HEAD_DIM)
    )

    var n_q = Int(ATTN_Q_LEN) * q_cols
    var n_kv = Int(ATTN_KV_LEN) * kv_cols
    var n_s = Int(ATTN_Q_LEN) * Int(ATTN_KV_LEN)
    var hq = ctx.enqueue_create_host_buffer[DT](n_q)
    var hk = ctx.enqueue_create_host_buffer[DT](n_kv)
    var hv = ctx.enqueue_create_host_buffer[DT](n_kv)
    var hd = ctx.enqueue_create_host_buffer[DT](n_q)
    ctx.synchronize()
    for i in range(n_q):
        hq[i] = f32_data(q)[unsafe_offset=i]
    for i in range(n_kv):
        hk[i] = f32_data(k)[unsafe_offset=i]
        hv[i] = f32_data(v)[unsafe_offset=i]

    var dq = ctx.enqueue_create_buffer[DT](n_q)
    var dk = ctx.enqueue_create_buffer[DT](n_kv)
    var dv = ctx.enqueue_create_buffer[DT](n_kv)
    var ds = ctx.enqueue_create_buffer[DT](n_s)
    var dd = ctx.enqueue_create_buffer[DT](n_q)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
    ctx.synchronize()
    ctx.enqueue_function[attention_par_gpu[32]](
        dd.unsafe_ptr(),
        dq.unsafe_ptr(),
        dk.unsafe_ptr(),
        dv.unsafe_ptr(),
        ds.unsafe_ptr(),
        Int32(ATTN_HEADS),
        Int32(ATTN_KV_HEADS),
        Int32(ATTN_HEAD_DIM),
        Int32(ATTN_Q_LEN),
        Int32(ATTN_KV_LEN),
        grid_dim=(Int(ATTN_HEADS), Int(ATTN_Q_LEN), 1),
        block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var n = n_q
    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(dst_cpu), n)
    var scale = max_abs(f32_data(dst_cpu), n)
    if scale <= 0.0:
        raise Error("attention_par: the reference tensor is all zeros")
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "attention_par: gpu vs scalar max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    print("*** ATTENTION_PAR GPU/SCALAR DIFF PASS *** diff=", diff, " tol=", tol)


def q4_matmul_diff_check(mut ctx: DeviceContext) raises:
    """产品 q4 matmul 冠军 kernel（smem 版）的组合参照差分。

    参照 = `quantize_q4_0` → `dequant_q4_0`（产品量化往返）→ 标量 `linear`。
    不用 AVX2 当参照：SIMD 通道序影响舍入顺序，浮点逐位对不上；组合参照与
    GPU kernel 共享同一套 dequant 语义，差值只面对「融合 matmul 的累加」。
    bench（A100）里此检查长期 diff=0.0，门沿用同一判据。
    """
    var arena = Arena(1 << 21)
    var n_w = Int(Q4MM_OUT) * Int(Q4MM_INNER)
    var n_x = Int(Q4MM_ROWS) * Int(Q4MM_INNER)
    var n_d = Int(Q4MM_ROWS) * Int(Q4MM_OUT)
    var n_blocks = n_w // 32
    var n_bytes = n_blocks * 18
    var w_flat = scratch(arena, 1, n_w)
    var x = scratch(arena, Int(Q4MM_ROWS), Int(Q4MM_INNER))
    fill(w_flat, 21)
    fill(x, 22)
    var blocks = arena.alloc(n_bytes)
    var wf_raw = arena.alloc(n_w * 4)
    quantize_q4_0(blocks, f32_data(w_flat), n_w)
    dequant_q4_0(wf_raw.unsafe_bitcast[Float32](), blocks, n_w)
    var ref_raw = arena.alloc(n_d * 4)
    linear(
        rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT)),
        x,
        rows_view(wf_raw, Int(Q4MM_OUT), Int(Q4MM_INNER)),
    )

    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hw = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var hd = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    for i in range(n_x):
        hx[i] = f32_data(x)[unsafe_offset=i]
    for i in range(n_bytes):
        hw[i] = blocks[unsafe_offset=i]
    var dx = ctx.enqueue_create_buffer[DT](n_x)
    var dw = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var dd = ctx.enqueue_create_buffer[DT](n_d)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()
    ctx.enqueue_function[q4_matmul_smem_gpu[32, Q4MM_INNER]](
        dx.unsafe_ptr(),
        dw.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(Q4MM_INNER),
        Int32(Q4MM_OUT),
        grid_dim=(Int(Q4MM_OUT), Int(Q4MM_ROWS), 1),
        block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    var scale = max_abs(f32_data(rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    if scale <= 0.0:
        raise Error("q4_matmul: the reference tensor is all zeros")
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "q4_matmul: gpu vs reference max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    print("*** Q4_MATMUL GPU/REFERENCE DIFF PASS *** diff=", diff, " tol=", tol)




def rope2_diff_check(mut ctx: DeviceContext) raises:
    """rope2（q/k 合一发射）的零容差差分 —— 与 rope_gpu 逐元素同算术。

    覆盖两段平铺索引的分支：q 段与 k 段各自对照标量 rope 的输出。
    """
    var arena = Arena(1 << 18)
    var q = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_Q_COLS))
    var k = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_K_COLS))
    var cos = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_HEAD_DIM))
    var sin = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_HEAD_DIM))
    var oq_cpu = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_Q_COLS))
    var ok_cpu = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_K_COLS))
    fill(q, 41)
    fill(k, 42)
    fill(cos, 43)
    fill(sin, 44)
    rope(oq_cpu, ok_cpu, q, k, cos, sin, Int(ROPE_HEAD_DIM))

    var n_q = Int(ROPE_TOKENS) * Int(ROPE_Q_COLS)
    var n_k = Int(ROPE_TOKENS) * Int(ROPE_K_COLS)
    var n_t = Int(ROPE_TOKENS) * Int(ROPE_HEAD_DIM)
    var hq = ctx.enqueue_create_host_buffer[DT](n_q)
    var hk = ctx.enqueue_create_host_buffer[DT](n_k)
    var hc = ctx.enqueue_create_host_buffer[DT](n_t)
    var hs = ctx.enqueue_create_host_buffer[DT](n_t)
    var hqo = ctx.enqueue_create_host_buffer[DT](n_q)
    var hko = ctx.enqueue_create_host_buffer[DT](n_k)
    ctx.synchronize()
    for i in range(n_q):
        hq[i] = f32_data(q)[unsafe_offset=i]
    for i in range(n_k):
        hk[i] = f32_data(k)[unsafe_offset=i]
    for i in range(n_t):
        hc[i] = f32_data(cos)[unsafe_offset=i]
        hs[i] = f32_data(sin)[unsafe_offset=i]
    var dq = ctx.enqueue_create_buffer[DT](n_q)
    var dk = ctx.enqueue_create_buffer[DT](n_k)
    var dc = ctx.enqueue_create_buffer[DT](n_t)
    var ds = ctx.enqueue_create_buffer[DT](n_t)
    var dqo = ctx.enqueue_create_buffer[DT](n_q)
    var dko = ctx.enqueue_create_buffer[DT](n_k)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dc, src_buf=hc)
    ctx.enqueue_copy(dst_buf=ds, src_buf=hs)
    ctx.synchronize()
    var q_pairs = Int(ROPE_TOKENS) * (Int(ROPE_Q_COLS) // Int(ROPE_HEAD_DIM)) * (Int(ROPE_HEAD_DIM) // 2)
    var k_pairs = Int(ROPE_TOKENS) * (Int(ROPE_K_COLS) // Int(ROPE_HEAD_DIM)) * (Int(ROPE_HEAD_DIM) // 2)
    ctx.enqueue_function[rope2_gpu](
        dqo.unsafe_ptr(), dko.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(),
        dc.unsafe_ptr(), ds.unsafe_ptr(),
        Int32(ROPE_Q_COLS), Int32(ROPE_K_COLS), Int32(ROPE_HEAD_DIM), Int32(ROPE_TOKENS),
        grid_dim=(ceildiv(q_pairs + k_pairs, 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hqo, src_buf=dqo)
    ctx.enqueue_copy(dst_buf=hko, src_buf=dko)
    ctx.synchronize()

    for i in range(n_q):
        if hqo[i] != f32_data(oq_cpu)[unsafe_offset=i]:
            raise Error(
                "rope2(q): element " + String(i) + " gpu=" + String(hqo[i])
                + " cpu=" + String(f32_data(oq_cpu)[unsafe_offset=i])
            )
    for i in range(n_k):
        if hko[i] != f32_data(ok_cpu)[unsafe_offset=i]:
            raise Error(
                "rope2(k): element " + String(i) + " gpu=" + String(hko[i])
                + " cpu=" + String(f32_data(ok_cpu)[unsafe_offset=i])
            )
    arena.keep_alive()
    print("*** ROPE2 GPU/SCALAR DIFF PASS *** diff= 0.0 (zero tolerance)")


def copy2_diff_check(mut ctx: DeviceContext) raises:
    """copy2（KV append k/v 合一发射）的零容差差分。"""
    var arena = Arena(1 << 16)
    var n = 512
    var ks = scratch(arena, 1, n)
    var vs = scratch(arena, 1, n)
    fill(ks, 7)
    fill(vs, 9)
    var hks = ctx.enqueue_create_host_buffer[DT](n)
    var hvs = ctx.enqueue_create_host_buffer[DT](n)
    var hkd = ctx.enqueue_create_host_buffer[DT](n)
    var hvd = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        hks[i] = f32_data(ks)[unsafe_offset=i]
        hvs[i] = f32_data(vs)[unsafe_offset=i]
    var dks = ctx.enqueue_create_buffer[DT](n)
    var dvs = ctx.enqueue_create_buffer[DT](n)
    var dkd = ctx.enqueue_create_buffer[DT](n)
    var dvd = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_copy(dst_buf=dks, src_buf=hks)
    ctx.enqueue_copy(dst_buf=dvs, src_buf=hvs)
    ctx.synchronize()
    ctx.enqueue_function[copy2_gpu](
        dkd.unsafe_ptr(), dvd.unsafe_ptr(), dks.unsafe_ptr(), dvs.unsafe_ptr(),
        Int32(n), grid_dim=(ceildiv(n * 2, 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hkd, src_buf=dkd)
    ctx.enqueue_copy(dst_buf=hvd, src_buf=dvd)
    ctx.synchronize()
    for i in range(n):
        if hkd[i] != hks[i]:
            raise Error("copy2(k): element " + String(i))
        if hvd[i] != hvs[i]:
            raise Error("copy2(v): element " + String(i))
    arena.keep_alive()
    print("*** COPY2 GPU/REFERENCE DIFF PASS *** diff= 0.0 (zero tolerance)")


def rope_kv_diff_check(mut ctx: DeviceContext) raises:
    """rope_kv（rope q + rope k 直写 store 行 + v 拷贝，三合一）的零容差差分。

    参照 = 标量 rope 的 q/k 输出 + v 逐位拷贝。k 段目标不是中转缓冲而是
    store 行（本 kernel 的存在理由，账本七十）；q 段与 rope2 逐元素同算术。
    """
    var arena = Arena(1 << 18)
    var q = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_Q_COLS))
    var k = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_K_COLS))
    var v = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_K_COLS))
    var cos = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_HEAD_DIM))
    var sin = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_HEAD_DIM))
    var oq_cpu = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_Q_COLS))
    var ok_cpu = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_K_COLS))
    fill(q, 51)
    fill(k, 52)
    fill(v, 53)
    fill(cos, 54)
    fill(sin, 55)
    rope(oq_cpu, ok_cpu, q, k, cos, sin, Int(ROPE_HEAD_DIM))

    var n_q = Int(ROPE_TOKENS) * Int(ROPE_Q_COLS)
    var n_k = Int(ROPE_TOKENS) * Int(ROPE_K_COLS)
    var n_t = Int(ROPE_TOKENS) * Int(ROPE_HEAD_DIM)
    var hq = ctx.enqueue_create_host_buffer[DT](n_q)
    var hk = ctx.enqueue_create_host_buffer[DT](n_k)
    var hv = ctx.enqueue_create_host_buffer[DT](n_k)
    var hc = ctx.enqueue_create_host_buffer[DT](n_t)
    var hs = ctx.enqueue_create_host_buffer[DT](n_t)
    var hqo = ctx.enqueue_create_host_buffer[DT](n_q)
    var hko = ctx.enqueue_create_host_buffer[DT](n_k)
    var hvo = ctx.enqueue_create_host_buffer[DT](n_k)
    ctx.synchronize()
    for i in range(n_q):
        hq[i] = f32_data(q)[unsafe_offset=i]
    for i in range(n_k):
        hk[i] = f32_data(k)[unsafe_offset=i]
        hv[i] = f32_data(v)[unsafe_offset=i]
    for i in range(n_t):
        hc[i] = f32_data(cos)[unsafe_offset=i]
        hs[i] = f32_data(sin)[unsafe_offset=i]
    var dq = ctx.enqueue_create_buffer[DT](n_q)
    var dk = ctx.enqueue_create_buffer[DT](n_k)
    var dv = ctx.enqueue_create_buffer[DT](n_k)
    var dc = ctx.enqueue_create_buffer[DT](n_t)
    var ds = ctx.enqueue_create_buffer[DT](n_t)
    var dqo = ctx.enqueue_create_buffer[DT](n_q)
    var dkr = ctx.enqueue_create_buffer[DT](n_k)
    var dvr = ctx.enqueue_create_buffer[DT](n_k)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
    ctx.enqueue_copy(dst_buf=dc, src_buf=hc)
    ctx.enqueue_copy(dst_buf=ds, src_buf=hs)
    ctx.synchronize()
    var q_pairs = Int(ROPE_TOKENS) * (Int(ROPE_Q_COLS) // Int(ROPE_HEAD_DIM)) * (Int(ROPE_HEAD_DIM) // 2)
    var k_pairs = Int(ROPE_TOKENS) * (Int(ROPE_K_COLS) // Int(ROPE_HEAD_DIM)) * (Int(ROPE_HEAD_DIM) // 2)
    ctx.enqueue_function[rope_kv_gpu](
        dqo.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
        dkr.unsafe_ptr(), dvr.unsafe_ptr(), dc.unsafe_ptr(), ds.unsafe_ptr(),
        Int32(ROPE_Q_COLS), Int32(ROPE_K_COLS), Int32(ROPE_HEAD_DIM), Int32(ROPE_TOKENS),
        grid_dim=(ceildiv(q_pairs + k_pairs + n_k, 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hqo, src_buf=dqo)
    ctx.enqueue_copy(dst_buf=hko, src_buf=dkr)
    ctx.enqueue_copy(dst_buf=hvo, src_buf=dvr)
    ctx.synchronize()

    for i in range(n_q):
        if hqo[i] != f32_data(oq_cpu)[unsafe_offset=i]:
            raise Error(
                "rope_kv(q): element " + String(i) + " gpu=" + String(hqo[i])
                + " cpu=" + String(f32_data(oq_cpu)[unsafe_offset=i])
            )
    for i in range(n_k):
        if hko[i] != f32_data(ok_cpu)[unsafe_offset=i]:
            raise Error(
                "rope_kv(k): element " + String(i) + " gpu=" + String(hko[i])
                + " cpu=" + String(f32_data(ok_cpu)[unsafe_offset=i])
            )
        if hvo[i] != hv[i]:
            raise Error("rope_kv(v): element " + String(i))
    arena.keep_alive()
    print("*** ROPE_KV GPU/SCALAR DIFF PASS *** diff= 0.0 (zero tolerance)")


def q4_fused_diff_check(mut ctx: DeviceContext) raises:
    """fused q4 matmul（bias 尾缀）的组合参照差分。

    参照 = quantize → dequant → 标量 linear → f32 加 bias（与 CPU 通路
    project 的顺序一致：linear 先写回 f32，bias 是其后的 f32 加法）。
    kernel 尾缀同序（Float32(smem[0]) + bias[o]），差分预期 diff=0.0；
    无偏置实例化（HAS_BIAS=False）与 smem 冠军同构，由 q4_matmul 检查
    的语义覆盖，这里只测 bias 路径。
    """
    var arena = Arena(1 << 21)
    var n_w = Int(Q4MM_OUT) * Int(Q4MM_INNER)
    var n_x = Int(Q4MM_ROWS) * Int(Q4MM_INNER)
    var n_d = Int(Q4MM_ROWS) * Int(Q4MM_OUT)
    var n_blocks = n_w // 32
    var n_bytes = n_blocks * 18
    var w_flat = scratch(arena, 1, n_w)
    var x = scratch(arena, Int(Q4MM_ROWS), Int(Q4MM_INNER))
    var bias = scratch(arena, 1, Int(Q4MM_OUT))
    fill(w_flat, 21)
    fill(x, 22)
    fill(bias, 5)
    var blocks = arena.alloc(n_bytes)
    var wf_raw = arena.alloc(n_w * 4)
    quantize_q4_0(blocks, f32_data(w_flat), n_w)
    dequant_q4_0(wf_raw.unsafe_bitcast[Float32](), blocks, n_w)
    var ref_raw = arena.alloc(n_d * 4)
    linear(
        rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT)),
        x,
        rows_view(wf_raw, Int(Q4MM_OUT), Int(Q4MM_INNER)),
    )
    # CPU 侧的 bias 加法（f32，逐元素）—— 与 project 的顺序一致
    for i in range(n_d):
        var o = i % Int(Q4MM_OUT)
        f32_data(rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT)))[unsafe_offset=i] += f32_data(bias)[unsafe_offset=o]

    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hw = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var hb = ctx.enqueue_create_host_buffer[DT](Int(Q4MM_OUT))
    var hd = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    for i in range(n_x):
        hx[i] = f32_data(x)[unsafe_offset=i]
    for i in range(n_bytes):
        hw[i] = blocks[unsafe_offset=i]
    for i in range(Int(Q4MM_OUT)):
        hb[i] = f32_data(bias)[unsafe_offset=i]
    var dx = ctx.enqueue_create_buffer[DT](n_x)
    var dw = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var db = ctx.enqueue_create_buffer[DT](Int(Q4MM_OUT))
    var dd = ctx.enqueue_create_buffer[DT](n_d)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.synchronize()
    ctx.enqueue_function[q4_matmul_fused_gpu[32, Q4MM_INNER, True]](
        dx.unsafe_ptr(),
        dw.unsafe_ptr(),
        dd.unsafe_ptr(),
        db.unsafe_ptr(),
        Int32(Q4MM_INNER),
        Int32(Q4MM_OUT),
        grid_dim=(Int(Q4MM_OUT), Int(Q4MM_ROWS), 1),
        block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    var scale = max_abs(f32_data(rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    if scale <= 0.0:
        raise Error("q4_fused: the reference tensor is all zeros")
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "q4_fused: gpu vs reference max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    print("*** Q4_FUSED GPU/REFERENCE DIFF PASS *** diff=", diff, " tol=", tol)


def linear_block_diff_check(mut ctx: DeviceContext) raises:
    """linear_block（lm_head 块内合并读）的相对容差差分。

    树形归约与标量串行累加是合法重组差（rmsnorm 重写、q4 champion 同一
    性质）；判据相对容差 1e-5。
    """
    var arena = Arena(1 << 22)
    var n_w = Int(Q4MM_OUT) * Int(Q4MM_INNER)
    var n_x = Int(Q4MM_ROWS) * Int(Q4MM_INNER)
    var n_d = Int(Q4MM_ROWS) * Int(Q4MM_OUT)
    var w_view = rows_view(arena.alloc(n_w * 4), Int(Q4MM_OUT), Int(Q4MM_INNER))
    var x = scratch(arena, Int(Q4MM_ROWS), Int(Q4MM_INNER))
    fill(w_view, 31)
    fill(x, 32)
    var ref_raw = arena.alloc(n_d * 4)
    linear(
        rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT)),
        x,
        w_view,
    )

    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hw = ctx.enqueue_create_host_buffer[DT](n_w)
    var hd = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    for i in range(n_x):
        hx[i] = f32_data(x)[unsafe_offset=i]
    for i in range(n_w):
        hw[i] = f32_data(w_view)[unsafe_offset=i]
    var dx = ctx.enqueue_create_buffer[DT](n_x)
    var dw = ctx.enqueue_create_buffer[DT](n_w)
    var dd = ctx.enqueue_create_buffer[DT](n_d)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.synchronize()
    ctx.enqueue_function[linear_block_gpu[256]](
        dx.unsafe_ptr(),
        dw.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(Q4MM_INNER),
        Int32(Q4MM_OUT),
        grid_dim=(Int(Q4MM_OUT), Int(Q4MM_ROWS), 1),
        block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    var scale = max_abs(f32_data(rows_view(ref_raw, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    if scale <= 0.0:
        raise Error("linear_block: the reference tensor is all zeros")
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "linear_block: gpu vs reference max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    print("*** LINEAR_BLOCK GPU/REFERENCE DIFF PASS *** diff=", diff, " tol=", tol)



def q4_dual_fat_diff_check(mut ctx: DeviceContext) raises:
    """dual-fat（gate/up 双矩阵融合）的组合参照差分。

    参照 = 两份独立的 quantize -> dequant -> 标量 linear（同一 x 行）。
    kernel 每 CTA 出 a/b 各 OUTS 个输出，累加序与冠军逐位同构；差分预期
    与 q4_matmul 检查同级（实测 0.0）。
    """
    var arena = Arena(1 << 21)
    var n_w = Int(Q4MM_OUT) * Int(Q4MM_INNER)
    var n_x = Int(Q4MM_ROWS) * Int(Q4MM_INNER)
    var n_d = Int(Q4MM_ROWS) * Int(Q4MM_OUT)
    var n_blocks = n_w // 32
    var n_bytes = n_blocks * 18
    var wa_flat = scratch(arena, 1, n_w)
    var wb_flat = scratch(arena, 1, n_w)
    var x = scratch(arena, Int(Q4MM_ROWS), Int(Q4MM_INNER))
    fill(wa_flat, 21)
    fill(wb_flat, 23)
    fill(x, 22)
    var blocks_a = arena.alloc(n_bytes)
    var blocks_b = arena.alloc(n_bytes)
    var wa_raw = arena.alloc(n_w * 4)
    var wb_raw = arena.alloc(n_w * 4)
    quantize_q4_0(blocks_a, f32_data(wa_flat), n_w)
    quantize_q4_0(blocks_b, f32_data(wb_flat), n_w)
    dequant_q4_0(wa_raw.unsafe_bitcast[Float32](), blocks_a, n_w)
    dequant_q4_0(wb_raw.unsafe_bitcast[Float32](), blocks_b, n_w)
    var ref_a = arena.alloc(n_d * 4)
    var ref_b = arena.alloc(n_d * 4)
    linear(
        rows_view(ref_a, Int(Q4MM_ROWS), Int(Q4MM_OUT)),
        x,
        rows_view(wa_raw, Int(Q4MM_OUT), Int(Q4MM_INNER)),
    )
    linear(
        rows_view(ref_b, Int(Q4MM_ROWS), Int(Q4MM_OUT)),
        x,
        rows_view(wb_raw, Int(Q4MM_OUT), Int(Q4MM_INNER)),
    )

    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hwa = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var hwb = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var hda = ctx.enqueue_create_host_buffer[DT](n_d)
    var hdb = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    for i in range(n_x):
        hx[i] = f32_data(x)[unsafe_offset=i]
    for i in range(n_bytes):
        hwa[i] = blocks_a[unsafe_offset=i]
        hwb[i] = blocks_b[unsafe_offset=i]
    var dx = ctx.enqueue_create_buffer[DT](n_x)
    var dwa = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var dwb = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var dda = ctx.enqueue_create_buffer[DT](n_d)
    var ddb = ctx.enqueue_create_buffer[DT](n_d)
    # 占位 bias 指针：HAS_BIAS=False 不读，但全部指针须两两不同源（六十五）
    var dum_a = ctx.enqueue_create_buffer[DT](4)
    var dum_b = ctx.enqueue_create_buffer[DT](4)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dwa, src_buf=hwa)
    ctx.enqueue_copy(dst_buf=dwb, src_buf=hwb)
    ctx.synchronize()
    ctx.enqueue_function[q4_matmul_dual_fat_gpu[32, Q4MM_INNER, 4, False]](
        dx.unsafe_ptr(),
        dwa.unsafe_ptr(),
        dwb.unsafe_ptr(),
        dda.unsafe_ptr(),
        ddb.unsafe_ptr(),
        dum_a.unsafe_ptr(),
        dum_b.unsafe_ptr(),
        Int32(Q4MM_INNER),
        Int32(Q4MM_OUT),
        grid_dim=(ceildiv(Int(Q4MM_OUT), 4), Int(Q4MM_ROWS), 1),
        block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hda, src_buf=dda)
    ctx.enqueue_copy(dst_buf=hdb, src_buf=ddb)
    ctx.synchronize()

    var diff_a = max_abs_diff(hda.unsafe_ptr(), f32_data(rows_view(ref_a, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    var diff_b = max_abs_diff(hdb.unsafe_ptr(), f32_data(rows_view(ref_b, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    var scale = max_abs(f32_data(rows_view(ref_a, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_d)
    if scale <= 0.0:
        raise Error("q4_dual_fat: the reference tensor is all zeros")
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff_a > tol or diff_b > tol:
        raise Error(
            "q4_dual_fat: gpu vs reference max_abs_diff a="
            + String(diff_a)
            + " b="
            + String(diff_b)
            + " tolerance="
            + String(tol)
        )
    print("*** Q4_DUAL_FAT GPU/REFERENCE DIFF PASS *** diff_a=", diff_a, " diff_b=", diff_b, " tol=", tol)



def qkv_tri_diff_check(mut ctx: DeviceContext) raises:
    """qkv 三融合（gate/up dual 的三矩阵版）的组合参照差分。

    参照 = 三份独立的 quantize -> dequant -> 标量 linear（同一 x 行）。
    kernel 每 CTA 出 q/k/v 各一个输出（k/v 输出数小于 q 时尾部 CTA 只出
    部分），累加序与冠军逐位同构；差分预期 0.0。bias 走 HAS_BIAS=False +
    三个互不相同的 dummy 占位（aliasing 规）。
    """
    var arena = Arena(1 << 22)
    var n_wq = Int(Q4MM_OUT) * Int(Q4MM_INNER)
    var kv_out = 8
    var n_wkv = kv_out * Int(Q4MM_INNER)
    var n_x = Int(Q4MM_ROWS) * Int(Q4MM_INNER)
    var n_xb = n_x * 4
    var x = scratch(arena, Int(Q4MM_ROWS), Int(Q4MM_INNER))
    fill(x, 25)
    var nb_q = n_wq // 32
    var nb_kv = n_wkv // 32
    var n_dq = Int(Q4MM_ROWS) * Int(Q4MM_OUT)
    var n_dkv = Int(Q4MM_ROWS) * kv_out
    ctx.synchronize()
    # 参照侧：三份权重分别量化、反量化、标量 linear
    var dq_blocks = arena.alloc(nb_q * 18)
    var dk_blocks = arena.alloc(nb_kv * 18)
    var dv_blocks = arena.alloc(nb_kv * 18)
    var wq_raw = arena.alloc(n_wq * 4)
    var wk_raw = arena.alloc(n_wkv * 4)
    var wv_raw = arena.alloc(n_wkv * 4)
    var ref_q = arena.alloc(n_dq * 4)
    var ref_k = arena.alloc(n_dkv * 4)
    var ref_v = arena.alloc(n_dkv * 4)
    var wq_flat = scratch(arena, 1, n_wq)
    var wk_flat = scratch(arena, 1, n_wkv)
    var wv_flat = scratch(arena, 1, n_wkv)
    fill(wq_flat, 41)
    fill(wk_flat, 43)
    fill(wv_flat, 45)
    quantize_q4_0(dq_blocks, f32_data(wq_flat), n_wq)
    quantize_q4_0(dk_blocks, f32_data(wk_flat), n_wkv)
    quantize_q4_0(dv_blocks, f32_data(wv_flat), n_wkv)
    dequant_q4_0(wq_raw.unsafe_bitcast[Float32](), dq_blocks, n_wq)
    dequant_q4_0(wk_raw.unsafe_bitcast[Float32](), dk_blocks, n_wkv)
    dequant_q4_0(wv_raw.unsafe_bitcast[Float32](), dv_blocks, n_wkv)
    linear(
        rows_view(ref_q, Int(Q4MM_ROWS), Int(Q4MM_OUT)),
        x,
        rows_view(wq_raw, Int(Q4MM_OUT), Int(Q4MM_INNER)),
    )
    linear(
        rows_view(ref_k, Int(Q4MM_ROWS), kv_out),
        x,
        rows_view(wk_raw, kv_out, Int(Q4MM_INNER)),
    )
    linear(
        rows_view(ref_v, Int(Q4MM_ROWS), kv_out),
        x,
        rows_view(wv_raw, kv_out, Int(Q4MM_INNER)),
    )

    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hwq = ctx.enqueue_create_host_buffer[DType.uint8](nb_q * 18)
    var hwk = ctx.enqueue_create_host_buffer[DType.uint8](nb_kv * 18)
    var hwv = ctx.enqueue_create_host_buffer[DType.uint8](nb_kv * 18)
    var hdq = ctx.enqueue_create_host_buffer[DT](n_dq)
    var hdk = ctx.enqueue_create_host_buffer[DT](n_dkv)
    var hdv = ctx.enqueue_create_host_buffer[DT](n_dkv)
    ctx.synchronize()
    for i in range(n_x):
        hx[i] = f32_data(x)[unsafe_offset=i]
    for i in range(nb_q * 18):
        hwq[i] = dq_blocks[unsafe_offset=i]
    for i in range(nb_kv * 18):
        hwk[i] = dk_blocks[unsafe_offset=i]
        hwv[i] = dv_blocks[unsafe_offset=i]
    var dx = ctx.enqueue_create_buffer[DT](n_x)
    var dwq = ctx.enqueue_create_buffer[DType.uint8](nb_q * 18)
    var dwk = ctx.enqueue_create_buffer[DType.uint8](nb_kv * 18)
    var dwv = ctx.enqueue_create_buffer[DType.uint8](nb_kv * 18)
    var ddq = ctx.enqueue_create_buffer[DT](n_dq)
    var ddk = ctx.enqueue_create_buffer[DT](n_dkv)
    var ddv = ctx.enqueue_create_buffer[DT](n_dkv)
    var dum_q = ctx.enqueue_create_buffer[DT](4)
    var dum_k = ctx.enqueue_create_buffer[DT](4)
    var dum_v = ctx.enqueue_create_buffer[DT](4)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dwq, src_buf=hwq)
    ctx.enqueue_copy(dst_buf=dwk, src_buf=hwk)
    ctx.enqueue_copy(dst_buf=dwv, src_buf=hwv)
    ctx.synchronize()
    ctx.enqueue_function[q4_matmul_qkv_gpu[32, Q4MM_INNER, False]](
        dx.unsafe_ptr(),
        dwq.unsafe_ptr(),
        dwk.unsafe_ptr(),
        dwv.unsafe_ptr(),
        ddq.unsafe_ptr(),
        ddk.unsafe_ptr(),
        ddv.unsafe_ptr(),
        dum_q.unsafe_ptr(),
        dum_k.unsafe_ptr(),
        dum_v.unsafe_ptr(),
        Int32(Q4MM_INNER),
        Int32(Q4MM_OUT),
        Int32(kv_out),
        grid_dim=(Int(Q4MM_OUT), Int(Q4MM_ROWS), 1),
        block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hdq, src_buf=ddq)
    ctx.enqueue_copy(dst_buf=hdk, src_buf=ddk)
    ctx.enqueue_copy(dst_buf=hdv, src_buf=ddv)
    ctx.synchronize()

    var diff_q = max_abs_diff(hdq.unsafe_ptr(), f32_data(rows_view(ref_q, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_dq)
    var diff_k = max_abs_diff(hdk.unsafe_ptr(), f32_data(rows_view(ref_k, Int(Q4MM_ROWS), kv_out)), n_dkv)
    var diff_v = max_abs_diff(hdv.unsafe_ptr(), f32_data(rows_view(ref_v, Int(Q4MM_ROWS), kv_out)), n_dkv)
    var scale = max_abs(f32_data(rows_view(ref_q, Int(Q4MM_ROWS), Int(Q4MM_OUT))), n_dq)
    if scale <= 0.0:
        raise Error("qkv_tri: the reference tensor is all zeros")
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff_q > tol or diff_k > tol or diff_v > tol:
        raise Error(
            "qkv_tri: gpu vs reference diff q="
            + String(diff_q)
            + " k="
            + String(diff_k)
            + " v="
            + String(diff_v)
            + " tolerance="
            + String(tol)
        )
    print("*** QKV_TRI GPU/REFERENCE DIFF PASS *** diff_q=", diff_q, " diff_k=", diff_k, " diff_v=", diff_v, " tol=", tol)



def f16_weight_ref(
    mut arena: Arena,
    mut ctx: DeviceContext,
    seed: Int,
    n_w: Int,
    mut hq4: HostBuffer[DType.uint8],
    mut hf16: HostBuffer[DType.float16],
) raises -> None:
    """量化一个假权重矩阵到调用方给的 host 缓冲：q4 块流 + f16 预反量化。

    门共用件：q4 版喂 CPU dequant 参照，f16 版喂 GPU（模拟装载期
    预反量化，数值 = f32 精确反量化值舍入进 f16）。
    """
    var n_blocks = n_w // 32
    ctx.synchronize()
    var v = scratch(arena, 1, n_w)
    fill(v, seed)
    quantize_q4_0(hq4.unsafe_ptr(), f32_data(v), n_w)
    var dq = arena.alloc(n_w * 4)
    dequant_q4_0(dq.unsafe_bitcast[Float32](), hq4.unsafe_ptr(), n_w)
    var dq32 = dq.unsafe_bitcast[Float32]()
    for i in range(n_w):
        hf16[i] = Float16(dq32[unsafe_offset=i])


def f16_fat_diff_check(mut ctx: DeviceContext) raises:
    """f16 fat（down/o 形状）门项：参照 = CPU dequant（f32 精确）linear。

    f16 权重舍入（每权重 rel err ≤ 2^-11）使差分下限不再是 0 —— 容差
    F16_REL 实测定标（首次运行取实测差，加安全系数后固化）。
    """
    var arena = Arena(1 << 26)
    comptime FF_IN = 4864
    comptime FF_OUT = 896
    var n_w = Int(FF_OUT) * Int(FF_IN)
    var n_x = Int(FF_IN)
    var n_d = Int(FF_OUT)
    var x = scratch(arena, 1, Int(FF_IN))
    fill(x, 27)
    var hq4 = ctx.enqueue_create_host_buffer[DType.uint8](n_w // 32 * 18)
    var hf16 = ctx.enqueue_create_host_buffer[DType.float16](n_w)
    f16_weight_ref(arena, ctx, 29, n_w, hq4, hf16)
    var wraw = arena.alloc(n_w * 4)
    dequant_q4_0(wraw.unsafe_bitcast[Float32](), hq4.unsafe_ptr(), n_w)
    var ref_raw = arena.alloc(n_d * 4)
    linear(
        rows_view(ref_raw, 1, Int(FF_OUT)),
        x,
        rows_view(wraw, Int(FF_OUT), Int(FF_IN)),
    )
    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hd = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    for i in range(n_x):
        hx[i] = f32_data(x)[unsafe_offset=i]
    var n_bytes = n_w // 32 * 18
    var dq16 = ctx.enqueue_create_buffer[DType.float16](n_w)
    var hq4d = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    ctx.synchronize()
    for i in range(n_bytes):
        hq4d[i] = hq4[i]
    var dq4 = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    ctx.enqueue_copy(dst_buf=dq16, src_buf=hf16)
    ctx.enqueue_copy(dst_buf=dq4, src_buf=hq4d)
    ctx.synchronize()
    # GPU 侧预反量化（与 session 装载同路径）
    var w16 = ctx.enqueue_create_buffer[DType.float16](n_w)
    ctx.enqueue_function[q4_dequant_f16_gpu](
        dq4.unsafe_ptr(), w16.unsafe_ptr(), Int32(n_w),
        grid_dim=(ceildiv(n_w, 256), 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    var dd = ctx.enqueue_create_buffer[DT](n_d)
    var dd2 = ctx.enqueue_create_buffer[DT](4)
    var xg = ctx.enqueue_create_buffer[DT](n_x)
    ctx.enqueue_copy(dst_buf=xg, src_buf=hx)
    ctx.synchronize()
    # 第一步：GPU 反量化 vs host 转换 —— 分离 dequant / matmul 责任
    var hwchk = ctx.enqueue_create_host_buffer[DType.float16](n_w)
    ctx.enqueue_copy(dst_buf=hwchk, src_buf=w16)
    ctx.synchronize()
    var wdiff = Float32(0.0)
    for i in range(n_w):
        var a = Float32(hwchk[i])
        var b = Float32(hf16[i])
        var d = a - b
        if d < 0.0:
            d = -d
        if d > wdiff:
            wdiff = d
    print("[f16_fat] wdiff(gpu dequant vs host f16)=", wdiff)
    ctx.enqueue_function[f16_matmul_fat_gpu[256, FF_IN, 2, False]](
        xg.unsafe_ptr(), w16.unsafe_ptr(), dd.unsafe_ptr(), dd2.unsafe_ptr(),
        Int32(FF_IN), Int32(FF_OUT),
        grid_dim=(ceildiv(Int(FF_OUT), 2), 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()
    var diff256 = max_abs_diff(hd.unsafe_ptr(), f32_data(rows_view(ref_raw, 1, Int(FF_OUT))), n_d)
    # 对照 1：T64（同形状）—— 分离 THREADS 因素
    ctx.enqueue_function[f16_matmul_fat_gpu[64, FF_IN, 2, False]](
        xg.unsafe_ptr(), w16.unsafe_ptr(), dd.unsafe_ptr(), dd2.unsafe_ptr(),
        Int32(FF_IN), Int32(FF_OUT),
        grid_dim=(ceildiv(Int(FF_OUT), 2), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()
    var diff64 = max_abs_diff(hd.unsafe_ptr(), f32_data(rows_view(ref_raw, 1, Int(FF_OUT))), n_d)
    # 对照 2：短 INNER（4864 换 896 形不便利，直接用同一权重前 896 列不成立 —— 跳过）
    print("[f16_fat] diff256=", diff256, " diff64=", diff64)
    var diff = diff256
    var scale = max_abs(f32_data(rows_view(ref_raw, 1, Int(FF_OUT))), n_d)
    if scale <= 0.0:
        raise Error("f16_fat: the reference tensor is all zeros")
    var rel = diff / scale
    print("[f16_fat] diff=", diff, " scale=", scale, " rel=", rel)
    var tol = F16_REL * scale
    # ACC 判据（账本七十）：dst 预填已知图案，ACC=True 后应为
    # "图案 + 参照"（f32 加法顺序与 add_into 通路逐位一致）。
    var hpre = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    for i in range(n_d):
        hpre[i] = Float32((i % 17) - 8) * Float32(0.031)
    ctx.enqueue_copy(dst_buf=dd, src_buf=hpre)
    ctx.synchronize()
    ctx.enqueue_function[f16_matmul_fat_gpu[256, FF_IN, 2, False, True]](
        xg.unsafe_ptr(), w16.unsafe_ptr(), dd.unsafe_ptr(), dd2.unsafe_ptr(),
        Int32(FF_IN), Int32(FF_OUT),
        grid_dim=(ceildiv(Int(FF_OUT), 2), 1, 1), block_dim=(256, 1, 1),
    )
    var hd_acc = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.enqueue_copy(dst_buf=hd_acc, src_buf=dd)
    ctx.synchronize()
    var diff_acc = Float64(0)
    var scale_acc = Float64(0)
    for i in range(n_d):
        var expect = Float64(hpre[i]) + Float64(f32_data(rows_view(ref_raw, 1, Int(FF_OUT)))[unsafe_offset=i])
        var d = abs(Float64(hd_acc[i]) - expect)
        if d > diff_acc:
            diff_acc = d
        var s = abs(expect)
        if s > scale_acc:
            scale_acc = s
    if scale_acc <= 0.0:
        raise Error("f16_fat_acc: the reference tensor is all zeros")
    if diff_acc > F16_REL * scale_acc:
        raise Error(
            "f16_fat_acc: diff=" + String(diff_acc) + " tol=" + String(F16_REL * scale_acc)
        )
    print("[f16_fat_acc] rel=", diff_acc / scale_acc)
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "f16_fat: gpu vs reference diff=" + String(diff) + " tolerance=" + String(tol)
        )
    print("*** F16_FAT GPU/REFERENCE DIFF PASS *** diff=", diff, " tol=", tol)



def f16_dual_tri_head_diff_check(mut ctx: DeviceContext) raises:
    """f16 dual（gate/up 形）/ tri（q/k/v 形）/ head 的组合参照差分。

    参照 = f32 精确反量化的标量 linear（同一 x 行、同一量化权重）。
    f16 舍入噪声由 F16_REL 覆盖（实测单 matmul rel ~3e-5，判据 1e-3 留
    30 倍余量）。tri 的 bias 走 HAS_BIAS=False + 三个 dummy（aliasing 规）。
    """
    var arena = Arena(1 << 27)
    # ---- dual：gate 形 4864x896，两份权重 ----
    var n_wg = 4864 * 896
    var n_x = 896
    var hq4g_a = ctx.enqueue_create_host_buffer[DType.uint8](n_wg // 32 * 18)
    var hf16g_a = ctx.enqueue_create_host_buffer[DType.float16](n_wg)
    f16_weight_ref(arena, ctx, 61, n_wg, hq4g_a, hf16g_a)
    var hq4g_b = ctx.enqueue_create_host_buffer[DType.uint8](n_wg // 32 * 18)
    var hf16g_b = ctx.enqueue_create_host_buffer[DType.float16](n_wg)
    f16_weight_ref(arena, ctx, 63, n_wg, hq4g_b, hf16g_b)
    var x = scratch(arena, 1, n_x)
    fill(x, 62)
    var wraw_a = arena.alloc(n_wg * 4)
    dequant_q4_0(wraw_a.unsafe_bitcast[Float32](), hq4g_a.unsafe_ptr(), n_wg)
    var ref_a = arena.alloc(4864 * 4)
    linear(
        rows_view(ref_a, 1, 4864),
        x,
        rows_view(wraw_a, 4864, 896),
    )
    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hd_a = ctx.enqueue_create_host_buffer[DT](4864)
    var hd_b = ctx.enqueue_create_host_buffer[DT](4864)
    var hd_act = ctx.enqueue_create_host_buffer[DT](4864)
    ctx.synchronize()
    for i in range(n_x):
        hx[i] = f32_data(x)[unsafe_offset=i]
    var xg = ctx.enqueue_create_buffer[DT](n_x)
    var da = ctx.enqueue_create_buffer[DT](4864)
    var db = ctx.enqueue_create_buffer[DT](4864)
    var dact = ctx.enqueue_create_buffer[DT](4864)
    var wa = ctx.enqueue_create_buffer[DType.float16](n_wg)
    var wb = ctx.enqueue_create_buffer[DType.float16](n_wg)
    var dum1 = ctx.enqueue_create_buffer[DT](4)
    var dum2 = ctx.enqueue_create_buffer[DT](4)
    ctx.enqueue_copy(dst_buf=xg, src_buf=hx)
    ctx.enqueue_copy(dst_buf=wa, src_buf=hf16g_a)
    ctx.enqueue_copy(dst_buf=wb, src_buf=hf16g_b)
    ctx.synchronize()
    # HAS_ACT=True（账本七十）：swiglu 尾缀与 gate/up 同发射。
    ctx.enqueue_function[f16_matmul_dual_gpu[64, 896, 4, False, True]](
        xg.unsafe_ptr(), wa.unsafe_ptr(), wb.unsafe_ptr(),
        da.unsafe_ptr(), db.unsafe_ptr(), dum1.unsafe_ptr(), dum2.unsafe_ptr(),
        dact.unsafe_ptr(),
        Int32(896), Int32(4864),
        grid_dim=(ceildiv(4864, 4), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd_a, src_buf=da)
    ctx.enqueue_copy(dst_buf=hd_b, src_buf=db)
    ctx.enqueue_copy(dst_buf=hd_act, src_buf=dact)
    ctx.synchronize()
    var diff_a = max_abs_diff(hd_a.unsafe_ptr(), f32_data(rows_view(ref_a, 1, 4864)), 4864)
    var scale_a = max_abs(f32_data(rows_view(ref_a, 1, 4864)), 4864)
    if scale_a <= 0.0:
        raise Error("f16_dual: the reference tensor is all zeros")
    if diff_a > F16_REL * scale_a:
        raise Error(
            "f16_dual: diff=" + String(diff_a) + " tol=" + String(F16_REL * scale_a)
        )
    print("[f16_dual] rel=", diff_a / scale_a)
    # act 尾缀判据：silu(gate)*up 以 GPU 自身的 gate/up 输出为输入（两者
    # 已过参照门），f64 同序重算 —— 钉的是尾缀的算术与舍入顺序。
    var diff_act = Float64(0)
    var scale_act = Float64(0)
    for i in range(4864):
        var g = Float64(hd_a[i])
        var expect = Float32(
            g / (Float64(1) + exp(-g)) * Float64(hd_b[i])
        )
        var d = abs(Float64(hd_act[i]) - Float64(expect))
        if d > diff_act:
            diff_act = d
        var s = abs(Float64(expect))
        if s > scale_act:
            scale_act = s
    if scale_act <= 0.0:
        raise Error("f16_dual_act: the reference tensor is all zeros")
    if diff_act > F16_REL * scale_act:
        raise Error(
            "f16_dual_act: diff="
            + String(diff_act)
            + " tol="
            + String(F16_REL * scale_act)
        )
    print("[f16_dual_act] rel=", diff_act / scale_act)

    # ---- tri：q 16 出 / kv 12 出，同 x ----
    var n_wq = 16 * 896
    var n_wkv = 12 * 896
    var hq4q = ctx.enqueue_create_host_buffer[DType.uint8](n_wq // 32 * 18)
    var hf16q = ctx.enqueue_create_host_buffer[DType.float16](n_wq)
    f16_weight_ref(arena, ctx, 71, n_wq, hq4q, hf16q)
    var hq4k = ctx.enqueue_create_host_buffer[DType.uint8](n_wkv // 32 * 18)
    var hf16k = ctx.enqueue_create_host_buffer[DType.float16](n_wkv)
    f16_weight_ref(arena, ctx, 73, n_wkv, hq4k, hf16k)
    var hq4v = ctx.enqueue_create_host_buffer[DType.uint8](n_wkv // 32 * 18)
    var hf16v = ctx.enqueue_create_host_buffer[DType.float16](n_wkv)
    f16_weight_ref(arena, ctx, 75, n_wkv, hq4v, hf16v)
    var wraw_q = arena.alloc(n_wq * 4)
    var wraw_k = arena.alloc(n_wkv * 4)
    var wraw_v = arena.alloc(n_wkv * 4)
    dequant_q4_0(wraw_q.unsafe_bitcast[Float32](), hq4q.unsafe_ptr(), n_wq)
    dequant_q4_0(wraw_k.unsafe_bitcast[Float32](), hq4k.unsafe_ptr(), n_wkv)
    dequant_q4_0(wraw_v.unsafe_bitcast[Float32](), hq4v.unsafe_ptr(), n_wkv)
    var ref_q = arena.alloc(16 * 4)
    var ref_k = arena.alloc(12 * 4)
    var ref_v = arena.alloc(12 * 4)
    linear(
        rows_view(ref_q, 1, 16),
        x,
        rows_view(wraw_q, 16, 896),
    )
    linear(
        rows_view(ref_k, 1, 12),
        x,
        rows_view(wraw_k, 12, 896),
    )
    linear(
        rows_view(ref_v, 1, 12),
        x,
        rows_view(wraw_v, 12, 896),
    )
    var hd_q = ctx.enqueue_create_host_buffer[DT](16)
    var hd_k = ctx.enqueue_create_host_buffer[DT](12)
    var hd_v = ctx.enqueue_create_host_buffer[DT](12)
    ctx.synchronize()
    var dq = ctx.enqueue_create_buffer[DT](16)
    var dk = ctx.enqueue_create_buffer[DT](12)
    var dv = ctx.enqueue_create_buffer[DT](12)
    var wq = ctx.enqueue_create_buffer[DType.float16](n_wq)
    var wk = ctx.enqueue_create_buffer[DType.float16](n_wkv)
    var wv = ctx.enqueue_create_buffer[DType.float16](n_wkv)
    var dum3 = ctx.enqueue_create_buffer[DT](4)
    var dum4 = ctx.enqueue_create_buffer[DT](4)
    var dum5 = ctx.enqueue_create_buffer[DT](4)
    ctx.enqueue_copy(dst_buf=wq, src_buf=hf16q)
    ctx.enqueue_copy(dst_buf=wk, src_buf=hf16k)
    ctx.enqueue_copy(dst_buf=wv, src_buf=hf16v)
    ctx.synchronize()
    ctx.enqueue_function[f16_matmul_qkv_gpu[32, 896, False]](
        xg.unsafe_ptr(), wq.unsafe_ptr(), wk.unsafe_ptr(), wv.unsafe_ptr(),
        dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
        dum3.unsafe_ptr(), dum4.unsafe_ptr(), dum5.unsafe_ptr(),
        Int32(896), Int32(16), Int32(12),
        grid_dim=(16, 1, 1), block_dim=(32, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd_q, src_buf=dq)
    ctx.enqueue_copy(dst_buf=hd_k, src_buf=dk)
    ctx.enqueue_copy(dst_buf=hd_v, src_buf=dv)
    ctx.synchronize()
    var diff_q = max_abs_diff(hd_q.unsafe_ptr(), f32_data(rows_view(ref_q, 1, 16)), 16)
    var scale_q = max_abs(f32_data(rows_view(ref_q, 1, 16)), 16)
    if scale_q <= 0.0:
        raise Error("f16_tri: the reference tensor is all zeros")
    if diff_q > F16_REL * scale_q:
        raise Error(
            "f16_tri: diff=" + String(diff_q) + " tol=" + String(F16_REL * scale_q)
        )
    print("[f16_tri] rel=", diff_q / scale_q)

    # ---- head：896 -> 16，f16 权重直接喂（模拟 embed16 转换后） ----
    # 参照与 dual/tri 同构：量化权重 f32 精确反量化 + 标量 linear。
    var n_wh = 16 * 896
    var hq4h = ctx.enqueue_create_host_buffer[DType.uint8](n_wh // 32 * 18)
    var hf16h = ctx.enqueue_create_host_buffer[DType.float16](n_wh)
    f16_weight_ref(arena, ctx, 81, n_wh, hq4h, hf16h)
    var wraw_h = arena.alloc(n_wh * 4)
    dequant_q4_0(wraw_h.unsafe_bitcast[Float32](), hq4h.unsafe_ptr(), n_wh)
    var ref_h = arena.alloc(16 * 4)
    linear(
        rows_view(ref_h, 1, 16),
        x,
        rows_view(wraw_h, 16, 896),
    )
    var hd_h = ctx.enqueue_create_host_buffer[DT](16)
    var dh = ctx.enqueue_create_buffer[DT](16)
    var wh = ctx.enqueue_create_buffer[DType.float16](n_wh)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=wh, src_buf=hf16h)
    ctx.synchronize()
    ctx.enqueue_function[linear_fat_f16_gpu[128, 896, 8]](
        xg.unsafe_ptr(), wh.unsafe_ptr(), dh.unsafe_ptr(), Int32(16),
        grid_dim=(2, 1, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd_h, src_buf=dh)
    ctx.synchronize()
    var diff_h = max_abs_diff(hd_h.unsafe_ptr(), f32_data(rows_view(ref_h, 1, 16)), 16)
    var scale_h = max_abs(f32_data(rows_view(ref_h, 1, 16)), 16)
    if scale_h <= 0.0:
        raise Error("f16_head: the reference tensor is all zeros")
    if diff_h > F16_REL * scale_h:
        raise Error(
            "f16_head: diff=" + String(diff_h) + " tol=" + String(F16_REL * scale_h)
        )
    print("[f16_head] rel=", diff_h / scale_h)
    # 与 f16_fat 同一个坑：arena 的最后一次使用在上面 alloc 处，不
    # keep_alive 会在函数尾 munmap，之后的 GPU/读全部踩已释放映射。
    arena.keep_alive()
    print("*** F16 DUAL/TRI/HEAD GPU/REFERENCE DIFF PASS ***")


def add_copy_diff_check(mut ctx: DeviceContext) raises:
    """add 与 copy 的零容差差分 —— 纯 f32 加法/搬运，差值即读错内存。

    注意 `arena.keep_alive()` 的位置：它是**生命周期标记**（"内存用到这里"），
    之后的 arena 读取就是 use-after-free（本次调试实测：放在 helper 读取之前
    会段错误在 AsyncRT/比较代码里，症状极具误导性）。必须放在所有 arena 读取
    之后、函数末尾。
    """
    var arena = Arena(1 << 18)
    var a = scratch(arena, 1, Int(EW_N))
    var b = scratch(arena, 1, Int(EW_N))
    var dst_cpu = scratch(arena, 1, Int(EW_N))
    fill(a, 31)
    fill(b, 32)
    add(dst_cpu, a, b)

    var src_cpu = scratch(arena, 1, Int(COPY_N))
    fill(src_cpu, 33)
    var cp_ref = arena.alloc(Int(COPY_N) * 4)
    var pcp = cp_ref.unsafe_bitcast[Float32]()
    var psrc = f32_data(src_cpu)
    for i in range(Int(COPY_N)):
        pcp[unsafe_offset=i] = psrc[unsafe_offset=i]

    var na = Int(EW_N)
    var nc = Int(COPY_N)
    var ha = ctx.enqueue_create_host_buffer[DT](na)
    var hb = ctx.enqueue_create_host_buffer[DT](na)
    var hc = ctx.enqueue_create_host_buffer[DT](nc)
    var hd = ctx.enqueue_create_host_buffer[DT](na)
    var he = ctx.enqueue_create_host_buffer[DT](nc)
    ctx.synchronize()
    for i in range(na):
        ha[i] = f32_data(a)[unsafe_offset=i]
        hb[i] = f32_data(b)[unsafe_offset=i]
    for i in range(nc):
        hc[i] = psrc[unsafe_offset=i]

    var da = ctx.enqueue_create_buffer[DT](na)
    var db = ctx.enqueue_create_buffer[DT](na)
    var dc = ctx.enqueue_create_buffer[DT](nc)
    var dd = ctx.enqueue_create_buffer[DT](na)
    var de = ctx.enqueue_create_buffer[DT](nc)
    ctx.enqueue_copy(dst_buf=da, src_buf=ha)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.enqueue_copy(dst_buf=dc, src_buf=hc)
    ctx.synchronize()
    ctx.enqueue_function[add_gpu](
        dd.unsafe_ptr(), da.unsafe_ptr(), db.unsafe_ptr(), Int32(EW_N),
        grid_dim=(ceildiv(na, 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_function[copy_gpu](
        de.unsafe_ptr(), dc.unsafe_ptr(), Int32(COPY_N),
        grid_dim=(ceildiv(nc, 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.enqueue_copy(dst_buf=he, src_buf=de)
    ctx.synchronize()

    var diff_add = max_abs_diff(hd.unsafe_ptr(), f32_data(dst_cpu), na)
    if diff_add != 0.0:
        raise Error("add: diff=" + String(diff_add) + " (zero tolerance)")
    # add_into：原地残差（dst 即源之一）。dst 从 a 的副本出发，加 b，比 dst_cpu。
    var hsum = ctx.enqueue_create_host_buffer[DT](na)
    ctx.synchronize()
    for i in range(na):
        hsum[i] = f32_data(a)[unsafe_offset=i]
    var dsum = ctx.enqueue_create_buffer[DT](na)
    ctx.enqueue_copy(dst_buf=dsum, src_buf=hsum)
    ctx.synchronize()
    ctx.enqueue_function[add_into_gpu](
        dsum.unsafe_ptr(), db.unsafe_ptr(), Int32(EW_N),
        grid_dim=(ceildiv(na, 64), 1, 1), block_dim=(64, 1, 1),
    )
    var hsum2 = ctx.enqueue_create_host_buffer[DT](na)
    ctx.enqueue_copy(dst_buf=hsum2, src_buf=dsum)
    ctx.synchronize()
    var diff_into = max_abs_diff(hsum2.unsafe_ptr(), f32_data(dst_cpu), na)
    if diff_into != 0.0:
        raise Error("add_into: diff=" + String(diff_into) + " (zero tolerance)")
    var diff_cp = max_abs_diff(he.unsafe_ptr(), pcp, nc)
    if diff_cp != 0.0:
        raise Error("copy: diff=" + String(diff_cp) + " (zero tolerance)")
    arena.keep_alive()
    print("*** ADD/ADD_INTO/COPY GPU/SCALAR DIFF PASS *** diff= 0.0 (zero tolerance)")


def swiglu_diff_check(mut ctx: DeviceContext) raises:
    """swiglu 的相对容差差分 —— f64 中间量 + exp，exp 已知 1 ulp。"""
    var arena = Arena(1 << 18)
    var gate = scratch(arena, 1, Int(EW_N))
    var up = scratch(arena, 1, Int(EW_N))
    var dst_cpu = scratch(arena, 1, Int(EW_N))
    fill(gate, 34)
    fill(up, 35)
    swiglu(dst_cpu, gate, up)

    var n = Int(EW_N)
    var hg = ctx.enqueue_create_host_buffer[DT](n)
    var hu = ctx.enqueue_create_host_buffer[DT](n)
    var hd = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        hg[i] = f32_data(gate)[unsafe_offset=i]
        hu[i] = f32_data(up)[unsafe_offset=i]
    var dg = ctx.enqueue_create_buffer[DT](n)
    var du = ctx.enqueue_create_buffer[DT](n)
    var dd = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_copy(dst_buf=dg, src_buf=hg)
    ctx.enqueue_copy(dst_buf=du, src_buf=hu)
    ctx.synchronize()
    ctx.enqueue_function[swiglu_gpu](
        dd.unsafe_ptr(), dg.unsafe_ptr(), du.unsafe_ptr(), Int32(EW_N),
        grid_dim=(ceildiv(Int(EW_N), 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(dst_cpu), n)
    var scale = max_abs(f32_data(dst_cpu), n)
    if scale <= 0.0:
        raise Error("swiglu: the reference tensor is all zeros")
    arena.keep_alive()
    if diff > REL_TOLERANCE * scale:
        raise Error("swiglu: diff=" + String(diff) + " tol=" + String(REL_TOLERANCE * scale))
    print("*** SWIGLU GPU/SCALAR DIFF PASS *** diff=", diff, " tol=", REL_TOLERANCE * scale)


def rope_diff_check(mut ctx: DeviceContext) raises:
    """rope 的零容差差分 —— 纯 f32 乘加，同时是 FMA 折叠的观察点。

    `a*c - b*s` 是经典折叠模式：若门红在个位 ulp，就是编译器做了 FMA 契约
    （linear 的 diff=0.0 说明点积模式没折叠，这个模式此前没被测过）。红则把
    实际差值记录进账本，判据换相对容差 —— 那是合法结论，不是门失效。
    """
    var arena = Arena(1 << 18)
    var q = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_Q_COLS))
    var k = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_K_COLS))
    var cos = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_HEAD_DIM))
    var sin = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_HEAD_DIM))
    var oq_cpu = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_Q_COLS))
    var ok_cpu = scratch(arena, Int(ROPE_TOKENS), Int(ROPE_K_COLS))
    fill(q, 41)
    fill(k, 42)
    fill(cos, 43)
    fill(sin, 44)
    rope(oq_cpu, ok_cpu, q, k, cos, sin, Int(ROPE_HEAD_DIM))

    var n_q = Int(ROPE_TOKENS) * Int(ROPE_Q_COLS)
    var n_k = Int(ROPE_TOKENS) * Int(ROPE_K_COLS)
    var n_t = Int(ROPE_TOKENS) * Int(ROPE_HEAD_DIM)
    var hq = ctx.enqueue_create_host_buffer[DT](n_q)
    var hk = ctx.enqueue_create_host_buffer[DT](n_k)
    var hc = ctx.enqueue_create_host_buffer[DT](n_t)
    var hs = ctx.enqueue_create_host_buffer[DT](n_t)
    var hqo = ctx.enqueue_create_host_buffer[DT](n_q)
    var hko = ctx.enqueue_create_host_buffer[DT](n_k)
    ctx.synchronize()
    for i in range(n_q):
        hq[i] = f32_data(q)[unsafe_offset=i]
    for i in range(n_k):
        hk[i] = f32_data(k)[unsafe_offset=i]
    for i in range(n_t):
        hc[i] = f32_data(cos)[unsafe_offset=i]
        hs[i] = f32_data(sin)[unsafe_offset=i]
    var dq = ctx.enqueue_create_buffer[DT](n_q)
    var dk = ctx.enqueue_create_buffer[DT](n_k)
    var dc = ctx.enqueue_create_buffer[DT](n_t)
    var ds = ctx.enqueue_create_buffer[DT](n_t)
    var dqo = ctx.enqueue_create_buffer[DT](n_q)
    var dko = ctx.enqueue_create_buffer[DT](n_k)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dc, src_buf=hc)
    ctx.enqueue_copy(dst_buf=ds, src_buf=hs)
    ctx.synchronize()
    var q_pairs = Int(ROPE_TOKENS) * (Int(ROPE_Q_COLS) // Int(ROPE_HEAD_DIM)) * (Int(ROPE_HEAD_DIM) // 2)
    var k_pairs = Int(ROPE_TOKENS) * (Int(ROPE_K_COLS) // Int(ROPE_HEAD_DIM)) * (Int(ROPE_HEAD_DIM) // 2)
    ctx.enqueue_function[rope_gpu](
        dqo.unsafe_ptr(), dq.unsafe_ptr(), dc.unsafe_ptr(), ds.unsafe_ptr(),
        Int32(ROPE_Q_COLS), Int32(ROPE_HEAD_DIM), Int32(ROPE_TOKENS),
        grid_dim=(ceildiv(q_pairs, 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_function[rope_gpu](
        dko.unsafe_ptr(), dk.unsafe_ptr(), dc.unsafe_ptr(), ds.unsafe_ptr(),
        Int32(ROPE_K_COLS), Int32(ROPE_HEAD_DIM), Int32(ROPE_TOKENS),
        grid_dim=(ceildiv(k_pairs, 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hqo, src_buf=dqo)
    ctx.enqueue_copy(dst_buf=hko, src_buf=dko)
    ctx.synchronize()

    for i in range(n_q):
        if hqo[i] != f32_data(oq_cpu)[unsafe_offset=i]:
            raise Error(
                "rope(q): element " + String(i) + " gpu=" + String(hqo[i])
                + " cpu=" + String(f32_data(oq_cpu)[unsafe_offset=i])
                + " (suspect FMA contraction - record and switch to relative)"
            )
    for i in range(n_k):
        if hko[i] != f32_data(ok_cpu)[unsafe_offset=i]:
            raise Error(
                "rope(k): element " + String(i) + " gpu=" + String(hko[i])
                + " cpu=" + String(f32_data(ok_cpu)[unsafe_offset=i])
            )
    arena.keep_alive()
    print("*** ROPE GPU/SCALAR DIFF PASS *** diff= 0.0 (zero tolerance, no FMA contraction)")


def gather_diff_check(mut ctx: DeviceContext) raises:
    """embedding gather 的零容差差分。参照是 run 里 fp32 行拷贝的同构循环
    （run 无独立 gather 函数；q8 路由已被 dequant_q8_0 门覆盖）。"""
    var arena = Arena(1 << 18)
    var table = scratch(arena, Int(GATHER_VOCAB), Int(GATHER_COLS))
    fill(table, 45)
    var ids = List[Int]()
    ids.append(3)
    ids.append(0)
    ids.append(2)
    var dst_ref = arena.alloc(Int(GATHER_TOKENS) * Int(GATHER_COLS) * 4)
    var pd = dst_ref.unsafe_bitcast[Float32]()
    var pt = f32_data(table)
    for t in range(Int(GATHER_TOKENS)):
        var tok = ids[t]
        for c in range(Int(GATHER_COLS)):
            pd[unsafe_offset=t * Int(GATHER_COLS) + c] = pt[
                unsafe_offset=tok * Int(GATHER_COLS) + c
            ]

    var n_tab = Int(GATHER_VOCAB) * Int(GATHER_COLS)
    var n_d = Int(GATHER_TOKENS) * Int(GATHER_COLS)
    var ht = ctx.enqueue_create_host_buffer[DT](n_tab)
    var hi = ctx.enqueue_create_host_buffer[DType.int32](Int(GATHER_TOKENS))
    var hd = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    for i in range(n_tab):
        ht[i] = pt[unsafe_offset=i]
    for i in range(Int(GATHER_TOKENS)):
        hi[i] = Int32(ids[i])
    var dt = ctx.enqueue_create_buffer[DT](n_tab)
    var di = ctx.enqueue_create_buffer[DType.int32](Int(GATHER_TOKENS))
    var dd = ctx.enqueue_create_buffer[DT](n_d)
    ctx.enqueue_copy(dst_buf=dt, src_buf=ht)
    ctx.enqueue_copy(dst_buf=di, src_buf=hi)
    ctx.synchronize()
    ctx.enqueue_function[gather_gpu](
        dd.unsafe_ptr(), dt.unsafe_ptr(), di.unsafe_ptr(),
        Int32(GATHER_COLS), Int32(GATHER_TOKENS),
        grid_dim=(ceildiv(n_d, 64), 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    for i in range(n_d):
        if hd[i] != pd[unsafe_offset=i]:
            raise Error(
                "gather: element " + String(i) + " gpu=" + String(hd[i])
                + " ref=" + String(pd[unsafe_offset=i])
            )
    arena.keep_alive()
    print("*** GATHER GPU/REFERENCE DIFF PASS *** diff= 0.0 (zero tolerance)")


def main() raises:
    # 整门共享一个 DeviceContext：每个 check 各建一个会在第 ~10 个上下文处把
    # AsyncRT 拖崩（实测 crash，栈在 libAsyncRTMojoBindings 的 syscall 上）。
    # 上下文生命周期与门同长，check 之间顺序复用。
    var ctx = DeviceContext()
    rope2_diff_check(ctx)
    rmsnorm_diff_check(ctx)
    dequant_q4_diff_check(ctx)
    dequant_q8_diff_check(ctx)
    linear_diff_check(ctx)
    attention_diff_check(ctx)
    attention_par_diff_check(ctx)
    q4_matmul_diff_check(ctx)
    add_copy_diff_check(ctx)
    swiglu_diff_check(ctx)
    rope_diff_check(ctx)
    copy2_diff_check(ctx)
    rope_kv_diff_check(ctx)
    gather_diff_check(ctx)
    q4_fused_diff_check(ctx)
    linear_block_diff_check(ctx)
    q4_dual_fat_diff_check(ctx)
    qkv_tri_diff_check(ctx)
    f16_fat_diff_check(ctx)
    f16_dual_tri_head_diff_check(ctx)
