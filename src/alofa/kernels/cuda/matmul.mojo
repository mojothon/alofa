"""GPU 融合 q4 matmul —— 差分门之外的第一步：性能基线。

**为什么参照不是 AVX2**：`kernels/cpu/avx2.mojo` 的 `_matmul_q4_wide` 按
`j % 8` 切 SIMD 通道、`d` 折进 x、部分和跨通道归约 —— 累加顺序与标量版不同，
舍入也不同，拿它当差分参照会把"顺序不同"和"真算错"搅在一起。这里的参照是
**产品 dequant_q4_0 + 产品 linear** 的组合（两者都已被零容差/容差门钉死，
见 `tests/gpu/test_cuda_diff.mojo` 的账）。

**为什么这版还是一行一线程**：与 `basic.mojo` 的 `linear_gpu` 同一条纪律 ——
先让"GPU 算的数与 CPU 是同一个"成立，再谈性能。权重流是只读的、每个输出
元素独享自己那一行，一行一线程没有共享内存、没有规约顺序变量；性能数字
（GB/s）作为基线记录在账本里，优化的空间留给下一版。
"""

from std.gpu import global_idx, block_idx, thread_idx
from std.math import exp, sqrt
from std.memory import stack_allocation

from max.gpu import AddressSpace, barrier
from max.gpu.compute.mma import mma

from alofa.kernels.cpu.quant import Q4_BLOCK, Q4_BYTES, Q4_SCALE_BYTES
from alofa.kernels.cuda.basic import _fp16_bits_to_f32_device


def q4_matmul_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """`dst[row, o] = Σ_k dequant_q4_0(w[o, k]) · x[row, k]`，一个 (行, 输出列)
    一个线程。

    `w_blocks` 按行主序存块流：第 `o` 个输出行有 `inner/32` 个连续的 q4_0 块
    （与 AVX2 通路同一个布局假设）。`inner` 必须是 32 的倍数 —— Qwen 的
    896 与 4864 都满足；这不是新约束，标量版入口同样要求。
    """
    var o = Int(global_idx.x)
    var row = Int(global_idx.y)
    if o >= Int(out_cols):
        return
    var x_base = row * Int(inner)
    var nblk = Int(inner) // Q4_BLOCK
    var w_row = o * nblk
    var acc = Float64(0)
    for b in range(nblk):
        var base = (w_row + b) * Q4_BYTES
        var low = Int(w_blocks[unsafe_offset=base])
        var high = Int(w_blocks[unsafe_offset=base + 1])
        var d = _fp16_bits_to_f32_device(low | (high << 8))
        var kb = b * Q4_BLOCK
        for j in range(16):
            var byte = Int(w_blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
            acc += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(
                x[unsafe_offset=x_base + kb + j]
            )
            acc += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(
                x[unsafe_offset=x_base + kb + j + 16]
            )
    dst[unsafe_offset=row * Int(out_cols) + o] = Float32(acc)


def q4_matmul_block_gpu[THREADS: Int](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """`q4_matmul_gpu` 的 **block 规约版**：一个输出元素由 `THREADS` 个线程算。

    五十二的账说清了瓶颈：一行一线程在 rows=1 下只有 out_cols 个线程（A100
    容量的 ~2%），延迟主导 —— q4 读 7× 少的字节却比 fp32 慢，因为每线程串行
    解包 16 轮 nibble。这里把占用率提 `THREADS` 倍：每个 (行, 输出列) 分一个
    CUDA block，线程按块 stride 分工，`Float64` 部分和写共享内存做树规约。

    **规约顺序因此与串行版不同** —— 判据从"逐位一致"换成相对容差 1e-5（与
    AVX2 门同一个数）；确定性仍在：分工是静态的、树规约是固定形状，同输入
    重跑同一个结果。

    `THREADS` 必须是 2 的幂（树规约要求）。取值与形状有关：gate/up（inner=
    896 → 28 块）用 32；down（inner=4864 → 152 块）用 32 或 128 都能铺开。
    """
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var x_base = row * Int(inner)
    var nblk = Int(inner) // Q4_BLOCK
    var w_row = o * nblk
    var acc = Float64(0)
    var b = tid
    while b < nblk:
        var base = (w_row + b) * Q4_BYTES
        var low = Int(w_blocks[unsafe_offset=base])
        var high = Int(w_blocks[unsafe_offset=base + 1])
        var d = _fp16_bits_to_f32_device(low | (high << 8))
        var kb = b * Q4_BLOCK
        for j in range(16):
            var byte = Int(w_blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
            acc += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(
                x[unsafe_offset=x_base + kb + j]
            )
            acc += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(
                x[unsafe_offset=x_base + kb + j + 16]
            )
        b += THREADS

    var smem = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
    smem[tid] = acc
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        dst[unsafe_offset=row * Int(out_cols) + o] = Float32(smem[0])


def q4_matmul_fused_gpu[THREADS: Int, INNER: Int, HAS_BIAS: Bool](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """smem 冠军 + **bias 尾缀融合**（账本六十五）。

    decode 每层 21 次发射里 3 次是 q/k/v 的 bias 加——把加法挪进 matmul 的
    写回（`tid==0` 处），省 3 次发射/层。数值顺序与 CPU 通路**逐句一致**：
    标量侧是 `linear` 写回 f32(acc) 之后再 f32 加 bias，这里同样是
    `Float32(smem[0]) + bias[o]` 的 f32 加法 —— 差分预期 diff=0.0。
    `bias` 参数在 HAS_BIAS=False 的实例化里不读（传什么都行，约定传 dst）。
    """
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[unsafe_offset=k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    var nblk = Int(inner) // Q4_BLOCK
    var w_row = o * nblk
    var acc = Float64(0)
    var b = tid
    while b < nblk:
        var base = (w_row + b) * Q4_BYTES
        var low = Int(w_blocks[unsafe_offset=base])
        var high = Int(w_blocks[unsafe_offset=base + 1])
        var d = _fp16_bits_to_f32_device(low | (high << 8))
        var kb = b * Q4_BLOCK
        for j in range(16):
            var byte = Int(w_blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
            acc += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j])
            acc += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j + 16])
        b += THREADS

    var smem = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
    smem[unsafe_offset=tid] = acc
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] += smem[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        var v = Float32(smem[unsafe_offset=0])
        if HAS_BIAS:
            v = v + bias[unsafe_offset=o]
        dst[unsafe_offset=row * Int(out_cols) + o] = v


def q4_matmul_fat_gpu[THREADS: Int, INNER: Int, OUTS: Int, HAS_BIAS: Bool](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """fat-CTA：一个 block 处理 OUTS 个输出，x smem 与 x 载入相位摊薄（账本六十六）。

    六十五的证据：冠军每 CTA 只做 1 个输出，x 载入相位（28 次标量读 +
    barrier）摊不薄，SM 上并发 CTA 数撑不满。这里 OUTS 个输出共享同一份
    x smem：CTA 数降 OUTS 倍，每线程权重工作量升 OUTS 倍，载入相位的
    代价被 OUTS 个输出分摊。

    归约布局 `[OUTS][THREADS]`（sacc[oo*THREADS+tid]）：树归约每轮一次
    barrier 同时归约全部 OUTS 个部分和 —— barrier 数仍是 log2(THREADS)
    而不是 OUTS 倍。每输出的累加序与冠军逐位同构（线程按块跨步、树形
    归约），差分预期与 q4_matmul 检查一致。

    发射约定：`grid_dim=(ceildiv(out_cols, OUTS), rows)`、
    `block_dim=(THREADS,1,1)`；HAS_BIAS 语义同 fused（False 时 bias 占位
    须传不同源指针，六十五的 aliasing 规）。
    """
    var o0 = Int(block_idx.x) * OUTS
    var row = Int(block_idx.y)
    if o0 >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[unsafe_offset=k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    var nblk = Int(inner) // Q4_BLOCK
    var sacc = stack_allocation[OUTS * THREADS, Float64, address_space = AddressSpace.SHARED]()
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        var a = Float64(0)
        if o < Int(out_cols):
            var w_row = o * nblk
            var b = tid
            while b < nblk:
                var base = (w_row + b) * Q4_BYTES
                var low = Int(w_blocks[unsafe_offset=base])
                var high = Int(w_blocks[unsafe_offset=base + 1])
                var d = _fp16_bits_to_f32_device(low | (high << 8))
                var kb = b * Q4_BLOCK
                for j in range(16):
                    var byte = Int(w_blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
                    a += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j])
                    a += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j + 16])
                b += THREADS
        sacc[unsafe_offset=oo * THREADS + tid] = a
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for oo in range(OUTS):
                sacc[unsafe_offset=oo * THREADS + tid] += sacc[
                    unsafe_offset=oo * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                var v = Float32(sacc[unsafe_offset=oo * THREADS])
                if HAS_BIAS:
                    v = v + bias[unsafe_offset=o]
                dst[unsafe_offset=row * Int(out_cols) + o] = v


def q4_matmul_dual_fat_gpu[THREADS: Int, INNER: Int, OUTS: Int, HAS_BIAS: Bool](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_a: Pointer[UInt8, MutUntrackedOrigin],
    w_b: Pointer[UInt8, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    bias_a: Pointer[Float32, MutUntrackedOrigin],
    bias_b: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """gate/up 双矩阵 fat-CTA（账本六十六）：两 matmul 共享同一行 x。

    gate 和 up 的输入是同一个 normed 行 —— 分开跑就是两份 x smem 载入
    相位 + 两份 L2 x 流量。这里一个 CTA 处理 w_a 的 OUTS 个输出和 w_b
    的 OUTS 个输出，x 只载一次。每输出累加序与冠军逐位同构。

    发射约定：`grid_dim=(ceildiv(out_cols, OUTS), rows)`（两个矩阵输出
    数相同，各出 OUTS 个）、`block_dim=(THREADS,1,1)`。HAS_BIAS=False 时
    两个 bias 指针不读，但 aliasing 规要求全部指针两两不同源（六十五）。
    """
    var o0 = Int(block_idx.x) * OUTS
    var row = Int(block_idx.y)
    if o0 >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[unsafe_offset=k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    var nblk = Int(inner) // Q4_BLOCK
    var sacc = stack_allocation[2 * OUTS * THREADS, Float64, address_space = AddressSpace.SHARED]()
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        var a = Float64(0)
        if o < Int(out_cols):
            var w_row = o * nblk
            var b = tid
            while b < nblk:
                var base = (w_row + b) * Q4_BYTES
                var low = Int(w_a[unsafe_offset=base])
                var high = Int(w_a[unsafe_offset=base + 1])
                var d = _fp16_bits_to_f32_device(low | (high << 8))
                var kb = b * Q4_BLOCK
                for j in range(16):
                    var byte = Int(w_a[unsafe_offset=base + Q4_SCALE_BYTES + j])
                    a += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j])
                    a += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j + 16])
                b += THREADS
        sacc[unsafe_offset=oo * THREADS + tid] = a
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        var a = Float64(0)
        if o < Int(out_cols):
            var w_row = o * nblk
            var b = tid
            while b < nblk:
                var base = (w_row + b) * Q4_BYTES
                var low = Int(w_b[unsafe_offset=base])
                var high = Int(w_b[unsafe_offset=base + 1])
                var d = _fp16_bits_to_f32_device(low | (high << 8))
                var kb = b * Q4_BLOCK
                for j in range(16):
                    var byte = Int(w_b[unsafe_offset=base + Q4_SCALE_BYTES + j])
                    a += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j])
                    a += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j + 16])
                b += THREADS
        sacc[unsafe_offset=(OUTS + oo) * THREADS + tid] = a
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for oo in range(2 * OUTS):
                sacc[unsafe_offset=oo * THREADS + tid] += sacc[
                    unsafe_offset=oo * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                var v = Float32(sacc[unsafe_offset=oo * THREADS])
                if HAS_BIAS:
                    v = v + bias_a[unsafe_offset=o]
                dst_a[unsafe_offset=row * Int(out_cols) + o] = v
                var v2 = Float32(sacc[unsafe_offset=(OUTS + oo) * THREADS])
                if HAS_BIAS:
                    v2 = v2 + bias_b[unsafe_offset=o]
                dst_b[unsafe_offset=row * Int(out_cols) + o] = v2


def q4_matmul_qkv_gpu[THREADS: Int, INNER: Int, HAS_BIAS: Bool](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_q: Pointer[UInt8, MutUntrackedOrigin],
    w_k: Pointer[UInt8, MutUntrackedOrigin],
    w_v: Pointer[UInt8, MutUntrackedOrigin],
    dst_q: Pointer[Float32, MutUntrackedOrigin],
    dst_k: Pointer[Float32, MutUntrackedOrigin],
    dst_v: Pointer[Float32, MutUntrackedOrigin],
    bias_q: Pointer[Float32, MutUntrackedOrigin],
    bias_k: Pointer[Float32, MutUntrackedOrigin],
    bias_v: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    q_out: Int32,
    kv_out: Int32,
) -> None:
    """q/k/v 三 matmul 融合（账本六十七）：三者读同一 normed 行。

    一个 CTA 负责输出列 o，最多出三个输出：q（o < q_out）、k、v（o <
    kv_out，k/v 形状相同）。三分支显式展开 —— 不用指针三元选择（账本
    六十五：GPU 代码里会静默不写）。每输出累加序与冠军逐位同构，差分
    预期 0.0。x smem 只载一次，三次发射并一次。

    发射约定：`grid_dim=(max(q_out, kv_out), rows)`、
    `block_dim=(THREADS,1,1)`。HAS_BIAS=False 时三个 bias 不读，但
    aliasing 规要求全部指针两两不同源（六十五/六十六）。
    """
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[unsafe_offset=k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    var nblk = Int(inner) // Q4_BLOCK
    var sacc = stack_allocation[3 * THREADS, Float64, address_space = AddressSpace.SHARED]()
    @parameter
    for seg in range(3):
        var a = Float64(0)
        var active = False
        var w = w_q
        comptime if seg == 0:
            active = o < Int(q_out)
            w = w_q
        elif seg == 1:
            active = o < Int(kv_out)
            w = w_k
        else:
            active = o < Int(kv_out)
            w = w_v
        if active:
            var w_row = o * nblk
            var b = tid
            while b < nblk:
                var base = (w_row + b) * Q4_BYTES
                var low = Int(w[unsafe_offset=base])
                var high = Int(w[unsafe_offset=base + 1])
                var d = _fp16_bits_to_f32_device(low | (high << 8))
                var kb = b * Q4_BLOCK
                for j in range(16):
                    var byte = Int(w[unsafe_offset=base + Q4_SCALE_BYTES + j])
                    a += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j])
                    a += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(sx[unsafe_offset=kb + j + 16])
                b += THREADS
        sacc[unsafe_offset=seg * THREADS + tid] = a
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for seg in range(3):
                sacc[unsafe_offset=seg * THREADS + tid] += sacc[
                    unsafe_offset=seg * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        if o < Int(q_out):
            var v = Float32(sacc[unsafe_offset=tid])
            if HAS_BIAS:
                v = v + bias_q[unsafe_offset=o]
            dst_q[unsafe_offset=row * Int(q_out) + o] = v
        if o < Int(kv_out):
            var vk = Float32(sacc[unsafe_offset=THREADS + tid])
            var vv = Float32(sacc[unsafe_offset=2 * THREADS + tid])
            if HAS_BIAS:
                vk = vk + bias_k[unsafe_offset=o]
                vv = vv + bias_v[unsafe_offset=o]
            dst_k[unsafe_offset=row * Int(kv_out) + o] = vk
            dst_v[unsafe_offset=row * Int(kv_out) + o] = vv


def f16_bmm_fat_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, ACC: Bool, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """真 GEMM 分块 fat（账本八十九，prefill 主战场）：每 block 计算
    OUTS 个输出 × RB 行 —— 权重每 RB 行只读一次（w 载入提出 rr 环），
    x tile 走 L2（t×INNER f32 ≈ MB 级，L2 40MB 装得下）。

    与 batched-GEMV 版（八十八，grid y=t 每行重读全部权重）的差别
    就是 w 流量：t×矩阵 → ceil(t/RB)×矩阵。每输出累加序与 fat 核
    同构（k 升序 tid 步 THREADS*8、j 升序、THREADS 树收口）⇒ 与旧
    路径逐位一致。ACC 语义同 fat（残差就地累加）。RB=8 时 acc 16
    路 f64 全寄存器；sacc 树收口 16KB smem。
    """
    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, OUTS * RB](0)
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        if o < Int(out_cols):
            var w_base = o * Int(INNER)
            # k 迭代 @parameter 展开（账本九十 MLP 实验）：5 个 w load
            # 地址独立 ⇒ 编译器可全部提前发射（while 形态每线程每次
            # 仅 1 load 在飞，in-flight 字节差带宽需求两个数量级）。
            @parameter
            for it in range(THREADS * 8 * 0 + (INNER + THREADS * 8 - 1) // (THREADS * 8)):
                var k = tid * 8 + it * THREADS * 8
                if k < Int(INNER):
                    var wv = w.load[8, alignment=16](w_base + k)
                    @parameter
                    for rr in range(RB):
                        var r = r0 + rr
                        if r < Int(n_rows):
                            var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                            comptime if AF == DType.float64:
                                @parameter
                                for j in range(8):
                                    acc[oo * RB + rr] += Scalar[AF](
                                        Float64(xv[j]) * Float64(Float32(wv[j]))
                                    )
                            else:
                                # f32 快路（账本九十）：SIMD 乘 + 树收口。
                                # 组内树序与标量链合法重组差（f32 容差）。
                                var prod = xv * wv.cast[DType.float32]()
                                acc[oo * RB + rr] += Scalar[AF](prod.reduce_add())
    var sacc = stack_allocation[
        OUTS * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = Float32(
                            sacc[unsafe_offset=(oo * RB + rr) * THREADS]
                        )
                        comptime if ACC:
                            dst[unsafe_offset=r * Int(out_cols) + o] = (
                                dst[unsafe_offset=r * Int(out_cols) + o] + v
                            )
                        else:
                            dst[unsafe_offset=r * Int(out_cols) + o] = v


def f16_bmm_dual_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_a: Pointer[Float16, MutUntrackedOrigin],
    w_b: Pointer[Float16, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    dst_act: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """真 GEMM 分块 dual（账本八十九）：gate/up 双链 × RB 行，w 载入
    每 RB 行一次。x 为预归一化流（rmsnorm 前置单发，t 行一次 ——
    norm 树序与 in-kernel NORM 合法重组差，门容差覆盖）。每输出累加
    序与 dual 核同构（k 升序、j 升序、THREADS 树）⇒ 逐位同。act 尾
    同款：sigmoid(g)·g·up。RB=8 时 acc 32 路 f64。"""
    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, 2 * OUTS * RB](0)
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        if o < Int(out_cols):
            var w_base = o * Int(INNER)
            var k = tid * 8
            while k < Int(INNER):
                var wav = w_a.load[8, alignment=16](w_base + k)
                var wbv = w_b.load[8, alignment=16](w_base + k)
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                        comptime if AF == DType.float64:
                            @parameter
                            for j in range(8):
                                acc[(oo * RB + rr) * 2] += Scalar[AF](
                                    Float64(xv[j]) * Float64(Float32(wav[j]))
                                )
                                acc[(oo * RB + rr) * 2 + 1] += Scalar[AF](
                                    Float64(xv[j]) * Float64(Float32(wbv[j]))
                                )
                        else:
                            # f32 快路同 fat（账本九十）
                            var wa32 = wav.cast[DType.float32]()
                            var wb32 = wbv.cast[DType.float32]()
                            acc[(oo * RB + rr) * 2] += Scalar[AF](
                                (xv * wa32).reduce_add()
                            )
                            acc[(oo * RB + rr) * 2 + 1] += Scalar[AF](
                                (xv * wb32).reduce_add()
                            )
                k += THREADS * 8
    var sacc = stack_allocation[
        2 * OUTS * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(2 * OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(2 * OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = Float32(
                            sacc[unsafe_offset=(oo * RB + rr) * 2 * THREADS]
                        )
                        var v2 = Float32(
                            sacc[unsafe_offset=((oo * RB + rr) * 2 + 1) * THREADS]
                        )
                        dst_a[unsafe_offset=r * Int(out_cols) + o] = v
                        dst_b[unsafe_offset=r * Int(out_cols) + o] = v2
                        var g = Float64(v)
                        dst_act[unsafe_offset=r * Int(out_cols) + o] = Float32(
                            g / (Float64(1) + exp(-g)) * Float64(v2)
                        )


def f16_bmm_qkv_gpu[THREADS: Int, INNER: Int, RB: Int, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_q: Pointer[Float16, MutUntrackedOrigin],
    w_k: Pointer[Float16, MutUntrackedOrigin],
    w_v: Pointer[Float16, MutUntrackedOrigin],
    bias_q: Pointer[Float32, MutUntrackedOrigin],
    bias_k: Pointer[Float32, MutUntrackedOrigin],
    bias_v: Pointer[Float32, MutUntrackedOrigin],
    dst_q: Pointer[Float32, MutUntrackedOrigin],
    dst_k: Pointer[Float32, MutUntrackedOrigin],
    dst_v: Pointer[Float32, MutUntrackedOrigin],
    q_out: Int32,
    kv_out: Int32,
    n_rows: Int32,
) -> None:
    """真 GEMM 分块 qkv（账本八十九）：q/k/v 三链 × RB 行（OUTS=1，
    grid x = max(q_out, kv_out)，o 守卫分派三路），w 每 RB 行一次。
    x 为预归一化流。每输出累加序与 qkv 核同构 ⇒ 逐位同；bias 尾加
    同款。acc 24 路 f64。"""
    var o = Int(block_idx.x)
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, 4 * RB](0)
    @parameter
    for seg in range(3):
        var active = False
        var w = w_q
        comptime if seg == 0:
            active = o < Int(q_out)
            w = w_q
        elif seg == 1:
            active = o < Int(kv_out)
            w = w_k
        else:
            active = o < Int(kv_out)
            w = w_v
        if active:
            var w_base = o * Int(INNER)
            var k = tid * 8
            while k < Int(INNER):
                var wv = w.load[8, alignment=16](w_base + k)
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                        comptime if AF == DType.float64:
                            @parameter
                            for j in range(8):
                                acc[seg * RB + rr] += Scalar[AF](
                                    Float64(xv[j]) * Float64(Float32(wv[j]))
                                )
                        else:
                            # f32 快路同 fat（账本九十）
                            var prod = xv * wv.cast[DType.float32]()
                            acc[seg * RB + rr] += Scalar[AF](prod.reduce_add())
                k += THREADS * 8
    var sacc = stack_allocation[
        3 * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(3 * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(3 * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        if o < Int(q_out):
            @parameter
            for rr in range(RB):
                var r = r0 + rr
                if r < Int(n_rows):
                    var v = Float32(sacc[unsafe_offset=rr * THREADS])
                    v = v + bias_q[unsafe_offset=o]
                    dst_q[unsafe_offset=r * Int(q_out) + o] = v
        if o < Int(kv_out):
            @parameter
            for rr in range(RB):
                var r = r0 + rr
                if r < Int(n_rows):
                    var vk = Float32(sacc[unsafe_offset=(RB + rr) * THREADS])
                    var vv = Float32(sacc[unsafe_offset=(2 * RB + rr) * THREADS])
                    vk = vk + bias_k[unsafe_offset=o]
                    vv = vv + bias_v[unsafe_offset=o]
                    dst_k[unsafe_offset=r * Int(kv_out) + o] = vk
                    dst_v[unsafe_offset=r * Int(kv_out) + o] = vv

def f16_matmul_fat_si_gpu[THREADS: Int, INNER: Int, OUTS: Int, SPLIT: Int](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dpart: Pointer[Float64, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """split-inner fat（账本八十七）：归约维切 SPLIT 份，块数 ×SPLIT。

    down 现状 448 块 × 128 线程、x smem 19KB ⇒ 驻留 ~7 块/SM，延迟
    界（管线真值 670GB/s）。分块后每块只装 x 的 1/SPLIT chunk（smem
    4.9KB）与 w 行的 1/SPLIT，块数 ×SPLIT；块内链序（k 升序、j 升序）
    与树结构与旧同构，跨块加法序由 combine 的 s 升序固定 ⇒ f64 重结
    合，走容差门。INNER 与 INNER//SPLIT 须为 8 的倍数。
    """
    var o0 = Int(block_idx.x) * OUTS
    var s = Int(block_idx.y)
    if o0 >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var ch = INNER // SPLIT
    var sx = stack_allocation[INNER // SPLIT, Float32, address_space = AddressSpace.SHARED]()
    var x_base = s * ch
    # x chunk 向量载入（16B 对齐：s*ch 与 ch 均为 4 的倍数）
    var k = tid * 4
    while k < ch:
        var xv = x.load[4](x_base + k)
        @parameter
        for j in range(4):
            sx[unsafe_offset=k + j] = xv[j]
        k += THREADS * 4
    barrier()
    var sacc = stack_allocation[OUTS * THREADS, Float64, address_space = AddressSpace.SHARED]()
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        var a = Float64(0)
        if o < Int(out_cols):
            var w_base = o * Int(INNER) + x_base
            k = tid * 8
            while k < ch:
                var wv = w.load[8, alignment=16](w_base + k)
                @parameter
                for j in range(8):
                    a += Float64(sx[unsafe_offset=k + j]) * Float64(Float32(wv[j]))
                k += THREADS * 8
        sacc[unsafe_offset=oo * THREADS + tid] = a
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for oo in range(OUTS):
                sacc[unsafe_offset=oo * THREADS + tid] += sacc[
                    unsafe_offset=oo * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                dpart[unsafe_offset=o * SPLIT + s] = sacc[
                    unsafe_offset=oo * THREADS
                ]


def fat_si_comb_gpu[OUTS: Int, SPLIT: Int, ACC: Bool](
    dpart: Pointer[Float64, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
) -> None:
    """split 收口微核（账本八十七）：每线程一输出，s 升序 f64 求和。

    SPLIT 份部分和按 s 升序固定加法序；ACC 语义同 fat（残差累加，
    舍入顺序与"先写 f32 再相加"逐位一致）。
    """
    var o0 = Int(block_idx.x) * OUTS
    var tid = Int(thread_idx.x)
    var o = o0 + tid
    if o < Int(out_cols):
        var a = Float64(0)
        @parameter
        for s in range(SPLIT):
            a += dpart[unsafe_offset=o * SPLIT + s]
        var v = Float32(a)
        if ACC:
            dst[unsafe_offset=o] = dst[unsafe_offset=o] + v
        else:
            dst[unsafe_offset=o] = v


def f16_matmul_fat_gpu[THREADS: Int, INNER: Int, OUTS: Int, HAS_BIAS: Bool, ACC: Bool = False](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """f16 预反量化 GEMV（账本六十八）：装载期把 q4 反量化成 f16。

    f16 字节流量是 q4 的 3.5 倍，但实测离带宽下限还差 8 倍 —— 瓶颈是
    18B 块的标量字节加载链而非带宽。f16 路径把内层循环简化成对齐的
    f16 读 + FMA：`load[8]`（16B 对齐：o*INNER 与 k 跨步都是 8 的倍数）
    一次取 8 个权重，寄存器宽度只有 load[16]-u8 的一半。

    数值：权重 f16 舍入（每权重 rel err ≤ 2^-11），f64 累加；门判据用
    f16 专用容差（账本六十八，实测定标）。INNER 必须是 8 的倍数
    （k=tid*8 起步、THREADS*8 跨步的覆盖条件）。
    """
    var o0 = Int(block_idx.x) * OUTS
    var row = Int(block_idx.y)
    if o0 >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[unsafe_offset=k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    var sacc = stack_allocation[OUTS * THREADS, Float64, address_space = AddressSpace.SHARED]()
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        var a = Float64(0)
        if o < Int(out_cols):
            var w_base = o * Int(INNER)
            k = tid * 8
            while k < Int(INNER):
                var wv = w.load[8, alignment=16](w_base + k)
                @parameter
                for j in range(8):
                    a += Float64(sx[unsafe_offset=k + j]) * Float64(Float32(wv[j]))
                k += THREADS * 8
        sacc[unsafe_offset=oo * THREADS + tid] = a
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for oo in range(OUTS):
                sacc[unsafe_offset=oo * THREADS + tid] += sacc[
                    unsafe_offset=oo * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                var v = Float32(sacc[unsafe_offset=oo * THREADS])
                if HAS_BIAS:
                    v = v + bias[unsafe_offset=o]
                # ACC（账本七十）：残差累加进 dst（o/down 直接写 d_x，省
                # 一次 add_into 发射）。舍入顺序与"先写 f32 proj 再
                # f32+f32 相加"逐位一致。
                if ACC:
                    var acc0 = dst[unsafe_offset=row * Int(out_cols) + o]
                    dst[unsafe_offset=row * Int(out_cols) + o] = acc0 + v
                else:
                    dst[unsafe_offset=row * Int(out_cols) + o] = v


def f16_matmul_dual_gpu[
    THREADS: Int, INNER: Int, OUTS: Int, HAS_BIAS: Bool, HAS_ACT: Bool = False, NORM: Bool = False
](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_a: Pointer[Float16, MutUntrackedOrigin],
    w_b: Pointer[Float16, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    bias_a: Pointer[Float32, MutUntrackedOrigin],
    bias_b: Pointer[Float32, MutUntrackedOrigin],
    dst_act: Pointer[Float32, MutUntrackedOrigin],
    rms_w: Pointer[Float32, MutUntrackedOrigin],
    eps: Float32,
    inner: Int32,
    out_cols: Int32,
) -> None:
    """gate/up 的 f16 dual-fat（账本六十八）：两矩阵共享 x smem，结构同
    q4_matmul_dual_fat，内层换成对齐 f16 向量读。

    HAS_ACT（账本七十）：tid==0 写回后顺带算 `act = silu(gate)*up` ——
    与 `swiglu_gpu` 同序（f64 中间量、`v/(1+exp(-v))`、乘 up、一次
    Float32 收尾），省一次发射。`dst_act` 在 HAS_ACT=False 时不用。

    NORM（账本七十一）：x 载入 smem 后就地 rmsnorm（同 tri/head 的 NORM
    块）—— 第二个 norm 发射也省掉。"""
    var o0 = Int(block_idx.x) * OUTS
    var row = Int(block_idx.y)
    if o0 >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[unsafe_offset=k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    comptime if NORM:
        var nlocal = Float64(0)
        var nk = tid
        while nk < Int(INNER):
            var nv = Float64(sx[unsafe_offset=nk])
            nlocal += nv * nv
            nk += THREADS
        var sred = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
        sred[unsafe_offset=tid] = nlocal
        barrier()
        var nstride = THREADS // 2
        while nstride > 0:
            if tid < nstride:
                sred[unsafe_offset=tid] = (
                    sred[unsafe_offset=tid] + sred[unsafe_offset=tid + nstride]
                )
            barrier()
            nstride //= 2
        var nscale = Float64(1.0) / sqrt(
            sred[unsafe_offset=0] / Float64(INNER) + Float64(eps)
        )
        nk = tid
        while nk < Int(INNER):
            sx[unsafe_offset=nk] = Float32(
                Float64(sx[unsafe_offset=nk])
                * nscale
                * Float64(rms_w[unsafe_offset=nk])
            )
            nk += THREADS
        barrier()
    var sacc = stack_allocation[2 * OUTS * THREADS, Float64, address_space = AddressSpace.SHARED]()
    # k 外层、行内层（账本七十四）：旧结构 8 条 (half,oo) 链顺序执行，
    # 每链 while 内同时只有 1 个 16B load 在飞 —— 访存延迟主导
    # （480GB/s，round 72 已排除占据率）。改为每步 8 个 load 背靠背
    # 发射、f64 链交错。每行链内求和序不变（k 升序、j 升序，操作数
    # 逐一同旧）⇒ 与旧结构逐位一致。
    var acc = InlineArray[Float64, 2 * OUTS](fill=Float64(0))
    var wrow = InlineArray[Pointer[Float16, MutUntrackedOrigin], 2 * OUTS](fill=w_a)
    var wbase = InlineArray[Int, 2 * OUTS](fill=0)
    var valid = InlineArray[Bool, 2 * OUTS](fill=False)
    @parameter
    for half in range(2):
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            valid[half * OUTS + oo] = o < Int(out_cols)
            comptime if half == 1:
                wrow[half * OUTS + oo] = w_b
            wbase[half * OUTS + oo] = o * Int(INNER)
    k = tid * 8
    while k < Int(INNER):
        var xv = InlineArray[Float64, 8](fill=Float64(0))
        @parameter
        for j in range(8):
            xv[j] = Float64(sx[unsafe_offset=k + j])
        @parameter
        for idx in range(2 * OUTS):
            if valid[idx]:
                var wv = wrow[idx].load[8, alignment=16](wbase[idx] + k)
                @parameter
                for j in range(8):
                    acc[idx] += xv[j] * Float64(Float32(wv[j]))
        k += THREADS * 8
    @parameter
    for idx in range(2 * OUTS):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for oo in range(2 * OUTS):
                sacc[unsafe_offset=oo * THREADS + tid] += sacc[
                    unsafe_offset=oo * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                var v = Float32(sacc[unsafe_offset=oo * THREADS])
                var v2 = Float32(sacc[unsafe_offset=(OUTS + oo) * THREADS])
                if HAS_BIAS:
                    v = v + bias_a[unsafe_offset=o]
                    v2 = v2 + bias_b[unsafe_offset=o]
                dst_a[unsafe_offset=row * Int(out_cols) + o] = v
                dst_b[unsafe_offset=row * Int(out_cols) + o] = v2
                comptime if HAS_ACT:
                    var g = Float64(v)
                    dst_act[unsafe_offset=row * Int(out_cols) + o] = Float32(
                        g / (Float64(1) + exp(-g)) * Float64(v2)
                    )


def f16_matmul_qkv_fat_gpu[
    THREADS: Int, INNER: Int, OUTS: Int, HAS_BIAS: Bool
](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst_q: Pointer[Float32, MutUntrackedOrigin],
    dst_k: Pointer[Float32, MutUntrackedOrigin],
    dst_v: Pointer[Float32, MutUntrackedOrigin],
    bias_q: Pointer[Float32, MutUntrackedOrigin],
    bias_k: Pointer[Float32, MutUntrackedOrigin],
    bias_v: Pointer[Float32, MutUntrackedOrigin],
    rms_w: Pointer[Float32, MutUntrackedOrigin],
    eps: Float32,
    inner: Int32,
    q_out: Int32,
    kv_out: Int32,
) -> None:
    """qkv decode 融合核（账本一百一十五-b）：fat 结构 + 就地 rmsnorm
    + 三路 bias，权重为 cat3 打包单矩阵 [q+2kv, INNER] f16。

    现役 tri 核的病：3 段串行（每段独立 7 屏障树 = 21 屏障/块），
    每线程仅 ~3 次 f16 load——冷 DRAM 下 220GB/s（段测）。fat 化后
    每线程负载同 o/down 族，隔离对决 1.51×（同钟同口径，探针
    decg115 v1 vs v11）。列映射同 tri：o < q_out → q；否则 k/v。
    数值：norm 归约序与 tri NORM 块一致（f64 树 over THREADS）；
    GEMM 累加序同 fat 族（k=tid*8 升序）——f64 重结合走容差门。
    INNER 须为 8 的倍数；grid=(q_out+2*kv_out)//OUTS。"""

    var o0 = Int(block_idx.x) * OUTS
    var total = Int(q_out) + 2 * Int(kv_out)
    if o0 >= total:
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[
        INNER, Float32, address_space = AddressSpace.SHARED
    ]()
    var k = tid
    while k < Int(INNER):
        sx[unsafe_offset=k] = x[unsafe_offset=k]
        k += THREADS
    barrier()
    var nlocal = Float64(0)
    var nk = tid
    while nk < Int(INNER):
        var nv = Float64(sx[unsafe_offset=nk])
        nlocal += nv * nv
        nk += THREADS
    var sred = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    sred[unsafe_offset=tid] = nlocal
    barrier()
    var nstride = THREADS // 2
    while nstride > 0:
        if tid < nstride:
            sred[unsafe_offset=tid] = (
                sred[unsafe_offset=tid] + sred[unsafe_offset=tid + nstride]
            )
        barrier()
        nstride //= 2
    var nscale = Float64(1.0) / sqrt(
        sred[unsafe_offset=0] / Float64(INNER) + Float64(eps)
    )
    nk = tid
    while nk < Int(INNER):
        sx[unsafe_offset=nk] = Float32(
            Float64(sx[unsafe_offset=nk])
            * nscale
            * Float64(rms_w[unsafe_offset=nk])
        )
        nk += THREADS
    barrier()
    var sacc = stack_allocation[
        OUTS * THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        var a = Float64(0)
        if o < total:
            var w_base = o * Int(INNER)
            k = tid * 8
            while k < Int(INNER):
                var wv = w.load[8, alignment=16](w_base + k)
                @parameter
                for j in range(8):
                    a += Float64(sx[unsafe_offset=k + j]) * Float64(
                        Float32(wv[j])
                    )
                k += THREADS * 8
        sacc[unsafe_offset=oo * THREADS + tid] = a
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for oo in range(OUTS):
                sacc[unsafe_offset=oo * THREADS + tid] += sacc[
                    unsafe_offset=oo * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < total:
                var v = Float32(sacc[unsafe_offset=oo * THREADS])
                comptime if HAS_BIAS:
                    if o < Int(q_out):
                        dst_q[unsafe_offset=o] = v + bias_q[unsafe_offset=o]
                    elif o < Int(q_out) + Int(kv_out):
                        var r = o - Int(q_out)
                        dst_k[unsafe_offset=r] = v + bias_k[unsafe_offset=r]
                    else:
                        var r = o - Int(q_out) - Int(kv_out)
                        dst_v[unsafe_offset=r] = v + bias_v[unsafe_offset=r]
                else:
                    if o < Int(q_out):
                        dst_q[unsafe_offset=o] = v
                    elif o < Int(q_out) + Int(kv_out):
                        var r = o - Int(q_out)
                        dst_k[unsafe_offset=r] = v
                    else:
                        var r = o - Int(q_out) - Int(kv_out)
                        dst_v[unsafe_offset=r] = v


def cat3_f16_gpu(
    dst: Pointer[Float16, MutUntrackedOrigin],
    a: Pointer[Float16, MutUntrackedOrigin],
    b: Pointer[Float16, MutUntrackedOrigin],
    c: Pointer[Float16, MutUntrackedOrigin],
    na: Int32,
    nb: Int32,
    nc: Int32,
) -> None:
    """三矩阵拼接（账本一百一十五-b，qkv 打包用，一次成本）。"""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(na):
        dst[unsafe_offset=i] = a[unsafe_offset=i]
    elif i < Int(na) + Int(nb):
        dst[unsafe_offset=i] = b[unsafe_offset=i - Int(na)]
    elif i < Int(na) + Int(nb) + Int(nc):
        dst[unsafe_offset=i] = c[unsafe_offset=i - Int(na) - Int(nb)]


def f16_matmul_qkv_gpu[THREADS: Int, INNER: Int, HAS_BIAS: Bool, NORM: Bool = False](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_q: Pointer[Float16, MutUntrackedOrigin],
    w_k: Pointer[Float16, MutUntrackedOrigin],
    w_v: Pointer[Float16, MutUntrackedOrigin],
    dst_q: Pointer[Float32, MutUntrackedOrigin],
    dst_k: Pointer[Float32, MutUntrackedOrigin],
    dst_v: Pointer[Float32, MutUntrackedOrigin],
    bias_q: Pointer[Float32, MutUntrackedOrigin],
    bias_k: Pointer[Float32, MutUntrackedOrigin],
    bias_v: Pointer[Float32, MutUntrackedOrigin],
    rms_w: Pointer[Float32, MutUntrackedOrigin],
    eps: Float32,
    inner: Int32,
    q_out: Int32,
    kv_out: Int32,
) -> None:
    """q/k/v 的 f16 三融合（账本六十八）：结构同 q4_matmul_qkv_gpu，
    内层换成对齐 f16 向量读。INNER 是 8 的倍数。

    NORM（账本七十一）：x 载入 smem 后就地 rmsnorm（同 `linear_fat_f16_gpu`
    的 NORM 块，账本七十一）—— 解码层链的第一发直接吃残差流 d_x，
    `rmsnorm` 单发整个省掉。"""
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[unsafe_offset=k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    comptime if NORM:
        var nlocal = Float64(0)
        var nk = tid
        while nk < Int(INNER):
            var nv = Float64(sx[unsafe_offset=nk])
            nlocal += nv * nv
            nk += THREADS
        var sred = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
        sred[unsafe_offset=tid] = nlocal
        barrier()
        var nstride = THREADS // 2
        while nstride > 0:
            if tid < nstride:
                sred[unsafe_offset=tid] = (
                    sred[unsafe_offset=tid] + sred[unsafe_offset=tid + nstride]
                )
            barrier()
            nstride //= 2
        var nscale = Float64(1.0) / sqrt(
            sred[unsafe_offset=0] / Float64(INNER) + Float64(eps)
        )
        nk = tid
        while nk < Int(INNER):
            sx[unsafe_offset=nk] = Float32(
                Float64(sx[unsafe_offset=nk])
                * nscale
                * Float64(rms_w[unsafe_offset=nk])
            )
            nk += THREADS
        barrier()
    var sacc = stack_allocation[3 * THREADS, Float64, address_space = AddressSpace.SHARED]()
    @parameter
    for seg in range(3):
        var a = Float64(0)
        var active = False
        var w = w_q
        comptime if seg == 0:
            active = o < Int(q_out)
            w = w_q
        elif seg == 1:
            active = o < Int(kv_out)
            w = w_k
        else:
            active = o < Int(kv_out)
            w = w_v
        if active:
            var w_base = o * Int(INNER)
            k = tid * 8
            while k < Int(INNER):
                var wv = w.load[8, alignment=16](w_base + k)
                @parameter
                for j in range(8):
                    a += Float64(sx[unsafe_offset=k + j]) * Float64(Float32(wv[j]))
                k += THREADS * 8
        sacc[unsafe_offset=seg * THREADS + tid] = a
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for seg in range(3):
                sacc[unsafe_offset=seg * THREADS + tid] += sacc[
                    unsafe_offset=seg * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        if o < Int(q_out):
            var v = Float32(sacc[unsafe_offset=tid])
            if HAS_BIAS:
                v = v + bias_q[unsafe_offset=o]
            dst_q[unsafe_offset=row * Int(q_out) + o] = v
        if o < Int(kv_out):
            var vk = Float32(sacc[unsafe_offset=THREADS + tid])
            var vv = Float32(sacc[unsafe_offset=2 * THREADS + tid])
            if HAS_BIAS:
                vk = vk + bias_k[unsafe_offset=o]
                vv = vv + bias_v[unsafe_offset=o]
            dst_k[unsafe_offset=row * Int(kv_out) + o] = vk
            dst_v[unsafe_offset=row * Int(kv_out) + o] = vv


def q4_matmul_smem_gpu[THREADS: Int, INNER: Int](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """**x 行协作载入共享内存**的 blk32 —— 当前的性能冠军。

    分解实验（tests/gpu/prof_q4_blk.mojo，账本五十五）把 ~40µs 拆开：f64
    无辜（f32_acc 全速）、barrier 骨架 12µs、大头在加载路径。逐个排除后
    真凶是 **x 的全局读取**：每个 CTA 重读整行 x，而线程读 `x[kb+j]` 时
    warp 内地址间隔 128B，一条指令 32 个散射 32B 扇区，放大 ~8×。权重
    byte load 反而是干净的（邻接块 ~600B，L1 第一笔后全命中）。

    修法教科书式：CTA 开头协作把 x 行载入 smem（相邻线程读相邻地址，真
    合并），点积全部从 smem 取。`INNER` 是 comptime（smem 大小要编译期
    确定；Qwen 的 896 与 4864 各实例化一次，4864×4B=19.5KB，远低于 A100
    每 CTA 上限）。累加结构与 blk32 一致 —— 差分实测 diff=0.0。
    """
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    var nblk = Int(inner) // Q4_BLOCK
    var w_row = o * nblk
    var acc = Float64(0)
    var b = tid
    while b < nblk:
        var base = (w_row + b) * Q4_BYTES
        var low = Int(w_blocks[unsafe_offset=base])
        var high = Int(w_blocks[unsafe_offset=base + 1])
        var d = _fp16_bits_to_f32_device(low | (high << 8))
        var kb = b * Q4_BLOCK
        for j in range(16):
            var byte = Int(w_blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
            acc += Float64(Float32((byte & 0x0F) - 8) * d) * Float64(sx[kb + j])
            acc += Float64(Float32(((byte >> 4) & 0x0F) - 8) * d) * Float64(sx[kb + j + 16])
        b += THREADS

    var smem = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
    smem[tid] = acc
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        dst[unsafe_offset=row * Int(out_cols) + o] = Float32(smem[0])


def q4_matmul_vec_gpu[THREADS: Int](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """`q4_matmul_block_gpu` 的**向量加载版**：块数据一次 128-bit 读。

    五十三的诊断：块规约版 ~40µs 全是执行时间，瓶颈是每线程逐字节读 18B 块
    —— warp 内 32 线程的 byte load 各落 18B 远，一条指令打 32 个散射 32B
    扇区，读放大 ~32×。这里每线程认领整块：2B scale 拆两次字节读、16B 数据
    用 `load[16]` 一次进寄存器，同一 warp 的向量加载只覆盖 ~18 个扇区
    （读放大降到 ~2×，L2 吸收）。

    累加结构与 block 版完全一致（线程内按块升序串行 + 树规约），正确性判据
    同 block 版：相对容差，预期依旧 diff=0.0。
    """
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var x_base = row * Int(inner)
    var nblk = Int(inner) // Q4_BLOCK
    var w_row = o * nblk
    var acc = Float64(0)
    var b = tid
    while b < nblk:
        var base = (w_row + b) * Q4_BYTES
        var low = Int(w_blocks[unsafe_offset=base])
        var high = Int(w_blocks[unsafe_offset=base + 1])
        var d = _fp16_bits_to_f32_device(low | (high << 8))
        var nib = w_blocks.load[16](base + Q4_SCALE_BYTES)
        var kb = b * Q4_BLOCK
        # `nib[j]` 是对 SIMD 寄存器的动态索引 —— 循环不展开就会被抖到 local
        # memory（首测比 byte 版还慢 1.6×，就是栽在这里）。@parameter 展开后
        # 索引静态化，16 个 nibble 全在寄存器里。
        @parameter
        for j in range(16):
            acc += Float64(Float32(Int(nib[j] & UInt8(0x0F)) - 8) * d) * Float64(
                x[unsafe_offset=x_base + kb + j]
            )
            acc += Float64(Float32(Int(nib[j] >> 4) - 8) * d) * Float64(
                x[unsafe_offset=x_base + kb + j + 16]
            )
        b += THREADS

    var smem = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
    smem[tid] = acc
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        dst[unsafe_offset=row * Int(out_cols) + o] = Float32(smem[0])


def f16_matmul_qkv_mc_gpu[
    THREADS: Int, INNER: Int, COLS: Int, HAS_BIAS: Bool, NORM: Bool = False,
](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_q: Pointer[Float16, MutUntrackedOrigin],
    w_k: Pointer[Float16, MutUntrackedOrigin],
    w_v: Pointer[Float16, MutUntrackedOrigin],
    dst_q: Pointer[Float32, MutUntrackedOrigin],
    dst_k: Pointer[Float32, MutUntrackedOrigin],
    dst_v: Pointer[Float32, MutUntrackedOrigin],
    bias_q: Pointer[Float32, MutUntrackedOrigin],
    bias_k: Pointer[Float32, MutUntrackedOrigin],
    bias_v: Pointer[Float32, MutUntrackedOrigin],
    rms_w: Pointer[Float32, MutUntrackedOrigin],
    eps: Float32,
    inner: Int32,
    q_out: Int32,
    kv_out: Int32,
) -> None:
    """多列 tri（账本八十四）：每块算 COLS 个输出（三段各自越界裁剪）
    —— x 的 smem 载入与 rmsnorm 摊薄 COLS 倍，块数 ÷COLS（896→112），
    每线程 COLS×3 条独立 16B load 在飞（旧版 3 条）。旧版 245GB/s
    病根：896 块每块只算 1 输出，latency 主导。
    数值：k 序（tid*8 步 THREADS*8）与树规约组序同旧 ⇒ 逐位同。
    """
    var o0 = Int(block_idx.x) * COLS
    var row = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(inner)
    var k = tid
    while k < Int(inner):
        sx[unsafe_offset=k] = x[unsafe_offset=x_base + k]
        k += THREADS
    barrier()
    comptime if NORM:
        var nlocal = Float64(0)
        var nk = tid
        while nk < Int(INNER):
            var nv = Float64(sx[unsafe_offset=nk])
            nlocal += nv * nv
            nk += THREADS
        var sred = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
        sred[unsafe_offset=tid] = nlocal
        barrier()
        var nstride = THREADS // 2
        while nstride > 0:
            if tid < nstride:
                sred[unsafe_offset=tid] = (
                    sred[unsafe_offset=tid] + sred[unsafe_offset=tid + nstride]
                )
            barrier()
            nstride //= 2
        var nscale = Float64(1.0) / sqrt(
            sred[unsafe_offset=0] / Float64(INNER) + Float64(eps)
        )
        nk = tid
        while nk < Int(INNER):
            sx[unsafe_offset=nk] = Float32(
                Float64(sx[unsafe_offset=nk])
                * nscale
                * Float64(rms_w[unsafe_offset=nk])
            )
            nk += THREADS
        barrier()
    var nq = Int(q_out)
    var nkv = Int(kv_out)
    var acc = InlineArray[Float64, 3 * COLS](fill=Float64(0))
    @parameter
    for seg in range(3):
        var w = w_q
        comptime if seg == 0:
            w = w_q
        elif seg == 1:
            w = w_k
        else:
            w = w_v
        var nseg = nq
        comptime if seg != 0:
            nseg = nkv
        var kk = tid * 8
        while kk < Int(INNER):
            @parameter
            for c in range(COLS):
                var o = o0 + c
                if o < nseg:
                    var wv = w.load[8, alignment=16](o * Int(INNER) + kk)
                    @parameter
                    for j in range(8):
                        acc[seg * COLS + c] += Float64(
                            sx[unsafe_offset=kk + j]
                        ) * Float64(Float32(wv[j]))
            kk += THREADS * 8
    var sacc = stack_allocation[
        3 * COLS * THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for i in range(3 * COLS):
        sacc[unsafe_offset=i * THREADS + tid] = acc[i]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for i in range(3 * COLS):
                sacc[unsafe_offset=i * THREADS + tid] += sacc[
                    unsafe_offset=i * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for c in range(COLS):
            var o = o0 + c
            if o < Int(q_out):
                var v = Float32(sacc[unsafe_offset=c * THREADS])
                if HAS_BIAS:
                    v = v + bias_q[unsafe_offset=o]
                dst_q[unsafe_offset=row * Int(q_out) + o] = v
            if o < Int(kv_out):
                var vk = Float32(sacc[unsafe_offset=COLS * THREADS + c * THREADS])
                var vv = Float32(
                    sacc[unsafe_offset=2 * COLS * THREADS + c * THREADS]
                )
                if HAS_BIAS:
                    vk = vk + bias_k[unsafe_offset=o]
                    vv = vv + bias_v[unsafe_offset=o]
                dst_k[unsafe_offset=row * Int(kv_out) + o] = vk
                dst_v[unsafe_offset=row * Int(kv_out) + o] = vv


def f16_matmul_dual_nx_gpu[
    THREADS: Int, INNER: Int, OUTS: Int, HAS_ACT: Bool = False,
](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_a: Pointer[Float16, MutUntrackedOrigin],
    w_b: Pointer[Float16, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    bias_a: Pointer[Float32, MutUntrackedOrigin],
    bias_b: Pointer[Float32, MutUntrackedOrigin],
    dst_act: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
) -> None:
    """dual 无 smem 变体（账本八十五）：x 已由 rmsnorm 单发归一（调用方
    保证），直读全局（行 3.5KB，L1 常驻）。去 sx 载入相 + NORM 相
    （1216 块 × 冗余 norm 树全免）—— 换 o2 时块数翻倍（2432 块 ×
    2 warp = 44 warp/SM，旧结构 smem+X 税锁死 22）。dot 循环 k 序/
    j 序/树序同 f16_matmul_dual_gpu ⇒ 对同一归一化输入逐位同。
    形状扫描（bench_dual_ab）证伪旋钮路线：t64o4 565GB/s 最优，
    t128o8 打平——瓶颈是 x 摊薄 vs occupancy 死锁，唯一破法出核。
    head T128O8 实测 1031GB/s 证明盒子能跑 >1TB/s。
    """
    var o0 = Int(block_idx.x) * OUTS
    var row = Int(block_idx.y)
    if o0 >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var acc = InlineArray[Float64, 2 * OUTS](fill=Float64(0))
    var wrow = InlineArray[Pointer[Float16, MutUntrackedOrigin], 2 * OUTS](fill=w_a)
    var wbase = InlineArray[Int, 2 * OUTS](fill=0)
    var valid = InlineArray[Bool, 2 * OUTS](fill=False)
    @parameter
    for half in range(2):
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            valid[half * OUTS + oo] = o < Int(out_cols)
            comptime if half == 1:
                wrow[half * OUTS + oo] = w_b
            wbase[half * OUTS + oo] = o * Int(INNER)
    var x_base = row * Int(INNER)
    var k = tid * 8
    while k < Int(INNER):
        var xv = InlineArray[Float64, 8](fill=Float64(0))
        @parameter
        for j in range(8):
            xv[j] = Float64(x[unsafe_offset=x_base + k + j])
        @parameter
        for idx in range(2 * OUTS):
            if valid[idx]:
                var wv = wrow[idx].load[8, alignment=16](wbase[idx] + k)
                @parameter
                for j in range(8):
                    acc[idx] += xv[j] * Float64(Float32(wv[j]))
        k += THREADS * 8
    var sacc = stack_allocation[
        2 * OUTS * THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(2 * OUTS):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for oo in range(2 * OUTS):
                sacc[unsafe_offset=oo * THREADS + tid] += sacc[
                    unsafe_offset=oo * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                var v = Float32(sacc[unsafe_offset=oo * THREADS])
                var v2 = Float32(sacc[unsafe_offset=(OUTS + oo) * THREADS])
                dst_a[unsafe_offset=row * Int(out_cols) + o] = v
                dst_b[unsafe_offset=row * Int(out_cols) + o] = v2
                comptime if HAS_ACT:
                    var g = Float64(v)
                    dst_act[unsafe_offset=row * Int(out_cols) + o] = Float32(
                        g / (Float64(1) + exp(-g)) * Float64(v2)
                    )


def f16_bmm_fat2_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, ACC: Bool, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """fat2（账本九十九）：fat 的 x 外层重排——(it, rr) 外环、(oo) 内环，
    x tile 每 (it,rr) 只载一次、w 每 (it,oo) 一次。x 的 L2 流量
    out_cols×t×its×4KB ÷ OUTS（原版与 OUTS 无关，down 段 2.34GB）。
    每输出累加序不变（it 升序、THREADS 树同构）⇒ 与 fat 逐位同。
    要求 out_cols % OUTS == 0。"""

    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, OUTS * RB](0)
    @parameter
    for it in range((INNER + THREADS * 8 - 1) // (THREADS * 8)):
        var k = tid * 8 + it * THREADS * 8
        if k < Int(INNER):
            @parameter
            for rr in range(RB):
                var r = r0 + rr
                if r < Int(n_rows):
                    var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                    @parameter
                    for oo in range(OUTS):
                        var o = o0 + oo
                        if o < Int(out_cols):
                            var wv = w.load[8, alignment=16](
                                o * Int(INNER) + k
                            )
                            comptime if AF == DType.float64:
                                @parameter
                                for j in range(8):
                                    acc[oo * RB + rr] += Scalar[AF](
                                        Float64(xv[j]) * Float64(Float32(wv[j]))
                                    )
                            else:
                                var prod = xv * wv.cast[DType.float32]()
                                acc[oo * RB + rr] += Scalar[AF](prod.reduce_add())
    var sacc = stack_allocation[
        OUTS * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = Float32(
                            sacc[unsafe_offset=(oo * RB + rr) * THREADS]
                        )
                        comptime if ACC:
                            dst[unsafe_offset=r * Int(out_cols) + o] = (
                                dst[unsafe_offset=r * Int(out_cols) + o] + v
                            )
                        else:
                            dst[unsafe_offset=r * Int(out_cols) + o] = v


def f16_bmm_dual2_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_a: Pointer[Float16, MutUntrackedOrigin],
    w_b: Pointer[Float16, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    dst_act: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """dual2（账本九十九）：dual 的 x 外层重排，同 fat2。x 流量 ÷OUTS。
    每输出累加序不变 ⇒ 与 dual 逐位同。要求 out_cols % OUTS == 0。"""

    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, 2 * OUTS * RB](0)
    @parameter
    for it in range((INNER + THREADS * 8 - 1) // (THREADS * 8)):
        var k = tid * 8 + it * THREADS * 8
        if k < Int(INNER):
            @parameter
            for rr in range(RB):
                var r = r0 + rr
                if r < Int(n_rows):
                    var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                    @parameter
                    for oo in range(OUTS):
                        var o = o0 + oo
                        if o < Int(out_cols):
                            var wav = w_a.load[8, alignment=16](
                                o * Int(INNER) + k
                            )
                            var wbv = w_b.load[8, alignment=16](
                                o * Int(INNER) + k
                            )
                            comptime if AF == DType.float64:
                                @parameter
                                for j in range(8):
                                    acc[(oo * RB + rr) * 2] += Scalar[AF](
                                        Float64(xv[j]) * Float64(Float32(wav[j]))
                                    )
                                    acc[(oo * RB + rr) * 2 + 1] += Scalar[AF](
                                        Float64(xv[j]) * Float64(Float32(wbv[j]))
                                    )
                            else:
                                var wa32 = wav.cast[DType.float32]()
                                var wb32 = wbv.cast[DType.float32]()
                                acc[(oo * RB + rr) * 2] += Scalar[AF](
                                    (xv * wa32).reduce_add()
                                )
                                acc[(oo * RB + rr) * 2 + 1] += Scalar[AF](
                                    (xv * wb32).reduce_add()
                                )
    var sacc = stack_allocation[
        2 * OUTS * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(2 * OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(2 * OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = Float32(
                            sacc[unsafe_offset=(oo * RB + rr) * 2 * THREADS]
                        )
                        var v2 = Float32(
                            sacc[unsafe_offset=((oo * RB + rr) * 2 + 1) * THREADS]
                        )
                        dst_a[unsafe_offset=r * Int(out_cols) + o] = v
                        dst_b[unsafe_offset=r * Int(out_cols) + o] = v2
                        var g = Float64(v)
                        dst_act[unsafe_offset=r * Int(out_cols) + o] = Float32(
                            g / (Float64(1) + exp(-g)) * Float64(v2)
                        )


def f16_bmm_fat3_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, ACC: Bool, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """fat3（账本九十九）：fat2 的 w 寄存器修正——(it) 外环：w tile
    先载入寄存器组（@parameter 展开），再 rr×oo FMA。x 流量 ÷OUTS、
    w 流量与 fat 持平。要求 out_cols % OUTS == 0。累加序同 ⇒ 逐位同。"""

    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, OUTS * RB](0)
    @parameter
    for it in range((INNER + THREADS * 8 - 1) // (THREADS * 8)):
        var k = tid * 8 + it * THREADS * 8
        if k < Int(INNER):
            var warr = InlineArray[SIMD[DType.float16, 8], OUTS](
                fill=w.load[8, alignment=16](o0 * Int(INNER) + k)
            )
            @parameter
            for oo in range(1, OUTS):
                warr[oo] = w.load[8, alignment=16](
                    (o0 + oo) * Int(INNER) + k
                )
            @parameter
            for rr in range(RB):
                var r = r0 + rr
                if r < Int(n_rows):
                    var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                    @parameter
                    for oo in range(OUTS):
                        comptime if AF == DType.float64:
                            @parameter
                            for j in range(8):
                                acc[oo * RB + rr] += Scalar[AF](
                                    Float64(xv[j])
                                    * Float64(Float32(warr[oo][j]))
                                )
                        else:
                            var prod = xv * warr[oo].cast[DType.float32]()
                            acc[oo * RB + rr] += Scalar[AF](prod.reduce_add())
    var sacc = stack_allocation[
        OUTS * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = Float32(
                            sacc[unsafe_offset=(oo * RB + rr) * THREADS]
                        )
                        comptime if ACC:
                            dst[unsafe_offset=r * Int(out_cols) + o] = (
                                dst[unsafe_offset=r * Int(out_cols) + o] + v
                            )
                        else:
                            dst[unsafe_offset=r * Int(out_cols) + o] = v


def f16_bmm_dual3_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_a: Pointer[Float16, MutUntrackedOrigin],
    w_b: Pointer[Float16, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    dst_act: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """dual3（账本九十九）：dual2 的 w 寄存器修正——w_a/w_b tile 各进
    寄存器组。x 流量 ÷OUTS、w 流量与 dual 持平。要求 out_cols % OUTS == 0。
    累加序同 ⇒ 逐位同。"""

    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, 2 * OUTS * RB](0)
    @parameter
    for it in range((INNER + THREADS * 8 - 1) // (THREADS * 8)):
        var k = tid * 8 + it * THREADS * 8
        if k < Int(INNER):
            var waa = InlineArray[SIMD[DType.float16, 8], OUTS](
                fill=w_a.load[8, alignment=16](o0 * Int(INNER) + k)
            )
            var wbb = InlineArray[SIMD[DType.float16, 8], OUTS](
                fill=w_b.load[8, alignment=16](o0 * Int(INNER) + k)
            )
            @parameter
            for oo in range(1, OUTS):
                waa[oo] = w_a.load[8, alignment=16](
                    (o0 + oo) * Int(INNER) + k
                )
                wbb[oo] = w_b.load[8, alignment=16](
                    (o0 + oo) * Int(INNER) + k
                )
            @parameter
            for rr in range(RB):
                var r = r0 + rr
                if r < Int(n_rows):
                    var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                    @parameter
                    for oo in range(OUTS):
                        comptime if AF == DType.float64:
                            @parameter
                            for j in range(8):
                                acc[(oo * RB + rr) * 2] += Scalar[AF](
                                    Float64(xv[j])
                                    * Float64(Float32(waa[oo][j]))
                                )
                                acc[(oo * RB + rr) * 2 + 1] += Scalar[AF](
                                    Float64(xv[j])
                                    * Float64(Float32(wbb[oo][j]))
                                )
                        else:
                            var wa32 = waa[oo].cast[DType.float32]()
                            var wb32 = wbb[oo].cast[DType.float32]()
                            acc[(oo * RB + rr) * 2] += Scalar[AF](
                                (xv * wa32).reduce_add()
                            )
                            acc[(oo * RB + rr) * 2 + 1] += Scalar[AF](
                                (xv * wb32).reduce_add()
                            )
    var sacc = stack_allocation[
        2 * OUTS * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(2 * OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(2 * OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(out_cols):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = Float32(
                            sacc[unsafe_offset=(oo * RB + rr) * 2 * THREADS]
                        )
                        var v2 = Float32(
                            sacc[unsafe_offset=((oo * RB + rr) * 2 + 1) * THREADS]
                        )
                        dst_a[unsafe_offset=r * Int(out_cols) + o] = v
                        dst_b[unsafe_offset=r * Int(out_cols) + o] = v2
                        var g = Float64(v)
                        dst_act[unsafe_offset=r * Int(out_cols) + o] = Float32(
                            g / (Float64(1) + exp(-g)) * Float64(v2)
                        )


def f16_bmm_q2_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_q: Pointer[Float16, MutUntrackedOrigin],
    bias_q: Pointer[Float32, MutUntrackedOrigin],
    dst_q: Pointer[Float32, MutUntrackedOrigin],
    q_out: Int32,
    n_rows: Int32,
) -> None:
    """q2（账本一百）：qkv 的 q 链独立成核 + 寄存器分块（同 fat3）——
    x tile 每 (it,rr) 一次、w tile 进寄存器组。累加序同 qkv（k 升序、
    THREADS 树）⇒ 逐位同；bias 尾加同款。要求 q_out % OUTS == 0。"""

    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, OUTS * RB](0)
    @parameter
    for it in range((INNER + THREADS * 8 - 1) // (THREADS * 8)):
        var k = tid * 8 + it * THREADS * 8
        if k < Int(INNER):
            var warr = InlineArray[SIMD[DType.float16, 8], OUTS](
                fill=w_q.load[8, alignment=16](o0 * Int(INNER) + k)
            )
            @parameter
            for oo in range(1, OUTS):
                warr[oo] = w_q.load[8, alignment=16](
                    (o0 + oo) * Int(INNER) + k
                )
            @parameter
            for rr in range(RB):
                var r = r0 + rr
                if r < Int(n_rows):
                    var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                    @parameter
                    for oo in range(OUTS):
                        comptime if AF == DType.float64:
                            @parameter
                            for j in range(8):
                                acc[oo * RB + rr] += Scalar[AF](
                                    Float64(xv[j])
                                    * Float64(Float32(warr[oo][j]))
                                )
                        else:
                            var prod = xv * warr[oo].cast[DType.float32]()
                            acc[oo * RB + rr] += Scalar[AF](prod.reduce_add())
    var sacc = stack_allocation[
        OUTS * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(q_out):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var v = Float32(
                            sacc[unsafe_offset=(oo * RB + rr) * THREADS]
                        )
                        v = v + bias_q[unsafe_offset=o]
                        dst_q[unsafe_offset=r * Int(q_out) + o] = v


def f16_bmm_kv2_gpu[THREADS: Int, INNER: Int, OUTS: Int, RB: Int, AF: DType = DType.float64](
    x: Pointer[Float32, MutUntrackedOrigin],
    w_k: Pointer[Float16, MutUntrackedOrigin],
    w_v: Pointer[Float16, MutUntrackedOrigin],
    bias_k: Pointer[Float32, MutUntrackedOrigin],
    bias_v: Pointer[Float32, MutUntrackedOrigin],
    dst_k: Pointer[Float32, MutUntrackedOrigin],
    dst_v: Pointer[Float32, MutUntrackedOrigin],
    kv_out: Int32,
    n_rows: Int32,
) -> None:
    """kv2（账本一百）：qkv 的 k/v 双链成核 + 寄存器分块（同 dual3）。
    累加序同 qkv ⇒ 逐位同；bias 尾加同款。要求 kv_out % OUTS == 0。"""

    var o0 = Int(block_idx.x) * OUTS
    var r0 = Int(block_idx.y) * RB
    var tid = Int(thread_idx.x)
    var acc = SIMD[AF, 2 * OUTS * RB](0)
    @parameter
    for it in range((INNER + THREADS * 8 - 1) // (THREADS * 8)):
        var k = tid * 8 + it * THREADS * 8
        if k < Int(INNER):
            var wka = InlineArray[SIMD[DType.float16, 8], OUTS](
                fill=w_k.load[8, alignment=16](o0 * Int(INNER) + k)
            )
            var wva = InlineArray[SIMD[DType.float16, 8], OUTS](
                fill=w_v.load[8, alignment=16](o0 * Int(INNER) + k)
            )
            @parameter
            for oo in range(1, OUTS):
                wka[oo] = w_k.load[8, alignment=16](
                    (o0 + oo) * Int(INNER) + k
                )
                wva[oo] = w_v.load[8, alignment=16](
                    (o0 + oo) * Int(INNER) + k
                )
            @parameter
            for rr in range(RB):
                var r = r0 + rr
                if r < Int(n_rows):
                    var xv = x.load[8, alignment=32](r * Int(INNER) + k)
                    @parameter
                    for oo in range(OUTS):
                        comptime if AF == DType.float64:
                            @parameter
                            for j in range(8):
                                acc[(oo * RB + rr) * 2] += Scalar[AF](
                                    Float64(xv[j])
                                    * Float64(Float32(wka[oo][j]))
                                )
                                acc[(oo * RB + rr) * 2 + 1] += Scalar[AF](
                                    Float64(xv[j])
                                    * Float64(Float32(wva[oo][j]))
                                )
                        else:
                            var k32 = wka[oo].cast[DType.float32]()
                            var v32 = wva[oo].cast[DType.float32]()
                            acc[(oo * RB + rr) * 2] += Scalar[AF](
                                (xv * k32).reduce_add()
                            )
                            acc[(oo * RB + rr) * 2 + 1] += Scalar[AF](
                                (xv * v32).reduce_add()
                            )
    var sacc = stack_allocation[
        2 * OUTS * RB * THREADS, AF, address_space = AddressSpace.SHARED
    ]()
    @parameter
    for idx in range(2 * OUTS * RB):
        sacc[unsafe_offset=idx * THREADS + tid] = acc[idx]
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            @parameter
            for idx in range(2 * OUTS * RB):
                sacc[unsafe_offset=idx * THREADS + tid] += sacc[
                    unsafe_offset=idx * THREADS + tid + stride
                ]
        barrier()
        stride //= 2
    if tid == 0:
        @parameter
        for oo in range(OUTS):
            var o = o0 + oo
            if o < Int(kv_out):
                @parameter
                for rr in range(RB):
                    var r = r0 + rr
                    if r < Int(n_rows):
                        var vk = Float32(
                            sacc[unsafe_offset=(oo * RB + rr) * 2 * THREADS]
                        )
                        var vv = Float32(
                            sacc[unsafe_offset=((oo * RB + rr) * 2 + 1) * THREADS]
                        )
                        vk = vk + bias_k[unsafe_offset=o]
                        vv = vv + bias_v[unsafe_offset=o]
                        dst_k[unsafe_offset=r * Int(kv_out) + o] = vk
                        dst_v[unsafe_offset=r * Int(kv_out) + o] = vv



def x_to_f16_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    xh: Pointer[Float16, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """x 预转 f16（账本一百零四）：MMA 路径的操作数前置转换。"""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        xh[i] = Float16(x[i])
def pack_w_mma_gpu[INNER: Int, OUTS: Int, WARP: Int](
    w: Pointer[Float16, MutUntrackedOrigin],
    wp: Pointer[Float16, MutUntrackedOrigin],
) -> None:
    """w 打包为 mma-fragment 连续布局（账本一百零七）。原始布局
    [out_cols, INNER] 下每个 bv fragment 是「8 行 × 16B」散段（扇区
    效率 ~25%）；打包后同一 (col_block, k_step, warp, tile) 的 32 lane
    × 4 half 是连续 256B，一次 warp 载入吃满两条 128B 事务。

    打包序（out_cols 按 COLS=WARP*OUTS*8 分块）：
      idx = ((cb_idx*(INNER//16) + ks)*COLS + (warp*OUTS + j)*32 + lane)*4 + hi
    元素 = w[(cb_idx*COLS + warp*8 + j*WARP*8 + gid)*INNER + ks*16 +
             tig*2 + (hi//2)*8 + (hi%2)]，其中 lane = gid*4 + tig。
    与 f16_mma_down_pk_gpu / f16_mma_dual_pk_gpu 的载入公式严格互逆；
    一次性成本（载入后跑一次），不进热路径。要求 INNER % 16 == 0、
    out_cols % COLS == 0。"""

    comptime COLS = WARP * OUTS * 8
    var cb_idx = Int(block_idx.x)
    var ks = Int(block_idx.y)
    var i = Int(thread_idx.x)
    while i < COLS * 16:
        var wj = i // 128
        var lane = (i // 4) % 32
        var hi = i % 4
        var wq = wj // OUTS
        var j = wj % OUTS
        var col = cb_idx * COLS + wq * 8 + j * (WARP * 8) + (lane >> 2)
        var koff = (lane % 4) * 2 + (hi // 2) * 8 + (hi % 2)
        wp[
            unsafe_offset=(cb_idx * (INNER // 16) + ks) * (COLS * 16)
            + (wj * 32 + lane) * 4 + hi
        ] = w[unsafe_offset=col * INNER + ks * 16 + koff]
        i += 128


def f16_mma_down_gpu[INNER: Int, CH: Int, OUTS: Int, WARP: Int](
    xh: Pointer[Float16, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """down MMA（账本一百零六，WARP 占用率版）：tensor core
    m16n8k16 f16×f16→f32。块 = WARP 个 warp × 每 warp OUTS 个
    16×8 输出 tile，覆盖 WARP*OUTS*32 列 × 16 行；x tile f16 按
    CH 元素分块进 smem（16×CH×2B，CH=608 → 19KB），acc 寄存器跨
    块连续（f32 重组非逐位）；余数 tile（INNER % CH）av 直读全局。
    k 双链展开（步奇偶拆 accA/accB）。生产配置 [4864, 608, 2, 8]：
    128 列/块、256 线程、网格 7×t/16 —— 相比 128 线程版（[.., 4, 4]）
    每块 warp 翻倍，280µs vs 323µs（一百零六扫描）。要求
    out_cols % (WARP*OUTS*32) == 0（hidden=896 ✓）、CH % 32 == 0、
    INNER % 16 == 0、ceildiv(n_rows,16)*16 ≤ 缓冲行数。
    ACC 残加语义同 fat3(ACC=True)。仅 AF=float32 快路径使用
    （f64 精确路径仍走 fat3，门逐位不变）。"""

    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ch in range(INNER // CH):
        var i = tid
        while i < 16 * CH:
            sxf[unsafe_offset=i] = xh[
                unsafe_offset=(r0 + i // CH) * INNER + ch * CH + i % CH
            ]
            i += WARP * 32
        barrier()
        @parameter
        for st in range(CH // 16):
            var av = SIMD[DType.float16, 8](
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 9],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 9],
            )
            @parameter
            for j in range(OUTS):
                var col = cb + warp * 8 + j * (WARP * 8)
                var bv = SIMD[DType.float16, 4](
                    w[
                        unsafe_offset=(col + gid) * INNER + ch * CH
                        + st * 16 + tig * 2
                    ],
                    w[
                        unsafe_offset=(col + gid) * INNER + ch * CH
                        + st * 16 + tig * 2 + 1
                    ],
                    w[
                        unsafe_offset=(col + gid) * INNER + ch * CH
                        + st * 16 + tig * 2 + 8
                    ],
                    w[
                        unsafe_offset=(col + gid) * INNER + ch * CH
                        + st * 16 + tig * 2 + 9
                    ],
                )
                var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                comptime if st % 2 == 0:
                    mma(dv, av, bv, accA[j])
                    accA[j] = dv
                else:
                    mma(dv, av, bv, accB[j])
                    accB[j] = dv
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var bv = SIMD[DType.float16, 4](
                w[unsafe_offset=(col + gid) * INNER + k + tig * 2],
                w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 1],
                w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 8],
                w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 9],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        if r0 + gid < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] + acc[0]
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] + acc[1]
        if r0 + gid + 8 < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] + acc[2]
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] + acc[3]


def f16_mma_down_pk_gpu[INNER: Int, CH: Int, OUTS: Int, WARP: Int](
    xh: Pointer[Float16, MutUntrackedOrigin],
    wp: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """down MMA 打包版（账本一百零七）：与 f16_mma_down_gpu 逐参数
    同构、mma 顺序完全一致（数值逐位同），唯一差别是 w 从
    pack_w_mma_gpu[INNER, OUTS, WARP] 的 fragment 连续布局读取——
    每 warp 每次载入 256B 连续段而非 8×16B 散段。"""

    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ch in range(INNER // CH):
        var i = tid
        while i < 16 * CH:
            sxf[unsafe_offset=i] = xh[
                unsafe_offset=(r0 + i // CH) * INNER + ch * CH + i % CH
            ]
            i += WARP * 32
        barrier()
        @parameter
        for st in range(CH // 16):
            var av = SIMD[DType.float16, 8](
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 9],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 9],
            )
            @parameter
            for j in range(OUTS):
                var col = cb + warp * 8 + j * (WARP * 8)
                var pk0 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + ch * (CH // 16) + st) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
                var bv = SIMD[DType.float16, 4](
                    wp[unsafe_offset=pk0],
                    wp[unsafe_offset=pk0 + 1],
                    wp[unsafe_offset=pk0 + 2],
                    wp[unsafe_offset=pk0 + 3],
                )
                var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                comptime if st % 2 == 0:
                    mma(dv, av, bv, accA[j])
                    accA[j] = dv
                else:
                    mma(dv, av, bv, accB[j])
                    accB[j] = dv
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var pk1 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + (k // 16)) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
            var bv = SIMD[DType.float16, 4](
                wp[unsafe_offset=pk1],
                wp[unsafe_offset=pk1 + 1],
                wp[unsafe_offset=pk1 + 2],
                wp[unsafe_offset=pk1 + 3],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        if r0 + gid < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] + acc[0]
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] + acc[1]
        if r0 + gid + 8 < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] + acc[2]
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] + acc[3]





def f16_mma_down_pk2_gpu[INNER: Int, CH: Int, OUTS: Int, WARP: Int](
    xh: Pointer[Float16, MutUntrackedOrigin],
    wp: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """down_pk 的 **x smem 双缓冲**版（账本一百一十二-b）：原版每
    chunk 的 19.4KB x tile 载入被尾部 barrier 与上一 chunk 的 mma
    完全串行化；双缓冲在 mma 相位顶部预取下一 chunk 到另一块 smem
    （2×16×CH×2B，CH=608 时 38.8KB < 48KB），载入与计算重叠。
    mma 顺序、accA/accB 分组、store 全部不动 ⇒ **数值逐位同**。
    要求 32 * CH <= 24576（48KB smem 上限）。"""

    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf0 = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()
    var sxf1 = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    # 预取 chunk 0 → buf0
    var i = tid
    while i < 16 * CH:
        sxf0[unsafe_offset=i] = xh[
            unsafe_offset=(r0 + i // CH) * INNER + i % CH
        ]
        i += WARP * 32
    barrier()

    @parameter
    for ch in range(INNER // CH):
        comptime if ch % 2 == 0:
            # 消费 buf0；预取 ch+1 → buf1（buf1 读已受上轮 barrier 保护）
            comptime if ch + 1 < INNER // CH:
                var ip = tid
                while ip < 16 * CH:
                    sxf1[unsafe_offset=ip] = xh[
                        unsafe_offset=(
                            r0 + ip // CH
                        ) * INNER + (ch + 1) * CH + ip % CH
                    ]
                    ip += WARP * 32
            @parameter
            for st in range(CH // 16):
                var av = SIMD[DType.float16, 8](
                    sxf0[unsafe_offset=gid * CH + st * 16 + tig * 2],
                    sxf0[unsafe_offset=gid * CH + st * 16 + tig * 2 + 1],
                    sxf0[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2],
                    sxf0[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 1],
                    sxf0[unsafe_offset=gid * CH + st * 16 + tig * 2 + 8],
                    sxf0[unsafe_offset=gid * CH + st * 16 + tig * 2 + 9],
                    sxf0[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 8],
                    sxf0[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 9],
                )
                @parameter
                for j in range(OUTS):
                    var col = cb + warp * 8 + j * (WARP * 8)
                    var pk0 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + ch * (CH // 16) + st) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
                    var bv = SIMD[DType.float16, 4](
                        wp[unsafe_offset=pk0],
                        wp[unsafe_offset=pk0 + 1],
                        wp[unsafe_offset=pk0 + 2],
                        wp[unsafe_offset=pk0 + 3],
                    )
                    var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                    comptime if st % 2 == 0:
                        mma(dv, av, bv, accA[j])
                        accA[j] = dv
                    else:
                        mma(dv, av, bv, accB[j])
                        accB[j] = dv
        else:
            comptime if ch + 1 < INNER // CH:
                var ip = tid
                while ip < 16 * CH:
                    sxf0[unsafe_offset=ip] = xh[
                        unsafe_offset=(
                            r0 + ip // CH
                        ) * INNER + (ch + 1) * CH + ip % CH
                    ]
                    ip += WARP * 32
            @parameter
            for st in range(CH // 16):
                var av = SIMD[DType.float16, 8](
                    sxf1[unsafe_offset=gid * CH + st * 16 + tig * 2],
                    sxf1[unsafe_offset=gid * CH + st * 16 + tig * 2 + 1],
                    sxf1[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2],
                    sxf1[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 1],
                    sxf1[unsafe_offset=gid * CH + st * 16 + tig * 2 + 8],
                    sxf1[unsafe_offset=gid * CH + st * 16 + tig * 2 + 9],
                    sxf1[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 8],
                    sxf1[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 9],
                )
                @parameter
                for j in range(OUTS):
                    var col = cb + warp * 8 + j * (WARP * 8)
                    var pk0 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + ch * (CH // 16) + st) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
                    var bv = SIMD[DType.float16, 4](
                        wp[unsafe_offset=pk0],
                        wp[unsafe_offset=pk0 + 1],
                        wp[unsafe_offset=pk0 + 2],
                        wp[unsafe_offset=pk0 + 3],
                    )
                    var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                    comptime if st % 2 == 0:
                        mma(dv, av, bv, accA[j])
                        accA[j] = dv
                    else:
                        mma(dv, av, bv, accB[j])
                        accB[j] = dv
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var pk1 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + (k // 16)) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
            var bv = SIMD[DType.float16, 4](
                wp[unsafe_offset=pk1],
                wp[unsafe_offset=pk1 + 1],
                wp[unsafe_offset=pk1 + 2],
                wp[unsafe_offset=pk1 + 3],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        if r0 + gid < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] + acc[0]
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] + acc[1]
        if r0 + gid + 8 < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] + acc[2]
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] + acc[3]


def f16_mma_down_spl_gpu[
    INNER: Int, CH: Int, OUTS: Int, WARP: Int, S: Int,
](
    xh: Pointer[Float16, MutUntrackedOrigin],
    wp: Pointer[Float16, MutUntrackedOrigin],
    part: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """down_pk 的 **split-K** 版（账本一百一十二-b）：t=128 时 down 只有
    112 块 ≈ 108 SM，每 SM 1 块 8 warp（占用 12.5%），藏不住冷权重的
    ~600 周期 DRAM 延迟（实测每 mma ~576 周期，双缓冲/bv 预取均只
    +3%）。沿 K 切 S 段（grid.z = split），块数 ×S；每块把 K 段累进
    自己的 partial（**裸 store，不读 dst**），随后 f32_add4_gpu 归并并
    加残差。打包权重布局不变（k 绝对步进进索引）；x smem 单缓冲。
    要求 (INNER // S) % CH == 0；partial 按 [S][n_rows][out_cols]。
    分部求和改变整体累加序 ⇒ 与 pk 差 ~1e-4 级（门容差覆盖）。"""

    var split = Int(block_idx.z)
    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    comptime KTOT = INNER // 16
    comptime KPS = KTOT // S
    comptime STEPS = CH // 16
    var ks0 = split * KPS

    @parameter
    for ks in range(KPS):
        comptime if ks % STEPS == 0:
            var i = tid
            while i < 16 * CH:
                sxf[unsafe_offset=i] = xh[
                    unsafe_offset=(r0 + i // CH) * INNER
                    + (ks0 + ks) * 16 + i % CH
                ]
                i += WARP * 32
            barrier()
        var av = SIMD[DType.float16, 8](
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 1],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 1],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 8],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 9],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 8],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var pk1 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + (ks0 + ks)) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
            var bv = SIMD[DType.float16, 4](
                wp[unsafe_offset=pk1],
                wp[unsafe_offset=pk1 + 1],
                wp[unsafe_offset=pk1 + 2],
                wp[unsafe_offset=pk1 + 3],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv
        # 段末同步：下一次 tile 重载前，全部 warp 须读完 sxf
        comptime if (ks + 1) % STEPS == 0:
            barrier()

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        var base = (split * Int(n_rows) + r0) * Int(out_cols) + col
        if r0 + gid < Int(n_rows):
            part[unsafe_offset=base + gid * Int(out_cols) + tig * 2] = acc[0]
            part[unsafe_offset=base + gid * Int(out_cols) + tig * 2 + 1] = acc[
                1
            ]
        if r0 + gid + 8 < Int(n_rows):
            part[
                unsafe_offset=base + (gid + 8) * Int(out_cols) + tig * 2
            ] = acc[2]
            part[
                unsafe_offset=base + (gid + 8) * Int(out_cols) + tig * 2 + 1
            ] = acc[3]


def f32_adds_gpu[S: Int](
    dst: Pointer[Float32, MutUntrackedOrigin],
    part: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """split-K 归并（账本一百一十二-b）：dst[i] += Σ_s part[s][i]，
    part 按 [S][n] 连续布局，S 编译期（必须与 spl 的 S 一致——
    不一致会静默丢段）。grid = ceil(n/256)，256 线程。"""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var nn = Int(n)
        var acc = Float32(0)
        for s in range(S):
            acc += part[unsafe_offset=s * nn + i]
        dst[unsafe_offset=i] += acc

def cat2_f16_gpu(
    a: Pointer[Float16, MutUntrackedOrigin],
    b: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float16, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """拼接两个 f16 数组：dst[0..n) = a，dst[n..2n) = b（账本
    一百一十三-b：k/v 权重拼成 [2*kv_dim, inner] 再进 pack）。"""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = a[unsafe_offset=i]
        dst[unsafe_offset=Int(n) + i] = b[unsafe_offset=i]


def cat2_f32_gpu(
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """f32 版拼接（k/v bias 拼成 [2*kv_dim]）。"""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = a[unsafe_offset=i]
        dst[unsafe_offset=Int(n) + i] = b[unsafe_offset=i]


def f16_mma_gemm2_spl_gpu[
    INNER: Int, CH: Int, WARP: Int, S: Int,
](
    xh: Pointer[Float16, MutUntrackedOrigin],
    wp: Pointer[Float16, MutUntrackedOrigin],
    part_a: Pointer[Float32, MutUntrackedOrigin],
    part_b: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """kv2 双输出 packed GEMM 的 **split-K** 版（账本一百一十四-a）：
    一百一十三-b 的 gemm2 在流水内实测慢 ~1ms——探针 L2 热（每层
    打包权重仅 458KB，4 套轮转 ≪ 40MB L2），流水内每层的 k/v 打包
    权重都被 ~37MB/层的权重流冲出 L2，真冷 DRAM 下 16 块发射藏不住
    延迟（down 同病）。沿 K 切 S 段（grid.z = split），块数 ×S；
    partial 裸写 [S][n_rows][out_cols]（a/b 两侧分开），随后
    f32_mergb2_gpu 归并加 bias。要求 (INNER // S) % CH == 0 且
    INNER % CH == 0；分部求和改变整体累加序 ⇒ 与标量差 ~1e-4 级。"""

    comptime OUTS = 2
    var split = Int(block_idx.z)
    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    comptime KTOT = INNER // 16
    comptime KPS = KTOT // S
    comptime STEPS = CH // 16
    var ks0 = split * KPS

    @parameter
    for ks in range(KPS):
        comptime if ks % STEPS == 0:
            var i = tid
            while i < 16 * CH:
                sxf[unsafe_offset=i] = xh[
                    unsafe_offset=(r0 + i // CH) * INNER
                    + (ks0 + ks) * 16 + i % CH
                ]
                i += WARP * 32
            barrier()
        var av = SIMD[DType.float16, 8](
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 1],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 1],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 8],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 9],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 8],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var pk = (
                (cb // (WARP * OUTS * 8)) * (INNER // 16) + (ks0 + ks)
            ) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
            var bv = SIMD[DType.float16, 4](
                wp[unsafe_offset=pk],
                wp[unsafe_offset=pk + 1],
                wp[unsafe_offset=pk + 2],
                wp[unsafe_offset=pk + 3],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv
        comptime if (ks + 1) % STEPS == 0:
            barrier()

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        var is_a = col < Int(out_cols)
        var pp = part_a if is_a else part_b
        var cbase = col - Int(out_cols) if not is_a else col
        var base = (split * Int(n_rows) + r0) * Int(out_cols) + cbase
        if r0 + gid < Int(n_rows):
            pp[unsafe_offset=base + gid * Int(out_cols) + tig * 2] = acc[0]
            pp[unsafe_offset=base + gid * Int(out_cols) + tig * 2 + 1] = acc[
                1
            ]
        if r0 + gid + 8 < Int(n_rows):
            pp[
                unsafe_offset=base + (gid + 8) * Int(out_cols) + tig * 2
            ] = acc[2]
            pp[
                unsafe_offset=base + (gid + 8) * Int(out_cols) + tig * 2 + 1
            ] = acc[3]


def f32_mergb_gpu[S: Int, HAS_BIAS: Bool](
    dst: Pointer[Float32, MutUntrackedOrigin],
    part: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n: Int32,
) -> None:
    """q2 split-K 归并（账本一百一十七-b）：dst[i] = Σ_s part[s][i]
    + bias[i % out_cols]。part 按 [S][n] 连续布局，S 编译期（必须
    与 spl 的 S 一致——不一致静默丢段）。mergb2 的单路版。
    HAS_BIAS（一百八十四）：False 时纯归并，bias 不读（占位指针）。"""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var nn = Int(n)
        var acc = Float32(0)
        for s in range(S):
            acc += part[unsafe_offset=s * nn + i]
        comptime if HAS_BIAS:
            acc += bias[unsafe_offset=i % Int(out_cols)]
        dst[unsafe_offset=i] = acc


def f32_mergsw_gpu[S: Int](
    dst_act: Pointer[Float32, MutUntrackedOrigin],
    part_a: Pointer[Float32, MutUntrackedOrigin],
    part_b: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """dual split-K 收口（账本一百一十八）：dst_act[i] =
    silu(Σ_s part_a[s][i]) * Σ_s part_b[s][i]——f64 中间量，序同
    swiglu_gpu / dual_pk 尾部。part 按 [S][n] 连续布局，S 编译期
    （必须与 spl 的 S 一致——不一致静默丢段）。gate/up 的 f32
    全量落盘被跳过（prefill 的 down 只吃 act）。"""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var nn = Int(n)
        var sa = Float32(0)
        var sb = Float32(0)
        for s in range(S):
            sa += part_a[unsafe_offset=s * nn + i]
            sb += part_b[unsafe_offset=s * nn + i]
        var g = Float64(sa)
        dst_act[unsafe_offset=i] = Float32(
            g / (Float64(1) + exp(-g)) * Float64(sb)
        )


def f32_mergb2_gpu[S: Int, HAS_BIAS: Bool](
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    part_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    part_b: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n: Int32,
) -> None:
    """kv2 split-K 归并（账本一百一十四-a）：dst_a[i] = Σ part_a +
    bias[i % out_cols]，dst_b[i] = Σ part_b + bias[out_cols + i %
    out_cols]（k/v 两侧 bias 在拼接 bias 里相邻）。part 按
    [S][n] 连续，S 编译期（必须与 spl 的 S 一致——不一致静默丢段）。
    HAS_BIAS（一百八十四）：False 时纯归并，bias 不读（占位指针）。"""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var nn = Int(n)
        var aa = Float32(0)
        var ab = Float32(0)
        for s in range(S):
            aa += part_a[unsafe_offset=s * nn + i]
            ab += part_b[unsafe_offset=s * nn + i]
        comptime if HAS_BIAS:
            var c = i % Int(out_cols)
            aa += bias[unsafe_offset=c]
            ab += bias[unsafe_offset=Int(out_cols) + c]
        dst_a[unsafe_offset=i] = aa
        dst_b[unsafe_offset=i] = ab


def f16_mma_gemm2_pk_gpu[INNER: Int, CH: Int, WARP: Int, HAS_BIAS: Bool](
    xh: Pointer[Float16, MutUntrackedOrigin],
    wp: Pointer[Float16, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """**双输出** packed GEMM（账本一百一十三-b）：把 k/v 两个
    [kv_dim, inner] 权重拼成 [2*kv_dim, inner] 打包后一发算完，
    dst = [x@Wk^T + bk | x@Wv^T + bv] 按列拆到 dst_a/dst_b。OUTS=2
    时块 0 的列全落在 [0, out_cols)（→ k），块 1 全落在
    [out_cols, 2*out_cols)（→ v），要求 out_cols % (WARP*16) == 0
    保证块内列不跨界。主循环与 f16_mma_gemm_pk_gpu 逐指令相同。
    HAS_BIAS（一百八十四，bias-optional）：False 时不读 bias 指针
    （bias-optional 家族传占位，同 q4_matmul_fused_gpu 的约定）。"""

    comptime OUTS = 2
    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ch in range(INNER // CH):
        var i = tid
        while i < 16 * CH:
            sxf[unsafe_offset=i] = xh[
                unsafe_offset=(r0 + i // CH) * INNER + ch * CH + i % CH
            ]
            i += WARP * 32
        barrier()
        @parameter
        for st in range(CH // 16):
            var av = SIMD[DType.float16, 8](
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 9],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 9],
            )
            @parameter
            for j in range(OUTS):
                var col = cb + warp * 8 + j * (WARP * 8)
                var pk0 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + ch * (CH // 16) + st) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
                var bv = SIMD[DType.float16, 4](
                    wp[unsafe_offset=pk0],
                    wp[unsafe_offset=pk0 + 1],
                    wp[unsafe_offset=pk0 + 2],
                    wp[unsafe_offset=pk0 + 3],
                )
                var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                comptime if st % 2 == 0:
                    mma(dv, av, bv, accA[j])
                    accA[j] = dv
                else:
                    mma(dv, av, bv, accB[j])
                    accB[j] = dv
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var pk1 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + (k // 16)) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
            var bv = SIMD[DType.float16, 4](
                wp[unsafe_offset=pk1],
                wp[unsafe_offset=pk1 + 1],
                wp[unsafe_offset=pk1 + 2],
                wp[unsafe_offset=pk1 + 3],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        # 双输出拆分：j 的列区间整块落在 k 侧或 v 侧
        var is_a = col < Int(out_cols)
        var dp = dst_a if is_a else dst_b
        var cbase = col - Int(out_cols) if not is_a else col
        var b0 = Float32(0)
        var b1 = Float32(0)
        comptime if HAS_BIAS:
            b0 = bias[unsafe_offset=col + tig * 2]
            b1 = bias[unsafe_offset=col + tig * 2 + 1]
        if r0 + gid < Int(n_rows):
            dp[
                unsafe_offset=(r0 + gid) * Int(out_cols) + cbase + tig * 2
            ] = acc[0] + b0
            dp[
                unsafe_offset=(r0 + gid) * Int(out_cols) + cbase + tig * 2 + 1
            ] = acc[1] + b1
        if r0 + gid + 8 < Int(n_rows):
            dp[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + cbase + tig * 2
            ] = acc[2] + b0
            dp[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + cbase + tig * 2 + 1
            ] = acc[3] + b1


def f16_mma_gemm_pk_gpu[
    INNER: Int, CH: Int, OUTS: Int, WARP: Int, HAS_BIAS: Bool
](
    xh: Pointer[Float16, MutUntrackedOrigin],
    wp: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """通用 packed GEMM（账本一百零八）：`dst = x @ wp^T (+ bias)`，
    wp 来自 pack_w_mma_gpu[INNER, OUTS, WARP]，几何/主循环与
    f16_mma_down_pk_gpu 逐指令相同，只是尾部从 ACC 累加改为裸写
    （HAS_BIAS 时叠 f32 bias，索引 = 全局列）。供 q2/o 的 896×896
    小 GEMM 使用；要求 out_cols % (WARP*OUTS*8) == 0。"""

    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ch in range(INNER // CH):
        var i = tid
        while i < 16 * CH:
            sxf[unsafe_offset=i] = xh[
                unsafe_offset=(r0 + i // CH) * INNER + ch * CH + i % CH
            ]
            i += WARP * 32
        barrier()
        @parameter
        for st in range(CH // 16):
            var av = SIMD[DType.float16, 8](
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 9],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 9],
            )
            @parameter
            for j in range(OUTS):
                var col = cb + warp * 8 + j * (WARP * 8)
                var pk0 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + ch * (CH // 16) + st) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
                var bv = SIMD[DType.float16, 4](
                    wp[unsafe_offset=pk0],
                    wp[unsafe_offset=pk0 + 1],
                    wp[unsafe_offset=pk0 + 2],
                    wp[unsafe_offset=pk0 + 3],
                )
                var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                comptime if st % 2 == 0:
                    mma(dv, av, bv, accA[j])
                    accA[j] = dv
                else:
                    mma(dv, av, bv, accB[j])
                    accB[j] = dv
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var pk1 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + (k // 16)) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
            var bv = SIMD[DType.float16, 4](
                wp[unsafe_offset=pk1],
                wp[unsafe_offset=pk1 + 1],
                wp[unsafe_offset=pk1 + 2],
                wp[unsafe_offset=pk1 + 3],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        var b0 = Float32(0)
        var b1 = Float32(0)
        comptime if HAS_BIAS:
            b0 = bias[unsafe_offset=col + tig * 2]
            b1 = bias[unsafe_offset=col + tig * 2 + 1]
        if r0 + gid < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = acc[0] + b0
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = acc[1] + b1
        if r0 + gid + 8 < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = acc[2] + b0
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = acc[3] + b1


def f16_mma_dual_pk_gpu[INNER: Int, CH: Int, WARP: Int](
    xh: Pointer[Float16, MutUntrackedOrigin],
    w_a: Pointer[Float16, MutUntrackedOrigin],
    w_b: Pointer[Float16, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    dst_act: Pointer[Float32, MutUntrackedOrigin],
    dst_act16: Pointer[Float16, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """dual MMA 打包版（账本一百零七）：与 f16_mma_dual_gpu 同构同
    mma 序，w_a/w_b 均从 pack_w_mma_gpu[INNER, 2, WARP] 布局读取。
    OUTS 固定 2（每 warp 每矩阵 2 个 16×8 tile，块 WARP*16 列）。"""

    var cb = Int(block_idx.x) * WARP * 8 * 2
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA1 = InlineArray[SIMD[DType.float32, 4], 2](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB1 = InlineArray[SIMD[DType.float32, 4], 2](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accA2 = InlineArray[SIMD[DType.float32, 4], 2](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB2 = InlineArray[SIMD[DType.float32, 4], 2](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ch in range(INNER // CH):
        var i = tid
        while i < 16 * CH:
            sxf[unsafe_offset=i] = xh[
                unsafe_offset=(r0 + i // CH) * INNER + ch * CH + i % CH
            ]
            i += WARP * 32
        barrier()
        @parameter
        for ks in range(CH // 32):
            @parameter
            for u in range(2):
                var k = ch * CH + (ks * 2 + u) * 16
                var ko = (ks * 2 + u) * 16
                var av = SIMD[DType.float16, 8](
                    sxf[unsafe_offset=gid * CH + ko + tig * 2],
                    sxf[unsafe_offset=gid * CH + ko + tig * 2 + 1],
                    sxf[unsafe_offset=(gid + 8) * CH + ko + tig * 2],
                    sxf[unsafe_offset=(gid + 8) * CH + ko + tig * 2 + 1],
                    sxf[unsafe_offset=gid * CH + ko + tig * 2 + 8],
                    sxf[unsafe_offset=gid * CH + ko + tig * 2 + 9],
                    sxf[unsafe_offset=(gid + 8) * CH + ko + tig * 2 + 8],
                    sxf[unsafe_offset=(gid + 8) * CH + ko + tig * 2 + 9],
                )
                @parameter
                for j in range(2):
                    var col = cb + warp * 8 + j * (WARP * 8)
                    var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                    var pk2 = ((cb // (WARP * 16)) * (INNER // 16) + (k // 16)) * (WARP * 16 * 16) + ((warp * 2 + j) * 32 + lane) * 4
                    var bv = SIMD[DType.float16, 4](
                        w_a[unsafe_offset=pk2],
                        w_a[unsafe_offset=pk2 + 1],
                        w_a[unsafe_offset=pk2 + 2],
                        w_a[unsafe_offset=pk2 + 3],
                    )
                    comptime if u % 2 == 0:
                        mma(dv, av, bv, accA1[j])
                        accA1[j] = dv
                    else:
                        mma(dv, av, bv, accB1[j])
                        accB1[j] = dv
                    var dv2 = SIMD[DType.float32, 4](0, 0, 0, 0)
                    var pk3 = ((cb // (WARP * 16)) * (INNER // 16) + (k // 16)) * (WARP * 16 * 16) + ((warp * 2 + j) * 32 + lane) * 4
                    var bv2 = SIMD[DType.float16, 4](
                        w_b[unsafe_offset=pk3],
                        w_b[unsafe_offset=pk3 + 1],
                        w_b[unsafe_offset=pk3 + 2],
                        w_b[unsafe_offset=pk3 + 3],
                    )
                    comptime if u % 2 == 0:
                        mma(dv2, av, bv2, accA2[j])
                        accA2[j] = dv2
                    else:
                        mma(dv2, av, bv2, accB2[j])
                        accB2[j] = dv2
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(2):
            var col = cb + warp * 8 + j * (WARP * 8)
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            var pk4 = ((cb // (WARP * 16)) * (INNER // 16) + (k // 16)) * (WARP * 16 * 16) + ((warp * 2 + j) * 32 + lane) * 4
            var bv = SIMD[DType.float16, 4](
                w_a[unsafe_offset=pk4],
                w_a[unsafe_offset=pk4 + 1],
                w_a[unsafe_offset=pk4 + 2],
                w_a[unsafe_offset=pk4 + 3],
            )
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA1[j])
                accA1[j] = dv
            else:
                mma(dv, av, bv, accB1[j])
                accB1[j] = dv
            var dv2 = SIMD[DType.float32, 4](0, 0, 0, 0)
            var pk5 = ((cb // (WARP * 16)) * (INNER // 16) + (k // 16)) * (WARP * 16 * 16) + ((warp * 2 + j) * 32 + lane) * 4
            var bv2 = SIMD[DType.float16, 4](
                w_b[unsafe_offset=pk5],
                w_b[unsafe_offset=pk5 + 1],
                w_b[unsafe_offset=pk5 + 2],
                w_b[unsafe_offset=pk5 + 3],
            )
            comptime if ks % 2 == 0:
                mma(dv2, av, bv2, accA2[j])
                accA2[j] = dv2
            else:
                mma(dv2, av, bv2, accB2[j])
                accB2[j] = dv2

    @parameter
    for j in range(2):
        var col = cb + warp * 8 + j * (WARP * 8)
        var v1 = accA1[j] + accB1[j]
        var v2 = accA2[j] + accB2[j]
        if r0 + gid < Int(n_rows):
            dst_a[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = v1[0]
            dst_b[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = v2[0]
            var g0 = Float64(v1[0])
            var a0 = Float32(g0 / (Float64(1) + exp(-g0)) * Float64(v2[0]))
            dst_act[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = a0
            dst_act16[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = Float16(a0)
            dst_a[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = v1[1]
            dst_b[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = v2[1]
            var g1 = Float64(v1[1])
            var a1 = Float32(g1 / (Float64(1) + exp(-g1)) * Float64(v2[1]))
            dst_act[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = a1
            dst_act16[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = Float16(a1)
        if r0 + gid + 8 < Int(n_rows):
            dst_a[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = v1[2]
            dst_b[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = v2[2]
            var g2 = Float64(v1[2])
            var a2 = Float32(g2 / (Float64(1) + exp(-g2)) * Float64(v2[2]))
            dst_act[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = a2
            dst_act16[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = Float16(a2)
            dst_a[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = v1[3]
            dst_b[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = v2[3]
            var g3 = Float64(v1[3])
            var a3 = Float32(g3 / (Float64(1) + exp(-g3)) * Float64(v2[3]))
            dst_act[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = a3
            dst_act16[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = Float16(a3)


def f16_mma_dual_gpu[INNER: Int, CH: Int, WARP: Int](
    xh: Pointer[Float16, MutUntrackedOrigin],
    w_a: Pointer[Float16, MutUntrackedOrigin],
    w_b: Pointer[Float16, MutUntrackedOrigin],
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    dst_act: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """dual MMA（账本一百零四/一百零六）：down MMA 同构分块（CH=608），
    每 warp 每矩阵 2 个 16×8 tile，块覆盖 WARP*16 列，w_a/w_b 各自
    fragment、av 两矩阵共用。尾融合 dst_act = silu(gate)·up（公式同
    dual3）。要求 out_cols % (WARP*16) == 0（inter=4864 ✓）。
    f32 快路径生产版见 f16_mma_dual_pk_gpu（一百零七 w 打包）。"""

    var cb = Int(block_idx.x) * WARP * 8 * 2
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA1 = InlineArray[SIMD[DType.float32, 4], 2](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB1 = InlineArray[SIMD[DType.float32, 4], 2](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accA2 = InlineArray[SIMD[DType.float32, 4], 2](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB2 = InlineArray[SIMD[DType.float32, 4], 2](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ch in range(INNER // CH):
        var i = tid
        while i < 16 * CH:
            sxf[unsafe_offset=i] = xh[
                unsafe_offset=(r0 + i // CH) * INNER + ch * CH + i % CH
            ]
            i += WARP * 32
        barrier()
        @parameter
        for ks in range(CH // 32):
            @parameter
            for u in range(2):
                var k = ch * CH + (ks * 2 + u) * 16
                var ko = (ks * 2 + u) * 16
                var av = SIMD[DType.float16, 8](
                    sxf[unsafe_offset=gid * CH + ko + tig * 2],
                    sxf[unsafe_offset=gid * CH + ko + tig * 2 + 1],
                    sxf[unsafe_offset=(gid + 8) * CH + ko + tig * 2],
                    sxf[unsafe_offset=(gid + 8) * CH + ko + tig * 2 + 1],
                    sxf[unsafe_offset=gid * CH + ko + tig * 2 + 8],
                    sxf[unsafe_offset=gid * CH + ko + tig * 2 + 9],
                    sxf[unsafe_offset=(gid + 8) * CH + ko + tig * 2 + 8],
                    sxf[unsafe_offset=(gid + 8) * CH + ko + tig * 2 + 9],
                )
                @parameter
                for j in range(2):
                    var col = cb + warp * 8 + j * (WARP * 8)
                    var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                    var bv = SIMD[DType.float16, 4](
                        w_a[
                            unsafe_offset=(col + gid) * INNER + k + tig * 2
                        ],
                        w_a[
                            unsafe_offset=(col + gid) * INNER + k + tig * 2 + 1
                        ],
                        w_a[
                            unsafe_offset=(col + gid) * INNER + k + tig * 2 + 8
                        ],
                        w_a[
                            unsafe_offset=(col + gid) * INNER + k + tig * 2 + 9
                        ],
                    )
                    comptime if u % 2 == 0:
                        mma(dv, av, bv, accA1[j])
                        accA1[j] = dv
                    else:
                        mma(dv, av, bv, accB1[j])
                        accB1[j] = dv
                    var dv2 = SIMD[DType.float32, 4](0, 0, 0, 0)
                    var bv2 = SIMD[DType.float16, 4](
                        w_b[
                            unsafe_offset=(col + gid) * INNER + k + tig * 2
                        ],
                        w_b[
                            unsafe_offset=(col + gid) * INNER + k + tig * 2 + 1
                        ],
                        w_b[
                            unsafe_offset=(col + gid) * INNER + k + tig * 2 + 8
                        ],
                        w_b[
                            unsafe_offset=(col + gid) * INNER + k + tig * 2 + 9
                        ],
                    )
                    comptime if u % 2 == 0:
                        mma(dv2, av, bv2, accA2[j])
                        accA2[j] = dv2
                    else:
                        mma(dv2, av, bv2, accB2[j])
                        accB2[j] = dv2
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(2):
            var col = cb + warp * 8 + j * (WARP * 8)
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            var bv = SIMD[DType.float16, 4](
                w_a[unsafe_offset=(col + gid) * INNER + k + tig * 2],
                w_a[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 1],
                w_a[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 8],
                w_a[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 9],
            )
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA1[j])
                accA1[j] = dv
            else:
                mma(dv, av, bv, accB1[j])
                accB1[j] = dv
            var dv2 = SIMD[DType.float32, 4](0, 0, 0, 0)
            var bv2 = SIMD[DType.float16, 4](
                w_b[unsafe_offset=(col + gid) * INNER + k + tig * 2],
                w_b[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 1],
                w_b[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 8],
                w_b[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 9],
            )
            comptime if ks % 2 == 0:
                mma(dv2, av, bv2, accA2[j])
                accA2[j] = dv2
            else:
                mma(dv2, av, bv2, accB2[j])
                accB2[j] = dv2

    @parameter
    for j in range(2):
        var col = cb + warp * 8 + j * (WARP * 8)
        var v1 = accA1[j] + accB1[j]
        var v2 = accA2[j] + accB2[j]
        if r0 + gid < Int(n_rows):
            dst_a[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = v1[0]
            dst_b[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = v2[0]
            var g0 = Float64(v1[0])
            dst_act[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = Float32(
                g0 / (Float64(1) + exp(-g0)) * Float64(v2[0])
            )
            dst_a[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = v1[1]
            dst_b[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = v2[1]
            var g1 = Float64(v1[1])
            dst_act[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = Float32(
                g1 / (Float64(1) + exp(-g1)) * Float64(v2[1])
            )
        if r0 + gid + 8 < Int(n_rows):
            dst_a[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = v1[2]
            dst_b[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = v2[2]
            var g2 = Float64(v1[2])
            dst_act[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = Float32(
                g2 / (Float64(1) + exp(-g2)) * Float64(v2[2])
            )
            dst_a[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = v1[3]
            dst_b[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = v2[3]
            var g3 = Float64(v1[3])
            dst_act[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = Float32(
                g3 / (Float64(1) + exp(-g3)) * Float64(v2[3])
            )


def f16_mma_head_gpu[INNER: Int, CH: Int, OUTS: Int, WARP: Int](
    xh: Pointer[Float16, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """lm_head MMA（账本一百六十一）：`f16_mma_down_gpu` 的**直写版**
    ——down 核写死残差累加（o 直连残差语义），lm_head 的 logits 无残差
    基值，dst 直写（= acc，非 +=）。权重吃行主 [out_cols, INNER] f16
    （d_embed16 原布局，**免 pack**）；x 为 rmsnorm 直出的 f16
    （d_xnormb16）。约束同 down 核：out_cols % (WARP*OUTS*8) == 0、
    CH % 32 == 0、INNER % 16 == 0、ceildiv(n_rows,16)*16 ≤ 缓冲行数。
    LMHEAD 生产配置 [896, 608, 2, 8]：128 列/块、grid vocab/128 ×
    ceildiv(t,16)——对比 fat 核 [256, 896, 8]（53.6 万块、56% 线程
    空转、f64 累加）每块 128 输出 × tensor core。"""

    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ch in range(INNER // CH):
        var i = tid
        while i < 16 * CH:
            sxf[unsafe_offset=i] = xh[
                unsafe_offset=(r0 + i // CH) * INNER + ch * CH + i % CH
            ]
            i += WARP * 32
        barrier()
        @parameter
        for st in range(CH // 16):
            var av = SIMD[DType.float16, 8](
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 9],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 9],
            )
            @parameter
            for j in range(OUTS):
                var col = cb + warp * 8 + j * (WARP * 8)
                var bv = SIMD[DType.float16, 4](
                    w[
                        unsafe_offset=(col + gid) * INNER + ch * CH
                        + st * 16 + tig * 2
                    ],
                    w[
                        unsafe_offset=(col + gid) * INNER + ch * CH
                        + st * 16 + tig * 2 + 1
                    ],
                    w[
                        unsafe_offset=(col + gid) * INNER + ch * CH
                        + st * 16 + tig * 2 + 8
                    ],
                    w[
                        unsafe_offset=(col + gid) * INNER + ch * CH
                        + st * 16 + tig * 2 + 9
                    ],
                )
                var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                comptime if st % 2 == 0:
                    mma(dv, av, bv, accA[j])
                    accA[j] = dv
                else:
                    mma(dv, av, bv, accB[j])
                    accB[j] = dv
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var bv = SIMD[DType.float16, 4](
                w[unsafe_offset=(col + gid) * INNER + k + tig * 2],
                w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 1],
                w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 8],
                w[unsafe_offset=(col + gid) * INNER + k + tig * 2 + 9],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        if r0 + gid < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = acc[0]
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = acc[1]
        if r0 + gid + 8 < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = acc[2]
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = acc[3]


def f16_mma_head_pk_gpu[INNER: Int, CH: Int, OUTS: Int, WARP: Int](
    xh: Pointer[Float16, MutUntrackedOrigin],
    wp: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """lm_head MMA 打包版（账本一百六十三）：与 f16_mma_head_gpu 逐
    参数同构、mma 顺序完全一致（数值逐位同），唯一差别是 w 从
    pack_w_mma_gpu[INNER, OUTS, WARP] 的 fragment 连续布局读取——
    head 直读版每 bv 是 8×16B 散段（扇区效率 ~25%，一百六十二孤立
    口径 1317µs vs 理论 78µs），打包后每 warp 载入 256B 连续段。
    尾部写为**直写**（= acc，logits 无残差基值）——与 down_pk 的
    唯一差别。d_embedp 由 pack_w_mma_gpu 载入后一次性打包（静态
    权重，d_embed16 仍保留供 gather 查表）。"""

    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    @parameter
    for ch in range(INNER // CH):
        var i = tid
        while i < 16 * CH:
            sxf[unsafe_offset=i] = xh[
                unsafe_offset=(r0 + i // CH) * INNER + ch * CH + i % CH
            ]
            i += WARP * 32
        barrier()
        @parameter
        for st in range(CH // 16):
            var av = SIMD[DType.float16, 8](
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 1],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=gid * CH + st * 16 + tig * 2 + 9],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 8],
                sxf[unsafe_offset=(gid + 8) * CH + st * 16 + tig * 2 + 9],
            )
            @parameter
            for j in range(OUTS):
                var col = cb + warp * 8 + j * (WARP * 8)
                var pk0 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + ch * (CH // 16) + st) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
                var bv = SIMD[DType.float16, 4](
                    wp[unsafe_offset=pk0],
                    wp[unsafe_offset=pk0 + 1],
                    wp[unsafe_offset=pk0 + 2],
                    wp[unsafe_offset=pk0 + 3],
                )
                var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
                comptime if st % 2 == 0:
                    mma(dv, av, bv, accA[j])
                    accA[j] = dv
                else:
                    mma(dv, av, bv, accB[j])
                    accB[j] = dv
        barrier()

    @parameter
    for ks in range((INNER % CH) // 16):
        var k = (INNER // CH) * CH + ks * 16
        var av = SIMD[DType.float16, 8](
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 1],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid) * INNER + k + tig * 2 + 9],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 8],
            xh[unsafe_offset=(r0 + gid + 8) * INNER + k + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var pk1 = ((cb // (WARP * OUTS * 8)) * (INNER // 16) + (k // 16)) * (WARP * OUTS * 8 * 16) + ((warp * OUTS + j) * 32 + lane) * 4
            var bv = SIMD[DType.float16, 4](
                wp[unsafe_offset=pk1],
                wp[unsafe_offset=pk1 + 1],
                wp[unsafe_offset=pk1 + 2],
                wp[unsafe_offset=pk1 + 3],
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        if r0 + gid < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2
            ] = acc[0]
            dst[
                unsafe_offset=(r0 + gid) * Int(out_cols) + col + tig * 2 + 1
            ] = acc[1]
        if r0 + gid + 8 < Int(n_rows):
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2
            ] = acc[2]
            dst[
                unsafe_offset=(r0 + gid + 8) * Int(out_cols) + col + tig * 2 + 1
            ] = acc[3]


def f32_adds_v4_gpu[S: Int](
    dst: Pointer[Float32, MutUntrackedOrigin],
    part: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """adds 的 SIMD4 版（账本一百六十五）：每线程 4 连续元素，
    load/store[4] 借宽——原版 1 元素/线程在 25088 元素上要 98 块
    ×256 线程恰好 1:1，DRAM 延迟主导；v4 版 grid 缩到 1/4、每线程
    两条 16B 事务。要求 n % 4 == 0（out_cols 是 128 倍数 ✓）。
    逐元素 IEEE 加法序与标量版逐位相同。grid = ceildiv(n//4, 256)。"""
    var i4 = (Int(block_idx.x) * 256 + Int(thread_idx.x)) * 4
    var nn = Int(n)
    if i4 < nn:
        var acc = SIMD[DType.float32, 4](0, 0, 0, 0)
        for s in range(S):
            acc += part.load[4](s * nn + i4)
        dst.store[4](i4, dst.load[4](i4) + acc)


def f32_mergb_v4_gpu[S: Int, HAS_BIAS: Bool](
    dst: Pointer[Float32, MutUntrackedOrigin],
    part: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n: Int32,
) -> None:
    """mergb 的 SIMD4 版（账本一百六十五）：结构同 adds_v4——
    Σ_s SIMD 累加 + bias SIMD 载入（4 连续 i 的 i%out_cols 也连续，
    要求 out_cols % 4 == 0 ✓）。grid = ceildiv(n//4, 256)。
    HAS_BIAS（一百八十四）：False 时纯归并，bias 不读（占位指针）。"""
    var i4 = (Int(block_idx.x) * 256 + Int(thread_idx.x)) * 4
    var nn = Int(n)
    if i4 < nn:
        var acc = SIMD[DType.float32, 4](0, 0, 0, 0)
        for s in range(S):
            acc += part.load[4](s * nn + i4)
        comptime if HAS_BIAS:
            var c = i4 % Int(out_cols)
            acc += bias.load[4](c)
        dst.store[4](i4, acc)


def f32_mergb2_v4_gpu[S: Int, HAS_BIAS: Bool](
    dst_a: Pointer[Float32, MutUntrackedOrigin],
    part_a: Pointer[Float32, MutUntrackedOrigin],
    dst_b: Pointer[Float32, MutUntrackedOrigin],
    part_b: Pointer[Float32, MutUntrackedOrigin],
    bias: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n: Int32,
) -> None:
    """mergb2 的 SIMD4 版（账本一百六十五）：k/v 双路并行 SIMD
    累加——两路 load[4] 交错发出，DRAM 延迟重叠。bias 双段各一载。
    grid = ceildiv(n//4, 256)。
    HAS_BIAS（一百八十四）：False 时纯归并，bias 不读（占位指针）。"""
    var i4 = (Int(block_idx.x) * 256 + Int(thread_idx.x)) * 4
    var nn = Int(n)
    if i4 < nn:
        var aa = SIMD[DType.float32, 4](0, 0, 0, 0)
        var ab = SIMD[DType.float32, 4](0, 0, 0, 0)
        for s in range(S):
            aa += part_a.load[4](s * nn + i4)
            ab += part_b.load[4](s * nn + i4)
        comptime if HAS_BIAS:
            var c = i4 % Int(out_cols)
            aa += bias.load[4](c)
            ab += bias.load[4](Int(out_cols) + c)
        dst_a.store[4](i4, aa)
        dst_b.store[4](i4, ab)

def q4_pack_scales_gpu(
    blocks: Pointer[UInt8, MutUntrackedOrigin],
    wps: Pointer[UInt8, MutUntrackedOrigin],
    nblk: Int32,
) -> None:
    """q4_0 块头 scale 抽取（账本一百七十二）：一块一线程，把块头
    2B fp16 位模式原样拷进 wps（2B/块）。装载期一次性。nblk 运行时
    （块数随权重形状变化，不进 comptime）。"""
    var b = Int(global_idx.x)
    if b < Int(nblk):
        wps[unsafe_offset = b * 2] = blocks[unsafe_offset = b * 18]
        wps[unsafe_offset = b * 2 + 1] = blocks[unsafe_offset = b * 18 + 1]


def pack_w_mma_q4_gpu[INNER: Int, OUTS: Int, WARP: Int](
    blocks: Pointer[UInt8, MutUntrackedOrigin],
    wpq: Pointer[UInt8, MutUntrackedOrigin],
) -> None:
    """q4_0 块流 -> nibble fragment 布局（账本一百七十二）。与
    pack_w_mma_gpu 同一槽位公式，但每槽存 4bit nib（0-15 原值，
    反量化在 GEMM 侧做），两槽拼一字节——每线程每迭代处理一个字节
    （槽对 2j/2j+1），单写者无半字节竞态。scale 由
    q4_pack_scales_gpu 单独抽取。装载期一次性。
    要求 INNER % 32 == 0、out_cols % COLS == 0（同 pack_w_mma）。"""
    comptime COLS = WARP * OUTS * 8
    var cb_idx = Int(block_idx.x)
    var ks = Int(block_idx.y)
    var j = Int(thread_idx.x)
    var nbytes = COLS * 16 // 2
    # 块基址（每 (cb_idx, ks) 段连续——与 pack_w_mma 的槽序同构）
    var base = (cb_idx * (INNER // 16) + ks) * nbytes
    while j < nbytes:
        var n0 = _q4_slot_nib(
            blocks, 2 * j, cb_idx, ks, INNER, OUTS, WARP
        )
        var n1 = _q4_slot_nib(
            blocks, 2 * j + 1, cb_idx, ks, INNER, OUTS, WARP
        )
        wpq[unsafe_offset = base + j] = UInt8(n0 | (n1 << 4))
        j += 128


@always_inline
def _q4_slot_nib(
    blocks: Pointer[UInt8, MutUntrackedOrigin],
    i: Int,
    cb_idx: Int,
    ks: Int,
    INNER: Int,
    OUTS: Int,
    WARP: Int,
) -> Int:
    """pack_w_mma_gpu 槽位公式的 q4 nib 版（一百七十二）：槽 i ->
    (col, koff) -> 全局权重 idx -> 块内 nib（0-15）。"""
    var COLS = WARP * OUTS * 8
    var wj = i // 128
    var lane = (i // 4) % 32
    var hi = i % 4
    var wq = wj // OUTS
    var jj = wj % OUTS
    var col = cb_idx * COLS + wq * 8 + jj * (WARP * 8) + (lane >> 2)
    var koff = (lane % 4) * 2 + (hi // 2) * 8 + hi % 2
    var gi = col * INNER + ks * 16 + koff
    var b = gi // 32
    var idx = gi % 32
    var byte = blocks[unsafe_offset = b * 18 + 2 + idx % 16]
    if idx < 16:
        return Int(byte) & 0x0F
    return (Int(byte) >> 4) & 0x0F


def f16_mma_down_spl_q4_gpu[
    INNER: Int, CH: Int, OUTS: Int, WARP: Int, S: Int,
](
    xh: Pointer[Float16, MutUntrackedOrigin],
    wpq: Pointer[UInt8, MutUntrackedOrigin],
    wps: Pointer[UInt8, MutUntrackedOrigin],
    part: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
    n_rows: Int32,
) -> None:
    """down_spl 的 **q4 直读** 版（账本一百七十二）：与
    f16_mma_down_spl_gpu 逐行同构（同一 ks 循环/x 重载条件/barrier
    位置/mma 序），唯一差别是 bv 从 nibble fragment 布局读 2B + 块
    scale，核内反量化 `(nib-8)*d`（f32 精确乘后舍入 f16——与装载期
    q4_dequant_f16_gpu 逐位同）——**数值逐位同于 f16 链**。bv 载入
    流量 8B -> 2B+2B(scale)（3x 减）。要求 INNER % 32 == 0、
    (INNER // S) % CH == 0（同 down_spl）；wpq/wps 由
    pack_w_mma_q4_gpu / q4_pack_scales_gpu 装载期产出。"""

    var split = Int(block_idx.z)
    var cb = Int(block_idx.x) * WARP * OUTS * 8
    var r0 = Int(block_idx.y) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var gid = lane >> 2
    var tig = lane % 4

    var sxf = stack_allocation[
        16 * CH, Float16, address_space = AddressSpace.SHARED
    ]()

    var accA = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )
    var accB = InlineArray[SIMD[DType.float32, 4], OUTS](
        fill=SIMD[DType.float32, 4](0, 0, 0, 0)
    )

    comptime KTOT = INNER // 16
    comptime KPS = KTOT // S
    comptime STEPS = CH // 16
    comptime COLS = WARP * OUTS * 8
    var ks0 = split * KPS

    @parameter
    for ks in range(KPS):
        comptime if ks % STEPS == 0:
            var i = tid
            while i < 16 * CH:
                sxf[unsafe_offset=i] = xh[
                    unsafe_offset=(r0 + i // CH) * INNER
                    + (ks0 + ks) * 16 + i % CH
                ]
                i += WARP * 32
            barrier()
        var av = SIMD[DType.float16, 8](
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 1],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 1],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 8],
            sxf[unsafe_offset=gid * CH + (ks % STEPS) * 16 + tig * 2 + 9],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 8],
            sxf[unsafe_offset=(gid + 8) * CH + (ks % STEPS) * 16 + tig * 2 + 9],
        )
        @parameter
        for j in range(OUTS):
            var col = cb + warp * 8 + j * (WARP * 8)
            var pkq = (
                (cb // COLS) * (INNER // 16) + (ks0 + ks)
            ) * (COLS * 16) + ((warp * OUTS + j) * 32 + lane) * 4
            var q0 = wpq[unsafe_offset=pkq // 2]
            var q1 = wpq[unsafe_offset=pkq // 2 + 1]
            var sb = (col + gid) * (INNER // 32) + (ks0 + ks) // 2
            var dl = Int(wps[unsafe_offset=sb * 2])
            var dh = Int(wps[unsafe_offset=sb * 2 + 1])
            var d = _fp16_bits_to_f32_device(dl | (dh << 8))
            var bv = SIMD[DType.float16, 4](
                Float16(Float32(Int(q0 & 0x0F) - 8) * d),
                Float16(Float32(Int(q0 >> 4) - 8) * d),
                Float16(Float32(Int(q1 & 0x0F) - 8) * d),
                Float16(Float32(Int(q1 >> 4) - 8) * d),
            )
            var dv = SIMD[DType.float32, 4](0, 0, 0, 0)
            comptime if ks % 2 == 0:
                mma(dv, av, bv, accA[j])
                accA[j] = dv
            else:
                mma(dv, av, bv, accB[j])
                accB[j] = dv
        # 段末同步：下一次 tile 重载前，全部 warp 须读完 sxf
        comptime if (ks + 1) % STEPS == 0:
            barrier()

    @parameter
    for j in range(OUTS):
        var col = cb + warp * 8 + j * (WARP * 8)
        var acc = accA[j] + accB[j]
        var base = (split * Int(n_rows) + r0) * Int(out_cols) + col
        if r0 + gid < Int(n_rows):
            part[unsafe_offset=base + gid * Int(out_cols) + tig * 2] = acc[0]
            part[unsafe_offset=base + gid * Int(out_cols) + tig * 2 + 1] = acc[
                1
            ]
        if r0 + gid + 8 < Int(n_rows):
            part[
                unsafe_offset=base + (gid + 8) * Int(out_cols) + tig * 2
            ] = acc[2]
            part[
                unsafe_offset=base + (gid + 8) * Int(out_cols) + tig * 2 + 1
            ] = acc[3]