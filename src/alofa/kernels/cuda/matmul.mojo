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
from std.memory import stack_allocation

from max.gpu import AddressSpace, barrier

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
