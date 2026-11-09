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

from std.gpu import global_idx

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
