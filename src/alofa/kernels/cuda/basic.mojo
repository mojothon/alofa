"""GPU 基础 kernel —— 逐行算子，一行一线程的最简形态。

**为什么一行一线程（而不是块内规约）**：当前唯一的判据是差分门
（`tests/gpu/test_cuda_diff.mojo`）——「GPU 算的数与标量后端算的数是同一个」。
块内规约会引入共享内存与浮点累加顺序两个正确性变量，数值对不上时分不清是布局
读错还是规约写错。性能内核是之后的事：那时换的是实现，不是门的判据。

算法与累加顺序必须与 `kernels/cpu/scalar.mojo` 的同名算子**逐句对应**：
Float64 累加、先 rsqrt 再乘。任何自己发明的顺序都会把真正的差异藏进舍入里。
"""

from std.gpu import global_idx
from std.math import sqrt
from std.memory import bitcast

from alofa.kernels.cpu.quant import Q4_BLOCK, Q4_BYTES, Q4_SCALE_BYTES
from alofa.kernels.cpu.quant import Q8_BLOCK, Q8_BYTES, Q8_SCALE_BYTES


def rmsnorm_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    cols: Int32,
    rows: Int32,
    eps: Float32,
) -> None:
    """`dst = x / sqrt(mean(x²) + eps) * w`，一行一个线程。

    与 `kernels/cpu/scalar.mojo` 的 `rmsnorm` 同一个算法与同一个顺序：求和用
    `Float64`、每行算一次 `1/sqrt` 再乘 —— 差分门要的正是「两者应当逐位接近」。
    """
    var row = Int(global_idx.x)
    if row >= Int(rows):
        return
    var base = row * Int(cols)
    var sum_sq = Float64(0)
    for i in range(Int(cols)):
        var v = Float64(x[unsafe_offset=base + i])
        sum_sq += v * v
    var scale = Float64(1.0) / sqrt(sum_sq / Float64(Int(cols)) + Float64(eps))
    for i in range(Int(cols)):
        dst[unsafe_offset=base + i] = Float32(
            Float64(x[unsafe_offset=base + i]) * scale * Float64(w[unsafe_offset=i])
        )


def _fp16_bits_to_f32_device(half: Int) -> Float32:
    """设备端 fp16 位模式 → fp32，与 `kernels/cpu/quant.mojo` 的
    `fp16_bits_to_f32` **同一个算法与同一个顺序**（指数/尾数用整数移位与 2 的
    整数次幂拼出来，全程精确、无舍入）。

    与主机版的唯一差别：设备 kernel 不能 raise。指数全 1（inf/NaN）在权重块里
    不可能出现 —— `quantize_q4_0` 在主机侧就会拒绝溢出的缩放因子 —— 这里用位
    拼接给出与 IEEE 语义一致的 inf/NaN，不让坏数据被静默吞掉。
    """
    var exponent = (half >> 10) & 0x1F
    var mantissa = half & 0x3FF
    var negative = ((half >> 15) & 1) != 0

    var magnitude = Float32(0)
    if exponent == 0:
        if mantissa != 0:
            # 次正规：mantissa × 2^-24。
            magnitude = Float32(mantissa) / Float32(1 << 24)
    elif exponent == 31:
        # inf/NaN：符号位 | 指数全 1 | 尾数左移 13 位（对齐到 fp32 尾数高位）。
        var bits = (((half >> 15) & 1) << 31) | 0x7F800000 | (mantissa << 13)
        magnitude = bitcast[DType.float32, 1](UInt32(bits))
    else:
        # 正规数：(1024 + mantissa) × 2^(exponent-25)。
        var significand = Float32(mantissa + 1024)
        var shift = exponent - 25
        if shift >= 0:
            magnitude = significand * Float32(1 << shift)
        else:
            magnitude = significand / Float32(1 << (-shift))

    if negative:
        return -magnitude
    return magnitude


def dequant_q4_0_gpu(
    blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n_blocks: Int32,
) -> None:
    """q4_0 反量化，一个块（32 个值）一个线程。

    与 `kernels/cpu/quant.mojo` 的 `dequant_q4_0` 逐句对应：缩放因子在块头
    `Q4_SCALE_BYTES` 字节（fp16 位模式，低字节在前），随后 16 字节、每字节装
    两个值（低 4 位在前）。结果只由 nibble 与缩放因子决定，两者都精确 —— 差分
    判据因此可以是**零容差**（见 `tests/gpu/test_cuda_diff.mojo`）。
    """
    var b = Int(global_idx.x)
    if b >= Int(n_blocks):
        return
    var base = b * Q4_BYTES
    var low = Int(blocks[unsafe_offset=base])
    var high = Int(blocks[unsafe_offset=base + 1])
    var d = _fp16_bits_to_f32_device(low | (high << 8))
    for j in range(16):
        var byte = Int(blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
        dst[unsafe_offset=b * Q4_BLOCK + j] = Float32((byte & 0x0F) - 8) * d
        dst[unsafe_offset=b * Q4_BLOCK + j + 16] = Float32(
            ((byte >> 4) & 0x0F) - 8
        ) * d


def dequant_q8_0_gpu(
    blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    n_blocks: Int32,
) -> None:
    """q8_0 反量化，一个块（32 个值）一个线程。

    与 `dequant_q4_0_gpu` 同一条差分理由：8 位码与 fp16 缩放都是精确的，判据
    可以是零容差。与 q4_0 的一个差别不能漏：q8_0 存的是**有符号**码（补码），
    `>= 128` 要减 256 做符号扩展 —— 漏了这步所有负权重整体跳到 +128 那侧，
    零容差门会立刻红（`kernels/cpu/quant.mojo` 的 `_int8_from_byte` 原话）。
    """
    var b = Int(global_idx.x)
    if b >= Int(n_blocks):
        return
    var base = b * Q8_BYTES
    var low = Int(blocks[unsafe_offset=base])
    var high = Int(blocks[unsafe_offset=base + 1])
    var d = _fp16_bits_to_f32_device(low | (high << 8))
    for j in range(Int(Q8_BLOCK)):
        var q = Int(blocks[unsafe_offset=base + Q8_SCALE_BYTES + j])
        if q >= 128:
            q = q - 256
        dst[unsafe_offset=b * Q8_BLOCK + j] = Float32(q) * d


def linear_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """`dst[row, o] = Σ_k x[row, k] * w[o, k]`，一个 (行, 输出列) 一个线程。

    与 `kernels/cpu/scalar.mojo` 的 `_gemm` 逐句对应：`w` 按 `[out, in]` 存
    （一行 `w` 连续），累加 `Float64`、`k` 升序、最后一次性 `Float32(acc)` ——
    差分门要的正是同一个顺序，任何重排都会把真差异藏进舍入里。

    grid 取 `(out_cols, rows)` 正好铺满，不空转；边界判断照写（与 rmsnorm
    同一个习惯，防以后 grid 复用时的越界）。
    """
    var o = Int(global_idx.x)
    var row = Int(global_idx.y)
    if o >= Int(out_cols):
        return
    var x_base = row * Int(inner)
    var w_base = o * Int(inner)
    var acc = Float64(0)
    for k in range(Int(inner)):
        acc += Float64(x[unsafe_offset=x_base + k]) * Float64(
            w[unsafe_offset=w_base + k]
        )
    dst[unsafe_offset=row * Int(out_cols) + o] = Float32(acc)
