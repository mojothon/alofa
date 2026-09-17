"""Q4_0: 解量化，以及把解量化融进 matmul 的那个循环。

为什么这个文件的接口是裸指针而不是 `TensorView`
--------------------------------------------

`core/tensor.mojo` 的 `TensorView` 描述的是"元素类型固定、步长规则"的数组。
q4_0 的块流不是这种东西：它的元素是 32 个一组共享一个 fp16 缩放的 4 位量化值，
`core/dtype.mojo` 里也没有对应的 `DT_*`（块格式的元数据在 `QUANT_*` 那几组
函数里，不在 `DT_*` 里）。硬给它造一个 `DT_Q4_0` 会让 `TensorView` 从此必须
回答"步长是多少字节"这种对块格式没有意义的问题。所以块流就用原始字节寻址，
入口处显式校验长度与对齐。

块布局（GGML q4_0，一个块 32 个值 / 18 字节）
------------------------------------------

    [0:2]    fp16 缩放 d，小端
    [2:18]   16 字节，每字节装两个 4 位量化值：低半字节是第 j 个，
             高半字节是第 j+16 个
    还原值   v = (nibble - 8) * d

这个布局是**外部事实**（GGML 的 q4_0 就这么存），所以解量化是可以被外部检验
的；而"fp32 怎么压成 q4_0"是我方离线的格式转换，没有外部参照 —— 两者的可信
度不同，`docs/plan/capability-ledger.md` 里分开标注，不许笼统写成"与 llama.cpp
一致"。

为什么 fp16 → fp32 手写位运算而不是用 `Float16`
--------------------------------------------

两条理由，都不是洁癖：

- **位模式是这里唯一重要的事**。缩放因子一旦读错，`v` 就整体错一个倍率，而
  这类错误在"输出还算流利"的时候看不出来。手写解码把"半精度怎么变成单精度"
  摊开成指数、尾数、符号三段，读代码的人不必信任任何库的行为。
- **它是精确的**。半精度能表示的数，单精度都能精确表示，转换不该引入任何
  舍入 —— 于是解量化后的值只取决于 nibble 与 d，这就是这一层能做**零容差**
  比较的底气。

为什么融合 matmul 用 `Float64` 累加
----------------------------------

与 `scalar.mojo` 的 `_gemm` 同一个理由（它也是 `Float64`）：这一层的作用是当
判据，不是当最快的路径。`cols` 是 896，fp32 顺序累加的误差与"融合是否写对"
想要区分的误差是同一个量级，那就分不清了。

Run:
    pixi run mojo run -O0 -I src tests/unit/test_q4_parity.mojo
"""

from alofa.core.dtype import (
    QUANT_Q4_0,
    quant_block_size,
    quant_bytes_per_block,
    quant_scale_bytes,
)
from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_SHAPE_MISMATCH,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.tensor import F32Ptr

comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]

comptime Q4_BLOCK = quant_block_size[QUANT_Q4_0]()
comptime Q4_BYTES = quant_bytes_per_block[QUANT_Q4_0]()
comptime Q4_SCALE_BYTES = quant_scale_bytes[QUANT_Q4_0]()


def fp16_bits_to_f32(half: Int) raises AlofaError -> Float32:
    """把一个 IEEE 半精度的位模式精确转成单精度。

    半精度能表示的每个数单精度都能精确表示，所以这个转换是**精确的**，不含
    任何舍入；指数与尾数用整数移位与 2 的整数次幂（也是精确的）拼出来。
    """
    var exponent = (half >> 10) & 0x1F
    var mantissa = half & 0x3FF
    var negative = ((half >> 15) & 1) != 0

    var magnitude = Float32(0)
    if exponent == 0:
        if mantissa == 0:
            magnitude = Float32(0)
        else:
            # 次正规：mantissa × 2^-24（尾数 × 2^-10，再乘最小指数 2^-14）。
            magnitude = Float32(mantissa) / Float32(1 << 24)
    elif exponent == 31:
        # 无穷与 NaN 不该出现在权重块里。出现说明块流坏了（对齐错、读到别的
        # 张量去了），与其让一个 ±inf 缩放因子悄悄把整行权重变成 inf，不如
        # 在这里就报出来。
        raise AlofaError(
            ERR_UNSUPPORTED,
            "a q4_0 scale is infinity or NaN",
            "half_bits=" + String(half),
        )
    else:
        # 正规数：值 = (1 + mantissa/1024) × 2^(exponent-15)
        #           = (1024 + mantissa) × 2^(exponent-25)
        var significand = Float32(mantissa + 1024)
        var shift = exponent - 25
        if shift >= 0:
            magnitude = significand * Float32(1 << shift)
        else:
            magnitude = significand / Float32(1 << (-shift))

    if negative:
        return -magnitude
    return magnitude


def f32_to_fp16_bits(value: Float32) raises AlofaError -> Int:
    """把一个单精度数舍到最近的半精度位模式（并列取偶，IEEE 默认舍入）。

    与 `fp16_bits_to_f32` 相反方向，也就不可能再是精确的：这里**会**丢尾数，
    所以"取最近、并列取偶"必须显式写出来 —— 换成截断会让每个块的缩放因子都
    偏小一点，而偏小的缩放因子让整块权重等比缩小，**看起来完全正常**。量化
    后的网络依旧流利，只是悄悄变了个人；这类错误正是量化门最该挡住的。
    """
    var bits = Int(value.to_bits())
    var sign = (bits >> 31) & 1
    var magnitude = bits & 0x7FFFFFFF
    var exponent = (bits >> 23) & 0xFF

    if exponent == 0xFF:
        raise AlofaError(
            ERR_UNSUPPORTED,
            "infinity and NaN cannot be a quantisation scale",
            "value=" + String(value),
        )
    if magnitude == 0:
        return 0

    var encoded = 0
    if magnitude < 0x33800000:
        # 小于 2^-25：连半精度的次正规都够不着，归零。这类块整块都是 0 或
        # 极度接近 0，缩放因子取 0 与取 2^-24 的差别小于一个量化台阶。
        encoded = 0
    elif magnitude < 0x38800000:
        # 次正规区间。值 = (2^23 + 尾数) × 2^(e-23)，而次正规的单位是 2^-24，
        # 于是要右移 -(e+1) 位；移掉的那部分按"取最近、并列取偶"决定进位。
        var e = exponent - 127
        var significand = (magnitude & 0x7FFFFF) | 0x800000
        var shift = -(e + 1)
        var kept = significand >> shift
        var dropped = significand & ((1 << shift) - 1)
        var half = 1 << (shift - 1)
        if dropped > half or (dropped == half and (kept & 1) != 0):
            kept += 1
        encoded = kept
    else:
        # 正规数。把"减偏置 112"与"右移 13 位"合成一步：先加上半个目标单位
        # （0x0FFF）与并列修正，再整体移位，舍入与进位就一起发生了。
        encoded = (magnitude + 0x0FFF + ((magnitude >> 13) & 1) - 0x38000000) >> 13
        if encoded >= 0x7C00:
            raise AlofaError(
                ERR_UNSUPPORTED,
                "too large to be a quantisation scale in fp16",
                "value=" + String(value),
            )
    return (sign << 15) | encoded


def block_scale(blocks: U8Ptr, block_base: Int) raises AlofaError -> Float32:
    """第 `block_base` 字节处的那个块的 fp16 缩放因子。"""
    var low = Int(blocks[unsafe_offset=block_base])
    var high = Int(blocks[unsafe_offset=block_base + 1])
    return fp16_bits_to_f32(low | (high << 8))


def dequant_q4_0(dst: F32Ptr, blocks: U8Ptr, n: Int) raises AlofaError:
    """把 `n` 个量化值还原成 fp32。`n` 必须是 32 的倍数。

    结果只由 nibble 与缩放因子决定，两者都是精确的，所以这一层**可以**做零
    容差比较 —— `tests/unit/test_q4_parity.mojo` 就那么做。用容差反而是放水：
    它会把"半字节高低位装反了"这类错，伪装成"舍入差一点"。
    """
    if n <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT, "element count must be positive", "n=" + String(n)
        )
    if n % Q4_BLOCK != 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "element count must be a whole number of blocks",
            "n=" + String(n) + " block=" + String(Q4_BLOCK),
        )
    var n_blocks = n // Q4_BLOCK
    for b in range(n_blocks):
        var base = b * Q4_BYTES
        var d = block_scale(blocks, base)
        for j in range(16):
            var byte = Int(blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
            var low_nibble = byte & 0x0F
            var high_nibble = (byte >> 4) & 0x0F
            dst[unsafe_offset=b * Q4_BLOCK + j] = Float32(low_nibble - 8) * d
            dst[unsafe_offset=b * Q4_BLOCK + j + 16] = Float32(high_nibble - 8) * d


comptime Q4_MSE_ROUNDS = 2


def quantize_q4_0(blocks: U8Ptr, src: F32Ptr, n: Int) raises AlofaError:
    """把 `n` 个 fp32 压成 q4_0 块流（每 32 个值 18 字节）。

    算法与 `scripts/dump_q4_reference.py` 里那段**逐步对应**，任何一步改了都会
    让 `tests/unit/test_q4_parity.mojo` 的逐字节比较红。

    先按朴素 q4_0 起个头：

        amax = 块内 |x| 的最大值；d = amax / 7；q = clamp(round(x/d)+8, 0, 15)

    再按 MSE 修 d（`Q4_MSE_ROUNDS` 轮）：对**固定的 q**，令 `s = q - 8`，重建
    误差 `Σ (x - d·s)²` 对 d 求极小的解是

        d* = Σ(x·s) / Σ(s²)

    朴素写法为了让最大值够到台阶顶，把台阶钉在分布最稀疏的地方，实测 q/k/v/o
    权重的相对 L2 误差约 10%；MSE 解允许最大值被裁掉一点点，把台阶挪到分布
    真正密集的地方。**块布局与解量化规则一点没变**，所以那条零容差的解量化门
    继续适用 —— 变的是"台阶放哪儿"，不是"块怎么读"。

    三个容易写错的地方：
    - **量化用的 d 是 fp32 的 d**，存进块里的才是 fp16 的那个（先用 fp16 的 d
      去除，量化结果会整体偏大）；
    - `round` 取**最近、并列取偶**（IEEE 默认），截断会让每个值平均偏小半个台阶；
    - 两个 Σ 必须**按 j 递增顺序**用 `Float64` 累加，与参考脚本的
      `np.add.accumulate` 同序。敢这么依赖顺序是因为乘积 `x·s` 在 fp64 里是
      精确的（fp32 的 24 位尾数乘上 `|s| ≤ 8`），于是加法顺序是这里唯一的自
      由度，连 FMA 收缩也改变不了它。
    """
    if n <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT, "element count must be positive", "n=" + String(n)
        )
    if n % Q4_BLOCK != 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "element count must be a whole number of blocks",
            "n=" + String(n) + " block=" + String(Q4_BLOCK),
        )
    var n_blocks = n // Q4_BLOCK
    for b in range(n_blocks):
        var start = b * Q4_BLOCK
        var amax = Float32(0)
        for j in range(Q4_BLOCK):
            var v = src[unsafe_offset=start + j]
            var a = v if v >= Float32(0) else -v
            if a > amax:
                amax = a
        var d = amax / Float32(7)
        if d != Float32(0):
            # 最小二乘修 scale：对当前这组 nibble，Σ(x - d·s)² 的极小点就是
            # Σ(x·s)/Σ(s²)。全零块（d == 0）没有台阶可挪，跳过。
            for _ in range(Q4_MSE_ROUNDS):
                var num = Float64(0)
                var den = Float64(0)
                for j in range(Q4_BLOCK):
                    var x = src[unsafe_offset=start + j]
                    var s = Float64(_quantise_one(x, d) - 8)
                    num += Float64(x) * s
                    den += s * s
                if den > Float64(0):
                    d = Float32(num / den)
        var half = f32_to_fp16_bits(d)
        var base = b * Q4_BYTES
        blocks[unsafe_offset=base] = UInt8(half & 0xFF)
        blocks[unsafe_offset=base + 1] = UInt8((half >> 8) & 0xFF)

        for j in range(16):
            var lo = 8
            var hi = 8
            if d != Float32(0):
                lo = _quantise_one(src[unsafe_offset=start + j], d)
                hi = _quantise_one(src[unsafe_offset=start + j + 16], d)
            blocks[unsafe_offset=base + Q4_SCALE_BYTES + j] = UInt8(
                (lo & 0x0F) | ((hi & 0x0F) << 4)
            )


def _quantise_one(value: Float32, d: Float32) -> Int:
    """`clamp(round(value / d) + 8, 0, 15)`，取整取最近、并列取偶。"""
    var scaled = value / d
    var floor_value = Float32(Int(scaled))
    if floor_value > scaled:
        floor_value -= Float32(1)
    var below = Int(floor_value)
    var fraction = scaled - floor_value
    var rounded = below
    if fraction > Float32(0.5):
        rounded = below + 1
    elif fraction == Float32(0.5):
        # 恰好半个台阶：取偶数那一侧。
        rounded = below if (below % 2 == 0) else below + 1
    var q = rounded + 8
    if q < 0:
        return 0
    if q > 15:
        return 15
    return q


def _matmul_q4(
    dst: F32Ptr,
    x: F32Ptr,
    blocks: U8Ptr,
    rows: Int,
    cols: Int,
    bias: F32Ptr,
    has_bias: Bool,
) raises AlofaError:
    """`dst[r] = Σ_c w[r,c] · x[c]`，`w` 以 q4_0 块流给出，行主序 `[rows, cols]`。

    解量化发生在内层的寄存器里：读到 nibble 就直接乘上缩放与激活累加，**不物化
    解量化后的权重**。这不是为了快（虽然它确实省一次全量写入），而是因为"融合"
    本身就是这个 kernel 要被检验的那件事 —— 若先解量化再走通用 matmul，被测的
    就只是通用 matmul 了。

    `bias` 只在 `has_bias` 为真时读取；不带的那个包装函数按 `scalar.mojo`
    同样的约定把 `dst` 传进来占位。
    """
    if rows <= 0 or cols <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "matrix dimensions must be positive",
            "rows=" + String(rows) + " cols=" + String(cols),
        )
    if cols % Q4_BLOCK != 0:
        raise AlofaError(
            ERR_SHAPE_MISMATCH,
            "inner dimension must be a whole number of blocks",
            "cols=" + String(cols) + " block=" + String(Q4_BLOCK),
        )
    var blocks_per_row = cols // Q4_BLOCK
    for row in range(rows):
        var acc = Float64(0)
        if has_bias:
            acc = Float64(bias[unsafe_offset=row])
        var row_block = row * blocks_per_row
        for b in range(blocks_per_row):
            var base = (row_block + b) * Q4_BYTES
            var d = block_scale(blocks, base)
            var x_base = b * Q4_BLOCK
            for j in range(16):
                var byte = Int(blocks[unsafe_offset=base + Q4_SCALE_BYTES + j])
                var w0 = Float32((byte & 0x0F) - 8) * d
                var w1 = Float32(((byte >> 4) & 0x0F) - 8) * d
                acc += Float64(w0) * Float64(x[unsafe_offset=x_base + j])
                acc += Float64(w1) * Float64(x[unsafe_offset=x_base + j + 16])
        dst[unsafe_offset=row] = Float32(acc)


def matmul_q4_f32(
    dst: F32Ptr, x: F32Ptr, blocks: U8Ptr, rows: Int, cols: Int
) raises AlofaError:
    """`dst = W · x`，`W` 为 q4_0 块流，无偏置。"""
    _matmul_q4(dst, x, blocks, rows, cols, dst, False)


def matmul_q4_f32_bias(
    dst: F32Ptr, x: F32Ptr, blocks: U8Ptr, rows: Int, cols: Int, bias: F32Ptr
) raises AlofaError:
    """`dst = W · x + bias`；Qwen2 的 q/k/v 投影都带偏置。"""
    _matmul_q4(dst, x, blocks, rows, cols, bias, True)
