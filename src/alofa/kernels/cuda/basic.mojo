"""GPU 基础 kernel —— 逐行算子，一行一线程的最简形态。

**为什么一行一线程（而不是块内规约）**：当前唯一的判据是差分门
（`tests/gpu/test_cuda_diff.mojo`）——「GPU 算的数与标量后端算的数是同一个」。
块内规约会引入共享内存与浮点累加顺序两个正确性变量，数值对不上时分不清是布局
读错还是规约写错。性能内核是之后的事：那时换的是实现，不是门的判据。

算法与累加顺序必须与 `kernels/cpu/scalar.mojo` 的同名算子**逐句对应**：
Float64 累加、先 rsqrt 再乘。任何自己发明的顺序都会把真正的差异藏进舍入里。
"""

from std.gpu import global_idx, thread_idx, block_idx, grid_dim
from std.math import exp, log, sqrt
from std.memory import bitcast, stack_allocation

from max.gpu import AddressSpace, barrier

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
    """`dst = x / sqrt(mean(x²) + eps) * w`，一行一个 block、块内 256 线程。

    历史：最初是"一行一个线程"——当时唯一判据是差分门，块内规约会引入共享
    内存与浮点累加顺序两个正确性变量。decode 计时（账本六十四）抓到了代价：
    单线程串行读 896 个 f64 ≈ 100µs/次，57 次 norm 占 5.8ms/token。

    现在的实现：一行一个 block，256 线程分块累加（`Float64`）+ 共享内存树形
    归约求 `sum_sq`，`1/sqrt` 每行一次，写回阶段全线程分条。**累加顺序变量**：
    树形归约与标量的顺序求和有合法重组差（与 q4 champion 同一性质），门的
    相对容差 1e-5 覆盖余量巨大。发射约定：`grid_dim=(rows,1,1)`、
    `block_dim=(256,1,1)`。
    """
    var row = Int(block_idx.x)
    if row >= Int(rows):
        return
    var tid = Int(thread_idx.x)
    var base = row * Int(cols)
    var n = Int(cols)

    var local = Float64(0)
    var i = tid
    while i < n:
        var v = Float64(x[unsafe_offset=base + i])
        local += v * v
        i += 256

    var smem = stack_allocation[256, Float64, address_space = AddressSpace.SHARED]()
    smem[unsafe_offset=tid] = local
    barrier()
    var stride = 128
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] = (
                smem[unsafe_offset=tid] + smem[unsafe_offset=tid + stride]
            )
        barrier()
        stride //= 2

    var scale = Float64(1.0) / sqrt(
        smem[unsafe_offset=0] / Float64(n) + Float64(eps)
    )
    i = tid
    while i < n:
        dst[unsafe_offset=base + i] = Float32(
            Float64(x[unsafe_offset=base + i]) * scale * Float64(w[unsafe_offset=i])
        )
        i += 256


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


def linear_block_gpu[THREADS: Int](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    inner: Int32,
    out_cols: Int32,
) -> None:
    """lm_head 专用的**块内合并读** linear（账本六十五）。

    `linear_gpu` 是一个 (行, 输出) 一个线程 —— warp 内 32 线程各读 w 的
    不同行，零合并，lm_head（151936x896，544MB）实测 1.14ms = 3× 内存
    下限。这里一行一个 block：线程按 k 分条**合并**读 x 与 w 的同一行，
    共享内存树形归约（与 rmsnorm 重写同一形态）。代价：累加从串行变树形
    —— 合法重组差，门的相对容差覆盖（rmsnorm 重写的同一判据逻辑）。

    发射约定：`grid_dim=(out_cols, rows)`、`block_dim=(THREADS,1,1)`，
    THREADS=256（896/256 = 3.5 → 每线程 3-4 个元素）。
    """
    var o = Int(block_idx.x)
    var row = Int(block_idx.y)
    if o >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var x_base = row * Int(inner)
    var w_base = o * Int(inner)
    var local = Float64(0)
    var k = tid
    while k < Int(inner):
        local += Float64(x[unsafe_offset=x_base + k]) * Float64(w[unsafe_offset=w_base + k])
        k += THREADS
    var smem = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
    smem[unsafe_offset=tid] = local
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] += smem[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        dst[unsafe_offset=row * Int(out_cols) + o] = Float32(smem[unsafe_offset=0])


def linear_fat_gpu[THREADS: Int, INNER: Int, OUTS: Int](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    out_cols: Int32,
) -> None:
    """lm_head 的 fat-CTA 版（账本六十六）。

    linear_block（六十五）一行一 block：151936 个 CTA 各自重读同一 x 行
    （896x4B），x 经 L2 的总流量与权重同量级 —— 1.02ms 里约一半是它。
    这里一个 block 处理 OUTS 个输出：x 只读一次（进 smem），L2 流量降
    OUTS 倍。INNER comptime（smem 容量需要；本架构 hidden=896 固定）。
    累加序：线程按 inner 跨步、树形归约 —— 与 linear_block 同构，重组差
    由门的相对容差覆盖（同一判据逻辑）。

    发射约定：`grid_dim=(ceildiv(out_cols, OUTS), rows)`、
    `block_dim=(THREADS,1,1)`。
    """
    var o0 = Int(block_idx.x) * OUTS
    var row = Int(block_idx.y)
    if o0 >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(INNER)
    var k = tid
    while k < Int(INNER):
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
            k = tid
            while k < Int(INNER):
                a += Float64(sx[unsafe_offset=k]) * Float64(w[unsafe_offset=w_base + k])
                k += THREADS
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
                dst[unsafe_offset=row * Int(out_cols) + o] = Float32(sacc[unsafe_offset=oo * THREADS])


def linear_fat_f16_gpu[THREADS: Int, INNER: Int, OUTS: Int, NORM: Bool = False](
    x: Pointer[Float32, MutUntrackedOrigin],
    w: Pointer[Float16, MutUntrackedOrigin],
    dst: Pointer[Float32, MutUntrackedOrigin],
    rms_w: Pointer[Float32, MutUntrackedOrigin],
    eps: Float32,
    out_cols: Int32,
) -> None:
    """lm_head 的 f16 版（账本六十八）：权重 f16 使字节流量减半。

    结构同 linear_fat_gpu（x smem 共享 + [OUTS][THREADS] 归约），内层
    `load[8]` f16（行基址 o*INNER*2B 16B 对齐，INNER 为 8 的倍数）。
    数值：权重 f16 舍入 + f64 累加，f16 专用容差覆盖（与 matmul f16
    同一判据族）。

    NORM（账本七十一）：x 载入 smem 后就地 rmsnorm —— 反正每块都载整行，
    归一化系数各块冗余算一遍，换掉单独的 rmsnorm 发射。f64 累加 + 树形
    归约同 `rmsnorm_gpu`，分块顺序随 THREADS 不同（合法重组差，容差门
    覆盖；rmsnorm 单发的零容差判据仍由 session 原语路径保住）。
    """
    var o0 = Int(block_idx.x) * OUTS
    var row = Int(block_idx.y)
    if o0 >= Int(out_cols):
        return
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[INNER, Float32, address_space = AddressSpace.SHARED]()
    var x_base = row * Int(INNER)
    var k = tid
    while k < Int(INNER):
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
    var sacc = stack_allocation[OUTS * THREADS, Float64, address_space = AddressSpace.SHARED]()
    @parameter
    for oo in range(OUTS):
        var o = o0 + oo
        var a = Float64(0)
        if o < Int(out_cols):
            var w_base = o * Int(INNER)
            k = tid * 8
            while k < Int(INNER):
                var wv = w.load[8](w_base + k)
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
                dst[unsafe_offset=row * Int(out_cols) + o] = Float32(sacc[unsafe_offset=oo * THREADS])


def argmax_rb_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    out_idx: Pointer[Int32, MutUntrackedOrigin],
    vocab: Int32,
) -> None:
    """行 argmax（账本一百二十）：grid=(B,)，块 b 对 [b][vocab] 行
    求 argmax 直写 out[b]。首现并列语义与 host/argmax_f32 一致。
    256 线程跨步 + smem 树。"""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = b * Int(vocab)
    var best = Float32(-3.402823466e38)
    var bi = Int32(-1)
    var i = tid
    while i < Int(vocab):
        var v = x[unsafe_offset=base + i]
        if v > best:
            best = v
            bi = Int32(i)
        i += 256
    var sval = stack_allocation[
        256, Float32, address_space = AddressSpace.SHARED
    ]()
    var sidx = stack_allocation[
        256, Int32, address_space = AddressSpace.SHARED
    ]()
    sval[unsafe_offset=tid] = best
    sidx[unsafe_offset=tid] = bi
    barrier()
    var stride = 128
    while stride > 0:
        if tid < stride:
            var v2 = sval[unsafe_offset=tid + stride]
            var i2 = sidx[unsafe_offset=tid + stride]
            var v1 = sval[unsafe_offset=tid]
            var i1 = sidx[unsafe_offset=tid]
            if (v2 > v1) or ((v2 == v1) and (i2 < i1)):
                sval[unsafe_offset=tid] = v2
                sidx[unsafe_offset=tid] = i2
        barrier()
        stride //= 2
    if tid == 0:
        out_idx[unsafe_offset=b] = sidx[unsafe_offset=0]
def sample_topp_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    out_idx: Pointer[Int32, MutUntrackedOrigin],
    vocab: Int32,
    temp: Float32,
    top_p: Float32,
    seed: Int32,
) -> None:
    """行 top-p（nucleus）温度采样（账本一百二十八）。

    无排序二分阈值：M(τ)=Σ_{logit≥τ} exp((logit-m)/T)/Z 随 τ 单调
    递减（m=行最大 logit 稳定常数，Z 全词表一趟归一）；24 趟二分找
    最大 τ 使归一化质量 M(τ) ≥ top_p —— 最小 nucleus = {logit ≥ τ*}。核内 Gumbel-max
    （score=logit+T·G）相对分数自动完成核内重归一 ≡ softmax(logits/T)
    在核上的归一化采样。top_p=1 退化为 sample_rb（核=全词表）。
    """
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = b * Int(vocab)

    # 行最大（稳定常数；核必含 argmax：logit==m ≥ 任何收敛 τ）
    var mx = Float32(-3.402823466e38)
    var i = tid
    while i < Int(vocab):
        var v = x[unsafe_offset=base + i]
        if v > mx:
            mx = v
        i += 256
    var sm = stack_allocation[
        256, Float32, address_space = AddressSpace.SHARED
    ]()
    sm[unsafe_offset=tid] = mx
    barrier()
    var stride = 128
    while stride > 0:
        if tid < stride:
            var v2 = sm[unsafe_offset=tid + stride]
            var v1 = sm[unsafe_offset=tid]
            if v2 > v1:
                sm[unsafe_offset=tid] = v2
        barrier()
        stride //= 2
    var m = sm[unsafe_offset=0]

    # 全词表归一化 Z = Σ exp((logit-m)/T)（一趟）
    var zacc = Float32(0)
    i = tid
    while i < Int(vocab):
        zacc += exp((x[unsafe_offset=base + i] - m) / temp)
        i += 256
    sm[unsafe_offset=tid] = zacc
    barrier()
    var zs = 128
    while zs > 0:
        if tid < zs:
            sm[unsafe_offset=tid] += sm[unsafe_offset=tid + zs]
        barrier()
        zs //= 2
    var zinv = Float32(1) / sm[unsafe_offset=0]

    # 二分 τ：归一化质量 acc·zinv ≥ top_p → lo=τ（核可更小）；否则 hi=τ
    var lo = m - Float32(80) * temp
    var hi = m
    var it = 0
    while it < 24:
        var tau = (lo + hi) / Float32(2)
        var acc = Float32(0)
        i = tid
        while i < Int(vocab):
            var v = x[unsafe_offset=base + i]
            if v >= tau:
                acc += exp((v - m) / temp)
            i += 256
        sm[unsafe_offset=tid] = acc
        barrier()
        var s2 = 128
        while s2 > 0:
            if tid < s2:
                sm[unsafe_offset=tid] += sm[unsafe_offset=tid + s2]
            barrier()
            s2 //= 2
        if sm[unsafe_offset=0] * zinv >= top_p:
            lo = tau
        else:
            hi = tau
        it += 1

    # 核内 Gumbel-max（与 sample_rb 同款 LCG/并列语义）
    var best = Float32(-3.402823466e38)
    var bi = Int32(-1)
    var state = bitcast[DType.uint32](seed) ^ (UInt32(b) * 2654435761) ^ (
        UInt32(tid) * 40503
    )
    i = tid
    while i < Int(vocab):
        var v = x[unsafe_offset=base + i]
        if v >= lo:
            state = state * 1664525 + 1013904223
            var ux = (state >> 8) & 16777215
            var u = (Float32(ux) + Float32(0.5)) / Float32(16777216)
            var sc = v + temp * (-log(-log(u)))
            if sc > best:
                best = sc
                bi = Int32(i)
        i += 256
    var sv = stack_allocation[
        256, Float32, address_space = AddressSpace.SHARED
    ]()
    var si = stack_allocation[
        256, Int32, address_space = AddressSpace.SHARED
    ]()
    sv[unsafe_offset=tid] = best
    si[unsafe_offset=tid] = bi
    barrier()
    var s3 = 128
    while s3 > 0:
        if tid < s3:
            var v2 = sv[unsafe_offset=tid + s3]
            var i2 = si[unsafe_offset=tid + s3]
            var v1 = sv[unsafe_offset=tid]
            var i1 = si[unsafe_offset=tid]
            if (v2 > v1) or ((v2 == v1) and (i2 < i1)):
                sv[unsafe_offset=tid] = v2
                si[unsafe_offset=tid] = i2
        barrier()
        s3 //= 2
    if tid == 0:
        out_idx[unsafe_offset=b] = si[unsafe_offset=0]


def sample_rb_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    out_idx: Pointer[Int32, MutUntrackedOrigin],
    vocab: Int32,
    temp: Float32,
    seed: Int32,
) -> None:
    """行温度采样（账本一百二十三）：grid=(B,)，Gumbel-max 技巧
    ——argmax(logit + T·G) 精确等价 softmax(logits/T) 采样。
    每线程 LCG（state 由 seed/row/tid 派生）逐元素出 u，
    Gumbel g=-log(-log(u))；temp=0 走纯 argmax（与 argmax_rb 同）。
    并列取小下标（首现语义一致）。"""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = b * Int(vocab)
    var best = Float32(-3.402823466e38)
    var bi = Int32(-1)
    var state = bitcast[DType.uint32](seed) ^ (UInt32(b) * 2654435761) ^ (
        UInt32(tid) * 40503
    )
    var i = tid
    while i < Int(vocab):
        var v = x[unsafe_offset=base + i]
        if temp > Float32(0):
            state = state * 1664525 + 1013904223
            var ux = (state >> 8) & 16777215
            var u = (Float32(ux) + Float32(0.5)) / Float32(16777216)
            v = v + temp * (-log(-log(u)))
        if v > best:
            best = v
            bi = Int32(i)
        i += 256
    var sval = stack_allocation[
        256, Float32, address_space = AddressSpace.SHARED
    ]()
    var sidx = stack_allocation[
        256, Int32, address_space = AddressSpace.SHARED
    ]()
    sval[unsafe_offset=tid] = best
    sidx[unsafe_offset=tid] = bi
    barrier()
    var stride = 128
    while stride > 0:
        if tid < stride:
            var v2 = sval[unsafe_offset=tid + stride]
            var i2 = sidx[unsafe_offset=tid + stride]
            var v1 = sval[unsafe_offset=tid]
            var i1 = sidx[unsafe_offset=tid]
            if (v2 > v1) or ((v2 == v1) and (i2 < i1)):
                sval[unsafe_offset=tid] = v2
                sidx[unsafe_offset=tid] = i2
        barrier()
        stride //= 2
    if tid == 0:
        out_idx[unsafe_offset=b] = sidx[unsafe_offset=0]


def sample_topp_row_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    out_idx: Pointer[Int32, MutUntrackedOrigin],
    vocab: Int32,
    temps: Pointer[Float32, MutUntrackedOrigin],
    topps: Pointer[Float32, MutUntrackedOrigin],
    seeds: Pointer[Int32, MutUntrackedOrigin],
) -> None:
    """**per-row 参数**统一采样核（账本一百三十六）：grid=(B,)，行 b 的
    参数从数组读——temp=0 走纯 argmax；temp>0 且 top_p≥1 走全词表
    Gumbel；否则 nucleus 二分。**种子不再派生自行号**（host 算好直传
    seeds[b]）——采样结果与行号解耦，合批编排（padding/压缩/行号漂移）
    不再影响重放语义；t=1 时与 sample_topp_gpu/sample_rb_gpu 逐位同
    （b=0 时 b·C 派生项为零）。

    GpuService 的批行池因此**合批条件彻底消失**：任意 temp/top_p 组合
    的一批行一个 launch 搞定，每行自选分支。
    """
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = b * Int(vocab)
    var temp = temps[unsafe_offset=b]
    var top_p = topps[unsafe_offset=b]
    var seed = seeds[unsafe_offset=b]

    # 纯 argmax 分支（与 argmax_rb 逐位同：无 LCG、并列取小下标）。
    if temp <= Float32(0):
        var best0 = Float32(-3.402823466e38)
        var bi0 = Int32(-1)
        var i0 = tid
        while i0 < Int(vocab):
            var v0 = x[unsafe_offset=base + i0]
            if v0 > best0:
                best0 = v0
                bi0 = Int32(i0)
            i0 += 256
        var s0v = stack_allocation[
            256, Float32, address_space = AddressSpace.SHARED
        ]()
        var s0i = stack_allocation[
            256, Int32, address_space = AddressSpace.SHARED
        ]()
        s0v[unsafe_offset=tid] = best0
        s0i[unsafe_offset=tid] = bi0
        barrier()
        var s0 = 128
        while s0 > 0:
            if tid < s0:
                var v2 = s0v[unsafe_offset=tid + s0]
                var i2 = s0i[unsafe_offset=tid + s0]
                var v1 = s0v[unsafe_offset=tid]
                var i1 = s0i[unsafe_offset=tid]
                if (v2 > v1) or ((v2 == v1) and (i2 < i1)):
                    s0v[unsafe_offset=tid] = v2
                    s0i[unsafe_offset=tid] = i2
            barrier()
            s0 //= 2
        if tid == 0:
            out_idx[unsafe_offset=b] = s0i[unsafe_offset=0]
        return

    # ---- temp>0：行最大（稳定常数） ----
    var mx = Float32(-3.402823466e38)
    var i = tid
    while i < Int(vocab):
        var v = x[unsafe_offset=base + i]
        if v > mx:
            mx = v
        i += 256
    var sm = stack_allocation[
        256, Float32, address_space = AddressSpace.SHARED
    ]()
    sm[unsafe_offset=tid] = mx
    barrier()
    var stride = 128
    while stride > 0:
        if tid < stride:
            var v2 = sm[unsafe_offset=tid + stride]
            var v1 = sm[unsafe_offset=tid]
            if v2 > v1:
                sm[unsafe_offset=tid] = v2
        barrier()
        stride //= 2
    var m = sm[unsafe_offset=0]

    var lo = m - Float32(80) * temp
    if top_p < Float32(1):
        # nucleus：归一化 Z + 24 趟二分 τ（与 sample_topp_gpu 同算序）
        var zacc = Float32(0)
        i = tid
        while i < Int(vocab):
            zacc += exp((x[unsafe_offset=base + i] - m) / temp)
            i += 256
        sm[unsafe_offset=tid] = zacc
        barrier()
        var zs = 128
        while zs > 0:
            if tid < zs:
                sm[unsafe_offset=tid] += sm[unsafe_offset=tid + zs]
            barrier()
            zs //= 2
        var zinv = Float32(1) / sm[unsafe_offset=0]

        var hi = m
        var it = 0
        while it < 24:
            var tau = (lo + hi) / Float32(2)
            var acc = Float32(0)
            i = tid
            while i < Int(vocab):
                var v = x[unsafe_offset=base + i]
                if v >= tau:
                    acc += exp((v - m) / temp)
                i += 256
            sm[unsafe_offset=tid] = acc
            barrier()
            var s2 = 128
            while s2 > 0:
                if tid < s2:
                    sm[unsafe_offset=tid] += sm[unsafe_offset=tid + s2]
                barrier()
                s2 //= 2
            if sm[unsafe_offset=0] * zinv >= top_p:
                lo = tau
            else:
                hi = tau
            it += 1

    # 核内 Gumbel-max（种子直传，无行号派生）
    var best = Float32(-3.402823466e38)
    var bi = Int32(-1)
    var state = bitcast[DType.uint32](seed) ^ (UInt32(tid) * 40503)
    i = tid
    while i < Int(vocab):
        var v = x[unsafe_offset=base + i]
        if v >= lo:
            state = state * 1664525 + 1013904223
            var ux = (state >> 8) & 16777215
            var u = (Float32(ux) + Float32(0.5)) / Float32(16777216)
            var sc = v + temp * (-log(-log(u)))
            if sc > best:
                best = sc
                bi = Int32(i)
        i += 256
    var sv = stack_allocation[
        256, Float32, address_space = AddressSpace.SHARED
    ]()
    var si = stack_allocation[
        256, Int32, address_space = AddressSpace.SHARED
    ]()
    sv[unsafe_offset=tid] = best
    si[unsafe_offset=tid] = bi
    barrier()
    var s3 = 128
    while s3 > 0:
        if tid < s3:
            var v2 = sv[unsafe_offset=tid + s3]
            var i2 = si[unsafe_offset=tid + s3]
            var v1 = sv[unsafe_offset=tid]
            var i1 = si[unsafe_offset=tid]
            if (v2 > v1) or ((v2 == v1) and (i2 < i1)):
                sv[unsafe_offset=tid] = v2
                si[unsafe_offset=tid] = i2
        barrier()
        s3 //= 2
    if tid == 0:
        out_idx[unsafe_offset=b] = si[unsafe_offset=0]


def argmax_f32_gpu(
    x: Pointer[Float32, MutUntrackedOrigin],
    part_val: Pointer[Float32, MutUntrackedOrigin],
    part_idx: Pointer[Int32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """词表 argmax 相一（账本一百一十六-b）：每块扫一段，块内跨步
    遍历 + smem 树（并列取小下标，与 host 首现语义一致），partials
    写 (val, idx) 两数组。配 argmax_comb_gpu 收口。"""
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var nblocks = Int(grid_dim.x)
    var chunk = (Int(n) + nblocks - 1) // nblocks
    var lo = bid * chunk
    var hi = min(lo + chunk, Int(n))
    var best = Float32(-3.402823466e38)
    var bi = Int32(-1)
    var i = lo + tid
    while i < hi:
        var v = x[unsafe_offset=i]
        if v > best:
            best = v
            bi = Int32(i)
        i += 256
    var sval = stack_allocation[
        256, Float32, address_space = AddressSpace.SHARED
    ]()
    var sidx = stack_allocation[
        256, Int32, address_space = AddressSpace.SHARED
    ]()
    sval[unsafe_offset=tid] = best
    sidx[unsafe_offset=tid] = bi
    barrier()
    var stride = 128
    while stride > 0:
        if tid < stride:
            var v2 = sval[unsafe_offset=tid + stride]
            var i2 = sidx[unsafe_offset=tid + stride]
            var v1 = sval[unsafe_offset=tid]
            var i1 = sidx[unsafe_offset=tid]
            if (v2 > v1) or ((v2 == v1) and (i2 < i1)):
                sval[unsafe_offset=tid] = v2
                sidx[unsafe_offset=tid] = i2
        barrier()
        stride //= 2
    if tid == 0:
        part_val[unsafe_offset=bid] = sval[unsafe_offset=0]
        part_idx[unsafe_offset=bid] = sidx[unsafe_offset=0]


def argmax_comb_gpu(
    part_val: Pointer[Float32, MutUntrackedOrigin],
    part_idx: Pointer[Int32, MutUntrackedOrigin],
    out_idx: Pointer[Int32, MutUntrackedOrigin],
    nb: Int32,
) -> None:
    """词表 argmax 相二（账本一百一十六-b）：单块树收 partials，
    直写 d_token（4 字节回传替代 608KB logits D2H）。"""
    var tid = Int(thread_idx.x)
    var sval = stack_allocation[
        256, Float32, address_space = AddressSpace.SHARED
    ]()
    var sidx = stack_allocation[
        256, Int32, address_space = AddressSpace.SHARED
    ]()
    var best = Float32(-3.402823466e38)
    var bi = Int32(-1)
    var i = tid
    while i < Int(nb):
        var v = part_val[unsafe_offset=i]
        if v > best:
            best = v
            bi = part_idx[unsafe_offset=i]
        elif (v == best) and (part_idx[unsafe_offset=i] < bi):
            bi = part_idx[unsafe_offset=i]
        i += 256
    sval[unsafe_offset=tid] = best
    sidx[unsafe_offset=tid] = bi
    barrier()
    var stride = 128
    while stride > 0:
        if tid < stride:
            var v2 = sval[unsafe_offset=tid + stride]
            var i2 = sidx[unsafe_offset=tid + stride]
            var v1 = sval[unsafe_offset=tid]
            var i1 = sidx[unsafe_offset=tid]
            if (v2 > v1) or ((v2 == v1) and (i2 < i1)):
                sval[unsafe_offset=tid] = v2
                sidx[unsafe_offset=tid] = i2
        barrier()
        stride //= 2
    if tid == 0:
        out_idx[unsafe_offset=0] = sidx[unsafe_offset=0]


def hbm_probe_f16_gpu[THREADS: Int](
    src: Pointer[Float16, MutUntrackedOrigin],
    sink: Pointer[Float64, MutUntrackedOrigin],
    n_elem: Int32,
) -> None:
    """HBM 纯流式探针（账本八十七，诊断工具）：每块连续条带 f16
    load[8] 累加，f64 树收口写 1 值防 DCE。量盒子对 f16 权值布局的
    真实带宽上限（与 head/dual/down 的 670-800GB/s 对照）。"""
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var per_block = Int(n_elem) // Int(grid_dim.x)
    var base = bid * per_block
    var acc = Float64(0)
    var k = tid * 8
    while k < per_block:
        var wv = src.load[8](base + k)
        @parameter
        for j in range(8):
            acc += Float64(Float32(wv[j]))
        k += THREADS * 8
    var sred = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
    sred[unsafe_offset=tid] = acc
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            sred[unsafe_offset=tid] += sred[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        sink[unsafe_offset=bid] = sred[unsafe_offset=0]
