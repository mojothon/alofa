"""GPU 因果 GQA attention —— 与标量 `attention` 逐句对应的串行版。

这是差分门的对象：一个线程算一个 `(head, t)`，三遍循环（scores → max →
exp+sum → V 累加 / total）与 `kernels/cpu/scalar.mojo` 的 `attention` 同一个
顺序、同一个 `Float64` 累加位宽、同一个因果掩码（`upto = kv_len - q_len + t`，
掩码外的 scores 只写不读）。规约全串行 ⇒ 判据只面对一个新变量：**exp**（GPU
走 libdevice、CPU 走 libm，两者不保证同 ulp）。门用 1e-5 相对容差，并把实际
差值打出来 —— 若 exp 的差异真的存在，这个数字就是它的量级记录。

性能版（块内并行规约）等这个串行版在门里立足之后再做：串行版就是它的对照臂。
"""

from std.gpu import global_idx
from std.math import exp, sqrt

from alofa.kernels.cpu.scalar import LOWEST_FP32


def attention_gpu(
    dst: Pointer[Float32, MutUntrackedOrigin],
    q: Pointer[Float32, MutUntrackedOrigin],
    k: Pointer[Float32, MutUntrackedOrigin],
    v: Pointer[Float32, MutUntrackedOrigin],
    scores: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    q_len: Int32,
    kv_len: Int32,
) -> None:
    """`dst[t, head*hd+d] = softmax(q·kᵀ/√hd, 掩码) · v`，因果掩码，GQA 分组。

    与标量版唯一的结构差别是没有 `head_lo/head_hi` 分片参数 —— GPU 的并行
    单位就是 head×t，不需要按头切。`scores` 用法与标量一致：一整块
    `q_len × kv_len` 的 scratch，每 (head, t) 写一行、读两遍、再覆盖。
    """
    var head = Int(global_idx.x)
    var t = Int(global_idx.y)
    if head >= Int(n_heads) or t >= Int(q_len):
        return
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var q_head_base = head * Int(head_dim)
    var kv_head_base = kv_head * Int(head_dim)
    var q_cols = Int(n_heads) * Int(head_dim)
    var kv_cols = Int(n_kv_heads) * Int(head_dim)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var upto = Int(kv_len) - Int(q_len) + t
    var row = t * Int(kv_len)

    var best = LOWEST_FP32
    for j in range(upto + 1):
        var acc = Float64(0)
        for d in range(Int(head_dim)):
            acc += Float64(q[unsafe_offset=t * q_cols + q_head_base + d]) * Float64(
                k[unsafe_offset=j * kv_cols + kv_head_base + d]
            )
        var s = Float32(acc) * scale
        scores[unsafe_offset=row + j] = s
        if s > best:
            best = s

    var total = Float64(0)
    for j in range(upto + 1):
        var e = Float64(exp(Float64(scores[unsafe_offset=row + j] - best)))
        scores[unsafe_offset=row + j] = Float32(e)
        total += e

    for d in range(Int(head_dim)):
        var acc = Float64(0)
        for j in range(upto + 1):
            acc += Float64(scores[unsafe_offset=row + j]) * Float64(
                v[unsafe_offset=j * kv_cols + kv_head_base + d]
            )
        dst[unsafe_offset=t * q_cols + q_head_base + d] = Float32(acc / total)
