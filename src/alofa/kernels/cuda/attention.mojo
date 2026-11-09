"""GPU 因果 GQA attention —— 与标量 `attention` 逐句对应的串行版。

这是差分门的对象：一个线程算一个 `(head, t)`，三遍循环（scores → max →
exp+sum → V 累加 / total）与 `kernels/cpu/scalar.mojo` 的 `attention` 同一个
顺序、同一个 `Float64` 累加位宽、同一个因果掩码（`upto = kv_len - q_len + t`，
掩码外的 scores 只写不读）。规约全串行 ⇒ 判据只面对一个新变量：**exp**（GPU
走 libdevice、CPU 走 libm，两者不保证同 ulp）。门用 1e-5 相对容差，并把实际
差值打出来 —— 若 exp 的差异真的存在，这个数字就是它的量级记录。

性能版（块内并行规约）等这个串行版在门里立足之后再做：串行版就是它的对照臂。
"""

from std.gpu import block_idx, global_idx, thread_idx
from std.math import exp, sqrt
from std.memory import stack_allocation

from max.gpu import AddressSpace, barrier

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
    var row = (head * Int(q_len) + t) * Int(kv_len)

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


def attention_par_gpu[THREADS: Int](
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
    """decode 形状的性能版：一个 `(head, t)` 一个 CUDA block，块内三遍并行。

    顺序变量的账（相对串行版，也是相对标量）：
    - **scores 点积**：线程认领 `j = tid, tid+THREADS, …`，但单个 j 内仍按 d
      升序串行 —— 与标量同序 ⇒ **scores 逐位一致**；
    - **max**：浮点比较无舍入，树规约与串行精确相等 —— 顺序自由；
    - **exp + total**：total 改为树规约 —— 这是唯一新增的舍入顺序变量（叠加
      已知的 expf 1 ulp），容差覆盖；
    - **V 累加**：每线程认领一个 d（要求 `head_dim <= THREADS`），j 升序串行
      —— 与标量同序。

    Qwen 0.5B decode：head_dim=64 ≤ THREADS，n_heads=14 个 block 各 64 线程
    （串行版是 14 个线程各扛 kv_len 全部工作）。`scores` 是
    `n_heads × q_len × kv_len`，行偏移 `(head*q_len + t)*kv_len` —— 头之间
    并行，行必须私有（共享行的覆写竞态在 bench decode 形状实测 17% 偏差）。
    """
    var head = Int(block_idx.x)
    var t = Int(block_idx.y)
    if head >= Int(n_heads) or t >= Int(q_len):
        return
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var q_head_base = head * Int(head_dim)
    var kv_head_base = kv_head * Int(head_dim)
    var q_cols = Int(n_heads) * Int(head_dim)
    var kv_cols = Int(n_kv_heads) * Int(head_dim)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var upto = Int(kv_len) - Int(q_len) + t
    var row = (head * Int(q_len) + t) * Int(kv_len)

    var smax = stack_allocation[THREADS, Float32, address_space = AddressSpace.SHARED]()
    var stot = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()

    var best_local = LOWEST_FP32
    var j = tid
    while j <= upto:
        var acc = Float64(0)
        for d in range(Int(head_dim)):
            acc += Float64(q[unsafe_offset=t * q_cols + q_head_base + d]) * Float64(
                k[unsafe_offset=j * kv_cols + kv_head_base + d]
            )
        var s = Float32(acc) * scale
        scores[unsafe_offset=row + j] = s
        if s > best_local:
            best_local = s
        j += THREADS

    smax[tid] = best_local
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            if smax[tid + stride] > smax[tid]:
                smax[tid] = smax[tid + stride]
        barrier()
        stride //= 2
    var best = smax[0]

    var total_local = Float64(0)
    j = tid
    while j <= upto:
        var e = Float64(exp(Float64(scores[unsafe_offset=row + j] - best)))
        scores[unsafe_offset=row + j] = Float32(e)
        total_local += e
        j += THREADS

    stot[tid] = total_local
    barrier()
    stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            stot[tid] += stot[tid + stride]
        barrier()
        stride //= 2
    var total = stot[0]

    if tid < Int(head_dim):
        var acc = Float64(0)
        for jj in range(upto + 1):
            acc += Float64(scores[unsafe_offset=row + jj]) * Float64(
                v[unsafe_offset=jj * kv_cols + kv_head_base + tid]
            )
        dst[unsafe_offset=t * q_cols + q_head_base + tid] = Float32(acc / total)


def attn_rope_gpu[THREADS: Int](
    dst: Pointer[Float32, MutUntrackedOrigin],
    q_rot: Pointer[Float32, MutUntrackedOrigin],
    q_src: Pointer[Float32, MutUntrackedOrigin],
    k_src: Pointer[Float32, MutUntrackedOrigin],
    v_src: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    v_store: Pointer[Float32, MutUntrackedOrigin],
    cos_row: Pointer[Float32, MutUntrackedOrigin],
    sin_row: Pointer[Float32, MutUntrackedOrigin],
    scores: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
) -> None:
    """decode 专用：rope(q) + rope(k 直写 store 行) + v 拷贝 + attention
    四合一（账本七十三）—— `rope_kv_gpu` 的活折进 attention 头部，层链
    再省 1 发射。

    跨块竞态的解法在案：当前 token 的 rope 后 k / 原 v **每块算一份
    smem 副本**（兄弟查询块共享同一 kv_head），只有组内首块
    （`head % group == 0`）写 store 行 —— 同一 launch 内读别的块刚写的
    store 行没有可见性保证，直读是竞态；smem 副本与 store 行值逐位相同
    （rope 同式同序），scores/V 相位对 `j == upto`（当前 token）读 smem、
    `j < upto` 读 store。q 每块 rope 自己的头。

    顺序账（相对 rope_kv + attention_par 分离发射）：rope 段逐元素同式
    （t=0，表行 = cos_row/sin_row）⇒ q_rot/store 行**逐位一致**；attention
    三相位与 `attention_par_gpu` 全同（scores 单 j 内 d 升序、max 无舍入、
    total 树规约、V 每 d 一线程 j 升序）⇒ 差分面不变。约束：
    `head_dim <= THREADS`（V 相位每线程一个 d，smem 行也按 THREADS 分配）。
    发射约定：`grid_dim=(n_heads,1,1)`，q_len=1（decode）。
    """
    var head = Int(block_idx.x)
    if head >= Int(n_heads):
        return
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var q_head_base = head * Int(head_dim)
    var kv_head_base = kv_head * Int(head_dim)
    var q_cols = Int(n_heads) * Int(head_dim)
    var kv_cols = Int(n_kv_heads) * Int(head_dim)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var upto = Int(kv_len) - 1
    var row = head * Int(kv_len)

    var smax = stack_allocation[THREADS, Float32, address_space = AddressSpace.SHARED]()
    var stot = stack_allocation[THREADS, Float64, address_space = AddressSpace.SHARED]()
    var s_k = stack_allocation[THREADS, Float32, address_space = AddressSpace.SHARED]()
    var s_v = stack_allocation[THREADS, Float32, address_space = AddressSpace.SHARED]()

    # rope q（tid < half，与 rope_kv 的 q 段同式：a*cos - b*sin / b*cos + a*sin）。
    var half = Int(head_dim) // 2
    if tid < half:
        var d = tid
        var a = q_src[unsafe_offset=q_head_base + d]
        var b = q_src[unsafe_offset=q_head_base + d + half]
        q_rot[unsafe_offset=q_head_base + d] = a * cos_row[
            unsafe_offset=d
        ] - b * sin_row[unsafe_offset=d]
        q_rot[unsafe_offset=q_head_base + d + half] = (
            b * cos_row[unsafe_offset=d + half]
            + a * sin_row[unsafe_offset=d + half]
        )
    # rope k + v 拷贝：smem 副本人人有，store 行组内首块写。
    if tid < half:
        var d = tid
        var a = k_src[unsafe_offset=kv_head_base + d]
        var b = k_src[unsafe_offset=kv_head_base + d + half]
        var r1 = a * cos_row[unsafe_offset=d] - b * sin_row[unsafe_offset=d]
        var r2 = b * cos_row[unsafe_offset=d + half] + a * sin_row[unsafe_offset=d + half]
        s_k[unsafe_offset=d] = r1
        s_k[unsafe_offset=d + half] = r2
        if head % group == 0:
            k_store[unsafe_offset=upto * kv_cols + kv_head_base + d] = r1
            k_store[unsafe_offset=upto * kv_cols + kv_head_base + d + half] = r2
    if tid < Int(head_dim):
        var vv = v_src[unsafe_offset=kv_head_base + tid]
        s_v[unsafe_offset=tid] = vv
        if head % group == 0:
            v_store[unsafe_offset=upto * kv_cols + kv_head_base + tid] = vv
    barrier()

    # scores 相位：j < upto 读 store（旧 token，此前 launch 已写定）；
    # j == upto（当前 token）读 smem 副本 —— 由 `tid == upto % THREADS`
    # 的线程在循环后补算（max 无舍入，并入 best_local 的次序自由）。
    var best_local = LOWEST_FP32
    var j = tid
    while j < upto:
        var acc = Float64(0)
        var kbase = j * kv_cols + kv_head_base
        for d in range(Int(head_dim)):
            acc += Float64(q_rot[unsafe_offset=q_head_base + d]) * Float64(
                k_store[unsafe_offset=kbase + d]
            )
        var s = Float32(acc) * scale
        scores[unsafe_offset=row + j] = s
        if s > best_local:
            best_local = s
        j += THREADS
    if tid == upto % THREADS:
        var acc = Float64(0)
        for d in range(Int(head_dim)):
            acc += Float64(q_rot[unsafe_offset=q_head_base + d]) * Float64(
                s_k[unsafe_offset=d]
            )
        var s = Float32(acc) * scale
        scores[unsafe_offset=row + upto] = s
        if s > best_local:
            best_local = s

    smax[tid] = best_local
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            if smax[tid + stride] > smax[tid]:
                smax[tid] = smax[tid + stride]
        barrier()
        stride //= 2
    var best = smax[0]

    var total_local = Float64(0)
    j = tid
    while j <= upto:
        var e = Float64(exp(Float64(scores[unsafe_offset=row + j] - best)))
        scores[unsafe_offset=row + j] = Float32(e)
        total_local += e
        j += THREADS

    stot[tid] = total_local
    barrier()
    stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            stot[tid] += stot[tid + stride]
        barrier()
        stride //= 2
    var total = stot[0]

    # V 相位：j < upto 读 store，当前 token 读 smem（j 升序不变）。
    if tid < Int(head_dim):
        var acc = Float64(0)
        for jj in range(upto):
            acc += Float64(scores[unsafe_offset=row + jj]) * Float64(
                v_store[unsafe_offset=jj * kv_cols + kv_head_base + tid]
            )
        acc += Float64(scores[unsafe_offset=row + upto]) * Float64(
            s_v[unsafe_offset=tid]
        )
        dst[unsafe_offset=q_head_base + tid] = Float32(acc / total)
