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


def attn_rope_gpu[
    THREADS: Int, VEC: Bool = False, cut: Int = 0, V2: Bool = False,
    smode: Int = 0,
](
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

    # q 行预载（账本八十一 smode≥1）：q 每块恒定，原实现每 j 重读
    # 128 条 global LDG —— 一次搬进 smem（f64），点积改 smem 读。
    # 值 Float64(q_rot[...]) 逐位同原式 ⇒ scores bit-exact。
    var s_q = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    comptime if smode >= 1:
        if tid < Int(head_dim):
            s_q[tid] = Float64(q_rot[unsafe_offset=q_head_base + tid])
        barrier()

    # scores 相位：j < upto 读 store（旧 token，此前 launch 已写定）；
    # j == upto（当前 token）读 smem 副本 —— 由 `tid == upto % THREADS`
    # 的线程在循环后补算（max 无舍入，并入 best_local 的次序自由）。
    var best_local = LOWEST_FP32
    var j = tid
    comptime if cut & 1 == 0 and smode == 4:
        # smode 4（账本八十一）：2 线程/j 半维点积 —— 线程对 (2j, 2j+1)
        # 各算 head_dim 一半（16B 向量读、成对覆盖连续行），smem 对合
        # 并。⚠️ d 求和序变（两半 f64 重结合），非 bit-exact（门 1e-5
        # + argmax 判据；V2 先例 diff 未翻位）。
        var s_pair = stack_allocation[
            THREADS, Float64, address_space = AddressSpace.SHARED
        ]()
        var half_hd = Int(head_dim) // 2
        var c8h = half_hd // 8
        var half = tid % 2
        var jstep = THREADS // 2
        var rounds = (upto + jstep - 1) // jstep
        for r4 in range(rounds):
            var jj = r4 * jstep + tid // 2
            var acc4 = Float64(0)
            if jj < upto:
                var kb4 = jj * kv_cols + kv_head_base + half * half_hd
                for c in range(c8h):
                    var kv8 = k_store.load[8](kb4 + c * 8)
                    @parameter
                    for dd in range(8):
                        acc4 += s_q[half * half_hd + c * 8 + dd] * Float64(
                            kv8[dd]
                        )
                for d in range(c8h * 8, half_hd):
                    acc4 += s_q[half * half_hd + d] * Float64(
                        k_store[unsafe_offset=kb4 + d]
                    )
            s_pair[tid] = acc4
            barrier()
            if half == 0 and jj < upto:
                var s4 = Float32(s_pair[tid] + s_pair[tid + 1]) * scale
                scores[unsafe_offset=row + jj] = s4
                if s4 > best_local:
                    best_local = s4
            barrier()
    comptime if cut & 1 == 0 and smode != 4:
        while j < upto:
            var acc = Float64(0)
            var kbase = j * kv_cols + kv_head_base
            # VEC（账本七十九）：k 行 16B 向量读（load[8]，matmul 同款）——
            # 每 j 16 条向量 load 替代 128 条标量 2B 读，load 相互独立可
            # 全部在飞。求和序 d 升序不变（8 元组内 dd 升序）⇒ scores
            # 逐位同旧，下游 bit-exact。
            comptime if smode == 2:
                var c8 = Int(head_dim) // 8
                var cur = k_store.load[8](kbase)
                for c in range(c8):
                    var nxi = c + 1
                    if nxi >= c8:
                        nxi = c8 - 1
                    var nxt = k_store.load[8](kbase + nxi * 8)
                    @parameter
                    for dd in range(8):
                        acc += s_q[c * 8 + dd] * Float64(cur[dd])
                    cur = nxt
                for d in range(c8 * 8, Int(head_dim)):
                    acc += s_q[d] * Float64(k_store[unsafe_offset=kbase + d])
            elif smode == 1:
                var c8 = Int(head_dim) // 8
                for c in range(c8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += s_q[c * 8 + dd] * Float64(kv8[dd])
                for d in range(c8 * 8, Int(head_dim)):
                    acc += s_q[d] * Float64(k_store[unsafe_offset=kbase + d])
            elif VEC:
                var c8 = Int(head_dim) // 8
                for c in range(c8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += Float64(
                            q_rot[unsafe_offset=q_head_base + c * 8 + dd]
                        ) * Float64(Float32(kv8[dd]))
                for d in range(c8 * 8, Int(head_dim)):
                    acc += Float64(q_rot[unsafe_offset=q_head_base + d]) * Float64(
                        k_store[unsafe_offset=kbase + d]
                    )
            else:
                for d in range(Int(head_dim)):
                    acc += Float64(q_rot[unsafe_offset=q_head_base + d]) * Float64(
                        k_store[unsafe_offset=kbase + d]
                    )
            var s = Float32(acc) * scale
            scores[unsafe_offset=row + j] = s
            if s > best_local:
                best_local = s
            j += THREADS
    comptime if cut & 1 == 0:
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

    var best = Float32(0)
    var stride = THREADS // 2
    comptime if cut & 2 == 0:
        smax[tid] = best_local
        barrier()
        stride = THREADS // 2
        while stride > 0:
            if tid < stride:
                if smax[tid + stride] > smax[tid]:
                    smax[tid] = smax[tid + stride]
            barrier()
            stride //= 2
        best = smax[0]

    var total_local = Float64(0)
    comptime if cut & 4 == 0:
        j = tid
        while j <= upto:
            var e = Float64(exp(Float64(scores[unsafe_offset=row + j] - best)))
            scores[unsafe_offset=row + j] = Float32(e)
            total_local += e
            j += THREADS

    var total = Float64(1)
    comptime if cut & 8 == 0:
        stot[tid] = total_local
        barrier()
        stride = THREADS // 2
        while stride > 0:
            if tid < stride:
                stot[tid] += stot[tid + stride]
            barrier()
            stride //= 2
        total = stot[0]

    # V 相位：j < upto 读 store，当前 token 读 smem（j 升序不变）。
    # V2（账本八十）：THREADS > head_dim 时按 j 分组并行（组 g 处理
    # j ≡ g (mod groups)，组内升序），部分和进 smem 后按组序合并 ——
    # 串行链 ÷ groups。⚠️ j 求和序变（f64 重结合），非 bit-exact，
    # 以门 1e-5 容差 + argmax 不变为判据（分相实测 V 占 kernel 35%）。
    comptime if cut & 16 == 0:
        comptime if V2:
            var s_vg = stack_allocation[
                THREADS, Float64, address_space = AddressSpace.SHARED
            ]()
            var groups = Int(THREADS) // Int(head_dim)
            var d = tid % Int(head_dim)
            var g = tid // Int(head_dim)
            var acc2 = Float64(0)
            var jj2 = g
            while jj2 < upto:
                acc2 += Float64(scores[unsafe_offset=row + jj2]) * Float64(
                    v_store[
                        unsafe_offset=jj2 * kv_cols + kv_head_base + d
                    ]
                )
                jj2 += groups
            if g == 0:
                acc2 += Float64(scores[unsafe_offset=row + upto]) * Float64(
                    s_v[unsafe_offset=d]
                )
            s_vg[tid] = acc2
            barrier()
            if tid < Int(head_dim):
                var acc = s_vg[tid]
                for g2 in range(1, groups):
                    acc += s_vg[tid + g2 * Int(head_dim)]
                dst[unsafe_offset=q_head_base + tid] = Float32(acc / total)
        else:
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


def attn_rs_gpu[
    THREADS: Int, VEC: Bool = True, TILES: Int = 4,
](
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
    """K1（账本八十二，跨块 j 分块三趟之首趟）：rope(q/k) + scores
    分块。grid=(n_heads, TILES) —— 八十一判决：14 块锁 per-SM 天花板
    （scores 15.4µs@kv300），j 分块 ×TILES 倍 SM。每块自算当前 token
    的 k rope smem 副本（七十三解法保留，免同 launch 跨块竞态），仅
    tile 0 写 store 行（跨 kernel 可见，K2 直读 store 行无竞态）。
    scores 点积同 VEC 式同序（d 升序）⇒ 逐位同旧。
    """
    var head = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var q_head_base = head * Int(head_dim)
    var kv_head_base = kv_head * Int(head_dim)
    var kv_cols = Int(n_kv_heads) * Int(head_dim)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var upto = Int(kv_len) - 1
    var row = head * Int(kv_len)
    var nsc = upto + 1
    var chunk = (nsc + TILES - 1) // TILES
    var j0 = tile * chunk
    var j1 = j0 + chunk
    if j1 > nsc:
        j1 = nsc

    var s_k = stack_allocation[
        THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
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
    if tid < half:
        var d = tid
        var a = k_src[unsafe_offset=kv_head_base + d]
        var b = k_src[unsafe_offset=kv_head_base + d + half]
        var r1 = a * cos_row[unsafe_offset=d] - b * sin_row[unsafe_offset=d]
        var r2 = b * cos_row[unsafe_offset=d + half] + a * sin_row[
            unsafe_offset=d + half
        ]
        s_k[unsafe_offset=d] = r1
        s_k[unsafe_offset=d + half] = r2
        if tile == 0:
            k_store[unsafe_offset=upto * kv_cols + kv_head_base + d] = r1
            k_store[unsafe_offset=upto * kv_cols + kv_head_base + d + half] = r2
    if tid < Int(head_dim):
        if tile == 0:
            v_store[unsafe_offset=upto * kv_cols + kv_head_base + tid] = (
                v_src[unsafe_offset=kv_head_base + tid]
            )
    barrier()

    var jj = j0 + tid
    while jj < j1:
        var acc = Float64(0)
        if jj < upto:
            comptime if VEC:
                var kbase = jj * kv_cols + kv_head_base
                var c8 = Int(head_dim) // 8
                for c in range(c8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += Float64(
                            q_rot[unsafe_offset=q_head_base + c * 8 + dd]
                        ) * Float64(Float32(kv8[dd]))
                for d in range(c8 * 8, Int(head_dim)):
                    acc += Float64(
                        q_rot[unsafe_offset=q_head_base + d]
                    ) * Float64(k_store[unsafe_offset=kbase + d])
        else:
            for d in range(Int(head_dim)):
                acc += Float64(q_rot[unsafe_offset=q_head_base + d]) * Float64(
                    s_k[unsafe_offset=d]
                )
        scores[unsafe_offset=row + jj] = Float32(acc) * scale
        jj += THREADS


def attn_vt_gpu[
    THREADS: Int, TILES: Int = 4,
](
    scores: Pointer[Float32, MutUntrackedOrigin],
    v_store: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tot: Pointer[Float64, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
) -> None:
    """K2（三趟之二）：V 相位分块 + 块内冗余 softmax。gmax 无舍入序
    自由；gsum 树序与旧 total 相同（j=tid 步 THREADS）⇒ 逐位同；exp
    写回 scores（旧流程同款，免二次 exp）。V 部分和按 tile×组重结合
    （非 bit-exact，门 1e-5 容差 + argmax 判据）。j==upto 直读
    v_store —— K1（上一 launch）已写定，跨 kernel 可见无竞态。
    """
    var head = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var kv_head_base = kv_head * Int(head_dim)
    var kv_cols = Int(n_kv_heads) * Int(head_dim)
    var upto = Int(kv_len) - 1
    var row = head * Int(kv_len)
    var nsc = upto + 1
    var chunk = (nsc + TILES - 1) // TILES
    var t0 = tile * chunk
    var t1 = t0 + chunk
    if t1 > nsc:
        t1 = nsc

    var smax = stack_allocation[
        THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    var stot = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var best = LOWEST_FP32
    var j = tid
    while j <= upto:
        var s = scores[unsafe_offset=row + j]
        if s > best:
            best = s
        j += THREADS
    smax[tid] = best
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            if smax[tid + stride] > smax[tid]:
                smax[tid] = smax[tid + stride]
        barrier()
        stride //= 2
    var gmax = smax[0]

    var total_local = Float64(0)
    j = tid
    while j <= upto:
        var e = Float64(exp(Float64(scores[unsafe_offset=row + j] - gmax)))
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
    if tile == 0 and tid == 0:
        tot[unsafe_offset=head] = stot[0]

    var s_g = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var hd = Int(head_dim)
    var groups = THREADS // hd
    var d = tid % hd
    var g = tid // hd
    var acc = Float64(0)
    var jj = t0 + g
    while jj < t1:
        acc += Float64(scores[unsafe_offset=row + jj]) * Float64(
            v_store[unsafe_offset=jj * kv_cols + kv_head_base + d]
        )
        jj += groups
    s_g[tid] = acc
    barrier()
    if g == 0:
        var acc2 = s_g[tid]
        for g2 in range(1, groups):
            acc2 += s_g[tid + g2 * hd]
        vpart[unsafe_offset=(head * TILES + tile) * hd + d] = Float32(acc2)


def attn_comb_gpu[TILES: Int = 4](
    dst: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tot: Pointer[Float64, MutUntrackedOrigin],
    n_heads: Int32,
    head_dim: Int32,
) -> None:
    """K3（三趟之三）：tile 部分和合并 + 归一。14 块 × head_dim 线程，
    每 d 一次 TILES 加法 —— 微核（~1µs）。"""
    var head = Int(block_idx.x)
    var d = Int(thread_idx.x)
    var hd = Int(head_dim)
    if d < hd:
        var acc = Float64(0)
        for t in range(TILES):
            acc += Float64(vpart[unsafe_offset=(head * TILES + t) * hd + d])
        dst[unsafe_offset=head * hd + d] = Float32(
            acc / tot[unsafe_offset=head]
        )


def attn_vt2_gpu[
    THREADS: Int, TILES: Int = 8,
](
    scores: Pointer[Float32, MutUntrackedOrigin],
    v_store: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tsum: Pointer[Float64, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
) -> None:
    """K2'（账本八十三，冗余 softmax 消除）：gmax 每块冗余**只读**
    （无 exp 无写回）；exp 只算本 tile（smem e_s 缓存，全行恰 1×），
    tile 局部和树规约后写 tsum[head*TILES+tile]，gsum 推迟给 K3 合并
    —— 砍掉旧 K2 的 8× exp + 8× scores 写回 + gsum 树（10 屏障）。
    exp 值逐位同旧（同 s 同 gmax）；gsum 变 tile 和合并（重结合，
    门 1e-5 容差判据）。空 tile（t0 >= nsc）各循环空转，exp 不触发。
    """
    var head = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var kv_head_base = kv_head * Int(head_dim)
    var kv_cols = Int(n_kv_heads) * Int(head_dim)
    var upto = Int(kv_len) - 1
    var row = head * Int(kv_len)
    var nsc = upto + 1
    var chunk = (nsc + TILES - 1) // TILES
    var t0 = tile * chunk
    var t1 = t0 + chunk
    if t1 > nsc:
        t1 = nsc

    var stot = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var e_s = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    # gmax（冗余只读；max 无舍入序自由）。
    var smax = stack_allocation[
        THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    var best = LOWEST_FP32
    var j = tid
    while j <= upto:
        var s = scores[unsafe_offset=row + j]
        if s > best:
            best = s
        j += THREADS
    smax[tid] = best
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            if smax[tid + stride] > smax[tid]:
                smax[tid] = smax[tid + stride]
        barrier()
        stride //= 2
    var gmax = smax[0]

    # 本 tile exp（smem 缓存：j ↔ tid = j - t0，chunk <= THREADS）+
    # tile 局部和（独立 scratch 树规约）。
    var tl = Float64(0)
    var jt = t0 + tid
    if jt < t1:
        var e = Float64(exp(Float64(scores[unsafe_offset=row + jt] - gmax)))
        e_s[unsafe_offset=jt - t0] = e
        tl += e
    stot[tid] = tl
    barrier()
    stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            stot[tid] += stot[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        tsum[unsafe_offset=head * TILES + tile] = stot[0]
    barrier()

    # V 部分和（tile 范围，组式；exp 复用 e_s 缓存）
    var s_g = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var hd = Int(head_dim)
    var groups = THREADS // hd
    var d = tid % hd
    var g = tid // hd
    var acc = Float64(0)
    var jj = t0 + g
    while jj < t1:
        acc += e_s[unsafe_offset=jj - t0] * Float64(
            v_store[unsafe_offset=jj * kv_cols + kv_head_base + d]
        )
        jj += groups
    s_g[tid] = acc
    barrier()
    if g == 0:
        var acc2 = s_g[tid]
        for g2 in range(1, groups):
            acc2 += s_g[tid + g2 * hd]
        vpart[unsafe_offset=(head * TILES + tile) * hd + d] = Float32(acc2)


def attn_comb2_gpu[TILES: Int = 8](
    dst: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tsum: Pointer[Float64, MutUntrackedOrigin],
    n_heads: Int32,
    head_dim: Int32,
) -> None:
    """K3'：tile 部分和与 tile 和双双合并，就地归一（gsum =
    Σ tile 和，树序重结合）。"""
    var head = Int(block_idx.x)
    var d = Int(thread_idx.x)
    var hd = Int(head_dim)
    if d < hd:
        var acc = Float64(0)
        var gsum = Float64(0)
        for t in range(TILES):
            acc += Float64(vpart[unsafe_offset=(head * TILES + t) * hd + d])
            gsum += tsum[unsafe_offset=head * TILES + t]
        dst[unsafe_offset=head * hd + d] = Float32(acc / gsum)




def attn_dec_gpu[THREADS: Int, TILES: Int, GROUP: Int, SKIP: Int = 0](
    q_src: Pointer[Float32, MutUntrackedOrigin],
    k_src: Pointer[Float32, MutUntrackedOrigin],
    v_src: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    v_store: Pointer[Float32, MutUntrackedOrigin],
    cos_row: Pointer[Float32, MutUntrackedOrigin],
    sin_row: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tsum: Pointer[Float64, MutUntrackedOrigin],
    tmax: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    kv_len: Int32,
) -> None:
    """decode 融合分组核（账本一百一十五-a）：三趟的 K1/K2 并一枚——
    rope(q/k)+v 拷贝+scores+softmax 部分和+V 部分和一次发射，仅
    comb2 归并另发。一百一十四-c 勘探：attention 段 0.5ms/步且不随
    kv_len 增长 ⇒ 发射+延迟主导；现成单发 attn_rope 14 块延迟链证伪。
    grid=(n_kv_heads, TILES)，块内 GROUP 个 q-head 槽共享 k/v（GQA
    k 流量 ÷GROUP）：scores 按 (key, 8 路 dim 分片) 布点 + 3 屏障树
    规约；softmax 用流式局部基准（tile 局部 max，合并重标度交给
    comb2——免全局 gmax 往返，数值差 ~1e-7，门容差覆盖）；V 相
    (head, dim) 布点串行 keys（chunk 随 TILES 减半）。vpart/tsum
    布局与 attn_comb2_gpu 完全一致（head = kv_head*GROUP + h），K3'
    直接复用。tile 0 独写 store 行；当前 token 的 k/v 各块自算（s_k
    入 smem / v_src 直读）免同 launch 跨块竞态（七十三解法）。要求
    head_dim==64、THREADS==512（SLOTS=8 ≥ GROUP=7）、kv_len ≤
    TILES*64。q_rot 全局不再写（fused 内无人读，省一路往返）。"""

    comptime HD = 64
    comptime HALF = HD // 2
    comptime SLOTS = THREADS // HD
    comptime KC = 8
    comptime CHMAX = THREADS // KC

    var kv_head = Int(block_idx.y)
    var tile = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var kv_head_base = kv_head * HD
    var kv_cols = Int(n_kv_heads) * HD
    var scale = Float32(1.0) / sqrt(Float32(HD))
    var upto = Int(kv_len) - 1
    var nsc = upto + 1
    var chunk = (nsc + TILES - 1) // TILES
    var j0 = tile * chunk
    var j1 = j0 + chunk
    if j1 > nsc:
        j1 = nsc
    var jn = j1 - j0

    var s_q = stack_allocation[
        SLOTS * HD, Float32, address_space = AddressSpace.SHARED
    ]()
    var s_k = stack_allocation[
        HD, Float32, address_space = AddressSpace.SHARED
    ]()
    var s_sc = stack_allocation[
        CHMAX * SLOTS * KC, Float32, address_space = AddressSpace.SHARED
    ]()
    var s_e = stack_allocation[
        SLOTS * HD, Float64, address_space = AddressSpace.SHARED
    ]()
    var s_redf = stack_allocation[
        THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    var s_red64 = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var s_m = stack_allocation[
        SLOTS, Float32, address_space = AddressSpace.SHARED
    ]()

    # 1) rope q → smem（每线程自维，配对直读免交换）；rope k_new → s_k
    #    （tile 0 落 store 行）；v 拷贝（tile 0）。
    if tid < GROUP * HD:
        var h = tid // HD
        var d = tid % HD
        var hb = (kv_head * GROUP + h) * HD
        var qv: Float32
        if d < HALF:
            var a = q_src[unsafe_offset=hb + d]
            var b = q_src[unsafe_offset=hb + d + HALF]
            qv = a * cos_row[unsafe_offset=d] - b * sin_row[unsafe_offset=d]
        else:
            var a = q_src[unsafe_offset=hb + d - HALF]
            var b = q_src[unsafe_offset=hb + d]
            qv = b * cos_row[unsafe_offset=d] + a * sin_row[unsafe_offset=d]
        s_q[unsafe_offset=h * HD + d] = qv
    if tid < HALF:
        var d = tid
        var a = k_src[unsafe_offset=kv_head_base + d]
        var b = k_src[unsafe_offset=kv_head_base + d + HALF]
        var r1 = a * cos_row[unsafe_offset=d] - b * sin_row[unsafe_offset=d]
        var r2 = b * cos_row[unsafe_offset=d + HALF] + a * sin_row[
            unsafe_offset=d + HALF
        ]
        s_k[unsafe_offset=d] = r1
        s_k[unsafe_offset=d + HALF] = r2
        if tile == 0:
            k_store[unsafe_offset=upto * kv_cols + kv_head_base + d] = r1
            k_store[unsafe_offset=upto * kv_cols + kv_head_base + d + HALF] = r2
    if tid < HD:
        if tile == 0:
            v_store[unsafe_offset=upto * kv_cols + kv_head_base + tid] = v_src[
                unsafe_offset=kv_head_base + tid
            ]
    barrier()

    # 2) scores：(key, 8 路 dim 分片) 布点，3 屏障树规约到 (jl, h, c=0)。
    var jl = tid // KC
    var c = tid % KC
    if jl < jn:
        var jj = j0 + jl
        var kv8: SIMD[DType.float32, 8]
        if jj == upto:
            kv8 = s_k.load[8](c * KC)
        else:
            kv8 = k_store.load[8](jj * kv_cols + kv_head_base + c * KC)
        for h in range(GROUP):
            var qv8 = s_q.load[8](h * HD + c * KC)
            var p = Float64(0)
            for dd in range(KC):
                p += Float64(qv8[dd]) * Float64(kv8[dd])
            s_sc[unsafe_offset=(jl * SLOTS + h) * KC + c] = Float32(p)
    barrier()
    var st = KC // 2
    while st > 0:
        if jl < jn and c < st:
            for h in range(GROUP):
                s_sc[unsafe_offset=(jl * SLOTS + h) * KC + c] += s_sc[
                    unsafe_offset=(jl * SLOTS + h) * KC + c + st
                ]
        barrier()
        st //= 2

    # 3) tile 局部 max（流式 softmax 基准）。
    if SKIP < 1 and tid < GROUP * HD:
        var h = tid // HD
        var u = tid % HD
        var m = LOWEST_FP32
        var uu = u
        while uu < jn:
            var sc = s_sc[unsafe_offset=(uu * SLOTS + h) * KC] * scale
            if sc > m:
                m = sc
            uu += HD
        s_redf[unsafe_offset=h * HD + u] = m
    barrier()
    var st2 = HALF
    while st2 > 0:
        if tid < GROUP * st2:
            var h = tid // st2
            var u = tid % st2
            if s_redf[unsafe_offset=h * HD + u + st2] > s_redf[
                unsafe_offset=h * HD + u
            ]:
                s_redf[unsafe_offset=h * HD + u] = s_redf[
                    unsafe_offset=h * HD + u + st2
                ]
        barrier()
        st2 //= 2
    if tid < GROUP:
        s_m[unsafe_offset=tid] = s_redf[unsafe_offset=tid * HD]
    barrier()

    # 4) exp（局部基准）+ tile 局部和 → tsum。
    var e = Float64(0)
    if SKIP < 2 and tid < GROUP * HD:
        var h = tid // HD
        var u = tid % HD
        if u < jn:
            e = exp(
                Float64(
                    s_sc[unsafe_offset=(u * SLOTS + h) * KC] * scale
                    - s_m[unsafe_offset=h]
                )
            )
            s_e[unsafe_offset=h * HD + u] = e
    s_red64[unsafe_offset=tid] = e
    barrier()
    st2 = CHMAX // 2
    while st2 > 0:
        if tid < GROUP * st2:
            var h = tid // st2
            var u = tid % st2
            s_red64[unsafe_offset=h * CHMAX + u] += s_red64[
                unsafe_offset=h * CHMAX + u + st2
            ]
        barrier()
        st2 //= 2
    if tid < GROUP:
        tsum[unsafe_offset=(kv_head * GROUP + tid) * TILES + tile] = s_red64[
            unsafe_offset=tid * CHMAX
        ]
    if tid < GROUP:
        tmax[unsafe_offset=(kv_head * GROUP + tid) * TILES + tile] = s_m[
            unsafe_offset=tid
        ]

    # 5) V 部分和：(head, dim) 布点，串行 tile keys（chunk 随 TILES 减半，
    #    见探针 TILES 扫描）；当前 token 直读 v_src 免竞态。
    var h5 = tid // HD
    var d5 = tid % HD
    if SKIP < 3 and h5 < GROUP:
        var acc = Float64(0)
        var uu = 0
        while uu < jn:
            var jj = j0 + uu
            var vv: Float32
            if jj == upto:
                vv = v_src[unsafe_offset=kv_head_base + d5]
            else:
                vv = v_store[unsafe_offset=jj * kv_cols + kv_head_base + d5]
            acc += s_e[unsafe_offset=h5 * HD + uu] * Float64(vv)
            uu += 1
        var head = kv_head * GROUP + h5
        vpart[unsafe_offset=(head * TILES + tile) * HD + d5] = Float32(acc)


def attn_comb3_gpu[TILES: Int](
    dst: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tsum: Pointer[Float64, MutUntrackedOrigin],
    tmax: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    head_dim: Int32,
) -> None:
    """K3''（账本一百一十五-a，流式 softmax 归并）：attn_dec_gpu 的
    tile 部分和各带局部 max 基准，合并须重标度——M = max m_t，
    w_t = exp(m_t - M)，dst = (Σ vpart_t·w_t)/(Σ tsum_t·w_t)。
    comb2 假设全局 gmax 基准一致，直接相加对流式部分和是错的
    （探针 maxdiff 0.04 的真凶）。"""

    var head = Int(block_idx.x)
    var d = Int(thread_idx.x)
    var hd = Int(head_dim)
    var s_w = stack_allocation[
        TILES, Float64, address_space = AddressSpace.SHARED
    ]()
    # w_t 每 head 全 d 共享——线程 0 算一次入 smem（原版每 d 重复
    # TILES 次 f64 exp，64× 冗余，f64 exp 软件模拟是计时的主嫌）。
    if d == 0:
        var m = LOWEST_FP32
        for t in range(TILES):
            var mt = tmax[unsafe_offset=head * TILES + t]
            if mt > m:
                m = mt
        for t in range(TILES):
            s_w[unsafe_offset=t] = exp(
                Float64(tmax[unsafe_offset=head * TILES + t] - m)
            )
    barrier()
    if d < hd:
        var acc = Float64(0)
        var gsum = Float64(0)
        for t in range(TILES):
            var w = s_w[unsafe_offset=t]
            acc += Float64(vpart[unsafe_offset=(head * TILES + t) * hd + d]) * w
            gsum += tsum[unsafe_offset=head * TILES + t] * w
        dst[unsafe_offset=head * hd + d] = Float32(acc / gsum)


def kv_store_b_gpu[THREADS: Int](
    q_rot: Pointer[Float32, MutUntrackedOrigin],
    q_src: Pointer[Float32, MutUntrackedOrigin],
    k_src: Pointer[Float32, MutUntrackedOrigin],
    v_src: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    v_store: Pointer[Float32, MutUntrackedOrigin],
    cos_tab: Pointer[Float32, MutUntrackedOrigin],
    sin_tab: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    pos0: Int32,
) -> None:
    """prefill KV 落位核（账本八十八）：rope q -> q_rot、rope k ->
    store 行 pos0+b、v 拷贝。grid=(n_heads, 1, B)。

    单独一发的原因：行 b 的 scores 要读行 pos0..pos0+b-1 的 store
    行 —— 同 launch 跨 block 写读是竞态（decode 无此问题：只读
    旧 launch 的行 + 当前行走 smem）。store 与 scores 分跨 launch
    边界，跨 kernel 可见无竞态。k rope 按 kv_head 分组冗余
    （group 个 head 块写同一行，同值幂等）。"""
    var b = Int(block_idx.z)
    var head = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var hd = Int(head_dim)
    var q_base = b * Int(n_heads) * hd + head * hd
    var kv_head_base = kv_head * hd
    var kv_cols = Int(n_kv_heads) * hd
    var upto = Int(pos0) + b
    var half = hd // 2
    var cos_row = cos_tab + (Int(pos0) + b) * hd
    var sin_row = sin_tab + (Int(pos0) + b) * hd
    if tid < half:
        var d = tid
        var a = q_src[unsafe_offset=q_base + d]
        var bb = q_src[unsafe_offset=q_base + d + half]
        q_rot[unsafe_offset=q_base + d] = a * cos_row[
            unsafe_offset=d
        ] - bb * sin_row[unsafe_offset=d]
        q_rot[unsafe_offset=q_base + d + half] = (
            bb * cos_row[unsafe_offset=d + half]
            + a * sin_row[unsafe_offset=d + half]
        )
        var ka = k_src[unsafe_offset=b * kv_cols + kv_head_base + d]
        var kb = k_src[unsafe_offset=b * kv_cols + kv_head_base + d + half]
        k_store[unsafe_offset=upto * kv_cols + kv_head_base + d] = (
            ka * cos_row[unsafe_offset=d] - kb * sin_row[unsafe_offset=d]
        )
        k_store[unsafe_offset=upto * kv_cols + kv_head_base + d + half] = (
            kb * cos_row[unsafe_offset=d + half]
            + ka * sin_row[unsafe_offset=d + half]
        )
    if tid < hd:
        v_store[unsafe_offset=upto * kv_cols + kv_head_base + tid] = (
            v_src[unsafe_offset=b * kv_cols + kv_head_base + tid]
        )


def attn_rs_b_gpu[
    THREADS: Int, VEC: Bool = True, TILES: Int = 8,
](
    q_rot: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    scores: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
    pos0: Int32,
) -> None:
    """K1 scores 相的 batch 版（账本八十八）：grid=(n_heads, TILES, B)。
    只算点积（rope/store 在 kv_store_b_gpu，上一 launch，跨 kernel
    可见）。scores 行 = (b*n_heads+head)*kv_len；因果界 upto=pos0+b，
    j==upto 直读 store 行（kv_store_b 已写定）。逐行算术与单 token
    K1 同构（同序同树）。"""
    var b = Int(block_idx.z)
    var head = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var hd = Int(head_dim)
    var q_base = b * Int(n_heads) * hd + head * hd
    var kv_head_base = kv_head * hd
    var kv_cols = Int(n_kv_heads) * hd
    var upto = Int(pos0) + b
    var row = (b * Int(n_heads) + head) * Int(kv_len)
    var nsc = upto + 1
    var chunk = (nsc + TILES - 1) // TILES
    var j0 = tile * chunk
    var j1 = j0 + chunk
    if j1 > nsc:
        j1 = nsc
    var scale = Float32(1.0) / sqrt(Float32(hd))

    var jj = j0 + tid
    while jj < j1:
        var acc = Float64(0)
        if jj < upto:
            comptime if VEC:
                var kbase = jj * kv_cols + kv_head_base
                var c8 = hd // 8
                for c in range(c8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += Float64(
                            q_rot[unsafe_offset=q_base + c * 8 + dd]
                        ) * Float64(Float32(kv8[dd]))
                for d in range(c8 * 8, hd):
                    acc += Float64(
                        q_rot[unsafe_offset=q_base + d]
                    ) * Float64(k_store[unsafe_offset=kbase + d])
        else:
            for d in range(hd):
                acc += Float64(q_rot[unsafe_offset=q_base + d]) * Float64(
                    k_store[unsafe_offset=upto * kv_cols + kv_head_base + d]
                )
        scores[unsafe_offset=row + jj] = Float32(acc) * scale
        jj += THREADS

def attn_vt2_b_gpu[
    THREADS: Int, TILES: Int = 8,
](
    scores: Pointer[Float32, MutUntrackedOrigin],
    v_store: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tsum: Pointer[Float64, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
    pos0: Int32,
) -> None:
    """K2' 的 batch 版（账本八十八）：grid=(n_heads, TILES, B)，行界与
    scores/vpart/tsum 偏移同 K1b 约定。"""
    var b = Int(block_idx.z)
    var head = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var kv_head_base = kv_head * Int(head_dim)
    var kv_cols = Int(n_kv_heads) * Int(head_dim)
    var upto = Int(pos0) + b
    var row = (b * Int(n_heads) + head) * Int(kv_len)
    var nsc = upto + 1
    var chunk = (nsc + TILES - 1) // TILES
    var t0 = tile * chunk
    var t1 = t0 + chunk
    if t1 > nsc:
        t1 = nsc

    var stot = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var e_s = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    # gmax（冗余只读；max 无舍入序自由）。
    var smax = stack_allocation[
        THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    var best = LOWEST_FP32
    var j = tid
    while j <= upto:
        var s = scores[unsafe_offset=row + j]
        if s > best:
            best = s
        j += THREADS
    smax[tid] = best
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            if smax[tid + stride] > smax[tid]:
                smax[tid] = smax[tid + stride]
        barrier()
        stride //= 2
    var gmax = smax[0]

    var tl = Float64(0)
    var jt = t0 + tid
    if jt < t1:
        var e = Float64(exp(Float64(scores[unsafe_offset=row + jt] - gmax)))
        e_s[unsafe_offset=jt - t0] = e
        tl += e
    stot[tid] = tl
    barrier()
    stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            stot[tid] += stot[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        tsum[unsafe_offset=(b * Int(n_heads) + head) * TILES + tile] = stot[0]
    barrier()

    var s_g = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var hd = Int(head_dim)
    var groups = THREADS // hd
    var d = tid % hd
    var g = tid // hd
    var acc = Float64(0)
    var jj = t0 + g
    while jj < t1:
        acc += e_s[unsafe_offset=jj - t0] * Float64(
            v_store[unsafe_offset=jj * kv_cols + kv_head_base + d]
        )
        jj += groups
    s_g[tid] = acc
    barrier()
    if g == 0:
        var acc2 = s_g[tid]
        for g2 in range(1, groups):
            acc2 += s_g[tid + g2 * hd]
        vpart[unsafe_offset=(
            (b * Int(n_heads) + head) * TILES + tile
        ) * hd + d] = Float32(acc2)


def attn_comb2_b_gpu[TILES: Int = 8](
    dst: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tsum: Pointer[Float64, MutUntrackedOrigin],
    n_heads: Int32,
    head_dim: Int32,
) -> None:
    """K3' 的 batch 版（账本八十八）：grid=(n_heads, 1, B)。"""
    var b = Int(block_idx.z)
    var head = Int(block_idx.x)
    var d = Int(thread_idx.x)
    var hd = Int(head_dim)
    if d < hd:
        var acc = Float64(0)
        var gsum = Float64(0)
        for t in range(TILES):
            acc += Float64(
                vpart[unsafe_offset=((b * Int(n_heads) + head) * TILES + t) * hd + d]
            )
            gsum += tsum[unsafe_offset=(b * Int(n_heads) + head) * TILES + t]
        dst[unsafe_offset=b * Int(n_heads) * hd + head * hd + d] = Float32(acc / gsum)


def attn_fused_b_gpu[
    THREADS: Int, VEC: Bool = True, TILES: Int = 1,
](
    q_rot: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    v_store: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tsum: Pointer[Float64, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
    pos0: Int32,
) -> None:
    """flash 融合（账本一百零二）：rs+vt2 单核（TILES=1 限定）——
    scores 算进 smem（f32 无损）+ gmax/e_s/V 同核完成。逐位同
    rs+vt2（算序同构，vpart/tsum maxdiff=0.0 已验）。
    ⚠️ 性能负结果（A/B 交替 30 rep，kv=512）：fused 532µs vs 分离
    482µs（-10%）——点积循环与 V 累加同函数 ⇒ 寄存器分配按峰值
    分配，占用率下降抵消费省（launch+显存往返仅 ~5µs）。保留作
    参照，勿接入 session；若要救活需 phase 级 noinline 或拆核。"""

    var b = Int(block_idx.z)
    var head = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var hd = Int(head_dim)
    var q_base = b * Int(n_heads) * hd + head * hd
    var kv_head_base = kv_head * hd
    var kv_cols = Int(n_kv_heads) * hd
    var upto = Int(pos0) + b
    var row = (b * Int(n_heads) + head) * Int(kv_len)
    var nsc = upto + 1
    var chunk = (nsc + TILES - 1) // TILES
    var j0 = tile * chunk
    var j1 = j0 + chunk
    if j1 > nsc:
        j1 = nsc
    var scale = Float32(1.0) / sqrt(Float32(hd))

    var s_s = stack_allocation[
        THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    var smax = stack_allocation[
        THREADS, Float32, address_space = AddressSpace.SHARED
    ]()

    # phase A：scores（同 rs 算序）+ 线程局部 max
    var best = LOWEST_FP32
    var jj = j0 + tid
    while jj < j1:
        var acc = Float64(0)
        if jj < upto:
            comptime if VEC:
                var kbase = jj * kv_cols + kv_head_base
                var c8 = hd // 8
                for c in range(c8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += Float64(
                            q_rot[unsafe_offset=q_base + c * 8 + dd]
                        ) * Float64(Float32(kv8[dd]))
                for d in range(c8 * 8, hd):
                    acc += Float64(
                        q_rot[unsafe_offset=q_base + d]
                    ) * Float64(k_store[unsafe_offset=kbase + d])
            if not VEC:
                for d in range(hd):
                    acc += Float64(
                        q_rot[unsafe_offset=q_base + d]
                    ) * Float64(k_store[unsafe_offset=jj * kv_cols + kv_head_base + d])
        else:
            for d in range(hd):
                acc += Float64(q_rot[unsafe_offset=q_base + d]) * Float64(
                    k_store[unsafe_offset=upto * kv_cols + kv_head_base + d]
                )
        var s = Float32(acc) * scale
        s_s[unsafe_offset=jj - j0] = s
        if s > best:
            best = s
        jj += THREADS
    smax[tid] = best
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            if smax[tid + stride] > smax[tid]:
                smax[tid] = smax[tid + stride]
        barrier()
        stride //= 2
    var gmax = smax[0]

    # phase B：e_s + tsum（同 vt2）
    var e_s = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var stot = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var tl = Float64(0)
    var jt = j0 + tid
    if jt < j1:
        var e = Float64(exp(Float64(s_s[unsafe_offset=jt - j0] - gmax)))
        e_s[unsafe_offset=jt - j0] = e
        tl += e
    stot[tid] = tl
    barrier()
    stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            stot[tid] += stot[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        tsum[unsafe_offset=(b * Int(n_heads) + head) * TILES + tile] = stot[0]
    barrier()

    # phase C：V 累加（同 vt2 phase3）
    var s_g = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var groups = THREADS // hd
    var d = tid % hd
    var g = tid // hd
    var acc = Float64(0)
    var jj2 = j0 + g
    while jj2 < j1:
        acc += e_s[unsafe_offset=jj2 - j0] * Float64(
            v_store[unsafe_offset=jj2 * kv_cols + kv_head_base + d]
        )
        jj2 += groups
    s_g[tid] = acc
    barrier()
    if g == 0:
        var acc2 = s_g[tid]
        for g2 in range(1, groups):
            acc2 += s_g[tid + g2 * hd]
        vpart[unsafe_offset=(
            (b * Int(n_heads) + head) * TILES + tile
        ) * hd + d] = Float32(acc2)


def attn_rs_b4_gpu[
    THREADS: Int, ROWS: Int = 4, VEC: Bool = True,
](
    q_rot: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    scores: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
    pos0: Int32,
    n_rows: Int32,
) -> None:
    """K1 scores 的「多行共块」版（账本一百零八）：grid=(n_heads, 1,
    ceil(t/ROWS))，块内 THREADS/ROWS 个线程服务一行。原 rs 在 t=128、
    THREADS=512 时只有 128 线程活跃（3/4 空转）；本版把 ROWS=4 行塞进
    一块，所有线程都干活。

    逐元素算序与 attn_rs_b_gpu **完全同构**（同一 j 仍由同一线程按
    8-wide chunk + f64 累加走完全部 hd 维），因此 scores 逐位相等
    （实测 maxdiff 0.0，0 个非零）。

    ⚠️ 性能负结果（t=128、kv=128、14 头，批量 20 连发）：rs 113µs vs
    rs4(4 行/块) 122µs vs rs4(2 行/块, 256 线程) 117µs——**无收益**。
    ⇒ RS 不是占用率受限（空转线程不花钱），而是受「每线程重复读 64 维
    q + 每 key 读 256B k」的访存模式限制（每层 ~116MB 冗余流量 ≒
    113µs @1TB/s）。真要提速需按 kv_head 分组（同组 7 个 head 共享一次
    k 读）或把 q 行进 smem/shfl 广播。保留作参照。"""

    comptime TPR = THREADS // ROWS
    var tid = Int(thread_idx.x)
    var r = tid // TPR
    var lane = tid % TPR
    var b = Int(block_idx.z) * ROWS + r
    if b >= Int(n_rows):
        return
    var head = Int(block_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var hd = Int(head_dim)
    var q_base = b * Int(n_heads) * hd + head * hd
    var kv_head_base = kv_head * hd
    var kv_cols = Int(n_kv_heads) * hd
    var upto = Int(pos0) + b
    var row = (b * Int(n_heads) + head) * Int(kv_len)
    var nsc = upto + 1
    var scale = Float32(1.0) / sqrt(Float32(hd))

    var jj = lane
    while jj < nsc:
        var acc = Float64(0)
        if jj < upto:
            comptime if VEC:
                var kbase = jj * kv_cols + kv_head_base
                var c8 = hd // 8
                for c in range(c8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += Float64(
                            q_rot[unsafe_offset=q_base + c * 8 + dd]
                        ) * Float64(Float32(kv8[dd]))
                for d in range(c8 * 8, hd):
                    acc += Float64(
                        q_rot[unsafe_offset=q_base + d]
                    ) * Float64(k_store[unsafe_offset=kbase + d])
        else:
            for d in range(hd):
                acc += Float64(q_rot[unsafe_offset=q_base + d]) * Float64(
                    k_store[unsafe_offset=upto * kv_cols + kv_head_base + d]
                )
        scores[unsafe_offset=row + jj] = Float32(acc) * scale
        jj += TPR


def attn_rs_f32_gpu[
    THREADS: Int, VEC: Bool = True,
](
    q_rot: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    scores: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
    pos0: Int32,
) -> None:
    """K1 scores 的 **f32 累加**版（账本一百零九）：与 attn_rs_b_gpu 逐
    指令同构，只把点积累加器从 Float64 换成 Float32（scores 本就存 f32，
    f64 只在累加内部）。grid/线程映射/因果界完全不变。

    动机：RS 144µs 对应 14.7M MAC（理论 ~5µs），差 30×——怀疑每 key
    64 次**依赖链 f64 FMA**（A100 f64 吞吐 1/2、f32→f64 转换 1/4）。
    f32 版是否翻正由 probe 判定；数值上仅累加精度差 ~1e-7 相对量级，
    远小于 f32 快路径自身 5e-3 的门容差。"""

    var b = Int(block_idx.z)
    var head = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var group = Int(n_heads) // Int(n_kv_heads)
    var kv_head = head // group
    var hd = Int(head_dim)
    var q_base = b * Int(n_heads) * hd + head * hd
    var kv_head_base = kv_head * hd
    var kv_cols = Int(n_kv_heads) * hd
    var upto = Int(pos0) + b
    var row = (b * Int(n_heads) + head) * Int(kv_len)
    var nsc = upto + 1
    var scale = Float32(1.0) / sqrt(Float32(hd))

    var jj = tid
    while jj < nsc:
        var acc = Float32(0)
        if jj < upto:
            comptime if VEC:
                var kbase = jj * kv_cols + kv_head_base
                var c8 = hd // 8
                for c in range(c8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    var q8 = q_rot.load[8](q_base + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += q8[dd] * Float32(kv8[dd])
                for d in range(c8 * 8, hd):
                    acc += q_rot[unsafe_offset=q_base + d] * k_store[
                        unsafe_offset=kbase + d
                    ]
        else:
            for d in range(hd):
                acc += q_rot[unsafe_offset=q_base + d] * k_store[
                    unsafe_offset=upto * kv_cols + kv_head_base + d
                ]
        scores[unsafe_offset=row + jj] = acc * scale
        jj += THREADS


def attn_rs_h_gpu[
    THREADS: Int, HMAX: Int = 8, VEC: Bool = True,
](
    q_rot: Pointer[Float32, MutUntrackedOrigin],
    k_store: Pointer[Float32, MutUntrackedOrigin],
    scores: Pointer[Float32, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
    pos0: Int32,
) -> None:
    """K1 scores 的 **kv_head 分组**版（账本一百零九）：grid =
    (n_kv_heads, 1, t)，一块服务同一 kv head 下的一组 query head
    （GQA 里 group = n_heads/n_kv_heads 个，本模型 = 7）。

    动机（nsys 实测）：原 rs 每层 79.9µs 对应 14.7M MAC，且 f32/f64 累加
    只差 4% ⇒ 不是算力問題，而是访存——原版每个 (head, b) 块都把 k 行
    整读一遍，同一 kv head 的 7 个 query head 重复读 7 次（每层 ~57MB
    冗余）。本版把这些 head 塞进同一块：**k 流量 ÷group**，且块数
    1792 → 256（更少块、更多活）。

    线程映射：h_off = tid % HMAX（HMAX ≥ group，多的空转），
    lane = tid / HMAX，键 jj = lane, lane+LPR, …。同一 (head, key) 的
    点积仍由单线程按 8-wide chunk + f64 累加走完全部 hd 维 ⇒ **算序
    与 attn_rs_b_gpu 完全一致，scores 逐位同**。无 barrier ⇒ 早退安全。"""

    comptime LPR = THREADS // HMAX
    var b = Int(block_idx.z)
    var kv_head = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var h_off = tid % HMAX
    var lane = tid // HMAX
    var group = Int(n_heads) // Int(n_kv_heads)
    if h_off >= group:
        return
    var head = kv_head * group + h_off
    var hd = Int(head_dim)
    var q_base = b * Int(n_heads) * hd + head * hd
    var kv_head_base = kv_head * hd
    var kv_cols = Int(n_kv_heads) * hd
    var upto = Int(pos0) + b
    var row = (b * Int(n_heads) + head) * Int(kv_len)
    var nsc = upto + 1
    var scale = Float32(1.0) / sqrt(Float32(hd))

    var jj = lane
    while jj < nsc:
        var acc = Float64(0)
        if jj < upto:
            comptime if VEC:
                var kbase = jj * kv_cols + kv_head_base
                var c8 = hd // 8
                for c in range(c8):
                    var kv8 = k_store.load[8](kbase + c * 8)
                    @parameter
                    for dd in range(8):
                        acc += Float64(
                            q_rot[unsafe_offset=q_base + c * 8 + dd]
                        ) * Float64(Float32(kv8[dd]))
                for d in range(c8 * 8, hd):
                    acc += Float64(
                        q_rot[unsafe_offset=q_base + d]
                    ) * Float64(k_store[unsafe_offset=kbase + d])
        else:
            for d in range(hd):
                acc += Float64(q_rot[unsafe_offset=q_base + d]) * Float64(
                    k_store[unsafe_offset=upto * kv_cols + kv_head_base + d]
                )
        scores[unsafe_offset=row + jj] = Float32(acc) * scale
        jj += LPR


def attn_vt2_h_gpu[
    THREADS: Int, HMAX: Int = 8, TILES: Int = 8, FUSED: Bool = False,
](
    scores: Pointer[Float32, MutUntrackedOrigin],
    v_store: Pointer[Float32, MutUntrackedOrigin],
    vpart: Pointer[Float32, MutUntrackedOrigin],
    tsum: Pointer[Float64, MutUntrackedOrigin],
    n_heads: Int32,
    n_kv_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
    pos0: Int32,
    dst: Pointer[Float32, MutUntrackedOrigin],
) -> None:
    """K2' 的 **kv_head 分组**版（账本一百一十一）：grid =
    (n_kv_heads, TILES, B)，一块服务同一 kv head 下的一组 query head
    （group = n_heads/n_kv_heads 个，本模型 7）。要求
    THREADS // HMAX == head_dim（每槽 64 线程恰对应 64 个输出维），
    且 group <= HMAX（多余槽空转但仍参与 barrier）。

    动机（nsys 实测）：VT2 45.5µs/层与 RS 同病——每个 (head,b,tile)
    块都把同一 kv head 的 v 行整读一遍，7 倍冗余（每层 ~57MB）。分组
    后同块 7 个槽读相同 v 行（L1 命中），DRAM 流量 ÷group。

    结构：每槽 = THREADS//HMAX = 64 线程 = 1 个 head。gmax 仍全行
    冗余只读（max 无舍入序自由，值与原版逐位同）；exp/sum 的树从
    512 宽改 64 宽、PV 从 8 组跨步改单线程整段累加 ⇒ **求和树形改变，
    结果 ~1e-7 级漂移（非逐位同），由门容差覆盖**。e_s 按槽分条
    [HMAX][THREADS]（f64，32KB），chunk 上限 THREADS（nsc ≤ 8*512）。"""

    comptime TPP = THREADS // HMAX
    var b = Int(block_idx.z)
    var kv_head = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var slot = tid // TPP
    var lane = tid % TPP
    var group = Int(n_heads) // Int(n_kv_heads)
    var active = slot < group
    var head = kv_head * group + slot
    var hd = Int(head_dim)
    var kv_head_base = kv_head * hd
    var kv_cols = Int(n_kv_heads) * hd
    var upto = Int(pos0) + b
    var row = (b * Int(n_heads) + head) * Int(kv_len)
    var nsc = upto + 1
    var chunk = (nsc + TILES - 1) // TILES
    var t0 = tile * chunk
    var t1 = t0 + chunk
    if t1 > nsc:
        t1 = nsc

    var smax = stack_allocation[
        THREADS, Float32, address_space = AddressSpace.SHARED
    ]()
    var stot = stack_allocation[
        THREADS, Float64, address_space = AddressSpace.SHARED
    ]()
    var e_s = stack_allocation[
        HMAX * THREADS, Float64, address_space = AddressSpace.SHARED
    ]()

    # ---- gmax（全行，槽内 64 宽树）----
    var best = LOWEST_FP32
    if active:
        var j = lane
        while j < nsc:
            var s = scores[unsafe_offset=row + j]
            if s > best:
                best = s
            j += TPP
    smax[tid] = best
    barrier()
    var stride = TPP // 2
    while stride > 0:
        if lane < stride:
            if smax[tid + stride] > smax[tid]:
                smax[tid] = smax[tid + stride]
        barrier()
        stride //= 2
    var gmax = smax[slot * TPP]

    # ---- exp + sum（本 tile 的 chunk）----
    var tl = Float64(0)
    if active:
        # 跨步覆盖整个 chunk（chunk 可达 THREADS，> TPP 时单遍会漏 key）
        var jt = t0 + lane
        while jt < t1:
            var e = Float64(exp(Float64(scores[unsafe_offset=row + jt] - gmax)))
            e_s[unsafe_offset=slot * THREADS + (jt - t0)] = e
            tl += e
            jt += TPP
    stot[tid] = tl
    barrier()
    stride = TPP // 2
    while stride > 0:
        if lane < stride:
            stot[tid] += stot[tid + stride]
        barrier()
        stride //= 2
    if active and lane == 0:
        tsum[unsafe_offset=(b * Int(n_heads) + head) * TILES + tile] = stot[
            slot * TPP
        ]

    # ---- PV（groups = TPP//hd = 1：每线程一维，整段累加）----
    # 一百一十二：TILES==1 时 chunk 即整行，acc/sum 本身就是 COMB 的
    # 全部工作——直接尾写 dst，省掉每层一发 COMB（13.6µs）。f64 acc
    # 不经 f32 vpart 中转，与 COMB 相差 ~1e-7（门内）。
    var d = lane
    var acc = Float64(0)
    if active:
        var jj = t0
        while jj < t1:
            acc += e_s[unsafe_offset=slot * THREADS + (jj - t0)] * Float64(
                v_store[unsafe_offset=jj * kv_cols + kv_head_base + d]
            )
            jj += 1
        comptime if FUSED and TILES == 1:
            dst[unsafe_offset=b * Int(n_heads) * hd + head * hd + d] = (
                Float32(acc / stot[slot * TPP])
            )
        else:
            vpart[unsafe_offset=(
                (b * Int(n_heads) + head) * TILES + tile
            ) * hd + d] = Float32(acc)
