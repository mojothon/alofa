"""GPU 逐元素小算子 —— rope / swiglu / add / embedding gather / 连续拷贝。

接线设计（`docs/plan/06-device-context.md` §3.4）步骤 1 的五个小 kernel。全部
一线程一元素的平铺并行，与标量实现逐句对应：

- `add_gpu` / `copy_gpu` / `gather_gpu`：纯 f32 搬运与加法，判据**零容差**
  （任何差值都是读错内存）；
- `rope_gpu`：纯 f32 乘减/乘加。有一个已知风险要在门里看：`a*c - b*s` 是
  经典的 FMA 折叠模式，若 GPU 编译器折叠了它，差分门会以零容差红出来 ——
  那正是门想要的信息（linear 的 diff=0.0 说明点积路径没折叠，但这个模式
  没被测过）；
- `swiglu_gpu`：`Float64` 中间量 + exp，与标量同序；exp 已知 1 ulp
  （libdevice vs libm）⇒ 判据相对容差。
"""

from std.gpu import global_idx, block_idx, thread_idx

from alofa.kernels.cuda.basic import _fp16_bits_to_f32_device
from std.math import exp


def add_gpu(
    dst: Pointer[Float32, MutUntrackedOrigin],
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """`dst[i] = a[i] + b[i]`，残差路。与 `scalar.add` 同一个 f32 加法。"""
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    dst[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def copy_gpu(
    dst: Pointer[Float32, MutUntrackedOrigin],
    src: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """连续 f32 拷贝 —— KV cache 的 append（对应 host 的 `copy_into`）。"""
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    dst[unsafe_offset=i] = src[unsafe_offset=i]


def swiglu_gpu(
    dst: Pointer[Float32, MutUntrackedOrigin],
    gate: Pointer[Float32, MutUntrackedOrigin],
    up: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """`dst = silu(gate) * up`，与 `scalar.swiglu` 同序：f64 中间量、
    `v / (1 + exp(-v))`、乘 up、一次 `Float32` 收尾。"""
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var v = Float64(gate[unsafe_offset=i])
    dst[unsafe_offset=i] = Float32(
        v / (Float64(1) + exp(-v)) * Float64(up[unsafe_offset=i])
    )


def rope_gpu(
    dst: Pointer[Float32, MutUntrackedOrigin],
    src: Pointer[Float32, MutUntrackedOrigin],
    cos_table: Pointer[Float32, MutUntrackedOrigin],
    sin_table: Pointer[Float32, MutUntrackedOrigin],
    cols: Int32,
    head_dim: Int32,
    tokens: Int32,
) -> None:
    """半旋转 rope，与 `scalar.rope` 逐句同构；q 和 k 各启动一次（同一 kernel）。

    一线程一个 `(token, head, d)` 三元组（`d` 只走前半 `half`），从平铺
    `global_idx` 还原三维下标。算术与标量同一个 f32 表达式序；cos/sin 表按
    token 索引（`[tokens, head_dim]`），两半分开读 —— 与标量一样不假设表在
    两半间重复。
    """
    var i = Int(global_idx.x)
    var half = Int(head_dim) // 2
    var heads = Int(cols) // Int(head_dim)
    var n_pairs = Int(tokens) * heads * half
    if i >= n_pairs:
        return
    var d = i % half
    var head = (i // half) % heads
    var t = i // (half * heads)
    var table = t * Int(head_dim)
    var base = t * Int(cols) + head * Int(head_dim)
    var a = src[unsafe_offset=base + d]
    var b = src[unsafe_offset=base + d + half]
    dst[unsafe_offset=base + d] = a * cos_table[unsafe_offset=table + d] - b * sin_table[
        unsafe_offset=table + d
    ]
    dst[unsafe_offset=base + d + half] = b * cos_table[
        unsafe_offset=table + d + half
    ] + a * sin_table[unsafe_offset=table + d + half]


def gather_gpu(
    dst: Pointer[Float32, MutUntrackedOrigin],
    table: Pointer[Float32, MutUntrackedOrigin],
    ids: Pointer[Int32, MutUntrackedOrigin],
    cols: Int32,
    tokens: Int32,
) -> None:
    """embedding 行拷贝：`dst[t*cols+c] = table[ids[t]*cols+c]`。

    对应 `run` 里的 fp32 行拷贝（其文档原话：无算术可错，按索引查词表）；
    q8 embedding 路由不在这 —— 那是 dequant_q8_0 通路，已被门覆盖。
    """
    var i = Int(global_idx.x)
    if i >= Int(tokens) * Int(cols):
        return
    var t = i // Int(cols)
    var c = i % Int(cols)
    var token = Int(ids[unsafe_offset=t])
    dst[unsafe_offset=i] = table[unsafe_offset=token * Int(cols) + c]


def add_into_gpu(
    dst: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """`dst[i] = dst[i] + b[i]`，原地残差累加。

    与 `add_gpu` 的区别是目标就是源之一 —— 解_ALLOC aliasing 检查的约束
    （同一缓冲不能同时以两个 mutable 参数进 kernel 发射，账本六十三）；
    逐元素同下标读写，每线程先读后写自己那一份，语义与 `dst = a + b`
    （a 即 dst）完全一致。
    """
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    dst[unsafe_offset=i] = dst[unsafe_offset=i] + b[unsafe_offset=i]


def rope2_gpu(
    q_dst: Pointer[Float32, MutUntrackedOrigin],
    k_dst: Pointer[Float32, MutUntrackedOrigin],
    q_src: Pointer[Float32, MutUntrackedOrigin],
    k_src: Pointer[Float32, MutUntrackedOrigin],
    cos_table: Pointer[Float32, MutUntrackedOrigin],
    sin_table: Pointer[Float32, MutUntrackedOrigin],
    q_cols: Int32,
    k_cols: Int32,
    head_dim: Int32,
    tokens: Int32,
) -> None:
    """q 和 k 的 rope 合成一次发射（账本六十五：省 1 发射/层）。

    平铺索引跨两段：前 `tokens*q_heads*half` 个是 q，其余是 k。每元素的
    算术与 `rope_gpu`/标量逐句相同 —— 包括**表的两半分开读**（第二半的
    写回用 `table + d + half`，不假设表在两半间重复；初版漏了这一点，
    差分门抓出，账本六十五）。
    """
    var i = Int(global_idx.x)
    var half = Int(head_dim) // 2
    var q_heads = Int(q_cols) // Int(head_dim)
    var k_heads = Int(k_cols) // Int(head_dim)
    var q_pairs = Int(tokens) * q_heads * half
    var k_pairs = Int(tokens) * k_heads * half
    if i >= q_pairs + k_pairs:
        return
    # 注意：不要用「指针三元选择」写这个分支 —— 本版 Mojo 的 GPU 代码里
    # `ptr_a if cond else ptr_b` 误编译成静默不写（账本六十五：standalone
    # 复现，输出全零、无 fault）。两个分支各自完整展开。
    if i < q_pairs:
        var j = i
        var d = j % half
        var head = (j // half) % q_heads
        var t = j // (half * q_heads)
        var table = t * Int(head_dim)
        var base = t * Int(q_cols) + head * Int(head_dim)
        var a = q_src[unsafe_offset=base + d]
        var b = q_src[unsafe_offset=base + d + half]
        q_dst[unsafe_offset=base + d] = a * cos_table[
            unsafe_offset=table + d
        ] - b * sin_table[unsafe_offset=table + d]
        q_dst[unsafe_offset=base + d + half] = (
            b * cos_table[unsafe_offset=table + d + half]
            + a * sin_table[unsafe_offset=table + d + half]
        )
    else:
        var j = i - q_pairs
        var d = j % half
        var head = (j // half) % k_heads
        var t = j // (half * k_heads)
        var table = t * Int(head_dim)
        var base = t * Int(k_cols) + head * Int(head_dim)
        var a = k_src[unsafe_offset=base + d]
        var b = k_src[unsafe_offset=base + d + half]
        k_dst[unsafe_offset=base + d] = a * cos_table[
            unsafe_offset=table + d
        ] - b * sin_table[unsafe_offset=table + d]
        k_dst[unsafe_offset=base + d + half] = (
            b * cos_table[unsafe_offset=table + d + half]
            + a * sin_table[unsafe_offset=table + d + half]
        )


def copy2_gpu(
    k_dst: Pointer[Float32, MutUntrackedOrigin],
    v_dst: Pointer[Float32, MutUntrackedOrigin],
    k_src: Pointer[Float32, MutUntrackedOrigin],
    v_src: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """KV append 的 k/v 两笔拷贝合成一次发射（账本六十五：省 1 发射/层）。

    前 `n` 个元素走 k，其余走 v —— 与两次 `copy_gpu` 逐位一致。
    """
    var i = Int(global_idx.x)
    if i < Int(n):
        k_dst[unsafe_offset=i] = k_src[unsafe_offset=i]
    elif i < Int(n) * 2:
        var j = i - Int(n)
        v_dst[unsafe_offset=j] = v_src[unsafe_offset=j]


def rope_kv_gpu(
    q_rot: Pointer[Float32, MutUntrackedOrigin],
    q_src: Pointer[Float32, MutUntrackedOrigin],
    k_src: Pointer[Float32, MutUntrackedOrigin],
    v_src: Pointer[Float32, MutUntrackedOrigin],
    k_row: Pointer[Float32, MutUntrackedOrigin],
    v_row: Pointer[Float32, MutUntrackedOrigin],
    cos_row: Pointer[Float32, MutUntrackedOrigin],
    sin_row: Pointer[Float32, MutUntrackedOrigin],
    q_cols: Int32,
    kv_cols: Int32,
    head_dim: Int32,
    tokens: Int32,
) -> None:
    """rope(q) + rope(k 直写 KV store 行) + v 拷贝，三合一发射（账本七十）。

    decode 里 `d_k_rot` 只喂 copy2 —— rope 后的 k 直接写进本 token 的
    store 行（k_row/v_row 由调用方偏移到 pos 行），中转缓冲整个省掉。
    平铺索引三段：q rope（与 rope2 逐元素同算术，含表两半分开读）、
    k rope（同上，目标换 store 行）、v 拷贝。三段输出互不重叠，无竞态。
    """
    var i = Int(global_idx.x)
    var half = Int(head_dim) // 2
    var q_heads = Int(q_cols) // Int(head_dim)
    var k_heads = Int(kv_cols) // Int(head_dim)
    var q_pairs = Int(tokens) * q_heads * half
    var k_pairs = Int(tokens) * k_heads * half
    if i >= q_pairs + k_pairs + Int(tokens) * Int(kv_cols):
        return
    if i < q_pairs:
        var j = i
        var d = j % half
        var head = (j // half) % q_heads
        var t = j // (half * q_heads)
        var table = t * Int(head_dim)
        var base = t * Int(q_cols) + head * Int(head_dim)
        var a = q_src[unsafe_offset=base + d]
        var b = q_src[unsafe_offset=base + d + half]
        q_rot[unsafe_offset=base + d] = a * cos_row[
            unsafe_offset=table + d
        ] - b * sin_row[unsafe_offset=table + d]
        q_rot[unsafe_offset=base + d + half] = (
            b * cos_row[unsafe_offset=table + d + half]
            + a * sin_row[unsafe_offset=table + d + half]
        )
    elif i < q_pairs + k_pairs:
        var j = i - q_pairs
        var d = j % half
        var head = (j // half) % k_heads
        var t = j // (half * k_heads)
        var table = t * Int(head_dim)
        var base = t * Int(kv_cols) + head * Int(head_dim)
        var a = k_src[unsafe_offset=base + d]
        var b = k_src[unsafe_offset=base + d + half]
        k_row[unsafe_offset=base + d] = a * cos_row[
            unsafe_offset=table + d
        ] - b * sin_row[unsafe_offset=table + d]
        k_row[unsafe_offset=base + d + half] = (
            b * cos_row[unsafe_offset=table + d + half]
            + a * sin_row[unsafe_offset=table + d + half]
        )
    else:
        var j = i - q_pairs - k_pairs
        v_row[unsafe_offset=j] = v_src[unsafe_offset=j]


def q4_dequant_f16_gpu(
    blocks: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Float16, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """q4_0 块流 -> f16 权重（账本六十八，装载期用）。

    一线程一权重：块 b = i//32，块内 idx = i%32；字节 = 块基址 +2 +
    (idx%16)，idx<16 取低半字节、否则取高半字节（与 CPU dequant_q4_0 的
    布局一致）；scale 是块头的 fp16 位模式。数值 = (nib-8)*d 精确 f32
    乘法后舍入进 f16 —— 与 host 侧 dequant + Float16() 舍入逐位相同。
    """
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var b = i // 32
    var idx = i % 32
    var base = b * 18
    var low = Int(blocks[unsafe_offset=base])
    var high = Int(blocks[unsafe_offset=base + 1])
    var d = _fp16_bits_to_f32_device(low | (high << 8))
    # 不用三元选择（六十五：GPU 代码里三元误编译前科，int 版同族风险）
    var nib = 0
    if idx < 16:
        nib = Int(blocks[unsafe_offset=base + 2 + idx]) & 0x0F
    else:
        nib = (Int(blocks[unsafe_offset=base + 2 + idx - 16]) >> 4) & 0x0F
    dst[unsafe_offset=i] = Float16(Float32(nib - 8) * d)


def f32_to_f16_gpu(
    src: Pointer[Float32, MutUntrackedOrigin],
    dst: Pointer[Float16, MutUntrackedOrigin],
    n: Int32,
) -> None:
    """f32 -> f16 逐元素（账本六十八，lm_head 权重装载期用）。"""
    var i = Int(global_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = Float16(src[unsafe_offset=i])
