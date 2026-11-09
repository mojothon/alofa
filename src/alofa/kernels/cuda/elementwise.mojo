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

from std.gpu import global_idx
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
