"""Qwen2 forward, in one file, with one code path for both phases.

Prefill (a whole prompt at once) and decode (one token at a time, reading the
key-value store) share everything here. They differ only in how many rows the
views have and in where the rotary table starts — which is the point: two code
paths would be two things to fix when they disagree, and a disagreement
between them is one of the classic ways a decode path silently diverges.

The parameters are whatever `scripts/dump_model_reference.py` wrote, named
exactly as Hugging Face names them (`model.layers.0.self_attn.q_proj.weight`
and so on). No renaming, no re-laying-out, no transpose on load: a parameter's
name in the file is its name in the reference, so a name typo fails loudly at
construction rather than producing a plausible wrong answer.

Buffers are allocated once, at construction, and the per-call views are just
different row counts over the same memory. A forward pass allocates nothing.

Run:
    pixi run mojo build -O2 -I src tests/unit/test_model_parity.mojo -o /tmp/mp
"""

from std.math import cos, pow, sin
from std.runtime.asyncrt import TaskGroup, parallelism_level

from alofa.core.dtype import DT_FP32
from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    ERR_PARSE,
    ERR_SHAPE_MISMATCH,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import (
    MAX_RANK,
    F32Ptr,
    Shape,
    TensorView,
    f32_data,
    rows_view,
    shape2,
)
from alofa.core.text import parse_float64, parse_int
from alofa.kernels.cpu.quant import (
    Q4_BLOCK,
    Q4_BYTES,
    matmul_q4_f32,
    matmul_q4_f32_bias,
    quantize_q4_0,
)

from alofa.kernels.cpu.avx2 import (
    matmul_q4_f32 as matmul_q4_f32_vec,
    matmul_q4_f32_bias as matmul_q4_f32_bias_vec,
)
from alofa.kernels.cpu.avx2 import (
    add as add_vec,
    attention as attention_vec,
    linear as linear_vec,
    linear_bias as linear_bias_vec,
    rmsnorm as rmsnorm_vec,
    rope as rope_vec,
    swiglu as swiglu_vec,
)
from alofa.kernels.cpu.scalar import (
    add,
    attention,
    linear,
    linear_bias,
    rmsnorm,
    rope,
    swiglu,
)
from alofa.model.config import json_bool, json_float, json_has, json_int
from alofa.model.loader import TensorFile, config_value

comptime EMBED = "model.embed_tokens.weight"
comptime FINAL_NORM = "model.norm.weight"
comptime OUTPUT = "lm_head.weight"
comptime COS = "rope_cos"
comptime SIN = "rope_sin"

# 量化槽位。一层的七个投影矩阵按前向里的顺序编号，输出投影单独占最后一格；
# 索引由层号直接算出，运行期不做任何字符串查找。
comptime Q4_SLOT_Q = 0
comptime Q4_SLOT_K = 1
comptime Q4_SLOT_V = 2
comptime Q4_SLOT_O = 3
comptime Q4_SLOT_GATE = 4
comptime Q4_SLOT_UP = 5
comptime Q4_SLOT_DOWN = 6
comptime Q4_SLOTS_PER_LAYER = 7


# 输出投影没有量化槽位（`Q4_NO_SLOT`）—— 见 `enable_q4` 里的说明。
comptime Q4_NO_SLOT = -1


# --- CPU 后端 ---------------------------------------------------------------
#
# 标量后端与向量后端发的是**同一份语义**：同样的内存布局、同样的算子契约，
# 区别只是"用哪份实现"。所以它适合做成**编译期**参数而不是运行期字段 ——
# 运行期可改的后端意味着每个算子前面多一次分支，也意味着"这次到底跑的是哪个
# 后端"变成了一个要在日志里回答的问题。
#
# ⚠️ 不是所有算子都有向量版本：`rope` 与 `attention` 本轮只有标量实现，整网
# 跑在 `BACKEND_AVX2` 上时它们仍然走标量。这一点写在 `backend_label` 的文档与
# 账本里，不假装覆盖。（量化通路 2026-09-20 起**有**向量实现了，见
# `q4_matmul_k`；那条注释曾在三处说过它「只有标量」，现已一并改掉。）
#
# ⚠️ 后端是**方法**的参数，不是结构体的参数。不是设计偏好，是被编译器逼的：
# Mojo 1.0.0（ed45d567）在"参数化结构体 + 会抛错误的构造函数"上会直接把编译
# 器进程搞崩（`var m = QwenForward[BACKEND_AVX2](...)` 这一行就够），最小复现
# 见 `tests/fixtures/bad_backend.mojo` 的注释。放在方法上（`prefill[backend]`
# / `step[backend]`）语义其实更准：后端是**这一次前向**的属性，不是模型实例
# 的属性 —— 权重和 KV 缓存都不关心后端是什么。
comptime BACKEND_SCALAR = 0
comptime BACKEND_AVX2 = 1


def uses_vector_backend[backend: Int]() -> Bool:
    """`backend` 是不是向量后端。

    未知取值必须在编译期就炸掉：写成 `comptime if backend == BACKEND_AVX2`
    时，一个拼错的常量会**静默退化成标量后端** —— 门照样绿，但它绿的原因是
    根本没跑向量路径。
    """
    comptime assert (
        backend == BACKEND_SCALAR or backend == BACKEND_AVX2
    ), "unknown cpu backend"
    return backend == BACKEND_AVX2


def backend_label[backend: Int]() -> String:
    """后端的名字（`"scalar"` / `"avx2"`），供门断言"跑的是哪个后端"。

    它读的是**与算子分发同一个条件**（`uses_vector_backend`），所以"名字写着
    avx2、实际发的是标量"需要把同一个判断改两遍才做得到 —— 两条路径共用一次
    判断，比各写一遍更难说谎。

    ⚠️ 它说明的是"这一次前向用了哪套算子实现"，**不等于**"每个算子都向量化
    了"：`rope` 与 `attention` 仍然只有标量实现，向量后端里它们走标量。
    （量化通路曾经也在这个名单里，2026-09-20 起已随 `q4_matmul_k` 分后端。）
    """
    comptime if uses_vector_backend[backend]():
        return "avx2"
    else:
        return "scalar"


def rmsnorm_k[backend: Int](
    dst: TensorView, x: TensorView, w: TensorView, eps: Float32
) raises AlofaError:
    """按 `backend` 分发的 RMSNorm。"""
    comptime if uses_vector_backend[backend]():
        rmsnorm_vec(dst, x, w, eps)
    else:
        rmsnorm(dst, x, w, eps)


def linear_k[backend: Int](dst: TensorView, x: TensorView, w: TensorView) raises AlofaError:
    """按 `backend` 分发的投影（无偏置）。"""
    comptime if uses_vector_backend[backend]():
        linear_vec(dst, x, w)
    else:
        linear(dst, x, w)


def linear_bias_k[backend: Int](
    dst: TensorView, x: TensorView, w: TensorView, b: TensorView
) raises AlofaError:
    """按 `backend` 分发的投影（带偏置）。"""
    comptime if uses_vector_backend[backend]():
        linear_bias_vec(dst, x, w, b)
    else:
        linear_bias(dst, x, w, b)


def q4_matmul_k[backend: Int](
    dst: F32Ptr, x: F32Ptr, blocks: RawPtr, rows: Int, cols: Int
) raises AlofaError:
    """按 `backend` 分发的 q4_0 融合投影（无偏置）。

    两条实现的契约与数值完全一致（向量版对同一份 fixture 的九条用例与标量版同为
    逐位一致，见 `tests/unit/test_q4_matmul_vec.mojo`），差别只是速度。
    """
    comptime if uses_vector_backend[backend]():
        matmul_q4_f32_vec(dst, x, blocks, rows, cols)
    else:
        matmul_q4_f32(dst, x, blocks, rows, cols)


def q4_matmul_bias_k[backend: Int](
    dst: F32Ptr, x: F32Ptr, blocks: RawPtr, rows: Int, cols: Int, bias: F32Ptr
) raises AlofaError:
    """按 `backend` 分发的 q4_0 融合投影（带偏置）。"""
    comptime if uses_vector_backend[backend]():
        matmul_q4_f32_bias_vec(dst, x, blocks, rows, cols, bias)
    else:
        matmul_q4_f32_bias(dst, x, blocks, rows, cols, bias)


# ---------------------------------------------------------------------------
# 把一次投影切成几片并发跑
#
# 一次 decode 的输出是 `out` 个标量，第 c 个是 `w` 的第 c 行与 `x` 的点积 —— 它们
# 之间**没有任何依赖**，这就是全部的并行度。切 `inner`（那个归约维）会引入跨片
# 归约（原子加，或再一趟合并），那是**额外**的同步，不是这里要拿的东西。
#
# ⚠️ Mojo 1.0.0 没有线程模块：`thread` / `threading` / `concurrent` / `parallel`
# 逐个 import 过，全是 `unable to locate module`。可用的只有 `runtime.asyncrt`
# 这层协程 + `TaskGroup`，`parallelism_level()` 报的就是核数。所以下面量到的
# 「并发收益」里**含这套运行时的调度开销** —— 它若把收益吃掉，那也是结论。
#
# ⚠️ 分片视图是**栈上的局部量**，协程按值捕获（`TensorView` 是 Copyable），所以
# 这里一次堆分配都没有。这不是洁癖：一次 decode 有 **169 次**投影（24 层 × 7 +
# 输出投影），每片一次 malloc 就把调度开销从 µs 级抬到十 µs 级。
# ---------------------------------------------------------------------------

# 片数的上限。超过核数再加片只会让最后几片排队，还多付调度。
comptime MAX_SHARDS = 16

# 实测过的最多片数。`default_shards()` 取它作上限：只启用量过的那一档。
comptime SHARDS_MEASURED = 8

# prefill（批 > 1）实测过的最佳片数与量过的行数上界。
#
# 为什么 prefill 要单独一档：`prefill` 按**输出行**切片，而 `avx2._gemm` 的权重
# 复用只在**块内**发生（`_gemm_tile[RB]`，`RB` ∈ 8/4/2/1）。于是"片数"在这里是
# 一个真旋钮，两个方向相反：
#
#     片数少 → 每片行数多 → 块内复用充分 → 权重被读的遍数少
#     片数多 → 线程并行度高 → 聚合带宽高
#
# 实测（`scripts/bench_prefill.mojo`，Qwen2.5-0.5B fp32 / avx2，每 token 时间，
# 3 趟 min…max，轮外层片数内层交错，**三次运行**）：
#
#     n = 8    1 片 35.5–49.8 / 2 片 24.0–28.9 / 4 片 19.8–25.3 / 8 片 32.3–67.6
#     n = 16   1 片 33.2–39.1 / 2 片 17.8–20.5 / 4 片 13.7–17.1 / 8 片 22.8–37.8
#     n = 32   1 片 32.3–38.0 / 2 片 17.6–38.2 / 4 片 10.4–16.0 / 8 片 14.2–20.7
#
# → **4 片在三档上都不是最差、在 16/32 上最好**；1 片最差（并发仍然要）。
#   即在这个 8 核机器上，**复用比并行度更值钱**。
#
# **更长的 prompt（同脚本，n = 32/48/64/96/128，三次运行）** —— 找 4 片与 8 片的
# **换手点**。按"权重被读的遍数 = 片数 × ceil(每片行数 / 8)"，n ≥ 48 时四档的遍数
# 就拉平了（都是 8 遍），所以换手点应该在 32 与 64 之间：
#
#     n = 32   4 片 9.0–16.1 / 10.3–11.3 / 11.2–17.7  8 片 9.7–12.7 / 11.6–11.8 / 10.8–15.9
#     n = 48   4 片 10.8–11.6 / 11.6–12.0 / 11.0–11.4  8 片 11.6–11.8 / 15.9–21.2 / 11.6–15.6
#     n = 64   4 片 8.8–15.2 / 9.2–13.0 / 8.5–11.6     8 片 7.9–8.8 / 10.0–14.4 / 8.0–9.3
#     n = 96   4 片 8.6–13.6 / 10.6–18.5 / 8.9–10.9    8 片 9.4–11.2 / 11.2–15.8 / 8.1–10.3
#     n = 128  4 片 9.6–10.7 / 9.5–11.0 / 9.0–14.3     8 片 10.0–10.8 / 10.1–12.0 / 7.8–9.1
#
# → **n = 48 上 4 片确定更好**（2/3 次区间不重合：12.01 < 15.88、11.37 < 11.58；
#   第三次也同向，只差 0.04 ms 没分开）。**n ≥ 64 量不出结论**（区间全都重合，
#   只有 n=64 那一次 8 片不重合地更好 → 不跨运行复现）。
#
# ⚠️ 换手点**随机器负载挪**：n = 32 这一档在负载低时（loadavg 5.6–7.0）三次里只有
#   一次分开，在负载高时（loadavg 8–12，上一轮）三次里两次分开 —— 8 片在机器忙时
#   掉得更多。**4 片从来没被量成更差**，这是选它的第二个理由。
#
# ⚠️ `n > 48` **量不出差别** → 退回调用方给的片数（与加这个上限之前逐位相同），
#    不假装量过。这与 `SHARDS_MEASURED` 是同一个规矩：只启用量过的那一档。
comptime PREFILL_SHARDS_MEASURED = 4
comptime PREFILL_ROWS_MEASURED = 48


def prefill_shards(rows: Int, shards: Int) -> Int:
    """prefill 这次用几片。

    只在量过的行数范围内（`rows <= PREFILL_ROWS_MEASURED`）把片数压到实测最好的
    那一档；超出就原样返回 —— 那里没量过，改动前的样子就是最不坏的选择。

    ⚠️ 它只改**并行度**，不改任何一次浮点运算：每个输出各自一个累加器，切分只
    决定"谁算哪几行"。故片数不同的 prefill 必须**逐位相等**（`bench_prefill.mojo`
    的自检与 `tests/unit/test_parallel_shards.mojo` 都在守这条）。
    """
    if rows > PREFILL_ROWS_MEASURED:
        return shards
    if shards <= PREFILL_SHARDS_MEASURED:
        return shards
    return PREFILL_SHARDS_MEASURED


def shard_count(n_out: Int, shards: Int) -> Int:
    """这次投影实际切成几片。

    片数不许超过输出个数：多出来的片是空的，可**空片也要付一次调度**。
    `shards <= 1` 一律退化成 1 片 —— 那条路就是原来的直呼，语义与性能都不变。
    """
    if shards <= 1:
        return 1
    var n = shards
    if n > n_out:
        n = n_out
    if n > MAX_SHARDS:
        n = MAX_SHARDS
    return n


def shard_shape(a: Int, b: Int) -> Shape:
    """二维 fp32 视图的形状，且**不抛错**。

    协程里用不了 `shape2`（它抛错，而 `TaskGroup.create_task` 只收不抛错的协
    程），所以分片视图的形状在这里现造。代价是它只支持二维 —— 一次投影本来也
    只有二维。
    """
    var dims = InlineArray[Int, MAX_RANK](fill=1)
    dims[0] = a
    dims[1] = b
    return Shape(dims, 2)


# ⚠️ 协程的参数只能是**平凡值**（指针与整数）。这不是风格问题：把 `TensorView`
# 直接传进协程，实测会**静默写到别处去** —— 参数槽在 `wait()` 之前就失效了，而
# 每片算的又是同一个值，于是"对不对"看起来像随机的（2026-09-20 复现：把分片视
# 图绑成循环外的具名变量就对，绑在循环里就错）。所以分片视图一律在协程**内**
# 用指针和整数现造。
async def _linear_shard[backend: Int](
    d: RawPtr,
    x: RawPtr,
    w: RawPtr,
    b: RawPtr,
    rows: Int,
    cols: Int,
    inner: Int,
    has_bias: Bool,
):
    """一片输出上的投影（带不带偏置由 `has_bias` 决定）。

    ⚠️ 参数不能叫 `out`：它是参数传递约定关键字，写进参数表会被当成 `out` 约定
    解析（"expected argument name"），故这里用 `cols`。

    ⚠️ `linear_k` 是 `raises` 的，而 `TaskGroup.create_task` 只收**不抛错**的协程
    （`RaisingCoroutine` 传不进去）。分片只是把同一次计算切成几段，段内的形状
    规则与全量那次**一模一样**，契约由调用方在进协程之前用同一次直呼验过 ——
    这里的 `except` 不是"吞掉错误"。
    """
    var dv = TensorView(d, shard_shape(rows, cols), DT_FP32)
    var xv = TensorView(x, shard_shape(rows, inner), DT_FP32)
    var wv = TensorView(w, shard_shape(cols, inner), DT_FP32)
    var bv = TensorView(b, shard_shape(1, cols), DT_FP32)
    try:
        if has_bias:
            linear_bias_k[backend](dv, xv, wv, bv)
        else:
            linear_k[backend](dv, xv, wv)
    except err:
        _ = err


async def _q4_shard[backend: Int](
    d: F32Ptr,
    x: F32Ptr,
    blocks: RawPtr,
    rows: Int,
    cols: Int,
    bias: F32Ptr,
    has_bias: Bool,
):
    """同上，q4_0 通路：一片输出行，配上它在块流里那一段。"""
    try:
        if has_bias:
            q4_matmul_bias_k[backend](d, x, blocks, rows, cols, bias)
        else:
            q4_matmul_k[backend](d, x, blocks, rows, cols)
    except err:
        _ = err


def linear_k_shards[backend: Int](
    dst: TensorView, x: TensorView, w: TensorView, shards: Int
) raises AlofaError:
    """把投影按输出切成 `shards` 片并发跑。

    批为 1（decode）时切**输出列**：`w` 的每片是一段连续的行，`dst` 的每片是一
    段连续的元素。批大于 1（prefill）时列不连续（`dst` 是行主序），于是改切
    **输出行**：`dst` 与 `x` 各自切一段连续的行，`w` 共享。
    """
    if dst.shape.rank() != 2 or x.shape.rank() != 2 or w.shape.rank() != 2:
        linear_k[backend](dst, x, w)
        return
    var rows = dst.shape.dims[0]
    var out = dst.shape.dims[1]
    var inner = x.shape.dims[1]
    var split = out if rows == 1 else rows
    # decode（批 = 1）切输出列，片数越多聚合带宽越高；prefill 切输出行，片数
    # 越多反而把权重读得越碎 —— 那一档实测过，见 `prefill_shards()`。
    var want = shards if rows == 1 else prefill_shards(rows, shards)
    var n = shard_count(split, want)
    if n <= 1:
        linear_k[backend](dst, x, w)
        return

    var d0 = dst.data.unsafe_offset(dst.byte_offset)
    var x0 = x.data.unsafe_offset(x.byte_offset)
    var w0 = w.data.unsafe_offset(w.byte_offset)
    var tg = TaskGroup()
    var per = split // n
    var rem = split - per * n
    var c0 = 0
    for k in range(n):
        var cnt = per + (rem if k == n - 1 else 0)
        if rows == 1:
            tg.create_task(
                _linear_shard[backend](
                    d0.unsafe_offset(c0 * 4),
                    x0,
                    w0.unsafe_offset(c0 * inner * 4),
                    d0,
                    1,
                    cnt,
                    inner,
                    False,
                )
            )
        else:
            tg.create_task(
                _linear_shard[backend](
                    d0.unsafe_offset(c0 * out * 4),
                    x0.unsafe_offset(c0 * inner * 4),
                    w0,
                    d0,
                    cnt,
                    out,
                    inner,
                    False,
                )
            )
        c0 += cnt
    tg.wait()


def linear_bias_k_shards[backend: Int](
    dst: TensorView, x: TensorView, w: TensorView, b: TensorView, shards: Int
) raises AlofaError:
    """同上，带偏置。

    偏置跟着输出列走：切列时每片只取自己那几列的偏置，切行时整份偏置每片都要
    （一行里的每一列都要加它自己那个偏置）。
    """
    if dst.shape.rank() != 2 or x.shape.rank() != 2 or w.shape.rank() != 2:
        linear_bias_k[backend](dst, x, w, b)
        return
    var rows = dst.shape.dims[0]
    var out = dst.shape.dims[1]
    var inner = x.shape.dims[1]
    var split = out if rows == 1 else rows
    var want = shards if rows == 1 else prefill_shards(rows, shards)
    var n = shard_count(split, want)
    if n <= 1:
        linear_bias_k[backend](dst, x, w, b)
        return

    var d0 = dst.data.unsafe_offset(dst.byte_offset)
    var x0 = x.data.unsafe_offset(x.byte_offset)
    var w0 = w.data.unsafe_offset(w.byte_offset)
    var b0 = b.data.unsafe_offset(b.byte_offset)
    var tg = TaskGroup()
    var per = split // n
    var rem = split - per * n
    var c0 = 0
    for k in range(n):
        var cnt = per + (rem if k == n - 1 else 0)
        if rows == 1:
            tg.create_task(
                _linear_shard[backend](
                    d0.unsafe_offset(c0 * 4),
                    x0,
                    w0.unsafe_offset(c0 * inner * 4),
                    b0.unsafe_offset(c0 * 4),
                    1,
                    cnt,
                    inner,
                    True,
                )
            )
        else:
            tg.create_task(
                _linear_shard[backend](
                    d0.unsafe_offset(c0 * out * 4),
                    x0.unsafe_offset(c0 * inner * 4),
                    w0,
                    b0,
                    cnt,
                    out,
                    inner,
                    True,
                )
            )
        c0 += cnt
    tg.wait()


def q4_matmul_k_shards[backend: Int](
    dst: F32Ptr, x: F32Ptr, blocks: RawPtr, rows: Int, cols: Int, shards: Int
) raises AlofaError:
    """同上，q4_0 通路：按输出行切，`blocks` 里每行是 `cols/Q4_BLOCK` 个块。"""
    var n = shard_count(rows, shards)
    if n <= 1:
        q4_matmul_k[backend](dst, x, blocks, rows, cols)
        return
    var row_bytes = cols // Q4_BLOCK * Q4_BYTES
    var tg = TaskGroup()
    var per = rows // n
    var rem = rows - per * n
    var c0 = 0
    for k in range(n):
        var cnt = per + (rem if k == n - 1 else 0)
        # 偏置指针在 has_bias=False 时不会被读，这里传 `dst` 只是占位。
        tg.create_task(
            _q4_shard[backend](
                dst.unsafe_offset(c0),
                x,
                blocks.unsafe_offset(c0 * row_bytes),
                cnt,
                cols,
                dst,
                False,
            )
        )
        c0 += cnt
    tg.wait()


def q4_matmul_bias_k_shards[backend: Int](
    dst: F32Ptr,
    x: F32Ptr,
    blocks: RawPtr,
    rows: Int,
    cols: Int,
    bias: F32Ptr,
    shards: Int,
) raises AlofaError:
    """同上，带偏置。"""
    var n = shard_count(rows, shards)
    if n <= 1:
        q4_matmul_bias_k[backend](dst, x, blocks, rows, cols, bias)
        return
    var row_bytes = cols // Q4_BLOCK * Q4_BYTES
    var tg = TaskGroup()
    var per = rows // n
    var rem = rows - per * n
    var c0 = 0
    for k in range(n):
        var cnt = per + (rem if k == n - 1 else 0)
        tg.create_task(
            _q4_shard[backend](
                dst.unsafe_offset(c0),
                x,
                blocks.unsafe_offset(c0 * row_bytes),
                cnt,
                cols,
                bias.unsafe_offset(c0),
                True,
            )
        )
        c0 += cnt
    tg.wait()


def default_shards() -> Int:
    """默认的片数：核数，但**不超过 8**。

    8 是本机（8 核）实测到最好的那一档：`scripts/bench_model_shards.mojo` 给
    `shards=8` **1.66–1.84×**（批 = 1 的单流 decode，fp32/avx2，两次运行复现）。
    ⚠️ 超过 8 的片数**没有量过**，所以它不是一个"越多越好"的证据：这里取 8 是
    "只启用量过的那一档"。

    ⚠️ 批大于 1 时分片改切**输出行**，那条路**没量过**（同上，只量了批 = 1）。
    批越大，串行段（注意力 / RMSNorm / RoPE / 残差）占比越高，加速只会更小。
    """
    var n = parallelism_level()
    if n < 1:
        return 1
    if n > SHARDS_MEASURED:
        return SHARDS_MEASURED
    return n


def add_k[backend: Int](dst: TensorView, a: TensorView, b: TensorView) raises AlofaError:
    """按 `backend` 分发的逐元素加。"""
    comptime if uses_vector_backend[backend]():
        add_vec(dst, a, b)
    else:
        add(dst, a, b)


def swiglu_k[backend: Int](
    dst: TensorView, gate: TensorView, up: TensorView
) raises AlofaError:
    """按 `backend` 分发的 SwiGLU。"""
    comptime if uses_vector_backend[backend]():
        swiglu_vec(dst, gate, up)
    else:
        swiglu(dst, gate, up)


def rope_k[backend: Int](
    out_q: TensorView,
    out_k: TensorView,
    q: TensorView,
    k: TensorView,
    cos: TensorView,
    sin: TensorView,
    head_dim: Int,
) raises AlofaError:
    """按 `backend` 分发的旋转位置编码。"""
    comptime if uses_vector_backend[backend]():
        rope_vec(out_q, out_k, q, k, cos, sin, head_dim)
    else:
        rope(out_q, out_k, q, k, cos, sin, head_dim)


def attention_k[backend: Int](
    dst: TensorView,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    scores: TensorView,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
) raises AlofaError:
    """按 `backend` 分发的注意力。"""
    comptime if uses_vector_backend[backend]():
        attention_vec(dst, q, k, v, scores, n_heads, n_kv_heads, head_dim)
    else:
        attention(dst, q, k, v, scores, n_heads, n_kv_heads, head_dim)


def rope_table(
    dst: RawPtr,
    rows: Int,
    head_dim: Int,
    theta: Float64,
    want_cos: Bool,
) raises AlofaError:
    """Fill a `[rows, head_dim]` rotary table from the configuration's `theta`.

    A checkpoint does not contain this table — it is a function of `rope_theta`
    and the head dimension, and the reference builds it at run time. So it is
    derived here too: `inv_freq[i] = theta ** (-2i / head_dim)` over the first
    half of a head, repeated across the second half, one row per position.

    The frequency and the angle are kept in fp32 because that is what the
    reference computes them in, and a table that disagreed with the reference's
    by an fp32 epsilon at position 0 disagrees by more than an epsilon at
    position 500. `cos` / `sin` are taken in fp64 and rounded once.

    `rows` is the stream's length, not the checkpoint's `max_position_embeddings`:
    positions beyond the table are refused by `run` before they are looked up.
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise AlofaError(
            ERR_UNSUPPORTED,
            "head dimension must be positive and even",
            "head_dim=" + String(head_dim),
        )
    if rows <= 0:
        raise AlofaError(
            ERR_OUT_OF_RANGE, "rotary table needs at least one row", "rows=" + String(rows)
        )
    var half = head_dim // 2
    var out = dst.unsafe_bitcast[Float32]()
    for pos in range(rows):
        var row = pos * head_dim
        for i in range(half):
            var inverse = Float32(1.0) / Float32(
                pow(theta, Float64(2 * i) / Float64(head_dim))
            )
            var angle = Float32(pos) * inverse
            var value = (
                Float32(cos(Float64(angle)))
                if want_cos
                else Float32(sin(Float64(angle)))
            )
            # Both halves of a head carry the same value: the repetition is how
            # the reference lays the table out, and `rope` indexes both halves
            # rather than assuming they match (see its docstring).
            out[unsafe_offset=row + i] = value
            out[unsafe_offset=row + i + half] = value


def q4_slot_count(n_layers: Int) -> Int:
    return n_layers * Q4_SLOTS_PER_LAYER


def q4_weight_name(layer: Int, slot: Int) raises AlofaError -> String:
    """第 `layer` 层第 `slot` 号槽位的参数名，顺序与前向里的调用顺序一致。

    写成模块级函数而不是方法，是因为构造函数里就要用它 —— 那时 `self` 的字段
    还没全部初始化，Mojo 不允许在那一刻调用 `self` 上的方法。
    """
    if layer < 0:
        raise AlofaError(
            ERR_OUT_OF_RANGE, "layer is out of range", "layer=" + String(layer)
        )
    var p = "model.layers." + String(layer) + "."
    if slot == Q4_SLOT_Q:
        return p + "self_attn.q_proj.weight"
    if slot == Q4_SLOT_K:
        return p + "self_attn.k_proj.weight"
    if slot == Q4_SLOT_V:
        return p + "self_attn.v_proj.weight"
    if slot == Q4_SLOT_O:
        return p + "self_attn.o_proj.weight"
    if slot == Q4_SLOT_GATE:
        return p + "mlp.gate_proj.weight"
    if slot == Q4_SLOT_UP:
        return p + "mlp.up_proj.weight"
    if slot == Q4_SLOT_DOWN:
        return p + "mlp.down_proj.weight"
    raise AlofaError(ERR_OUT_OF_RANGE, "no such slot", "slot=" + String(slot))


struct QwenConfig(Movable):
    """The architecture as numbers, read from a `key <tab> value` file."""

    var n_layers: Int
    var hidden: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var intermediate: Int
    var vocab: Int
    var eps: Float32
    var rope_theta: Float64
    var tied_output: Bool

    def __init__(out self, config_path: String) raises AlofaError:
        var is_json = config_path.byte_length() >= 5
        if is_json:
            var cb = config_path.as_bytes()
            var suffix = ".json".as_bytes()
            var start = len(cb) - len(suffix)
            for i in range(len(suffix)):
                if cb[start + i] != suffix[i]:
                    is_json = False
        if is_json:
            self.n_layers = json_int(config_path, "num_hidden_layers")
            self.hidden = json_int(config_path, "hidden_size")
            self.n_heads = json_int(config_path, "num_attention_heads")
            self.n_kv_heads = json_int(config_path, "num_key_value_heads")
            # Qwen2.5's `config.json` has no `head_dim` — the reference derives
            # it, and `validate` below is what catches a hidden size that the
            # head count does not divide.
            if json_has(config_path, "head_dim"):
                self.head_dim = json_int(config_path, "head_dim")
            else:
                self.head_dim = self.hidden // self.n_heads
            self.intermediate = json_int(config_path, "intermediate_size")
            self.vocab = json_int(config_path, "vocab_size")
            self.eps = Float32(json_float(config_path, "rms_norm_eps"))
            self.rope_theta = json_float(config_path, "rope_theta")
            self.tied_output = json_bool(config_path, "tie_word_embeddings")
        else:
            self.n_layers = parse_int(config_value(config_path, "n_layers"))
            self.hidden = parse_int(config_value(config_path, "hidden"))
            self.n_heads = parse_int(config_value(config_path, "n_heads"))
            self.n_kv_heads = parse_int(config_value(config_path, "n_kv_heads"))
            self.head_dim = parse_int(config_value(config_path, "head_dim"))
            self.intermediate = parse_int(config_value(config_path, "intermediate"))
            self.vocab = parse_int(config_value(config_path, "vocab"))
            self.eps = Float32(parse_float64(config_value(config_path, "eps")))
            self.rope_theta = parse_float64(config_value(config_path, "rope_theta"))
            self.tied_output = parse_int(
                config_value(config_path, "tie_word_embeddings")
            ) != 0
        self.validate(config_path)

    def validate(imm self, where: String) raises AlofaError:
        """Reject a configuration that cannot be a Qwen2, with the reason.

        Checked at construction and not again per token: the numbers below are
        the assumptions the forward pass is written against, and a silent
        violation would surface as a shape error several layers in.
        """
        if self.n_heads * self.head_dim != self.hidden:
            raise AlofaError(
                ERR_PARSE,
                "heads * head_dim must equal hidden",
                "config=" + where,
            )
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise AlofaError(
                ERR_PARSE,
                "n_kv_heads must be positive and divide n_heads",
                "config=" + where,
            )
        if self.n_layers <= 0 or self.intermediate <= 0 or self.vocab <= 0:
            raise AlofaError(
                ERR_PARSE, "layer, feed-forward and vocabulary sized must be positive", "config=" + where
            )
        if self.eps <= 0:
            raise AlofaError(ERR_PARSE, "rms norm eps must be positive", "config=" + where)

    def kv_dim(imm self) -> Int:
        """Channels in a key or value row."""
        return self.n_kv_heads * self.head_dim


struct QwenForward(Movable):
    """Parameters, buffers and the key-value store of one model instance.

    Not thread-safe and not reentrant: one instance is one generation stream,
    and `kv_len` is where it is in that stream. Concurrency is the runtime
    layer's problem, above this one.

    后端（`BACKEND_SCALAR` / `BACKEND_AVX2`）不是实例的属性，而是 `prefill` /
    `step` / `run` 这些方法的参数 —— 见上面那段说明。`rope` 与 `attention`
    目前只有标量实现。
    """

    var cfg: QwenConfig
    var params: TensorFile
    # The output projection's name. Tied embeddings (`tie_word_embeddings`)
    # mean the checkpoint has no `lm_head.weight` — the reference reads the
    # embedding matrix for both — so the name is resolved once, at load, and
    # is not re-derived for every token.
    var head: String
    var cos: TensorView
    var sin: TensorView
    # Owns the rotary tables when the parameter file does not carry them.
    var rope_arena: Arena
    var kv_arena: Arena
    var act_arena: Arena
    var k_store: List[RawPtr]
    var v_store: List[RawPtr]
    var hidden_ptr: RawPtr
    var normed_ptr: RawPtr
    var q_ptr: RawPtr
    var k_ptr: RawPtr
    var v_ptr: RawPtr
    var q_rot_ptr: RawPtr
    var k_rot_ptr: RawPtr
    var attn_ptr: RawPtr
    var proj_ptr: RawPtr
    var gate_ptr: RawPtr
    var up_ptr: RawPtr
    var act_ptr: RawPtr
    var scores_ptr: RawPtr
    var logits_ptr: RawPtr
    var kv_len: Int
    var max_tokens: Int
    # 量化后的投影矩阵：每个槽位一段 q4_0 块流，索引直接由层号算出，运行期不
    # 做任何名字查找。空列表 = 走 fp32 通路。
    var q4_arena: Arena
    var q4_blocks: List[RawPtr]
    var q4_enabled: Bool
    # 一次投影切成几片并发跑。1 = 不切（与加这个字段之前完全同一条代码路径）；
    # 默认是核数（上限 8），见 `default_shards()`。
    # 它是**实例**的属性而不是方法的参数：一次前向里有 169 次投影，让调用方每
    # 次都把这个数传一遍，只是给了 169 个把它传错的机会。
    var shards: Int

    def __init__(
        out self,
        params_dir: String,
        config_path: String,
        max_tokens: Int,
        quantize: Bool = False,
    ) raises AlofaError:
        if max_tokens <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "max_tokens must be positive",
                "max_tokens=" + String(max_tokens),
            )
        self.cfg = QwenConfig(config_path)
        self.params = TensorFile(params_dir)
        self.max_tokens = max_tokens
        self.kv_len = 0
        # 默认按核数切（`default_shards()`，上限是实测过的 8）。端到端实测见
        # `scripts/bench_model_shards.mojo`：批 = 1 的单流 decode 快 1.66–1.84×。
        # 想退回不切就显式 `set_shards(1)` —— 那条路与分片之前逐位相同。
        self.shards = default_shards()

        # A checkpoint that ties its embeddings ships no `lm_head.weight`; one
        # that does not tie them and still lacks it is a different failure, and
        # saying which is which is the whole point of this branch.
        if self.params.has(OUTPUT):
            self.head = OUTPUT
        elif self.cfg.tied_output:
            self.head = EMBED
        else:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "output projection is missing and embeddings are not tied",
                "name=" + OUTPUT,
            )

        # The rotary table is a function of `rope_theta`, not a parameter, so a
        # real checkpoint directory does not contain it. Our own exports do
        # carry the dumped one; either is read, and only one of them is used.
        var table_rows = max_tokens
        var table_cols = self.cfg.head_dim
        var derives_table = not (self.params.has(COS) and self.params.has(SIN))
        self.rope_arena = Arena(
            table_rows * table_cols * 2 * 4 if derives_table else 1
        )
        if derives_table:
            var cos_raw = self.rope_arena.alloc(table_rows * table_cols * 4)
            var sin_raw = self.rope_arena.alloc(table_rows * table_cols * 4)
            rope_table(cos_raw, table_rows, table_cols, self.cfg.rope_theta, True)
            rope_table(sin_raw, table_rows, table_cols, self.cfg.rope_theta, False)
            var table_dims = List[Int]()
            table_dims.append(table_rows)
            table_dims.append(table_cols)
            self.cos = TensorView(cos_raw, Shape(table_dims), DT_FP32)
            self.sin = TensorView(sin_raw, Shape(table_dims), DT_FP32)
        else:
            self.cos = self.params.view(COS)
            self.sin = self.params.view(SIN)

        var kv_dim = self.cfg.kv_dim()
        var store_bytes = max_tokens * kv_dim * 4
        # The store is one allocation per layer per side: a generation reads
        # rows `0..kv_len` of it, so contiguity across positions is what lets
        # attention see the whole history as one view.
        self.kv_arena = Arena(store_bytes * self.cfg.n_layers * 2)
        self.k_store = List[RawPtr]()
        self.v_store = List[RawPtr]()
        for _ in range(self.cfg.n_layers):
            self.k_store.append(self.kv_arena.alloc(store_bytes))
            self.v_store.append(self.kv_arena.alloc(store_bytes))

        var h = self.cfg.hidden
        var inter = self.cfg.intermediate
        # 六条 hidden 宽的缓冲（残差、两个归一化输出、q、旋转后的 q、注意力与
        # 投影输出）+ 三条 kv 宽 + 三条前馈宽。少算一条就是构造时的容量错误，
        # 而不是运行时的越界，所以这里写死成显式倍数而不是"差不多"。
        var act_bytes = max_tokens * (h * 6 + inter * 3 + self.cfg.kv_dim() * 3) * 4
        var logits_bytes = self.cfg.vocab * 4
        self.act_arena = Arena(act_bytes + logits_bytes + max_tokens * max_tokens * 4)
        self.hidden_ptr = self.act_arena.alloc(max_tokens * h * 4)
        self.normed_ptr = self.act_arena.alloc(max_tokens * h * 4)
        self.q_ptr = self.act_arena.alloc(max_tokens * h * 4)
        self.q_rot_ptr = self.act_arena.alloc(max_tokens * h * 4)
        self.attn_ptr = self.act_arena.alloc(max_tokens * h * 4)
        self.proj_ptr = self.act_arena.alloc(max_tokens * h * 4)
        self.k_ptr = self.act_arena.alloc(max_tokens * kv_dim * 4)
        self.k_rot_ptr = self.act_arena.alloc(max_tokens * kv_dim * 4)
        self.v_ptr = self.act_arena.alloc(max_tokens * kv_dim * 4)
        self.gate_ptr = self.act_arena.alloc(max_tokens * inter * 4)
        self.up_ptr = self.act_arena.alloc(max_tokens * inter * 4)
        self.act_ptr = self.act_arena.alloc(max_tokens * inter * 4)
        self.scores_ptr = self.act_arena.alloc(max_tokens * max_tokens * 4)
        self.logits_ptr = self.act_arena.alloc(logits_bytes)

        # 量化块的空间按张量**逐个**估算，不留"差不多"：容量算少了会在最后一
        # 层崩，那时已经跑了二十几层。
        self.q4_enabled = False
        self.q4_blocks = List[RawPtr]()
        var q4_bytes = 0
        for layer in range(self.cfg.n_layers):
            for slot in range(Q4_SLOTS_PER_LAYER):
                var n = self.params.numel(q4_weight_name(layer, slot))
                if n % Q4_BLOCK != 0:
                    raise AlofaError(
                        ERR_SHAPE_MISMATCH,
                        "a quantised matrix must be a whole number of blocks",
                        "name=" + q4_weight_name(layer, slot) + " numel=" + String(n),
                    )
                q4_bytes += n // Q4_BLOCK * Q4_BYTES
        self.q4_arena = Arena(q4_bytes + q4_slot_count(self.cfg.n_layers) * 64)
        if quantize:
            self.enable_q4()

        # Every parameter the forward pass names is checked now. A missing
        # parameter discovered on the last layer of a long generation is far
        # harder to debug than one discovered before anything runs.
        self.require(EMBED)
        self.require(FINAL_NORM)
        # `OUTPUT` is not required here on purpose: a tied checkpoint has no
        # `lm_head.weight`, and `head` already names what the output projection
        # is (`self.require(self.head)` would ask the same question twice).
        for layer in range(self.cfg.n_layers):
            var p = "model.layers." + String(layer) + "."
            self.require(p + "input_layernorm.weight")
            self.require(p + "post_attention_layernorm.weight")
            self.require(p + "self_attn.q_proj.weight")
            self.require(p + "self_attn.q_proj.bias")
            self.require(p + "self_attn.k_proj.weight")
            self.require(p + "self_attn.k_proj.bias")
            self.require(p + "self_attn.v_proj.weight")
            self.require(p + "self_attn.v_proj.bias")
            self.require(p + "self_attn.o_proj.weight")
            self.require(p + "mlp.gate_proj.weight")
            self.require(p + "mlp.up_proj.weight")
            self.require(p + "mlp.down_proj.weight")

    def set_shards(mut self, shards: Int) raises AlofaError:
        """设置一次投影切成几片并发跑。`shards <= 1` 表示不切。

        为什么是显式设置而不是让它默认等于核数：这是一台 **8 核常年 runq 6–27**
        的机器，"核数"在这里不等于"我能用多少核"。默认值保持 1（与加这个字段
        之前完全同一条代码路径），要看并发的收益必须由调用方**明说**。
        """
        if shards < 0 or shards > MAX_SHARDS:
            raise AlofaError(
                ERR_OUT_OF_RANGE,
                "shard count must be between 0 and the maximum",
                "shards=" + String(shards) + " max=" + String(MAX_SHARDS),
            )
        self.shards = shards

    def enable_q4(mut self) raises AlofaError:
        """把所有投影矩阵就地压成 q4_0，之后的前向走量化通路。

        ⚠️ fp32 原始权重**仍然映射着**：这样一来同一进程里可以直接拿 fp32 的结
        果当参照，但常驻内存会是 fp32 + q4 两份 —— 内存门量的是默认通路，不是
        这条。要省内存得先丢掉原始映射，那是加载器的事，本轮没做。
        """
        for layer in range(self.cfg.n_layers):
            for slot in range(Q4_SLOTS_PER_LAYER):
                self.quantize_one(layer * Q4_SLOTS_PER_LAYER + slot)
        # 输出投影（`lm_head`，与 embedding 绑定）**不量化**，这不是省事：
        # 本轮量过，连同它一起压成 4 bit，教师强制贪心一致率只有 0.77 —— 词表
        # 那一维的 151936 个 logits 全靠这一步的精度决定谁最大，4 bit 在这里
        # 省下的内存（约 76 MB，占 27%）远不值那个损失。
        self.q4_enabled = True

    def quantize_one(mut self, index: Int) raises AlofaError:
        var name = q4_weight_name(
            index // Q4_SLOTS_PER_LAYER, index % Q4_SLOTS_PER_LAYER
        )
        var n = self.params.numel(name)
        var blocks = self.q4_arena.alloc(n // Q4_BLOCK * Q4_BYTES)
        quantize_q4_0(blocks, self.params.ptr(name), n)
        self.q4_blocks.append(blocks)

    def project[backend: Int = BACKEND_SCALAR](
        mut self,
        dst: TensorView,
        x: TensorView,
        name: String,
        slot: Int,
        bias: TensorView,
        has_bias: Bool,
    ) raises AlofaError:
        """一次投影：走 q4_0 块流，或者走 fp32 通路。

        `name` 与 `slot` 是同一件事的两种写法（名字给 fp32 通路查表，槽位给量
        化通路算索引），两条通路必须指向同一个矩阵 —— 这是为了让**切换通路不改
        变语义**这件事在调用点就看得见，而不是藏在两个分支里。

        ⚠️ 只有 fp32 通路分后端；量化通路（`matmul_q4_f32`）本轮只有标量实现，
        `backend` 参数在量化通路上不起作用，这一点没写在调用点上 —— 因为"量化
        通路还有没有向量实现"是一个会在下一轮变的事实，而签名不该跟着翻烧饼。
        （2026-09-20：它变了。现在两条通路都按 `backend` 分发。）
        """
        if not self.q4_enabled or slot == Q4_NO_SLOT:
            if has_bias:
                linear_bias_k_shards[backend](
                    dst, x, self.params.view(name), bias, self.shards
                )
            else:
                linear_k_shards[backend](dst, x, self.params.view(name), self.shards)
            return

        var t_rows = dst.shape.dims[0]
        var out = dst.shape.dims[1]
        var cols = x.shape.dims[1]
        if t_rows != x.shape.dims[0]:
            raise AlofaError(
                ERR_SHAPE_MISMATCH,
                "a projection consumes one input row per output row",
                "name=" + name,
            )
        if t_rows <= 0 or out <= 0 or cols <= 0:
            raise AlofaError(
                ERR_SHAPE_MISMATCH,
                "a projection needs positive dimensions",
                "name=" + name,
            )
        var blocks = self.q4_blocks[slot]
        # 量化核做的是**矩阵乘向量**（一次给一个输出行），这里要的是一整个
        # token 批次，于是逐行喂进去。不在核里塞二维支持，是因为核的那份契约
        # （以及它对着的 fixture）就是一维的；为了省一个循环把两处契约一起改
        # 掉，换来的只会是"两边都说不清自己在算什么"。
        var dst_p = f32_data(dst)
        var x_p = f32_data(x)
        for r in range(t_rows):
            var out_row = dst_p.unsafe_offset(r * out * 4)
            var in_row = x_p.unsafe_offset(r * cols * 4)
            if has_bias:
                q4_matmul_bias_k_shards[backend](
                    out_row, in_row, blocks, out, cols, f32_data(bias), self.shards
                )
            else:
                q4_matmul_k_shards[backend](out_row, in_row, blocks, out, cols, self.shards)

    def require(imm self, name: String) raises AlofaError:
        """Fail with the parameter's name if it is not in the file."""
        if not self.params.has(name):
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "parameter is missing", "name=" + name
            )

    def keep_alive(self):
        """State that the arenas and the mapping are still in use here.

        Every pointer this struct hands out (logits, and any view a caller
        builds from it) is untracked, and Mojo ends a value's lifetime at its
        last use — see `Arena.keep_alive` and `MappedFile.keep_alive`.
        """
        self.kv_arena.keep_alive()
        self.act_arena.keep_alive()
        self.rope_arena.keep_alive()
        self.params.keep_alive()

    def reset(mut self):
        """Forget the stored history; the next call starts a new stream."""
        self.kv_len = 0

    def prefill[backend: Int = BACKEND_SCALAR](
        mut self, ids: List[Int]
    ) raises AlofaError -> F32Ptr:
        """Run a whole prompt and return the logits for its last position.

        The stored history is assumed to be empty, so this is the start of a
        stream; `step` continues it.

        `backend` 选这一趟前向用哪套算子实现，默认是标量；`BACKEND_AVX2` 走向量
        后端（`rope` / `attention` 仍然只有标量实现）。
        """
        if self.kv_len != 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "prefill must start from an empty history; call reset first",
                "kv_len=" + String(self.kv_len),
            )
        return self.run[backend](ids)

    def step[backend: Int = BACKEND_SCALAR](
        mut self, token: Int
    ) raises AlofaError -> F32Ptr:
        """Consume one token and return the logits for the next one."""
        var ids = List[Int]()
        ids.append(token)
        return self.run[backend](ids)

    def run[backend: Int = BACKEND_SCALAR](
        mut self, new_ids: List[Int]
    ) raises AlofaError -> F32Ptr:
        """The one forward path, over `new_ids`, extending the stored history.

        Steps, in the order the reference performs them: gather embeddings,
        then per block — input norm, q/k/v projections, rotary, append to the
        store, attention over the whole store, output projection, residual,
        feed-forward norm, gate and up projections, activation, down
        projection, residual — then the final norm and the output projection on
        the last row only.
        """
        var t = len(new_ids)
        if t == 0:
            raise AlofaError(ERR_INVALID_ARGUMENT, "no tokens given", "")
        var pos0 = self.kv_len
        var kv_len = pos0 + t
        if kv_len > self.max_tokens:
            raise AlofaError(
                ERR_OUT_OF_RANGE,
                "history would exceed the allocated length",
                "kv_len=" + String(kv_len) + " max=" + String(self.max_tokens),
            )

        var h = self.cfg.hidden
        var kv_dim = self.cfg.kv_dim()
        var hidden = rows_view(self.hidden_ptr, t, h)
        var normed = rows_view(self.normed_ptr, t, h)

        # Embedding gather. A row copy rather than a kernel: there is no
        # arithmetic to get wrong, and the vocabulary is looked up by index.
        var embed = self.params.view(EMBED)
        if embed.shape.dims[1] != h:
            raise AlofaError(
                ERR_SHAPE_MISMATCH,
                "embedding width does not match hidden",
                "name=" + EMBED,
            )
        var pe = f32_data(embed)
        var ph = f32_data(hidden)
        for i in range(t):
            var token = new_ids[i]
            if token < 0 or token >= self.cfg.vocab:
                raise AlofaError(
                    ERR_OUT_OF_RANGE,
                    "token id is outside the vocabulary",
                    "id=" + String(token),
                )
            var src = token * h
            var dst = i * h
            for j in range(h):
                ph[unsafe_offset=dst + j] = pe[unsafe_offset=src + j]

        var table_cos = self.cos.slice_dim(0, pos0, t)
        var table_sin = self.sin.slice_dim(0, pos0, t)

        for layer in range(self.cfg.n_layers):
            var p = "model.layers." + String(layer) + "."
            var q = rows_view(self.q_ptr, t, h)
            var k = rows_view(self.k_ptr, t, kv_dim)
            var v = rows_view(self.v_ptr, t, kv_dim)
            var q_rot = rows_view(self.q_rot_ptr, t, h)
            var k_rot = rows_view(self.k_rot_ptr, t, kv_dim)
            var attn_out = rows_view(self.attn_ptr, t, h)
            var proj = rows_view(self.proj_ptr, t, h)

            rmsnorm_k[backend](normed, hidden, self.params.view(p + "input_layernorm.weight"), self.cfg.eps)
            self.project[backend](
                q,
                normed,
                p + "self_attn.q_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_Q,
                self.params.view(p + "self_attn.q_proj.bias"),
                True,
            )
            self.project[backend](
                k,
                normed,
                p + "self_attn.k_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_K,
                self.params.view(p + "self_attn.k_proj.bias"),
                True,
            )
            self.project[backend](
                v,
                normed,
                p + "self_attn.v_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_V,
                self.params.view(p + "self_attn.v_proj.bias"),
                True,
            )
            rope_k[backend](q_rot, k_rot, q, k, table_cos, table_sin, self.cfg.head_dim)

            # Append this step's keys and values, then attend over everything
            # stored so far — the store is what makes the two phases the same
            # computation at different row counts.
            copy_into(self.k_store[layer], pos0 * kv_dim, k_rot, t * kv_dim)
            copy_into(self.v_store[layer], pos0 * kv_dim, v, t * kv_dim)

            attention_k[backend](
                attn_out,
                q_rot,
                rows_view(self.k_store[layer], kv_len, kv_dim),
                rows_view(self.v_store[layer], kv_len, kv_dim),
                rows_view(self.scores_ptr, t, kv_len),
                self.cfg.n_heads,
                self.cfg.n_kv_heads,
                self.cfg.head_dim,
            )
            self.project[backend](
                proj,
                attn_out,
                p + "self_attn.o_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_O,
                proj,
                False,
            )
            add_k[backend](hidden, hidden, proj)

            rmsnorm_k[backend](normed, hidden, self.params.view(p + "post_attention_layernorm.weight"), self.cfg.eps)
            var gate = rows_view(self.gate_ptr, t, self.cfg.intermediate)
            var up = rows_view(self.up_ptr, t, self.cfg.intermediate)
            var activated = rows_view(self.act_ptr, t, self.cfg.intermediate)
            self.project[backend](
                gate,
                normed,
                p + "mlp.gate_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_GATE,
                gate,
                False,
            )
            self.project[backend](
                up,
                normed,
                p + "mlp.up_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_UP,
                up,
                False,
            )
            swiglu_k[backend](activated, gate, up)
            self.project[backend](
                proj,
                activated,
                p + "mlp.down_proj.weight",
                layer * Q4_SLOTS_PER_LAYER + Q4_SLOT_DOWN,
                proj,
                False,
            )
            add_k[backend](hidden, hidden, proj)

        # Only the last position's logits are ever asked for, so only that row
        # goes through the projection — 151936 outputs per token is the single
        # most expensive line in the pass.
        var last_hidden = hidden.slice_dim(0, t - 1, 1)
        var last_normed = rows_view(self.normed_ptr, 1, h)
        rmsnorm_k[backend](last_normed, last_hidden, self.params.view(FINAL_NORM), self.cfg.eps)
        var logits = rows_view(self.logits_ptr, 1, self.cfg.vocab)
        # Copied out of `self` first: the projection reads the name while the
        # call writes through `self`, and the borrow checker is right that a
        # field cannot be both.
        var head_name = self.head
        self.project[backend](logits, last_normed, head_name, Q4_NO_SLOT, logits, False)
        self.kv_len = kv_len
        return f32_data(logits)

    def argmax(self, logits: F32Ptr) -> Int:
        """Index of the largest logit; ties go to the lowest index.

        Ties resolved to the lowest index, as `torch.argmax` does, so that a
        comparison of two argmaxes is a comparison of sequences and not of
        tie-breaking habits.
        """
        var best = 0
        var best_value = Float32(-3.4028234663852886e38)
        for i in range(self.cfg.vocab):
            var v = logits[unsafe_offset=i]
            if v > best_value:
                best_value = v
                best = i
        return best


def copy_into(dst: RawPtr, dst_elem: Int, src: TensorView, n: Int) raises AlofaError:
    """Copy `n` floats of `src` into raw memory at element `dst_elem`.

    An element loop rather than a bulk copy: `n` is one step's worth of keys
    and values, and the arithmetic-free version is the one that cannot be
    wrong about element sizes.
    """
    if n > src.numel():
        raise AlofaError(
            ERR_OUT_OF_RANGE,
            "copy longer than the source",
            "n=" + String(n) + " src=" + String(src.numel()),
        )
    var pd = dst.unsafe_bitcast[Float32]()
    var ps = f32_data(src)
    for i in range(n):
        pd[unsafe_offset=dst_elem + i] = ps[unsafe_offset=i]
