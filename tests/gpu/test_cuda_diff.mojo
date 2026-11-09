"""CUDA kernel 与标量后端的**逐值差分**门 —— **只能在 A100 远程验证机上运行**。

本地开发机是 Maxwell sm_52，现代 CUDA 栈与 MAX 均不支持，因此本文件**故意不加入
`pixi run test`**（本地跑必然失败，会污染能力账本的信号）。运行方式：

    ./scripts/a100.sh gpu                          # 先挑一张空闲卡（共享机）
    ./scripts/a100.sh run 4 tests/gpu/test_cuda_diff.mojo

覆盖四个 kernel：`rmsnorm` 与 `linear`（相对容差 1e-5）、`dequant_q4_0` 与
`dequant_q8_0`（**零容差**）。

本门**只验正确性、不报性能**（账本 §4 给这条路线定的解锁条件③）：GPU 与标量后端算
**同一个**输入，逐值比 fp32 **相对**容差 1e-5 —— 与 AVX2 门同一个数，不是「GPU 可以
宽一点」。

2026-11-09 实测结果（A100，卡 0，`mojo run -I src`）：
    *** RMSNORM   GPU/SCALAR DIFF PASS *** diff= 0.0  tol= 5.0350770950317384e-05
    *** DEQUANT Q4_0 GPU/SCALAR DIFF PASS *** diff= 0.0
    *** DEQUANT Q8_0 GPU/SCALAR DIFF PASS *** diff= 0.0
    *** LINEAR    GPU/SCALAR DIFF PASS *** diff= 0.0  tol= 0.024
    （四个 kernel 全部与标量**逐位一致**，比容差要求还干净 —— linear 也 diff=0.0
    说明 GPU 编译器没把乘加折叠成改变舍入的 FMA。注意远程直跑必须带 `-I src`，
    否则 `alofa.verify` 会报 "does not refer to a nested package"。另外 `ref` 是
    Mojo 关键字，变量名避开。）

为什么 kernel 里一个线程要独立算完一整行（看起来很浪费）：本门只想让「GPU 算的数与
CPU 算的数是同一个」这句话成立。写成块内规约会引入**共享内存**和**浮点累加顺序**
两个额外的正确性变量，一旦数值对不上就分不清是布局读错还是规约写错。等这条路走通
了再谈性能内核 —— 那时换的是实现，不是这个门的判据。
"""

from std.gpu import global_idx
from std.math import ceildiv, sqrt

from max.gpu.host import DeviceContext
from alofa.core.dtype import DT_FP32
from alofa.core.memory import Arena
from alofa.core.tensor import Shape, TensorView, f32_data
from alofa.kernels.cpu.quant import (
    Q4_BLOCK,
    Q4_BYTES,
    Q8_BLOCK,
    Q8_BYTES,
    dequant_q4_0,
    dequant_q8_0,
    quantize_q4_0,
    quantize_q8_0,
)
from alofa.kernels.cpu.scalar import linear, rmsnorm
from alofa.kernels.cuda.basic import (
    dequant_q4_0_gpu,
    dequant_q8_0_gpu,
    linear_gpu,
    rmsnorm_gpu,
)
from alofa.verify.compare import max_abs, max_abs_diff

comptime DT = DType.float32
comptime ROWS = 8
comptime COLS = 896
# q4_0 反量化差分的规模：8 个块（256 个值），块结构与 `kernels/cpu/quant.mojo` 一致。
comptime Q4_BLOCKS = 8
comptime Q4_N = Q4_BLOCKS * Q4_BLOCK
# q8_0 差分：同样 8 个块（256 个值）。
comptime Q8_BLOCKS = 8
comptime Q8_N = Q8_BLOCKS * Q8_BLOCK
# linear 差分的规模：Qwen 0.5B 的 hidden=896，输出列取 16（差分门不追规模）。
comptime LM_ROWS = 4
comptime LM_INNER = 896
comptime LM_OUT = 16
comptime EPS = Float32(1e-6)
comptime BLOCK = 64
# 与 AVX2 门**同一个**容差。
comptime REL_TOLERANCE = Float64(1e-5)


def scratch(mut arena: Arena, rows: Int, cols: Int) raises -> TensorView:
    var dims = List[Int]()
    dims.append(rows)
    dims.append(cols)
    var raw = arena.alloc(rows * cols * 4)
    return TensorView(raw, Shape(dims), DT_FP32)


def fill(t: TensorView, seed: Int) raises -> None:
    """确定性的、不整齐的输入。

    不用随机数，也不取 0/1 这类特殊值：这个门的判据是数值，输入本身必须可复现，
    且要能撑起 `ref_max` 的规模（全零输入会让相对容差退化成绝对容差，门形同虚设）。
    """
    var p = f32_data(t)
    for i in range(t.numel()):
        var k = i + seed * 7919
        p[unsafe_offset=i] = Float32((k % 251) - 125) * Float32(0.023)


def rmsnorm_diff_check() raises:
    var arena = Arena(1 << 20)
    var x = scratch(arena, ROWS, COLS)
    var w = scratch(arena, 1, COLS)
    var dst_cpu = scratch(arena, ROWS, COLS)
    fill(x, 1)
    fill(w, 2)

    # 参照：CPU 标量后端（不是另写一份，是产品代码本身）
    rmsnorm(dst_cpu, x, w, EPS)

    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DT](ROWS * COLS)
    var hw = ctx.enqueue_create_host_buffer[DT](COLS)
    var hd = ctx.enqueue_create_host_buffer[DT](ROWS * COLS)
    ctx.synchronize()
    var px = f32_data(x)
    var pw = f32_data(w)
    for i in range(ROWS * COLS):
        hx[i] = px[unsafe_offset=i]
    for i in range(COLS):
        hw[i] = pw[unsafe_offset=i]

    var dx = ctx.enqueue_create_buffer[DT](ROWS * COLS)
    var dw = ctx.enqueue_create_buffer[DT](COLS)
    var dd = ctx.enqueue_create_buffer[DT](ROWS * COLS)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)

    ctx.enqueue_function[rmsnorm_gpu](
        dx.unsafe_ptr(),
        dw.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(COLS),
        Int32(ROWS),
        EPS,
        grid_dim=(ceildiv(ROWS, BLOCK), 1, 1),
        block_dim=(BLOCK,),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var n = ROWS * COLS
    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(dst_cpu), n)
    var scale = max_abs(f32_data(dst_cpu), n)
    if scale < 1.0:
        scale = 1.0
    var tol = REL_TOLERANCE * scale
    # arena 必须活到上面这些指针都不再用为止：Mojo 在值的**最后一次使用处**析构，而
    # `arena` 变量的最后一次使用是第三个 `scratch(...)` —— 比 `fill` 还早。不在这里
    # 声明，`Arena.__deinit__` 会在 `fill(x, 1)` 之前就把整块映射 `munmap` 掉，于是
    # 第一个写就落到已释放的地址上（实测：段错误，栈指向 `fill`，看起来像 GPU 的锅）。
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "rmsnorm: gpu vs scalar max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    # 参照张量若全为零，上面的相对容差会退化成绝对容差，门就形同虚设。
    if scale <= 0.0:
        raise Error("rmsnorm: the reference tensor is all zeros")
    print("*** RMSNORM GPU/SCALAR DIFF PASS *** diff=", diff, " tol=", tol)


def dequant_q4_diff_check() raises:
    """q4_0 反量化的 GPU/标量差分，判据是**零容差**。

    为什么与 rmsnorm 不同：反量化结果只由 nibble 与缩放因子决定，两者都是精确
    的（`kernels/cpu/quant.mojo` 的 docstring 说得更狠 —— 用容差反而是放水，会把
    "半字节高低位装反"这类错伪装成"舍入差一点"）。差一个 ULP 就是错，没有第二句话。
    """
    var arena = Arena(1 << 20)
    var n_values = Int(Q4_N)
    var n_bytes = Int(Q4_BLOCKS) * Int(Q4_BYTES)
    var src_view = scratch(arena, 1, n_values)
    fill(src_view, 3)
    var blocks = arena.alloc(n_bytes)
    var ref_out = arena.alloc(n_values * 4).unsafe_bitcast[Float32]()
    # 块从**产品代码的量化器**来：顺带验了 GPU 读的块布局与量化器写的一致。
    quantize_q4_0(blocks, f32_data(src_view), n_values)
    # 参照：CPU 标量反量化（不是另写一份，是产品代码本身）。
    dequant_q4_0(ref_out, blocks, n_values)

    var ctx = DeviceContext()
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var hd = ctx.enqueue_create_host_buffer[DT](n_values)
    ctx.synchronize()
    for i in range(n_bytes):
        hb[i] = blocks[unsafe_offset=i]

    var db = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var dd = ctx.enqueue_create_buffer[DT](n_values)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.enqueue_function[dequant_q4_0_gpu](
        db.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(Q4_BLOCKS),
        grid_dim=(Int(Q4_BLOCKS), 1, 1),
        block_dim=(Int(Q4_BLOCK),),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var diff = Float64(0)
    for i in range(n_values):
        var d = Float64(hd[i]) - Float64(ref_out[unsafe_offset=i])
        if d < 0.0:
            d = -d
        if d > diff:
            diff = d
    # 与 rmsnorm 同一个坑：arena 的最后一次使用在上面 alloc 处，不 keep_alive 会被
    # 提前 munmap（见 rmsnorm_diff_check 里的注释）。
    arena.keep_alive()
    if diff != 0.0:
        raise Error(
            "dequant_q4_0: gpu vs scalar diff=" + String(diff) + " (zero tolerance)"
        )
    print("*** DEQUANT Q4_0 GPU/SCALAR DIFF PASS *** diff=", diff)


def dequant_q8_diff_check() raises:
    """q8_0 反量化的 GPU/标量差分，判据与 q4_0 一样是**零容差**。"""
    var arena = Arena(1 << 20)
    var n_values = Int(Q8_BLOCKS) * Int(Q8_BLOCK)
    var n_bytes = Int(Q8_BLOCKS) * Int(Q8_BYTES)
    var src_view = scratch(arena, 1, n_values)
    fill(src_view, 6)
    var blocks = arena.alloc(n_bytes)
    var ref_out = arena.alloc(n_values * 4).unsafe_bitcast[Float32]()
    quantize_q8_0(blocks, f32_data(src_view), n_values)
    dequant_q8_0(ref_out, blocks, n_values)

    var ctx = DeviceContext()
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var hd = ctx.enqueue_create_host_buffer[DT](n_values)
    ctx.synchronize()
    for i in range(n_bytes):
        hb[i] = blocks[unsafe_offset=i]
    var db = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var dd = ctx.enqueue_create_buffer[DT](n_values)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.enqueue_function[dequant_q8_0_gpu](
        db.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(Q8_BLOCKS),
        grid_dim=(Int(Q8_BLOCKS), 1, 1),
        block_dim=(Int(Q8_BLOCK),),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var diff = Float64(0)
    for i in range(n_values):
        var d = Float64(hd[i]) - Float64(ref_out[unsafe_offset=i])
        if d < 0.0:
            d = -d
        if d > diff:
            diff = d
    arena.keep_alive()
    if diff != 0.0:
        raise Error(
            "dequant_q8_0: gpu vs scalar diff=" + String(diff) + " (zero tolerance)"
        )
    print("*** DEQUANT Q8_0 GPU/SCALAR DIFF PASS *** diff=", diff)


def linear_diff_check() raises:
    """linear（无偏置 GEMM）的 GPU/标量差分，相对容差与 rmsnorm 同一个数。

    kernel 与 `_gemm` 同一个累加顺序（Float64、k 升序、末尾一次 Float32），预期
    也是逐位一致；判据仍用 1e-5 相对容差 —— 它是「同一算法」的门，不是「同一
    指令序列」的门，若某天 GPU 编译器把乘加折叠成 FMA 使差分变非零，门会把
    差值打出来让人看，而不是默默放行。
    """
    var arena = Arena(1 << 21)
    var x = scratch(arena, LM_ROWS, LM_INNER)
    var w = scratch(arena, LM_OUT, LM_INNER)
    var dst_cpu = scratch(arena, LM_ROWS, LM_OUT)
    fill(x, 4)
    fill(w, 5)
    # 参照：CPU 标量 `linear`（产品代码本身，`w` 按 [out, in] 存）。
    linear(dst_cpu, x, w)

    var ctx = DeviceContext()
    var n_x = Int(LM_ROWS) * Int(LM_INNER)
    var n_w = Int(LM_OUT) * Int(LM_INNER)
    var n_d = Int(LM_ROWS) * Int(LM_OUT)
    var hx = ctx.enqueue_create_host_buffer[DT](n_x)
    var hw = ctx.enqueue_create_host_buffer[DT](n_w)
    var hd = ctx.enqueue_create_host_buffer[DT](n_d)
    ctx.synchronize()
    var px = f32_data(x)
    var pw = f32_data(w)
    for i in range(n_x):
        hx[i] = px[unsafe_offset=i]
    for i in range(n_w):
        hw[i] = pw[unsafe_offset=i]

    var dx = ctx.enqueue_create_buffer[DT](n_x)
    var dw = ctx.enqueue_create_buffer[DT](n_w)
    var dd = ctx.enqueue_create_buffer[DT](n_d)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dw, src_buf=hw)
    ctx.enqueue_function[linear_gpu](
        dx.unsafe_ptr(),
        dw.unsafe_ptr(),
        dd.unsafe_ptr(),
        Int32(LM_INNER),
        Int32(LM_OUT),
        grid_dim=(Int(LM_OUT), Int(LM_ROWS), 1),
        block_dim=(1, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hd, src_buf=dd)
    ctx.synchronize()

    var n = n_d
    var diff = max_abs_diff(hd.unsafe_ptr(), f32_data(dst_cpu), n)
    var scale = max_abs(f32_data(dst_cpu), n)
    if scale < 1.0:
        scale = 1.0
    var tol = REL_TOLERANCE * scale
    arena.keep_alive()
    if diff > tol:
        raise Error(
            "linear: gpu vs scalar max_abs_diff="
            + String(diff)
            + " tolerance="
            + String(tol)
            + " ref_max="
            + String(scale)
        )
    if scale <= 0.0:
        raise Error("linear: the reference tensor is all zeros")
    print("*** LINEAR GPU/SCALAR DIFF PASS *** diff=", diff, " tol=", tol)


def main() raises:
    rmsnorm_diff_check()
    dequant_q4_diff_check()
    dequant_q8_diff_check()
    linear_diff_check()
