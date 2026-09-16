"""A100 GPU kernel 冒烟测试 —— **只能在 A100 远程验证机上运行**。

本地开发机 GPU 是 Maxwell sm_52，现代 CUDA 栈与 MAX 均不支持，因此本文件
**故意不加入 `pixi run test`**（本地跑必然失败，会污染能力账本的信号）。

运行方式（见 scripts/a100.sh）：
    ./scripts/a100.sh gpu                 # 先挑一张空闲卡（共享机，0-3 常满载）
    ./scripts/a100.sh run 4 tests/gpu/vecadd.mojo

2026-09-16 实测结果（GPU 4）：
    DeviceContext created
    *** GPU VECADD PASS on A100 *** c[0]= 0.0   c[999]= 2997.0      # 999 + 2×999 ✓

Mojo 1.0 GPU 语法要点（实测踩坑，见 docs/plan/02-architecture.md §11.3）：
  - 线程/块索引      → `from std.gpu import ...`（与 std.sys 同构）
  - DeviceContext    → `from max.gpu.host import ...`（**不是** std.gpu.host）
  - 指针             → `Pointer[T, MutUntrackedOrigin]`；`UnsafePointer` 已废弃
  - kernel 标量参数  → 必须 `Int32`/`Int64`；`Int`/`UInt` 不 conform `DevicePassable`
"""

from std.gpu import global_idx
from std.math import ceildiv
from max.gpu.host import DeviceContext

comptime SIZE = 1000
comptime BLOCK = 256
comptime DT = DType.float32


def vec_add(
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    c: Pointer[Float32, MutUntrackedOrigin],
    n: Int64,
) -> None:
    var tid = global_idx.x
    if tid < Int(n):
        c[tid] = a[tid] + b[tid]


def main() raises:
    var ctx = DeviceContext()
    print("DeviceContext created")

    # 主机侧准备：a[i] = i, b[i] = 2i  →  期望 c[i] = 3i
    var host_a = ctx.enqueue_create_host_buffer[DT](SIZE)
    var host_b = ctx.enqueue_create_host_buffer[DT](SIZE)
    ctx.synchronize()
    for i in range(SIZE):
        host_a[i] = Float32(i)
        host_b[i] = Float32(i) * 2.0

    var dev_a = ctx.enqueue_create_buffer[DT](SIZE)
    var dev_b = ctx.enqueue_create_buffer[DT](SIZE)
    var dev_c = ctx.enqueue_create_buffer[DT](SIZE)
    ctx.enqueue_copy(dst_buf=dev_a, src_buf=host_a)
    ctx.enqueue_copy(dst_buf=dev_b, src_buf=host_b)

    ctx.enqueue_function[vec_add](
        dev_a.unsafe_ptr(),
        dev_b.unsafe_ptr(),
        dev_c.unsafe_ptr(),
        Int64(SIZE),
        grid_dim=(ceildiv(SIZE, BLOCK), 1, 1),
        block_dim=(BLOCK,),
    )
    ctx.synchronize()

    var host_c = ctx.enqueue_create_host_buffer[DT](SIZE)
    ctx.enqueue_copy(dst_buf=host_c, src_buf=dev_c)
    ctx.synchronize()

    var bad = 0
    for i in range(SIZE):
        if host_c[i] != Float32(i) + Float32(i) * 2.0:
            bad += 1
    if bad != 0:
        raise Error("VECADD MISMATCH count=" + String(bad))

    print("*** GPU VECADD PASS on A100 *** c[0]=", host_c[0], " c[999]=", host_c[999])
