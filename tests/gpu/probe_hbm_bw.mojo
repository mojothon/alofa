"""HBM 纯流式带宽探针（账本八十七）：f16 布局上限 vs 权值族实测。

head/dual/down 管线真值 670-800GB/s —— 若纯流式也只有 ~800，
f16 权值族已到盒子结构顶，2ms 目标须换字节口径（MMA/量化）；
若纯流式 ~1.4TB/s+，则仍有结构空间。512MB f16，18992 块。
"""

from alofa.core.ffi.posix import monotonic_ns
from alofa.kernels.cuda.basic import hbm_probe_f16_gpu
from max.gpu.host import DeviceContext

comptime MB = 512
comptime N_ELEM = MB * 1024 * 1024 // 2
comptime BLOCKS = 2374
comptime THREADS = 256


def main() raises:
    var ctx = DeviceContext()
    var d_src = ctx.enqueue_create_buffer[DType.float16](N_ELEM)
    var d_out = ctx.enqueue_create_buffer[DType.float64](BLOCKS)
    ctx.synchronize()
    for rep in range(5):
        var t0 = monotonic_ns()
        ctx.enqueue_function[hbm_probe_f16_gpu[THREADS]](
            d_src.unsafe_ptr(),
            d_out.unsafe_ptr(),
            Int32(N_ELEM),
            grid_dim=(BLOCKS, 1, 1),
            block_dim=(THREADS, 1, 1),
        )
        ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6
        var gbs = Float64(MB) / ms * 1.024
        print("rep ", rep, " ", ms, " ms  ", gbs, " GB/s")
    # 校验和（防 DCE + 确认读真发生了）
    var h_out = ctx.enqueue_create_host_buffer[DType.float64](BLOCKS)
    ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
    ctx.synchronize()
    var chk = Float64(0)
    for i in range(BLOCKS):
        chk += h_out[i]
    print("checksum=", chk)
