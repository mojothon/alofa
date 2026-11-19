"""账本一百五十三/一百五十四：批行 rmsnorm 的链上净时微基准。

背景：ALOFA_PROFILE 分段测得 tri 段 ~30µs/次（批行 [32,896]）——但该口径
每段尾强制 synchronize，把流水线排空的时间塞进了段里。本基准拆几种口径：

1. burst：N 次连续发射 + 末尾一次 sync —— 均摊链上净时（生产口径，
   enqueue 异步流水线）
2. percall-sync：每次发射后 sync —— profile 口径（含排空）
3. 形状扫：rows 1/8/32/64、block_dim 64/128/256/512

不依赖模型权重，buffer 直接构造。地板参考：32×896×4B×2 读写 + 896×4B w
≈ 0.26µs @900GB/s。
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.kernels.cuda.basic import rmsnorm_gpu

comptime COLS = 896
comptime ROWS = 32


struct NormBench:
    """buffer 挂字段 + mut self 方法（与 GpuSession 同款 origin 模式）。"""

    var ctx: DeviceContext
    var d_x: DeviceBuffer[DType.float32]
    var d_w: DeviceBuffer[DType.float32]
    var d_dst: DeviceBuffer[DType.float32]

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        var n = ROWS * COLS
        self.d_x = self.ctx.enqueue_create_buffer[DType.float32](n)
        self.d_w = self.ctx.enqueue_create_buffer[DType.float32](COLS)
        self.d_dst = self.ctx.enqueue_create_buffer[DType.float32](n)
        self.ctx.synchronize()

    def _launch(mut self, rows: Int, tpb: Int) raises:
        self.ctx.enqueue_function[rmsnorm_gpu](
            self.d_x.unsafe_ptr(),
            self.d_w.unsafe_ptr(),
            self.d_dst.unsafe_ptr(),
            Int32(COLS),
            Int32(rows),
            Float32(1e-5),
            grid_dim=(rows, 1, 1),
            block_dim=(tpb, 1, 1),
        )

    def burst(mut self, rows: Int, tpb: Int, reps: Int) raises -> Float64:
        var t0 = monotonic_ns()
        for _k in range(reps):
            self._launch(rows, tpb)
        self.ctx.synchronize()
        return Float64(monotonic_ns() - t0) / 1e6 / Float64(reps)

    def percall(mut self, reps: Int) raises -> Float64:
        var t0 = monotonic_ns()
        for _k in range(reps):
            self._launch(ROWS, 256)
            self.ctx.synchronize()
        return Float64(monotonic_ns() - t0) / 1e6 / Float64(reps)


def main() raises:
    var bench = NormBench()

    # 预热（JIT）
    _ = bench.burst(ROWS, 256, 50)

    # 口径 1：burst 均摊（生产口径）
    var b1 = 1e30
    for _k in range(3):
        var v = bench.burst(ROWS, 256, 2000)
        if v < b1:
            b1 = v
    print("burst per-launch us:", b1 * 1e3)

    # 口径 2：逐次 sync（profile 口径）
    var p = bench.percall(200)
    print("percall-sync per-launch us:", p * 1e3)

    # 口径 3：rows 扫（tpb=256）
    var v1 = bench.burst(1, 256, 800)
    print("burst rows=1 us:", v1 * 1e3)
    var v8 = bench.burst(8, 256, 800)
    print("burst rows=8 us:", v8 * 1e3)
    var v32 = bench.burst(32, 256, 800)
    print("burst rows=32 us:", v32 * 1e3)
    var v64 = bench.burst(64, 256, 800)
    print("burst rows=64 us:", v64 * 1e3)

    # 口径 4：block_dim 扫（rows=32）
    var t64 = bench.burst(ROWS, 64, 800)
    print("burst tpb=64 us:", t64 * 1e3)
    var t128 = bench.burst(ROWS, 128, 800)
    print("burst tpb=128 us:", t128 * 1e3)
    var t256 = bench.burst(ROWS, 256, 800)
    print("burst tpb=256 us:", t256 * 1e3)
    var t512 = bench.burst(ROWS, 512, 800)
    print("burst tpb=512 us:", t512 * 1e3)
