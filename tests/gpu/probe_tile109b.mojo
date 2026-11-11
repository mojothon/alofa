"""账本一百零九-f：down/dual 两个**真实 packed 核**在冷权重（8 份独立
打包缓冲轮转）下的分块扫描，GPU 1 满时钟。

down: INNER=4864 → out 896（56 块 @COLS128 / 112 块 @COLS64）
dual: INNER=896 → out 4864（304 块 @COLS128 / 608 块 @COLS64）

每配置 8 份权重轮转、REP=6，取最小。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.kernels.cuda.matmul import (
    f16_mma_down_pk_gpu,
    f16_mma_dual_pk_gpu,
    pack_w_mma_gpu,
)

comptime T = 128
comptime NSET = 8
comptime REP = 5

# down: 6 配置；dual: 4 配置
comptime ND = 6
comptime NU = 4


def main() raises:
    var ctx = DeviceContext()
    comptime DIN = 4864
    comptime DOUT = 896
    comptime UIN = 896
    comptime UOUT = 4864

    var hxd = ctx.enqueue_create_host_buffer[DType.float16](T * DIN)
    var hxu = ctx.enqueue_create_host_buffer[DType.float16](T * UIN)
    var hwd = ctx.enqueue_create_host_buffer[DType.float16](DOUT * DIN)
    var hwu = ctx.enqueue_create_host_buffer[DType.float16](UOUT * UIN)
    ctx.synchronize()
    for i in range(T * DIN):
        hxd[i] = Float16(Float64((i * 13) % 23 - 11) / 16.0)
    for i in range(T * UIN):
        hxu[i] = Float16(Float64((i * 11) % 19 - 9) / 16.0)
    for i in range(DOUT * DIN):
        hwd[i] = Float16(Float64((i * 7) % 11 - 5) / 32.0)
    for i in range(UOUT * UIN):
        hwu[i] = Float16(Float64((i * 5) % 13 - 6) / 32.0)

    var dxd = ctx.enqueue_create_buffer[DType.float16](T * DIN)
    var dxu = ctx.enqueue_create_buffer[DType.float16](T * UIN)
    var dwd = ctx.enqueue_create_buffer[DType.float16](DOUT * DIN)
    var dwu = ctx.enqueue_create_buffer[DType.float16](UOUT * UIN)
    var dxbd = ctx.enqueue_create_buffer[DType.float32](T * DOUT)
    var dga = ctx.enqueue_create_buffer[DType.float32](T * UOUT)
    var dgb = ctx.enqueue_create_buffer[DType.float32](T * UOUT)
    var dact = ctx.enqueue_create_buffer[DType.float32](T * UOUT)
    ctx.enqueue_copy(dst_buf=dxd, src_buf=hxd)
    ctx.enqueue_copy(dst_buf=dxu, src_buf=hxu)
    ctx.enqueue_copy(dst_buf=dwd, src_buf=hwd)
    ctx.enqueue_copy(dst_buf=dwu, src_buf=hwu)
    ctx.synchronize()

    var wpd = List[DeviceBuffer[DType.float16]]()
    for i in range(ND * NSET):
        wpd.append(ctx.enqueue_create_buffer[DType.float16](DOUT * DIN))
    var wpu1 = List[DeviceBuffer[DType.float16]]()
    var wpu2 = List[DeviceBuffer[DType.float16]]()
    for i in range(NU * NSET):
        wpu1.append(ctx.enqueue_create_buffer[DType.float16](UOUT * UIN))
        wpu2.append(ctx.enqueue_create_buffer[DType.float16](UOUT * UIN))

    @parameter
    for v in range(ND):
        comptime W8 = v != 2 and v != 3
        comptime OS = 1 if (v == 1 or v == 3 or v == 5) else 2
        comptime COLS = (8 if W8 else 4) * OS * 8
        for s in range(NSET):
            ctx.enqueue_function[
                pack_w_mma_gpu[DIN, OS, 8 if W8 else 4]
            ](
                dwd.unsafe_ptr(), wpd[v * NSET + s].unsafe_ptr(),
                grid_dim=(DOUT // COLS, DIN // 16, 1), block_dim=(128, 1, 1),
            )
    @parameter
    for v in range(NU):
        comptime W8 = v != 1
        comptime OS = 2
        comptime COLS = (8 if W8 else 4) * OS * 8
        for s in range(NSET):
            ctx.enqueue_function[
                pack_w_mma_gpu[UIN, OS, 8 if W8 else 4]
            ](
                dwu.unsafe_ptr(), wpu1[v * NSET + s].unsafe_ptr(),
                grid_dim=(UOUT // COLS, UIN // 16, 1), block_dim=(128, 1, 1),
            )
            ctx.enqueue_function[
                pack_w_mma_gpu[UIN, OS, 8 if W8 else 4]
            ](
                dwu.unsafe_ptr(), wpu2[v * NSET + s].unsafe_ptr(),
                grid_dim=(UOUT // COLS, UIN // 16, 1), block_dim=(128, 1, 1),
            )
    ctx.synchronize()

    var bd = InlineArray[Float64, ND](fill=1e30)
    var bu = InlineArray[Float64, NU](fill=1e30)
    for rep in range(4):
        @parameter
        for v in range(ND):
            comptime W8 = v != 2 and v != 3
            comptime OS = 1 if (v == 1 or v == 3 or v == 5) else 2
            comptime COLS = (8 if W8 else 4) * OS * 8
            comptime CHV = 1216 if (v == 4 or v == 5) else 608
            var t0 = monotonic_ns()
            for r in range(REP):
                for s in range(NSET):
                    ctx.enqueue_function[
                        f16_mma_down_pk_gpu[DIN, CHV, OS, 8 if W8 else 4]
                    ](
                        dxd.unsafe_ptr(), wpd[v * NSET + s].unsafe_ptr(),
                        dxbd.unsafe_ptr(), Int32(DOUT), Int32(T),
                        grid_dim=(DOUT // COLS, (T + 15) // 16, 1),
                        block_dim=(256, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP * NSET)
            if per < bd[v]:
                bd[v] = per
        @parameter
        for v in range(NU):
            comptime W8 = v != 1
            comptime OS = 2
            comptime COLS = (8 if W8 else 4) * OS * 8
            comptime CHV = 896 if v >= 2 else 608
            var t1 = monotonic_ns()
            for r in range(REP):
                for s in range(NSET):
                    ctx.enqueue_function[
                        f16_mma_dual_pk_gpu[UIN, CHV, 8 if W8 else 4]
                    ](
                        dxu.unsafe_ptr(), wpu1[v * NSET + s].unsafe_ptr(),
                        wpu2[v * NSET + s].unsafe_ptr(), dga.unsafe_ptr(),
                        dgb.unsafe_ptr(), dact.unsafe_ptr(),
                        Int32(UOUT), Int32(T),
                        grid_dim=(UOUT // COLS, (T + 15) // 16, 1),
                        block_dim=(256, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t1) / 1e6 / Float64(REP * NSET)
            if per < bu[v]:
                bu[v] = per

    @parameter
    for v in range(ND):
        comptime W8 = v != 2 and v != 3
        comptime OS = 1 if (v == 1 or v == 3 or v == 5) else 2
        comptime CHV = 1216 if (v == 4 or v == 5) else 608
        print("down cfg", v, " WARP:", 8 if W8 else 4, " OUTS:", OS,
              " COLS:", (8 if W8 else 4) * OS * 8, " CH:", CHV,
              " us/call:", bd[v])
    @parameter
    for v in range(NU):
        comptime W8 = v != 1
        comptime OS = 2
        comptime CHV = 896 if v >= 2 else 608
        print("dual cfg", v, " WARP:", 8 if W8 else 4, " OUTS:", OS,
              " COLS:", (8 if W8 else 4) * OS * 8, " CH:", CHV,
              " us/call:", bu[v])
