"""账本一百一十五-b：decode GEMM 冷转隔离探针（64 套轮转 >L2）。

形状与生产逐字节同（f16 权重）：
  qkvp: 1152x896（q+k/v 打包单矩阵——绕开现役 tri 核 3 段串行树
        21 屏障的结构病；生产只需 cat3 一次 + 指针偏移接 attention）
  o:    896x896    down: 4864x896
扫 fat 家族几何 (T, OUTS)。64 套 × 12.4MB = 793MB ≫ L2 ⇒ 每读必冷。
流水内基线：tri-qkv 9.4µs/层(220GB/s) o 5.3µs(301) down 13.3µs(656)。

aliasing 检查（账本六十三加强版）：同 buffer 派生的两个指针即使落
局部变量也会被判定同源 ⇒ dst/bias 必须用独立 buffer + comptime
分支直传 unsafe_ptr()。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.matmul import f16_matmul_fat_gpu, f16_matmul_qkv_gpu

comptime NSET = 64
comptime INNER = 896
comptime INNER_FF = 4864
comptime NQKV = 1152
comptime ROUNDS = 4


def main() raises:
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](INNER_FF)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](NQKV)
    var hout = ctx.enqueue_create_host_buffer[DType.float32](NQKV)
    var href = ctx.enqueue_create_host_buffer[DType.float32](NQKV)
    ctx.synchronize()
    for i in range(INNER_FF):
        hx[i] = Float32(Float64((i * 13) % 23 - 11) / 16.0)
    for i in range(NQKV):
        hb[i] = Float32(Float64((i * 7) % 11 - 5) / 32.0)
    var dx = ctx.enqueue_create_buffer[DType.float32](INNER_FF)
    var dbias = ctx.enqueue_create_buffer[DType.float32](NQKV)
    var dzero = ctx.enqueue_create_buffer[DType.float32](NQKV)
    var dout = ctx.enqueue_create_buffer[DType.float32](NQKV)
    var dref = ctx.enqueue_create_buffer[DType.float32](NQKV)
    var dq2 = ctx.enqueue_create_buffer[DType.float32](INNER)
    var dk2 = ctx.enqueue_create_buffer[DType.float32](128)
    var dv2 = ctx.enqueue_create_buffer[DType.float32](128)
    var bq2 = ctx.enqueue_create_buffer[DType.float32](INNER)
    var bk2 = ctx.enqueue_create_buffer[DType.float32](128)
    var bv2 = ctx.enqueue_create_buffer[DType.float32](128)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dbias, src_buf=hb)
    ctx.synchronize()

    var wtotal = NSET * (NQKV * INNER + INNER * INNER + INNER_FF * INNER)
    var hw = ctx.enqueue_create_host_buffer[DType.float16](wtotal)
    ctx.synchronize()
    var off = 0
    for s in range(NSET):
        for i in range(NQKV * INNER):
            hw[off + i] = Float16(
                Float64((s * 31 + i * 17) % 29 - 14) / 16.0
            )
        off += NQKV * INNER
        for i in range(INNER * INNER):
            hw[off + i] = Float16(
                Float64((s * 31 + i * 13) % 23 - 11) / 16.0
            )
        off += INNER * INNER
        for i in range(INNER_FF * INNER):
            hw[off + i] = Float16(
                Float64((s * 31 + i * 11) % 19 - 9) / 16.0
            )
        off += INNER_FF * INNER
    var wall = ctx.enqueue_create_buffer[DType.float16](wtotal)
    ctx.enqueue_copy(dst_buf=wall, src_buf=hw)
    ctx.synchronize()
    var o_off = NSET * NQKV * INNER
    var d_off = o_off + NSET * INNER * INNER

    @parameter
    for v in range(11):
        # v: 0-3 qkvp (T128 O2 / T128 O1 / T256 O2 / T256 O1)
        #    4-7 o 同布局    8-10 down (128 O2 / 256 O2 / 512 O2)
        comptime IS_DOWN = v >= 8
        comptime INW = INNER_FF if IS_DOWN else INNER
        comptime TT = (
            512 if v == 10 else
            (256 if (v % 4) >= 2 else 128)
        )
        comptime OUTS = 1 if (v % 2) == 1 else 2
        var woff = 0
        var cols = NQKV
        var wstride = NQKV * INNER
        comptime if v >= 4 and v < 8:
            woff = o_off
            cols = INNER
            wstride = INNER * INNER
        comptime if IS_DOWN:
            woff = d_off
            cols = INNER
            wstride = INNER_FF * INNER

        # 基准（T128/O2 现役几何）→ dref
        comptime if v < 4:
            ctx.enqueue_function[
                f16_matmul_fat_gpu[128, INNER, 2, True, False]
            ](
                dx.unsafe_ptr(), wall.unsafe_ptr() + woff,
                dref.unsafe_ptr(), dbias.unsafe_ptr(),
                Int32(INNER), Int32(cols),
                grid_dim=(cols // 2, 1, 1), block_dim=(128, 1, 1),
            )
        elif v < 8:
            ctx.enqueue_function[
                f16_matmul_fat_gpu[128, INNER, 2, True, False]
            ](
                dx.unsafe_ptr(), wall.unsafe_ptr() + woff,
                dref.unsafe_ptr(), dzero.unsafe_ptr(),
                Int32(INNER), Int32(cols),
                grid_dim=(cols // 2, 1, 1), block_dim=(128, 1, 1),
            )
        else:
            ctx.enqueue_function[
                f16_matmul_fat_gpu[128, INNER_FF, 2, True, False]
            ](
                dx.unsafe_ptr(), wall.unsafe_ptr() + woff,
                dref.unsafe_ptr(), dzero.unsafe_ptr(),
                Int32(INNER_FF), Int32(cols),
                grid_dim=(cols // 2, 1, 1), block_dim=(128, 1, 1),
            )
        # 变体几何 → dout
        comptime if not IS_DOWN:
            ctx.enqueue_function[
                f16_matmul_fat_gpu[TT, INNER, OUTS, True, False]
            ](
                dx.unsafe_ptr(), wall.unsafe_ptr() + woff,
                dout.unsafe_ptr(),
                (dbias.unsafe_ptr() if v < 4 else dzero.unsafe_ptr()),
                Int32(INNER), Int32(cols),
                grid_dim=(cols // OUTS, 1, 1), block_dim=(TT, 1, 1),
            )
        else:
            ctx.enqueue_function[
                f16_matmul_fat_gpu[TT, INNER_FF, OUTS, True, False]
            ](
                dx.unsafe_ptr(), wall.unsafe_ptr() + woff,
                dout.unsafe_ptr(), dzero.unsafe_ptr(),
                Int32(INNER_FF), Int32(cols),
                grid_dim=(cols // OUTS, 1, 1), block_dim=(TT, 1, 1),
            )
        ctx.enqueue_copy(dst_buf=hout, src_buf=dout)
        ctx.enqueue_copy(dst_buf=href, src_buf=dref)
        ctx.synchronize()
        var md = 0.0
        for i in range(cols):
            var d = Float64(hout[i]) - Float64(href[i])
            if d < 0:
                d = -d
            if d > md:
                md = d

        var best = 1e30
        for rep in range(ROUNDS):
            var t0 = monotonic_ns()
            for i in range(NSET):
                var wp = wall.unsafe_ptr() + woff + i * wstride
                comptime if v < 4:
                    ctx.enqueue_function[
                        f16_matmul_fat_gpu[TT, INNER, OUTS, True, False]
                    ](
                        dx.unsafe_ptr(), wp, dout.unsafe_ptr(),
                        dbias.unsafe_ptr(), Int32(INNER), Int32(cols),
                        grid_dim=(cols // OUTS, 1, 1),
                        block_dim=(TT, 1, 1),
                    )
                elif v < 8:
                    ctx.enqueue_function[
                        f16_matmul_fat_gpu[TT, INNER, OUTS, True, False]
                    ](
                        dx.unsafe_ptr(), wp, dout.unsafe_ptr(),
                        dzero.unsafe_ptr(), Int32(INNER), Int32(cols),
                        grid_dim=(cols // OUTS, 1, 1),
                        block_dim=(TT, 1, 1),
                    )
                else:
                    ctx.enqueue_function[
                        f16_matmul_fat_gpu[TT, INNER_FF, OUTS, True, False]
                    ](
                        dx.unsafe_ptr(), wp, dout.unsafe_ptr(),
                        dzero.unsafe_ptr(), Int32(INNER_FF), Int32(cols),
                        grid_dim=(cols // OUTS, 1, 1),
                        block_dim=(TT, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e3 / Float64(NSET)
            if per < best:
                best = per
        var mb = Float64(cols) * Float64(INW) * 2.0 / 1e6
        print("v", v, "T", TT, "O", OUTS, "cols", cols, "us:", best,
              "GB/s:", mb / best * 1e3, "maxdiff:", md)

    # v11：tri 核隔离基线（解包 q/k/v 各 96 套，同钟同口径）
    var hw3 = ctx.enqueue_create_host_buffer[DType.float16](
        NSET * (INNER * INNER + 2 * 128 * INNER)
    )
    ctx.synchronize()
    var off3 = 0
    for s in range(NSET):
        for i in range(INNER * INNER):
            hw3[off3 + i] = hw[
                s * NQKV * INNER + i
            ]
        off3 += INNER * INNER
        for i in range(2 * 128 * INNER):
            hw3[off3 + i] = hw[
                s * NQKV * INNER + INNER * INNER + i
            ]
        off3 += 2 * 128 * INNER
    var w3q = ctx.enqueue_create_buffer[DType.float16](NSET * INNER * INNER)
    var w3k = ctx.enqueue_create_buffer[DType.float16](NSET * 128 * INNER)
    var w3v = ctx.enqueue_create_buffer[DType.float16](NSET * 128 * INNER)
    # 三份独立上传（同 buffer 派生指针会被 aliasing 检查判同源）
    var hwq = ctx.enqueue_create_host_buffer[DType.float16](
        NSET * INNER * INNER
    )
    var hwk = ctx.enqueue_create_host_buffer[DType.float16](
        NSET * 128 * INNER
    )
    var hwv = ctx.enqueue_create_host_buffer[DType.float16](
        NSET * 128 * INNER
    )
    ctx.synchronize()
    var o3 = 0
    for s in range(NSET):
        for i in range(INNER * INNER):
            hwq[o3 + i] = hw3[s * (INNER * INNER + 2 * 128 * INNER) + i]
        for i in range(128 * INNER):
            hwk[o3 + i] = hw3[
                s * (INNER * INNER + 2 * 128 * INNER)
                + INNER * INNER + i
            ]
            hwv[o3 + i] = hw3[
                s * (INNER * INNER + 2 * 128 * INNER)
                + INNER * INNER + 128 * INNER + i
            ]
        o3 += INNER * INNER
    ctx.enqueue_copy(dst_buf=w3q, src_buf=hwq)
    ctx.enqueue_copy(dst_buf=w3k, src_buf=hwk)
    ctx.enqueue_copy(dst_buf=w3v, src_buf=hwv)
    ctx.synchronize()
    var best3 = 1e30
    for rep in range(ROUNDS):
        var t0 = monotonic_ns()
        for i in range(NSET):
            var qo = i * INNER * INNER
            var ko = i * 128 * INNER
            ctx.enqueue_function[
                f16_matmul_qkv_gpu[128, INNER, True, True]
            ](
                dx.unsafe_ptr(), (w3q.unsafe_ptr() + qo),
                (w3k.unsafe_ptr() + ko), (w3v.unsafe_ptr() + ko),
                dq2.unsafe_ptr(), dk2.unsafe_ptr(), dv2.unsafe_ptr(),
                bq2.unsafe_ptr(), bk2.unsafe_ptr(), bv2.unsafe_ptr(),
                dzero.unsafe_ptr(), Float32(1e-6),
                Int32(INNER), Int32(INNER), Int32(128),
                grid_dim=(INNER, 1, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        var per = Float64(monotonic_ns() - t0) / 1e3 / Float64(NSET)
        if per < best3:
            best3 = per
    var mb3 = Float64(NQKV) * Float64(INNER) * 2.0 / 1e6
    print("v11 tri-qkv us:", best3, "GB/s:", mb3 / best3 * 1e3)
