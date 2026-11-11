"""账本一百一十七-b：prefill GEMM 冷转几何扫型（t=128 生产形状）。

五核皆 f32 FMA 寄存器分块（无张量核），f32 峰 19.5TFLOPS：
  q2 0.206G / o 0.206G / kv2 0.058G / dual 1.11G / down 1.11G per layer
  ⇒ f32 下限 ~3.3ms vs 实际 prefill 11.6ms ⇒ 3.5× 在 FMA 效率。
扫 (OUTS, RB) 网格扩展。48 套 × 21MB ≈ 1GB 冷转。
⚠️ aliasing：kv2 的 w_k/w_v、dual3 的 w_a/w_b 必须独立 buffer。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.matmul import (
    f16_bmm_q2_gpu, f16_bmm_kv2_gpu, f16_bmm_fat3_gpu, f16_bmm_dual3_gpu,
)

comptime NSET = 48
comptime T = 128
comptime INNER = 896
comptime INNER_FF = 4864
comptime QO = 896
comptime KVO = 128
comptime ROUNDS = 3


def main() raises:
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * INNER_FF)
    ctx.synchronize()
    for i in range(T * INNER_FF):
        hx[i] = Float32(Float64((i * 13) % 23 - 11) / 16.0)
    var dx1 = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dx2 = ctx.enqueue_create_buffer[DType.float32](T * INNER_FF)
    ctx.enqueue_copy(dst_buf=dx1, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dx2, src_buf=hx)
    ctx.synchronize()

    comptime WQ = QO * INNER
    comptime WK = KVO * INNER
    comptime WGA = INNER_FF * INNER
    comptime WDN = INNER_FF * INNER
    var hwq = ctx.enqueue_create_host_buffer[DType.float16](NSET * WQ)
    var hwk = ctx.enqueue_create_host_buffer[DType.float16](NSET * WK)
    var hwv = ctx.enqueue_create_host_buffer[DType.float16](NSET * WK)
    var hwo = ctx.enqueue_create_host_buffer[DType.float16](NSET * WQ)
    var hwga = ctx.enqueue_create_host_buffer[DType.float16](NSET * WGA)
    var hwup = ctx.enqueue_create_host_buffer[DType.float16](NSET * WGA)
    var hwdn = ctx.enqueue_create_host_buffer[DType.float16](NSET * WDN)
    var hbq = ctx.enqueue_create_host_buffer[DType.float32](QO)
    var hbk = ctx.enqueue_create_host_buffer[DType.float32](KVO)
    var hbv = ctx.enqueue_create_host_buffer[DType.float32](KVO)
    ctx.synchronize()
    for i in range(QO):
        hbq[i] = Float32(0.01)
    for i in range(KVO):
        hbk[i] = Float32(0.01)
        hbv[i] = Float32(0.02)
    for s in range(NSET):
        var o6 = s * WQ
        var ok = s * WK
        var og = s * WGA
        for i in range(WQ):
            hwq[o6 + i] = Float16(Float64((s * 31 + i * 17) % 29 - 14) / 16.0)
            hwo[o6 + i] = Float16(
                Float64((s * 37 + i * 13) % 23 - 11) / 16.0
            )
        for i in range(WK):
            hwk[ok + i] = Float16(
                Float64((s * 29 + i * 19) % 31 - 15) / 16.0
            )
            hwv[ok + i] = Float16(
                Float64((s * 23 + i * 11) % 27 - 13) / 16.0
            )
        for i in range(WGA):
            hwga[og + i] = Float16(
                Float64((s * 41 + i * 7) % 25 - 12) / 16.0
            )
            hwup[og + i] = Float16(
                Float64((s * 43 + i * 5) % 21 - 10) / 16.0
            )
        for i in range(WDN):
            hwdn[og + i] = Float16(
                Float64((s * 47 + i * 3) % 33 - 16) / 16.0
            )
    var dwq = ctx.enqueue_create_buffer[DType.float16](NSET * WQ)
    var dwk = ctx.enqueue_create_buffer[DType.float16](NSET * WK)
    var dwv = ctx.enqueue_create_buffer[DType.float16](NSET * WK)
    var dwo = ctx.enqueue_create_buffer[DType.float16](NSET * WQ)
    var dwga = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwup = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwdn = ctx.enqueue_create_buffer[DType.float16](NSET * WDN)
    var dbq = ctx.enqueue_create_buffer[DType.float32](QO)
    var dbk = ctx.enqueue_create_buffer[DType.float32](KVO)
    var dbv = ctx.enqueue_create_buffer[DType.float32](KVO)
    ctx.enqueue_copy(dst_buf=dwq, src_buf=hwq)
    ctx.enqueue_copy(dst_buf=dwk, src_buf=hwk)
    ctx.enqueue_copy(dst_buf=dwv, src_buf=hwv)
    ctx.enqueue_copy(dst_buf=dwo, src_buf=hwo)
    ctx.enqueue_copy(dst_buf=dwga, src_buf=hwga)
    ctx.enqueue_copy(dst_buf=dwup, src_buf=hwup)
    ctx.enqueue_copy(dst_buf=dwdn, src_buf=hwdn)
    ctx.enqueue_copy(dst_buf=dbq, src_buf=hbq)
    ctx.enqueue_copy(dst_buf=dbk, src_buf=hbk)
    ctx.enqueue_copy(dst_buf=dbv, src_buf=hbv)
    ctx.synchronize()

    var dq = ctx.enqueue_create_buffer[DType.float32](T * QO)
    var dk = ctx.enqueue_create_buffer[DType.float32](T * KVO)
    var dv = ctx.enqueue_create_buffer[DType.float32](T * KVO)
    var do_ = ctx.enqueue_create_buffer[DType.float32](T * QO)
    var dga = ctx.enqueue_create_buffer[DType.float32](T * INNER_FF)
    var dup = ctx.enqueue_create_buffer[DType.float32](T * INNER_FF)
    var dact = ctx.enqueue_create_buffer[DType.float32](T * INNER_FF)
    var ddn = ctx.enqueue_create_buffer[DType.float32](T * QO)
    ctx.synchronize()

    # v: 0-3 q2 (4,8)/(2,8)/(4,4)/(2,4)  4 kv2 (2,8)
    #    5-6 o (4,8)/(2,8)  7 dual (2,8,T64)  8-10 down (4,8)/(2,8)/(4,4)
    @parameter
    for v in range(11):
        comptime OUTS = 4
        comptime RB = 8
        comptime if v == 1 or v == 3 or v == 6 or v == 9:
            OUTS = 2
        comptime if v == 2 or v == 3 or v == 10:
            RB = 4
        var best = 1e30
        for rep in range(ROUNDS):
            var t0 = monotonic_ns()
            for i in range(NSET):
                var iq = i * WQ
                var ik = i * WK
                var ig = i * WGA
                comptime if v <= 3:
                    ctx.enqueue_function[
                        f16_bmm_q2_gpu[T, INNER, OUTS, RB, DType.float32]
                    ](
                        dx1.unsafe_ptr(), dwq.unsafe_ptr() + iq,
                        dbq.unsafe_ptr(), dq.unsafe_ptr(), Int32(QO),
                        Int32(T),
                        grid_dim=(QO // OUTS, (T + RB - 1) // RB, 1),
                        block_dim=(128, 1, 1),
                    )
                elif v == 4:
                    ctx.enqueue_function[
                        f16_bmm_kv2_gpu[T, INNER, OUTS, RB, DType.float32]
                    ](
                        dx1.unsafe_ptr(), dwk.unsafe_ptr() + ik,
                        dwv.unsafe_ptr() + ik,
                        dbk.unsafe_ptr(), dbv.unsafe_ptr(),
                        dk.unsafe_ptr(), dv.unsafe_ptr(),
                        Int32(KVO), Int32(T),
                        grid_dim=(KVO // OUTS, (T + RB - 1) // RB, 1),
                        block_dim=(128, 1, 1),
                    )
                elif v <= 6:
                    ctx.enqueue_function[
                        f16_bmm_fat3_gpu[
                            T, INNER, OUTS, RB, True, DType.float32
                        ]
                    ](
                        dx1.unsafe_ptr(), dwo.unsafe_ptr() + iq,
                        do_.unsafe_ptr(), Int32(QO), Int32(T),
                        grid_dim=(QO // OUTS, (T + RB - 1) // RB, 1),
                        block_dim=(128, 1, 1),
                    )
                elif v == 7:
                    ctx.enqueue_function[
                        f16_bmm_dual3_gpu[
                            64, INNER, OUTS, RB, DType.float32
                        ]
                    ](
                        dx1.unsafe_ptr(), dwga.unsafe_ptr() + ig,
                        dwup.unsafe_ptr() + ig,
                        dga.unsafe_ptr(), dup.unsafe_ptr(),
                        dact.unsafe_ptr(), Int32(INNER_FF), Int32(T),
                        grid_dim=(INNER_FF // OUTS, (T + RB - 1) // RB, 1),
                        block_dim=(64, 1, 1),
                    )
                else:
                    ctx.enqueue_function[
                        f16_bmm_fat3_gpu[
                            T, INNER_FF, OUTS, RB, True, DType.float32
                        ]
                    ](
                        dx2.unsafe_ptr(), dwdn.unsafe_ptr() + ig,
                        ddn.unsafe_ptr(), Int32(QO), Int32(T),
                        grid_dim=(QO // OUTS, (T + RB - 1) // RB, 1),
                        block_dim=(128, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e3 / Float64(NSET)
            if per < best:
                best = per
        comptime KER = INNER_FF if v >= 8 else INNER
        comptime COLS = (
            KVO if v == 4 else
            (INNER_FF if (v == 7 or v >= 8) else QO)
        )
        var gf = 2.0 * Float64(COLS) * Float64(KER) * Float64(T) / 1e9
        var mb = Float64(COLS) * Float64(KER) * 2.0 / 1e6
        print("v", v, "O", OUTS, "RB", RB, "us:", best,
              "TF:", gf / best * 1e3, "wGB/s:", mb / best * 1e3)
