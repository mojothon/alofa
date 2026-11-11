"""账本一百一十七-b：q2 split-K 隔离对决（t=128 生产形状，48 套
冷转 >L2）。

背景：prefill f32 快路径五核中 kv2（S=8）/o/down（S=4）已 split-K
（一百一十三/一十四），唯独 q2 还是 f16_mma_gemm_pk 56 块（每 SM
半块，占 12.5% 藏不住冷权重 ~600 周期 DRAM 延迟——down_spl docstring
判词）。down_spl 核通用，q2 直用：S=4 → 224 块、S=8 → 448 块，
收口用新核 f32_mergb_gpu（mergb2 单路版）。
⚠️ 探针必须 48 套冷转（一百一十四-a 教训：L2 热假阳性）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.elementwise import swiglu_gpu
from alofa.kernels.cuda.matmul import (
    pack_w_mma_gpu, f16_mma_gemm_pk_gpu, f16_mma_down_spl_gpu,
    f16_mma_gemm2_spl_gpu, f16_mma_dual_pk_gpu,
    f32_mergb_gpu, f32_mergb2_gpu, f32_adds_gpu, f32_mergsw_gpu,
    f16_mma_gemm_pk_gpu,
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
    # x16 直接生成 f16（生产是 x_to_f16 转的，数值不影响计时）
    print("s0")
    var hx16 = ctx.enqueue_create_host_buffer[DType.float16](
        T * INNER_FF
    )
    print("s1")
    ctx.synchronize()
    for i in range(T * INNER_FF):
        hx16[i] = Float16(Float64((i * 13) % 23 - 11) / 16.0)
    var dx1 = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    var dx2 = ctx.enqueue_create_buffer[DType.float16](T * INNER_FF)
    ctx.enqueue_copy(dst_buf=dx1, src_buf=hx16)
    ctx.enqueue_copy(dst_buf=dx2, src_buf=hx16)
    ctx.synchronize()

    comptime WQ = QO * INNER
    comptime WKV = 2 * KVO * INNER
    comptime WGA = INNER_FF * INNER
    comptime WDN = INNER_FF * INNER
    var hwq = ctx.enqueue_create_host_buffer[DType.float16](NSET * WQ)
    var hwkv = ctx.enqueue_create_host_buffer[DType.float16](NSET * WKV)
    print("s2")
    var hwga = ctx.enqueue_create_host_buffer[DType.float16](NSET * WGA)
    var hwup = ctx.enqueue_create_host_buffer[DType.float16](NSET * WGA)
    var hwdn = ctx.enqueue_create_host_buffer[DType.float16](NSET * WDN)
    print("s3")
    var hbq = ctx.enqueue_create_host_buffer[DType.float32](QO)
    var hbkv = ctx.enqueue_create_host_buffer[DType.float32](2 * KVO)
    ctx.synchronize()
    print("s4")
    for i in range(QO):
        hbq[i] = Float32(0.01)
    for i in range(2 * KVO):
        hbkv[i] = Float32(0.02)
    for s in range(NSET):
        for i in range(WQ):
            hwq[s * WQ + i] = Float16(
                Float64((s * 31 + i * 17) % 29 - 14) / 16.0
            )
        for i in range(WKV):
            hwkv[s * WKV + i] = Float16(
                Float64((s * 23 + i * 11) % 27 - 13) / 16.0
            )
        for i in range(WGA):
            hwga[s * WGA + i] = Float16(
                Float64((s * 41 + i * 7) % 25 - 12) / 16.0
            )
            hwup[s * WGA + i] = Float16(
                Float64((s * 43 + i * 5) % 21 - 10) / 16.0
            )
        for i in range(WDN):
            hwdn[s * WDN + i] = Float16(
                Float64((s * 47 + i * 3) % 33 - 16) / 16.0
            )
    var dwq = ctx.enqueue_create_buffer[DType.float16](NSET * WQ)
    var dwkv = ctx.enqueue_create_buffer[DType.float16](NSET * WKV)
    var dwga = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwup = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwdn = ctx.enqueue_create_buffer[DType.float16](NSET * WDN)
    # 打包权重（q/kv 同 pack[896,2,8] 布局；down pack[4864,2,8]）
    var dwpq = ctx.enqueue_create_buffer[DType.float16](NSET * WQ)
    var dwpkv = ctx.enqueue_create_buffer[DType.float16](NSET * WKV)
    var dwpdn = ctx.enqueue_create_buffer[DType.float16](NSET * WDN)
    var dwpga = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwpup = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwpga1 = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwpup1 = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwpga16 = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwpup16 = ctx.enqueue_create_buffer[DType.float16](NSET * WGA)
    var dwpq2 = ctx.enqueue_create_buffer[DType.float16](NSET * WQ)
    print("s5")
    var dbq = ctx.enqueue_create_buffer[DType.float32](QO)
    var dbkv = ctx.enqueue_create_buffer[DType.float32](2 * KVO)
    ctx.enqueue_copy(dst_buf=dwq, src_buf=hwq)
    ctx.enqueue_copy(dst_buf=dwkv, src_buf=hwkv)
    ctx.enqueue_copy(dst_buf=dwga, src_buf=hwga)
    ctx.enqueue_copy(dst_buf=dwup, src_buf=hwup)
    ctx.enqueue_copy(dst_buf=dwdn, src_buf=hwdn)
    ctx.enqueue_copy(dst_buf=dbq, src_buf=hbq)
    ctx.enqueue_copy(dst_buf=dbkv, src_buf=hbkv)
    ctx.synchronize()
    print("s6")
    for i in range(NSET):
        ctx.enqueue_function[pack_w_mma_gpu[896, 2, 8]](
            dwq.unsafe_ptr() + i * WQ,
            dwpq.unsafe_ptr() + i * WQ,
            grid_dim=(QO // 128, INNER // 16, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[896, 2, 8]](
            dwkv.unsafe_ptr() + i * WKV,
            dwpkv.unsafe_ptr() + i * WKV,
            grid_dim=((2 * KVO) // 128, INNER // 16, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[896, 2, 8]](
            dwga.unsafe_ptr() + i * WGA,
            dwpga.unsafe_ptr() + i * WGA,
            grid_dim=(INNER_FF // 128, INNER // 16, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[896, 2, 8]](
            dwup.unsafe_ptr() + i * WGA,
            dwpup.unsafe_ptr() + i * WGA,
            grid_dim=(INNER_FF // 128, INNER // 16, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[4864, 2, 8]](
            dwdn.unsafe_ptr() + i * WDN,
            dwpdn.unsafe_ptr() + i * WDN,
            grid_dim=(QO // 128, INNER_FF // 16, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[896, 1, 8]](
            dwga.unsafe_ptr() + i * WGA,
            dwpga1.unsafe_ptr() + i * WGA,
            grid_dim=(INNER_FF // 64, INNER // 16, 1),
            block_dim=(128, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[896, 1, 8]](
            dwup.unsafe_ptr() + i * WGA,
            dwpup1.unsafe_ptr() + i * WGA,
            grid_dim=(INNER_FF // 64, INNER // 16, 1),
            block_dim=(128, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[896, 2, 16]](
            dwga.unsafe_ptr() + i * WGA,
            dwpga16.unsafe_ptr() + i * WGA,
            grid_dim=(INNER_FF // 256, INNER // 16, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[896, 2, 16]](
            dwup.unsafe_ptr() + i * WGA,
            dwpup16.unsafe_ptr() + i * WGA,
            grid_dim=(INNER_FF // 256, INNER // 16, 1),
            block_dim=(256, 1, 1),
        )
    ctx.synchronize()
    print("s7 packs done")
    ctx.enqueue_copy(dst_buf=dwpq2, src_buf=dwpq)
    ctx.synchronize()
    print("setup ok")

    var dq = ctx.enqueue_create_buffer[DType.float32](T * QO)
    var dqp = ctx.enqueue_create_buffer[DType.float32](8 * T * QO)
    var dk = ctx.enqueue_create_buffer[DType.float32](T * KVO)
    var dv = ctx.enqueue_create_buffer[DType.float32](T * KVO)
    var dkp = ctx.enqueue_create_buffer[DType.float32](8 * T * KVO)
    var dvp = ctx.enqueue_create_buffer[DType.float32](8 * T * KVO)
    var ddn = ctx.enqueue_create_buffer[DType.float32](T * QO)
    var ddnp = ctx.enqueue_create_buffer[DType.float32](4 * T * QO)
    var dga = ctx.enqueue_create_buffer[DType.float32](T * INNER_FF)
    var dup = ctx.enqueue_create_buffer[DType.float32](T * INNER_FF)
    var dact = ctx.enqueue_create_buffer[DType.float32](T * INNER_FF)
    var dgp = ctx.enqueue_create_buffer[DType.float32](4 * T * INNER_FF)
    var dup_p = ctx.enqueue_create_buffer[DType.float32](4 * T * INNER_FF)
    ctx.synchronize()

    # v0 q2_pk(56块)  v1 q2_spl4(224)  v2 q2_spl8(448)
    # v3 kv2_spl8(现状参照)  v4 dual_pk(现状参照)  v5 down_spl4(现状参照)
    @parameter
    for v in range(10):
        var best = 1e30
        for rep in range(ROUNDS):
            var t0 = monotonic_ns()
            for i in range(NSET):
                var wq = dwpq.unsafe_ptr() + i * WQ
                var wkv = dwpkv.unsafe_ptr() + i * WKV
                var wdn = dwpdn.unsafe_ptr() + i * WDN
                comptime if v == 0:
                    ctx.enqueue_function[
                        f16_mma_gemm_pk_gpu[896, 896, 2, 8, True]
                    ](
                        dx1.unsafe_ptr(), wq, dq.unsafe_ptr(),
                        dbq.unsafe_ptr(), Int32(QO), Int32(T),
                        grid_dim=(QO // 128, T // 16, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 1:
                    ctx.enqueue_function[
                        f16_mma_down_spl_gpu[896, 224, 2, 8, 4]
                    ](
                        dx1.unsafe_ptr(), wq, dqp.unsafe_ptr(),
                        Int32(QO), Int32(T),
                        grid_dim=(QO // 128, T // 16, 4),
                        block_dim=(256, 1, 1),
                    )
                    ctx.enqueue_function[f32_mergb_gpu[4]](
                        dq.unsafe_ptr(), dqp.unsafe_ptr(),
                        dbq.unsafe_ptr(), Int32(QO), Int32(T * QO),
                        grid_dim=((T * QO + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 2:
                    ctx.enqueue_function[
                        f16_mma_down_spl_gpu[896, 112, 2, 8, 8]
                    ](
                        dx1.unsafe_ptr(), wq, dqp.unsafe_ptr(),
                        Int32(QO), Int32(T),
                        grid_dim=(QO // 128, T // 16, 8),
                        block_dim=(256, 1, 1),
                    )
                    ctx.enqueue_function[f32_mergb_gpu[8]](
                        dq.unsafe_ptr(), dqp.unsafe_ptr(),
                        dbq.unsafe_ptr(), Int32(QO), Int32(T * QO),
                        grid_dim=((T * QO + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 3:
                    ctx.enqueue_function[
                        f16_mma_gemm2_spl_gpu[896, 112, 8, 8]
                    ](
                        dx1.unsafe_ptr(), wkv,
                        dkp.unsafe_ptr(), dvp.unsafe_ptr(),
                        Int32(KVO), Int32(T),
                        grid_dim=((2 * KVO) // 128, T // 16, 8),
                        block_dim=(256, 1, 1),
                    )
                    ctx.enqueue_function[f32_mergb2_gpu[8]](
                        dk.unsafe_ptr(), dkp.unsafe_ptr(),
                        dv.unsafe_ptr(), dvp.unsafe_ptr(),
                        dbkv.unsafe_ptr(), Int32(KVO), Int32(T * KVO),
                        grid_dim=((T * KVO + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 4:
                    ctx.enqueue_function[
                        f16_mma_dual_pk_gpu[896, 896, 8]
                    ](
                        dx1.unsafe_ptr(),
                        (dwpga.unsafe_ptr() + i * WGA),
                        (dwpup.unsafe_ptr() + i * WGA),
                        dga.unsafe_ptr(), dup.unsafe_ptr(),
                        dact.unsafe_ptr(), Int32(INNER_FF), Int32(T),
                        grid_dim=(INNER_FF // 128, T // 16, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 6 or v == 7:
                    comptime SS = 2 if v == 6 else 4
                    comptime CHS = 448 if v == 6 else 224
                    ctx.enqueue_function[
                        f16_mma_down_spl_gpu[896, CHS, 2, 8, SS]
                    ](
                        dx1.unsafe_ptr(),
                        (dwpga.unsafe_ptr() + i * WGA),
                        dgp.unsafe_ptr(),
                        Int32(INNER_FF), Int32(T),
                        grid_dim=(INNER_FF // 128, T // 16, SS),
                        block_dim=(256, 1, 1),
                    )
                    ctx.enqueue_function[
                        f16_mma_down_spl_gpu[896, CHS, 2, 8, SS]
                    ](
                        dx1.unsafe_ptr(),
                        (dwpup.unsafe_ptr() + i * WGA),
                        dup_p.unsafe_ptr(),
                        Int32(INNER_FF), Int32(T),
                        grid_dim=(INNER_FF // 128, T // 16, SS),
                        block_dim=(256, 1, 1),
                    )
                    ctx.enqueue_function[f32_mergsw_gpu[SS]](
                        dact.unsafe_ptr(), dgp.unsafe_ptr(),
                        dup_p.unsafe_ptr(), Int32(T * INNER_FF),
                        grid_dim=((T * INNER_FF + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                elif v == 9:
                    ctx.enqueue_function[
                        f16_mma_dual_pk_gpu[896, 896, 16]
                    ](
                        dx1.unsafe_ptr(),
                        (dwpga16.unsafe_ptr() + i * WGA),
                        (dwpup16.unsafe_ptr() + i * WGA),
                        dga.unsafe_ptr(), dup.unsafe_ptr(),
                        dact.unsafe_ptr(), Int32(INNER_FF), Int32(T),
                        grid_dim=(INNER_FF // 256, T // 16, 1),
                        block_dim=(512, 1, 1),
                    )
                elif v == 8:
                    ctx.enqueue_function[
                        f16_mma_gemm_pk_gpu[896, 896, 1, 8, False]
                    ](
                        dx1.unsafe_ptr(),
                        (dwpga1.unsafe_ptr() + i * WGA),
                        dga.unsafe_ptr(),
                        dbq.unsafe_ptr(), Int32(INNER_FF), Int32(T),
                        grid_dim=(INNER_FF // 64, T // 16, 1),
                        block_dim=(256, 1, 1),
                    )
                    ctx.enqueue_function[
                        f16_mma_gemm_pk_gpu[896, 896, 1, 8, False]
                    ](
                        dx1.unsafe_ptr(),
                        (dwpup1.unsafe_ptr() + i * WGA),
                        dup.unsafe_ptr(),
                        dbq.unsafe_ptr(), Int32(INNER_FF), Int32(T),
                        grid_dim=(INNER_FF // 64, T // 16, 1),
                        block_dim=(256, 1, 1),
                    )
                    ctx.enqueue_function[swiglu_gpu](
                        dga.unsafe_ptr(), dup.unsafe_ptr(),
                        dact.unsafe_ptr(), Int32(T * INNER_FF),
                        grid_dim=((T * INNER_FF + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    ctx.enqueue_function[
                        f16_mma_down_spl_gpu[4864, 1216, 2, 8, 4]
                    ](
                        dx2.unsafe_ptr(), wdn, ddnp.unsafe_ptr(),
                        Int32(QO), Int32(T),
                        grid_dim=(QO // 128, T // 16, 4),
                        block_dim=(256, 1, 1),
                    )
                    ctx.enqueue_function[f32_adds_gpu[4]](
                        ddn.unsafe_ptr(), ddnp.unsafe_ptr(),
                        Int32(T * QO),
                        grid_dim=((T * QO + 255) // 256, 1, 1),
                        block_dim=(256, 1, 1),
                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e3 / Float64(NSET)
            if per < best:
                best = per
        print("v", v, "us:", best)

    # 正确性抽查（v6 checksum 与 v4 同序同公式，差 ~1e-4 级）
    var hq = ctx.enqueue_create_host_buffer[DType.float32](T * QO)
    ctx.enqueue_copy(dst_buf=hq, src_buf=dq)
    ctx.synchronize()
    var csum = Float32(0)
    for i in range(1024):
        csum += hq[i]
    print("q2_spl8 checksum(1024):", csum)
