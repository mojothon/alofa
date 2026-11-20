"""账本一百六十二：前向主体逐核孤立微基准（t=28 生产形状）——
子打点 sync 口径失真，改用 20 连发单 sync ÷ 20、5 轮取最小的
孤立口径，归因 GEMM 本体 vs 卫星核。"""

from max.gpu.host import DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.kernels.cuda.basic import rmsnorm_gpu
from alofa.kernels.cuda.matmul import (
    x_to_f16_gpu,
    cat2_f16_gpu,
    pack_w_mma_gpu,
    f16_mma_down_spl_gpu,
    f16_mma_gemm2_spl_gpu,
    f16_mma_dual_pk_gpu,
    f16_mma_down_pk_gpu,
    f16_mma_head_gpu,
    f16_mma_head_pk_gpu,
    f32_adds_gpu,
    f32_mergb_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 28
    comptime INNER = 896
    comptime INTER = 4864
    comptime QOUT = 896
    comptime KVD = 128
    comptime HID = 896
    comptime REP = 20
    comptime ROUNDS = 5

    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](INNER)
    ctx.synchronize()
    for i in range(T * INNER):
        hx[i] = Float32(Float64((i * 13) % 23 - 11) / 8.0)
    for i in range(INNER):
        hw[i] = Float32(1)

    var dxf = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dx16 = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    var dattn16 = ctx.enqueue_create_buffer[DType.float16](T * HID)
    var dact16 = ctx.enqueue_create_buffer[DType.float16](T * INTER)
    var dnorm_w = ctx.enqueue_create_buffer[DType.float32](INNER)
    var dxnorm = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dxnorm16 = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    ctx.enqueue_copy(dst_buf=dxf, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dnorm_w, src_buf=hw)
    ctx.synchronize()

    var dq = ctx.enqueue_create_buffer[DType.float32](T * QOUT)
    var dk = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    var dqb = ctx.enqueue_create_buffer[DType.float32](QOUT)
    var dkb = ctx.enqueue_create_buffer[DType.float32](2 * KVD)
    var dqpart = ctx.enqueue_create_buffer[DType.float32](8 * T * QOUT)
    var dkpart = ctx.enqueue_create_buffer[DType.float32](8 * T * KVD)
    var dvpart = ctx.enqueue_create_buffer[DType.float32](8 * T * KVD)
    var dopart = ctx.enqueue_create_buffer[DType.float32](4 * T * HID)
    var dxb = ctx.enqueue_create_buffer[DType.float32](T * HID)
    var dgate = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dup = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dact = ctx.enqueue_create_buffer[DType.float32](T * INTER)
    var dlogits = ctx.enqueue_create_buffer[DType.float32](T * 151936)

    var dw_kv = ctx.enqueue_create_buffer[DType.float16](2 * KVD * INNER)
    var dwp_q = ctx.enqueue_create_buffer[DType.float16](QOUT * INNER)
    var dwp_kv = ctx.enqueue_create_buffer[DType.float16](2 * KVD * INNER)
    var dwp_o = ctx.enqueue_create_buffer[DType.float16](HID * INNER)
    var dwp_g = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwp_u = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwp_dn = ctx.enqueue_create_buffer[DType.float16](HID * INTER)
    var dembed = ctx.enqueue_create_buffer[DType.float16](151936 * INNER)
    ctx.synchronize()

    # 一次性 pack 其余权重（源数据无所谓，bench 计时不依赖数值）
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
        dwp_q.unsafe_ptr(), dwp_kv.unsafe_ptr(),
        grid_dim=(2 * KVD // 128, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
        dwp_q.unsafe_ptr(), dwp_o.unsafe_ptr(),
        grid_dim=(HID // 128, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
        dwp_u.unsafe_ptr(), dwp_g.unsafe_ptr(),
        grid_dim=(INTER // 128, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
        dwp_g.unsafe_ptr(), dwp_u.unsafe_ptr(),
        grid_dim=(INTER // 128, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INTER, 2, 8]](
        dwp_g.unsafe_ptr(), dwp_dn.unsafe_ptr(),
        grid_dim=(HID // 128, INTER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()
    print("probe162: setup done")

    var best = 0.0
    var t0 = 0
    var us = 0.0

    # ---- rmsnorm（t=28）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[rmsnorm_gpu](
                dxf.unsafe_ptr(), dnorm_w.unsafe_ptr(),
                dxnorm.unsafe_ptr(), dxnorm16.unsafe_ptr(),
                Int32(INNER), Int32(T), Float32(1e-6),
                grid_dim=(T, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        var t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[rmsnorm_gpu](
                dxf.unsafe_ptr(), dnorm_w.unsafe_ptr(),
                dxnorm.unsafe_ptr(), dxnorm16.unsafe_ptr(),
                Int32(INNER), Int32(T), Float32(1e-6),
                grid_dim=(T, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("rmsnorm_t28 =", best, "us")

    # ---- x_to_f16（25088 元素）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[x_to_f16_gpu](
                dxf.unsafe_ptr(), dx16.unsafe_ptr(), Int32(T * INNER),
                grid_dim=((T * INNER + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[x_to_f16_gpu](
                dxf.unsafe_ptr(), dx16.unsafe_ptr(), Int32(T * INNER),
                grid_dim=((T * INNER + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("x_to_f16_25088 =", best, "us")

    # ---- q2 spl（q GEMM 本体）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_down_spl_gpu[INNER, 112, 2, 8, 8]](
                dx16.unsafe_ptr(), dwp_q.unsafe_ptr(), dqpart.unsafe_ptr(),
                Int32(QOUT), Int32(T),
                grid_dim=(QOUT // 128, 2, 8), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_down_spl_gpu[INNER, 112, 2, 8, 8]](
                dx16.unsafe_ptr(), dwp_q.unsafe_ptr(), dqpart.unsafe_ptr(),
                Int32(QOUT), Int32(T),
                grid_dim=(QOUT // 128, 2, 8), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("q2_spl_q_gemm =", best, "us")

    # ---- mergb S=8（q merge+bias）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f32_mergb_gpu[8]](
                dq.unsafe_ptr(), dqpart.unsafe_ptr(), dqb.unsafe_ptr(),
                Int32(QOUT), Int32(T * QOUT),
                grid_dim=((T * QOUT + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f32_mergb_gpu[8]](
                dq.unsafe_ptr(), dqpart.unsafe_ptr(), dqb.unsafe_ptr(),
                Int32(QOUT), Int32(T * QOUT),
                grid_dim=((T * QOUT + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("mergb8_q =", best, "us")

    # ---- kv2 spl（kv GEMM 本体）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_gemm2_spl_gpu[INNER, 112, 8, 8]](
                dx16.unsafe_ptr(), dwp_kv.unsafe_ptr(),
                dkpart.unsafe_ptr(), dvpart.unsafe_ptr(),
                Int32(2 * KVD), Int32(T),
                grid_dim=(2 * KVD // 128, 2, 8), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_gemm2_spl_gpu[INNER, 112, 8, 8]](
                dx16.unsafe_ptr(), dwp_kv.unsafe_ptr(),
                dkpart.unsafe_ptr(), dvpart.unsafe_ptr(),
                Int32(2 * KVD), Int32(T),
                grid_dim=(2 * KVD // 128, 2, 8), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("kv2_spl_kv_gemm =", best, "us")

    # ---- o spl（o GEMM 本体）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_down_spl_gpu[INNER, 224, 2, 8, 4]](
                dattn16.unsafe_ptr(), dwp_o.unsafe_ptr(), dopart.unsafe_ptr(),
                Int32(HID), Int32(T),
                grid_dim=(HID // 128, 2, 4), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_down_spl_gpu[INNER, 224, 2, 8, 4]](
                dattn16.unsafe_ptr(), dwp_o.unsafe_ptr(), dopart.unsafe_ptr(),
                Int32(HID), Int32(T),
                grid_dim=(HID // 128, 2, 4), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("o_spl_o_gemm =", best, "us")

    # ---- adds S=4（o 残差合并）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f32_adds_gpu[4]](
                dxb.unsafe_ptr(), dopart.unsafe_ptr(), Int32(T * HID),
                grid_dim=((T * HID + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f32_adds_gpu[4]](
                dxb.unsafe_ptr(), dopart.unsafe_ptr(), Int32(T * HID),
                grid_dim=((T * HID + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("adds4_o =", best, "us")

    # ---- dual_pk（ff GEMM 本体，gate+up+swiglu）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 896, 8]](
                dx16.unsafe_ptr(), dwp_g.unsafe_ptr(), dwp_u.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 896, 8]](
                dx16.unsafe_ptr(), dwp_g.unsafe_ptr(), dwp_u.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("dual_pk_ff =", best, "us")

    # ---- down_pk（down GEMM 本体）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_down_pk_gpu[INTER, 896, 2, 8]](
                dact16.unsafe_ptr(), dwp_dn.unsafe_ptr(), dxb.unsafe_ptr(),
                Int32(HID), Int32(T),
                grid_dim=(HID // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_down_pk_gpu[INTER, 896, 2, 8]](
                dact16.unsafe_ptr(), dwp_dn.unsafe_ptr(), dxb.unsafe_ptr(),
                Int32(HID), Int32(T),
                grid_dim=(HID // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("down_pk_down =", best, "us")

    # ---- head MMA（对照新核）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_head_gpu[INNER, 608, 2, 8]](
                dx16.unsafe_ptr(), dembed.unsafe_ptr(), dlogits.unsafe_ptr(),
                Int32(151936), Int32(T),
                grid_dim=(151936 // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_head_gpu[INNER, 608, 2, 8]](
                dx16.unsafe_ptr(), dembed.unsafe_ptr(), dlogits.unsafe_ptr(),
                Int32(151936), Int32(T),
                grid_dim=(151936 // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("head_mma =", best, "us")

    print("probe162 done")

    # ==== 一百六十二-b：A/B 变体 ====
    var dwp_g4 = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwp_u4 = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwp_g2 = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    var dwp_u2 = ctx.enqueue_create_buffer[DType.float16](INTER * INNER)
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 4]](
        dwp_g.unsafe_ptr(), dwp_g4.unsafe_ptr(),
        grid_dim=(INTER // 64, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 4]](
        dwp_u.unsafe_ptr(), dwp_u4.unsafe_ptr(),
        grid_dim=(INTER // 64, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 2]](
        dwp_g.unsafe_ptr(), dwp_g2.unsafe_ptr(),
        grid_dim=(INTER // 32, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 2]](
        dwp_u.unsafe_ptr(), dwp_u2.unsafe_ptr(),
        grid_dim=(INTER // 32, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()

    # ---- dual_pk WARP=4（152 块）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 896, 4]](
                dx16.unsafe_ptr(), dwp_g4.unsafe_ptr(), dwp_u4.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 64, 2, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 896, 4]](
                dx16.unsafe_ptr(), dwp_g4.unsafe_ptr(), dwp_u4.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 64, 2, 1), block_dim=(128, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("dual_pk_ff_w4 =", best, "us")

    # ---- dual_pk WARP=2（304 块）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 896, 2]](
                dx16.unsafe_ptr(), dwp_g2.unsafe_ptr(), dwp_u2.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 32, 2, 1), block_dim=(64, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 896, 2]](
                dx16.unsafe_ptr(), dwp_g2.unsafe_ptr(), dwp_u2.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 32, 2, 1), block_dim=(64, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("dual_pk_ff_w2 =", best, "us")

    # ---- down_spl 生产对照（S=8/CH=608/OUTS=1）----
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_down_spl_gpu[INTER, 608, 1, 8, 8]](
                dact16.unsafe_ptr(), dwp_dn.unsafe_ptr(), dopart.unsafe_ptr(),
                Int32(HID), Int32(T),
                grid_dim=(HID // 64, 2, 8), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_down_spl_gpu[INTER, 608, 1, 8, 8]](
                dact16.unsafe_ptr(), dwp_dn.unsafe_ptr(), dopart.unsafe_ptr(),
                Int32(HID), Int32(T),
                grid_dim=(HID // 64, 2, 8), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("down_spl_down =", best, "us")

    print("probe162b done")

    # ==== 一百六十三：head pack 化 A/B ====
    var dwp_e = ctx.enqueue_create_buffer[DType.float16](151936 * INNER)
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
        dembed.unsafe_ptr(), dwp_e.unsafe_ptr(),
        grid_dim=(151936 // 128, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_head_pk_gpu[INNER, 608, 2, 8]](
                dx16.unsafe_ptr(), dwp_e.unsafe_ptr(), dlogits.unsafe_ptr(),
                Int32(151936), Int32(T),
                grid_dim=(151936 // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_head_pk_gpu[INNER, 608, 2, 8]](
                dx16.unsafe_ptr(), dwp_e.unsafe_ptr(), dlogits.unsafe_ptr(),
                Int32(151936), Int32(T),
                grid_dim=(151936 // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("head_pk =", best, "us")

    print("probe163 done")

    # ==== 一百六十四：CH/OUTS/S 扫描 ====
    # head_pk CH=896（INNER%CH==0 无余数路径；当前 608 有 18 步全局散读）
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_head_pk_gpu[INNER, 896, 2, 8]](
                dx16.unsafe_ptr(), dwp_e.unsafe_ptr(), dlogits.unsafe_ptr(),
                Int32(151936), Int32(T),
                grid_dim=(151936 // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_head_pk_gpu[INNER, 896, 2, 8]](
                dx16.unsafe_ptr(), dwp_e.unsafe_ptr(), dlogits.unsafe_ptr(),
                Int32(151936), Int32(T),
                grid_dim=(151936 // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("head_pk_ch896 =", best, "us")

    # head_pk OUTS=4（重 pack [896,4,8]）
    var dwp_e4 = ctx.enqueue_create_buffer[DType.float16](151936 * INNER)
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 4, 8]](
        dembed.unsafe_ptr(), dwp_e4.unsafe_ptr(),
        grid_dim=(151936 // 256, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_head_pk_gpu[INNER, 608, 4, 8]](
                dx16.unsafe_ptr(), dwp_e4.unsafe_ptr(), dlogits.unsafe_ptr(),
                Int32(151936), Int32(T),
                grid_dim=(151936 // 256, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_head_pk_gpu[INNER, 608, 4, 8]](
                dx16.unsafe_ptr(), dwp_e4.unsafe_ptr(), dlogits.unsafe_ptr(),
                Int32(151936), Int32(T),
                grid_dim=(151936 // 256, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("head_pk_outs4 =", best, "us")

    # dual_pk CH=448（INNER=896%448==0 无余数；smem 14KB）
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 448, 8]](
                dx16.unsafe_ptr(), dwp_g.unsafe_ptr(), dwp_u.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 448, 8]](
                dx16.unsafe_ptr(), dwp_g.unsafe_ptr(), dwp_u.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("dual_pk_ff_ch448 =", best, "us")

    # dual_pk CH=224（smem 7KB）
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 224, 8]](
                dx16.unsafe_ptr(), dwp_g.unsafe_ptr(), dwp_u.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_dual_pk_gpu[INNER, 224, 8]](
                dx16.unsafe_ptr(), dwp_g.unsafe_ptr(), dwp_u.unsafe_ptr(),
                dgate.unsafe_ptr(), dup.unsafe_ptr(),
                dact.unsafe_ptr(), dact16.unsafe_ptr(),
                Int32(INTER), Int32(T),
                grid_dim=(INTER // 128, 2, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("dual_pk_ff_ch224 =", best, "us")

    # q2_spl S=4（CH=224）+ mergb4 对照
    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f16_mma_down_spl_gpu[INNER, 224, 2, 8, 4]](
                dx16.unsafe_ptr(), dwp_q.unsafe_ptr(), dqpart.unsafe_ptr(),
                Int32(QOUT), Int32(T),
                grid_dim=(QOUT // 128, 2, 4), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f16_mma_down_spl_gpu[INNER, 224, 2, 8, 4]](
                dx16.unsafe_ptr(), dwp_q.unsafe_ptr(), dqpart.unsafe_ptr(),
                Int32(QOUT), Int32(T),
                grid_dim=(QOUT // 128, 2, 4), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("q2_spl_s4 =", best, "us")

    best = 1e30
    for _r in range(ROUNDS):
        for _w in range(2):
            ctx.enqueue_function[f32_mergb_gpu[4]](
                dq.unsafe_ptr(), dqpart.unsafe_ptr(), dqb.unsafe_ptr(),
                Int32(QOUT), Int32(T * QOUT),
                grid_dim=((T * QOUT + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        t0 = monotonic_ns()
        for _k in range(REP):
            ctx.enqueue_function[f32_mergb_gpu[4]](
                dq.unsafe_ptr(), dqpart.unsafe_ptr(), dqb.unsafe_ptr(),
                Int32(QOUT), Int32(T * QOUT),
                grid_dim=((T * QOUT + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        us = Float64(monotonic_ns() - t0) / 1e3 / REP
        if us < best:
            best = us
    print("mergb4_q =", best, "us")

    print("probe164 done")


def probe_variants() -> None:
    """一百六十二-b：A/B 变体——dual_pk WARP/占用扫描、down_spl CH 对照。"""
    pass
