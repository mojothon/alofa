"""账本一百一十四-a：kv2 split-K（f16_mma_gemm2_spl_gpu + f32_mergb2_gpu）
vs 标量 vs 一百一十三-b 的 gemm2——**96 套权重轮转（44MB > 40MB L2）**。
一百一十三-b 的教训：4 套（1.8MB）全在 L2 里，「冷权重」是假的；流水内
每层 k/v 打包权重都被 ~37MB/层 权重流冲出 L2，真冷下 16 块藏不住延迟。
本探针 96 套轮转让每次读都真冷。缓冲先全建后全拷（避开 AsyncRT 交错悬崖）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.kernels.cuda.matmul import (
    cat2_f16_gpu,
    cat2_f32_gpu,
    f16_bmm_kv2_gpu,
    f16_mma_gemm2_pk_gpu,
    f16_mma_gemm2_spl_gpu,
    f32_mergb2_gpu,
    pack_w_mma_gpu,
    x_to_f16_gpu,
)

comptime T = 128
comptime INNER = 896
comptime KVD = 128
comptime NSET = 96
comptime REP = 3


def main() raises:
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](T * INNER)
    var hwk = ctx.enqueue_create_host_buffer[DType.float16](KVD * INNER)
    var hwv = ctx.enqueue_create_host_buffer[DType.float16](KVD * INNER)
    var hbk = ctx.enqueue_create_host_buffer[DType.float32](KVD)
    var hbv = ctx.enqueue_create_host_buffer[DType.float32](KVD)
    ctx.synchronize()
    for i in range(T * INNER):
        hx[i] = Float32(Float64((i * 13) % 23 - 11) / 16.0)
    for i in range(KVD * INNER):
        hwk[i] = Float16(Float64((i * 7) % 11 - 5) / 32.0)
        hwv[i] = Float16(Float64((i * 5) % 17 - 8) / 6.0)
    for i in range(KVD):
        hbk[i] = Float32(Float64((i * 3) % 9 - 4) / 8.0)
        hbv[i] = Float32(Float64((i * 11) % 13 - 6) / 8.0)

    var dx = ctx.enqueue_create_buffer[DType.float32](T * INNER)
    var dx16 = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    var dwk = ctx.enqueue_create_buffer[DType.float16](KVD * INNER)
    var dwv = ctx.enqueue_create_buffer[DType.float16](KVD * INNER)
    var dbk = ctx.enqueue_create_buffer[DType.float32](KVD)
    var dbv = ctx.enqueue_create_buffer[DType.float32](KVD)
    var kvraw = ctx.enqueue_create_buffer[DType.float16](2 * KVD * INNER)
    var wpkv = ctx.enqueue_create_buffer[DType.float16](2 * KVD * INNER)
    var bkv = ctx.enqueue_create_buffer[DType.float32](2 * KVD)
    var kb_ref = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    var vb_ref = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    var kb_new = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    var vb_new = ctx.enqueue_create_buffer[DType.float32](T * KVD)
    var part_a = ctx.enqueue_create_buffer[DType.float32](8 * T * KVD)
    var part_b = ctx.enqueue_create_buffer[DType.float32](8 * T * KVD)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)
    ctx.enqueue_copy(dst_buf=dwk, src_buf=hwk)
    ctx.enqueue_copy(dst_buf=dwv, src_buf=hwv)
    ctx.enqueue_copy(dst_buf=dbk, src_buf=hbk)
    ctx.enqueue_copy(dst_buf=dbv, src_buf=hbv)
    ctx.synchronize()

    # 96 套 w16 与打包权重：先全建（连续分配），再全拷/全打包
    var wks = List[DeviceBuffer[DType.float16]]()
    var wvs = List[DeviceBuffer[DType.float16]]()
    var wpks = List[DeviceBuffer[DType.float16]]()
    for s in range(NSET):
        wks.append(ctx.enqueue_create_buffer[DType.float16](KVD * INNER))
        wvs.append(ctx.enqueue_create_buffer[DType.float16](KVD * INNER))
        wpks.append(ctx.enqueue_create_buffer[DType.float16](2 * KVD * INNER))
    ctx.synchronize()
    for s in range(NSET):
        ctx.enqueue_copy(dst_buf=wks[s], src_buf=hwk)
        ctx.enqueue_copy(dst_buf=wvs[s], src_buf=hwv)
    ctx.synchronize()
    for s in range(NSET):
        ctx.enqueue_function[cat2_f16_gpu](
            wks[s].unsafe_ptr(), wvs[s].unsafe_ptr(), kvraw.unsafe_ptr(),
            Int32(KVD * INNER),
            grid_dim=(2 * KVD * INNER // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
            kvraw.unsafe_ptr(), wpks[s].unsafe_ptr(),
            grid_dim=(2 * KVD // 128, INNER // 16, 1), block_dim=(128, 1, 1),
        )
    # 正确性对照用的 wpkv 也要打包（一百一十三-b 探针在此踩过：
    # 只打 wpks 不打 wpkv，maxdiff 0.44 全是垃圾权重）
    ctx.enqueue_function[cat2_f16_gpu](
        dwk.unsafe_ptr(), dwv.unsafe_ptr(), kvraw.unsafe_ptr(),
        Int32(KVD * INNER),
        grid_dim=(2 * KVD * INNER // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, 2, 8]](
        kvraw.unsafe_ptr(), wpkv.unsafe_ptr(),
        grid_dim=(2 * KVD // 128, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[cat2_f32_gpu](
        dbk.unsafe_ptr(), dbv.unsafe_ptr(), bkv.unsafe_ptr(), Int32(KVD),
        grid_dim=(2 * KVD // 256 + 1, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[x_to_f16_gpu](
        dx.unsafe_ptr(), dx16.unsafe_ptr(), Int32(T * INNER),
        grid_dim=(T * INNER // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # 参照：标量 kv2（session 同款 tri_t=64）
    ctx.enqueue_function[f16_bmm_kv2_gpu[64, INNER, 2, 4, DType.float32]](
        dx.unsafe_ptr(), dwk.unsafe_ptr(), dwv.unsafe_ptr(),
        dbk.unsafe_ptr(), dbv.unsafe_ptr(),
        kb_ref.unsafe_ptr(), vb_ref.unsafe_ptr(),
        Int32(KVD), Int32(T),
        grid_dim=(KVD // 2, T // 4, 1), block_dim=(64, 1, 1),
    )
    # gemm2（一百一十三-b）
    ctx.enqueue_function[f16_mma_gemm2_pk_gpu[INNER, 896, 8]](
        dx16.unsafe_ptr(), wpkv.unsafe_ptr(),
        kb_new.unsafe_ptr(), vb_new.unsafe_ptr(), bkv.unsafe_ptr(),
        Int32(KVD), Int32(T),
        grid_dim=(2 * KVD // 128, T // 16, 1), block_dim=(256, 1, 1),
    )
    var hA = ctx.enqueue_create_host_buffer[DType.float32](T * KVD)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](T * KVD)
    var hC = ctx.enqueue_create_host_buffer[DType.float32](T * KVD)
    ctx.enqueue_copy(dst_buf=hA, src_buf=kb_ref)
    ctx.enqueue_copy(dst_buf=hB, src_buf=kb_new)
    ctx.synchronize()
    var mxk2 = 0.0
    for i in range(T * KVD):
        var d = Float64(hB[i]) - Float64(hA[i])
        if d < 0:
            d = -d
        if d > mxk2:
            mxk2 = d
    # spl S=8/CH=112 + 归并
    ctx.enqueue_function[f16_mma_gemm2_spl_gpu[INNER, 112, 8, 8]](
        dx16.unsafe_ptr(), wpkv.unsafe_ptr(),
        part_a.unsafe_ptr(), part_b.unsafe_ptr(),
        Int32(KVD), Int32(T),
        grid_dim=(2 * KVD // 128, T // 16, 8), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[f32_mergb2_gpu[8]](
        kb_new.unsafe_ptr(), part_a.unsafe_ptr(),
        vb_new.unsafe_ptr(), part_b.unsafe_ptr(),
        bkv.unsafe_ptr(), Int32(KVD), Int32(T * KVD),
        grid_dim=(T * KVD // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hC, src_buf=kb_new)
    ctx.synchronize()
    var mxk8 = 0.0
    for i in range(T * KVD):
        var d = Float64(hC[i]) - Float64(hA[i])
        if d < 0:
            d = -d
        if d > mxk8:
            mxk8 = d
    # spl S=4/CH=224 + 归并
    ctx.enqueue_function[f16_mma_gemm2_spl_gpu[INNER, 224, 8, 4]](
        dx16.unsafe_ptr(), wpkv.unsafe_ptr(),
        part_a.unsafe_ptr(), part_b.unsafe_ptr(),
        Int32(KVD), Int32(T),
        grid_dim=(2 * KVD // 128, T // 16, 4), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[f32_mergb2_gpu[4]](
        kb_new.unsafe_ptr(), part_a.unsafe_ptr(),
        vb_new.unsafe_ptr(), part_b.unsafe_ptr(),
        bkv.unsafe_ptr(), Int32(KVD), Int32(T * KVD),
        grid_dim=(T * KVD // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=hC, src_buf=kb_new)
    ctx.synchronize()
    var mxk4 = 0.0
    for i in range(T * KVD):
        var d = Float64(hC[i]) - Float64(hA[i])
        if d < 0:
            d = -d
        if d > mxk4:
            mxk4 = d
    print("gemm2 k maxdiff:", mxk2, " spl8 k:", mxk8, " spl4 k:", mxk4)

    # 计时：96 套轮转（真冷）。v=0 标量 / 1 gemm2 / 2 spl8 / 3 spl4
    var best = InlineArray[Float64, 4](fill=1e30)
    for rep in range(5):
        @parameter
        for v in range(4):
            var t0 = monotonic_ns()
            for r in range(REP):
                for s in range(NSET):
                    comptime if v == 0:
                        ctx.enqueue_function[
                            f16_bmm_kv2_gpu[64, INNER, 2, 4, DType.float32]
                        ](
                            dx.unsafe_ptr(), wks[s].unsafe_ptr(),
                            wvs[s].unsafe_ptr(), dbk.unsafe_ptr(),
                            dbv.unsafe_ptr(), kb_ref.unsafe_ptr(),
                            vb_ref.unsafe_ptr(), Int32(KVD), Int32(T),
                            grid_dim=(KVD // 2, T // 4, 1),
                            block_dim=(64, 1, 1),
                        )
                    elif v == 1:
                        ctx.enqueue_function[
                            f16_mma_gemm2_pk_gpu[INNER, 896, 8]
                        ](
                            dx16.unsafe_ptr(), wpks[s].unsafe_ptr(),
                            kb_new.unsafe_ptr(), vb_new.unsafe_ptr(),
                            bkv.unsafe_ptr(), Int32(KVD), Int32(T),
                            grid_dim=(2 * KVD // 128, T // 16, 1),
                            block_dim=(256, 1, 1),
                        )
                    elif v == 2:
                        ctx.enqueue_function[
                            f16_mma_gemm2_spl_gpu[INNER, 112, 8, 8]
                        ](
                            dx16.unsafe_ptr(), wpks[s].unsafe_ptr(),
                            part_a.unsafe_ptr(), part_b.unsafe_ptr(),
                            Int32(KVD), Int32(T),
                            grid_dim=(2 * KVD // 128, T // 16, 8),
                            block_dim=(256, 1, 1),
                        )
                        ctx.enqueue_function[f32_mergb2_gpu[8]](
                            kb_new.unsafe_ptr(), part_a.unsafe_ptr(),
                            vb_new.unsafe_ptr(), part_b.unsafe_ptr(),
                            bkv.unsafe_ptr(), Int32(KVD), Int32(T * KVD),
                            grid_dim=(T * KVD // 256, 1, 1),
                            block_dim=(256, 1, 1),
                        )
                    else:
                        ctx.enqueue_function[
                            f16_mma_gemm2_spl_gpu[INNER, 224, 8, 4]
                        ](
                            dx16.unsafe_ptr(), wpks[s].unsafe_ptr(),
                            part_a.unsafe_ptr(), part_b.unsafe_ptr(),
                            Int32(KVD), Int32(T),
                            grid_dim=(2 * KVD // 128, T // 16, 4),
                            block_dim=(256, 1, 1),
                        )
                        ctx.enqueue_function[f32_mergb2_gpu[4]](
                            kb_new.unsafe_ptr(), part_a.unsafe_ptr(),
                            vb_new.unsafe_ptr(), part_b.unsafe_ptr(),
                            bkv.unsafe_ptr(), Int32(KVD), Int32(T * KVD),
                            grid_dim=(T * KVD // 256, 1, 1),
                            block_dim=(256, 1, 1),
                        )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e6 / Float64(REP * NSET)
            if per < best[v]:
                best[v] = per
    print("scalar:", best[0], " gemm2:", best[1], " spl8:", best[2],
          " spl4:", best[3], " us/call")
