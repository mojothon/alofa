"""账本一百七十二：down_spl q4 直读正确性 probe——A 链（生产现有：
q4_dequant_f16 -> pack_w_mma -> f16_mma_down_spl）vs B 链（新：
pack_w_mma_q4 + q4_pack_scales -> f16_mma_down_spl_q4），partial 归并
后逐位比对。数值必须逐位同（反量化数学不变，仅载入布局压缩）。"""

from std.math import ceildiv
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.elementwise import q4_dequant_f16_gpu
from alofa.kernels.cuda.matmul import (
    f16_mma_down_spl_gpu,
    f16_mma_down_spl_q4_gpu,
    f32_adds_gpu,
    pack_w_mma_gpu,
    pack_w_mma_q4_gpu,
    q4_pack_scales_gpu,
)


def main() raises:
    var ctx = DeviceContext()
    comptime INNER = 4864
    comptime OUTC = 896
    comptime CH = 608
    comptime OUTS = 1
    comptime WARP = 8
    comptime S = 8
    comptime T = 28
    comptime NBLK = OUTC * INNER // 32
    comptime COLS = WARP * OUTS * 8
    comptime NCB = OUTC // COLS
    comptime NPK = NCB * (INNER // 16) * COLS * 16

    # 伪造 q4 块流：scale 位模式取 0x2800-0x37FF（~0.03-0.5，无
    # denormal/inf），nib 均匀 0-15
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](NBLK * 18)
    var seed = UInt32(12345)
    for b in range(NBLK):
        seed = seed * 1664525 + 1013904223
        var sb = UInt16((seed >> 8) % 0x0FFF) + UInt16(0x2800)
        hb[b * 18] = UInt8(UInt32(sb) & 0xFF)
        hb[b * 18 + 1] = UInt8((UInt32(sb) >> 8) & 0xFF)
        for k in range(16):
            seed = seed * 1664525 + 1013904223
            var byte = UInt8((seed >> 10) & 0xFF)
            hb[b * 18 + 2 + k] = byte
    var db = ctx.enqueue_create_buffer[DType.uint8](NBLK * 18)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)

    # x：f16 随机（位模式 0x3800-0x3BFF ~ 0.5-1.5 与小负数）
    var hx = ctx.enqueue_create_host_buffer[DType.float16](T * INNER)
    for i in range(T * INNER):
        seed = seed * 1664525 + 1013904223
        var bits = UInt16((seed >> 6) & 0x03FF) + UInt16(0x3800)
        hx[i] = Float16(bits)
    var dx = ctx.enqueue_create_buffer[DType.float16](T * INNER)
    ctx.enqueue_copy(dst_buf=dx, src_buf=hx)

    # A 链：q4 -> f16 -> pack
    var dw16 = ctx.enqueue_create_buffer[DType.float16](OUTC * INNER)
    var dwp = ctx.enqueue_create_buffer[DType.float16](NPK)
    ctx.enqueue_function[q4_dequant_f16_gpu](
        db.unsafe_ptr(), dw16.unsafe_ptr(), Int32(OUTC * INNER),
        grid_dim=(ceildiv(OUTC * INNER, 256), 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[pack_w_mma_gpu[INNER, OUTS, WARP]](
        dw16.unsafe_ptr(), dwp.unsafe_ptr(),
        grid_dim=(NCB, INNER // 16, 1), block_dim=(128, 1, 1),
    )

    # B 链：q4 -> nibble pack + scales
    var dwpq = ctx.enqueue_create_buffer[DType.uint8](NPK // 2)
    var dwps = ctx.enqueue_create_buffer[DType.uint8](NBLK * 2)
    ctx.enqueue_function[pack_w_mma_q4_gpu[INNER, OUTS, WARP]](
        db.unsafe_ptr(), dwpq.unsafe_ptr(),
        grid_dim=(NCB, INNER // 16, 1), block_dim=(128, 1, 1),
    )
    ctx.enqueue_function[q4_pack_scales_gpu[NBLK]](
        db.unsafe_ptr(), dwps.unsafe_ptr(),
        grid_dim=(ceildiv(NBLK, 256), 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # 两链 forward：down_spl（f16 权重）vs down_spl_q4
    var partA = ctx.enqueue_create_buffer[DType.float32](S * T * OUTC)
    var partB = ctx.enqueue_create_buffer[DType.float32](S * T * OUTC)
    var dstA = ctx.enqueue_create_buffer[DType.float32](T * OUTC)
    var dstB = ctx.enqueue_create_buffer[DType.float32](T * OUTC)
    comptime GY = ceildiv(T, 16)
    ctx.enqueue_memset[DType.float32](dstA, 0.0)
    ctx.enqueue_memset[DType.float32](dstB, 0.0)
    ctx.synchronize()
    ctx.enqueue_function[
        f16_mma_down_spl_gpu[INNER, CH, OUTS, WARP, S]
    ](
        dx.unsafe_ptr(), dwp.unsafe_ptr(), partA.unsafe_ptr(),
        Int32(OUTC), Int32(T),
        grid_dim=(NCB, GY, S), block_dim=(WARP * 32, 1, 1),
    )
    ctx.enqueue_function[
        f16_mma_down_spl_q4_gpu[INNER, CH, OUTS, WARP, S]
    ](
        dx.unsafe_ptr(), dwpq.unsafe_ptr(), dwps.unsafe_ptr(),
        partB.unsafe_ptr(), Int32(OUTC), Int32(T),
        grid_dim=(NCB, GY, S), block_dim=(WARP * 32, 1, 1),
    )
    ctx.enqueue_function[f32_adds_gpu[S]](
        dstA.unsafe_ptr(), partA.unsafe_ptr(), Int32(T * OUTC),
        grid_dim=(ceildiv(T * OUTC, 256), 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[f32_adds_gpu[S]](
        dstB.unsafe_ptr(), partB.unsafe_ptr(), Int32(T * OUTC),
        grid_dim=(ceildiv(T * OUTC, 256), 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    var ha = ctx.enqueue_create_host_buffer[DType.float32](T * OUTC)
    var hbb = ctx.enqueue_create_host_buffer[DType.float32](T * OUTC)
    ctx.enqueue_copy(dst_buf=ha, src_buf=dstA)
    ctx.enqueue_copy(dst_buf=hbb, src_buf=dstB)
    ctx.synchronize()

    var mism = 0
    var maxd = 0.0
    for i in range(T * OUTC):
        var a = Float64(ha[i])
        var b = Float64(hbb[i])
        var d = a - b
        if d < 0.0:
            d = -d
        if d > 0.0:
            mism += 1
            if d > maxd:
                maxd = d
    print("elements =", T * OUTC)
    print("mismatch =", mism)
    print("maxdiff  =", maxd)
    if mism == 0:
        print("probe172 BIT-EXACT PASS")
    else:
        print("probe172 FAIL")
