"""账本一百七十四-b：spec 行化三核恒等映射逐位同 probe——A 链
（原核 kv_store_bb / attn_rs_h_bb / attn_vt2_h_bb）vs B 链（spec 版
rowseq[b]=b、rowj[b]=0 恒等映射），全输出（q_rot/k_store/v_store/
scores/dst/dst16）逐位比对。恒等映射下必须逐位同。"""

from max.gpu.host import DeviceContext
from alofa.kernels.cuda.attention import (
    attn_rs_h_bb_gpu,
    attn_rs_h_bb_gpu_spec,
    attn_vt2_h_bb_gpu,
    attn_vt2_h_bb_gpu_spec,
    kv_store_bb_gpu,
    kv_store_bb_gpu_spec,
)


def main() raises:
    var ctx = DeviceContext()
    comptime B = 4
    comptime H = 8
    comptime HKV = 4
    comptime HD = 64
    comptime MKV = 96
    comptime KC = HKV * HD
    comptime ATTN_T = 512
    comptime QN = B * H * HD
    comptime SN = B * MKV * KC
    comptime SC = B * H * MKV

    seed = UInt32(987654321)

    # cos/sin 表（128 个位置，覆盖 pos+j）
    var hcos = ctx.enqueue_create_host_buffer[DType.float32](128 * HD)
    var hsin = ctx.enqueue_create_host_buffer[DType.float32](128 * HD)
    for i in range(128 * HD):
        seed = seed * 1664525 + 1013904223
        hcos[i] = Float32(Float64(Int32(seed >> 8)) / 4294967296.0 - 0.5)
        seed = seed * 1664525 + 1013904223
        hsin[i] = Float32(Float64(Int32(seed >> 8)) / 4294967296.0 - 0.5)
    var dcos = ctx.enqueue_create_buffer[DType.float32](128 * HD)
    var dsin = ctx.enqueue_create_buffer[DType.float32](128 * HD)
    ctx.enqueue_copy(dst_buf=dcos, src_buf=hcos)
    ctx.enqueue_copy(dst_buf=dsin, src_buf=hsin)

    # pos_arr（不等值）与 KV store 预填（历史位置）
    var hpos = ctx.enqueue_create_host_buffer[DType.int32](B)
    var posv = [10, 37, 5, 63]
    for i in range(B):
        hpos[i] = Int32(posv[i])
    var dpos = ctx.enqueue_create_buffer[DType.int32](B)
    ctx.enqueue_copy(dst_buf=dpos, src_buf=hpos)

    var hq = ctx.enqueue_create_host_buffer[DType.float32](QN)
    var hks = ctx.enqueue_create_host_buffer[DType.float32](B * KC)
    var hvs = ctx.enqueue_create_host_buffer[DType.float32](B * KC)
    var hst = ctx.enqueue_create_host_buffer[DType.float32](SN)
    for i in range(QN):
        seed = seed * 1664525 + 1013904223
        hq[i] = Float32(Float64(Int32(seed >> 9)) / 4194304.0 - 1.0)
    for i in range(B * KC):
        seed = seed * 1664525 + 1013904223
        hks[i] = Float32(Float64(Int32(seed >> 9)) / 4194304.0 - 1.0)
        seed = seed * 1664525 + 1013904223
        hvs[i] = Float32(Float64(Int32(seed >> 9)) / 4194304.0 - 1.0)
    for i in range(SN):
        seed = seed * 1664525 + 1013904223
        hst[i] = Float32(Float64(Int32(seed >> 9)) / 4194304.0 - 1.0)

    var dq = ctx.enqueue_create_buffer[DType.float32](QN)
    var dks = ctx.enqueue_create_buffer[DType.float32](B * KC)
    var dvs = ctx.enqueue_create_buffer[DType.float32](B * KC)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dks, src_buf=hks)
    ctx.enqueue_copy(dst_buf=dvs, src_buf=hvs)

    var dsta = ctx.enqueue_create_buffer[DType.float32](SN)
    var dstb = ctx.enqueue_create_buffer[DType.float32](SN)
    ctx.enqueue_copy(dst_buf=dsta, src_buf=hst)
    ctx.enqueue_copy(dst_buf=dstb, src_buf=hst)
    var vsta = ctx.enqueue_create_buffer[DType.float32](SN)
    var vstb = ctx.enqueue_create_buffer[DType.float32](SN)
    ctx.enqueue_copy(dst_buf=vsta, src_buf=hst)
    ctx.enqueue_copy(dst_buf=vstb, src_buf=hst)

    var qrota = ctx.enqueue_create_buffer[DType.float32](QN)
    var qrotb = ctx.enqueue_create_buffer[DType.float32](QN)
    var sca = ctx.enqueue_create_buffer[DType.float32](SC)
    var scb = ctx.enqueue_create_buffer[DType.float32](SC)

    # rs/vt2 共用缓冲（A、B 两链各一套，避免交叉污染）
    var vpart = ctx.enqueue_create_buffer[DType.float32](H * 8 * HD)
    var tsum = ctx.enqueue_create_buffer[DType.float64](H * 8)
    var dsta_o = ctx.enqueue_create_buffer[DType.float32](QN)
    var dsta_o16 = ctx.enqueue_create_buffer[DType.float16](QN)
    var dstb_o = ctx.enqueue_create_buffer[DType.float32](QN)
    var dstb_o16 = ctx.enqueue_create_buffer[DType.float16](QN)

    # 恒等 rowmap
    var hrs = ctx.enqueue_create_host_buffer[DType.int32](B)
    var hrj = ctx.enqueue_create_host_buffer[DType.int32](B)
    for i in range(B):
        hrs[i] = Int32(i)
        hrj[i] = Int32(0)
    var drs = ctx.enqueue_create_buffer[DType.int32](B)
    var drj = ctx.enqueue_create_buffer[DType.int32](B)
    ctx.enqueue_copy(dst_buf=drs, src_buf=hrs)
    ctx.enqueue_copy(dst_buf=drj, src_buf=hrj)

    # A 链：原核
    ctx.enqueue_function[kv_store_bb_gpu[ATTN_T]](
        qrota.unsafe_ptr(), dq.unsafe_ptr(), dks.unsafe_ptr(),
        dvs.unsafe_ptr(), dsta.unsafe_ptr(), vsta.unsafe_ptr(),
        dcos.unsafe_ptr(), dsin.unsafe_ptr(), dpos.unsafe_ptr(),
        Int32(H), Int32(HKV), Int32(HD), Int32(MKV),
        grid_dim=(H, 1, B), block_dim=(ATTN_T, 1, 1),
    )
    ctx.enqueue_function[attn_rs_h_bb_gpu[ATTN_T, 8, True]](
        qrota.unsafe_ptr(), dsta.unsafe_ptr(), sca.unsafe_ptr(),
        dpos.unsafe_ptr(), Int32(H), Int32(HKV), Int32(HD), Int32(MKV),
        grid_dim=(HKV, 1, B), block_dim=(ATTN_T, 1, 1),
    )
    ctx.enqueue_function[
        attn_vt2_h_bb_gpu[ATTN_T, 8, 1, True]
    ](
        sca.unsafe_ptr(), vsta.unsafe_ptr(), vpart.unsafe_ptr(),
        tsum.unsafe_ptr(), dpos.unsafe_ptr(), Int32(H), Int32(HKV),
        Int32(HD), Int32(MKV), dsta_o.unsafe_ptr(),
        dsta_o16.unsafe_ptr(),
        grid_dim=(HKV, 1, B), block_dim=(ATTN_T, 1, 1),
    )

    # B 链：spec 核恒等映射
    ctx.enqueue_function[kv_store_bb_gpu_spec[ATTN_T]](
        qrotb.unsafe_ptr(), dq.unsafe_ptr(), dks.unsafe_ptr(),
        dvs.unsafe_ptr(), dstb.unsafe_ptr(), vstb.unsafe_ptr(),
        dcos.unsafe_ptr(), dsin.unsafe_ptr(), dpos.unsafe_ptr(),
        Int32(H), Int32(HKV), Int32(HD), Int32(MKV),
        drs.unsafe_ptr(), drj.unsafe_ptr(),
        grid_dim=(H, 1, B), block_dim=(ATTN_T, 1, 1),
    )
    ctx.enqueue_function[attn_rs_h_bb_gpu_spec[ATTN_T, 8, True]](
        qrotb.unsafe_ptr(), dstb.unsafe_ptr(), scb.unsafe_ptr(),
        dpos.unsafe_ptr(), Int32(H), Int32(HKV), Int32(HD), Int32(MKV),
        drs.unsafe_ptr(), drj.unsafe_ptr(),
        grid_dim=(HKV, 1, B), block_dim=(ATTN_T, 1, 1),
    )
    ctx.enqueue_function[
        attn_vt2_h_bb_gpu_spec[ATTN_T, 8, 1, True]
    ](
        scb.unsafe_ptr(), vstb.unsafe_ptr(), vpart.unsafe_ptr(),
        tsum.unsafe_ptr(), dpos.unsafe_ptr(), Int32(H), Int32(HKV),
        Int32(HD), Int32(MKV),
        drs.unsafe_ptr(), drj.unsafe_ptr(),
        dstb_o.unsafe_ptr(), dstb_o16.unsafe_ptr(),
        grid_dim=(HKV, 1, B), block_dim=(ATTN_T, 1, 1),
    )
    ctx.synchronize()

    var mismatch = 0


    var hqa = ctx.enqueue_create_host_buffer[DType.float32](QN)
    var hqb = ctx.enqueue_create_host_buffer[DType.float32](QN)
    var hsa = ctx.enqueue_create_host_buffer[DType.float32](SN)
    var hsb = ctx.enqueue_create_host_buffer[DType.float32](SN)
    var hva = ctx.enqueue_create_host_buffer[DType.float32](SN)
    var hvb = ctx.enqueue_create_host_buffer[DType.float32](SN)
    var hca = ctx.enqueue_create_host_buffer[DType.float32](SC)
    var hcb = ctx.enqueue_create_host_buffer[DType.float32](SC)
    var hoa = ctx.enqueue_create_host_buffer[DType.float32](QN)
    var hob = ctx.enqueue_create_host_buffer[DType.float32](QN)
    var hoa16 = ctx.enqueue_create_host_buffer[DType.float16](QN)
    var hob16 = ctx.enqueue_create_host_buffer[DType.float16](QN)
    ctx.enqueue_copy(dst_buf=hqa, src_buf=qrota)
    ctx.enqueue_copy(dst_buf=hqb, src_buf=qrotb)
    ctx.enqueue_copy(dst_buf=hsa, src_buf=dsta)
    ctx.enqueue_copy(dst_buf=hsb, src_buf=dstb)
    ctx.enqueue_copy(dst_buf=hva, src_buf=vsta)
    ctx.enqueue_copy(dst_buf=hvb, src_buf=vstb)
    ctx.enqueue_copy(dst_buf=hca, src_buf=sca)
    ctx.enqueue_copy(dst_buf=hcb, src_buf=scb)
    ctx.enqueue_copy(dst_buf=hoa, src_buf=dsta_o)
    ctx.enqueue_copy(dst_buf=hob, src_buf=dstb_o)
    ctx.enqueue_copy(dst_buf=hoa16, src_buf=dsta_o16)
    ctx.enqueue_copy(dst_buf=hob16, src_buf=dstb_o16)
    ctx.synchronize()

    for i in range(QN):
        if hqa[i] != hqb[i]:
            mismatch += 1
        if Float64(hoa[i]) != Float64(hob[i]):
            mismatch += 1
        if Float64(hoa16[i]) != Float64(hob16[i]):
            mismatch += 1
    for i in range(SN):
        if hsa[i] != hsb[i]:
            mismatch += 1
        if hva[i] != hvb[i]:
            mismatch += 1
    for i in range(SC):
        if hca[i] != hcb[i]:
            mismatch += 1
    print("mismatch =", mismatch)
    if mismatch == 0:
        print("probe175 BIT-EXACT PASS")
    else:
        print("probe175 FAIL")
