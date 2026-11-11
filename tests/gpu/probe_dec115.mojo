"""账本一百一十五-a：attn_dec_gpu（融合分组 decode 核，2 发射）vs
三趟（rs+vt2+comb2，3 发射）。正确性 = dst maxdiff（流式局部 softmax
基准重标度，期望 ~1e-6 级）；计时 = kv_len 512 交替轮取 min + TILES
扫描（8/16/32）。"""

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext
from alofa.kernels.cuda.attention import (
    attn_comb2_gpu,
    attn_comb3_gpu,
    attn_dec_gpu,
    attn_rs_gpu,
    attn_vt2_gpu,
)

comptime MB = 512
comptime NH = 14
comptime NKV = 2
comptime HD = 64
comptime GROUP = 7


def main() raises:
    var ctx = DeviceContext()
    var hq = ctx.enqueue_create_host_buffer[DType.float32](NH * HD)
    var hk = ctx.enqueue_create_host_buffer[DType.float32](NKV * HD)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](NKV * HD)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](HD)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](HD)
    var hks = ctx.enqueue_create_host_buffer[DType.float32](MB * NKV * HD)
    var hvs = ctx.enqueue_create_host_buffer[DType.float32](MB * NKV * HD)
    ctx.synchronize()
    for i in range(NH * HD):
        hq[i] = Float32(Float64((i * 13) % 23 - 11) / 16.0)
    for i in range(NKV * HD):
        hk[i] = Float32(Float64((i * 7) % 11 - 5) / 8.0)
        hv[i] = Float32(Float64((i * 5) % 17 - 8) / 16.0)
    for i in range(HD):
        hc[i] = Float32(Float64((i * 3) % 9 - 4) / 8.0)
        hs[i] = Float32(Float64((i * 11) % 13 - 6) / 8.0)
    for r in range(MB):
        for i in range(NKV * HD):
            var seed = Float64((r * 31 + i * 17) % 29 - 14)
            hks[r * NKV * HD + i] = Float32(seed / 16.0)
            hvs[r * NKV * HD + i] = Float32(seed / 32.0)

    var dq = ctx.enqueue_create_buffer[DType.float32](NH * HD)
    var dk = ctx.enqueue_create_buffer[DType.float32](NKV * HD)
    var dv = ctx.enqueue_create_buffer[DType.float32](NKV * HD)
    var dc = ctx.enqueue_create_buffer[DType.float32](HD)
    var dsn = ctx.enqueue_create_buffer[DType.float32](HD)
    var dqr = ctx.enqueue_create_buffer[DType.float32](NH * HD)
    var dks = ctx.enqueue_create_buffer[DType.float32](MB * NKV * HD)
    var dvs = ctx.enqueue_create_buffer[DType.float32](MB * NKV * HD)
    var dsc = ctx.enqueue_create_buffer[DType.float32](NH * MB)
    var dvp = ctx.enqueue_create_buffer[DType.float32](NH * 32 * HD)
    var dts = ctx.enqueue_create_buffer[DType.float64](NH * 32)
    var dtm = ctx.enqueue_create_buffer[DType.float32](NH * 32)
    var dst_r = ctx.enqueue_create_buffer[DType.float32](NH * HD)
    var dst_n = ctx.enqueue_create_buffer[DType.float32](NH * HD)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
    ctx.enqueue_copy(dst_buf=dc, src_buf=hc)
    ctx.enqueue_copy(dst_buf=dsn, src_buf=hs)
    ctx.enqueue_copy(dst_buf=dks, src_buf=hks)
    ctx.enqueue_copy(dst_buf=dvs, src_buf=hvs)
    ctx.synchronize()

    var hA = ctx.enqueue_create_host_buffer[DType.float32](NH * HD)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](NH * HD)

    @parameter
    for T in range(3):
        comptime TF = 8 * (1 << T)
        for kv_len in [1, 17, 64, 512]:
            ctx.enqueue_function[
                attn_rs_gpu[512, True, 8]
            ](
                dqr.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(),
                dv.unsafe_ptr(), dks.unsafe_ptr(), dvs.unsafe_ptr(),
                dc.unsafe_ptr(), dsn.unsafe_ptr(), dsc.unsafe_ptr(),
                Int32(NH), Int32(NKV), Int32(HD), Int32(kv_len),
                grid_dim=(NH, 8, 1), block_dim=(512, 1, 1),
            )
            ctx.enqueue_function[
                attn_vt2_gpu[512, 8]
            ](
                dsc.unsafe_ptr(), dvs.unsafe_ptr(), dvp.unsafe_ptr(),
                dts.unsafe_ptr(), Int32(NH), Int32(NKV), Int32(HD),
                Int32(kv_len),
                grid_dim=(NH, 8, 1), block_dim=(512, 1, 1),
            )
            ctx.enqueue_function[
                attn_comb2_gpu[8]
            ](
                dst_r.unsafe_ptr(), dvp.unsafe_ptr(), dts.unsafe_ptr(),
                Int32(NH), Int32(HD),
                grid_dim=(NH, 1, 1), block_dim=(512, 1, 1),
            )
            ctx.enqueue_function[
                attn_dec_gpu[512, TF, GROUP]
            ](
                dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
                dks.unsafe_ptr(), dvs.unsafe_ptr(), dc.unsafe_ptr(),
                dsn.unsafe_ptr(), dvp.unsafe_ptr(), dts.unsafe_ptr(),
                dtm.unsafe_ptr(), Int32(NH), Int32(NKV), Int32(kv_len),
                grid_dim=(NKV, TF, 1), block_dim=(512, 1, 1),
            )
            ctx.enqueue_function[
                attn_comb3_gpu[TF]
            ](
                dst_n.unsafe_ptr(), dvp.unsafe_ptr(), dts.unsafe_ptr(),
                dtm.unsafe_ptr(), Int32(NH), Int32(HD),
                grid_dim=(NH, 1, 1), block_dim=(512, 1, 1),
            )
            ctx.enqueue_copy(dst_buf=hA, src_buf=dst_r)
            ctx.enqueue_copy(dst_buf=hB, src_buf=dst_n)
            ctx.synchronize()
            var mx = 0.0
            for i in range(NH * HD):
                var d = Float64(hB[i]) - Float64(hA[i])
                if d < 0:
                    d = -d
                if d > mx:
                    mx = d
            print("T", TF, " kv_len", kv_len, " maxdiff:", mx)

    # 计时：kv_len 512，交替轮取 min（tri 3 发射 vs fused 2 发射）
    var kv_len = 512
    var btri = 1e30
    var bdec = InlineArray[Float64, 5](fill=1e30)
    for rep in range(6):
        @parameter
        for v in range(6):
            var t0 = monotonic_ns()
            for r in range(50):
                comptime if v == 0:
                    ctx.enqueue_function[
                        attn_rs_gpu[512, True, 8]
                    ](
                        dqr.unsafe_ptr(), dq.unsafe_ptr(), dk.unsafe_ptr(),
                        dv.unsafe_ptr(), dks.unsafe_ptr(), dvs.unsafe_ptr(),
                        dc.unsafe_ptr(), dsn.unsafe_ptr(), dsc.unsafe_ptr(),
                        Int32(NH), Int32(NKV), Int32(HD), Int32(kv_len),
                        grid_dim=(NH, 8, 1), block_dim=(512, 1, 1),
                    )
                    ctx.enqueue_function[
                        attn_vt2_gpu[512, 8]
                    ](
                        dsc.unsafe_ptr(), dvs.unsafe_ptr(), dvp.unsafe_ptr(),
                        dts.unsafe_ptr(), Int32(NH), Int32(NKV), Int32(HD),
                        Int32(kv_len),
                        grid_dim=(NH, 8, 1), block_dim=(512, 1, 1),
                    )
                    ctx.enqueue_function[
                        attn_comb2_gpu[8]
                    ](
                        dst_r.unsafe_ptr(), dvp.unsafe_ptr(),
                        dts.unsafe_ptr(), Int32(NH), Int32(HD),
                        grid_dim=(NH, 1, 1), block_dim=(512, 1, 1),
                    )
                else:
                    comptime if v == 1:
                        # 块数实验：TILES=160（320 块）SKIP=1 裸相
                        ctx.enqueue_function[
                            attn_dec_gpu[512, 160, GROUP, 1]
                        ](
                            dq.unsafe_ptr(), dk.unsafe_ptr(),
                            dv.unsafe_ptr(), dks.unsafe_ptr(),
                            dvs.unsafe_ptr(), dc.unsafe_ptr(),
                            dsn.unsafe_ptr(), dvp.unsafe_ptr(),
                            dts.unsafe_ptr(), dtm.unsafe_ptr(),
                            Int32(NH), Int32(NKV), Int32(kv_len),
                            grid_dim=(160, NKV, 1), block_dim=(512, 1, 1),
                        )
                    else:
                        comptime if v == 2:
                            ctx.enqueue_function[
                                attn_dec_gpu[512, 16, GROUP]
                            ](
                                dq.unsafe_ptr(), dk.unsafe_ptr(),
                                dv.unsafe_ptr(), dks.unsafe_ptr(),
                                dvs.unsafe_ptr(), dc.unsafe_ptr(),
                                dsn.unsafe_ptr(), dvp.unsafe_ptr(),
                                dts.unsafe_ptr(), dtm.unsafe_ptr(),
                                Int32(NH), Int32(NKV), Int32(kv_len),
                                grid_dim=(NKV, 16, 1), block_dim=(512, 1, 1),
                            )
                            # 计时拆分：本轮只发 dec 核（comb3 时另测）
                            if False:
                                ctx.enqueue_function[
                                    attn_comb3_gpu[16]
                                ](
                                    dst_n.unsafe_ptr(), dvp.unsafe_ptr(),
                                    dts.unsafe_ptr(), dtm.unsafe_ptr(),
                                    Int32(NH), Int32(HD),
                                    grid_dim=(NH, 1, 1), block_dim=(512, 1, 1),
                                )
                        else:
                            comptime if v == 3:
                                ctx.enqueue_function[
                                    attn_dec_gpu[512, 16, GROUP, 1]
                                ](
                                    dq.unsafe_ptr(), dk.unsafe_ptr(),
                                    dv.unsafe_ptr(), dks.unsafe_ptr(),
                                    dvs.unsafe_ptr(), dc.unsafe_ptr(),
                                    dsn.unsafe_ptr(), dvp.unsafe_ptr(),
                                    dts.unsafe_ptr(), dtm.unsafe_ptr(),
                                    Int32(NH), Int32(NKV), Int32(kv_len),
                                    grid_dim=(16, NKV, 1),
                                    block_dim=(512, 1, 1),
                                )
                            else:
                                comptime if v == 4:
                                    ctx.enqueue_function[
                                        attn_dec_gpu[512, 16, GROUP, 2]
                                    ](
                                        dq.unsafe_ptr(), dk.unsafe_ptr(),
                                        dv.unsafe_ptr(), dks.unsafe_ptr(),
                                        dvs.unsafe_ptr(), dc.unsafe_ptr(),
                                        dsn.unsafe_ptr(), dvp.unsafe_ptr(),
                                        dts.unsafe_ptr(), dtm.unsafe_ptr(),
                                        Int32(NH), Int32(NKV),
                                        Int32(kv_len),
                                        grid_dim=(16, NKV, 1),
                                        block_dim=(512, 1, 1),
                                    )
                                else:
                                    ctx.enqueue_function[
                                        attn_dec_gpu[512, 16, GROUP, 3]
                                    ](
                                        dq.unsafe_ptr(), dk.unsafe_ptr(),
                                        dv.unsafe_ptr(), dks.unsafe_ptr(),
                                        dvs.unsafe_ptr(), dc.unsafe_ptr(),
                                        dsn.unsafe_ptr(), dvp.unsafe_ptr(),
                                        dts.unsafe_ptr(), dtm.unsafe_ptr(),
                                        Int32(NH), Int32(NKV),
                                        Int32(kv_len),
                                        grid_dim=(16, NKV, 1),
                                        block_dim=(512, 1, 1),
                                    )
            ctx.synchronize()
            var per = Float64(monotonic_ns() - t0) / 1e3 / Float64(50)
            comptime if v == 0:
                if per < btri:
                    btri = per
            else:
                if per < bdec[v - 1]:
                    bdec[v - 1] = per
    print("tri:", btri, " skip1T160(320块):", bdec[0], " dec16裸核:", bdec[1],
          " skip1/2/3:", bdec[2], bdec[3], bdec[4])
