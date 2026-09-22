"""q4 整批通路（`matmul_q4_f32_rows`）的核级差分门。

被测的这一条：**一个块只解量化一次、`RB` 个 token 行共用**（行复用的杠杆，
`scripts/bench_q4_rows.mojo` 实测 RB=8 保守端 2.38×）。它换访存与指令数，
**不许换一个比特** —— 这是本门的主判据，也是它的全部意义。

三重判据：

① **逐位等于同一后端的单行通路**（`avx2.matmul_q4_f32` 逐行调）：这是最紧的一条，
   为零容差。
② **数值等于标量实现**（`quant.matmul_q4_f32`，f64 累加，另一套代码）：① 是**同源**
   比较 —— 若哪天 `lo` / `hi` 半字节装反了，两边一起反，①照样绿。判据沿用全项目
   统一的 `1e-5 × max(1, |ref|)`。
③ **哨兵 + 守卫区**：批越大越容易出现「某一行压根没被写」或「写到 `dst` 外面」。
   2026-09-22 修好的那个 bug 就是这一类 —— 调用点写成 `dst_p.unsafe_offset(r * out * 4)`
   （类型化指针按元素走，那个 `* 4` 是 RawPtr 按字节的习惯误抄过来）。门里把那个
   写法**原样复原**一遍，断言检查必须判红：这条同时也是那个 bug 的常驻红测。

三条负向对照（一个不会失败的门等于没门）：

- N1 针对①：改掉结果里的一个元素 → 逐位比较必须红。
- N2 针对③：用旧的错误步长（`rows` 的 4 倍）写一遍 → 哨兵/守卫检查必须红。
- N3 针对③的余数行：只写前 `RB` 的整数倍那几行（漏掉尾数行）→ 哨兵检查必须红。

⚠️ 形状故意造得**除不尽**：`cols` 取 3 / 5 个块而不是整十，`n_tok` 取 `RB` 的
整数倍 ±1（含 1、`RB-1`、`2·RB+1`），逼着尾数行那条后备路径进到达不到的地方。
"""

from std.testing import TestSuite, assert_equal

from alofa.core.dtype import DT_FP32
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena
from alofa.core.tensor import F32Ptr, TensorView, f32_data, shape2
from alofa.kernels.cpu.avx2 import Q4_BLOCK, Q4_BYTES, matmul_q4_f32_rows
from alofa.kernels.cpu.avx2 import matmul_q4_f32_bias as matmul_q4_f32_bias_vec
from alofa.kernels.cpu.avx2 import matmul_q4_f32_rows_band
from alofa.kernels.cpu.avx2 import matmul_q4_f32_bias_rows_band
from alofa.kernels.cpu.avx2 import matmul_q4_f32 as matmul_q4_f32_vec
from alofa.kernels.cpu.quant import matmul_q4_f32, matmul_q4_f32_bias
from alofa.kernels.cpu.quant import quantize_q4_0

comptime GUARD = 64  # `dst` 之后的守卫区（float 个数）
comptime SENTINEL = Float32(-999)
comptime REL_TOL = Float64(1e-5)  # 沿用全项目的统一判据


def ptr_of(raw: RawPtr, n: Int) raises -> F32Ptr:
    return f32_data(TensorView(raw, shape2(n, 1), DT_FP32))


def paint(p: F32Ptr, n: Int) raises -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = SENTINEL
        i += 1


def seed_w(p: F32Ptr, rows: Int, cols: Int) raises -> None:
    """权重：0.03125 是整的二进制小数，好量化也好复现。"""
    var i = 0
    while i < rows * cols:
        p[unsafe_offset=i] = Float32(((i * 37) % 101) - 50) * Float32(0.03125)
        i += 1


def seed_x(p: F32Ptr, n: Int) raises -> None:
    var i = 0
    while i < n:
        p[unsafe_offset=i] = Float32(((i * 17) % 53) - 26) * Float32(0.125)
        i += 1


def bitwise_diffs(a: F32Ptr, b: F32Ptr, n: Int) raises -> Int:
    var bad = 0
    var i = 0
    while i < n:
        if a[unsafe_offset=i] != b[unsafe_offset=i]:
            bad += 1
        i += 1
    return bad


def scalar_diffs(a: F32Ptr, b: F32Ptr, n: Int) raises -> Int:
    """容差之外的条目数：`|a-b| > 1e-5 × max(1, |b|)`，`b` 是标量参照。"""
    var bad = 0
    var i = 0
    while i < n:
        var av = Float64(a[unsafe_offset=i])
        var bv = Float64(b[unsafe_offset=i])
        var diff = av - bv if av > bv else bv - av
        var scale = bv if bv > Float64(1) else Float64(1)
        if scale < Float64(1):
            scale = Float64(1)
        if diff > REL_TOL * scale:
            bad += 1
        i += 1
    return bad


def sentinel_rows(p: F32Ptr, n_tok: Int, rows: Int) raises -> Int:
    """批里**整行仍是哨兵**的行数（= 这一行一次都没被写）。"""
    var cnt = 0
    var t = 0
    while t < n_tok:
        var whole = True
        var o = 0
        while o < rows:
            if p[unsafe_offset=t * rows + o] != SENTINEL:
                whole = False
            o += 1
        if whole:
            cnt += 1
        t += 1
    return cnt


def guard_dirty(p: F32Ptr, n_tok: Int, rows: Int) raises -> Int:
    var dirty = 0
    var i = n_tok * rows
    var end = i + GUARD
    while i < end:
        if p[unsafe_offset=i] != SENTINEL:
            dirty += 1
        i += 1
    return dirty


def check_case[RB: Int](n_tok: Int, rows: Int, cols: Int) raises -> Int:
    """一档形状；返回**坏条目数之和**（0 = 全过）。"""
    var area = n_tok * rows + GUARD
    var blocks_bytes = rows * (cols // Q4_BLOCK) * Q4_BYTES
    var arena = Arena(
        rows * cols * 4
        + blocks_bytes
        + n_tok * cols * 4
        + area * 4
        + n_tok * rows * 4
        + 8192
    )
    var w_raw = arena.alloc(rows * cols * 4)
    var b_raw = arena.alloc(blocks_bytes)
    var x_raw = arena.alloc(n_tok * cols * 4)
    var d_raw = arena.alloc(area * 4)  # 被测：带守卫区
    var r_raw = arena.alloc(n_tok * rows * 4)  # 参照：单行通路
    var s_raw = arena.alloc(n_tok * rows * 4)  # 参照：标量实现

    var pw = ptr_of(w_raw, rows * cols)
    seed_w(pw, rows, cols)
    quantize_q4_0(b_raw.unsafe_bitcast[UInt8](), pw, rows * cols)
    var x = ptr_of(x_raw, n_tok * cols)
    seed_x(x, n_tok * cols)
    var d = ptr_of(d_raw, area)
    var refv = ptr_of(r_raw, n_tok * rows)
    var refs = ptr_of(s_raw, n_tok * rows)

    # ---- 被测 ----
    paint(d, area)
    matmul_q4_f32_rows[RB](d, x, b_raw.unsafe_bitcast[UInt8](), n_tok, rows, cols)

    # ---- 参照：同一后端的单行通路 / 标量实现 ----
    var t = 0
    while t < n_tok:
        matmul_q4_f32_vec(
            refv.unsafe_offset(t * rows),
            x.unsafe_offset(t * cols),
            b_raw.unsafe_bitcast[UInt8](),
            rows,
            cols,
        )
        matmul_q4_f32(
            refs.unsafe_offset(t * rows),
            x.unsafe_offset(t * cols),
            b_raw.unsafe_bitcast[UInt8](),
            rows,
            cols,
        )
        t += 1

    var bad = 0
    bad += bitwise_diffs(d, refv, n_tok * rows)  # ①
    bad += scalar_diffs(d, refs, n_tok * rows)  # ②
    bad += sentinel_rows(d, n_tok, rows)  # ③
    bad += guard_dirty(d, n_tok, rows)  # ③
    if bad != 0:
        print(
            "    档 RB=",
            RB,
            " 批=",
            n_tok,
            " out=",
            rows,
            " inner=",
            cols,
            " → 坏条目 ",
            bad,
        )
    arena.keep_alive()
    return bad


def check_old_stride() raises -> Bool:
    """N2：2026-09-22 那个 bug 的步长。检查**必须**判红，返回 True。

    调用点写成 `dst_p.unsafe_offset(r * rows * 4)`（把 RawPtr 按字节的习惯抄到了
    类型化指针上）：第 0 行写在正确位置，`r` 大于等于 1 的那些行跑到 `4r·rows` 上 —— 于是
    预期区域内的那些行**一个都没被写**，而写出去的那几份落在守卫区里。
    """
    comptime rows = 32
    comptime cols = 96
    comptime n_tok = 4
    comptime area = n_tok * rows * 4 + GUARD  # 故意放大，让越界写留在自己的分配里
    var blocks_bytes = rows * (cols // Q4_BLOCK) * Q4_BYTES
    var arena = Arena(rows * cols * 4 + blocks_bytes + n_tok * cols * 4 + area * 4 + 8192)
    var w_raw = arena.alloc(rows * cols * 4)
    var b_raw = arena.alloc(blocks_bytes)
    var x_raw = arena.alloc(n_tok * cols * 4)
    var d_raw = arena.alloc(area * 4)
    var pw = ptr_of(w_raw, rows * cols)
    seed_w(pw, rows, cols)
    quantize_q4_0(b_raw.unsafe_bitcast[UInt8](), pw, rows * cols)
    var x = ptr_of(x_raw, n_tok * cols)
    seed_x(x, n_tok * cols)
    var d = ptr_of(d_raw, area)
    paint(d, area)

    var r = 0
    while r < n_tok:
        # ↓ 就是 bug 那两行（旧写法当化石留在这里）
        matmul_q4_f32_vec(
            d.unsafe_offset(r * rows * 4),
            x.unsafe_offset(r * cols),
            b_raw.unsafe_bitcast[UInt8](),
            rows,
            cols,
        )
        r += 1

    var caught = sentinel_rows(d, n_tok, rows) > 0 and guard_dirty(d, n_tok, rows) > 0
    print(
        "    N2 旧步长（`r*rows*4`）：未写行数 ",
        sentinel_rows(d, n_tok, rows),
        " 守卫区被踩 ",
        guard_dirty(d, n_tok, rows),
    )
    arena.keep_alive()
    return caught


def check_tail_skip() raises -> Bool:
    """N3：漏掉尾数行（只写 `RB` 的整数倍那几行）。哨兵检查**必须**判红。"""
    comptime rows = 24
    comptime cols = 96
    comptime n_tok = 10  # RB=4 → 8 + 2 尾数行
    comptime RB = 4
    comptime area = n_tok * rows + GUARD
    var blocks_bytes = rows * (cols // Q4_BLOCK) * Q4_BYTES
    var arena = Arena(rows * cols * 4 + blocks_bytes + n_tok * cols * 4 + area * 4 + 8192)
    var w_raw = arena.alloc(rows * cols * 4)
    var b_raw = arena.alloc(blocks_bytes)
    var x_raw = arena.alloc(n_tok * cols * 4)
    var d_raw = arena.alloc(area * 4)
    var pw = ptr_of(w_raw, rows * cols)
    seed_w(pw, rows, cols)
    quantize_q4_0(b_raw.unsafe_bitcast[UInt8](), pw, rows * cols)
    var x = ptr_of(x_raw, n_tok * cols)
    seed_x(x, n_tok * cols)
    var d = ptr_of(d_raw, area)
    paint(d, area)

    var full = n_tok - n_tok % RB
    matmul_q4_f32_rows[RB](d, x, b_raw.unsafe_bitcast[UInt8](), full, rows, cols)

    var missed = sentinel_rows(d, n_tok, rows)
    print("    N3 只写整数倍那几行：未写行数 ", missed, "（应为 ", n_tok - full, "）")
    arena.keep_alive()
    return missed == n_tok - full and missed > 0


def check_bias_band_case[RB: Int](
    n_tok: Int, rows: Int, cols: Int, bands: Int
) raises -> Int:
    """**带偏置**那一族（q/k/v）：分片的整批必须逐位等于「分片 × 逐行」。

    它走的是另一条核（`_matmul_q4_wide_rows`，f64 通道累加），与无偏置那条（`_matmul_q4_halves_rows`，
    f32 半块累加）**不是同一个** —— 所以这里不复用上面那个函数，而是把它整份重来一遍：
    两条通路各自都有可能"把它自己那份改对了、另一份改坏了"。
    """
    var area = n_tok * rows + GUARD
    var blocks_bytes = rows * (cols // Q4_BLOCK) * Q4_BYTES
    var arena = Arena(
        rows * cols * 4
        + blocks_bytes
        + n_tok * cols * 4
        + area * 4
        + n_tok * rows * 4
        + rows * 4
        + 8192
    )
    var w_raw = arena.alloc(rows * cols * 4)
    var b_raw = arena.alloc(blocks_bytes)
    var x_raw = arena.alloc(n_tok * cols * 4)
    var d_raw = arena.alloc(area * 4)
    var r_raw = arena.alloc(n_tok * rows * 4)
    var s_raw = arena.alloc(n_tok * rows * 4)
    var bias_raw = arena.alloc(rows * 4)
    var pw = ptr_of(w_raw, rows * cols)
    seed_w(pw, rows, cols)
    quantize_q4_0(b_raw.unsafe_bitcast[UInt8](), pw, rows * cols)
    var x = ptr_of(x_raw, n_tok * cols)
    seed_x(x, n_tok * cols)
    var bias = ptr_of(bias_raw, rows)
    var i = 0
    while i < rows:
        bias[unsafe_offset=i] = Float32((i * 5) % 17 - 8) * Float32(0.0625)
        i += 1
    var d = ptr_of(d_raw, area)
    var refv = ptr_of(r_raw, n_tok * rows)
    var refs = ptr_of(s_raw, n_tok * rows)
    paint(d, area)
    paint(refv, n_tok * rows)
    paint(refs, n_tok * rows)

    var row_bytes = cols // Q4_BLOCK * Q4_BYTES
    var per = rows // bands
    var rem = rows - per * bands
    var c0 = 0
    for k in range(bands):
        var cnt = per + (rem if k == bands - 1 else 0)
        var band_blocks = b_raw.unsafe_offset(c0 * row_bytes)
        # `blocks` 由核按字节偏、`dst` 由调用方按元素偏，**`bias` 也得跟着偏** ——
        # 漏偏的话第一带是对的、后面每个带都错，而差分门会把这件事说成"数不对"。
        matmul_q4_f32_bias_rows_band[RB](
            d.unsafe_offset(c0),
            x,
            band_blocks,
            n_tok,
            cnt,
            cols,
            bias.unsafe_offset(c0),
            rows,
        )
        var t = 0
        while t < n_tok:
            matmul_q4_f32_bias_vec(
                refv.unsafe_offset(t * rows + c0),
                x.unsafe_offset(t * cols),
                band_blocks,
                cnt,
                cols,
                bias.unsafe_offset(c0),
            )
            matmul_q4_f32_bias(
                refs.unsafe_offset(t * rows + c0),
                x.unsafe_offset(t * cols),
                band_blocks,
                cnt,
                cols,
                bias.unsafe_offset(c0),
            )
            t += 1
        c0 += cnt

    var bad = bitwise_diffs(d, refv, n_tok * rows)
    bad += scalar_diffs(d, refs, n_tok * rows)
    bad += sentinel_rows(d, n_tok, rows)
    bad += guard_dirty(d, n_tok, rows)
    if bad != 0:
        print(
            "    偏置 RB=",
            RB,
            " 批=",
            n_tok,
            " out=",
            rows,
            " 分片=",
            bands,
            " → 坏条目 ",
            bad,
        )
    arena.keep_alive()
    return bad


def check_band_case[RB: Int](n_tok: Int, rows: Int, cols: Int, bands: Int) raises -> Int:
    """分片的走法：每个输出行带一口气做完整批；必须与「分片 × 逐行」逐位一致。

    `dst` 里每个 token 相隔 `rows` 个元素，而这个带只占其中的 `cnt` 个 —— 于是
    这里的 `dst_stride` 与 `rows`（这一次算的行数）**不相等**，今天核里那两个
    步长必须分开传。
    """
    var area = n_tok * rows + GUARD
    var blocks_bytes = rows * (cols // Q4_BLOCK) * Q4_BYTES
    var arena = Arena(
        rows * cols * 4
        + blocks_bytes
        + n_tok * cols * 4
        + area * 4
        + n_tok * rows * 4
        + 8192
    )
    var w_raw = arena.alloc(rows * cols * 4)
    var b_raw = arena.alloc(blocks_bytes)
    var x_raw = arena.alloc(n_tok * cols * 4)
    var d_raw = arena.alloc(area * 4)
    var r_raw = arena.alloc(n_tok * rows * 4)
    var pw = ptr_of(w_raw, rows * cols)
    seed_w(pw, rows, cols)
    quantize_q4_0(b_raw.unsafe_bitcast[UInt8](), pw, rows * cols)
    var x = ptr_of(x_raw, n_tok * cols)
    seed_x(x, n_tok * cols)
    var d = ptr_of(d_raw, area)
    var refv = ptr_of(r_raw, n_tok * rows)
    paint(d, area)
    paint(refv, n_tok * rows)

    var row_bytes = cols // Q4_BLOCK * Q4_BYTES
    var per = rows // bands
    var rem = rows - per * bands
    var c0 = 0
    for k in range(bands):
        var cnt = per + (rem if k == bands - 1 else 0)
        # RawPtr 的 `unsafe_offset` 按**字节**走（另一个数量级的坑，见 avx2 文件头）
        var band_blocks = b_raw.unsafe_offset(c0 * row_bytes)
        matmul_q4_f32_rows_band[RB](
            d.unsafe_offset(c0), x, band_blocks, n_tok, cnt, cols, rows
        )
        var t = 0
        while t < n_tok:
            matmul_q4_f32_vec(
                refv.unsafe_offset(t * rows + c0),
                x.unsafe_offset(t * cols),
                band_blocks,
                cnt,
                cols,
            )
            t += 1
        c0 += cnt

    var bad = bitwise_diffs(d, refv, n_tok * rows)
    bad += sentinel_rows(d, n_tok, rows)
    bad += guard_dirty(d, n_tok, rows)
    if bad != 0:
        print(
            "    行带 RB=",
            RB,
            " 批=",
            n_tok,
            " out=",
            rows,
            " 分片=",
            bands,
            " → 坏条目 ",
            bad,
        )
    arena.keep_alive()
    return bad


def test_rows_match_single_row_calls() raises:
    """RB 行复用必须与逐个 token 行调用**逐位一致**，且每一行都被写、没写出去。

    形状除不尽是故意的：`cols` = 3 / 5 个块，批含 `< RB` 与 `2·RB+1`。
    """
    var bad = 0
    comptime RB8 = 8
    bad += check_case[8](1, 7, 96)  # 批 1：比 RB 小
    bad += check_case[8](7, 7, 96)  # RB-1
    bad += check_case[8](RB8, 7, 96)  # 正好一个组
    bad += check_case[8](RB8 + 1, 13, 160)  # 一组 + 1 行尾数
    bad += check_case[8](2 * RB8 + 1, 5, 96)  # 两组 + 1 行尾数
    bad += check_case[2](3, 9, 160)  # 换 RB
    bad += check_case[4](5, 9, 96)
    # 分片的走法（`dst` 里每个 token 隔 `rows` 个元素，这一次只写其中的一段输出行）
    bad += check_band_case[8](8, 9, 96, 3)  # out 除不尽 3 片
    bad += check_band_case[8](10, 7, 160, 2)  # 批不是 RB 的整数倍
    bad += check_band_case[4](1, 7, 96, 3)  # 批 = 1
    # 带偏置（q/k/v）那条：另一套累加结构，RB 也另行挑
    bad += check_bias_band_case[2](2, 9, 96, 3)
    bad += check_bias_band_case[2](10, 7, 160, 2)  # 批不是 RB 的整数倍
    bad += check_bias_band_case[4](1, 9, 96, 3)  # 批 = 1
    assert_equal(
        bad,
        0,
        "q4 整批通路与逐行调用不一致（或漏行 / 越界）",
    )


def test_negative_controls_stay_red() raises:
    """三条负向对照：改一个比特 / 旧的 4 倍步长 / 漏掉尾数行，检查必须判红。"""
    var n1 = 0
    comptime rows = 8
    comptime cols = 96
    var arena = Arena(rows * cols * 4 + 4 * 4096 + 8192)
    var a_raw = arena.alloc(rows * 4)
    var b_raw = arena.alloc(rows * 4)
    var pa = ptr_of(a_raw, rows)
    var pb = ptr_of(b_raw, rows)
    var i = 0
    while i < rows:
        pa[unsafe_offset=i] = Float32(i + 1) * Float32(0.5)
        pb[unsafe_offset=i] = Float32(i + 1) * Float32(0.5)
        i += 1
    assert_equal(bitwise_diffs(pa, pb, rows), 0, "两份相同的结果不该有差异")
    pa[unsafe_offset=rows - 1] = pa[unsafe_offset=rows - 1] + Float32(1e-3)
    # N1：改动必须被逐位比较抓住
    assert_equal(
        bitwise_diffs(pa, pb, rows) > 0,
        True,
        "N1 失效：改了一个元素，逐位比较却看不出来",
    )
    arena.keep_alive()

    assert_equal(
        check_old_stride(),
        True,
        "N2 失效：旧的 4 倍步长没被哨兵/守卫检查判红 —— 2026-09-22 那个 bug 会回来",
    )
    assert_equal(
        check_tail_skip(),
        True,
        "N3 失效：漏掉尾数行没被哨兵检查抓住",
    )


def test_stride_guard_catches_overlap() raises:
    """`dst_stride < rows` 会让相邻两个 token 的输出带叠在一起；必须拦下来。

    这条守的是**写坏别人的份**：分开传步长之后，这个错误不会再长成"数不对"，而是
    直接 overwrite 到别人的区域上 —— 差分门事后才看见，那时候守东西已经晚了。
    """
    var arena = Arena(4096)
    var raw = arena.alloc(4096)
    var p = ptr_of(raw, 256)
    var raised = False
    try:
        matmul_q4_f32_rows_band[8](p, p, raw.unsafe_bitcast[UInt8](), 4, 16, 96, 8)
    except err:
        raised = True
    assert_equal(
        raised,
        True,
        "dst_stride 比行数小却没有报错 —— 两个 token 的输出会叠在一起",
    )
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
