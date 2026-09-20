#!/usr/bin/env python3
"""导出**专打向量化实现**的 q4_0 matmul 夹具（`tests/fixtures/.../q4vec/`）。

为什么要单独一份夹具，而不是扩 `q4/` 那份
------------------------------------------

`q4/` 那份（2026-09-17）是给解量化/量化/融合 matmul 的**正确性**用的，四个用例
全是真实投影 q/k/v/o，`cols` 一律 896 —— 也就是**每行正好 28 个块**。对标量实现
够了，对向量实现它有个静默缺陷：

    **28 能被 1、2、4、7、14、28 整除** —— 任何"按 2/4 块展开、余数走标量尾巴"
    的写法，四个用例全套绕过尾巴。而尾巴偏偏是 SIMD 最容易写错的一段（剩下的
    材料不够一个向量，容易顺手多读或少读一个块）。`rows` 同样是 128/896，都能被
    4 整除 → 行方向的尾部也没被看过。

所以这份夹具按"**把尾巴露出来**"挑形状：块/行取 1、3、5、7、11、17、28、152，
行数取 1、2、3、5、7、13 —— 对 2/4/8/16 任一展开倍数都有用例产生余数。

其中两条是给合成数据兜底的：`real13` 用**真实权重** q_w 前 13 行 × 真实激活
`norm_out[0,:]`；`down_like` 用 down_proj 的**真实内存长度**（cols=4864）× 真实
激活 `swiglu_out[0,:]`。合成数据再花哨，也可能漏掉真实权重那种"块间 amax 差两个
量级"的分布。

还得覆盖三种容易被向量化写错的数值情形：

- **全零块（d == 0）**：若把 d 当普通值参与比较/运算，容易出 NaN 或污染整行；
- **全部值踩在台阶顶（±amax）**：nibble 只会是 0 或 15，漏掉 `-8` 偏移的实现在
  这里会偏约 8/7 倍 —— 没这条，漏偏移可以一直躲在"输出还算流利"里；
- **动态范围极大**（一堆 1e-8 夹一个 1.0）：几乎整块都落在代表 0 的台阶上，期望
  值几乎由单个值的累加顺序决定，相对差会被放大。

期望值的算法与 `dump_q4_reference.py` 完全一致（解出的 fp32 → fp64 → 乘 → fp64
求和），两条路共用一个参照物，谁对谁错不靠记忆。

用法：
    /home/rontom/anaconda3/bin/python scripts/dump_q4_matmul_cases.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# 量化/解量化/读 f32 复用参考脚本：算法必须与它逐步一致，否则 `test_q4_parity`
# 会红 —— 没必要再抄一份。
from dump_q4_reference import (  # noqa: E402
    BLK_BYTES,
    BLOCK,
    FIX,
    dequant_q4_0,
    load_fp32,
    quantize_q4_0,
)

OUT = os.path.join(FIX, "q4vec")

LAYER0_TSV = os.path.join(FIX, "layer0", "tensors.tsv")
LAYER0_BIN = os.path.join(FIX, "layer0", "tensors.f32")


def _layer0(name: str) -> np.ndarray:
    return load_fp32(LAYER0_TSV, LAYER0_BIN, name)


def _alternating(rows: int, cols: int, scale: float = 0.5) -> np.ndarray:
    """+a, -a, +a, …：点积强烈相消，期望值接近 0。"""
    flat = np.arange(rows * cols, dtype=np.float32) % 2
    return np.where(flat == 0, scale, -scale).astype(np.float32).reshape(rows, cols)


def _clipped(rng, rows: int, cols: int) -> np.ndarray:
    """全部取 ±1：量化后 nibble 只会是 0 或 15。"""
    sign = np.where(rng.random((rows, cols)) < 0.5, -1.0, 1.0)
    return sign.astype(np.float32)


def _dynrange(rows: int, cols: int) -> np.ndarray:
    """一堆 1e-8 里每隔 17 个夹一个 1.0。"""
    w = np.full((rows, cols), 1e-8, dtype=np.float32)
    w.reshape(-1)[::17] = 1.0
    return w


def _build_weight(kind: str, rng, rows: int, cols: int) -> np.ndarray:
    if kind == "zeros":
        return np.zeros((rows, cols), dtype=np.float32)
    if kind == "gaussian":
        return (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
    if kind == "alternating":
        return _alternating(rows, cols)
    if kind == "clipped":
        return _clipped(rng, rows, cols)
    if kind == "dynrange":
        return _dynrange(rows, cols)
    if kind == "constant":
        return np.full((rows, cols), 0.3, dtype=np.float32)
    if kind == "real_q_w":
        full = _layer0("q_w")
        if full.shape[0] < rows:
            raise SystemExit("q_w 行数不足")
        return np.ascontiguousarray(full[:rows, :]).astype(np.float32)
    raise SystemExit("unknown weight kind " + kind)


def _build_act(kind: str, rng, cols: int) -> np.ndarray:
    if kind == "gaussian":
        return (rng.standard_normal(cols) * 0.03).astype(np.float32)
    if kind == "ones":
        return np.full(cols, 0.5, dtype=np.float32)
    if kind == "alternating":
        return _alternating(1, cols).reshape(-1)
    if kind == "tiny":
        return np.full(cols, 1e-6, dtype=np.float32)
    if kind == "real_norm_out":
        return np.ascontiguousarray(_layer0("norm_out")[0, :cols]).astype(np.float32)
    if kind == "real_swiglu":
        return np.ascontiguousarray(_layer0("swiglu_out")[0, :cols]).astype(np.float32)
    raise SystemExit("unknown act kind " + kind)


# （名字, 行, 列, 权重构造, 激活构造）
CASES = [
    ("one_block", 1, 32, "zeros", "gaussian"),
    ("odd_blocks", 2, 96, "constant", "ones"),
    ("tail5", 3, 160, "clipped", "gaussian"),
    ("tail7", 5, 224, "alternating", "alternating"),
    ("tail11", 7, 352, "dynrange", "ones"),
    ("tail17", 13, 544, "gaussian", "gaussian"),
    ("const_row", 3, 96, "clipped", "tiny"),
    ("real13", 13, 896, "real_q_w", "real_norm_out"),
    ("down_like", 2, 4864, "gaussian", "real_swiglu"),
]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    blocks_blob = bytearray()
    act_blob = bytearray()
    exp_blob = bytearray()
    cases_lines = []
    act_lines = []
    out_lines = []

    for idx, (name, rows, cols, w_kind, a_kind) in enumerate(CASES):
        if cols % BLOCK != 0:
            raise SystemExit("%s 的 cols=%d 不能被 32 整除" % (name, cols))
        # 每个用例一个独立的固定种子：加/删用例都不影响别的用例产出的字节。
        rng = np.random.default_rng(20260920 + idx)

        w = _build_weight(w_kind, rng, rows, cols)
        act = _build_act(a_kind, rng, cols)
        flat = np.ascontiguousarray(w).reshape(-1)

        raw = quantize_q4_0(flat)
        block_off = len(blocks_blob) // BLK_BYTES
        blocks_blob += raw

        act_off = len(act_blob) // 4
        act_blob += act.astype("<f4").tobytes()

        back = dequant_q4_0(raw, flat.size)
        prod = back.reshape(rows, cols).astype(np.float64) * act.astype(np.float64)
        expected = prod.sum(axis=1).astype(np.float64)
        out_off = len(exp_blob) // 8
        exp_blob += expected.astype("<f8").tobytes()

        n_blocks = flat.size // BLOCK
        cases_lines.append(
            "%s\t%d\t%d\t%d\t%d\t%s\n" % (name, rows, cols, block_off, n_blocks, w_kind)
        )
        act_lines.append("%s\t%d\t%d\t%s\n" % (name, act_off, cols, a_kind))
        out_lines.append("%s\t%d\t%d\n" % (name, out_off, rows))

        zero_blocks = int(np.sum(back.reshape(-1, BLOCK).max(axis=1) == 0))
        peak = float(np.max(np.abs(expected))) if expected.size else 0.0
        print(
            "  %-11s %3d×%-5d 块 %4d  |ref|max=%.6g  零块 %d"
            % (name, rows, cols, n_blocks, peak, zero_blocks)
        )

    with open(os.path.join(OUT, "blocks.bin"), "wb") as f:
        f.write(bytes(blocks_blob))
    with open(os.path.join(OUT, "act.f32"), "wb") as f:
        f.write(bytes(act_blob))
    with open(os.path.join(OUT, "out_expected.f64"), "wb") as f:
        f.write(bytes(exp_blob))

    def write_tsv(name, lines):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.writelines(lines)

    # cases.tsv 第 6 列（权重构造方式）只作存档用，测试不读它 —— 夹具里留着
    # "这一例是用什么构造的"，是为了让人不必反推脚本就能知道期望值的来路。
    write_tsv("cases.tsv", cases_lines)
    write_tsv("acts.tsv", act_lines)
    write_tsv("out_expected.tsv", out_lines)

    print(
        "q4vec 夹具已写入 %s（%d 用例，块流 %d 字节）"
        % (OUT, len(CASES), len(blocks_blob))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
