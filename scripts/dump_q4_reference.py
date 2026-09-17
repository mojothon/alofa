#!/usr/bin/env python3
"""导出 q4_0 量化的参考 fixture（供 `tests/unit/test_q4_parity.mojo` 差分）。

为什么这个脚本和别的导出脚本不一样
----------------------------------

别的差分门里，参照物是**参考实现的真实输入/输出**（HF 的算子输出、真实
logits）。量化这里做不到，原因不是省事，是本质上不同：

- **量化（fp32 → q4_0）是我方离线的格式转换**，不是某个外部实现的行为。
  Mojo 侧**永远不量化**，只解量化 —— 所以这一步没有"另一家"可对照。
- **解量化（q4_0 → fp32）有外部事实可依**：块布局（32 值 / 18 字节、fp16
  缩放、低半字节在前、`v = (nibble - 8) * d`）是 GGML q4_0 的真实布局。
  于是这一半是**格式一致性检验**，不是自证。
- **融合 matmul 的期望值**由 fp32 参考权重解出的期望值与 fp64 累加求出。

所以本脚本里每一步数学都**显式写在代码里**，不用任何库的"帮我们算一下"
的便利函数（`np.round` 的取整方向、`astype(float16)` 的舍入都由 IEEE 规定，
是确定的）。账本里"量化 dequant"这一行的证据要照这样理解：**布局与解量化
有外部格式可依；量化步本身是自证的**，不许笼统写成"与 llama.cpp 一致"。

用法：
    python scripts/dump_q4_reference.py          # 小切片（layer0 的真实权重）
    python scripts/dump_q4_reference.py --full   # 全量权重（约 1.9 GB → 约 250 MB）
"""

import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(ROOT, "tests", "fixtures", "qwen2.5-0.5b")

# GGML q4_0：一个块 32 个值，18 字节。
#   [0:2]  fp16 缩放 d（小端）
#   [2:18] 16 字节，每字节装两个 4 位量化值：低半字节是第 j 个，高半字节是第 j+16 个
#   还原：v = (nibble - 8) * d
BLOCK = 32
BLK_BYTES = 18

# MSE 选 scale 的迭代轮数：收益几乎全在第一轮，两轮已经收敛到最后一两位。
MSE_ROUNDS = 2


def _quantise_with(blocks: np.ndarray, d: np.ndarray) -> np.ndarray:
    """用给定的块缩放 d 把每个值量化成 nibble 码（0..15）。

    `np.round` 取**最近、并列取偶**，与 Mojo 侧 `_quantise_one` 同一规则；
    换成截断会让每个值平均偏小半个台阶。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(d[:, None] != 0, blocks / d[:, None], np.float32(0.0))
    return np.clip(np.round(scaled) + np.float32(8.0), 0, 15).astype(np.uint8)


def quantize_q4_0(flat: np.ndarray) -> bytes:
    """fp32 一维数组 → q4_0 块流。长度必须是 32 的倍数。

    块布局与解量化规则**一点没变**（仍是 18 字节 32 值、fp16 缩放、低半字节
    在前、`v = (nibble - 8) * d`），变的是**缩放因子怎么选**：

    旧（朴素 q4_0）：
        amax = max(|x|)                    块内绝对值的最大值
        d    = amax / 7                    让最大值恰好够到 nibble 15（台阶 +7）
        q    = clamp(round(x / d) + 8, 0, 15)

    新（按 MSE 选）：先用 `amax / 7` 起个头量化一次，然后对**固定的 q**，令
    `s = q - 8`，把重建误差 `Σ (x - d·s)²` 对 d 求极小，得最小二乘解

        d* = Σ(x·s) / Σ(s²)

    用 d* 重新量化，重复 `MSE_ROUNDS` 轮（收益几乎全在第一轮，两轮已收敛到
    最后一两位）。

    为什么值得：朴素写法为了"让最大值够得着台阶顶"，把台阶钉在分布最稀疏的
    地方，块里绝大多数小值因此只有很低的分辨率（实测 q/k/v/o 权重相对 L2 误差
    约 10%）。MSE 解允许最大值被裁到台阶顶（裁掉一点点，比让整个块变粗便宜），
    把台阶挪到分布真正密集的地方。**量化仍然是有损的**，只是损在更划算的地方。

    两个必须与 Mojo 侧（`kernels/cpu/quant.mojo`）逐字节一致的地方：
    - 量化用的 d 是 **fp32 的 d**，存进块的才是 fp16 的那个（先用 fp16 的 d
      去除，量化结果会整体偏大）；
    - `round` 取**最近、并列取偶**（IEEE 默认），截断会让每个值平均偏小半个台阶。
    """
    flat = np.ascontiguousarray(flat, dtype=np.float32)
    if flat.size % BLOCK != 0:
        raise ValueError("长度必须是 %d 的倍数，收到 %d" % (BLOCK, flat.size))
    n_blocks = flat.size // BLOCK
    blocks = flat.reshape(n_blocks, BLOCK)
    x64 = blocks.astype(np.float64)

    amax = np.max(np.abs(blocks), axis=1)
    d = np.where(amax > 0, amax / np.float32(7.0), np.float32(0.0)).astype(np.float32)

    for _ in range(MSE_ROUNDS):
        s = _quantise_with(blocks, d).astype(np.float64) - 8.0
        # 顺序累加：Mojo 侧是 j = 0..31 的朴素循环，这里用 accumulate 保证同序
        # （顺带说明为什么敢这么依赖：x·s 在 fp64 里是精确的 —— fp32 的 24 位
        # 尾数乘上 |s| ≤ 8 的 3 位，用不了 53 位 —— 所以即便编译器把它收缩成
        # FMA，两边算出的也是同一个数，剩下的只有相加顺序这一个自由度）。
        num = np.add.accumulate(x64 * s, axis=1)[:, -1]
        den = np.add.accumulate(s * s, axis=1)[:, -1]
        ok = (den > 0.0) & (d > np.float32(0.0))
        d = np.where(
            ok, num / np.where(ok, den, 1.0), d.astype(np.float64)
        ).astype(np.float32)

    d16 = d.astype(np.float16)
    q = _quantise_with(blocks, d)

    lo = q[:, 0:16]
    hi = q[:, 16:32]
    packed = (lo & np.uint8(0x0F)) | ((hi & np.uint8(0x0F)) << np.uint8(4))

    # 块是**交错**的：每个块 18 字节 = [fp16 d][16 字节 packed]。
    # 先把所有缩放因子排在前面会让"按 18 字节读块"的解法读到垃圾 —— 只有一个
    # 块时恰好对得上，所以这种错不会被小样例发现。
    out = np.empty((n_blocks, BLK_BYTES), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:BLK_BYTES] = packed
    return out.tobytes()


def dequant_q4_0(raw: bytes, n: int) -> np.ndarray:
    """解量化的**独立实现**：重排字节、按布局还原。

    与 `quantize_q4_0` 里写块的代码不共享任何中间变量 —— 它从字节流出发，
    走的是"读盘"这条路。这样测试里 Mojo 对上它，等于对上"一个从字节流
    独立解出来的人"，而不是对上"量化时的中间数组"。
    """
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, BLK_BYTES)
    d = arr[:, 0:2].copy().view(np.float16).astype(np.float32)
    packed = arr[:, 2:BLK_BYTES].astype(np.uint8)
    lo = (packed & np.uint8(0x0F)).astype(np.float32)
    hi = (packed >> np.uint8(4)).astype(np.float32)
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    values = (q - np.float32(8.0)) * d.reshape(-1, 1)
    return values.reshape(-1)[:n].astype(np.float32)


def read_tensors_tsv(path: str):
    """`name <tab> shape <tab> offset <tab> numel`（偏移量以**字节**计）。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, shape, offset, numel = line.split("\t")
            rows.append((name, shape, int(offset), int(numel)))
    return rows


def load_fp32(tsv_path: str, bin_path: str, name: str) -> np.ndarray:
    for n, shape, offset, numel in read_tensors_tsv(tsv_path):
        if n != name:
            continue
        dims = [int(x) for x in shape.split("x")]
        with open(bin_path, "rb") as f:
            f.seek(offset)
            raw = f.read(numel * 4)
        return np.frombuffer(raw, dtype="<f4").reshape(dims).astype(np.float32)
    raise KeyError(name)


def fmt(v: float) -> str:
    return "%.12f" % v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full",
        action="store_true",
        help="量化全量权重（约 1.9 GB）到 weights_q4/，供端到端门使用",
    )
    args = ap.parse_args()

    if args.full:
        return dump_full()

    src_tsv = os.path.join(FIX, "layer0", "tensors.tsv")
    src_bin = os.path.join(FIX, "layer0", "tensors.f32")
    out_dir = os.path.join(FIX, "q4")
    os.makedirs(out_dir, exist_ok=True)

    # 四个真实投影矩阵（layer 0）：in 维都是 896，都能被 32 整除。
    cases = [("q_w", 896, 896), ("k_w", 128, 896), ("v_w", 128, 896), ("o_w", 896, 896)]

    # 激活向量取 layer 0 第一个位置的真实归一化输出（896 维），
    # 于是点积的参照物是"真实权重 × 真实激活"，不是随手造的数。
    norm_out = load_fp32(src_tsv, src_bin, "norm_out")
    act = norm_out[0, :].astype(np.float32)
    assert act.shape[0] == 896

    blocks_blob = bytearray()
    deq_blob = bytearray()
    exp_blob = bytearray()
    orig_blob = bytearray()
    cases_lines = []
    deq_lines = []
    out_lines = []
    orig_lines = []

    for name, rows, cols in cases:
        w = load_fp32(src_tsv, src_bin, name)
        if w.shape != (rows, cols):
            raise SystemExit("%s 形状 %s 与预期 (%d,%d) 不符" % (name, w.shape, rows, cols))
        flat = np.ascontiguousarray(w).reshape(-1)
        raw = quantize_q4_0(flat)
        n_blocks = flat.size // BLOCK
        block_off = len(blocks_blob) // BLK_BYTES
        blocks_blob += raw

        back = dequant_q4_0(raw, flat.size)
        deq_off = len(deq_blob) // 4
        deq_blob += back.astype("<f4").tobytes()

        # 融合 matmul 的期望值：解出的权重 × 激活，fp64 累加。
        prod = back.reshape(rows, cols).astype(np.float64) * act.astype(np.float64)
        expected = prod.sum(axis=1).astype(np.float64)
        out_off = len(exp_blob) // 8
        exp_blob += expected.astype("<f8").tobytes()

        cases_lines.append(
            "%s\t%d\t%d\t%d\t%d\n" % (name, rows, cols, block_off, n_blocks)
        )
        deq_lines.append("%s\t%d\t%d\n" % (name, deq_off, flat.size))
        out_lines.append("%s\t%d\t%d\n" % (name, out_off, rows))

        # 原始 fp32 权重一并落盘：测试要能证明"量化确实是有损的"，否则
        # "解量化逐位相等"有可能只是因为压根没量化、直接抄了原值。
        orig_off = len(orig_blob) // 4
        orig_blob += np.ascontiguousarray(flat).astype("<f4").tobytes()
        orig_lines.append("%s\t%d\t%d\n" % (name, orig_off, flat.size))

        # 量化误差：解出的权重与原始 fp32 权重差多少（只报告，不作断言）。
        err = float(np.max(np.abs(back - flat)))
        rel = float(np.linalg.norm(back - flat) / np.linalg.norm(flat))
        print("  %s  %d×%d  块 %d  最大绝对误差 %.6f  相对误差 %.4f%%" % (
            name, rows, cols, n_blocks, err, rel * 100.0))

    with open(os.path.join(out_dir, "blocks.bin"), "wb") as f:
        f.write(bytes(blocks_blob))
    with open(os.path.join(out_dir, "dequant.f32"), "wb") as f:
        f.write(bytes(deq_blob))
    with open(os.path.join(out_dir, "out_expected.f64"), "wb") as f:
        f.write(bytes(exp_blob))
    with open(os.path.join(out_dir, "act.f32"), "wb") as f:
        f.write(act.astype("<f4").tobytes())
    with open(os.path.join(out_dir, "orig.f32"), "wb") as f:
        f.write(bytes(orig_blob))

    def write_tsv(name, lines):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.writelines(lines)

    write_tsv("cases.tsv", cases_lines)
    write_tsv("dequant.tsv", deq_lines)
    write_tsv("out_expected.tsv", out_lines)
    write_tsv("orig.tsv", orig_lines)
    with open(os.path.join(out_dir, "act.tsv"), "w", encoding="utf-8") as f:
        f.write("norm_out_row0\t0\t%d\n" % act.shape[0])

    print("q4 切片 fixture 已写入 %s" % out_dir)
    return 0


def dump_full() -> int:
    """把全量权重量化成 q4_0，供端到端门使用。

    只量化二维矩阵（rows × cols，cols 能被 32 整除）；一维的 norm 权重与
    embedding 保持 fp32 —— 它们占比极小，量化只会平白损伤质量。
    """
    src_dir = os.path.join(FIX, "weights")
    src_tsv = os.path.join(src_dir, "tensors.tsv")
    src_bin = os.path.join(src_dir, "tensors.f32")
    out_dir = os.path.join(FIX, "weights_q4")
    os.makedirs(out_dir, exist_ok=True)

    rows_all = read_tensors_tsv(src_tsv)
    q4_blob = open(os.path.join(out_dir, "tensors.q4"), "wb")
    fp32_blob = open(os.path.join(out_dir, "tensors.f32"), "wb")
    index = []
    q4_bytes = 0
    fp32_elems = 0

    for name, shape, offset, numel in rows_all:
        dims = [int(x) for x in shape.split("x")]
        if len(dims) == 2 and dims[1] % BLOCK == 0:
            with open(src_bin, "rb") as f:
                f.seek(offset)
                raw = f.read(numel * 4)
            flat = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            blob = quantize_q4_0(flat)
            q4_blob.write(blob)
            index.append((name, shape, "q4_0", q4_bytes // BLK_BYTES, flat.size // BLOCK))
            q4_bytes += len(blob)
        else:
            with open(src_bin, "rb") as f:
                f.seek(offset)
                raw = f.read(numel * 4)
            fp32_blob.write(raw)
            index.append((name, shape, "fp32", fp32_elems, numel))
            fp32_elems += numel

    q4_blob.close()
    fp32_blob.close()

    with open(os.path.join(out_dir, "tensors.tsv"), "w", encoding="utf-8") as f:
        for name, shape, kind, off, count in index:
            f.write("%s\t%s\t%s\t%d\t%d\n" % (name, shape, kind, off, count))

    print("全量 q4 权重已写入 %s（q4 %d 块 / fp32 %d 元素）" % (
        out_dir, q4_bytes // BLK_BYTES, fp32_elems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
