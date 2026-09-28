#!/usr/bin/env python3
"""导出 q8_0 量化的参考 fixture（供 `tests/unit/test_q8_parity.mojo` 差分）。

为什么要有 q8_0
---------------
2026-09-28 与 llama.cpp 的同机同 prompt 对比量出：alofa 的 q4_0 档只有对方的
0.54…0.60×，而**同流量对照**（把 llama.cpp 的嵌入/输出张量也拖回 F32，它从
25.60 掉到 13.44，与 alofa 的 13.83…15.37 重合）证明差距**不在 q4 内核**，而在
**输出投影的存储宽度**：alofa 的 `lm_head` 是独立 fp32 副本（544 MB），llama.cpp
因 tied embedding 只存一份且压成 Q8_0（≈165 MB）。

输出投影压到 q4_0 是量过、否掉的（教师强制贪心一致率 0.8691 → 0.77）；8 bit 是
它在质量上还站得住的最窄一档（相对 L2 约 0.4%，q4_0 是 10.3%）。

本脚本与 `dump_q4_reference.py` 同一套规矩
------------------------------------------
- **量化（fp32 → q8_0）是我方离线的格式转换**，没有"另一家"可对照 ⇒ 算法显式写
  在代码里，不用任何库的"帮我们算一下"的便利函数；
- **解量化（q8_0 → fp32）有外部事实可依**：块布局（32 值 / 34 字节、fp16 缩放、
  其后 32 个字节各一个**有符号** 8 位码、`v = q · d`）是 GGML q8_0 的真实布局
  ⇒ 这一半是**格式一致性检验**；
- 解量化在这里是**独立实现**：从字节流出发，与量化那段不共享任何中间变量。

用法：
    python scripts/dump_q8_reference.py
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(ROOT, "tests", "fixtures", "qwen2.5-0.5b")

# GGML q8_0：一个块 32 个值，34 字节。
#   [0:2]    fp16 缩放 d（小端）
#   [2:34]   32 个字节，每个装一个**有符号** 8 位量化值（补码）
#   还原：v = q * d
BLOCK = 32
BLK_BYTES = 34

# 从 `lm_head.weight` 取的行数 —— 这条门服务的就是输出投影，所以用它的真实权重。
HEAD_ROWS = 64


def quantize_q8_0(flat: np.ndarray) -> bytes:
    """fp32 一维数组 → q8_0 块流。长度必须是 32 的倍数。

        amax = max(|x|)                 块内绝对值的最大值
        d    = amax / 127               让最大值恰好够到 +127
        q    = clamp(round(x / d), -128, 127)

    ⚠️ 没有照 q4_0 那样按 MSE 修 d：256 级台阶下相对 L2 约 0.4%，台阶已细到
    "往哪儿挪"没有收益可榨。

    两个必须与 Mojo 侧（`kernels/cpu/quant.mojo`）逐字节一致的地方：
    - 量化用的 d 是 **fp32 的 d**，存进块的才是 fp16 的那个；
    - `round` 取**最近、并列取偶**（IEEE 默认，numpy 的 `np.round` 同规则）。
    """
    flat = np.ascontiguousarray(flat, dtype=np.float32)
    if flat.size % BLOCK != 0:
        raise ValueError("长度必须是 %d 的倍数，收到 %d" % (BLOCK, flat.size))
    n_blocks = flat.size // BLOCK
    blocks = flat.reshape(n_blocks, BLOCK)

    amax = np.max(np.abs(blocks), axis=1)
    d = np.where(amax > 0, amax / np.float32(127.0), np.float32(0.0)).astype(
        np.float32
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(d[:, None] != 0, blocks / d[:, None], np.float32(0.0))
    q = np.clip(np.round(scaled), -128, 127).astype(np.int8)

    # 块是**交错**的：每个块 34 字节 = [fp16 d][32 字节码]。把所有缩放因子排在
    # 前面会让"按 34 字节读块"的解法读到垃圾 —— 只有一个块时恰好对得上，所以这
    # 种错不会被小样例发现。
    out = np.empty((n_blocks, BLK_BYTES), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:BLK_BYTES] = q.view(np.uint8)
    return out.tobytes()


def dequant_q8_0(raw: bytes, n: int) -> np.ndarray:
    """解量化的**独立实现**：重排字节、按布局还原。

    与 `quantize_q8_0` 里写块的代码不共享任何中间变量 —— 它从字节流出发，走的
    是"读盘"这条路。这样测试里 Mojo 对上它，等于对上"一个从字节流独立解出来的
    人"，而不是对上"量化时的中间数组"。
    """
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, BLK_BYTES)
    d = arr[:, 0:2].copy().view(np.float16).astype(np.float32)
    # ⚠️ 必须是 int8：当成无符号读，负权重会整体跳到 +128 那侧。
    q = arr[:, 2:BLK_BYTES].astype(np.int8).astype(np.float32)
    values = q * d.reshape(-1, 1)
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


def main() -> None:
    tsv = os.path.join(FIX, "weights", "tensors.tsv")
    binp = os.path.join(FIX, "weights", "tensors.f32")
    head = load_fp32(tsv, binp, "lm_head.weight")
    cols = head.shape[1]

    # 取前 HEAD_ROWS 行：真实权重，且是这条门将来要服务的那个矩阵。
    orig = np.ascontiguousarray(head[:HEAD_ROWS, :]).reshape(-1)
    assert orig.size % BLOCK == 0, orig.size

    blocks = quantize_q8_0(orig)
    deq = dequant_q8_0(blocks, orig.size)

    rel = np.linalg.norm(deq - orig) / np.linalg.norm(orig)
    print("lm_head 前 %d 行（%d 个值）：相对 L2 误差 %.4f%%" % (
        HEAD_ROWS, orig.size, rel * 100.0))

    out_dir = os.path.join(FIX, "q8")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "orig.f32"), "wb") as f:
        f.write(orig.astype("<f4").tobytes())
    with open(os.path.join(out_dir, "blocks.bin"), "wb") as f:
        f.write(blocks)
    with open(os.path.join(out_dir, "dequant.f32"), "wb") as f:
        f.write(deq.astype("<f4").tobytes())

    # 融合 matmul 的期望：真实激活 × **解量化后**的权重，fp64 累加。
    # 用解量化后的权重（而不是原始 fp32）是为了让判据只盯住**累加顺序**这一件事
    # —— 解量化本身是精确的，已经被上面那条零容差门单独覆盖；若这里拿原始 fp32
    # 当期望，量化误差会混进来，1e-5 的判据立刻被吃掉（q8 的相对 L2 就有 0.5%）。
    with open(os.path.join(FIX, "q4", "act.f32"), "rb") as f:
        act = np.frombuffer(f.read(), dtype="<f4")
    if act.size < cols:
        raise SystemExit("act.f32 只有 %d 个值，需要 %d" % (act.size, cols))
    act = act[:cols].astype(np.float64)
    expected = deq.astype(np.float64).reshape(HEAD_ROWS, cols) @ act
    with open(os.path.join(out_dir, "act.f32"), "wb") as f:
        f.write(act.astype("<f4").tobytes())
    with open(os.path.join(out_dir, "out_expected.f64"), "wb") as f:
        f.write(expected.astype("<f8").tobytes())

    with open(os.path.join(out_dir, "cases.tsv"), "w", encoding="utf-8") as f:
        f.write("name\trows\tcols\tblocks\tblock_bytes\n")
        f.write("lm_head_head\t%d\t%d\t%d\t%d\n" % (
            HEAD_ROWS, cols, orig.size // BLOCK, BLK_BYTES))
    print("已写入 %s" % out_dir)


if __name__ == "__main__":
    main()
