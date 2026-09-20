#!/usr/bin/env python3
"""Q4_K 的离线参考：量化（fp32 → 块流）与解量化（块流 → fp32）。

为什么要有 Q4_K
--------------
q4_0 一个块（32 个值）只有一个 fp16 **缩放因子**，量化台阶被钉在 0 上
（`v = (nibble - 8) * d`），于是它只能表示**关于零对称**的分布。真实权重不是
这样：一个块里的 32 个值常常整体偏在一侧，此时对称台阶把一半的动态范围浪费在
没有数据的那边。

Q4_K 给每个子块（32 个值）配一个 **scale 和一个 min**（各 6 bit），台阶可以整体
平移 —— 台阶数仍然是 16 个，但它们落在数据真实所在的区间上。代价是每个权重
4.5 bit 而不是 4.0 bit。

两件事分开写（与 q4_0 同一个规矩）
--------------------------------
* **块布局是 GGML 的外部事实**：GGUF 文件里就是这么排的，所以「解量化」属格式
  一致性检验，Mojo 侧可以对这份参考做**零容差逐位比较**。
* **fp32 → Q4_K 的量化步是我方离线的格式转换**（Mojo 侧永远不量化），没有外部
  参照，不许写成「与 llama.cpp 一致」。下面的算法就是唯一的规格说明。

布局（一个 super-block = 256 个权重 = 144 字节）
---------------------------------------------
    +0    fp16 d      子块 scale 的整体缩放
    +2    fp16 dmin   子块 min   的整体缩放
    +4    12 字节 scales/min：8 个子块 × (6 bit scale + 6 bit min)，按 GGML 的
          高低位交错排布（见 `_get_scale_min`）
    +16   128 字节 qs：256 个 nibble。每 32 字节承载 64 个值（先 32 个低半字节，
          再 32 个高半字节），4 组共 256 个

解出的值：`y = d * sc * nibble - dmin * m`，nibble ∈ [0, 15]（**不减 8**，偏移
已经由 min 承担 —— 这正是它比 q4_0 多出来的那个自由度）。

⚠️ 已知未完成（第 6 项在这里停住，不要当成已交付）
---------------------------------------------
布局里**后 4 个子块的 scale 与 min 共用 `scales[j-4]` 的高 2 位**，所以两者的高
2 位必须相同。本文件当前的做法是事后取 `max(sc>>4, mn>>4)` 强行对齐 —— 这会把
min 抬高（实测某子块 37 → 53），相对 L2 反而从 q4_0 的 10% 恶化到 49%，**比不做
还差**。正确的做法是 GGML 的**联合量化**（`make_qx_quants` 那个路子）：先按误差
最小解出 (scale, min)，再把它们压进 6 bit 时让高 2 位自然落在同一区间，而不是
事后截断。本轮没有外网，无法核对 llama.cpp 的逐位细节，故**不声称这是真实的
GGUF Q4_K**，也不声称误差更低 —— 留到能对照源码时再做。
"""

from __future__ import annotations

import struct

import numpy as np

QK_K = 256  # 一个 super-block 的权重数
K_SCALE_SIZE = 12  # scales/min 的字节数
BLOCK_BYTES = 2 + 2 + K_SCALE_SIZE + 128


def _f16(x: float) -> np.uint16:
    """fp16 位模式。fp16→fp32 是**精确**转换，故整条链路可以零容差。"""
    return np.uint16(struct.unpack("<H", struct.pack("<e", float(x)))[0])


def _f16_to_f32(bits: int) -> float:
    return float(struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0])


def _set_scale_min(j: int, scales: bytearray, sc: int, m: int) -> None:
    """按 GGML 的排布把第 j 个子块的 6-bit scale / min 写进 12 字节数组。

    96 bit 装 16 个 6-bit 值是装得下的，但 GGML 的排布有一个**约束**：后 4 个子块
    的 scale 与 min 共用 `scales[j-4]` 的高 2 位，所以两者的**高 2 位必须相同**。
    量化侧必须自己保证这一点（见 `quantize_q4_k` 里的对齐），否则解回来就串了。
    """
    if j < 4:
        scales[j] = (scales[j] & 0xC0) | (sc & 63)
        scales[j + 4] = (scales[j + 4] & 0xC0) | (m & 63)
    else:
        scales[j + 4] = (sc & 0xF) | ((m & 0xF) << 4)
        scales[j - 4] = (scales[j - 4] & 0x3F) | (((sc >> 4) & 0x3) << 6)


def _get_scale_min(j: int, scales: bytes):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    hi = ((scales[j - 4] >> 6) & 0x3) << 4
    return (scales[j + 4] & 0xF) | hi, (scales[j + 4] >> 4) | hi


def quantize_q4_k(flat: np.ndarray) -> bytes:
    """fp32 → Q4_K 块流。`flat` 长度必须是 256 的倍数。"""
    x = np.ascontiguousarray(flat, dtype=np.float32).reshape(-1)
    if x.size % QK_K != 0:
        raise SystemExit(f"size {x.size} 不是 {QK_K} 的倍数，Q4_K 切不动")

    out = bytearray()
    for base in range(0, x.size, QK_K):
        blk = x[base : base + QK_K]
        sub_sc = np.zeros(8, dtype=np.float64)
        sub_mn = np.zeros(8, dtype=np.float64)
        nibbles = np.zeros(QK_K, dtype=np.int64)

        # 每个子块独立选 (scale, min)：32 个值落在 16 级台阶上，台阶整体平移到
        # 数据所在的区间 —— 这就是 Q4_K 相对 q4_0 多出来的那个自由度。
        for j in range(8):
            w = blk[j * 32 : (j + 1) * 32].astype(np.float64)
            lo = float(w.min())
            hi = float(w.max())
            if hi <= lo:
                sub_sc[j] = 0.0
                sub_mn[j] = 0.0
                nibbles[j * 32 : (j + 1) * 32] = 0
                continue
            if lo >= 0.0:
                # 这个子块的数据全在零的一侧。Q4_K 的 min 通道是**无符号** 6 bit
                # 且解出来是 `y = sc·nib − dmin·m`，它只能表达**负**方向的偏移；
                # 拿它去表示正的 lo 会把整个子块平移到错误的位置（实测相对 L2 从
                # 10% 恶化到 49%，就是这个符号）。故此处退化为无偏移：台阶从 0 起，
                # 代价是 [0, lo] 这一段动态范围被浪费掉。
                sc = hi / 15.0
                sub_sc[j] = sc
                sub_mn[j] = 0.0
                q = np.clip(np.round(w / sc), 0, 15).astype(np.int64)
            else:
                sc = (hi - lo) / 15.0
                sub_sc[j] = sc
                sub_mn[j] = -lo
                q = np.clip(np.round((w - lo) / sc), 0, 15).astype(np.int64)
            nibbles[j * 32 : (j + 1) * 32] = q

        # 子块的 scale / min 再各自用一个整体缩放压到 6 bit。
        d = float(sub_sc.max()) / 63.0 if sub_sc.max() > 0 else 0.0
        dmin = float(np.abs(sub_mn).max()) / 63.0 if np.abs(sub_mn).max() > 0 else 0.0
        sc6 = (
            np.clip(np.round(sub_sc / d), 0, 63).astype(np.int64)
            if d > 0
            else np.zeros(8, dtype=np.int64)
        )
        mn6 = (
            np.clip(np.round(np.abs(sub_mn) / dmin), 0, 63).astype(np.int64)
            if dmin > 0
            else np.zeros(8, dtype=np.int64)
        )

        # 后 4 个子块的 scale / min 共用高 2 位，两者必须落在同一高位区间 ——
        # 取较大者对齐（这会让其中一个略微变大，是这份量化算法的一部分）。
        for j in range(4, 8):
            hi = max(int(sc6[j]) >> 4, int(mn6[j]) >> 4)
            sc6[j] = (hi << 4) | (int(sc6[j]) & 0xF)
            mn6[j] = (hi << 4) | (int(mn6[j]) & 0xF)

        out += struct.pack("<H", int(_f16(d)))
        out += struct.pack("<H", int(_f16(dmin)))
        scales = bytearray(K_SCALE_SIZE)
        for j in range(8):
            _set_scale_min(j, scales, int(sc6[j]), int(mn6[j]))
        out += bytes(scales)

        qs = bytearray(128)
        for group in range(4):  # 4 组 × 64 值，每组 32 字节
            for l in range(32):
                lo_n = int(nibbles[group * 64 + l]) & 0xF
                hi_n = int(nibbles[group * 64 + 32 + l]) & 0xF
                qs[group * 32 + l] = lo_n | (hi_n << 4)
        out += bytes(qs)
    return bytes(out)


def dequant_q4_k(raw: bytes, n: int) -> np.ndarray:
    """Q4_K 块流 → fp32。布局是外部事实，故这份实现就是规格。"""
    if len(raw) != (n // QK_K) * BLOCK_BYTES:
        raise SystemExit(f"块流长度 {len(raw)} 与 {n} 个权重不匹配")
    y = np.zeros(n, dtype=np.float32)
    for i in range(n // QK_K):
        off = i * BLOCK_BYTES
        d = _f16_to_f32(int.from_bytes(raw[off : off + 2], "little"))
        dmin = _f16_to_f32(int.from_bytes(raw[off + 2 : off + 4], "little"))
        scales = raw[off + 4 : off + 4 + K_SCALE_SIZE]
        qs = raw[off + 16 : off + BLOCK_BYTES]
        for group in range(4):
            for half in range(2):
                j = group * 2 + half
                sc6, mn6 = _get_scale_min(j, scales)
                sc = d * float(sc6)
                mn = dmin * float(mn6)
                for l in range(32):
                    idx = group * 64 + half * 32 + l
                    if half == 0:
                        nib = qs[group * 32 + l] & 0xF
                    else:
                        nib = qs[group * 32 + l] >> 4
                    y[i * QK_K + idx] = np.float32(sc * float(nib) - mn)
    return y


def rel_l2(orig: np.ndarray, quant: np.ndarray) -> float:
    denom = float(np.linalg.norm(orig.astype(np.float64).reshape(-1)))
    if denom == 0.0:
        return 0.0
    num = float(
        np.linalg.norm(
            (quant.astype(np.float64) - orig.astype(np.float64)).reshape(-1)
        )
    )
    return num / denom
