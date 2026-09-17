#!/usr/bin/env python3
"""把 HuggingFace 的采样语义导出成**可逐位核对**的参考对，供 Mojo 侧差分。

与 `dump_reference.py` / `dump_model_reference.py` 同一条纪律：**参照物是导出
的 fixture，不是实时调 Python** —— 被测代码变了，答案不会跟着变。

为什么这份脚本要自己实现一遍流水线（而不是只调 HF）
-------------------------------------------------

HF 的 warper 全在 **fp32** 里跑，且 `torch.softmax` / `cumsum` 的浮点累加顺序
没有定义。要达成"采样出的 token id 精确相等"，两侧必须用**同一套显式定义的
fp64 算法**，否则差异来自 libm 的最后 1 ulp 而非语义。

所以这里做两件事，分别钉住不同的风险：

1. **语义对不对** —— 用 HF 真实的 `RepetitionPenaltyLogitsProcessor` 与
   `_get_logits_warper` 返回的 warper 列表跑一遍，比较**存活集合**。集合是离散
   的，不受 1 ulp 影响；任何语义误解（顺序颠倒、阈值方向反了、min_tokens_to_keep
   漏了）都会让集合直接不等。
2. **数值对不对** —— 用显式定义的 fp64 流水线算出概率向量与采样 id，Mojo 侧
   镜像同一算法，逐 id 相等。

并且脚本会对"刀刃"做体检：top_k 的第 k 与第 k+1 名间隔、top_p 的累积概率与
阈值的间隔、min_p 的概率与阈值的间隔，任一过小就直接失败。**绝不允许把建立
在刀刃上的 fixture 交出去**，否则差分门会随机闪红，然后被人"顺手"放宽。

变换顺序（已核对 transformers 4.41.0 源码）
-------------------------------------------

`GenerationMixin.sample` 里先跑 `logits_processor` 再跑 `logits_warper`；
`_get_logits_warper` 的构造顺序是 temperature → top_k → top_p → min_p →
（typical_p / epsilon / eta）→ LogitNormalization。故：

    logit bias → repetition_penalty → frequency → presence → temperature
    → top_k → top_p → min_p → softmax(fp64) → 逆 CDF

其中 logit bias / frequency / presence **HF 4.41 没有对应 processor**，采用
OpenAI / vLLM 的加法语义，属**语义自证**，账本须注明不是 HF 对齐。

为什么自研 SplitMix64 而不用 `std.random`
-----------------------------------------

`std.random` 即便存在，其算法也无法在 Python 侧逐位复现，就做不到"采样逐 id
相等"。SplitMix64 是纯整数运算 + 一次确定的缩放，两侧可逐位一致。

导出物（默认落在 `tests/fixtures/qwen2.5-0.5b/sampler/`）

| 文件 | 内容 |
|---|---|
| `manifest.tsv` | `key <tab> value`：seed / 维度 / 生成器，给人核对 |
| `cases.tsv` | 每个用例的参数：case, prompt, temperature, top_k, top_p, min_p, rep, freq, pres, bias, history |
| `bias.tsv` | `bias_set <tab> index <tab> value` |
| `history.tsv` | `name <tab> token ids(,)` —— 取自真实 greedy 生成 |
| `probs.f32` / `probs.tsv` | 每个用例最终概率向量（fp32，按偏移索引） |
| `sets.tsv` | 每个用例的存活集合：个数 + 前若干个下标 + 全体下标的 FNV-1a 指纹 |
| `draws.tsv` | seed + 前 64 个 uint64 原始输出（把 PRNG 错位与逆 CDF 错位分开） |
| `sampled.tsv` | 每个用例用同一 PRNG 采出的 id 序列 |
| `dist.tsv` / `dist_ref.tsv` | 分布检验用例的 N、阈值，以及参考侧算出的 TVD / 卡方 |
| `dist_support.tsv` / `dist_probs.f32` / `dist_decoy.f32` | 分布检验的支撑集、理论分布与**诱饵分布** |
| `ties/` | 大量并列取值的合成 logits，只比集合（并列上的排序未定义） |

用法：

    /home/rontom/anaconda3/bin/python scripts/dump_sampler_reference.py

注意：本机没有外网；参考栈必须是 `/home/rontom/anaconda3/bin/python`
（pixi 环境里的 python 没有 torch）。
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

OUT_DEFAULT = Path("tests/fixtures/qwen2.5-0.5b")

# PRNG 种子。刻意选一个"看起来不像 0/1"的值：全零种子会让 SplitMix64 的前几
# 个输出偏小，掩盖位混合的错误。
SEED = 0x1A2B3C4D5E6F7788

N_EXPORT_DRAWS = 64  # draws.tsv 导出的原始输出个数
N_SAMPLE_IDS = 32  # 每个用例采多少个 id（逐 id 相等）
N_DIST_SAMPLES = 50000  # 分布检验的采样次数

# 分布检验阈值。序列是固定的，所以这不是"统计意义上的 flaky 阈值"，而是给
# 参考值留出工程余量：参考 TVD 约 0.018，阈值 0.05；卡方自由度 31 时期望约 31。
DIST_TVD_UPPER = 0.05
DIST_CHI2_CRITICAL = 80.0
# 诱饵分布必须被判失败，否则这个门恒真。
DIST_DECOY_TVD_LOWER = 0.15

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

# 刀刃体检的最小间隔。低于它就说明这条用例会被 1 ulp 级别的差异翻转，
# 那不是被测实现的问题，是 fixture 的问题。
MIN_TOPK_GAP = 1e-4
MIN_TOPP_MARGIN = 1e-9
MIN_MINP_MARGIN = 1e-12


class SplitMix64:
    """确定性位源。任何常量改动都会让两侧不再逐位一致。"""

    GAMMA = 0x9E3779B97F4A7C15
    MIX1 = 0xBF58476D1CE4E5B9
    MIX2 = 0x94D049BB133111EB

    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + self.GAMMA) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * self.MIX1) & MASK64
        z = ((z ^ (z >> 27)) * self.MIX2) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def next_uniform(self) -> float:
        """[0, 1) 上的均匀数：取高 53 位后按 2^-53 缩放。"""
        return (self.next_u64() >> 11) * (2.0**-53)


def fnv1a64_indices(indices: np.ndarray) -> int:
    """全体下标的一个 64 位指纹，用来做**精确集合比较**而不必存整个集合。

    温度很高时存活集合有十几万个元素，全存下来既浪费又让人读不了；指纹让
    测试仍能断言"集合完全相同"，这在语义上比逐元素比对概率更硬。
    """
    h = FNV_OFFSET
    for i in indices:
        v = int(i) & MASK64
        for _ in range(8):
            h = ((h ^ (v & 0xFF)) * FNV_PRIME) & MASK64
            v >>= 8
    return h


# --------------------------------------------------------------------------
# 显式定义的 fp64 流水线。Mojo 侧必须逐步镜像。
# --------------------------------------------------------------------------


def softmax64(x: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """存活元素上的 fp64 softmax；非存活处**恰好为 0**。

    求和用 `np.add.accumulate`：它必须产出每一个部分和，因此是严格顺序累加，
    不会像 `np.sum` 那样走 pairwise。Mojo 侧同样顺序累加，于是逐位一致。
    """
    m = float(x[alive].max())
    e = np.zeros_like(x)
    e[alive] = np.exp(x[alive] - m)
    total = float(np.add.accumulate(e)[-1])
    if total == 0.0:
        raise SystemExit("softmax 的全为零：存活集合是空的")
    return e / total


def argsort_desc(x: np.ndarray) -> np.ndarray:
    """降序，并列按下标升序 —— 稳定排序把这个顺序定死。"""
    return np.argsort(-x, kind="stable")


def argsort_asc(x: np.ndarray) -> np.ndarray:
    return np.argsort(x, kind="stable")


def apply_top_k(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= x.shape[0]:
        return np.ones(x.shape[0], dtype=bool)
    order = argsort_desc(x)
    if k < x.shape[0]:
        gap = float(x[order[k - 1]] - x[order[k]])
        if gap < MIN_TOPK_GAP:
            raise SystemExit(
                "top_k 的第 %d 名与第 %d 名间隔仅 %.3e，这条用例立在刀刃上"
                % (k, k + 1, gap)
            )
    thr = float(x[order[k - 1]])
    return x >= thr


def apply_top_p(x: np.ndarray, alive: np.ndarray, top_p: float, keep: int) -> np.ndarray:
    if top_p >= 1.0:
        return alive.copy()
    p = softmax64(x, alive)
    order = argsort_asc(x)
    vals = np.where(alive[order], p[order], 0.0)
    cum = np.add.accumulate(vals)
    cutoff = 1.0 - top_p
    remove = (cum <= cutoff) & alive[order]
    remove[-keep:] = False  # 升序最后一个即最大值，永不移除
    # 刀刃体检：只看真正处在边界上的那一个元素
    live_pos = np.nonzero(remove)[0]
    margins = []
    if live_pos.size:
        b = int(live_pos[-1])
        margins.append(abs(float(cum[b]) - cutoff))
        if b + 1 < cum.shape[0]:
            margins.append(abs(float(cum[b + 1]) - cutoff))
        if min(margins) < MIN_TOPP_MARGIN:
            raise SystemExit(
                "top_p 的累积概率距阈值仅 %.3e，这条用例立在刀刃上" % min(margins)
            )
    out = alive.copy()
    out[order[remove]] = False
    return out


def apply_min_p(x: np.ndarray, alive: np.ndarray, min_p: float, keep: int) -> np.ndarray:
    if min_p <= 0.0:
        return alive.copy()
    p = softmax64(x, alive)
    thr = min_p * float(p.max())
    order = argsort_desc(x)
    remove = (p < thr) & alive
    remove[order[:keep]] = False  # 降序前 keep 个永不移除
    cand = np.nonzero(remove)[0]
    if cand.size:
        margin = float(np.min(np.abs(p[cand] - thr)))
        if margin < MIN_MINP_MARGIN:
            raise SystemExit(
                "min_p 的概率距阈值仅 %.3e，这条用例立在刀刃上" % margin
            )
    out = alive.copy()
    out[remove] = False
    return out


def pipeline(
    logits32: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    rep: float,
    freq: float,
    pres: float,
    bias_pairs: list[tuple[int, float]],
    history: list[int],
    keep: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """完整流水线。返回 `(probs, alive)`。

    `history` 只含"已生成"的 token，语义与 HF 的 `input_ids` 一致（HF 会连
    prompt 一起算，但传什么是调用方的事）。
    """
    x = preprocess64(logits32, rep, freq, pres, bias_pairs, history)

    if temperature == 0.0:
        # 退化：整团概率压到 argmax 上。这与"温度趋于 0"的极限一致，
        # 而不是去真的做除法。
        best = int(np.argmax(x))
        probs = np.zeros(x.shape[0], dtype=np.float64)
        probs[best] = 1.0
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[best] = True
        return probs, mask

    if temperature != 1.0:
        x = x / np.float64(temperature)

    alive = apply_top_k(x, top_k)
    alive = apply_top_p(x, alive, top_p, keep)
    alive = apply_min_p(x, alive, min_p, keep)

    return softmax64(x, alive), alive


def preprocess64(
    logits32: np.ndarray,
    rep: float,
    freq: float,
    pres: float,
    bias_pairs: list[tuple[int, float]],
    history: list[int],
) -> np.ndarray:
    """processor 阶段：bias → repetition → frequency → presence，全在 fp64。

    这一段单独抽出来，是因为 HF 的对照需要它：这四项里 HF 只有 repetition，
    所以我们先把四项都算好、转成 fp32 交给 HF，再让它只负责 warper 阶段。
    """
    x = logits32.astype(np.float64)

    for idx, value in bias_pairs:
        x[idx] = x[idx] + value

    if rep != 1.0:
        # 至多每个 token 施加一次（HF 的文档明确这么写，其实现也是 gather +
        # scatter：重复下标最终只落一次）。对重复 token 反复施加是我们自己
        # 发明的语义，不是 HF 的。
        for t in sorted(set(history)):
            if x[t] < 0:
                x[t] = x[t] * rep
            else:
                x[t] = x[t] / rep

    if freq != 0.0 or pres != 0.0:
        counts: dict[int, int] = {}
        for t in history:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            x[t] = x[t] - freq * c
            if c > 0:
                x[t] = x[t] - pres

    return x


def inverse_cdf(probs: np.ndarray, u: float) -> int:
    """逆 CDF：顺序累加，第一个使累加和超过 u 的下标。"""
    acc = 0.0
    for i in range(probs.shape[0]):
        acc += float(probs[i])
        if u < acc:
            return i
    return probs.shape[0] - 1


# --------------------------------------------------------------------------
# HF 侧的存活集合（语义对照）
# --------------------------------------------------------------------------


def hf_alive_set(
    pre32: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    rep: float,
    history: list[int],
) -> np.ndarray:
    """用 HF 真实的 processor / warper 跑一遍，返回"哪些下标还活着"。

    `pre32` 是**已经施加过** logit bias / frequency / presence 的 fp32 logits
    —— 这三项 HF 没有对应 processor，由我们先算好再交给它。于是这个对照检验
    的是两件事：① warper 的语义与顺序是否与 HF 一致；② 我们是否把那三项放
    在了"processor 阶段"（warpers 之前），而不是塞到某个 warper 中间。

    只比集合：HF 在 fp32 里算，我们在 fp64 里算，逐元素比概率会引入 libm 的
    最后 1 ulp 噪声；但**存活集合是离散的**，语义错了它一定不等。
    """
    import torch
    from transformers import (
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        MinPLogitsWarper,
    )

    scores = torch.from_numpy(pre32[None, :].copy())

    if rep != 1.0:
        input_ids = torch.tensor([history], dtype=torch.long)
        scores = RepetitionPenaltyLogitsProcessor(penalty=rep)(input_ids, scores)

    # 顺序取自 `GenerationMixin._get_logits_warper`（transformers 4.41.0）：
    # temperature → top_k → top_p → min_p。值为 1.0 / 0 / 1.0 / 0 时 HF 根本
    # 不构造对应 warper，这里照做。
    warpers = []
    if temperature not in (0.0, 1.0):
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
    if min_p > 0.0:
        warpers.append(MinPLogitsWarper(min_p=min_p, min_tokens_to_keep=1))
    for warp in warpers:
        scores = warp(torch.zeros((1, 1), dtype=torch.long), scores)

    return torch.isfinite(scores[0]).numpy()


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------


def load_logits(root: Path) -> tuple[np.ndarray, int, int]:
    """读 `logits_last.f32`：形状由文件长度与 greedy 行数反推，不写死维度。"""
    greedy_lines = [
        line for line in (root / "greedy.tsv").read_text().splitlines() if line.strip()
    ]
    n_prompts = len(greedy_lines)
    raw = (root / "logits_last.f32").read_bytes()
    if len(raw) % (n_prompts * 4) != 0:
        raise SystemExit("logits_last.f32 的长度无法被 prompt 数整除")
    vocab = len(raw) // (n_prompts * 4)
    arr = np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(n_prompts, vocab)
    return arr, n_prompts, vocab


def fmt(value: float) -> str:
    """定点写法：Mojo 侧的解析只认十进制小数，不接受指数。"""
    return "%.12f" % value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--history-tokens", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.out)
    out = root / "sampler"
    out.mkdir(parents=True, exist_ok=True)

    logits, n_prompts, vocab = load_logits(root)
    greedy_rows = [
        [int(t) for t in line.split(",")]
        for line in (root / "greedy.tsv").read_text().splitlines()
        if line.strip()
    ]

    # history：取第 1 条 prompt 的真实 greedy 前缀。用真实生成序列而不是随
    # 手编的 id，惩罚项才会落在模型真正会去的那些 token 上。
    histories = {"h0": greedy_rows[1][: args.history_tokens]}

    # logit bias：压住 argmax、抬高一个次优、再抬一个频率很低的。三个方向
    # 都覆盖：被压下去的应当真的被压下去，被抬上来的应当跨过 top_k 边界。
    argmax0 = int(np.argmax(logits[0]))
    order0 = np.argsort(-logits[0], kind="stable")
    biases = {
        "b0": [
            (argmax0, -6.0),
            (int(order0[3]), 4.0),
            (int(order0[400]), 8.0),
        ]
    }

    cases = [
        # name, prompt, T, top_k, top_p, min_p, rep, freq, pres, bias, history
        ("c0", 0, 0.0, 0, 1.0, 0.0, 1.0, 0.0, 0.0, "-", "-"),
        ("c1", 0, 0.7, 0, 1.0, 0.0, 1.0, 0.0, 0.0, "-", "-"),
        ("c2", 0, 0.7, 50, 1.0, 0.0, 1.0, 0.0, 0.0, "-", "-"),
        ("c3", 0, 0.7, 50, 0.9, 0.0, 1.0, 0.0, 0.0, "-", "-"),
        ("c4", 0, 0.7, 50, 0.9, 0.05, 1.0, 0.0, 0.0, "-", "-"),
        ("c5", 1, 0.8, 0, 1.0, 0.0, 1.15, 0.0, 0.0, "-", "h0"),
        ("c6", 1, 0.8, 0, 1.0, 0.0, 1.0, 0.3, 0.2, "-", "h0"),
        ("c7", 2, 0.9, 40, 1.0, 0.0, 1.0, 0.0, 0.0, "b0", "-"),
        # 分布检验专用：top_k 把支撑集压到 32，于是经验分布与理论分布都能
        # 在有限样本下被真正检验，而不是被长尾稀释成噪声。
        ("c8", 3, 1.0, 32, 1.0, 0.0, 1.0, 0.0, 0.0, "-", "-"),
    ]

    prob_chunks: list[np.ndarray] = []
    prob_rows: list[tuple[str, int, int]] = []
    set_rows: list[tuple[str, int, str, int]] = []
    sampled_rows: list[tuple[str, list[int]]] = []
    offset = 0

    for name, prompt, T, top_k, top_p, min_p, rep, freq, pres, bias_name, hist_name in cases:
        row = logits[prompt]
        bias_pairs = biases.get(bias_name, []) if bias_name != "-" else []
        history = histories.get(hist_name, []) if hist_name != "-" else []

        probs, alive = pipeline(
            row, T, top_k, top_p, min_p, rep, freq, pres, bias_pairs, history
        )

        # 语义对照：HF 的真实 warper 必须给出**同一个**存活集合。
        # bias / frequency / presence 由我们先算进 pre32（HF 没有这三样），
        # repetition 留给 HF 自己的 processor 施加。
        if T != 0.0:
            pre32 = preprocess64(row, 1.0, freq, pres, bias_pairs, history).astype(
                np.float32
            )
            ref_alive = hf_alive_set(pre32, T, top_k, top_p, min_p, rep, history)
            if not np.array_equal(ref_alive, alive):
                ours = set(np.nonzero(alive)[0].tolist())
                theirs = set(np.nonzero(ref_alive)[0].tolist())
                raise SystemExit(
                    "%s：存活集合与 HF 不一致（我们多 %d 个、少 %d 个，例如 %s）"
                    % (
                        name,
                        len(ours - theirs),
                        len(theirs - ours),
                        sorted(ours ^ theirs)[:8],
                    )
                )
        else:
            # 温度为 0 时 HF 的 warper 不接受 0，改为直接核对 argmax。
            if int(np.argmax(row)) != int(np.argmax(probs)):
                raise SystemExit("%s：温度 0 未退化为 argmax" % name)
            if int(np.argmax(row)) != greedy_rows[prompt][0]:
                raise SystemExit("%s：温度 0 的结果与 greedy.tsv 的首 token 不一致" % name)

        idx = np.nonzero(alive)[0]
        prob_chunks.append(probs.astype(np.float32))
        prob_rows.append((name, vocab, offset))
        offset += vocab
        head = ",".join(str(int(i)) for i in idx[:16])
        set_rows.append((name, int(idx.size), head, fnv1a64_indices(idx)))

        rng = SplitMix64(SEED)
        ids = [inverse_cdf(probs, rng.next_uniform()) for _ in range(N_SAMPLE_IDS)]
        if idx.size and not all(int(i) in set(idx.tolist()) for i in ids):
            raise SystemExit("%s：采样落到了存活集合之外" % name)
        sampled_rows.append((name, ids))

    with (out / "probs.f32").open("wb") as f:
        for chunk in prob_chunks:
            f.write(chunk.astype("<f4").tobytes())
    with (out / "probs.tsv").open("w", encoding="utf-8") as f:
        for name, dims, off in prob_rows:
            f.write("%s\t%d\t%d\t%d\n" % (name, dims, off, dims))

    with (out / "cases.tsv").open("w", encoding="utf-8") as f:
        for name, prompt, T, top_k, top_p, min_p, rep, freq, pres, bias_name, hist_name in cases:
            f.write(
                "%s\t%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                % (
                    name,
                    prompt,
                    fmt(T),
                    top_k,
                    fmt(top_p),
                    fmt(min_p),
                    fmt(rep),
                    fmt(freq),
                    fmt(pres),
                    bias_name,
                    hist_name,
                )
            )

    with (out / "bias.tsv").open("w", encoding="utf-8") as f:
        for name, pairs in biases.items():
            for index, value in pairs:
                f.write("%s\t%d\t%s\n" % (name, index, fmt(value)))

    with (out / "history.tsv").open("w", encoding="utf-8") as f:
        for name, ids in histories.items():
            f.write("%s\t%s\n" % (name, ",".join(str(i) for i in ids)))

    with (out / "sets.tsv").open("w", encoding="utf-8") as f:
        for name, count, head, digest in set_rows:
            f.write("%s\t%d\t%s\t%d\n" % (name, count, head, digest))

    with (out / "logits.tsv").open("w", encoding="utf-8") as f:
        for i in range(n_prompts):
            f.write("p%d\t%d\t%d\t%d\n" % (i, vocab, i * vocab, vocab))

    # PRNG 的原始输出单独导出：若 draws 就对不上，那是位混合错；若 draws 对
    # 而 sampled 对不上，那是逆 CDF 或概率向量错。与 layer0 逐算子差分同理。
    rng = SplitMix64(SEED)
    draws = [rng.next_u64() for _ in range(N_EXPORT_DRAWS)]
    with (out / "draws.tsv").open("w", encoding="utf-8") as f:
        f.write("seed\t%d\n" % SEED)
        for i, value in enumerate(draws):
            f.write("%d\t%d\n" % (i, value))

    with (out / "sampled.tsv").open("w", encoding="utf-8") as f:
        for name, ids in sampled_rows:
            f.write("%s\t%s\n" % (name, ",".join(str(i) for i in ids)))

    # ---- 分布检验 ----
    dist_case = "c8"
    d = next(c for c in cases if c[0] == dist_case)
    _, prompt, T, top_k, top_p, min_p, rep, freq, pres, bias_name, hist_name = d
    probs, alive = pipeline(
        logits[prompt], T, top_k, top_p, min_p, rep, freq, pres, [], []
    )
    support = np.nonzero(alive)[0]
    support = support[np.argsort(-probs[support], kind="stable")]
    p_support = probs[support].astype(np.float64)
    total = float(np.add.accumulate(p_support)[-1])
    p_support = p_support / total
    # 落盘是 fp32，所以采样必须也用**落盘后**的那个数，而不是重归一化前的
    # fp64 值：否则两侧差的不是算法，而是存储精度。
    p_support = p_support.astype(np.float32).astype(np.float64)

    # 诱饵：温度减半后重新归一化。支撑集不变（正比例缩放保序），但形状明显
    # 更尖 —— 用**错误的**理论分布去做检验，门必须判失败，否则门恒真。
    decoy_logits = logits[prompt].astype(np.float64) / np.float64(T * 0.5)
    m = float(decoy_logits[support].max())
    e = np.exp(decoy_logits[support] - m)
    decoy = e / float(np.add.accumulate(e)[-1])
    decoy = decoy.astype(np.float32).astype(np.float64)

    rng = SplitMix64(SEED)
    cum = np.add.accumulate(p_support)
    counts = np.zeros(support.shape[0], dtype=np.int64)
    for _ in range(N_DIST_SAMPLES):
        u = rng.next_uniform()
        hit = int(np.searchsorted(cum, u, side="right"))
        # 累积和可能因 fp32 存储而略小于 1，此时 u 落在末尾之后 —— 与 Mojo
        # 侧 `pick_from` 的兜底一致：给最后一个。
        if hit >= support.shape[0]:
            hit = support.shape[0] - 1
        counts[hit] += 1

    obs = counts.astype(np.float64)
    exp = p_support * N_DIST_SAMPLES
    tvd = 0.5 * float(np.abs(obs / N_DIST_SAMPLES - p_support).sum())
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    exp_decoy = decoy * N_DIST_SAMPLES
    tvd_decoy = 0.5 * float(np.abs(obs / N_DIST_SAMPLES - decoy).sum())
    chi2_decoy = float(((obs - exp_decoy) ** 2 / exp_decoy).sum())

    if tvd >= DIST_TVD_UPPER:
        raise SystemExit("分布检验的 TVD %.6f 已超过阈值 %.6f" % (tvd, DIST_TVD_UPPER))
    if chi2 >= DIST_CHI2_CRITICAL:
        raise SystemExit("分布检验的卡方 %.3f 已超过阈值 %.3f" % (chi2, DIST_CHI2_CRITICAL))
    if tvd_decoy < DIST_DECOY_TVD_LOWER:
        raise SystemExit(
            "诱饵分布的 TVD 只有 %.6f（需 ≥ %.6f）：这个门区分不出错误分布"
            % (tvd_decoy, DIST_DECOY_TVD_LOWER)
        )

    with (out / "dist_probs.f32").open("wb") as f:
        f.write(p_support.astype("<f4").tobytes())
        f.write(decoy.astype("<f4").tobytes())
    with (out / "dist_probs.tsv").open("w", encoding="utf-8") as f:
        f.write("%s\t%d\t%d\t%d\n" % (dist_case, support.shape[0], 0, support.shape[0]))
        f.write("%s_decoy\t%d\t%d\t%d\n" % (dist_case, support.shape[0], support.shape[0], support.shape[0]))
    with (out / "dist_support.tsv").open("w", encoding="utf-8") as f:
        f.write("%s\t%s\n" % (dist_case, ",".join(str(int(i)) for i in support)))
    with (out / "dist.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "%s\t%d\t%s\t%s\t%s\n"
            % (dist_case, N_DIST_SAMPLES, fmt(DIST_TVD_UPPER), fmt(DIST_CHI2_CRITICAL), fmt(DIST_DECOY_TVD_LOWER))
        )
    with (out / "dist_ref.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "%s\t%s\t%s\t%s\t%s\n"
            % (dist_case, fmt(tvd), fmt(chi2), fmt(tvd_decoy), fmt(chi2_decoy))
        )

    # ---- 并列取值的合成用例：只比集合 ----
    write_ties(out)

    with (out / "manifest.tsv").open("w", encoding="utf-8") as f:
        f.write("generator\tscripts/dump_sampler_reference.py\n")
        f.write("seed\t%d\n" % SEED)
        f.write("vocab\t%d\n" % vocab)
        f.write("n_prompts\t%d\n" % n_prompts)
        f.write("n_cases\t%d\n" % len(cases))
        f.write("n_export_draws\t%d\n" % N_EXPORT_DRAWS)
        f.write("n_sample_ids\t%d\n" % N_SAMPLE_IDS)
        f.write("n_dist_samples\t%d\n" % N_DIST_SAMPLES)
        f.write("min_tokens_to_keep\t1\n")

    print("导出完成：%s" % out)
    print("  用例 %d 条，词表 %d，分布检验 TVD %.6f / 卡方 %.3f（诱饵 %.6f）"
          % (len(cases), vocab, tvd, chi2, tvd_decoy))
    for name, count, _, _ in set_rows:
        print("  %s 存活 %d" % (name, count))
    return 0


def write_ties(out: Path) -> None:
    """并列取值上的集合检验。

    `torch.sort` 在并列值上的顺序未定义，所以这里**只比集合**：top_k 的存活
    集合在语义上是"值 ≥ 第 k 大者"的全部下标，与并列怎么排无关。若哪天有人
    把 `>=` 写成 `>`，这块会立刻红 —— 而用真实 logits 几乎测不出来。
    """
    import torch
    from transformers import TopKLogitsWarper

    n = 64
    rng = np.random.default_rng(20260917)
    row = np.round(rng.normal(0.0, 2.0, size=n) * 4.0) / 4.0  # 大量并列
    row = row.astype(np.float32)
    if len(set(row.tolist())) >= n:
        raise SystemExit("并列用例没有构造出并列值，脚本需要重新设计")

    ties_dir = out / "ties"
    ties_dir.mkdir(parents=True, exist_ok=True)
    with (ties_dir / "logits.f32").open("wb") as f:
        f.write(struct.pack("<%df" % n, *row.tolist()))

    ks = [4, 8, 16, 32]
    with (ties_dir / "cases.tsv").open("w", encoding="utf-8") as f:
        for k in ks:
            f.write("t%d\t%d\n" % (k, k))
    with (ties_dir / "sets.tsv").open("w", encoding="utf-8") as f:
        for k in ks:
            scores = torch.from_numpy(row[None, :].copy())
            processed = TopKLogitsWarper(top_k=k, min_tokens_to_keep=1)(
                torch.zeros((1, 1), dtype=torch.long), scores
            )
            alive = np.nonzero(torch.isfinite(processed[0]).numpy())[0]
            # 集合定义：值 ≥ 第 k 大者。与 HF 交叉核对，二者必须一致。
            order = np.argsort(-row, kind="stable")
            thr = float(row[order[k - 1]])
            expect = np.nonzero(row >= thr)[0]
            if not np.array_equal(np.sort(alive), np.sort(expect)):
                raise SystemExit(
                    "并列用例 k=%d：HF 的存活集合与定义不符（%d vs %d）"
                    % (k, alive.size, expect.size)
                )
            f.write(
                "t%d\t%d\t%s\n" % (k, int(alive.size), ",".join(str(int(i)) for i in alive))
            )
    with (ties_dir / "manifest.tsv").open("w", encoding="utf-8") as f:
        f.write("n\t%d\n" % n)
        f.write("distinct_values\t%d\n" % len(set(row.tolist())))


if __name__ == "__main__":
    sys.exit(main())
