#!/usr/bin/env python3
"""诊断：量化到底伤在哪个投影上。

第 7 项要求「先定位为什么量化输出投影反而更差」。「更差」有两种完全不同的原因，
不分开就无从下手：

1. **误差更大** —— 这个矩阵的权重分布不适合 q4_0（每 32 个值共用一个 fp16
   scale），量化后相对 L2 误差本来就高。
2. **位置更敏感** —— 误差差不多大，但它写在残差流上，会被后面每一层继续放大；
   而别的投影的误差被 attention 的 softmax / RMSNorm 部分吸收掉了。

只测「全量量化的一致率」分不开这两者。所以这里是**逐个投影单独量化**：每个投影
单独切成 q4_0 再解回来，其余保持 fp32，然后跑与 `test_q4_greedy.mojo` **同一套**
教师强制一致率（喂真值、比 argmax、只统计生成段）。谁的量化最伤一致率，谁就是
优先要换块布局的地方；再看它的静态误差排第几，就知道属于上面哪一种。

「输出投影」这个词有两个候选（`o_proj` 注意力输出投影 / `lm_head` 词表输出投影），
两个都测，不再靠猜。

用法（必须用有 torch 的解释器，pixi 的没有）：

    /home/rontom/anaconda3/bin/python scripts/diag_q4_proj.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_q4_reference import dequant_q4_0, quantize_q4_0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "qwen2.5-0.5b"

# 每个投影在 transformers 里的后缀；`lm_head` 单独处理。
PROJECTIONS = [
    ("q_proj", "self_attn.q_proj"),
    ("k_proj", "self_attn.k_proj"),
    ("v_proj", "self_attn.v_proj"),
    ("o_proj", "self_attn.o_proj"),
    ("gate_proj", "mlp.gate_proj"),
    ("up_proj", "mlp.up_proj"),
    ("down_proj", "mlp.down_proj"),
]


def model_path() -> str:
    hits = sorted(
        glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/"
            )
        )
    )
    if not hits:
        raise SystemExit("no local Qwen2.5-0.5B snapshot; cannot run offline")
    return hits[-1]


def load_prompts():
    """`prompts.tsv`：每行是「prompt 的 hex」+「它的 token ids」两列。"""
    out = []
    for line in (FIXTURE / "prompts.tsv").read_text().splitlines():
        if not line.strip():
            continue
        _, ids = line.split("\t")
        out.append([int(x) for x in ids.split(",") if x])
    return out


def load_greedy():
    out = []
    for line in (FIXTURE / "greedy.tsv").read_text().splitlines():
        if not line.strip():
            continue
        out.append([int(x) for x in line.split(",") if x])
    return out


def q4_roundtrip(arr: np.ndarray) -> np.ndarray:
    """fp32 → q4_0 块流 → fp32。块按展平后的每 32 个连续值切。"""
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    if flat.size % 32 != 0:
        raise SystemExit(f"size {flat.size} 不是 32 的倍数，q4_0 切不动")
    return dequant_q4_0(quantize_q4_0(flat), flat.size).reshape(arr.shape)


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


def agreement(model, prompt_ids, expected, torch):
    """与 `test_q4_greedy.mojo` 同构：prefill 一步，之后每步喂**真值**再比 argmax。

    只统计生成段（`expected` 里的每一个位置），prompt 段不计入 —— 门里也是这样。
    """
    import torch as _t  # noqa: F401

    ids = _t.tensor([prompt_ids], dtype=_t.long)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    total = 0
    agreed = 0
    token = int(_t.argmax(out.logits[0, -1]).item())
    total += 1
    if token == expected[0]:
        agreed += 1
    for i in range(1, len(expected)):
        out = model(
            input_ids=_t.tensor([[expected[i - 1]]], dtype=_t.long),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        token = int(_t.argmax(out.logits[0, -1]).item())
        total += 1
        if token == expected[i]:
            agreed += 1
    return agreed, total


def rate_of(model, prompts, greedy, torch) -> float:
    a = 0
    t = 0
    for p, g in zip(prompts, greedy):
        da, dt = agreement(model, p, g, torch)
        a += da
        t += dt
    return a / t if t else 0.0


def set_weight(model, suffix: str, fn) -> int:
    """对所有层里名字以 `suffix` 结尾的 weight 应用 `fn`，返回改了几处。"""
    import torch

    n = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(suffix + ".weight"):
                new = fn(param.detach().cpu().numpy())
                param.copy_(torch.from_numpy(np.ascontiguousarray(new)))
                n += 1
    return n


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_grad_enabled(False)
    # 快速模式（口径会变，只用于**相对**比较，不再是门的那 512 步）：
    #   --prompts=N   只用前 N 条 prompt
    #   --only=a,b    只测这几个投影（写 `suffix`，lm_head 写 "lm_head"）
    n_prompt = 10 ** 9
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--prompts="):
            n_prompt = int(a.split("=", 1)[1])
        elif a.startswith("--only="):
            only = a.split("=", 1)[1].split(",")
    prompts = load_prompts()[:n_prompt]
    greedy = load_greedy()[:n_prompt]
    assert len(prompts) == len(greedy), "one greedy row per prompt"

    print(f"prompts={len(prompts)}  生成段总步数={sum(len(g) for g in greedy)}")
    print()

    base = AutoModelForCausalLM.from_pretrained(
        model_path(), torch_dtype=torch.float32
    )
    base.eval()

    # 参照物：fp32 对自己的一致率。它必须是 1.0 —— 参考序列本来就是 fp32 贪心
    # 解码出来的；不是 1.0 说明这条诊断链本身没搭对（比如 KV cache 用错了）。
    fp32_rate = rate_of(base, prompts, greedy, torch)
    print(f"fp32 自洽     一致率 {fp32_rate:.4f}   （必须为 1.0，否则诊断无效）")
    if fp32_rate < 0.999:
        print("!! fp32 对自己的一致率不是 1.0 —— 下面的数字没有意义，先修诊断脚本")
        return 1
    print()

    rows = []
    for label, suffix in PROJECTIONS:
        if only is not None and suffix not in only:
            continue
        m = AutoModelForCausalLM.from_pretrained(
            model_path(), torch_dtype=torch.float32
        )
        m.eval()
        errs = []

        def _fn(a, errs=errs):
            q = q4_roundtrip(a)
            errs.append(rel_l2(a, q))
            return q

        n = set_weight(m, suffix, _fn)
        r = rate_of(m, prompts, greedy, torch)
        rows.append((label, n, float(np.mean(errs)), r))
        print(f"{label:<10} {n:>3} 处  相对L2 {np.mean(errs) * 100:6.2f}%  一致率 {r:.4f}")
        del m

    # 两个「输出投影」候选分别单独测：o_proj 已在上面，这里补 lm_head。
    if only is None or "lm_head" in only:
        m = AutoModelForCausalLM.from_pretrained(
            model_path(), torch_dtype=torch.float32
        )
        m.eval()
        err = 0.0
        with torch.no_grad():
            for name, param in m.named_parameters():
                if name == "lm_head.weight":
                    lm = param.detach().cpu().numpy()
                    q = q4_roundtrip(lm)
                    err = rel_l2(lm, q)
                    param.copy_(torch.from_numpy(np.ascontiguousarray(q)))
        r = rate_of(m, prompts, greedy, torch)
        rows.append(("lm_head", 1, err, r))
        print(
            f"{'lm_head':<10}   1 处  相对L2 {err * 100:6.2f}%  一致率 {r:.4f}"
        )
        del m

    print()
    print("=== 按一致率从差到好 ===")
    for label, n, e, r in sorted(rows, key=lambda x: x[3]):
        print(f"  {label:<10} 一致率 {r:.4f}   相对L2 {e * 100:6.2f}%   ({n} 处)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
