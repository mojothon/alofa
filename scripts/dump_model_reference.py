#!/usr/bin/env python3
"""把 HuggingFace 的 Qwen2.5-0.5B 前向拆成**逐算子的参考对**，供 Mojo 侧差分。

与 `dump_reference.py`（分词 + 末位 logits）不同，这份脚本导出的是**算子级**
的中间张量：每一个 kernel 都拿到自己的输入与"权威答案"，于是第一次对不上时
能立刻定位到是哪一个算子错，而不是只知道"整网错了"。

权威答案从哪来：**全部来自 HF 的 hook 捕获，脚本里不做任何数学**。
`apply_rotary_pos_emb` 不是模块（挂不了 hook），所以它用一次函数替换来捕获
输入与输出；除此之外，每个张量都是某个 `nn.Module` 的真实输入或真实输出。
这条约束是刻意的：只要脚本自己算了一步，参照物就不再独立于被测实现。

为什么用 fp32 裸二进制 + TSV，而不是 npz / safetensors / JSON：

- Mojo 侧没有这些格式的解析器，也不该为了读测试 fixture 去写一个；
- TSV 可以直接 `git diff`，第三方能肉眼核对形状与偏移；
- 权重与激活混在一个 payload 里，靠 `tensors.tsv` 的偏移索引，避免几十个小文件。

导出物（默认落在 `tests/fixtures/qwen2.5-0.5b/`）：

| 文件 | 内容 |
|---|---|
| `config.tsv` | `key <tab> value` 的架构参数（Mojo 不解析 JSON） |
| `ids.tsv` | 切片 prompt 的 token id，每行一个 |
| `layer0/tensors.tsv` | `name <tab> dims(以 x 连接) <tab> 字节偏移 <tab> 元素数` |
| `layer0/ops.tsv` | `算子 <tab> 输入张量(,) <tab> 输出张量` —— 给人看的对照表 |
| `layer0/tensors.f32` | 上述所有张量按偏移顺序拼接的 fp32 小端裸数据 |

用法：

    python3 scripts/dump_model_reference.py                  # 第 0 层切片（默认）
    python3 scripts/dump_model_reference.py --layer 1        # 换一层
    python3 scripts/dump_model_reference.py --tokens 32      # 更长的因果掩码

注意：本机没有外网，必须用 `HF_HUB_OFFLINE=1` + 已缓存的模型；参考栈是
`/home/rontom/anaconda3/bin/python`（pixi 环境里的 python 没有 torch）。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B"

# 切片 prompt：故意选代码（数字、缩进、换行混排），16 个 token 足以让因果掩码
# 的下三角与对角线都被真正走到 —— 5 个 token 的短句只能验证到掩码的一角。
SLICE_PROMPT = (
    "def fibonacci(n):\n"
    "    if n < 2:\n"
    "        return n\n"
    "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
)
SLICE_TOKENS = 16


class Store:
    """按顺序收集张量，最后一次性写成 payload + 索引。

    名字重复直接报错而不是覆盖：一个被静默覆盖的张量会让差分测试拿错参照物，
    而它看起来只是"这个算子碰巧对得上"。
    """

    def __init__(self) -> None:
        self.order: list[str] = []
        self.arrays: dict[str, "object"] = {}

    def add(self, name: str, tensor) -> None:
        import numpy as np
        import torch

        if name in self.arrays:
            raise SystemExit("张量名重复: %s" % name)
        if isinstance(tensor, (tuple, list)):
            raise SystemExit("%s 是序列而不是张量，取哪一个元素要写清楚" % name)
        # 立刻转成 C 连续的 fp32 numpy：后面所有写法都按 numpy 来，避免一半 torch
        # 一半 numpy 的隐式规则（torch 的 .size 是方法、.astype 不存在）。
        arr = tensor.detach().to(torch.float32).cpu().numpy()
        arr = np.ascontiguousarray(arr.astype(np.float32))
        if not np.isfinite(arr).all():
            raise SystemExit("%s 含 NaN/Inf，参照物不可用" % name)
        self.order.append(name)
        self.arrays[name] = arr

    def write(self, out: Path, aliases: list[tuple[str, str]] = ()) -> int:
        offsets: dict[str, int] = {}
        payload = bytearray()
        for name in self.order:
            arr = self.arrays[name]
            # 每个张量从 4 字节边界开始：Mojo 侧按 fp32 索引，非对齐偏移会让
            # 一次加载跨到上一个张量的尾巴上。
            while len(payload) % 4 != 0:
                payload += b"\x00"
            offsets[name] = len(payload)
            payload += arr.tobytes()

        (out / "tensors.f32").write_bytes(bytes(payload))

        with (out / "tensors.tsv").open("w", encoding="utf-8") as f:
            for name in self.order:
                arr = self.arrays[name]
                dims = "x".join(str(d) for d in arr.shape)
                f.write(
                    "%s\t%s\t%d\t%d\n"
                    % (name, dims, offsets[name], int(arr.size))
                )
            # 绑定的输出投影与词嵌入是同一块存储：各写一行索引，指向同一个
            # 偏移。加载侧因此不需要知道 tie 这回事，也就没有"忘了 tie"的 bug。
            for alias, target in aliases:
                arr = self.arrays[target]
                dims = "x".join(str(d) for d in arr.shape)
                f.write(
                    "%s\t%s\t%d\t%d\n"
                    % (alias, dims, offsets[target], int(arr.size))
                )
        return len(payload)


def capture_slice(model, tokenizer, layer_idx: int, n_tokens: int, store: Store) -> dict:
    """跑一次前向，把第 `layer_idx` 层的算子边界全部录下来。"""
    import numpy as np
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

    ids = tokenizer.encode(SLICE_PROMPT, add_special_tokens=False)[:n_tokens]
    if len(ids) < 2:
        raise SystemExit("切片 token 太少，因果掩码形同没有")

    layers = model.model.layers
    if layer_idx >= len(layers):
        raise SystemExit("层号越界: %d >= %d" % (layer_idx, len(layers)))
    layer = layers[layer_idx]
    attn = layer.self_attn
    mlp = layer.mlp

    handles = []

    def on(name_in, name_out):
        def hook(module, args, output):
            if name_in:
                store.add(name_in, args[0])
            if name_out:
                store.add(name_out, output)

        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(on("", "embed_out")))
    handles.append(layer.input_layernorm.register_forward_hook(on("", "norm_out")))
    handles.append(layer.post_attention_layernorm.register_forward_hook(on("hidden1", "norm2_out")))

    # q/k/v 的**输出**是 RoPE 之前的投影结果；RoPE 之后的版本另由函数替换捕获。
    handles.append(attn.q_proj.register_forward_hook(on("", "q_out")))
    handles.append(attn.k_proj.register_forward_hook(on("", "k_out")))
    handles.append(attn.v_proj.register_forward_hook(on("", "v_out")))
    # o_proj 的**输入**就是注意力算子的输出 —— 这是注意力唯一干净的边界。
    handles.append(attn.o_proj.register_forward_hook(on("attn_out", "o_out")))
    handles.append(mlp.gate_proj.register_forward_hook(on("", "gate_out")))
    handles.append(mlp.up_proj.register_forward_hook(on("", "up_out")))
    handles.append(mlp.act_fn.register_forward_hook(on("", "silu_out")))
    # down_proj 的输入就是 HF 自己算的 silu(gate) * up，不需要脚本去乘一遍。
    handles.append(mlp.down_proj.register_forward_hook(on("swiglu_out", "")))

    original_rope = qwen2_mod.apply_rotary_pos_emb
    # `apply_rotary_pos_emb` 是模块级函数，替换它会影响**所有**层；所以只有第
    # `layer_idx` 次调用才录制。（钩子那边不存在这个问题：钩子挂在具体的模块上。）
    state = {"calls": 0}

    def patched_rope(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        out = original_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
        mine = state["calls"] == layer_idx
        state["calls"] += 1
        if not mine:
            return out
        # q: [1, heads, T, head_dim] -> [T, heads * head_dim]（batch 恒为 1）
        store.add("q_rot", out[0].transpose(1, 2).reshape(q.shape[2], -1))
        store.add("k_rot", out[1].transpose(1, 2).reshape(k.shape[2], -1))
        # cos/sin 是 [T, head_dim]（这个版本已经没有 batch 维）—— 存的是"按位置
        # 展开后"的表，Mojo 侧按 t 直接取行，不再复现 inv_freq。
        store.add("cos", cos if cos.dim() == 2 else cos[0])
        store.add("sin", sin if sin.dim() == 2 else sin[0])
        return out

    qwen2_mod.apply_rotary_pos_emb = patched_rope
    try:
        with torch.no_grad():
            input_ids = torch.tensor([ids], dtype=torch.long)
            # q/k/v 的输出是 [1, T, H]，存成 [T, H] 去掉无意义的 batch 维。
            _ = model(input_ids)
    finally:
        qwen2_mod.apply_rotary_pos_emb = original_rope
        for h in handles:
            h.remove()

    # 展平 batch 维：所有这些量都是 batch=1，留着它只会让 Mojo 侧多一层索引。
    for name in ("embed_out", "norm_out", "hidden1", "norm2_out", "q_out", "k_out",
                 "v_out", "attn_out", "o_out", "gate_out", "up_out", "silu_out",
                 "swiglu_out"):
        arr = store.arrays[name]
        if arr.ndim == 3 and arr.shape[0] == 1:
            store.arrays[name] = np.ascontiguousarray(arr[0])

    # 自检：钩子挂错位置是这类脚本最典型的失败（挂到上一层、挂成输出而不是输入），
    # 而它不会报错，只会让 Mojo 侧"怎么都对不上"。这里只用**不参与导出**的临时
    # 量做恒等式检查，导出的 swiglu_out / hidden1 仍然是 HF 的原始张量。
    a = store.arrays
    swiglu_check = a["silu_out"] * a["up_out"]
    hidden1_check = a["embed_out"] + a["o_out"]

    def close(name: str, expected) -> None:
        d = float(np.abs(a[name] - expected).max())
        if d > 1e-5:
            raise SystemExit("钩子可能挂错了：%s 与恒等式相差 %g" % (name, d))

    close("swiglu_out", swiglu_check)
    close("hidden1", hidden1_check)
    return {"ids": ids, "n_tokens": len(ids)}


def write_ops(out: Path) -> None:
    """人工可读的算子对照表：哪一列张量喂给哪个算子、应该对上哪一个。"""
    rows = [
        ("rmsnorm", "embed_out,norm_w", "norm_out"),
        ("linear_bias", "norm_out,q_w,q_b", "q_out"),
        ("linear_bias", "norm_out,k_w,k_b", "k_out"),
        ("linear_bias", "norm_out,v_w,v_b", "v_out"),
        ("rope", "q_out,k_out,cos,sin", "q_rot,k_rot"),
        ("attention", "q_rot,k_rot,v_out", "attn_out"),
        ("linear", "attn_out,o_w", "o_out"),
        ("add", "embed_out,o_out", "hidden1"),
        ("rmsnorm", "hidden1,norm2_w", "norm2_out"),
        ("silu", "gate_out", "silu_out"),
        ("swiglu", "gate_out,up_out", "swiglu_out"),
    ]
    with (out / "ops.tsv").open("w", encoding="utf-8") as f:
        for op, ins, outs in rows:
            f.write("%s\t%s\t%s\n" % (op, ins, outs))


def write_config(cfg, out: Path) -> None:
    n_heads = int(cfg.num_attention_heads)
    hidden = int(cfg.hidden_size)
    rows = [
        ("n_layers", int(cfg.num_hidden_layers)),
        ("hidden", hidden),
        ("n_heads", n_heads),
        ("n_kv_heads", int(getattr(cfg, "num_key_value_heads", n_heads))),
        ("head_dim", hidden // n_heads),
        ("intermediate", int(cfg.intermediate_size)),
        ("vocab", int(cfg.vocab_size)),
        ("eps", float(cfg.rms_norm_eps)),
        ("rope_theta", float(getattr(cfg, "rope_theta", 10000.0))),
        ("tie_word_embeddings", int(bool(getattr(cfg, "tie_word_embeddings", False)))),
        ("max_position_embeddings", int(getattr(cfg, "max_position_embeddings", 0))),
    ]
    with (out / "config.tsv").open("w", encoding="utf-8") as f:
        for key, value in rows:
            # 浮点数一律定点写法：Mojo 侧的解析只认十进制小数，而 `repr(1e-06)`
            # 会给科学计数法，多一个解析分支就多一处出错的地方。
            text = value if isinstance(value, int) else "%.12f" % value
            f.write("%s\t%s\n" % (key, text))


def dump_full(model, tokenizer, out: Path, max_positions: int, greedy_tokens: int) -> dict:
    """全部参数 + 数值门参考：这是整网前向差分的输入与答案。

    参数按 `state_dict` 的顺序写进 `weights/`，绑定的输出投影写成别名行。
    旋转表直接取 `rotary_emb` 缓存的前 `max_positions` 行 —— 复现 inv_freq
    是加载器的事故源，导出已经算好的表则让 Mojo 侧只做查表。

    greedy 参考用 KV cache 逐 token 生成（快约两个数量级），再用**无 cache
    的全序列重放**核对前 8 个 token —— cache 路径与全序列路径在参考实现里
    就应当一致，这里只是把"应当"变成"已核对"。
    """
    import torch
    from dump_reference import PROMPTS
    from transformers import DynamicCache

    weights_dir = out / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    store = Store()
    seen: dict[int, str] = {}
    aliases: list[tuple[str, str]] = []
    for name, tensor in model.state_dict().items():
        ptr = tensor.detach().data_ptr()
        if ptr in seen:
            aliases.append((name, seen[ptr]))
            continue
        seen[ptr] = name
        store.add(name, tensor)

    if not any(alias == "lm_head.weight" for alias, _ in aliases) and (
        "lm_head.weight" not in store.arrays
    ):
        if not model.config.tie_word_embeddings:
            raise SystemExit("输出投影未绑定也未单独存储，导出策略需要重新考虑")
        aliases.append(("lm_head.weight", "model.embed_tokens.weight"))

    # 旋转表挂在每个注意力模块下（这份版本里不在顶层模型上），各层配置相同，
    # 取第 0 层的即可。
    rotary = model.model.layers[0].self_attn.rotary_emb
    store.add("rope_cos", rotary.cos_cached[:max_positions])
    store.add("rope_sin", rotary.sin_cached[:max_positions])
    payload_bytes = store.write(weights_dir, aliases)

    prompts = [
        {"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)}
        for text in PROMPTS
    ]
    (out / "prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # 同一份数据的两种写法：JSON 给人核对，TSV 给 Mojo 读。Mojo 侧没有 JSON
    # 解析器，也不该为了读一个测试基准去写一个。
    with (out / "prompts.tsv").open("w", encoding="utf-8") as f:
        for item in prompts:
            f.write(
                "%s\t%s\n"
                % (
                    item["text"].encode("utf-8").hex(),
                    ",".join(str(i) for i in item["ids"]),
                )
            )

    greedy = []
    logits_rows = []
    with torch.no_grad():
        for item in prompts:
            input_ids = torch.tensor([item["ids"]], dtype=torch.long)
            cache = DynamicCache()
            step = model(input_ids, past_key_values=cache, use_cache=True)
            logits_rows.append(step.logits[0, -1].to(torch.float32).numpy())

            nxt = int(step.logits[0, -1].argmax())
            generated = []
            for _ in range(greedy_tokens):
                generated.append(nxt)
                x = torch.tensor([[nxt]], dtype=torch.long)
                step = model(x, past_key_values=cache, use_cache=True)
                nxt = int(step.logits[0, -1].argmax())
            greedy.append(generated)

            # 重放核对：无 cache 全序列前向的前 8 个 greedy token 必须一致。
            replay = list(item["ids"])
            replayed = []
            for _ in range(min(8, greedy_tokens)):
                x = torch.tensor([replay], dtype=torch.long)
                nxt = int(model(x).logits[0, -1].argmax())
                replayed.append(nxt)
                replay.append(nxt)
            if replayed != generated[: len(replayed)]:
                raise SystemExit(
                    "cache 路径与全序列路径不一致：%r vs %r" % (replayed, generated[: len(replayed)])
                )

    vocab_size = int(logits_rows[0].shape[0])
    with (out / "logits_last.f32").open("wb") as f:
        for row in logits_rows:
            f.write(struct.pack("<%df" % vocab_size, *row.tolist()))
    (out / "greedy.json").write_text(json.dumps(greedy), encoding="utf-8")
    with (out / "greedy.tsv").open("w", encoding="utf-8") as f:
        for row in greedy:
            f.write("%s\n" % ",".join(str(i) for i in row))

    return {
        "n_prompts": len(prompts),
        "vocab_size": vocab_size,
        "greedy_tokens": greedy_tokens,
        "max_positions": max_positions,
        "weights_bytes": payload_bytes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default="tests/fixtures/qwen2.5-0.5b")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=SLICE_TOKENS)
    ap.add_argument(
        "--full",
        action="store_true",
        help="导出全部参数与数值门参考（约 2 GB，产物在 weights/ 下，不进版本库）",
    )
    ap.add_argument("--max-positions", type=int, default=512)
    ap.add_argument("--greedy", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.out)
    slice_dir = out / ("layer%d" % args.layer)
    slice_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        # eager 注意力才是逐行可读的 matmul → mask → softmax → matmul；sdpa
        # 把这些融进一个内核，我们想差分的是公式本身而不是融合后的结果。
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()

    write_config(model.config, out)

    store = Store()
    info = capture_slice(model, tokenizer, args.layer, args.tokens, store)

    layer = model.model.layers[args.layer]
    store.add("norm_w", layer.input_layernorm.weight)
    store.add("norm2_w", layer.post_attention_layernorm.weight)
    store.add("q_w", layer.self_attn.q_proj.weight)
    store.add("q_b", layer.self_attn.q_proj.bias)
    store.add("k_w", layer.self_attn.k_proj.weight)
    store.add("k_b", layer.self_attn.k_proj.bias)
    store.add("v_w", layer.self_attn.v_proj.weight)
    store.add("v_b", layer.self_attn.v_proj.bias)
    store.add("o_w", layer.self_attn.o_proj.weight)

    payload_bytes = store.write(slice_dir)
    write_ops(slice_dir)
    (slice_dir / "ids.tsv").write_text(
        "".join("%d\n" % i for i in info["ids"]), encoding="utf-8"
    )

    manifest = {
        "model": args.model,
        "layer": args.layer,
        "n_tokens": info["n_tokens"],
        "n_tensors": len(store.order),
        "payload_bytes": payload_bytes,
        "python": sys.version.split()[0],
        "generated_with": "scripts/dump_model_reference.py",
    }
    if args.full:
        manifest.update(
            dump_full(model, tokenizer, out, args.max_positions, args.greedy)
        )
    (slice_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("切片产物: %.2f MB -> %s" % (payload_bytes / 1e6, slice_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
