# Install

## Prerequisites

| Requirement | Notes |
|---|---|
| Linux x86-64 | the CPU path uses AVX2; the scalar backend works everywhere |
| [pixi](https://pixi.sh) | the only build tool — it pins Mojo and all deps per `pixi.toml` |
| NVIDIA GPU (A100-class) + CUDA driver | only for the GPU path; kernels compile through Mojo's CUDA target, no separate nvcc project |
| ~2 GB disk for weights | only for model-level gates and serving with real weights |

## Clone and bootstrap

```bash
git clone https://github.com/mojothon/alofa && cd alofa
pixi run test       # full CPU suite + the capability-ledger gate, ~4.5 min
```

`pixi run test` is the front door: if it is green, your environment is
correct end to end (tokenizer parity, layer0 parity, scheduler, engine,
HTTP gates, ledger gate).

## Weights

alofa runs **Qwen2.5-0.5B**. Three weight forms exist:

1. **Test fixtures** (in-repo): `tests/fixtures/qwen2.5-0.5b/` — fp32
   operator-level reference payloads exported from HuggingFace, used by the
   numeric gates. No download needed; `pixi run test` works out of the box.
2. **Full weights for serving / model gates** (default
   `ALOFA_WEIGHTS=tests/fixtures/qwen2.5-0.5b-hf`): pull the HF snapshot and
   build the fp32 dump directory:

   ```bash
   huggingface-cli download Qwen/Qwen2.5-0.5B   # lands in ~/.cache/huggingface
   pixi run python scripts/build_real_model_dir.py
   ```

3. **HF safetensors directly**: `config.json` + safetensors (bf16 widened to
   fp32 in place) can be loaded straight from the snapshot directory —
   point `ALOFA_WEIGHTS` at it.

## Remote GPU host

GPU development targets a **shared** 6×A100 host. It has no outbound
network, so the pixi environment is synced over rsync and only idle cards
may be used:

```bash
./scripts/a100.sh gpu      # pick an idle card (0-3 are often busy)
./scripts/a100.sh sync     # sync the project, including .pixi
./scripts/a100.sh run 3 tests/gpu/vecadd.mojo   # smoke test on GPU 3
```

Everything in the ledger marked "仅远程验证机" must be reproduced this way.

## Verify the GPU path

```bash
./scripts/a100.sh run 3 tests/gpu/test_session_decode.mojo   # session gate
./scripts/a100.sh run 3 tests/gpu/test_gpu_service.mojo      # service gate
```

Both print an explicit `PASS` banner; the full 23-gate GPU suite is listed in
[developing.md](developing.md#gates).
