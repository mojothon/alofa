"""GPU 路的服务入口：`GpuService`（GpuSession 后端）→ `ChatHandler` →
reactor 循环 —— HTTP、/v1/chat/completions、SSE 流式全部复用 srv 层。

与 `serve.mojo`（CPU 路）的差异
------------------------------
* 权重：GPU 路 q4 fixture（`ALOFA_GPU_WEIGHTS` / `ALOFA_GPU_CONFIG`，
  默认 `tests/fixtures/qwen2.5-0.5b/{weights,config.tsv}`）——q4 上传 +
  f16 预反量化（账本六十八/一百二十五三段化装载）。
* 上下文窗口固定 512（prefill_step 的形状实参是 comptime，见
  `srv/gpu_service.mojo` 的 alias；`ALOFA_MAX_TOKENS` 在 GPU 路不生效）。
* 引擎线程路（`ALOFA_ENGINE_THREADS`）GPU v0 **不接**：`spawn_twin`
  已实现（自己的 session + 共享 ctx），但多线程共用一条 ctx 的并发
  安全未验证——GPU 路先走串行（`Loop.run`）。
* 采样：温度 / top_p 在 GPU tail（`sample_rb` / `sample_topp`，账本
  一百二十三/一百二十八）；top_p 的 OpenAI 参数面接线是后续工作。

用法（远程机）：
    CUDA_VISIBLE_DEVICES=3 ALOFA_RUN_SECONDS=120 \
        mojo run -I src src/alofa/serve_gpu.mojo
    curl -s localhost:8000/v1/chat/completions -d '...'
"""

from flare.net import IpAddr, SocketAddr

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from alofa.srv.config import ServeConfig, env_int, env_str
from alofa.srv.gpu_service import GpuService
from alofa.srv.loop import Loop
from alofa.srv.openai import ChatHandler
from alofa.tokenizer import load_tokenizer_json


def main() raises:
    var cfg = ServeConfig.from_env()
    var gpu_weights = env_str(
        "ALOFA_GPU_WEIGHTS", "tests/fixtures/qwen2.5-0.5b/weights"
    )
    var gpu_config = env_str(
        "ALOFA_GPU_CONFIG", "tests/fixtures/qwen2.5-0.5b/config.tsv"
    )
    # ⚠️ GPU v0 窗口固定 512：prefill_step/generate_batch 的形状实参里
    # max_tokens 是 comptime（`srv/gpu_service.mojo`），环境变量改不动它。
    comptime WINDOW = 512
    # 池大小：132 时是 N 条独立 session（显存 ≈ N × 3.36GB，上限 8）；
    # 134 重写后 N = **单 session 的批行池容量**（n_rows，上限 GPU_BMAX=64），
    # 权重/KV 只有一份，显存与 N 无关——上限随之放宽到 64。默认 1 =
    # 一百二十九的单流行为不变。
    var n_sessions = env_int("ALOFA_GPU_SESSIONS", 1, 1, 64)

    print(
        "alofa serve (gpu) — single process, HTTP with SSE streaming,"
        " GpuSession q4/f16 backend, session pool, interleaved streams"
    )
    var model = QwenForward(gpu_weights, gpu_config, WINDOW, True)
    var tokenizer = load_tokenizer_json(cfg.tokenizer_json)
    var svc = GpuService(
        model,
        n_sessions,
        WINDOW,
        tokenizer^,
        151936,
        cfg.seed,
        gpu_weights,
        gpu_config,
        cfg.tokenizer_json,
    )
    print(
        "  model loaded: " + cfg.model_name + " (gpu q4 path, window "
        + String(WINDOW) + ", session pool " + String(n_sessions) + ")"
    )
    print(
        "  spec (prompt-lookup solo greedy): "
        + ("on" if svc.spec_on else "off")
        + " (ALOFA_GPU_SPEC)"
    )
    if cfg.engine_threads != 0:
        print(
            "  [warn] ALOFA_ENGINE_THREADS ignored on the gpu path (v0"
            " serial; spawn_twin exists but ctx thread-safety is"
            " unverified)"
        )

    var handler = ChatHandler(svc^, cfg.model_name)
    var addr = SocketAddr(IpAddr.parse(cfg.host), cfg.port)
    var loop = Loop.bind_addr(addr)
    print("  listening on " + cfg.host + ":" + String(Int(cfg.port)))
    var served = loop.run(handler, cfg.max_requests, Int32(-1))
    print("  served " + String(served) + " requests")
    loop.close()
