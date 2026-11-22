"""启动序列最小门（账本一百三十九）：serve_gpu 在 GpuService 构造段
稳定崩 CUDA_ERROR_INVALID_VALUE，而 test_gpu_service 门（同构造）全
绿——逐步复制 serve_gpu 的启动序列并打印，二分定位崩溃步。

端口取 ServeConfig.from_env()（默认 8000）。共享主机上 8000 常被
别的服务占着：`ALOFA_PORT=<空闲端口> a100.sh run <gpu> 本文件`
（a100.sh 已透传 ALOFA_* 运行时变量）。"""

from flare.net import IpAddr, SocketAddr

from alofa.model.arch.qwen import QwenForward
from alofa.srv.config import ServeConfig
from alofa.srv.gpu_service import GpuService
from alofa.srv.loop import Loop
from alofa.srv.openai import ChatHandler
from alofa.tokenizer import load_tokenizer_json


def main() raises:
    var cfg = ServeConfig.from_env()
    print("[boot] config ok, port", Int(cfg.port), "seed", cfg.seed)
    var model = QwenForward(
        "tests/fixtures/qwen2.5-0.5b/weights",
        "tests/fixtures/qwen2.5-0.5b/config.tsv",
        512, True,
    )
    print("[boot] model ok")
    var tok = load_tokenizer_json(cfg.tokenizer_json)
    print("[boot] tokenizer ok")
    var svc = GpuService(
        model^, 8, 512, tok^, 151936, 42,
        "tests/fixtures/qwen2.5-0.5b/weights",
        "tests/fixtures/qwen2.5-0.5b/config.tsv",
        cfg.tokenizer_json,
    )
    print("[boot] gpu service ok")
    var handler = ChatHandler(svc^, cfg.model_name)
    print("[boot] handler ok")
    var addr = SocketAddr(IpAddr.parse(cfg.host), cfg.port)
    var loop = Loop.bind_addr(addr)
    print("[boot] loop bound", cfg.host + ":" + String(Int(cfg.port)))
    _ = handler
    _ = loop
    print("[boot] ALL STEPS OK (no run)")
