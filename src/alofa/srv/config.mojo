"""服务运行配置：环境变量 → 值，默认与旧编译期常量一字不差。

为什么是环境变量而不是命令行：Mojo 1.0 的 `sys.argv` 在编译产物里是空的
（`serve.mojo` 的文件头写了这段历史），而 `std.os.getenv` 在编译产物里实测
可用（`tests/capability/test_deps.mojo` 钉着这一条）。这也正好是 systemd
`Environment=` 的注入路径（`scripts/deploy/alofa.service`）。

规则：没设的变量取默认；设了但值不合法（非整数 / 越界）直接报错。部署配置
静默回退默认值，是把「配错了」变成「跑在错的端口上」——那更难查。
"""

from std.os import getenv

from alofa.core.error import ERR_INVALID_ARGUMENT, AlofaError
from alofa.core.text import parse_int, read_text


def env_str(name: String, default: String) -> String:
    """读一个环境变量；没设（或设成空串）取默认。"""
    var raw = getenv(name, "")
    if raw.byte_length() == 0:
        return default
    return raw^


def env_int(name: String, default: Int, lo: Int, hi: Int) raises -> Int:
    """读一个整数环境变量，范围外的值与不是整数的值一样报错。"""
    var raw = getenv(name, "")
    var value = default
    if raw.byte_length() > 0:
        try:
            value = parse_int(raw)
        except err:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "environment variable is not an integer",
                name + "=" + raw + " (" + String(err) + ")",
            )
    if value < lo or value > hi:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "environment variable is out of range",
            name + "=" + String(value) + " not in [" + String(lo) + ", "
            + String(hi) + "]",
        )
    return value


@fieldwise_init
struct ServeConfig(Movable):
    """一次进程启动的全部可调项。默认值 = 2026-09 之前的编译期常量。"""

    var host: String
    var port: UInt16
    var weights_dir: String
    var config_path: String
    var weights_file: String
    var tokenizer_json: String
    var model_name: String
    var max_tokens: Int
    var max_prompt_tokens: Int
    var seed: Int
    # 0 = 自动：单进程 = 核数（与现状一致）；多 worker = 1（fork 后 asyncrt
    # 不可用，见 serve.mojo）。>0 = 每个 worker 固定用这么多分片。
    var shards: Int
    var workers: Int
    # 0 = 一直跑（生产）；>0 = 到点自己优雅收工（门与压测用）。
    var run_seconds: Int
    # 优雅退出宽限：SIGTERM 之后给在途请求这么多毫秒答完，再不退就 SIGKILL。
    var grace_ms: Int
    # 0 = 无限答；>0 = 每个 worker 答满这个数就退出（门用）。
    var max_requests: Int
    # 每个 worker（单进程时就是那一个进程）里 **engine 线程的条数** = 同时能生成的
    # 条数（roadmap 3.2c）。
    #
    # **默认 1** = 生成走 engine 线程（`Loop.run_threaded`），reactor 循环只管
    # accept/解析/回写 —— 一条 5 s 的生成不再卡住别的连接（账本四十：`test-engine`
    # 的"生成期间循环还在 accept"性质 + 三种断开性质全绿）。设 **0** 退回串行路
    # （`Loop.run`，生成就在循环上）：单请求路径少一次跨线程往返，但要独占循环。
    # 两条路的历史病都已根因修复：堆腐坏 = flare `Mutex`（账本三十九，邮箱已换
    # `RawMutex`，soak FAKE 0/6、真模型 0/3）；断开串味 = `ends_sse_stream` 把
    # 非流式应答误标流结束（账本四十）。
    #
    # ⚠️ **N 条 = N 份权重**：`handler.spawn_twin` 是重新加载一份（现在还没有"共享
    # 只读权重"那一层），所以把它调大之前先算内存。想并发又想让权重真的共享，加
    # worker（进程级共享靠 fork 的写时复制）。
    var engine_threads: Int
    # 权重档：0 = fp32（默认），1 = 体走 q4_0（加载时在线量化），
    # 2 = 体 q4_0 + 输出头 q8_0。
    #
    # ⚠️ **默认留 0**，因为 0 是"与参考实现逐字节相同"的那一档：q4 是有损的，
    # 教师强制贪心一致率也是按量化档单独量的（见账本）。但 0 **不是**流量最小的
    # 那一档 —— 每 token 要读 1975.8 MB，而账本里 q4_0+q8_head 只有 345.9 MB
    # （5.7×）。decode 是带宽受限的，所以这个档位直接决定每 token 的墙钟。
    # ⚠️ 逐字节稳定（2026-11-04 账本三十一修订）：q4 档"批==单"在 mub 开关两侧
    # 都成立（mub 宽带==窄带逐位，并发 vs 单跑 8/8 一致）；mub 只是与 fp32 参考
    # 实现不逐位 —— 那是量化档的本性。MUB=0 只在与 fp32 对齐的场合才需要。
    var weight_bits: Int

    @staticmethod
    def from_env() raises -> ServeConfig:
        var weights = env_str("ALOFA_WEIGHTS", "tests/fixtures/qwen2.5-0.5b-hf")
        # 权重形态解析（一百八十）：目录下有 `model.safetensors` → 单文件路，
        # 传文件路径；只有 `model.safetensors.index.json` + N 片（大 checkpoint
        # 的标准形态）→ 分片路，TensorFile 拿目录自己去读 index。
        var weights_file = weights + "/model.safetensors"
        try:
            _ = read_text(weights + "/model.safetensors.index.json")
            weights_file = weights
        except:
            pass
        return ServeConfig(
            host=env_str("ALOFA_HOST", "127.0.0.1"),
            port=UInt16(env_int("ALOFA_PORT", 8000, 1, 65535)),
            weights_dir=weights,
            config_path=weights + "/config.json",
            # `QwenForward` 的 `params_dir` 要是权重文件自己（见 serve.mojo 的
            # 历史注释），不是目录。
            weights_file=weights_file,
            tokenizer_json=env_str(
                "ALOFA_TOKENIZER", "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
            ),
            model_name=env_str("ALOFA_MODEL_NAME", "qwen2.5-0.5b"),
            # ⚠️ 这是**上下文窗口**（prompt + 生成的总槽位），不是生成步数上限：
            # `QwenForward` 拿它分配 KV。单个请求的生成步数 = 窗口 − prompt。
            max_tokens=env_int("ALOFA_MAX_TOKENS", 256, 1, 4096),
            max_prompt_tokens=env_int("ALOFA_MAX_PROMPT_TOKENS", 128, 1, 4096),
            seed=env_int("ALOFA_SEED", 1234, 0, 2_147_483_647),
            shards=env_int("ALOFA_SHARDS", 0, 0, 64),
            workers=env_int("ALOFA_WORKERS", 1, 1, 64),
            run_seconds=env_int("ALOFA_RUN_SECONDS", 0, 0, 86_400),
            grace_ms=env_int("ALOFA_GRACE_MS", 120_000, 1, 3_600_000),
            max_requests=env_int("ALOFA_MAX_REQUESTS", 0, 0, 2_000_000_000),
            # 默认 1（账本四十：两条路的历史病都已根因修复，重门 + soak 全绿后翻转）；
            # 0 = 串行（生成在循环上），仍是受支持的退路。
            engine_threads=env_int("ALOFA_ENGINE_THREADS", 1, 0, 16),
            weight_bits=env_int("ALOFA_WEIGHT_BITS", 0, 0, 2),
        )
