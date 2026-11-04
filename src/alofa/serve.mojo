"""OpenAI 兼容的 HTTP 服务入口（单进程或多 worker，非流式 + SSE 流式）。

用法
----
    pixi run serve                                        # 单进程（默认）
    ALOFA_WORKERS=4 ALOFA_HOST=0.0.0.0 pixi run serve     # 多 worker（P3.4）

配置来自环境变量（`srv/config.mojo`）：端口、监听地址、权重目录、tokenizer、
worker 数、生成上限、分片数、优雅退出宽限……默认值与 2026-09 之前的编译期
常量一字不差。为什么是环境变量而不是命令行：Mojo 1.0 的 `sys.argv` 在编译产物
里是空的，而 `std.os.getenv` 实测可用（`tests/capability/test_deps.mojo`）。

权重只加载一次
--------------
多 worker 模式下这一条更重要：权重在 fork **之前**加载，子进程靠写时复制共享
同一份（fork 时父进程还没跑过任何前向），每个 worker 的私有成本只有会被写的
页（KV/激活）。单进程模式的加载路径与此完全相同。

每次请求先 `reset()`
--------------------
这修的是一个安静的 bug：复用不重置的话，`prefill` 在第二个请求上会**报错**
（好查），但 `step` 会**安静地把两个请求缝在一起**（难查，输出看起来像人话）。
`reset()` 每请求必调 —— 多 worker 下每个 worker 的语义与单进程一致。

采样
----
`temperature <= 0` 走贪心（`model.argmax`，逐 token 确定性）；`> 0` 走
`Sampler`（温度生效，seed 每请求重置 —— 同一请求体在同一 worker 上可复现，
但**不保证**跨 worker 相同：每个 worker 都是独立的 `next_id` 计数器）。

流式（`stream=true`）
--------------------
逐 token 吐 SSE 帧（见 `srv/sse.mojo`）。前向是**惰性**的：第一次
`stream_next` 才 prefill —— 这样 `stream_begin` 里那点校验（prompt 分词为空 /
过长）还能在响应头发出去之前回成 400。

每步的文本用**整段前缀**解码再取新增的那一截（`stream_emitted`），而不是"把这
个 token 单独解码"：多字节字符可能跨两个 token，单独解码时前半截是不完整的
UTF-8，`decode` 会把它丢掉 —— 那样流式拼出来的文本会**不等于**非流式那份。
整段前缀解码的代价是每步重解一遍（≤ 256 步，几十 KB），换来的是两边逐字节
一致。

替代物
------
`llama.cpp` 的 server。差异：一个 worker 一条 reactor 循环（`srv/loop.mojo`），
上面同时挂着多条连接（旧形态是「一个 worker 同时只处理一条连接」，多 worker 只修
进程数、修不了单 worker 的串行）。生成默认**就在这条循环上**（`ALOFA_ENGINE_THREADS=0`
= 串行，一次生成占住这条循环直到它结束）。`>=1` 才把生成搬到 `srv/engine_thread.mojo`
的 engine 线程上（roadmap 3.2b → 3.2c），同时能生成几条 = 那个数（每多一条就多一份权
重，那是 3.2c 明文写下的代价）—— ⚠️ **那条路是实验的**：真模型 + 客户端中途断开会把
请求搞坏（堆被踩，见账本 §8），所以它不默认开。兼容
面是 OpenAI 的 `/v1/chat/completions`：不改代码、只改 `base_url` 就能用官方 SDK
打到它。
"""

from std.collections import List
from std.os import getenv

from flare.net import IpAddr, SocketAddr

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    ERR_STALLED,
    AlofaError,
)
from alofa.core.rng import Rng
from alofa.core.tensor import F32Ptr
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward, q4_mub_enabled
from alofa.model.profiler import ALOFA_PROFILE, K_DELIV, K_DETOK, K_PICK, K_SETTLE, K_WALL
from alofa.runtime.kv import MAX_SEQ_TOKENS
from alofa.runtime.sampler import LogitBias, SampleParams, Sampler
from alofa.srv.config import env_int, env_str, ServeConfig
from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import bytes_of_text, bytes_to_text
from alofa.srv.loop import MAX_CONNS, Loop
from alofa.srv.master import run_workers
from alofa.core.ffi.mem import raise_stack_limit
from alofa.core.memory import Arena
from alofa.engine.core import MAX_PROMPT, EngineCore
from alofa.engine.executor import MAX_BATCH, MAX_GEN, NO_TOKEN, int_map
from alofa.engine.scheduler import SchedConfig
from alofa.srv.openai import NO_REQUEST, ChatHandler, Completion, Service
from alofa.srv.sse import StreamToken
from alofa.tokenizer import Tokenizer, load_tokenizer_json


# 批调度这一侧的配置。⚠️ `capacity_blocks` 是**故意的小**：一块是 16 个 token 的
# K/V，1024 块在 0.5B 上是几百 MB；而引擎同一时刻最多 `MAX_BATCH`(= 8) 条、每条最多
# `MAX_PROMPT` + `MAX_GEN` = 160 个 token —— 80 块就够，给 256 块是三倍余量（前缀
# 缓存也记在这本账上）。
comptime BATCH_ROWS = 64
comptime BATCH_BLOCK = 16
comptime BATCH_CAP_BLOCKS = 256
comptime BATCH_WATERMARK = 900
comptime BATCH_MAX_WAIT = 8
# 一条流最多推引擎多少拍还没等到下一个 token，就算**卡住**（`ERR_STALLED`）。
#
# 为什么必须有上界：那个循环原来只有两个出口 —— 引擎交出 token，或 `has_work()`
# 变假。两条都不成立时（无调试信息的 -O2 下实测就会），客户端一个字节都收不到，
# 而服务端还在算：从外面看是"服务活着、请求卡死"，分不清是死循环还是还在排队。
# 有了上界，"等不到"变成一帧**指名错误**，卡在哪条请求上一眼看得见。
#
# 为什么是 256：正常每 token 一拍（批满时加上 prefill 分块也不过十几拍），256 是
# 一个数量级的余量 —— 上界只用来兜住"永远不前进"，不该在正常的排队等待时先跳出来。
comptime MAX_PROGRESS_TICKS = 256


def load_weights(weights_file: String, config_path: String, max_tokens: Int, weight_bits: Int) raises -> QwenForward:
    """按 `weight_bits` 装一份模型（0 = fp32，1 = 体 q4_0，2 = 体 q4_0 + 头 q8_0）。

    ⚠️ 量化是**加载后在线做的**：fp32 原始映射仍然留着（`enable_q4` 的注释里写了
    为什么先不丢），所以这一档省的是每 token 的**读流量**，不是常驻内存。
    """
    var quantize = weight_bits >= 1
    var model = QwenForward(weights_file, config_path, max_tokens, quantize)
    if weight_bits >= 2:
        model.enable_q8_output()
    # 剂量-响应档位（账本三十二）：`ALOFA_JUNK_ARM=3` = 双份加料（数值逐位相同，
    # 只多 ~30% 指令）。同二进制两臂，重启即切 —— 判"解码体离带宽墙的差，是核内
    # 指令还是访存"：加料后时间线性涨 ⇒ 指令受限；平坦 ⇒ 访存受限。
    if env_str("ALOFA_JUNK_ARM", "0") == "3":
        model.set_junk_arm(3)
        print("  [srv] note: ALOFA_JUNK_ARM=3 (dose-response probe)")
    if weight_bits > 0 and q4_mub_enabled():
        # mub 与精确通路不是逐位一致（多一层激活量化，核级差 1e-7 量级、贪心一致
        # 率单钉）。但"批==单"这条硬判据它**过得去**：行带核每行的累加链与 RB/RU
        # 分组无关（每行仍按块序、行末归约），宽带与窄带逐位相同 —— 2026-11-04
        # （账本三十一）实测：温 0 下 8 条相同/不同 prompt 并发 vs 各自单跑都是
        # 8/8 逐字节一致，且各宽度都快于 `ALOFA_Q4_MUB=0`（宽 8 聚合 54–56 vs
        # 47–49 tok/s）。二十三那笔"随批组合漂移"的账，祸根是宽带**退精确通路**
        # 的回退，回退改成降窄带后就不存在了。
        print(
            "  [srv] note: ALOFA_Q4_MUB is on (default); batch-vs-single greedy "
            "streams stay byte-identical, mub differs from the fp32 reference"
        )
    return model^


def batch_default() -> Bool:
    """批路的默认开关：`ALOFA_BATCH=0` 关掉，其余（含没设）都是开。

    只认 `0`/非 `0` 两个态，不解析其它值 —— 一个"关批调度"的开关不需要三态，
    多出来的态只是多一种写错的方式（写错就退到默认，也就是开着，而那是**安全**
    的那一侧：开着若有问题，门会先红）。
    """
    var raw = getenv("ALOFA_BATCH", "")
    if raw.byte_length() == 0:
        return True
    return raw != "0"


struct ModelService(Service, Twinable):
    """一次加载、多次请求：每次 complete 都从干净状态开始。

    流式有**两条**路（这不是偷懒，是两条路的形状不同）：

    * **批路**（`EngineCore`）：一次前向推进多条请求，是 p99 那条收益的来源。它只
      收"装得进引擎"的请求 —— prompt ≤ `MAX_PROMPT`(= 128)、要的新 token ≤
      `MAX_GEN`(= 32)、同一时刻 ≤ `MAX_BATCH`(= 8) 条，而且**目前只收贪心**
      （`temperature <= 0`）。
    * **单流老路**（`model.prefill` / `model.step`）：形状越界或带温度的请求走它。
      它一次只握一条（第二条会被指名拒绝，而不是静默串行）。

    ⚠️ **默认 `temperature` 是 1.0**（`srv/openai.mojo`），所以**默认请求走的是老
    路** —— 批调度今天只覆盖显式要贪心的客户端。这条不是疏漏，是"采样批化"还没
    被逐 token 验过（验过之后才会把它也搬上批路）；它写在这里，也该写进能力账本。

    两条路**可以同时**有流在跑：批路的 K/V 在引擎自己的池里，老路的在 model 里，
    而 `QwenForward` 没有跨调用的位置状态（`t` 是每次前向的局部变量），所以交错
    调用同一份权重是安全的 —— 这也是它们能共用一份权重的原因。
    """

    var model: QwenForward
    var tokenizer: Tokenizer
    var vocab: Int
    var max_tokens: Int
    var max_prompt_tokens: Int
    var seed: Int
    # 采样器与随机数：流式要跨 `stream_next` 调用留着（`Sampler` 的缓冲区按构造
    # 一次分配，逐 token 重建等于每步几十 KB 的分配）。非流式那条路没动 —— 它
    # 是已经验过的路径，不为了省一点内存去改它。
    var sampler: Sampler
    var params: SampleParams
    var rng: Rng
    # 流式在途状态。
    var stream_ids: List[Int]
    var stream_out: List[Int]
    var stream_steps: Int
    var stream_count: Int
    var stream_last: Int
    var stream_emitted: Int
    # 二分诊断刀第二把（账本三十五）：`ALOFA_ET_CUT_TICK=1` 时 `_batch_next` 不推
    # `_batch_tick`（decide→forward→settle 整段缺席），交 done 终点。_admit/engine
    # 记账/submit 与整条 mailbox/collect 管道保真。
    var cut_tick: Bool
    # 第三把刀：`ALOFA_ET_CUT_DECIDE=1` 时 `_batch_tick` 不跑 `decide`（前向缺席）
    # 也不 pick，只做 `step_choices` 清 `NO_TOKEN` + `settle`（与 except 路同款）。
    # 活⇒写者在 decide/forward 或 pick；崩⇒写者在 choices 写或 settle。
    var cut_decide: Bool
    var stream_temp: Float64
    var stream_started: Bool
    # 在途那条流的**请求号**（`NO_REQUEST` = 没有）。这一整块状态只属于它 ——
    # `stream_next` / `stream_end` 每次都核对，号对不上就红而不是"接着给另一条流
    # 生成"。它服务的是下面那条**单流老路**（采样，或装不进引擎的请求）；
    # 批路的状态在 `b_*` 那几张表里。
    var stream_request: Int

    # 批路的开关。**默认开**。
    #
    # ⚠️ 这个开关不是给运维的旋钮，是给**门**留的：同一条 prompt 要能分别走两条
    # 路（一批 N 条 vs 一条一条跑），才对得上"逐 token 相等"那笔账 —— 没它，两条
    # 路只能各自跑，差异会被"反正 prompt 不一样"永远盖住。
    #
    # 关掉它（`ALOFA_BATCH=0`）就是退回批调度之前那条路：一条一条跑。所以它也
    # 是"批调度出问题时"的退路 —— 明着慢，而不是静默变慢。
    var batch_enabled: Bool
    # 批路开窗统计（`ALOFA_BATCH_STATS=1` 才开，默认零输出）：一拍一行太吵、没有又
    # 判不了"批到底拼起来没有" —— tick 级探针说 2.36×，端到端却只多 7%，第一个要
    # 看的数就是**宽度**。`window_*` 每 64 拍清一次，`ticks` 是本次进程的总数。
    var batch_stat_on: Bool
    var batch_ticks: Int
    var t_step_last: Int
    var batch_win_ticks: Int
    var batch_win_widths: Int
    var batch_win_max: Int

    # 批调度（P2 已 verified 的那条执行器）：一次前向推进**多条**请求。
    var engine: EngineCore
    # 批路每条流的状态，**按槽位**索引（不是按 request 号 —— 引擎自己只有
    # `MAX_BATCH` 个槽位，`find` 是线性查找）。
    #   `b_req` — 这个槽位在给谁生成（`NO_REQUEST` = 空着）
    #   `b_out` — 生成的 token，扁平存放：槽位 s 的第 k 个在 `s * MAX_GEN + k`
    #   `b_n` / `b_steps` — 已交出几个 / 一共要几个
    #   `b_emitted` — 已经发出去的**字节**数（增量文本靠它，理由见 `_new_text`）
    #   `b_temp` — 这条请求的温度（贪心是 ≤ 0）
    #   `b_rng` — 这条请求**自己的**随机源状态（`Rng.state`）
    #
    # ⚠️ 随机源为什么必须**每条请求一份**：批路是多条流交错推进的，一个共享的随机
    # 源会让"这条流这一步抽到什么"取决于**别人**问了几步 —— 于是同一条 prompt 在
    # 批里与单跑会给出不同答案，而"批调度只是换了执行顺序"这件事就不成立了。状态
    # 存成 `UInt64` 而不是 `Rng`：平行基础类型列表，`Rng` 只在用它的那一步里现造。
    var b_req: List[Int]
    var b_out: List[Int]
    var b_n: List[Int]
    var b_steps: List[Int]
    var b_emitted: List[Int]
    var b_temp: List[Float64]
    var b_rng: List[UInt64]

    # 等待队列：槽位满了的请求**排在这里**，而不是被拒绝。
    # 它是"只 append + 一个头指针"的：条目一旦被放进批表就不再回看，`w_head` 追上
    # 队尾时整表清空（那时队列是空的）。不这么做就需要列表的删除操作，而"删中间
    # 一项"正是这类定长簿记里最容易写错的地方。
    #   `w_req` — 在给谁等（`NO_REQUEST` = 已经划掉，比如对端断了）
    #   `w_prompt` / `w_steps` / `w_temp` — 轮到它时重新 submit 要的东西
    var w_req: List[Int]
    var w_prompt: List[String]
    var w_steps: List[Int]
    var w_temp: List[Float64]
    var w_head: Int
    # 一共让多少条请求排过队。它是**给门读**的：门要能说"这里确实排过队"，否则
    # "没有被拒绝"可能只是"恰好没人超额"。
    var waited: Int

    # 再造一份时要用的三个路径（`spawn_twin`）。留着它们而不是留一份
    # `ServeConfig`：配置里有 host/port 那些和"加载一份权重"无关的东西，而这里
    # 要的只是"从哪儿加载"。
    var weights_file: String
    var config_path: String
    var tokenizer_json: String
    # 权重档（见 `ServeConfig.weight_bits`）：`spawn_twin` 要**按同一档**重新加载，
    # 否则同进程里两种精度的权重并存，答出来的 token 视走哪条路而定。
    var weight_bits: Int

    def __init__(
        out self,
        var model: QwenForward,
        var tokenizer: Tokenizer,
        vocab: Int,
        max_tokens: Int,
        max_prompt_tokens: Int,
        seed: Int,
        weights_file: String,
        config_path: String,
        tokenizer_json: String,
        weight_bits: Int,
    ) raises:
        self.model = model^
        self.tokenizer = tokenizer^
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = seed
        # 这三个是**借用**（不带 `var` 的参数在 Mojo 1.0 里是借用，不是拥有的），
        # 所以这里是拷贝而不是 `^` —— 从借用里转移会直接编译不过。
        self.weights_file = weights_file
        self.config_path = config_path
        self.tokenizer_json = tokenizer_json
        self.weight_bits = weight_bits
        self.sampler = Sampler(vocab, MAX_SEQ_TOKENS)
        self.params = SampleParams()
        self.rng = Rng(UInt64(seed))
        self.stream_ids = List[Int]()
        self.stream_out = List[Int]()
        self.stream_steps = 0
        self.stream_count = 0
        self.stream_last = 0
        self.stream_emitted = 0
        self.cut_tick = env_str("ALOFA_ET_CUT_TICK", "0") == "1"
        self.cut_decide = env_str("ALOFA_ET_CUT_DECIDE", "0") == "1"
        self.engine.cut_fwd = env_str("ALOFA_ET_CUT_FWD", "0") == "1"
        self.stream_temp = 0.0
        self.stream_started = False
        self.stream_request = NO_REQUEST
        self.batch_enabled = batch_default()
        self.batch_stat_on = getenv("ALOFA_BATCH_STATS", "") == "1"
        self.batch_ticks = 0
        self.t_step_last = 0
        self.batch_win_ticks = 0
        self.batch_win_widths = 0
        self.batch_win_max = 0
        # 批路：形状参数全部来自**这份权重**（不从配置里再抄一遍 —— 抄错了会静默
        # 算错，而从同一个 cfg 读出来的是同一份真值）。
        self.engine = EngineCore(
            SchedConfig(
                BATCH_ROWS,
                BATCH_ROWS,
                BATCH_BLOCK,
                BATCH_CAP_BLOCKS,
                BATCH_WATERMARK,
                BATCH_MAX_WAIT,
            ),
            self.model.cfg.hidden,
            self.model.cfg.intermediate,
            self.model.cfg.kv_dim(),
            self.model.cfg.n_layers,
            self.model.cfg.n_heads,
            self.model.cfg.n_kv_heads,
            self.model.cfg.head_dim,
            self.model.cfg.vocab,
            self.model.cfg.eps,
            self.max_tokens,
        )
        self.b_req = List[Int]()
        self.b_out = List[Int]()
        self.b_n = List[Int]()
        self.b_steps = List[Int]()
        self.b_emitted = List[Int]()
        self.b_temp = List[Float64]()
        self.b_rng = List[UInt64]()
        self.w_req = List[Int]()
        self.w_prompt = List[String]()
        self.w_steps = List[Int]()
        self.w_temp = List[Float64]()
        self.w_head = 0
        self.waited = 0
        for i in range(MAX_BATCH):
            self.b_req.append(NO_REQUEST)
            self.b_n.append(0)
            self.b_steps.append(0)
            self.b_emitted.append(0)
            self.b_temp.append(0.0)
            self.b_rng.append(UInt64(0))
        for i in range(MAX_BATCH * MAX_GEN):
            self.b_out.append(0)

    @staticmethod
    def load(cfg: ServeConfig) raises -> ModelService:
        # 注意 params_dir 是权重**文件**，不是目录（真实 HF 目录里参数就是
        # `model.safetensors` 这一份，目录下没有散着的分片）。
        var model = load_weights(cfg.weights_file, cfg.config_path, cfg.max_tokens, cfg.weight_bits)
        var vocab = model.cfg.vocab
        var tokenizer = load_tokenizer_json(cfg.tokenizer_json)
        return ModelService(
            model^,
            tokenizer^,
            vocab,
            cfg.max_tokens,
            cfg.max_prompt_tokens,
            cfg.seed,
            cfg.weights_file,
            cfg.config_path,
            cfg.tokenizer_json,
            cfg.weight_bits,
        )

    def spawn_twin(self, index: Int) raises -> Int:
        """engine 线程池里的"再加载一份"（roadmap 3.2c）。

        ⚠️ **一份 = 一份权重**：这里是**重新加载**（`QwenForward` 从权重文件再来一
        次），不是"共享只读权重 + 各自一份 KV"。所以 `ALOFA_ENGINE_THREADS=N` 的
        内存是 N 份权重 —— 这个代价写在 `srv/engine_thread.mojo` 的文件头，启动时
        也会跟着实验警示一起打出来，不是疏漏。共享那一层要动到 model（把"只读的
        权重"和"每条流的状态"分开），那是下一件事。

        分片数跟着原型走（多 worker 下它是 1，见 `main` 里那段 fork 的说明）。
        """
        var model = load_weights(
            self.weights_file, self.config_path, self.max_tokens, self.weight_bits
        )
        model.shards = self.model.shards
        var tokenizer = load_tokenizer_json(self.tokenizer_json)
        var twin = ModelService(
            model^,
            tokenizer^,
            self.vocab,
            self.max_tokens,
            self.max_prompt_tokens,
            self.seed,
            self.weights_file,
            self.config_path,
            self.tokenizer_json,
            self.weight_bits,
        )
        return heap_place(twin^)

    def complete(
        mut self, prompt: String, max_tokens: Int, temperature: Float64
    ) raises -> Completion:
        self.model.reset()
        var ids = self.tokenizer.encode(prompt)
        if len(ids) == 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "the prompt tokenized to nothing", prompt
            )
        if len(ids) > self.max_prompt_tokens:
            raise AlofaError(
                ERR_CAPACITY,
                "the prompt is longer than this server reads",
                "n=" + String(len(ids)),
            )

        # 模型那侧的 `max_tokens` 是**上下文窗口**（prompt + 生成的总槽位，
        # 见 `QwenForward` 的 KV 分配）—— 生成步数不能越过窗口减 prompt。
        var steps = min(max_tokens, self.max_tokens - len(ids))
        if steps < 0:
            steps = 0
        var out = List[Int]()
        var rng = Rng(UInt64(self.seed))
        # ⚠️ 采样器**复用 `self.sampler` 那一份**，不在这里现造一个：`Sampler` 的构造
        # 会按 `vocab` 建一块 arena（`vocab*16 + vocab*8 + …`，0.5B 上是 **3.65 MB**），
        # 每请求建毁一次等于每请求一次几 MB 的申请+释放。`build` 每一步都按 logits 全量
        # 重建分布，不留上一步的东西 —— 批路本来就是这么用 `self.sampler` 的。
        var n_vocab = self.vocab
        var params = SampleParams()
        params.temperature = temperature

        var logits = self.model.prefill[BACKEND_AVX2](ids)
        for t in range(steps):
            var next_id = 0
            if temperature <= 0.0:
                next_id = self.model.argmax(logits)
            else:
                self.sampler.build(logits, n_vocab, params, List[LogitBias](), out)
                next_id = self.sampler.pick(n_vocab, rng.next_uniform())
            out.append(next_id)
            if t + 1 < steps:
                logits = self.model.step[BACKEND_AVX2](next_id)
        var text = self.tokenizer.decode(out)
        return Completion(text, len(ids), len(out))

    def stream_begin(
        mut self,
        prompt: String,
        max_tokens: Int,
        temperature: Float64,
        request: Int,
    ) raises:
        """开始一次流式生成：**只** reset、分词与校验，不走前向。

        前向留在第一次 `stream_next`（惰性），因为这里是唯一还能把请求的错回成
        400 的地方 —— 响应头一旦发出去，退路就只剩下"给流一个终点"。
        """
        # 上一条流没被收尾（对端断了 → 不一定有人通知）就先收掉它：这一整块状态
        # 是**一整份**，不收就只是被下面覆盖掉（今天看不出来），接了批调度之后占
        # 的是表里一个位置。
        if self.stream_request != NO_REQUEST and self.stream_request != request:
            self.stream_request = NO_REQUEST
        var ids = self.tokenizer.encode(prompt)
        if len(ids) == 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "the prompt tokenized to nothing", prompt
            )
        if len(ids) > self.max_prompt_tokens:
            raise AlofaError(
                ERR_CAPACITY,
                "the prompt is longer than this server reads",
                "n=" + String(len(ids)),
            )
        # 与非流式同一条规则：模型那侧的 `max_tokens` 是上下文窗口，生成步数不
        # 能越过窗口减 prompt。
        var steps = min(max_tokens, self.max_tokens - len(ids))
        if steps < 0:
            steps = 0

        # ---- 批路：贪心**与采样**都收，只要形状装得进引擎 ----
        # 三个上限都是引擎自己的（`submit` 会指名拒绝越界的，所以这里是**先**判
        # 断再决定走哪条路，而不是碰运气）：prompt ≤ `MAX_PROMPT`、新 token ≤
        # `MAX_GEN`、同时在批 ≤ `MAX_BATCH`。越界的请求不是错，只是走老路。
        #
        # 采样也收，是因为"选哪个 token"这一步已经不在引擎里了（`engine.decide`
        # 只跑前向，选谁由这边按槽位答）—— 每条请求带自己的温度与自己的随机源，
        # 所以批里的采样与单跑的采样抽的是**同一个**序列。
        var slot = self._free_slot()
        # 批路的形状上限还有**第四条**：一条序列的**总长**。
        #
        # `MAX_GEN`（32）和 `MAX_SEQ_TOKENS`（64，`runtime/kv` 的序列上限，引擎与
        # KV 房间用的是同一个常数）不是一回事 —— 前者管"生成多少"，后者管"prompt
        # + 生成加起来多少"。少了这一条，`prompt 34`（chat 模板带 system 段）`+ 32`
        # 会进批路，而引擎在 `hist` 撞到 64（= 4 个 16-token 块）时扩不出第 5 块：
        # 客户端看到的是流走到一半断掉，日志里是
        # `capacity: the request has fewer blocks than it has history`
        # （账本第十五条，实测 108/122 条这么死）。
        #
        # 装不下的按上下文窗口的规矩**截短**（客户端的 `finish_reason` 会说
        # `length`），那与 `steps = min(max_tokens, 窗口 - prompt)` 是同一条语义；
        # 静默塞进去则是"服务自己把自己毒死"。
        var batch_steps = min(steps, MAX_SEQ_TOKENS - len(ids))
        if (
            self.batch_enabled
            and steps > 0
            and len(ids) <= MAX_PROMPT
            and steps <= MAX_GEN
            and batch_steps > 0
        ):
            if slot >= 0:
                self._submit(slot, request, ids, batch_steps, temperature)
                return
            # 槽位满了 → **排队**而不是拒绝（为什么是排队、以及它为什么必须有上
            # 界，写在 `_enqueue` 上）。队列也满了才落到老路去 —— 那里会指名拒绝
            # （`capacity`），那是有上界的拒绝，不是无上界的排队。
            if self._enqueue(request, prompt, batch_steps, temperature):
                return

        # ---- 老路（采样，或装不进引擎的请求）：一次只握一条 ----
        # 第二条采样流**指名拒绝**而不是悄悄串行：串行的话客户端看到的是"偶发地
        # 特别慢"，而那是最难归因的一种慢。
        if self.stream_request != NO_REQUEST and self.stream_request != request:
            raise AlofaError(
                ERR_CAPACITY,
                "this service runs one sampling stream at a time",
                "streaming=" + String(self.stream_request),
            )
        self.model.reset()
        self.stream_ids = ids^
        self.stream_steps = steps
        self.stream_count = 0
        self.stream_last = 0
        self.stream_emitted = 0
        self.stream_temp = temperature
        self.stream_started = False
        self.params.temperature = temperature
        self.rng = Rng(UInt64(self.seed))
        self.stream_out.clear()
        self.stream_request = request

    def stream_next(mut self, request: Int) raises -> StreamToken:
        """走一步，返回这一步**新增**的文本。

        第一次调用才 prefill（`stream_started`），之后每次 `step` 上一步的
        token —— 与非流式那条路走的前向次数**完全相同**，所以流不改变数值结果。
        """
        # 有空位就先把排队的放进来：槽位是在别人走完的那一刻空出来的，而"轮到谁"
        # 不该由调用方记着 —— 否则排在第 9 位的那条要等有人恰好问它，才知道自己
        # 已经能进了。
        self._admit()
        if self._waiting_of(request) >= 0:
            # 还在等：它排着队，可引擎里没有它的位置。这一帧**必须**说"在等"而不
            # 是"结束" —— 后者会让客户端拿到一个连 `[DONE]` 都没有的空连接。
            return StreamToken("", False, True)
        var slot = self._slot_of(request)
        if slot >= 0:
            var t_deliv = self.model.prof.now()
            var tok = self._batch_next(slot, request)
            _ = self.model.prof.tick(K_DELIV, t_deliv)
            return tok^
        if self.stream_request != request:
            # 号对不上 = 有人拿另一条流的号来问这一步。静默答下去就是"接着给别人
            # 生成"，而那一边的客户端看到的是一条完全正常的流 —— 只是内容属于别人。
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "this service is already streaming another request",
                "asked="
                + String(request)
                + " streaming="
                + String(self.stream_request),
            )
        if self.stream_count >= self.stream_steps:
            # 已经交完了还被问：仍然回答"结束"。路由可能多问一次（它才知道要了
            # 多少），重新开始或报错都会让这条流失去终点。
            return StreamToken("", True, False)
        var next_id = 0
        if not self.stream_started:
            var logits = self.model.prefill[BACKEND_AVX2](self.stream_ids)
            next_id = self._pick(logits)
            self.stream_started = True
        else:
            var logits = self.model.step[BACKEND_AVX2](self.stream_last)
            next_id = self._pick(logits)
        self.stream_out.append(next_id)
        self.stream_last = next_id
        self.stream_count += 1
        return StreamToken(self._new_text(), False, False)

    def stream_end(mut self, request: Int) raises:
        """把 `request` 占的东西还回来（**幂等**：号对不上就什么也不做）。

        批路要还的是引擎里那个**槽位**：`cancel` 只是把取消排进下一拍，所以这里
        必须再跑一拍让它落地 —— 少了那一拍，槽位一直占着，第 `MAX_BATCH` + 1 条
        请求会收到 "the engine is full"，而真正的原因（某条连接断了）在别处。
        """
        var slot = self._slot_of(request)
        if slot >= 0:
            if self.engine.find(request) >= 0:
                if not self.engine.is_done(request):
                    self.engine.cancel(request)
                    # ⚠️ 这一拍也必须是 `_batch_tick`（按每条请求**自己的**策略选），
                    # 不能是 `engine.tick`：那个是写死贪心的，而这一拍发生在**别人**
                    # 断开的时候 —— 用它就会给批里正在采样的请求塞一个 argmax 的
                    # token。那不是"算错一点"，是那条流的答案从此分叉，而日志里什么
                    # 都没有。
                    self._batch_tick()
                # ⚠️ 槽位要**显式**还回去：引擎那边"走完"只是 `ST_DONE`，它自己不
                # 回收（槽位是输出数组的下标，什么时候能复用只有调用方知道）。少了
                # 这一步，这台服务一辈子只能服务 `MAX_BATCH` 条请求 —— 第 9 条无论
                # 什么时候来，收到的都是 "the engine is full"。
                self.engine.release(request)
            self.b_req[slot] = NO_REQUEST
            self.b_n[slot] = 0
            self.b_steps[slot] = 0
            self.b_emitted[slot] = 0
            # 让出一个槽位之后**立刻**把排队的接进来：等待的那条流正在等的正是这
            # 一刻，放到下一次 `stream_next` 才接就等于让它多等一轮轮询。
            self._admit()
            return
        # 它也可能还在**队列**里（还没等到槽位，对端就断了）：划掉它，不然它会占
        # 着一个位置，直到被 admit 到一条已经不存在的连接上。
        if self._cancel_wait(request):
            self._admit()
            return
        if self.stream_request != request:
            return
        self.stream_request = NO_REQUEST

    def _submit(
        mut self,
        slot: Int,
        request: Int,
        imm ids: List[Int],
        steps: Int,
        temperature: Float64,
    ) raises:
        """把一条请求放进批表的 `slot`（编码已经做好，这里只 submit + 记状态）。"""
        var arena = Arena(MAX_PROMPT * 8 + 64)
        var toks = int_map(arena.alloc(MAX_PROMPT * 8))
        for i in range(len(ids)):
            toks[unsafe_offset=i] = ids[i]
        self.engine.submit(request, toks, len(ids), steps)
        # ⚠️ `Arena` 在**指针最后一次使用处**就析构（这是它的设计），所以这行不
        # 写，`toks` 在 `submit` 内部读到的是已经释放的内存 —— 实测表现是
        # `stream_begin` 段错误，而不是"算错了"。
        arena.keep_alive()
        self.b_req[slot] = request
        self.b_steps[slot] = steps
        self.b_n[slot] = 0
        self.b_emitted[slot] = 0
        self.b_temp[slot] = temperature
        # 每条请求从**同一个种子**起 —— 与老路 `self.rng = Rng(seed)` 同一个
        # 起点，否则"批里 == 单跑"这条判据从第一步起就不可能成立。
        self.b_rng[slot] = UInt64(self.seed)

    def _enqueue(
        mut self, request: Int, prompt: String, steps: Int, temperature: Float64
    ) -> Bool:
        """排到队尾。返回 False = 队列也满了（那时调用方才真的拒绝）。

        为什么是**排队**而不是拒绝：引擎的槽位数是形状决定的（`MAX_BATCH` = 8），
        而放宽它不兑换吞吐（实测 `rows` 8 / 16 / 32 的每行耗时重合 —— 见账本），
        所以"第 9 条并发"本来就不该靠加槽位解决，它该等。

        ⚠️ 排队必须**有上界**，否则"排队"就是把 OOM 推迟到半夜：一条连接最多一条
        在途流，所以队长的上界就是连接数 —— 这不是估的，是数的。
        """
        if len(self.w_req) - self.w_head >= MAX_CONNS:
            return False
        self.w_req.append(request)
        self.w_prompt.append(prompt)
        self.w_steps.append(steps)
        self.w_temp.append(temperature)
        self.waited += 1
        return True

    def _waiting_of(self, request: Int) -> Int:
        """`request` 在等待队列里的下标（-1 = 它没在等）。"""
        for i in range(self.w_head, len(self.w_req)):
            if self.w_req[i] == request:
                return i
        return -1

    def _cancel_wait(mut self, request: Int) -> Bool:
        """把还在排队的 `request` 划掉（对端断了）。划掉而不是删除：这条队列只
        append，`_admit` 会跳过划掉的条目。"""
        var i = self._waiting_of(request)
        if i < 0:
            return False
        self.w_req[i] = NO_REQUEST
        return True

    def _admit(mut self) raises:
        """有几个空位就放几条进来，队首优先（先来的先服务）。"""
        while self.w_head < len(self.w_req):
            if self.w_req[self.w_head] == NO_REQUEST:
                self.w_head += 1
                continue
            var slot = self._free_slot()
            if slot < 0:
                break
            var request = self.w_req[self.w_head]
            # 轮到它才编码：队列里存的是 prompt。`ids` 只在 submit 那一刻有用，
            # 提前编好就得把一批不定长的表一直挂在这里。
            var ids = self.tokenizer.encode(self.w_prompt[self.w_head])
            # ⚠️ **批表有空位 ≠ 引擎有空位**：一条流走完时，批表这边立刻空了，而
            # 引擎那个槽位要到**下一拍**才是 `ST_FREE`。所以这里必须接住"满了"：
            # 队首**不动**，下一轮再来 —— 这本来就是排队该有的样子（"轮到我时再
            # 试"），而不是"我保证现在一定进得去"。
            try:
                self._submit(
                    slot,
                    request,
                    ids,
                    self.w_steps[self.w_head],
                    self.w_temp[self.w_head],
                )
            except err:
                # 只吞"满了"这一件事：别的错（形状、重复 id）吞了就是把它变成
                # 一场永远等不到的排队。
                if String(err).find("capacity") < 0:
                    raise err
                # ⚠️ 空转一拍，否则这个队永远排不到头：引擎把槽位从"走完"收回
                # 是**在一拍里**做的，而所有活跃请求都走完之后就没人再拍了 —— 于
                # 是槽位一直停在"走完"，下一轮问还是"满了"。这一拍与 `stream_end`
                # 里那一拍是同一个理由（引擎的状态机只在一拍里前进）。
                self._batch_tick()
                break
            self.w_head += 1
        if self.w_head >= len(self.w_req):
            # 队尾已经追平：整表清空、头指针归零。不清的话这个只 append 的列表会
            # 一直涨，涨到上界就再也放不进新的 —— 而它其实早就空了。
            self.w_req.clear()
            self.w_prompt.clear()
            self.w_steps.clear()
            self.w_temp.clear()
            self.w_head = 0

    def _free_slot(self) -> Int:
        """批表里第一个空位（-1 = 满了）。满了不是错 —— 请求会排队。"""
        for i in range(MAX_BATCH):
            if self.b_req[i] == NO_REQUEST:
                return i
        return -1

    def _slot_of(self, request: Int) -> Int:
        """`request` 在批表里的槽位（-1 = 它不在批路上）。"""
        for i in range(MAX_BATCH):
            if self.b_req[i] == request:
                return i
        return -1

    def _batch_tick(mut self) raises -> Int:
        """批路走一拍：**前向交给引擎，选谁由这边决定**。

        为什么不直接调 `engine.tick`：那个 tick 里的 `argmax` 是**写死**的。采样
        要的两样东西引擎都不该持有 —— 每条请求自己的随机源（`b_rng`）与自己的历
        史（`b_out`，采样器拿它做重复惩罚）—— 所以这一拍拆成三步：`decide`（排
        一拍 + 跑前向）→ 逐个槽位问「谁欠一个 token、它的 logits 在哪」→ `settle`
        （落地）。贪心那条路走的是**同一个** `decide` / `settle`，只是决策是一行
        `argmax`，所以两条路的"一拍"不可能走偏。
        """
        if self.cut_decide:
            var chosen0 = self.engine.step_choices()
            for i in range(MAX_BATCH):
                chosen0[unsafe_offset=i] = NO_TOKEN
            self.engine.settle(chosen0)
            return 0
        var rows = self.engine.decide[BACKEND_AVX2](self.model)
        var chosen = self.engine.step_choices()
        for i in range(MAX_BATCH):
            chosen[unsafe_offset=i] = NO_TOKEN
        var t_pick = self.model.prof.now()
        try:
            if rows > 0:
                for i in range(MAX_BATCH):
                    if not self.engine.step_owes(i):
                        continue
                    var req = self.engine.step_request(i)
                    var slot = self._slot_of(req)
                    if slot < 0:
                        # 引擎在替一条这边不认得的请求跑。留 `NO_TOKEN` 会让它原地不动
                        # —— 客户端看到的是"偶发地特别慢"，最难归因的那一种，所以宁可红。
                        raise AlofaError(
                            ERR_INVALID_ARGUMENT,
                            "the engine is generating for a request this service does"
                            + " not hold",
                            "request=" + String(req),
                        )
                    chosen[unsafe_offset=i] = self._batch_pick(
                        slot, self.engine.step_logits(i)
                    )
        except err:
            # 决策中途抛：这一拍**也必须落地**（`chosen` 全是 `NO_TOKEN` = 一个
            # token 都不算）。少了它，`decide` 借走的缓冲没人还，执行器从此进不了
            # 下一次 `plan` —— 症状是"这条流断了"之后**每一条**请求都秒失败，而
            # 归因落在几千行之后（账本第十五条）。
            try:
                self.engine.settle(chosen)
            except err2:
                print(
                    "  [srv] the step could not be settled after a failure: "
                    + String(err2)
                )
            raise err.copy()
        _ = self.model.prof.tick(K_PICK, t_pick)
        var t_settle = self.model.prof.now()
        self.engine.settle(chosen)
        _ = self.model.prof.tick(K_SETTLE, t_settle)
        comptime if ALOFA_PROFILE:
            # 服务端步墙钟：相邻两次 forward 完成的间隔。客户端（Python 收流）
            # 的开销不进这个数 —— 它没资格算进服务端的账。
            var t_now = self.model.prof.now()
            if self.t_step_last != 0:
                _ = self.model.prof.tick(K_WALL, self.t_step_last)
            self.t_step_last = t_now
        if self.batch_stat_on:
            self.batch_ticks += 1
            self.batch_win_ticks += 1
            self.batch_win_widths += rows
            if rows > self.batch_win_max:
                self.batch_win_max = rows
            if self.batch_win_ticks >= 64:
                print(
                    "  [srv] batch: ticks="
                    + String(self.batch_ticks)
                    + " window_ticks="
                    + String(self.batch_win_ticks)
                    + " window_width_sum="
                    + String(self.batch_win_widths)
                    + " window_width_max="
                    + String(self.batch_win_max)
                    + " running="
                    + String(self.engine.running_count())
                    + " waiting="
                    + String(self.engine.waiting_count())
                    + " queued="
                    + String(len(self.w_req))
                    + " blocks="
                    + String(self.engine.blocks_used())
                    + "+"
                    + String(self.engine.blocks_wanted())
                    + "/"
                    + String(self.engine.capacity_blocks())
                    + " cached="
                    + String(self.engine.cached_blocks())

                    + " preempts="
                    + String(self.engine.preempt_total())

                )
                comptime if ALOFA_PROFILE:
                    self.model.dump_profile()
                self.batch_win_ticks = 0
                self.batch_win_widths = 0
                self.batch_win_max = 0
        return rows

    def _batch_pick(mut self, slot: Int, logits: F32Ptr) raises AlofaError -> Int:
        """批路给 `slot` 选这一步的 token：贪心，或按**它的**温度采样。

        与老路 `_pick` 是同一个分支、同一个 `sampler`、同一份参数，唯一的结构差异
        是**随机源**：老路一份 `self.rng`（一次一条流，无所谓），批路每条请求一份
        （`b_rng`）—— 多条流交错推进时，一个共享的随机源会让"这条流这一步抽到什
        么"取决于**别人**问了几步，于是同一条 prompt 在批里与单跑会给出不同答案。

        ⚠️ 历史是 `b_out` 的前 `b_n` 个，与老路的 `stream_out` 对应：`sampler.build`
        拿它做重复惩罚。少了它，批里的采样会与单跑分叉 —— 而且只在生成出重复词的
        那一段才看得出来，是最容易被"跑一遍看着没问题"放过去的一种错。
        """
        if self.b_temp[slot] <= 0.0:
            return self.model.argmax(logits)
        var hist = List[Int]()
        for j in range(self.b_n[slot]):
            hist.append(self.b_out[slot * MAX_GEN + j])
        self.params.temperature = self.b_temp[slot]
        self.sampler.build(logits, self.vocab, self.params, List[LogitBias](), hist)
        var rng = Rng(self.b_rng[slot])
        var u = rng.next_uniform()
        self.b_rng[slot] = rng.state
        return self.sampler.pick(self.vocab, u)

    def _batch_next(mut self, slot: Int, request: Int) raises -> StreamToken:
        """批路走一步：引擎还没交出第 `n` 个 token 就推它，直到交出或没活干。

        一次 `tick` 推进**所有**在批里的请求 —— 这正是批调度的收益所在：八条流的
        第 k 个 token 是同一次前向算出来的。
        """
        # ⚠️ `want` 是**引擎交出的 token 数**，不是"我调了几次"。早先这里记的是
        # 调用次数（`b_n = want + 1`），与引擎的 `n_out` 一旦错开 —— 一拍里引擎
        # 可能已经推进了两步、或者这一拍纯粹是"回炉重填" —— 就会出现"缓冲里有
        # token、`_batch_text` 却交空串"（文本在上一帧已经发完了）。而空串在
        # `step_stream` 那里是**流结束**的暗号：客户端于是拿到一条正文被截短、
        # 却带着 `finish_reason` 与 `[DONE]` 的"正常"流 —— 实测 8 并发同 prompt
        # 下 7/8 条短一截，温 0 也不一致（丢的是尾巴，不是算错）。
        if self.cut_tick:
            # 二分刀（见 cut_tick 字段注释）：活⇒写者在 _batch_tick 内。
            return StreamToken("", True, False)
        var want = self.b_n[slot]
        var spins = 0
        while self.engine.n_output(request) <= want:
            if not self.engine.has_work():
                break
            self._batch_tick()
            spins += 1
            if spins >= MAX_PROGRESS_TICKS:
                # 指名拒绝而不是继续等：等到天荒地老在客户端就是"永远没有
                # `[DONE]`"，而那是最难归因的一种失败。报出来至少知道卡在哪条
                # 请求、要的是第几个 token。
                raise AlofaError(
                    ERR_STALLED,
                    "the engine stopped making progress on this request",
                    "req="
                    + String(request)
                    + " want="
                    + String(want)
                    + " ticks="
                    + String(spins),
                )
        if self.engine.n_output(request) <= want:
            # 引擎交不出更多了：这条流到终点了（也可能被抢占后没再排上 —— 无论哪
            # 一种，"给一个终点"都比"一直等"好：后者在客户端是永远等不到 `[DONE]`）。
            if self.batch_stat_on:
                # 为什么"交不出更多"是个必须说清的事：它决定了客户端看到的是
                # "答完了"还是"答到一半"。`n_out` 掉回 `want` 以下只有两种来路 ——
                # 引擎按 `max_new` 收工（正常），或请求被抢占/整槽回炉（错）。
                print(
                    "  [srv] batch end: req="
                    + String(request)
                    + " want="
                    + String(want)
                    + " n_out="
                    + String(self.engine.n_output(request))
                    + " in_slot="
                    + String(self.engine.find(request))
                    + " done="
                    + String(self.engine.is_done(request))
                    + " preempts="
                    + String(self.engine.preempt_total())
                    + " running="
                    + String(self.engine.running_count())
                    + " waiting="
                    + String(self.engine.waiting_count())
                )
            return StreamToken("", True, False)
        var arena = Arena(MAX_GEN * 8 + 64)
        var dest = int_map(arena.alloc(MAX_GEN * 8))
        var n = self.engine.output(request, dest)
        for i in range(n):
            self.b_out[slot * MAX_GEN + i] = dest[unsafe_offset=i]
        arena.keep_alive()  # 同上：指针用完之前 arena 不许析构
        var t_detok = self.model.prof.now()
        var text = self._batch_text(slot, n)
        _ = self.model.prof.tick(K_DETOK, t_detok)
        # 口径跟引擎对齐：`b_n` 是**已经交出的 token 数**（`_batch_pick` 拿它当
        # 历史的长度，也拿它判断"到 max_new 了没有"）。
        self.b_n[slot] = n
        if text.byte_length() == 0 and n < self.b_steps[slot]:
            # 交了 token 但这段没有新文本（比如 tokens 都落在上一帧的同一个
            # 多字节字符里）。**这不是终点** —— 再要一步，绝不把空串交出去：
            # 空串在 `step_stream` 那里就是"流结束"。
            want = n
            if not self.engine.has_work():
                return StreamToken("", True, False)
            return self._batch_next(slot, request)
        return StreamToken(text, n >= self.b_steps[slot], False)

    def _batch_text(mut self, slot: Int, n: Int) raises -> String:
        """批路这一步新增的文本 —— 与 `_new_text` **同一个**算法（整段前缀解码减去
        已经发出去的字节），只是源头是批表那一段。

        两边必须同一个算法：不一样的话，流式拼出来的文本会**不等于**非流式那份，
        而"多字节字符被切成两半"正是它要防的。
        """
        var ids = List[Int]()
        for i in range(n):
            ids.append(self.b_out[slot * MAX_GEN + i])
        var full = bytes_of_text(self.tokenizer.decode(ids))
        var sent = self.b_emitted[slot]
        if sent > len(full):
            # 不该发生（前缀只会变长）。真发生了就从头对齐，而不是带着一个错的
            # 偏移量继续 —— 那个偏移会让后面每一帧都错。
            sent = 0
        self.b_emitted[slot] = len(full)
        return bytes_to_text(full, sent, len(full))

    def _pick(mut self, logits: F32Ptr) raises -> Int:
        """从一行 logits 里取一个 id：贪心或按温度采样（与非流式同一个分支）。"""
        if self.stream_temp <= 0.0:
            return self.model.argmax(logits)
        self.sampler.build(
            logits, self.vocab, self.params, List[LogitBias](), self.stream_out
        )
        return self.sampler.pick(self.vocab, self.rng.next_uniform())

    def _new_text(mut self) raises -> String:
        """这一步新增的文本 = 整段前缀解码的结果减去已经发出去的那些字节。

        为什么不"把这个 token 单独解码"：多字节字符可能跨两个 token，单独解码
        时前半截是不完整的 UTF-8，会被 `decode` 丢掉 —— 那样流式拼出来的文本
        会**不等于**非流式那份。整段前缀解码的代价是每步重解一遍（步数 ≤ 256，
        几十 KB），换来的是两边逐字节一致。
        """
        var full = bytes_of_text(self.tokenizer.decode(self.stream_out))
        var sent = self.stream_emitted
        if sent > len(full):
            # 不该发生（前缀只会变长）。真发生了就从头对齐，而不是带着一个错的
            # 偏移量继续 —— 那个偏移会让后面每一帧都错。
            sent = 0
        self.stream_emitted = len(full)
        return bytes_to_text(full, sent, len(full))


def main() raises:
    # engine 线程是 pthread 默认栈（= soft RLIMIT_STACK，本机 8 MiB）；批前向的
    # 调用链在 8 MiB 上栈溢出（2026-11-04 钉死，账本三十五）。必须在任何线程
    # spawn 之前抬 —— 它只影响之后创建的线程。`ALOFA_STACK_MB=0` 关掉。
    var stack_mb = env_int("ALOFA_STACK_MB", 64, 0, 1024)
    if stack_mb > 0:
        var got = raise_stack_limit(stack_mb * 1024 * 1024)
        print(
            "  [srv] stack limit: "
            + String(got // (1024 * 1024))
            + " MiB (engine threads spawn after this)"
        )
    var cfg = ServeConfig.from_env()
    if cfg.workers > 1:
        print(
            "alofa serve — " + String(cfg.workers) + " workers, HTTP with SSE"
            + " streaming, a reactor loop per worker"
        )
    else:
        print(
            "alofa serve — single process, HTTP with SSE streaming, a reactor"
            + " loop with many connections"
        )
    var service = ModelService.load(cfg)
    if getenv("ALOFA_PROFILE", "") == "1":
        service.model.set_profile(True)
        print("  [srv] profiling on（每 64 拍随统计行 dump）")
    if cfg.shards > 0:
        # 显式给了分片数就尊重它（含单进程模式）。
        service.model.shards = cfg.shards
    elif cfg.workers > 1:
        # fork 之后 asyncrt 不可用：父进程加载模型时 `default_shards()` 碰过
        # 并发运行时，子进程继承的线程池状态是坏的 —— 实测 shards=2/8 都会让
        # 第一个前向**死等**（不是慢），直到被 master SIGKILL。所以多 worker
        # 默认每 worker 单分片；要更多分片 = `ALOFA_SHARDS`（后果自负）。
        # 单进程不受影响（没有 fork）。
        service.model.shards = 1
    print("  model loaded: " + cfg.model_name + " (vocab " + String(service.vocab) + ")")
    if cfg.engine_threads >= 1:
        # ⚠️ 这是**警示**不是装饰：engine 线程路已知会在"真模型 + 客户端中途断开"
        # 时把请求搞坏（`Job` 的缓冲被别人 free —— 那次排查的硬事实在账本 §8），所以
        # 默认不开（`ALOFA_ENGINE_THREADS=0` = 生成就在循环上）。把它打开的人必须
        # **当场**看见，而不是等它把请求搞坏之后才知道。
        var extra = ""
        if cfg.engine_threads > 1:
            # 多一条就多**一份权重**（`spawn_twin` 是重新加载一份），而"为什么内存翻
            # 了几倍"在 ps 里看不出来 —— 它只显示一个进程。
            extra = (
                "; each one reloads the weights: "
                + String(cfg.engine_threads)
                + " copies of them"
            )
        print(
            "  [warn] engine threads: "
            + String(cfg.engine_threads)
            + " (EXPERIMENTAL — a client leaving mid-generation is known to"
            + " corrupt the heap; the supported path is ALOFA_ENGINE_THREADS=0)"
            + extra
        )

    var addr = SocketAddr(IpAddr.parse(cfg.host), cfg.port)
    var handler = ChatHandler(service^, cfg.model_name)

    if cfg.workers > 1:
        # master 不监听；每个 worker 自己 bind(SO_REUSEPORT)，内核分发连接。
        # 进程编排的细节（停止通知、唤醒连接、fail-fast）见 srv/master.mojo。
        print("  listening on " + cfg.host + ":" + String(Int(cfg.port)))
        var report = run_workers(
            handler,
            addr,
            cfg.port,
            cfg.workers,
            cfg.max_requests,
            cfg.run_seconds,
            cfg.grace_ms,
            cfg.engine_threads,
        )
        if report.abnormal > 0 or report.forced > 0:
            raise AlofaError(
                ERR_CAPACITY,
                "the worker fleet did not shut down cleanly",
                "abnormal=" + String(report.abnormal) + " forced=" + String(
                    report.forced
                ),
            )
    else:
        # 一条 reactor 循环（`srv/loop.mojo`）：「一条 worker 同时只处理一条连接」
        # 这个限制到此为止 —— 慢客户端、流水线、keep-alive 的空闲连接都不再互相挡
        # 路；生成仍然是一条一条来（见 loop 文件头）。监听只在单进程这条路上开：
        # 多 worker 时 master 不监听，每个 worker fork 之后自己 bind。
        # 两条路都在，选哪条看 `cfg.engine_threads`：
        #
        # - `0`（**默认**）= `Loop.run`：生成就在这条循环上，不过线程、不碰邮箱。
        # - `>=1` = `Loop.run_threaded`：生成在 `engines` 条 engine 线程上
        #   （`srv/engine_thread.mojo`），这条循环上只剩 I/O，一次前向不再把别的连接的
        #   读写一起按住（roadmap 3.2b/3.2c）。
        #
        # ⚠️ **为什么默认不是 1**：实测（2026-09-25）线程那条路会把请求搞坏 —— 同一份
        # 二进制上 8 条几乎每条挂到超时，而走 `run` 是 8/8 全过、~1 s、零报错。性质还没
        # 修好，而"能用的服务"与"并发生成"是两件事，不该让前者等后者。`run` 保留着另一
        # 半用途：它是那条性质唯一的负向对照（`pixi run test-engine` 两种都跑）。
        var loop = Loop.bind_addr(addr)
        print("  listening on " + cfg.host + ":" + String(Int(cfg.port)))
        var served = 0
        if cfg.engine_threads == 0:
            served = loop.run(handler, cfg.max_requests, Int32(-1))
        else:
            served = loop.run_threaded(
                handler, cfg.max_requests, Int32(-1), cfg.engine_threads
            )
        print("  served " + String(served) + " requests")
        loop.close()
