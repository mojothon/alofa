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
进程数、修不了单 worker 的串行）。生成**不在**这条循环上（`srv/engine_thread.mojo`，
roadmap 3.2b），循环只做 I/O；同时能生成几条 = `ALOFA_ENGINE_THREADS`（默认 1 —— 每
多一条就多一份权重，那是 3.2c 明文写下的代价）。兼容
面是 OpenAI 的 `/v1/chat/completions`：不改代码、只改 `base_url` 就能用官方 SDK
打到它。
"""

from std.collections import List
from std.os import getenv

from flare.net import IpAddr, SocketAddr

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    AlofaError,
)
from alofa.core.rng import Rng
from alofa.core.tensor import F32Ptr
from alofa.model.arch.qwen import BACKEND_AVX2, QwenForward
from alofa.runtime.kv import MAX_SEQ_TOKENS
from alofa.runtime.sampler import LogitBias, SampleParams, Sampler
from alofa.srv.config import ServeConfig
from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import bytes_of_text, bytes_to_text
from alofa.srv.loop import MAX_CONNS, Loop
from alofa.srv.master import run_workers
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
        self.sampler = Sampler(vocab, MAX_SEQ_TOKENS)
        self.params = SampleParams()
        self.rng = Rng(UInt64(seed))
        self.stream_ids = List[Int]()
        self.stream_out = List[Int]()
        self.stream_steps = 0
        self.stream_count = 0
        self.stream_last = 0
        self.stream_emitted = 0
        self.stream_temp = 0.0
        self.stream_started = False
        self.stream_request = NO_REQUEST
        self.batch_enabled = batch_default()
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
        var model = QwenForward(cfg.weights_file, cfg.config_path, cfg.max_tokens)
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
        )

    def spawn_twin(self, index: Int) raises -> Int:
        """engine 线程池里的"再加载一份"（roadmap 3.2c）。

        ⚠️ **一份 = 一份权重**：这里是**重新加载**（`QwenForward` 从权重文件再来一
        次），不是"共享只读权重 + 各自一份 KV"。所以 `ALOFA_ENGINE_THREADS=N` 的
        内存是 N 份权重 —— 这个代价写在 `srv/engine_thread.mojo` 的文件头与
        `config.mojo` 的启动打印里，不是疏漏。共享那一层要动到 model（把"只读的
        权重"和"每条流的状态"分开），那是下一件事。

        分片数跟着原型走（多 worker 下它是 1，见 `main` 里那段 fork 的说明）。
        """
        var model = QwenForward(self.weights_file, self.config_path, self.max_tokens)
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
        var sampler = Sampler(self.vocab, MAX_SEQ_TOKENS)
        var params = SampleParams()
        params.temperature = temperature

        var logits = self.model.prefill[BACKEND_AVX2](ids)
        for t in range(steps):
            var next_id = 0
            if temperature <= 0.0:
                next_id = self.model.argmax(logits)
            else:
                sampler.build(
                    logits, self.vocab, params, List[LogitBias](), out
                )
                next_id = sampler.pick(self.vocab, rng.next_uniform())
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
        if (
            self.batch_enabled
            and steps > 0
            and len(ids) <= MAX_PROMPT
            and steps <= MAX_GEN
        ):
            if slot >= 0:
                self._submit(slot, request, ids, steps, temperature)
                return
            # 槽位满了 → **排队**而不是拒绝（为什么是排队、以及它为什么必须有上
            # 界，写在 `_enqueue` 上）。队列也满了才落到老路去 —— 那里会指名拒绝
            # （`capacity`），那是有上界的拒绝，不是无上界的排队。
            if self._enqueue(request, prompt, steps, temperature):
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
            return self._batch_next(slot, request)
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
        var rows = self.engine.decide[BACKEND_AVX2](self.model)
        var chosen = self.engine.step_choices()
        for i in range(MAX_BATCH):
            chosen[unsafe_offset=i] = NO_TOKEN
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
        self.engine.settle(chosen)
        return rows

    def _batch_pick(mut self, slot: Int, logits: F32Ptr) raises -> Int:
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
        var want = self.b_n[slot]
        while self.engine.n_output(request) <= want:
            if not self.engine.has_work():
                break
            self._batch_tick()
        if self.engine.n_output(request) <= want:
            # 引擎交不出更多了：这条流到终点了（也可能被抢占后没再排上 —— 无论哪
            # 一种，"给一个终点"都比"一直等"好：后者在客户端是永远等不到 `[DONE]`）。
            return StreamToken("", True, False)
        var arena = Arena(MAX_GEN * 8 + 64)
        var dest = int_map(arena.alloc(MAX_GEN * 8))
        var n = self.engine.output(request, dest)
        for i in range(n):
            self.b_out[slot * MAX_GEN + i] = dest[unsafe_offset=i]
        arena.keep_alive()  # 同上：指针用完之前 arena 不许析构
        var text = self._batch_text(slot, n)
        self.b_n[slot] = want + 1
        return StreamToken(text, self.b_n[slot] >= self.b_steps[slot], False)

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
    if cfg.engine_threads > 1:
        # 这条打印不是装饰：`engines` 条的代价是 **N 份权重**（`spawn_twin` 是重新
        # 加载一份），而"为什么内存翻了几倍"在 ps 里看不出来 —— 它只显示一个进程。
        print(
            "  engine threads: "
            + String(cfg.engine_threads)
            + " per worker (each one reloads the weights: "
            + String(cfg.engine_threads)
            + " copies of them)"
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
        # 生成在**另一条线程**上（`srv/engine_thread.mojo`）：这条循环上只剩 I/O，
        # 一次前向不再把别的连接的读写一起按住。`run`（生成就在这条循环上）保留着
        # —— 它是这条性质唯一的负向对照（`pixi run test-engine` 两种都跑）。
        var loop = Loop.bind_addr(addr)
        print("  listening on " + cfg.host + ":" + String(Int(cfg.port)))
        var served = loop.run_threaded(
            handler, cfg.max_requests, Int32(-1), cfg.engine_threads
        )
        print("  served " + String(served) + " requests")
        loop.close()
