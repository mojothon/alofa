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
from alofa.srv.loop import Loop
from alofa.srv.master import run_workers
from alofa.core.memory import Arena
from alofa.engine.core import MAX_PROMPT, EngineCore
from alofa.engine.executor import MAX_BATCH, MAX_GEN, int_map
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
    var b_req: List[Int]
    var b_out: List[Int]
    var b_n: List[Int]
    var b_steps: List[Int]
    var b_emitted: List[Int]

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
        for i in range(MAX_BATCH):
            self.b_req.append(NO_REQUEST)
            self.b_n.append(0)
            self.b_steps.append(0)
            self.b_emitted.append(0)
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

        # ---- 批路：贪心，且形状装得进引擎 ----
        # 三个上限都是引擎自己的（`submit` 会指名拒绝越界的，所以这里是**先**判
        # 断再决定走哪条路，而不是碰运气）：prompt ≤ `MAX_PROMPT`、新 token ≤
        # `MAX_GEN`、同时在批 ≤ `MAX_BATCH`。越界的请求不是错，只是走老路。
        var slot = self._free_slot()
        if (
            self.batch_enabled
            and temperature <= 0.0
            and steps > 0
            and len(ids) <= MAX_PROMPT
            and steps <= MAX_GEN
            and slot >= 0
        ):
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
            return StreamToken("", True)
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
        return StreamToken(self._new_text(), False)

    def stream_end(mut self, request: Int) raises:
        """把 `request` 占的东西还回来（**幂等**：号对不上就什么也不做）。

        批路要还的是引擎里那个**槽位**：`cancel` 只是把取消排进下一拍，所以这里
        必须再跑一拍让它落地 —— 少了那一拍，槽位一直占着，第 `MAX_BATCH` + 1 条
        请求会收到 "the engine is full"，而真正的原因（某条连接断了）在别处。
        """
        var slot = self._slot_of(request)
        if slot >= 0:
            if self.engine.find(request) >= 0:
                self.engine.cancel(request)
                self.engine.tick[BACKEND_AVX2](self.model)
            self.b_req[slot] = NO_REQUEST
            self.b_n[slot] = 0
            self.b_steps[slot] = 0
            self.b_emitted[slot] = 0
            return
        if self.stream_request != request:
            return
        self.stream_request = NO_REQUEST

    def _free_slot(self) -> Int:
        """批表里第一个空位（-1 = 满了）。满了不是错 —— 请求会走老路。"""
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

    def _batch_next(mut self, slot: Int, request: Int) raises -> StreamToken:
        """批路走一步：引擎还没交出第 `n` 个 token 就推它，直到交出或没活干。

        一次 `tick` 推进**所有**在批里的请求 —— 这正是批调度的收益所在：八条流的
        第 k 个 token 是同一次前向算出来的。
        """
        var want = self.b_n[slot]
        while self.engine.n_output(request) <= want:
            if not self.engine.has_work():
                break
            self.engine.tick[BACKEND_AVX2](self.model)
        if self.engine.n_output(request) <= want:
            # 引擎交不出更多了：这条流到终点了（也可能被抢占后没再排上 —— 无论哪
            # 一种，"给一个终点"都比"一直等"好：后者在客户端是永远等不到 `[DONE]`）。
            return StreamToken("", True)
        var arena = Arena(MAX_GEN * 8 + 64)
        var dest = int_map(arena.alloc(MAX_GEN * 8))
        var n = self.engine.output(request, dest)
        for i in range(n):
            self.b_out[slot * MAX_GEN + i] = dest[unsafe_offset=i]
        arena.keep_alive()  # 同上：指针用完之前 arena 不许析构
        var text = self._batch_text(slot, n)
        self.b_n[slot] = want + 1
        return StreamToken(text, self.b_n[slot] >= self.b_steps[slot])

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
