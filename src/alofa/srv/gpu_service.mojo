"""GPU 后端的 `Service` 实现：把 **session 池**（N 份 `GpuSession` 共享
同一份权重与同一个 `DeviceContext`，账本一百二十五/一百二十六）接到
`ChatHandler` 的 OpenAI/SSE 之下——HTTP、`/v1/chat/completions`、流式
SSE 全部复用 srv 层，本模块只回答「怎么生成、怎么并发」。

并发模型（一百三十二）
----------------------
**池，不是线程**。reactor 循环一次只调一个 `stream_next`，但池里 N 条
session 各自持有独立的 KV 与流相位——`stream_begin` 从池里取一条空闲
session，第二条流落在另一条 session 上，两条流**交替推进、互不覆盖**
（126 验过：共享 ctx 的多 session KV 独立、穿插确定性逐位成立）。
吞吐还是串行的（每次前向 t=1，GPU 把调用排成队），并发的是**进展**：
一条 20 s 的生成不再把其它请求挡在池外。跨流合批（一次前向推 N 条
流各一枚，t=N 行摊薄权重读）是下一步，不在本层。

池满的策略：`stream_begin` / `complete` 取不到空闲 session 就**指名
拒绝**（`ERR_CAPACITY`）——`stream_begin` 的错发生在响应头之前，
`ChatHandler` 把它回成 400，客户端可以立刻重试；不做排队帧（waiting
那条路要动 loop 的队列语义，留给合批那一轮一起做）。

流式相位与 CPU 路同构（`stream_begin` 只准备，前向留在第一次
`stream_next`——那是唯一还能把请求的错回成 400 的地方）：

* prefill：prompt 去尾一枚，`prefill_step(batch=False)` 一段算完
  （prompt 尾枚留给解码步，这样首 token 与后续 token 走**同一条**
  采样路径）；
* 解码步：`prefill_step(batch=True, t=1)`，位置在 `h_posarr[0]`
  （批路的位置语义是逐行的，不读 `kv_pos`——账本一百二十九）；
* 每步种子 = `seed + count`：同一条流逐步去相关，重放（同种子）逐位
  复现。

采样全在 GPU tail（`sample_temp` / `sample_topp`，账本一百二十三/
一百二十八）：temp=0 走 argmax，temp>0 走 Gumbel-max，top_p<1 走
nucleus 截断。文本帧走**整段前缀解码**（与 CPU 路 `_new_text` 同法，
账本一百三十一）：多字节字符跨 token 不产生乱码帧。

eos 走导出层（`Tokenizer.special_id("<|im_end|>")`，账本一百三十一）；
`spawn_twin`（engine 线程路）给孪生体**一条**自己的 session（不放大
池——N 线程 × M session = N×M 份 KV，账没有这么记的）并共享同一个
`DeviceContext`——多线程共用一条 ctx 的并发安全尚未验证。
"""

from max.gpu.host import DeviceContext

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import bytes_of_text, bytes_to_text
from alofa.srv.openai import NO_REQUEST, Completion, Service
from alofa.srv.sse import StreamToken
from alofa.tokenizer import Tokenizer, load_tokenizer_json

from alofa.core.error import ERR_CAPACITY, ERR_INVALID_ARGUMENT, AlofaError


# prefill_step 的形状实参（comptime，见各门）。⚠️ 目前钉在
# Qwen2.5-0.5B fixture 上（hidden=896 / ff=4864 / 24 层）；换模型要
# 连着这几个 alias 一起换，session 里的运行时形状不会替你查——抄错
# 的症状是静默算错（与 serve.mojo「从同一份 cfg 读真值」的教训同源）。
alias GPU_INNER = 896
alias GPU_FF = 4864
alias GPU_BMAX = 64
alias GPU_HQ = 4
alias GPU_KV_DIM = 128
alias GPU_HEAD_DIM = 128
alias GPU_ATTN_T = 128
alias GPU_MAX_TOKENS = 512
alias GPU_MB = 1
alias GPU_ATTN_TILES = 8

# eos 钉值兜底（导出层查不到时用）：Qwen2.5 chat 的 `<|im_end|>`
# （= 151645，fixture special.tsv 第 3 行钉着十六进制）。
alias DEFAULT_EOS = 151645

# `complete()` 在池里占坑的标记：它没有 `request` 号可挂，用这个
# 区别于 NO_REQUEST（空闲）与任何流号。拿 -1 会跟 NO_REQUEST 撞。
alias COMPLETE_MARKER = -2


struct GpuStream(Movable):
    """池里一个坑位的**流相位**。`request == NO_REQUEST` 即空闲。

    一条流的全部可变状态都住在这里——池化之后不再有"整块状态只属于
    一条流"的单份字段，找流靠 `request` 号（号对不上就红，协议契约
    与单流版一致）。
    """

    var request: Int
    var prompt: List[Int]
    var out: List[Int]
    var emitted: Int
    var steps: Int
    var count: Int
    var last: Int
    var temp: Float32
    var topp: Float32
    var seed: Int
    var started: Bool
    var pos: Int

    def __init__(out self):
        self.request = NO_REQUEST
        self.prompt = List[Int]()
        self.out = List[Int]()
        self.emitted = 0
        self.steps = 0
        self.count = 0
        self.last = 0
        self.temp = 0.0
        self.topp = 1.0
        self.seed = 0
        self.started = False
        self.pos = 0


struct GpuService(Service, Movable, Deinitable, Twinable):
    var sessions: List[GpuSession]
    var streams: List[GpuStream]
    var n_sessions: Int
    var tokenizer: Tokenizer
    var vocab: Int
    var max_tokens: Int
    var seed: Int
    var eos: Int
    # spawn_twin 要用的三个路径（与 ModelService 同款：存路径不存模型，
    # 孪生体按同一份权重重开）。
    var weights_dir: String
    var config_path: String
    var tokenizer_json: String

    def __init__(
        out self,
        model: QwenForward,
        n_sessions: Int,
        max_tokens: Int,
        var tokenizer: Tokenizer,
        vocab: Int,
        seed: Int,
        weights_dir: String,
        config_path: String,
        tokenizer_json: String,
    ) raises:
        # N 份 session **共享同一份权重**（model 借用）与**同一个
        # ctx**（每进程 DeviceContext 上限 2，账本一百二十四/一百二十五
        # ——多路复用必须同 ctx）。
        self.sessions = List[GpuSession]()
        var ctx = DeviceContext()
        for _i in range(n_sessions):
            self.sessions.append(GpuSession(model, max_tokens, ctx))
        self.streams = List[GpuStream]()
        for _i in range(n_sessions):
            self.streams.append(GpuStream())
        self.n_sessions = n_sessions
        self.tokenizer = tokenizer^
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.seed = seed
        # eos 走导出层（tokenizer 的 added tokens 里查 `<|im_end|>`），
        # 钉值只是查不到时的兜底——换 fixture 不再静默死在错的 id 上。
        var exported = self.tokenizer.special_id("<|im_end|>")
        self.eos = exported if exported >= 0 else DEFAULT_EOS
        self.weights_dir = weights_dir
        self.config_path = config_path
        self.tokenizer_json = tokenizer_json

    def _slot_of(self, request: Int) -> Int:
        var slot = 0
        while slot < self.n_sessions:
            if self.streams[slot].request == request:
                return slot
            slot += 1
        return -1

    def _free_slot(self) -> Int:
        var slot = 0
        while slot < self.n_sessions:
            if self.streams[slot].request == NO_REQUEST:
                return slot
            slot += 1
        return -1

    def spawn_twin(self, index: Int) raises -> Int:
        """engine 线程路的那份"自己"：**一条**自己的 session（不放大
        池——N 线程 × M session = N×M 份 KV）+ 共享 ctx。多线程共用
        一条 ctx 未验证——与 engines>=1 的实验旗一致。"""
        var model = QwenForward(
            self.weights_dir, self.config_path, self.max_tokens, True
        )
        var twin = GpuService(
            model,
            1,
            self.max_tokens,
            load_tokenizer_json(self.tokenizer_json),
            self.vocab,
            self.seed + index,
            self.weights_dir,
            self.config_path,
            self.tokenizer_json,
        )
        return heap_place(twin^)

    # ---- Service 协议 ----------------------------------------------------

    def complete(
        mut self,
        prompt: String,
        max_tokens: Int,
        temperature: Float64,
        top_p: Float64,
    ) raises -> Completion:
        """非流式：一次 `generate_batch` 到终点（连续批处理 v2，单行）。
        在池里占一个坑（`COMPLETE_MARKER`），算完就还。"""
        var slot = self._free_slot()
        if slot < 0:
            raise AlofaError(
                ERR_CAPACITY,
                "all gpu sessions are busy",
                "sessions=" + String(self.n_sessions),
            )
        self.streams[slot].request = COMPLETE_MARKER
        var ids = self.tokenizer.encode(prompt)
        if len(ids) == 0:
            self.streams[slot].request = NO_REQUEST
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "prompt tokenizes to nothing", ""
            )
        if len(ids) >= self.max_tokens:
            self.streams[slot].request = NO_REQUEST
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "prompt exceeds the KV window",
                "prompt=" + String(len(ids)) + " cap=" + String(self.max_tokens),
            )
        var lens = List[Int]()
        lens.append(len(ids))
        var out = self.sessions[slot].generate_batch[
            GPU_INNER, GPU_FF, GPU_BMAX, GPU_HQ, GPU_KV_DIM, GPU_HEAD_DIM,
            GPU_ATTN_T, GPU_MAX_TOKENS, GPU_MB, GPU_ATTN_TILES,
            DType.float32
        ](
            ids^,
            lens^,
            max_tokens,
            self.eos,
            List[Int](),
            List[Int](),
            Float32(temperature),
            Int32(self.seed),
            Float32(top_p),
        )
        self.streams[slot].request = NO_REQUEST
        # eos 停行之后，generate_batch 把 **eos 本身**记进该行（其后剩余
        # 步以 NO_TOKEN(=-1) 填充，executor 的哨兵）——两样都不进 decode：
        # eos 的文本是控制序列，不是内容；决算只数真 token。这样
        # complete 与流式（eos 那帧不吐文本）严格同链同文本。
        var real = List[Int]()
        for i in range(len(out)):
            if out[i] >= 0 and out[i] != self.eos:
                real.append(out[i])
        var n_out = len(real)
        var text = self.tokenizer.decode(real^)
        return Completion(text, len(ids), n_out)

    def stream_begin(
        mut self,
        prompt: String,
        max_tokens: Int,
        temperature: Float64,
        top_p: Float64,
        request: Int,
    ) raises:
        """从池里占一个坑，只准备，不走前向（前向在第一次 stream_next
        ——那里还能回 400）。池满 = 指名拒绝。"""
        var slot = self._free_slot()
        if slot < 0:
            raise AlofaError(
                ERR_CAPACITY,
                "all gpu sessions are busy",
                "asked=" + String(request)
                + " sessions=" + String(self.n_sessions),
            )
        var ids = self.tokenizer.encode(prompt)
        if len(ids) == 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "prompt tokenizes to nothing", ""
            )
        if len(ids) >= self.max_tokens:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "prompt exceeds the KV window",
                "prompt=" + String(len(ids)) + " cap=" + String(self.max_tokens),
            )
        var steps = max_tokens
        if steps > self.max_tokens - len(ids):
            steps = self.max_tokens - len(ids)
        # KV 回绕：这条 session 的上一条流的 KV 全部作废（池内复用）。
        self.sessions[slot].kv_pos = 0
        for i in range(GPU_BMAX):
            self.sessions[slot].h_posarr[i] = Int32(0)
        self.sessions[slot].ctx.synchronize()
        var st = GpuStream()
        st.request = request
        st.prompt = ids^
        st.steps = steps
        st.temp = Float32(temperature)
        st.topp = Float32(top_p)
        st.seed = self.seed
        self.streams[slot] = st^

    def stream_next(mut self, request: Int) raises -> StreamToken:
        var slot = self._slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "stream phase mismatch",
                "asked=" + String(request) + " (not in the pool)",
            )
        # ⚠️ 就地变异：List 下标不能 move-out（origin 规则）——读写字段
        # 都直接走 `self.streams[slot].x` 这个可变引用。
        if self.streams[slot].count >= self.streams[slot].steps:
            # 交完了还被问：仍然回答"结束"（路由可能多问一次）。
            return StreamToken("", True, False)
        var tok = 0
        if not self.streams[slot].started:
            # prefill：prompt 去尾一枚（尾枚走解码步，与后续同一条采样路）。
            var n = len(self.streams[slot].prompt)
            if n > 1:
                var pre = List[Int]()
                for i in range(n - 1):
                    pre.append(self.streams[slot].prompt[i])
                _ = self.sessions[slot].prefill_step[
                    GPU_INNER, GPU_FF, GPU_BMAX, GPU_HQ, GPU_KV_DIM,
                    GPU_HEAD_DIM, GPU_ATTN_T, GPU_MAX_TOKENS, GPU_MB,
                    GPU_ATTN_TILES,
                    DType.float32, True, True, True, True, True, True, True,
                    True, False, False
                ](pre^, False, True, False)
                self.sessions[slot].ctx.synchronize()
            self.streams[slot].pos = n - 1
            # 首 token 的解码步喂的是 prompt 尾枚（不是 0）。
            self.streams[slot].last = self.streams[slot].prompt[n - 1]
            self.streams[slot].started = True
        # 解码步：批路 t=1，位置写 h_posarr[0]（批路不读 kv_pos）。
        var one = List[Int]()
        one.append(self.streams[slot].last)
        self.sessions[slot].h_posarr[0] = Int32(self.streams[slot].pos)
        _ = self.sessions[slot].prefill_step[
            GPU_INNER, GPU_FF, GPU_BMAX, GPU_HQ, GPU_KV_DIM,
            GPU_HEAD_DIM, GPU_ATTN_T, GPU_MAX_TOKENS, GPU_MB,
            GPU_ATTN_TILES,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](
            one^, False, False, False,
            self.streams[slot].temp,
            Int32(self.streams[slot].seed + self.streams[slot].count),
            self.streams[slot].topp,
        )
        self.sessions[slot].ctx.synchronize()
        tok = Int(self.sessions[slot].h_tokout[0])
        if tok == self.eos:
            # eos 不吐（它的文本是控制序列，不是内容）：直接收流。
            # done 后继续问仍 done=True（协议契约）。
            self.streams[slot].count = self.streams[slot].steps
            return StreamToken("", True, False)
        self.streams[slot].out.append(tok)
        self.streams[slot].last = tok
        self.streams[slot].count += 1
        self.streams[slot].pos += 1
        # 整段前缀解码减已发字节（CPU 路 `_new_text` 同法）。
        var full = bytes_of_text(self.tokenizer.decode(self.streams[slot].out))
        var emitted = self.streams[slot].emitted
        if emitted > len(full):
            # 不该发生（前缀只会变长）。真发生了就从头对齐。
            emitted = 0
        var text = bytes_to_text(full, emitted, len(full))
        self.streams[slot].emitted = len(full)
        if self.streams[slot].count >= self.streams[slot].steps:
            return StreamToken(text, True, False)
        return StreamToken(text, False, False)

    def stream_end(mut self, request: Int) raises:
        """把坑还回池里。幂等：号不在池里就什么也不做（走完自己收 +
        对端断开再收是常态）。"""
        var slot = self._slot_of(request)
        if slot >= 0:
            self.streams[slot].request = NO_REQUEST
            self.streams[slot].prompt = List[Int]()
            self.streams[slot].out = List[Int]()
