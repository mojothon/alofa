"""GPU 后端的 `Service` 实现：把 `GpuSession`（q4/f16 全 GPU 路）接到
`ChatHandler` 的 OpenAI/SSE 之下 —— HTTP、/v1/chat/completions、流式
SSE 全部复用 srv 层，本模块只回答「怎么生成」。

与 CPU 路（`serve.mojo` 的 `ModelService`）的关系
------------------------------------------------
CPU 路有批路/单流两条；GPU 路先只做**单流**（一次一条流，第二条指名
拒绝）：吞吐来自 session 内部的 B 路批解码（f16 路实测 ~11×），服务
并发留给后面的「多 session 多路复用」（账本一百二十五/一百二十六）。

流式的相位与 CPU 路同构（`stream_begin` 只准备，前向留在第一次
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
nucleus 截断。CPU `Sampler` 不参与。

`spawn_twin`（engine 线程路）给孪生体一份**自己的 session**（q4 重传
一遍）并共享同一个 `DeviceContext`——多线程共用一条 ctx 的并发安全
尚未验证，与 `engines>=1` 本身的实验性质一致。
"""

from max.gpu.host import DeviceContext

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.openai import NO_REQUEST, Completion, Service
from alofa.srv.sse import StreamToken
from alofa.tokenizer import Tokenizer, load_tokenizer_json

from alofa.core.error import ERR_INVALID_ARGUMENT, AlofaError


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

# eos 默认值：Qwen2.5 chat 的 `<|im_end|>`（= 151645，fixture
# special.tsv 第 3 行钉着十六进制 3c7c696d5f656e647c3e）。它是**可写字段**
# （门里用"取链中一枚冒充 eos"来钉终止分支），不是编译期常量——真正
# 的 eos 该从 tokenizer/配置里导出，导出层建好前先用这个值。
alias DEFAULT_EOS = 151645


struct GpuService(Service, Movable, Deinitable, Twinable):
    var session: GpuSession
    var tokenizer: Tokenizer
    var vocab: Int
    var max_tokens: Int
    var seed: Int
    # spawn_twin 要用的三个路径（与 ModelService 同款：存路径不存模型，
    # 孪生体按同一份权重重开一条 session）。
    var weights_dir: String
    var config_path: String
    var tokenizer_json: String

    # 在途流状态（单流：整块状态只属于一条流，号对不上就红）。
    var s_request: Int
    var s_prompt: List[Int]
    var s_steps: Int
    var s_count: Int
    var s_last: Int
    var s_temp: Float32
    var s_topp: Float32
    var s_seed: Int
    var s_started: Bool
    var s_pos: Int
    var eos: Int

    def __init__(
        out self,
        var session: GpuSession,
        var tokenizer: Tokenizer,
        vocab: Int,
        max_tokens: Int,
        seed: Int,
        weights_dir: String,
        config_path: String,
        tokenizer_json: String,
    ) raises:
        self.session = session^
        self.tokenizer = tokenizer^
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.seed = seed
        self.weights_dir = weights_dir
        self.config_path = config_path
        self.tokenizer_json = tokenizer_json
        self.s_request = NO_REQUEST
        self.s_prompt = List[Int]()
        self.s_steps = 0
        self.s_count = 0
        self.s_last = 0
        self.s_temp = 0.0
        self.s_topp = 1.0
        self.s_seed = seed
        self.s_started = False
        self.s_pos = 0
        self.eos = DEFAULT_EOS

    def spawn_twin(self, index: Int) raises -> Int:
        """engine 线程路的那份"自己"：自己的 session（q4 重传）+ 共享
        ctx。多线程共用一条 ctx 未验证——与 engines>=1 的实验旗一致。"""
        var model = QwenForward(self.weights_dir, self.config_path, self.max_tokens, True)
        var twin_session = GpuSession(model, self.max_tokens, self.session.ctx)
        var tokenizer = load_tokenizer_json(self.tokenizer_json)
        var twin = GpuService(
            twin_session^,
            tokenizer^,
            self.vocab,
            self.max_tokens,
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
        """非流式：一次 `generate_batch` 到终点（连续批处理 v2，单行）。"""
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
        var lens = List[Int]()
        lens.append(len(ids))
        var out = self.session.generate_batch[
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
        # eos 停行之后，generate_batch 把** eos 本身**记进该行（其后剩余步
        # 以 NO_TOKEN(=-1) 填充，executor 的哨兵）——两样都不进 decode：
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
        """只准备，不走前向（前向在第一次 stream_next——那里还能回 400）。"""
        if self.s_request != NO_REQUEST:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "this service is already streaming another request",
                "asked=" + String(request) + " streaming=" + String(self.s_request),
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
        # KV 回绕：上一条流的 KV 全部作废（单流复用同一份 store）。
        self.session.kv_pos = 0
        for i in range(GPU_BMAX):
            self.session.h_posarr[i] = Int32(0)
        self.session.ctx.synchronize()
        self.s_prompt = ids^
        self.s_steps = steps
        self.s_count = 0
        self.s_last = 0
        self.s_temp = Float32(temperature)
        self.s_topp = Float32(top_p)
        self.s_seed = self.seed
        self.s_started = False
        self.s_pos = 0
        self.s_request = request

    def stream_next(mut self, request: Int) raises -> StreamToken:
        if request != self.s_request:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "stream phase mismatch",
                "asked=" + String(request) + " streaming=" + String(self.s_request),
            )
        if self.s_count >= self.s_steps:
            # 交完了还被问：仍然回答"结束"（路由可能多问一次）。
            return StreamToken("", True, False)
        var tok = 0
        if not self.s_started:
            # prefill：prompt 去尾一枚（尾枚走解码步，与后续同一条采样路）。
            var n = len(self.s_prompt)
            if n > 1:
                var pre = List[Int]()
                for i in range(n - 1):
                    pre.append(self.s_prompt[i])
                _ = self.session.prefill_step[
                    GPU_INNER, GPU_FF, GPU_BMAX, GPU_HQ, GPU_KV_DIM,
                    GPU_HEAD_DIM, GPU_ATTN_T, GPU_MAX_TOKENS, GPU_MB,
                    GPU_ATTN_TILES,
                    DType.float32, True, True, True, True, True, True, True,
                    True, False, False
                ](pre^, False, True, False)
                self.session.ctx.synchronize()
            self.s_pos = n - 1
            # 首 token 的解码步喂的是 prompt 尾枚（不是 0）。
            self.s_last = self.s_prompt[n - 1]
            self.s_started = True
        # 解码步：批路 t=1，位置写 h_posarr[0]（批路不读 kv_pos）。
        var one = List[Int]()
        one.append(self.s_last)
        self.session.h_posarr[0] = Int32(self.s_pos)
        _ = self.session.prefill_step[
            GPU_INNER, GPU_FF, GPU_BMAX, GPU_HQ, GPU_KV_DIM,
            GPU_HEAD_DIM, GPU_ATTN_T, GPU_MAX_TOKENS, GPU_MB,
            GPU_ATTN_TILES,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](
            one^, False, False, False,
            self.s_temp, Int32(self.s_seed + self.s_count), self.s_topp,
        )
        self.session.ctx.synchronize()
        tok = Int(self.session.h_tokout[0])
        if tok == self.eos:
            # eos 不吐（它的文本是控制序列，不是内容）：直接收流。
            # done 后继续问仍 done=True（协议契约）。
            self.s_count = self.s_steps
            return StreamToken("", True, False)
        self.s_last = tok
        self.s_count += 1
        self.s_pos += 1
        var text = self.tokenizer.decode(_make_one(tok))
        if self.s_count >= self.s_steps:
            return StreamToken(text, True, False)
        return StreamToken(text, False, False)

    def stream_end(mut self, request: Int) raises:
        """幂等：号对不上就什么也不做（走完自己收 + 对端断开再收是常态）。"""
        if self.s_request == request:
            self.s_request = NO_REQUEST


def _make_one(tok: Int) -> List[Int]:
    var l = List[Int]()
    l.append(tok)
    return l^
