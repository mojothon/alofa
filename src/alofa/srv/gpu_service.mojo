"""GPU 后端的 `Service` 实现：**批行池 + tick 合批**（账本一百三十四）
接到 `ChatHandler` 的 OpenAI/SSE 之下——HTTP、`/v1/chat/completions`、
流式 SSE 全部复用 srv 层，本模块只回答「怎么生成、怎么并发」。

并发模型（一百三十二 → 一百三十四两步演化）
-------------------------------------------
**池（132）**：N 份 GpuSession 各持独立 KV——并发的是进展不是吞吐
（每次前向 t=1）。133 修掉跨路径 KV 断层后，池交替同链成立。

**批行池 + tick（134，本层）**：合批的正确形态不是跨 session（一次
batch 前向只读一个 session 的 store），而是**单 session 的 N 个批行**
——`GpuStream` 的下标就是 `store_b` 的批行号（行间 KV 天然隔离，正是
133 统一后的布局；`prefill_step` 的 `row_base` 参让任意行子集可前向）。

推进模型 = `generate_batch` 同构的**相位混合 tick**：

* `_tick()` 一次前向推进所有活跃行各一步：prompt 相位的行喂
  `prompt[pidx]`（KV 落位 pidx），生成相位的行喂 `last`（链式）；
  空闲行 padding（喂 0、pos 0，tokout 丢弃）。尾枚 prompt token 那
  步的 tokout 就是 gen[0]——首帧与后续帧走**同一条**链，与
  `generate_batch` 逐位同构（parity 门钉）。
* `stream_next` 需要帧时触发 tick：帧存进行状态（`frame_ready`），
  其他行同 tick 产生的帧等它们自己来领——单线程 reactor 交替问
  N 条流，实际前向次数 ≈ max(各流步数) 而不是总和。
* **合批条件**：活跃行全部 temp==0（argmax 无参数分歧、行号无关
  确定）。temp>0 的采样参数是整批标量、Gumbel 种子按紧密行号派生
  （`seed ^ b*C`）——行号随编排漂移会破坏重放语义，故带采样参数的
  流退化为逐行 t=1（`row_base=slot`，种子语义与 129 逐位一致）。
  per-row 参数核（temp/seed/topp 数组化）是合批的下一刀。

池满策略不变：`stream_begin` 取不到空闲行就指名拒绝（`ERR_CAPACITY`
——响应头之前，`ChatHandler` 回 400）。`complete` 需要整个 session
（`generate_batch` 清全表 posarr 并从行 0 起排布）——有任何活跃流时
同样指名拒绝。

流式相位（131/133 的语义全部保留）：

* 每步种子 = `seed + count`：同一条流逐步去相关，重放（同种子）逐位
  复现（逐行模式下 `row_base=slot` 不改核内行号 b=0，种子不变）。
* 采样全在 GPU tail（temp=0 argmax / Gumbel-max / nucleus）。
* 文本帧走**字节级整段前缀解码**（`decode_bytes` + `utf8_tail_safe_len`
  帧尾边界截断，账本一百三十三）：多字节跨 token 不产生乱码帧。

`spawn_twin`（engine 线程路）给孪生体一个**单行**的自己（一份 KV）
并共享同一 ctx——多线程共用一条 ctx 的并发安全尚未验证。
"""

from max.gpu.host import DeviceContext

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from alofa.srv.engine_thread import Twinable, heap_place
from alofa.srv.http import bytes_to_text, utf8_tail_safe_len
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
    """批行池里一个**行位**的流相位。`request == NO_REQUEST` 即空闲。

    行号即 `store_b` 的批行号——一条流的所有 KV 都住 store_b 的这一
    行，行间互不越界（133 统一后的布局天然保证）。找流靠 `request`
    号（号对不上就红，协议契约与单流版一致）。

    `frame_*` 是 tick 存下的待领帧：tick 推进所有活跃行，每行把产出
    存好，`stream_next` 来领——交替问流时一 tick 伺候全部。
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
    # 生成链上游（下一 tick 喂给模型的 token）：收行时推进。⚠️ 与
    # `last` 分离——last 是「已领的最新一枚」（出队时更新，门的逐位
    # 同链收集靠它），tick 的组行跨多步时若读 last 会喂进陈旧 token
    # （134 门实测：B 第 2 枚 220 = 对 token 0 的响应）。
    var cur: Int
    # prompt 游标：pidx < plen = prompt 相位（喂 prompt[pidx]）；
    # pidx == plen = 生成相位（喂 last）。尾枚步的 tokout 是 gen[0]。
    var pidx: Int
    # 帧队列（⚠️ 不能是单槽）：tick 推进所有活跃行，一行一帧；当
    # 别的行的领帧循环一次跨多步时，本行的中间帧会被覆盖（134 门
    # 实测 B 丢前两帧）。队列存每帧的 emitted 终值（字节）与 done
    # 标志；`stream_next` 按游标出队，文本用 [prev_end, end) 从整段
    # 前缀里现切——out 不变时 decode_bytes 恒定，切片即帧。
    var frame_ends: List[Int]
    var frame_dones: List[Bool]
    # 每帧对应的 token（eos 帧记推进前的 last——`last` 的语义是
    # 「已领的最新一枚」，由 `stream_next` 出队时更新；收行时直接
    # 推 last 会让它跑到帧前面，门的逐位同链收集就错了）。
    var frame_tokens: List[Int]
    var frame_head: Int

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
        self.cur = 0
        self.pidx = 0
        self.frame_ends = List[Int]()
        self.frame_dones = List[Bool]()
        self.frame_tokens = List[Int]()
        self.frame_head = 0


struct GpuService(Service, Movable, Deinitable, Twinable):
    var session: GpuSession
    var streams: List[GpuStream]
    var n_rows: Int
    var tokenizer: Tokenizer
    var vocab: Int
    var max_tokens: Int
    var seed: Int
    var eos: Int
    # 合批口径（门钉「真合批发生」）：tick 总数 / 合批 tick / 逐行步。
    var n_ticks: Int
    var n_batched: Int
    var n_single: Int
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
        # 单 session、N 个批行（参数名留 n_sessions 兼容 serve_gpu 的
        # ALOFA_GPU_SESSIONS——语义已是「并发行容量」）。行间 KV 隔离
        # 由 store_b 的 [row][pos][kv] 布局天然保证（133）。
        var ctx = DeviceContext()
        self.session = GpuSession(model, max_tokens, ctx)
        self.streams = List[GpuStream]()
        var n_rows = n_sessions
        if n_rows > GPU_BMAX:
            n_rows = GPU_BMAX
        if n_rows < 1:
            n_rows = 1
        for _i in range(n_rows):
            self.streams.append(GpuStream())
        self.n_rows = n_rows
        self.tokenizer = tokenizer^
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.seed = seed
        # eos 走导出层（tokenizer 的 added tokens 里查 `<|im_end|>`），
        # 钉值只是查不到时的兜底——换 fixture 不再静默死在错的 id 上。
        var exported = self.tokenizer.special_id("<|im_end|>")
        self.eos = exported if exported >= 0 else DEFAULT_EOS
        self.n_ticks = 0
        self.n_batched = 0
        self.n_single = 0
        self.weights_dir = weights_dir
        self.config_path = config_path
        self.tokenizer_json = tokenizer_json

    def _slot_of(self, request: Int) -> Int:
        var slot = 0
        while slot < self.n_rows:
            if self.streams[slot].request == request:
                return slot
            slot += 1
        return -1

    def _free_slot(self) -> Int:
        var slot = 0
        while slot < self.n_rows:
            if self.streams[slot].request == NO_REQUEST:
                return slot
            slot += 1
        return -1

    def _any_active(self) -> Bool:
        """有流占着行（complete 需要整个 session，见下）。"""
        var slot = 0
        while slot < self.n_rows:
            if self.streams[slot].request != NO_REQUEST:
                return True
            slot += 1
        return False

    def spawn_twin(self, index: Int) raises -> Int:
        """engine 线程路的那份"自己"：**单行**的自己（一份 KV）+ 共享
        ctx。多线程共用一条 ctx 未验证——与 engines>=1 的实验旗一致。"""
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
        """非流式：占一行自驱动到终点（一百三十七）。

        旧实现独占整个 session（generate_batch 清全表 posarr 从行 0
        起排布），流活跃时指名拒绝——HTTP 非流式与流式互斥是正确性债。
        本刀 complete 改为 `stream_begin` -> `stream_next` 循环 ->
        `stream_end` 的封装：它就是 tick 批行池里的一条普通活跃行，
        **与并发流共存**（行满才拒），采样语义由 per-row 核保证（t=1
        时种子派生与旧路径逐位同）。

        拼接语义：逐帧领到 done，各帧是 UTF-8 安全边界的增量切片，
        拼接 == 整段 decode（⑥ 判据）；eos 帧不吐文本（与流式同），
        completion_tokens 用本行 `out` 的真 token 枚数。"""
        var ids = self.tokenizer.encode(prompt)
        var n_in = len(ids)
        if n_in == 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "prompt tokenizes to nothing", ""
            )
        if n_in >= self.max_tokens:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "prompt exceeds the KV window",
                "prompt=" + String(n_in) + " cap=" + String(self.max_tokens),
            )
        self.stream_begin(prompt, max_tokens, temperature, top_p,
                          COMPLETE_MARKER)
        var cslot = self._slot_of(COMPLETE_MARKER)
        var text = String("")
        var done = False
        while not done:
            var fr = self.stream_next(COMPLETE_MARKER)
            text += fr.text
            if fr.done:
                done = True
        var n_out = len(self.streams[cslot].out)
        self.stream_end(COMPLETE_MARKER)
        return Completion(text, n_in, n_out)

    def stream_begin(
        mut self,
        prompt: String,
        max_tokens: Int,
        temperature: Float64,
        top_p: Float64,
        request: Int,
    ) raises:
        """占一个批行，只准备，不走前向（前向在第一次 stream_next
        ——那里还能回 400）。行满 = 指名拒绝。"""
        var slot = self._free_slot()
        if slot < 0:
            raise AlofaError(
                ERR_CAPACITY,
                "all gpu rows are busy",
                "asked=" + String(request) + " rows=" + String(self.n_rows),
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
        # 只清**本行**的 posarr（批行池里别的行可能是活跃流的——清全表
        # 是 132 池模型的遗产，本层严禁）。h_posarr 是 host pinned，
        # 写入即时生效，forward 时随整块上传。
        self.session.h_posarr[slot] = Int32(0)
        var st = GpuStream()
        st.request = request
        st.prompt = ids^
        st.steps = steps
        st.temp = Float32(temperature)
        st.topp = Float32(top_p)
        st.seed = self.seed
        self.streams[slot] = st^

    def _frame_of(mut self, slot: Int) raises -> String:
        """出队一帧：`emitted` 前移到队头帧的 end，文本 = 两 end 之间
        的字节（字节级 + 帧尾边界截断在入队时已完成，账本一百三十
        三）。"""
        var end = self.streams[slot].frame_ends[self.streams[slot].frame_head]
        var prev = self.streams[slot].emitted
        if prev > end:
            # 不该发生（end 只会前移）。真发生了就从头对齐。
            prev = 0
        var text = bytes_to_text(
            self.tokenizer.decode_bytes(self.streams[slot].out), prev, end
        )
        self.streams[slot].emitted = end
        return text

    def _forward_rows(
        mut self, first_slot: Int, slots: List[Int], batched: Bool
    ) raises:
        """一次前向推进 `slots`（行 k ↔ slot first_slot+k；None 位置
        的行由调用方以 padding 补齐）。`batched` = 全 argmax（无参数
        分歧）；False = 单行带自己的采样参数。

        ⚠️ 就地变异：List 下标不能 move-out（origin 规则）——流状态
        全走 `self.streams[slot].x` 可变引用。
        """
        var t = len(slots)
        # 组行：prompt 相位喂 prompt[pidx]，生成相位喂 last；padding
        # 行喂 0（tokout 丢弃）。posarr[slot] = pos（喂 token 的位置）。
        var toks = List[Int]()
        var fed_gen = List[Bool]()
        var k = 0
        while k < t:
            var slot = slots[k]
            if slot >= 0:
                if self.streams[slot].pidx < len(self.streams[slot].prompt):
                    toks.append(self.streams[slot].prompt[self.streams[slot].pidx])
                    fed_gen.append(
                        self.streams[slot].pidx == len(self.streams[slot].prompt) - 1
                    )
                else:
                    toks.append(self.streams[slot].cur)
                    fed_gen.append(True)
                self.session.h_posarr[first_slot + k] = Int32(
                    self.streams[slot].pos
                )
                # per-row 采样参数（一百三十六）：padding 行给 temp=0
                # （核内 argmax 分支，tokout 丢弃）；活跃行种子 = 流种子
                # + 该步 count——与旧逐行 t=1 传参逐位同（种子 host 直
                # 传、行号无关），重放语义保持。
                self.session.h_stemp[first_slot + k] = self.streams[slot].temp
                self.session.h_stopps[first_slot + k] = self.streams[slot].topp
                self.session.h_sseeds[first_slot + k] = Int32(
                    self.streams[slot].seed + self.streams[slot].count
                )
            else:
                # padding 行：喂 0、pos 0——KV 写它自己行位的位置 0，
                # 不越入别的行（store_b 行隔离）。
                toks.append(0)
                fed_gen.append(False)
                self.session.h_posarr[first_slot + k] = Int32(0)
                self.session.h_stemp[first_slot + k] = Float32(0)
                self.session.h_stopps[first_slot + k] = Float32(1)
                self.session.h_sseeds[first_slot + k] = Int32(0)
            k += 1
        # per-row 采样（一百三十六）：一个 launch 吃任意 temp/top_p
        # 组合——argmax 是 temp=0 的核内特例（与 argmax_rb 逐位同），
        # 种子 host 直传与行号解耦，**合批条件彻底消失**：任何活跃行
        # 组合一次前向，每行与它单独前向逐位同。
        _ = self.session.prefill_step[
            GPU_INNER, GPU_FF, GPU_BMAX, GPU_HQ, GPU_KV_DIM,
            GPU_HEAD_DIM, GPU_ATTN_T, GPU_MAX_TOKENS, GPU_MB,
            GPU_ATTN_TILES,
            DType.float32, True, True, True, True, True, True, True,
            True, False, True
        ](
            toks^, False, False, False, 0.0, Int32(0), 1.0,
            0, 1,  # layer_gap 默认
            first_slot,
            True,  # per_row_sampling
        )
        self.session.ctx.synchronize()
        # 收行：prompt 相位步（非尾枚）的 tokout 丢弃；尾枚步与生成步
        # 的 tokout 是生成 token，产帧。
        k = 0
        while k < t:
            var slot = slots[k]
            if slot >= 0 and fed_gen[k]:
                var tv = Int(self.session.h_tokout[first_slot + k])
                if tv == self.eos:
                    # eos 不吐（它的文本是控制序列，不是内容）：直接收流。
                    # 帧队列照样入队（done 帧的 end 不前移 ⇒ 空文本）；
                    # token 记推进前的 last（eos 不进链，与 129 语义同）。
                    self.streams[slot].count = self.streams[slot].steps
                    self.streams[slot].frame_ends.append(
                        self.streams[slot].emitted
                    )
                    self.streams[slot].frame_dones.append(True)
                    self.streams[slot].frame_tokens.append(
                        self.streams[slot].last
                    )
                else:
                    self.streams[slot].out.append(tv)
                    self.streams[slot].cur = tv
                    self.streams[slot].count += 1
                    self.streams[slot].pos += 1
                    if self.streams[slot].pidx < len(self.streams[slot].prompt):
                        # 尾枚步：prompt 游标收尾。
                        self.streams[slot].pidx = len(self.streams[slot].prompt)
                    # 帧尾边界截断在此完成：end = 最后一个完整
                    # UTF-8 字符边界。⚠️ 只入队 end，**不动 emitted**
                    # ——emitted 是"已发给客户端的字节"，只在出队时
                    # 前移；入队就推进会吞掉 [prev, end) 的帧文本
                    # （切片变空，134 门实测）。
                    var full = self.tokenizer.decode_bytes(
                        self.streams[slot].out
                    )
                    var safe = utf8_tail_safe_len(full)
                    self.streams[slot].frame_ends.append(safe)
                    self.streams[slot].frame_dones.append(
                        self.streams[slot].count >= self.streams[slot].steps
                    )
                    self.streams[slot].frame_tokens.append(tv)
            elif slot >= 0:
                # prompt 相位步：只推游标与位置（KV 已写）。
                self.streams[slot].pidx += 1
                self.streams[slot].pos += 1
            k += 1

    def _tick(mut self) raises:
        """推进所有活跃行各一步。全 argmax → 合批（t = 最大活跃行号+1，
        padding 补齐，row_base=0）；有采样参数 → 逐行（每行 t=1，
        row_base=slot）。"""
        self.n_ticks += 1
        # 活跃 = 占着行且未交完（done 的行不再推进）。
        var act = List[Int]()
        var max_slot = -1
        var i = 0
        while i < self.n_rows:
            var st_active = (
                self.streams[i].request != NO_REQUEST
                and self.streams[i].count < self.streams[i].steps
            )
            if st_active:
                act.append(i)
                max_slot = i
            i += 1
        if max_slot < 0:
            return
        # 合批（一百三十六：per-row 采样核让合批条件彻底消失——任意
        # temp/top_p 组合一批搞定，每行与它单独前向逐位同）。
        self.n_batched += 1
        var slots = List[Int]()
        var k = 0
        var ai = 0
        while k <= max_slot:
            if ai < len(act) and act[ai] == k:
                slots.append(k)
                ai += 1
            else:
                slots.append(-1)  # padding
            k += 1
        self._forward_rows(0, slots^, True)

    def stream_next(mut self, request: Int) raises -> StreamToken:
        var slot = self._slot_of(request)
        if slot < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "stream phase mismatch",
                "asked=" + String(request) + " (not in the pool)",
            )
        # 无待领帧且流未完 → tick（会顺带推进别的活跃行——合批的
        # 意义所在）。帧队列保证本行的每一帧都在，不管别的行被 tick
        # 跨越了多少步。
        while (
            self.streams[slot].frame_head >= len(self.streams[slot].frame_ends)
            and self.streams[slot].count < self.streams[slot].steps
        ):
            self._tick()
        if self.streams[slot].frame_head >= len(self.streams[slot].frame_ends):
            # 交完了（帧全领完）还被问：仍然回答"结束"。
            return StreamToken("", True, False)
        var fd = self.streams[slot].frame_dones[self.streams[slot].frame_head]
        var text = self._frame_of(slot)
        # last = 本帧 token（不是收行时的最新 tick——tick 会跨多步，
        # 门按 last 收集逐位同链，last 必须跟着领帧走）。
        self.streams[slot].last = self.streams[slot].frame_tokens[
            self.streams[slot].frame_head
        ]
        self.streams[slot].frame_head += 1
        # ⚠️ done 帧也带文本：末帧的最后几枚 token 的字节就在这一段
        # 里——返回空串会把它们永远吞掉。「不吐文本」只属于 eos 帧
        # （eos 入队时 end 不前移 ⇒ 空文本）。134 门实测：末帧 4 字节
        # 被吞，join 恰为 decode 的前缀。
        return StreamToken(text, fd, False)

    def stream_end(mut self, request: Int) raises:
        """把行还回池里。幂等：号不在池里就什么也不做（走完自己收 +
        对端断开再收是常态）。"""
        var slot = self._slot_of(request)
        if slot >= 0:
            self.streams[slot].request = NO_REQUEST
            self.streams[slot].prompt = List[Int]()
            self.streams[slot].out = List[Int]()
            self.streams[slot].frame_ends = List[Int]()
            self.streams[slot].frame_dones = List[Bool]()
            self.streams[slot].frame_tokens = List[Int]()
            self.streams[slot].frame_head = 0
