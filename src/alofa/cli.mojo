"""One prompt, end to end: tokenizer → model → engine → sampler → text.

为什么这个文件必须存在
----------------------
在此之前，每一层都有自己的门，而且**都是绿的**：分词器对得上 HuggingFace，模型
对得上 HuggingFace，调度器对得上 Python 参照，批执行器对得上串行执行。但没有任何
一处把它们串起来跑过一次 —— `test_engine_core.mojo` 自己就写明它 "without a model:
the argmax is supplied by the test"。

那是一个理性的选择（引擎循环的测点是时序，不是数值），代价是**接缝从未被执行过**。
而接缝恰恰是历史欠账所在：2026-09-17 记过一次"各层各自绿、拼起来崩到 0/512"。
这个文件的职责就是让那条路径每天都能被走一遍。

它跑的是**真权重**（`tests/fixtures/qwen2.5-0.5b/weights`，fp32 裸 dump），不是玩具
形状。之所以可能，是因为那 1.9 GB 的切片就躺在本机；也因此它**不进 `pixi run test`**
—— 见 `pixi.toml` 里 `generate` 任务的说明。

用法：

    pixi run generate
    pixi run generate -- "The capital of France is"

⚠️ **要测速度别用上面的捷径**：`pixi run generate` 是 `-O0`（为了 CI 里编译快），
`-O2` 才配跟外部基准比。测速度照下面这句话来：

    pixi run mojo build -O2 -I src src/alofa/cli.mojo -o target/cli && ./target/cli

一次完整运行 = 5 趟演示 + `BENCH_REPS`×4 趟 A/B（默认 16 趟），所以 `-O0` 下会跑
上好几分钟 —— 那是 `-O0` 的代价，不是卡住了。CI 不跑这个任务（见 `pixi.toml`）。

量化通路（q4_0）
----------------
2026-09-18 起这里也跑一遍 q4_0，并把两者的 tok/s 打在同一张表里 —— 因为"量化到底
有没有用"这件事**只有在这条真实路径上才问得出来**：`test_q4_parity` 证明的是块格式
逐位对，`test_q4_greedy` 证明的是整网还认得自己，两者都是**正确性**；而"自回归循环
是不是更快"要的是同一个进程、同一份权重、同一个 prompt 的 A/B，缺一样就变成在比噪声。

**天花板不是 8×，也不是 4×，而是约 2.65×**，理由写在 `enable_q4` 里，这里重述一遍
因为它决定怎么看结果表：输出投影 `lm_head` 故意不量化（压它之后教师强制贪心一致率
掉到 0.77），而它一家就占了每 token 权重的 544.5 MB / 1976 MB = **27.6%**。剩下
72.4% 从 4 字节压到 4.5 bit（18 字节 / 32 个值 = **7.11×**，不是"4 bit 所以 4×"），
所以最低限度的耗时是原来的

    (1431 / 7.11 + 544.5) / 1976 = (201 + 544.5) / 1976 = 0.377 → **2.65×**

⚠️ 1976 MB 是**权重文件的字节数**（1,976,393,216 B = 494.1M 参数），不是估算：每个
decode step 把 24 层（357.85M 参数）与 `lm_head`（136.13M 参数）各读一遍，而 embed
只在 prefill 时按行查表（`lm_head` 与它绑定共用同一份字节，offset 都是 0）。账本里
曾用过 2.108 GB 这个数（每层记 65.1 MB，实际 59.6 MB），**偏高 6%**，别再引用。

能不能到那个 2.65×，取决于解量化核（`matmul_q4_f32`，2026-09-20 起**已有向量实现**）
跟不跟得上：它每个 nibble 都要 mask/shift → 转 f32 → 乘 scale → 转 f64 → 累加，字节
省了 7.11× 而每字节的活多了十几倍。这张表就是要回答这个 —— 两个方向都可能，不许为了
好看选一边。

**2026-09-20 晚的实测答案是：现在到不了，因为瓶颈换人了。** 核级（`down_proj`
4864×896，同进程相邻测量、best-of-3）：`fp32/avx2` = 1.485 ms（11.74 GB/s = 读峰值
的 78%，**访存受限**）而 `q4_0/avx2` = 1.539 ms（1.59 GB/s = 自己访存地板的 10.6%，
**算术受限**）→ 量化把省下的字节全花在解量化上了，**比 fp32 慢 3.6%**。所以端到端
A/B 不是"量不出结论"，而是"效应（±5%）比本机噪声（±10%）还小"。

怎么读那张表（2026-09-20 起）
----------------------------
`main` 最后那张 A/B 表是**重复 + 交错**的：四个变体在同一轮里各跑一遍，轮与轮之间
转顺序，一共 `BENCH_REPS` 轮。这不是仪式感 —— 同一份二进制、同一台机器、同一个
prompt，`q4/fp32（avx2 档）` 在六次里给过 0.61 / 1.01 / 1.59 / 1.72 / 1.89 /
2.16×，**方向都翻过**。所以判定只有一条：**两个区间重不重合**。重合就是"这份数据
区分不出两者"，照旧记"量不出结论"；不重合才写谁快。从区间里挑好看的那半边，比不测
更糟。

表里的数是**只 decode** 的，而且是**每 token 时间的中位数**：

- prefill 是整段 prompt 的一次前向，属于固定成本；混进分母，tok/s 就随 `n_new`
  漂移，而且会把真正要测的每 token 差别稀释掉。它被单独记在 `prefill` 一行。
- 中位数而不是均值：一个被打断的 token 能把 16 个样本的均值推走 6%，推不动
  中位数。表同时给出均值口径，**两个口径同向**才写结论（挡"口径挑选"）。
- 开跑前有一轮**预热**被丢掉：权重是 mmap 的，第一次读到哪一页就在那儿付一次
  缺页，而一个 decode step 会把全部权重读一遍 —— 缺页量的是内核的页管理，不是
  被测的 kernel。

⚠️ **加轮数不是让结论变好看的旋钮**：区间取 min…max，轮数越多区间越宽、判定越
保守。要分开区间只能降抖动，也就是上面这三条。
"""

from std.collections import List

from alofa.core.error import ERR_CAPACITY, ERR_IO, ERR_INVALID_ARGUMENT, AlofaError
from alofa.core.ffi import monotonic_ns
from alofa.core.memory import Arena
from alofa.core.rng import Rng
from alofa.core.text import parse_int, read_text
from alofa.engine.core import MAX_PROMPT, EngineCore
# MAX_BATCH 必须从 `executor` 拿：它经 `engine/batch.mojo` 是 **8**，而
# `engine/scheduler.mojo` 里那个同名常量是 **32**。从 scheduler 导入会拿 32 去
# 遍历只有 8 个槽的 `ex.live`（实测 `Assert Error: index 8`）—— 同一个名字两个
# 取值，抄错一处就崩在别人的数组里。
from alofa.engine.executor import MAX_BATCH, MAX_GEN, NO_TOKEN, IntPtr, int_map
from alofa.engine.scheduler import SchedConfig
from alofa.model.arch.qwen import (
    BACKEND_AVX2,
    BACKEND_SCALAR,
    QwenForward,
    backend_label,
)
from alofa.runtime.kv import MAX_SEQ_TOKENS
from alofa.runtime.sampler import LogitBias, SampleParams, Sampler
from alofa.tokenizer.tokenizer import load_tokenizer

# 真实架构（取自 `tests/fixtures/qwen2.5-0.5b/config.tsv`）。
comptime HIDDEN = 896
comptime INTER = 4864
comptime KV_DIM = 128
comptime N_LAYERS = 24
comptime N_HEADS = 14
comptime N_KV_HEADS = 2
comptime HEAD_DIM = 64
comptime VOCAB = 151936
# rope 表只导出了前 512 行（`dump_model_reference.py --max-positions`），
# 传 32768 会让查表越界。
comptime MAX_POS = 512

# 引擎的容量上限见 `engine/executor.mojo` / `engine/core.mojo`。
comptime ROWS = 64
comptime BLOCK = 16
# 一块 16 token，一条序列最多 `MAX_SEQ_TOKENS`（64）→ 4 块足够；给 64 块是为了让
# 水印（900‰ → 57 块）永远够不着，于是这条切片不会去走抢占路径 —— 抢占另有门。
comptime CAP_BLOCKS = 64
comptime WATERMARK = 900
comptime MAX_WAIT = 8

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime DEFAULT_PROMPT = "The capital of France is"
comptime DEFAULT_NEW = 16
comptime MAX_TICKS = 256
# 固定种子：采样路径要能复现，否则它既不能作为门也不能作为回归基线。
comptime SEED = UInt64(20260918)

# A/B 的轮数与档数。`BENCH_REPS` 取 **4** 不是随手定的：四档按 `(rep + k) % 4`
# 轮换时，每档恰好各占一次第一/第二/第三/第四个位置，于是"这次排在前面所以快"
# 这个偏差被摊平，而不是由某一次的顺序决定。少于 3 轮不给区间（那不叫区间），
# 多于 4 轮只是多花时间 —— 一轮四档，本机一趟几秒到十几秒。
comptime BENCH_REPS = 4
comptime N_ARMS = 4
# A/B 的生成长度：`MAX_GEN`（`engine/executor.mojo`）= **32** 是引擎给一条请求的上限，
# 所以 32 就是这批测量能拿到的最长生成 —— 演示用的 `DEFAULT_NEW` 仍是 16（跑得快、
# 打印短），但**测量**要更长的样本：每多一个 token 就多一个 decode step 的样本，
# 而每 token 是 memory-bound 的同一次权重流，多采样只会把偶发的慢 token 摊薄。
comptime BENCH_NEW = 32
# 预热只跑 2 个 token 就够：**一个 decode step 会把全部权重读一遍**，所以两个 token
# 就把 mmap 的每一页摸过了。它的结果**丢弃**，因为它量的是内核的缺页处理。
comptime BENCH_WARMUP_NEW = 2
# 每 token 要从内存里流过去的字节数。**fp32 档是整个文件**：一个 decode step 会把
# 24 层与 `lm_head` 的权重各读一遍（`ROWS=1` 时 matmul 是矩阵×向量，没有复用）。
# q4 档只把 24 层的七个投影压成 q4_0 —— **`lm_head` 不量化**（见
# `QwenForward.enable_q4`：一起压成 4 bit 会让教师强制贪心一致率掉到 0.77）。
#   文件 = 1.976 GB = 494.1M 参数；`lm_head` 与 `embed_tokens` **绑定共用**
#     （`tensors.tsv` 里两者 offset 都是 0），所以文件里只有一份 136.13M 参数。
#   24 层七槽 = 357.85M 参数 = 1.431 GB ；`lm_head` = 136.13M 参数 = 0.544 GB
#   q4_0 是 **32 个值 18 字节**（16 B nibble + 2 B fp16 scale）= 每值 0.5625 B
#     → 相对 fp32 是 **7.11×**，不是"4 bit 所以 4×" —— 那个直觉是错的。
#   fp32 档每 token = 1.976 GB（norm/bias 那 0.29 MB 忽略）
#   q4   档每 token = 1.431 / 7.11 + 0.544 = 0.201 + 0.544 = **0.746 GB**
# ⚠️ 所以"量化把每 token 字节降到 1/4"是错的：真实是**降到 38%（2.65×）**，而且在
#    q4 通路里未量化的 `lm_head` 自己就占 **73%**。这是"该不该开量化"的关键数。
comptime BYTES_PER_TOKEN_FP32 = 1.976e9
comptime BYTES_PER_TOKEN_Q4 = 0.746e9
# 本机实测读带宽（`scripts/bench_decode_roofline.mojo`）= 12.0 GB/s。把 tok/s 换成
# "占了天花板多少"，是因为**谁贴着天花板谁就脆**：贴着天花板的档对任何扰动（邻居、
# 缺页、TLB）都是 1:1 敏感，离天花板远的档几乎不动 —— 这本身就是"为什么只有 fp32
# 在抖"的一半答案。
comptime READ_BW_GB_S = 12.0
comptime PROC_STAT_PATH = "/proc/self/stat"
comptime PROC_LOADAVG_PATH = "/proc/loadavg"


def join_ids(ids: List[Int]) -> String:
    """把 id 列表拼成可打印的 `[a, b, c]`（Mojo 没有现成的列表格式化）。"""
    var out = "["
    for i in range(len(ids)):
        if i > 0:
            out += ", "
        out += String(ids[i])
    return out + "]"


def fixed2(x: Float64) -> String:
    """两位小数。tok/s 打全精度只会把表撑破，而那多出来的位数是假的精度。"""
    var a = x
    if a < Float64(0):
        a = -a
    var n = Int(a * 100.0 + 0.5)
    var frac = n % 100
    var tail = String(frac)
    if frac < 10:
        tail = "0" + tail
    return String(n // 100) + "." + tail


def sort_ns(mut s: List[Int]):
    """插入排序。样本量是几十（一趟生成的 token 数），O(n²) 无所谓；不用 `List.sort`
    是为了不引它对元素类型的要求 —— 这里只要能排 `Int`。
    """
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v


def median_ns(s: List[Int]) -> Int:
    """每 token 时间的**中位数**，µs 以上量级取整到 ns 就够。

    用中位数而不是均值：一个被 OS 打断的 token 能把 16 个样本的均值推走 6%，推不动
    中位数。偶数个样本取偏大的那个中间值 —— 差一个位置不影响任何结论。
    """
    if len(s) == 0:
        return 0
    return s[len(s) // 2]


def book_step(
    mut prefill_ns: Int,
    mut decode_ns: Int,
    mut n_decode: Int,
    mut steps: List[Int],
    rows: Int,
    dt: Int,
):
    """把一拍的时间记到它该去的地方。

    prefill 那一拍跑的是整段 prompt（`rows > 1`），decode 每拍一行 —— 分开是因为
    前者是**一次**的固定成本，后者才是**每 token** 的成本。

    ⚠️ 这个判据要求 prompt 长于一个 token 且不跨 `ROWS` 分块：本文件的 prompt 是
    5 个 token，`ROWS` 是 64，成立。`rows == 0` 的那拍是纯记账（调度器在传消息，
    见 `has_work` 的注释），它不产出 token，两边都不记。
    """
    if rows > 1:
        prefill_ns += dt
    elif rows == 1:
        decode_ns += dt
        n_decode += 1
        steps.append(dt)


struct SliceRun(Copyable, Movable):
    """一次切片的测量结果：拿它算比值，而不是把 tok/s 抄到纸上再比。

    `setup_ns` 与 `engine_ns` 分开存是有意的：量化（这里是 `enable_q4`）是**每次
    加载付一次**的成本，它混进 tok/s 会让"q4 更快"这个结论在短生成上随 `n_new`
    漂移 —— 16 个 token 和 256 个 token 会算出两个不同的"胜利者"。真实部署里量化
    出来的块是要落盘的（那是加载器的事，本轮没做），所以它根本不该进解码的成本。

    `prefill_ns` 与 `decode_ns` 同样分开（2026-09-20 起）：prefill 是整段 prompt 的
    **一次**前向，是固定成本；decode 才是**每 token** 的成本。混在一起，tok/s 就随
    `n_new` 漂移（16 个 token 与 32 个 token 不是一个数），也就没法跟 llama.cpp 的
    eval time 对齐；分开之后还多看见一件事：prefill 到底占掉多少。

    `p50_ns` / `step_lo_ns` / `step_hi_ns` 记**每 token** 时间的中位数与区间。中位数
    是这次测量的主统计量：一个被打断的 token 能把 16 个样本的均值推走 6%，推不动
    中位数。
    """

    var text: String
    var engine_ns: Int
    var setup_ns: Int
    var n_out: Int
    var prefill_ns: Int
    var decode_ns: Int
    var n_decode: Int
    var p50_ns: Int
    var step_lo_ns: Int
    var step_hi_ns: Int

    def __init__(
        out self,
        text: String,
        engine_ns: Int,
        setup_ns: Int,
        n_out: Int,
        prefill_ns: Int,
        decode_ns: Int,
        n_decode: Int,
        p50_ns: Int,
        step_lo_ns: Int,
        step_hi_ns: Int,
    ):
        self.text = text
        self.engine_ns = engine_ns
        self.setup_ns = setup_ns
        self.n_out = n_out
        self.prefill_ns = prefill_ns
        self.decode_ns = decode_ns
        self.n_decode = n_decode
        self.p50_ns = p50_ns
        self.step_lo_ns = step_lo_ns
        self.step_hi_ns = step_hi_ns

    def tok_per_s(self) -> Float64:
        """旧口径：含 prefill 的整段时间。只用于演示打印，**判定不认它**。"""
        if self.engine_ns <= 0:
            return Float64(0)
        return Float64(self.n_out) * 1000000000.0 / Float64(self.engine_ns)

    def decode_tok_per_s(self) -> Float64:
        """只算 decode 的均值：prefill 那一次不在分母里。"""
        if self.decode_ns <= 0 or self.n_decode <= 0:
            return Float64(0)
        return Float64(self.n_decode) * 1000000000.0 / Float64(self.decode_ns)

    def p50_tok_per_s(self) -> Float64:
        """主统计量：每 token 时间的**中位数**取倒数。"""
        if self.p50_ns <= 0:
            return Float64(0)
        return 1000000000.0 / Float64(self.p50_ns)

    def print_line(imm self):
        print("   engine : " + String(self.engine_ns // 1000000) + " ms")
        if self.setup_ns > 0:
            print(
                "   setup  : "
                + String(self.setup_ns // 1000000)
                + " ms (量化，每次加载付一次，不含在 tok/s 里)"
            )
        print(
            "   prefill: "
            + String(self.prefill_ns // 1000000)
            + " ms (整段 prompt 一次前向，固定成本)"
        )
        var n = 1
        if self.n_decode > 0:
            n = self.n_decode
        print(
            "   decode : "
            + String(self.decode_ns // 1000000)
            + " ms / "
            + String(self.n_decode)
            + " token = "
            + fixed2(Float64(self.decode_ns) / Float64(n) / 1000000.0)
            + " ms/token (mean) ; "
            + fixed2(Float64(self.p50_ns) / 1000000.0)
            + " ms/token (中位数) ; 单个 token "
            + fixed2(Float64(self.step_lo_ns) / 1000000.0)
            + " … "
            + fixed2(Float64(self.step_hi_ns) / 1000000.0)
            + " ms"
        )
        print(
            "   tok/s  : "
            + String(self.p50_tok_per_s())
            + " (只 decode, 中位数) / "
            + String(self.decode_tok_per_s())
            + " (只 decode, mean) / "
            + String(self.tok_per_s())
            + " (含 prefill, 旧口径)"
        )


def run_slice[backend: Int = BACKEND_SCALAR](
    mut model: QwenForward,
    prompt: String,
    n_new: Int,
    params: SampleParams,
    greedy: Bool,
    verbose: Bool = True,
    setup_ns: Int = 0,
) raises -> SliceRun:
    """跑**一趟**垂直切片，返回该趟的测量（含文本）。

    模型由调用方给进来，不在函数里加载 —— 因为 A/B 要在同一份权重上跑很多趟，
    而加载是个一边界的动作（mmap 2 GB）：留在里面就变成每趟都在量页缓存了。

    `greedy=True` 走 `run`（贪心，确定性基线）；否则走 `run_sampled`，token 由
    sampler 按 `params` 抽。两条路都跑一遍是有意的：只跑前者等于没接 sampler，
    而 sampler 恰恰是接缝里唯一从未被执行过的一段。

    `backend` 是**编译期**参数（`BACKEND_SCALAR` / `BACKEND_AVX2`），不是运行期
    字段 —— 后端是这一趟前向的属性，放在函数上而不是结构体上，是因为参数化结构体
    会崩编译器，也因为权重和 KV 缓存本来就不关心后端。

    `setup_ns` 是调用方付掉的准备时间（加载 / 量化），只用来打印，**不进 tok/s**。
    `verbose=False` 关掉逐趟的六行打印：重复 A/B 一共十几趟，每趟六行会把表淹掉。
    """
    var tok = load_tokenizer(FIXTURE)
    var ids = tok.encode(prompt)
    if len(ids) == 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT, "the prompt tokenized to nothing", prompt
        )
    if len(ids) > MAX_PROMPT:
        raise AlofaError(
            ERR_CAPACITY,
            "the prompt is longer than the engine keeps",
            "n=" + String(len(ids)),
        )
    if n_new <= 0 or n_new > MAX_GEN:
        raise AlofaError(
            ERR_CAPACITY, "max_new is outside what the engine records", ""
        )

    if verbose:
        print("   prompt : " + prompt)
        print("   ids    : " + join_ids(ids))

    # 模型实例是**复用**的（重复 A/B 在同一份权重上跑很多趟），而 KV 缓存是它的
    # 字段：上一趟的历史不清掉，这一趟的 prefill 会带着旧位置一起算，而且
    # `prefill` 自己就要求从空历史开始。
    model.reset()

    var cfg = SchedConfig(ROWS, ROWS, BLOCK, CAP_BLOCKS, WATERMARK, MAX_WAIT)
    var engine = EngineCore(
        cfg,
        HIDDEN,
        INTER,
        KV_DIM,
        N_LAYERS,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        VOCAB,
        1e-6,
        MAX_POS,
        ROWS,
    )

    # prompt 与输出都放在这块 arena 里：引擎收的是**指针**（它要能在抢占后重喂），
    # 而指针指向的内存必须活到整轮生成结束 —— 见文件末尾的 `keep_alive`。
    var arena = Arena(MAX_PROMPT * 8 + MAX_GEN * 8 + 64)
    var toks = int_map(arena.alloc(MAX_PROMPT * 8))
    for i in range(len(ids)):
        toks[unsafe_offset=i] = ids[i]

    engine.submit(1, toks, len(ids), n_new)
    # 计时只包住引擎的 prefill + decode，**不含**权重 mmap 与分词：那两样是固定开销，
    # 混进分母会让 tok/s 随生成长度漂移，也就没法跟 llama.cpp 的 eval time 对齐。
    var t0 = monotonic_ns()
    var ticks = 0
    var prefill_ns = 0
    var decode_ns = 0
    var n_decode = 0
    var steps = List[Int]()
    if greedy:
        # `engine.run` 内部就是 `while has_work: tick`，这里把同一个循环展开写一遍，
        # 唯一的区别是**逐拍计时** —— prefill 与 decode 必须分开记（理由见
        # `SliceRun`）。展开在这里而不是给引擎加计时，是因为 `engine/core.mojo` 有
        # 零分配源码门，而记一串每拍的时间需要一个 `List`。
        while ticks < MAX_TICKS and engine.has_work():
            var t1 = monotonic_ns()
            var rows = engine.tick[backend](model)
            var dt = monotonic_ns() - t1
            ticks += 1
            book_step(prefill_ns, decode_ns, n_decode, steps, rows, dt)
    else:
        var sampler = Sampler(VOCAB, MAX_SEQ_TOKENS)
        var rng = Rng(SEED)
        # 采样循环**内联在这里**，不抽成 `run_sampled(core, ...)`：把 `EngineCore`
        # 当参数传会按值拷贝它，而它带着一块 arena —— 副本析构后副本里的指针就
        # 悬了（实测：第一次访问 `ex.live[i]` 即崩）。本项目里所有对引擎的循环
        # 都是它自己的方法，没有自由函数接收大结构体的先例，与其猜借用语法，
        # 不如让循环留在拥有这块内存的作用域里。
        while ticks < MAX_TICKS and engine.has_work():
            var t1 = monotonic_ns()
            var rows = engine.prepare()
            var chosen = engine.chosen
            for i in range(MAX_BATCH):
                chosen[unsafe_offset=i] = NO_TOKEN
            if rows > 0:
                engine.ex.forward[backend](model)
                for i in range(MAX_BATCH):
                    if engine.ex.live[i] != 1 or engine.ex.served[i] == 0:
                        continue
                    var req = engine.ex.req[i]
                    # 重复惩罚要知道这条请求说过什么，而那些 token 在引擎里：
                    # `submit` 时拷进来的 prompt，加上这轮已经生成的。
                    var slot = engine.find(req)
                    var history = List[Int]()
                    if slot >= 0:
                        for j in range(engine.p_len[slot]):
                            history.append(
                                engine.prompts[unsafe_offset=slot * MAX_PROMPT + j]
                            )
                        for j in range(engine.n_out[slot]):
                            history.append(engine.out[slot * MAX_GEN + j])
                    sampler.build(
                        engine.ex.logits_of(req),
                        engine.ex.vocab,
                        params,
                        List[LogitBias](),
                        history,
                    )
                    chosen[unsafe_offset=i] = sampler.pick(
                        engine.ex.vocab, rng.next_uniform()
                    )
            engine.settle(chosen)
            var dt = monotonic_ns() - t1
            ticks += 1
            book_step(prefill_ns, decode_ns, n_decode, steps, rows, dt)

    var dest = int_map(arena.alloc(MAX_GEN * 8))
    var got = engine.output(1, dest)
    var out = List[Int]()
    for i in range(got):
        out.append(dest[unsafe_offset=i])

    var elapsed_ns = monotonic_ns() - t0
    var text = tok.decode(out)
    # arena 必须活到所有指针都不再用为止：Mojo 的值在**最后一次使用处**析构，
    # 而这里最后一次使用正是上面的读取。
    arena.keep_alive()
    sort_ns(steps)
    var step_lo = 0
    var step_hi = 0
    if len(steps) > 0:
        step_lo = steps[0]
        step_hi = steps[len(steps) - 1]
    var run = SliceRun(
        text,
        elapsed_ns,
        setup_ns,
        got,
        prefill_ns,
        decode_ns,
        n_decode,
        median_ns(steps),
        step_lo,
        step_hi,
    )
    if verbose:
        print("   ticks  : " + String(ticks))
        print("   out ids: " + join_ids(out))
        print("   preempt: " + String(engine.preempt_total()))
        run.print_line()
    return run^


def generate[backend: Int = BACKEND_SCALAR](
    prompt: String,
    n_new: Int,
    params: SampleParams,
    greedy: Bool,
    quantize: Bool = False,
) raises -> SliceRun:
    """加载权重（可选量化）后跑一趟切片。

    `quantize=True` 把 24 层的七个投影槽位压成 q4_0（`lm_head` 除外 —— 见
    `enable_q4`），之后的前向走量化通路。量化这一步的时间单独记在 `setup_ns`，
    **不进 tok/s**：它是"每次加载付一次"，而 tok/s 要量的是每个 token 的成本，
    混在一起结论就会随生成长度漂移。
    """
    var t_setup = monotonic_ns()
    var model = QwenForward(
        FIXTURE + "/weights", FIXTURE + "/config.tsv", ROWS, quantize
    )
    var setup_ns = monotonic_ns() - t_setup
    var run = run_slice[backend](
        model, prompt, n_new, params, greedy, True, setup_ns
    )
    return run^


def sample_min(s: List[Float64]) -> Float64:
    var v = s[0]
    for i in range(1, len(s)):
        if s[i] < v:
            v = s[i]
    return v


def sample_max(s: List[Float64]) -> Float64:
    var v = s[0]
    for i in range(1, len(s)):
        if s[i] > v:
            v = s[i]
    return v


def span_text(s: List[Float64]) -> String:
    return fixed2(sample_min(s)) + " … " + fixed2(sample_max(s))


def verdict(q: List[Float64], f: List[Float64]) -> String:
    """两串 tok/s 的比较：区间重不重合是唯一的判据。

    比值区间取最保守的两端（`q_min/f_max … q_max/f_min`），因为它假设两次测量
    的抖动方向相反 —— 这正是 2026-09-20 那六次里实际发生的事。均值谁大不作数。
    """
    var q_lo = sample_min(q)
    var q_hi = sample_max(q)
    var f_lo = sample_min(f)
    var f_hi = sample_max(f)
    if q_lo <= Float64(0) or f_lo <= Float64(0):
        return "有一档量到 0 tok/s，这一轮不作判定"
    var text = (
        fixed2(q_lo / f_hi) + "× … " + fixed2(q_hi / f_lo) + "×   区间"
    )
    if q_lo > f_hi:
        return text + "不重合 → q4 更快"
    if q_hi < f_lo:
        return text + "不重合 → q4 更慢"
    return text + "重合 → 这份数据区分不出两者（照旧记「量不出结论」）"


def direction(q: List[Float64], f: List[Float64]) -> Int:
    """区间比较的**方向**：+1 = q 更快且区间不重合；-1 = q 更慢且不重合；0 = 重合。

    0 是默认值：量不出来的时候不猜方向。有一档量到 0 tok/s（时钟没走 / 没跑到）
    也算 0 —— 那不是"慢"，那是没量成。
    """
    var q_lo = sample_min(q)
    var q_hi = sample_max(q)
    var f_lo = sample_min(f)
    var f_hi = sample_max(f)
    if q_lo <= Float64(0) or f_lo <= Float64(0):
        return 0
    if q_lo > f_hi:
        return 1
    if q_hi < f_lo:
        return -1
    return 0


def combined(
    q_p50: List[Float64],
    f_p50: List[Float64],
    q_mean: List[Float64],
    f_mean: List[Float64],
) -> String:
    """两个口径（每 token 中位数 / decode 均值）**同向且都排除 1.0** 才给结论。

    默认值是"量不出结论"。一个口径说快、另一个含 1.0 时取好看的那个，等于自己造
    结论 —— 这条就是挡它的：两个口径必须**一致地**把默认值推翻。
    """
    var d_p50 = direction(q_p50, f_p50)
    var d_mean = direction(q_mean, f_mean)
    if d_p50 != 0 and d_p50 == d_mean:
        if d_p50 > 0:
            return "两个口径同向 → q4 更快"
        return "两个口径同向 → q4 更慢"
    if d_p50 != 0 or d_mean != 0:
        return "两个口径不同向 → 量不出结论（不许取好看的那个）"
    return "两个口径都含 1.0 → 量不出结论"


def ratio_list(q: List[Float64], f: List[Float64]) -> List[Float64]:
    """同轮配对：第 i 轮的 q 除以第 i 轮的 f。

    **交错的目的就是让这个比值有意义**：同一轮里四档挨着跑，机器慢下来的时候四档
    一起慢，比值里只剩档与档的差别。而各档自己的 min…max（非配对区间）把"这一轮
    整体快/慢"也算进了抖动 —— 2026-09-20 晚那张表里四档从第 1 轮到第 4 轮一起上升
    约 3–7%，非配对的区间就是被这个共模因子撑宽的。
    """
    var out = List[Float64]()
    for i in range(len(f)):
        if f[i] > Float64(0):
            out.append(q[i] / f[i])
        else:
            out.append(Float64(0))
    return out^


def verdict_ratio(r: List[Float64]) -> String:
    """一串配对比值的判定：整串都在 1.0 之上/之下才算有方向。"""
    var lo = sample_min(r)
    var hi = sample_max(r)
    if lo <= Float64(0):
        return "有一轮没量成，不作判定"
    if lo > Float64(1):
        return fixed2(lo) + "× … " + fixed2(hi) + "×   整串 > 1.0 → q4 更快"
    if hi < Float64(1):
        return fixed2(lo) + "× … " + fixed2(hi) + "×   整串 < 1.0 → q4 更慢"
    return fixed2(lo) + "× … " + fixed2(hi) + "×   整串含 1.0 → 量不出结论"


def arm_name(i: Int) -> String:
    if i == 0:
        return "fp32/scalar"
    if i == 1:
        return "fp32/avx2  "
    if i == 2:
        return "q4_0/scalar"
    return "q4_0/avx2  "


def run_arm(
    arm: Int,
    mut model_f32: QwenForward,
    mut model_q4: QwenForward,
    n_new: Int,
) raises -> SliceRun:
    """跑第 `arm` 档一趟。

    这个 `if` 链省不掉：`backend` 是**编译期**参数，没法用运行期变量去挑，只能每个
    取值各写一遍调用。档位与模型/后端的对应关系只写在这里一处。
    """
    if arm == 0:
        return run_slice[BACKEND_SCALAR](
            model_f32, DEFAULT_PROMPT, n_new, SampleParams(), True, False
        )
    if arm == 1:
        return run_slice[BACKEND_AVX2](
            model_f32, DEFAULT_PROMPT, n_new, SampleParams(), True, False
        )
    if arm == 2:
        return run_slice[BACKEND_SCALAR](
            model_q4, DEFAULT_PROMPT, n_new, SampleParams(), True, False
        )
    return run_slice[BACKEND_AVX2](
        model_q4, DEFAULT_PROMPT, n_new, SampleParams(), True, False
    )


def track_step(mut lo: List[Int], mut hi: List[Int], arm: Int, r: SliceRun):
    """把这一趟的「最快 / 最慢单个 token」并进该档的全轮区间。"""
    if r.step_lo_ns <= 0:
        return
    if lo[arm] == 0 or r.step_lo_ns < lo[arm]:
        lo[arm] = r.step_lo_ns
    if hi[arm] == 0 or r.step_hi_ns > hi[arm]:
        hi[arm] = r.step_hi_ns


def pad_left(s: String, width: Int) -> String:
    """右对齐到 `width` 列。表要能扫，列就得对齐；只用于 ASCII 数字。"""
    var out = s
    while out.byte_length() < width:
        out = " " + out
    return out


def proc_counts() raises -> List[Int]:
    """/proc/self/stat 的四个计数：minor faults / major faults / utime / stime。

    为什么是这四个
    --------------
    fp32 档的权重是**文件映射**（`MappedFile`，1.976 GB），q4 档的权重在**匿名**
    内存里（`q4_arena`，约 0.36 GB）。文件页是可以被内核回收的干净页，回收之后再
    访问就要重新缺页：还在页缓存里是 minor，已经掉出去就得读盘（major）。匿名页
    没有这回事 —— 它只有被换出去才会 major fault，而本机 swap 基本没动。

    所以这四个数是"抖动是不是缺页造成的"的判据：

    - **Δmajflt > 0 且集中在 fp32 档** → 文件页被回收、回落时读了盘，机制找到了。
    - **Δmajflt = 0 而 Δminflt 很大** → 只是重新建立映射（页缓存还在），代价小得多。
    - **两者都平** → 抖动不在页这一层，得往带宽争抢那边找。

    `stime`（内核态时间）是第二个独立信号：缺页、回收、建页表都花在内核态，所以
    掉速那一档如果 `stime` 也一起涨，就是内核在替它干活。

    ⚠️ `utime`/`stime` 的单位是**时钟滴答**，这里按 `CLK_TCK=100` 折成 ms（本机
    如此）。这个换算只影响"多少 ms"，不影响"有没有"。
    """
    var text = read_text(PROC_STAT_PATH)
    # `comm` 在括号里而且**可能含空格**，所以从**最后一个** `)` 之后开始切字段：
    # 这样就不必假设进程名里没有空格。本进程名是 `cli`，但这条不该依赖它。
    var tail = ""
    for part in text.split(")"):
        tail = String(part)
    var toks = List[String]()
    for t in tail.split(" "):
        var word = String(t)
        if word.byte_length() > 0:
            toks.append(word)
    # 切完第 0 个是 `state`（第 3 个字段），所以第 N 个字段的下标是 N − 3。
    if len(toks) < 13:
        raise AlofaError(
            ERR_IO,
            "unexpected layout in /proc/self/stat",
            "tokens=" + String(len(toks)) + " path=" + PROC_STAT_PATH,
        )
    var out = List[Int]()
    out.append(parse_int(toks[7]))  # 10: minflt
    out.append(parse_int(toks[9]))  # 12: majflt
    out.append(parse_int(toks[11]))  # 14: utime
    out.append(parse_int(toks[12]))  # 15: stime
    return out^


def runq_len() raises -> Int:
    """/proc/loadavg 第 4 个字段里 `/` 左边的数：此刻**可运行**的进程数。

    这是**邻居**的信号，不是本进程的：本机 loadavg 常年 10–12（8 核），VS Code 的
    node 进程各占 20–30% CPU。它证明不了谁抢走了什么，但掉速那一档如果同时 runq
    高，至少说明"那段时间机器上还有别人"；反过来，如果掉速与 runq 无关，就该往
    别的机制上找。
    """
    var text = read_text(PROC_LOADAVG_PATH)
    var toks = List[String]()
    for t in text.split(" "):
        var word = String(t)
        if word.byte_length() > 0:
            toks.append(word)
    if len(toks) < 4:
        raise AlofaError(
            ERR_IO, "unexpected layout in /proc/loadavg", "text=" + text
        )
    for part in String(toks[3]).split("/"):
        return parse_int(String(part))
    raise AlofaError(ERR_IO, "no run queue in /proc/loadavg", "text=" + text)


def gb_text(bytes_per_token: Float64, s: List[Float64]) -> String:
    """把一串 tok/s 换成"每 token 流了多少 GB/s"以及占读带宽天花板的百分比。"""
    var lo = bytes_per_token * sample_min(s) / 1000000000.0
    var hi = bytes_per_token * sample_max(s) / 1000000000.0
    return (
        fixed2(lo)
        + " … "
        + fixed2(hi)
        + " GB/s = 天花板的 "
        + String(Int(lo * 100.0 / READ_BW_GB_S))
        + "% … "
        + String(Int(hi * 100.0 / READ_BW_GB_S))
        + "%"
    )


def bench_ab(reps: Int) raises:
    """重复加交错的四档 A/B：本文件里唯一可以用来回答「该不该开量化」的表。

    三条纪律，缺一条这张表就只是噪声：

    1. **同一进程**：两份权重各加载一次，之后十几趟都跑在它们上面。跨进程比
       tok/s 是在比页缓存和邻居负载。
    2. **重复取区间**：四个 arm 各跑一遍时 `fp32/avx2` 给过 1.63–4.52 tok/s、
       比值 0.61–2.16×（**方向都翻过**）；改成同进程交错重复后收到 3.64–4.53 与
       0.98–1.29×。差的那部分就是这条纪律要付的钱，单次是碰运气。
    3. **交错 + 转顺序**：四个变体在同一轮里各跑一遍，且第 `rep` 轮从第 `rep`
       档起跑（`arm = (rep + k) % N_ARMS`）。顺序固定的话"谁排在前面"就变成
       "谁更快"，而那不是通路的属性。
    4. **prefill 与 decode 分开记**：prefill 是整段 prompt 的**一次**前向，是固定
       成本；decode 才是每 token 的成本。混在一起，tok/s 会随 `n_new` 漂移，而且
       那一次性的开销会把真正要测的每 token 差别稀释掉（16 个 token 时它占约
       1/17）。所以这里的样本是**只 decode** 的数。
    5. **主统计量是每 token 时间的中位数**，不是均值：一个被 OS 打断的 token 能把
       16 个样本的均值推走 6%，推不动中位数。表同时给均值口径作对照。
    6. **预热轮不计入样本**：权重是 mmap 的，第一次读到哪一页就在那儿付一次缺页，
       而**一个 decode step 会把全部权重读一遍** —— 所以两个 token 就够把每一页
       摸过。缺页量的是内核的页管理，不是被测的 kernel。

    ⚠️ **加轮数不是让结论变好看的旋钮**：区间取 min…max，轮数越多区间只会越宽
    （min 更低、max 更高），判定只会更保守。要分开区间只能**降抖动**，也就是上面
    的第 4/5/6 条。

    判定：**主口径是同轮配对比值**（见 `ratio_list`）—— 交错就是为了让它成立；
    各档自己的非配对区间作为**保守对照**保留，但它会被"这一轮整体快慢"这个共模
    因子撑宽，所以它更保守、不是更准。两个口径都要打印，谁都不能单独拿走结论 ——
    先定规矩再看数，是为了挡住"口径挑选"。
    """
    if reps <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT, "the A/B needs at least one round", ""
        )
    print("")
    print(
        "== 重复交错 A/B（"
        + String(reps)
        + " 轮 × 四档，同一进程、同一份权重、同一个 prompt） =="
    )
    var t_load = monotonic_ns()
    var model_f32 = QwenForward(FIXTURE + "/weights", FIXTURE + "/config.tsv", ROWS)
    var t_quant = monotonic_ns()
    var model_q4 = QwenForward(
        FIXTURE + "/weights", FIXTURE + "/config.tsv", ROWS, True
    )
    var t_done = monotonic_ns()
    # mmap 是**惰性**的：这个数常常是 0 ms，它不是加载成本，只是"登记了一次映射"
    # —— 真正的读页落在下面那个**预热轮**里，而预热轮不计入样本。（2026-09-20
    # 上午的那张表让第 1 轮替 mmap 付了这笔钱，那也是它的四档各跑一遍不可信的
    # 第二个理由。）
    print(
        "   权重加载 fp32 : "
        + String((t_quant - t_load) // 1000000)
        + " ms (mmap，惰性；真正的读页在第 1 轮里)"
    )
    print(
        "   量化 q4_0     : "
        + String((t_done - t_quant) // 1000000)
        + " ms ← 每次加载付一次，不含在 tok/s 里"
    )

    # 预热轮：先把 mmap 的页摸一遍，结果**丢弃**。
    print("   预热轮（" + String(BENCH_WARMUP_NEW) + " 新 token，结果丢弃）")
    for k in range(N_ARMS):
        _ = run_arm(k, model_f32, model_q4, BENCH_WARMUP_NEW)

    # 八串样本：下标 = 轮次。每轮给每串各追加一个值，所以 `s0[i]` 就是第 i 轮的
    # `fp32/scalar` —— 轮换只改**跑的顺序**，不改写入的串。
    # `s*` 是主口径（每 token 中位数），`m*` 是同一批样本的 decode 均值：两个口径
    # 必须同向才算有结论（见 `combined`）—— 这是挡"口径挑选"的。
    var s0 = List[Float64]()
    var s1 = List[Float64]()
    var s2 = List[Float64]()
    var s3 = List[Float64]()
    var m0 = List[Float64]()
    var m1 = List[Float64]()
    var m2 = List[Float64]()
    var m3 = List[Float64]()
    # 每档「全轮最快 / 最慢的单个 token」：用来分辨抖动是**偶发的慢 token**（两端
    # 拉得很开而中位数稳）还是**整轮一起慢**（中位数自己在抖）。
    var lo_ns = List[Int]()
    var hi_ns = List[Int]()
    for i in range(N_ARMS):
        lo_ns.append(0)
        hi_ns.append(0)

    # 内核侧证据，下标 `rep * N_ARMS + arm`。读 `/proc` 的代价是每档两次小读，
    # 相对每档几十秒的解码可以忽略；它换来的是"掉速那一档到底发生了什么"。
    var d_min = List[Int]()
    var d_maj = List[Int]()
    var d_sys = List[Int]()
    var runq = List[Int]()
    for i in range(reps * N_ARMS):
        d_min.append(0)
        d_maj.append(0)
        d_sys.append(0)
        runq.append(0)

    for rep in range(reps):
        print("   -- 第 " + String(rep + 1) + " 轮")
        for k in range(N_ARMS):
            var arm = (rep + k) % N_ARMS
            var c0 = proc_counts()
            var q0 = runq_len()
            var r = run_arm(arm, model_f32, model_q4, BENCH_NEW)
            var c1 = proc_counts()
            var q1 = runq_len()
            var slot = rep * N_ARMS + arm
            d_min[slot] = c1[0] - c0[0]
            d_maj[slot] = c1[1] - c0[1]
            d_sys[slot] = (c1[3] - c0[3]) * 10
            runq[slot] = q0
            if q1 > q0:
                runq[slot] = q1
            if arm == 0:
                s0.append(r.p50_tok_per_s())
                m0.append(r.decode_tok_per_s())
            elif arm == 1:
                s1.append(r.p50_tok_per_s())
                m1.append(r.decode_tok_per_s())
            elif arm == 2:
                s2.append(r.p50_tok_per_s())
                m2.append(r.decode_tok_per_s())
            else:
                s3.append(r.p50_tok_per_s())
                m3.append(r.decode_tok_per_s())
            track_step(lo_ns, hi_ns, arm, r)

    print("")
    print(
        "   每轮 tok/s（"
        + String(BENCH_NEW)
        + " 新 token，**只 decode**，每 token 时间的中位数）："
    )
    print("   轮    fp32/scalar   fp32/avx2     q4_0/scalar   q4_0/avx2")
    for i in range(reps):
        print(
            "   "
            + String(i + 1)
            + "     "
            + fixed2(s0[i])
            + "          "
            + fixed2(s1[i])
            + "         "
            + fixed2(s2[i])
            + "        "
            + fixed2(s3[i])
        )
    print("")
    print("   区间 min … max（tok/s，中位数口径 / 均值口径）：")
    print("   fp32/scalar : " + span_text(s0) + "   /   " + span_text(m0))
    print("   fp32/avx2   : " + span_text(s1) + "   /   " + span_text(m1))
    print("   q4_0/scalar : " + span_text(s2) + "   /   " + span_text(m2))
    print("   q4_0/avx2   : " + span_text(s3) + "   /   " + span_text(m3))
    print("")
    print("   单个 token 的时间（全轮合并，ms）：")
    for i in range(N_ARMS):
        print(
            "   "
            + arm_name(i)
            + " : "
            + fixed2(Float64(lo_ns[i]) / 1000000.0)
            + " … "
            + fixed2(Float64(hi_ns[i]) / 1000000.0)
        )
    print("")
    print(
        "   内核侧证据（每档：Δmajor 缺页 / Δminor 缺页 / 内核态时间 / 邻居 runq）："
    )
    print("   轮  档            Δmaj    Δmin      Δstime   runq")
    for rep in range(reps):
        for k in range(N_ARMS):
            var slot = rep * N_ARMS + k
            print(
                "   "
                + String(rep + 1)
                + "   "
                + arm_name(k)
                + " : "
                + pad_left(String(d_maj[slot]), 6)
                + "  "
                + pad_left(String(d_min[slot]), 7)
                + "  "
                + pad_left(String(d_sys[slot]) + " ms", 9)
                + "  "
                + pad_left(String(runq[slot]), 5)
            )
    print("")
    print(
        "   推算的每 token 流量（字节/token ÷ 每 token 时间 = GB/s；"
        "天花板 = 本机读带宽 "
        + fixed2(READ_BW_GB_S)
        + " GB/s）："
    )
    print("   fp32/scalar : " + gb_text(BYTES_PER_TOKEN_FP32, s0))
    print("   fp32/avx2   : " + gb_text(BYTES_PER_TOKEN_FP32, s1))
    print("   q4_0/scalar : " + gb_text(BYTES_PER_TOKEN_Q4, s2))
    print("   q4_0/avx2   : " + gb_text(BYTES_PER_TOKEN_Q4, s3))
    print(
        "   （fp32 档每 token 流 1.976 GB；q4 档 0.746 GB = 24 层 0.201 GB + "
        "未量化的 `lm_head` 0.544 GB，后者自己就占 73%）"
    )
    print("")
    print(
        "   配对比值（同轮内 q4 ÷ fp32：消掉「这一轮整体快慢」这个共模因子）—— 主口径"
    )
    var pr_s = ratio_list(s2, s0)
    var pr_v = ratio_list(s3, s1)
    for i in range(len(pr_s)):
        print(
            "   第 "
            + String(i + 1)
            + " 轮  标量档 "
            + fixed2(pr_s[i])
            + "×   avx2 档 "
            + fixed2(pr_v[i])
            + "×"
        )
    print("   标量档 : " + verdict_ratio(pr_s))
    print("   avx2 档: " + verdict_ratio(pr_v) + "  ← 这条才是「该不该开量化」的答案")
    print("")
    print("   非配对区间（各档自己的 min…max）—— 保守对照，被上面的共模漂移撑宽了：")
    print("   q4 / fp32（同标量档）")
    print("       中位数口径 : " + verdict(s2, s0))
    print("       均值口径   : " + verdict(m2, m0))
    print("       → " + combined(s2, s0, m2, m0))
    print("   q4 / fp32（同 avx2 档）← 这条才是「该不该开量化」的答案")
    print("       中位数口径 : " + verdict(s3, s1))
    print("       均值口径   : " + verdict(m3, m1))
    print("       → " + combined(s3, s1, m3, m1))


def main() raises:
    print("alofa — vertical slice (tokenizer → model → engine → sampler)")
    # 两个后端各跑一遍贪心：同一 prompt 下二者文本必须逐字相同，那是 AVX2 档能拿
    # 来当基线的资格线 —— 向量算子与标量算子一旦语义分叉，它只是个跑得快点的错答案。
    print("[greedy fp32] " + backend_label[BACKEND_SCALAR]())
    var greedy = generate[BACKEND_SCALAR](
        DEFAULT_PROMPT, DEFAULT_NEW, SampleParams(), True
    )
    print("   text   : " + greedy.text)
    print("[greedy fp32] " + backend_label[BACKEND_AVX2]())
    var greedy_vec = generate[BACKEND_AVX2](
        DEFAULT_PROMPT, DEFAULT_NEW, SampleParams(), True
    )
    print("   text   : " + greedy_vec.text)
    print("[sampled fp32] seed=" + String(Int(SEED)))
    var sampled = generate[BACKEND_SCALAR](
        DEFAULT_PROMPT, DEFAULT_NEW, SampleParams(), False
    )
    print("   text   : " + sampled.text)
    # 量化通路**两个后端都跑**：2026-09-20 起量化核（`matmul_q4_f32`）自己也有向量
    # 实现了（见 `q4_matmul_k`），所以 "AVX2 + q4" 不再是混跑。这一档更不能省：
    # 只报 q4/scalar 会得出"量化快 1.5×"的结论，而真正要回答的是它在今天最快的
    # 那条路（fp32/avx2）面前值不值得开 —— 少跑这一档，那句 1.5× 就是挑出来的数字。
    print("[greedy q4_0] " + backend_label[BACKEND_SCALAR]())
    var greedy_q4 = generate[BACKEND_SCALAR](
        DEFAULT_PROMPT, DEFAULT_NEW, SampleParams(), True, True
    )
    print("   text   : " + greedy_q4.text)
    print("[greedy q4_0] " + backend_label[BACKEND_AVX2]())
    var greedy_q4_vec = generate[BACKEND_AVX2](
        DEFAULT_PROMPT, DEFAULT_NEW, SampleParams(), True, True
    )
    print("   text   : " + greedy_q4_vec.text)

    # 这里只报**绝对值**，不报比值：四档各跑了一遍，而本机同日绝对 tok/s 抖 2.8×
    # （`fp32/avx2` 1.63–4.52），2026-09-20 用这四档算出来的 `q4/fp32（avx2 档）`
    # 六次给过 0.61 / 1.01 / 1.59 / 1.72 / 1.89 / 2.16× —— **方向都翻过**。
    # 把单次比值印在屏幕上就是给人挑数字，所以比值只由下面那张重复交错的表给出。
    print("")
    print("== 同进程 A/B（单次，仅作参考；判定见下面那张表） ==")
    print(
        "   fp32/scalar  : "
        + fixed2(greedy.p50_tok_per_s())
        + " tok/s (只 decode, 中位数)"
    )
    print(
        "   fp32/avx2    : "
        + fixed2(greedy_vec.p50_tok_per_s())
        + " tok/s (只 decode, 中位数)"
    )
    print(
        "   q4_0/scalar  : "
        + fixed2(greedy_q4.p50_tok_per_s())
        + " tok/s (只 decode, 中位数)"
    )
    print(
        "   q4_0/avx2    : "
        + fixed2(greedy_q4_vec.p50_tok_per_s())
        + " tok/s (只 decode, 中位数)"
    )
    # 这一行是**报告**，不是门：量化会改数值，文本不同是预期的
    # （`test_q4_greedy` 量过：教师强制贪心一致率 0.80）。把它打出来是为了让
    # "快了多少"和"歪了多少"永远出现在同一个视野里 —— 只看前者会得出假结论。
    print("   文本一致     : " + String(greedy.text == greedy_q4.text))

    bench_ab(BENCH_REPS)
