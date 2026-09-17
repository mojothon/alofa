"""权重不能多占一份：加载并生成之后，峰值 RSS 必须在权重 footprints 的 1.15 倍内。

这个门能证明什么、不能证明什么（必须照实说）
--------------------------------------------

**能证明**：权重没有被多留一份。拷贝、转置后留着原副本、或者把整网权重常驻
解量化成 fp32 —— 这些都会让常驻量奔向 2×，而这个门在 1.15× 就红。

**不能证明**：权重是零拷贝的。mmap 之后触碰过的文件页算进 RSS，`read()` 到
堆上同样算进 RSS —— 两者的常驻量都是约 1× 权重，**RSS 这个量本身区分不了
它们**。所以别拿这个门去宣称"证明了 mmap 无拷贝"；那条结论需要别的证据
（比如比较缺页次数或映射区间），本轮没有做。

这个区分不是文字游戏。一个门的价值等于它真正能拦住的错误，把"能拦住的"说
成"已经证明的"，就会让后来者以为零拷贝这条已经被看过了 —— 而这恰恰是本账本
要防的那种含糊。

**为什么用峰值而不是此刻的值**：内核可以随时回收干净的文件页，于是"此刻常驻"
会往下走，同一个程序量两次可能第二次更小。若用它判上界，门就变成看内核心情。
`VmHWM` 只涨不落，记的是最紧张的那个瞬间，那才是内存门该问的问题。

**负向对照**：同一进程里再拷一份同样大小的权重，门必须变红。一个从没被验证
过会红的门，绿灯不说明任何事。

需要 `tests/fixtures/qwen2.5-0.5b/weights/`（约 1.9 GB 本地导出），因此
**故意不进 `pixi run test`**：

    pixi run test-memory
"""

from std.testing import TestSuite, assert_true

from alofa.core.error import ERR_IO, AlofaError
from alofa.core.ffi.mem import mlock
from alofa.core.memory import Arena
from alofa.core.text import parse_int, read_text
from alofa.model.arch.qwen import QwenForward
from alofa.verify.rss import peak_rss_bytes, rss_bytes

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights/"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime PROMPTS = "tests/fixtures/qwen2.5-0.5b/prompts.tsv"

# 门：峰值 RSS ≤ 1.15 × 权重字节。
# 1.15 是留给"运行时本身 + 激活 + KV"的余量：Mojo 运行时与这套测试的静态
# 开销是几十 MB 量级，激活与 KV 在 64 token 上下文下也是 MB 量级，相对 1.9 GB
# 的权重都远不到 15%。真正要抓的是 2× —— 多留一份直接翻倍，15% 的余量足以
# 把它挡住，又不会因为几十 MB 的常客开销而误报。
comptime GATE_NUM = 115
comptime GATE_DEN = 100

comptime GEN_TOKENS = 32
comptime MAX_TOKENS = 64


def lines_of(path: String) raises AlofaError -> List[String]:
    var text = read_text(path)
    var out = List[String]()
    for span in text.split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        out.append(line)
    return out^


def weight_bytes() raises AlofaError -> Int:
    """权重实际占的字节数：按**字节偏移去重**后的总和。

    去重要紧。绑定权重（`tie_word_embeddings`）在索引里是两条**同偏移**的别名
    行（`lm_head.weight` 与 `model.embed_tokens.weight`），不去重就会把 embedding
    那 544 MB 数两遍 —— 分母凭空大 27%，1.15× 的门实际成了 1.47×，门就松了。
    """
    var total = 0
    var seen = List[Int]()
    for line in lines_of(WEIGHTS + "tensors.tsv"):
        var fields = List[String]()
        for span in line.split("\t"):
            fields.append(String(span))
        var offset = parse_int(fields[2])
        var duplicate = False
        for i in range(len(seen)):
            if seen[i] == offset:
                duplicate = True
                break
        if duplicate:
            continue
        seen.append(offset)
        total += parse_int(fields[3]) * 4
    if total <= 0:
        raise AlofaError(ERR_IO, "权重索引是空的", WEIGHTS + "tensors.tsv")
    return total


def prompt_ids() raises AlofaError -> List[Int]:
    """第一条 prompt 的 token id。"""
    var first = lines_of(PROMPTS)[0]
    var fields = List[String]()
    for span in first.split("\t"):
        fields.append(String(span))
    var ids = List[Int]()
    for span in fields[1].split(","):
        var piece = String(span)
        if piece.byte_length() == 0:
            continue
        ids.append(parse_int(piece))
    if len(ids) == 0:
        raise AlofaError(ERR_IO, "prompt 是空的", PROMPTS)
    return ids^


def test_peak_rss_stays_within_the_weight_footprint() raises:
    """加载权重 + 生成 32 个 token 之后的峰值 RSS 不得超过 1.15 × 权重。"""
    var wbytes = weight_bytes()
    var budget = wbytes * GATE_NUM / GATE_DEN

    var forward = QwenForward(WEIGHTS, CONFIG, MAX_TOKENS)
    var ids = prompt_ids()
    var logits = forward.prefill(ids)
    var next_token = forward.argmax(logits)
    for _ in range(GEN_TOKENS - 1):
        var step_logits = forward.step(next_token)
        next_token = forward.argmax(step_logits)

    # ⚠️ 寿命问题，不是仪式。Mojo 在**最后一次使用处**析构值：生成循环一结束，
    # `forward` 就到寿命，`MappedFile` 会立刻 `munmap`，权重那 1.9 GB 当场从
    # 地址空间消失（本轮实测：此刻 RSS 从 1.99 GB 掉到 10 MB）。于是"多留一份"
    # 的负向对照加进去峰值纹丝不动 —— 它只是把已经空掉的坑重新填满。**是这个
    # 对照抓出了测量本身的 bug**，主断言从头到尾都是绿的；这正是对照的意义。
    #
    # 而 `keep_alive()` 是**空函数**，内联之后可能被当成"没有使用"，光靠它钉
    # 不住（本轮实测：只加它，映射照样没了）。所以末尾另有一次**真正的使用**
    # —— 打印 `forward` 的一个字段 —— 把寿命真正撑到测量之后。
    forward.keep_alive()

    forward.keep_alive()

    var peak = peak_rss_bytes()
    print(
        "观测：权重 "
        + String(wbytes)
        + " 字节，加载并生成 "
        + String(GEN_TOKENS)
        + " token 后此刻 RSS "
        + String(rss_bytes())
        + " 峰值 RSS "
        + String(peak)
        + " 字节，倍率 "
        + String(Float64(peak) / Float64(wbytes))
    )
    assert_true(
        peak <= budget,
        "峰值 RSS "
        + String(peak)
        + " 超过预算 "
        + String(budget)
        + "（权重 "
        + String(wbytes)
        + "，倍率 "
        + String(Float64(peak) / Float64(wbytes))
        + "）—— 权重很可能被多留了一份",
    )
    # 光有上界不够，还得确认权重**真的在内存里**：若映射在量之前就没了，此刻
    # 常驻量只剩十几 MB，1.15× 的上界照样绿，那种绿什么都没说明。
    assert_true(
        rss_bytes() >= wbytes,
        "此刻 RSS 只有 "
        + String(rss_bytes())
        + "，小于权重 "
        + String(wbytes)
        + " —— 量的时候权重已经不在内存里了，上面那个上界是白过的",
    )
    # 反向的绊线：门通过了，还得确认它是**因为权重真的常驻才通过的**。若加载
    # 变成惰性的、压根没触碰权重，常驻量会很小，1.15× 的门照样绿 —— 那种绿
    # 什么都没说明，所以这里要求至少权重的量级确实进来了。
    assert_true(
        peak >= wbytes,
        "峰值 RSS 只有 "
        + String(peak)
        + "，小于权重 "
        + String(wbytes)
        + " —— 权重根本没进内存，上面那个门是白过的",
    )

    # 负向对照：真的再留一份，门必须红。
    #
    # **用 `mlock` 而不是"写一遍"**。手写循环会被编译器整个优化掉（store-to-load
    # 转发：写一个字节再读回来，编译器直接把读替换成刚写的值，一片页都没真正
    # 触碰）；换成 `memset` 也照样没了 —— 它是 LLVM 认识的固有函数，后面跟着
    # 释放时会被当成死存储删掉。两种写法本轮都实测过：峰值 RSS 一模一样，对
    # 照看着做了，其实没做。
    #
    # `mlock` 是编译器不认识语义的外部调用，删不掉；而且它不只是"写"，它让
    # 内核把整段**锁进物理内存**，于是这一份必须常驻 —— 这正是"多留了一份"
    # 该有的效果。
    var arena = Arena(wbytes + 4096)
    var duplicate = arena.alloc(wbytes)
    if mlock(duplicate, wbytes) != 0:
        raise AlofaError(
            ERR_IO,
            "负向对照没能锁住那份额外的内存，做不出来就不许说验证过",
            "mlock failed bytes=" + String(wbytes),
        )
    var peak_after = peak_rss_bytes()
    print(
        "对照：多锁一份 "
        + String(wbytes)
        + " 字节后峰值 RSS "
        + String(peak_after)
        + " 字节（预算 "
        + String(budget)
        + "）"
    )
    assert_true(
        peak_after > budget,
        "多留了一份权重后峰值 RSS 仍只有 "
        + String(peak_after)
        + "（预算 "
        + String(budget)
        + "）—— 这个门抓不住多出来的那一份，等于没有",
    )
    # 不止要"变红"，还要"红得对"：多出来的必须是**整整一份**，而不是碰巧
    # 越过了 1.15 的线。只断言超标的话，一个漏掉一半的观测也能混过去。
    var want_two = wbytes * 2 * 95 / 100
    assert_true(
        peak_after >= want_two,
        "多留一份之后峰值 RSS 只有 "
        + String(peak_after)
        + "，离两份（约 "
        + String(want_two)
        + "）还差得远 —— 权重的那一份在量的时候已经不在了",
    )
    # 见上面关于寿命的注释：这两次读取是 `forward` 与 `arena` 的**最后一次真
    # 正使用**，必须发生在所有测量之后，否则两个对象会在最后一个被测点之前
    # 就被析构，量到的数就成了空坑里的数。
    print(
        "收尾：模型仍在（层数 "
        + String(forward.cfg.n_layers)
        + "），对照副本首字节 "
        + String(Int(duplicate[0]))
    )
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
