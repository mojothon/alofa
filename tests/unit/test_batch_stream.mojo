"""批调度 vs 单流：同一条 prompt，两条路必须**逐 token 相等**（贪心与采样都要）。

批调度那笔收益（一条 engine 线程同时推进多条流）只有在这道门绿了之后才算拿到
—— 因为"快"和"算错"在服务端日志里是**一样**的：两条路都不报错，只是其中一条给
出的 token 与另一条不同。而"逐 token 相等"是唯一能把它钉住的判据：一批 N 条一起
前向，与这 N 条各自单跑，必须给出同一个 token 序列。

判据写成**相等**而不是"接近"：这里比的是 token id，是整数 —— 一个 token 不同就是
不同，没有"差不多"的余地。logits 才需要容差，而 logits 差一点点会在自回归里滚成
完全不同的答案，拿 logits 当判据等于没判。

两条路：批路走 `EngineCore`（一次前向推进整批），老路走 `model.prefill` / `step`
（一次一条）。开关是 `ModelService.batch_enabled`（默认开，`ALOFA_BATCH=0` 关）—
— 它存在的理由就是这道门：**同一条 prompt 要能分别走两条路**才能对起来，没它
的话两条路只能各自跑，差异会被"反正 prompt 不一样"永远盖住。

采样那条路**额外**要验的是随机源：批路是多条流交错推进的，若它们共用一个随机
源，那么"这条流这一步抽到什么"就取决于**别人**问了几步 —— 于是同一条 prompt 在
批里与单跑会给出不同答案，而"批调度只是换了执行顺序"这件事就不成立了。所以采
样那两条一条比"批 vs 单跑"，一条比"批里四条**同 prompt** 的流是否互相一致"（共
享随机源时它们必然分叉）。

怎么跑（重资产：要一份真权重，所以 build 之后跑，不进 `pixi run test`）：

    pixi run mojo build -O2 -I src tests/unit/test_batch_stream.mojo -o target/batch_stream
    ALOFA_WEIGHTS=tests/fixtures/qwen2.5-0.5b-hf ./target/batch_stream

⚠️ 门**必须**同时守住"两条路真的是两条路"（最后那条对照）：如果批路静默退回了
老路，上面那条相等就是同一条路和自己比 —— 门会绿，但一个字节也没验到。
"""

from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from alofa.engine.executor import MAX_GEN
from alofa.serve import ModelService
from alofa.srv.config import ServeConfig

comptime PROMPT_A = "The capital of France is"
comptime PROMPT_B = "中国的首都是"
comptime PROMPT_C = "2 + 2 ="
comptime PROMPT_D = "Once upon a time"
# ≤ `MAX_GEN`(32)：引擎一次只收这么多新 token，越界的请求会退回老路（那时这道门
# 比的就不是批路了）。
comptime STEPS = 8
# 采样那条路的温度：> 0 才走采样器（`temperature <= 0` 是贪心分支）。
comptime TEMP = 0.7


def _prompts() -> List[String]:
    var ps = List[String]()
    ps.append(PROMPT_A)
    ps.append(PROMPT_B)
    ps.append(PROMPT_C)
    ps.append(PROMPT_D)
    return ps^


def _solo_run(mut svc: ModelService, prompt: String, temp: Float64) raises -> List[Int]:
    """老路：一条一条跑（`batch_enabled` 由调用方设成 False），返回 token 序列。"""
    svc.stream_begin(prompt, STEPS, temp, 0)
    var guard = 0
    while guard < 512:
        guard += 1
        if svc.stream_next(0).done:
            break
    var out = List[Int]()
    for i in range(len(svc.stream_out)):
        out.append(svc.stream_out[i])
    svc.stream_end(0)
    return out^


def _batch_run(
    mut svc: ModelService, ps: List[String], temp: Float64
) raises -> List[Int]:
    """批路：`len(ps)` 条同时 submit 再交错推进；返回扁平序列（第 i 条从
    `i * STEPS` 起，不足的位填 `-1`）。

    交错推进是**故意**的：服务里就是这么走的（每条连接一帧一往返），所以门要照
    着服务的节奏问，而不是一次把一条问完再问下一条 —— 交错正是"共享随机源"会露
    馅的地方。
    """
    for i in range(len(ps)):
        svc.stream_begin(ps[i], STEPS, temp, i)
    var out = List[Int]()
    for i in range(len(ps) * STEPS):
        out.append(-1)
    var finished = List[Int]()
    for i in range(len(ps)):
        finished.append(0)
    var remaining = len(ps)
    var guard = 0
    while remaining > 0 and guard < 4096:
        guard += 1
        for i in range(len(ps)):
            if finished[i] == 1:
                continue
            if not svc.stream_next(i).done:
                continue
            # ⚠️ 一条流走完就**立刻**把槽位还回去（真实服务里 `ChatHandler` 也是
            # 这么做的）：槽位不释放，排在后面的请求永远进不来 —— 门会一直空转到
            # `guard` 用尽，看起来像"排队没生效"，其实是门自己把槽位攥着。
            var slot = svc._slot_of(i)
            var n = svc.b_n[slot]
            for k in range(STEPS):
                if k < n:
                    out[i * STEPS + k] = svc.b_out[slot * MAX_GEN + k]
            svc.stream_end(i)
            finished[i] = 1
            remaining -= 1
    return out^


def _solo_all(
    mut svc: ModelService, ps: List[String], temp: Float64
) raises -> List[Int]:
    """同一批 prompt 在**老路**上一条一条跑（扁平化，与 `_batch_run` 同形状）。"""
    var out = List[Int]()
    for i in range(len(ps)):
        var seq = _solo_run(svc, ps[i], temp)
        for k in range(STEPS):
            if k < len(seq):
                out.append(seq[k])
            else:
                out.append(-1)
    return out^


def _compare(mut batched: List[Int], mut solo: List[Int], what: String) raises:
    """逐 token 比 —— 一处不同就说清**第几条的第几个**。"""
    for i in range(len(batched)):
        assert_equal(
            batched[i],
            solo[i],
            what
            + " token differs: stream="
            + String(i // STEPS)
            + " step="
            + String(i % STEPS)
            + " batch="
            + String(batched[i])
            + " solo="
            + String(solo[i]),
        )


def test_a_batch_of_four_agrees_with_four_solo_runs() raises:
    """四条不同 prompt 一起批跑（贪心），必须等于它们各自单跑 —— **逐 token**。

    这四条是不同长度、不同语言的 prompt（英文 / 中文 / 数字 / 散文），所以批路
    里的 prefill 是**真的**凑成了一批（而不是四条恰好一样长）。
    """
    var svc = ModelService.load(ServeConfig.from_env())
    var ps = _prompts()
    svc.batch_enabled = True
    var batched = _batch_run(svc, ps, 0.0)
    svc.batch_enabled = False
    var solo = _solo_all(svc, ps, 0.0)
    _compare(batched, solo, "greedy batch=4")


def test_a_batch_of_one_agrees_with_one_solo_run() raises:
    """批里**只有一条**时也要相等（贪心）—— 最小形状。

    它和上面那条不是重复：批=1 是"批路本身对不对"（没有批间的相互影响可怪），
    批=4 才是"合批有没有改变结果"。分开跑的话，一旦红，能直接看出是哪一个。
    """
    var svc = ModelService.load(ServeConfig.from_env())
    var ps = List[String]()
    ps.append(PROMPT_B)
    svc.batch_enabled = True
    var batched = _batch_run(svc, ps, 0.0)
    svc.batch_enabled = False
    var solo = _solo_all(svc, ps, 0.0)
    _compare(batched, solo, "greedy batch=1")


def test_a_sampling_batch_agrees_with_solo_runs() raises:
    """采样（`temperature` > 0）也要：批 4 条 == 各自单跑，**逐 token**。

    这条比贪心那条多验两样东西，都是采样专属的：
    * **随机源每条请求一份** —— 共用的话，"这条流这一步抽到什么"取决于别人问了
      几步，批里与单跑必然分叉。
    * **历史跟着自己的请求走** —— 采样器拿它做重复惩罚，拿错的话只在生成出重复
      词的那一段才分叉（最容易"跑一遍看着没问题"）。
    """
    var svc = ModelService.load(ServeConfig.from_env())
    var ps = _prompts()
    svc.batch_enabled = True
    var batched = _batch_run(svc, ps, TEMP)
    svc.batch_enabled = False
    var solo = _solo_all(svc, ps, TEMP)
    _compare(batched, solo, "sampling batch=4")


def test_four_identical_sampling_streams_agree() raises:
    """四条**同 prompt、同种子**的采样流一起批跑，必须给出同一个序列。

    这是"随机源每条请求一份"的专属对照，且不靠老路：同 prompt 同种子同温度，四
    条流的每一步面对的是同一份 logits、同一个分布 → 它们**必须**抽到同一个数。
    共用一个随机源时它们一定分叉（交错推进让每条流抽到的是序列里不同位置的数），
    所以这一条红了可以直接归因到随机源，不用先怀疑前向。
    """
    var svc = ModelService.load(ServeConfig.from_env())
    var ps = List[String]()
    for i in range(4):
        ps.append(PROMPT_A)
    svc.batch_enabled = True
    var batched = _batch_run(svc, ps, TEMP)
    for k in range(STEPS):
        assert_equal(
            batched[k],
            batched[1 * STEPS + k],
            "two identical sampling streams drew differently at step=" + String(k),
        )
        assert_equal(
            batched[k],
            batched[3 * STEPS + k],
            "two identical sampling streams drew differently at step=" + String(k),
        )


def test_requests_beyond_the_batch_wait_instead_of_being_refused() raises:
    """第 9 条之后不是"拒绝"，是**排队** —— 且排过队的答案与单跑一致。

    判据分两半，缺一半这条门就不成立：
    * **不被拒绝**：10 条请求一起推进（引擎只有 `MAX_BATCH` = 8 个槽位），全部跑
      完、每条都给满 `STEPS` 个 token。以前第 9 条收到的是 `capacity`（"this
      service runs one sampling stream at a time"），客户端只能自己重试。
    * **确实排过队**：`svc.waited` 必须 ≥ 2。少了这一半，"10 条都跑完了"也可能只
      是"恰好没人超额" —— 门会在逻辑上变绿而一个字节也没验到。
    """
    var svc = ModelService.load(ServeConfig.from_env())
    var ps = List[String]()
    for i in range(10):
        if i % 2 == 0:
            ps.append(PROMPT_A)
        else:
            ps.append(PROMPT_C)
    svc.batch_enabled = True
    var batched = _batch_run(svc, ps, TEMP)
    for i in range(10):
        for k in range(STEPS):
            assert_true(
                batched[i * STEPS + k] >= 0,
                "第 "
                + String(i)
                + " 条没有跑完（被拒了？）step="
                + String(k),
            )
    assert_true(svc.waited >= 2, "应该有请求排过队，waited=" + String(svc.waited))
    svc.batch_enabled = False
    var solo = _solo_all(svc, ps, TEMP)
    _compare(batched, solo, "queued")


def test_the_two_paths_really_are_two_paths() raises:
    """批路必须**真的**进引擎，老路必须**真的**没进。

    常驻负向对照：如果批路静默退回了老路（比如形状判断写错、`batch_enabled` 没
    接上），上面几条就是同一条路和自己比 —— 它们会绿，而门一个字节也没验到。这
    一条把"两条路不同"本身钉住。
    """
    var svc = ModelService.load(ServeConfig.from_env())
    svc.batch_enabled = True
    svc.stream_begin(PROMPT_A, STEPS, 0.0, 0)
    assert_true(svc._slot_of(0) >= 0, "批路必须进引擎（却没进）")
    svc.stream_end(0)
    svc.batch_enabled = False
    svc.stream_begin(PROMPT_A, STEPS, 0.0, 0)
    assert_true(svc._slot_of(0) < 0, "老路不许进引擎（却进了）")
    svc.stream_end(0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
