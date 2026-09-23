"""批调度 vs 单流：同一条 prompt，两条路必须**逐 token 相等**。

批调度那笔收益（一条 engine 线程同时推进多条流）只有在这道门绿了之后才算拿到
—— 因为"快"和"算错"在服务端日志里是**一样**的：两条路都不报错，只是其中一条给
出的 token 与另一条不同。而"逐 token 相等"是唯一能把它钉住的判据：一批 N 条一起
前向，与这 N 条各自单跑，必须给出同一个 token 序列。

判据写成**相等**而不是"接近"：这里比的是 token id，是整数 —— 一个 token 不同就是
不同，没有"差不多"的余地。logits 才需要容差，而 logits 差一点点会在自回归里滚成
完全不同的答案，所以拿 logits 当判据等于没判。

两条路：批路走 `EngineCore`（一次前向推进整批），老路走 `model.prefill` / `step`
（一次一条）。开关是 `ModelService.batch_enabled`（默认开，`ALOFA_BATCH=0` 关）—
— 它存在的理由就是这道门：**同一条 prompt 要能分别走两条路**才能对起来，没它
的话两条路只能各自跑，差异会被"反正 prompt 不一样"永远盖住。

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


def _prompts() -> List[String]:
    var ps = List[String]()
    ps.append(PROMPT_A)
    ps.append(PROMPT_B)
    ps.append(PROMPT_C)
    ps.append(PROMPT_D)
    return ps^


def _solo_run(mut svc: ModelService, prompt: String) raises -> List[Int]:
    """老路：一条一条跑（`batch_enabled` 由调用方设成 False），返回 token 序列。"""
    svc.stream_begin(prompt, STEPS, 0.0, 0)
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


def _batch_run(mut svc: ModelService, ps: List[String]) raises -> List[Int]:
    """批路：`len(ps)` 条同时 submit 再交错推进；返回扁平序列（第 i 条从
    `i * STEPS` 起，不足的位填 `-1`）。

    交错推进是**故意**的：服务里就是这么走的（每条连接一帧一往返），所以门要照
    着服务的节奏问，而不是一次把一条问完再问下一条。
    """
    for i in range(len(ps)):
        svc.stream_begin(ps[i], STEPS, 0.0, i)
    var finished = List[Int]()
    for i in range(len(ps)):
        finished.append(0)
    var remaining = len(ps)
    var guard = 0
    while remaining > 0 and guard < 512:
        guard += 1
        for i in range(len(ps)):
            if finished[i] == 1:
                continue
            if svc.stream_next(i).done:
                finished[i] = 1
                remaining -= 1
    var out = List[Int]()
    for i in range(len(ps)):
        var slot = svc._slot_of(i)
        var n = svc.b_n[slot]
        for k in range(STEPS):
            if k < n:
                out.append(svc.b_out[slot * MAX_GEN + k])
            else:
                out.append(-1)
        svc.stream_end(i)
    return out^


def _solo_all(mut svc: ModelService, ps: List[String]) raises -> List[Int]:
    """同一批 prompt 在**老路**上一条一条跑（扁平化，与 `_batch_run` 同形状）。"""
    var out = List[Int]()
    for i in range(len(ps)):
        var seq = _solo_run(svc, ps[i])
        for k in range(STEPS):
            if k < len(seq):
                out.append(seq[k])
            else:
                out.append(-1)
    return out^


def test_a_batch_of_four_agrees_with_four_solo_runs() raises:
    """四条不同 prompt 一起批跑，必须等于它们各自单跑 —— **逐 token**。

    这四条是不同长度、不同语言的 prompt（英文 / 中文 / 数字 / 散文），所以批路
    里的 prefill 是**真的**凑成了一批（而不是四条恰好一样长）。
    """
    var svc = ModelService.load(ServeConfig.from_env())
    var ps = _prompts()
    svc.batch_enabled = True
    var batched = _batch_run(svc, ps)
    svc.batch_enabled = False
    var solo = _solo_all(svc, ps)
    for i in range(len(ps)):
        for k in range(STEPS):
            assert_equal(
                batched[i * STEPS + k],
                solo[i * STEPS + k],
                "token differs: prompt="
                + String(i)
                + " step="
                + String(k)
                + " batch="
                + String(batched[i * STEPS + k])
                + " solo="
                + String(solo[i * STEPS + k]),
            )


def test_a_batch_of_one_agrees_with_one_solo_run() raises:
    """批里**只有一条**时也要相等 —— 最小形状。

    它和上面那条不是重复：批=1 是"批路本身对不对"（没有批间的相互影响可怪），
    批=4 才是"合批有没有改变结果"。分开跑的话，一旦红，能直接看出是哪一个。
    """
    var svc = ModelService.load(ServeConfig.from_env())
    var ps = List[String]()
    ps.append(PROMPT_B)
    svc.batch_enabled = True
    var batched = _batch_run(svc, ps)
    svc.batch_enabled = False
    var solo = _solo_all(svc, ps)
    for k in range(STEPS):
        assert_equal(batched[k], solo[k], "batch=1 token differs at step=" + String(k))


def test_the_two_paths_really_are_two_paths() raises:
    """批路必须**真的**进引擎，老路必须**真的**没进。

    常驻负向对照：如果批路静默退回了老路（比如形状判断写错、`batch_enabled` 没
    接上），上面两条就是同一条路和自己比 —— 它们会绿，而门一个字节也没验到。这
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
