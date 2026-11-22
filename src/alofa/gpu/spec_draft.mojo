"""n-gram 草稿源（一百七十四-c 第四刀）：prompt-lookup speculation
的 host 侧实现——零额外权重，从「prompt + 已生成链」里找与最近上
下文匹配的 n-gram，取其后缀 token 作草稿链。

设计（与账本 174 开题的③一致）
------------------------------
* 草稿源可插拔：本模块是起步实现（prompt-lookup），接口（输入链 +
  输出草稿链）预留后续换小模型草稿源。
* 贪心解码（det 门）下验证 = argmax 前缀匹配（`GpuSession.spec_step`
  的 trust_all=False 路径），无采样漂移——草稿质量只影响接受率，
  不影响输出的正确性（接受即贪心链逐位同）。
* 匹配策略：扫描「最近的 n-gram 命中」的连续后缀链（最近历史最能
  预测未来；重复模式文本通常单命中，链语义保证草稿是连贯续写）。

遥测口径（174-d A/B 用）
------------------------
* `n_calls`   验证 tick 数（含无草稿的退化 tick——退化为 t=1）
* `n_hits`    命中草稿的 tick 数（hit_rate = n_hits / n_calls）
* `n_drafted` 提议草稿 token 总数
* `n_accepted` 被接受的草稿 token 总数（accept_rate = accepted/drafted）
* `n_tokens`  生成 token 总数（不含 prompt；tokens_per_tick = n_tokens/n_calls
  ——spec 的收益口径：>1 即每 tick 多产）
"""

from alofa.core.error import ERR_INVALID_ARGUMENT, AlofaError


struct SpecTelemetry(Movable):
    """spec 生成的草稿命中/接受遥测（174-d 生产 A/B 的口径面）。"""

    var n_calls: Int
    var n_hits: Int
    var n_drafted: Int
    var n_accepted: Int
    var n_tokens: Int

    def __init__(out self):
        self.n_calls = 0
        self.n_hits = 0
        self.n_drafted = 0
        self.n_accepted = 0
        self.n_tokens = 0

    def hit_rate(self) -> Float64:
        """有草稿可验的 tick 占比——草稿源的覆盖面。"""
        return Float64(self.n_hits) / Float64(max(self.n_calls, 1))

    def accept_rate(self) -> Float64:
        """被接受的草稿占比——草稿源的质量。"""
        return Float64(self.n_accepted) / Float64(max(self.n_drafted, 1))

    def tokens_per_tick(self) -> Float64:
        """每验证 tick 的产出 token 数——spec 净收益的直接口径。"""
        return Float64(self.n_tokens) / Float64(max(self.n_calls, 1))


struct SpecGenResult(Movable):
    """`GpuSession.generate_spec` 的返回：生成链 + 遥测 + 前向数。"""

    var tokens: List[Int]
    var tel: SpecTelemetry
    var n_ticks: Int

    def __init__(
        out self, var tokens: List[Int], var tel: SpecTelemetry,
        n_ticks: Int,
    ):
        self.tokens = tokens^
        self.tel = tel^
        self.n_ticks = n_ticks


def spec_lookup(
    ctx: List[Int], k: Int, ngram: Int, mut drafts: List[Int]
) raises -> Bool:
    """prompt-lookup 草稿查询：在 `ctx`（prompt + 已生成链）里找与
    末尾 `ngram` 个 token 匹配的位置，取**最近一次命中**的连续后缀
    链（至多 `k` 枚）追加进 `drafts`。返回是否命中。

    纯 host 逻辑，确定性：同 ctx 同输出。扫描升序、记录最后命中的
    链起点——等价于「从最近的历史里找模板」。命中链短于 k（模板
    贴近 ctx 末尾）时取可得的部分——验证 tick 的行数相应变少，无
    浪费。
    """
    if k <= 0 or ngram <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT, "spec_lookup k/ngram must be positive", ""
        )
    var n = len(ctx)
    if n <= ngram:
        return False
    # 升序扫描，记录**最后**一次命中的链起点（最近历史最能预测未来）。
    var ext = -1
    var i = 0
    while i + ngram < n:
        var ok = True
        for j in range(ngram):
            if ctx[i + j] != ctx[n - ngram + j]:
                ok = False
                break
        if ok:
            ext = i + ngram
        i += 1
    if ext < 0:
        return False
    var end = ext + k
    if end > n:
        end = n
    for p in range(ext, end):
        drafts.append(ctx[p])
    return True
