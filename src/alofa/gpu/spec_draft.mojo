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


def _win_hash(win: List[Int]) -> Int:
    """n-gram 窗口滚动哈希（2^62 掩码防溢出）。碰撞 → 错草稿，验证
    层拒绝只浪费预算行——正确性不依赖哈希唯一性。"""
    var h = 0
    for t in win:
        h = ((h * 1000003) + t) & ((1 << 62) - 1)
    return h


struct SpecTable(Movable):
    """会话级共享输出 n-gram 表（一百七十五-补四）——跨流草稿源。

    175-补三画像暴露 prompt-lookup 在真问答域 hit≈0（草稿只能来自
    prompt 内部复现）。本表把**每条流的产出**（贪心与采样一致收，
    _emit_token 侧钩子）以滑动 n-gram 窗口写入共享 Dict[Int, Int]
    （窗口滚动哈希 → 下一 token），查询时**链式跟随**重建完整草稿
    链：同前缀的历史生成被逐位重放（重复请求 / 并发同 prompt /
    多轮对话），异前缀自然脱靶。正确性不受表污染——草稿经 spec
    验证（argmax 前缀匹配），错草稿只浪费预算行；表在单个 service
    生命周期内单调增长，cap 封顶后停止写入（确定性：同读写序列同
    表态）。
    """

    var map: Dict[Int, Int]
    var cap: Int
    var n_inserts: Int
    var n_shared_hits: Int

    def __init__(out self, cap: Int = 1 << 18):
        self.map = Dict[Int, Int]()
        self.cap = cap
        self.n_inserts = 0
        self.n_shared_hits = 0

    def insert(mut self, win: List[Int], nxt: Int) raises:
        """写一条（窗口哈希 → 下一 token）。同键覆写（最近生成最
        能代表当前分布）；cap 满即停写——只影响新键召回，旧键照常。"""
        if len(self.map) >= self.cap:
            return
        self.map[_win_hash(win)] = nxt
        self.n_inserts += 1

    def follow(
        mut self, tail: List[Int], k: Int, mut drafts: List[Int]
    ) raises -> Bool:
        """从 ctx 末 n-gram 窗口 `tail` 起链式跟随，至多 `k` 枚草稿。
        窗口滑动推进（追加入草稿 token），任一环脱靶即止。"""
        var win = List[Int]()
        for t in tail:
            win.append(t)
        var any = False
        for _ in range(k):
            var h = _win_hash(win)
            if h not in self.map:
                break
            var nxt = self.map[h]
            drafts.append(nxt)
            any = True
            for j in range(len(win) - 1):
                win[j] = win[j + 1]
            win[len(win) - 1] = nxt
        return any


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
