"""前向阶段计时器 —— **编译期**门，不是一个运行时 `if`。

默认 `ALOFA_PROFILE = False` ⇒ 所有打点在编译期折叠成 0：二进制里既没有
`monotonic_ns()` 也没有累加，生产路径零成本。要测量就把常量改成 True 重新
`mojo build`。

为什么不做运行时开关：打点落在每 token 168 次投影的热路径上，一个运行时的
`if self.prof_on` 无法被折叠掉，而且它会挡住内联（见 `block_scale` 那个坑）。
编译期门还有个好处 —— 关着的时候代码**根本不在二进制里**，不用论证它有多便宜。

用法，调用点始终一行：

    var t = self.prof.now()          # 段开始
    ...
    t = self.prof.tick(K_NORM, t)    # 段结束并累加；返回新起点，可串联
    _ = self.prof.tick(K_HEAD, t)    # 不串联就丢弃
    self.prof.count_token()          # 一个 token 走完

⚠️ 带**副作用**的测量（比如"额外做一次 `params.view` 看它多贵"）不能只靠
`now()`/`tick()` —— 那两次调用会折叠掉，但中间的副作用还在。那种地方必须整段
包在 `comptime if ALOFA_PROFILE:` 里。
"""

from alofa.core.ffi import monotonic_ns


comptime ALOFA_PROFILE = False
"""编译期开关。False ⇒ 打点不进二进制。改这里必须重新 build。"""


comptime K_NORM = 0
comptime K_ROPE_ATTN = 1
comptime K_ACT = 2
comptime K_RESID = 3
comptime K_HEAD = 4
comptime K_VIEW = 5
comptime K_PREP = 6
comptime K_GEMM = 7
comptime K_ROPE = 8


def _f2(x: Float64) -> String:
    """两位小数。本机没有 sprintf，整数拼是唯一不引入新依赖的写法。"""
    var v = x
    var whole = Int(v)
    var frac = Int((v - Float64(whole)) * 100.0 + 0.5)
    if frac >= 100:
        whole += 1
        frac -= 100
    var fs = String(frac)
    if fs.byte_length() < 2:
        fs = "0" + fs
    return String(whole) + "." + fs


struct ForwardProf(Movable):
    """前向各阶段的纳秒累加器。默认档所有 `now`/`tick` 折叠为 `return 0`。"""

    var on: Bool
    var norm: Int
    var rope_attn: Int
    var rope: Int
    var act: Int
    var resid: Int
    var head: Int
    var view: Int
    var prep: Int
    var gemm: Int
    var tokens: Int

    def __init__(out self):
        self.on = False
        self.norm = 0
        self.rope_attn = 0
        self.rope = 0
        self.act = 0
        self.resid = 0
        self.head = 0
        self.view = 0
        self.prep = 0
        self.gemm = 0
        self.tokens = 0

    @always_inline
    def now(self) -> Int:
        """段起点。`ALOFA_PROFILE=False` 时恒为 0。"""
        comptime if not ALOFA_PROFILE:
            return 0
        return monotonic_ns()

    @always_inline
    def tick(mut self, k: Int, t0: Int) -> Int:
        """把 `now() - t0` 累加进第 `k` 个桶，返回新起点便于串联。"""
        comptime if not ALOFA_PROFILE:
            return 0
        var t1 = monotonic_ns()
        if k == K_NORM:
            self.norm += t1 - t0
        elif k == K_ROPE_ATTN:
            self.rope_attn += t1 - t0
        elif k == K_ACT:
            self.act += t1 - t0
        elif k == K_RESID:
            self.resid += t1 - t0
        elif k == K_HEAD:
            self.head += t1 - t0
        elif k == K_VIEW:
            self.view += t1 - t0
        elif k == K_PREP:
            self.prep += t1 - t0
        elif k == K_GEMM:
            self.gemm += t1 - t0
        elif k == K_ROPE:
            self.rope += t1 - t0
        return t1

    @always_inline
    def count_token(mut self):
        """一个 token 走完。默认档是空操作。"""
        comptime if ALOFA_PROFILE:
            self.tokens += 1

    def reset(mut self):
        """清零各桶（`set_profile` 会调）。"""
        self.norm = 0
        self.rope_attn = 0
        self.rope = 0
        self.act = 0
        self.resid = 0
        self.head = 0
        self.view = 0
        self.prep = 0
        self.gemm = 0
        self.tokens = 0

    def dump(imm self):
        """打印每 token 各阶段毫秒。投影相关的按次数摊到"每次"。"""
        comptime if not ALOFA_PROFILE:
            return
        if self.tokens <= 0:
            print("   [profile] 没有样本（ALOFA_PROFILE 未开或没跑过 step）")
            return
        var n = Float64(self.tokens)
        var norm = Float64(self.norm) / n / 1000000.0
        var ra = Float64(self.rope_attn) / n / 1000000.0
        var rp = Float64(self.rope) / n / 1000000.0
        var act = Float64(self.act) / n / 1000000.0
        var resid = Float64(self.resid) / n / 1000000.0
        var head = Float64(self.head) / n / 1000000.0
        print(
            "   [profile] 每 token 各段（ms）· 样本 "
            + String(self.tokens)
            + " 个 token"
        )
        print("     rmsnorm ×2/层        " + _f2(norm))
        print("     attn 核（分数+softmax+AV） " + _f2(ra))
        print("     rope+KVcopy         " + _f2(rp))
        print("     swiglu              " + _f2(act))
        print("     残差 add ×2/层      " + _f2(resid))
        print("     输出投影 lm_head    " + _f2(head))
        print("     —— 以上合计（非投影串行部分） " + _f2(norm + rp + ra + act + resid))
        # 168 = 7 次投影/层 × 24 层；96 = 其中无 bias 的 4 次/层（o/gate/up/down）
        var per_proj = n * 168.0
        var per_nb = n * 96.0
        print("     【project 内部】")
        print(
            "       params.view 查表（fp32 档才有）  "
            + _f2(Float64(self.view) / per_proj / 1000000.0 * 1000.0)
            + " µs/次 · 合计 "
            + _f2(Float64(self.view) / n / 1000000.0)
            + " ms"
        )
        print(
            "       核前准备（q4 档，96 次/tok）      "
            + _f2(Float64(self.prep) / per_nb / 1000000.0 * 1000.0)
            + " µs/次 · 合计 "
            + _f2(Float64(self.prep) / n / 1000000.0)
            + " ms"
        )
        print(
            "       核本身（q4 无 bias，96 次/tok）   "
            + _f2(Float64(self.gemm) / per_nb / 1000000.0 * 1000.0)
            + " µs/次 · 合计 "
            + _f2(Float64(self.gemm) / n / 1000000.0)
            + " ms"
        )
