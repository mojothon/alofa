"""合批性能口径基准（账本一百三十五）：给 v0 跨流合批的收益定基线。

两个口径：

1. **session 层微基准**——`prefill_step` 批路 t=1/2/4/8 的每步墙钟
   （argmax、padding 行补齐）：验证「一次前向推 N 行」的摊薄比——
   t 行的一次前向若 ≈ t=1 的一次前向（权重读主导），则合批吞吐
   ≈ t 倍。
2. **GpuService 层端到端**——两流（EN 24 帧 / CJK 16 帧）：
   串行（先 A 后 B 各独跑）vs 交替（tick 合批）的总墙钟与加速比。

计时 = `monotonic_ns`（仓库 bench 惯例），预热后取均值；GpuService
层每次先 `stream_end` 清场，冷热两个时段分开报。
"""

from alofa.gpu.session import GpuSession
from alofa.model.arch.qwen import QwenForward
from alofa.srv.gpu_service import GpuService
from alofa.tokenizer import load_tokenizer_json

from alofa.core.ffi.posix import monotonic_ns
from max.gpu.host import DeviceContext

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
comptime CJK = "给我讲一个简短的睡前故事。"
comptime EN = "Once upon a time, in a land far away,"

comptime INNER = 896
comptime FF = 4864
comptime BMAX = 64
comptime HQ = 4
comptime KVDIM = 128
comptime HDim = 128
comptime ATTN_T = 128
comptime MT = 512
comptime MB = 1
comptime TILES = 8


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var ctx = DeviceContext()
    var session = GpuSession(model, 512, ctx)

    # ---- 口径 1：session 层 t=1/2/4/8 每步延迟 ----
    print("== session-layer step delay (argmax, ms/step) ==")
    var base = Float64(0)
    for t in [1, 2, 4, 8, 16]:
        var toks = List[Int]()
        for _b in range(t):
            toks.append(100)
        var warm = 5
        var iters = 30
        # 预热
        for s in range(warm):
            for b in range(t):
                session.h_posarr[b] = Int32(s)
            _ = session.prefill_step[
                INNER, FF, BMAX, HQ, KVDIM, HDim, ATTN_T, MT, MB, TILES,
                DType.float32, True, True, True, True, True, True, True,
                True, False, True
            ](toks^, False, False, False, 0.0, Int32(0), 1.0, 0, 1, 0)
            session.ctx.synchronize()
        # 测量
        var t0 = monotonic_ns()
        for s in range(iters):
            for b in range(t):
                session.h_posarr[b] = Int32(warm + s)
            _ = session.prefill_step[
                INNER, FF, BMAX, HQ, KVDIM, HDim, ATTN_T, MT, MB, TILES,
                DType.float32, True, True, True, True, True, True, True,
                True, False, True
            ](toks^, False, False, False, 0.0, Int32(0), 1.0, 0, 1, 0)
            session.ctx.synchronize()
        var ms = Float64(monotonic_ns() - t0) / 1e6 / Float64(iters)
        print("  t=", t, "  per-step:", ms, "ms")
        if t == 1:
            base = ms
        else:
            var ideal = base  # t 行理想 = 与 t=1 同价（权重读主导）
            print("    -> per-row cost:", ms / Float64(t), "ms (t=1:",
                  base, "ms; batch-efficiency vs serial:",
                  base * Float64(t) / ms, "x)")

    # ---- 口径 2：GpuService 端到端（串行 vs 交替合批）----
    print("== service-level end-to-end (EN 24f / CJK 16f) ==")
    var tok2 = load_tokenizer_json(TOKJSON)
    var svc = GpuService(
        model, 8, 512, tok2^, 151936, 42, WEIGHTS, CONFIG, TOKJSON,
    )

    # 串行：A 独跑 + B 独跑
    var t0 = monotonic_ns()
    svc.stream_begin(EN, 24, 0.0, 1.0, 1)
    var da = False
    while not da:
        var fr = svc.stream_next(1)
        if fr.done:
            da = True
    svc.stream_end(1)
    var t1 = monotonic_ns()
    svc.stream_begin(CJK, 16, 0.0, 1.0, 2)
    var db = False
    while not db:
        var fr2 = svc.stream_next(2)
        if fr2.done:
            db = True
    svc.stream_end(2)
    var serial_ms = Float64(monotonic_ns() - t1) / 1e6
    var a_ms = Float64(t1 - t0) / 1e6
    print("  serial: A", a_ms, "ms + B", serial_ms, "ms = total",
          a_ms + serial_ms, "ms")

    # 交替（合批）：两条同时活跃
    svc.stream_begin(EN, 24, 0.0, 1.0, 11)
    svc.stream_begin(CJK, 16, 0.0, 1.0, 12)
    var da2 = False
    var db2 = False
    var t2 = monotonic_ns()
    while (not da2) or (not db2):
        if not da2:
            var fa = svc.stream_next(11)
            if fa.done:
                da2 = True
        if not db2:
            var fb = svc.stream_next(12)
            if fb.done:
                db2 = True
    svc.stream_end(11)
    svc.stream_end(12)
    var batched_ms = Float64(monotonic_ns() - t2) / 1e6
    print("  batched interleaved total:", batched_ms, "ms")
    print("  speedup vs serial:", (a_ms + serial_ms) / batched_ms, "x")
    print("  ticks:", svc.n_ticks, "batched:", svc.n_batched,
          "single:", svc.n_single)
