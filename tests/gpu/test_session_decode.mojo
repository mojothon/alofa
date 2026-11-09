"""GpuSession 端到端对数：单 token decode，GPU 全层 vs 产品 `run`（账本六十三）。

设计 §5 步骤 3 的验收：`GpuSession.decode_step` 跑完 28 层（gather → 每层
rmsnorm/q/k/v/rope/append/attention/o/残差/gate/up/swiglu/down/残差 → 终
norm → lm_head），logits 与 `model.run`（BACKEND_SCALAR，product 通路本身）
对数。判据：相对容差 1e-5 × ref_max —— 两侧累加序不同（shards f64 vs smem
f32），28 层非线性传播后的合法重组差要落在这个带内；**错了就是错**，任何
超出 1e-5 的差都按 bug 查，不放宽。

Run（A100）:
    mojo run -I src tests/gpu/test_session_decode.mojo
"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession
from alofa.verify.compare import max_abs, max_abs_diff

comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
comptime TOKEN = 42
# smem kernel 的编译期列数：hidden=896（每层 6 个投影），inter=4864（down）。
comptime INNER = 896
comptime INNER_FF = 4864
comptime REL_TOLERANCE = Float64(1e-5)


def main() raises:
    var model = QwenForward(WEIGHTS, CONFIG, 64, True)
    var session = GpuSession(model, 64)

    # 多步 decode：CPU 一次 prefill 三个 token，GPU 逐步 decode —— 终 logits
    # 对数（同时验证 pos>0 的 rope 表行与 kv_len>1 的 attention/append）。
    var ids = List[Int]()
    ids.append(TOKEN)
    ids.append(17)
    ids.append(900)
    var cpu_logits = model.run(ids)
    var gpu_logits: Pointer[Float32, MutUntrackedOrigin] = (
        session.decode_step[INNER, INNER_FF](TOKEN)
    )
    gpu_logits = session.decode_step[INNER, INNER_FF](17)
    gpu_logits = session.decode_step[INNER, INNER_FF](900)

    var n = model.cfg.vocab
    var diff = max_abs_diff(gpu_logits, cpu_logits, n)
    var scale = max_abs(cpu_logits, n)
    if scale <= 0.0:
        raise Error("logits: the reference is all zeros")
    if diff > REL_TOLERANCE * scale:
        raise Error(
            "logits: diff=" + String(diff) + " tol=" + String(REL_TOLERANCE * scale)
            + " ref_max=" + String(scale)
        )
    print(
        "*** SESSION DECODE GPU/CPU PARITY PASS *** diff=",
        diff,
        " tol=",
        REL_TOLERANCE * scale,
    )
    # 样本对照（防"比到自己"的假绿）：两侧各打头四个与最大值。
    var pg = gpu_logits
    var pc = cpu_logits
    for i in range(4):
        print("logits[", i, "] gpu=", pg[unsafe_offset=i], " cpu=", pc[unsafe_offset=i])
    var best = 0
    for i in range(1, n):
        if pc[unsafe_offset=i] > pc[unsafe_offset=best]:
            best = i
    print(
        "argmax=", best,
        " gpu_max=", pg[unsafe_offset=best],
        " cpu_max=", pc[unsafe_offset=best],
    )
    model.params.keep_alive()
