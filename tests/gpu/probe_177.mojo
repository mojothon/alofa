"""账本一百七十四-c 诊断探针 177：generate_batch 自污染定位。

genA → 快照 slot0 K rows0-11（全层）→ genB → 快照 → 逐层逐行比对。
genB 与 genA 无任何显式中间操作；若 genB 已错而 K 快照全同，
则分叉在 genB 步 11 的前向内部（读到了快照之外的隐藏状态）。"""

from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession
from alofa.tokenizer import load_tokenizer_json


def main() raises:
    comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b/weights"
    comptime CONFIG = "tests/fixtures/qwen2.5-0.5b/config.tsv"
    comptime TOKJSON = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
    comptime PROMPT = "Once upon a time, in a land far away,"

    var model = QwenForward(WEIGHTS, CONFIG, 512, True)
    var sess = GpuSession(model, 512)
    var tok = load_tokenizer_json(TOKJSON)
    var pids = tok.encode(PROMPT)
    var p = len(pids)
    print("prompt tokens:", p)

    # genA
    var jfA = List[Int]()
    var jlA = List[Int]()
    var rgA = sess.generate_batch[896, 4864](
        pids, [p], 12, -1, jfA, jlA, 0.0, 0, 1.0
    )
    print("genA:", rgA[0], rgA[1], rgA[2], rgA[3])

    # 快照 K rows 0-11 全层
    var snapA = List[Float64]()
    sess.ctx.synchronize()
    for l in range(sess.n_layers):
        var kbA = sess.ctx.enqueue_create_host_buffer[DType.float32](
            64 * 512 * 128
        )
        sess.ctx.enqueue_copy(dst_buf=kbA, src_buf=sess.d_k_store_b[l])
        sess.ctx.synchronize()
        for rr in range(12):
            for c in range(128):
                snapA.append(Float64(kbA[rr * 128 + c]))

    # 毒饵：K/V store rows 13-63 全层毒成 1e30
    for l in range(sess.n_layers):
        var kp = sess.ctx.enqueue_create_host_buffer[DType.float32](
            64 * 512 * 128
        )
        sess.ctx.enqueue_copy(dst_buf=kp, src_buf=sess.d_k_store_b[l])
        var vp = sess.ctx.enqueue_create_host_buffer[DType.float32](
            64 * 512 * 128
        )
        sess.ctx.enqueue_copy(dst_buf=vp, src_buf=sess.d_v_store_b[l])
        sess.ctx.synchronize()
        for c in range(128):
            kp[12 * 128 + c] = 1.0e30
            vp[12 * 128 + c] = 1.0e30
        sess.ctx.enqueue_copy(dst_buf=sess.d_k_store_b[l], src_buf=kp)
        sess.ctx.enqueue_copy(dst_buf=sess.d_v_store_b[l], src_buf=vp)
    sess.ctx.synchronize()
    print("poisoned K/V rows 13-63")

    # genB（无任何显式中间操作）
    var jfB = List[Int]()
    var jlB = List[Int]()
    var rgB = sess.generate_batch[896, 4864](
        pids, [p], 12, -1, jfB, jlB, 0.0, 0, 1.0
    )
    print("genB:", rgB[0], rgB[1], rgB[2], rgB[3])

    var snapB = List[Float64]()
    sess.ctx.synchronize()
    for l in range(sess.n_layers):
        var kbB = sess.ctx.enqueue_create_host_buffer[DType.float32](
            64 * 512 * 128
        )
        sess.ctx.enqueue_copy(dst_buf=kbB, src_buf=sess.d_k_store_b[l])
        sess.ctx.synchronize()
        for rr in range(12):
            for c in range(128):
                snapB.append(Float64(kbB[rr * 128 + c]))

    var ndiff = 0
    for l in range(sess.n_layers):
        for rr in range(12):
            var dsum = Float64(0)
            for c in range(128):
                dsum += abs(
                    snapA[l * 12 * 128 + rr * 128 + c]
                    - snapB[l * 12 * 128 + rr * 128 + c]
                )
            if dsum > 0:
                print("Kdiff layer", l, "row", rr, ": sum =", dsum)
                ndiff += 1
    print("Kdiff total:", ndiff)
