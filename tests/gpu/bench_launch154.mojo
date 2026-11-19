"""账本一百五十四：prefill_step 发射开销剖析（借 GpuSession 的 buffer）。

背景：profile=1/2 双口径都显示 tri（rmsnorm）段 ~30µs/次发射，而
decode_step 的同类发射 ~3-5µs。本基准借 GpuSession 的批行 buffer
（d_xb[32,896] → d_xnormb）重复发射批行 rmsnorm，与单行形状对照：

- 形状 A：批行 [32,896]，grid=(32,1,1) —— prefill_step 批模式同款
- 形状 B：单行 [1,896]，grid=(1,1,1) —— decode_step 同款
- 形状 C：批行 buffer 但 rows=1、grid=(1,1,1) —— 排除 rows 参数因素

同一个 kernel、同一个 ctx、同一调用代码——差异只在形状参数。
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from alofa.core.ffi.posix import monotonic_ns
from alofa.kernels.cuda.basic import rmsnorm_gpu
from alofa.model.arch.qwen import QwenForward
from alofa.gpu.session import GpuSession

comptime COLS = 896


struct Launcher:
    """持 GpuSession（其 buffers/ctx 是久经验证的析构单元），只借字段。
    ctx 提为直属字段（self.s.ctx 两层链会触发 enqueue_function 的
    capturing 特化失败——单层 self.ctx 与 GpuSession 方法同构）。"""

    var s: GpuSession
    var ctx: DeviceContext

    def __init__(out self, model: QwenForward) raises:
        self.ctx = DeviceContext()
        self.s = GpuSession(model, 512, self.ctx)

    def shoot_batch(mut self, rows: Int, reps: Int) raises -> Float64:
        """批行 buffer：d_xb → d_xnormb（prefill_step 批模式同款）。"""
        var t0 = monotonic_ns()
        for _k in range(reps):
            self.ctx.enqueue_function[rmsnorm_gpu](
                self.s.d_xb.unsafe_ptr(),
                self.s.d_norms[0].unsafe_ptr(),
                self.s.d_xnormb.unsafe_ptr(),
                Pointer[Float16, MutUntrackedOrigin](),
                Int32(COLS),
                Int32(rows),
                self.s.eps,
                grid_dim=(rows, 1, 1),
                block_dim=(256, 1, 1),
            )
        self.ctx.synchronize()
        return Float64(monotonic_ns() - t0) / 1e6 / Float64(reps)

    def shoot_single(mut self, reps: Int) raises -> Float64:
        """单行 buffer：d_x → d_normed（decode_step 同款）。"""
        var t0 = monotonic_ns()
        for _k in range(reps):
            self.ctx.enqueue_function[rmsnorm_gpu](
                self.s.d_x.unsafe_ptr(),
                self.s.d_norms[0].unsafe_ptr(),
                self.s.d_normed.unsafe_ptr(),
                Pointer[Float16, MutUntrackedOrigin](),
                Int32(COLS),
                Int32(1),
                self.s.eps,
                grid_dim=(1, 1, 1),
                block_dim=(256, 1, 1),
            )
        self.ctx.synchronize()
        return Float64(monotonic_ns() - t0) / 1e6 / Float64(reps)


def main() raises:
    var model = QwenForward("tests/fixtures/qwen2.5-0.5b/weights",
        "tests/fixtures/qwen2.5-0.5b/config.tsv", 512, True)
    var lb = Launcher(model)

    # 预热
    _ = lb.shoot_batch(32, 50)

    # 形状 A：批行 [32,896]，grid=(32,1,1)
    var a1 = 1e30
    for _k in range(3):
        var v = lb.shoot_batch(32, 1000)
        if v < a1:
            a1 = v
    print("batch-rows(32) launch us:", a1 * 1e3)

    # 形状 B：单行 [1,896]，grid=(1,1,1)
    var b1 = 1e30
    for _k in range(3):
        var v = lb.shoot_single(1000)
        if v < b1:
            b1 = v
    print("single-row(1) launch us:", b1 * 1e3)

    # 形状 C：批行 buffer 但 grid=(1,1,1)
    var c1 = 1e30
    for _k in range(3):
        var v = lb.shoot_batch(1, 1000)
        if v < c1:
            c1 = v
    print("batch-buf rows=1 launch us:", c1 * 1e3)
