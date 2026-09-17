"""这份文件**必须编译失败**：未知的 CPU 后端不能静默退化成标量。

它不是一个测试，是一道编译期红测，由 `pixi run test-backend-guard` 驱动：
先尝试编译它，再要求编译错误里出现 `unknown cpu backend`。

为什么需要一道"必须编译不过"的门：`prefill` / `step` / `run` 的算子分发写成
`comptime if uses_vector_backend[backend]()`，于是**任何**不是 `BACKEND_AVX2`
的取值都会走进标量分支。拼错一个常量（`prefill[7]`）得到的不是报错，而是一个
安静跑在标量后端上、所有门全绿的"向量后端" —— 这正是最该被挡住的那类失败：
它不红，它只是把要验的东西换掉了。

顺带说明为什么后端是**方法**参数而不是结构体参数（同一件事的另一个痕迹）：
Mojo 1.0.0（ed45d567）在"参数化结构体 + 会抛错误的构造函数"上会直接把编译器
进程搞崩 —— 最小复现是

    struct S[n: Int]:
        def __init__(out self) raises: ...
    var s = S[0]()      # 编译器在这里 crash

放到方法上（`prefill[backend]`）既能绕开这个崩溃，语义也更准：后端是这一次
前向的属性，不是模型实例的属性。

Run:
    pixi run test-backend-guard
"""

from alofa.model.arch.qwen import QwenForward

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime WEIGHTS = FIXTURE + "/weights"
comptime CONFIG = FIXTURE + "/config.tsv"


def main() raises:
    # `7` 不是任何已知后端。它必须被 `uses_vector_backend` 里的 comptime
    # assert 挡在编译期，而不是在下面这行悄悄走标量分支。
    var model = QwenForward(WEIGHTS, CONFIG, 8)
    var ids = List[Int]()
    ids.append(1)
    var logits = model.prefill[7](ids)
    print(Float32(logits[unsafe_offset=0]))
