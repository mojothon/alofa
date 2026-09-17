"""零分配门的常驻负向对照 —— 这个文件**故意违规**，不参与编译。

`tests/unit/test_paged_attention.mojo` 把它当**文本**读，只做字符串扫描，所以它
不必能编译通过；它存在的意义是让"扫描确实会报红"成为一条常驻断言，而不是我口头
说过的一句话。一个不会失败的门等于没有门。

它模仿的是最容易被顺手加进分页内核的两种会增长的容器：一个按 token 累积的
`List`（每来一段 run 就 append 一次），和一处只为拼错误信息而建的 `String`
（内核的错误信息一律是固定的，见 `src/alofa/kernels/cpu/paged.mojo` 的说明）。
"""

from alofa.core.tensor import TensorView


struct GrowablePagedView:
    """每条 run 一段：`List` 会随上下文长度增长，正是内核里不该有的东西。"""

    var runs: List[Int]
    var names: List[String]

    def __init__(out self):
        self.runs = List[Int]()
        self.names = List[String]()


def describe(view: TensorView) raises -> String:
    return String("rows=") + String(view.numel())
