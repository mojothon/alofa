"""零分配门的常驻负向对照 —— 这个文件**故意违规**，不参与编译。

`tests/unit/test_batch_pool.mojo` 把它当**文本**读，只做字符串扫描，所以它不必能
编译通过。它存在的意义是让"扫描确实会报红"成为一条常驻断言，而不是我说过的一句话。

它模仿的是最容易被顺手加进批张量池的三种会增长的东西：

- `List`：按参与请求累积借用，每来一个请求 append 一次 —— 忙碌循环里无边界的
  增长正是这一层要挡住的。
- `Dict`：把请求 id 映射到行区间，确实好用，而它的每一次插入都在分配。
- `String`：只为拼一句好看的错误信息。这里的写法是具名错误码加固定句子，
  见 `src/alofa/engine/batch.mojo` 的说明。
"""




struct GrowableBatchView:
    """一个会随批次增长的表：正是批张量池里不该有的东西。"""

    var rows: List[Int]
    var by_request: Dict[Int, Int]
    var labels: List[String]

    def __init__(out self):
        self.rows = List[Int]()
        self.by_request = Dict[Int, Int]()
        self.labels = List[String]()


def describe(request: Int, rows: Int) -> String:
    return String("request=") + String(request) + String(" rows=") + String(rows)
