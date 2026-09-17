"""零分配门的常驻负向对照 —— 这个文件**故意违规**，不参与编译。

`tests/unit/test_batch_executor.mojo` 把它当**文本**读，只做字符串扫描，所以它不必
能编译通过。它的存在让"扫描确实会报红"成为一条常驻断言，而不是我说过的一句话：忙碌
循环那一节的源码扫描若哪天失效（拼错的 token、被注释掉的行、换了文件），这条会先红。

它模仿的是最容易顺手加进忙碌循环的四种会增长的东西：

- `List`：按参与请求累积借用，每来一个请求 append 一次 —— 每步都增长正是这一层要
  挡住的，池子大小固定的意义就是没有这种增长。
- `Dict`：把请求 id 映射到行区间，确实好用，而它的每一次插入都在分配。
- `Set`：为去重构造的集合，只为一次查询。
- `DynamicVector`：看起来像"固定"的容器，其实超容量就重新分配。
"""


struct GrowablePlan:
    """一个会随批次增长的执行计划：正是执行器里不该有的东西。"""

    var row_owner: List[Int]
    var by_request: Dict[Int, Int]
    var touched: Set[Int]
    var scratch: DynamicVector[Int]

    def __init__(out self):
        self.row_owner = List[Int]()
        self.by_request = Dict[Int, Int]()
        self.touched = Set[Int]()
        self.scratch = DynamicVector[Int]()

    def note(mut self, request: Int, row: Int) raises:
        self.row_owner.append(row)
        self.by_request[request] = row
        self.touched.add(request)
        self.scratch.push_back(row)
