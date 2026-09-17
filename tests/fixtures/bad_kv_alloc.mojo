"""抗原：给 KV 视图加一个会增长的堆容器。

这份文件**故意不参与编译**（它不在任何包的导入链上），只被
`tests/unit/test_kv_pool.mojo` 当作文本扫描：零分配门必须判它违规。若扫描判不出
来，那条门就是恒真的，证明不了任何事。
"""

from alofa.runtime.kv import MAX_BLOCKS


struct GrowableKvView:
    """一个"看起来合理"的写法：块表用 List，随请求增长。

    它之所以必须被拦下，不是因为 List 慢，而是因为一旦块表能增长，"稳态零堆分配"
    就从类型保证退化成了"写代码的人记得住" —— 而后者没有任何东西在守。
    """

    var blocks: List[Int]
    var refcnt: List[Int]

    def __init__(out self):
        self.blocks = List[Int]()
        self.refcnt = List[Int]()
        for i in range(MAX_BLOCKS):
            self.refcnt.append(0)

    def retain(mut self, block: Int):
        self.blocks.append(block)
        self.refcnt[block] += 1
