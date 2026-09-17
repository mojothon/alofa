"""零分配门的常驻负向对照 —— 这个文件**故意违规**，不参与编译。

`tests/unit/test_kv_room.mojo` 把它当**文本**读，只做字符串扫描，所以它不必能编译
通过。它的存在让"扫描确实会报红"成为一条常驻断言，而不是我说过的一句话：忙碌循环
那一节的源码扫描若哪天失效（拼错的 token、被注释掉的行、换了文件），这条会先红。

它模仿的是最容易顺手加进 KV 房间的四种会增长的东西：

- `List`：把这一拍被驱逐的节点记下来，回头一起上报 —— 每拍都增长，正是这一层要
  挡住的；房间的账全是编译期定长的 `InlineArray`，就是为了没有这种增长。
- `Dict`：请求 id → 块列表。好用，而每次插入都在分配。
- `DynamicVector`：看起来像"固定"的容器，其实超容量就重新分配。
- `alloc(`：直接向 arena 要一块"临时"的空间，而忙碌循环里没有临时。
"""

from std.collections import Dict, List, Set

from alofa.core.memory import Arena


struct GrowableRoom:
    """一个会随并发增长的房间：正是房间里不该有的东西。"""

    var evicted: List[Int]
    var by_request: Dict[Int, Int]
    var touched: Set[Int]
    var scratch: DynamicVector[Int]
    var arena: Arena

    def __init__(out self):
        self.evicted = List[Int]()
        self.by_request = Dict[Int, Int]()
        self.touched = Set[Int]()
        self.scratch = DynamicVector[Int]()
        self.arena = Arena(1 << 16)

    def reclaim(mut self, need: Int) raises:
        var tmp = self.arena.alloc(need * 8)
        self.evicted.append(need)
        self.by_request[need] = 1
        self.touched.add(need)
        self.scratch.push_back(need)
        _ = tmp
