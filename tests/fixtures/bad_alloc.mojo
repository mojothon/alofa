"""零分配门的抗原：这个文件**永远不会被编译**，只会被 `test_scheduler.mojo`
当作文本扫描。

它模拟的是"有人给调度器加了一个会增长的堆容器"这一回归。扫描器必须判它违规，
否则那条门恒真 —— 一个不会失败的门等于没有门。
"""

from alofa.core.memory import Arena


struct LeakyScheduler:
    var buf: List[Int]

    def __init__(out self):
        self.buf = List[Int]()
        var arena = Arena(4096)
        arena.keep_alive()

    def step(mut self):
        self.buf.append(1)
        var text = String(1)
        print(text)
