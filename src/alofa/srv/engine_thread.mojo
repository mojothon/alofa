"""把前向从事件循环上搬下来：N 条独占线程（默认 1 条）+ 两个邮箱（请求进、结果出）。

它改的是哪一处
--------------
`srv/loop.mojo` 的循环现在只做 I/O：读字节、切请求、写字节、收连接。前向（一次
`handler.handle`，以及流的一帧 `handler.stream_next`）全部在**另一条线程**上跑
（roadmap 3.2b）。改之前，一次前向会占住这条循环 —— 一个 20 秒的生成期间，另外
99 条连接连"读一个请求"都做不到；改之后那 20 秒里循环照常 accept / 读 / 写。

并发生成（`engines > 1` 时：线程 N 条，每条一份 handler）
--------------------------------------------------------
**同时生成的条数 = `engines`**（roadmap 3.2c）：`engines` 条 engine 线程，每条一份
**handler**（`Twinable.spawn_twin`）。两条 2 秒的生成在 `engines=2` 下是 2 秒，在
`engines=1` 下是 4 秒 —— 门里两种都量着（`pixi run test-engine`）。

为什么每条线程必须**各有一份** handler（不能 N 条线程共用一份 + 一把锁）：handler
里面装的是"一条生成的相位"（`QwenForward` 的 KV、`ChatHandler` 的流状态）。锁得住
内存、挡不住语义 —— 两条生成共用一份 KV 会互相写坏，而它的症状只是"偶尔答错一次"。

代价（明文写在这儿，不是疏漏）：**N 条 = N 份权重**。`ModelService.spawn_twin` 是
重新加载一份（现在还没有"只读权重共享一层"那个接缝），所以 `engines=2` 就要两份
权重的内存。默认 `engines=0`（`ALOFA_ENGINE_THREADS=0` = 生成就跑在 reactor 循环上，
串行）；`>=1` 才走这条路 —— ⚠️ 那条路现在是**实验**的（真模型 + 客户端中途断开会把
请求搞坏，见账本 §8），所以服务启动时会对着它当场打 `[warn]`。这条代价只在你把它调大
之后才出现。

一条流要一直在同一条线程上（`Job.thread`）
------------------------------------------
流的相位在**那份 handler** 里，所以一帧的 job 必须回到"开始这条流的那条线程"。
换一条线程去生成下一帧 = 从一条不存在的流上取一帧 —— 症状是客户端收到响应头之后
什么都没有。所以 job 带上 thread，而"哪个槽位归哪条线程"由循环在**第一次派活**时
定下来（`Loop.owner_for`）之后不变 —— 变了就会让一条连接上的多个请求换顺序应答。

两条线、一把锁、以及锁为什么只有一把
------------------------------------
- **handler 只在 engine 线程上被碰**，所以 handler 内部（权重、KV、流的相位）
  **没有锁** —— 这正是"独占线程"这四个字的全部内容，也是它没有退化成"到处加锁"
  的原因。
- 锁只在**交接点**：邮箱（请求进、结果出）。临界区里只有几次 `append` / 取首
  元素，没有 I/O、没有前向 —— 一把锁护住两个方向，是因为这两个方向的临界区都
  短到不值得分开。
- **结果回来靠 wakeup**（`Reactor.wakeup()`，跨线程；能力门
  `tests/capability/test_reactor.mojo` 钉着它），**不是**靠 poll 超时"顺带"发现 ——
  后者会让每条流的每一帧都晚一个 poll 周期（50 ms）。

⚠️ 箱子必须活到 join 之后（这条踩过一次，崩在 `pthread_mutex_lock`）
------------------------------------------------------------------
Mojo 是**用完即析构**（ASAP）：一个值在它最后一次被提到之后就结束生命。所以
"取了箱子的地址 → spawn → join"这条顺序里，箱子的最后一次被提到是 spawn 那一
行 —— join 还没开始，箱子已经被析构，而线程正在用它里面的 mutex。症状不是报错，
是**在新线程里崩**。所以调用方必须在 join **之后**再碰一次箱子（见
`Loop.run_threaded` 末尾那个"生命周期锚点"）。用 `malloc` 把箱子放到堆上也能绕
开它（`flare.runtime.scheduler` 就是那么做的），但那要手工管释放；栈上的箱子
加一个锚点更短，也更容易在 review 里看见。

三个不这么做就会偶尔错一次的坑
------------------------------
1. **槽位会换主人**：一条连接被收掉之后，它的槽位号会给下一条连接用。在途的结果
   只带槽位号的话，上一条连接的应答会发给**下一条**连接 —— 一次"偶发的响应内容
   不对"，而且只在"客户端恰好在生成期间断开"时出现。所以每个槽位带一个 **gen**
   （`drop` 时自增），结果回来先对 gen，对不上就丢。
2. **engine 线程不能让异常跑出去**：pthread 没有异常通道，一条冒出去的异常等于
   "那条连接永远等不到应答，而且没人知道为什么"。所以入口里每一条路都自己兜住，
   错误变成一帧（流）或一个 500（请求）。
3. **邮箱必须有上界**：循环派活快、engine 生成慢的时候，无界队列会把"客户端的
   数量"变成"进程的内存"。满了就**不派**（请求留在连接缓冲里，下一轮再来）——
   那才是背压。派出去又塞不进邮箱的形状是错的（请求已经被 `take` 消费掉了，没有
   回头路），所以派活之前先问 `has_room()`。
"""

from std.collections import List
from std.ffi import external_call
from std.memory import Pointer
from std.sys import size_of

from flare.runtime import Reactor
from flare.runtime._thread import ThreadHandle
from flare.runtime.mutex import Mutex

from alofa.srv.http import (
    SSE_CONTENT_TYPE,
    HttpResponse,
    bytes_of_text,
    json_response,
    parse_request,
    serialize,
    serialize_stream_head,
)
from alofa.srv.server import Handler

# pthread 的 `void *`。与 `flare.runtime._thread` 里那个同名别名是同一个类型
# （`Pointer[UInt8, MutUntrackedOrigin]`），跨线程只认这一个形状。
comptime OpaquePtr = Pointer[UInt8, MutUntrackedOrigin]

# handler 的类型约束。`Deinitable` 不是摆设：箱子要在 `join` 之后才析构，而那
# 需要调用方把它留住 —— 没析构函数的类型留不住（也用不着留）。
comptime EngineHandler = Handler & Movable & Deinitable & Twinable


trait Twinable:
    """能再造一份"自己"的（engine 线程池要每条线程一份，见文件头）。

    ⚠️ 为什么交出来的是一个**地址**而不是一份值：Mojo 1.0 的 trait 方法**不能把
    `Self` 当返回类型**（`def twin(self) -> Self` 在 trait 里连解析都过不去，
    `out twin: Self` 一样），trait 也不支持类型参数（`trait T[X]` 报"参数不能出现
    在 trait 里"）。三条路都堵着，所以只剩"自己 malloc 一份、把地址交出来"，由
    调用方用它（`heap_take`）取回并负责释放。
    """

    def spawn_twin(self, index: Int) raises -> Int:
        """再造一份自己（给第 `index` 条 engine 线程），放在堆上，返回它的地址。

        一份 = 一份权重与一条生成的相位（不是"共享只读权重 + 各自一份 KV"）：那样
        做要动到 model 那一层，而这一层要的是"语义上就是另一个 handler"。代价写
        在文件头：**N 条线程 = N 份权重**。

        ⚠️ 为什么要带 `index`：**每条线程一份的计数器必须错开**。响应里的
        `chatcmpl-N` 是 handler 自己数的，两份 handler 都从 1 开始 = 两条并发生成
        交回同一个 id —— 客户端按 id 去重就会丢一条。所以孪生出来的那份把起点拨到
        `1 + index * ID_STRIDE`（见 `srv/openai.mojo`）。
        """
        ...


def heap_place[T: Movable](var value: T) raises -> Int:
    """把一份值放到堆上，返回地址（`spawn_twin` 那一半的通用写法）。"""
    var raw = external_call["malloc", Int](size_of[T]())
    var slot = Pointer[T, MutUntrackedOrigin](unsafe_from_address=raw)
    slot.unsafe_write(value^)
    return raw


def heap_take[T: Movable](raw: Int) raises -> T:
    """把 `heap_place` 放上去的那份取回来，并还掉那块内存。

    取回来而不是"就地用"：值到了栈上，退出作用域时会**析构**（权重、KV 都还回
    去），而留在堆上不取等于那份内存要等进程结束才归还。
    """
    var slot = Pointer[T, MutUntrackedOrigin](unsafe_from_address=raw)
    var out = slot.unsafe_take_pointee()
    _ = external_call["free", NoneType](raw)
    return out^

# 一个 job 是"这条连接上要 engine 做的一件事"。
# - `JOB_REQUEST`：一个完整请求（非流式一次答完；流式只交回头）。
# - `JOB_STREAM_STEP`：在途那条流的**一帧**。一帧一次往返，所以流也走邮箱 ——
#   engine 不知道对端收得动收不动，而循环知道（它握着发送队列）。
comptime JOB_REQUEST = 0
comptime JOB_STREAM_STEP = 1
# - `JOB_STREAM_END`：一条流**没走完**就结束了（对端断了 / 连接被丢），让 engine 把它
#   那一份收掉（`Handler.stream_end`，幂等）。
#
#   为什么它是**一件事**而不是循环直接调 handler：handler 归 engine 线程独占（见
#   `EngineBox` 的注释），循环这一侧从头到尾碰不得 —— 从循环调一次 `stream_end`
#   等于把"每条线程一份相位"这件事撕开一个口子，而症状会是"偶尔答错一帧"那种最难
#   归因的失败。派一件活过去，收尾仍然发生在持有相位的那条线程上。
comptime JOB_STREAM_END = 2

# 一个 result 是"engine 交回来的字节"。
# - `RES_BYTES`：排进这条连接的发送队列，这条请求**还没答完**（流的一帧）。
# - `RES_DONE`：这条请求答完了（`served` 在这里 +1）。`close` 决定写完要不要关。
comptime RES_BYTES = 0
comptime RES_DONE = 1
# 一条流的**第一截**（只有响应头）。循环拿到它才知道"这个槽位上开始了一条流"，
# 接下来要一帧一次往返地派 `JOB_STREAM_STEP`。
#
# 为什么它不能复用 `RES_BYTES`：一帧也是 `RES_BYTES`，而"这一截是流的头"与"这一截
# 是一段字节"是两个判断。合成一个的话，循环不知道该开始喂帧 —— 症状是客户端只收到
# 一个响应头，然后一直挂到空闲超时（看起来像"服务偶尔不答流式请求"）。
comptime RES_STREAM_HEAD = 2

# 邮箱里最多有多少个 job。它是**有意的**上界（见文件头第 3 条）：等于"最多有多少
# 条请求在排队等生成"，而不是"最多能连多少条"（那是 `MAX_CONNS`）。
comptime MAILBOX_CAP = 256

# engine 线程没活干时睡多久。睡是为了不空转满一个核；1 ms 是"空转的代价"与"派活
# 的延迟"之间的一个数 —— 结果回来的那条路**不走这里**（走 wakeup），所以这个数
# 不影响任何一条响应的延迟，只影响"活派进来之后 engine 多久睁眼"。
comptime ENGINE_IDLE_US = 1000


def opaque_of(imm addr: Int) -> OpaquePtr:
    return OpaquePtr(unsafe_from_address=addr)


# 过邮箱的那两样东西自带的**自证**：一个魔数（放在结构体的**第一个**字段 —— 被前一块
# 写穿时它先坏）加一份"内层 List 有多长"的副本（长度对不上 = `List` 那三个字被踩了）。
#
# 为什么要有它：实测踩踏的表现是"几十条之后在别处随机发作"（`Job.slot` 变成一个堆地址、
# 请求行开头乱码、随机 abort）。那种失败没法归因 —— 它离真凶太远。有了自证，被踩的那
# 一次会在**过邮箱的那一刻**变红，而且能分出"进邮箱之前就坏了"与"在邮箱里坏的"。
comptime MAX_JOB_BYTES = 8192
comptime MAX_RES_BYTES = 16384
comptime JOB_MAGIC = 0xA10FA001
comptime RES_MAGIC = 0xA10FA002


struct Job(Movable):
    """一件派给 engine 的事。

    `slot` 与 `gen` 一起才认得一条连接：号会复用，gen 不会。
    """

    var magic: Int
    var slot: Int
    var gen: Int
    var kind: Int
    # 这件活归哪条 engine 线程（见文件头"一条流要一直在同一条线程上"）：流的相位
    # 在那条线程的 handler 里，换线程 = 从一条不存在的流上取下一帧。
    var thread: Int
    # 请求的**原始字节**（头 + 体，正好一段），只有 `JOB_REQUEST` 用得上。
    #
    # 为什么不是直接带一个 `HttpRequest`：请求是 `Conn` 从入站缓冲上切出来的，它
    # 带着那条缓冲的借用关系（`List` 不 ImplicitlyCopyable 那一族报错、以及
    # "field destroyed out of the middle of a value" 都是它惹的），挪不进邮箱；
    # 而拷出来的一份 `List[UInt8]` 是自己的，可以。代价是 engine 线程要**再解析
    # 一次**（`parse_request` 只做切分与拷贝，一次几百字节，而它本来就要跑一次
    # 前向），换来的是"跨线程只过一种形状的东西"。
    var raw: InlineArray[UInt8, MAX_JOB_BYTES]
    var n_raw: Int

    def __init__(out self, slot: Int, gen: Int, kind: Int, thread: Int):
        self.magic = JOB_MAGIC
        self.slot = slot
        self.gen = gen
        self.kind = kind
        self.thread = thread
        self.raw = InlineArray[UInt8, MAX_JOB_BYTES](fill=0)
        self.n_raw = 0

    def __init__(
        out self, slot: Int, gen: Int, kind: Int, thread: Int, var raw: List[UInt8]
    ):
        """带上请求的那一个（`JOB_REQUEST`）。"""
        self.magic = JOB_MAGIC
        self.slot = slot
        self.gen = gen
        self.kind = kind
        self.thread = thread
        self.raw = InlineArray[UInt8, MAX_JOB_BYTES](fill=0)
        self.n_raw = len(raw)
        if self.n_raw > MAX_JOB_BYTES:
            print("  [srv] a request was truncated to MAX_JOB_BYTES")
            self.n_raw = MAX_JOB_BYTES
        var i = 0
        while i < self.n_raw:
            self.raw[i] = raw[i]
            i += 1

    def intact(self) -> Bool:
        """自己还像一件活吗（过邮箱**前后**都该为真）。

        `JOB_REQUEST` 还要看**内容的头一个字节**：HTTP 方法就是那几个大写字母
        （POST/GET/OPTIONS…）。长度对而内容坏是最需要盯住的一种 —— 实测那条
        `[line=p?!/chat/completions HTTP/1.1]` 正是这个形状：**尾巴是好的、开头是
        乱码**，长度一点没变。只看长度会把它漏掉。
        """
        if self.magic != JOB_MAGIC:
            return False
        if self.slot < 0 or self.kind < JOB_REQUEST or self.kind > JOB_STREAM_END:
            return False
        if self.thread < 0 or self.n_raw < 0 or self.n_raw > MAX_JOB_BYTES:
            return False
        if self.kind == JOB_REQUEST and self.n_raw > 0:
            var head = self.raw[0]
            if head < 65 or head > 90:  # 'A'..'Z'
                return False
        return True

    def dump(self) -> String:
        var head = "head="
        var i = 0
        while i < 8 and i < self.n_raw:
            head += String(Int(self.raw[i])) + ","
            i += 1
        return (
            "magic="
            + String(self.magic)
            + " slot="
            + String(self.slot)
            + " gen="
            + String(self.gen)
            + " kind="
            + String(self.kind)
            + " thread="
            + String(self.thread)
            + " raw="
            + String(self.n_raw)
            + "/"
            + String(self.n_raw)
            + " "
            + head
        )


struct Result(Movable):
    """engine 交回来的一截字节。

    `kind` 与 `close` 是两个独立的判断，合成一个会让"流结束"与"响应说完要关连接"
    变成同一件事 —— 它们在 SSE 上**恰好同时为真**，于是那条路永远测不出另一个写
    错了。
    """

    var magic: Int
    var slot: Int
    var gen: Int
    var kind: Int
    var close: Bool
    var bytes: InlineArray[UInt8, MAX_RES_BYTES]
    var n_bytes: Int

    def __init__(out self):
        self.magic = RES_MAGIC
        self.slot = -1
        self.gen = -1
        self.kind = RES_DONE
        self.close = False
        self.bytes = InlineArray[UInt8, MAX_RES_BYTES](fill=0)
        self.n_bytes = 0

    def __init__(
        out self,
        slot: Int,
        gen: Int,
        kind: Int,
        close: Bool,
        var bytes: List[UInt8],
    ):
        self.magic = RES_MAGIC
        self.slot = slot
        self.gen = gen
        self.kind = kind
        self.close = close
        self.bytes = InlineArray[UInt8, MAX_RES_BYTES](fill=0)
        self.n_bytes = len(bytes)
        if self.n_bytes > MAX_RES_BYTES:
            print("  [srv] a response was truncated to MAX_RES_BYTES")
            self.n_bytes = MAX_RES_BYTES
        var i = 0
        while i < self.n_bytes:
            self.bytes[i] = bytes[i]
            i += 1

    def intact(self) -> Bool:
        """自己还像一截应答吗（过邮箱**前后**都该为真）。"""
        return (
            self.magic == RES_MAGIC
            and self.slot >= -1
            and self.kind >= RES_BYTES
            and self.kind <= RES_STREAM_HEAD
            and self.n_bytes >= 0
            and self.n_bytes <= MAX_RES_BYTES
        )

    def dump(self) -> String:
        return (
            "magic="
            + String(self.magic)
            + " slot="
            + String(self.slot)
            + " gen="
            + String(self.gen)
            + " kind="
            + String(self.kind)
            + " bytes="
            + String(self.n_bytes)
            + "/"
            + String(self.n_bytes)
        )


struct Mailbox(Movable):
    """两个方向各一个 FIFO，一把锁护住。

    为什么锁在这一层而不是在循环里：临界区的边界必须**看得见** —— 只有这几行是
    共享的，前向与 I/O 都不在里面。锁的范围一旦跟着调用方走，就会有人把它抱过一次
    `poll`。
    """

    var mutex: Mutex
    var jobs: List[Job]
    var results: List[Result]
    var stop: Bool
    # 已经认领出去的 engine 编号数（编号只能**在线程里**领：spawn 之前分好的话，
    # 你不知道哪条线程先跑起来）。它借邮箱这把锁 —— 认领只发生一次，不值得再开
    # 一把。
    var threads: Int
    # 诊断计数：四个数一起看，才能分开"活没被 engine 取走"（pushed > taken）与
    # "取走之后结果丢了"（pushed == taken）。只在卡住时读，不进热路径的判定。
    var jobs_pushed: Int
    var jobs_taken: Int
    var results_pushed: Int
    var results_taken: Int
    var engine_busy: Int
    var engine_polls: Int
    var engine_runs: Int
    # 账**第一次**对不上的地方（0 = 没对不上；1/2/3/4 = 在 `push_job`/`take_job`/
    # `push_result`/`take_result` 里第一次发现）。
    #
    # 为什么需要它：`results_pushed` 只有 `push_result` 一处自增、`jobs_taken` 只有
    # `take_job` 一处自增，所以"一件活回一截应答"意味着这两个数**恒等**。实测出现过
    # `rpush=162 > jtake=160` —— 单靠打印分不清是"我读错了"还是"字段被踩了"，而这两条
    # 的后续动作完全相反。下面那条恒等式（pushed - taken == len(queue)）一旦破，就是
    # 字段或 `List` 被写穿，不是逻辑错。
    var vio: Int
    # 最近 24 件**被取走的活** / **交回的结果**，编码成 `slot*1e6+gen*1e4+kind*1e2+n`。
    # 卡住时把这两条打出来，就能回答"那一帧到底进没进 engine"：进了取活环、
    # 没进结果环 = 丢在 `_run_job`/`push_result` 之间；两个环都没有 = `take_job`
    # 根本没看见它（而邮箱已经空了 —— 那就是被别的路径吃掉了）。
    var tj: List[Int]
    var tr: List[Int]

    def __init__(out self) raises:
        self.mutex = Mutex()
        self.jobs = List[Job]()
        self.results = List[Result]()
        # 一次把容量留到上界，**让这两个 `List` 永不扩容**：扩容要把每个元素搬到新
        # 缓冲上，而元素是带一块自有 `List` 的 `Job`/`Result` —— 不搬就没有那次机会。
        #
        # ⚠️ 它是**被量过才留下来的**，不是顺手：这里原本是为了治那个 UAF（活躺在邮箱
        # 里的时候字节被踩），假设是"扩容时旧缓冲上那份已被挪走的元素又被析构了一次，
        # 于是把新元素正在用的缓冲还了回去"。实测**否掉**：留够容量之后自证照红（20 条
        # 里 5 次），服务仍旧死。留着它只因为"热路径上不 realloc"本身成立，而这条否掉
        # 的记录比那点分配更值钱 —— 下一个人不必再猜一遍。
        self.jobs.reserve(MAILBOX_CAP)
        self.results.reserve(MAILBOX_CAP)
        self.stop = False
        self.threads = 0
        self.jobs_pushed = 0
        self.jobs_taken = 0
        self.results_pushed = 0
        self.results_taken = 0
        # engine 自己在干什么：`busy=1` = 正在 `_run_job` 里（卡在这里 = 引擎不动了）；
        # `polls` = 空转问活的次数（在涨 = 线程活着且闲着）；`runs` = 干完的件数。
        # 这三个数放在这里，是因为只有它们能把"`take` 走了却没结果"分成两半。
        self.engine_busy = 0
        self.engine_polls = 0
        self.engine_runs = 0
        self.vio = 0
        self.tj = List[Int]()
        self.tr = List[Int]()

    def claim_thread(mut self) raises -> Int:
        """领一个 engine 编号（0, 1, 2 …）。第一条起来的线程不一定是第一条 spawn
        的 —— 所以编号不能在 spawn 之前分好。"""
        self.mutex.lock()
        var mine = self.threads
        self.threads += 1
        self.mutex.unlock()
        return mine

    # ── 请求方向（循环 → engine） ──────────────────────────────────────────

    def has_room(self) raises -> Bool:
        """还能派一个吗。派之前**先问**（见文件头第 3 条）：派出去再发现塞不进
        来，那个请求已经被 `take` 消费掉了，没有回头路。"""
        self.mutex.lock()
        var room = len(self.jobs) < MAILBOX_CAP
        self.mutex.unlock()
        return room

    def push_job(mut self, var job: Job) raises -> Bool:
        # 进邮箱**之前**先自证：这里就坏 = 踩它的在循环那一侧（构造它的地方到这儿之间）。
        if not job.intact():
            print("  [srv] a job was already damaged before the mailbox: " + job.dump())
        self.mutex.lock()
        if len(self.jobs) >= MAILBOX_CAP:
            self.mutex.unlock()
            return False
        self.jobs.append(job^)
        self.jobs_pushed += 1
        if self.vio == 0 and self.jobs_pushed - self.jobs_taken != len(self.jobs):
            self.vio = 1
        self.mutex.unlock()
        return True

    def take_job(mut self, thread: Int, mut job: Job) raises -> Bool:
        """取**这条线程**最老的一个 job，写进 `job`（调用方先初始化好它 —— Mojo
        不允许 `out` 参数与返回值同时存在，而"取到了吗"必须能返回）。

        只取归自己的那一个：流的相位在自己那份 handler 里，替别人干一帧等于从一条
        不存在的流上取字节（见文件头）。扫一遍而不是"每线程一个队列"，是因为
        队列一共就 `MAILBOX_CAP` 个，而每线程一个 `List` 会把"一个上界"变成"N 个
        上界"（线程越多能排队的越多 —— 那正是这个上界要挡的东西）。

        顺序是**派发的顺序**：不保持它，一条连接上的第二个请求就可能先于别人的
        第一个请求被生成 —— 那是一次"看起来随机的慢"。
        """
        self.mutex.lock()
        var i = 0
        while i < len(self.jobs):
            if self.jobs[i].thread == thread:
                job = self.jobs.pop(i)
                self.jobs_taken += 1
                self.tj.append(
                    job.slot * 1000000 + job.gen * 10000 + job.kind * 100
                )
                if len(self.tj) > 24:
                    _ = self.tj.pop(0)
                if self.vio == 0 and self.jobs_pushed - self.jobs_taken != len(
                    self.jobs
                ):
                    self.vio = 2
                self.mutex.unlock()
                # 出邮箱**之后**再自证：进来时是好的、出去时坏了 = 踩它的在邮箱里
                # （`List` 的搬运、或者别处这次写穿到这儿）。
                if not job.intact():
                    print("  [srv] a job was damaged in the mailbox: " + job.dump())
                return True
            i += 1
        self.mutex.unlock()
        return False

    # ── 结果方向（engine → 循环） ──────────────────────────────────────────

    def push_result(mut self, var res: Result) raises:
        if not res.intact():
            print(
                "  [srv] a result was already damaged before the mailbox: "
                + res.dump()
            )
        self.mutex.lock()
        self.tr.append(
            res.slot * 1000000 + res.gen * 10000 + res.kind * 100 + min(
                res.n_bytes, 99
            )
        )
        if len(self.tr) > 24:
            _ = self.tr.pop(0)
        self.results.append(res^)
        self.results_pushed += 1
        if self.vio == 0 and self.results_pushed - self.results_taken != len(
            self.results
        ):
            self.vio = 3
        self.mutex.unlock()

    def first_result(mut self, mut slot: Int, mut gen: Int) raises -> Bool:
        """队首那个结果是给谁的 —— **只看不取**（两个值都写进调用方的变量）。

        循环要先看一眼：队首那条连接正在背压的话，结果必须**留在邮箱里**（取出来
        就排不进发送队列了，而丢掉它等于让客户端永远等下去）。
        """
        self.mutex.lock()
        if len(self.results) == 0:
            self.mutex.unlock()
            slot = -1
            gen = -1
            return False
        slot = self.results[0].slot
        gen = self.results[0].gen
        self.mutex.unlock()
        return True

    def take_result(mut self, mut res: Result) raises -> Bool:
        self.mutex.lock()
        if len(self.results) == 0:
            self.mutex.unlock()
            return False
        res = self.results.pop(0)
        self.results_taken += 1
        if self.vio == 0 and self.results_pushed - self.results_taken != len(
            self.results
        ):
            self.vio = 4
        self.mutex.unlock()
        if not res.intact():
            print("  [srv] a result was damaged in the mailbox: " + res.dump())
        return True

    # ── 在途与停止 ────────────────────────────────────────────────────────

    def pending(self) raises -> Int:
        """还有多少件"派出去但没答完"的事（含已经生成好、还没排进发送队列的）。

        循环用它算"还能派几个"：多派一个请求是允许的（答完会被算进 `served`），
        多派到永远答不完是不允许的 —— 那会让"答完 N 条就停"的循环真的停不下来。
        """
        self.mutex.lock()
        var n = len(self.jobs) + len(self.results)
        self.mutex.unlock()
        return n

    def begin_job(mut self) raises:
        """记"我进 `_run_job` 了"。

        ⚠️ **写也必须在锁内**：只读加锁是不够的 —— 读侧拿到的是"别的线程写的普通字段"，
        而这类写不保证对读侧可见（`-O2` 还会把读侧的读提到循环外，见 `read_counters`）。
        `busy` 罩着整段生成，锁不能抱那么久，所以拆成一头一尾两次很短的加锁更新。
        """
        self.mutex.lock()
        self.engine_busy = 1
        self.mutex.unlock()

    def end_job(mut self) raises:
        """记"这一件干完了"。`busy` 归零与 `runs` 自增**在同一个临界区**里：分开写会
        让读侧看见"不忙但没干完"（或反过来）的切片，而这两个数正是要对账的。"""
        self.mutex.lock()
        self.engine_busy = 0
        self.engine_runs += 1
        self.mutex.unlock()

    def tick_poll(mut self) raises:
        """记一次空转轮询（同样要加锁，理由见 `begin_job`）。"""
        self.mutex.lock()
        self.engine_polls += 1
        self.mutex.unlock()

    def read_counters(
        mut self,
        mut jpush: Int,
        mut jtake: Int,
        mut rpush: Int,
        mut rtake: Int,
        mut busy: Int,
        mut polls: Int,
        mut runs: Int,
        mut jq: Int,
        mut rq: Int,
        mut vio: Int,
    ) raises:
        """七个诊断数**在锁内**一次读完。

        为什么非要这一层：这七个字段都是**跨线程**的（engine 线程写、循环读），直接
        读字段不只是"数可能不对"—— `-O2` 在这一侧看不到任何写，于是它把那次读提到
        循环外，之后每一轮都拿到同一个陈旧值。实测正是这样把 `busy=1` 钉在打印里，
        让人误以为 engine 卡在 `_run_job` 里（其实它在空转，`polls` 一直在涨）。

        七个一起读还有一个理由：分开读会拿到**不同时刻**的切片，而这几个数是要互相
        对账的（`jtake` 比 `rpush` 多出来的那件就是丢的那件）。
        """
        self.mutex.lock()
        jpush = self.jobs_pushed
        jtake = self.jobs_taken
        rpush = self.results_pushed
        rtake = self.results_taken
        busy = self.engine_busy
        polls = self.engine_polls
        runs = self.engine_runs
        # `jpush - jtake` 必须等于 `jq`（每个派出去的活要么还在队里、要么已被取走）
        # —— 对账的三个数一起读，破了就是字段/列表被写穿，见 `vio`。
        jq = len(self.jobs)
        rq = len(self.results)
        vio = self.vio
        self.mutex.unlock()

    def counters_text(mut self) raises -> String:
        """十一个诊断数**在锁内**拼成一个串（不走 `mut` 出参）。

        为什么要另开一条：`read_counters` 的十一个 `mut` 出参在调用方是十一个局部
        变量，而服务里打出来的数字出现过**在锁内不可能成立的组合**（`rpush > jtake`
        —— `push_result` 只跟着被取走的活走）。最小复现（`scripts/tmp_mailbox_race.mojo`）
        证明锁与 `List` 这一层不丢更新，那么剩下的可能就只有"出参这条路把值传错了"。
        两条路同时打出来，一样就排除、不一样就定案。
        """
        self.mutex.lock()
        var s = (
            "jpush="
            + String(self.jobs_pushed)
            + " jtake="
            + String(self.jobs_taken)
            + " rpush="
            + String(self.results_pushed)
            + " rtake="
            + String(self.results_taken)
            + " busy="
            + String(self.engine_busy)
            + " polls="
            + String(self.engine_polls)
            + " runs="
            + String(self.engine_runs)
            + " jq="
            + String(len(self.jobs))
            + " rq="
            + String(len(self.results))
            + " vio="
            + String(self.vio)
        )
        self.mutex.unlock()
        return s^

    def read_trace(mut self, mut jobs: List[Int], mut res: List[Int]) raises:
        """拷一份最近取走的活与交回的结果（编码见 `tj` 的注释）。"""
        self.mutex.lock()
        for e in self.tj:
            jobs.append(e)
        for e in self.tr:
            res.append(e)
        self.mutex.unlock()

    def set_stop(mut self) raises:
        self.mutex.lock()
        self.stop = True
        self.mutex.unlock()

    def stopped(self) raises -> Bool:
        self.mutex.lock()
        var s = self.stop
        self.mutex.unlock()
        return s


struct EngineBox(Movable):
    """engine 线程的全部家当：邮箱、reactor 的地址、handler 的地址。

    handler **只留一个地址**，不是因为地址更优雅：Mojo 的函数参数没有 `owned`
    约定（`owned` 不是关键字），所以一个 `mut handler: H` 是**借用**，挪不进箱子 ——
    handler 于是留在调用方（`Loop.run_threaded`）那里，它的生命周期天然覆盖
    `join`（调用方要等 join 完才返回）。"这条线程独占它"这件事仍然成立：地址只有
    engine 线程在用，循环这一侧从头到尾没碰过 handler。
    """

    var mailbox: Mailbox
    var reactor_addr: Int
    var handler_addr: Int

    def __init__(out self, reactor_addr: Int, handler_addr: Int) raises:
        self.mailbox = Mailbox()
        self.reactor_addr = reactor_addr
        self.handler_addr = handler_addr

    def stop(mut self) raises:
        """让 engine 线程退出：**先设标记，再 join**。

        反过来（先 join）会等到天荒地老 —— engine 只在"没活 + 看到了标记"时才
        退出，而标记没设它永远看不到。
        """
        self.mailbox.set_stop()


def queue_request(
    mut box: EngineBox, slot: Int, gen: Int, thread: Int, var raw: List[UInt8]
) raises -> Bool:
    """把一个请求派给 engine 的第 `thread` 条线程。返回 False = 邮箱满了（那就不
    派，见文件头第 3 条）。

    它是个函数而不是内联的三行，是因为"把字节挪进 job"这一步只在这个形状下能编译
    （挪进 `Job` 的是一个**参数**，不是从别的 struct 中间挖出来的一块）。
    """
    var job = Job(slot, gen, JOB_REQUEST, thread, raw^)
    return box.mailbox.push_job(job^)


struct EngineThreads(Movable):
    """N 条 engine 线程的 handle。

    为什么放在**堆**上：N 是运行时的数（`ALOFA_ENGINE_THREADS`），而 `List` 装不
    下这类带资源的值（`List` 要整体搬动元素）。
    """

    var raw: Int
    var n: Int

    def __init__(out self, raw: Int, n: Int):
        self.raw = raw
        self.n = n

    @staticmethod
    def alloc(n: Int) raises -> EngineThreads:
        return EngineThreads(
            external_call["malloc", Int](size_of[ThreadHandle]() * n), n
        )

    def put(mut self, i: Int, var handle: ThreadHandle) raises:
        var slot = Pointer[ThreadHandle, MutUntrackedOrigin](
            unsafe_from_address=self.raw
        )
        slot.unsafe_offset(i).unsafe_write(handle^)

    def join_all(mut self) raises:
        """等这 N 条线程都退出来。**每条都要 join**：漏一条等于那条线程还在用箱子
        里的锁，而箱子马上就要被析构（见文件头那一段）。"""
        if self.n == 0:
            return
        var slot = Pointer[ThreadHandle, MutUntrackedOrigin](
            unsafe_from_address=self.raw
        )
        var i = 0
        while i < self.n:
            _ = slot.unsafe_offset(i)[].join()
            i += 1
        _ = external_call["free", NoneType](self.raw)
        self.raw = 0
        self.n = 0


def spawn_engines[H: EngineHandler](
    mut box: EngineBox, engines: Int
) raises -> EngineThreads:
    """起 `engines` 条 engine 线程（每条一份 handler，见文件头"并发生成"）。

    ⚠️ 返回的这批 handle **必须**被 join，而且**箱子必须活到 join 之后**（见文件
    头那一段：Mojo 用完即析构，spawn 那一行是箱子的最后一次被提到，之后它就完了
    —— 而线程还在用它）。调用方在 join 之后碰一次箱子即可。
    """
    # ⚠️ `engines` 是参数（不可改）：改它要另起一个局部变量。
    var count = engines
    if count <= 0:
        count = 1
    var addr = Int(Pointer(to=box))
    var threads = EngineThreads.alloc(count)
    var i = 0
    while i < count:
        threads.put(i, ThreadHandle.spawn[_engine_entry[H]](opaque_of(addr)))
        i += 1
    return threads^


def _engine_entry[H: EngineHandler](arg: OpaquePtr) -> OpaquePtr:
    """pthread 的入口。**不许让异常跑出去**（pthread 没有异常通道）。

    返回值没有意义（循环不读它），但要给一个 —— 空指针是从运行时零造出来的，
    因为 `Pointer` 不接受字面量 0。
    """
    var box = arg.unsafe_bitcast[EngineBox]()
    var index = -1
    try:
        index = box[].mailbox.claim_thread()
    except err:
        # 连"我是第几条"都问不出来就别干了：两条线程共用一份 handler 会互相写坏
        # 一条生成的相位，而那只是"偶尔答错一次" —— 比"这条线程的活永远没人干"
        # 更难查。
        print("  [srv] an engine thread could not claim its index: " + String(err))
        var zero = 0
        return OpaquePtr(unsafe_from_address=zero)
    _engine_main[H](box, index)
    var zero = 0
    return OpaquePtr(unsafe_from_address=zero)


def _engine_main[H: EngineHandler](
    box: Pointer[EngineBox, MutUntrackedOrigin], index: Int
):
    """第 `index` 条 engine 线程的主循环：有活就干，没活就睡，看到标记且没活了就退。

    **0 号用原型那份 handler**（调用方手上那一份，地址在箱子里），其余的用自己
    `spawn_twin` 出来的一份 —— 每条线程一份生成的相位（见文件头"并发生成"）。

    **退出前要把活干完**（先取活，取不到才看标记）：循环那边已经把"答完 N 条"
    当成退出条件了，engine 提前退出会让最后几条请求永远等不到应答 —— 而那条
    连接看起来只是"慢"，没有任何报错。
    """
    var handler = Pointer[H, MutUntrackedOrigin](
        unsafe_from_address=box[].handler_addr
    )
    var twin_addr = 0
    if index > 0:
        try:
            twin_addr = handler[].spawn_twin(index)
            handler = Pointer[H, MutUntrackedOrigin](unsafe_from_address=twin_addr)
        except err:
            print("  [srv] an engine handler could not be duplicated: " + String(err))
            return
    while True:
        var job = Job(-1, -1, JOB_REQUEST, index)
        var got = False
        try:
            got = box[].mailbox.take_job(index, job)
        except err:
            print("  [srv] the mailbox could not be read: " + String(err))
        if got:
            try:
                box[].mailbox.begin_job()
            except err:
                print("  [srv] the engine counters could not be written: " + String(err))
            # 默认值就是"失败"那一截（关连接）：上行会覆盖它。先备好再跑，是因为
            # "可能没初始化"在这条线程上的后果是一次未定义行为，不是一条编译警告。
            var res = Result(job.slot, job.gen, RES_DONE, True, List[UInt8]())
            try:
                res = _run_job[H](handler, job^)
            except err:
                # ⚠️ **一件取走的活必须有应答**。少了这一截，循环那边 `inflight` 永远是
                # 1，而邮箱里一件都没有 —— 那是一条挂到天亮的请求，且没有任何报错：循环
                # 不会再派帧，`reap` 又跳过 `streaming=1` 的槽位，空闲超时也收不到它。
                # 炸也要给客户端一个终点。
                #
                # （这句曾经引过一组"实测数字"：16 件活只交回 15 截、`busy` 卡在 1。那组
                # 数字是**未加锁读**造出来的假象 —— 计数改成锁内读之后四数是对平的。不
                # 变量本身照样成立，但"它已经被实测证实"这句话不成立。）
                print("  [srv] a job could not be run: " + String(err))
            try:
                box[].mailbox.push_result(res^)
            except err:
                print("  [srv] a result could not be queued: " + String(err))
            try:
                box[].mailbox.end_job()
            except err:
                print("  [srv] the engine counters could not be written: " + String(err))
            _wake[H](box)
            continue
        try:
            box[].mailbox.tick_poll()
        except err:
            print("  [srv] the engine counters could not be written: " + String(err))
        # 停不停是**唯一**的退出条件（`while True` 没有别的出口）。连"该退出了吗"都问
        # 不出来的时候，退是唯一不会更坏的选择 —— 继续跑下去只会让 `join` 挂住。
        try:
            if box[].mailbox.stopped():
                break
        except err:
            print("  [srv] the stop flag could not be read, stopping: " + String(err))
            break
        _ = external_call["usleep", Int32](Int32(ENGINE_IDLE_US))
    # 把自己那份 handler 还回去（0 号用的是调用方那一份，不属于这条线程 —— 它归
    # 调用方管，而调用方的生命周期本来就盖到 join 之后）。
    if twin_addr != 0:
        try:
            _ = heap_take[H](twin_addr)
        except err:
            print("  [srv] an engine handler could not be released: " + String(err))


def _run_job[H: EngineHandler](
    handler: Pointer[H, MutUntrackedOrigin], var job: Job
) -> Result:
    """跑一件活，交回一截字节。这一层不许抛（见 `_engine_entry`）。"""
    if job.kind == JOB_STREAM_STEP:
        return _stream_step[H](handler, job^)
    if job.kind == JOB_STREAM_END:
        return _stream_end[H](handler, job^)
    return _answer[H](handler, job^)


def _stream_end[H: EngineHandler](
    handler: Pointer[H, MutUntrackedOrigin], var job: Job
) -> Result:
    """把一条**没走完**的流在 engine 这一侧收掉。

    交回的那一截不会被任何人收到（连接已经不在了，`collect` 靠 `gen` 对不上把它丢
    掉）—— 它存在的唯一理由是**记账**：循环派了活就要等到应答，`inflight` 才有机会
    归零。少了这一截，槽位上的"还有一件事在途"永远是真的。
    """
    try:
        handler[].stream_end(job.slot)
    except err:
        print("  [srv] a stream could not be released: " + String(err))
    return Result(job.slot, job.gen, RES_DONE, True, List[UInt8]())


def _answer[H: EngineHandler](
    handler: Pointer[H, MutUntrackedOrigin], var job: Job
) -> Result:
    """一个请求 → 一截字节（流式只交头，帧由 `_stream_step` 一帧一帧给）。

    与 `srv/loop.mojo` 的 `answer` 是同一套语义搬过来的：**错误也要变成一截
    字节**。engine 线程里没有"让异常冒出去"这条路 —— 冒出去等于这条连接被静默
    丢掉，客户端只能看到一个挂住的请求。
    """
    var slot = job.slot
    var gen = job.gen
    # 默认值就是"失败"那一个：handler 成功会覆盖它。反过来写（先声明、再在
    # `except` 里补上）编译器会认为它可能没被初始化 —— 而"可能没初始化"在 engine
    # 线程里的后果是一次未定义行为，不是一条编译警告。
    var response = json_response(
        500,
        "{\"error\":{\"message\":\"the request could not be"
        + " answered\",\"type\":\"server_error\",\"code\":"
        + "\"internal\"}}",
        True,
    )
    # 默认值也是"关"：解析不了这个请求的时候，它没有第二次机会，而一条不知该不该
    # 关的连接会一直占着槽位（`MAX_CONNS` 有上界）。
    var close = True
    try:
        var raw = List[UInt8]()
        var i = 0
        while i < job.n_raw:
            raw.append(job.raw[i])
            i += 1
        var request = parse_request(raw)
        close = not request.keep_alive()
        response = handler[].handle(job.slot, request)
    except err:
        # 500 不带细节进响应（内部原因属于日志），但消息要打出来 —— 一个没有原因
        # 的 500 只能靠猜。
        print("  [srv] request failed: " + String(err))
    # 对端说要关，就必须在响应里也说要关：只在写完以后默默关掉，对端会以为这条
    # 连接还能用 —— 那是一次看起来像"偶发连接重置"的失败。
    if close:
        response.close = True
    var text: String
    try:
        if response.content_type == SSE_CONTENT_TYPE:
            # 只有头：帧由 `_stream_step` 一帧一帧喂。
            text = serialize_stream_head(response.status)
            return Result(slot, gen, RES_STREAM_HEAD, False, bytes_of_text(text))
        text = serialize(response)
    except err:
        print("  [srv] response could not be written: " + String(err))
        return Result(slot, gen, RES_DONE, True, List[UInt8]())
    return Result(slot, gen, RES_DONE, response.close, bytes_of_text(text))


def _stream_step[H: EngineHandler](
    handler: Pointer[H, MutUntrackedOrigin], var job: Job
) -> Result:
    """流走一步：一帧，或者一个空收尾（空收尾 = 流结束，靠关连接定界）。"""
    var frame: String
    try:
        frame = handler[].stream_next(job.slot)
    except err:
        # 前向炸了也要给这条流一个终点：少了它，客户端永远等不到 `[DONE]`。
        print("  [srv] stream aborted: " + String(err))
        frame = ""
    if frame.byte_length() == 0:
        return Result(job.slot, job.gen, RES_DONE, True, List[UInt8]())
    return Result(job.slot, job.gen, RES_BYTES, False, bytes_of_text(frame))


def _wake[H: EngineHandler](box: Pointer[EngineBox, MutUntrackedOrigin]):
    """叫醒正在 `poll` 里等着的循环。

    这是"结果不会被拖到一个 poll 周期之后"的唯一保障（能力门里钉的是同一个原语）。
    叫不醒的失败形态很安静：一切正常，只是每条流的每一帧都晚 50 ms。
    """
    var reactor = Pointer[Reactor, MutUntrackedOrigin](
        unsafe_from_address=box[].reactor_addr
    )
    try:
        reactor[].wakeup()
    except err:
        print("  [srv] the reactor could not be woken: " + String(err))
