"""一条 reactor 事件循环上跑 N 条连接：读、切请求、排队、写，都不阻塞。

与 `srv/server.mojo` 的差别只有一处，但它会改到每一行的写法
----------------------------------------------------------
那边是"accept 一条 → 在这条连接上一直读到有完整请求 → 生成 → 写完"。在这条循环上，
**任何一次阻塞都是全体阻塞**：一个连上不发数据的对端会让另外 99 条连接一起等，一个慢
客户端（发得出去、收得慢）会把生成侧卡在 `write` 里，同样连累所有人。所以这里每条连接
只有一个"现在能读/能写吗"的问题，答案来自 reactor；读不到完整请求就是**这条连接本轮没
事干**，而不是"停在这儿等"。

代价（`run` 那一版，明文写在这儿，不是疏漏）
----------------------------------------------
**请求仍然是串行生成的，而且就生成在这条循环上**：一次 `handler.handle` 会占住这条
循环（`active_stream` 那条也是）。所以那一版换来的不是"同时生成 N 条"，而是"一条
连接的 I/O 不再连累别人" —— 慢客户端、流水线、keep-alive 的空闲连接都交给 reactor，
生成仍然一条一条来，而且**生成期间这条循环什么都做不了**。

`run_threaded`：把生成搬到另一条线程（roadmap 3.2b）
--------------------------------------------------
`run` 与 `run_threaded` 两条入口都在，前者留着当**负向对照**（同一个门换回 `run`
就红，见 `tests/unit/test_engine_thread_server.mojo`）。后者把 `handler` 交给一条
独占线程，这条循环只做 I/O：请求进邮箱、结果出邮箱，结果回来靠 `Reactor.wakeup()`。

它换来的那一处：**生成期间循环还在干活** —— 一个 20 秒的生成不再让另外 99 条连接
连"读一个请求"都做不到。

它**没有**换来的那一处（同样明文写在这儿）：**同时生成的条数仍然是 1**。只有一条
engine 线程，一次处理一个 job。并发生成是紧跟着的 **3.2c**（`spawn_engines`：N 条
engine 线程，每条一份 handler）—— 那条路的代价是 N 份权重，所以在它之前这里先停在
1。另一条路是让 engine 一次吃下一批（P2 的批量调度已经在了），那才是**不**多占权重
的并发生成，还没做；邮箱的形状（`push_job` 一次一个、`pending()` 看在途）已经为它留
了接缝。

⚠️ 邮箱这一版多了一个 `run` 没有的状态：**在途**。一条连接把请求派出去之后，它的
缓冲是空的、也没有字节要写 —— 单看 `Conn` 它"什么都没干"，于是空闲超时会把它收掉
（生成慢不是一条连接该死的理由，收掉它客户端就收到一个半截响应）。所以 `inflight`
要参与 `reap` 的判断，而结果回来时还要对一次 `gen`（槽位会换主人：不对的话，上一条
连接的应答会发给下一条）。

四个必须自己守住的护栏
----------------------
1. **非阻塞 + EAGAIN**：监听 fd 与每条连接都设 `O_NONBLOCK`，`EAGAIN` 不是错误而是
   "这轮到这儿"。漏了它，`send` 会在慢客户端上把整条循环堵死 —— 而那正是这次改造要
   修的东西。
2. **监听 fd 必须接到 EAGAIN**：`accept` 只接一条就走的话，backlog 里剩下的连接会让
   level-triggered 事件一直报可读 —— 循环变成满核空转（测试的门：`tests/capability/
   test_reactor.mojo`）。
3. **背压要改 interest**：队列满了就不再要"可读"（否则空转），也不再生成下一帧
   （否则内存按客户端数量涨）。降下来再改回去。
4. **槽位有限**（`MAX_CONNS`）：满了就接进来立刻关，不放进 backlog —— 留在 backlog
   里同样会空转，而"接了又关"至少让对端立刻知道。
"""

from std.collections import List
from std.memory import Pointer

from flare.net import SocketAddr
from flare.runtime import INTEREST_READ, INTEREST_WRITE, Event, Reactor
from flare.runtime.reuseport import bind_reuseport
from flare.tcp import TcpListener, TcpStream, accept_fd

from alofa.core.error import ERR_IO, AlofaError
from alofa.core.ffi.posix import monotonic_ms, set_nonblocking
from alofa.srv.conn import (
    TAKEN_CLOSED,
    TAKEN_NEED_MORE,
    TAKEN_REQUEST,
    WRITE_CHUNK,
    Conn,
    Taken,
)
from alofa.srv.engine_thread import (
    JOB_REQUEST,
    JOB_STREAM_STEP,
    JOB_STREAM_END,
    RES_BYTES,
    RES_DONE,
    RES_STREAM_HEAD,
    EngineHandler,
    EngineBox,
    Job,
    Result,
    queue_request,
    spawn_engines,
)
from alofa.srv.http import (
    SSE_CONTENT_TYPE,
    HttpRequest,
    HttpResponse,
    bytes_of_text,
    json_response,
    serialize,
    serialize_stream_head,
)
from alofa.srv.server import Handler, stop_requested

# 同时挂在一条循环上的连接数上限。它是**有意的**上界：每条连接带一个发送队列
# （`SEND_HIGH_WATER` 1 MiB），无上界等于把内存交给客户端决定。
comptime MAX_CONNS = 128

# token 的选择：`WAKEUP_TOKEN` 被 reactor 留着用（不能撞），socket 的 token 从
# `CONN_TOKEN_BASE` 开始，等于"槽位号 + 基数" —— 事件里除了一个整数没有别的东西
# 能指回连接，所以这个映射必须是唯一的。
comptime LISTENER_TOKEN = 1
comptime CONN_TOKEN_BASE = 100

comptime READ_CHUNK = 4096
comptime IDLE_TIMEOUT_MS = 30000
# 空闲时的 poll 超时：它同时是"多久看一次停止标记"和"多久查一次空闲连接"的粒度。
comptime POLL_IDLE_MS = 50
# 一轮里最多接多少条新连接。上限是为了"接不完"的时候不会一直绕 —— 剩下的连接下一轮
# 还在 backlog 里，而空转比慢更贵。
comptime ACCEPT_PER_ROUND = 32


def is_would_block(imm err: String) -> Bool:
    """flare 把 `EAGAIN`/`EWOULDBLOCK` 报成 `Timeout(...)`

    非阻塞 socket 上它是**常态**而不是错误：读空了、写满了都报它。这里只按名字认
    （与项目里其它地方一致：比名字不比消息），因为 errno 拿不到 —— flare 把它转成
    异常的时候只留了名字。
    """
    return err.find("Timeout(") == 0


struct Loop(Movable):
    """一个 listening socket + 一张连接表 + 一条事件循环。

    连接表是**平行的基础类型列表**（`conns` / `streams` / `live` / `last_seen` /
    `interest`），不是一张 `List[Conn]` 的索引表：槽位号就是下标，关掉一条连接只是
    把 `live` 置 0 让槽位可复用 —— 下标不变，token 也就不会错位。
    """

    var listener: TcpListener
    var lfd: Int32
    var reactor: Reactor
    var conns: List[Conn]
    var streams: List[TcpStream]
    var live: List[Int]
    var last_seen: List[Int]
    var interest: List[Int]
    # 每个槽位"是不是正在写一条流"（`run` 那一版同时只可能有一条：handler 的流
    # 状态只有一份，`progress` 在有流时会立刻返回 —— 而 `run_threaded` 那一版能同
    # 时有 `engines` 条，每条归一条 engine 线程，那份 handler 是它自己的）。
    #
    # 为什么是**按槽位**的一列而不是一个 `active_stream`：`engines > 1` 时两条流
    # 可以同时存在（各在一条 engine 线程上、各用一份 handler）。
    var streaming: List[Int]
    var served: Int
    # ── 下面这三个只有 `run_threaded` 用（生成在另一条线程上的时候） ──
    # 槽位的**代号**：`drop` 时自增。号会复用，代号不会 —— 在途的结果靠它认主人
    # （不认的话，上一条连接的应答会发给下一条）。
    var gen: List[Int]
    # 这条连接有没有派出去还没答完的事。空闲超时必须跳过它：请求已经离开缓冲，单看
    # 连接它"什么都没干"，而生成慢不是一条连接该死的理由。
    #
    # 它同时也是"流能不能派下一帧"：**一个槽位上最多一件在途的事**（一个请求、或
    # 一帧），所以"在途"为 0 才能派下一个 —— 这既保证了一帧一次往返，也保证了一条
    # 连接上的多个请求**按发出的顺序**被应答（两条 engine 线程快慢不同，同时在途
    # 两个请求 = 可能换顺序）。
    var inflight: List[Int]
    # 这个槽位归第几条 engine 线程（第一次派活时定下来，之后不变）。-1 = 还没定。
    var owner: List[Int]
    # engine 线程数（`run_threaded` 的 `engines`）与"下一个派给谁"的游标。
    var engines: Int
    var next_engine: Int
    # 诊断用：这个槽位已经报过一次"卡住"了吗（0 = 还没报）。只上诊断，不参与判定。
    var stall_seen: List[Int]
    # 这个槽位上有一条**没走完**的流等着 engine 去收尾（`JOB_STREAM_END`）：
    # `-1` = 没有，`>= 0` = 该交给第几条 engine 线程。记线程号而不是记"有/没有"，
    # 是因为 `drop` 会把 `owner` 清成 -1 —— 不在这里留住它，收尾就不知道该派给谁
    # （派错线程 = 从一条没有那条相位的线程上收一条不存在的流）。
    #
    # 为什么是"登记一个待办"而不是 `drop` 时直接派：`drop` 也被 `reap` / `drop_all` 调，
    # 那些地方手上没有邮箱；而真正能派活的是 `dispatch`。分开之后"连接被丢"与"通知
    # engine"各在自己的位置上，`drop` 也不必为了派活把 `Handler` 的类型参数拖进来。
    var release_pending: List[Int]
    # 那份待办**已经派出去了吗**（`JOB_STREAM_END` 进邮箱 = 1）。
    #
    # 为什么要有它：待办必须等**它自己**的回执才清，而 `collect` 收到的一截应答分不
    # 出是"那件活的"还是"收尾的" —— 于是原来它对**每一截**应答都清一次待办，结果
    # 是：断开时那件在途的活一回来就把待办抹了，`JOB_STREAM_END` 从此再也没人派，
    # engine 那边那个号一直占着（占满的表现是"新请求被拒"，而真正的原因在别处）。
    var release_sent: List[Int]

    def __init__(out self, var listener: TcpListener) raises:
        self.listener = listener^
        self.lfd = Int32(self.listener.as_raw_fd())
        self.reactor = Reactor()
        self.conns = List[Conn]()
        self.streams = List[TcpStream]()
        self.live = List[Int]()
        self.last_seen = List[Int]()
        self.interest = List[Int]()
        self.streaming = List[Int]()
        self.served = 0
        self.gen = List[Int]()
        self.inflight = List[Int]()
        self.owner = List[Int]()
        self.stall_seen = List[Int]()
        self.release_pending = List[Int]()
        self.release_sent = List[Int]()
        self.engines = 1
        self.next_engine = 0

    @staticmethod
    def bind(port: UInt16) raises -> Loop:
        """绑到 127.0.0.1（`SO_REUSEPORT`，多 worker 各 bind 同一个端口）。"""
        return Loop.bind_addr(SocketAddr.localhost(port))

    @staticmethod
    def bind_addr(addr: SocketAddr) raises -> Loop:
        return Loop.adopt_listener(bind_reuseport(addr))

    @staticmethod
    def adopt_listener(var listener: TcpListener) raises -> Loop:
        """接手一个已经 bind 好的监听 socket（master 给 worker 的就是这种）。"""
        var lfd = listener.as_raw_fd()
        # ⚠️ `listener` 必须活到最后：Mojo 是用完即析构（ASAP），如果
        # `as_raw_fd()` 是它的最后一次使用，监听 socket 会在下一句之前被关掉。
        var rc = set_nonblocking(Int32(lfd))
        if rc != 0:
            raise AlofaError(
                ERR_IO,
                "the listener could not be made non-blocking",
                "rc=" + String(Int(rc)),
            )
        var loop = Loop(listener^)
        loop.reactor.register(Int32(lfd), UInt64(LISTENER_TOKEN), INTEREST_READ)
        return loop^

    def close(mut self):
        self.listener.close()

    # ── 主循环 ──────────────────────────────────────────────────────────────

    def run[H: Handler](
        mut self, mut handler: H, max_requests: Int, stop_fd: Int32
    ) raises -> Int:
        """跑到答完 `max_requests` 条（`<= 0` 表示一直跑）或停止标记被关掉。

        返回实际答完的条数 —— 门靠这个数停下一个本来不会停的循环。

        一轮的顺序是**固定的**：先看事件（谁可读/可写）→ 再推进每一条连接（切请求、
        生成、写）→ 最后收尾（该关的关）。顺序不能换：先推进再收尾，才会把"刚写完
        的响应"与"因此可以关掉的连接"在同一轮里收掉。
        """
        while max_requests <= 0 or self.served < max_requests:
            if stop_requested(stop_fd):
                break
            # 有东西要写、或者有流在途：poll 立刻返回（别让等待拖住写）。否则等一小
            # 会 —— 省 CPU，同时给"停止标记/空闲超时"一个检查点。
            var timeout = POLL_IDLE_MS
            if self.busy():
                timeout = 0
            var events = List[Event]()
            var got = self.reactor.poll(timeout, events)
            var i = 0
            while i < got:
                var token = events[i].token
                if not events[i].is_wakeup():
                    if token == LISTENER_TOKEN:
                        self.accept_ready()
                    elif token >= CONN_TOKEN_BASE:
                        var slot = Int(token - CONN_TOKEN_BASE)
                        if slot < len(self.live) and self.live[slot] == 1:
                            if events[i].is_readable():
                                self.read_ready(slot)
                            if events[i].is_writable() and self.live[slot] == 1:
                                self.flush(slot)
                i += 1
            self.progress[H](handler)
            self.reap()
        # 收尾：把手上还没写完的字节发出去再关。直接关会截掉客户端正在等的那截响应。
        self.drain_all()
        self.drop_all()
        return self.served

    def run_threaded[H: EngineHandler](
        mut self,
        mut handler: H,
        max_requests: Int,
        stop_fd: Int32,
        engines: Int = 1,
    ) raises -> Int:
        """与 `run` 同一件事，但生成在**别的线程**上（`srv/engine_thread.mojo`）。

        这一条循环上只剩下 I/O：读字节、切请求、把请求派进邮箱、把结果排进发送
        队列、写。差别集中在一处 —— `run` 里那次 `handler.handle` 在这儿变成"派一
        个 job"，而"结果回来了"这件事由 `Reactor.wakeup()` 打断 poll 告诉我们。

        `engines` 是 engine 线程的条数，**也是同时能生成的条数**（roadmap 3.2c）：
        每条线程一份 handler（`Twinable.spawn_twin`），所以两条 2 秒的生成在
        `engines=2` 下是 2 秒、在 `engines=1` 下是 4 秒（门里两种都量着）。代价写
        在 `engine_thread.mojo` 的文件头：**N 条 = N 份权重**。

        一轮的顺序仍然是固定的：事件 → **收结果** → **派活** → 收尾。收在派前，
        是因为派活要看"还有多少在途"（`budget`），而结果取回来之后在途才准。
        """
        # ⚠️ `engines` 是参数（不可改）：改它要另起一个局部变量。
        var count = engines
        if count <= 0:
            count = 1
        self.engines = count
        var box = EngineBox(
            Int(Pointer(to=self.reactor)), Int(Pointer(to=handler))
        )
        var threads = spawn_engines[H](box, count)
        while max_requests <= 0 or self.served < max_requests:
            if stop_requested(stop_fd):
                break
            # ⚠️ 超时归零的判据是"还有字节没发出去"，**不是** `busy()`：后者把"有流在跑"
            # 也算进去，而线程这条路上帧是 engine 产的、它产完会叫醒我们 —— 为此死转只会
            # 白烧一个核（实测 utime +68/秒）。见 `has_pending_writes`。
            var timeout = POLL_IDLE_MS
            if self.has_pending_writes():
                timeout = 0
            var events = List[Event]()
            var got = self.reactor.poll(timeout, events)
            self.on_events(events, got)
            self.collect[H](box)
            # 还能派几个：`max_requests` 是"答完就停"的那个数，在途的也算在里面
            # —— 否则最后几轮会把请求全派出去，然后循环在"答完"之前就退了。
            var budget = 1 << 30
            if max_requests > 0:
                budget = max_requests - self.served - box.mailbox.pending()
            self.dispatch[H](box, budget)
            self.guard_stalls[H](box)
            self.sync_all()
            self.reap()
        box.stop()
        threads.join_all()
        # ⚠️ 生命周期锚点：箱子必须活到 join **之后**（Mojo 用完即析构，spawn 那一
        # 行是它最后一次被提到 —— 那之后线程还在用里面的 mutex）。这一行是唯一让
        # 它活下来的东西，别把它"整理"掉。
        var leftover = box.mailbox.pending()
        self.drain_all()
        self.drop_all()
        if leftover > 0:
            print(
                "  [srv] "
                + String(leftover)
                + " request(s) were still in flight when the loop stopped"
            )
        return self.served

    def on_events(mut self, imm events: List[Event], got: Int) raises:
        """事件 → 动作。**不认识的事件就当没看见**：token 是唯一的映射，认错等于
        把一条连接的字节写到另一条上。"""
        var i = 0
        while i < got:
            var token = events[i].token
            if not events[i].is_wakeup():
                if token == LISTENER_TOKEN:
                    self.accept_ready()
                elif token >= CONN_TOKEN_BASE:
                    var slot = Int(token - CONN_TOKEN_BASE)
                    if slot < len(self.live) and self.live[slot] == 1:
                        if events[i].is_readable():
                            self.read_ready(slot)
                        if events[i].is_writable() and self.live[slot] == 1:
                            self.flush(slot)
            i += 1

    def collect[H: EngineHandler](mut self, mut box: EngineBox) raises:
        """把 engine 交回来的结果排进对应连接的发送队列。

        两处"看起来可以省"的判断都不能省：
        - **先看队首是谁**（`first_result`）：那条连接正在背压的话，结果必须**留在
          邮箱里** —— 取出来就排不进队列，而丢掉它等于让客户端永远等下去；
        - **取回来再对一次 `gen`**：槽位可能已经换过主人，对不上就必须丢。
        """
        while True:
            var slot = -1
            var gen = -1
            try:
                if not box.mailbox.first_result(slot, gen):
                    return
            except err:
                print("  [srv] the mailbox could not be read: " + String(err))
                return
            if slot < 0 or slot >= len(self.live) or self.live[slot] == 0:
                # 连接已经不在了（客户端走了）：结果没有主人，丢掉。
                var dead = Result()
                try:
                    _ = box.mailbox.take_result(dead)
                except:
                    pass
                continue
            if self.conns[slot].backpressure():
                self.sync_interest(slot)
                return
            var res = Result()
            try:
                if not box.mailbox.take_result(res):
                    # 上一步刚看见有一条，这一步却取不到 —— 结果队列只被这一条循环消
                    # 费，所以这不该发生。打出来是为了让它从"静默丢一条应答"变成"红"。
                    print("  [srv] a result vanished between peek and take")
                    return
            except err:
                print("  [srv] a result could not be taken: " + String(err))
                return
            self.inflight[slot] = 0
            # 待办只认**它自己**的回执：这一截是 `JOB_STREAM_END` 的答案才清。
            #
            # 对每一截应答都清（原来的写法）等于把待办提前抹掉：断开时那件在途的活
            # 一回来就把待办清了，于是 `JOB_STREAM_END` 从此没人派 —— engine 那边那个
            # 请求号一直占着（占满的表现是"新请求被拒"，而原因在别处），而槽位还被
            # `has_room` 当成空位。
            if self.release_sent[slot] == 1:
                self.release_pending[slot] = -1
                self.release_sent[slot] = 0
            if res.gen != self.gen[slot] or self.live[slot] == 0:
                # 槽位换过主人：这个结果属于**上一条**连接。发给这一条，就是那种
                # "偶发的响应内容不对"，而且只在客户端恰好中途断开时出现。
                continue
            if res.n_bytes > 0:
                # 字节在**这一条线程**上拷出来（分配与释放都在这边）：跨线程过的只是
                # 定长内联的那份，没有堆块跟着活一起走。
                var out = List[UInt8]()
                var i = 0
                while i < res.n_bytes:
                    out.append(res.bytes[i])
                    i += 1
                try:
                    self.conns[slot].enqueue(out^)
                except err:
                    print("  [srv] connection dropped: " + String(err))
                    self.drop(slot)
                    continue
            self.last_seen[slot] = monotonic_ms()
            # 在途那一帧回来了（一帧一次往返），所以可以再派一帧。少了这一行，流会
            # 在第一帧之后**停住**：`dispatch` 以为还有一帧在途，而那一帧早回来了 ——
            # 症状是客户端收到头 + 一帧，然后一直挂到空闲超时。
            if res.kind == RES_STREAM_HEAD:
                # 这一条开始了一条流：接下来由 `dispatch` 一帧一次往返地喂它。
                # `streaming` 是**按槽位**标记的，所以 `engines` 条流可以同时在
                # 跑 —— 各归一条 engine 线程，各用一份 handler（`run` 那一版同时只有
                # 一条：`progress` 在有流时会立刻返回）。
                self.streaming[slot] = 1
            if res.kind == RES_DONE:
                if res.close:
                    self.conns[slot].close_when_drained()
                self.streaming[slot] = 0
                self.served += 1

    def owner_for(mut self, slot: Int) -> Int:
        """这个槽位归第几条 engine 线程。第一次派活时定下来，**之后不变**。

        为什么"定了就不变"：一条连接上的多个请求必须**按发出的顺序**被应答，而两条
        engine 线程快慢不同 —— 换线程等于换顺序（症状是偶尔一次"响应内容和请求对
        不上"）。流的相位也要求它（一帧必须回到开始那条流的线程上，见
        `engine_thread.mojo` 的文件头）。

        分配是**轮转**（不是按槽位号取模）：槽位号是"第一个空位"，而空位的分布随
        连接活多久变 —— 取模会让"两条新连接常常落在同一条线程上"，轮转不会。
        """
        if self.owner[slot] >= 0:
            return self.owner[slot]
        var pick = self.next_engine % self.engines
        self.next_engine += 1
        self.owner[slot] = pick
        return pick

    def dispatch[H: EngineHandler](
        mut self, mut box: EngineBox, mut budget: Int
    ) raises:
        """把连接上的活派给 engine。

        **派之前先问邮箱有没有位置**（`has_room`）：请求一旦 `take` 出来就被消费
        掉了，塞不进邮箱等于把它弄丢。

        派活不再"一条连接一轮只答一个"（那是 `progress` 的写法）：现在生成不在这一
        条线程上，多派几个只是让 engine 的队列长一点 —— 而顺序仍然是扫描顺序
        （FIFO），一条连了十个请求的连接不会因此插到别人前面。

        两段：**先喂在途的流**（一帧一次往返，帧的延迟直接是客户端看到的延迟），
        **再派新请求**。一起做而不是二选一，是因为 `engines > 1` 时"一条流在跑"不
        再意味着"别的槽位只能等着" —— 流占的是它自己那条线程。
        """
        # 0) 先派"收尾"（连接被丢时 engine 那一份还没还回去）。它排在喂帧**之前**：
        #    一个已经没有客户端的槽位不该继续占着引擎侧的请求号 —— 而喂帧是让活
        #    下去，收尾是让活停下来，后者先走才不会把号越占越满。
        for slot in range(len(self.live)):
            var thread = self.release_pending[slot]
            if thread < 0 or self.inflight[slot] == 1:
                continue
            var closing = Job(slot, self.gen[slot], JOB_STREAM_END, thread)
            try:
                # ⚠️ 派出去**先不消待办**：`stream_end` 是按请求号（= 槽位）找流的，这件
                # 活还在飞的时候把槽位给新连接，engine 收掉的就会是**新**那条流（更
                # 糟的是重复释放）。待办要等 `collect` 看见回执才清 —— 那时才真的收完。
                if box.mailbox.push_job(closing^):
                    self.inflight[slot] = 1
                    # 派出去了：从现在起回来的那一截应答**才是**这份待办的回执
                    # （`collect` 据此清它，见 `release_sent`）。
                    self.release_sent[slot] = 1
                # 邮箱满了就留着下一轮再派：这是一件"迟早要办"的事，不是"现在必须
                # 办完"的事，而丢掉它就等于把引擎侧那个位置漏掉。
            except err:
                print("  [srv] a stream release could not be queued: " + String(err))
        # 1) 在途的流：每条一帧（归它的那条线程）。
        for slot in range(len(self.live)):
            if self.live[slot] == 0 or self.streaming[slot] != 1:
                continue
            # 一个槽位上最多一件在途的事（见 `inflight` 的注释）：在途为 0 才能派
            # 下一帧。少了这个判断，流会**停住**（`dispatch` 以为还有一帧在途）。
            if self.inflight[slot] == 1:
                continue
            if self.conns[slot].backpressure():
                self.sync_interest(slot)
                continue
            var step = Job(slot, self.gen[slot], JOB_STREAM_STEP, self.owner[slot])
            try:
                if box.mailbox.push_job(step^):
                    self.inflight[slot] = 1
            except err:
                print("  [srv] a stream step could not be queued: " + String(err))
        if budget <= 0:
            return
        var room = False
        try:
            room = box.mailbox.has_room()
        except err:
            print("  [srv] the mailbox could not be checked: " + String(err))
        if not room:
            return
        for slot in range(len(self.live)):
            if budget <= 0:
                break
            if self.live[slot] == 0:
                continue
            # 一个槽位上一次只派一件（见 `inflight`）：既保证同一条连接上的多个请
            # 求按发出的顺序被应答，也让"在途"这个数不用再按线程拆开记。
            if self.inflight[slot] == 1:
                continue
            if self.conns[slot].backpressure():
                self.sync_interest(slot)
                continue
            # ⚠️ **先把字节留在连接的缓冲里，等确认邮箱有位置再取**。
            #
            # 反过来（先 `take_with_raw` 再 `push_job`）会静默丢请求：那份字节一旦被
            # 从入站缓冲上切下来，就交给了一个 `Job`；`push_job` 说"满了"的时候，那个
            # `Job` 连同字节一起被销毁 —— 请求从此消失，客户端挂到超时，而服务端一行
            # 报错都没有。原来那句"满了就留着下一轮再派"是不成立的：字节已经没了，
            # 没有下一轮。
            #
            # 循环开头那次 `has_room()` 不够：这一轮里排在前面的槽位已经占了位置。
            var room_now = False
            try:
                room_now = box.mailbox.has_room()
            except err:
                print("  [srv] the mailbox could not be checked: " + String(err))
                return
            if not room_now:
                return
            # 请求要过线程，过的是**原始字节**而不是 `HttpRequest`：后者带着入站缓冲
            # 的借用关系，挪不进邮箱（见 `srv/conn.mojo` 的 `take_with_raw`）。
            var raw = List[UInt8]()
            var taken: Taken
            try:
                taken = self.conns[slot].take_with_raw(raw)
            except err:
                # 一个坏对端只带走它自己那条连接。
                print("  [srv] connection dropped: " + String(err))
                self.drop(slot)
                continue
            if taken.state == TAKEN_NEED_MORE:
                continue
            if taken.state == TAKEN_CLOSED:
                self.conns[slot].close_when_drained()
                if self.conns[slot].drained():
                    self.drop(slot)
                continue
            var queued = False
            try:
                # ⚠️ `gen` 与 `owner` 先拷成局部值：`owner_for` 会写 `self`，而把
                # `self.gen[slot]` 直接塞进同一个实参列表 = 一个不可变借用和一个
                # 可变借用同时活着（编译器按别名拒绝它）。
                var gen = self.gen[slot]
                var thread = self.owner_for(slot)
                queued = queue_request(box, slot, gen, thread, raw^)
            except err:
                print("  [srv] a request could not be queued: " + String(err))
            # 用 `continue` 而不是 `break`：邮箱满是一个**全局**上界，而这一轮里别的
            # 槽位可能刚好还有位置（`continue` 只是多扫几个槽位，代价可以忽略）。
            if not queued:
                continue
            self.inflight[slot] = 1
            budget -= 1

    def guard_stalls[H: EngineHandler](mut self, mut box: EngineBox) raises:
        """一个槽位"在途"却不再前进：先打几帧状态（隔 1 s，最多 4 帧），然后收掉它。

        只在**已经卡住**时才执行（热路径只多一次扫槽位），所以它不会像"每次派活都打
        印"那样把时序改掉 —— 那正是这次排查踩过的坑：加两行常规打印，现象就消失了。

        **收掉的三条判据必须同时成立**（少一条就会误杀）：
        - 这个槽位记着活（`inflight` 或 `streaming`）；
        - 超过 6 s 没等到结果（4 帧、每帧隔 1 s，最后一帧才动手）；
        - **引擎这边没有活**：`busy`（引擎正卡在一件活里）为 0 **且** `pending()`
          （邮箱里还压着待办的）为 0，都在锁内看。这一条把"引擎正在算"与"这条只是排
          在别人后面"一起排除在外 —— 还在算的那条不能收，否则 40× 慢（见记忆）会变成
          真的失败；排队的同样不能收，否则 `engines=1` 时第二条生成会被砍。

        为什么非收不可：卡住时实测邮箱是空的、engine 在空转，循环却一直以为有一帧在
        途 ⇒ 不会再派帧；而 `reap` 会跳过 `streaming=1` 的槽位，空闲超时也收不到它。
        两边都不收，客户端就只能挂到超时 —— SSE 没有 `Content-Length`，只能靠关连接
        定界，于是"流其实吐完了"也照样挂。
        """
        var now = monotonic_ms()
        for slot in range(len(self.live)):
            # 没有主人的槽位不看：它可能正在等 engine 收尾（`release_pending`），那个
            # 在途是有意为之，不是卡住。
            if self.live[slot] == 0 or self.stall_seen[slot] >= 4:
                continue
            if self.inflight[slot] != 1 and self.streaming[slot] != 1:
                continue
            # 诊断数必须**在锁内**读（见 `Mailbox.read_counters`）：直接读跨线程字段，
            # `-O2` 会把那次读提到循环外，于是每一帧都拿到同一个陈旧值 —— 那样打出来的
            # 数字不是"当时的状态"，只会把排查带偏。
            var jpush = 0
            var jtake = 0
            var rpush = 0
            var rtake = 0
            var busy = 0
            var polls = 0
            var runs = 0
            try:
                box.mailbox.read_counters(
                    jpush, jtake, rpush, rtake, busy, polls, runs
                )
            except err:
                print("  [srv] the engine counters could not be read: " + String(err))
            # ⚠️ **引擎正在一件活里、或邮箱里还压着活 ⇒ 这一条只是排在别人后面，不是卡
            # 住**。少了这一条，`engines=1` 时第二条生成会被误杀：它从派出去那一刻起就
            # "在途"，而第一条要跑满一整次生成（实测 5 s 那条把 6 s 的判据顶掉了）。
            #
            # 边界（诚实写下来）：这样一条**卡在 `handle` 里永远不出来**的活就收不掉了
            # —— 从外面看它和"正在算"没有区别。宁可留这个洞，也不能把排队中的活砍了：
            # 后者是**必现**的，前者是要引擎自己坏掉才出现。
            if busy != 0 or box.mailbox.pending() > 0:
                continue
            # 隔 1 s 再打一帧：只看一帧分不清"engine 还活着但在原地转"与"已经死了"。
            if now - self.last_seen[slot] <= 3000 + 1000 * self.stall_seen[slot]:
                continue
            self.stall_seen[slot] += 1
            print(
                "  [dbg] stalled slot="
                + String(slot)
                + " gen="
                + String(self.gen[slot])
                + " owner="
                + String(self.owner[slot])
                + " inflight="
                + String(self.inflight[slot])
                + " streaming="
                + String(self.streaming[slot])
                + " pending="
                + String(self.conns[slot].pending_len())
                + " mailbox="
                + String(box.mailbox.pending())
                + " jpush="
                + String(jpush)
                + " jtake="
                + String(jtake)
                + " rpush="
                + String(rpush)
                + " rtake="
                + String(rtake)
                + " busy="
                + String(busy)
                + " polls="
                + String(polls)
                + " runs="
                + String(runs)
                + " n="
                + String(self.stall_seen[slot])
            )
            if self.stall_seen[slot] == 4:
                # 最后一帧之后收掉它：清掉"还在等它"的标记 → 写完就关（`reap` 见到
                # `closing` 且队列空就丢槽位）。客户端看到的是"流没走到 `[DONE]` 就断
                # 了" —— 断言该看的是"没有请求能挂死"，不是"每条都必须走完"。
                #
                # ⚠️ 差一步：这里**没有**调 handler 的 `stream_end`（`EngineBox` 的约定
                # 是"循环不碰 handler"），所以引擎那一侧的请求号/槽位还占着。按 `drop`
                # 那条 TODO，把 handler 带进来才能连这一步一起收干净。
                self.inflight[slot] = 0
                self.streaming[slot] = 0
                self.conns[slot].close_when_drained()

    def sync_all(mut self) raises:
        """把所有连接的 interest 改成它现在真的用得上的（背压的落点）。"""
        for slot in range(len(self.live)):
            if self.live[slot] == 1:
                self.sync_interest(slot)

    def busy(self) -> Bool:
        """这轮 poll 该立刻返回吗（有字节要写、或有流在途）。"""
        if self.streaming_slot() >= 0:
            return True
        return self.has_pending_writes()

    def has_pending_writes(self) -> Bool:
        """有还没发出去的字节吗。**只看发送队列，不看有没有流在跑**。

        为什么拆出这一条：线程那条路上"有流在跑"**不能**再让 poll 超时归零 —— 帧是
        engine 线程产出来的，它产完会 `_wake` 把这条循环叫醒（`_wake` 的注释里那句
        "叫不醒就每帧晚 50 ms"说的正是这件事）。照旧归零的话，整条流期间这条循环都在
        死转：实测主线程 `state=R`、utime **+68/秒**（≈ 烧掉 0.7 个核），而 engine 那边
        在睡 —— 一个核在为一个"没有活要干"的循环转。

        真正需要立刻返回的只有一种情况：手上还有字节没发出去（`flush` 要靠 writable
        事件，而事件只有 poll 才拿得到）。
        """
        for slot in range(len(self.live)):
            if self.live[slot] == 1 and self.conns[slot].pending_len() > 0:
                return True
        return False

    # ── accept / read / write ───────────────────────────────────────────────

    def accept_ready(mut self) raises:
        """把 backlog 里的连接全接进来（直到 EAGAIN）。

        只接一条就走会让 level-triggered 事件一直报可读 —— 那不是"慢一点"，是满核
        空转。接到 EAGAIN 才是"这次真的没了"。
        """
        var round = 0
        while round < ACCEPT_PER_ROUND:
            round += 1
            var conn: TcpStream
            try:
                conn = accept_fd(self.lfd)
            except:
                return  # EAGAIN：backlog 空了
            var fd = Int32(conn.raw_fd())
            # 槽位满了：立刻关掉。留在 backlog 里同样会空转，而"接了又关"至少让对端
            # 马上知道（而不是等一个永远不来的应答）。
            if not self.has_room():
                conn.close()
                continue
            # 非阻塞在这里设：漏了它，`send` 会在慢客户端上把整条循环堵死 —— 而"慢
            # 客户端不再连累别人"正是这条循环存在的理由。
            if set_nonblocking(fd) != 0:
                conn.close()
                continue
            var slot = self.adopt(conn^, fd)
            if slot < 0:
                continue  # 登记不上（`adopt` 已经把它关掉并打过招呼了）
            self.last_seen[slot] = monotonic_ms()

    def has_room(self) -> Bool:
        """还有槽位吗。判据在**移动之前**问 —— 移动之后 `conn` 与槽位里的那份是同一
        个 fd，再"关掉它"就变成关两次（中间这个号可能已经被别人拿走了）。

        ⚠️ **等着收尾的槽位不算空位**：那条流在 engine 那边还没还回去，把槽位给新
        连接会让 `stream_end` 收到新那条流上（见 `adopt`）。宁可拒一条新连接，也不
        能把一条正在跑的流收掉。
        """
        if len(self.live) < MAX_CONNS:
            return True
        for slot in range(len(self.live)):
            if self.live[slot] == 0 and self.release_pending[slot] < 0:
                return True
        return False

    def adopt(mut self, var conn: TcpStream, fd: Int32) raises -> Int:
        """把一条新连接放进一个槽位（调用方已经确认有空位）。

        ⚠️ **等着收尾的槽位不给新连接**（`release_pending`）：那份收尾还在 engine 那边
        排队或已经在飞，而 `stream_end` 是按请求号（= 槽位）找流的 —— 这时候把槽位给
        新连接，被收掉的就是**新**那条流。宁可让这条连接去别的槽位（都没有就让
        `has_room` 把它拒掉）。
        """
        var target = -1
        for slot in range(len(self.live)):
            if self.live[slot] == 0 and self.release_pending[slot] < 0:
                target = slot
                break
        if target >= 0:
            var slot = target
            self.streams[slot] = conn^
            var fresh = Conn()
            self.conns[slot] = fresh^
            self.live[slot] = 1
            self.interest[slot] = INTEREST_READ
            self.inflight[slot] = 0
            self.streaming[slot] = 0
            self.owner[slot] = -1
            # 上一条连接的收尾**不再等**：这条槽位现在归新连接，而 `stream_end`
            # 是按请求号（= 槽位）找流的 —— 再派一次就会收掉**新**那条流。
            self.release_pending[slot] = -1
            self.release_sent[slot] = 0
            # ⚠️ 代号**不动**：换主人正是靠它自增来标记的（见 `drop`）。
            try:
                self.reactor.register(
                    fd, UInt64(CONN_TOKEN_BASE + slot), INTEREST_READ
                )
            except err:
                # 注册不上（EEXIST = 上一条连接的注册没摘干净，EBADF = fd 已经关了）
                # 只该带走这一条连接。让它冒出去 = 一次 `epoll_ctl` 失败就让整个服
                # 务没了 —— 而"一个倒霉的客户端不该带走所有人"正是这条循环的理由。
                print("  [srv] a connection could not be registered: " + String(err))
                self.drop(slot)
                return -1
            return slot
        if len(self.live) >= MAX_CONNS:
            return -1
        var fresh = Conn()
        self.conns.append(fresh^)
        self.streams.append(conn^)
        self.live.append(1)
        self.last_seen.append(monotonic_ms())
        self.interest.append(INTEREST_READ)
        self.gen.append(0)
        self.inflight.append(0)
        self.streaming.append(0)
        self.owner.append(-1)
        self.stall_seen.append(0)
        self.release_pending.append(-1)
        self.release_sent.append(0)
        try:
            self.reactor.register(
                fd, UInt64(CONN_TOKEN_BASE + len(self.live) - 1), INTEREST_READ
            )
        except err:
            print("  [srv] a connection could not be registered: " + String(err))
            self.drop(len(self.live) - 1)
            return -1
        return len(self.live) - 1

    def read_ready(mut self, slot: Int) raises:
        """这条连接上有字节了：读一截交给 `Conn`（一次一截，剩下的下一轮再读 ——
        事件是 level-triggered，没读完会再报一次，而"一条连接把一轮读满"会让别的
        连接饿着）。"""
        if self.conns[slot].backpressure():
            return
        var buf = List[UInt8](capacity=READ_CHUNK)
        buf.resize(READ_CHUNK, 0)
        var n: Int
        try:
            n = self.streams[slot].read(buf.unsafe_ptr(), READ_CHUNK)
        except err:
            if is_would_block(String(err)):
                return
            self.drop(slot)
            return
        if n == 0:
            # 对端关了。keep-alive 的连接就是这么结束的，不是错误。
            self.conns[slot].mark_peer_closed()
        else:
            var got = List[UInt8](capacity=n)
            var i = 0
            while i < n:
                got.append(buf[i])
                i += 1
            self.conns[slot].feed(got)
        self.last_seen[slot] = monotonic_ms()

    def flush(mut self, slot: Int) raises:
        """把队列里的一截写给对端；写多少由 `send(2)` 说了算。"""
        var pending = self.conns[slot].pending_len()
        if pending == 0:
            return
        var want = WRITE_CHUNK
        if pending < WRITE_CHUNK:
            want = pending
        var n: Int
        try:
            n = self.streams[slot].write(self.conns[slot].pending_view(want))
        except err:
            if is_would_block(String(err)):
                return
            # 对端没了（BrokenPipe / ConnectionReset）：只带走这一条连接。让异常冒出
            # `run` 等于"一个客户端挂了 = 服务退出"。
            self.drop(slot)
            return
        if n > 0:
            self.conns[slot].advance(n)
            self.last_seen[slot] = monotonic_ms()

    # ── 推进 ────────────────────────────────────────────────────────────────

    def progress[H: Handler](mut self, mut handler: H) raises:
        """每一条连接往前走一步：流在途的喂一帧，别的试着切一个请求来答。"""
        # 一次一条流（handler 的流状态只有一份）：有流在途就先喂它，这一轮不再答
        # 新的请求 —— 否则第二条流会从第一条的中间接着取帧。
        var s = self.streaming_slot()
        if s >= 0:
            self.step_stream[H](handler, s)
            return
        for slot in range(len(self.live)):
            if self.live[slot] == 0:
                continue
            if self.conns[slot].backpressure():
                self.sync_interest(slot)
                continue
            var taken: Taken
            try:
                taken = self.conns[slot].take()
            except err:
                # 一个坏对端（或发了一半就走的对端）只能带走它自己那条连接。让异常冒出
                # `run` 意味着"一个畸形请求 = 整个服务退出"，而一条循环服务所有人正是
                # 这一版要的形态。
                print("  [srv] connection dropped: " + String(err))
                self.drop(slot)
                continue
            if taken.state == TAKEN_NEED_MORE:
                continue
            if taken.state == TAKEN_CLOSED:
                # 没有要收的，也没有要发的（有要发的由 `reap` 等它写完）。
                self.conns[slot].close_when_drained()
                if self.conns[slot].drained():
                    self.drop(slot)
                continue
            self.answer[H](handler, slot, taken.request^)
            # 一条连接一轮只答一个请求：流水线上的第二个下一轮再答，否则一条连了
            # 十个请求的连接会把一轮占满。
            break
        for slot in range(len(self.live)):
            if self.live[slot] == 1:
                self.sync_interest(slot)

    def answer[H: Handler](mut self, mut handler: H, slot: Int, imm req: HttpRequest) raises:
        """答一个请求：生成 → 排队（**不写**，写由 `flush` 在可写时做）。

        给 handler 的 request 号就是**槽位**：两条连接不可能同号，所以 handler 那
        边的在途状态"属于谁"是能对得上的（见 `srv/server.mojo` 里 `Handler` 的
        文档）。
        """
        var response: HttpResponse
        try:
            response = handler.handle(slot, req)
        except err:
            # 500 不带细节进响应：内部原因属于日志，而这一版还没有日志分级，所以只在
            # 这里打印。消息仍然要打出来 —— 一个没有原因的 500 只能靠猜。
            print("  [srv] request failed: " + String(err))
            response = json_response(
                500,
                "{\"error\":{\"message\":\"the request could not be"
                + " answered\",\"type\":\"server_error\",\"code\":"
                + "\"internal\"}}",
                True,
            )
        # 对端说要关，就必须在响应里也说要关：只在写完以后默默关掉，对端会以为这条
        # 连接还能用，于是下一个请求发到一个正在被关的 socket 上 —— 那是一次看起来
        # 像"偶发连接重置"的失败。
        if not req.keep_alive():
            response.close = True
        if response.content_type == SSE_CONTENT_TYPE:
            # 只有头：帧由 `step_stream` 一帧一帧喂。头先排队（排在流的前面），所以
            # 客户端看到的顺序是头 → 帧 → `[DONE]` → 关连接。
            self.conns[slot].enqueue(bytes_of_text(serialize_stream_head(response.status)))
            self.streaming[slot] = 1
            return
        self.conns[slot].enqueue(bytes_of_text(serialize(response)))
        if response.close:
            self.conns[slot].close_when_drained()
        self.served += 1

    def streaming_slot(self) -> Int:
        """正在写流的那个槽位（-1 = 没有）。`run` 那一版用它维持"一次一条流"。"""
        for slot in range(len(self.live)):
            if self.live[slot] == 1 and self.streaming[slot] == 1:
                return slot
        return -1

    def step_stream[H: Handler](mut self, mut handler: H, slot: Int) raises:
        """给 `slot` 那条流喂**一帧**。

        一帧一轮（而不是一次把整条流生成完）：生成出来的字节要等客户端收，一次生成完
        等于把整条流先攒在内存里 —— 那是发送队列存在的意义要防的东西。
        """
        if slot < 0 or self.live[slot] == 0:
            return
        if self.conns[slot].backpressure():
            return
        var frame: String
        try:
            frame = handler.stream_next(slot)
        except err:
            print("  [srv] stream aborted: " + String(err))
            frame = ""
        if frame.byte_length() == 0:
            # 流结束。靠关连接定界（没有 `Content-Length`），所以这里是"写完就关"
            # 而不是"再写一帧"。
            self.conns[slot].close_when_drained()
            self.streaming[slot] = 0
            self.served += 1
            return
        self.conns[slot].enqueue(bytes_of_text(frame))

    def sync_interest(mut self, slot: Int) raises:
        """把这条连接关心的事件改成"它现在真的用得上的"。

        这一步是**背压的落点**：队列满了就撤掉"可读"（不然 level-triggered 会一直报
        可读而循环不再读 —— 满核空转），并且加上"可写"（要它把队列排空）。
        """
        if self.live[slot] == 0:
            return
        var want = INTEREST_READ
        if self.conns[slot].backpressure():
            want = INTEREST_WRITE
        elif self.conns[slot].pending_len() > 0:
            want = INTEREST_READ | INTEREST_WRITE
        if want == self.interest[slot]:
            return
        self.interest[slot] = want
        self.reactor.modify(Int32(self.streams[slot].raw_fd()), want)

    # ── 收尾 ────────────────────────────────────────────────────────────────

    def reap(mut self):
        """关掉该关的连接：写完该写的、或者空转太久的。"""
        var now = monotonic_ms()
        for slot in range(len(self.live)):
            if self.live[slot] == 0:
                continue
            if self.conns[slot].should_close():
                self.drop(slot)
                continue
            if now - self.last_seen[slot] <= IDLE_TIMEOUT_MS:
                continue
            # 空闲超时只收**真的没事干**的连接。三个例外，每一个都是"收错了会变成
            # 一次看起来随机的失败"：
            # - 有完整请求在排队：生成慢不等于这条连接该死（丢了它 = 客户端收到一个
            #   半截响应/连接重置，而它什么也没做错）；
            # - 还有字节没写出去：那是我们在等客户端收；
            # - 正在写流的那条：流的步子就是慢。
            if self.conns[slot].peek() == TAKEN_REQUEST:
                continue
            if self.conns[slot].pending_len() > 0:
                continue
            if self.streaming[slot] == 1:
                continue
            # 请求已经派给 engine 了（`run_threaded`）：缓冲是空的、也没有字节要
            # 写，单看连接它"什么都没干" —— 但它正在等一个应答。收掉它，客户端收到
            # 的就是一个半截响应，而它什么也没做错。
            if self.inflight[slot] > 0:
                continue
            # 剩下的才是"连着又不说话"（含"发了一半就不发了"）：`MAX_CONNS` 有上界，
            # 占满等于拒绝服务 —— 而那是外部能触发的。
            self.drop(slot)

    def drop(mut self, slot: Int):
        """关掉一条连接、把槽位还回去。

        一条**没走完**的流在这里被登记成待收尾（`release_pending`），由 `dispatch`
        派一件 `JOB_STREAM_END` 给 engine：状态是按 `request`（= 槽位）索引的，不显式
        交还，那个位置就一直占着 —— 占满的表现是"新请求被拒"，而它离真正的原因
        （有人断了连接）隔着一层。

        ⚠️ 这里**不能**直接调 handler：handler 归 engine 线程独占（`EngineBox` 的注
        释），从循环这一侧调一次，等于给"每条线程一份相位"撕开一个口子。
        """
        if self.live[slot] == 0:
            return
        var fd = Int32(self.streams[slot].raw_fd())
        try:
            self.reactor.unregister(fd)
        except:
            pass
        self.streams[slot].close()
        self.live[slot] = 0
        var fresh = Conn()
        self.conns[slot] = fresh^
        self.last_seen[slot] = monotonic_ms()
        # 换主人：代号自增。在途的结果回来时对不上代号 → 丢掉（它属于上一条连接）。
        self.gen[slot] += 1
        # ⚠️ 判据必须在 `inflight` 清零**之前**取出来：清零之后再问"它在途吗"，答案永远
        # 是"不在" —— 那样非流式请求被中途断开时就登记不上待收尾，engine 那一份相位一直
        # 占着这个号（等到同槽的下一条请求撞上来，两条活的相位叠在一起）。
        var was_inflight = self.inflight[slot]
        self.inflight[slot] = 0
        # 流跟着连接一起没了（在途的那一帧回来时对不上代号，会被 `collect` 丢掉）。
        # ⚠️ 但**engine 那一份还在**：登记待收尾，交给 `dispatch` 派 `JOB_STREAM_END`。
        # 只有"确实有一条流、且它已经有归属"才登记 —— 没有归属说明它还没派出去过，
        # engine 那边本来就没有东西要收。
        # ⚠️ 判据是"**engine 那边还占着这个号**"，不是"这条连接上开着一条流"。
        #
        # 一件活还在飞（或还排在邮箱里）的时候 `streaming` 可能还是 0 —— 流头都还没
        # 回来。那种情况下不登记，槽位下一轮就能被新连接拿走（它只避 `release_pending`），
        # 而新请求用的**是同一个请求号**（`handle(job.slot, …)`，请求号 = 槽位）⇒ 两件
        # 活共用一个号：engine 那侧的状态被第二个覆盖、被第一个释放。实测那次堆踩穿
        # （`Job.raw` 的 `magic`/长度全对、头 8 字节变成别人的堆地址，最后 abort 在
        # `Conn::enqueue` 的 realloc 上）只有客户端中途断开时才出现，形状正对这一条。
        #
        # `stream_end` 是幂等的（"号对不上就什么也不做"），所以给一个其实没有流的号
        # 派这一件是安全的 —— 宁可多派一次，也不能漏。
        if (
            self.streaming[slot] == 1 or was_inflight == 1
        ) and self.owner[slot] >= 0:
            self.release_pending[slot] = self.owner[slot]
            self.release_sent[slot] = 0
        self.streaming[slot] = 0
        # 归属也作废：下一个主人重新轮转（留着旧的会让"每条线程各几条"慢慢失衡）。
        self.owner[slot] = -1

    def drain_all(mut self) raises:
        """收尾：把手上还没写完的字节都发出去（有上限，不无限等）。"""
        var rounds = 0
        while rounds < 1000 and self.busy():
            rounds += 1
            for slot in range(len(self.live)):
                if self.live[slot] == 1 and self.conns[slot].pending_len() > 0:
                    self.flush(slot)

    def drop_all(mut self):
        for slot in range(len(self.live)):
            if self.live[slot] == 1:
                self.drop(slot)
