"""Master/worker 进程编排：SO_REUSEPORT 多 worker 的进程层（P3.4）。

形态对应 `docs/plan/02-architecture.md` §6.2：master 不监听、只管进程；每个
worker 自己 `bind(SO_REUSEPORT)` 同一端口（flare 的 `bind_reuseport`），内核按
连接四元组哈希分发，没有用户态负载均衡器。每个 worker 是独立的模型副本
+ 独立的一条 reactor 循环（`srv/loop.mojo`）—— 与单进程版同一个循环，只是
数量变成了 N。

为什么权重在 fork 之前加载
--------------------------
加载一次要十几秒（4.94 亿个 bf16 就地放宽成 fp32），而 fork 是纯内存操作：
子进程靠写时复制共享父进程已加载的那份，每个 worker 的私有成本只有会被
写的页（KV/激活缓冲）。注意这是「父进程从未跑过前向」为前提的 —— 加载路径
不碰并发运行时，asyncrt 的线程池（若存在）是子进程第一次前向时各自惰性建
的。这条前提由真权重冒烟（账本里 `ALOFA_WORKERS=2` 那条）看守。

停止通知为什么是「一个被关掉的描述符」而不是管道
------------------------------------------------
Mojo 没有模块级可变全局，信号处理器带不了任何状态；往管道里写字节又要
`read(2)`/`write(2)` —— 那两个符号 stdlib 已用别的签名声明过，再声明一次会在
lowering 阶段撞车（flare 为此专门绕道 dlopen）。这里用的记号是：worker 把
监听 fd `dup2` 到固定高位（`STOP_FD`），SIGTERM/SIGINT 处理器只做一次
`close(2)`（异步信号安全表内、无锁、无分配），服务循环轮询「这个 fd 还开着
吗」。选 900 这个号，是因为 worker 一次只处理一条连接，正常路径的 fd 都在
个位数循环里转，900 不会被别人复用；万一被复用，误报的方向是「没停」——
晚停，而不是不停，那是安全的失败方向。

master 怎么知道该停了
----------------------
- 到点收工（`run_seconds > 0`，门与压测用）：轮询里直接看钟。
- 收到 SIGTERM/SIGINT（生产路径，systemd 发的）：注册一个**什么都不做**的
  处理器，把默认的「进程立刻死亡」换成「打断 usleep」—— nanosleep 不受
  SA_RESTART 保护，usleep 返回 -1 就是信号到了。

醒不过来怎么办
--------------
标记和管道都一样有这个问题：等待会一直等下去。worker 现在不是阻塞在 accept
上，而是阻塞在 `epoll_wait` 上 —— 循环每一轮都会（最多 `POLL_IDLE_MS` 一次）
回到标记检查处，所以标记自己就能让它退出。master 仍然在发完 SIGTERM 之后补
几条「唤醒连接」（连上立刻关）：它们是给「等待更久」的那些形态留的兜底，多
发几条没有代价。SO_REUSEPORT 按四元组挑 worker，一条
唤醒连接只醒一个，所以发「存活数 + 余量」条。在途请求还没答完的 worker 不受
影响：它答完手头这条连接再看到标记。宽限（`grace_ms`）之后仍不退的，SIGKILL
兜底 —— 那「forced」会计数上报，不该无声发生。

worker 在运行期死了怎么办
--------------------------
fail-fast：master 停掉全部、上报异常数、以非零码退出，交给 systemd 的
`Restart=` 接管恢复。半套集群继续服务比快速重启更难排障。
"""

from std.collections import List
from std.ffi import external_call

from flare.net import SocketAddr
from flare.tcp import TcpStream
from flare.runtime.reuseport import bind_reuseport

from alofa.core.error import ERR_IO, ERR_INVALID_ARGUMENT, AlofaError
from alofa.core.ffi.posix import dup_to, fd_is_open, monotonic_ms
from alofa.srv.loop import Loop
from alofa.srv.engine_thread import EngineHandler

comptime SIG_INT = 2
comptime SIG_KILL = 9
comptime SIG_TERM = 15

# 停止标记的固定描述符号（文件头写了为什么是 900 这个高位号）。
comptime STOP_FD = Int32(900)

comptime POLL_US = 100_000
# SIGTERM 之后等处理器跑完、标记关掉，再发唤醒连接。
comptime TERM_SETTLE_US = 150_000
# 唤醒连接数 = 存活 worker 数 + 这个余量（哈希分发，多发几条才保险）。
comptime WAKE_EXTRA = 4

# wait4 的 WNOHANG。取状态用 `wait4` 而不是 `waitpid`：flare 自己声明过
# `waitpid`（状态参数传的是字面 0），同一符号两套签名会撞 lowering。
comptime WNOHANG = 1


def on_worker_stop(sig: Int32) -> None:
    """worker 的 SIGTERM/SIGINT 处理器：close 掉停止标记，仅此而已。

    只做 `close(2)`：异步信号安全表内、无锁、无分配。它带不了状态（Mojo
    没有模块级可变全局，处理器也拿不到闭包），状态由「标记 fd 被关掉」本身
    承载，服务循环轮询读到。
    """
    _ = external_call["close", Int32](STOP_FD)


def on_master_stop(sig: Int32) -> None:
    """master 的处理器什么都不做：把默认「进程死亡」换成「打断 usleep」。"""
    pass


@fieldwise_init
struct RunReport(Movable):
    """收工账：异常退出的 worker 数，与被 SIGKILL 兜底的数。

    `forced > 0` 意味着优雅路径没走通、靠 SIGKILL 硬停 —— 门要盯着这个数，
    否则「优雅退出」坏了也能无声通过。
    """

    var abnormal: Int
    var forced: Int


def reap_status(pid: Int32) -> Int32:
    """`wait4(pid, &status, WNOHANG)` 的原始 status；-1 = 还在跑（或出错）。

    解码归调用方：低 7 位非 0 = 被信号带走；否则退出码在位 8..15。
    """
    var status = Int32(0)
    var got = external_call["wait4", Int32](
        pid, Pointer(to=status), Int32(WNOHANG), Int32(0)
    )
    if got != pid:
        return Int32(-1)
    return status


def wake_idle(addr: SocketAddr, times: Int) -> None:
    """连上就关：让阻塞在 accept 上的 worker 醒一次，好走到标记检查处。

    连 0.0.0.0 在 Linux 上等于连 127.0.0.1，master 不需要区分监听地址。
    连不上（worker 都退了）不是错误，忽略即可。
    """
    for _ in range(times):
        try:
            var conn = TcpStream.connect(addr)
            conn.close()
        except err:
            pass


def worker_main[H: EngineHandler](
    mut handler: H,
    addr: SocketAddr,
    port: UInt16,
    max_requests: Int,
    engines: Int = 1,
) raises:
    """一个 worker 进程的全部工作：bind → 标记 → 接信号 → 服务循环。

    `engines` 是这个 worker 里 engine 线程的条数（roadmap 3.2c）。线程是在
    **fork 之后**才起的（`srv/engine_thread.mojo` 的 `spawn_engines`），所以这里
    不存在"fork 带着线程"的问题；而 `engines > 1` 时每条线程各一份权重（见
    `run_workers` 那段警告）。
    """
    # 父进程（master）死了也别当孤儿：内核会替我们补一发 SIGTERM，走进
    # 下面的优雅路径而不是漂着。
    _ = external_call["prctl", Int32](Int32(1), Int32(SIG_TERM), Int64(0))
    var listener = bind_reuseport(addr)
    if dup_to(listener.as_raw_fd(), STOP_FD) < 0:
        raise AlofaError(ERR_IO, "could not arm the stop marker", "")
    # 先把标记立起来，再挂处理器：反过来会出现「信号到了、标记还没立」的窗口。
    _ = external_call["signal", Int32](Int32(SIG_TERM), on_worker_stop)
    _ = external_call["signal", Int32](Int32(SIG_INT), on_worker_stop)
    # 一个 worker 现在跑的是一条 reactor 循环（`srv/loop.mojo`）**加 `engines` 条
    # engine 线程**（`srv/engine_thread.mojo`）：它同时挂着多条连接，而且一次前向
    # 不再把别的连接的读写一起按住 —— "一个慢客户端 = 一个 worker 下线"不再成立。
    # 同时能生成的条数 = `engines`（默认 1）。
    var loop = Loop.adopt_listener(listener^)
    var served = loop.run_threaded(handler, max_requests, STOP_FD, engines)
    print(
        "  [worker " + String(external_call["getpid", Int32]()) + "] served "
        + String(served) + " requests"
    )
    loop.close()


def begin_stop(imm pids: List[Int32], imm exited: List[Bool], imm addr: SocketAddr):
    """停止三步曲的第一步：SIGTERM 全员 → 等处理器跑完 → 唤醒连接。"""
    print("  [master] stopping workers: SIGTERM, drain, then SIGKILL if needed")
    for i in range(len(pids)):
        if not exited[i]:
            _ = external_call["kill", Int32](pids[i], Int32(SIG_TERM))
    _ = external_call["usleep", Int32](Int32(TERM_SETTLE_US))
    var alive = 0
    for i in range(len(pids)):
        if not exited[i]:
            alive += 1
    if alive > 0:
        wake_idle(addr, alive + WAKE_EXTRA)


def supervise(
    mut pids: List[Int32],
    mut exited: List[Bool],
    mut codes: List[Int],
    mut signaled: List[Bool],
    imm addr: SocketAddr,
    run_seconds: Int,
    grace_ms: Int,
    fail_fast_on_clean_exit: Bool,
) -> RunReport:
    """盯着 N 个 worker 直到全部退出。返回 abnormal / forced 计数。

    子进程账目用四个平行列表而不是 `List[SomeStruct]`：Mojo 的容器元素
    不隐式拷贝，基础类型没这个问题。
    """
    var abnormal = 0
    var forced = 0
    var stopping = False
    var killed = False
    var start = monotonic_ms()
    var grace_until = 0
    while True:
        # nanosleep 不受 SA_RESTART 保护：注册过的信号一来，usleep 返回 -1。
        var interrupted = external_call["usleep", Int32](Int32(POLL_US)) == Int32(
            -1
        )
        var alive = 0
        var do_stop = False
        for i in range(len(pids)):
            if exited[i]:
                continue
            var status = reap_status(pids[i])
            if status == Int32(-1):
                alive += 1
                continue
            exited[i] = True
            signaled[i] = (status & Int32(127)) != Int32(0)
            codes[i] = Int((status >> 8) & Int32(255))
            if signaled[i]:
                if stopping:
                    forced += 1
                else:
                    print(
                        "  [master] worker " + String(pids[i])
                        + " died by signal"
                    )
                    abnormal += 1
                    do_stop = True
            elif codes[i] != 0:
                print(
                    "  [master] worker " + String(pids[i]) + " exited with "
                    + "code " + String(codes[i])
                )
                abnormal += 1
                do_stop = True
            elif fail_fast_on_clean_exit and not stopping:
                # stopping 期间的干净退出是预期路径（停止三步曲干的就是这个），
                # 不算异常。
                print(
                    "  [master] worker " + String(pids[i]) + " exited while"
                    + " the fleet should be running"
                )
                do_stop = True
        if stopping:
            if alive == 0:
                break
            if not killed and monotonic_ms() >= grace_until:
                for i in range(len(pids)):
                    if not exited[i]:
                        print(
                            "  [master] escalating to SIGKILL for worker "
                            + String(pids[i])
                        )
                        _ = external_call["kill", Int32](
                            pids[i], Int32(SIG_KILL)
                        )
                killed = True
            continue
        if do_stop or interrupted:
            begin_stop(pids, exited, addr)
            stopping = True
            grace_until = monotonic_ms() + grace_ms
            continue
        if run_seconds > 0 and monotonic_ms() - start >= run_seconds * 1000:
            begin_stop(pids, exited, addr)
            stopping = True
            grace_until = monotonic_ms() + grace_ms
            continue
        if alive == 0:
            # 每个 worker 答满 max_requests 自然收工（门模式）。
            break
    return RunReport(abnormal, forced)


def run_workers[H: EngineHandler](
    mut handler: H,
    imm addr: SocketAddr,
    port: UInt16,
    workers: Int,
    max_requests: Int,
    run_seconds: Int,
    grace_ms: Int,
    engines: Int = 1,
) raises -> RunReport:
    """fork `workers` 个进程跑同一个 handler，盯到全部退出。

    `handler` 是可变借用：fork 让每个子进程拿到自己的一份拷贝（写时复制），
    母进程里的原件不动 —— 权重共享正是靠这一点。

    ⚠️ `engines > 1` 时那份"共享"就打了折：每个 worker 里每条 engine 线程**各一份
    权重**（`handler.spawn_twin` 是重新加载一份），所以总共是 `workers × engines`
    份。想并发而不想付这个代价，就加 worker（进程级共享是真的共享）。
    """
    if workers <= 0:
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "workers must be positive",
            "workers=" + String(workers),
        )
    _ = external_call["signal", Int32](Int32(SIG_TERM), on_master_stop)
    _ = external_call["signal", Int32](Int32(SIG_INT), on_master_stop)

    var pids = List[Int32]()
    var exited = List[Bool]()
    var codes = List[Int]()
    var signaled = List[Bool]()
    for _ in range(workers):
        var pid = external_call["fork", Int32]()
        if pid < 0:
            raise AlofaError(ERR_IO, "fork failed", "")
        if pid == 0:
            var code = Int32(0)
            try:
                worker_main(handler, addr, port, max_requests, engines)
            except err:
                print("  [worker] exited with an error: " + String(err))
                code = Int32(1)
            # `_exit`：fork 出来的子进程不走父进程的退出路径（析构噪音）。
            _ = external_call["_exit", Int32](code)
            # 不可达 —— `_exit` 不返回，但编译器不知道；这行只为了让子进程的
            # 控制流在这里结束。
            return RunReport(0, 0)
        pids.append(pid)
        exited.append(False)
        codes.append(0)
        signaled.append(False)

    return supervise(
        pids,
        exited,
        codes,
        signaled,
        addr,
        run_seconds,
        grace_ms,
        max_requests <= 0,
    )
