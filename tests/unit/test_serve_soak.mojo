"""真权重服务的**流式浸泡门**：一个进程连续答 N 条 SSE，不许错、不许断、不许自己死。

它盯的是哪条性质
----------------
2026-09-24 手工冒烟（`pixi run serve` + curl）时，真权重服务在**连续**流式请求下
先后吐出：

- `invalid_argument(1): the scheduler asked to decode a request that is not resident`
- `invalid_argument(1): duplicate request id`
- `HTTP/1.1 500` + `{"code":"internal"}`
- 流被截掉（没有 `data: [DONE]`），最后**进程崩溃**（栈停在 `_alloc_bytes`）

而现有的 SSE / reactor / engine 三条端到端门**全部用替身 handler**（不带权重），
替身跑 200 条都不崩 —— 也就是说"真权重 + 连续多条流"这条路径从来没有被走过。
**单条**请求一直是好的（非流式 2 s、流式 1.4 s），所以任何只发一条的门都看不见它。

为什么必须 fork 一个子进程来跑服务
----------------------------------
崩溃会把整个进程带走。同一个进程里既跑服务又跑断言的话，失败的表现是"门自己死了"
（退出码里看不出是被杀还是崩），而不是"门红了"；fork 之后，父进程能看见**子进程
自己先没了**，那才是可断言的事实。

为什么判据里有"不许出现 error 帧"这一条
----------------------------------------
上面那三条报错里，前两条是**帧里的 error**（`data: {"error": …}`），HTTP 状态行
仍然是 `200 OK`。只查状态码的话它们全绿 —— 而线上客户端看到的是"流了一半然后
一个错误帧"，那正是账本里"半流式比明确拒绝更糟"那一类。

为什么这一条要用 `-O2` 而不是 `-O0` 构建
----------------------------------------
（详见 `pixi run test-soak` 与 `pixi.toml` 里那行注释。）最初的记载是"崩溃只在无调试
信息的 -O2 codegen 下现形"，2026-09-24 实测**修正**了它：

- 带 `-g` 的 `-O2` **能**复现，只是形状不同：第 3 个请求拿到 `duplicate request id
  (scheduler)` 的 error 帧（根因已在 `EngineCore.release` 修掉）；修好之后 `-g` 下
  40 条全绿（`test-soak-g`）。所以 **`-g` 不是"复现不出来"，是"换了个说法"**。
- **不**带 `-g` 的那一条至今仍红：流式请求 30 s（实测把超时抬到 120 s 也一样）一个
  字节都不回来，服务子进程还活着、在**算**（`STAT=Rl`，不是死锁）。而同一份逻辑只
  因门的常量改动（预热条数 0/2、流数 0/1/40）就在"通过 / 卡死"之间翻转 —— 这是 UB
  的形状，不是死锁也不是普通的慢。
- ASan 不可用：它 abort 在 Mojo stdlib 自己的 `List.__init__`（`safetensors.mojo:169`
  那个 `List[UInt8]`），报 `beg is not aligned by 8` —— 是 stdlib 容器注解与 ASan 的
  集成误报，进程在**加载权重**阶段就被掐死，根本没跑到流式。本机也没有 valgrind。
  别再在这两条路上花时间。

门必须跑在**出事的那个编译选项**上，否则它证明的是别的东西。也就是说：在"无 -g 的那
一个"修好之前，`pixi run test-soak` 是**红的**，且它红得对 —— 别为了让它绿去改判据。

为什么不在 `pixi run test` 里
------------------------------
它 fork（JIT 下的 fork 会崩编译器 → 必须先 build），而且要加载 988 MB bf16
（就地放宽成 fp32）、跑几十条真前向 → 独立入口 `pixi run test-soak`。
"""

from std.collections import List
from std.ffi import external_call
from std.testing import assert_true

from flare.net import IpAddr, SocketAddr
from flare.tcp import TcpStream

from alofa.core.ffi.posix import monotonic_ms
from alofa.serve import ModelService
from alofa.srv.config import ServeConfig
from alofa.srv.http import bytes_of_text, bytes_to_text
from alofa.srv.loop import Loop
from alofa.srv.openai import ChatHandler

comptime PORT = UInt16(18140)
comptime WEIGHTS = "tests/fixtures/qwen2.5-0.5b-hf"
comptime TOKENIZER = "tests/fixtures/qwen2.5-0.5b/tokenizer.json"
# 连发多少条。手工冒烟时崩在第 ~25 条，取 40 是留一倍余量：门要在"确实坏了"的那条
# 路径上**稳定**红，而不是靠运气 —— 一条会 flaky 的门比没有门更糟（它会被人
# `--no-verify` 掉，顺带把旁边那条真的门也一起废掉）。
comptime SOAK_STREAMS = 40
# 每条请求生成几个 token。取小值是为了让一轮 soak 在几分钟内跑完：这条门盯的是
# "第 N 条之后服务还活着"，不是生成质量（质量有别的门）。
comptime GEN_TOKENS = 4
# ⚠️ 当前是 **1**，不是"跟线上对齐"的 8 —— 这是一个**按住的变量**，不是随手填的值：
# shards=8 时 fork 之后 asyncrt 不可用（父进程碰过并发运行时）会让**连非流式都卡**，
# 那是另一条坑（与"流式卡死"不是一个病）。先把并发度按住，才能干净地盯流式这一条。
# 要回到 8 的话，先确认 `test-soak-g`（-g 那条）在 8 下也全绿，否则两个病会混在一起。
comptime SOAK_SHARDS = 1
# 请求体里带不带 `"temperature":0`。**这是"走哪条路"的开关**（见 `serve.mojo` 的
# `ModelService`）：
#   - 空字符串 = 不带 → 服务默认 `temperature=1.0` → 走**单流老路**（`model.prefill`
#     / `model.step`，一次只握一条）。
#   - `"temperature":0` = 显式贪心 → 走**批路**（`EngineCore`，一次前向推进多条）。
# 手工冒烟那天崩的是**老路**（默认参数），所以默认留空；把它填上就是"批路也这样吗"。
# 两条路都能红 / 都不能红，是把范围砍掉一半的最快办法。
comptime SOAK_TEMP_FIELD = ""
# soak 之前先发一条**非流式**请求当预热（`True` = 发）。
#
# 为什么要它：2026-09-24 手工冒烟时，"服务起来后第一个请求就是流式"那次**卡死**
# （180 s 无响应），而后来同一台服务上的流式只用了 1.44 s —— 差别只在于中间夹过
# 非流式请求。所以"第一条流式"和"第 N 条流式"可能是**两件事**，拿它当开关能把
# 这个疑问一次性钉死：
#   - 开了预热就全绿 → 病在"流式第一次走到的那条初始化路径"，范围骤减；
#   - 开了预热照样红 → 病在流式本身，与第几条无关。
#
# 2026-09-24 实测（**无 -g** 那一条）把这个开关的答案记下来，别再重做一遍：
#   - `WARMUP=0, STREAMS=1` → 第一条流式 **1736 ms 通过**；
#   - `WARMUP=2`（非流式）→ 之后的流式**卡死**（30 s / 120 s 都不回来）；
#   - `WARMUP=2` 但预热**换成流式** → 第一条（作为预热的那条）就卡。
# 也就是"第几条才卡"随构建翻转 —— 卡不卡与"前面跑过几条"没有稳定关系，这条开关
# 没能把范围砍掉一半，它量到的是**不稳定**。而 `STREAMS=0`（只发非流式）在无 -g 下
# 全过 ⇒ 非流式那条路本身是好的。
# ⚠️ 它不是"把门弄绿"的手段：预热之后那 40 条**照旧**要全干净，判据一条没少。
comptime SOAK_WARMUP = 2
# 卡住之后**先等这么久再收尸**，期间服务子进程还活着 —— 这是留给外面采样用的窗口：
# "卡住了"有两类（死循环 = 烧 CPU；死锁 = 0% CPU），判据完全不同，而从客户端只看
# 得见"没字节来"。窗口里可以从 `/proc/<pid>/task/*/stat` 看每线程的 utime/stime
# 增量、用 gdb 抓栈。默认 0 = 关（正常跑门时不该有这段停顿）。
comptime STUCK_HOLD_MS = 0
# 客户端接收超时：服务没答就报错，而不是把这个门挂住（挂住的门看起来像"还在跑"）。
comptime CLIENT_TIMEOUT_MS = 30000
# 等子进程就绪的上限：真权重（988 MB bf16 → fp32 放宽）加载实测约 30 s，给 4 分钟是
# 为了让"加载慢"不被误判成"没起来"—— 这两件事的处置完全不同。
comptime READY_TIMEOUT_MS = 240000
comptime READY_POLL_MS = 200
comptime READ_CHUNK = 4096
comptime HEALTH_REQUEST = (
    "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
)


def connect(port: UInt16) raises -> TcpStream:
    """连上服务，并给客户端自己设一个接收超时。"""
    var conn = TcpStream.connect(SocketAddr.localhost(port))
    conn.set_recv_timeout(CLIENT_TIMEOUT_MS)
    return conn^


def read_more(mut conn: TcpStream, mut buf: List[UInt8]) raises -> Int:
    var base = len(buf)
    buf.resize(base + READ_CHUNK, 0)
    var got = conn.read(buf.unsafe_ptr().unsafe_offset(base), READ_CHUNK)
    buf.resize(base + got, 0)
    return got


def drain(mut conn: TcpStream, mut buf: List[UInt8]) raises:
    """一直读到 EOF。流式响应靠关连接定界，所以 EOF 就是它的结尾。

    服务没关（也没答）的话这里会撞上接收超时并抛错 —— 那正是"卡住了"该有的失败
    方式：挂住比报错糟得多。
    """
    while read_more(conn, buf) > 0:
        pass


def send_text(mut conn: TcpStream, imm text: String) raises:
    var sent = conn.write(text.as_bytes())
    assert_true(sent == text.byte_length(), "the write was short")


def post_request(imm body: String) -> String:
    """`Content-Length` 由体算出来，不手填 —— 手填的长度是这条门最该抓的那类错，
    而它一旦出现在**测试**里，门就变成"两个错互相印证"。
    """
    return (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        + "Host: localhost\r\n"
        + "Content-Length: "
        + String(body.byte_length())
        + "\r\n\r\n"
        + body
    )


def stream_body() -> String:
    return (
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        + SOAK_TEMP_FIELD
        + "\"max_tokens\":"
        + String(GEN_TOKENS)
        + ",\"stream\":true}"
    )


def warmup_body() -> String:
    """预热用的**非流式**请求：同一个 prompt，只是不带 `stream`。

    （见 `SOAK_WARMUP` 的注释：它要分辨的是"流式第一次"还是"流式本身"。）
    """
    return (
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        + SOAK_TEMP_FIELD
        + "\"max_tokens\":"
        + String(GEN_TOKENS)
        + "}"
    )


def long_body() -> String:
    """一条 prompt 超过 `max_prompt_tokens`（128）的请求。

    它是这条门的**负向对照**：它必须被**在响应头之前**拒成 400。若它也返回 200，
    说明"零 error 帧"只是因为门根本没看内容 —— 一个把请求全丢掉的实现也能零错误。
    """
    var text = "hello world "
    for _ in range(60):
        text += "hello world "
    return (
        "{\"messages\":[{\"role\":\"user\",\"content\":\""
        + text
        + "\"}],\"max_tokens\":"
        + String(GEN_TOKENS)
        + ",\"stream\":true}"
    )


def child_serve() raises:
    """子进程：加载真权重，然后跑一条 reactor 循环（`run_threaded`）。

    分片由 `SOAK_SHARDS` 定，与 `pixi run serve` 的默认值（自动 = 核数）对齐。
    （fork 之后 asyncrt 不可用这条坑是真的，但那是**父进程碰过并发运行时**才成立；
    这个父进程只做 socket，不加载模型、不碰 asyncrt —— 所以子进程自己初始化是安
    全的。真出现"第一个前向死等"的话，那本身是一条要记的发现，不是调这个数绕过去。）
    """
    var cfg = ServeConfig(
        host="127.0.0.1",
        port=PORT,
        weights_dir=WEIGHTS,
        config_path=WEIGHTS + "/config.json",
        weights_file=WEIGHTS + "/model.safetensors",
        tokenizer_json=TOKENIZER,
        model_name="qwen2.5-0.5b",
        max_tokens=256,
        max_prompt_tokens=128,
        seed=1234,
        shards=SOAK_SHARDS,
        workers=1,
        run_seconds=0,
        grace_ms=1000,
        max_requests=0,
        engine_threads=1,
    )
    var service = ModelService.load(cfg)
    service.model.shards = SOAK_SHARDS
    var handler = ChatHandler(service^, "qwen2.5-0.5b")
    var loop = Loop.bind_addr(SocketAddr(IpAddr.parse("127.0.0.1"), PORT))
    _ = loop.run_threaded(handler, 0, Int32(-1), 1)
    loop.close()


def wait_ready() raises -> Bool:
    """轮询 `/health` 直到服务答话（或超时）。

    为什么是轮询而不是"睡固定 30 秒"：加载时间随机器负载漂（本机长期过载），睡固定
    时长会在忙的时候变成"服务还没起来就开始发请求"，症状是连接被拒 —— 那看起来
    像服务坏了，其实是门没等。
    """
    var began = monotonic_ms()
    while monotonic_ms() - began < READY_TIMEOUT_MS:
        try:
            var conn = connect(PORT)
            var buf = List[UInt8]()
            send_text(conn, HEALTH_REQUEST)
            drain(conn, buf)
            var text = bytes_to_text(buf, 0, len(buf))
            if text.find("\"status\":\"ok\"") >= 0:
                return True
        except:
            pass
        _ = external_call["usleep", Int32](Int32(READY_POLL_MS * 1000))
    return False


def child_running(pid: Int32) -> Bool:
    """子进程还在跑吗（`waitpid` 的 `WNOHANG`）。

    ⚠️ **不能用 `kill(pid, 0)`**：进程崩溃之后会变成**僵尸**，僵尸仍然存在，
    `kill(pid, 0)` 照样返回 0 —— 那样"服务已经死了"会被判成"还活着"，而"进程没了"
    正是这条门最要紧的那条判据（前三条只描述症状，只有它是崩溃本身）。

    顺带：`WNOHANG` 返回 > 0 时子进程已被回收，之后的收尸会拿不到它 —— 这没有
    影响，收尸那一步只是确保不留下会跟下一轮抢端口的残留。
    """
    # 第二个参数是 `int *status`，这里传 **NULL**：退出码对这条判据不重要（无论是
    # 崩溃还是正常退出，都算"服务自己没了"）。声明成 `Int64` 是为了让它占满一个
    # 指针宽度的寄存器 —— 声明成 `Int32` 的话高 32 位不由这个调用决定，而那正是
    # "传半个指针"这类错的样子。
    var got = external_call["waitpid", Int32](pid, Int64(0), Int32(1))
    return got == 0


def kill_child(pid: Int32):
    """把服务子进程收掉（SIGKILL）。

    ⚠️ 不收的后果不是"多一个进程"那么轻：`Loop` 的监听 socket 是 **SO_REUSEPORT**
    的，下一轮新 fork 出来的服务会和上一轮遗留的那个**一起**监听同一个端口，连接
    被内核随机分给其中一个 —— 症状是"请求偶尔没人答""响应只有半截"，看起来完全
    像间歇性 bug。所以**失败的路径上也要收**。
    """
    _ = external_call["kill", Int32](pid, Int32(9))
    _ = external_call["usleep", Int32](Int32(50000))


def one_stream(pid: Int32, index: Int, imm body: String) raises -> String:
    """发一条流式请求，读到 EOF（连接被关）为止。

    读不到头时**先分辨"卡住"和"死了"再报错**：这两件事的处置完全不同（前者是
    生成没推进，后者是进程没了），而它们从客户端看都是"没字节来"。分辨的办法是
    回头问一次 `waitpid`（`child_running`）—— 这一步只在失败路径上跑，不影响
    门本身的判据。
    """
    var conn = connect(PORT)
    var buf = List[UInt8]()
    send_text(conn, post_request(body))
    try:
        drain(conn, buf)
    except err:
        # 先说清楚**卡在哪一步**再抛：index 是 -2（预热）/ -1（对照）/ >=0（第几条流式）。
        # 没有这一行的话，失败只是 `Timeout: recv`，看不出是预热就卡还是第 17 条才卡。
        print(
            "  [soak] STUCK on "
            + ("warmup (non-stream)" if index == -2 else "stream " + String(index))
            + "; service still alive="
            + String(Int(child_running(pid)))
        )
        if child_running(pid):
            if STUCK_HOLD_MS > 0:
                print(
                    "  [soak] holding the service for "
                    + String(STUCK_HOLD_MS)
                    + "ms (pid="
                    + String(Int(pid))
                    + ") for sampling"
                )
                # 卡住之后再问两件事（各走一条**新**连接，所以不受卡住那条的影响）：
                # ① `/health` —— 循环整体还答不答？② 再来一条流式 —— 是"只卡那一条"
                # 还是"从此不服务"？两者都卡 = 循环整体停了；health 秒回 = 循环在转，
                # 病在派发/在途状态。
                var during = one_raw(pid, HEALTH_REQUEST)
                print("  [soak] during-stuck /health -> " + during)
                var again = one_raw(pid, post_request(stream_body()))
                print("  [soak] during-stuck stream -> " + again)
                # 一次 `usleep` 睡 90 秒在 Linux 上会被 EINVAL 悄悄放过去（于是窗口
                # 变成 0 秒），所以按 1 秒一段睡。
                for _ in range(STUCK_HOLD_MS // 1000):
                    _ = external_call["sleep", UInt32](UInt32(1))
            raise err
        assert_true(
            False,
            "the service exited on its own while answering stream "
            + String(index)
            + " (it was there for the request, then it was gone)",
        )
    return bytes_to_text(buf, 0, len(buf))


def one_raw(pid: Int32, imm raw: String) raises -> String:
    """发一段**原始** HTTP（不是 JSON body 包装），拿到什么回什么；卡住就回 `<stuck>`。

    只在"已经卡住"之后用：那时服务的循环还在不在转，是两类完全不同的病（整体不
    服务 vs 只卡那一条），而这两类从外面看都是"没字节来"。
    """
    var conn = connect(PORT)
    var buf = List[UInt8]()
    send_text(conn, raw)
    try:
        drain(conn, buf)
    except:
        return "<stuck>"
    return bytes_to_text(buf, 0, len(buf))


def assert_clean_stream(index: Int, imm res: String) raises:
    """钉住一条流式响应的形状与"没有 error 帧"。"""
    assert_true(
        res.find("HTTP/1.1 200 OK\r\n") == 0,
        "stream " + String(index) + " is not a 200: " + res,
    )
    assert_true(
        res.find("Content-Type: text/event-stream\r\n") > 0,
        "stream " + String(index) + " did not announce SSE: " + res,
    )
    # 流式响应不许带 Content-Length（写第一帧时长度未知）。
    assert_true(
        res.find("Content-Length") < 0,
        "stream " + String(index) + " claimed a length: " + res,
    )
    # 这一条是这条门的重点：那三条已知报错都是**帧里的 error**，状态行仍是 200。
    assert_true(
        res.find("\"error\"") < 0,
        "stream "
        + String(index)
        + " carried an error frame (the service is already broken): "
        + res,
    )
    assert_true(
        res.find("data: [DONE]\r\n\r\n") >= 0,
        "stream " + String(index) + " was cut off before [DONE]: " + res,
    )


def main() raises:
    print(
        "  [soak] "
        + String(SOAK_STREAMS)
        + " streams × "
        + String(GEN_TOKENS)
        + " tokens, real weights ("
        + WEIGHTS
        + ")"
    )
    var pid = external_call["fork", Int32]()
    if pid == 0:
        child_serve()
        # 直接 `_exit`：子进程不再跑一遍父进程的析构（fork 出来的 Mojo 运行时在退出
        # 路径上会打一段栈，那是噪音，不是这个门要盯的东西）。
        _ = external_call["_exit", Int32](Int32(0))
        return

    try:
        assert_true(
            wait_ready(),
            "the service never answered /health in "
            + String(READY_TIMEOUT_MS)
            + "ms — either the weights are missing or it died while loading",
        )
        print("  [soak] service ready")

        # 预热发**两条非流式**，并且它们自己也要被断言（否则"预热失败"会以"第一条
        # 流式卡住"的样子出现，把两件事混成一件）。
        #
        # 为什么是**两条**：2026-09-24 抓到的最短复现是"非流式 → 流式(OK) → 流式(卡)"，
        # 于是要分辨的是「第二个**请求**就卡」还是「第二条**流式**才卡」：
        #   - 第二条非流式也卡 → 病在"第二次请求"（与流式无关），范围在通用请求路径；
        #   - 两条非流式都过、第二条流式卡 → 病在流式的**状态回收**（第一条流没还回去）。
        for w in range(SOAK_WARMUP):
            var warm = one_stream(pid, -2, warmup_body())
            assert_true(
                warm.find("HTTP/1.1 200 OK\r\n") == 0,
                "the warmup request " + String(w) + " was not a 200: " + warm,
            )
            assert_true(
                warm.find("\"finish_reason\"") >= 0,
                "the warmup response " + String(w) + " had no finish_reason: " + warm,
            )
            print("  [soak] warmup " + String(w) + ": non-stream answered")

        for i in range(SOAK_STREAMS):
            assert_true(
                child_running(pid),
                "the service died on its own before stream "
                + String(i)
                + " (it must survive all "
                + String(SOAK_STREAMS)
                + ")",
            )
            var began = monotonic_ms()
            var res = one_stream(pid, i, stream_body())
            assert_clean_stream(i, res)
            # 逐条打印（而不是每 10 条）：出事时要知道的是"卡在**第几条**"和"前面
            # 每条各用了多久"—— 那条耗时曲线本身就是线索（是恒定慢、还是越跑越慢）。
            print(
                "  [soak] stream "
                + String(i)
                + ": "
                + String(monotonic_ms() - began)
                + "ms"
            )

        # 负向对照（常驻，**放在 soak 之后**）：超长 prompt 必须在响应头之前被拒成
        # 400。它若也返回 200，说明上面的"零 error 帧"只是因为门根本没看内容。
        # 放后面的理由：它是"一条被拒绝的请求"，先跑它会让 soak 的起点带上一点
        # 状态；而这条门要盯的性质是"连续 N 条**正常**请求"，不该被它搅进去。
        var rejected = one_stream(pid, -1, long_body())
        assert_true(
            rejected.find("HTTP/1.1 400") == 0,
            "an over-long prompt was not rejected before the head: " + rejected,
        )
        print("  [soak] control: over-long prompt rejected with 400")
    except err:
        kill_child(pid)
        raise err

    kill_child(pid)
    print("  [soak] " + String(SOAK_STREAMS) + " streams clean, service alive")
