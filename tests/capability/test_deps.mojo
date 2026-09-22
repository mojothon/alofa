"""外部纯 Mojo 依赖的可用性门。

这些 import 写在模块顶层，因此**"文件能否编译"本身就是断言**：
一旦 flare 或 json 的 API 发生破坏性变更，本文件编译失败 → CI 失败。

这直接服务于能力账本 §1.3：alofa 决定*依赖*而非*自研*服务层（A7），
那么"依赖是否还在"就必须由 CI 持续验证，而不是写在文档里靠人记着。

已实测（2026-09-16，alofa pixi 环境）：
  - `import flare`                          OK
  - `from flare.runtime import Reactor`     OK
  - `from flare.http import HttpServer`     OK
  - `import json`                           OK
2026-09-21 追加（单进程非流式 HTTP 用的就是这条）：
  - `from flare.net import SocketAddr`      OK
  - `from flare.tcp import TcpListener`     OK（阻塞式，不是 reactor 那条）
  - `from flare.tcp import TcpStream`       OK
2026-09-21 追加（多 worker，P3.4）：
  - `from flare.runtime.reuseport import bind_reuseport` OK（同端口二次 bind 实测成功）
  - `from std.os import getenv`             OK（编译产物里运行时读取实测成功）
未实测 / 不存在：
  - `from flare.net import TcpListener`     ✗ package 'net' does not contain 'TcpListener'
  - `from json import parse / Document`     ✗ 顶层不直接暴露
"""

import flare
import json
from flare.http import HttpServer
from flare.net import SocketAddr
from flare.runtime import Reactor
from flare.runtime.reuseport import bind_reuseport
from flare.tcp import TcpListener, TcpStream
from std.os import getenv

from std.testing import TestSuite, assert_true


def test_flare_package_available() raises:
    """flare 顶层包可用 —— L5/L6 服务层的地基。"""
    assert_true(True, "flare package imports")
    print("  flare 0.2.x available (network runtime: reactor/scheduler/timer_wheel/reuseport)")


def test_flare_reactor_type_available() raises:
    """`flare.runtime.Reactor` 可解析 —— 事件循环不是纸上谈兵。"""
    assert_true(True, "flare.runtime.Reactor imports")
    print("  flare.runtime.Reactor resolved")


def test_flare_http_server_type_available() raises:
    """`flare.http.HttpServer` 可解析 —— HTTP 层可直接依赖。"""
    assert_true(True, "flare.http.HttpServer imports")
    print("  flare.http.HttpServer resolved")


def test_flare_blocking_tcp_available() raises:
    """`flare.tcp` 的阻塞接口可解析 —— 单进程非流式 HTTP 的传输层是它。

    为什么钉这一条而不是只钉 `HttpServer`：`srv/` 这一版走的是**阻塞**那条路
    （`srv/server.mojo` 的文件头写了为什么），而阻塞接口和 reactor 是 flare 里两套
    不同的东西。只钉 reactor， flare 哪天把阻塞接口挪走/改名，这条依赖就断了而门还绿。
    """
    assert_true(True, "flare.tcp imports")
    print("  flare.tcp.TcpListener / TcpStream resolved (blocking transport)")
    print("  flare.net.SocketAddr resolved")


def test_json_package_available() raises:
    """JSON 包可用 —— config 与 OpenAI API 的序列化依赖。"""
    assert_true(True, "json package imports")
    print("  json available (SIMD two-pass parse + comptime reflection serde)")


def test_flare_reuseport_binds_two_listeners_on_one_port() raises:
    """SO_REUSEPORT 不只是能 import：同端口第二个 bind 也必须成功。

    多 worker 的地基是「每个 worker 各自 bind 同一端口、内核分发连接」——
    那在 reuseport 没真正生效时是 EADDRINUSE。这条门把第二个 bind 真做一遍，
    import 探测不到这一层。（`srv/master.mojo` 的多 worker 走的就是它。）
    """
    comptime P = UInt16(18992)
    var a = bind_reuseport(SocketAddr.localhost(P))
    var b = bind_reuseport(SocketAddr.localhost(P))
    b.close()
    a.close()
    print("  bind_reuseport: two listeners bound " + String(Int(P)) + " (SO_REUSEPORT)")


def test_std_os_getenv_contract() raises:
    """`std.os.getenv` 的契约：没设的变量取默认值。

    配置层（`srv/config.mojo`）整个建立在它上面。编译产物里运行时读取这一
    行为已单独实测（`ALOFA_WORKERS` 等部署变量就是这么进来的）；这条钉的是
    API 契约本身。
    """
    var v = getenv("ALOFA_DEFINITELY_UNSET_9F2", "fallback")
    assert_true(v == "fallback", "unset variable must yield the default")
    print("  std.os.getenv: unset -> default")


def main() raises:
    print("Running external-dependency capability gate...")
    TestSuite.discover_tests[__functions_in_module()]().run()
