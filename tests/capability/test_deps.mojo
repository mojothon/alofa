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
未实测 / 不存在：
  - `from flare.net import TcpListener`     ✗ package 'net' does not contain 'TcpListener'
  - `from json import parse / Document`     ✗ 顶层不直接暴露
"""

import flare
import json
from flare.http import HttpServer
from flare.runtime import Reactor

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


def test_json_package_available() raises:
    """JSON 包可用 —— config 与 OpenAI API 的序列化依赖。"""
    assert_true(True, "json package imports")
    print("  json available (SIMD two-pass parse + comptime reflection serde)")


def main() raises:
    print("Running external-dependency capability gate...")
    TestSuite.discover_tests[__functions_in_module()]().run()
