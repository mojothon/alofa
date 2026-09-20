"""N/M 快照 CI 门 —— 账本里写的数字，必须是这次真的跑出来的数字。

为什么这道门单独存在
--------------------
`test_ledger.mojo` 只校验 "evidence 文件存在"。够用了吗？不够。账本里曾经写着

    `evidence:tests/unit/test_engine_core.mojo`（14/14）

而同一份文件在账本别处还写着 10/10 和 15/15。文件当然存在，所以那道门一直是绿的
—— 三个互不相容的数字并存了很久。腐烂的不是"有没有测试"，而是**数字本身**。

所以这里多做一步：拿 `pixi run test` 落下的实测条数（`target/test_counts.tsv`）去
核对每个 `?count=N`。

自检门
------
和 `test_ledger.mojo` 一样，这里同样必须有负面用例。三种失效方式都要被证明能抓：

  1. 数字漂了（写了 15，实测 17）
  2. 声明了 count，但那个套件根本没在本次清单里（路径写错 / 没跑到）
  3. 跑了却没登记 count

第 3 条尤其重要：若只有"写了才查"，最省事的做法就变成"干脆一个都不写"，这道门会
在最宽松的地方失效。

为什么不顺便重跑一遍所有套件
----------------------------
因为 `pixi run test` 已经跑过了，并在它的过程中顺手写了这份清单 —— 于是核对是免费
的。注意：清单缺失时这里**必须失败**，而不是静默通过：静默跳过会让它退化成装饰。
"""

from std.io import FileHandle
from std.testing import TestSuite, assert_true

from ledger import (
    evidence_count,
    evidence_path,
    file_exists,
    parse_count,
    validate_ledger_with_counts,
    validate_text_with_counts,
)

comptime LEDGER = "docs/plan/capability-ledger.md"
comptime COUNTS = "target/test_counts.tsv"


def read_counts() raises -> String:
    var handle = FileHandle(COUNTS, "r")
    return handle.read()


def test_counts_artifact_exists() raises:
    """先决条件：没有实测清单就不能算通过，否则这道门会悄悄变成"没跑"。

    清单由 `pixi run test` 顺带产出，所以正常顺序是先 test 再 check-counts。
    """
    assert_true(
        file_exists(COUNTS),
        "缺少 "
        + COUNTS
        + " —— 请先跑 `pixi run test`（它会在过程中产出这份清单）；"
        + "这道门不接受「没数据就算通过」。",
    )


def test_real_ledger_counts_match_measurement() raises:
    """真账本里每个 `?count=N` 都必须与本次实测条数一致（0 错误）。"""
    var errors = validate_ledger_with_counts(LEDGER, read_counts(), False)
    assert_true(
        errors == 0,
        "账本 N/M 快照校验失败，错误数 = "
        + String(errors)
        + "（详见上方 [ledger] 输出；跑 `pixi run ledger-sync` 可批量刷新）",
    )


def test_drifted_count_is_caught() raises:
    """数字漂了必须被发现 —— 这是本门存在的全部理由。"""
    var counts = "tests/unit/test_scheduler.mojo\t17\n"
    var text = "| 能力 | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=15` |"
    var errors = validate_text_with_counts(text, "<inline>", False, counts)
    assert_true(errors == 1, "漂移应报 1 处错误，实际 = " + String(errors))


def test_unregistered_count_is_caught() raises:
    """跑过的套件没登记 count → 报错。否则"不写"就是绕过这道门的最短路径。"""
    var counts = "tests/unit/test_scheduler.mojo\t17\n"
    var text = "| 能力 | `verified` | `evidence:tests/unit/test_scheduler.mojo` |"
    var errors = validate_text_with_counts(text, "<inline>", False, counts)
    assert_true(errors == 1, "未登记 count 应报 1 处错误，实际 = " + String(errors))


def test_count_without_corresponding_run_is_caught() raises:
    """声明了 count，但该套件没出现在清单里 → 报错（路径错 / 没跑到）。"""
    var counts = "tests/unit/test_core_log.mojo\t8\n"
    var text = "| 能力 | `verified` | `evidence:tests/unit/test_scheduler.mojo?count=17` |"
    var errors = validate_text_with_counts(text, "<inline>", False, counts)
    assert_true(errors == 1, "清单里查不到时应报 1 处错误，实际 = " + String(errors))


def test_verified_remote_skips_count_check() raises:
    """`verified-remote` 的套件只在 A100 上执行，本机清单里不会有它，不该报错。"""
    var counts = "tests/unit/test_core_log.mojo\t8\n"
    var text = "| GPU kernel | `verified-remote` | `evidence:tests/gpu/vecadd.mojo` |"
    var errors = validate_text_with_counts(text, "<inline>", False, counts)
    assert_true(errors == 0, "verified-remote 应跳过核验，实际错误数 = " + String(errors))


def test_static_mode_does_not_require_counts() raises:
    """不带清单时只做静态校验 —— 保证 `test_ledger.mojo` 那条老门不受影响。"""
    var counts = "tests/unit/test_core_log.mojo\t8\n"
    var text = "| 能力 | `verified` | `evidence:tests/unit/test_core_log.mojo`（8/8 通过） |"
    var errors = validate_text_with_counts(text, "<inline>", False, "")
    assert_true(errors == 0, "静态模式下不该报 N/M 相关的错，实际 = " + String(errors))


def test_parse_count_handles_digits() raises:
    assert_true(parse_count("17") == 17, "应解析出 17")
    assert_true(parse_count("2048") == 2048, "应解析出 2048")
    # 数字后面紧跟其他字符（例如收尾的反引号）时要停住
    assert_true(parse_count("13`<br>") == 13, "应取前导数字 13")
    assert_true(parse_count("abc") == -1, "非数字应返回 -1")
    assert_true(parse_count("") == -1, "空串应返回 -1")


def test_evidence_path_and_count_split() raises:
    var token = "tests/unit/test_scheduler.mojo?count=17"
    assert_true(
        evidence_path(token) == "tests/unit/test_scheduler.mojo",
        "应剥掉 ?count= 后缀",
    )
    assert_true(evidence_count(token) == 17, "应取到 17")
    assert_true(
        evidence_path("tests/unit/test_core_log.mojo") == "tests/unit/test_core_log.mojo",
        "没有后缀时原样返回",
    )
    assert_true(
        evidence_count("tests/unit/test_core_log.mojo") == -1,
        "没有后缀时 count 应为 -1",
    )


def main() raises:
    print("Running ledger N/M snapshot gate...")
    TestSuite.discover_tests[__functions_in_module()]().run()
