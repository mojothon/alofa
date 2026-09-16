"""能力账本 CI 门 —— 账本能不能自我证明。

这组测试是 alofa 的 P0 门本身，跑三层：

1. **真账本必须通过** —— 所有 `verified` 类条目都有真实凭证。
2. **坏账本必须被拒绝**（`tests/fixtures/ledger_bad.md`）—— 这是**自检门**：
   证明第 1 条的"通过"不是因为校验器是个空壳。
3. **最小用例** —— 直接构造"标了 `verified` 但文件不存在"的行，断言它被抓。

第 2 条是整套机制的关键。一个只会通过的门等于没有门：如果有人把
`file_exists` 检查删掉，真账本依然"通过"，但第 2、3 条会立刻失败。

自证循环
--------
`docs/plan/capability-ledger.md` 里「能力账本 CI 校验」这条的 `evidence:` 指向
**本文件**。也就是说：账本用自己声明的门来证明自己可信。
"""

from std.testing import TestSuite, assert_true

from ledger import validate_ledger, validate_text


def test_real_ledger_passes() raises:
    """真账本必须 0 错误 —— 每个 verified 都要有真实凭证。"""
    var errors = validate_ledger("docs/plan/capability-ledger.md", False)
    assert_true(
        errors == 0,
        "账本校验失败，错误数 = " + String(errors) + "（详见上方 [ledger] 输出）",
    )


def test_bad_ledger_fixture_is_rejected() raises:
    """自检门：含 4 处违规的 fixture 必须被拒绝。这证明门本身有效。"""
    var errors = validate_ledger("tests/fixtures/ledger_bad.md", False)
    assert_true(
        errors >= 4,
        "坏 fixture 应至少报 4 处违规，实际 = "
        + String(errors)
        + " → 校验器可能被改弱了",
    )


def test_verified_with_missing_file_is_rejected() raises:
    """最小用例：标 `verified` 但 evidence 文件不存在 → 必须报错。"""
    var text = "| 编造能力 | `verified` | `evidence:tests/no_such_file.mojo` |"
    var errors = validate_text(text, "<inline>", False)
    assert_true(errors == 1, "应报 1 处错误，实际 = " + String(errors))


def test_verified_without_evidence_is_rejected() raises:
    """标 `verified` 却没给 evidence → 必须报错。"""
    var text = "| 编造能力 | `verified` | 我觉得它应该能跑 |"
    var errors = validate_text(text, "<inline>", False)
    assert_true(errors == 1, "应报 1 处错误，实际 = " + String(errors))


def test_verified_env_requires_probe() raises:
    """`verified-env` 必须给出可复现的观测命令，否则报错。"""
    var text = "| 编造的硬件 | `verified-env` | 据说是 8 张卡 |"
    var errors = validate_text(text, "<inline>", False)
    assert_true(errors == 1, "应报 1 处错误，实际 = " + String(errors))


def test_misspelled_label_is_caught() raises:
    """`verifed` 这类拼写错误不能被静默放过。"""
    var text = "| 编造能力 | `verifed` | `evidence:tests/no_such_file.mojo` |"
    var errors = validate_text(text, "<inline>", False)
    assert_true(errors == 1, "拼错标签应被捕获，实际错误数 = " + String(errors))


def test_placeholder_evidence_is_rejected() raises:
    """`evidence:待补` 这类占位符不能蒙混过关。"""
    var text = "| 编造能力 | `verified` | `evidence:待补` |"
    var errors = validate_text(text, "<inline>", False)
    assert_true(errors == 1, "占位符应被拒绝，实际错误数 = " + String(errors))


def main() raises:
    print("Running capability ledger gate...")
    TestSuite.discover_tests[__functions_in_module()]().run()
