"""Tests for the roofline skeleton.

Every expected value here is an integer that follows from the inputs by hand:
4 MB in 1 ms is 4 GB/s, and against a 8 GB/s peak that is 500 permille. A
floating-point tolerance would let a wrong formula pass as long as it was
close, which is the opposite of what a measurement type is for.

The bottleneck tests use peaks chosen so the machine's balance point is exactly
1 flop/byte, which makes the expected side obvious by inspection.

Run:
    pixi run mojo run -I src tests/unit/test_verify_roofline.mojo
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.verify.roofline import (
    BOTTLENECK_BALANCED,
    BOTTLENECK_COMPUTE,
    BOTTLENECK_MEMORY,
    Counter,
    Roofline,
    StopWatch,
    bottleneck_name,
)

# A machine whose balance point is exactly 1 flop per byte.
comptime PEAK_BANDWIDTH = 1_000_000_000
comptime PEAK_FLOPS = 1_000_000_000


def test_bandwidth_is_bytes_over_time() raises:
    # 4,000,000 bytes in 1,000,000 ns = 4,000,000,000 bytes/s.
    var counter = Counter("copy", 4_000_000, 0, 1_000_000)
    assert_equal(counter.bandwidth_bytes_per_s(), 4_000_000_000)


def test_throughput_is_flops_over_time() raises:
    # 2,000,000 flops in 1,000,000 ns = 2,000,000,000 flops/s.
    var counter = Counter("matmul", 0, 2_000_000, 1_000_000)
    assert_equal(counter.flops_per_s(), 2_000_000_000)


def test_utilizations_are_exact_permille() raises:
    var roofline = Roofline(8_000_000_000, 4_000_000_000)
    # 4 GB/s of an 8 GB/s peak is half; 2 GFLOP/s of a 4 GFLOP/s peak is half.
    var counter = Counter("mixed", 4_000_000, 2_000_000, 1_000_000)
    assert_equal(roofline.bandwidth_utilization_permille(counter), 500)
    assert_equal(roofline.compute_utilization_permille(counter), 500)


def test_utilization_can_exceed_the_peak() raises:
    """A peak that is too low must read as over 1000, not as clamped."""
    var roofline = Roofline(1_000_000_000, 1_000_000_000)
    var counter = Counter("fast", 2_000_000, 0, 1_000_000)
    assert_equal(roofline.bandwidth_utilization_permille(counter), 2000)


def test_memory_bound_when_intensity_is_below_the_balance_point() raises:
    var roofline = Roofline(PEAK_BANDWIDTH, PEAK_FLOPS)
    # 1 flop per 1000 bytes, against a machine that wants 1 flop per byte.
    var counter = Counter("stream", 1000, 1, 1_000_000)
    assert_equal(roofline.bottleneck(counter), BOTTLENECK_MEMORY)


def test_compute_bound_when_intensity_is_above_the_balance_point() raises:
    var roofline = Roofline(PEAK_BANDWIDTH, PEAK_FLOPS)
    # 1000 flops per byte, against a machine that wants 1 flop per byte.
    var counter = Counter("dense", 1, 1000, 1_000_000)
    assert_equal(roofline.bottleneck(counter), BOTTLENECK_COMPUTE)


def test_balanced_at_the_balance_point() raises:
    var roofline = Roofline(PEAK_BANDWIDTH, PEAK_FLOPS)
    var counter = Counter("knee", 1000, 1000, 1_000_000)
    assert_equal(roofline.bottleneck(counter), BOTTLENECK_BALANCED)


def test_bottleneck_survives_realistic_magnitudes() raises:
    """真实量级下 cross-multiply 曾经**整数溢出**，把带宽受限判成 balanced。

    这里用一次真前向的量级：`bytes` 到 5e8、峰值到 1e10，两侧乘积就是 2.5e18 与
    5e18。它们本身还装得进 Int64，但旧实现要再乘 1000（容差是千分比），一乘就
    越过 Int64 上界绕回负数 —— 负的 `difference` 恒小于等于任何东西，于是判定变成
    了「永远 balanced」。

    ⚠️ 这个 bug **只能**在真实量级上出现：本文件其余用例都用 4 MB 和 1e9 这种数，
    乘积到不了 1e18，所以它们全绿，而真正跑起来的时候这道门是哑的。留这条大数用
    例不是为了覆盖率好看，是为了让「万一有人把整数改窄」当场被抓住。

    期望值是可以手算的：平衡点是 1 GB/s 对 1 GFLOP/s，即 1 flop/byte；这条的算术
    强度是 0.5 flops/byte，只有平衡点的一半 —— 差了一倍，当然不该判成 balanced。
    """
    # 一台真机器的两个峰值：10 GB/s 带宽、70 GFLOP/s 算力 → 平衡点 7 flops/byte。
    var roofline = Roofline(10_000_000_000, 70_000_000_000)
    # `bytes × 算力峰值` = 5e8 × 7e10 = 3.5e19，溢出。
    var counter = Counter("real_scale", 500_000_000, 250_000_000, 50_000_000)
    assert_equal(roofline.bottleneck(counter), BOTTLENECK_MEMORY)


def test_realistic_magnitudes_also_classify_compute() raises:
    """同一量级的算力受限也必须照判 —— 别让修一边把另一边弄坏。

    与上一条反着来，而且换一边溢出：`flops × 带宽峰值` = 1e9 × 1e10 = 1e19，越过
    Int64 上界后绕回**正数**，rhs 也可能被比下去 —— 溢出不是「总是判 balanced」，
    是「判什么都可能」，所以两个方向都得留。
    """
    var roofline = Roofline(10_000_000_000, 70_000_000_000)
    # 强度 10 flops/byte，高于平衡点 7。
    var counter = Counter("real_scale_flops", 100_000_000, 1_000_000_000, 50_000_000)
    assert_equal(roofline.bottleneck(counter), BOTTLENECK_COMPUTE)


def test_bottleneck_names_are_stable() raises:
    assert_equal(bottleneck_name(BOTTLENECK_MEMORY), "memory")
    assert_equal(bottleneck_name(BOTTLENECK_COMPUTE), "compute")
    assert_equal(bottleneck_name(BOTTLENECK_BALANCED), "balanced")


def test_peaks_must_be_supplied() raises:
    """A zero peak would silently make every utilization meaningless."""
    var caught = "no-error"
    try:
        var roofline = Roofline(0, PEAK_FLOPS)
        _ = roofline.compute_utilization_permille(Counter("x", 1, 1, 1))
    except err:
        caught = err.name()
    assert_equal(caught, "invalid_argument")


def test_a_zero_length_interval_is_rejected() raises:
    var caught = "no-error"
    try:
        var counter = Counter("instant", 1000, 1000, 0)
        _ = counter.bandwidth_bytes_per_s()
    except err:
        caught = err.name()
    assert_equal(caught, "invalid_argument")


def test_negative_counts_are_rejected() raises:
    var caught = "no-error"
    try:
        var counter = Counter("backwards", -1, 0, 1000)
        _ = counter.bandwidth_bytes_per_s()
    except err:
        caught = err.name()
    assert_equal(caught, "invalid_argument")


def test_stopwatch_measures_a_positive_interval() raises:
    var watch = StopWatch()
    var spin = 0
    while spin < 100000:
        spin += 1
    assert_true(watch.elapsed_ns() > 0, "a measured interval must be positive")
    var counter = watch.counter("spin", 0, spin)
    assert_equal(counter.label, "spin")


def test_report_carries_achieved_and_utilization_together() raises:
    """A utilization on its own hides both the measurement and the peak."""
    var roofline = Roofline(PEAK_BANDWIDTH, PEAK_FLOPS)
    # 1 MB in 1 ms is 1 GB/s, which is the whole peak: 1000 permille.
    var counter = Counter("copy", 1_000_000, 1_000_000, 1_000_000)
    var line = roofline.report(counter)
    assert_true(
        line.find("bytes_moved=1000000") != -1, "report must carry the raw counts"
    )
    assert_true(
        line.find("bandwidth_utilization_permille=1000") != -1,
        "report must carry the utilization",
    )
    assert_true(
        line.find("bottleneck=balanced") != -1, "report must name the bottleneck"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
