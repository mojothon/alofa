"""Roofline measurement skeleton.

A roofline report says two things about a measured interval: how much of the
machine's memory bandwidth it used, and how much of its arithmetic throughput.
Everything else — tokens per second, requests per second — is a consequence of
those two and of the workload, which is why the ledger asks for utilization
rather than a headline number.

**This module contains no performance numbers and no device constants.** Both
peaks are constructor arguments. A peak that is assumed rather than measured
would turn every utilization computed from it into an unverified claim, which
is the one thing this repository's capability ledger is built to reject.

Ratios are integers in permille (thousandths) rather than floats. A utilization
is a measurement, not a real number: it will be compared against a threshold,
printed, and asserted on, and every one of those is exact with integers and
needs a tolerance with floats. The roofline comparison itself is a
cross-multiplication — `flops * peak_bandwidth` against
`bytes * peak_flops` — which decides memory-bound vs compute-bound without
ever dividing.

Run:
    pixi run mojo run -I src tests/unit/test_verify_roofline.mojo
"""

from alofa.core.error import ERR_INVALID_ARGUMENT, AlofaError
from alofa.core.ffi import monotonic_ns
from alofa.core.log import LEVEL_INFO, Logger, log_line

comptime BOTTLENECK_MEMORY = 0
comptime BOTTLENECK_COMPUTE = 1
comptime BOTTLENECK_BALANCED = 2

# How close to the machine's balance point a measurement has to be before it is
# called balanced. Without a band, a workload sitting one flop away from the
# knee would be reported as bound by whichever side it happened to fall on.
comptime BALANCE_TOLERANCE_PERMILLE = 50


def bottleneck_name(kind: Int) -> String:
    """The name used in reports: `memory`, `compute`, or `balanced`."""
    if kind == BOTTLENECK_COMPUTE:
        return "compute"
    elif kind == BOTTLENECK_MEMORY:
        return "memory"
    return "balanced"


struct Counter(Copyable, Movable):
    """One measured interval: what moved, what was computed, how long it took.

    `bytes_moved` is bytes crossing the memory hierarchy, not bytes of a
    tensor — a counter that counted its input twice by reading it twice would
    report the wrong bandwidth, and the whole point of the measurement is to
    catch exactly that.
    """

    var label: String
    var bytes_moved: Int
    var flops: Int
    var elapsed_ns: Int

    def __init__(
        out self,
        label: String,
        bytes_moved: Int,
        flops: Int,
        elapsed_ns: Int,
    ) raises AlofaError:
        if bytes_moved < 0 or flops < 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "counts cannot be negative",
                "bytes_moved="
                + String(bytes_moved)
                + " flops="
                + String(flops),
            )
        # A zero-duration interval cannot be turned into a rate. Rejecting it
        # here is better than reporting an infinite bandwidth, which would look
        # like a very fast machine rather than a broken measurement.
        if elapsed_ns <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "interval must have a positive duration",
                "elapsed_ns=" + String(elapsed_ns),
            )
        self.label = label
        self.bytes_moved = bytes_moved
        self.flops = flops
        self.elapsed_ns = elapsed_ns

    def bandwidth_bytes_per_s(self) -> Int:
        """Bytes per second, truncated to a whole number."""
        return self.bytes_moved * 1_000_000_000 // self.elapsed_ns

    def flops_per_s(self) -> Int:
        """Floating-point operations per second, truncated to a whole number."""
        return self.flops * 1_000_000_000 // self.elapsed_ns


struct Roofline(Copyable, Movable):
    """A machine's two ceilings, both supplied by the caller."""

    var peak_bandwidth_bytes_per_s: Int
    var peak_flops_per_s: Int

    def __init__(
        out self, peak_bandwidth_bytes_per_s: Int, peak_flops_per_s: Int
    ) raises AlofaError:
        if peak_bandwidth_bytes_per_s <= 0 or peak_flops_per_s <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "peaks must be positive and supplied by the caller",
                "peak_bandwidth_bytes_per_s="
                + String(peak_bandwidth_bytes_per_s)
                + " peak_flops_per_s="
                + String(peak_flops_per_s),
            )
        self.peak_bandwidth_bytes_per_s = peak_bandwidth_bytes_per_s
        self.peak_flops_per_s = peak_flops_per_s

    def bandwidth_utilization_permille(self, counter: Counter) -> Int:
        """Fraction of peak bandwidth used, in thousandths."""
        return (
            counter.bandwidth_bytes_per_s() * 1000
            // self.peak_bandwidth_bytes_per_s
        )

    def compute_utilization_permille(self, counter: Counter) -> Int:
        """Fraction of peak arithmetic throughput used, in thousandths."""
        return counter.flops_per_s() * 1000 // self.peak_flops_per_s

    def bottleneck(self, counter: Counter) -> Int:
        """Whether the interval was bound by bandwidth, by arithmetic, or neither.

        Compares the workload's arithmetic intensity (`flops / bytes`) with the
        machine's balance point (`peak_flops / peak_bandwidth`) by
        cross-multiplying, so the answer never depends on a division.
        """
        var lhs = counter.flops * self.peak_bandwidth_bytes_per_s
        var rhs = counter.bytes_moved * self.peak_flops_per_s

        var difference = lhs - rhs
        if difference < 0:
            difference = -difference
        var larger = lhs
        if rhs > larger:
            larger = rhs

        if difference * 1000 <= BALANCE_TOLERANCE_PERMILLE * larger:
            return BOTTLENECK_BALANCED
        elif lhs < rhs:
            return BOTTLENECK_MEMORY
        return BOTTLENECK_COMPUTE

    def report(self, counter: Counter, ts_ms: Int = 0) -> String:
        """Render one JSONL record for a measured interval.

        Same field set as every other alofa log line, so a consumer that reads
        logs can read measurements without a second parser. The achieved rates
        are recorded alongside the utilizations: a utilization alone is
        unfalsifiable, since it hides both the measurement and the peak it was
        divided by.
        """
        return log_line(
            ts_ms,
            LEVEL_INFO,
            "roofline",
            "roofline.sample",
            counter.label,
            "bytes_moved="
            + String(counter.bytes_moved)
            + " flops="
            + String(counter.flops)
            + " elapsed_ns="
            + String(counter.elapsed_ns)
            + " bandwidth_bytes_per_s="
            + String(counter.bandwidth_bytes_per_s())
            + " flops_per_s="
            + String(counter.flops_per_s())
            + " bandwidth_utilization_permille="
            + String(self.bandwidth_utilization_permille(counter))
            + " compute_utilization_permille="
            + String(self.compute_utilization_permille(counter))
            + " bottleneck="
            + bottleneck_name(self.bottleneck(counter)),
        )

    def emit(self, counter: Counter):
        """Write `report` through a logger. Never raises."""
        Logger("roofline", LEVEL_INFO).info(
            "roofline.sample",
            counter.label,
            "bandwidth_utilization_permille="
            + String(self.bandwidth_utilization_permille(counter))
            + " compute_utilization_permille="
            + String(self.compute_utilization_permille(counter))
            + " bottleneck="
            + bottleneck_name(self.bottleneck(counter)),
        )


struct StopWatch(Copyable, Movable):
    """A `CLOCK_MONOTONIC` interval, ready to become a `Counter`.

    Monotonic rather than wall time: an interval that jumps when the system
    clock is adjusted would make a utilization look better or worse for a
    reason that has nothing to do with the code being measured.
    """

    var started_ns: Int

    def __init__(out self):
        self.started_ns = monotonic_ns()

    def restart(mut self) -> Int:
        """Start a new interval, returning the one just finished."""
        var now = monotonic_ns()
        var elapsed = now - self.started_ns
        self.started_ns = now
        return elapsed

    def elapsed_ns(self) -> Int:
        """Nanoseconds since the interval started, without ending it."""
        return monotonic_ns() - self.started_ns

    def counter(
        self, label: String, bytes_moved: Int, flops: Int
    ) raises AlofaError -> Counter:
        """Close the interval into a `Counter`, raising if no time passed."""
        return Counter(label, bytes_moved, flops, self.elapsed_ns())
