"""The verification pillar.

`verify/` answers "is what alofa produced the same as what the reference
produced, and how far from the machine's limits is it?" Layers L0–L6 must
never depend on this package: verification is not part of running a model, and
a layer that imported it could not be reasoned about on its own.

This package depends *downward* only — on `alofa.core` for error reporting,
timing, and log formatting.

What is here in P0 is a skeleton: the shape of a roofline measurement, with no
numbers in it. `docs/plan/capability-ledger.md` requires every performance
claim to be reported as a roofline utilization on named hardware, and a
utilization is only meaningful next to a peak that the caller states. Peaks are
therefore parameters, never constants — there is deliberately no table of
device bandwidths in this file, because a number copied from a spec sheet is
exactly the kind of unverified claim the ledger exists to prevent.

Run:
    pixi run mojo run -I src tests/unit/test_verify_roofline.mojo
"""

from .roofline import (
    BALANCE_TOLERANCE_PERMILLE,
    BOTTLENECK_BALANCED,
    BOTTLENECK_COMPUTE,
    BOTTLENECK_MEMORY,
    Counter,
    Roofline,
    StopWatch,
    bottleneck_name,
)
