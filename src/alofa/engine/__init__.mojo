"""L4 engine layer: deciding *what* to run, before anything is run.

This layer exists because of one observation about the reference projects: their
schedulers are tangled with their models, so the interesting edges — a preemption
storm, a zero budget, a cancel that lands on the same tick as an arrival — can
only be reached once you have weights, a GPU and a server. Which in practice
means they are never reached at all.

So the rule here is: **the scheduler touches no model, no I/O and no clock.**
It takes events in, and it emits a pure data structure out. Everything about it
is therefore reproducible from a file, which is what `engine/trace.mojo` and
`tests/unit/test_scheduler.mojo` do.

The layer depends on `core` only. Nothing in L0–L3 may depend on it.

Run:
    pixi run mojo run -I src tests/unit/test_scheduler.mojo
"""
