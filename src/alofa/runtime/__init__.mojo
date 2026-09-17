"""L3 runtime layer: turning one step of a forward pass into what a caller wants.

Everything here is about a *single* sequence. Nothing in this layer knows about
concurrency, batching or serving — those are L4 and above, and the reason they
are separate is that the hard part of a runtime is scheduling, while the hard
part of this layer is being *numerically defensible*.

The layer depends only on `core`. If a module here ever needs `kernels` or
`model`, that is a sign the thing being written belongs one layer up.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_sampler_parity.mojo
"""
