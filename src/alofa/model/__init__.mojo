"""L2 model layer: architectures and their parameters.

An architecture module here owns one model family end to end — configuration,
parameters, and the forward pass over L1 compute — and knows nothing about
concurrency, serving or verification. It is loaded directly
(`from alofa.model.arch.qwen import ...`).
"""
