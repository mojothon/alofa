"""Architecture implementations for the L2 model layer.

One family per module, and one family to start with: Qwen2, done deeply
(configuration, parameters, prefill and decode) rather than several families
done shallowly. A second family is added when the first one has passed its
differential gates, not before.
"""
