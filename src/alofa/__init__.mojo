"""alofa — a verifiable LLM inference engine in pure Mojo.

Layers build strictly upward: `core` (L0) → `kernels` (L1) → `model` (L2) →
`runtime` (L3) → `engine` (L4) → `srv` (L5/L6). `verify/` and `tokenizer/`
are orthogonal pillars that no layer may depend on.

What alofa claims to be able to do is recorded in
`docs/plan/capability-ledger.md`, and that ledger is enforced by CI — see
`pixi run check-ledger`.
"""

from .core import AlofaError, Logger
from .verify import Counter, Roofline
