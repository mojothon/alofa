"""Turning a row of logits into one token, in a way that can be checked.

Why this is written the way it is
---------------------------------

Every one of these transformations is three lines of arithmetic. None of them
is hard. What is hard is being able to say, out loud and with evidence, that
our `top_p` is the same `top_p` as Hugging Face's — because the failure mode
in this area is not a crash. It is a sampler that produces fluent text with a
subtly wrong distribution, and nobody notices until someone measures.

So the rule this file follows is: **the arithmetic is chosen to be reproducible,
not to be fast or clever.**

- Everything accumulates in `Float64`. A probability vector over 151936 entries
  summed in fp32 has a visible rounding error, and then "is our top-p cut the
  same as theirs?" becomes unanswerable.
- The order of operations is the order Hugging Face applies them, verified
  against `transformers` 4.41 rather than against a blog post.
- Removal is represented by a sentinel (`LOGIT_FLOOR`) rather than by a mask
  array, because that is literally what HF does — it fills with `-inf` — and
  mirroring it means the sentinel cannot accidentally behave differently at a
  boundary.

Where the semantics come from
-----------------------------

`temperature`, `top_k`, `top_p`, `min_p` and `repetition_penalty` have HF
counterparts and are checked against them (`scripts/dump_sampler_reference.py`
runs the real warpers and compares **surviving sets**).

`logit_bias`, `frequency_penalty` and `presence_penalty` have **no** HF
counterpart in 4.41. They follow the OpenAI / vLLM semantics: an additive
offset applied before anything else, and an additive penalty proportional to
how often a token has already appeared. Those three are *self-attested* —
pinned by explicit boundary cases rather than by an external implementation —
and the capability ledger says so. Conflating the two kinds of evidence would
be the easy mistake here.

Two penalties that look alike and are not
-----------------------------------------

`repetition_penalty` is **multiplicative** (divide if positive, multiply if
negative) and applies at most once per distinct token. `frequency_penalty` and
`presence_penalty` are **additive** and scale with the token's count. They are
kept under separate field names on purpose: a single `penalty` field would let
a caller switch between two incompatible semantics without noticing.

Allocation
----------

`Sampler` owns every buffer it needs and allocates once, at construction.
`build` and `pick` allocate nothing, which is what lets a P2 steady state
reach zero per-token allocation.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_sampler_parity.mojo
"""

from std.math import exp

from alofa.core.error import (
    ERR_CAPACITY,
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    AlofaError,
)
from alofa.core.memory import Arena

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]

# What a removed entry is set to. Not `-inf`, because subtracting `-inf` from
# `-inf` is a NaN and that would turn a mask bug into a silent poisoned
# distribution; and not a small finite number, because `exp` of it has to
# underflow to exactly zero for the softmax to agree with the reference's.
# 1e300 is far below any real logit and `exp(-1e300)` is 0.
comptime LOGIT_FLOOR = Float64(-1.0e300)


struct LogitBias(Copyable, Movable):
    """An additive offset on one vocabulary entry, applied before everything else."""

    var index: Int
    var value: Float64

    def __init__(out self, index: Int, value: Float64):
        self.index = index
        self.value = value


struct SampleParams(Copyable, Movable):
    """One step's worth of transformation settings.

    Defaults mean "do nothing": temperature 1, no filtering, no penalties. A
    caller that sets nothing gets plain sampling from the model's own
    distribution.
    """

    var temperature: Float64
    var top_k: Int
    var top_p: Float64
    var min_p: Float64
    var repetition_penalty: Float64
    var frequency_penalty: Float64
    var presence_penalty: Float64
    var min_tokens_to_keep: Int

    def __init__(out self):
        self.temperature = 1.0
        self.top_k = 0
        self.top_p = 1.0
        self.min_p = 0.0
        self.repetition_penalty = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.min_tokens_to_keep = 1


def softmax_f64(dst: F64Ptr, x: F64Ptr, n: Int) raises AlofaError:
    """`dst = softmax(x)` in double precision, accumulating strictly in order.

    The subtraction of the row maximum happens before `exp`, so the largest
    term is exactly 1 and nothing overflows. Entries at `LOGIT_FLOOR`
    underflow to exactly zero, which is what makes "removed" and "probability
    zero" the same thing without a second bookkeeping array.

    The order of accumulation is part of the contract: a different summation
    order gives a different last bit, and the differential gate compares
    against a reference that sums in this order.
    """
    var biggest = LOGIT_FLOOR
    for i in range(n):
        if x[unsafe_offset=i] > biggest:
            biggest = x[unsafe_offset=i]
    var total = Float64(0)
    for i in range(n):
        var term = exp(x[unsafe_offset=i] - biggest)
        dst[unsafe_offset=i] = term
        total += term
    if total == Float64(0):
        raise AlofaError(
            ERR_INVALID_ARGUMENT,
            "cannot normalise an empty distribution",
            "n=" + String(n),
        )
    for i in range(n):
        dst[unsafe_offset=i] = dst[unsafe_offset=i] / total


def pick_from(weights: F64Ptr, m: Int, u: Float64) -> Int:
    """Inverse CDF: the first index whose cumulative weight passes `u`.

    `u` is expected in [0, 1). Accumulation is in order, which is the same
    rule the reference uses, so the two agree on which side of a boundary a
    draw lands even when `u` is close to one.
    """
    var acc = Float64(0)
    for i in range(m):
        acc += weights[unsafe_offset=i]
        if u < acc:
            return i
    return m - 1


struct Sampler(Movable):
    """Applies the transformation chain to one row of logits, repeatedly.

    Buffers are owned here and sized at construction. `build` then fills them
    in place, so a generation loop allocates nothing per token.
    """

    var arena: Arena
    var capacity: Int
    var max_history: Int
    var values: F64Ptr
    var weights: F64Ptr
    var order: I32Ptr
    var aux: I32Ptr
    var hist_index: I32Ptr
    var hist_count: I32Ptr

    def __init__(out self, capacity: Int, max_history: Int) raises AlofaError:
        """Reserve room for rows of `capacity` and histories of `max_history`."""
        if capacity <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "capacity must be positive",
                "capacity=" + String(capacity),
            )
        if max_history <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "max_history must be positive",
                "max_history=" + String(max_history),
            )
        var wanted = capacity * 16 + capacity * 8 + max_history * 8 + 4096
        self.arena = Arena(wanted)
        self.capacity = capacity
        self.max_history = max_history
        self.values = self.arena.alloc(capacity * 8).unsafe_bitcast[Float64]()
        self.weights = self.arena.alloc(capacity * 8).unsafe_bitcast[Float64]()
        self.order = self.arena.alloc(capacity * 4).unsafe_bitcast[Int32]()
        self.aux = self.arena.alloc(capacity * 4).unsafe_bitcast[Int32]()
        self.hist_index = self.arena.alloc(max_history * 4).unsafe_bitcast[Int32]()
        self.hist_count = self.arena.alloc(max_history * 4).unsafe_bitcast[Int32]()
        self.arena.keep_alive()

    def probs(self) -> F64Ptr:
        """The distribution produced by the last `build`."""
        return self.weights

    def is_alive(self, i: Int) -> Bool:
        """Whether entry `i` survived the filters in the last `build`."""
        return self.values[unsafe_offset=i] > LOGIT_FLOOR

    def alive_count(self, n: Int) -> Int:
        """How many entries survived. This is the number the set gate asserts on."""
        var count = 0
        for i in range(n):
            if self.values[unsafe_offset=i] > LOGIT_FLOOR:
                count += 1
        return count

    def _tally(mut self, history: List[Int]) raises AlofaError -> Int:
        """Distinct tokens of `history` with their counts; returns how many.

        Linear in the number of *distinct* tokens per step, so the whole pass
        is quadratic in that number. For a decode loop that is bounded by the
        context length and small in practice; if it ever stops being small this
        is the function to replace, and the fix is a hash map, not a tolerance.
        """
        var distinct = 0
        for i in range(len(history)):
            var token = history[i]
            var at = -1
            for j in range(distinct):
                if Int(self.hist_index[unsafe_offset=j]) == token:
                    at = j
                    break
            if at < 0:
                if distinct >= self.max_history:
                    raise AlofaError(
                        ERR_CAPACITY,
                        "history is longer than the sampler was built for",
                        "len=" + String(len(history)) + " max=" + String(self.max_history),
                    )
                self.hist_index[unsafe_offset=distinct] = Int32(token)
                self.hist_count[unsafe_offset=distinct] = 1
                distinct += 1
            else:
                self.hist_count[unsafe_offset=at] = self.hist_count[unsafe_offset=at] + 1
        return distinct

    def _sort_descending(mut self, n: Int):
        """Order indices by value descending, ties by index ascending.

        A stable bottom-up merge sort. Stability is not a nicety: the
        reference breaks ties by lower index, and `min_tokens_to_keep` keeps
        "the largest one" — so with ties, which index survives depends
        entirely on this ordering being the same on both sides.
        """
        var order = self.order
        var aux = self.aux
        var values = self.values
        for i in range(n):
            order[unsafe_offset=i] = Int32(i)
        var width = 1
        while width < n:
            var start = 0
            while start < n:
                var mid = start + width
                if mid > n:
                    mid = n
                var end = start + width * 2
                if end > n:
                    end = n
                var a = start
                var b = mid
                var k = start
                while a < mid and b < end:
                    var left = Int(order[unsafe_offset=a])
                    var right = Int(order[unsafe_offset=b])
                    if values[unsafe_offset=left] >= values[unsafe_offset=right]:
                        aux[unsafe_offset=k] = order[unsafe_offset=a]
                        a += 1
                    else:
                        aux[unsafe_offset=k] = order[unsafe_offset=b]
                        b += 1
                    k += 1
                while a < mid:
                    aux[unsafe_offset=k] = order[unsafe_offset=a]
                    a += 1
                    k += 1
                while b < end:
                    aux[unsafe_offset=k] = order[unsafe_offset=b]
                    b += 1
                    k += 1
                start = end
            var swap = self.order
            self.order = self.aux
            self.aux = swap
            order = self.order
            aux = self.aux
            width = width * 2

    def build(
        mut self,
        logits: F32Ptr,
        n: Int,
        params: SampleParams,
        bias: List[LogitBias],
        history: List[Int],
    ) raises AlofaError:
        """Fill `probs()` with the distribution for this step.

        The order here is the order Hugging Face uses: processors first
        (bias, repetition, frequency, presence), then warpers (temperature,
        top-k, top-p, min-p), then the final softmax.
        """
        if n <= 0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT, "row length must be positive", "n=" + String(n)
            )
        if n > self.capacity:
            raise AlofaError(
                ERR_CAPACITY,
                "row is longer than the sampler was built for",
                "n=" + String(n) + " capacity=" + String(self.capacity),
            )

        var values = self.values
        for i in range(n):
            values[unsafe_offset=i] = Float64(logits[unsafe_offset=i])

        for b in bias:
            if b.index < 0 or b.index >= n:
                raise AlofaError(
                    ERR_OUT_OF_RANGE,
                    "bias index is outside the row",
                    "index=" + String(b.index) + " n=" + String(n),
                )
            values[unsafe_offset=b.index] = values[unsafe_offset=b.index] + b.value

        var distinct = self._tally(history)

        if params.repetition_penalty != 1.0:
            if params.repetition_penalty <= 0.0:
                raise AlofaError(
                    ERR_INVALID_ARGUMENT,
                    "repetition_penalty must be positive",
                    "value=" + String(params.repetition_penalty),
                )
            for j in range(distinct):
                var token = Int(self.hist_index[unsafe_offset=j])
                if token < 0 or token >= n:
                    raise AlofaError(
                        ERR_OUT_OF_RANGE,
                        "history token is outside the row",
                        "token=" + String(token) + " n=" + String(n),
                    )
                if values[unsafe_offset=token] < Float64(0):
                    values[unsafe_offset=token] = (
                        values[unsafe_offset=token] * params.repetition_penalty
                    )
                else:
                    values[unsafe_offset=token] = (
                        values[unsafe_offset=token] / params.repetition_penalty
                    )

        if params.frequency_penalty != 0.0 or params.presence_penalty != 0.0:
            for j in range(distinct):
                var token = Int(self.hist_index[unsafe_offset=j])
                if token < 0 or token >= n:
                    raise AlofaError(
                        ERR_OUT_OF_RANGE,
                        "history token is outside the row",
                        "token=" + String(token) + " n=" + String(n),
                    )
                var seen = Float64(Int(self.hist_count[unsafe_offset=j]))
                values[unsafe_offset=token] = (
                    values[unsafe_offset=token] - params.frequency_penalty * seen
                )
                values[unsafe_offset=token] = (
                    values[unsafe_offset=token] - params.presence_penalty
                )

        if params.temperature == 0.0:
            # Degenerate case: the whole distribution collapses onto the
            # maximum. Doing the division and then taking an argmax would give
            # the same index, but it would also give infinities on the way
            # there, and a NaN if the row ever contained a tie at the top.
            var best = 0
            for i in range(1, n):
                if values[unsafe_offset=i] > values[unsafe_offset=best]:
                    best = i
            for i in range(n):
                self.weights[unsafe_offset=i] = Float64(0)
                if i != best:
                    values[unsafe_offset=i] = LOGIT_FLOOR
            self.weights[unsafe_offset=best] = Float64(1)
            return

        if params.temperature < 0.0:
            raise AlofaError(
                ERR_INVALID_ARGUMENT,
                "temperature must not be negative",
                "temperature=" + String(params.temperature),
            )
        if params.temperature != 1.0:
            for i in range(n):
                values[unsafe_offset=i] = values[unsafe_offset=i] / params.temperature

        var keep = params.min_tokens_to_keep
        if keep < 1:
            keep = 1

        var needs_order = (
            (params.top_k > 0 and params.top_k < n)
            or params.top_p < 1.0
            or params.min_p > 0.0
        )
        if needs_order:
            self._sort_descending(n)

        if params.top_k > 0 and params.top_k < n:
            # HF removes `score < kth`, so everything *equal* to the kth
            # largest survives. Writing `>` here instead of `>=` would drop
            # tied entries and is exactly the kind of thing the ties fixture
            # exists to catch.
            var kth = values[unsafe_offset=Int(self.order[unsafe_offset=params.top_k - 1])]
            for i in range(n):
                if values[unsafe_offset=i] < kth:
                    values[unsafe_offset=i] = LOGIT_FLOOR

        if params.top_p < 1.0:
            if params.top_p <= 0.0:
                raise AlofaError(
                    ERR_INVALID_ARGUMENT,
                    "top_p must be positive",
                    "top_p=" + String(params.top_p),
                )
            softmax_f64(self.weights, self.values, n)
            var taken = 0
            var acc = Float64(0)
            for j in range(n):
                acc += self.weights[unsafe_offset=Int(self.order[unsafe_offset=j])]
                taken = j + 1
                if acc >= params.top_p:
                    break
            if taken < keep:
                taken = keep
            for j in range(taken, n):
                values[unsafe_offset=Int(self.order[unsafe_offset=j])] = LOGIT_FLOOR

        if params.min_p > 0.0:
            softmax_f64(self.weights, self.values, n)
            var biggest = Float64(0)
            for i in range(n):
                if self.weights[unsafe_offset=i] > biggest:
                    biggest = self.weights[unsafe_offset=i]
            var floor_p = params.min_p * biggest
            for i in range(n):
                if self.weights[unsafe_offset=i] < floor_p:
                    self.aux[unsafe_offset=i] = 1
                else:
                    self.aux[unsafe_offset=i] = 0
            for j in range(keep):
                if j >= n:
                    break
                self.aux[unsafe_offset=Int(self.order[unsafe_offset=j])] = 0
            for i in range(n):
                if self.aux[unsafe_offset=i] != 0:
                    values[unsafe_offset=i] = LOGIT_FLOOR

        softmax_f64(self.weights, self.values, n)

    def pick(self, n: Int, u: Float64) -> Int:
        """Draw one index from the distribution built by the last `build`."""
        return pick_from(self.weights, n, u)
