"""A source of bit patterns that can be reproduced elsewhere, bit for bit.

Why this exists instead of reaching for the standard library
------------------------------------------------------------

The whole point of a differential gate is that two implementations, written on
different sides of a fixture, agree **exactly**. A generator taken from the
standard library cannot be mirrored on the reference side: its algorithm is not
part of any interface, it may change between releases, and reproducing it means
re-deriving it from a shipped binary. Whatever it produces can therefore only
be eyeballed, never asserted on -- which is another way of saying it cannot be
gated.

SplitMix64 is a handful of integer operations over three published constants.
That is short enough to be written twice, in two languages, and the two copies
can then be checked against each other. If they ever disagree, the disagreement
is a real difference in arithmetic, not an unspecified behaviour.

What this is not
----------------

SplitMix64 is a counter-based mixer with good enough statistical behaviour for
replay, and nothing more. It is not cryptographically secure, and it is not
what one would pick for a simulation that has to survive a serious test suite.
Nothing was traded away to get here, because those guarantees were never on
offer: the requirement is the opposite one -- **the same seed must give the
same sequence forever**, including after a compiler upgrade.

Constants
---------

The three multipliers and the increment are part of the definition. Changing
any of them changes every downstream number, which is why they are `comptime`
and why there is no parameter to override them.
"""

comptime SPLITMIX64_GAMMA = UInt64(0x9E3779B97F4A7C15)
comptime SPLITMIX64_MIX1 = UInt64(0xBF58476D1CE4E5B9)
comptime SPLITMIX64_MIX2 = UInt64(0x94D049BB133111EB)

# 2^-53. The top 53 bits of a 64-bit word are the widest range that every
# double still maps one-to-one onto, so this is the usual way to turn a word
# into a number in [0, 1) without ever producing 1.0.
comptime U53_SCALE = Float64(1.1102230246251565e-16)


struct Rng(Copyable, Movable):
    """A reproducible bit source, seeded by one 64-bit word.

    The state advances by an odd constant before every draw, which is what
    keeps successive outputs from being related by a fixed linear map. Each
    draw is then two xor-shift-and-multiply rounds followed by one xor-shift;
    the multiplications are what spread a change in a low bit up into the high
    bits that actually get used.
    """

    var state: UInt64

    def __init__(out self, seed: UInt64):
        """Start from `seed`. There is no good seed to search for here."""
        self.state = seed

    def next_u64(mut self) -> UInt64:
        """Consume one step and return the mixed 64-bit word."""
        self.state = self.state + SPLITMIX64_GAMMA
        var z = self.state
        z = (z ^ (z >> 30)) * SPLITMIX64_MIX1
        z = (z ^ (z >> 27)) * SPLITMIX64_MIX2
        return z ^ (z >> 31)

    def next_uniform(mut self) -> Float64:
        """One draw in [0, 1), using the high 53 bits.

        The shift happens while the value is still an integer, so the
        conversion to floating point is exact and the only rounding is the one
        in the final multiply -- the same on any machine with IEEE doubles.
        """
        return Float64(self.next_u64() >> 11) * U53_SCALE
