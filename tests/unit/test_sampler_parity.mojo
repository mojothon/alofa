"""The sampler must be the same sampler as the reference, not a similar one.

Three separate things can go wrong in a sampler, and they need three separate
kinds of evidence:

1. **The transformation is wrong.** A filter applied in the wrong order, or a
   threshold compared with `>` instead of `>=`, still produces a distribution
   — just not the right one. Caught here by comparing the probability vector
   against the reference, and more sharply by comparing the *surviving set*,
   which is discrete and therefore has no tolerance to hide behind.

2. **The selection is wrong on ties.** Sorting is not defined on equal values,
   so a real logits row will not catch this; a synthetic row with many equal
   values will. That is what the `ties/` fixture is for, and why its assertion
   is on the set rather than on the values.

3. **The draw is wrong.** A sampler can produce the right distribution and
   still take from it incorrectly — which is exactly the shape of a decoder
   that picks tokens by exact match instead of by rejection sampling. Caught
   twice over: sampled ids must match **token for token** against a reference
   running the same reproducible bit source, and the empirical distribution of
   many draws must match the theoretical one by TVD and chi-square.

The distribution test is the one that keeps the others honest: a deliberately
wrong theoretical distribution must be *rejected*. A goodness-of-fit gate that
accepts any distribution is worse than no gate, because it produces a green
tick that means nothing.

Run:
    pixi run mojo run -O0 -I src tests/unit/test_sampler_parity.mojo
"""

from std.testing import TestSuite, assert_true

from alofa.core.error import ERR_IO, AlofaError
from alofa.core.memory import Arena
from alofa.core.mmap import MappedFile
from alofa.core.rng import Rng
from alofa.core.text import parse_float64, parse_int, read_text
from alofa.runtime.sampler import (
    F64Ptr,
    LogitBias,
    SampleParams,
    Sampler,
    pick_from,
)

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b/sampler/"
comptime LOGITS = "tests/fixtures/qwen2.5-0.5b/logits_last.f32"
comptime GREEDY = "tests/fixtures/qwen2.5-0.5b/greedy.tsv"

# Probabilities live in [0, 1] and are stored as fp32, whose step near 0.5 is
# about 6e-8. This is roughly sixteen steps of that storage precision: wide
# enough to absorb the last-bit difference between two `exp` implementations,
# and about five orders of magnitude too narrow to absorb a semantic mistake.
# The set assertions below have no tolerance at all.
comptime PROB_TOL = Float64(1e-6)

comptime FNV_OFFSET = UInt64(0xCBF29CE484222325)
comptime FNV_PRIME = UInt64(0x100000001B3)


def abs_f64(x: Float64) -> Float64:
    if x >= Float64(0):
        return x
    return -x


def lines_of(path: String) raises AlofaError -> List[String]:
    """Non-empty lines of a text file, in order."""
    var text = read_text(path)
    var out = List[String]()
    for span in text.split("\n"):
        var line = String(span)
        if line.byte_length() == 0:
            continue
        out.append(line)
    return out^


def fields_of(line: String) raises AlofaError -> List[String]:
    """Tab-separated fields of one line."""
    var out = List[String]()
    for span in line.split("\t"):
        out.append(String(span))
    return out^


def ids_from_csv(text: String) raises AlofaError -> List[Int]:
    """A comma-separated run of integers."""
    var ids = List[Int]()
    if text.byte_length() == 0:
        return ids^
    for span in text.split(","):
        var part = String(span)
        if part.byte_length() == 0:
            continue
        ids.append(parse_int(part))
    return ids^


struct CaseRow(Copyable, Movable):
    """One row of `cases.tsv`."""

    var name: String
    var prompt: Int
    var params: SampleParams
    var bias_name: String
    var hist_name: String

    def __init__(out self):
        self.name = ""
        self.prompt = 0
        self.params = SampleParams()
        self.bias_name = "-"
        self.hist_name = "-"


def load_cases() raises AlofaError -> List[CaseRow]:
    """Every case, with its parameters parsed into a `SampleParams`."""
    var out = List[CaseRow]()
    for line in lines_of(FIXTURE + "cases.tsv"):
        var f = fields_of(line)
        var row = CaseRow()
        row.name = f[0]
        row.prompt = parse_int(f[1])
        row.params.temperature = parse_float64(f[2])
        row.params.top_k = parse_int(f[3])
        row.params.top_p = parse_float64(f[4])
        row.params.min_p = parse_float64(f[5])
        row.params.repetition_penalty = parse_float64(f[6])
        row.params.frequency_penalty = parse_float64(f[7])
        row.params.presence_penalty = parse_float64(f[8])
        row.bias_name = f[9]
        row.hist_name = f[10]
        out.append(row.copy())
    return out^


def load_bias(name: String) raises AlofaError -> List[LogitBias]:
    """The offsets belonging to one bias set."""
    var out = List[LogitBias]()
    if name == "-":
        return out^
    for line in lines_of(FIXTURE + "bias.tsv"):
        var f = fields_of(line)
        if f[0] != name:
            continue
        out.append(LogitBias(parse_int(f[1]), parse_float64(f[2])))
    return out^


def load_history(name: String) raises AlofaError -> List[Int]:
    """The token ids of one history; empty when the case has none."""
    var empty = List[Int]()
    if name == "-":
        return empty^
    for line in lines_of(FIXTURE + "history.tsv"):
        var f = fields_of(line)
        if f[0] != name:
            continue
        return ids_from_csv(f[1])
    raise AlofaError(ERR_IO, "history set is not in the fixture", "name=" + name)


def sets_field(name: String, column: Int) raises AlofaError -> String:
    """One column of the row named `name` in `sets.tsv`."""
    for line in lines_of(FIXTURE + "sets.tsv"):
        var f = fields_of(line)
        if f[0] == name:
            return f[column]
    raise AlofaError(ERR_IO, "case is missing from sets.tsv", "name=" + name)


def expected_sampled(name: String) raises AlofaError -> List[Int]:
    """Reference ids for one case."""
    var empty = List[Int]()
    for line in lines_of(FIXTURE + "sampled.tsv"):
        var f = fields_of(line)
        if f[0] != name:
            continue
        return ids_from_csv(f[1])
    return empty^


def fnv1a_of_alive(sampler: Sampler, n: Int) raises -> UInt64:
    """Fingerprint of the surviving indices, in ascending order.

    Comparing counts alone would pass on "right number of survivors, wrong
    ones". Comparing a fingerprint catches that without storing a 151936
    element index list in the fixture.
    """
    var h = FNV_OFFSET
    for i in range(n):
        if not sampler.is_alive(i):
            continue
        var v = UInt64(i)
        for _ in range(8):
            h = (h ^ (v & UInt64(0xFF))) * FNV_PRIME
            v = v >> 8
    return h


def vocab_size() raises AlofaError -> Int:
    for line in lines_of(FIXTURE + "logits.tsv"):
        var f = fields_of(line)
        return parse_int(f[1])
    raise AlofaError(ERR_IO, "no logits rows in the fixture", "")


def reference_seed() raises AlofaError -> UInt64:
    var first = fields_of(lines_of(FIXTURE + "draws.tsv")[0])
    return UInt64(parse_int(first[1]))


def test_rng_matches_reference_draws() raises:
    """The bit source must produce the reference's exact words.

    If this fails while the sampled ids also fail, the bug is here; if only
    the ids fail, the bit source is fine and the inverse CDF is not. That is
    the whole reason the raw draws are exported separately.
    """
    var lines = lines_of(FIXTURE + "draws.tsv")
    var rng = Rng(reference_seed())
    var checked = 0
    for i in range(1, len(lines)):
        var f = fields_of(lines[i])
        var want = UInt64(parse_int(f[1]))
        var got = rng.next_u64()
        assert_true(got == want, "draw " + String(i - 1) + " does not match")
        checked += 1
    assert_true(checked == 64, "expected 64 exported draws, got " + String(checked))


def test_uniform_draws_stay_in_the_unit_interval() raises:
    """`next_uniform` is in [0, 1); never 1, which would break the inverse CDF."""
    var rng = Rng(UInt64(0x1234567890ABCDEF))
    var smallest = Float64(1)
    var largest = Float64(0)
    for _ in range(4096):
        var u = rng.next_uniform()
        assert_true(u >= Float64(0) and u < Float64(1), "draw outside [0, 1)")
        if u < smallest:
            smallest = u
        if u > largest:
            largest = u
    assert_true(largest > Float64(0.5), "draws are bunched near zero")
    assert_true(smallest < Float64(0.5), "draws are bunched near one")


def test_probability_vectors_match_reference() raises:
    """Every case's distribution, element by element, against the reference."""
    var n = vocab_size()
    var logits = MappedFile(LOGITS)
    var probs = MappedFile(FIXTURE + "probs.f32")
    var sampler = Sampler(n, 4096)
    var cases = load_cases()
    var checked = 0

    for row in cases:
        var offset = -1
        for line in lines_of(FIXTURE + "probs.tsv"):
            var f = fields_of(line)
            if f[0] == row.name:
                offset = parse_int(f[2])
        assert_true(offset >= 0, "no reference probabilities for " + row.name)

        var src = logits.ptr().unsafe_offset(row.prompt * n * 4).unsafe_bitcast[
            Float32
        ]()
        var want = probs.ptr().unsafe_offset(offset * 4).unsafe_bitcast[Float32]()
        sampler.build(
            src, n, row.params, load_bias(row.bias_name), load_history(row.hist_name)
        )
        var got = sampler.probs()
        var worst = Float64(0)
        for i in range(n):
            var d = Float64(want[unsafe_offset=i]) - got[unsafe_offset=i]
            if d < Float64(0):
                d = -d
            if d > worst:
                worst = d
        assert_true(
            worst <= PROB_TOL,
            row.name + " probabilities differ by " + String(worst),
        )
        checked += 1
        sampler.arena.keep_alive()
    assert_true(checked == 9, "expected 9 cases, got " + String(checked))
    logits.keep_alive()
    probs.keep_alive()


def test_surviving_sets_match_reference() raises:
    """Which entries survive each filter: exact count and exact index set.

    No tolerance. A filter that keeps the right number of the wrong tokens is
    a real and easy bug, and only the second assertion catches it.
    """
    var n = vocab_size()
    var logits = MappedFile(LOGITS)
    var sampler = Sampler(n, 4096)
    var cases = load_cases()

    for row in cases:
        var src = logits.ptr().unsafe_offset(row.prompt * n * 4).unsafe_bitcast[
            Float32
        ]()
        sampler.build(
            src, n, row.params, load_bias(row.bias_name), load_history(row.hist_name)
        )
        var want_count = parse_int(sets_field(row.name, 1))
        var got_count = sampler.alive_count(n)
        assert_true(
            got_count == want_count,
            row.name
            + " should keep "
            + String(want_count)
            + " entries, got "
            + String(got_count),
        )
        var want_digest = UInt64(parse_int(sets_field(row.name, 3)))
        var got_digest = fnv1a_of_alive(sampler, n)
        assert_true(
            got_digest == want_digest,
            row.name + " keeps the wrong entries (index fingerprint differs)",
        )
        sampler.arena.keep_alive()
    logits.keep_alive()


def test_sampled_ids_match_token_for_token() raises:
    """Thirty-two draws per case, identical ids in identical order.

    This is the assertion that would catch a decoder taking tokens by exact
    match instead of by rejection sampling: the distribution would still look
    plausible, but the ids would not line up.
    """
    var n = vocab_size()
    var seed = reference_seed()
    var logits = MappedFile(LOGITS)
    var sampler = Sampler(n, 4096)
    var cases = load_cases()

    for row in cases:
        var src = logits.ptr().unsafe_offset(row.prompt * n * 4).unsafe_bitcast[
            Float32
        ]()
        sampler.build(
            src, n, row.params, load_bias(row.bias_name), load_history(row.hist_name)
        )
        var want = expected_sampled(row.name)
        var rng = Rng(seed)
        for i in range(len(want)):
            var got = sampler.pick(n, rng.next_uniform())
            assert_true(
                got == want[i],
                row.name
                + " draw "
                + String(i)
                + " should be "
                + String(want[i])
                + ", got "
                + String(got),
            )
        sampler.arena.keep_alive()
    logits.keep_alive()


def test_temperature_zero_equals_the_reference_greedy_token() raises:
    """Degenerate temperature is argmax, and that argmax is the reference's.

    Cross-checked against `greedy.tsv`, which a different script produced from
    a different run — so this is a genuine cross-fixture assertion, not a
    restatement of something this fixture computed itself.
    """
    var n = vocab_size()
    var logits = MappedFile(LOGITS)
    var sampler = Sampler(n, 4096)
    var cases = load_cases()
    var row = cases[0].copy()
    assert_true(row.params.temperature == Float64(0), "first case must use temperature 0")

    var src = logits.ptr().unsafe_offset(row.prompt * n * 4).unsafe_bitcast[Float32]()
    var no_bias = List[LogitBias]()
    var no_history = List[Int]()
    sampler.build(src, n, row.params, no_bias, no_history)
    assert_true(sampler.alive_count(n) == 1, "temperature 0 must leave one entry")

    var greedy_ids = ids_from_csv(lines_of(GREEDY)[0])
    var got = sampler.pick(n, Float64(0))
    assert_true(
        got == greedy_ids[0],
        "temperature 0 should give the reference's first greedy token "
        + String(greedy_ids[0])
        + ", got "
        + String(got),
    )
    sampler.arena.keep_alive()
    logits.keep_alive()


def test_ties_keep_every_entry_at_the_threshold() raises:
    """With equal values, `top_k` must keep *all* of them, not an arbitrary k.

    Real logits essentially never tie, so a `>` written where `>=` belongs
    would survive every model-based test and then misbehave on a quantized
    model, where ties are common. The assertion is on the set because the
    *order* within a tie group is genuinely undefined.
    """
    var cases = lines_of(FIXTURE + "ties/cases.tsv")
    var expect_lines = lines_of(FIXTURE + "ties/sets.tsv")
    var raw = MappedFile(FIXTURE + "ties/logits.f32")
    var src = raw.ptr().unsafe_bitcast[Float32]()
    var n = 64
    var sampler = Sampler(n, 64)
    var no_bias = List[LogitBias]()
    var no_history = List[Int]()
    var checked = 0

    for line in cases:
        var f = fields_of(line)
        var name = f[0]
        var params = SampleParams()
        params.top_k = parse_int(f[1])

        var want = List[Int]()
        for want_line in expect_lines:
            var wf = fields_of(want_line)
            if wf[0] != name:
                continue
            want = ids_from_csv(wf[2])

        sampler.build(src, n, params, no_bias, no_history)
        var got_count = sampler.alive_count(n)
        assert_true(
            got_count == len(want),
            name
            + " should keep "
            + String(len(want))
            + " entries, got "
            + String(got_count),
        )
        for i in range(len(want)):
            assert_true(
                sampler.is_alive(want[i]),
                name + " is missing index " + String(want[i]),
            )
        checked += 1
        sampler.arena.keep_alive()
    assert_true(checked == 4, "expected 4 tie cases, got " + String(checked))
    raw.keep_alive()


def distribution_stats(
    weights: F64Ptr, m: Int, counts: List[Int], total: Int
) -> List[Float64]:
    """TVD and chi-square of the observed counts against `weights`."""
    var out = List[Float64]()
    var tvd = Float64(0)
    var chi2 = Float64(0)
    var n = Float64(total)
    for i in range(m):
        var observed = Float64(counts[i])
        var expected = weights[unsafe_offset=i] * n
        var d = observed / n - weights[unsafe_offset=i]
        if d < Float64(0):
            d = -d
        tvd += d
        var diff = observed - expected
        chi2 += diff * diff / expected
    out.append(tvd / Float64(2))
    out.append(chi2)
    return out^


def test_empirical_distribution_matches_theoretical() raises:
    """Fifty thousand draws: the empirical distribution must be the theoretical one.

    The sequence is fixed, so this is a deterministic assertion, not a
    statistical one that can flake. Matching the reference's TVD and
    chi-square proves the whole chain — bit source, inverse CDF and the
    distribution itself — agrees end to end.
    """
    var spec = fields_of(lines_of(FIXTURE + "dist.tsv")[0])
    var name = spec[0]
    var samples = parse_int(spec[1])
    var tvd_upper = parse_float64(spec[2])
    var chi2_critical = parse_float64(spec[3])
    var decoy_lower = parse_float64(spec[4])

    var ref_row = fields_of(lines_of(FIXTURE + "dist_ref.tsv")[0])
    var want_tvd = parse_float64(ref_row[1])
    var want_chi2 = parse_float64(ref_row[2])
    var want_tvd_decoy = parse_float64(ref_row[3])

    var support_line = fields_of(lines_of(FIXTURE + "dist_support.tsv")[0])
    assert_true(name == support_line[0], "distribution case and support disagree")
    var support = ids_from_csv(support_line[1])
    var m = len(support)

    # Weights are copied out of the fixture into one arena: `pick_from` walks a
    # pointer, and the point of the gate is that both sides read the *same*
    # numbers the fixture stores, not separately rounded copies of them.
    var arena = Arena(m * 8 * 2 + 64)
    var weights = arena.alloc(m * 8).unsafe_bitcast[Float64]()
    var decoy = arena.alloc(m * 8).unsafe_bitcast[Float64]()
    var stored = MappedFile(FIXTURE + "dist_probs.f32")
    var base = stored.ptr().unsafe_bitcast[Float32]()
    for i in range(m):
        weights[unsafe_offset=i] = Float64(base[unsafe_offset=i])
    for i in range(m):
        decoy[unsafe_offset=i] = Float64(base[unsafe_offset=m + i])

    var rng = Rng(reference_seed())
    var counts = List[Int]()
    for _ in range(m):
        counts.append(0)
    for _ in range(samples):
        var hit = pick_from(weights, m, rng.next_uniform())
        counts[hit] = counts[hit] + 1

    var stats = distribution_stats(weights, m, counts, samples)
    var got_tvd = stats[0]
    var got_chi2 = stats[1]
    assert_true(
        abs_f64(got_tvd - want_tvd) <= Float64(1e-9),
        "TVD should be " + String(want_tvd) + ", got " + String(got_tvd),
    )
    assert_true(
        abs_f64(got_chi2 - want_chi2) <= Float64(1e-6),
        "chi-square should be " + String(want_chi2) + ", got " + String(got_chi2),
    )
    assert_true(got_tvd <= tvd_upper, "TVD exceeds the gate: " + String(got_tvd))
    assert_true(
        got_chi2 <= chi2_critical, "chi-square exceeds the gate: " + String(got_chi2)
    )

    # Negative control: the same counts tested against a *wrong* theory. If
    # this ever passes, the gate above is not measuring anything.
    var decoy_stats = distribution_stats(decoy, m, counts, samples)
    assert_true(
        abs_f64(decoy_stats[0] - want_tvd_decoy) <= Float64(1e-9),
        "decoy TVD should be " + String(want_tvd_decoy) + ", got " + String(decoy_stats[0]),
    )
    assert_true(
        decoy_stats[0] >= decoy_lower,
        "a wrong distribution was accepted (TVD only " + String(decoy_stats[0]) + ")",
    )
    assert_true(
        decoy_stats[0] > got_tvd * Float64(4),
        "the decoy is not clearly distinguishable from the truth",
    )
    assert_true(
        decoy_stats[1] > chi2_critical,
        "chi-square did not reject the decoy: " + String(decoy_stats[1]),
    )
    stored.keep_alive()
    arena.keep_alive()


def test_probabilities_sum_to_one() raises:
    """A distribution that does not sum to one is not a distribution."""
    var n = vocab_size()
    var logits = MappedFile(LOGITS)
    var sampler = Sampler(n, 4096)
    var cases = load_cases()
    for row in cases:
        var src = logits.ptr().unsafe_offset(row.prompt * n * 4).unsafe_bitcast[
            Float32
        ]()
        sampler.build(
            src, n, row.params, load_bias(row.bias_name), load_history(row.hist_name)
        )
        var total = Float64(0)
        var p = sampler.probs()
        for i in range(n):
            total += p[unsafe_offset=i]
        assert_true(
            abs_f64(total - Float64(1)) <= Float64(1e-5),
            row.name + " probabilities sum to " + String(total),
        )
        sampler.arena.keep_alive()
    logits.keep_alive()


def test_an_out_of_range_bias_index_is_rejected() raises:
    """A bias pointing outside the row is a caller bug, so it must be named."""
    var n = 64
    var arena = Arena(n * 4 + 64)
    var src = arena.alloc(n * 4).unsafe_bitcast[Float32]()
    for i in range(n):
        src[unsafe_offset=i] = Float32(i)
    var sampler = Sampler(n, 8)
    var no_history = List[Int]()
    var bad = List[LogitBias]()
    bad.append(LogitBias(n + 1, Float64(1)))
    var rejected = False
    try:
        sampler.build(src, n, SampleParams(), bad, no_history)
    except err:
        rejected = err.name() == "out_of_range"
    assert_true(rejected, "an out-of-range bias index must raise out_of_range")
    arena.keep_alive()


def test_a_row_longer_than_the_capacity_is_rejected() raises:
    """Silently sampling from a truncated row would be far worse than an error."""
    var arena = Arena(64 * 4 + 64)
    var src = arena.alloc(64 * 4).unsafe_bitcast[Float32]()
    var sampler = Sampler(16, 8)
    var no_bias = List[LogitBias]()
    var no_history = List[Int]()
    var rejected = False
    try:
        sampler.build(src, 32, SampleParams(), no_bias, no_history)
    except err:
        rejected = err.name() == "capacity"
    assert_true(rejected, "an oversized row must raise capacity")
    arena.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
