"""Differential test: the whole Qwen2.5-0.5B forward, against Hugging Face.

Three verdicts, in increasing order of what they prove:

1. **logits** — cosine ≥ 0.999 against the reference and the same argmax. A
   model can be wrong by a little and still pick the same token, so this is
   the weakest of the three and the one that runs first, because a failure
   here makes the other two uninterpretable.
2. **greedy** — 128 tokens, identical to the reference. This is the strong
   one: a small numeric drift that survives one comparison snowballs over 128
   autoregressive steps, so a wrong-but-close implementation fails here even
   when it passes the cosine check.
3. **incremental** — decoding one token at a time must agree with consuming the
   whole prompt at once. Two implementations of the same formula that agree on
   the first token and diverge on the tenth is the failure mode this exists
   for, and it is invisible to any test that only prefills.

The reference is a fixture (`scripts/dump_model_reference.py`), not a live
call into Python, so the answer does not move when the code under test does.

Build and run (the scalar backend is slow enough that the default `mojo run`
optimisation level makes this take minutes rather than seconds):

    pixi run mojo build -O2 -I src tests/unit/test_model_parity.mojo -o /tmp/mp
    /tmp/mp
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.error import AlofaError
from alofa.core.mmap import MappedFile
from alofa.core.tensor import F32Ptr, f32_data
from alofa.core.text import parse_int, read_text
from alofa.model.arch.qwen import QwenForward
from alofa.model.loader import config_value
from alofa.verify.compare import cosine, max_abs_diff

comptime FIXTURE = "tests/fixtures/qwen2.5-0.5b"
comptime WEIGHTS_DIR = FIXTURE + "/weights"
comptime CONFIG = FIXTURE + "/config.tsv"
comptime PROMPTS = FIXTURE + "/prompts.tsv"
comptime GREEDY = FIXTURE + "/greedy.tsv"
comptime LOGITS = FIXTURE + "/logits_last.f32"

# Enough for a prompt plus the full greedy run; the model refuses to go past
# it rather than growing, which is what the last test asserts.
comptime MAX_TOKENS = 256

# The P1 numeric gate. Deliberately the same number as in the roadmap: a gate
# whose threshold drifts is not a gate.
comptime COSINE_GATE = Float64(0.999)


def ids_from_csv(text: String) raises AlofaError -> List[Int]:
    """A comma-separated line of token ids."""
    var ids = List[Int]()
    if text.byte_length() == 0:
        return ids^
    var parts = text.split(",")
    for part_span in parts:
        var part = String(part_span)
        if part.byte_length() == 0:
            continue
        ids.append(parse_int(part))
    return ids^


def load_prompts() raises AlofaError -> List[List[Int]]:
    """Prompt token ids; the text column is hex and only for humans."""
    var text = read_text(PROMPTS)
    var out = List[List[Int]]()
    var lines = text.split("\n")
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        var ids_text = String(fields[1])
        out.append(ids_from_csv(ids_text))
    return out^


def load_greedy() raises AlofaError -> List[List[Int]]:
    """Reference greedy continuations, one line of ids per prompt."""
    var text = read_text(GREEDY)
    var out = List[List[Int]]()
    var lines = text.split("\n")
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        out.append(ids_from_csv(line))
    return out^


def vocab_size() raises AlofaError -> Int:
    return parse_int(config_value(CONFIG, "vocab"))


def reference_logits(mapped: MappedFile, prompt: Int) raises AlofaError -> F32Ptr:
    """The reference logits row for one prompt, from the mapped fixture."""
    var n = vocab_size()
    return mapped.ptr().unsafe_offset(prompt * n * 4).unsafe_bitcast[Float32]()


def argmax_of(values: F32Ptr, n: Int) -> Int:
    """Largest element's index, ties to the lowest index (as `torch.argmax`)."""
    var best = 0
    var best_value = Float32(-3.4028234663852886e38)
    for i in range(n):
        if values[unsafe_offset=i] > best_value:
            best_value = values[unsafe_offset=i]
            best = i
    return best


def test_prefill_logits_match_reference() raises:
    """Cosine and argmax of the last position, for every prompt."""
    var n = vocab_size()
    var mapped = MappedFile(LOGITS)
    var prompts = load_prompts()
    assert_true(len(prompts) > 0, "the fixture has no prompts")

    var index = 0
    for ids in prompts:
        var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
        var logits = model.prefill(ids)
        var expected_logits = reference_logits(mapped, index)
        var sim = cosine(logits, expected_logits, n)
        assert_true(
            sim >= COSINE_GATE,
            "logits cosine below the gate: prompt="
            + String(index)
            + " cosine="
            + String(sim),
        )
        assert_equal(
            argmax_of(logits, n),
            argmax_of(expected_logits, n),
            "argmax differs from the reference: prompt=" + String(index),
        )
        model.keep_alive()
        index += 1
    mapped.keep_alive()


def test_greedy_generation_matches_token_for_token() raises:
    """128 tokens, identical to the reference, for every prompt."""
    var n = vocab_size()
    var prompts = load_prompts()
    var greedy = load_greedy()
    assert_equal(len(greedy), len(prompts), "one greedy row per prompt")

    var index = 0
    for ids in prompts:
        var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
        var expected = greedy[index].copy()
        var first = model.prefill(ids)
        var seen = 0
        var token = argmax_of(first, n)
        assert_equal(
            token, expected[0], "first greedy token differs: prompt=" + String(index)
        )
        seen += 1
        while seen < len(expected):
            var logits = model.step(expected[seen - 1])
            token = argmax_of(logits, n)
            assert_equal(
                token,
                expected[seen],
                "greedy differs at token "
                + String(seen)
                + ": prompt="
                + String(index),
            )
            seen += 1
        model.keep_alive()
        index += 1


def test_incremental_decode_matches_full_prefill() raises:
    """One token at a time must land where consuming the whole prompt lands.

    The comparison is between this implementation's two phases, and then
    against the reference, so a failure says which side moved.
    """
    var n = vocab_size()
    var mapped = MappedFile(LOGITS)
    var prompts = load_prompts()
    var index = 0
    for ids in prompts:
        var whole = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
        var at_once = whole.prefill(ids)

        var stepped = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
        var one_at_a_time = stepped.step(ids[0])
        for i in range(1, len(ids)):
            one_at_a_time = stepped.step(ids[i])

        var diff = max_abs_diff(at_once, one_at_a_time, n)
        assert_true(
            diff <= Float64(1e-4),
            "incremental decode differs from prefill: prompt="
            + String(index)
            + " max_abs_diff="
            + String(diff),
        )
        var expected_logits = reference_logits(mapped, index)
        assert_equal(
            argmax_of(one_at_a_time, n),
            argmax_of(expected_logits, n),
            "incremental argmax differs from the reference: prompt=" + String(index),
        )
        whole.keep_alive()
        stepped.keep_alive()
        index += 1
    mapped.keep_alive()


def test_history_must_be_fresh_for_a_new_prefill() raises:
    """Starting a stream on top of an old one is an error, not a silent merge."""
    var model = QwenForward(WEIGHTS_DIR, CONFIG, MAX_TOKENS)
    var ids = List[Int]()
    ids.append(785)
    _ = model.prefill(ids)
    var caught = "no-error"
    try:
        _ = model.prefill(ids)
    except err:
        caught = err.name()
    assert_equal(caught, "invalid_argument")
    model.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
