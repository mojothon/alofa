"""End-to-end: one directory, laid out the way Hugging Face publishes it.

The other gates read our own exports — a TSV index next to an fp32 payload,
with the rotary tables already materialised and `lm_head.weight` written out
even though the embeddings are tied. Those exports exist because the fixtures
have to be diffable, but a directory like that is not what a user has: a user
has `config.json`, `model.safetensors` and `tokenizer.json`, and three things
about that directory are different from ours.

1. **The payload is bfloat16.** Every tensor, including the 136M-element
   embedding. Widening it is exact (bf16 is the top half of an fp32) but it
   cannot happen inside a read-only mapping, so the file owns the widened copy.
2. **There is no `lm_head.weight`.** `tie_word_embeddings` is true, so the
   output projection *is* the embedding matrix. A loader that required the name
   would refuse every Qwen2 checkpoint.
3. **There are no rotary tables.** `rope_cos` / `rope_sin` are a function of
   `rope_theta`, which the reference builds at run time; only our exporter
   writes them down.

So this suite loads the real directory, and answers the only question that
matters about it: does the answer come out the same? The reference is the same
fixture the other gates use — logits from Hugging Face running this checkpoint
in fp32, and its greedy continuation — so "the same" means cosine ≥ 0.999 with
the same argmax, and then token-for-token agreement for 16 generated tokens.

It is not part of `pixi run test`: it widens 494M parameters and runs a real
forward. Build and run it the way `test-model` is run:

    pixi run test-model-dir

The directory itself is assembled by `scripts/build_real_model_dir.py`, which
links the cached checkpoint rather than copying a gigabyte into the repository.
"""

from std.testing import TestSuite, assert_equal, assert_true

from alofa.core.mmap import MappedFile
from alofa.core.tensor import F32Ptr, f32_data
from alofa.core.text import parse_float64, parse_int, read_text
from alofa.model.arch.qwen import QwenConfig, QwenForward
from alofa.model.config import json_int, json_value
from alofa.model.loader import TensorFile, config_value
from alofa.model.safetensors import SafeTensorFile
from alofa.tokenizer.tokenizer_json import load_tokenizer_json
from alofa.verify.compare import cosine, max_abs_diff

# The directory under test: a real checkpoint's files, nothing of ours.
comptime REAL_DIR = "tests/fixtures/qwen2.5-0.5b-hf"
comptime REAL_CONFIG = REAL_DIR + "/config.json"
comptime REAL_WEIGHTS = REAL_DIR + "/model.safetensors"
comptime REAL_TOKENIZER = REAL_DIR + "/tokenizer.json"

# The reference: Hugging Face's answer for the same checkpoint, exported by
# `scripts/dump_model_reference.py` / `dump_reference.py`.
comptime REF = "tests/fixtures/qwen2.5-0.5b"
comptime REF_CONFIG = REF + "/config.tsv"
comptime REF_WEIGHTS = REF + "/weights"
comptime PROMPTS = REF + "/prompts.tsv"
comptime GREEDY = REF + "/greedy.tsv"
comptime LOGITS = REF + "/logits_last.f32"

comptime MAX_TOKENS = 256

# Deliberately the same gate as the numeric model gate: a threshold that moves
# to fit the code under test is not a threshold.
comptime COSINE_GATE = Float64(0.999)

# How many generated tokens are compared. Sixteen is enough for a numeric
# difference to compound into a different token, and short enough that this
# suite stays a gate someone will actually run.
comptime GREEDY_TOKENS = 16

# The rotary tables this suite derives are compared against the ones the
# reference handed out. Measured: 5.96e-08, i.e. one fp32 ulp, over 256x64 —
# the reference computes the angle in fp32 and this code computes it in fp64,
# so one ulp is the smallest gap the two could disagree by. The gate sits at
# 1e-6: above the ulp, far below the O(1) difference a wrong exponent, a
# missing duplication of the halves, or a row indexed by anything other than
# absolute position would produce.
comptime ROPE_TOLERANCE = Float64(1e-6)


def hex_value(byte: UInt8) -> Int:
    var value = Int(byte)
    if value >= 48 and value <= 57:
        return value - 48
    if value >= 97 and value <= 102:
        return value - 87
    if value >= 65 and value <= 70:
        return value - 55
    return -1


def decode_hex(imm text: String, mut sink: List[UInt8]) raises:
    """Decode an even-length hex string. The fixture hex-encodes prompt text so
    newlines and tabs survive a line-oriented file."""
    var raw = text.as_bytes()
    var index = 0
    while index + 1 < len(raw):
        var high = hex_value(raw[index])
        var low = hex_value(raw[index + 1])
        if high < 0 or low < 0:
            raise Error("not a hex digit at " + String(index))
        sink.append(UInt8(high * 16 + low))
        index += 2
    if index != len(raw):
        raise Error("hex string has an odd length")


def parse_id_list(imm text: String, mut sink: List[Int]) raises:
    """Parse `785,6722,315`; the fixture writes one line per prompt."""
    var raw = text.as_bytes()
    var index = 0
    while index <= len(raw):
        var stop = index
        while stop < len(raw) and raw[stop] != 44:
            stop += 1
        if stop > index:
            var value = 0
            var digit = index
            while digit < stop:
                var number = Int(raw[digit]) - 48
                if number < 0 or number > 9:
                    raise Error("non-digit in the expected ids")
                value = value * 10 + number
                digit += 1
            sink.append(value)
        if stop >= len(raw):
            return
        index = stop + 1


def load_prompt_texts() raises -> List[String]:
    """The prompt texts, hex-encoded in the fixture so newlines survive."""
    var out = List[String]()
    var lines = read_text(PROMPTS).split("\n")
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        var bytes = List[UInt8]()
        decode_hex(String(line.split("\t")[0]), bytes)
        out.append(String(unsafe_from_utf8=bytes))
    return out^


def load_prompt_ids() raises -> List[List[Int]]:
    """The ids Hugging Face produced for those texts."""
    var out = List[List[Int]]()
    var lines = read_text(PROMPTS).split("\n")
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        var ids = List[Int]()
        parse_id_list(String(line.split("\t")[1]), ids)
        out.append(ids^)
    return out^


def load_greedy_row(row: Int) raises -> List[Int]:
    """One reference greedy continuation, as ids."""
    var lines = read_text(GREEDY).split("\n")
    var ids = List[Int]()
    parse_id_list(String(lines[row]), ids)
    return ids^


def vocab_size() raises -> Int:
    return parse_int(config_value(REF_CONFIG, "vocab"))


def argmax_of(values: F32Ptr, n: Int) -> Int:
    """Largest element's index, ties to the lowest index (as `torch.argmax`)."""
    var best = 0
    var best_value = Float32(-3.4028234663852886e38)
    for i in range(n):
        if values[unsafe_offset=i] > best_value:
            best_value = values[unsafe_offset=i]
            best = i
    return best


def test_directory_is_the_checkpoint_hugging_face_publishes() raises:
    """The three files, in the shapes that make this suite worth running.

    If the fixture were quietly re-exported as fp32, or `lm_head.weight` were
    added to it, every test below would still pass and none of them would be
    testing what it claims to. This one is the check on the fixture.
    """
    assert_equal(json_value(REAL_CONFIG, "torch_dtype"), "bfloat16")
    assert_equal(json_int(REAL_CONFIG, "vocab_size"), 151936)
    var file = SafeTensorFile(REAL_WEIGHTS)
    assert_true(
        file.has("model.embed_tokens.weight"), "the directory has no embeddings"
    )
    assert_true(
        file.has("model.layers.0.self_attn.q_proj.bias"),
        "Qwen2 biases are missing from the directory",
    )
    assert_true(
        not file.has("lm_head.weight"),
        "the directory carries lm_head.weight, so tied embeddings are untested",
    )
    file.keep_alive()


def test_config_matches_the_reference_export() raises:
    """The real `config.json` and our exported `config.tsv` name one model."""
    var cfg = QwenConfig(REAL_CONFIG)
    assert_equal(cfg.n_layers, parse_int(config_value(REF_CONFIG, "n_layers")))
    assert_equal(cfg.hidden, parse_int(config_value(REF_CONFIG, "hidden")))
    assert_equal(cfg.n_heads, parse_int(config_value(REF_CONFIG, "n_heads")))
    assert_equal(cfg.n_kv_heads, parse_int(config_value(REF_CONFIG, "n_kv_heads")))
    assert_equal(cfg.head_dim, parse_int(config_value(REF_CONFIG, "head_dim")))
    assert_equal(cfg.intermediate, parse_int(config_value(REF_CONFIG, "intermediate")))
    assert_equal(cfg.vocab, parse_int(config_value(REF_CONFIG, "vocab")))
    assert_true(
        cfg.rope_theta == parse_float64(config_value(REF_CONFIG, "rope_theta")),
        "rope_theta disagrees with the export",
    )
    assert_true(cfg.tied_output, "the checkpoint ties its embeddings")


def test_tokenizer_encodes_the_reference_prompts() raises:
    """The ids the reference consumed, produced from the same text here.

    This is the seam between the two halves of the engine: the model fixture
    was built by feeding Hugging Face's ids in, so if this tokenizer produced
    different ids the rest of the suite would be comparing against a reference
    for a different prompt.
    """
    var tokenizer = load_tokenizer_json(REAL_TOKENIZER)
    var texts = load_prompt_texts()
    var expected = load_prompt_ids()
    assert_equal(len(texts), len(expected), "one id row per prompt text")

    for index in range(len(texts)):
        var got = tokenizer.encode(texts[index])
        assert_equal(
            len(got),
            len(expected[index]),
            "token count differs: prompt=" + String(index),
        )
        for j in range(len(got)):
            assert_equal(
                got[j],
                expected[index][j],
                "token differs: prompt="
                + String(index)
                + " position="
                + String(j),
            )


def test_prefill_logits_match_the_reference() raises:
    """Last-position logits for every prompt, from the real directory."""
    var vocab = vocab_size()
    var mapped = MappedFile(LOGITS)
    var prompts = load_prompt_ids()
    var model = QwenForward(REAL_WEIGHTS, REAL_CONFIG, MAX_TOKENS)

    for index in range(len(prompts)):
        model.reset()
        var logits = model.prefill(prompts[index].copy())
        var expected = mapped.ptr().unsafe_offset(
            index * vocab * 4
        ).unsafe_bitcast[Float32]()
        var sim = cosine(logits, expected, vocab)
        assert_true(
            sim >= COSINE_GATE,
            "logits cosine below the gate: prompt="
            + String(index)
            + " cosine="
            + String(sim),
        )
        assert_equal(
            argmax_of(logits, vocab),
            argmax_of(expected, vocab),
            "argmax differs from the reference: prompt=" + String(index),
        )

    model.keep_alive()
    mapped.keep_alive()


def test_greedy_generation_matches_the_reference() raises:
    """Sixteen generated tokens, identical to Hugging Face's continuation.

    The ids are fed from the reference rather than from our own argmax: this
    asks "given the same history, does the next token agree", so a mismatch
    names the position where the two implementations parted.
    """
    var vocab = vocab_size()
    var prompts = load_prompt_ids()
    var prompt = prompts[0].copy()
    var expected = load_greedy_row(0)
    assert_true(len(expected) >= GREEDY_TOKENS, "the reference row is too short")

    var model = QwenForward(REAL_WEIGHTS, REAL_CONFIG, MAX_TOKENS)
    var logits = model.prefill(prompt)
    var token = argmax_of(logits, vocab)
    assert_equal(token, expected[0], "first greedy token differs")
    var seen = 1
    while seen < GREEDY_TOKENS:
        logits = model.step(expected[seen - 1])
        token = argmax_of(logits, vocab)
        assert_equal(
            token,
            expected[seen],
            "greedy differs at token " + String(seen),
        )
        seen += 1
    model.keep_alive()


def test_derived_rotary_table_matches_the_reference_one() raises:
    """The tables this run derived, against the ones Hugging Face handed out.

    The real directory has no `rope_cos` / `rope_sin`, so `QwenForward` builds
    them from `rope_theta`. Our export happens to carry the reference's, which
    makes the derivation checkable: a wrong exponent, a missing duplication of
    the halves, or a table indexed by batch position instead of absolute
    position all show up here as a difference of order one.

    What the tolerance is for is stated next to `ROPE_TOLERANCE`: the observed
    gap is one fp32 ulp, and the gate is set one and a half orders of magnitude
    above it — not because a larger gap would be acceptable, but because an
    exact-match gate would be a gate on libm rather than on the formula.
    """
    var dumped = TensorFile(REF_WEIGHTS)
    var model = QwenForward(REAL_WEIGHTS, REAL_CONFIG, MAX_TOKENS)

    var cols = model.cos.shape.dims[1]
    var rows = model.cos.shape.dims[0]
    var dumped_rows = dumped.view("rope_cos").shape.dims[0]
    if dumped_rows < rows:
        rows = dumped_rows

    var expected_cos = f32_data(dumped.view("rope_cos"))
    var expected_sin = f32_data(dumped.view("rope_sin"))
    var got_cos = f32_data(model.cos)
    var got_sin = f32_data(model.sin)
    var diff_cos = max_abs_diff(got_cos, expected_cos, rows * cols)
    var diff_sin = max_abs_diff(got_sin, expected_sin, rows * cols)
    # Printed, not just asserted: the number is what tells the next reader
    # whether the tolerance is still about rounding or has started to hide
    # something.
    print(
        "rotary table over "
        + String(rows)
        + "x"
        + String(cols)
        + ": max_abs_diff cos="
        + String(diff_cos)
        + " sin="
        + String(diff_sin)
    )
    assert_true(
        diff_cos <= ROPE_TOLERANCE,
        "derived cos table differs from the reference: max_abs_diff="
        + String(diff_cos),
    )
    assert_true(
        diff_sin <= ROPE_TOLERANCE,
        "derived sin table differs from the reference: max_abs_diff="
        + String(diff_sin),
    )
    model.keep_alive()
    dumped.keep_alive()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
