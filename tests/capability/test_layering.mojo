"""Gate: the L0 core layer must not know that upper layers exist.

The layering rule is one-directional — `core` → `kernels` → `model` →
`runtime` → `engine` → `srv`. A dependency that ran backwards would be easy to
add and hard to see: `core/memory.mojo` mentioning attention would still
compile, and only start to hurt when somebody tried to reuse the core layer
on a backend that has no such concept.

So the rule is checked here, on every run, by reading each core file and
looking for upper-layer vocabulary — including in comments, because a comment
that explains a core type in terms of a model concept is how the coupling
starts. Mojo 1.0 has no directory traversal, so the file list is written out;
`test_core_files_are_listed_and_readable` is what stops that list from
silently going stale.

The last test is the one that keeps this gate honest: it feeds the checker a
string that does violate the rule and asserts the checker says so. A gate that
cannot fail is worse than no gate.

Run:
    pixi run mojo run -I src tests/capability/test_layering.mojo
"""

from std.io import FileHandle
from std.testing import TestSuite, assert_equal, assert_true

comptime CORE_ROOT = "src/alofa/core/"


def core_files() -> List[String]:
    """Every file in the core layer. Update this when a file is added."""
    var files = List[String]()
    files.append(CORE_ROOT + "__init__.mojo")
    files.append(CORE_ROOT + "dtype.mojo")
    files.append(CORE_ROOT + "error.mojo")
    files.append(CORE_ROOT + "log.mojo")
    files.append(CORE_ROOT + "memory.mojo")
    files.append(CORE_ROOT + "mmap.mojo")
    files.append(CORE_ROOT + "rng.mojo")
    files.append(CORE_ROOT + "tensor.mojo")
    files.append(CORE_ROOT + "text.mojo")
    files.append(CORE_ROOT + "ffi/__init__.mojo")
    files.append(CORE_ROOT + "ffi/mem.mojo")
    files.append(CORE_ROOT + "ffi/posix.mojo")
    return files^


def forbidden_terms() -> List[String]:
    """Vocabulary that belongs to a layer above L0, or to a pillar.

    Import paths are included because a comment is a hint while an import is a
    facts-on-the-ground dependency.
    """
    var terms = List[String]()
    terms.append("attention")
    terms.append("kv cache")
    terms.append("kvcache")
    terms.append("paged")
    terms.append("radix")
    terms.append("sampler")
    terms.append("logits")
    terms.append("scheduler")
    terms.append("engine")
    terms.append("prefill")
    terms.append("decode")
    terms.append("embedding")
    terms.append("rope")
    terms.append("rmsnorm")
    terms.append("swiglu")
    terms.append("softmax")
    terms.append("matmul")
    terms.append("qwen")
    terms.append("llama")
    terms.append("transformer")
    terms.append("tokenizer")
    terms.append("openai")
    terms.append("http")
    terms.append("cuda")
    terms.append("model")
    terms.append("weight")
    terms.append("kernel")
    terms.append("batch")
    terms.append("inference")
    terms.append("from alofa.kernels")
    terms.append("from alofa.model")
    terms.append("from alofa.runtime")
    terms.append("from alofa.engine")
    terms.append("from alofa.srv")
    terms.append("from alofa.tokenizer")
    terms.append("from alofa.verify")
    return terms^


def same_letter_ignoring_case(left: UInt8, right: UInt8) -> Bool:
    var a = left
    var b = right
    # ASCII upper-case to lower-case; nothing else in these files is cased.
    if a >= 65 and a <= 90:
        a = a + 32
    if b >= 65 and b <= 90:
        b = b + 32
    return a == b


def contains_ignoring_case(haystack: String, needle: String) -> Bool:
    """Substring search that treats `A` and `a` as the same letter."""
    var n = haystack.byte_length()
    var m = needle.byte_length()
    if m == 0 or m > n:
        return False
    var hay = haystack.as_bytes()
    var nee = needle.as_bytes()
    var start = 0
    while start <= n - m:
        var offset = 0
        while offset < m:
            if not same_letter_ignoring_case(hay[start + offset], nee[offset]):
                break
            offset += 1
        if offset == m:
            return True
        start += 1
    return False


def violations_in(text: String) -> String:
    """Every forbidden term appearing in `text`, as one detail string.

    Collecting all of them rather than failing on the first means one run
    reports the whole mess instead of making the next person fix it one word
    at a time.
    """
    var found = String("")
    for term in forbidden_terms():
        if contains_ignoring_case(text, term):
            if found.byte_length() > 0:
                found += ","
            found += term
    return found


def read_text(path: String) raises -> String:
    var handle = FileHandle(path, "r")
    var text = handle.read()
    handle.close()
    return text


def test_core_files_are_listed_and_readable() raises:
    """A gate over a stale or empty file list is not a gate."""
    var files = core_files()
    assert_true(len(files) > 0, "the core file list must not be empty")
    for path in files:
        var text = read_text(path)
        assert_true(
            text.byte_length() > 0, "core file is empty or missing: " + path
        )


def test_core_does_not_name_upper_layer_concepts() raises:
    for path in core_files():
        var found = violations_in(read_text(path))
        assert_equal(found, "", "upper-layer vocabulary in " + path + ": " + found)


def test_the_gate_itself_is_not_vacuous() raises:
    """The checker must reject text that does break the rule."""
    var offending = String("def run(): return engine.prefill(batch)")
    assert_true(
        violations_in(offending).byte_length() > 0,
        "the layering checker accepted offending text, so it proves nothing",
    )


def test_a_clean_line_passes() raises:
    assert_equal(violations_in("var data = mmap_anonymous(1024)"), "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
