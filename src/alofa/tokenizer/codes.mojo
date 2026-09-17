"""Named error codes for the tokenizer pillar.

Same reasoning as `core.error`, kept local rather than imported: `tokenizer/` is
an orthogonal pillar and stays compilable on its own, so it carries its own
codes instead of reaching into L0 for them. Codes are `comptime` so a misspelled
name fails to compile, and once persisted they are a contract — **append only,
never renumber**.
"""

comptime TOK_ERR_NONE = 0
comptime TOK_ERR_MISSING_BYTE_TOKEN = 1
comptime TOK_ERR_MISSING_MERGE_TOKEN = 2
comptime TOK_ERR_BAD_VOCAB = 3
comptime TOK_ERR_BAD_MERGES = 4
comptime TOK_ERR_BAD_FIXTURE_LINE = 5
comptime TOK_ERR_IO = 6
comptime TOK_ERR_UNKNOWN_ID = 7

comptime TOK_ERR_LAST = 7


def tokenizer_error_name(code: Int) -> String:
    """Identifier of `code`, or "unknown" — so an unnamed code is discoverable."""
    if code == TOK_ERR_NONE:
        return "none"
    elif code == TOK_ERR_MISSING_BYTE_TOKEN:
        return "missing_byte_token"
    elif code == TOK_ERR_MISSING_MERGE_TOKEN:
        return "missing_merge_token"
    elif code == TOK_ERR_BAD_VOCAB:
        return "bad_vocab"
    elif code == TOK_ERR_BAD_MERGES:
        return "bad_merges"
    elif code == TOK_ERR_BAD_FIXTURE_LINE:
        return "bad_fixture_line"
    elif code == TOK_ERR_IO:
        return "io"
    elif code == TOK_ERR_UNKNOWN_ID:
        return "unknown_id"
    return "unknown"


def tokenizer_error(code: Int, detail: String) -> Error:
    """Build a tokenizer error whose message names the code that failed."""
    return Error("tokenizer_error=" + tokenizer_error_name(code) + ": " + detail)
