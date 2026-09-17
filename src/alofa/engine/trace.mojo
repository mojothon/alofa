"""Turning a scheduling decision into bytes, and bytes back into a decision.

The format here is a contract, not an implementation detail: the fixtures under
`tests/fixtures/scheduler/` are produced by `scripts/dump_scheduler_reference.py`,
which is an *independent* implementation of the same policy. If the two ever
disagree by one character, the gate fails — which is the whole point of having a
reference at all.

Three rules the format follows:

- **No floats.** The threshold is permille, the digest is an integer, and every
  count is an integer. A float would make the gate a tolerance gate, and a
  tolerance gate for a decision function is not a gate.
- **No key ordering, no whitespace beyond one space.** Fields are written in a
  fixed order and read by position, so two implementations cannot disagree about
  what `dict` iteration order should have been.
- **Empty is `-`, not an empty string.** An empty field is indistinguishable
  from a missing one when you are diffing bytes by eye at two in the morning.

This file allocates (`String` per line). That is fine and intended: recording a
trace is opt-in, and `Scheduler.step` — the thing that runs per token — does not
call any of it.

Run:
    pixi run mojo run -I src tests/unit/test_scheduler.mojo
"""

from alofa.core.error import ERR_PARSE, AlofaError
from alofa.core.text import parse_int
from alofa.engine.scheduler import Action, SchedConfig, SchedInput

comptime EMPTY = "-"

comptime CFG_PREFIX = "CFG"
comptime IN_PREFIX = "IN="
comptime OUT_PREFIX = "OUT="


def parts_of(text: String, sep: String) -> List[String]:
    """Split keeping empty fields as empty strings (they mean something here)."""
    var out = List[String]()
    for span in text.split(sep):
        out.append(String(span))
    return out^


def value_after(text: String, sep: String) raises AlofaError -> String:
    """Everything after the *first* `sep`.

    Splitting on every occurrence would be wrong on a line like `IN=A=1:2:3`,
    where the second `=` is inside the payload, not a separator.
    """
    if text.find(sep) < 0:
        raise AlofaError(ERR_PARSE, "missing '" + sep + "' in '" + text + "'")
    var pieces = List[String]()
    for span in text.split(sep, 1):
        pieces.append(String(span))
    if len(pieces) < 2:
        raise AlofaError(ERR_PARSE, "empty value in '" + text + "'")
    return pieces[1]


def config_line(cfg: SchedConfig) -> String:
    """`CFG b=64 c=16 s=16 k=64 w=900 t=8` — one line, fixed key order."""
    return (
        CFG_PREFIX
        + " b="
        + String(cfg.token_budget)
        + " c="
        + String(cfg.max_chunk)
        + " s="
        + String(cfg.block_size)
        + " k="
        + String(cfg.capacity_blocks)
        + " w="
        + String(cfg.watermark_permille)
        + " t="
        + String(cfg.max_wait_ticks)
    )


def parse_config(line: String) raises AlofaError -> SchedConfig:
    """Read back what `config_line` wrote, tolerating nothing else."""
    var fields = parts_of(line, " ")
    if len(fields) != 7:
        raise AlofaError(ERR_PARSE, "config line must have CFG plus six fields")
    if fields[0] != CFG_PREFIX:
        raise AlofaError(ERR_PARSE, "config line must start with CFG")
    var budget = parse_int(value_after(fields[1], "="))
    var chunk = parse_int(value_after(fields[2], "="))
    var block_size = parse_int(value_after(fields[3], "="))
    var capacity = parse_int(value_after(fields[4], "="))
    var watermark = parse_int(value_after(fields[5], "="))
    var max_wait = parse_int(value_after(fields[6], "="))
    return SchedConfig(budget, chunk, block_size, capacity, watermark, max_wait)


def input_line(inp: SchedInput) -> String:
    """`IN=A=1:300:8,2:64:4|C=3|F=-` — arrivals, cancels, completions."""
    var arrivals = EMPTY
    if inp.n_arrived > 0:
        arrivals = ""
        for i in range(inp.n_arrived):
            if i > 0:
                arrivals += ","
            arrivals += (
                String(inp.arr_id[i])
                + ":"
                + String(inp.arr_prompt[i])
                + ":"
                + String(inp.arr_max_new[i])
            )
    var cancels = EMPTY
    if inp.n_cancelled > 0:
        cancels = ""
        for i in range(inp.n_cancelled):
            if i > 0:
                cancels += ","
            cancels += String(inp.cancelled[i])
    var finished = EMPTY
    if inp.n_finished > 0:
        finished = ""
        for i in range(inp.n_finished):
            if i > 0:
                finished += ","
            finished += String(inp.finished[i])
    return IN_PREFIX + "A=" + arrivals + "|C=" + cancels + "|F=" + finished


def action_line(act: Action) -> String:
    """`OUT=t=7 P=1:0:16,2:0:8 D=1 X=3 F=- C=2 K=1993893780`."""
    var prefill = EMPTY
    if act.n_prefill > 0:
        prefill = ""
        for i in range(act.n_prefill):
            if i > 0:
                prefill += ","
            prefill += (
                String(act.p_req[i])
                + ":"
                + String(act.p_start[i])
                + ":"
                + String(act.p_end[i])
            )
    var decode = EMPTY
    if act.n_decode > 0:
        decode = ""
        for i in range(act.n_decode):
            if i > 0:
                decode += ","
            decode += String(act.decode[i])
    var preempted = EMPTY
    if act.n_preempted > 0:
        preempted = ""
        for i in range(act.n_preempted):
            if i > 0:
                preempted += ","
            preempted += String(act.preempted[i])
    var finished = EMPTY
    if act.n_finished > 0:
        finished = ""
        for i in range(act.n_finished):
            if i > 0:
                finished += ","
            finished += String(act.finished[i])
    return (
        OUT_PREFIX
        + "t="
        + String(act.tick_seq)
        + " P="
        + prefill
        + " D="
        + decode
        + " X="
        + preempted
        + " F="
        + finished
        + " C="
        + String(act.preempt_total)
        + " K="
        + String(act.state_digest)
    )


def trace_line(inp: SchedInput, act: Action) -> String:
    """One recorded tick: what came in, what went out."""
    return input_line(inp) + " " + action_line(act)


def parse_input(text: String) raises AlofaError -> SchedInput:
    """Read back what `input_line` wrote.

    `A=-` / `C=-` / `F=-` mean "nothing this tick". Anything else that fails to
    parse is a broken fixture, and a broken fixture must stop the run rather
    than quietly become an empty event list.
    """
    if not text.startswith(IN_PREFIX):
        raise AlofaError(ERR_PARSE, "input line must start with IN=")
    var body = value_after(text, "=")
    var groups = parts_of(body, "|")
    if len(groups) != 3:
        raise AlofaError(ERR_PARSE, "input line must have three groups")
    var inp = SchedInput()
    for g in range(3):
        var field = groups[g]
        var items = value_after(field, "=")
        if items == EMPTY:
            continue
        for piece in items.split(","):
            var item = String(piece)
            if g == 0:
                var triple = parts_of(item, ":")
                if len(triple) != 3:
                    raise AlofaError(ERR_PARSE, "arrival needs id:prompt:max_new")
                inp.add_arrival(
                    parse_int(triple[0]),
                    parse_int(triple[1]),
                    parse_int(triple[2]),
                )
            else:
                var req = parse_int(item)
                if g == 1:
                    inp.add_cancel(req)
                else:
                    inp.add_finished(req)
    return inp^
