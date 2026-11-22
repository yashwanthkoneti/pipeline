"""Pipe tables, parsed positionally - and an honest account of when the
parse cannot be trusted.

The format is markdown-ish: |a|b|c| per row, sometimes with a |---|---|
separator, sometimes with leading or trailing empty cells.

The trap is `.strip("|")`, which is greedy. On `||Amount|Registration|`
it eats the empty first cell, every column shifts left by one, and each
header then labels its neighbour's value:

    Registration=$25-$200      <- wrong column
    ?=No                       <- fell off the end

That is worse than losing the header, because it reads as confirmation.
Strip exactly one pipe from each end and an empty cell keeps its place.

The same reasoning, one level up, is why `assess` exists. Real chunks
come out of PDF extraction, and the pipes are not clean: rows carry
different numbers of them, the header reappears mid-table at a page
break, merged cells arrive as blanks under the row above, and borderless
tables arrive with no pipes at all. Three of those four STILL PARSE.
They produce a confident-looking grid with a value under the wrong
header, `has_unstructured_table` stays False, so nothing is flagged and
the LLM reader is never called. A table that falls through to prose is
the safe failure; a table that parses wrongly is the dangerous one.

So each parse is assessed. Repeated header rows are dropped, which is
unambiguous. Everything else ambiguous is recorded as a note and the
table is marked unreliable - which routes it to the LLM reader, and
where that is off or rejects the result, puts the warning in front of
the judge rather than letting a tidy grid speak for itself.

Nothing here guesses. A merged cell carried down as a blank is NOT
forward-filled: the value might belong to the row above, or the row
might genuinely have no value, and inventing the answer manufactures
exactly the misreading TABLE_MISREAD is meant to catch.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from ..domain.models import Table

PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# a tab-delimited row with at least two columns. Cheap to support and it
# is what a copy out of a spreadsheet produces.
TAB_ROW = re.compile(r"^[^\t\n]*\t[^\t\n]*(?:\t|$)")
# HTML table markup. The space-aligned case is not a regex any more -
# see _gap_columns.
_HTML_TABLE = re.compile(r"<\s*table\b|<\s*t[rdh]\b", re.I)
UNSTRUCTURED_TABLE = _HTML_TABLE
# |---|:--:|---| and friends: separator chars only, and at least one dash
_SEP_ROW = re.compile(r"^[\s|:-]*-[\s|:-]*$")
_ESCAPED = "\x00"

# Thresholds for the quality assessment. Named, because each one was
# chosen against a real failure and a bare number in a condition hides
# which.
MAX_BLANK_RATIO = 0.25          # merged cells carried down as blanks
MIN_COLUMNS = 2                 # one column is a list, not a table
GAP_MIN_WIDTH = 2               # a column gap in borderless output


def is_pipe_row(line: str) -> bool:
    return bool(PIPE_ROW.match(line))


def is_table_row(line: str) -> bool:
    """A pipe row, or a tab-delimited one."""
    return bool(PIPE_ROW.match(line)) or bool(TAB_ROW.match(line))


def split_row(line: str) -> List[str]:
    """One pipe row -> cells, position preserved.

    An escaped `\\|` inside a cell is data, not a boundary.
    """
    raw = str(line)
    if "|" not in raw and "\t" in raw:
        return [c.strip() for c in raw.rstrip("\n").split("\t")]
    s = raw.strip().replace(r"\|", _ESCAPED)
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.replace(_ESCAPED, "|").strip() for c in s.split("|")]


def _same_row(a: Sequence[str], b: Sequence[str]) -> bool:
    """Two rows carrying the same text, ignoring case and spacing."""
    norm = lambda r: [re.sub(r"\s+", " ", str(c or "")).strip().lower()
                      for c in r]
    return norm(a) == norm(b)


def assess(header: Sequence[str], rows: Sequence[Sequence[str]],
           raw_widths: Sequence[int], repeated_headers: int = 0
           ) -> Tuple[Tuple[str, ...], bool]:
    """What this parse had to guess. Returns (notes, reliable).

    Reliable means: a reader can take `header[j]` as the label for
    `row[j]`. Each note below is a case where that binding may not hold,
    and the whole point is to say so rather than to emit a grid that
    looks the same either way.

    Two different problems, with two different remedies, so they are not
    merged into one flag:

      the columns may be misaligned -> unreliable, and the LLM reader
      is given the original layout to try instead.

      the columns are right but the CONTENT is ambiguous, as with a
      merged cell arriving as a blank -> a note. Re-reading cannot help;
      the LLM sees the same blank and is forbidden to fill it. Only the
      judge can be told, so it is told.
    """
    notes: List[str] = []
    unreliable = False

    if repeated_headers:
        # Unambiguous, so repaired rather than flagged. Worth recording
        # because it dates the table to a PDF page break, which is also
        # where column widths tend to shift.
        notes.append(
            f"{repeated_headers} body row(s) repeated the header text and "
            f"were dropped (a page break in the source PDF)")

    distinct = sorted(set(raw_widths))
    if len(distinct) > 1:
        notes.append(
            f"rows arrived with different cell counts {distinct}; short "
            f"rows were padded on the right, so a value may be sitting "
            f"under the wrong column")
        unreliable = True

    generic = [i + 1 for i, h in enumerate(header)
               if re.fullmatch(r"col\d+", str(h or ""))]
    if generic:
        notes.append(
            f"column(s) {generic} have no header, so their values cannot "
            f"be bound to anything")
        unreliable = True

    if header and len(header) < MIN_COLUMNS:
        notes.append(
            "only one column was found; the delimiters in the source are "
            "probably not pipes")
        unreliable = True

    # A merged cell in the source arrives as a blank under the row that
    # held the value. The grid is a faithful rendering; what is ambiguous
    # is whether the blank means "same as above" or "nothing". Never
    # guessed, because forward-filling the wrong one manufactures the
    # misreading this audit exists to catch.
    carried = [i + 1 for i, r in enumerate(rows)
               if r and not str(r[0] or "").strip()
               and str(header[0] if header else "").strip()]
    if carried:
        notes.append(
            f"row(s) {carried} have no value in the first column "
            f"('{header[0]}'). In the source this is usually a cell merged "
            f"with the row above, so the blank may inherit that row's value "
            f"- or may genuinely be empty. This parse does not decide "
            f"which, so do not read a blank here as 'no value'")

    cells = [c for r in rows for c in r]
    if cells:
        blank = sum(1 for c in cells if not str(c or "").strip())
        ratio = blank / len(cells)
        if ratio > MAX_BLANK_RATIO:
            notes.append(
                f"{ratio:.0%} of body cells are blank, too many for a "
                f"table this shape; the delimiters in the source are "
                f"probably not lining up with the columns")
            unreliable = True

    return tuple(notes), not unreliable


def parse_table(lines: Sequence[str]) -> Table | None:
    """Pipe rows -> a Table with every row the same width.

    Rows are padded, never truncated: a short row keeps its cells in
    their original columns instead of sliding left. A blank header
    becomes col1, col2 so the position stays visible.

    A body row that repeats the header is dropped - that is a page break
    in the source PDF, not data. Everything else the parse had to guess
    is recorded on the table rather than smoothed over.
    """
    rows = [split_row(l) for l in lines if not _SEP_ROW.match(str(l).strip())]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return None

    head_raw, body_raw = rows[0], rows[1:]
    kept = [r for r in body_raw if not _same_row(r, head_raw)]
    repeated = len(body_raw) - len(kept)
    if not kept:
        return None

    raw_widths = [len(r) for r in [head_raw] + kept]
    width = max(raw_widths)
    padded = [tuple(r + [""] * (width - len(r))) for r in [head_raw] + kept]
    header = tuple(h if h else f"col{i + 1}"
                   for i, h in enumerate(padded[0]))
    body = tuple(padded[1:])
    notes, reliable = assess(header, body, raw_widths, repeated)
    return Table(header=header, rows=body, notes=notes, reliable=reliable)


def split_segments(value: str) -> List[Tuple[str, object]]:
    """Walk a field value and return ordered ('text', s) / ('table', Table).

    Order matters. A sentence printed after a table must stay after it,
    or a qualifier such as "the APAC row is superseded from July" lands
    against the wrong rows and the audit inherits the misreading it is
    looking for.
    """
    segs: List[Tuple[str, object]] = []
    run: List[str] = []
    buf: List[str] = []

    def flush_table():
        table = parse_table(run)
        if table is not None:
            segs.append(("table", table))
        else:
            buf.extend(run)          # a lone pipe row is not a table
        run.clear()

    def flush_text():
        s = "\n".join(buf).strip()
        if s:
            segs.append(("text", s))
        buf.clear()

    lines = str(value).split("\n")
    for i, line in enumerate(lines):
        if is_table_row(line):
            if buf:
                flush_text()
            run.append(line)
            continue
        # A blank line inside a table does not end it. Treating it as a
        # boundary split the header off the body, and a header alone is
        # not a table, so the whole thing fell through to prose and the
        # judge lost the column bindings.
        if run and not line.strip():
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if is_table_row(nxt):
                continue
        if run:
            flush_table()
        buf.append(line)
    if run:
        flush_table()
    flush_text()
    return segs


def render_table(table: Table) -> str:
    """Re-emit a table so a quote of one row carries its own meaning.

    An unreliable parse says so, here, where the judge will read it. A
    grid looks equally authoritative whether or not its columns line up,
    and a judge that cannot tell will flag the AGENT for a misreading
    that belongs to this parser.
    """
    head = " | ".join(table.header)
    lines = [f"TABLE  columns: {head}",
             "  (a value belongs ONLY to the row it is printed under)"]
    lines.extend("  " + r for r in table.as_rows())
    if not table.reliable:
        lines.append(
            "  CAUTION: this layout is not trustworthy, for the reason(s) "
            "below. Do NOT treat a cell sitting under a header as proof "
            "the policy binds them, and do not raise a table misreading "
            "on this table alone.")
    for note in table.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def _decorative_pipes(line: str) -> str:
    """Blank out pipes that are drawing a border, not delimiting cells.

    `pdftotext` sometimes keeps the left rule of a table and drops the
    rest, so a row arrives as `|Band  Approver  Notice`. One pipe is a
    border; it carries no column information and it stopped the old
    detector dead, because the pattern required the line to start with a
    non-pipe. Replaced with a space rather than removed, so every
    remaining character keeps its column offset.
    """
    s = str(line)
    if s.count("|") >= MIN_COLUMNS:
        return s
    return s.replace("|", " ")


def _gap_columns(lines: Sequence[str]) -> List[Tuple[int, int]]:
    """Character ranges that are blank on EVERY one of these lines.

    This is what makes a borderless table detectable without guessing.
    Prose wraps, so its spaces land in different places on each line;
    a column layout puts the gap at the same offset line after line.
    Runs touching either margin are dropped - a shared indent is not a
    column boundary.
    """
    rows = [str(l) for l in lines]
    if len(rows) < 2:
        return []
    width = max(len(r) for r in rows)
    if width < 4:
        return []
    padded = [r.ljust(width) for r in rows]
    blank = [all(p[i] == " " for p in padded) for i in range(width)]

    runs: List[Tuple[int, int]] = []
    start = None
    for i, is_blank in enumerate(blank + [False]):
        if is_blank and start is None:
            start = i
        elif not is_blank and start is not None:
            if i - start >= GAP_MIN_WIDTH:
                runs.append((start, i))
            start = None
    return [(a, b) for a, b in runs if a > 0 and b < width]


def _is_column_layout(lines: Sequence[str]) -> bool:
    """Do these lines share an interior gap AND straddle it, every one?

    The second half is not pedantry. A gap blank on every line can sit
    entirely to the right of a short line's text, so `("Band    Approver",
    "x")` shares one without being a table. Requiring every line to carry
    content on both sides of the gap is what distinguishes a column
    layout from a long line next to a short one.
    """
    gaps = _gap_columns(lines)
    if not gaps:
        return False
    left, right = gaps[0][0], gaps[-1][1]
    width = max(len(l) for l in lines)
    return all(str(l).ljust(width)[:left].strip()
               and str(l).ljust(width)[right:].strip() for l in lines)


def looks_like_unstructured_table(text: str) -> bool:
    """A table shape this parser cannot split, so route it onward.

    Either HTML table markup, or two-plus consecutive lines forming a
    column layout - the signature of a table printed without borders, or
    with only its left rule surviving extraction.

    The window is a SUFFIX, shrunk from the front until it holds a
    layout. A chunk's metadata arrives as one long dense line directly
    above the table, and a window that keeps it can never share a gap
    with the rows below, so every real borderless table went undetected.
    """
    body = str(text or "")
    if _HTML_TABLE.search(body):
        return True

    run: List[str] = []
    for raw in body.split("\n"):
        line = _decorative_pipes(raw)
        if not line.strip() or is_table_row(raw):
            run = []
            continue
        run.append(line.rstrip())
        while len(run) >= 2 and not _is_column_layout(run):
            run.pop(0)
        if len(run) >= 2:
            return True
    return False
