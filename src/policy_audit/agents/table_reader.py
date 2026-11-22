"""Structure a table with an LLM, then verify the result in code.

Why this is not simply "let the model read the table".

The audit's job includes catching the agent binding a cell to the wrong
row. If a model does that binding FOR the judge and makes the same
mistake, the check passes and nothing is learned. So the model here is
not asked what the table MEANS - it is asked only to lay the table out,
and the layout is then checked:

  every cell it emits must appear verbatim in the raw chunk text
  the row widths must be consistent
  no row may be invented, and none dropped silently

Invention is therefore impossible. MISALIGNMENT is not - a model could
put real values under the wrong headers and every cell would still
verify. That residual risk is why:

  * the deterministic parser stays authoritative wherever it succeeds.
    A pipe table is parsed in code and the model is never consulted.
  * the model is used only where the parser found no table at all, where
    the judge's alternative is unstructured prose and anything is better.
  * a table the model structured is MARKED, so a finding that rests on
    one can be weighted differently from one resting on a parsed table.

Set PA_LLM_TABLES=off to disable, fallback (default) for the above, or
all to run it on every table - which is only useful with
compare_tables.py, to find out whether the deterministic parse is
actually wrong on your data rather than assumed to be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..domain.models import Table
from ..parsing.tables import assess
from .client import JsonAgent

_SYSTEM = """\
You lay out tables. You do not interpret them.

You are given a fragment of a policy document containing a table the
automatic parser could not split, or split wrongly. Return it as a grid.

Rules, and the first is absolute:

1. COPY CELL TEXT VERBATIM. Every cell you output must appear in the
   fragment exactly as you write it. Do not reword, expand, abbreviate,
   fix spelling or convert units. A cell you cannot copy exactly must be
   returned as an empty string.
2. Keep the row order and the column order of the original.
3. Every row must have the same number of cells as the header. Use an
   empty string for a blank cell - never shift a value into a
   neighbouring column to fill a gap.
4. If a cell spans several lines in the original, join it with a single
   space.
5. If the fragment holds more than one table, return the largest.
6. If there is no table, return an empty header and no rows.

These fragments come out of PDF extraction, so expect damage. What to do
with each kind:

  BORDERLESS. No pipes at all, columns held apart by runs of spaces.
  The column boundary is the vertical gap that repeats down the lines,
  not any single wide space. Read down the fragment, not across one row.

  A SURVIVING LEFT RULE. Every line starts with one `|` and has no
  others. That pipe is a border. It is not a column boundary and it does
  not mean the first column is empty.

  RAGGED PIPES. Rows carry different numbers of pipes. Decide each
  column from the header and the values that line up beneath it. Do not
  assume the Nth pipe is the Nth boundary on every row.

  THE HEADER REPEATED MID-TABLE. A page break reprints the column
  headings inside the body. Emit the header once and drop the repeat; it
  is not a data row.

  MERGED CELLS. A value that applies to several rows is printed once, on
  the first of them, and the rows under it are blank in that column.
  LEAVE THEM BLANK. Do not copy the value down, and do not move a value
  up into a blank. What the blank means is not yours to decide.

Do not explain anything. Return JSON only.
"""

_USER = """\
## FRAGMENT
{fragment}

## RETURN
{{"header": ["<column 1>", "<column 2>", ...],
  "rows": [["<cell>", "<cell>", ...], ...]}}
"""

_PUNCT = re.compile(r"[^a-z0-9]+")
# the stand-in this module writes for a header the model left blank
_PLACEHOLDER = re.compile(r"col\d+")


def _flat(text: str) -> str:
    return _PUNCT.sub(" ", str(text or "").lower()).strip()


@dataclass(frozen=True)
class TableReadResult:
    table: Optional[Table]
    verified: bool
    reason: str = ""
    cells_checked: int = 0
    cells_found: int = 0

    @property
    def coverage(self) -> float:
        return (round(self.cells_found / self.cells_checked, 3)
                if self.cells_checked else 0.0)


def verify_table(table: Table, fragment: str, *,
                 floor: float = 1.0) -> TableReadResult:
    """Is EVERY non-empty cell actually in the source text?

    The default is 1.0, not a tolerance. A 0.9 floor accepted a table
    with one paraphrased cell out of twelve, and one paraphrased policy
    value is the entire problem - it reads like a quote, cannot be
    checked later, and a model that paraphrased once will paraphrase
    again. Lower it only for a calibration run.
    """
    haystack = _flat(fragment)
    if not table.header:
        return TableReadResult(None, False, "no header returned")

    widths = {len(r) for r in table.rows}
    if widths and widths != {len(table.header)}:
        return TableReadResult(
            None, False,
            f"row widths {sorted(widths)} do not match the header width "
            f"{len(table.header)} - a shifted column cannot be trusted")

    checked = found = 0
    missing: List[str] = []
    for cell in list(table.header) + [c for r in table.rows for c in r]:
        cell = str(cell or "").strip()
        if not cell:
            continue
        # `col3` is OUR placeholder for a header the model left empty.
        # It asserts nothing about the source, so checking it against the
        # source rejected every otherwise-perfect read with the wrong
        # reason - "the model rewrote the table" - and sent the judge
        # back to prose. The unbindable column is a reliability problem,
        # which `assess` already reports, not a verbatim one.
        if _PLACEHOLDER.fullmatch(cell):
            continue
        checked += 1
        if _flat(cell) in haystack:
            found += 1
        elif len(missing) < 4:
            missing.append(cell)

    if not checked:
        return TableReadResult(None, False, "every cell was empty")
    coverage = found / checked
    if coverage < floor:
        return TableReadResult(
            None, False,
            f"only {coverage:.0%} of cells appear verbatim in the source "
            f"(not found: {missing}) - the model rewrote the table rather "
            f"than laying it out",
            checked, found)
    return TableReadResult(table, True, "", checked, found)


def _to_table(payload: Dict[str, Any]) -> Optional[Table]:
    header = payload.get("header")
    rows = payload.get("rows")
    if not isinstance(header, list) or not header:
        return None
    head = tuple(str(h or "").strip() or f"col{i + 1}"
                 for i, h in enumerate(header))
    body: List[Tuple[str, ...]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, list):
                continue
            # Deliberately NOT padded. A short row is ambiguous - we
            # cannot know which column the model dropped - and padding it
            # hid exactly the shifted-column case the width check below
            # exists to reject.
            body.append(tuple(str(c if c is not None else "").strip()
                              for c in row))
    # The model's layout gets the same assessment as the parser's. A
    # headerless column is unbindable whoever produced it, and a merged
    # cell the model correctly left blank is just as ambiguous to read,
    # so the note has to travel either way.
    notes, reliable = assess(head, body, [len(head)] + [len(r) for r in body])
    return Table(header=head, rows=tuple(body), notes=notes,
                 reliable=reliable)


class TableReader:
    """One call per fragment, only where the parser found nothing."""

    def __init__(self, agent: JsonAgent, *, verify_floor: float = 0.9,
                 max_fragment_chars: int = 6000) -> None:
        self.agent = agent
        self.verify_floor = verify_floor
        self.max_fragment_chars = max_fragment_chars
        self.stats = {"attempted": 0, "verified": 0, "rejected": 0}

    def read(self, fragment: str) -> TableReadResult:
        fragment = str(fragment or "")[:self.max_fragment_chars]
        if not fragment.strip():
            return TableReadResult(None, False, "empty fragment")

        self.stats["attempted"] += 1
        payload = self.agent.ask(_SYSTEM, _USER.format(fragment=fragment),
                                 expect=["header"])
        if "_error" in payload:
            self.stats["rejected"] += 1
            return TableReadResult(None, False,
                                   f"not structured: {payload['_error']}")

        table = _to_table(payload)
        if table is None:
            self.stats["rejected"] += 1
            return TableReadResult(None, False, "no usable grid returned")

        result = verify_table(table, fragment, floor=self.verify_floor)
        self.stats["verified" if result.verified else "rejected"] += 1
        return result
