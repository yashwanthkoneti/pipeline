"""A structured view of the retrieved policy, built without the corpus.

Corpus matching was dropped on purpose. Knowing which titles exist
elsewhere never told us whether a gap was a corpus gap or a retrieval
gap, and the "missing document" suggestions it produced were noise. The
one thing it arguably helped - spotting an over-eager guardrail - turns
out to be answerable from the request and the response alone.

What does help is telling the screener the SHAPE of what it was handed.
It cannot know what the corpus contains, but it can be told that the
retrieved policy sets out four different approvers depending on amount
band. That single fact turns "the answer looks fine" into "the policy
varies by amount and the answer named no amount", which is the commonest
real defect in conditional policy and the one that otherwise gets
mislabelled as a misread question.

All of this is derived deterministically from the parsed chunks. No
model call, nothing to drift.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from ..domain.models import Chunk, Ontology
from .chunks import section_of

# Column or phrase names that mark a dimension the policy varies by.
# Deliberately generic - these are the axes institutional policy keys on,
# not anything specific to one bank's vocabulary.
_DIMENSION_WORDS = (
    "region", "country", "jurisdiction", "location", "geography",
    "entity", "legal entity", "business", "division", "desk",
    "category", "type", "class", "classification", "kind",
    "tier", "band", "level", "threshold", "limit", "amount", "value",
    "role", "grade", "seniority", "title", "position",
    "product", "instrument", "asset", "channel",
    "severity", "priority", "risk rating", "client", "customer",
    "period", "duration", "term", "frequency",
)

# Columns that hold an OUTCOME rather than a dimension. A table with a
# dimension column and an outcome column is a conditional rule, which is
# exactly what an unqualified answer tends to flatten.
_OUTCOME_WORDS = (
    "approver", "approval", "required", "requirement", "permitted",
    "allowed", "action", "outcome", "result", "process", "treatment",
    "escalation", "owner", "responsible", "signatory", "authority",
    "notice", "deadline", "timeline",
)

_CONDITIONAL_PROSE = re.compile(
    r"\b(?:depend(?:s|ing) on|varies by|based on (?:the )?(?:applicable|"
    r"relevant)?|whichever|as applicable|for each|in the case of|"
    r"where the|if the|unless the|subject to the)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).strip()


def _matches_any(text: str, words: Sequence[str]) -> List[str]:
    """Only ever called on a column header, so the text is short."""
    low = _norm(text)
    return [w for w in words if w in low]


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    seen, out = set(), []
    for v in values:
        v = str(v).strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return tuple(out)


def build_ontology(chunks: Sequence[Chunk]) -> Ontology:
    """Derive documents, sections, table columns and conditional axes."""
    documents = _unique(c.doc_title for c in chunks if c.doc_title != "?")
    sections = _unique(section_of(c) for c in chunks)

    columns: List[str] = []
    dimensions: List[str] = []
    alternatives = 0

    for chunk in chunks:
        for table in chunk.tables:
            columns.extend(table.header)
            dim_cols = [h for h in table.header
                        if _matches_any(h, _DIMENSION_WORDS)]
            out_cols = [h for h in table.header
                        if _matches_any(h, _OUTCOME_WORDS)]
            # a dimension column plus an outcome column plus more than one
            # row IS a conditional rule, whatever the prose says
            if dim_cols and len(table.rows) >= 2:
                dimensions.extend(dim_cols)
                if out_cols:
                    alternatives = max(alternatives, len(table.rows))
            elif len(table.rows) >= 2 and table.header:
                # no recognised dimension name, but a multi-row table is
                # still several cases; name the first column as the axis
                dimensions.append(table.header[0])
                alternatives = max(alternatives, len(table.rows))

        # The prose scan that used to live here is gone. It read every
        # field looking for "depends on the applicable region" and such,
        # which cost about three quarters of this function's time and
        # told us almost nothing a table had not already said: if the
        # policy varies, it varies in a table, and that is detected
        # above from structure alone. Prose that mentions a dimension
        # without tabulating it is usually describing scope, not
        # branching.

    # several documents covering the same section are alternatives for the
    # same obligation, which is the non-table form of the same problem
    if len(documents) > 1 and sections:
        by_section: dict[str, int] = {}
        for chunk in chunks:
            sec = _norm(section_of(chunk))
            if sec:
                by_section[sec] = by_section.get(sec, 0) + 1
        if by_section:
            alternatives = max(alternatives, max(by_section.values()))

    return Ontology(
        documents=documents,
        sections=sections,
        table_columns=_unique(columns),
        conditional_dimensions=_unique(dimensions),
        alternative_count=alternatives,
    )
