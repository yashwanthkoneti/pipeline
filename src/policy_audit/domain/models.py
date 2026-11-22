"""Data carried between stages. Dataclasses only - no I/O, no LLM calls.

Keeping these pure is what makes the pipeline testable without a model
endpoint: every stage is a function from one of these to another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .taxonomy import Adjudication, Factor, Gate, Route, Suspicion, Theme


# --------------------------------------------------------------------------
# parsed sources
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Table:
    """A pipe table from a chunk, with position preserved.

    header and rows always have the same width. A blank cell is still a
    cell: dropping it shifts every column after it and each header then
    labels its neighbour's value, which is indistinguishable from the
    model misreading the table.
    """

    header: Tuple[str, ...]
    rows: Tuple[Tuple[str, ...], ...]
    # What the parse had to repair or guess, in words. A table that
    # parsed "successfully" but wrongly is more dangerous than one that
    # fell through to prose, because a tidy grid reads as confirmation.
    # These notes travel into the text the judge reads.
    notes: Tuple[str, ...] = ()
    # False when a note means a value may be sitting under the wrong
    # header. The enrichment pass routes these to the LLM reader, and
    # where it cannot, render_table warns the judge instead.
    reliable: bool = True

    def as_rows(self) -> List[str]:
        """One line per row, every cell named by its own column.

        A model reading '|EMEA|50,000|' has to hold the column order in
        mind and track which row it is on. Naming the column beside each
        value removes both steps, and a quote of one row then carries its
        own meaning instead of being a bare cell value.
        """
        out = []
        for i, row in enumerate(self.rows, 1):
            cells = [f"{self.header[j]}={v or '(blank)'}"
                     for j, v in enumerate(row)]
            out.append(f"row {i}: " + "; ".join(cells))
        return out


@dataclass(frozen=True)
class Chunk:
    """One retrieved document fragment, parsed into fields."""

    index: int
    doc_title: str
    fields: Tuple[Tuple[str, str], ...]      # ordered key/value pairs
    tables: Tuple[Table, ...]
    text: str                                 # rendered, tables expanded
    # a table shape the parser could not structure (HTML, space-aligned).
    # It still reaches the judge as prose, but flagged, because an
    # unstructured table is exactly where a misread hides.
    has_unstructured_table: bool = False
    # this chunk's table was laid out by a model, not by the parser. Cell
    # text was verified against the source, so nothing was invented - but
    # the column alignment is the model's, and that is the error
    # TABLE_MISREAD hunts, so a finding resting on one is knowable.
    model_structured_table: bool = False

    def field(self, *names: str) -> str:
        low = {k.lower(): v for k, v in self.fields}
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        for n in names:
            for k, v in low.items():
                if n.lower() in k:
                    return v
        return ""


@dataclass(frozen=True)
class Turn:
    role: str                                 # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Ontology:
    """A structured view of what the retrieved text actually covers.

    The screener does not know the corpus, so it cannot tell whether a
    gap is a corpus gap or a retrieval gap - and guessing at that was
    what made corpus matching useless. What it CAN be told is the shape
    of what it was given: which documents, which sections, which tables,
    and which dimensions the policy keys on. That last one is what turns
    "the answer looks fine" into "the policy varies by region and the
    answer named no region".
    """

    documents: Tuple[str, ...]
    sections: Tuple[str, ...]
    table_columns: Tuple[str, ...]
    # dimensions the retrieved policy appears to vary by
    conditional_dimensions: Tuple[str, ...]
    # distinct alternatives found for the same obligation, if any
    alternative_count: int = 0

    def render(self) -> str:
        lines = ["## WHAT THE RETRIEVED POLICY COVERS"]
        if self.documents:
            lines.append("documents: " + "; ".join(self.documents))
        if self.sections:
            lines.append("sections: " + "; ".join(self.sections[:12]))
        if self.table_columns:
            lines.append("table columns: " + "; ".join(self.table_columns))
        if self.conditional_dimensions:
            lines.append(
                "the policy here varies by: "
                + "; ".join(self.conditional_dimensions)
                + "  <- an answer that does not name which case it applies "
                  "to may be stating one branch as the rule")
        if self.alternative_count > 1:
            lines.append(f"distinct alternatives found for the same "
                         f"obligation: {self.alternative_count}")
        if len(lines) == 1:
            lines.append("(nothing retrieved)")
        return "\n".join(lines)


@dataclass(frozen=True)
class Clarification:
    """A parsed clarification template."""

    marker: str
    questions: Tuple[Tuple[str, Tuple[str, ...]], ...]   # (text, options)

    @property
    def is_clarification(self) -> bool:
        """The marker alone. A clarification whose body parsed badly is
        still a clarification, and requiring questions here moved those
        rows into `answered` where every answer check then ran against a
        response that answered nothing."""
        return bool(self.marker)

    @property
    def body_parsed(self) -> bool:
        """Did the questions and options come out? Diagnostic only."""
        return bool(self.questions) and self.n_options > 0

    @property
    def n_questions(self) -> int:
        return len(self.questions)

    @property
    def n_options(self) -> int:
        return sum(len(o) for _, o in self.questions)


@dataclass
class Interaction:
    """One row of the audit input, fully prepared for the screener."""

    transaction_id: str
    conversation_id: str
    request: str
    response: str
    history: Tuple[Turn, ...]
    chunks: Tuple[Chunk, ...]
    ontology: Ontology
    route: Route
    clarification: Optional[Clarification] = None
    # this turn answers the agent's own clarifying question, so the user
    # has NARROWED the request rather than added to it
    is_askback_answer: bool = False
    askback_question: str = ""
    # `1a, 2b, 3c` resolved against the previous turn's option list, and
    # the readable form of the request that produces. Without this the
    # judge is handed a request with no subject in it.
    askback_selections: Tuple[Tuple[str, str], ...] = ()
    resolved_request: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)

    @property
    def history_text(self) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.history)

    @property
    def has_retrieval(self) -> bool:
        return bool(self.chunks)


# --------------------------------------------------------------------------
# screening
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Pointer:
    """Where the screener thinks the evidence lives.

    This is the cost mechanism. The screener says "chunk 2, the
    pre-clearance table" and code cuts a narrow span out of it, so the
    adjudicator reads a few hundred characters instead of the whole
    retrieval. It is also a precision mechanism: a screener that cannot
    say where to look has not found anything specific.
    """

    source: str                 # "chunks" | "response" | "request" | "history"
    chunk_index: Optional[int] = None
    locator: str = ""           # section name, table row, or a short quote


@dataclass(frozen=True)
class ScreenResult:
    check_id: str
    suspicion: Suspicion
    reason: str = ""
    pointers: Tuple[Pointer, ...] = ()

    @property
    def escalates(self) -> bool:
        from .taxonomy import ESCALATING
        return self.suspicion in ESCALATING


@dataclass(frozen=True)
class GateOutcome:
    gate: Gate
    results: Tuple[ScreenResult, ...]
    # a gate may end the audit: a correctly refused request has no answer
    # to examine, so the answer gate never runs
    stop: bool = False
    note: str = ""


# --------------------------------------------------------------------------
# adjudication
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Evidence:
    """What the adjudicator must produce to confirm anything.

    Both statements are COMPLETE sentences, not fragments. A fragment
    like "No" or "$25 and up to $200 | No" is meaningful where it sits
    and meaningless once lifted out, which is why a reviewer then has to
    reopen the trace and work the finding out again. The quote exists for
    verification; the two statements exist to be read.
    """

    answer_claims: str
    source_states: str
    source_ref: str = ""
    quote: str = ""

    def readable(self) -> bool:
        """Can a reader judge this without opening anything?"""
        def complete(s: str) -> bool:
            s = (s or "").strip()
            return bool(s) and "|" not in s and len(s.split()) >= 4
        return complete(self.answer_claims) and complete(self.source_states)


@dataclass(frozen=True)
class Finding:
    check_id: str
    adjudication: Adjudication
    theme: Optional[Theme]
    factors: Tuple[Factor, ...]
    evidence: Optional[Evidence]
    rationale: str = ""
    overturned_screener: bool = False
    grounded: bool = True

    @property
    def is_defect(self) -> bool:
        from .taxonomy import REPORTABLE
        return self.adjudication in REPORTABLE


@dataclass
class AuditResult:
    transaction_id: str
    route: Route
    findings: Tuple[Finding, ...]
    themes: Tuple[Theme, ...]
    factors: Tuple[Factor, ...]
    expected_behaviour: bool
    expected_reason: str = ""
    screened: Tuple[ScreenResult, ...] = ()
    gate_notes: Tuple[str, ...] = ()
    unproven: Tuple[str, ...] = ()
    screener_calls: int = 0
    adjudicator_calls: int = 0
    errors: Tuple[str, ...] = ()

    @property
    def needs_review(self) -> bool:
        return any(f.is_defect for f in self.findings)

    @property
    def primary_theme(self) -> Optional[Theme]:
        return self.themes[0] if self.themes else None
