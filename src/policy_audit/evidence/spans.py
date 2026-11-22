"""Cut narrow passages out of the sources, from the screener's pointers.

This is the cost mechanism and a precision mechanism at once.

Cost: the adjudicator reads a few hundred characters per check instead of
the whole retrieval, which is what makes a large model affordable on the
subset of rows that escalate.

Precision: a pointer that cannot be located is itself a finding about the
screener. If it pointed at a table row that does not exist, there was
nothing specific there, and the adjudicator is told so rather than being
handed the whole chunk and left to find something.

Table rows are never cut mid-table. A fragment of a table is the exact
material that produces a misread, so a pointer into a table brings the
header and the whole row with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..domain.models import Chunk, Interaction, Pointer, Table
from ..parsing.tables import render_table

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD.findall(str(text).lower()) if len(w) > 2]


def overlap(needle: str, haystack: str) -> float:
    """How much of needle's content appears in haystack.

    Verbatim containment first, then content-word overlap, because a
    locator is usually a paraphrase of a section heading rather than an
    exact substring.
    """
    n, h = _tokens(needle), _tokens(haystack)
    if not n:
        return 0.0
    if " ".join(n) in " ".join(h):
        return 1.0
    hs = set(h)
    return round(sum(1 for w in n if w in hs) / len(n), 3)


@dataclass(frozen=True)
class Span:
    source: str
    ref: str
    text: str
    match: float

    def render(self) -> str:
        return f"[{self.source} | {self.ref}]\n{self.text}"


def _table_span(table: Table, locator: str) -> Optional[str]:
    """A table row with its header, or the whole table.

    Narrowing to one row only helps when the locator identifies that row.
    A locator naming a COLUMN - "Approver", "Registration required" -
    matches every row equally, and picking the first then hides the very
    thing the pointer was about: that the policy has several branches.
    Those checks exist to catch one branch presented as the rule, so
    handing over one branch guarantees a rejection.

    So a single row is sent only when it beats the runner-up clearly.
    Otherwise the whole table goes, which is still bounded.
    """
    rows = table.as_rows()
    if not locator or not rows:
        return render_table(table)

    scored = sorted(((overlap(locator, line), line) for line in rows),
                    key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    if best_score < 0.3 or best_score - runner_up < 0.25:
        return render_table(table)

    head = " | ".join(table.header)
    return (f"TABLE  columns: {head}\n"
            "  (a value belongs ONLY to the row it is printed under)\n"
            f"  {best}")


def _window(text: str, locator: str, width: int) -> Tuple[str, float]:
    """A window of text around the best match for locator."""
    text = str(text or "")
    if not text:
        return "", 0.0
    if not locator:
        return text[:width], 0.0

    low, loc = text.lower(), locator.lower().strip()
    at = low.find(loc)
    score = 1.0
    if at < 0:
        # fall back to the best-matching line
        lines = text.split("\n")
        best_i, score = 0, 0.0
        for i, line in enumerate(lines):
            s = overlap(locator, line)
            if s > score:
                best_i, score = i, s
        if score < 0.3:
            return text[:width], score
        at = sum(len(l) + 1 for l in lines[:best_i])

    half = max(width // 2, 60)
    start = max(0, at - half)
    end = min(len(text), at + len(loc) + half)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix, score


def _chunk_span(chunk: Chunk, locator: str, width: int) -> Span:
    """Prefer a table row when the locator looks like one."""
    for table in chunk.tables:
        rows = table.as_rows()
        if any(overlap(locator, r) >= 0.45 for r in rows) or (
                locator and any(overlap(locator, h) >= 0.6
                                for h in table.header)):
            body = _table_span(table, locator)
            return Span(source="policy text",
                        ref=f"chunk {chunk.index} | {chunk.doc_title}",
                        text=body or "", match=1.0)
    body, score = _window(chunk.text, locator, width)
    return Span(source="policy text",
                ref=f"chunk {chunk.index} | {chunk.doc_title}",
                text=body, match=score)


def semantic_spans(ix: Interaction, locator: str, *, embedder,
                   width: int = 480, limit: int = 2) -> List[Span]:
    """Locate a passage by meaning. A FALLBACK, never the first attempt.

    Used only when the lexical locator cannot find the pointer, because
    the cost is an embedding call per row and it is worth paying only on
    the rows where lexical actually failed.

    Locating by meaning is safe; VERIFYING by meaning is not, and
    grounding stays lexical for that reason - see evidence/index.py.
    """
    from .index import PassageIndex, passages_from_chunks

    passages = passages_from_chunks(ix.chunks)
    if not passages or not str(locator or "").strip():
        return []
    index = PassageIndex.build(passages, embedder)
    hits = index.search(embedder.embed_one(locator), k=limit)
    return [Span(source="policy text",
                 ref=f"chunk {chunk_index} | {label}",
                 text=text[:width], match=score)
            for score, chunk_index, label, text in hits]


def extract_spans(ix: Interaction, pointers: Sequence[Pointer], *,
                  width: int = 480, limit: int = 3,
                  embedder=None, semantic_floor: float = 0.3) -> List[Span]:
    """Turn pointers into passages the adjudicator can read.

    Lexical first. If an embedder is supplied and lexical located
    nothing, fall back to meaning - which is the right order, since a
    verbatim hit is both cheaper and stronger evidence that the screener
    was pointing somewhere real.
    """
    spans: List[Span] = []
    for pointer in pointers:
        if len(spans) >= limit:
            break
        if pointer.source == "chunks":
            if not ix.chunks:
                continue
            if pointer.chunk_index is not None and \
                    0 <= pointer.chunk_index < len(ix.chunks):
                spans.append(_chunk_span(ix.chunks[pointer.chunk_index],
                                         pointer.locator, width))
                continue
            # no usable index: take the best-matching chunk
            best, best_score = None, 0.0
            for chunk in ix.chunks:
                score = overlap(pointer.locator, chunk.text)
                if score > best_score:
                    best, best_score = chunk, score
            if best is not None:
                spans.append(_chunk_span(best, pointer.locator, width))
        elif pointer.source == "response":
            body, score = _window(ix.response, pointer.locator, width)
            spans.append(Span("the response", "response", body, score))
        elif pointer.source == "request":
            spans.append(Span("the request", "this turn", ix.request, 1.0))
        elif pointer.source == "history":
            body, score = _window(ix.history_text, pointer.locator, width)
            spans.append(Span("history", "earlier turns", body, score))

    if embedder is not None and ix.chunks and not any(
            s.match >= semantic_floor for s in spans):
        for pointer in pointers:
            if pointer.source != "chunks" or not pointer.locator:
                continue
            found = semantic_spans(ix, pointer.locator, embedder=embedder,
                                   width=width, limit=limit)
            if found:
                return found
    return spans


def fallback_spans(ix: Interaction, check_quote_from: Sequence[str], *,
                   width: int = 480, limit: int = 3) -> List[Span]:
    """What to send when the screener pointed at nothing usable.

    Still bounded: the first chunks truncated to the window, not the whole
    retrieval. A screener that gave no pointer has not earned a full read,
    and the adjudicator is told the pointer was missing so it can weigh
    that.
    """
    spans: List[Span] = []
    if "chunks" in check_quote_from:
        for chunk in ix.chunks[:limit]:
            spans.append(Span("policy text",
                              f"chunk {chunk.index} | {chunk.doc_title}",
                              chunk.text[:width], 0.0))
    if "history" in check_quote_from and ix.history and len(spans) < limit:
        spans.append(Span("history", "earlier turns",
                          ix.history_text[-width:], 0.0))
    return spans[:limit]


def render_spans(spans: Sequence[Span]) -> str:
    if not spans:
        return ""
    return "\n\n".join(s.render() for s in spans)
