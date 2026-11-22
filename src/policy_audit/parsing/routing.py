"""Which of the five things did the agent do? Decided in code.

The route decides which checks can apply at all, so it must not be a
judgement call. A model asked "did it refuse?" will read any cautious
phrasing as a refusal; the response either carries the clarification
template or a known static reply, or it does not.

Fill STATIC_RESPONSES and the marker lists in config with the live
wording. `route_diagnostics` reports the distribution before a run, so a
wrong list shows up as an implausible count rather than as a silently
mislabelled corpus.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Sequence

from ..domain.models import Clarification
from ..domain.taxonomy import Route


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ",
                                      str(text).lower())).strip()


# Phrases that mark a refusal to ACT, as opposed to a refusal to answer.
# These are the "I am here to assist with policy documents" and "I cannot
# take that action" families.
ACTION_REFUSAL = re.compile(
    r"\b(?:cannot|can't|unable to|not able to|do not have the ability to)\b"
    r"[^.]{0,60}\b(?:perform|take|execute|carry out|submit|approve|process|"
    r"complete|action|make (?:any )?changes|update|cancel|book|raise)\b"
    r"|\bI (?:am|'m) not able to (?:do|perform|take)\b"
    r"|\bcould directly determine\b", re.I)

SCOPE_REFUSAL = re.compile(
    r"\bI (?:am|'m) here to (?:assist|help)\b"
    r"|\bI can only (?:assist|help|answer)\b"
    r"|\b(?:outside|beyond) (?:my|the) (?:scope|remit|knowledge)\b"
    r"|\bnot (?:able|equipped) to (?:assist|help|answer)\b"
    r"|\bI (?:do not|don't) have (?:access to|information on)\b", re.I)


def detect_route(response: str, clarification: Clarification,
                 static_responses: Sequence[str] = (),
                 has_marker: bool = False) -> Route:
    """Classify the response. Order matters, and clarification is first.

    The clarify marker outranks everything, including a static match. A
    sentence beginning "please clarify" means the clarification flow has
    started, and a canned phrase appearing somewhere in its preamble must
    not reclassify the row - static matching is a substring test, so a
    generic entry like "I am here to assist with policy documents" will
    otherwise swallow clarifications whole.

    It also does not depend on the body parsing. A clarification whose
    questions came back unreadable is still a clarification; calling it
    `answered` would turn every answer-shaped check loose on a response
    that answered nothing.
    """
    text = str(response or "").strip()
    if not text:
        return Route.STATIC

    if has_marker or clarification.is_clarification:
        return Route.CLARIFIED

    norm = _norm(text)
    for canned in static_responses:
        canned_norm = _norm(canned)
        if canned_norm and (norm == canned_norm or canned_norm in norm):
            return Route.STATIC

    if ACTION_REFUSAL.search(text):
        return Route.REFUSED_ACTION
    if SCOPE_REFUSAL.search(text):
        return Route.REFUSED_SCOPE

    return Route.ANSWERED


def route_diagnostics(routes: Sequence[Route]) -> Dict[str, object]:
    """Distribution plus the warnings worth seeing before spending tokens."""
    counts = Counter(r.value for r in routes)
    total = max(len(routes), 1)
    warnings = []
    if not counts.get(Route.CLARIFIED.value):
        warnings.append(
            "no row classified as clarified - if the agent does ask "
            "clarifying questions, config.clarify_markers does not match "
            "the live template, and every clarification check will be "
            "skipped")
    if counts.get(Route.ANSWERED.value, 0) / total > 0.97:
        warnings.append(
            "almost every row classified as answered - the refusal and "
            "static patterns are probably not matching the live wording")
    if counts.get(Route.STATIC.value, 0) / total > 0.5:
        warnings.append(
            "more than half of rows classified as static - check "
            "config.static_responses for an over-broad entry")
    return {"counts": dict(counts), "total": len(routes),
            "warnings": warnings}
