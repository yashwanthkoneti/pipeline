"""The costly agent: reads the passages, decides, and its word is final.

It runs only on what the screener escalated, so it can be the large
model. It sees narrow spans rather than the whole retrieval, which is
both what makes that affordable and what makes its quotes checkable.

It is free to overturn the screener, and the prompt tells it that
rejecting is the expected outcome - otherwise a model asked to
adjudicate its colleague's suspicion tends to ratify it.

Two things happen after it answers, in code, because they are the errors
that survive instruction:

  the evidence must be READABLE - two complete sentences, no fragments,
  no bare table cells. Unreadable evidence means a reviewer has to redo
  the work, which is the cost this whole design exists to remove.

  the quote must be GROUNDED in the source it claims. A confirmation
  citing the user's own words as policy is dropped, not softened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import Settings
from ..domain.checks import Check, CHECKS_BY_ID
from ..domain.models import (Evidence, Finding, Interaction, ScreenResult)
from ..domain.taxonomy import Adjudication, Factor
from ..evidence.grounding import check_grounding
from ..evidence.spans import extract_spans, fallback_spans, render_spans
from .client import JsonAgent
from .prompts import adjudicator_prompt

_DECISIONS = {
    "confirmed": Adjudication.CONFIRMED,
    "rejected": Adjudication.REJECTED,
    "unproven": Adjudication.UNPROVEN,
}


def _factor_for(check: Check, raw: object) -> Tuple[Factor, ...]:
    """The factor the adjudicator picked, if it is one this check offers.

    When it picks nothing usable and the check offers several, the honest
    answer is that we do not know which. Attributing all of them inflates
    every one of their counts from a single row, so the fallback factor
    carries it instead - and a large count there is a signal that the
    adjudicator is not being asked the question clearly enough.
    """
    name = str(raw or "").strip().lower()
    for factor in check.factors:
        if factor.value == name:
            return (factor,)
    if len(check.factors) == 1:
        return check.factors
    return (Factor.PROMPT_OR_ROUTING,)


class Adjudicator:
    def __init__(self, agent: JsonAgent, settings: Settings,
                 embedder=None) -> None:
        self.agent = agent
        self.settings = settings
        # supplied only when semantic_spans is on; span extraction tries
        # lexical first either way
        self.embedder = embedder

    def _spans(self, ix: Interaction, result: ScreenResult,
               check: Check) -> Tuple[str, bool]:
        spans = extract_spans(ix, result.pointers,
                              width=self.settings.span_window_chars,
                              limit=self.settings.max_spans_per_check,
                              embedder=self.embedder)
        usable = [s for s in spans
                  if s.text and s.match >= self.settings.span_min_match]
        if usable:
            return render_spans(usable), True
        if spans:
            return render_spans(spans), False
        return render_spans(fallback_spans(
            ix, check.quote_from,
            width=self.settings.span_window_chars,
            limit=self.settings.max_spans_per_check)), False

    def adjudicate(self, ix: Interaction,
                   result: ScreenResult) -> Optional[Finding]:
        check = CHECKS_BY_ID.get(result.check_id)
        if check is None:
            return None

        spans, located = self._spans(ix, result, check)
        system, user = adjudicator_prompt(ix, check, result.reason, spans)
        payload = self.agent.ask(system, user, expect=["decision"])

        if "_error" in payload:
            # An adjudication we could not obtain is NOT a defect. Silence
            # from the expensive model must not become an accusation.
            return Finding(
                check_id=check.id, adjudication=Adjudication.UNPROVEN,
                theme=None, factors=(), evidence=None,
                rationale=f"not adjudicated: {payload['_error']}")

        decision = _DECISIONS.get(
            str(payload.get("decision") or "").strip().lower(),
            Adjudication.UNPROVEN)
        rationale = str(payload.get("rationale") or "").strip()[:500]

        if decision is Adjudication.REJECTED:
            return Finding(
                check_id=check.id, adjudication=Adjudication.REJECTED,
                theme=None, factors=(), evidence=None,
                rationale=rationale or "rejected without a reason given",
                overturned_screener=True)

        evidence = Evidence(
            answer_claims=str(payload.get("answer_claims") or "").strip(),
            source_states=str(payload.get("source_states") or "").strip(),
            source_ref=str(payload.get("source_ref") or "").strip(),
            quote=str(payload.get("quote") or "").strip(),
        )

        if decision is Adjudication.UNPROVEN:
            return Finding(
                check_id=check.id, adjudication=Adjudication.UNPROVEN,
                theme=None, factors=(), evidence=evidence,
                rationale=rationale)

        # ---- confirmed: now earn it --------------------------------
        if self.settings.require_readable_evidence and \
                not evidence.readable():
            return Finding(
                check_id=check.id, adjudication=Adjudication.UNPROVEN,
                theme=None, factors=(), evidence=evidence,
                rationale=(
                    "confirmed but downgraded: the evidence is not two "
                    "complete statements, so a reviewer could not judge it "
                    "without reopening the trace"))

        needs_policy = "chunks" in check.quote_from
        grounding = check_grounding(
            evidence, ix, floor=self.settings.span_min_match,
            needs_policy_text=needs_policy)

        if not grounding.grounded and self.settings.strict_grounding:
            return Finding(
                check_id=check.id, adjudication=Adjudication.UNPROVEN,
                theme=None, factors=(), evidence=evidence,
                grounded=False,
                rationale=f"confirmed but downgraded: {grounding.note}")

        note = rationale
        if not located:
            note = (note + " | note: the screener gave no usable pointer, "
                           "so this was adjudicated on a bounded fallback "
                           "passage").strip(" |")

        return Finding(
            check_id=check.id, adjudication=Adjudication.CONFIRMED,
            theme=check.theme,
            factors=_factor_for(check, payload.get("factor")),
            evidence=evidence, rationale=note,
            grounded=grounding.grounded)

    def adjudicate_all(self, ix: Interaction,
                       escalations: Sequence[ScreenResult]) -> List[Finding]:
        findings: List[Finding] = []
        for result in escalations:
            finding = self.adjudicate(ix, result)
            if finding is not None:
                findings.append(finding)
        return findings
