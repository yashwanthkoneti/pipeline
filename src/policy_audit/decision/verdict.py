"""From findings to a verdict. Pure functions over confirmed findings.

Expected behaviour is the DEFAULT, not a case to be proved. A row is a
defect only when the adjudicator confirmed one with readable, grounded
evidence. Everything else - nothing suspected, suspicion rejected,
suspicion unproven, the adjudicator unreachable - comes out as expected
behaviour, with the reason recorded so the two are distinguishable later.

That ordering is the point. The agent being audited works from whatever
retrieval handed it and cannot choose its own corpus; a judge that
accuses it whenever the evidence is murky produces a queue nobody trusts
and wastes the time of the people who would have fixed the real defects.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from ..domain.models import AuditResult, Finding, Interaction, ScreenResult
from ..domain.taxonomy import (Adjudication, Factor,
                               FACTOR_DEFECT, FACTOR_OWNER,
                               Theme)

# Which theme leads when a row has several. Ordered by what a reader needs
# to know first: a wrong answer outranks a thin one, and both outrank the
# user simply not being served.
THEME_PRIORITY: Tuple[Theme, ...] = (
    Theme.INCORRECT,
    Theme.INCONSISTENT,
    Theme.UNSUBSTANTIATED,
    Theme.INCOMPLETE,
    Theme.IRRELEVANT,
)


def order_themes(themes: Sequence[Theme]) -> Tuple[Theme, ...]:
    seen = set()
    out = []
    for theme in THEME_PRIORITY:
        if theme in themes and theme not in seen:
            seen.add(theme)
            out.append(theme)
    return tuple(out)


def order_factors(factors: Sequence[Factor]) -> Tuple[Factor, ...]:
    seen = set()
    out = []
    for factor in factors:
        if factor not in seen:
            seen.add(factor)
            out.append(factor)
    return tuple(out)


def _expected_reason(screened: Sequence[ScreenResult],
                     findings: Sequence[Finding]) -> str:
    """Why this row came out clean. The distinction matters downstream.

    "nothing was suspected" and "we could not substantiate anything" both
    count as expected behaviour for the queue, but they are different
    facts about the audit, and reporting them as one number inflates the
    expected-behaviour share with rows we simply failed to decide.
    """
    if not screened:
        return "no check applied to this row"
    if not any(r.escalates for r in screened):
        return "the screener found nothing to look at"
    rejected = sum(1 for f in findings
                   if f.adjudication is Adjudication.REJECTED)
    unproven = sum(1 for f in findings
                   if f.adjudication is Adjudication.UNPROVEN)
    if rejected and not unproven:
        return (f"{rejected} suspicion(s) examined and rejected on the "
                f"evidence")
    if unproven and not rejected:
        return (f"{unproven} suspicion(s) could not be substantiated from "
                f"the text")
    return (f"{rejected} suspicion(s) rejected, {unproven} not "
            f"substantiated")


def build_result(ix: Interaction, screened: Sequence[ScreenResult],
                 findings: Sequence[Finding], *,
                 gate_notes: Sequence[str] = (),
                 errors: Sequence[str] = (),
                 screener_calls: int = 0,
                 adjudicator_calls: int = 0) -> AuditResult:
    defects = [f for f in findings if f.is_defect]
    themes = order_themes([f.theme for f in defects if f.theme])
    factors = order_factors([x for f in defects for x in f.factors])

    return AuditResult(
        transaction_id=ix.transaction_id,
        route=ix.route,
        findings=tuple(findings),
        themes=themes,
        factors=factors,
        expected_behaviour=not defects,
        expected_reason="" if defects else _expected_reason(screened,
                                                            findings),
        screened=tuple(screened),
        gate_notes=tuple(gate_notes),
        unproven=tuple(f.check_id for f in findings
                       if f.adjudication is Adjudication.UNPROVEN),
        screener_calls=screener_calls,
        adjudicator_calls=adjudicator_calls,
        errors=tuple(errors),
    )


# --------------------------------------------------------------------------
# run-level summary
# --------------------------------------------------------------------------
def defect_report(results: Sequence[AuditResult], *,
                  examples_per_factor: int = 2) -> List[Dict[str, object]]:
    """Per factor: what the defect IS, who owns it, and real instances.

    A count tells you how often; it does not tell you what to change. The
    evidence already holds two complete sentences per finding, so the
    example lines write themselves - and a reader who has never seen the
    data learns the defect from the report rather than from a tally.
    """
    by_factor: Dict[Factor, List[Dict[str, object]]] = {}
    for result in results:
        for finding in result.findings:
            if not finding.is_defect:
                continue
            for factor in (finding.factors or (Factor.PROMPT_OR_ROUTING,)):
                ev = finding.evidence
                by_factor.setdefault(factor, []).append({
                    "transaction_id": result.transaction_id,
                    "check": finding.check_id,
                    "theme": finding.theme.value if finding.theme else "",
                    "answer": ev.answer_claims if ev else "",
                    "source": ev.source_states if ev else "",
                    "ref": ev.source_ref if ev else "",
                })

    report = []
    for factor, rows in sorted(by_factor.items(), key=lambda kv: -len(kv[1])):
        report.append({
            "factor": factor.value,
            "owner": FACTOR_OWNER.get(factor, ""),
            "defect": FACTOR_DEFECT.get(factor, ""),
            "rows": len(rows),
            "examples": rows[:examples_per_factor],
            "others": [r["transaction_id"]
                       for r in rows[examples_per_factor:]],
        })
    return report


def unsubstantiated_report(results: Sequence[AuditResult]
                           ) -> List[Dict[str, object]]:
    """What the audit suspected and could not settle.

    This is the expected-behaviour population worth reading. "Could not
    be substantiated" is not the agent behaving well, it is the audit
    failing to decide, and the reason says which gate stopped it.
    """
    by_check: Dict[str, List[Dict[str, str]]] = {}
    for result in results:
        if not result.expected_behaviour:
            continue
        for finding in result.findings:
            if finding.adjudication is not Adjudication.UNPROVEN:
                continue
            by_check.setdefault(finding.check_id, []).append({
                "transaction_id": result.transaction_id,
                "why": finding.rationale,
            })
    return [{"check": check, "rows": len(rows),
             "why": rows[0]["why"] if rows else "",
             "transaction_ids": [r["transaction_id"] for r in rows[:5]]}
            for check, rows in sorted(by_check.items(),
                                      key=lambda kv: -len(kv[1]))]


def summarise(results: Sequence[AuditResult]) -> Dict[str, object]:
    """The numbers to look at before believing any of the others."""
    total = len(results) or 1
    expected = [r for r in results if r.expected_behaviour]
    defective = [r for r in results if not r.expected_behaviour]

    themes = Counter(t.value for r in defective for t in r.themes)
    factors = Counter(f.value for r in defective for f in r.factors)
    reasons = Counter(r.expected_reason for r in expected)

    screened = sum(len(r.screened) for r in results)
    escalated = sum(1 for r in results for s in r.screened if s.escalates)
    confirmed = sum(1 for r in results for f in r.findings if f.is_defect)
    rejected = sum(1 for r in results for f in r.findings
                   if f.adjudication is Adjudication.REJECTED)
    unproven = sum(1 for r in results for f in r.findings
                   if f.adjudication is Adjudication.UNPROVEN)

    warnings: List[str] = []
    if escalated and confirmed / max(escalated, 1) > 0.9:
        warnings.append(
            "the adjudicator confirmed almost everything the screener "
            "escalated. Either the screener is too conservative, or the "
            "adjudicator is ratifying rather than adjudicating - check a "
            "sample of confirmations by hand before trusting the rate.")
    if escalated and confirmed == 0:
        warnings.append(
            "nothing survived adjudication. Check whether evidence is "
            "being downgraded for grounding rather than genuinely "
            "rejected - the two mean very different things.")
    unsubstantiated = sum(v for k, v in reasons.items()
                          if "not be substantiated" in k)
    if unsubstantiated / total > 0.2:
        warnings.append(
            f"{unsubstantiated / total:.0%} of rows are expected behaviour "
            f"only because nothing could be substantiated. The "
            f"expected-behaviour share is that much softer than it looks; "
            f"quote it with that caveat.")

    return {
        "defect_report": defect_report(results),
        "unsubstantiated_report": unsubstantiated_report(results),
        "rows": len(results),
        "expected_behaviour": len(expected),
        "expected_behaviour_pct": round(100 * len(expected) / total, 1),
        "with_defects": len(defective),
        "themes": dict(themes.most_common()),
        "factors": dict(factors.most_common()),
        "expected_reasons": dict(reasons.most_common()),
        "checks_screened": screened,
        "escalated": escalated,
        "escalation_rate_pct": round(100 * escalated / max(screened, 1), 1),
        "confirmed": confirmed,
        "rejected_by_adjudicator": rejected,
        "unproven": unproven,
        "screener_calls": sum(r.screener_calls for r in results),
        "adjudicator_calls": sum(r.adjudicator_calls for r in results),
        "warnings": warnings,
    }
