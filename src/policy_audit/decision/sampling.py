"""A stratified sample of the expected-behaviour rows, for review.

This exists because of a deliberate asymmetry elsewhere: a defect the
audit cannot substantiate is recorded as expected behaviour. That keeps
the review queue trustworthy, and it means every miss leaves the system
silently. Without a sample there is no way to find out how often.

Why stratified rather than random. "Expected behaviour" is not one thing.
A row where the screener saw nothing at all and a row where a suspicion
was raised and could not be settled are both clean in the output and
carry completely different amounts of doubt. A flat 10% sample spends
most of its budget on the first kind, which is where misses are rarest.
Stratifying lets you measure a miss rate PER STRATUM, so the number you
get back is actionable: "of the rows we cleared because nothing could be
substantiated, a third were real defects" tells you what to fix.

The strata are ordered by doubt, highest first, and a row lands in the
first one that applies.

Selection is deterministic - a hash of the transaction id - so the same
rows are drawn on every rerun. A review queue that reshuffles when the
pipeline is rerun cannot be worked through.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple

from ..domain.models import AuditResult
from ..domain.taxonomy import Suspicion


class Stratum(str, Enum):
    """Ordered by how much doubt the clean verdict carries."""

    ERRORED = "errored"               # the audit itself failed on this row
    UNVERIFIABLE = "unverifiable"     # suspected, could not be settled
    NO_EVIDENCE = "no_evidence"       # nothing retrieved to check against
    UNSURE_SCREEN = "unsure_screen"   # the screener could not tell
    CLEARED = "cleared"               # adjudicator examined and rejected
    CLEAN = "clean"                   # the screener saw nothing


STRATUM_ORDER: Tuple[Stratum, ...] = (
    Stratum.ERRORED, Stratum.UNVERIFIABLE, Stratum.NO_EVIDENCE,
    Stratum.UNSURE_SCREEN, Stratum.CLEARED, Stratum.CLEAN,
)

STRATUM_MEANING: Dict[Stratum, str] = {
    Stratum.ERRORED:
        "the audit errored on this row, so it was never really judged",
    Stratum.UNVERIFIABLE:
        "something was suspected and could not be substantiated from the "
        "text",
    Stratum.NO_EVIDENCE:
        "clean, but nothing was retrieved, so most checks had nothing to "
        "verify against",
    Stratum.UNSURE_SCREEN:
        "the screener could not tell on at least one check",
    Stratum.CLEARED:
        "a suspicion was examined on the evidence and rejected",
    Stratum.CLEAN:
        "the screener found nothing to look at",
}


def classify(result: AuditResult, *, has_retrieval: bool) -> Stratum:
    """Which stratum this clean row belongs to. First match wins."""
    if result.errors:
        return Stratum.ERRORED
    if result.unproven:
        return Stratum.UNVERIFIABLE
    if not has_retrieval:
        return Stratum.NO_EVIDENCE
    if any(s.suspicion is Suspicion.UNSURE for s in result.screened):
        return Stratum.UNSURE_SCREEN
    if result.findings:
        return Stratum.CLEARED
    return Stratum.CLEAN


def _draw(transaction_id: str) -> float:
    """A stable number in [0, 1) from the id. Same rows every rerun."""
    digest = hashlib.sha256(str(transaction_id).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(1 << 48)


@dataclass(frozen=True)
class SampleDecision:
    in_sample: bool
    stratum: Stratum
    reason: str = ""


def stratified_sample(results: Sequence[AuditResult],
                      has_retrieval: Dict[str, bool], *,
                      rate: float = 0.10,
                      min_per_stratum: int = 2,
                      max_per_stratum: int = 0,
                      ) -> Dict[str, SampleDecision]:
    """Pick ~`rate` of each stratum of the expected-behaviour rows.

    min_per_stratum matters more than it looks. At 10% a stratum holding
    six rows draws nothing, and the strata carrying the most doubt are
    usually the small ones - which is exactly backwards. The floor makes
    every non-empty stratum measurable; it will push the overall rate
    above the nominal one on small runs, and that is the right trade.

    max_per_stratum caps a dominant stratum so the queue does not fill up
    with the kind of row that is least likely to be interesting. 0 = no
    cap.
    """
    buckets: Dict[Stratum, List[AuditResult]] = defaultdict(list)
    for result in results:
        if not result.expected_behaviour:
            continue
        stratum = classify(
            result, has_retrieval=has_retrieval.get(result.transaction_id,
                                                    False))
        buckets[stratum].append(result)

    decisions: Dict[str, SampleDecision] = {}
    for stratum in STRATUM_ORDER:
        rows = buckets.get(stratum, [])
        if not rows:
            continue
        target = max(int(round(len(rows) * rate)),
                     min(min_per_stratum, len(rows)))
        if max_per_stratum:
            target = min(target, max_per_stratum)
        # deterministic: lowest draw wins, so the set is stable and does
        # not depend on row order
        ranked = sorted(rows, key=lambda r: _draw(r.transaction_id))
        chosen = {r.transaction_id for r in ranked[:target]}
        for result in rows:
            picked = result.transaction_id in chosen
            decisions[result.transaction_id] = SampleDecision(
                in_sample=picked,
                stratum=stratum,
                reason=(f"{stratum.value}: {STRATUM_MEANING[stratum]}"
                        if picked else ""),
            )
    return decisions


def sample_report(decisions: Dict[str, SampleDecision]) -> Dict[str, object]:
    """Per-stratum counts, for the run summary."""
    totals: Dict[str, int] = defaultdict(int)
    picked: Dict[str, int] = defaultdict(int)
    for decision in decisions.values():
        totals[decision.stratum.value] += 1
        if decision.in_sample:
            picked[decision.stratum.value] += 1

    rows = []
    for stratum in STRATUM_ORDER:
        name = stratum.value
        if not totals.get(name):
            continue
        rows.append({
            "stratum": name,
            "expected_rows": totals[name],
            "sampled": picked[name],
            "rate_pct": round(100 * picked[name] / totals[name], 1),
            "why_it_was_clean": STRATUM_MEANING[stratum],
        })
    return {
        "expected_rows": sum(totals.values()),
        "sampled": sum(picked.values()),
        "overall_rate_pct": round(
            100 * sum(picked.values()) / max(sum(totals.values()), 1), 1),
        "by_stratum": rows,
    }
