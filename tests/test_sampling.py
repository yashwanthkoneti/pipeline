"""The review sample. This is the only thing that measures the misses.

A defect the audit cannot substantiate is recorded as expected
behaviour, by design. That makes the queue trustworthy and makes every
miss silent, so these properties are what stop the silence being total:

  every stratum of clean rows is represented
  small strata are not rounded away, since they carry the most doubt
  a defect row is never sampled - it is already in the queue
  the draw is stable across reruns
"""

from policy_audit.decision.sampling import (STRATUM_ORDER, Stratum, classify,
                                            sample_report,
                                            stratified_sample)
from policy_audit.domain.models import AuditResult, Finding, ScreenResult
from policy_audit.domain.taxonomy import (Adjudication, Route, Suspicion,
                                          Theme)


def clean(tid, *, errors=(), unproven=(), screened=(), findings=()):
    return AuditResult(
        transaction_id=tid, route=Route.ANSWERED, findings=tuple(findings),
        themes=(), factors=(), expected_behaviour=True,
        expected_reason="x", screened=tuple(screened),
        unproven=tuple(unproven), errors=tuple(errors))


def defect(tid):
    return AuditResult(
        transaction_id=tid, route=Route.ANSWERED,
        findings=(Finding("CONTRADICTS_POLICY", Adjudication.CONFIRMED,
                          Theme.INCORRECT, (), None),),
        themes=(Theme.INCORRECT,), factors=(), expected_behaviour=False)


def test_strata_are_assigned_by_highest_doubt_first():
    unsure = ScreenResult("X", Suspicion.UNSURE)
    assert classify(clean("a", errors=("boom",), unproven=("X",)),
                    has_retrieval=True) is Stratum.ERRORED
    assert classify(clean("b", unproven=("X",)),
                    has_retrieval=True) is Stratum.UNVERIFIABLE
    assert classify(clean("c"), has_retrieval=False) is Stratum.NO_EVIDENCE
    assert classify(clean("d", screened=(unsure,)),
                    has_retrieval=True) is Stratum.UNSURE_SCREEN
    assert classify(clean("e", findings=(
        Finding("X", Adjudication.REJECTED, None, (), None),)),
        has_retrieval=True) is Stratum.CLEARED
    assert classify(clean("f"), has_retrieval=True) is Stratum.CLEAN


def test_every_stratum_is_represented_even_when_tiny():
    """The small strata are the doubtful ones; 10% would round them away."""
    results = [clean(f"clean{i}") for i in range(200)]
    results += [clean(f"unv{i}", unproven=("X",)) for i in range(5)]
    results += [clean(f"err{i}", errors=("boom",)) for i in range(3)]
    retrieval = {r.transaction_id: True for r in results}

    picks = stratified_sample(results, retrieval, rate=0.10,
                              min_per_stratum=2)
    report = sample_report(picks)
    by = {row["stratum"]: row for row in report["by_stratum"]}

    assert by["clean"]["sampled"] == 20
    assert by["unverifiable"]["sampled"] == 2     # floor, not 0
    assert by["errored"]["sampled"] == 2
    assert report["sampled"] == 24


def test_defect_rows_are_never_sampled():
    results = [defect("d1"), defect("d2"), clean("c1"), clean("c2")]
    picks = stratified_sample(results, {r.transaction_id: True
                                        for r in results}, rate=1.0)
    assert "d1" not in picks
    assert "d2" not in picks
    assert set(picks) == {"c1", "c2"}


def test_draw_is_stable_across_reruns():
    results = [clean(f"t{i}") for i in range(60)]
    retrieval = {r.transaction_id: True for r in results}
    first = {k for k, v in stratified_sample(results, retrieval).items()
             if v.in_sample}
    shuffled = list(reversed(results))
    second = {k for k, v in stratified_sample(shuffled, retrieval).items()
              if v.in_sample}
    assert first == second
    assert first                                   # and not empty


def test_rate_zero_still_honours_the_floor_or_samples_nothing():
    results = [clean(f"t{i}") for i in range(40)]
    retrieval = {r.transaction_id: True for r in results}
    picks = stratified_sample(results, retrieval, rate=0.0,
                              min_per_stratum=0)
    assert not any(v.in_sample for v in picks.values())


def test_cap_limits_a_dominant_stratum():
    results = [clean(f"t{i}") for i in range(500)]
    retrieval = {r.transaction_id: True for r in results}
    picks = stratified_sample(results, retrieval, rate=0.10,
                              max_per_stratum=15)
    assert sum(1 for v in picks.values() if v.in_sample) == 15


def test_sampled_rows_carry_why_they_were_clean():
    results = [clean(f"u{i}", unproven=("X",)) for i in range(10)]
    picks = stratified_sample(results, {r.transaction_id: True
                                        for r in results}, rate=0.5)
    reasons = [v.reason for v in picks.values() if v.in_sample]
    assert reasons
    assert all("could not be substantiated" in r for r in reasons)


def test_strata_order_runs_from_most_to_least_doubtful():
    assert STRATUM_ORDER[0] is Stratum.ERRORED
    assert STRATUM_ORDER[-1] is Stratum.CLEAN
