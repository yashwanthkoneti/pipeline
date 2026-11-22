"""The cheap agent: four gates, recall-tuned.

It does the heavy lifting - one pass per gate over every row - and it is
allowed to be wrong in one direction only. A "suspect" it cannot justify
costs one adjudication call. A "clear" it should not have given is never
looked at again, so CLEAR is the only verdict that can hide a defect, and
the prompt says so.

Gates run in order and can stop the audit. A correctly refused request
never reaches the answer gate, so no answer-shaped check can attach
itself to a row where no answer was given - which is how
"answer_incomplete" used to end up as the cause of a clarification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from ..domain.checks import Check, checks_for
from ..domain.models import GateOutcome, Interaction, Pointer, ScreenResult
from ..domain.taxonomy import CALL_PLAN, Gate, Route, Suspicion
from .client import JsonAgent
from .prompts import screener_prompt

_VERDICTS = {
    "clear": Suspicion.CLEAR,
    "suspect": Suspicion.SUSPECT,
    "unsure": Suspicion.UNSURE,
}


@dataclass
class ScreenReport:
    facts: Dict[str, Any] = field(default_factory=dict)
    outcomes: List[GateOutcome] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    calls: int = 0

    @property
    def results(self) -> Tuple[ScreenResult, ...]:
        return tuple(r for o in self.outcomes for r in o.results)

    @property
    def escalations(self) -> Tuple[ScreenResult, ...]:
        return tuple(r for r in self.results if r.escalates)


def _pointers(raw: Any) -> Tuple[Pointer, ...]:
    out: List[Pointer] = []
    if not isinstance(raw, list):
        return ()
    for item in raw[:4]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "chunks").strip().lower()
        if source not in ("chunks", "response", "request", "history"):
            source = "chunks"
        chunk = item.get("chunk")
        try:
            chunk_index = int(chunk) if chunk is not None else None
        except (TypeError, ValueError):
            chunk_index = None
        out.append(Pointer(source=source, chunk_index=chunk_index,
                           locator=str(item.get("locator") or "").strip()))
    return tuple(out)


def _parse_checks(payload: Any, checks: Sequence[Check]) -> List[ScreenResult]:
    """Map the reply onto the checks we asked about.

    A check the model omitted becomes UNSURE rather than CLEAR. Silence is
    not evidence of absence, and defaulting it to clear would let a
    truncated reply quietly clear a row.
    """
    block = payload.get("checks") if isinstance(payload, dict) else None
    block = block if isinstance(block, dict) else {}
    results: List[ScreenResult] = []
    for check in checks:
        entry = block.get(check.id)
        if not isinstance(entry, dict):
            results.append(ScreenResult(
                check_id=check.id, suspicion=Suspicion.UNSURE,
                reason="the screener did not answer this check"))
            continue
        verdict = _VERDICTS.get(
            str(entry.get("verdict") or "").strip().lower(), Suspicion.UNSURE)
        results.append(ScreenResult(
            check_id=check.id,
            suspicion=verdict,
            reason=str(entry.get("reason") or "").strip()[:400],
            pointers=_pointers(entry.get("where")),
        ))
    return results


class Screener:
    def __init__(self, agent: JsonAgent) -> None:
        self.agent = agent

    def _ask(self, ix: Interaction, gates: Sequence[Gate],
             checks: Sequence[Check]) -> Tuple[Dict[str, Any], List[str]]:
        label = "+".join(g.value for g in gates)
        system, user = screener_prompt(ix, gates, checks)
        expect = ["checks"] if checks else None
        payload = self.agent.ask(system, user, expect=expect)
        errors = []
        if "_error" in payload:
            errors.append(f"{label}: {payload['_error']}")
        if "_partial" in payload:
            errors.append(f"{label}: reply incomplete "
                          f"({payload['_partial']})")
        return payload, errors

    def run(self, ix: Interaction) -> ScreenReport:
        report = ScreenReport()

        for batch in CALL_PLAN:
            # The answer gate is about an answer. On a route where none was
            # given there is nothing for it to examine, and asking anyway
            # invites invented findings.
            live = []
            for gate in batch:
                if gate is Gate.ANSWER and ix.route is not Route.ANSWERED:
                    report.outcomes.append(GateOutcome(
                        gate=gate, results=(), stop=True,
                        note=f"answer gate skipped: route is "
                             f"{ix.route.value}, so there is no answer to "
                             f"examine"))
                    continue
                live.append(gate)
            if not live:
                continue

            per_gate = {g: checks_for(g, ix.route) for g in live}
            checks = [c for g in live for c in per_gate[g]]
            if not checks and Gate.REQUEST not in live:
                for gate in live:
                    report.outcomes.append(GateOutcome(
                        gate=gate, results=(),
                        note=f"no checks apply at {gate.value} for route "
                             f"{ix.route.value}"))
                continue

            payload, errors = self._ask(ix, live, checks)
            report.calls += 1
            report.errors.extend(errors)

            if Gate.REQUEST in live:
                facts = payload.get("facts")
                report.facts = facts if isinstance(facts, dict) else {
                    k: v for k, v in payload.items()
                    if not str(k).startswith("_")}

            parsed = _parse_checks(payload, checks) if checks else []
            by_id = {r.check_id: r for r in parsed}
            for gate in live:
                report.outcomes.append(GateOutcome(
                    gate=gate,
                    results=tuple(by_id[c.id] for c in per_gate[gate]
                                  if c.id in by_id)))

        return report
