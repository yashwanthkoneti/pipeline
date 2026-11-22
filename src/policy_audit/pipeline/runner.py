"""Orchestration: screen, escalate, adjudicate, record.

The cascade in one place:

    prepare (deterministic)
      -> screener, 4 gates, every row
        -> escalations only
          -> adjudicator, narrow spans
            -> verdict, expected behaviour by default

Checkpointing is per row to a JSONL file, so a 500-row run that dies at
row 380 resumes at 380. That was not optional in practice - a run long
enough to matter is long enough to be interrupted.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence

from ..agents.adjudicator import Adjudicator
from ..agents.client import JsonAgent, LLM
from ..agents.screener import Screener
from ..config import Settings
from ..decision.verdict import build_result, summarise
from ..domain.models import AuditResult, Interaction
from ..domain.taxonomy import Adjudication


def _serialise(result: AuditResult) -> Dict[str, object]:
    return {
        "transaction_id": result.transaction_id,
        "route": result.route.value,
        "expected_behaviour": result.expected_behaviour,
        "expected_reason": result.expected_reason,
        "themes": [t.value for t in result.themes],
        "factors": [f.value for f in result.factors],
        "unproven": list(result.unproven),
        "gate_notes": list(result.gate_notes),
        "errors": list(result.errors),
        "screener_calls": result.screener_calls,
        "adjudicator_calls": result.adjudicator_calls,
        "screened": [
            {"check": s.check_id, "verdict": s.suspicion.value,
             "reason": s.reason,
             "where": [asdict(p) for p in s.pointers]}
            for s in result.screened
        ],
        "findings": [
            {"check": f.check_id,
             "decision": f.adjudication.value,
             "theme": f.theme.value if f.theme else None,
             "factors": [x.value for x in f.factors],
             "overturned_screener": f.overturned_screener,
             "grounded": f.grounded,
             "rationale": f.rationale,
             "evidence": (asdict(f.evidence) if f.evidence else None)}
            for f in result.findings
        ],
    }


def read_checkpoint_ids(path: str) -> set:
    """Just the transaction ids, without rebuilding every result.

    Read before preparation so finished rows can be skipped, which is
    what makes a resumed run cheap rather than merely correct.
    """
    ids = set()
    if not path or not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                tid = json.loads(line).get("transaction_id")
            except Exception:
                continue            # a half-written final line
            if tid:
                ids.add(str(tid))
    return ids


class Runner:
    def __init__(self, screener_llm: LLM, adjudicator_llm: LLM,
                 settings: Settings) -> None:
        self.settings = settings
        self.screener = Screener(JsonAgent(
            screener_llm, settings.screener_max_tokens,
            max_retries=settings.max_retries,
            retry_base_delay=settings.retry_base_delay,
            min_seconds_between_calls=settings.min_seconds_between_calls,
            name="screener"))
        embedder = None
        if settings.semantic_spans:
            from ..evidence.index import Embedder, have_faiss
            embedder = Embedder(settings.embedding_model,
                                base_url=settings.openai_base_url or None)
            print(f"semantic span location on "
                  f"({'faiss' if have_faiss() else 'numpy'} backend, "
                  f"{settings.embedding_model})")
        self.adjudicator = Adjudicator(JsonAgent(
            adjudicator_llm, settings.adjudicator_max_tokens,
            max_retries=settings.max_retries,
            retry_base_delay=settings.retry_base_delay,
            min_seconds_between_calls=settings.min_seconds_between_calls,
            name="adjudicator"), settings, embedder=embedder)

    # ---- one row ----------------------------------------------------
    def audit(self, ix: Interaction) -> AuditResult:
        report = self.screener.run(ix)
        escalations = report.escalations

        findings = []
        adjudicator_calls = 0
        if escalations and not self.settings.dry_run:
            findings = self.adjudicator.adjudicate_all(ix, escalations)
            adjudicator_calls = len(escalations)

        return build_result(
            ix, report.results, findings,
            gate_notes=[o.note for o in report.outcomes if o.note],
            errors=report.errors,
            screener_calls=report.calls,
            adjudicator_calls=adjudicator_calls,
        )

    # ---- a run ------------------------------------------------------
    def _load_checkpoint(self) -> Dict[str, Dict[str, object]]:
        path = self.settings.checkpoint_path
        if not path or not os.path.exists(path):
            return {}
        done: Dict[str, Dict[str, object]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue          # a half-written final line
                tid = row.get("transaction_id")
                if tid:
                    done[str(tid)] = row
        return done

    def _append_checkpoint(self, payload: Dict[str, object]) -> None:
        path = self.settings.checkpoint_path
        if not path:
            return
        # A dry run screens but never adjudicates, so its rows are
        # measurements, not verdicts. Recording them would make the next
        # real run skip every one of them as already audited and return
        # the same empty result - which looks exactly like the real run
        # having found nothing.
        if self.settings.dry_run:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def run(self, interactions: Sequence[Interaction], *,
            resume: bool = True,
            on_row=None) -> List[AuditResult]:
        if self.settings.dry_run:
            print("dry run: screening only, the adjudicator will not be "
                  "called and nothing will be checkpointed")
        done = self._load_checkpoint() if resume else {}
        if done:
            print(f"resuming: {len(done)} row(s) already audited")

        results: List[AuditResult] = []
        rows = list(interactions)
        if self.settings.limit:
            rows = rows[:self.settings.limit]

        for i, ix in enumerate(rows, 1):
            if ix.transaction_id in done:
                continue
            result = self.audit(ix)
            results.append(result)
            self._append_checkpoint(_serialise(result))
            if on_row:
                on_row(i, len(rows), result)
            elif i % self.settings.batch_size == 0:
                print(f"  {i}/{len(rows)} rows")
        return results

    @property
    def call_stats(self) -> Dict[str, object]:
        return {
            "screener": vars(self.screener.agent.stats),
            "adjudicator": vars(self.adjudicator.agent.stats),
        }
