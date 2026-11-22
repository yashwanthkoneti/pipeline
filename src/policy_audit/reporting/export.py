"""Two outputs, for two readers.

findings.json   nested, one object per row that has a defect, each
                finding carrying both complete statements. For reading.

findings.csv    one row per FINDING, not per interaction. For sorting,
                filtering and handing to whoever owns the factor.

A third, rows.csv, carries one row per interaction with the verdict, for
joining to whatever dashboard already exists.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Sequence

import pandas as pd

from ..config import SETTINGS, Settings
from ..decision.sampling import sample_report, stratified_sample
from ..decision.verdict import summarise
from ..domain.models import AuditResult, Interaction
from ..domain.taxonomy import FACTOR_OWNER, THEME_MEANING, Adjudication


def _evidence_fields(f) -> Dict[str, object]:
    """The four fields every output shares, from one place."""
    ev = f.evidence
    return {
        "answer_claims": ev.answer_claims if ev else "",
        "source_states": ev.source_states if ev else "",
        "source_ref": ev.source_ref if ev else "",
        "quote": ev.quote if ev else "",
    }


def _owners(factors) -> str:
    return "; ".join(sorted({FACTOR_OWNER.get(x, "") for x in factors}
                            - {""}))


def _finding_rows(result: AuditResult,
                  ix: Interaction | None) -> List[Dict[str, object]]:
    rows = []
    for f in result.findings:
        if not f.is_defect:
            continue
        row = {
            "transaction_id": result.transaction_id,
            "route": result.route.value,
            "theme": f.theme.value if f.theme else "",
            "theme_meaning": THEME_MEANING.get(f.theme, "") if f.theme else "",
            "factors": "; ".join(x.value for x in f.factors),
            "owners": _owners(f.factors),
            "check": f.check_id,
        }
        row.update(_evidence_fields(f))
        row.update({"grounded": f.grounded, "rationale": f.rationale,
                    "request": ix.request[:600] if ix else ""})
        rows.append(row)
    return rows


def _primary_finding(result: AuditResult):
    """The finding whose theme leads, so the main CSV reads without JSON."""
    for theme in [t for t in result.themes]:
        for f in result.findings:
            if f.is_defect and f.theme is theme:
                return f
    return next((f for f in result.findings if f.is_defect), None)


def export(results: Sequence[AuditResult],
           interactions: Sequence[Interaction],
           out_dir: str | None = None,
           settings: Settings = SETTINGS) -> Dict[str, object]:
    out_dir = out_dir or settings.output_dir
    os.makedirs(out_dir, exist_ok=True)
    by_id = {i.transaction_id: i for i in interactions}

    # A slice of the clean rows goes to review anyway. Without it, every
    # defect the audit could not substantiate leaves the system silently
    # and there is no way to find out how often that happens.
    sampled = stratified_sample(
        results,
        {i.transaction_id: bool(i.chunks) for i in interactions},
        rate=settings.sample_rate,
        min_per_stratum=settings.sample_min_per_stratum,
        max_per_stratum=settings.sample_max_per_stratum,
    )

    nested: List[Dict[str, object]] = []
    flat: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []

    for result in results:
        ix = by_id.get(result.transaction_id)
        pick = sampled.get(result.transaction_id)
        primary = _primary_finding(result)
        ev = primary.evidence if primary else None
        # the input columns first, so the file reads like the input with
        # the verdict appended and joins back without a key lookup
        row = {col: (ix.raw.get(col, "") if ix else "")
               for col in settings.carry_input_columns}
        row.update({
            "route": result.route.value,
            "expected_behaviour": result.expected_behaviour,
            "expected_reason": result.expected_reason,
            "sample_stratum": pick.stratum.value if pick else "",
            "in_review_sample": bool(pick and pick.in_sample),
            "sample_reason": pick.reason if pick else "",
            # what a reviewer should actually open: a confirmed defect, or
            # a sampled clean row
            "in_review_queue": bool(result.needs_review
                                    or (pick and pick.in_sample)),
            "primary_theme": (result.primary_theme.value
                              if result.primary_theme else ""),
            "themes": "; ".join(t.value for t in result.themes),
            "factors": "; ".join(f.value for f in result.factors),
            "owners": _owners(result.factors),
            "needs_review": result.needs_review,
            # the leading finding, spelled out. Two complete sentences, so
            # a row can be judged in the spreadsheet without opening the
            # JSON or the trace.
            "finding_check": primary.check_id if primary else "",
            **(_evidence_fields(primary) if primary else
               {"answer_claims": "", "source_states": "", "source_ref": "",
                "quote": ""}),
            "finding_count": sum(1 for f in result.findings if f.is_defect),
            # every finding, including any beyond the primary one, so
            # nothing is lost in a one-row-per-interaction file
            "all_findings_json": json.dumps(
                [{"check": f.check_id,
                  "theme": f.theme.value if f.theme else None,
                  "factors": [x.value for x in f.factors],
                  **_evidence_fields(f)}
                 for f in result.findings if f.is_defect],
                ensure_ascii=False) if primary else "",
            "examined_and_cleared": "; ".join(
                f"{f.check_id}:{f.adjudication.value}"
                for f in result.findings if not f.is_defect),
            "unproven_checks": "; ".join(result.unproven),
            "screener_calls": result.screener_calls,
            "adjudicator_calls": result.adjudicator_calls,
            "errors": "; ".join(result.errors),
        })
        rows.append(row)

        finding_rows = _finding_rows(result, ix)
        flat.extend(finding_rows)
        if not finding_rows:
            continue
        nested.append({
            "transaction_id": result.transaction_id,
            "route": result.route.value,
            "themes": [t.value for t in result.themes],
            "factors": [f.value for f in result.factors],
            "request": ix.request if ix else "",
            "response": ix.response if ix else "",
            "findings": [
                {"check": f.check_id,
                 "theme": f.theme.value if f.theme else None,
                 "factors": [x.value for x in f.factors],
                 "owners": _owners(f.factors),
                 **_evidence_fields(f),
                 "grounded": f.grounded,
                 "rationale": f.rationale}
                for f in result.findings if f.is_defect
            ],
            # what was suspected and did not survive, so a reviewer can see
            # the audit disagreed with itself rather than wondering
            "examined_and_cleared": [
                {"check": f.check_id, "decision": f.adjudication.value,
                 "why": f.rationale}
                for f in result.findings
                if f.adjudication in (Adjudication.REJECTED,
                                      Adjudication.UNPROVEN)
            ],
        })

    summary = summarise(results)
    summary["dry_run"] = settings.dry_run
    summary["review_sample"] = sample_report(sampled)
    queue = sum(1 for r in rows if r["in_review_queue"])
    summary["review_queue"] = queue
    summary["review_queue_pct"] = round(100 * queue / max(len(rows), 1), 1)
    doc = {"summary": summary, "interactions": nested}

    main_path = settings.output_path or os.path.join(out_dir,
                                                     "audit_output.csv")
    os.makedirs(os.path.dirname(main_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(main_path, index=False)
    paths = {"output_csv": main_path}

    if settings.write_detail_files:
        paths["findings_json"] = os.path.join(out_dir, "findings.json")
        paths["findings_csv"] = os.path.join(out_dir, "findings.csv")
        with open(paths["findings_json"], "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
        pd.DataFrame(flat).to_csv(paths["findings_csv"], index=False)

    return {"summary": summary, "paths": paths,
            "findings": len(flat), "rows": len(rows)}


def print_summary(summary: Dict[str, object]) -> None:
    print(f"\nrows                 {summary['rows']}")
    print(f"expected behaviour   {summary['expected_behaviour']} "
          f"({summary['expected_behaviour_pct']}%)")
    print(f"with defects         {summary['with_defects']}")
    print(f"\nchecks screened      {summary['checks_screened']}")
    print(f"escalated            {summary['escalated']} "
          f"({summary['escalation_rate_pct']}%)")
    print(f"confirmed            {summary['confirmed']}")
    print(f"rejected on review   {summary['rejected_by_adjudicator']}")
    print(f"unproven             {summary['unproven']}")
    print(f"\nscreener calls       {summary['screener_calls']}")
    print(f"adjudicator calls    {summary['adjudicator_calls']}")

    if summary.get("dry_run"):
        print("\n*** DRY RUN: the adjudicator was never called, so no "
              "finding could be confirmed. ***")
        print("    Nothing was checkpointed. Set PA_DRY_RUN=false for a "
              "real run.")

    sample = summary.get("review_sample") or {}
    if sample:
        print(f"\nreview queue         {summary.get('review_queue')} "
              f"({summary.get('review_queue_pct')}%)"
              f"  = confirmed defects + sampled clean rows")
        print(f"sampled from clean   {sample.get('sampled')} of "
              f"{sample.get('expected_rows')} "
              f"({sample.get('overall_rate_pct')}%)")
        for row in sample.get("by_stratum", []):
            print(f"  {row['stratum']:14s} {row['sampled']:4d}/"
                  f"{row['expected_rows']:<5d} {row['rate_pct']:5.1f}%  "
                  f"{row['why_it_was_clean'][:58]}")

    if summary["themes"]:
        print("\nWHAT USERS EXPERIENCED")
        for name, count in summary["themes"].items():        # type: ignore
            print(f"  {name:18s} {count}")

    report = summary.get("defect_report") or []
    if report:
        print("\n" + "=" * 70)
        print("DEFECTS BY CAUSE")
        print("=" * 70)
        for entry in report:                                 # type: ignore
            print(f"\n{entry['factor']}"
                  f"{'':<{max(1, 44 - len(entry['factor']))}}"
                  f"{entry['rows']} row(s)    owner: {entry['owner']}")
            print(f"  {entry['defect']}")
            for ex in entry["examples"]:
                print(f"\n  {ex['transaction_id']}   [{ex['check']}]")
                if ex["answer"]:
                    print(f"     answer: {ex['answer'][:96]}")
                if ex["source"]:
                    print(f"     policy: {ex['source'][:96]}")
                if ex["ref"]:
                    print(f"     where:  {ex['ref'][:96]}")
            if entry["others"]:
                shown = ", ".join(entry["others"][:6])
                more = (f"  (+{len(entry['others']) - 6} more)"
                        if len(entry["others"]) > 6 else "")
                print(f"\n  also {shown}{more}")

    unproven = summary.get("unsubstantiated_report") or []
    if unproven:
        print("\n" + "=" * 70)
        print("SUSPECTED BUT NOT SUBSTANTIATED")
        print("  Counted as expected behaviour. This is the audit failing "
              "to decide,")
        print("  not the agent behaving well - read these before quoting "
              "the share.")
        print("=" * 70)
        for entry in unproven:                               # type: ignore
            print(f"\n{entry['check']}"
                  f"{'':<{max(1, 44 - len(entry['check']))}}"
                  f"{entry['rows']} row(s)")
            print(f"  {entry['why'][:110]}")
            print(f"  {', '.join(entry['transaction_ids'])}")
    if summary["expected_reasons"]:
        print("\nwhy rows came out clean")
        for name, count in summary["expected_reasons"].items():  # type: ignore
            print(f"  {count:5d}  {name}")
    for warning in summary["warnings"]:                      # type: ignore
        print(f"\nWARNING: {warning}")
