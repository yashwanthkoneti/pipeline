"""Run the cascade on the cases that caused trouble, with no endpoint.

The fake models here are crude on purpose: the screener suspects almost
everything, the adjudicator confirms only what it can quote. That is
enough to show the architecture doing its job on the four rows that
produced wrong labels in the previous pipeline.

    python demo.py
"""

from __future__ import annotations

import json
import re

import pandas as pd

from policy_audit.config import Settings
from policy_audit.domain.checks import CHECKS
from policy_audit.pipeline.prepare import prepare, preflight
from policy_audit.pipeline.runner import Runner
from policy_audit.reporting.export import export, print_summary

GE_CHUNK = (
    "**__FILE_NAME__:**GE_Policy_v2.pdf"
    "**__DOCUMENT_TITLE__:**Gifts and Entertainment Policy"
    "**__SECTION__:**3.1 Registration**__TEXT__:**Thresholds by category:\n"
    "||Amount|Registration required|\n|:---|---:|:--:|\n"
    "|Gifts|$25 and up to $200|No|\n|Gifts|above $200|Yes|\n"
    "Registration is made through the G&E portal.")

APPROVAL_CHUNK = (
    "**__FILE_NAME__:**Approval_Authority_v1.pdf"
    "**__DOCUMENT_TITLE__:**Approval Authority Standard"
    "**__SECTION__:**2 Authority levels**__TEXT__:**Approver by amount:\n"
    "|Band|Approver|\n|---|---|\n|up to USD 10,000|Line Manager|\n"
    "|USD 10,000 to 100,000|Division Head|\n"
    "|above USD 100,000|Regional Committee|")

LEAVE_CHUNK = (
    "**__FILE_NAME__:**Leave_Policy_v5.pdf"
    "**__DOCUMENT_TITLE__:**Leave Policy"
    "**__SECTION__:**4 Compassionate leave**__TEXT__:**"
    "Employees may take up to five days of paid compassionate leave on the "
    "death of an immediate family member. Maternity and sick leave are "
    "covered in sections 2 and 3.")

CLARIFY = ("I need a little more detail. Please clarify:\n\n"
           "1. Which type of leave are you asking about ?\n\n"
           "option a: Maternity leave\n"
           "option b: Compassionate leave\n"
           "option c: Sick leave")

ROWS = [
    # 1. the table row the old pipeline mangled by dropping an empty cell
    dict(transaction_id="T001", conversation_id="C1",
         request_created_on="2026-11-01 09:00:00",
         request="Do I need to register a $150 gift?",
         response="Yes, registration is required for a $150 gift.",
         document_chunks=GE_CHUNK),

    # 2. conditional policy answered with one branch. Previously labelled
    #    "intent misread", which it is not - the agent understood fine.
    dict(transaction_id="T002", conversation_id="C2",
         request_created_on="2026-11-01 09:05:00",
         request="Who approves an expense?",
         response="Your Line Manager approves the expense.",
         document_chunks=APPROVAL_CHUNK),

    # 3. askback: the agent asked which leave, the user said compassionate.
    #    Previously flagged incomplete for not covering the other types.
    dict(transaction_id="T003", conversation_id="C3",
         request_created_on="2026-11-01 09:10:00",
         request="What leave am I entitled to?",
         response=CLARIFY, document_chunks=""),
    dict(transaction_id="T004", conversation_id="C3",
         request_created_on="2026-11-01 09:11:00",
         request="1b",
         response=("You may take up to five days of paid compassionate "
                   "leave on the death of an immediate family member."),
         document_chunks=LEAVE_CHUNK),

    # 4. a genuinely clean answer, to check the default really is clean
    dict(transaction_id="T005", conversation_id="C4",
         request_created_on="2026-11-01 09:20:00",
         request="Is registration needed for a $40 gift?",
         response=("No. For gifts from $25 up to $200 registration is not "
                   "required."),
         document_chunks=GE_CHUNK),

    # 5. no retrieval at all, answered confidently anyway
    dict(transaction_id="T006", conversation_id="C5",
         request_created_on="2026-11-01 09:25:00",
         request="What is the notice period for resignation?",
         response="The notice period is 90 days for all staff.",
         document_chunks=""),
]


class FakeScreener:
    """Suspects anything it has not been given reason to clear."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        if "GATE: REQUEST" in user:
            return json.dumps({"facts": {
                "asks_for": "policy_content",
                "in_policy_scope": True,
                "is_action_request": False,
                "answerable_as_stated": "Who approves" not in user,
                "parts": ["the question asked"]}})

        asked = [c.id for c in CHECKS if f"### {c.id}" in user]
        narrowed = "NARROWED" in user
        no_retrieval = "nothing was retrieved" in user
        checks = {}
        for cid in asked:
            verdict, locator = "clear", ""
            if cid == "TABLE_MISREAD" and "Registration required=No" in user \
                    and "required for a $150" in user:
                verdict, locator = "suspect", "Registration required"
            elif cid == "BRANCH_AS_RULE" and "varies by" in user:
                verdict, locator = "suspect", "Approver"
            elif cid == "UNGROUNDED_CLAIM" and no_retrieval:
                verdict, locator = "suspect", "notice period"
            elif cid == "PART_UNADDRESSED" and not narrowed \
                    and "leave" in user.lower():
                verdict, locator = "suspect", "leave"
            checks[cid] = {
                "verdict": verdict,
                "reason": "worth a closer look" if verdict != "clear" else "",
                "where": ([{"source": "chunks", "chunk": 0,
                            "locator": locator}] if verdict != "clear"
                          else []),
            }
        return json.dumps({"checks": checks})


class FakeAdjudicator:
    """Confirms only what it can quote out of the span it was given."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        check = re.search(r"check: (\w+)", user)
        cid = check.group(1) if check else ""

        if cid == "TABLE_MISREAD" and "Registration required=No" in user:
            return json.dumps({
                "decision": "confirmed",
                "answer_claims": "registration is required for a $150 gift",
                "source_states": ("for gifts from $25 up to $200 "
                                  "registration required is No"),
                "source_ref": "Gifts and Entertainment Policy, row 1",
                "quote": "Registration required=No",
                "factor": "table_misread",
                "rationale": "the $150 gift falls in the row marked No"})

        if cid == "BRANCH_AS_RULE" and "Approver=Division Head" in user:
            return json.dumps({
                "decision": "confirmed",
                "answer_claims": ("the Line Manager approves the expense, "
                                  "stated without qualification"),
                "source_states": ("the approver depends on the amount band: "
                                  "Line Manager up to USD 10,000, Division "
                                  "Head above that"),
                "source_ref": "Approval Authority Standard, section 2",
                "quote": "Band=USD 10,000 to 100,000; Approver=Division Head",
                "factor": "single_branch_presented_as_rule",
                "rationale": ("the answer names no amount, so it is one "
                              "branch presented as the rule")})

        if cid == "UNGROUNDED_CLAIM":
            return json.dumps({
                "decision": "confirmed",
                "answer_claims": ("the notice period for resignation is 90 "
                                  "days for all staff"),
                "source_states": "no retrieved policy text states this",
                "source_ref": "",
                "quote": "",
                "factor": "ungrounded_generation",
                "rationale": "nothing was retrieved for this request"})

        return json.dumps({
            "decision": "rejected",
            "rationale": ("the passage does not support the suspicion on a "
                          "careful read")})


def main() -> None:
    settings = Settings(checkpoint_path="", output_dir="out/demo",
                        output_path="out/demo/audit_output.csv",
                        span_min_match=0.35,
                        sample_rate=0.10, sample_min_per_stratum=1)
    df = pd.DataFrame(ROWS)
    interactions = prepare(df, settings)

    report = preflight(interactions, settings)
    print("=" * 66)
    print("PREFLIGHT  (no model calls)")
    print("=" * 66)
    for key in ("rows", "with_retrieval", "with_resolved_title",
                "with_tables", "clarification_responses", "askback_turns",
                "rows_with_conditional_policy", "routes"):
        print(f"  {key:30s} {report[key]}")
    for warning in report["warnings"]:
        print(f"\n  WARNING: {warning}")

    runner = Runner(FakeScreener(), FakeAdjudicator(), settings)
    results = runner.run(interactions, resume=False)

    print("\n" + "=" * 66)
    print("PER ROW")
    print("=" * 66)
    by_id = {i.transaction_id: i for i in interactions}
    for r in results:
        ix = by_id[r.transaction_id]
        verdict = ("EXPECTED" if r.expected_behaviour
                   else f"DEFECT: {r.primary_theme.value}")
        print(f"\n{r.transaction_id}  [{r.route.value}]  {verdict}")
        print(f"   asked : {ix.request[:60]}")
        if r.expected_behaviour:
            print(f"   why   : {r.expected_reason}")
        for f in r.findings:
            if not f.is_defect:
                continue
            ev = f.evidence
            print(f"   factor: {', '.join(x.value for x in f.factors)}")
            print(f"   answer: {ev.answer_claims}")
            print(f"   policy: {ev.source_states}")
            print(f"   ref   : {ev.source_ref or '(none)'}")

    out = export(results, interactions, settings.output_dir,
                 settings)
    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print_summary(out["summary"])
    for path in out["paths"].values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
