"""Prompt construction for both agents.

The two agents are written to pull in opposite directions, and that
asymmetry is the whole design:

  The screener is told to be SENSITIVE. Its job is to not miss things.
  "suspect" costs one adjudication call; a miss costs a defect shipped.
  It is told explicitly that suspect is not an accusation.

  The adjudicator is told to be STRICT. Its job is to confirm only what
  it can quote. It is told explicitly that rejecting the screener is a
  good outcome, so that overturning carries no implied cost.

Net effect: high recall at the screen, high precision at the verdict,
and the verdict is what reaches a human.
"""

from __future__ import annotations

from typing import Sequence

from ..domain.checks import Check
from ..domain.models import Interaction
from ..domain.taxonomy import Gate

# --------------------------------------------------------------------------
# shared framing
# --------------------------------------------------------------------------
_WHAT_IS_BEING_AUDITED = """\
You are auditing an assistant that answers questions about a bank's
internal policies. It has five legitimate behaviours, and ALL of them are
correct conduct:

  1. answer a policy question from the retrieved policy text
  2. ask the user a clarifying question when it needs a fact to answer
  3. decline a question that is not about policy
  4. stay inside the policy corpus rather than giving general advice
  5. refuse to perform an action, and explain the process instead

The assistant works from whatever the retrieval step handed it. It does
not choose its own documents. An answer limited by a thin corpus is not
the assistant's fault, and neither is a gap in the policy set.
"""

_WHAT_HISTORY_IS_FOR = """\
Use the full conversation history to understand what the current request
MEANS. Use the current request to decide what the answer had to COVER.

Those are different jobs. History tells you that "compassionate leave"
refers to the entitlement under the policy discussed earlier - that is
history doing its job. It does not add topics the answer owed the user. A
subject the user raised in an earlier turn, and has since moved on from,
is not something this answer had to cover.
"""

_ASKBACK_NOTE = """\
IMPORTANT - THIS TURN ANSWERS THE ASSISTANT'S OWN QUESTION.

The assistant asked: {question}
{selections}
So the user has just NARROWED their request. They have said which case
they want. The topics they did not pick are no longer in scope, and an
answer confined to what they picked is CORRECT and complete, not partial.

Judge the response against WHAT THE USER SELECTED. An answer addressing
an option they did not choose is answering a different question.
"""

_SELECTIONS = """
The user's reply is a code. It resolves to:
{resolved}
"""


def _sources(ix: Interaction, *, include_chunks: bool = True,
             include_history: bool = True,
             include_ontology: bool = True) -> str:
    """Assemble only the sources this gate actually reasons over.

    The request gate is the case worth noting: it asks what the user
    asked for, and whether a missing fact would change the answer. For
    the second part it needs to know what the policy VARIES BY - which
    the ontology states in a hundred tokens - not the policy text itself.
    Sending the bodies there duplicated the single largest block in the
    prompt for no decision it could affect.
    """
    parts = []
    if include_history and ix.history:
        parts.append("## CONVERSATION HISTORY (earlier turns only)\n"
                     + ix.history_text)
    parts.append(f"## CURRENT USER REQUEST\n{ix.request}")
    if include_ontology:
        parts.append(ix.ontology.render())
    if include_chunks:
        if ix.chunks:
            body = "\n\n".join(
                f"[chunk {c.index} | {c.doc_title}]"
                + ("\n(NOTE: the table below was laid out by a model, not "
                   "by the parser. Every cell was checked to appear in the "
                   "source, so no value is invented - but the column "
                   "alignment is the model's. Satisfy yourself a value "
                   "really sits in the row it is shown under before you "
                   "rely on it.)" if c.model_structured_table
                   else "\n(NOTE: this chunk appears to contain a table that "
                   "could not be split into rows. Read its alignment "
                   "carefully, and quote the whole row when you cite a "
                   "value from it.)" if c.has_unstructured_table else "")
                + f"\n{c.text}"
                for c in ix.chunks)
            parts.append("## RETRIEVED POLICY TEXT\n" + body)
        else:
            parts.append("## RETRIEVED POLICY TEXT\n(nothing was retrieved "
                         "for this request)")
    parts.append(f"## THE ASSISTANT'S RESPONSE\n{ix.response}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# screener
# --------------------------------------------------------------------------
_SCREENER_SYSTEM = """\
{what}
{history}
YOU ARE THE FIRST OF TWO REVIEWERS. You are the fast one.

Your job is to NOT MISS anything. A second, more careful reviewer sees
everything you mark "suspect", reads the exact passages, and decides. So
"suspect" is not an accusation - it is a request for a closer look, and
it costs almost nothing.

"clear" is different. Nothing you mark clear is ever looked at again, so
only mark clear when you can see there is no problem.

"unsure" is for when you were not given enough to tell. Use it freely. It
escalates exactly like suspect.

For anything you mark suspect or unsure, say WHERE to look. The second
reviewer is shown only the passages you point at, not the whole
retrieval, so a pointer with no location wastes the escalation.

Return JSON only. No preamble, no code fences.
"""

_GATE_FACTS = {
    Gate.REQUEST: """\
Describe what the user asked for in THIS turn. Facts only - you are not
judging the response at this gate.

Return:
{
  "asks_for": "policy_content" | "document_location" | "process" |
              "action" | "not_policy",
  "in_policy_scope": true | false,
  "is_action_request": true | false,
  "answerable_as_stated": true | false,
  "missing_fact": "<if not answerable, the ONE fact that is missing and
                   that changes the answer; else empty>",
  "parts": ["<each distinct thing THIS turn asks about, quoted from this
             turn's words>"]
}

On "answerable_as_stated": ask whether the correct answer could DIFFER
depending on something the user left unstated - their region, entity,
role, product, threshold band, or which of several processes applies. If
it could, answer false and name that fact. Plain, well-formed English can
still be false: judge whether the ANSWER is determined, not whether the
QUESTION is readable.

On "parts": quote from this turn only. A request about one topic has one
part. Do not add topics from the history or from the retrieved documents.
""",
}

_CHECK_BLOCK = """\
For each check below, answer "clear", "suspect" or "unsure".

{checks}

Return:
{{
  "checks": {{
    "<CHECK_ID>": {{
      "verdict": "clear" | "suspect" | "unsure",
      "reason": "<one short line; for suspect, what specifically looks wrong>",
      "where": [
        {{"source": "chunks" | "response" | "request" | "history",
          "chunk": <chunk number, only when source is chunks>,
          "locator": "<a section name, a table row, or a short phrase to
                       find - enough for the second reviewer to locate it>"}}
      ]
    }}
  }}
}}

Include every check id listed. "where" may be empty only when the verdict
is "clear".
"""


def _render_checks(checks: Sequence[Check]) -> str:
    out = []
    for c in checks:
        out.append(
            f"### {c.id}\n"
            f"{c.question}\n"
            f"  suspect if: {c.suspect_when}\n"
            f"  clear if:   {c.clear_when}")
    return "\n\n".join(out)


def screener_prompt(ix: Interaction, gates: Sequence[Gate],
                    checks: Sequence[Check]) -> tuple[str, str]:
    """(system, user) for one CALL, which may cover more than one gate.

    Gates that reason over the same sources are batched into a single
    call - context and route both read the conversation and the response
    and neither reads the policy text, so splitting them paid for a
    second system prompt and a second copy of the response to ask four
    questions instead of two.
    """
    gates = tuple(gates)
    system = _SCREENER_SYSTEM.format(what=_WHAT_IS_BEING_AUDITED,
                                     history=_WHAT_HISTORY_IS_FOR)
    blocks = []
    if ix.is_askback_answer:
        blocks.append(_ASKBACK_NOTE.format(
            question=ix.askback_question or "(a clarifying question)",
            selections=(_SELECTIONS.format(resolved=ix.resolved_request)
                        if ix.resolved_request else "")))
    blocks.append(_sources(
        ix,
        include_chunks=Gate.ANSWER in gates,
        include_history=any(g in gates for g in (Gate.CONTEXT, Gate.ROUTE,
                                                 Gate.ANSWER)),
    ))
    label = " + ".join(g.value.upper() for g in gates)
    for gate in gates:
        if gate in _GATE_FACTS:
            blocks.append(f"## GATE: {gate.value.upper()}\n"
                          + _GATE_FACTS[gate])
    if checks:
        blocks.append(f"## GATE: {label} - CHECKS\n"
                      + _CHECK_BLOCK.format(checks=_render_checks(checks)))
    return system, "\n\n".join(blocks)


# --------------------------------------------------------------------------
# adjudicator
# --------------------------------------------------------------------------
_ADJUDICATOR_SYSTEM = """\
{what}
{history}
YOU ARE THE SECOND OF TWO REVIEWERS, and the decision is yours.

A fast first reviewer flagged this as possibly wrong and pointed at the
passages below. It is often wrong. REJECTING its suspicion is a good
outcome and the expected one - most suspicions do not survive a careful
read. You are not here to agree with it.

Confirm a defect only if you can state BOTH of these as complete
sentences that a reader could judge without seeing anything else:

  what the answer claims
  what the policy text actually states on the same point

A fragment is not a sentence. "registration required" and
"$25 and up to $200 | No" are fragments: the first has no subject, the
second is a table cell with no column. A table cell means nothing lifted
out of its row - name the column and the row. If you cannot write both
sentences completely, you have not proved a defect: answer "rejected" or
"unproven".

  confirmed  a real defect, with both sentences and a verbatim quote
  rejected   the first reviewer was wrong; say why in one line
  unproven   might be real, but the text given cannot settle it

Weigh the cost correctly. A confirmed defect sends someone to investigate
and change a prompt. A confirmed defect that turns out to be nothing
costs their time and the credibility of every other finding. An
unreported defect costs one row. When the two are close, choose
"unproven".

Return JSON only. No preamble, no code fences.
"""

_ADJUDICATOR_TASK = """\
## WHAT THE FIRST REVIEWER SUSPECTED
check: {check_id}
question: {question}
its reasoning: {reason}
{note}
## THE PASSAGES IT POINTED AT
{spans}

## THE ASSISTANT'S RESPONSE
{response}

## THE USER'S REQUEST (this turn)
{request}
{history}
## YOUR DECISION
Return:
{{
  "decision": "confirmed" | "rejected" | "unproven",
  "answer_claims": "<complete sentence: what the answer asserts>",
  "source_states": "<complete sentence: what the policy text states on the
                     same point. For a missing-part finding, what the user
                     asked for that was not addressed. For an ungrounded
                     claim, 'no retrieved policy text states this'.>",
  "source_ref": "<document title, and the table row or section>",
  "quote": "<short verbatim fragment from the passage, for checking>",
  "factor": "<one of: {factors}; or empty if none fits>",
  "rationale": "<one line. If rejecting, what the first reviewer missed.>"
}}

Leave answer_claims and source_states empty only when rejecting.
"""


def adjudicator_prompt(ix: Interaction, check: Check, reason: str,
                       spans: str) -> tuple[str, str]:
    """(system, user) for one escalated check."""
    system = _ADJUDICATOR_SYSTEM.format(what=_WHAT_IS_BEING_AUDITED,
                                        history=_WHAT_HISTORY_IS_FOR)
    note = ""
    if check.adjudicator_note:
        note = f"\n## WATCH FOR\n{check.adjudicator_note}\n"
    if ix.is_askback_answer:
        note += "\n" + _ASKBACK_NOTE.format(
            question=ix.askback_question or "(a clarifying question)",
            selections=(_SELECTIONS.format(resolved=ix.resolved_request)
                        if ix.resolved_request else ""))
    history = ""
    if "history" in check.quote_from and ix.history:
        history = ("\n## CONVERSATION HISTORY (earlier turns only)\n"
                   + ix.history_text + "\n")
    user = _ADJUDICATOR_TASK.format(
        check_id=check.id,
        question=check.question,
        reason=reason or "(none given)",
        note=note,
        spans=spans or "(the first reviewer pointed at nothing specific)",
        response=ix.response,
        request=ix.request,
        history=history,
        factors=", ".join(f.value for f in check.factors) or "(none)",
    )
    return system, user
