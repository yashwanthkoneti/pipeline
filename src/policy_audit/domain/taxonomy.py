"""The labels an audit can produce, and nothing else.

Two layers, and keeping them apart is the whole point:

  THEME   what the USER experienced. Five of them, mutually comprehensible
          to anyone, defensible by quoting the request and the response.
          This is what goes in front of stakeholders.

  FACTOR  what the SYSTEM did that produced it. An inference about
          mechanism, which routes the fix to an owner.

A theme is an observation; a factor is an explanation. The two were
previously fused into one list of names, which meant a label like
"intent misinterpretation" asserted a mechanism while resting on evidence
that could only support an observation - and could not be defended in
review. Themes now carry no mechanism claim, and factors are never the
headline.

Every finding carries exactly one theme and zero or more factors.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Tuple


class Theme(str, Enum):
    """What the user experienced. The primary label."""

    INCORRECT = "Incorrect"
    INCOMPLETE = "Incomplete"
    IRRELEVANT = "Irrelevant"
    INCONSISTENT = "Inconsistent"
    UNSUBSTANTIATED = "Unsubstantiated"


THEME_MEANING: Dict[Theme, str] = {
    Theme.INCORRECT: "The answer states something the policy contradicts",
    Theme.INCOMPLETE: "The answer leaves out something the user asked for",
    Theme.IRRELEVANT: "The user did not get their question answered",
    Theme.INCONSISTENT: (
        "The answer conflicts with the policy set or with an earlier turn"
    ),
    Theme.UNSUBSTANTIATED: "The answer is not supported by the policy text",
}

class Factor(str, Enum):
    """Why it happened. Secondary, and never reported on its own.

    Thirteen, named to match how the fixes are owned. The checks stay
    more specific than this and roll up here - findings.csv carries the
    check id alongside the factor, so a coarse factor does not lose the
    detail, it just reports at the level someone can act on.
    """

    # ---- clarification, split four ways ----------------------------
    # One "unnecessary clarification" hid four different fixes. Stale
    # context is conversation memory; chunks-resolved is the routing
    # prompt failing to read retrieval; request-was-clear is its
    # threshold; ambiguity-missed is the opposite error entirely.
    AMBIGUITY_MISSED = "ambiguity_missed"
    CLARIFIED_ON_STALE_CONTEXT = "clarified_on_stale_context"
    CLARIFIED_THOUGH_CHUNKS_RESOLVED = (
        "clarified_though_chunks_resolved_ambiguity")
    CLARIFIED_THOUGH_REQUEST_WAS_CLEAR = (
        "clarified_though_request_was_clear")

    # ---- context ----------------------------------------------------
    CONTEXT_CONTINUITY_FAILURE = "context_continuity_failure"

    # ---- request understanding --------------------------------------
    INTENT_MISREAD = "intent_misread"
    INTENT_MISREAD_ACTION_DETECTOR = "intent_misread_action_detector"

    # ---- guardrail --------------------------------------------------
    GUARDRAIL_BLOCKED_ANSWERABLE = "guardrail_blocked_answerable"
    GUARDRAIL_REFUSED_POLICY_QUESTION = "guardrail_refused_policy_question"

    # ---- retrieval --------------------------------------------------
    RETRIEVAL_APPLICABILITY_MISS = "retrieval_applicability_miss"
    RETRIEVAL_PARTIAL_COVERAGE = "retrieval_partial_coverage"

    # ---- generation -------------------------------------------------
    # The roll-up for table misreads, dropped conditions, one branch
    # presented as the rule, and claims with nothing behind them. The
    # check id says which.
    GENERATION_OR_COMMUNICATION_FAILURE = (
        "generation_or_communication_failure")

    # ---- fallback ---------------------------------------------------
    # Used when a defect is confirmed but nothing more specific fits.
    # Worth watching: a large count here means the check set is missing
    # a category rather than that the cause is genuinely unclear.
    PROMPT_OR_ROUTING = "prompt_or_routing"


FACTOR_OWNER: Dict[Factor, str] = {
    Factor.AMBIGUITY_MISSED: "routing prompt",
    Factor.CLARIFIED_ON_STALE_CONTEXT: "conversation memory",
    Factor.CLARIFIED_THOUGH_CHUNKS_RESOLVED: "routing prompt",
    Factor.CLARIFIED_THOUGH_REQUEST_WAS_CLEAR: "routing prompt",
    Factor.CONTEXT_CONTINUITY_FAILURE: "conversation memory",
    Factor.INTENT_MISREAD: "intent detection / routing prompt",
    Factor.INTENT_MISREAD_ACTION_DETECTOR: "action detector",
    Factor.GUARDRAIL_BLOCKED_ANSWERABLE: "guardrail prompt",
    Factor.GUARDRAIL_REFUSED_POLICY_QUESTION: "guardrail prompt / scope",
    Factor.RETRIEVAL_APPLICABILITY_MISS: "retrieval tuning",
    Factor.RETRIEVAL_PARTIAL_COVERAGE: "retrieval tuning / corpus",
    Factor.GENERATION_OR_COMMUNICATION_FAILURE: "response prompt",
    Factor.PROMPT_OR_ROUTING: "routing prompt",
}

# One line a stakeholder can read, per factor. The summary pairs this
# with a real example from the run rather than a count.
FACTOR_DEFECT: Dict[Factor, str] = {
    Factor.AMBIGUITY_MISSED:
        "answered without a fact the answer turns on, and without asking "
        "for it or saying it was assumed",
    Factor.CLARIFIED_ON_STALE_CONTEXT:
        "asked about an earlier topic the user had already moved off",
    Factor.CLARIFIED_THOUGH_CHUNKS_RESOLVED:
        "asked the user for something the retrieved policy already settled",
    Factor.CLARIFIED_THOUGH_REQUEST_WAS_CLEAR:
        "asked for detail the request had already given",
    Factor.CONTEXT_CONTINUITY_FAILURE:
        "ignored or contradicted something established earlier in the "
        "conversation",
    Factor.INTENT_MISREAD:
        "answered a different question from the one asked",
    Factor.INTENT_MISREAD_ACTION_DETECTOR:
        "refused as an action request something that asked about a process",
    Factor.GUARDRAIL_BLOCKED_ANSWERABLE:
        "declined a policy question it had the material to answer",
    Factor.GUARDRAIL_REFUSED_POLICY_QUESTION:
        "declined a request for where to find a policy document",
    Factor.RETRIEVAL_APPLICABILITY_MISS:
        "used policy text that does not govern the user's situation",
    Factor.RETRIEVAL_PARTIAL_COVERAGE:
        "answered from text that covers only part of what was asked",
    Factor.GENERATION_OR_COMMUNICATION_FAILURE:
        "the answer departs from the policy text it was given",
    Factor.PROMPT_OR_ROUTING:
        "a confirmed defect with no more specific cause identified",
}


class Gate(str, Enum):
    """The four questions, asked in this order.

    Each gate can end the audit. A request that was correctly refused
    never reaches the answer gate, so the answer gate cannot invent a
    defect about an answer that does not exist - which is how
    "answer_incomplete" used to attach itself to clarifications.
    """

    REQUEST = "request"      # what was asked
    CONTEXT = "context"      # what the agent had to work with
    ROUTE = "route"          # what the agent chose to do
    ANSWER = "answer"        # whether the answer holds up


# Which gates share a call. Four gates, three calls: context and route
# reason over the same sources - the conversation, the request and the
# response, and neither needs the policy text - so asking them separately
# bought a second system prompt and a second copy of the response for
# nothing. The gates stay distinct in the taxonomy and in the output; only
# the call is shared.
CALL_PLAN: Tuple[Tuple[Gate, ...], ...] = (
    (Gate.REQUEST,),
    (Gate.CONTEXT, Gate.ROUTE),
    (Gate.ANSWER,),
)


class Route(str, Enum):
    ANSWERED = "answered"
    CLARIFIED = "clarified"
    REFUSED_SCOPE = "refused_scope"
    REFUSED_ACTION = "refused_action"
    STATIC = "static"


class Suspicion(str, Enum):
    """What the cheap screener can say about a check.

    Deliberately three-valued, and deliberately asymmetric in meaning:

      CLEAR     nothing here. Final - this is where the cost saving comes
                from, so the screener's bar for CLEAR is high.
      SUSPECT   something might be wrong. NOT a finding. Escalates.
      UNSURE    cannot tell from what I was given. Escalates, because
                "I could not tell" must never silently become "clean".
    """

    CLEAR = "clear"
    SUSPECT = "suspect"
    UNSURE = "unsure"


ESCALATING: FrozenSet[Suspicion] = frozenset({Suspicion.SUSPECT,
                                              Suspicion.UNSURE})


class Adjudication(str, Enum):
    """What the costly adjudicator can say. Its word is final.

      CONFIRMED  a defect, with a quoted span on both sides
      REJECTED   the screener was wrong, and here is why
      UNPROVEN   may well be real, but it cannot be evidenced from the
                 text, so it is not reported as a defect
    """

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNPROVEN = "unproven"


# Only CONFIRMED produces a finding. UNPROVEN is recorded for calibration
# but never reported as a defect, because a finding a reviewer cannot
# verify costs more than the defect it might have caught.
REPORTABLE: FrozenSet[Adjudication] = frozenset({Adjudication.CONFIRMED})
