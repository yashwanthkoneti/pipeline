"""The checks the screener runs, and what a confirmed one means.

Fourteen checks, not forty-six flags. The old set had grown to cover
every shape anyone had ever noticed, and most of it never produced a
finding anyone acted on - it produced labels. A check earns its place
here only if a confirmed instance is something an owner would change
code or prompt for.

Each check declares, in one place:

  gate        when it is asked
  routes      which routes it can apply to at all
  theme       what the user experienced, if it is confirmed
  factors     candidate mechanisms, narrowed by the adjudicator
  quote_from  which sources the adjudicator MUST quote to confirm it.
              A check whose evidence cannot be quoted from the sources
              listed here can never be confirmed, which is what stops
              an unverifiable finding reaching a review queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

from .taxonomy import Factor, Gate, Route, Theme

ANSWER_ROUTES: Tuple[Route, ...] = (Route.ANSWERED,)
REFUSED_ROUTES: Tuple[Route, ...] = (Route.REFUSED_SCOPE, Route.REFUSED_ACTION,
                                     Route.STATIC)
ALL_ROUTES: Tuple[Route, ...] = tuple(Route)


@dataclass(frozen=True)
class Check:
    id: str
    gate: Gate
    routes: Tuple[Route, ...]
    theme: Theme
    factors: Tuple[Factor, ...]
    # what the screener is asked, in its own words
    question: str
    # what makes it SUSPECT - written to catch, not to be sure
    suspect_when: str
    # what makes it CLEAR - written strictly, because CLEAR is final
    clear_when: str
    # sources the adjudicator must be able to quote from
    quote_from: Tuple[str, ...] = ("chunks", "response")
    # a note the adjudicator sees, for traps specific to this check
    adjudicator_note: str = ""


CHECKS: Tuple[Check, ...] = (

    # ---- gate 2: context -------------------------------------------
    Check(
        id="IGNORED_HISTORY",
        gate=Gate.CONTEXT,
        routes=ALL_ROUTES,
        theme=Theme.INCONSISTENT,
        factors=(Factor.CONTEXT_CONTINUITY_FAILURE,),
        question=("Did the response ignore a fact the user already gave in "
                  "an earlier turn?"),
        suspect_when=("an earlier user turn states a fact (their region, "
                      "entity, role, product, the document they are working "
                      "from) that the current response needed and did not "
                      "use, OR the response asks for something the user "
                      "already said"),
        clear_when=("the response uses every fact from history that bears "
                    "on it, or history contains no such fact"),
        quote_from=("history", "response"),
    ),

    # ---- gate 3: route ---------------------------------------------
    Check(
        id="CLARIFIED_ANSWERABLE",
        gate=Gate.ROUTE,
        routes=(Route.CLARIFIED,),
        theme=Theme.IRRELEVANT,
        # All four clarification factors are candidates; the
        # adjudicator picks which, since only a reader of the
        # passages can tell a stale topic from a clear request.
        factors=(Factor.CLARIFIED_THOUGH_REQUEST_WAS_CLEAR,
                 Factor.CLARIFIED_THOUGH_CHUNKS_RESOLVED,
                 Factor.CLARIFIED_ON_STALE_CONTEXT),
        question=("Was the clarification unnecessary - could this have been "
                  "answered as asked?"),
        suspect_when=("every question the clarification asks is either "
                      "already answered in this turn or in history, or asks "
                      "about something the policy answer does not turn on"),
        clear_when=("at least one question it asks is for a fact that is "
                    "genuinely missing and would change the answer"),
        quote_from=("request", "history", "response"),
        adjudicator_note=(
            "A clarification with several questions is only unnecessary if "
            "EVERY question it asks was already answered or irrelevant. One "
            "genuinely needed question makes the whole clarification "
            "warranted."),
    ),
    Check(
        id="GUARDRAIL_FALSE_BLOCK",
        gate=Gate.ROUTE,
        routes=REFUSED_ROUTES,
        theme=Theme.IRRELEVANT,
        factors=(Factor.GUARDRAIL_BLOCKED_ANSWERABLE,
                 Factor.GUARDRAIL_REFUSED_POLICY_QUESTION),
        question=("Was this a policy question the agent should have "
                  "answered, rather than declined?"),
        suspect_when=("the request asks about a policy, standard, process or "
                      "where to find a document, and the response declines, "
                      "deflects, or returns a generic 'I can only help with "
                      "policy documents' message"),
        clear_when=("the request asks the agent to perform an action, or "
                    "asks about something outside policy altogether, or "
                    "needs the user's own records to answer"),
        quote_from=("request", "response"),
    ),
    Check(
        id="ACTION_MISDETECTED",
        gate=Gate.ROUTE,
        routes=REFUSED_ROUTES,
        theme=Theme.IRRELEVANT,
        factors=(Factor.INTENT_MISREAD_ACTION_DETECTOR,),
        question=("Did the agent refuse as an action request something that "
                  "was a question about a process?"),
        suspect_when=("the request asks HOW something is done, WHAT the rule "
                      "is, or WHO approves, and the response refuses on the "
                      "grounds that it cannot take actions"),
        clear_when=("the request asks the agent itself to submit, approve, "
                    "cancel, escalate or otherwise do something"),
        quote_from=("request", "response"),
    ),
    Check(
        id="SHOULD_HAVE_ASKED",
        gate=Gate.ROUTE,
        routes=ANSWER_ROUTES,
        theme=Theme.UNSUBSTANTIATED,
        factors=(Factor.AMBIGUITY_MISSED,),
        question=("Did the agent answer a question that needed a fact the "
                  "user never supplied, without asking and without saying "
                  "so?"),
        suspect_when=("the correct answer depends on a fact the user did not "
                      "state - region, entity, role, product, threshold "
                      "band, complaint or approval type - and the response "
                      "picks one silently"),
        clear_when=("the answer holds whatever the unstated fact turns out "
                    "to be, or the response states which case it is "
                    "answering for, or it asks"),
        quote_from=("request", "response"),
        adjudicator_note=(
            "Naming the assumed fact is required. 'It assumed something' "
            "cannot be confirmed. If you cannot name the fact and show the "
            "answer turns on it, this is REJECTED."),
    ),

    # ---- gate 4: answer --------------------------------------------
    Check(
        id="ANSWERED_DIFFERENT_QUESTION",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.IRRELEVANT,
        factors=(Factor.INTENT_MISREAD,),
        question=("Did the answer respond to a different question from the "
                  "one asked?"),
        suspect_when=("the answer is about a different KIND of thing than "
                      "was asked - where a document is versus what it says, "
                      "how a process works versus whether something is "
                      "allowed - or about a sibling subject: a different "
                      "standard, a different region's rule, a different "
                      "product"),
        clear_when=("the answer addresses what this turn asked, even if it "
                    "answers it badly, thinly or without support"),
        quote_from=("request", "response"),
        adjudicator_note=(
            "Misreading requires that the request DETERMINED an answer. If "
            "the request was missing a fact and the agent picked one of "
            "several legitimate readings, that is an assumption, not a "
            "misreading - there was no single intent to miss, so REJECT "
            "this and let SHOULD_HAVE_ASKED carry it.\n"
            "Not a misread either: a thin answer, an unsupported answer, "
            "or the right question answered from the wrong document. In "
            "all of those the agent understood you. Confirm only if you "
            "can write 'the user asked for A; the answer provides B' with "
            "both halves filled from this row.\n"
            "On an askback turn, judge against WHAT THE USER SELECTED - "
            "answering an option they did not choose is answering a "
            "different question."),
    ),
    Check(
        id="CONTRADICTS_POLICY",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.INCORRECT,
        factors=(Factor.GENERATION_OR_COMMUNICATION_FAILURE,
                 Factor.RETRIEVAL_APPLICABILITY_MISS),
        question="Does the answer state something the policy text denies?",
        suspect_when=("a value, condition, approver, period or permission in "
                      "the answer differs from what the policy text states "
                      "on the same point"),
        clear_when=("every such statement matches the policy text, or the "
                    "policy text does not address that point"),
        adjudicator_note=(
            "A conflict about a VALUE (a number, an amount, a period, a "
            "named approver) is the strongest form. A difference in wording "
            "where both statements mean the same thing is not a conflict. "
            "If the response states the policy's position elsewhere in the "
            "same answer, this is REJECTED - you were shown a fragment."),
    ),
    Check(
        id="TABLE_MISREAD",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.INCORRECT,
        factors=(Factor.GENERATION_OR_COMMUNICATION_FAILURE,),
        question=("Does the answer attach a table value to the wrong row?"),
        suspect_when=("the answer cites a figure or outcome from a table and "
                      "pairs it with a different row than the one it belongs "
                      "to"),
        clear_when=("every table value the answer cites sits in the row it "
                    "is attributed to, or the answer cites no table"),
        adjudicator_note=(
            "Tables are given to you already split, one row per line, every "
            "cell written as <column>=<value>. A value belongs ONLY to the "
            "row it is printed under. Quote the row and the answer's claim."),
    ),
    Check(
        id="CONDITION_DROPPED",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.INCOMPLETE,
        factors=(Factor.GENERATION_OR_COMMUNICATION_FAILURE,
                 Factor.RETRIEVAL_PARTIAL_COVERAGE),
        question=("Does the answer drop a condition, exception or limit that "
                  "changes what the user should do?"),
        suspect_when=("the policy text attaches an unless / only if / except "
                      "/ subject to clause to the rule the answer gives, and "
                      "the answer omits it"),
        clear_when=("the answer carries every qualifier that bears on the "
                    "user's situation, or the rule has none"),
        adjudicator_note=(
            "The test is whether acting on the answer alone would put the "
            "user wrong. A qualifier that does not change what they should "
            "do is not a dropped condition."),
    ),
    Check(
        id="PART_UNADDRESSED",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.INCOMPLETE,
        factors=(Factor.GENERATION_OR_COMMUNICATION_FAILURE,),
        question=("Did the user ask about more than one thing IN THIS TURN, "
                  "and the answer address only some of them?"),
        suspect_when=("this turn names two or more distinct topics, "
                      "standards or questions, and the answer addresses "
                      "fewer than all of them"),
        clear_when=("this turn asks about one thing, or the answer addresses "
                    "everything this turn asks about"),
        quote_from=("request", "response"),
        adjudicator_note=(
            "The parts are what THIS turn asks for. Use the history to "
            "understand what this turn MEANS; use this turn to decide what "
            "the answer had to COVER. A topic the user raised in an earlier "
            "turn, or that appears in the retrieved documents but not in "
            "this turn, is NOT a part - and where this turn answers the "
            "agent's own clarifying question, the user has just NARROWED the "
            "request and the topics they ruled out are not parts. You must "
            "quote each missing part from this turn's text; if you cannot, "
            "this is REJECTED."),
    ),
    Check(
        id="BRANCH_AS_RULE",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.UNSUBSTANTIATED,
        factors=(Factor.GENERATION_OR_COMMUNICATION_FAILURE,),
        question=("Does the policy give several alternatives here, with the "
                  "answer presenting one as though it were the rule?"),
        suspect_when=("the policy text sets out more than one rule, "
                      "approver, threshold or process depending on some "
                      "dimension, and the answer states one without naming "
                      "the condition under which it applies"),
        clear_when=("the answer names the case it applies to, or the policy "
                    "has a single unconditional rule here"),
        adjudicator_note=(
            "This is the commonest real defect in conditional policy and it "
            "is NOT a misread question - the agent understood perfectly and "
            "answered too narrowly. Quote at least two of the alternatives "
            "from the policy text plus the answer's unqualified claim."),
    ),
    Check(
        id="UNGROUNDED_CLAIM",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.UNSUBSTANTIATED,
        factors=(Factor.RETRIEVAL_PARTIAL_COVERAGE,
                 Factor.GENERATION_OR_COMMUNICATION_FAILURE),
        question=("Does the answer state a policy rule that the policy text "
                  "does not contain?"),
        suspect_when=("the answer asserts a rule, threshold, period or "
                      "obligation that appears nowhere in the policy text "
                      "provided"),
        clear_when=("every rule the answer states appears in the policy "
                    "text, or the answer says plainly that it could not "
                    "find the relevant policy"),
        # The evidence here is the answer's own claim plus the ABSENCE of
        # supporting text. Requiring a policy quote would make the check
        # unconfirmable precisely when it is most clearly true - a row
        # with no retrieval at all.
        quote_from=("response",),
        adjudicator_note=(
            "Framing, structure and ordinary connective prose are not "
            "claims. Confirm this only for a statement a reader would act "
            "on. An answer that openly says the policy was not available is "
            "REJECTED - that is the agent behaving correctly."),
    ),
    Check(
        id="SELF_CONTRADICTION",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.INCONSISTENT,
        factors=(Factor.CONTEXT_CONTINUITY_FAILURE,),
        question=("Does the answer conflict with what the agent itself said "
                  "earlier in this conversation?"),
        suspect_when=("an earlier assistant turn stated a rule or outcome "
                      "and this response states a different one on the same "
                      "point, without saying anything changed"),
        clear_when=("no earlier turn addressed the same point, or both "
                    "statements agree, or the response explains the "
                    "difference"),
        quote_from=("history", "response"),
    ),
    Check(
        id="POLICY_SET_CONFLICT",
        gate=Gate.ANSWER,
        routes=ANSWER_ROUTES,
        theme=Theme.INCONSISTENT,
        factors=(Factor.RETRIEVAL_APPLICABILITY_MISS,),
        question=("Do two policy documents give conflicting rules, with the "
                  "answer following one without noting the conflict?"),
        suspect_when=("two retrieved documents state different rules on the "
                      "same point for the same situation, and the answer "
                      "picks one silently"),
        clear_when=("the documents agree, or they differ because they apply "
                    "to different situations, or the answer notes the "
                    "difference"),
        adjudicator_note=(
            "Two documents covering different regions, entities or product "
            "types do NOT conflict - they are conditional. A conflict means "
            "both claim to govern the SAME situation and disagree."),
    ),
)

CHECKS_BY_ID: Dict[str, Check] = {c.id: c for c in CHECKS}


def checks_for(gate: Gate, route: Route) -> Sequence[Check]:
    """The checks that can apply at this gate on this route.

    Route filtering happens here rather than in a prompt, so the screener
    is never asked whether an answer was incomplete on a row where no
    answer was given.
    """
    return [c for c in CHECKS if c.gate is gate and route in c.routes]
