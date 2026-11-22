"""The cascade, end to end, with a scripted model.

These are the properties the whole design rests on, so they are asserted
rather than assumed:

  a CLEAR screen never reaches the adjudicator (the cost saving)
  a SUSPECT screen is not a finding until adjudicated (the precision)
  the adjudicator can overturn the screener
  fragment evidence is downgraded even when confirmed
  evidence quoting the user as policy is downgraded
  an unreachable adjudicator produces expected behaviour, not a defect
  a non-answer route never gets an answer-shaped finding
"""

import json

from policy_audit.agents.client import ScriptedLLM, repair_truncated_json
from policy_audit.config import Settings
from policy_audit.domain.models import Interaction, Turn
from policy_audit.domain.taxonomy import Adjudication, Route, Theme
from policy_audit.parsing.chunks import parse_chunks
from policy_audit.parsing.clarification import parse_clarification
from policy_audit.parsing.ontology import build_ontology
from policy_audit.pipeline.runner import Runner

CHUNKS = ("**__DOCUMENT_TITLE__:**Gifts and Entertainment Policy"
          "**__SECTION__:**3.1 Registration**__TEXT__:**"
          "|Category|Amount|Registration required|\n|---|---|---|\n"
          "|Gifts|$25 and up to $200|No|\n|Gifts|above $200|Yes|")

SETTINGS = Settings(checkpoint_path="", span_min_match=0.4)


def make_ix(route=Route.ANSWERED, *, response="Registration is required "
            "for a $150 gift.", request="Do I need to register a $150 gift?",
            chunks=CHUNKS, history=()):
    parsed = tuple(parse_chunks(chunks)) if chunks else ()
    return Interaction(
        transaction_id="T1", conversation_id="C1",
        request=request, response=response,
        history=tuple(history), chunks=parsed,
        ontology=build_ontology(parsed), route=route,
        clarification=parse_clarification(response, ("please clarify",)),
    )


def screen(**verdicts):
    """A screener reply marking the named checks, clearing every other.

    Clearing the rest explicitly matters: an omitted check becomes UNSURE
    and escalates by design, so a helper that named only one check would
    escalate all ten and the assertion would be about the wrong thing.
    """
    from policy_audit.domain.checks import CHECKS
    checks = {c.id: {"verdict": "clear", "reason": "", "where": []}
              for c in CHECKS}
    for cid, verdict in verdicts.items():
        checks[cid] = {
            "verdict": verdict,
            "reason": "looks off",
            "where": [{"source": "chunks", "chunk": 0,
                       "locator": "Registration required"}],
        }
    return json.dumps({"checks": checks})


def adjudicate(decision, **fields):
    payload = {"decision": decision, "factor": "", "rationale": "because"}
    payload.update(fields)
    return json.dumps(payload)


# --------------------------------------------------------------------------
# cost: clear never escalates
# --------------------------------------------------------------------------
def test_all_clear_makes_no_adjudicator_call():
    screener = ScriptedLLM(default=screen())
    adjudicator = ScriptedLLM(default="{}")
    ix = make_ix()
    result = Runner(screener, adjudicator, SETTINGS).audit(ix)
    assert result.expected_behaviour
    assert result.adjudicator_calls == 0
    assert adjudicator.seen == []
    assert "found nothing" in result.expected_reason


def test_omitted_check_escalates_rather_than_clearing():
    """Silence is not evidence of absence."""
    screener = ScriptedLLM(default='{"checks": {}}')
    adjudicator = ScriptedLLM(default=adjudicate("rejected"))
    result = Runner(screener, adjudicator, SETTINGS).audit(make_ix())
    assert result.adjudicator_calls > 0
    assert all(f.adjudication is Adjudication.REJECTED
               for f in result.findings)


# --------------------------------------------------------------------------
# precision: suspect is not a finding
# --------------------------------------------------------------------------
def test_suspect_rejected_by_adjudicator_is_expected_behaviour():
    screener = ScriptedLLM(default=screen(TABLE_MISREAD="suspect"))
    adjudicator = ScriptedLLM(default=adjudicate(
        "rejected", rationale="the figure does sit in that row"))
    result = Runner(screener, adjudicator, SETTINGS).audit(make_ix())
    assert result.expected_behaviour
    assert not result.needs_review
    assert any(f.overturned_screener for f in result.findings)
    assert "rejected" in result.expected_reason


def test_confirmed_with_readable_grounded_evidence_is_a_defect():
    screener = ScriptedLLM(default=screen(TABLE_MISREAD="suspect"))
    adjudicator = ScriptedLLM(default=adjudicate(
        "confirmed",
        answer_claims="registration is required for a $150 gift",
        source_states=("for gifts from $25 up to $200 registration "
                       "required is No"),
        source_ref="Gifts and Entertainment Policy, row 1",
        quote="Registration required=No",
        factor="generation_or_communication_failure"))
    result = Runner(screener, adjudicator, SETTINGS).audit(make_ix())
    assert not result.expected_behaviour
    assert result.primary_theme is Theme.INCORRECT
    assert [f.value for f in result.factors] == [
        "generation_or_communication_failure"]


def test_fragment_evidence_is_downgraded_even_when_confirmed():
    """`$25 and up to $200 | No` is a cell, not a statement."""
    screener = ScriptedLLM(default=screen(TABLE_MISREAD="suspect"))
    adjudicator = ScriptedLLM(default=adjudicate(
        "confirmed",
        answer_claims="registration required",
        source_states="$25 and up to $200 | No",
        quote="No"))
    result = Runner(screener, adjudicator, SETTINGS).audit(make_ix())
    assert result.expected_behaviour
    assert result.unproven == ("TABLE_MISREAD",)
    assert any("two complete statements" in f.rationale
               for f in result.findings)


def test_evidence_quoting_the_user_as_policy_is_downgraded():
    screener = ScriptedLLM(default=screen(CONTRADICTS_POLICY="suspect"))
    adjudicator = ScriptedLLM(default=adjudicate(
        "confirmed",
        answer_claims="registration is required for a $150 gift",
        source_states="the user asked whether to register a $150 gift",
        quote="Do I need to register a $150 gift"))
    result = Runner(screener, adjudicator, SETTINGS).audit(make_ix())
    assert result.expected_behaviour
    assert any("the user's own words" in f.rationale
               for f in result.findings)


def test_unreachable_adjudicator_does_not_become_a_defect():
    screener = ScriptedLLM(default=screen(TABLE_MISREAD="suspect"))
    adjudicator = ScriptedLLM(default="not json at all")
    result = Runner(screener, adjudicator,
                    Settings(checkpoint_path="", max_retries=1)).audit(
        make_ix())
    assert result.expected_behaviour
    assert result.unproven == ("TABLE_MISREAD",)


# --------------------------------------------------------------------------
# route gating
# --------------------------------------------------------------------------
def test_clarified_row_is_never_asked_answer_questions():
    screener = ScriptedLLM(default=screen(CLARIFIED_ANSWERABLE="clear"))
    adjudicator = ScriptedLLM(default="{}")
    ix = make_ix(route=Route.CLARIFIED,
                 response="Please clarify:\n1. Which region ?\noption a: EMEA")
    result = Runner(screener, adjudicator, SETTINGS).audit(ix)
    asked = " ".join(screener.seen)
    assert "CONDITION_DROPPED" not in asked
    assert "PART_UNADDRESSED" not in asked
    assert any("no answer to examine" in n for n in result.gate_notes)


def test_answered_row_is_asked_answer_questions():
    screener = ScriptedLLM(default=screen(CONDITION_DROPPED="clear"))
    adjudicator = ScriptedLLM(default="{}")
    Runner(screener, adjudicator, SETTINGS).audit(make_ix())
    assert "CONDITION_DROPPED" in " ".join(screener.seen)


# --------------------------------------------------------------------------
# askback framing reaches the prompt
# --------------------------------------------------------------------------
def test_askback_turn_tells_the_screener_the_request_was_narrowed():
    template = ("Please clarify:\n1. Which leave type ?\n"
                "option a: Maternity\noption b: Compassionate")
    ix = make_ix(request="Compassionate leave",
                 history=(Turn("user", "What leave am I entitled to?"),
                          Turn("assistant", template)))
    assert ix.is_askback_answer is False  # history built by hand lacks marker
    # build it the way prepare() would, via detect_askback
    from policy_audit.parsing.clarification import detect_askback
    is_askback, asked = detect_askback(ix.history, ("please clarify",))
    assert is_askback
    ix2 = Interaction(**{**vars(ix), "is_askback_answer": True,
                         "askback_question": asked})
    screener = ScriptedLLM(default=screen(PART_UNADDRESSED="clear"))
    Runner(screener, ScriptedLLM(default="{}"), SETTINGS).audit(ix2)
    prompt = " ".join(screener.seen)
    assert "NARROWED" in prompt
    assert "no longer in scope" in prompt


# --------------------------------------------------------------------------
# json recovery
# --------------------------------------------------------------------------
def test_truncated_json_is_recovered():
    broken = ('{"checks": {"A": {"verdict": "clear"}, '
              '"B": {"verdict": "susp')
    out = repair_truncated_json(broken)
    assert out is not None
    assert out["checks"]["A"]["verdict"] == "clear"


def test_code_fences_are_stripped():
    assert repair_truncated_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_brace_inside_a_string_does_not_confuse_the_walker():
    out = repair_truncated_json('{"a": "a } brace", "b": 2}')
    assert out == {"a": "a } brace", "b": 2}


def test_dry_run_is_not_checkpointed(tmp_path):
    """Its rows were never adjudicated, so a real run must redo them."""
    ckpt = tmp_path / "checkpoint.jsonl"
    settings = Settings(checkpoint_path=str(ckpt), dry_run=True,
                        span_min_match=0.4)
    screener = ScriptedLLM(default=screen(TABLE_MISREAD="suspect"))
    runner = Runner(screener, ScriptedLLM(default="{}"), settings)
    result = runner.run([make_ix()], resume=False)
    assert result[0].adjudicator_calls == 0
    assert not ckpt.exists()        # nothing recorded


def test_real_run_is_checkpointed(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    settings = Settings(checkpoint_path=str(ckpt), span_min_match=0.4)
    runner = Runner(ScriptedLLM(default=screen(TABLE_MISREAD="suspect")),
                    ScriptedLLM(default=adjudicate("rejected")), settings)
    runner.run([make_ix()], resume=False)
    assert ckpt.exists()
    assert ckpt.read_text().strip()
