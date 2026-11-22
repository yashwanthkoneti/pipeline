"""Span extraction. The bugs here are silent, which makes them expensive.

A span that is too narrow makes a true finding unconfirmable; a span that
is too wide costs the money the cascade exists to save. The case worth
asserting is the one the demo caught: a pointer naming a COLUMN must not
be narrowed to one row, because the checks that point at columns exist to
catch one branch being presented as the rule.
"""

from policy_audit.domain.models import Interaction, Pointer
from policy_audit.domain.taxonomy import Route
from policy_audit.evidence.grounding import check_grounding, locate
from policy_audit.domain.models import Evidence
from policy_audit.evidence.spans import extract_spans, overlap, render_spans
from policy_audit.parsing.chunks import parse_chunks
from policy_audit.parsing.clarification import parse_clarification
from policy_audit.parsing.ontology import build_ontology

APPROVAL = ("**__DOCUMENT_TITLE__:**Approval Authority Standard"
            "**__SECTION__:**2 Authority levels**__TEXT__:**By amount:\n"
            "|Band|Approver|\n|---|---|\n|up to USD 10,000|Line Manager|\n"
            "|USD 10,000 to 100,000|Division Head|\n"
            "|above USD 100,000|Regional Committee|")


def make_ix(chunks=APPROVAL, request="Who approves an expense?",
            response="Your Line Manager approves the expense."):
    parsed = tuple(parse_chunks(chunks)) if chunks else ()
    return Interaction(
        transaction_id="T", conversation_id="C", request=request,
        response=response, history=(), chunks=parsed,
        ontology=build_ontology(parsed), route=Route.ANSWERED,
        clarification=parse_clarification(response, ("please clarify",)))


def test_column_pointer_returns_the_whole_table():
    """"Approver" matches every row equally - sending one hides the point."""
    ix = make_ix()
    spans = extract_spans(ix, [Pointer("chunks", 0, "Approver")])
    body = render_spans(spans)
    assert "Line Manager" in body
    assert "Division Head" in body
    assert "Regional Committee" in body


def test_distinctive_row_pointer_narrows_to_that_row():
    ix = make_ix()
    spans = extract_spans(ix, [Pointer("chunks", 0, "above USD 100,000")])
    body = render_spans(spans)
    assert "Regional Committee" in body
    assert "Line Manager" not in body


def test_table_span_always_carries_its_header():
    ix = make_ix()
    spans = extract_spans(ix, [Pointer("chunks", 0, "above USD 100,000")])
    body = render_spans(spans)
    # a bare cell means nothing; the column name must travel with it
    assert "Approver=" in body
    assert "columns:" in body


def test_bad_chunk_index_falls_back_to_the_best_match():
    ix = make_ix()
    spans = extract_spans(ix, [Pointer("chunks", 99, "Division Head")])
    assert spans and "Division Head" in spans[0].text


def test_no_chunks_yields_no_chunk_spans():
    ix = make_ix(chunks="")
    assert extract_spans(ix, [Pointer("chunks", 0, "anything")]) == []


def test_spans_are_bounded():
    long_chunk = ("**__DOCUMENT_TITLE__:**Big**__TEXT__:**"
                  + "filler sentence. " * 400 + "NEEDLE here.")
    ix = make_ix(chunks=long_chunk)
    spans = extract_spans(ix, [Pointer("chunks", 0, "NEEDLE")], width=300)
    assert spans
    assert "NEEDLE" in spans[0].text
    assert len(spans[0].text) < 600          # not the whole chunk


# --------------------------------------------------------------------------
# grounding
# --------------------------------------------------------------------------
def test_quote_present_in_policy_text_is_grounded():
    ix = make_ix()
    ev = Evidence(answer_claims="the Line Manager approves any expense",
                  source_states="the approver depends on the amount band",
                  quote="Approver=Division Head")
    assert check_grounding(ev, ix, floor=0.4).grounded


def test_quote_taken_from_the_user_is_called_out():
    ix = make_ix()
    ev = Evidence(answer_claims="the Line Manager approves",
                  source_states="the user asked who approves an expense",
                  quote="Who approves an expense")
    result = check_grounding(ev, ix, floor=0.4)
    assert not result.grounded
    assert result.cites_user_text
    assert "the user's own words" in result.note


def test_quote_found_nowhere_is_unverifiable():
    ix = make_ix()
    ev = Evidence(answer_claims="x happens",
                  source_states="escalation to the Oversight Panel is "
                                "mandatory for all cases",
                  quote="escalation to the Oversight Panel")
    result = check_grounding(ev, ix, floor=0.5)
    assert not result.grounded
    assert not result.cites_user_text


def test_check_that_does_not_rest_on_policy_text_is_exempt():
    """UNGROUNDED_CLAIM is true precisely when there is no policy text."""
    ix = make_ix(chunks="")
    ev = Evidence(answer_claims="the notice period is 90 days for all staff",
                  source_states="no retrieved policy text states this")
    assert check_grounding(ev, ix, needs_policy_text=False).grounded


def test_locate_is_robust_to_punctuation():
    assert locate("USD 10,000 to 100,000",
                  "band: USD 10,000 to 100,000 -- approver") == 1.0


def test_overlap_of_unrelated_text_is_low():
    assert overlap("compassionate leave entitlement",
                   "gift registration thresholds") < 0.3
