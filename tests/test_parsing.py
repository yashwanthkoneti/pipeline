"""The deterministic layer. These are the bugs that cost the most to find."""

from policy_audit.parsing.chunks import parse_chunks, parse_fields, split_chunks
from policy_audit.parsing.clarification import (answers_an_option,
                                                detect_askback,
                                                parse_clarification)
from policy_audit.parsing.ontology import build_ontology
from policy_audit.parsing.tables import (looks_like_unstructured_table,
                                         parse_table, render_table, split_row)
from policy_audit.domain.models import Turn

MARKERS = ("please clarify",)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def test_split_row_keeps_empty_leading_cell():
    """`.strip('|')` eats it and every column shifts left by one."""
    assert split_row("||Amount|Registration|") == ["", "Amount",
                                                   "Registration"]


def test_split_row_keeps_empty_trailing_cell():
    assert split_row("|Amount|Registration||") == ["Amount", "Registration",
                                                   ""]


def test_split_row_keeps_empty_middle_cell():
    assert split_row("|Band||Approver|") == ["Band", "", "Approver"]


def test_escaped_pipe_is_data():
    assert split_row(r"|Range|$25 \| $200|") == ["Range", "$25 | $200"]


def test_table_alignment_survives_blank_header_cell():
    table = parse_table(["||Amount|Registration required|",
                         "|:---|---:|:--:|",
                         "|Gifts|$25 and up to $200|No|",
                         "|Gifts|above $200|Yes|"])
    assert table is not None
    assert table.header == ("col1", "Amount", "Registration required")
    rendered = table.as_rows()
    # the value must stay attached to its OWN column
    assert "Amount=$25 and up to $200" in rendered[0]
    assert "Registration required=No" in rendered[0]
    assert "Registration required=Yes" in rendered[1]


def test_short_row_is_padded_not_slid_left():
    table = parse_table(["|A|B|C|", "|1|2|"])
    assert table is not None
    assert table.rows[0] == ("1", "2", "")


def test_separator_only_row_is_dropped_but_blank_row_is_not_a_separator():
    table = parse_table(["|A|B|", "|---|---|", "|  |  |", "|1|2|"])
    assert table is not None
    assert table.rows == (("1", "2"),)


def test_single_pipe_row_is_not_a_table():
    assert parse_table(["|just one row|"]) is None


# --------------------------------------------------------------------------
# chunks
# --------------------------------------------------------------------------
RAW = ("**__FILE_NAME__:**PAD_Std_v4.pdf**__DOCUMENT_TITLE__:**Personal "
       "Account Dealing Standard**__SECTION__:**4.2 Pre-clearance"
       "**__TEXT__:**Limits by region:\n"
       "|Region|Threshold|Approver|\n|---|---|---|\n"
       "|EMEA|USD 50,000|Regional CO|\n|APAC|USD 25,000|Global CO|\n"
       "Broad-based ETFs are exempt from the above.,"
       "**__FILE_NAME__:**GE_Pol_v2.pdf**__DOCUMENT_TITLE__:**Gifts and "
       "Entertainment Policy**__TEXT__:**Gifts above USD 100 need approval.")


def test_decorated_keys_are_undecorated():
    """`__FILE_NAME__` must become something a lookup can find.

    Case is kept so the rendered chunk reads naturally; lookups are
    case-insensitive. What must not survive is the underscores, which is
    what made every title resolve to "?".
    """
    fields = parse_fields("**__FILE_NAME__:**x.pdf")
    assert fields[0][0].lower() == "file name"


def test_field_lookup_is_case_and_shape_insensitive():
    chunk = parse_chunks("**__FILE_NAME__:**x.pdf")[0]
    assert chunk.field("file name") == "x.pdf"
    assert chunk.field("filename", "file name") == "x.pdf"


def test_chunks_split_on_repeated_key_not_on_commas():
    assert len(split_chunks(RAW)) == 2


def test_title_resolves_rather_than_becoming_a_question_mark():
    chunks = parse_chunks(RAW)
    assert chunks[0].doc_title == "Personal Account Dealing Standard"
    assert chunks[1].doc_title == "Gifts and Entertainment Policy"


def test_table_is_extracted_and_prose_order_preserved():
    chunk = parse_chunks(RAW)[0]
    assert len(chunk.tables) == 1
    body = chunk.text
    # the sentence that FOLLOWED the table must still follow it, or a
    # qualifier lands against the wrong rows
    assert body.index("row 1") < body.index("Broad-based ETFs are exempt")


def test_value_commas_do_not_break_parsing():
    raw = "**__ALPHA__:**one, two, three**__BETA__:**four"
    fields = {k.lower(): v for k, v in parse_fields(raw)}
    assert fields["alpha"] == "one, two, three"
    assert fields["beta"] == "four"


# --------------------------------------------------------------------------
# clarification and askback
# --------------------------------------------------------------------------
TEMPLATE = """To help you better, please clarify:

1. Which policy area are you asking about ?

option a: Personal Account Dealing
option b: Gifts and Entertainment
option c: Outside Business Interests

2. Which region applies to you ?

option a: EMEA
option b: APAC
option c: Americas

3. Are you an employee or a reviewer ?

option a: Employee
option b: Reviewer
option c: Other"""


def test_template_parses_three_questions_and_nine_options():
    c = parse_clarification(TEMPLATE, MARKERS)
    assert c.is_clarification
    assert c.n_questions == 3
    assert c.n_options == 9


def test_question_without_a_question_mark_is_still_a_question():
    c = parse_clarification("Please clarify:\n1. Which region\noption a: EMEA",
                            MARKERS)
    assert c.n_questions == 1
    assert c.n_options == 1


def test_option_with_empty_text_still_counts():
    c = parse_clarification("Please clarify:\n1. Which region ?\n"
                            "option a:\noption b:", MARKERS)
    assert c.n_options == 2


def test_no_marker_means_not_a_clarification():
    c = parse_clarification("Could you tell me which region you mean?",
                            MARKERS)
    assert not c.is_clarification


def test_askback_is_detected_from_the_previous_assistant_turn():
    history = (Turn("user", "What leave am I entitled to?"),
               Turn("assistant", TEMPLATE))
    is_askback, asked = detect_askback(history, MARKERS)
    assert is_askback
    assert "Which policy area" in asked


def test_askback_false_when_previous_turn_was_a_normal_answer():
    history = (Turn("user", "What leave am I entitled to?"),
               Turn("assistant", "You are entitled to 25 days."))
    assert detect_askback(history, MARKERS)[0] is False


def test_request_matching_an_offered_option_is_identified():
    history = (Turn("user", "What leave?"), Turn("assistant", TEMPLATE))
    assert answers_an_option("Gifts and Entertainment", history,
                             MARKERS) == "Gifts and Entertainment"


# --------------------------------------------------------------------------
# ontology
# --------------------------------------------------------------------------
def test_ontology_names_the_dimension_a_table_varies_by():
    ont = build_ontology(parse_chunks(RAW))
    assert "Personal Account Dealing Standard" in ont.documents
    assert "Region" in ont.table_columns
    # the screener must be told the policy varies by region, or an answer
    # that names no region looks fine
    assert any("region" in d.lower() for d in ont.conditional_dimensions)
    assert ont.alternative_count >= 2


def test_ontology_is_empty_without_retrieval():
    ont = build_ontology([])
    assert ont.documents == ()
    assert ont.conditional_dimensions == ()
    assert "nothing retrieved" in ont.render()


def test_unconditional_single_row_table_is_not_a_conditional_axis():
    raw = ("**__DOCUMENT_TITLE__:**Fees**__TEXT__:**"
           "|Fee|Amount|\n|---|---|\n|Annual|USD 100|")
    ont = build_ontology(parse_chunks(raw))
    assert ont.alternative_count == 0


# --------------------------------------------------------------------------
# table forms beyond the plain pipe case
# --------------------------------------------------------------------------
def test_blank_line_inside_a_table_does_not_end_it():
    """It used to split the header off the body, losing every binding."""
    from policy_audit.parsing.tables import split_segments
    tables = [p for k, p in split_segments(
        "|Band|Approver|\n|---|---|\n\n|up to 10k|Manager|") if k == "table"]
    assert tables
    assert "Approver=Manager" in tables[0].as_rows()[0]


def test_tab_separated_rows_parse():
    from policy_audit.parsing.tables import split_segments
    tables = [p for k, p in split_segments(
        "Band\tApprover\nup to 10k\tManager\nabove 10k\tDivision Head")
        if k == "table"]
    assert tables
    assert tables[0].header == ("Band", "Approver")
    assert "Approver=Division Head" in tables[0].as_rows()[1]


def test_unparseable_table_is_flagged_not_silently_dropped():
    """The judge must know a table resisted structuring."""
    html = parse_chunks("**__DOCUMENT_TITLE__:**P**__TEXT__:**"
                        "<table><tr><th>Band</th></tr>"
                        "<tr><td>10k</td></tr></table>")[0]
    spaced = parse_chunks("**__DOCUMENT_TITLE__:**P**__TEXT__:**"
                          "Band           Approver\n"
                          "up to 10k      Manager\n"
                          "above 10k      Division Head")[0]
    assert html.has_unstructured_table
    assert spaced.has_unstructured_table


def test_prose_and_parsed_tables_are_not_flagged():
    prose = parse_chunks("**__DOCUMENT_TITLE__:**P**__TEXT__:**The approver "
                         "is your line manager below the threshold.")[0]
    piped = parse_chunks("**__DOCUMENT_TITLE__:**P**__TEXT__:**"
                         "|Band|Approver|\n|---|---|\n|10k|Mgr|")[0]
    assert not prose.has_unstructured_table
    assert not piped.has_unstructured_table


def test_ontology_still_finds_the_axis_without_the_prose_scan():
    """The table says it; the prose scan was paying 1.5ms to agree."""
    ont = build_ontology(parse_chunks(
        "**__DOCUMENT_TITLE__:**Approval**__TEXT__:**"
        + "Narrative that mentions region and entity at length. " * 50
        + "\n|Band|Approver|\n|---|---|\n|up to 10k|Manager|\n"
          "|above 10k|Division Head|"))
    assert ont.conditional_dimensions == ("Band",)
    assert ont.alternative_count == 2


# --------------------------------------------------------------------------
# the live clarification template, and coded askback replies
# --------------------------------------------------------------------------
LIVE = """I need a few details before I can answer. Please clarify:

1. What type of instrument are you dealing in?

Option a: Listed equity
Option b: Exchange-traded fund (ETF)
Option c: Fixed income

2. Which region are you based in?

Option a: EMEA
Option b: APAC
Option c: Americas

3. Are you on the covered persons list?

Option a: Yes
Option b: No
Option c: I don't know"""


def test_capitalised_option_lines_parse():
    """`Option a:` is the live wording; a case-sensitive pattern found 0."""
    c = parse_clarification(LIVE, MARKERS)
    assert c.n_questions == 3
    assert c.n_options == 9


def test_marker_alone_makes_it_a_clarification():
    """A body that parses badly is still a clarification, not an answer."""
    c = parse_clarification("Please clarify: I could not follow that.",
                            MARKERS)
    assert c.is_clarification
    assert not c.body_parsed


def test_clarify_marker_outranks_a_static_match():
    from policy_audit.parsing.routing import detect_route
    from policy_audit.domain.taxonomy import Route
    text = ("I am here to assist with policy related documents. "
            "Please clarify:\n1. Which region ?\nOption a: EMEA")
    c = parse_clarification(text, MARKERS)
    route = detect_route(text, c, ("I am here to assist with policy "
                                   "related documents",), has_marker=True)
    assert route is Route.CLARIFIED


def test_coded_reply_resolves_to_the_options_it_names():
    from policy_audit.parsing.clarification import resolve_selections
    picks, readable = resolve_selections("1a, 2b, 3c",
                                         parse_clarification(LIVE, MARKERS))
    assert picks == {"1a": "Listed equity", "2b": "APAC",
                     "3c": "I don't know"}
    assert "Listed equity" in readable
    assert "APAC" in readable


def test_other_coded_shapes_resolve():
    from policy_audit.parsing.clarification import parse_selections
    for text in ("1a, 2b, 3c", "1a 2b 3c", "1-a, 2-b, 3-c", "Q1: a, Q2: b"):
        picks = parse_selections(text)
        assert picks[0] == (1, "a"), text
        assert picks[1] == (2, "b"), text


def test_bare_letters_are_taken_positionally():
    from policy_audit.parsing.clarification import parse_selections
    assert parse_selections("a, b, c") == [(1, "a"), (2, "b"), (3, "c")]


def test_an_out_of_range_selection_is_dropped_not_guessed():
    from policy_audit.parsing.clarification import resolve_selections
    picks, _ = resolve_selections("1a, 9z",
                                  parse_clarification(LIVE, MARKERS))
    assert picks == {"1a": "Listed equity"}


# --------------------------------------------------------------------------
# tables as they actually arrive out of PDF extraction
#
# The dangerous case in this group is not the table that falls through to
# prose - that one is at least visibly unparsed. It is the table that
# parses "successfully" and wrongly: a tidy grid with a value under the
# wrong header, has_unstructured_table False, so nothing is flagged and
# the LLM reader is never called. Every test below exists to make a bad
# parse announce itself.
# --------------------------------------------------------------------------
def test_ragged_pipe_counts_mark_the_parse_unreliable():
    t = parse_table(["|Band|Approver|Notice|",
                     "|up to 10k|Manager|",
                     "|10k-100k|Division Head|10 days|5 days|"])
    assert t is not None
    assert not t.reliable
    assert any("different cell counts" in n for n in t.notes)


def test_padding_a_short_row_invents_a_headerless_column_and_says_so():
    t = parse_table(["|Band|Approver|",
                     "|10k-100k|Division Head|5 days|"])
    assert t is not None and not t.reliable
    assert "col3" in t.header
    assert any("no header" in n for n in t.notes)


def test_header_repeated_at_a_page_break_is_dropped():
    t = parse_table(["|Band|Approver|", "|---|---|",
                     "|up to 10k|Manager|",
                     "|Band|Approver|",
                     "|above 100k|Committee|"])
    assert t is not None
    assert t.rows == (("up to 10k", "Manager"), ("above 100k", "Committee"))
    # dropping it is unambiguous, so the table stays usable
    assert t.reliable
    assert any("page break" in n for n in t.notes)


def test_repeated_header_match_ignores_case_and_spacing():
    t = parse_table(["|Band|Approver|",
                     "|up to 10k|Manager|",
                     "| band |  APPROVER |"])
    assert t is not None and len(t.rows) == 1


def test_a_table_that_is_only_a_repeated_header_is_not_a_table():
    assert parse_table(["|Band|Approver|", "|Band|Approver|"]) is None


def test_merged_cell_is_noted_but_never_forward_filled():
    t = parse_table(["|Region|Band|Approver|", "|---|---|---|",
                     "|EMEA|up to 10k|Manager|",
                     "||10k-100k|Division Head|"])
    assert t is not None
    # the blank stays a blank. Guessing it manufactures the exact
    # misreading the audit exists to catch.
    assert t.rows[1][0] == ""
    assert any("merged with the row above" in n for n in t.notes)
    # the grid itself is right, so re-reading it cannot help: not routed
    assert t.reliable


def test_a_clean_table_stays_reliable_and_unannotated():
    t = parse_table(["|Band|Approver|", "|---|---|",
                     "|up to 10k|Manager|", "|10k-100k|Division Head|"])
    assert t is not None and t.reliable and t.notes == ()


def test_an_unreliable_parse_warns_the_judge_in_the_rendered_text():
    t = parse_table(["|Band|Approver|Notice|", "|up to 10k|Manager|"])
    text = render_table(t)
    assert "CAUTION" in text
    assert "do not raise a table misreading" in text.lower()


def test_a_reliable_parse_carries_no_caution():
    t = parse_table(["|Band|Approver|", "|up to 10k|Manager|"])
    assert "CAUTION" not in render_table(t)


# ---- borderless tables, detected by alignment rather than space count ----
BORDERLESS = ("Band                 Approver             Notice\n"
              "up to USD 10,000     Line Manager         5 days\n"
              "USD 10,000-100,000   Division Head        10 days")


def test_borderless_pdf_table_is_detected():
    assert looks_like_unstructured_table(BORDERLESS)


def test_two_space_gaps_are_enough_when_they_line_up():
    assert looks_like_unstructured_table(
        "Band       Approver       Notice\n"
        "up to 10k  Line Manager   5 days\n"
        "10k-100k   Division Head  10 days")


def test_a_surviving_left_rule_does_not_hide_the_columns():
    # pdftotext keeps the table's left border and drops the rest. One
    # pipe is a border, not a delimiter, and it used to stop the
    # detector dead because the pattern needed a non-pipe first char.
    assert looks_like_unstructured_table(
        "|Band       Approver       Notice\n"
        "|up to 10k  Manager        5 days")


def test_prose_with_double_spaces_is_not_a_table():
    # the spaces are there, but they do not line up from line to line,
    # which is the whole difference between prose and a column layout
    assert not looks_like_unstructured_table(
        "The policy applies to all staff.  Registration is required.\n"
        "Managers approve up to 10k.  Above that, the committee decides.")


def test_a_long_line_beside_a_short_one_is_not_a_table():
    # the gap after "x" is blank on both lines, but "x" has nothing on
    # the far side of it, so there is no second column
    assert not looks_like_unstructured_table("Band                 Approver\nx")


def test_a_shared_indent_is_not_a_column_boundary():
    assert not looks_like_unstructured_table(
        "    Registration is required for gifts above the threshold.\n"
        "    The threshold is reviewed annually by the committee.")


def test_metadata_line_above_a_table_does_not_mask_it():
    # the detector runs per field value, and its window is a suffix. A
    # chunk's metadata is one long dense line directly above the table;
    # a window that kept it could never align, so no borderless table
    # was ever detected in a real chunk.
    chunk = ("**__FILE_NAME__:**Approval_v3.pdf"
             "**__DOCUMENT_TITLE__:**Approval Authority"
             "**__SECTION__:**2 Levels**__TEXT__:**Approver by band:\n"
             + BORDERLESS)
    parsed = parse_chunks(chunk)[0]
    assert not parsed.tables
    assert parsed.has_unstructured_table
