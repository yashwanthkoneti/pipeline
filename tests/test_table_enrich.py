"""The LLM table pass: when it is called, and what it is allowed to do.

Three outcomes, and the difference between them is the whole design:

  no table parsed, model verified   -> the model's grid is added
  bad parse, model verified         -> the bad grid is REPLACED
  bad parse, model rejected         -> the bad grid is KEPT, carrying its
                                       caution into the text the judge reads

and one non-outcome: a parse that is merely AMBIGUOUS (a merged cell) is
not re-read at all, because the model sees the same blank and is
forbidden to fill it. Only the judge can be told, so only the judge is.
"""

from __future__ import annotations

import json

from policy_audit.agents.client import JsonAgent, ScriptedLLM
from policy_audit.agents.table_reader import TableReader
from policy_audit.domain.models import Interaction, Ontology, Route
from policy_audit.parsing.chunks import parse_chunks
from policy_audit.pipeline.enrich import enrich_tables

ONT = Ontology((), (), (), ())

HEAD = ("**__FILE_NAME__:**Approval_v3.pdf"
        "**__DOCUMENT_TITLE__:**Approval Authority"
        "**__SECTION__:**2 Levels**__TEXT__:**")

BORDERLESS = HEAD + ("Approver by band:\n"
                     "Band                 Approver             Notice\n"
                     "up to USD 10,000     Line Manager         5 days\n"
                     "USD 10,000-100,000   Division Head        10 days")

RAGGED = HEAD + ("Limits:\n|Band|Approver|Notice|\n"
                 "|up to 10k|Manager|\n"
                 "|10k-100k|Division Head|10 days|5 days|\n")

MERGED = HEAD + ("By region:\n|Region|Band|Approver|\n|---|---|---|\n"
                 "|EMEA|up to 10k|Manager|\n||10k-100k|Division Head|\n")

CLEAN = HEAD + ("Bands:\n|Band|Approver|\n|---|---|\n"
                "|up to 10k|Manager|\n|10k-100k|Division Head|\n")


def _rows(chunk_text: str):
    chunks = parse_chunks(chunk_text)
    return [Interaction("T1", "C1", "who approves?", "Your manager.",
                        (), tuple(chunks), ONT, Route.ANSWERED)]


def _reader(reply: str, floor: float = 1.0) -> TableReader:
    return TableReader(JsonAgent(ScriptedLLM([reply]), 1200),
                       verify_floor=floor)


def _grid(header, rows) -> str:
    return json.dumps({"header": header, "rows": rows})


BORDERLESS_OK = _grid(
    ["Band", "Approver", "Notice"],
    [["up to USD 10,000", "Line Manager", "5 days"],
     ["USD 10,000-100,000", "Division Head", "10 days"]])

RAGGED_OK = _grid(["Band", "Approver", "Notice"],
                  [["up to 10k", "Manager", ""],
                   ["10k-100k", "Division Head", "10 days"]])

# "Manager" rewritten as "Line Manager": a paraphrase, and a paraphrased
# policy value is the entire problem - it reads like a quote and cannot
# be checked later.
RAGGED_PARAPHRASED = _grid(["Band", "Approver", "Notice"],
                           [["up to 10k", "Line Manager", ""],
                            ["10k-100k", "Division Head", "10 days"]])


def test_no_table_found_means_the_model_grid_is_added():
    rows = _rows(BORDERLESS)
    assert not rows[0].chunks[0].tables
    assert rows[0].chunks[0].has_unstructured_table

    stats = enrich_tables(rows, _reader(BORDERLESS_OK), mode="fallback")
    assert stats["verified"] == 1 and stats["no_table_found"] == 1
    chunk = rows[0].chunks[0]
    assert len(chunk.tables) == 1
    assert chunk.model_structured_table
    assert "found no table here" in chunk.text


def test_a_bad_parse_is_replaced_not_set_beside_the_good_one():
    rows = _rows(RAGGED)
    assert [t.reliable for t in rows[0].chunks[0].tables] == [False]

    stats = enrich_tables(rows, _reader(RAGGED_OK), mode="fallback")
    assert stats["bad_parse_replaced"] == 1
    chunk = rows[0].chunks[0]
    # the unreliable grid is gone, not sitting next to its replacement
    assert len(chunk.tables) == 1
    assert chunk.tables[0].reliable
    assert "col4" not in chunk.tables[0].header
    assert "REPLACED" in chunk.text


def test_a_rejected_read_keeps_the_bad_parse_and_its_warning():
    rows = _rows(RAGGED)
    stats = enrich_tables(rows, _reader(RAGGED_PARAPHRASED), mode="fallback")
    assert stats["rejected"] == 1 and stats["bad_parse_kept"] == 1
    chunk = rows[0].chunks[0]
    # nothing invented reached the judge, and the parse that remains says
    # it cannot be trusted
    assert not chunk.model_structured_table
    assert "CAUTION" in chunk.text
    assert "STRUCTURED BY MODEL" not in chunk.text


def test_an_ambiguous_but_correct_parse_is_not_re_read():
    rows = _rows(MERGED)
    before = rows[0].chunks[0].text
    stats = enrich_tables(rows, _reader(BORDERLESS_OK), mode="fallback")
    # no call: the model sees the same blank and may not fill it
    assert stats["chunks_attempted"] == 0
    assert rows[0].chunks[0].text == before
    assert "merged with the row above" in before


def test_a_clean_parse_costs_nothing():
    rows = _rows(CLEAN)
    stats = enrich_tables(rows, _reader(BORDERLESS_OK), mode="fallback")
    assert stats["chunks_attempted"] == 0


def test_off_makes_no_calls_at_all():
    rows = _rows(RAGGED)
    stats = enrich_tables(rows, _reader(RAGGED_OK), mode="off")
    assert stats["chunks_attempted"] == 0
    assert not rows[0].chunks[0].model_structured_table


def test_all_mode_re_reads_even_a_clean_parse():
    rows = _rows(CLEAN)
    reply = _grid(["Band", "Approver"],
                  [["up to 10k", "Manager"], ["10k-100k", "Division Head"]])
    stats = enrich_tables(rows, _reader(reply), mode="all")
    assert stats["chunks_attempted"] == 1


def test_the_model_grid_is_assessed_like_the_parsers():
    # a headerless column is unbindable whoever produced it
    rows = _rows(BORDERLESS)
    reply = _grid(["Band", "Approver", ""],
                  [["up to USD 10,000", "Line Manager", "5 days"],
                   ["USD 10,000-100,000", "Division Head", "10 days"]])
    enrich_tables(rows, _reader(reply), mode="fallback")
    table = rows[0].chunks[0].tables[0]
    assert not table.reliable
    assert any("no header" in n for n in table.notes)
    assert "CAUTION" in rows[0].chunks[0].text
