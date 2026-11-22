"""Optional LLM pass to structure tables the parser could not.

Kept out of prepare() on purpose. Preparation is deterministic and
testable without an endpoint, and this is neither. It runs after, as a
pass that rewrites the chunks of rows that need it and leaves the rest
alone.

Three modes, set by PA_LLM_TABLES:

  off        deterministic parsing only
  fallback   (default) two cases. Where the parser found NO table but the
             text looks like one - HTML, borderless columns, a surviving
             left rule - the judge's alternative is unstructured prose,
             so a verified structuring is strictly better. And where the
             parser DID produce a table but marked it unreliable, which
             is the more dangerous case: ragged pipe counts and headerless
             columns parse into a grid that looks exactly as authoritative
             as a correct one.
  all        every table, including ones the parser handled well. Only
             useful with compare_tables.py, to measure whether the parse
             is actually wrong on your data.

An unreliable parse is REPLACED, not supplemented. Showing the judge two
grids for one table and leaving it to pick is not caution, it is noise -
and the parse has already been established as the one that may have
shifted a column.

A table the model structured is marked on the chunk, and the mark travels
into the rendered text the judge reads. That matters: verification proves
no cell was invented, but a model could still place real values under the
wrong headers, and that is precisely the error TABLE_MISREAD hunts. So a
finding resting on a model-structured table is knowable as such rather
than indistinguishable from one resting on a parsed table.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from ..agents.table_reader import TableReader
from ..domain.models import Chunk, Interaction
from ..parsing.tables import render_table


def _rerender(chunk: Chunk, tables: Sequence, *, superseded: bool) -> str:
    """Rebuild the chunk text with the structured tables appended.

    The original text is kept. The model's layout is additive, so a
    reader - human or judge - can still see what it was derived from, and
    a misalignment is detectable rather than hidden behind a tidy grid.
    """
    parts = [chunk.text]
    for table in tables:
        head = ("STRUCTURED BY MODEL. The automatic parse of this table was "
                "unreliable and is REPLACED by the layout below; ignore any "
                "earlier grid for it."
                if superseded else
                "STRUCTURED BY MODEL (the automatic parser found no table "
                "here).")
        parts.append(
            head + " Cell text was verified to appear verbatim in the source "
            "above, so nothing is invented; the column alignment is the "
            "model's, not the parser's.\n" + render_table(table))
    return "\n\n".join(parts)


def enrich_tables(interactions: Sequence[Interaction], reader: TableReader,
                  *, mode: str = "fallback") -> Dict[str, int]:
    """Structure tables in place. Returns counts for the run summary."""
    stats = {"rows_touched": 0, "chunks_attempted": 0, "verified": 0,
             "rejected": 0, "no_table_found": 0, "bad_parse_replaced": 0,
             "bad_parse_kept": 0}
    if mode == "off":
        return stats

    for i, ix in enumerate(interactions):
        new_chunks: List[Chunk] = []
        touched = False
        for chunk in ix.chunks:
            bad = [t for t in chunk.tables if not t.reliable]
            if mode == "fallback":
                needs = bool(chunk.has_unstructured_table or bad)
            else:
                needs = bool(chunk.has_unstructured_table or chunk.tables)
            if not needs:
                new_chunks.append(chunk)
                continue

            stats["chunks_attempted"] += 1
            result = reader.read(chunk.text)
            if not result.verified or result.table is None:
                # The parse stays, unreliable and saying so. render_table
                # has already put the caution in the chunk text, which is
                # the whole reason that caution exists.
                stats["rejected"] += 1
                if bad:
                    stats["bad_parse_kept"] += 1
                new_chunks.append(chunk)
                continue

            stats["verified"] += 1
            touched = True
            if bad:
                stats["bad_parse_replaced"] += 1
                kept = tuple(t for t in chunk.tables if t.reliable)
            else:
                stats["no_table_found"] += 1
                kept = tuple(chunk.tables)
            new_chunks.append(replace(
                chunk,
                tables=kept + (result.table,),
                text=_rerender(chunk, [result.table], superseded=bool(bad)),
                model_structured_table=True,
            ))
        if touched:
            stats["rows_touched"] += 1
            interactions[i] = replace(ix, chunks=tuple(new_chunks))
    return stats


def compare_structuring(interactions: Sequence[Interaction],
                        reader: TableReader, *, limit: int = 25
                        ) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Run the model on tables the PARSER already handled, and diff.

    This answers "is the deterministic parse actually bad?" with evidence
    rather than impression. Where they agree, the parse is fine and the
    model is not needed. Where they differ, the diff says which one moved
    a value - and a cell appearing under a different header in the two
    layouts is the signal worth reading.
    """
    rows: List[Dict[str, object]] = []
    tally = {"compared": 0, "identical": 0, "differ": 0, "unverified": 0}

    for ix in interactions:
        for chunk in ix.chunks:
            if not chunk.tables or len(rows) >= limit:
                continue
            parsed = chunk.tables[0]
            result = reader.read(chunk.text)
            tally["compared"] += 1
            if not result.verified or result.table is None:
                tally["unverified"] += 1
                rows.append({"transaction_id": ix.transaction_id,
                             "chunk": chunk.index, "verdict": "unverified",
                             "why": result.reason})
                continue
            model = result.table
            same = (tuple(parsed.header) == tuple(model.header)
                    and tuple(parsed.rows) == tuple(model.rows))
            tally["identical" if same else "differ"] += 1
            rows.append({
                "transaction_id": ix.transaction_id,
                "chunk": chunk.index,
                "verdict": "identical" if same else "differ",
                "parser_header": " | ".join(parsed.header),
                "model_header": " | ".join(model.header),
                "parser_rows": len(parsed.rows),
                "model_rows": len(model.rows),
                "parser_row1": (parsed.as_rows()[0] if parsed.rows else ""),
                "model_row1": (model.as_rows()[0] if model.rows else ""),
            })
    return rows, tally
