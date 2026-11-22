"""Measure the table parser against the model, on your own data.

    python compare_tables.py

Everything comes from .env, like the main run. This script answers one
question with evidence instead of impression:

    is the deterministic parse actually wrong on my chunks?

It reads your input, finds chunks the parser DID produce a table for,
asks the cheap model to lay the same table out, and diffs the two. Three
outcomes, and each means something different:

    identical    the parse is right and the model adds nothing. If this
                 is most of them, leave PA_LLM_TABLES=fallback alone and
                 do not pay for a call per table.
    differ       one of them moved a value. Read these. The row printed
                 for each says which header a cell landed under in each
                 layout, which is the only thing worth looking at.
    unverified   the model's cells did not all appear in the source, so
                 it was rejected. Common on genuinely damaged tables and
                 not a failure of this script.

It also prints the parse-quality breakdown, which costs nothing: how many
chunks the parser flagged unreliable and why. If that number is high and
`differ` is low, the parser is being honest about hard tables rather than
quietly getting them wrong, and the fallback mode is already doing the
right thing.

Cost: one cheap-model call per table compared, capped by PA_COMPARE_LIMIT
(default 25). Nothing is written to the audit output and no checkpoint is
touched.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                            # so .env resolves
sys.path.insert(0, str(HERE / "src"))     # so it runs without pip install

from policy_audit.agents.client import JsonAgent, OpenAILLM   # noqa: E402
from policy_audit.agents.table_reader import TableReader      # noqa: E402
from policy_audit.config import ENV_FILE, SETTINGS            # noqa: E402
from policy_audit.pipeline.cli import load                    # noqa: E402
from policy_audit.pipeline.enrich import compare_structuring  # noqa: E402
from policy_audit.pipeline.prepare import prepare             # noqa: E402


def main() -> int:
    settings = SETTINGS
    limit = int(os.environ.get("PA_COMPARE_LIMIT", "25"))
    print(f"config from {ENV_FILE}")
    print(f"reading     {settings.input_path}", flush=True)
    df = load(settings.input_path)
    interactions = prepare(df, settings)
    print(f"            {len(interactions):,} row(s) prepared")

    # The free half of the answer first, so a run that is going nowhere
    # can be stopped before it costs anything.
    tables = [t for ix in interactions for c in ix.chunks for t in c.tables]
    unsplittable = sum(1 for ix in interactions for c in ix.chunks
                       if c.has_unstructured_table)
    reasons: Counter = Counter()
    for t in tables:
        if not t.reliable:
            for note in t.notes:
                reasons[note.split(";")[0].split(",")[0][:72]] += 1

    print("\n" + "=" * 70)
    print("WHAT THE PARSER THINKS OF ITS OWN WORK   (no model calls)")
    print("=" * 70)
    print(f"tables parsed          {len(tables)}")
    print(f"  of those, unreliable {sum(1 for t in tables if not t.reliable)}")
    print(f"table shapes unsplit   {unsplittable}")
    if reasons:
        print("\nwhy a parse was called unreliable")
        for reason, count in reasons.most_common():
            print(f"  {count:5d}  {reason}")
    notes_only = sum(1 for t in tables if t.reliable and t.notes)
    if notes_only:
        print(f"\n{notes_only} table(s) parsed cleanly but carry a note "
              f"(a merged cell, a repeated header). Those are read by the "
              f"judge, not re-read by a model.")

    if not tables:
        print("\nno parsed tables to compare")
        return 0
    if settings.preflight_only:
        print("\nPA_PREFLIGHT_ONLY is set; no model calls made")
        return 0

    llm = OpenAILLM(settings.screener_model, settings.temperature,
                    json_mode=settings.json_mode,
                    timeout=settings.request_timeout,
                    base_url=settings.openai_base_url or None)
    llm.smoke_test()
    reader = TableReader(
        JsonAgent(llm, settings.table_reader_max_tokens,
                  min_seconds_between_calls=settings.min_seconds_between_calls,
                  name="tables"),
        verify_floor=settings.table_verify_floor)

    print(f"\ncomparing up to {limit} table(s) against "
          f"{settings.screener_model}", flush=True)
    rows, tally = compare_structuring(interactions, reader, limit=limit)

    print("\n" + "=" * 70)
    print("PARSER vs MODEL")
    print("=" * 70)
    for key in ("compared", "identical", "differ", "unverified"):
        print(f"  {key:12s} {tally[key]}")

    differ = [r for r in rows if r["verdict"] == "differ"]
    if differ:
        print("\nwhere they disagree - read the headers, that is where a "
              "value moves")
        for row in differ:
            print(f"\n{row['transaction_id']}  chunk {row['chunk']}")
            print(f"  parser  {row['parser_header']}")
            print(f"          {str(row['parser_row1'])[:100]}")
            print(f"  model   {row['model_header']}")
            print(f"          {str(row['model_row1'])[:100]}")

    unverified = [r for r in rows if r["verdict"] == "unverified"]
    if unverified:
        print(f"\n{len(unverified)} rejected for not quoting the source:")
        for row in unverified[:5]:
            print(f"  {row['transaction_id']}  {str(row['why'])[:96]}")

    if tally["compared"]:
        agree = tally["identical"] / tally["compared"]
        print("\n" + "-" * 70)
        if agree >= 0.9:
            print(f"{agree:.0%} identical. The parser is handling your "
                  f"tables; leave PA_LLM_TABLES=fallback, which spends a "
                  f"call only where the parse is flagged or absent.")
        else:
            print(f"only {agree:.0%} identical. Read the disagreements "
                  f"above before changing anything - decide which layout "
                  f"is right on a few by hand, because 'differ' alone does "
                  f"not say the model won.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
