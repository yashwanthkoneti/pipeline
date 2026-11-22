"""Entry point. No arguments - everything comes from .env.

    policy-audit
    python -m policy_audit.pipeline.cli

Preflight runs before any model call and prints what it found. Every
warning it prints has silently invalidated a run before, so it is worth
reading rather than scrolling past: a clarify_markers list that matches
nothing empties a whole category and makes every askback turn look like
a fresh request, and chunk parsing that loses every title takes the
document name out of every finding.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

from ..agents.client import JsonAgent, OpenAILLM
from ..agents.table_reader import TableReader
from ..config import ENV_FILE, SETTINGS, Settings
from ..reporting.export import export, print_summary
from .enrich import enrich_tables
from .prepare import prepare, preflight
from .runner import Runner, read_checkpoint_ids


def _progress(done: int, total: int) -> None:
    print(f"            {done:,}/{total:,}", end="\r", flush=True)


def load(path: str) -> pd.DataFrame:
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def run(settings: Settings = SETTINGS) -> int:
    print(f"config from {ENV_FILE}")
    size = ""
    try:
        mb = os.path.getsize(settings.input_path) / 1e6
        size = f" ({mb:,.0f} MB)"
    except OSError:
        pass
    print(f"reading     {settings.input_path}{size}", flush=True)
    t0 = time.perf_counter()
    df = load(settings.input_path)
    print(f"            {len(df):,} rows in {time.perf_counter()-t0:.1f}s",
          flush=True)

    # Rows already in the checkpoint need no preparation. History is
    # still built from the whole frame; only their chunk parsing and
    # ontology are skipped, which is the part that costs anything.
    only = None
    if settings.resume:
        done = read_checkpoint_ids(settings.checkpoint_path)
        if done:
            tid = settings.col_transaction_id
            col = next((c for c in df.columns
                        if str(c).lower() == tid.lower()), tid)
            todo = {str(v) for v in df[col]} - done
            only = todo
            print(f"checkpoint  {len(done)} row(s) already audited, "
                  f"{len(todo)} to go")

    # prepare() is silent and can run for a while on large chunk
    # columns, which looks identical to a hang. Report it.
    n_todo = len(only) if only is not None else len(df)
    print(f"preparing   {n_todo:,} row(s)", flush=True)
    t0 = time.perf_counter()
    interactions = prepare(df, settings, only=only,
                           progress=_progress if n_todo >= 200 else None)
    print(f"            done in {time.perf_counter()-t0:.1f}s", flush=True)
    if only is not None and not interactions:
        print("\nnothing left to audit; delete the checkpoint to start over")
        return 0

    report = preflight(interactions, settings)
    print(f"\nrows                      {report['rows']}")
    print(f"with retrieval            {report['with_retrieval']}")
    print(f"  of those, titled        {report['with_resolved_title']}")
    print(f"  with a table            {report['with_tables']}")
    print(f"  table parsed badly      {report['unreliable_tables']}")
    print(f"  table not splittable    {report['unsplittable_tables']}")
    print(f"clarification responses   {report['clarification_responses']}")
    print(f"askback turns             {report['askback_turns']}")
    print(f"conditional policy rows   "
          f"{report['rows_with_conditional_policy']}")
    print(f"routes                    {report['routes']}")
    for warning in report["warnings"]:                  # type: ignore
        print(f"\nWARNING: {warning}")

    if settings.preflight_only:
        print("\nPA_PREFLIGHT_ONLY is set; no model calls made")
        return 0

    if report["warnings"]:
        print("\nproceeding with the warnings above. Fix them first if any "
              "look serious - they are cheaper to fix than a run.")

    print("\nchecking the models before spending anything on rows:")
    screener = OpenAILLM(settings.screener_model, settings.temperature,
                         json_mode=settings.json_mode,
                         timeout=settings.request_timeout,
                         base_url=settings.openai_base_url or None)
    adjudicator = OpenAILLM(settings.adjudicator_model, settings.temperature,
                            json_mode=settings.json_mode,
                            timeout=settings.request_timeout,
                            base_url=settings.openai_base_url or None)
    screener.smoke_test()
    if not settings.dry_run:
        adjudicator.smoke_test()

    # Tables, before any judging. A chunk whose table the parser could
    # not split, or split wrongly, is read by the cheap model here so the
    # gates see a grid instead of prose or a shifted one.
    if settings.llm_tables != "off":
        reader = TableReader(
            JsonAgent(screener, settings.table_reader_max_tokens,
                      min_seconds_between_calls=settings.min_seconds_between_calls,
                      name="tables"),
            verify_floor=settings.table_verify_floor)
        bad = sum(1 for ix in interactions for c in ix.chunks
                  if c.has_unstructured_table
                  or any(not t.reliable for t in c.tables))
        print(f"\ntables      {bad} chunk(s) need structuring "
              f"(mode={settings.llm_tables})", flush=True)
        t0 = time.perf_counter()
        tstats = enrich_tables(interactions, reader,
                              mode=settings.llm_tables)
        print(f"            {tstats['verified']} verified, "
              f"{tstats['rejected']} rejected, "
              f"{tstats['bad_parse_replaced']} bad parse(s) replaced, "
              f"{tstats['bad_parse_kept']} kept with a caution "
              f"in {time.perf_counter()-t0:.1f}s")
        if tstats["bad_parse_kept"]:
            print("            a kept bad parse carries its warning into "
                  "the chunk text, so the judge is told not to raise a "
                  "table misreading on it")

    runner = Runner(screener, adjudicator, settings)
    results = runner.run(interactions, resume=settings.resume)

    out = export(results, interactions, settings.output_dir)
    print_summary(out["summary"])                       # type: ignore
    print(f"\ncall stats {runner.call_stats}")
    for path in out["paths"].values():                  # type: ignore
        print(f"wrote {path}")
    return 0


def main() -> int:
    return run(SETTINGS)


if __name__ == "__main__":
    sys.exit(main())
