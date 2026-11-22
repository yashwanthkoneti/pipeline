"""Where does prepare() actually spend its time on YOUR data?

    python profile_prepare.py

Reads the input named in .env, reports the shape of the data and a
per-stage timing breakdown, and makes no model calls. Run it instead of
guessing - every slowness theory I had was wrong against synthetic data,
so the only useful numbers are yours.
"""

import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE / "src"))

import pandas as pd                                        # noqa: E402

from policy_audit.config import SETTINGS                   # noqa: E402
from policy_audit.parsing.chunks import parse_chunks       # noqa: E402
from policy_audit.parsing.clarification import (           # noqa: E402
    detect_askback, parse_clarification)
from policy_audit.parsing.ontology import build_ontology   # noqa: E402
from policy_audit.parsing.routing import detect_route      # noqa: E402
from policy_audit.pipeline.prepare import (build_histories,  # noqa: E402
                                           prepare)

LIMIT = int(os.environ.get("PA_PROFILE_ROWS", "200"))


def main() -> None:
    path = SETTINGS.input_path
    print(f"input {path}")
    t = time.perf_counter()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    t_read = time.perf_counter() - t
    print(f"read {len(df):,} rows in {t_read:.2f}s\n")

    # ---- the shape of the data, which is what decides the cost -------
    cols = {str(c).lower(): c for c in df.columns}
    ch = cols.get(SETTINGS.col_chunks.lower())
    rq = cols.get(SETTINGS.col_request.lower())
    rs = cols.get(SETTINGS.col_response.lower())
    cv = cols.get(SETTINGS.col_conversation_id.lower())

    print("SHAPE")
    for label, col in (("document_chunks", ch), ("request", rq),
                       ("response", rs)):
        if not col:
            print(f"  {label:16s} column not found")
            continue
        lens = df[col].astype(str).str.len()
        print(f"  {label:16s} mean {lens.mean():10,.0f}  "
              f"p95 {lens.quantile(.95):10,.0f}  max {lens.max():10,.0f} chars")
    if cv:
        per = df[cv].value_counts()
        print(f"  conversations    {len(per):,}  "
              f"turns: mean {per.mean():.1f}  max {per.max()}")
    if ch:
        total = df[ch].astype(str).str.len().sum()
        print(f"\n  total chunk text {total/1e6:,.1f} MB across the file")
        if df[ch].astype(str).str.len().max() > 2_000_000:
            print("  WARNING: at least one chunk cell is over 2 MB. That is "
                  "the cost. Truncate retrieval upstream rather than "
                  "parsing it here.")

    # ---- per-stage timing on a sample --------------------------------
    sample = df.head(LIMIT)
    print(f"\nTIMING on the first {len(sample)} rows")

    t = time.perf_counter(); build_histories(sample, SETTINGS)
    print(f"  build_histories   {time.perf_counter()-t:7.2f}s")

    stages = Counter()
    for raw in sample.to_dict("records"):
        chunks_raw = raw.get(ch) if ch else ""
        response = str(raw.get(rs) or "") if rs else ""

        t = time.perf_counter(); chunks = parse_chunks(chunks_raw)
        stages["parse_chunks"] += time.perf_counter() - t

        t = time.perf_counter(); build_ontology(chunks)
        stages["build_ontology"] += time.perf_counter() - t

        t = time.perf_counter()
        clar = parse_clarification(response, SETTINGS.clarify_markers)
        stages["parse_clarification"] += time.perf_counter() - t

        t = time.perf_counter()
        detect_route(response, clar, SETTINGS.static_responses)
        stages["detect_route"] += time.perf_counter() - t

    for name, secs in stages.most_common():
        per_row = secs / max(len(sample), 1)
        print(f"  {name:18s}{secs:7.2f}s   {per_row*1000:7.2f} ms/row   "
              f"-> {per_row*len(df):7.1f}s for all {len(df):,}")

    t = time.perf_counter(); prepare(sample, SETTINGS)
    t_prep = time.perf_counter() - t
    print(f"\n  prepare() total   {t_prep:7.2f}s   "
          f"{t_prep/max(len(sample),1)*1000:7.2f} ms/row   "
          f"-> {t_prep/max(len(sample),1)*len(df):7.1f}s for all {len(df):,}")
    print(f"\n  (set PA_PROFILE_ROWS to sample more or fewer than {LIMIT})")


if __name__ == "__main__":
    main()
