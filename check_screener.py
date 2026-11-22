"""Why did nothing escalate?  Run: python check_screener.py

Reads out/checkpoint.jsonl - no model calls, no cost - and reports what
the screener actually said, per check. Zero adjudications has only a few
possible causes and this separates them.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE / "src"))

from policy_audit.config import SETTINGS                  # noqa: E402

path = SETTINGS.checkpoint_path or "out/checkpoint.jsonl"
if not Path(path).is_file():
    print(f"no checkpoint at {path}")
    print("  -> nothing to inspect. Was PA_CHECKPOINT_PATH set?")
    raise SystemExit(1)

rows = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

print(f"checkpoint {path}")
print(f"rows       {len(rows)}\n")

print(f"PA_DRY_RUN is {SETTINGS.dry_run}"
      f"{'   <-- THIS IS WHY: dry run skips the adjudicator entirely'
        if SETTINGS.dry_run else ''}\n")

routes = Counter(r.get("route") for r in rows)
print(f"routes     {dict(routes)}")
calls_s = sum(r.get("screener_calls", 0) for r in rows)
calls_a = sum(r.get("adjudicator_calls", 0) for r in rows)
print(f"calls      screener {calls_s}   adjudicator {calls_a}\n")

errs = [e for r in rows for e in (r.get("errors") or [])]
if errs:
    print(f"ERRORS on {len(errs)} call(s) - the screener may not be "
          f"answering at all:")
    for msg, n in Counter(errs).most_common(5):
        print(f"  {n:4d}  {msg[:90]}")
    print()

verdicts = defaultdict(Counter)
reasons = defaultdict(list)
for r in rows:
    for s in r.get("screened") or []:
        verdicts[s["check"]][s["verdict"]] += 1
        if s["verdict"] != "clear" and s.get("reason"):
            reasons[s["check"]].append(s["reason"])

if not verdicts:
    print("NO CHECKS WERE RECORDED AT ALL.")
    print("  Either every row took a route with no applicable checks, or")
    print("  the screener replies were unparseable. Check 'errors' above.")
    raise SystemExit(0)

total = sum(sum(c.values()) for c in verdicts.values())
esc = sum(c["suspect"] + c["unsure"] for c in verdicts.values())
print(f"{'check':34s} {'clear':>7s} {'suspect':>8s} {'unsure':>7s}")
print("-" * 60)
for check in sorted(verdicts, key=lambda k: -sum(verdicts[k].values())):
    c = verdicts[check]
    print(f"{check:34s} {c['clear']:7d} {c['suspect']:8d} {c['unsure']:7d}")
print("-" * 60)
print(f"{'TOTAL':34s} {total - esc:7d} {esc:8d}")
print(f"\nescalation rate {100 * esc / max(total, 1):.1f}%")

print()
if esc == 0:
    print("The screener marked EVERY check clear.")
    print("  Plausible, if your agent is in good shape - but verify it is")
    print("  actually reading the prompt before believing it. Try:")
    print("    PA_LIMIT=5 PA_SCREENER_MODEL=<the larger model> python run.py")
    print("  If the larger model escalates on the same five rows, the")
    print("  small one is not following the instruction, not agreeing.")
elif calls_a == 0:
    print(f"{esc} check(s) escalated but the adjudicator was never called.")
    print("  That is PA_DRY_RUN=true, or the run stopped early.")
else:
    for check, rs in sorted(reasons.items(),
                            key=lambda kv: -len(kv[1]))[:3]:
        print(f"{check}: {rs[0][:100]}")
