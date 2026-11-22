"""Run the audit from Python. Put this at the project root and execute it.

    python run.py

Equivalent to the `policy-audit` command, but easier to run from an IDE
and easier to edit when you want to change one step.

It chdir's to its own directory first, because .env is resolved relative
to the working directory - which is the one thing that bites when the
same script is launched from a different folder.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                       # so .env and relative paths resolve
sys.path.insert(0, str(HERE / "src"))   # so it runs without pip install -e

from policy_audit.config import SETTINGS            # noqa: E402
from policy_audit.pipeline.cli import run           # noqa: E402

if __name__ == "__main__":
    print(f"project   {HERE}")
    print(f"env file  {HERE / '.env'}  "
          f"{'found' if (HERE / '.env').is_file() else 'MISSING'}")
    print(f"input     {SETTINGS.input_path}")
    print(f"screener  {SETTINGS.screener_model}")
    print(f"judge     {SETTINGS.adjudicator_model}")
    raise SystemExit(run(SETTINGS))
