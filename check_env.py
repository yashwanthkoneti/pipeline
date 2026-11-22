"""Why is my .env not being picked up?  Run: python check_env.py

Prints what the loader actually sees, in order, so the cause is visible
rather than guessed at.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

print(f"1. working directory   {Path.cwd()}")

print("\n2. files here that look like an env file:")
found = sorted(p.name for p in HERE.iterdir()
               if "env" in p.name.lower() and p.is_file())
for name in found:
    size = (HERE / name).stat().st_size
    marker = "  <-- this is the one that gets read" if name == ".env" else ""
    print(f"     {name:24s} {size:6d} bytes{marker}")
if ".env" not in found:
    print("     NO FILE NAMED EXACTLY '.env'")
    if "policy_audit.env" in found:
        print("     -> rename it:  mv policy_audit.env .env")
    if ".env.txt" in found:
        print("     -> Notepad added .txt:  mv .env.txt .env")
    if ".env.example" in found and len(found) == 1:
        print("     -> you may have edited .env.example instead;")
        print("        copy it:  cp .env.example .env")

print("\n3. what python-dotenv parses from it:")
try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:
    print("     python-dotenv NOT INSTALLED")
    print("     -> pip install python-dotenv")
    raise SystemExit(1)

values = dotenv_values(".env") if Path(".env").is_file() else {}
if not values:
    print("     nothing parsed (file missing or empty)")
else:
    for key in ("PA_INPUT_PATH", "PA_OUTPUT_PATH", "PA_SCREENER_MODEL",
                "OPENAI_API_KEY"):
        raw = values.get(key)
        if raw is None:
            print(f"     {key:22s} NOT PRESENT in the file")
        elif key == "OPENAI_API_KEY":
            print(f"     {key:22s} {'set' if raw and raw != 'sk-...' else 'STILL THE PLACEHOLDER sk-...'}")
        else:
            print(f"     {key:22s} {raw!r}")

print("\n4. was the value already in the real environment?")
# load_dotenv uses override=False, so a shell variable wins over the file
shadowed = [k for k in values if k in os.environ and os.environ[k] != values[k]]
if shadowed:
    print(f"     these shell variables OVERRIDE the file: {shadowed}")
    print("     -> unset them, or they keep winning")
else:
    print("     no - nothing in the shell is shadowing the file")

print("\n5. what Settings ends up with:")
import sys
sys.path.insert(0, str(HERE / "src"))
from policy_audit.config import SETTINGS
print(f"     input_path        {SETTINGS.input_path!r}")
print(f"     exists            {Path(SETTINGS.input_path).is_file()}")
print(f"     screener_model    {SETTINGS.screener_model!r}")

if SETTINGS.input_path == "data/audit_input.csv":
    print("\n  input_path is still the built-in default, so PA_INPUT_PATH")
    print("  is not reaching Settings. Steps 2-4 above say why.")
elif not Path(SETTINGS.input_path).is_file():
    print(f"\n  .env IS being read - the path in it just does not exist.")
    print(f"  Looked for: {Path(SETTINGS.input_path).resolve()}")
    print(f"  Paths are relative to {Path.cwd()}, so use an absolute path")
    print(f"  or put the CSV under that folder.")
else:
    print("\n  all good - run: python run.py")
