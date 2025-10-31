#!/usr/bin/env python3
"""Run all validation scripts for every language folder under `schemas/`.

For each subfolder in `schemas/` this script runs:
 - testing_validation/validate_features.py <folder>
 - testing_validation/validate_grammar.py <folder>
 - testing_validation/validate_grammar_extended.py <folder>

Outputs a short summary per language folder indicating pass/fail for each validator.
"""
from pathlib import Path
import subprocess
import sys
import os

sys.stdout.reconfigure(encoding="utf8")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
VALIDATORS = [
    (
        PROJECT_ROOT / "testing_validation" / "_validate_features.py",
        "_validate_features",
    ),
    (PROJECT_ROOT / "testing_validation" / "_validate_grammar.py", "_validate_grammar"),
    (
        PROJECT_ROOT / "testing_validation" / "_validate_grammar_extended.py",
        "_validate_grammar_extended",
    ),
]


def run_validator(script_path: Path, target_folder: Path) -> tuple[bool, str]:
    """Run a validator script on target_folder. Return (passed, output).
    Uses the same Python executable that's running this script.
    """
    # ensure script exists before running
    if not script_path.exists():
        return False, f"Validator script not found: {script_path}"

    try:
        # Ensure the child process uses UTF-8 for stdout/stderr so emojis and
        # other unicode characters don't raise encoding errors on Windows.
        env = dict(**os.environ)
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(
            [sys.executable, str(script_path), str(target_folder)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except Exception as e:
        return False, str(e)

    out = (proc.stdout or "") + (proc.stderr or "")
    passed = proc.returncode == 0
    return passed, out


def main():
    if not SCHEMAS_DIR.exists():
        print(f"Schemas directory not found: {SCHEMAS_DIR}")
        raise SystemExit(1)

    subfolders = [p for p in sorted(SCHEMAS_DIR.iterdir()) if p.is_dir()]
    if not subfolders:
        print(f"No subfolders found in {SCHEMAS_DIR}")
        raise SystemExit(0)

    for folder in subfolders:
        print(folder.name)
        for script, label in VALIDATORS:
            passed, output = run_validator(script, folder)
            status = "passed" if passed else "failed"
            print(f"  {label}: {status}")
            if not passed:
                # print a compact excerpt of the validator output for debugging
                excerpt = "\n".join((output.strip().splitlines())[:10])
                if excerpt:
                    print("    --- output ---")
                    for line in excerpt.splitlines():
                        print("    ", line)
                    print("    --------------")


if __name__ == "__main__":
    main()
