import sys
import json
import argparse
from pathlib import Path
from ruamel.yaml import YAML

# File suffixes that require negation handling
NEGATION_FILES = [
    "past.txt",
    "present.txt",
    "future.txt",
    "imperative.txt",
    "subjunctive.txt",
    "conditional.txt",
    "conditional-sentences.txt",
    "clauses.txt",
]

USAGE_FILES = [
    "past.txt",
    "present.txt",
    "future.txt",
    "subjunctive.txt",
    "conditional.txt",
]

# configure ruamel.yaml
yaml = YAML()
yaml.preserve_quotes = True
yaml.explicit_start = False
yaml.default_flow_style = False
yaml.width = 1000  # prevent line wrapping


def make_rule_name(applies: dict) -> str:
    """Generate canonical rule name from applies dict."""
    parts = []
    for feat in sorted(applies.keys()):
        val = applies[feat]
        if isinstance(val, list):
            val = "_".join(val)
        parts.append(f"{feat}_{val}")
    return "__".join(parts)


def validate_file(path: Path, fix: bool = False) -> list[str]:
    """Validate (and optionally fix) a single file, return list of errors."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.load(f)

    errors = []
    modified = False

    for category, cat_obj in data.items():
        rules = cat_obj.get("Rules", {})

        # --- A) validate rule names
        for rule_name in list(rules.keys()):
            applies = rules[rule_name].get("applies", {})
            if not applies:
                continue
            expected = make_rule_name(applies)
            if rule_name != expected:
                if fix:
                    rules[expected] = rules.pop(rule_name)
                    modified = True
                else:
                    errors.append(
                        f"[{category}] Rule name mismatch: '{rule_name}' should be '{expected}'"
                    )

        # --- B) check negation and usage requirements
        filename = path.name.lower()
        if any(filename.endswith(suffix) for suffix in NEGATION_FILES + USAGE_FILES):
            if "Negation" not in cat_obj:
                for rule_name, rule_obj in rules.items():
                    if "negation" not in rule_obj:
                        errors.append(
                            f"[{category}] Rule '{rule_name}' missing 'negation' field "
                            f"(required in {filename})"
                        )
            if any(filename.endswith(suffix) for suffix in USAGE_FILES):
                if "Usage" not in cat_obj:
                    errors.append(
                        f"[{category}] Missing top-level 'Usage' field "
                        f"(required in {filename})"
                    )

    # Write back if modified
    if fix and modified:
        if path.suffix == ".json":
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            with path.open("w", encoding="utf-8") as f:
                yaml.dump(data, f)

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with rule files")
    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix rule names in place"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    all_errors = 0
    for path in sorted(folder.glob("*.txt")):
        errors = validate_file(path, fix=args.fix)
        if errors:
            print(f"❌ Validation failed for {path}:")
            for e in errors:
                print("  -", e)
            all_errors += 1
        else:
            print(f"✅ {path}: all checks passed")

    if all_errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
