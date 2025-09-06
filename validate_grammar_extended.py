import sys
import json
import yaml
from pathlib import Path

# File suffixes that require negation handling
NEGATION_FILES = [
    "past.txt",
    "present.txt",
    "future.txt",
    "imperative.txt",
    "subjunctive.txt",
    "conditional.txt",
    "interrogatives.txt",
]


def make_rule_name(applies: dict) -> str:
    """Generate canonical rule name from applies dict."""
    parts = []
    for feat in sorted(applies.keys()):
        val = applies[feat]
        parts.append(f"{feat}-{val}")
    return "_".join(parts)


def validate_file(path: Path) -> list[str]:
    """Validate a single file, return list of error messages (empty if ok)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    errors = []

    for category, cat_obj in data.items():
        rules = cat_obj.get("Rules", {})

        # --- A) validate rule names
        for rule_name, rule_obj in rules.items():
            applies = rule_obj.get("applies", {})
            if not applies:
                continue
            expected = make_rule_name(applies)
            if rule_name != expected:
                errors.append(
                    f"[{category}] Rule name mismatch: '{rule_name}' should be '{expected}'"
                )

        # --- B) check negation and usage requirements
        filename = path.name.lower()
        if any(filename.endswith(suffix) for suffix in NEGATION_FILES):
            # Check for Negation
            if "Negation" not in cat_obj:
                for rule_name, rule_obj in rules.items():
                    if "negation" not in rule_obj:
                        errors.append(
                            f"[{category}] Rule '{rule_name}' missing 'negation' field "
                            f"(required in {filename})"
                        )
            # Check for Usage
            if "Usage" not in cat_obj:
                for rule_name, rule_obj in rules.items():
                    if "usage" not in rule_obj:
                        errors.append(
                            f"[{category}] Rule '{rule_name}' missing 'usage' field "
                            f"(required in {filename})"
                        )

    return errors


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_rules.py <folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    all_errors = 0
    for path in sorted(folder.glob("*.txt")):
        errors = validate_file(path)
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
