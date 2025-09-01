import sys
import json
import yaml
from pathlib import Path

# File suffixes that require negation handling
NEGATION_FILES = [
    "past.txt", "present.txt", "future.txt", 
    "imperative.txt","subjunctive.txt", "conditional.txt"
]

def make_rule_name(applies: dict) -> str:
    """Generate canonical rule name from applies dict."""
    parts = []
    for feat in sorted(applies.keys()):
        val = applies[feat]
        parts.append(f"{feat}-{val}")
    return "_".join(parts)

def validate_file(path: Path) -> None:
    # Load YAML or JSON
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    errors = []

    for category, cat_obj in data.items():
        rules = cat_obj.get("Rules", {})
        features = cat_obj.get("Features", {})

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

        # --- B) check negation requirement
        filename = path.name.lower()
        if any(filename.endswith(suffix) for suffix in NEGATION_FILES):
            if "Negation" in cat_obj:
                continue  # top-level Negation block satisfies requirement
            else:
                for rule_name, rule_obj in rules.items():
                    if "negation" not in rule_obj:
                        errors.append(
                            f"[{category}] Rule '{rule_name}' missing 'negation' field "
                            f"(required in {filename})"
                        )

    if errors:
        print(f"❌ Validation failed for {path}:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    else:
        print(f"✅ {path}: all checks passed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_rules.py <grammar_file1> [<grammar_file2> ...]")
        sys.exit(1)

    for file in sys.argv[1:]:
        validate_file(Path(file))
