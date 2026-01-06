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

        # --- B) check that 'Features' is exactly the set of features and their attributes in 'applies'
        expected_features = {}
        for rule_obj in rules.values():
            applies = rule_obj.get("applies", {})
            for feat, val in applies.items():
                if feat not in expected_features:
                    expected_features[feat] = set()
                if isinstance(val, list):
                    for v in val:
                        expected_features[feat].add(v)
                else:
                    expected_features[feat].add(val)
            # features may also be inside 'endings'['features']
            endings = rule_obj.get("endings", {}) # this is a list of endings
            for ending in endings:
                ending_features = ending.get("features", {})
                for feat, val in ending_features.items():
                    if feat not in expected_features:
                        expected_features[feat] = set()
                    if isinstance(val, list):
                        for v in val:
                            expected_features[feat].add(v)
                    else:
                        expected_features[feat].add(val)
            
        features_obj = cat_obj.get("Features", {})
        # Check for missing and extra features; fix if flag is set
        for feat, vals in expected_features.items():
            if feat not in features_obj:
                if fix:
                    # the attributes are just a sorted list
                    features_obj[feat] = sorted(list(vals))
                    modified = True
                else:
                    errors.append(
                        f"[{category}] Missing feature '{feat}' in 'Features' section"
                    )
            else:
                existing_vals = set(features_obj[feat])
                missing_vals = vals - existing_vals
                extra_vals = existing_vals - vals
                if missing_vals or extra_vals:
                    if fix:
                        features_obj[feat] = sorted(list(vals))
                        modified = True
                    else:
                        if missing_vals:
                            errors.append(
                                f"[{category}] Feature '{feat}' missing attributes "
                                f"{sorted(list(missing_vals))} in 'Features' section"
                            )
                        if extra_vals:
                            errors.append(
                                f"[{category}] Feature '{feat}' has extra attributes "
                                f"{sorted(list(extra_vals))} in 'Features' section"
                            )
                

        # --- C) check negation and usage requirements
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
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="File or folder with rule files")
    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix rule names in place"
    )
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(path.glob("*.txt"))
    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)

    all_errors = 0
    for p in paths:
        errors = validate_file(p, fix=args.fix)
        if errors:
            print(f"❌ Validation failed for {p}:")
            for e in errors:
                print("  -", e)
            all_errors += 1
        else:
            print(f"✅ {p}: all checks passed")

    sys.exit(1 if all_errors else 0)



if __name__ == "__main__":
    main()
