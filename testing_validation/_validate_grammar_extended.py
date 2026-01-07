import sys
import json
import argparse
import re
from io import StringIO
from pathlib import Path
from ruamel.yaml import YAML

# schema_first_lines.json is a normal json in the same directory
SCHEMA_FIRST_LINES = json.loads(Path("testing_validation/schema_first_lines.json").read_text(encoding="utf-8"))


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

# We'll still do our own merging, but this prevents ruamel from hard-failing
# if something slips through.
yaml.allow_duplicate_keys = True

def get_first_content_line(text: str) -> tuple[int, str] | None:
    """
    Returns (line_index, line_text) of the first non-empty, non-comment line.
    """
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return i, stripped
    return None


def make_rule_name(applies: dict) -> str:
    """Generate canonical rule name from applies dict."""
    parts = []
    for feat in sorted(applies.keys()):
        val = applies[feat]
        if isinstance(val, list):
            val = "_".join(val)
        parts.append(f"{feat}_{val}")
    return "__".join(parts)


# ---------------------------------------------------------------------
# Duplicate-key repair (text-level) for ruamel.yaml
# ---------------------------------------------------------------------

_KEY_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[^:#\n]+?)\s*:\s*(?P<val>.*?)\s*$")


def _parse_flow_list(val: str) -> list[str] | None:
    """
    Parse a simple one-line flow list like: [a, b, c]
    Returns list of items or None if not a flow list.
    """
    s = val.strip()
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [x.strip() for x in inner.split(",") if x.strip()]
    return None


def _format_flow_list(items: list[str]) -> str:
    # de-dup while preserving order
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return "[" + ", ".join(out) + "]"


def merge_duplicate_keys_text(text: str) -> tuple[str, bool]:
    """
    Best-effort fixer for YAML duplicate keys in the same mapping level.

    Strategy:
    - Track mapping contexts by indentation.
    - When a duplicate key line is found at the same indent/context:
        * merge values into a flow list on the first occurrence
        * comment out the duplicate occurrence line
    - Only handles single-line scalar values and single-line flow lists.
      (If value is empty / block scalar / complex, we skip.)
    """
    lines = text.splitlines(True)  # keep line endings
    changed = False

    # stack of (indent_len, context_dict)
    # context_dict: key -> {"first_idx": int, "values": [str]}
    stack: list[tuple[int, dict]] = [(-1, {})]

    def current_context(indent_len: int) -> dict:
        # unwind stack for dedent
        while stack and indent_len <= stack[-1][0]:
            stack.pop()
        # ensure a context exists for this indent
        if not stack or stack[-1][0] != indent_len:
            stack.append((indent_len, {}))
        return stack[-1][1]

    for i, raw in enumerate(lines):
        # ignore full-line comments and list items
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue

        m = _KEY_LINE_RE.match(raw.rstrip("\n\r"))
        if not m:
            continue

        indent = m.group("indent")
        key = m.group("key").strip()
        val = m.group("val")

        # skip keys with no value (could start a nested mapping)
        if val is None:
            continue
        val_stripped = val.strip()

        # skip complex/multiline indicators
        if val_stripped == "" or val_stripped in ("|", ">") or val_stripped.startswith(("&", "*", "{")):
            continue

        indent_len = len(indent)
        ctx = current_context(indent_len)

        # If value looks like an inline comment-only value, skip
        if val_stripped.startswith("#"):
            continue

        # If duplicate: merge
        if key in ctx:
            first_idx = ctx[key]["first_idx"]
            values = ctx[key]["values"]

            # merge current value into list
            cur_list = _parse_flow_list(val_stripped)
            if cur_list is not None:
                values.extend(cur_list)
            else:
                values.append(val_stripped)

            # update first line to be a flow list
            # first line might already be a flow list or scalar
            first_line = lines[first_idx].rstrip("\n\r")
            fm = _KEY_LINE_RE.match(first_line)
            if fm:
                first_val = fm.group("val").strip()
                first_list = _parse_flow_list(first_val)
                if first_list is not None:
                    merged = first_list + [v for v in values if v not in first_list]
                else:
                    merged = [first_val] + [v for v in values if v != first_val]

                new_first = f"{fm.group('indent')}{fm.group('key').strip()}: {_format_flow_list(merged)}"
                # preserve original newline
                nl = "\n" if lines[first_idx].endswith("\n") else ""
                lines[first_idx] = new_first + nl

            # comment out duplicate line
            nl = "\n" if lines[i].endswith("\n") else ""
            lines[i] = f"{indent}# FIX_REMOVED_DUPKEY: {key}: {val_stripped}{nl}"

            changed = True
        else:
            # first occurrence: remember it
            ctx[key] = {"first_idx": i, "values": []}

    return "".join(lines), changed


# ---------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------

def validate_file(path: Path, fix: bool = False) -> list[str]:
    """Validate (and optionally fix) a single file, return list of errors."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        text = path.read_text(encoding="utf-8")

        # Only in --fix mode, rewrite duplicates so ruamel can load it cleanly.
        if fix:
            text2, changed = merge_duplicate_keys_text(text)
            if changed:
                path.write_text(text2, encoding="utf-8")
                text = text2

        with StringIO(text) as f:
            data = yaml.load(f)

    errors = []
    modified = False

        # --- 0) First-line schema check
    # remove whats before the first _ and after the last _
    mod_path = path.name.split("_", 1)[-1].rsplit("_", 1)[0]
    expected_first = SCHEMA_FIRST_LINES.get(mod_path)
    if expected_first:
        text = path.read_text(encoding="utf-8")
        first = get_first_content_line(text)

        if first is None:
            errors.append(
                f"[{path.name}] File is empty; expected first line '{expected_first}'"
            )
        else:
            idx, actual = first
            if actual != expected_first:
                if fix:
                    lines = text.splitlines(True)
                    lines[idx] = expected_first + "\n"
                    path.write_text("".join(lines), encoding="utf-8")
                else:
                    errors.append(
                        f"[{path.name}] First line mismatch: "
                        f"expected '{expected_first}', got '{actual}'"
                    )

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
            endings = rule_obj.get("endings", {})  # this is a list of endings
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
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            with path.open("w", encoding="utf-8") as f:
                yaml.dump(data, f)

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="File or folder with rule files")
    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix in place"
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
