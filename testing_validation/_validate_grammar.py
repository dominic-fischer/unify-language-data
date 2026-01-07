import sys
import json
import yaml
import argparse
import re
from pathlib import Path
from jsonschema import Draft202012Validator
from yaml.parser import ParserError

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------
# YAML PRE-FIXING
# ---------------------------------------------------------------------

def _insert_commas_in_flow_sequences(text: str) -> str:
    """
    Repair common YAML flow sequence errors like:
      [a b c] -> [a, b, c]

    Conservative:
    - Only inside [...]
    - Skips quoted strings
    """
    out = []
    in_brackets = 0
    in_squote = False
    in_dquote = False
    prev_nonspace = ""

    i = 0
    while i < len(text):
        ch = text[i]

        if in_brackets > 0:
            if ch == "'" and not in_dquote:
                in_squote = not in_squote
            elif ch == '"' and not in_squote:
                in_dquote = not in_dquote

        if not in_squote and not in_dquote:
            if ch == "[":
                in_brackets += 1
                prev_nonspace = "["
                out.append(ch)
                i += 1
                continue
            if ch == "]" and in_brackets > 0:
                in_brackets -= 1
                prev_nonspace = "]"
                out.append(ch)
                i += 1
                continue

        if in_brackets > 0 and not in_squote and not in_dquote:
            if ch.isspace():
                j = i
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text):
                    nxt = text[j]
                    if prev_nonspace and prev_nonspace not in "[," and nxt not in "],":
                        out.append(",")
                        prev_nonspace = ","
                out.append(ch)
                i += 1
                continue

        out.append(ch)
        if not ch.isspace():
            prev_nonspace = ch
        i += 1

    return "".join(out)


def _comment_out_line(text: str, line_idx: int) -> str:
    lines = text.splitlines(True)
    if 0 <= line_idx < len(lines):
        if not lines[line_idx].lstrip().startswith("#"):
            lines[line_idx] = re.sub(
                r"^(\s*)",
                r"\1# FIX_REMOVED: ",
                lines[line_idx],
                count=1,
            )
    return "".join(lines)


def load_yaml_maybe_fix(filepath: Path, fix: bool):
    """
    Load YAML.
    In --fix mode, attempts to repair invalid YAML so schema fixing can proceed.
    """
    text = filepath.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(text)
        return ({} if data is None else data), text, False
    except yaml.YAMLError:
        if not fix:
            raise

    changed = False

    # Pass 1: try fixing flow sequences
    repaired = _insert_commas_in_flow_sequences(text)
    if repaired != text:
        text = repaired
        changed = True
        try:
            data = yaml.safe_load(text)
            return ({} if data is None else data), text, True
        except yaml.YAMLError:
            pass

    # Pass 2: comment out offending lines
    for _ in range(50):
        try:
            data = yaml.safe_load(text)
            return ({} if data is None else data), text, True
        except ParserError as e:
            mark = getattr(e, "problem_mark", None)
            if not mark:
                break
            text2 = _comment_out_line(text, mark.line)
            if text2 == text:
                break
            text = text2
            changed = True
        except yaml.YAMLError:
            break

    raise


# ---------------------------------------------------------------------
# SCHEMA FIXING
# ---------------------------------------------------------------------

def _empty_value_for_property(schema: dict, prop: str):
    if not isinstance(schema, dict):
        return {}

    prop_schema = schema.get("properties", {}).get(prop, {})
    t = prop_schema.get("type")

    if t == "object":
        return {}
    if t == "array":
        return []
    if t == "string":
        return ""
    if isinstance(t, list):
        if "object" in t:
            return {}
        if "array" in t:
            return []
        if "string" in t:
            return ""

    return {}


def _get_parent(root, path):
    if not path:
        return None, None
    cur = root
    for p in path[:-1]:
        try:
            cur = cur[p]
        except Exception:
            return None, None
    return cur, path[-1]


def _remove_at_path(root, path) -> bool:
    parent, key = _get_parent(root, path)
    if parent is None:
        return False
    try:
        if isinstance(parent, dict):
            parent.pop(key, None)
            return True
        if isinstance(parent, list) and isinstance(key, int):
            if 0 <= key < len(parent):
                parent.pop(key)
                return True
    except Exception:
        pass
    return False


def _apply_fix(data, err) -> bool:
    path = list(err.path)

    if err.validator == "required":
        missing = err.params.get("property")
        if isinstance(err.instance, dict):
            err.instance[missing] = _empty_value_for_property(err.schema, missing)
            return True
        return False

    if err.validator == "additionalProperties":
        extras = err.params.get("additionalProperties")
        if isinstance(err.instance, dict):
            if isinstance(extras, list):
                for k in extras:
                    err.instance.pop(k, None)
            elif isinstance(extras, str):
                err.instance.pop(extras, None)
            return True

    if path:
        return _remove_at_path(data, path)

    return False


def fix_data(data, schema):
    validator = Draft202012Validator(schema)
    changed = False

    for _ in range(200):
        errors = sorted(
            validator.iter_errors(data),
            key=lambda e: len(list(e.path)),
            reverse=True,
        )
        if not errors:
            return data, changed, []

        for err in errors:
            if _apply_fix(data, err):
                changed = True
                break
        else:
            return data, changed, errors

    return data, changed, list(validator.iter_errors(data))


# ---------------------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------------------

def validate_file(path: Path, schema, fix: bool):
    try:
        data, yaml_text, yaml_changed = load_yaml_maybe_fix(path, fix)
    except Exception as e:
        print(f"❌ {path.name}: YAML parse failed ({e})")
        return

    if fix and yaml_changed:
        path.write_text(yaml_text, encoding="utf-8")

    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))

    if not errors:
        print(f"✅ {path.name} valid")
        return

    if not fix:
        print(f"❌ {path.name} has {len(errors)} error(s):")
        for e in errors:
            loc = ".".join(map(str, e.path)) or "<root>"
            print(f"  - {loc}: {e.message}")
        return

    fixed, changed, remaining = fix_data(data, schema)

    if changed:
        path.write_text(
            yaml.safe_dump(
                fixed,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    if not remaining:
        print(f"🛠️  {path.name} fixed")
    else:
        print(f"⚠️  {path.name} partially fixed ({len(remaining)} remaining errors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    schema_path = Path(__file__).parent / "ref_schemas" / "grammar_schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    target = Path(args.path)

    if target.is_file():
        files = [target]
    else:
        files = sorted(target.glob("*.txt"))

    for f in files:
        validate_file(f, schema, args.fix)


if __name__ == "__main__":
    main()
