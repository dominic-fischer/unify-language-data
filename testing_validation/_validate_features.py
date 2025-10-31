import sys
import json
from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print(f">>>{path}")
    raise FileNotFoundError


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def check_features_in_applies(applies: dict, base_path: Path, errors: list[str]):
    unimorph_schema = load_json(
        base_path / "unimorph_schema_w_expl.json"
    )
    custom_path = base_path / "custom_schema.json"
    custom_schema = load_json(custom_path)

    modified_custom = False
    modified_lang = {}

    for feat, val in applies.items():
        vals = val if isinstance(val, list) else [val]

        if feat.startswith("u-"):
            clean_feat = feat
            if clean_feat not in unimorph_schema:
                errors.append(f"❌ Unknown UniMorph feature: {feat}")
                continue
            for v in vals:
                if v.startswith("x-"):
                    found = any(v in vs for vs in custom_schema.values())
                    if not found:
                        print(
                            f"❓ Value {v} not found in custom_schema.json — check if it should be added."
                        )
                else:
                    if v not in unimorph_schema[clean_feat]:
                        errors.append(f"❌ Unknown UniMorph value: {feat}={v}")

        elif feat.startswith("x-"):
            if feat not in custom_schema:
                print(f"🟢 Not yet in dict custom_schema.json, adding {feat}")
                custom_schema[feat] = []
                modified_custom = True
            for v in vals:
                if v not in custom_schema.get(feat, []):
                    print(f"🟢 Not yet in dict custom_schema.json, adding {feat}={v}")
                    custom_schema[feat].append(v)
                    modified_custom = True

        else:
            prefix = feat.split("-")[0]
            lang_dir = base_path
            lang_file = lang_dir / f"lang-{prefix}_schema.json"
            # if directory doesn't exist, create dir + file together
            if not lang_dir.exists():
                lang_dir.mkdir(parents=True, exist_ok=True)
                save_json(lang_file, {})  # create new file with empty dict

            schema = load_json(lang_file)

            if feat not in schema:
                print(f"🟢 Not yet in {lang_file.name}, adding {feat}")
                schema[feat] = []
                modified_lang[lang_file] = schema
            for v in vals:
                if v not in schema.get(feat, []):
                    print(f"🟢 Not yet in {lang_file.name}, adding {feat}={v}")
                    schema[feat].append(v)
                    modified_lang[lang_file] = schema

    # save changes if any
    if modified_custom:
        save_json(custom_path, custom_schema)
    if modified_lang:
        for f, schema in modified_lang.items():
            save_json(f, schema)


def validate_file(path: Path, base_path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.load(f)

    errors = []

    for _, cat_obj in data.items():
        # 1) Features
        for feat, vals in cat_obj.get("Features", {}).items():
            check_features_in_applies({feat: vals}, base_path, errors)

        # 2) Usage
        for _, usage_obj in cat_obj.get("Usage", {}).items():
            applies = usage_obj.get("applies", {})
            if applies:
                check_features_in_applies(applies, base_path, errors)

        # 3) Negation
        neg = cat_obj.get("Negation", {})
        if "applies" in neg:
            check_features_in_applies(neg["applies"], base_path, errors)

        # 4) Rules
        for _, rule_obj in cat_obj.get("Rules", {}).items():
            applies = rule_obj.get("applies", {})
            if applies:
                check_features_in_applies(applies, base_path, errors)

    return errors


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with rule files")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    # base_path points to the repository's reference schemas used for validation
    # (ref_schemas is located next to this script)
    base_path = Path(__file__).resolve().parent / "ref_schemas"

    all_errors = 0
    for path in sorted(folder.glob("*")):
        if path.suffix not in [".txt", ".yaml", ".yml", ".json"]:
            continue
        errors = validate_file(path, base_path)
        if errors:
            print(f"❌ {path}:")
            for e in errors:
                print("  -", e)
            all_errors += 1
        else:
            print(f"✅ {path}: all features/values validated")

    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
