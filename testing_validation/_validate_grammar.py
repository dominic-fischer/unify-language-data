import sys
import json
import yaml
from pathlib import Path
from jsonschema import Draft202012Validator


def validate_file(filepath: Path, schema) -> None:
    with open(filepath) as f:
        data = yaml.safe_load(f)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

    base = filepath.stem  # filename without extension
    if not errors:
        print(f"✅ {base} is valid")
    else:
        print(f"❌ {base} has {len(errors)} error(s):")
        for err in errors:
            location = ".".join(map(str, err.path)) or "<root>"
            print(f"  - At {location}: {err.message}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate.py <file-or-folder>")
        sys.exit(1)

    path = Path(sys.argv[1])

    # load schema once
    schema_path = (
        Path(__file__).resolve().parent / "ref_schemas" / "grammar_schema.json"
    )
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            print(f"No .txt files found in {path}")
            sys.exit(0)
    else:
        print(f"Error: {path} is not a file or folder")
        sys.exit(1)

    for filepath in files:
        validate_file(filepath, schema)



if __name__ == "__main__":
    main()
