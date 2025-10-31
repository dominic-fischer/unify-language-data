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
        print("Usage: python validate.py <folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a folder")
        sys.exit(1)

    # load schema once from the testing reference schemas
    schema_path = (
        Path(__file__).resolve().parent / "ref_schemas" / "grammar_schema.json"
    )
    with open(schema_path) as f:
        schema = json.load(f)

    # validate all .txt files in folder
    files = sorted(folder.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {folder}")
        sys.exit(0)

    for filepath in files:
        validate_file(filepath, schema)


if __name__ == "__main__":
    main()
