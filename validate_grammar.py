import sys
import json
import yaml
from pathlib import Path
from jsonschema import Draft202012Validator

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate.py <filename_without_txt>")
        sys.exit(1)

    # build path
    folder = "YAML_schemas_DE"
    base = sys.argv[1]
    filepath = Path(folder) / f"{base}.txt"

    # load schema
    with open("grammar_schema.json") as f:
        schema = json.load(f)

    # load data (YAML → Python dict)
    with open(filepath) as f:
        data = yaml.safe_load(f)

    # validate
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

    if not errors:
        print(f"✅ {base} is valid")
    else:
        print(f"❌ {base} has {len(errors)} error(s):")
        for err in errors:
            # Show *where* and *what*
            location = ".".join(map(str, err.path)) or "<root>"
            print(f"  - At {location}: {err.message}")

if __name__ == "__main__":
    main()
