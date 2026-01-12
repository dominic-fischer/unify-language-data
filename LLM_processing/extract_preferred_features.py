from pathlib import Path
import json
from typing import Dict, List, Any
import os
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "schemas" / "lang-de"
INPUT_UNIMORPH = BASE_DIR / "testing_validation" / "ref_schemas" / "unimorph_schema.json"
INPUT_CUSTOM = BASE_DIR / "testing_validation" / "ref_schemas" / "custom_schema.json"
OUT_PREFERRED_DIR = BASE_DIR / "prompt_annotations" / "preferred_features"
with open(INPUT_CUSTOM, encoding="utf8") as f:
    custom = json.load(f)
with open(INPUT_UNIMORPH, encoding="utf8") as f:
    unimorph = json.load(f)
data = {}

for path in INPUT_DIR.glob("*.txt"):
    with path.open("r", encoding="utf-8") as f:
        data[path.name] = yaml.safe_load(f)

from pathlib import Path

OUT_PREFERRED_DIR = Path(OUT_PREFERRED_DIR)
OUT_PREFERRED_DIR.mkdir(parents=True, exist_ok=True)

for fname, content in data.items():
    # fname is the input filename
    out_filename = Path(fname).stem + "_preferred_features.txt"
    out_path = OUT_PREFERRED_DIR / out_filename

    # top-level key in YAML
    k = next(iter(content))
    lookup_keys = content[k]["Features"].keys()

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{k}\n")

        for lk in lookup_keys:
            f.write(f"\t{lk}\n")
            if lk.startswith("u-"):
                vals = unimorph[lk]
                for vk, vv in vals.items():
                    f.write(f"\t\t{vk}\t({vv})\n")

            elif lk.startswith("x-"):
                vals = custom[lk]
                joined_vals = "\n\t\t".join(vals)
                f.write(f"\t\t{joined_vals}\n")

