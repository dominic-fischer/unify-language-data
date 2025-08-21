import json, gzip, sys
sys.stdout.reconfigure(encoding='utf-8')

path = "wiktionary_files/by_lang/chichewa.jsonl.gz"

with gzip.open(path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20: break
        try:
            word = json.loads(line)
        except Exception:
            continue  # skip bad line

        if word.get("pos") != "verb":
            continue

        for k, v in word.items():
            if k in ["inflection_templates", "forms"]:
                continue
            print(f"{k}:")
            if isinstance(v, list):
                for vv in v:
                    print("\t", vv)
            else:
                print("\t", v)
        print(f'\n{"-"*50}\n')
