import json, gzip, sys
sys.stdout.reconfigure(encoding='utf-8')

path = "wiktionary_files/by_lang/romanian.jsonl.gz"

with gzip.open(path, "rt", encoding="utf-8") as f:
    counter = {}
    counter_total = 0
    for i, line in enumerate(f):
        if i >= 5000: break
        try:
            word = json.loads(line)
        except Exception:
            continue  # skip bad line
        
        # check whether you can find a key "category" either in senses or in the top level and whether "lemmas" is in "categories"
        if "categories" in word:
            categories = word["categories"]
        else:
            categories = []
            if "senses" in word:
                for sense in word["senses"]:
                    if "categories" in sense:
                        categories.extend(sense["categories"])
        
        if not any("nouns" in cat.lower() for cat in categories):
            continue

        if word.get("pos") != "noun":
            continue

        for k, v in word.items():
            if k in ["forms"]:
               continue
            counter_total += 1

            print(f"{k}:")
            if isinstance(v, list):
                for vv in v:
                    print("\t", vv)
                    if isinstance(vv, dict) and "name" in vv:
                        name = vv["name"]
                        if name not in counter:
                            counter[name] = 0
                        counter[name] += 1
                        
            else:
                print("\t", v)
        print(f'\n{"-"*50}\n')

    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{k}: {v} ({v/counter_total:.2%})")
    print(f"Total: {counter_total}")
    print(f"Unique templates: {len(counter)}")
