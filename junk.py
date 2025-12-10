import json
with open("lang-de_to_italian_toc_mapping.json", encoding="utf-8") as f:
    data = json.load(f)

outfile = "italian_outfile.txt"

with open(outfile, "w", encoding="utf-8") as f:
    for k, v in data.items():
        f.write(k.upper())
        f.write("\n\n")
        for d in v["matches"]:
            f.write(d["full"])
            f.write("\n\n")
            f.write(d["content"])
            f.write("\n\n")
        f.write("- " * 50)
        
