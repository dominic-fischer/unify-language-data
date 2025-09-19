import pandas as pd
from pathlib import Path

# Path to your eng folder
unimorph_path = Path(r"C:\Users\Dominic-Asus\ProgrammingProject_HS25\eng\eng")

# Load UniMorph TSV (tab-separated: form, lemma, features)
df = pd.read_csv(unimorph_path, sep="\t", header=None, names=["form", "lemma", "features"])

# Simple lookup
def lookup(word):
    rows = df[df["form"] == word]
    if rows.empty:
        return f"No UniMorph entry for {word}"
    return rows

# Example
print(lookup("more slowly"))
