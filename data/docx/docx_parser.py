import sys
from pathlib import Path
import os

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Now import your classes
from data.TextParser import DocxParser

path = "data/docx/infiles/"

print(f"Looking for .docx files in {path}")
for file in os.listdir(path):
    # Join directory path with filename
    if not file.endswith(".docx"):
        continue

    print(f"Processing {file}...")
    file_path = os.path.join(path, file)
    docname = file.replace(".docx", "")

    # Create output directory for tables and text
    tables_dir = os.path.join("data/docx/infiles", f"{docname}_tables")
    out_dir = os.path.join("data/docx/outfiles")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Initialize parser with output directory
    parser = DocxParser(file_path)
    parser.output_folder = out_dir
    parser.title = docname  # Always use filename as title

    print(f"Found {len(parser.table_snippets)} tables")
    for i, table in enumerate(parser.table_snippets):
        # save the table to a csv
        out_file = os.path.join(tables_dir, f"table_{i}.csv")
        table_df = table["table"]
        table_df.to_csv(out_file, index=False)
        print(f"Saved table {i} to {out_file}")

    # This saves the final output (plaintext with tables inserted)
    parser.save_output()
