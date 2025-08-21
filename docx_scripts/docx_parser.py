
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now import your classes
from parsers import DocxParser

docname = "BANTU_Chichewa_NEW" 

# Replace with your actual DOCX path
parser = DocxParser(f"docx_files/docx_input_files/{docname}.docx")

for table in parser.table_snippets:
    # save the table to a csv
    # saving path = Path("docx_files/{docname}/table_X.csv")
    table_df = table["table"]
    table_index = table["index"]
    dir = Path(f"docx_files/{docname}_tables")
    dir.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(f"docx_files/{docname}_tables/table_{table_index}.csv", index=False)


# This saves the final output (plaintext with tables inserted)
parser.save_output()
