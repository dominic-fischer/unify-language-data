
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now import your classes
from parsers import DocxParser


# Replace with your actual DOCX path
parser = DocxParser("docx_files/docx_input_files/BANTU_Chichewa_NEW.docx")

# This saves the final output (plaintext with tables inserted)
parser.save_output()
