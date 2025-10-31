import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Now import your class
from data.TextParser import WikipediaParser

# Replace with the Wikipedia page title you want to process
parser = WikipediaParser("Zulu grammar")

# Save the final output (plaintext with tables inserted)
parser.output_folder = "data/wikipedia/outfiles"
parser.save_output()