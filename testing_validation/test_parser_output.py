# test_parser_output.py

import sys
import os
import builtins
from pathlib import Path
from difflib import unified_diff
from contextlib import contextmanager

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.TextParser import (
    DocxParser,
    WikipediaParser,
)  # Updated to import from unified 'parsers.py'


@contextmanager
def suppress_stdout():
    """
    Suppress all stdout and stderr output, including print() and C-level streams,
    but allow final test results (PASS/FAIL) to print after suppression.
    """

    def dummy_print(*args, **kwargs):
        pass

    original_print = builtins.print
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(os.devnull, "w") as devnull:
        try:
            builtins.print = dummy_print
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            builtins.print = original_print
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def compare_outputs(test_file, actual_file):
    """
    Compare test and actual output files and return True if they match.
    """
    with open(test_file, encoding="utf-8") as f1, open(
        actual_file, encoding="utf-8"
    ) as f2:
        test = f1.read()
        actual = f2.read()
    return test == actual


def test_docx_chichewa():
    with suppress_stdout():
        filepath = "data/docx/infiles/BANTU_Chichewa_NEW.docx"
        parser = DocxParser(filepath)
        parser.output_folder = ("data/docx/outfiles/")
        parser.title = filepath.split("/")[-1].replace(".docx", "")
        parser.save_output()
    result = compare_outputs(
        Path("testing_validation/ref_files/BANTU_Chichewa_NEW_plaintext_structured_w_tables.txt"),
        Path("data/docx/outfiles/BANTU_Chichewa_NEW_plaintext_structured_w_tables.txt"),
    )
    print(
        "✅ PASS: BANTU_Chichewa_NEW_plaintext_structured_w_tables.txt"
        if result
        else "❌ FAIL: BANTU_Chichewa_NEW_plaintext_structured_w_tables.txt"
    )


def test_wiki_zulu():
    with suppress_stdout():
        parser = WikipediaParser("Zulu grammar")
        parser.output_folder = ("data/wikipedia/outfiles/")
        parser.save_output()
    result = compare_outputs(
        Path("testing_validation/ref_files/Zulu_grammar_plaintext_structured_w_tables.txt"),
        Path("data/wikipedia/outfiles/Zulu_grammar_plaintext_structured_w_tables.txt"),
    )
    print(
        "✅ PASS: Zulu_grammar_plaintext_structured_w_tables.txt"
        if result
        else "❌ FAIL: Zulu_grammar_plaintext_structured_w_tables.txt"
    )


# ---- RUN ---- #
if __name__ == "__main__":
    test_docx_chichewa()
    test_wiki_zulu()
