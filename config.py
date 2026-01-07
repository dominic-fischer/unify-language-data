from __future__ import annotations

from pathlib import Path

import streamlit as st

# ----------------------------
# Config
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent

# Grammar lives in ONE place now
GRAMMAR_DIR = BASE_DIR / "outputs_gpt-5.2"

# Vocab lives here, files are *.jsonl.gz
VOCAB_DIR = BASE_DIR / "data" / "wiktionary" / "outfiles"

# Your language list (order used in UI where possible)
LANGS = ["chewa", "shona", "swahili", "zulu", "french", "italian", "portuguese", "romanian", "spanish", "german"]
LANG_SET = set(LANGS)

# Supported extensions
GRAMMAR_SUPPORTED = {".json", ".txt", ".yaml", ".yml"}
VOCAB_SUPPORTED = {".jsonl", ".jsonl.gz"}  # expect .jsonl.gz mostly

# Columns to hide by default (table should not show these)
HIDDEN_COLS = {"file", "rule_id", "category_in_doc", "details_json", "applies"}


def init_page() -> None:
    # Must be called before other Streamlit commands that output UI
    st.set_page_config(page_title="Linguistics Data Browser", layout="wide")
