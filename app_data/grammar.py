from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml  # pip install pyyaml

from config import GRAMMAR_DIR, GRAMMAR_SUPPORTED, LANG_SET
from app_data.files import iter_files


# ----------------------------
# Grammar parsing helpers
# ----------------------------
def parse_lang_from_filename(fp: Path) -> str:
    """
    Expects: <lang>_FORMAT_...
    Example: german_FORMAT_clauses_syntax.txt_mockoutputs.txt
    """
    name = fp.name.lower()
    m = re.match(r"^([a-z]+)_format_", name)
    if not m:
        return "Unknown"
    lang = m.group(1)
    return lang if lang in LANG_SET else "Unknown"


def read_real_category_first_line(fp: Path) -> str:
    """
    real category = first non-empty line, strip trailing ':'.
    """
    try:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                return s[:-1] if s.endswith(":") else s
    except Exception:
        pass
    return "Unknown"


def load_doc(path: Path) -> dict:
    """
    Loads YAML-ish .txt/.yaml/.yml or a .json file into a Python dict.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".txt", ".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


# ----------------------------
# Grammar normalization (schema-aware: pattern/patterns/forms/endings)
# ----------------------------
def normalize_grammar_files(files: List[Path]) -> pd.DataFrame:
    """
    Returns a rules dataframe.
    Each rule row has:
      - pattern: table-friendly display string (from pattern OR patterns/forms/endings summary)
      - details_json: full structured payload containing pattern/patterns/forms/endings for expanders
    """
    rules_rows: List[Dict[str, Any]] = []

    for fp in files:
        lang = parse_lang_from_filename(fp)
        real_category = read_real_category_first_line(fp)

        try:
            doc = load_doc(fp) or {}
        except Exception:
            continue

        if not isinstance(doc, dict) or not doc:
            continue

        for category_in_doc, payload in doc.items():
            if not isinstance(payload, dict):
                continue

            rules = payload.get("Rules", {}) or {}
            if not isinstance(rules, dict):
                continue

            for rule_id, rule in rules.items():
                if not isinstance(rule, dict):
                    continue

                applies = rule.get("applies", {}) or {}
                if not isinstance(applies, dict):
                    applies = {}

                # ---- schema-aware extraction
                pattern = rule.get("pattern")
                patterns = rule.get("patterns") or []
                forms = rule.get("forms") or []
                endings = rule.get("endings") or []

                if not isinstance(patterns, list):
                    patterns = []
                if not isinstance(forms, list):
                    forms = []
                if not isinstance(endings, list):
                    endings = []

                # Table-friendly "pattern" display
                if isinstance(pattern, str) and pattern.strip():
                    pattern_display = pattern.strip()
                else:
                    n_patterns = len(patterns)
                    n_forms = len(forms)
                    n_endings = len(endings)

                    parts = []
                    if n_patterns:
                        parts.append(f"{n_patterns} pattern blocks")
                    if n_forms:
                        parts.append(f"{n_forms} forms")
                    if n_endings:
                        parts.append(f"{n_endings} endings")

                    hint = ""
                    if n_patterns and isinstance(patterns[0], dict):
                        p0 = patterns[0].get("pattern")
                        if isinstance(p0, str) and p0.strip():
                            hint = p0.strip()

                    pattern_display = (hint if hint else " / ".join(parts)) or "—"

                details = {
                    "pattern": pattern if isinstance(pattern, str) else None,
                    "patterns": patterns,
                    "forms": forms,
                    "endings": endings,
                }
                details_json = json.dumps(details, ensure_ascii=False)

                rules_rows.append(
                    {
                        "file": fp.name,
                        "language": lang,
                        "real_category": real_category,
                        "category_in_doc": category_in_doc,
                        "rule_id": rule_id,
                        "applies": applies,
                        "applies_json": json.dumps(applies, ensure_ascii=False),
                        "pattern": pattern_display,
                        "note": rule.get("note"),
                        "negation": rule.get("negation"),
                        "examples": rule.get("examples", []),
                        "details_json": details_json,
                        "has_details": bool(patterns or forms or endings),
                    }
                )

    df = pd.DataFrame(rules_rows)

    if not df.empty:
        df["examples"] = df["examples"].apply(
            lambda xs: " | ".join(xs) if isinstance(xs, list) else ("" if xs is None else str(xs))
        )

    return df


@st.cache_data(show_spinner=True)
def load_grammar() -> pd.DataFrame:
    files = iter_files(GRAMMAR_DIR, recursive=True, exts=GRAMMAR_SUPPORTED)
    return normalize_grammar_files(files)


def build_file_map(rules_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """
    Map (language, real_category) -> file.
    """
    m: dict[tuple[str, str], str] = {}
    if rules_df.empty:
        return m
    for _, r in rules_df[["language", "real_category", "file"]].dropna().drop_duplicates().iterrows():
        key = (str(r["language"]), str(r["real_category"]))
        if key not in m:
            m[key] = str(r["file"])
    return m
