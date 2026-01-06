# app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import yaml  # pip install pyyaml


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Linguistics Data Browser", layout="wide")

GRAMMAR_DIR = Path("./data")
VOCAB_DIR = Path("./vocab/data")

GRAMMAR_SUPPORTED = {".json", ".txt", ".yaml", ".yml"}
VOCAB_SUPPORTED = {".jsonl", ".json", ".yaml", ".yml"}


# ----------------------------
# Helpers: loaders
# ----------------------------
def load_doc(path: Path) -> dict:
    """
    Loads a YAML-ish .txt/.yaml/.yml or a .json file into a Python dict.
    Your grammar example .txt is YAML-like, so this supports that.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".txt", ".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def load_jsonl(path: Path) -> List[dict]:
    """
    Loads JSONL: one JSON object per line.
    """
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ----------------------------
# Grammar normalization
# ----------------------------
def normalize_grammar_files(files: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      rules_df: one row per rule
      endings_df: one row per ending
      features_df: feature catalog (feature -> allowed values)
    """
    rules_rows: List[Dict[str, Any]] = []
    endings_rows: List[Dict[str, Any]] = []
    feature_catalog_rows: List[Dict[str, Any]] = []

    for fp in files:
        doc = load_doc(fp) or {}

        # doc structure: {CategoryName: {Features: {...}, Rules: {...}}}
        for category, payload in doc.items():
            payload = payload or {}

            feats = payload.get("Features", {}) or {}
            for feat_name, allowed_vals in feats.items():
                feature_catalog_rows.append(
                    {
                        "file": fp.name,
                        "category": category,
                        "feature": feat_name,
                        "allowed_values": allowed_vals,
                    }
                )

            rules = payload.get("Rules", {}) or {}
            for rule_id, rule in rules.items():
                rule = rule or {}
                applies = rule.get("applies", {}) or {}

                rules_rows.append(
                    {
                        "file": fp.name,
                        "category": category,
                        "rule_id": rule_id,
                        "applies": applies,
                        "pattern": rule.get("pattern"),
                        "note": rule.get("note"),
                        "negation": rule.get("negation"),
                        "examples": rule.get("examples", []),
                    }
                )

                # explode endings into their own table
                for e in (rule.get("endings") or []):
                    e = e or {}
                    e_feats = e.get("features", {}) or {}

                    row: Dict[str, Any] = {
                        "file": fp.name,
                        "category": category,
                        "rule_id": rule_id,
                        "form": e.get("form"),
                        "ending_note": e.get("note"),
                    }

                    # Flatten ending features into columns
                    for k, v in e_feats.items():
                        row[k] = v

                    # Also include applies features for context/filtering (prefixed)
                    for k, v in applies.items():
                        row[f"applies::{k}"] = v

                    endings_rows.append(row)

    rules_df = pd.DataFrame(rules_rows)
    endings_df = pd.DataFrame(endings_rows)
    features_df = pd.DataFrame(feature_catalog_rows)

    # Pretty display columns
    if not rules_df.empty:
        rules_df["applies_json"] = rules_df["applies"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        rules_df["examples"] = rules_df["examples"].apply(
            lambda xs: " | ".join(xs) if isinstance(xs, list) else ("" if xs is None else str(xs))
        )

    return rules_df, endings_df, features_df


@st.cache_data(show_spinner=True)
def load_grammar() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    files = sorted([p for p in GRAMMAR_DIR.glob("*") if p.suffix.lower() in GRAMMAR_SUPPORTED])
    rules_df, endings_df, features_df = normalize_grammar_files(files)

    meta_cols = {"file", "category", "rule_id", "form", "ending_note"}
    ending_feature_cols = [
        c for c in endings_df.columns if c not in meta_cols and not c.startswith("applies::")
    ]
    return rules_df, endings_df, features_df, sorted(ending_feature_cols)


# ----------------------------
# Vocab normalization (JSONL-friendly)
# ----------------------------
def normalize_vocab_files(files: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For Wiktionary-like JSONL:
      entries_df: one row per entry
      senses_df: one row per sense
    """
    entry_rows: List[Dict[str, Any]] = []
    sense_rows: List[Dict[str, Any]] = []

    for fp in files:
        suffix = fp.suffix.lower()

        if suffix == ".jsonl":
            entries = load_jsonl(fp)
        elif suffix in {".json", ".yaml", ".yml"}:
            # Fallback: accept a JSON/YAML that is either a list of dicts,
            # or a dict keyed by ids/lemmas.
            doc = load_doc(fp)
            if isinstance(doc, list):
                entries = doc
            elif isinstance(doc, dict):
                # convert {key: record} -> record + key
                entries = []
                for k, v in doc.items():
                    if isinstance(v, dict):
                        vv = dict(v)
                        vv["_key"] = k
                        entries.append(vv)
            else:
                entries = []
        else:
            continue

        for idx, e in enumerate(entries):
            if not isinstance(e, dict):
                continue

            word = e.get("word") or e.get("lemma") or e.get("_key")
            lang = e.get("lang")
            lang_code = e.get("lang_code")
            pos = e.get("pos")

            senses = e.get("senses") or []
            forms = e.get("forms") or []

            entry_rows.append(
                {
                    "file": fp.name,
                    "entry_index": idx,
                    "word": word,
                    "lang": lang,
                    "lang_code": lang_code,
                    "pos": pos,
                    "n_senses": len(senses) if isinstance(senses, list) else 0,
                    "n_forms": len(forms) if isinstance(forms, list) else 0,
                }
            )

            if isinstance(senses, list):
                for s_i, s in enumerate(senses):
                    if not isinstance(s, dict):
                        continue
                    glosses = s.get("glosses") or []
                    cats = s.get("categories") or []
                    links = s.get("links") or []

                    sense_rows.append(
                        {
                            "file": fp.name,
                            "entry_index": idx,
                            "sense_index": s_i,
                            "word": word,
                            "lang": lang,
                            "lang_code": lang_code,
                            "pos": pos,
                            "glosses": " | ".join(glosses) if isinstance(glosses, list) else str(glosses),
                            "categories": " | ".join(cats) if isinstance(cats, list) else str(cats),
                            "links": " | ".join([f"{a}->{b}" for a, b in links]) if isinstance(links, list) else str(links),
                        }
                    )

    return pd.DataFrame(entry_rows), pd.DataFrame(sense_rows)


@st.cache_data(show_spinner=True)
def load_vocab() -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted([p for p in VOCAB_DIR.glob("*") if p.suffix.lower() in VOCAB_SUPPORTED])
    return normalize_vocab_files(files)


# ----------------------------
# Filtering utilities
# ----------------------------
def build_constraints_ui(feature_cols: List[str], df: pd.DataFrame, key_prefix: str) -> Dict[str, List[Any]]:
    """
    Sidebar multiselects per feature column.
    AND across features, OR within selected values.
    """
    constraints: Dict[str, List[Any]] = {}
    st.sidebar.subheader("Constraints")

    feature_search = st.sidebar.text_input("Search feature/column", value="", key=f"{key_prefix}::featsearch")

    for col in feature_cols:
        if col not in df.columns:
            continue
        if feature_search and feature_search.lower() not in col.lower():
            continue

        vals = df[col].dropna().unique().tolist()
        # Keep only displayable scalar-ish values
        vals = [v for v in vals if isinstance(v, (str, int, float, bool)) and v != ""]
        vals = sorted(vals, key=lambda x: str(x))

        if not vals:
            continue

        picked = st.sidebar.multiselect(col, options=vals, default=[], key=f"{key_prefix}::{col}")
        if picked:
            constraints[col] = picked

    return constraints


def apply_constraints(df: pd.DataFrame, constraints: Dict[str, List[Any]]) -> pd.DataFrame:
    out = df
    for col, allowed in constraints.items():
        if col in out.columns and allowed:
            out = out[out[col].isin(allowed)]
    return out


# ----------------------------
# App UI
# ----------------------------
st.title("Linguistics Data Browser")

tab_grammar, tab_vocab = st.tabs(["Grammar", "Vocab"])


# ===== Grammar tab =====
with tab_grammar:
    st.subheader("Grammar")

    if not GRAMMAR_DIR.exists():
        st.error("Grammar directory ./data does not exist. Create it and add your grammar files.")
        st.stop()

    rules_df, endings_df, features_df, ending_feature_cols = load_grammar()

    if rules_df.empty and endings_df.empty:
        st.warning("No grammar rules/endings found after parsing. Check your files and schema consistency.")
        st.stop()

    # Scope filters
    st.sidebar.header("Grammar scope")
    g_files = sorted(set(rules_df["file"].unique().tolist() + endings_df["file"].unique().tolist()))
    g_cats = sorted(set(rules_df["category"].unique().tolist() + endings_df["category"].unique().tolist()))

    g_file_sel = st.sidebar.multiselect("Grammar files", options=g_files, default=g_files, key="g_files")
    g_cat_sel = st.sidebar.multiselect("Grammar categories", options=g_cats, default=g_cats, key="g_cats")

    view = st.radio("View", ["Endings (best for feature filtering)", "Rules (filter by applies)"], horizontal=True)

    if view.startswith("Endings"):
        df = endings_df.copy()
        df = df[df["file"].isin(g_file_sel) & df["category"].isin(g_cat_sel)]

        constraints = build_constraints_ui(ending_feature_cols, df, key_prefix="g_endings")
        df2 = apply_constraints(df, constraints)

        # Optional text search across a couple columns
        q = st.text_input("Search (form / rule_id)", value="", key="g_endings_search")
        if q:
            ql = q.lower()
            df2 = df2[
                df2["form"].fillna("").str.lower().str.contains(ql, regex=False)
                | df2["rule_id"].fillna("").str.lower().str.contains(ql, regex=False)
            ]

        st.caption(f"{len(df2):,} endings shown (within scope: {len(df):,})")

        # Put core columns first
        preferred = ["file", "category", "rule_id", "form", "ending_note"]
        rest = [c for c in df2.columns if c not in preferred]
        st.dataframe(
            df2[preferred + rest].sort_values(["category", "rule_id", "form"], kind="stable"),
            use_container_width=True,
            hide_index=True,
        )

    else:
        df = rules_df.copy()
        df = df[df["file"].isin(g_file_sel) & df["category"].isin(g_cat_sel)]

        # Expand applies dict into columns for filtering
        applies_wide = pd.json_normalize(df["applies"]).add_prefix("applies::")
        dfw = pd.concat([df.reset_index(drop=True), applies_wide.reset_index(drop=True)], axis=1)

        applies_cols = [c for c in dfw.columns if c.startswith("applies::")]
        constraints = build_constraints_ui(applies_cols, dfw, key_prefix="g_rules_applies")
        df2 = apply_constraints(dfw, constraints)

        q = st.text_input("Search (rule_id / pattern / note)", value="", key="g_rules_search")
        if q:
            ql = q.lower()
            df2 = df2[
                df2["rule_id"].fillna("").str.lower().str.contains(ql, regex=False)
                | df2["pattern"].fillna("").str.lower().str.contains(ql, regex=False)
                | df2["note"].fillna("").str.lower().str.contains(ql, regex=False)
            ]

        st.caption(f"{len(df2):,} rules shown (within scope: {len(dfw):,})")

        show_cols = ["file", "category", "rule_id", "applies_json", "pattern", "note", "negation", "examples"]
        show_cols = [c for c in show_cols if c in df2.columns]
        st.dataframe(df2[show_cols], use_container_width=True, hide_index=True)


# ===== Vocab tab =====
with tab_vocab:
    st.subheader("Vocab")

    if not VOCAB_DIR.exists():
        st.error("Vocab directory ./vocab/data does not exist. Create it and add your vocab files.")
        st.stop()

    entries_df, senses_df = load_vocab()

    if entries_df.empty:
        st.warning("No vocab entries found. (Expected .jsonl or .json/.yaml/.yml in ./vocab/data)")
        st.stop()

    # Scope filters
    st.sidebar.header("Vocab scope")
    v_files = sorted(entries_df["file"].dropna().unique().tolist())
    v_langs = sorted(entries_df["lang"].dropna().unique().tolist())
    v_pos = sorted(entries_df["pos"].dropna().unique().tolist())

    v_file_sel = st.sidebar.multiselect("Vocab files", options=v_files, default=v_files, key="v_files")
    v_lang_sel = st.sidebar.multiselect("Language", options=v_langs, default=v_langs, key="v_langs")
    v_pos_sel = st.sidebar.multiselect("POS", options=v_pos, default=v_pos, key="v_pos")

    view = st.radio("View", ["Entries", "Senses"], horizontal=True, key="v_view")

    if view == "Entries":
        df = entries_df.copy()
        df = df[
            df["file"].isin(v_file_sel)
            & df["lang"].isin(v_lang_sel)
            & df["pos"].isin(v_pos_sel)
        ]

        q = st.text_input("Search word (substring)", value="", key="v_entries_search")
        if q:
            df = df[df["word"].fillna("").str.contains(q, case=False, regex=False)]

        st.caption(f"{len(df):,} entries")

        st.dataframe(
            df.sort_values(["lang", "pos", "word"], kind="stable"),
            use_container_width=True,
            hide_index=True,
        )

    else:
        df = senses_df.copy()
        if df.empty:
            st.warning("No senses found in the vocab files.")
            st.stop()

        df = df[
            df["file"].isin(v_file_sel)
            & df["lang"].isin(v_lang_sel)
            & df["pos"].isin(v_pos_sel)
        ]

        gloss_q = st.text_input("Gloss contains", value="", key="v_gloss_search")
        if gloss_q:
            df = df[df["glosses"].fillna("").str.contains(gloss_q, case=False, regex=False)]

        cat_q = st.text_input("Category contains", value="", key="v_cat_search")
        if cat_q:
            df = df[df["categories"].fillna("").str.contains(cat_q, case=False, regex=False)]

        st.caption(f"{len(df):,} senses")

        st.dataframe(
            df.sort_values(["lang", "pos", "word", "sense_index"], kind="stable"),
            use_container_width=True,
            hide_index=True,
        )


# Footer
st.divider()
st.caption(
    "Tip: Streamlit tables support click-to-sort, column resizing, and built-in filtering controls in the dataframe UI."
)
