# app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import yaml  # pip install pyyaml

st.set_page_config(page_title="Grammar Browser", layout="wide")

DATA_DIR = Path("./data")  # put your ~100-200 files here
SUPPORTED = {".json", ".txt", ".yaml", ".yml"}


def load_doc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".txt", ".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def normalize_all(files: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rules_rows = []
    endings_rows = []
    feature_catalog_rows = []

    for fp in files:
        doc = load_doc(fp)
        # doc structure: {CategoryName: {Features: {...}, Rules: {...}}}
        for category, payload in (doc or {}).items():
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

                for e in (rule.get("endings") or []):
                    e_feats = (e.get("features") or {})
                    row = {
                        "file": fp.name,
                        "category": category,
                        "rule_id": rule_id,
                        "form": e.get("form"),
                        "ending_note": e.get("note"),
                    }
                    # flatten ending features into columns
                    for k, v in e_feats.items():
                        row[k] = v
                    # also optionally include applies features for context/filtering
                    for k, v in applies.items():
                        row[f"applies::{k}"] = v
                    endings_rows.append(row)

    rules_df = pd.DataFrame(rules_rows)
    endings_df = pd.DataFrame(endings_rows)
    features_df = pd.DataFrame(feature_catalog_rows)

    # Pretty formatting for display
    if not rules_df.empty:
        rules_df["applies_json"] = rules_df["applies"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        rules_df["examples"] = rules_df["examples"].apply(lambda xs: " | ".join(xs) if isinstance(xs, list) else str(xs))

    return rules_df, endings_df, features_df


@st.cache_data(show_spinner=True)
def load_everything() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    files = sorted([p for p in DATA_DIR.glob("*") if p.suffix.lower() in SUPPORTED])
    rules_df, endings_df, features_df = normalize_all(files)

    # Determine all feature columns present in endings_df (excluding metadata columns)
    meta_cols = {"file", "category", "rule_id", "form", "ending_note"}
    feature_cols = [c for c in endings_df.columns if c not in meta_cols and not c.startswith("applies::")]

    return rules_df, endings_df, features_df, sorted(feature_cols)


def build_constraints_ui(feature_cols: List[str], df: pd.DataFrame, prefix: str) -> Dict[str, List[str]]:
    """
    Create sidebar multiselects for features present in df.
    Returns dict: feature -> selected values.
    """
    constraints: Dict[str, List[str]] = {}
    st.sidebar.subheader("Feature constraints")

    # Small usability: only show features that have >1 distinct value
    candidates = []
    for f in feature_cols:
        if f in df.columns:
            vals = sorted([v for v in df[f].dropna().unique().tolist() if v != ""])
            if len(vals) >= 1:
                candidates.append((f, vals))

    # Optional: allow searching feature names by text
    feature_search = st.sidebar.text_input("Search feature", value="")
    for f, vals in candidates:
        if feature_search and feature_search.lower() not in f.lower():
            continue
        picked = st.sidebar.multiselect(f, options=vals, default=[], key=f"{prefix}::{f}")
        if picked:
            constraints[f] = picked
    return constraints


def apply_constraints(df: pd.DataFrame, constraints: Dict[str, List[str]]) -> pd.DataFrame:
    out = df
    for feat, allowed in constraints.items():
        if feat not in out.columns:
            continue
        out = out[out[feat].isin(allowed)]
    return out


# ---------- App ----------
st.title("Grammar Browser")

if not DATA_DIR.exists():
    st.error("DATA_DIR ./data does not exist. Create it and add your files.")
    st.stop()

rules_df, endings_df, features_df, ending_feature_cols = load_everything()

if rules_df.empty and endings_df.empty:
    st.warning("No rules/endings found after parsing. Check file formats.")
    st.stop()

# Top-level filters
files = sorted(set(rules_df["file"].unique().tolist() + endings_df["file"].unique().tolist()))
cats = sorted(set(rules_df["category"].unique().tolist() + endings_df["category"].unique().tolist()))

st.sidebar.header("Scope")
file_sel = st.sidebar.multiselect("Files", options=files, default=files)
cat_sel = st.sidebar.multiselect("Categories", options=cats, default=cats)

view = st.sidebar.radio("View", ["Endings (best for feature filtering)", "Rules (applies filtering)"])

if view.startswith("Endings"):
    df = endings_df.copy()
    df = df[df["file"].isin(file_sel) & df["category"].isin(cat_sel)]

    constraints = build_constraints_ui(ending_feature_cols, df, prefix="endings")
    df2 = apply_constraints(df, constraints)

    st.caption(f"{len(df2):,} endings (filtered) / {len(df):,} total in scope")

    st.dataframe(
        df2.sort_values(["category", "rule_id", "form"], kind="stable"),
        use_container_width=True,
        hide_index=True,
    )

else:
    df = rules_df.copy()
    df = df[df["file"].isin(file_sel) & df["category"].isin(cat_sel)]

    # For rules, constraints come from the applies dict, so build UI from applies keys/values
    st.sidebar.subheader("Applies constraints (rules)")
    # Explode applies into a temporary wide dataframe for filtering
    applies_wide = pd.json_normalize(df["applies"]).add_prefix("applies::")
    df_wide = pd.concat([df.reset_index(drop=True), applies_wide.reset_index(drop=True)], axis=1)

    applies_cols = [c for c in df_wide.columns if c.startswith("applies::")]
    # Build constraints UI for applies columns
    constraints = build_constraints_ui(applies_cols, df_wide, prefix="rules_applies")
    df2 = apply_constraints(df_wide, constraints)

    show_cols = ["file", "category", "rule_id", "applies_json", "pattern", "note", "negation", "examples"]
    st.caption(f"{len(df2):,} rules (filtered) / {len(df_wide):,} total in scope")
    st.dataframe(df2[show_cols], use_container_width=True, hide_index=True)
