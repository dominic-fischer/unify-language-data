from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if v == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def non_empty_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    if df.empty:
        return cols
    for c in df.columns:
        try:
            if (df[c].map(lambda x: not is_empty_value(x))).any():
                cols.append(c)
        except Exception:
            pass
    return cols


def _to_hashable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return v


def build_constraints_ui(feature_cols: List[str], df: pd.DataFrame, key_prefix: str) -> Dict[str, List[Any]]:
    """
    Sidebar multiselects per feature column.
    AND across features, OR within selected values.
    Safe with list/dict values (no .unique()).
    """
    constraints: Dict[str, List[Any]] = {}
    st.sidebar.subheader("Constraints")

    feature_search = st.sidebar.text_input("Search feature/column", value="", key=f"{key_prefix}::featsearch")

    for col in feature_cols:
        if col not in df.columns:
            continue
        if feature_search and feature_search.lower() not in col.lower():
            continue

        seen: Dict[str, Any] = {}
        for v in df[col].dropna().tolist():
            hv = _to_hashable(v)
            if hv is None or hv == "":
                continue
            seen[str(hv)] = hv

        vals = sorted(seen.values(), key=lambda x: str(x))
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
