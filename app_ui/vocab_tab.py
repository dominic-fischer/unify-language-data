from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from pathlib import Path
from app_data.bootstrap import ensure_zip_extracted

from config import LANGS
from app_data.vocab import VOCAB_DIR, VOCAB_POS_DIR

from app_data.vocab import (
    align_meanings_two_langs,
    get_category_options_for_langs_and_pos,
    get_pos_options_for_langs,
    load_vocab_entries_filtered,
    resolve_lang_files,
    resolve_lang_pos_files,
    vocab_record_meanings,
)
from app_ui.common import non_empty_columns


# ----------------------------
# Config helpers (local vs Streamlit Cloud)
# ----------------------------

def _get_config_value(key: str, default: str = "") -> str:
    """Get config from Streamlit secrets (Cloud) with safe local fallbacks.

    Resolution order:
      1) environment variable
      2) st.secrets
      3) default (hard-coded)
    """
    v = os.environ.get(key)
    if isinstance(v, str) and v.strip():
        return v.strip()

    try:
        v2 = st.secrets.get(key, "")
        if isinstance(v2, str) and v2.strip():
            return v2.strip()
    except StreamlitSecretNotFoundError:
        pass
    except Exception:
        pass

    return default


def _ensure_vocab_data(pos_zip_url: str, outfiles_zip_url: str) -> None:
    """Make sure BOTH datasets exist (POS shards + full outfiles)."""
    # POS shards: needed for POS/category discovery
    if not VOCAB_POS_DIR.exists():
        if not pos_zip_url:
            st.error(
                f"Missing vocab split data at {VOCAB_POS_DIR}. "
                "Configure OUTFILES_POS_ZIP_URL in Streamlit Secrets (or env var)."
            )
            st.stop()
        with st.spinner("Downloading vocab split data (one-time)..."):
            ensure_zip_extracted(pos_zip_url, VOCAB_POS_DIR)

    # Full dataset: ensures language mapping is stable and browsing works without POS
    if not VOCAB_DIR.exists():
        if not outfiles_zip_url:
            st.error(
                f"Missing full vocab data at {VOCAB_DIR}. "
                "Configure OUTFILES_ZIP_URL in Streamlit Secrets (or env var)."
            )
            st.stop()
        with st.spinner("Downloading full vocab data (one-time)..."):
            ensure_zip_extracted(outfiles_zip_url, VOCAB_DIR)


def render_vocab_details(details_json: str) -> None:
    """Show full vocab entry JSON plus a few common sections."""
    try:
        e = json.loads(details_json or "{}")
    except Exception:
        st.warning("Could not parse details_json.")
        return

    with st.expander("Raw entry (JSON)", expanded=False):
        st.json(e)

    for key, label in [
        ("etymology_text", "Etymology"),
        ("etymology", "Etymology"),
        ("etymologies", "Etymologies"),
        ("pronunciations", "Pronunciations"),
        ("forms", "Forms"),
        ("senses", "Senses / Meanings"),
        ("translations", "Translations"),
        ("related", "Related"),
        ("synonyms", "Synonyms"),
        ("antonyms", "Antonyms"),
        ("derived", "Derived terms"),
        ("descendants", "Descendants"),
        ("examples", "Examples"),
    ]:
        if key in e and e[key] not in (None, "", [], {}):
            with st.expander(label, expanded=(key == "senses")):
                st.json(e[key])


def render_vocab_tab() -> None:
    st.subheader("Vocab (Wiktionary)")

    # ---- URLs (secrets/env/default) ----
    pos_zip_url = _get_config_value(
        "OUTFILES_POS_ZIP_URL",
        "https://github.com/dominic-fischer/unify-language-data/releases/download/data-v1/outfiles_pos.zip",
    )
    outfiles_zip_url = _get_config_value(
        "OUTFILES_ZIP_URL",
        "https://github.com/dominic-fischer/unify-language-data/releases/download/data-v1/outfiles.zip",
    )

    # ---- Ensure BOTH datasets exist ----
    _ensure_vocab_data(pos_zip_url=pos_zip_url, outfiles_zip_url=outfiles_zip_url)

    vocab_mode = st.radio("Mode", ["Browse", "Compare by meaning"], horizontal=True, key="v_mode")

    # ----------------------------
    # Sidebar: languages first
    # ----------------------------
    st.sidebar.header("Vocab selection")
    v_langs = st.sidebar.multiselect("Languages", options=LANGS, default=[], key="v_langs")

    if not v_langs:
        st.info("Select at least one language in the sidebar.")
        st.stop()

    # Ensure there are source files for these languages
    src_files = resolve_lang_files(tuple(v_langs))
    if not src_files:
        st.warning("No vocab source files matched the selected languages.")
        st.stop()

    # POS options:
    # If 2 languages: intersection. Else: union.
    pos_options = get_pos_options_for_langs(tuple(v_langs))

    with st.sidebar:
        st.subheader("Filters")

        v_pos_val = st.selectbox("POS (optional)", options=[""] + pos_options, index=0, key="v_pos") or None

        # Categories depend on POS:
        cat_options = get_category_options_for_langs_and_pos(tuple(v_langs), v_pos_val)

        v_cat_val = st.selectbox(
            "Topic/category (optional)",
            options=[""] + cat_options,
            index=0,
            key="v_cat",
            disabled=(v_pos_val is None),
            help="Choose a POS first to see categories available within that POS.",
        ) or None

        limit_rows = st.slider("Max rows to load", 100, 5000, 1000, 100, key="v_limit_rows")

    # ----------------------------
    # Browse
    # ----------------------------
    if vocab_mode == "Browse":
        word_q = st.text_input("Search word (substring; optional)", value="", key="v_word_q")

        df = load_vocab_entries_filtered(
            langs=tuple(v_langs),
            pos=v_pos_val,
            cat=v_cat_val,
            word_query=word_q,
            limit_rows=limit_rows,
        )

        if df.empty:
            st.warning("No entries matched your filters.")
            st.stop()

        sort_cols = [c for c in ["lang", "pos", "word"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable")

        v_hidden = {"details_json"}
        cols = [c for c in non_empty_columns(df) if c not in v_hidden]

        scanned_files = resolve_lang_pos_files(tuple(v_langs), v_pos_val)

        st.caption(
            f"{len(df):,} entries loaded "
            f"(scanned: {', '.join([f.name for f in scanned_files])})."
        )

        event = st.dataframe(
            df[cols],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        sel = (event.selection or {}).get("rows", [])
        if sel:
            row = df.iloc[sel[0]]
            st.markdown("### Entry details")
            render_vocab_details(row.get("details_json", ""))

    # ----------------------------
    # Compare by meaning (two languages)
    # ----------------------------
    else:
        st.markdown(
            "Compare **two languages** using **meaning (gloss)** as the anchor. "
            "If glosses overlap, words align on the same row."
        )

        if len(v_langs) != 2:
            st.info("For meaning comparison, select exactly **2 languages** in the sidebar.")
            st.stop()

        lang1, lang2 = v_langs[0], v_langs[1]

        word_q = st.text_input("Optional: restrict words (substring)", value="", key="v_compare_word_q").strip()
        threshold = st.slider("Meaning overlap threshold", 0.10, 0.9, 0.35, 0.05, key="v_compare_thresh")

        scan_limit_each = st.slider("Scan limit per language", 50, 10000, 1000, 500, key="v_compare_scan_each")

        df1 = load_vocab_entries_filtered(
            langs=(lang1,),
            pos=v_pos_val,
            cat=v_cat_val,
            word_query=word_q if word_q else None,
            limit_rows=scan_limit_each,
        )
        df2 = load_vocab_entries_filtered(
            langs=(lang2,),
            pos=v_pos_val,
            cat=v_cat_val,
            word_query=word_q if word_q else None,
            limit_rows=scan_limit_each,
        )

        def to_word_gloss(df: pd.DataFrame) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            if df is None or df.empty:
                return out
            for _, r in df.iterrows():
                word = r.get("word") or ""
                if not isinstance(word, str) or not word.strip():
                    continue

                try:
                    entry = json.loads(r.get("details_json") or "{}")
                except Exception:
                    entry = {}

                meanings = vocab_record_meanings(entry, max_glosses=1)
                gloss = meanings[0] if meanings else (r.get("first_gloss") or "")
                if not isinstance(gloss, str) or not gloss.strip():
                    continue

                out.append((word.strip(), gloss.strip()))
            return out

        rows1 = to_word_gloss(df1)
        rows2 = to_word_gloss(df2)

        if not rows1 and not rows2:
            st.warning("No entries found for either language with the current filters.")
            st.stop()

        aligned = align_meanings_two_langs(rows1, rows2, threshold=threshold)
        df_align = pd.DataFrame(aligned)
        only_matches = st.checkbox("Only show matched meanings", value=True, key="v_only_matches")
        if only_matches:
            df_align = df_align[df_align["_sim"] >= threshold].copy()

        df_out = (
            df_align.groupby("meaning", as_index=False)
            .agg(
                **{
                    lang1: ("lang1", lambda xs: " | ".join([x for x in xs if isinstance(x, str) and x.strip()])),
                    lang2: ("lang2", lambda xs: " | ".join([x for x in xs if isinstance(x, str) and x.strip()])),
                    "_best_sim": ("_sim", "max"),
                }
            )
            .sort_values("_best_sim", ascending=False, kind="stable")
        )

        show_debug = st.checkbox("Show debug columns", value=False, key="v_compare_debug")
        if not show_debug:
            df_out = df_out.drop(columns=["_best_sim"], errors="ignore")

        st.caption(
            f"{len(df_out):,} meaning rows "
            f"(scanned up to {scan_limit_each:,} entries per language; threshold={threshold})."
        )
        st.dataframe(df_out, use_container_width=True, hide_index=True)
