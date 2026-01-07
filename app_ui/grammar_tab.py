from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from config import GRAMMAR_DIR, HIDDEN_COLS, LANGS
from app_data.grammar import build_file_map, load_grammar
from app_ui.common import apply_constraints, build_constraints_ui, is_empty_value, non_empty_columns


def render_rule_details(details_json: str) -> None:
    """
    Render patterns/forms/endings with expanders.
    """
    try:
        details = json.loads(details_json or "{}")
    except Exception:
        details = {}

    single_pattern = details.get("pattern")
    patterns = details.get("patterns") or []
    forms = details.get("forms") or []
    endings = details.get("endings") or []

    if single_pattern:
        with st.expander("Pattern", expanded=True):
            st.write(single_pattern)

    if patterns:
        with st.expander(f"Patterns ({len(patterns)})", expanded=True):
            for i, p in enumerate(patterns, 1):
                if isinstance(p, dict):
                    st.markdown(f"**#{i}**")
                    if p.get("pattern"):
                        st.write(p["pattern"])
                    if p.get("note"):
                        st.caption(p["note"])
                    if p.get("examples"):
                        ex = p["examples"]
                        if isinstance(ex, list):
                            st.write("\n".join([f"- {x}" for x in ex]))
                        else:
                            st.write(str(ex))
                    if p.get("forms"):
                        st.markdown("_Forms:_")
                        st.json(p["forms"])
                    if p.get("endings"):
                        st.markdown("_Endings:_")
                        st.json(p["endings"])
                    st.divider()
                else:
                    st.json(p)

    if forms:
        with st.expander(f"Forms ({len(forms)})"):
            st.json(forms)

    if endings:
        with st.expander(f"Endings ({len(endings)})"):
            st.json(endings)


def render_grammar_tab() -> None:
    st.subheader("Grammar")

    if not GRAMMAR_DIR.exists():
        st.error(f"Missing grammar directory: {GRAMMAR_DIR}")
        st.stop()

    rules_df = load_grammar()
    if rules_df.empty:
        st.warning("No grammar rules found after parsing.")
        st.stop()

    file_map = build_file_map(rules_df)

    mode = st.radio("Mode", ["Browse", "Compare languages"], horizontal=True)

    st.sidebar.header("Grammar selection")
    st.sidebar.markdown(
        "Select **one or more languages** and **one or more categories**.\n\n"
        "Matching files are chosen automatically."
    )

    langs_in_data = sorted(rules_df["language"].dropna().unique().tolist())
    lang_options = [l for l in LANGS if l in langs_in_data] + [l for l in langs_in_data if l not in LANGS]
    cat_options = sorted(rules_df["real_category"].dropna().unique().tolist())

    g_lang_sel = st.sidebar.multiselect("Languages", options=lang_options, default=[], key="g_langs")
    g_cat_sel = st.sidebar.multiselect("Categories (first line)", options=cat_options, default=[], key="g_cats")

    if not g_lang_sel or not g_cat_sel:
        st.info("Select at least one language and one category to load grammar data.")
        st.stop()

    selected_pairs = [(l, c) for l in g_lang_sel for c in g_cat_sel]
    selected_files = sorted({file_map.get(pair) for pair in selected_pairs if file_map.get(pair)})

    if not selected_files:
        st.warning("No files found for the selected languages + categories.")
        st.stop()

    rules_scope = rules_df[rules_df["file"].isin(selected_files)].copy()

    # Available columns for this scope (non-empty)
    available_cols = non_empty_columns(rules_scope)
    display_cols = [c for c in available_cols if c not in HIDDEN_COLS]

    # Always keep language + real_category visible (useful when multiselect)
    for must in ["language", "real_category"]:
        if must in rules_scope.columns and must not in display_cols:
            display_cols.insert(0, must)

    if mode == "Browse":
        # Expand applies dict into columns for filtering
        applies_wide = pd.json_normalize(rules_scope["applies"]).add_prefix("applies::")
        dfw = pd.concat([rules_scope.reset_index(drop=True), applies_wide.reset_index(drop=True)], axis=1)

        applies_cols = [c for c in dfw.columns if c.startswith("applies::")]
        applies_cols = [c for c in applies_cols if (dfw[c].map(lambda x: not is_empty_value(x))).any()]

        constraints = build_constraints_ui(applies_cols, dfw, key_prefix="g_rules_applies")
        df2 = apply_constraints(dfw, constraints)

        q = st.text_input("Search (pattern / note / examples)", value="", key="g_rules_search")
        if q:
            ql = q.lower()
            df2 = df2[
                df2["pattern"].fillna("").str.lower().str.contains(ql, regex=False)
                | df2["note"].fillna("").str.lower().str.contains(ql, regex=False)
                | df2["examples"].fillna("").str.lower().str.contains(ql, regex=False)
            ]

        cols_to_show = [c for c in display_cols if c in df2.columns]

        # If user selected constraints, include those applies columns too (if present)
        for c in constraints.keys():
            if c in df2.columns and c not in cols_to_show:
                cols_to_show.append(c)

        # Drop columns that are entirely empty after filtering
        cols_to_show = [c for c in cols_to_show if (df2[c].map(lambda x: not is_empty_value(x))).any()]

        st.caption(f"{len(df2):,} rule rows shown (from {len(selected_files)} selected files).")

        event = st.dataframe(
            df2[cols_to_show],
            use_container_width=True,
            hide_index=True,
            column_config={c: None for c in HIDDEN_COLS if c in df2.columns},
            on_select="rerun",
            selection_mode="single-row",
        )

        sel = (event.selection or {}).get("rows", [])
        if sel:
            row = df2.iloc[sel[0]]
            st.markdown("### Rule details")
            render_rule_details(row.get("details_json", ""))

    else:
        cats = sorted(rules_scope["real_category"].dropna().unique().tolist())
        chosen_cat = st.selectbox("Category", cats)

        langs_in_cat = sorted(
            rules_scope[rules_scope["real_category"] == chosen_cat]["language"].dropna().unique().tolist()
        )
        ordered_langs_in_cat = [l for l in LANGS if l in langs_in_cat] + [l for l in langs_in_cat if l not in LANGS]

        chosen_langs = st.multiselect(
            "Languages to compare",
            options=ordered_langs_in_cat,
            default=ordered_langs_in_cat[:2],
            key="g_compare_langs",
        )
        if not chosen_langs:
            st.info("Pick at least one language.")
            st.stop()

        compare_view = st.radio("Compare view", ["Matrix (patterns)", "Detail (one rule)"], horizontal=True)

        scoped_cat = rules_scope[rules_scope["real_category"] == chosen_cat].copy()

        if compare_view == "Matrix (patterns)":
            matrix = scoped_cat.pivot_table(
                index="rule_id",
                columns="language",
                values="pattern",
                aggfunc="first",
            )
            cols = [l for l in chosen_langs if l in matrix.columns]
            matrix = matrix[cols].reset_index()
            st.caption("Rule-by-language pattern matrix (blank means missing).")
            st.dataframe(matrix, use_container_width=True, hide_index=True)

        else:
            rule_ids = sorted(scoped_cat["rule_id"].dropna().unique().tolist())
            chosen_rule = st.selectbox("Rule", rule_ids)

            rule_rows = scoped_cat[scoped_cat["rule_id"] == chosen_rule]

            cols = st.columns(len(chosen_langs))
            for i, lang in enumerate(chosen_langs):
                with cols[i]:
                    st.markdown(f"### {lang}")
                    r = rule_rows[rule_rows["language"] == lang]
                    if r.empty:
                        st.info("No rule found for this language.")
                        continue
                    r0 = r.iloc[0]

                    if r0.get("pattern"):
                        st.markdown("**Pattern (summary)**")
                        st.write(r0.get("pattern") or "")

                    st.markdown("**Applies**")
                    st.code(r0.get("applies_json") or "{}")

                    if r0.get("note"):
                        st.markdown("**Note**")
                        st.write(r0["note"])

                    if r0.get("examples"):
                        st.markdown("**Examples**")
                        st.write(r0.get("examples") or "")

                    if r0.get("has_details"):
                        st.markdown("**Pattern / Forms / Endings**")
                        render_rule_details(r0.get("details_json", ""))
