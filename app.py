# app.py
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st
import yaml  # pip install pyyaml


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Linguistics Data Browser", layout="wide")

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


# ----------------------------
# Utilities: file discovery
# ----------------------------
def iter_files(root: Path, recursive: bool, exts: set[str]) -> List[Path]:
    if not root.exists():
        return []
    globber = root.rglob("*") if recursive else root.glob("*")
    out: List[Path] = []
    for p in globber:
        if not p.is_file():
            continue
        # special-case .jsonl.gz
        if p.name.lower().endswith(".jsonl.gz") and ".jsonl.gz" in exts:
            out.append(p)
        elif p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


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


# ----------------------------
# Vocab: streaming JSONL/JSONL.GZ
# ----------------------------
def iter_jsonl_records(path: Path) -> Iterable[dict]:
    """Yield dict records from .jsonl or .jsonl.gz without loading whole file into memory."""
    if path.name.lower().endswith(".jsonl.gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = path.open("r", encoding="utf-8")

    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def vocab_record_categories(entry: dict) -> List[str]:
    """
    Collect categories from entry.senses[].categories (if present).
    """
    cats: List[str] = []
    senses = entry.get("senses") or []
    if isinstance(senses, list):
        for s in senses:
            if isinstance(s, dict):
                sc = s.get("categories") or []
                if isinstance(sc, list):
                    for c in sc:
                        if isinstance(c, str):
                            cats.append(c)
    # de-dup while preserving order
    seen = set()
    out = []
    for c in cats:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def vocab_record_first_gloss(entry: dict) -> str:
    senses = entry.get("senses") or []
    if isinstance(senses, list):
        for s in senses:
            if isinstance(s, dict):
                glosses = s.get("glosses") or []
                if isinstance(glosses, list) and glosses:
                    g0 = glosses[0]
                    if isinstance(g0, str):
                        return g0
    return ""


def vocab_record_etymology(entry: dict) -> str:
    """
    Try common keys; keep best-effort (we also store full record in details_json).
    """
    for k in ["etymology_text", "etymology", "etymologies"]:
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        # some datasets store list of strings
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return " | ".join(v)
        # some store list of dicts
        if isinstance(v, list) and v and isinstance(v[0], dict):
            # try a common field
            txts = []
            for d in v:
                if isinstance(d, dict):
                    t = d.get("text") or d.get("etymology_text")
                    if isinstance(t, str) and t.strip():
                        txts.append(t.strip())
            if txts:
                return " | ".join(txts)
    return ""


@st.cache_data(show_spinner=True)
def build_vocab_index(max_unique_categories: int = 5000) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    One-time index scan (cached):
      - for each file: which langs/POS appear, plus a sample set of categories
    This enables "select lang/pos/category -> choose files" without loading everything.
    """
    files = iter_files(VOCAB_DIR, recursive=True, exts=VOCAB_SUPPORTED)
    rows: List[Dict[str, Any]] = []
    all_langs: set[str] = set()
    all_pos: set[str] = set()
    all_cats: set[str] = set()

    for fp in files:
        langs: set[str] = set()
        poss: set[str] = set()
        cats: set[str] = set()

        for e in iter_jsonl_records(fp):
            lang = e.get("lang")
            pos = e.get("pos")
            if isinstance(lang, str) and lang:
                langs.add(lang)
                all_langs.add(lang)
            if isinstance(pos, str) and pos:
                poss.add(pos)
                all_pos.add(pos)

            for c in vocab_record_categories(e):
                if len(cats) < max_unique_categories:
                    cats.add(c)
                if len(all_cats) < max_unique_categories:
                    all_cats.add(c)

        rows.append(
            {
                "file": str(fp.relative_to(VOCAB_DIR)),
                "langs": sorted(langs),
                "pos": sorted(poss),
                "categories": sorted(cats),
            }
        )

    idx_df = pd.DataFrame(rows)
    return idx_df, sorted(all_langs), sorted(all_pos), sorted(all_cats)


def select_vocab_files(idx_df: pd.DataFrame, lang: str | None, pos: str | None, cat: str | None) -> List[str]:
    """
    Pick the files that could contain requested (lang, pos, category).
    Uses the cached file index.
    """
    df = idx_df
    if lang:
        df = df[df["langs"].apply(lambda xs: isinstance(xs, list) and lang in xs)]
    if pos:
        df = df[df["pos"].apply(lambda xs: isinstance(xs, list) and pos in xs)]
    if cat:
        df = df[df["categories"].apply(lambda xs: isinstance(xs, list) and cat in xs)]
    return df["file"].tolist()


@st.cache_data(show_spinner=True)
def load_vocab_entries_filtered(
    selected_files: Tuple[str, ...],
    lang: str | None,
    pos: str | None,
    cat: str | None,
    word_query: str | None,
    limit_rows: int,
) -> pd.DataFrame:
    """
    Load only matching entries from selected files (streaming).
    Stores full record in details_json so user can view ALL info (etymology, senses, etc).
    """
    rows: List[Dict[str, Any]] = []
    q = (word_query or "").strip().lower()

    for rel in selected_files:
        fp = VOCAB_DIR / rel
        if not fp.exists():
            continue

        for e in iter_jsonl_records(fp):
            # filters
            if lang and e.get("lang") != lang:
                continue
            if pos and e.get("pos") != pos:
                continue
            if cat:
                cats = vocab_record_categories(e)
                if cat not in cats:
                    continue

            word = e.get("word") or e.get("lemma") or e.get("_key") or ""
            if q and isinstance(word, str):
                if q not in word.lower():
                    continue

            rows.append(
                {
                    "lang": e.get("lang"),
                    "pos": e.get("pos"),
                    "word": word,
                    "first_gloss": vocab_record_first_gloss(e),
                    "categories": " | ".join(vocab_record_categories(e)),
                    "etymology": vocab_record_etymology(e),
                    "n_senses": len(e.get("senses") or []) if isinstance(e.get("senses"), list) else 0,
                    "details_json": json.dumps(e, ensure_ascii=False),
                }
            )

            if len(rows) >= limit_rows:
                break

        if len(rows) >= limit_rows:
            break

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=True)
def load_vocab_word_exact(
    selected_files: Tuple[str, ...],
    word_exact: str,
    langs: Tuple[str, ...],
    pos: str | None,
    cat: str | None,
    limit_per_lang: int = 25,
) -> pd.DataFrame:
    """
    For compare: fetch entries where word == word_exact (case-insensitive) for selected languages.
    """
    target = word_exact.strip().lower()
    wanted_langs = set(langs)
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {l: 0 for l in wanted_langs}

    for rel in selected_files:
        fp = VOCAB_DIR / rel
        if not fp.exists():
            continue

        for e in iter_jsonl_records(fp):
            lang = e.get("lang")
            if lang not in wanted_langs:
                continue
            if pos and e.get("pos") != pos:
                continue
            if cat:
                cats = vocab_record_categories(e)
                if cat not in cats:
                    continue

            word = e.get("word") or e.get("lemma") or e.get("_key") or ""
            if not isinstance(word, str) or word.strip().lower() != target:
                continue

            if counts.get(lang, 0) >= limit_per_lang:
                continue

            rows.append(
                {
                    "lang": lang,
                    "pos": e.get("pos"),
                    "word": word,
                    "first_gloss": vocab_record_first_gloss(e),
                    "categories": " | ".join(vocab_record_categories(e)),
                    "etymology": vocab_record_etymology(e),
                    "n_senses": len(e.get("senses") or []) if isinstance(e.get("senses"), list) else 0,
                    "details_json": json.dumps(e, ensure_ascii=False),
                }
            )
            counts[lang] = counts.get(lang, 0) + 1

        # early exit if all languages hit limit
        if all(counts.get(l, 0) >= limit_per_lang for l in wanted_langs):
            break

    return pd.DataFrame(rows)


# ----------------------------
# Display/Filtering helpers
# ----------------------------
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

    feature_search = st.sidebar.text_input(
        "Search feature/column", value="", key=f"{key_prefix}::featsearch"
    )

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


def render_rule_details(details_json: str):
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


def render_vocab_details(details_json: str):
    """
    Show *all* information for a vocab entry, organized a bit.
    Falls back to raw JSON if keys are unknown.
    """
    try:
        e = json.loads(details_json or "{}")
    except Exception:
        st.warning("Could not parse details_json.")
        return

    # A few common sections
    with st.expander("Raw entry (JSON)", expanded=False):
        st.json(e)

    # Try to surface some common keys if they exist
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


# ----------------------------
# App UI
# ----------------------------
st.title("Linguistics Data Browser")

tab_grammar, tab_vocab = st.tabs(["Grammar", "Vocab"])


# ===== Grammar tab =====
with tab_grammar:
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

        langs_in_cat = sorted(rules_scope[rules_scope["real_category"] == chosen_cat]["language"].dropna().unique().tolist())
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


# ===== Vocab tab =====
with tab_vocab:
    st.subheader("Vocab (Wiktionary)")

    if not VOCAB_DIR.exists():
        st.error(f"Vocab directory missing: {VOCAB_DIR}")
        st.stop()

    vocab_mode = st.radio("Mode", ["Browse", "Compare words"], horizontal=True, key="v_mode")

    # Build cached per-file index so we can pick files by (lang/pos/category) without loading everything
    with st.spinner("Preparing vocab index (cached)..."):
        idx_df, all_langs, all_pos, all_cats = build_vocab_index()

    st.sidebar.header("Vocab selection")
    st.sidebar.markdown(
        "Pick **language**, optional **POS** and/or **category**.\n\n"
        "Only matching files are scanned, and only matching entries are loaded."
    )

    # Language: required for vocab (as requested)
    v_lang = st.sidebar.selectbox("Language", options=[""] + all_langs, index=0, key="v_lang")
    v_pos = st.sidebar.selectbox("POS (optional)", options=[""] + all_pos, index=0, key="v_pos")
    v_cat = st.sidebar.selectbox("Category (optional)", options=[""] + all_cats, index=0, key="v_cat")

    v_lang_val = v_lang if v_lang else None
    v_pos_val = v_pos if v_pos else None
    v_cat_val = v_cat if v_cat else None

    if not v_lang_val:
        st.info("Select a language in the sidebar to load vocab entries.")
        st.stop()

    # Determine which files to scan
    candidate_files = select_vocab_files(idx_df, v_lang_val, v_pos_val, v_cat_val)
    if not candidate_files:
        st.warning("No vocab files appear to contain that selection (based on index).")
        st.stop()

    # Safety/perf knob
    limit_rows = st.sidebar.slider("Max rows to load", min_value=100, max_value=5000, value=1000, step=100)

    if vocab_mode == "Browse":
        word_q = st.text_input("Search word (substring; optional)", value="", key="v_word_q")

        df = load_vocab_entries_filtered(
            selected_files=tuple(candidate_files),
            lang=v_lang_val,
            pos=v_pos_val,
            cat=v_cat_val,
            word_query=word_q,
            limit_rows=limit_rows,
        )

        if df.empty:
            st.warning("No entries matched your filters.")
            st.stop()

        # Show only non-empty columns for this selection (minus hidden)
        v_hidden = {"details_json"}  # don't show in table
        cols = [c for c in non_empty_columns(df) if c not in v_hidden]

        st.caption(f"{len(df):,} entries loaded (scanned {len(candidate_files)} file(s)).")

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

    else:
        # Compare words across languages:
        # user picks multiple languages, optional pos/category, and a word (exact match)
        st.markdown("Compare the same word across languages (exact match, case-insensitive).")

        compare_langs = st.multiselect(
            "Languages to compare",
            options=all_langs,
            default=[v_lang_val],
            key="v_compare_langs",
        )
        compare_langs = [l for l in compare_langs if l]

        word_exact = st.text_input("Word (exact)", value="", key="v_word_exact").strip()
        if not compare_langs or not word_exact:
            st.info("Choose at least one language and enter a word to compare.")
            st.stop()

        # Choose files that likely contain ANY of the compared languages (plus optional pos/cat)
        # We union the per-lang file sets.
        union_files: set[str] = set()
        for l in compare_langs:
            union_files.update(select_vocab_files(idx_df, l, v_pos_val, v_cat_val))

        if not union_files:
            st.warning("No files appear to contain that selection (based on index).")
            st.stop()

        df = load_vocab_word_exact(
            selected_files=tuple(sorted(union_files)),
            word_exact=word_exact,
            langs=tuple(compare_langs),
            pos=v_pos_val,
            cat=v_cat_val,
            limit_per_lang=25,
        )

        if df.empty:
            st.warning("No matching entries found for that word/language selection.")
            st.stop()

        # Side-by-side panels per language
        cols = st.columns(len(compare_langs))
        for i, lang in enumerate(compare_langs):
            with cols[i]:
                st.markdown(f"### {lang}")
                sub = df[df["lang"] == lang]
                if sub.empty:
                    st.info("No entry found.")
                    continue

                # If multiple entries per lang, show selectbox to choose which one
                if len(sub) > 1:
                    opts = list(range(len(sub)))
                    j = st.selectbox("Choose entry", options=opts, index=0, key=f"v_pick_{lang}")
                    row = sub.iloc[j]
                else:
                    row = sub.iloc[0]

                if row.get("pos"):
                    st.caption(f"POS: {row.get('pos')}")
                if row.get("first_gloss"):
                    st.markdown("**First gloss**")
                    st.write(row.get("first_gloss") or "")
                if row.get("etymology"):
                    st.markdown("**Etymology (summary)**")
                    st.write(row.get("etymology") or "")
                if row.get("categories"):
                    st.markdown("**Categories**")
                    st.write(row.get("categories") or "")

                st.markdown("**All details**")
                render_vocab_details(row.get("details_json", ""))


st.divider()
st.caption("Tip: if you change parsing logic, clear Streamlit cache (⋮ menu → Clear cache).")
